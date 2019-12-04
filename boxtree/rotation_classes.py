from __future__ import division

__copyright__ = "Copyright (C) 2019 Matt Wala"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from pytools import Record, memoize_method
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.cltypes  # noqa
from pyopencl.elementwise import ElementwiseTemplate
from mako.template import Template
from boxtree.tools import DeviceDataRecord, InlineBinarySearch
from boxtree.traversal import TRAVERSAL_PREAMBLE_MAKO_DEFS

import logging
logger = logging.getLogger(__name__)

from pytools import log_process


# {{{ rotation classes builder

# Note that these kernels compute translation classes first, and
# these get converted to rotation classes in a second step, from Python.

TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE = Template(r"""//CL:mako//
    #define LEVEL_TO_RAD(level) \
        (root_extent * 1 / (coord_t) (1 << (level + 1)))

    // Return an integer vector indicating the a translation direction
    // as a multiple of the box diameter.
    inline int_coord_vec_t get_translation_vector(
        coord_t root_extent,
        int level,
        coord_vec_t source_center,
        coord_vec_t target_center)
    {
        int_coord_vec_t result = (int_coord_vec_t) 0;
        coord_t diam = 2 * LEVEL_TO_RAD(level);
        %for i in range(dimensions):
            result.s${i} = rint((target_center.s${i} - source_center.s${i}) / diam);
        %endfor
        return result;
    }

    // Compute the translation class for the given translation vector.  The
    // translation class maps a translation vector (a_1, a_2, ..., a_d) into
    // a dense range of integers [0, ..., (4*n+3)^d - 1], where
    // d is the dimension and n is well_sep_is_n_away.
    //
    // This relies on the fact that the entries of the vector will
    // always be in the range [-2n-1,...,2n+1].
    //
    // The mapping from vector to class is:
    //
    //                         \~~   d                 k-1
    //     cls(a ,a ,...,a ) =  >      (2n+1+a ) (4n+3)
    //          1  2      d    /__ k=1        k
    //
    // Returns -1 on error.
    inline int get_translation_class(int_coord_vec_t vec, int well_sep_is_n_away)
    {
        int dim_bound = 2 * well_sep_is_n_away + 1;
        %for i in range(dimensions):
            if (!(-dim_bound <= vec.s${i} && vec.s${i} <= dim_bound))
            {
                return -1;
            }
        %endfor

        int result = 0;
        int base = 4 * well_sep_is_n_away + 3;
        int mult = 1;
        %for i in range(dimensions):
            result += (2 * well_sep_is_n_away + 1 + vec.s${i}) * mult;
            mult *= base;
        %endfor
        return result;
    }
    """ + str(InlineBinarySearch("box_id_t")),
    strict_undefined=True)


TRANSLATION_CLASS_FINDER_TEMPLATE = ElementwiseTemplate(
    arguments=r"""//CL:mako//
    /* input: */
    box_id_t *from_sep_siblings_lists,
    box_id_t *from_sep_siblings_starts,
    box_id_t *target_or_target_parent_boxes,
    int ntarget_or_target_parent_boxes,
    coord_t *box_centers,
    int aligned_nboxes,
    coord_t root_extent,
    box_level_t *box_levels,
    int well_sep_is_n_away,

    /* output: */
    int *translation_classes,
    int *translation_class_is_used,
    int *error_flag,
    """,

    operation=TRAVERSAL_PREAMBLE_MAKO_DEFS + r"""//CL:mako//
    // Find the target box for this source box.
    box_id_t source_box_id = from_sep_siblings_lists[i];

    size_t itarget_box = bsearch(
        from_sep_siblings_starts, 1 + ntarget_or_target_parent_boxes, i);

    box_id_t target_box_id = target_or_target_parent_boxes[itarget_box];

    // Ensure levels are the same.
    if (box_levels[source_box_id] != box_levels[target_box_id])
    {
        atomic_or(error_flag, 1);
        PYOPENCL_ELWISE_CONTINUE;
    }

    // Compute the translation vector and translation class.
    ${load_center("source_center", "source_box_id")}
    ${load_center("target_center", "target_box_id")}

    int_coord_vec_t vec = get_translation_vector(
        root_extent, box_levels[source_box_id], source_center, target_center);

    int translation_class = get_translation_class(vec, well_sep_is_n_away);

    // Ensure valid translation class.
    if (translation_class == -1)
    {
        atomic_or(error_flag, 1);
        PYOPENCL_ELWISE_CONTINUE;
    }

    translation_classes[i] = translation_class;
    translation_class_is_used[translation_class] = 1;
    """)


class _KernelInfo(Record):
    pass


class RotationClassesInfo(DeviceDataRecord):
    r"""Interaction lists to help with matrix precomputations for rotation-based
    translations ("point and shoot").

    .. attribute:: nfrom_sep_siblings_rotation_classes

       The number of distinct rotation classes.

    .. attribute:: from_sep_siblings_rotation_classes

        ``int32 [*]``

        A list, corresponding to *from_sep_siblings_lists* of *trav*, of
        the rotation class of each box pair.

    .. attribute:: from_sep_siblings_rotation_class_to_angle

        ``coord_t [nfrom_sep_siblings_rotation_classes]``

        Maps rotation classes in *from_sep_siblings_rotation_classes* to
        rotation angles. This represents the angle between box translation
        pairs and the *z*-axis.

    """

    @property
    def nfrom_sep_siblings_rotation_classes(self):
        return len(self.from_sep_siblings_rotation_class_to_angle)


class RotationClassesBuilder(object):
    """Build rotation classes for List 2 translations.
    """

    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernel_info(self, dimensions, well_sep_is_n_away,
            box_id_dtype, box_level_dtype, coord_dtype):
        coord_vec_dtype = cl.cltypes.vec_types[coord_dtype, dimensions]
        int_coord_vec_dtype = cl.cltypes.vec_types[np.dtype(np.int32), dimensions]

        # Make sure translation classes can fit inside a 32 bit integer.
        if (not
                (
                    self.ntranslation_classes(well_sep_is_n_away, dimensions)
                    <= 1 + np.iinfo(np.int32).max)):
            raise ValueError("would overflow")

        preamble = TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE.render(
                dimensions=dimensions)

        translation_class_finder = (
                TRANSLATION_CLASS_FINDER_TEMPLATE.build(
                    self.context,
                    type_aliases=(
                        ("int_coord_vec_t", int_coord_vec_dtype),
                        ("coord_vec_t", coord_vec_dtype),
                        ("coord_t", coord_dtype),
                        ("box_id_t", box_id_dtype),
                        ("box_level_t", box_level_dtype),
                    ),
                    var_values=(
                        ("dimensions", dimensions),
                    ),
                    more_preamble=preamble))

        return _KernelInfo(translation_class_finder=translation_class_finder)

    @staticmethod
    def vec_gcd(vec):
        """Return the GCD of a list of integers."""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        result = abs(vec[0])
        for elem in vec[1:]:
            result = gcd(result, abs(elem))
        return result

    def compute_rotation_classes(self,
            well_sep_is_n_away, dimensions, used_translation_classes):
        """Convert translation classes to a list of rotation classes and angles."""
        angle_to_rot_class = {}
        angles = []

        ntranslation_classes = (
                self.ntranslation_classes(well_sep_is_n_away, dimensions))

        translation_class_to_rot_class = (
                np.empty(ntranslation_classes, dtype=np.int32))

        translation_class_to_rot_class[:] = -1

        for cls in used_translation_classes:
            vec = self.translation_class_to_vector(
                    well_sep_is_n_away, dimensions, cls)

            # Normalize the translation vector (by dividing by its GCD).
            #
            # We need this before computing the cosine of the rotation angle,
            # because generally in in floating point arithmetic, if k is a
            # positive scalar and v is a vector, we can't assume
            #
            #   kv[-1] / sqrt(|kv|^2) == v[-1] / sqrt(|v|^2).
            #
            # Normalizing ensures vectors that are positive integer multiples of
            # each other get classified into the same equivalence class of
            # rotations.
            vec //= self.vec_gcd(vec)

            # Compute the rotation angle for the vector.
            norm = np.linalg.norm(vec)
            assert norm != 0
            angle = np.arccos(vec[-1] / norm)

            # Find the rotation class.
            if angle in angle_to_rot_class:
                rot_class = angle_to_rot_class[angle]
            else:
                rot_class = len(angles)
                angle_to_rot_class[angle] = rot_class
                angles.append(angle)

            translation_class_to_rot_class[cls] = rot_class

        return translation_class_to_rot_class, angles

    @staticmethod
    def ntranslation_classes(well_sep_is_n_away, dimensions):
        return (4 * well_sep_is_n_away + 3) ** dimensions

    @staticmethod
    def translation_class_to_vector(well_sep_is_n_away, dimensions, cls):
        # This computes the vector for the translation class, using the inverse
        # of the formula found in get_translation_class() defined in
        # TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE.
        result = np.zeros(dimensions, dtype=np.int32)
        shift = 2 * well_sep_is_n_away + 1
        base = 4 * well_sep_is_n_away + 3
        for i in range(dimensions):
            result[i] = cls % base - shift
            cls //= base
        return result

    @log_process(logger, "build m2l rotation classes")
    def __call__(self, queue, trav, tree, wait_for=None):
        """Returns a pair *info*, *evt* where info is a :class:`RotationClassesInfo`.
        """

        # {{{ compute translation classes for list 2

        well_sep_is_n_away = trav.well_sep_is_n_away
        dimensions = tree.dimensions
        coord_dtype = tree.coord_dtype

        knl_info = self.get_kernel_info(
                dimensions, well_sep_is_n_away, tree.box_id_dtype,
                tree.box_level_dtype, coord_dtype)

        ntranslation_classes = (
                self.ntranslation_classes(well_sep_is_n_away, dimensions))

        translation_classes_lists = cl.array.empty(
                queue, len(trav.from_sep_siblings_lists), dtype=np.int32)

        translation_class_is_used = cl.array.zeros(
                queue, ntranslation_classes, dtype=np.int32)

        error_flag = cl.array.zeros(queue, 1, dtype=np.int32)

        evt = knl_info.translation_class_finder(
                trav.from_sep_siblings_lists,
                trav.from_sep_siblings_starts,
                trav.target_or_target_parent_boxes,
                trav.ntarget_or_target_parent_boxes,
                tree.box_centers,
                tree.aligned_nboxes,
                tree.root_extent,
                tree.box_levels,
                well_sep_is_n_away,
                translation_classes_lists,
                translation_class_is_used,
                error_flag,
                queue=queue, wait_for=wait_for)

        if (error_flag.get()):
            raise ValueError("could not compute translation classes")

        # }}}

        # {{{ convert translation classes to rotation classes

        used_translation_classes = (
                np.flatnonzero(translation_class_is_used.get()))

        translation_class_to_rotation_class, rotation_angles = (
                self.compute_rotation_classes(
                    well_sep_is_n_away, dimensions, used_translation_classes))

        # There should be no more than 2^(d-1) * (2n+1)^d distinct rotation
        # classes, since that is an upper bound on the number of distinct
        # positions for list 2 boxes.
        d = dimensions
        n = well_sep_is_n_away
        assert len(rotation_angles) <= 2**(d-1) * (2*n+1)**d

        rotation_classes_lists = (
                cl.array.take(
                    cl.array.to_device(queue, translation_class_to_rotation_class),
                    translation_classes_lists))

        rotation_angles = cl.array.to_device(queue, np.array(rotation_angles))

        # }}}

        return RotationClassesInfo(
                from_sep_siblings_rotation_classes=rotation_classes_lists,
                from_sep_siblings_rotation_class_to_angle=rotation_angles,
                ).with_queue(None), evt

# }}}

# vim: filetype=pyopencl:fdm=marker
