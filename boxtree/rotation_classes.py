"""
Rotation classes data structure
-------------------------------

.. autoclass:: RotationClassesInfo

Build rotation classes
----------------------

.. autoclass:: RotationClassesBuilder
"""
from __future__ import annotations


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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pytools import log_process

from boxtree.array_context import PyOpenCLArrayContext, dataclass_array_container
from boxtree.translation_classes import TranslationClassesBuilder


if TYPE_CHECKING:
    from arraycontext import Array

logger = logging.getLogger(__name__)


# {{{ rotation classes builder

@dataclass_array_container
@dataclass(frozen=True)
class RotationClassesInfo:
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

    from_sep_siblings_rotation_classes: Array
    from_sep_siblings_rotation_class_to_angle: Array

    @property
    def nfrom_sep_siblings_rotation_classes(self):
        return len(self.from_sep_siblings_rotation_class_to_angle)


class RotationClassesBuilder:
    """Build rotation classes for List 2 translations.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, array_context: PyOpenCLArrayContext):
        self._setup_actx: PyOpenCLArrayContext = array_context
        self.tcb = TranslationClassesBuilder(array_context)

    @staticmethod
    def vec_gcd(vec) -> int:
        """Return the GCD of a list of integers."""
        import math

        # TODO: math.gcd supports a list of integers from >= 3.9
        result = abs(vec[0])
        for elem in vec[1:]:
            result = math.gcd(result, abs(elem))

        return result

    def compute_rotation_classes(self,
            well_sep_is_n_away: int, dimensions: int, used_translation_classes):
        """Convert translation classes to a list of rotation classes and angles."""
        angle_to_rot_class = {}
        angles = []

        ntranslation_classes_per_level = (
                self.tcb.ntranslation_classes_per_level(well_sep_is_n_away,
                    dimensions))

        translation_class_to_rot_class = (
                np.empty(ntranslation_classes_per_level, dtype=np.int32))

        translation_class_to_rot_class[:] = -1

        for cls in used_translation_classes:
            vec = self.tcb.translation_class_to_normalized_vector(
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

    @log_process(logger, "build m2l rotation classes")
    def __call__(self, actx, trav, tree, wait_for=None):
        """Returns a pair *info*, *evt* where info is a :class:`RotationClassesInfo`.
        """
        evt, translation_class_is_used, translation_classes_lists = \
            self.tcb.compute_translation_classes(actx, trav, tree, wait_for, False)

        d = tree.dimensions
        n = trav.well_sep_is_n_away

        # convert translation classes to rotation classes

        used_translation_classes = (
                np.flatnonzero(actx.to_numpy(translation_class_is_used)))

        translation_class_to_rotation_class, rotation_angles = (
                self.compute_rotation_classes(n, d, used_translation_classes))

        # There should be no more than 2^(d-1) * (2n+1)^d distinct rotation
        # classes, since that is an upper bound on the number of distinct
        # positions for list 2 boxes.
        assert len(rotation_angles) <= 2**(d-1) * (2*n+1)**d

        rotation_classes_lists = actx.from_numpy(
            translation_class_to_rotation_class
            )[translation_classes_lists]
        rotation_angles = actx.from_numpy(np.array(rotation_angles))

        info = RotationClassesInfo(
                from_sep_siblings_rotation_classes=rotation_classes_lists,
                from_sep_siblings_rotation_class_to_angle=rotation_angles,
                )

        return actx.freeze(info), evt

# }}}

# vim: filetype=pyopencl:fdm=marker
