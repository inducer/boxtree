from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Matt Wala
Copyright (C) 2018 Hao Gao
"""

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
import pyopencl as cl
import pyopencl.array  # noqa: F401
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from functools import partial
from pymbolic import var
import sys

if sys.version_info >= (3, 0):
    Template = partial(Template, strict_undefined=True)
else:
    Template = partial(Template, strict_undefined=True, disable_unicode=True)

if sys.version_info >= (3, 4):
    from abc import ABC, abstractmethod
else:
    from abc import ABCMeta, abstractmethod
    ABC = ABCMeta('ABC', (), {})


class TranslationCostModel:
    """Provides modeled costs for individual translations or evaluations."""

    def __init__(self, ncoeffs_fmm_by_level, uses_point_and_shoot):
        self.ncoeffs_fmm_by_level = ncoeffs_fmm_by_level
        self.uses_point_and_shoot = uses_point_and_shoot

    @staticmethod
    def direct():
        return var("c_p2p")

    def p2l(self, level):
        return var("c_p2l") * self.ncoeffs_fmm_by_level[level]

    def l2p(self, level):
        return var("c_l2p") * self.ncoeffs_fmm_by_level[level]

    def p2m(self, level):
        return var("c_p2m") * self.ncoeffs_fmm_by_level[level]

    def m2p(self, level):
        return var("c_m2p") * self.ncoeffs_fmm_by_level[level]

    def m2m(self, src_level, tgt_level):
        return var("c_m2m") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def l2l(self, src_level, tgt_level):
        return var("c_l2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2l(self, src_level, tgt_level):
        return var("c_m2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def e2e_cost(self, nsource_coeffs, ntarget_coeffs):
        if self.uses_point_and_shoot:
            return (
                    # Rotate the coordinate system to be z axis aligned.
                    nsource_coeffs ** (3 / 2)
                    # Translate the expansion along the z axis.
                    + nsource_coeffs ** (1 / 2) * ntarget_coeffs
                    # Rotate the coordinate system back.
                    + ntarget_coeffs ** (3 / 2))

        return nsource_coeffs * ntarget_coeffs


# {{{ translation cost model factories

def pde_aware_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation operators that make use of the
    knowledge that the potential satisfies a PDE.
    """
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])
    ncoeffs_fmm = (p_fmm + 1) ** (dim - 1)

    if dim == 3:
        uses_point_and_shoot = True
    else:
        uses_point_and_shoot = False

    return TranslationCostModel(
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=uses_point_and_shoot
    )


def taylor_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation based on Taylor expansions
    in Cartesian coordinates.
    """
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])
    ncoeffs_fmm = (p_fmm + 1) ** dim

    return TranslationCostModel(
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=False
    )

# }}}


class CostModel(ABC):
    def __init__(self, translation_cost_model_factory, calibration_params=None):
        """
        :arg translation_cost_model_factory: a function, which takes tree dimension
            and the number of tree levels as arguments, returns an object of
            :class:`TranslationCostModel`.
        :arg calibration_params: TODO
        """
        self.translation_cost_model_factory = translation_cost_model_factory
        if calibration_params is None:
            calibration_params = dict()
        self.calibration_params = calibration_params

    def with_calibration_params(self, calibration_params):
        """Return a copy of *self* with a new set of calibration parameters."""
        return type(self)(
                translation_cost_model_factory=self.translation_cost_model_factory,
                calibration_params=calibration_params)

    @abstractmethod
    def collect_direct_interaction_data(self, traversal):
        """Count the number of sources in direct interaction boxes.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :return: a :class:`dict` contains fields "nlist1_srcs_by_itgt_box",
            "nlist3close_srcs_by_itgt_box", and "nlist4close_srcs_by_itgt_box". Each
            of these fields is a :class:`numpy.ndarray` of shape
            (traversal.ntarget_boxes,), documenting the number of sources in list 1,
            list 3 close and list 4 close, respectively.
        """
        pass

    def count_direct(self, xlat_cost, traversal):
        """Count direct evaluations of each target box of *traversal*.

        :arg xlat_cost: a :class:`TranslationCostModel` object which specifies the
            translation cost.
        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :return: a :class:`numpy.ndarray` of shape (traversal.ntarget_boxes,).
        """
        tree = traversal.tree

        direct_interaction_data = self.collect_direct_interaction_data(traversal)
        nlist1_srcs_by_itgt_box = (
                direct_interaction_data["nlist1_srcs_by_itgt_box"])
        nlist3close_srcs_by_itgt_box = (
                direct_interaction_data["nlist3close_srcs_by_itgt_box"])
        nlist4close_srcs_by_itgt_box = (
                direct_interaction_data["nlist4close_srcs_by_itgt_box"])

        ntargets = tree.box_target_counts_nonchild[
            traversal.target_boxes
        ]

        return ntargets * (
                nlist1_srcs_by_itgt_box
                + nlist3close_srcs_by_itgt_box
                + nlist4close_srcs_by_itgt_box
                ) * xlat_cost.direct()


class CLCostModel(CostModel):
    def __init__(self, queue, translation_cost_model_factory,
                 calibration_params=None):
        self.queue = queue
        super().__init__(translation_cost_model_factory, calibration_params)

    def collect_direct_interaction_data(self, traversal):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)
        particle_id_dtype = tree.particle_id_dtype
        box_id_dtype = tree.box_id_dtype

        count_direct_interaction_knl = ElementwiseKernel(
            self.queue.context,
            Template("""
                ${particle_id_t} *srcs_by_itgt_box,
                ${box_id_t} *source_boxes_starts,
                ${box_id_t} *source_boxes_lists,
                ${particle_id_t} *box_source_counts_nonchild
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            Template("""
                ${particle_id_t} nsources = 0;
                ${box_id_t} source_boxes_start_idx = source_boxes_starts[i];
                ${box_id_t} source_boxes_end_idx = source_boxes_starts[i + 1];

                for(${box_id_t} cur_source_boxes_idx = source_boxes_start_idx;
                    cur_source_boxes_idx < source_boxes_end_idx;
                    cur_source_boxes_idx++)
                {
                    ${box_id_t} cur_source_box = source_boxes_lists[
                        cur_source_boxes_idx
                    ];
                    nsources += box_source_counts_nonchild[cur_source_box];
                }

                srcs_by_itgt_box[i] = nsources;
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            name="count_direct_interaction"
        )

        box_source_counts_nonchild_dev = cl.array.to_device(
            self.queue, tree.box_source_counts_nonchild
        )
        result = dict()

        # List 1
        nlist1_srcs_by_itgt_box_dev = cl.array.zeros(
            self.queue, (ntarget_boxes,), dtype=particle_id_dtype
        )
        neighbor_source_boxes_starts_dev = cl.array.to_device(
            self.queue, traversal.neighbor_source_boxes_starts
        )
        neighbor_source_boxes_lists_dev = cl.array.to_device(
            self.queue, traversal.neighbor_source_boxes_lists
        )

        count_direct_interaction_knl(
            nlist1_srcs_by_itgt_box_dev,
            neighbor_source_boxes_starts_dev,
            neighbor_source_boxes_lists_dev,
            box_source_counts_nonchild_dev
        )

        result["nlist1_srcs_by_itgt_box"] = nlist1_srcs_by_itgt_box_dev.get()

        # List 3 close
        if traversal.from_sep_close_smaller_starts is not None:
            nlist3close_srcs_by_itgt_box_dev = cl.array.zeros(
                self.queue, (ntarget_boxes,), dtype=particle_id_dtype
            )
            from_sep_close_smaller_starts_dev = cl.array.to_device(
                self.queue, traversal.from_sep_close_smaller_starts
            )
            from_sep_close_smaller_lists_dev = cl.array.to_device(
                self.queue, traversal.from_sep_close_smaller_lists
            )

            count_direct_interaction_knl(
                nlist3close_srcs_by_itgt_box_dev,
                from_sep_close_smaller_starts_dev,
                from_sep_close_smaller_lists_dev,
                box_source_counts_nonchild_dev
            )

            result["nlist3close_srcs_by_itgt_box"] = \
                nlist3close_srcs_by_itgt_box_dev.get()

        # List 4 close
        if traversal.from_sep_close_bigger_starts is not None:
            nlist4close_srcs_by_itgt_box_dev = cl.array.zeros(
                self.queue, (ntarget_boxes,), dtype=particle_id_dtype
            )
            from_sep_close_bigger_starts_dev = cl.array.to_device(
                self.queue, traversal.from_sep_close_bigger_starts
            )
            from_sep_close_bigger_lists_dev = cl.array.to_device(
                self.queue, traversal.from_sep_close_bigger_lists
            )

            count_direct_interaction_knl(
                nlist4close_srcs_by_itgt_box_dev,
                from_sep_close_bigger_starts_dev,
                from_sep_close_bigger_lists_dev,
                box_source_counts_nonchild_dev
            )

            result["nlist4close_srcs_by_itgt_box"] = \
                nlist4close_srcs_by_itgt_box_dev.get()

        return result


class PythonCostModel(CostModel):
    def collect_direct_interaction_data(self, traversal):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)

        # target box index -> nsources
        nlist1_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist3close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist4close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)

        for itgt_box in range(ntarget_boxes):
            nlist1_srcs = 0
            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nlist1_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist1_srcs_by_itgt_box[itgt_box] = nlist1_srcs

            nlist3close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                        traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nlist3close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist3close_srcs_by_itgt_box[itgt_box] = nlist3close_srcs

            nlist4close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                        traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nlist4close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist4close_srcs_by_itgt_box[itgt_box] = nlist4close_srcs

        result = dict()
        result["nlist1_srcs_by_itgt_box"] = nlist1_srcs_by_itgt_box
        result["nlist3close_srcs_by_itgt_box"] = nlist3close_srcs_by_itgt_box
        result["nlist4close_srcs_by_itgt_box"] = nlist4close_srcs_by_itgt_box

        return result
