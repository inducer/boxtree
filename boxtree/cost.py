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
from pytools import memoize_method
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
    def __init__(self, translation_cost_model_factory):
        """
        :arg translation_cost_model_factory: a function, which takes tree dimension
            and the number of tree levels as arguments, returns an object of
            :class:`TranslationCostModel`.
        """
        self.translation_cost_model_factory = translation_cost_model_factory

    @abstractmethod
    def process_direct(self, traversal, c_p2p):
        """Direct evaluation cost of each target box of *traversal*.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg c_p2p: calibration constant.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (traversal.ntarget_boxes,), with each entry represents the cost of the
            box.
        """
        pass

    @abstractmethod
    def process_list2(self, traversal, m2l_cost):
        """
        :param traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :param m2l_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) representing the translation cost of each level.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (ntarget_or_target_parent_boxes,), with each entry represents the cost
            of multipole-to-local translations to this box.
        """
        pass

    @staticmethod
    @abstractmethod
    def aggregate(per_box_result):
        """ Sum all entries of *per_box_result* into a number.

        :param per_box_result: an object of :class:`numpy.ndarray` or
            :class:`pyopencl.array.Array`, the result to be sumed.
        :return: a :class:`float`, the result of the sum.
        """
        pass


class CLCostModel(CostModel):
    """
    Note: For methods in this class, argument *traversal* should live on device
        memory.
    """
    def __init__(self, queue, translation_cost_model_factory):
        self.queue = queue
        super(CLCostModel, self).__init__(
            translation_cost_model_factory
        )

    # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

    @memoize_method
    def process_direct_knl(self, particle_id_dtype, box_id_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template("""
                double *direct_by_itgt_box,
                ${box_id_t} *source_boxes_starts,
                ${box_id_t} *source_boxes_lists,
                ${particle_id_t} *box_source_counts_nonchild,
                ${particle_id_t} *box_target_counts_nonchild,
                ${box_id_t} *target_boxes,
                double c_p2p
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            Template(r"""
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

                ${particle_id_t} ntargets = box_target_counts_nonchild[
                    target_boxes[i]
                ];

                direct_by_itgt_box[i] += (nsources * ntargets * c_p2p);
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            name="process_direct"
        )

    def process_direct(self, traversal, c_p2p):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)
        particle_id_dtype = tree.particle_id_dtype
        box_id_dtype = tree.box_id_dtype

        count_direct_interaction_knl = self.process_direct_knl(
            particle_id_dtype, box_id_dtype
        )

        direct_by_itgt_box_dev = cl.array.zeros(
            self.queue, (ntarget_boxes,), dtype=np.float64
        )

        # List 1
        count_direct_interaction_knl(
            direct_by_itgt_box_dev,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            traversal.tree.box_source_counts_nonchild,
            traversal.tree.box_target_counts_nonchild,
            traversal.target_boxes,
            c_p2p
        )

        # List 3 close
        if traversal.from_sep_close_smaller_starts is not None:
            count_direct_interaction_knl(
                direct_by_itgt_box_dev,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                traversal.tree.box_source_counts_nonchild,
                traversal.tree.box_target_counts_nonchild,
                traversal.target_boxes,
                c_p2p
            )

        # List 4 close
        if traversal.from_sep_close_bigger_starts is not None:
            count_direct_interaction_knl(
                direct_by_itgt_box_dev,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                traversal.tree.box_source_counts_nonchild,
                traversal.tree.box_target_counts_nonchild,
                traversal.target_boxes,
                c_p2p
            )

        return direct_by_itgt_box_dev

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    @memoize_method
    def process_list2_knl(self, box_id_dtype, box_level_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                double *nm2l,
                ${box_id_t} *target_or_target_parent_boxes,
                ${box_id_t} *from_sep_siblings_starts,
                ${box_level_t} *box_levels,
                double *m2l_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} start = from_sep_siblings_starts[i];
                ${box_id_t} end = from_sep_siblings_starts[i+1];
                ${box_level_t} ilevel = box_levels[target_or_target_parent_boxes[i]];

                nm2l[i] += (end - start) * m2l_cost[ilevel];
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            name="process_list2"
        )

    def process_list2(self, traversal, m2l_cost):
        tree = traversal.tree
        box_id_dtype = tree.box_id_dtype
        box_level_dtype = tree.box_level_dtype

        ntarget_or_target_parent_boxes = len(traversal.target_or_target_parent_boxes)
        nm2l = cl.array.zeros(
            self.queue, (ntarget_or_target_parent_boxes,), dtype=np.float64
        )

        process_list2_knl = self.process_list2_knl(box_id_dtype, box_level_dtype)
        process_list2_knl(
            nm2l,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            tree.box_levels,
            m2l_cost
        )

        return nm2l

    # }}}

    @staticmethod
    def aggregate(per_box_result):
        return cl.array.sum(per_box_result).get().reshape(-1)[0]


class PythonCostModel(CostModel):
    def process_direct(self, traversal, c_p2p):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)

        # target box index -> nsources
        direct_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.float64)

        for itgt_box in range(ntarget_boxes):
            nsources = 0

            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources += tree.box_source_counts_nonchild[src_ibox]

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                    traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2]
                )
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                    traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2]
                )
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            ntargets = tree.box_target_counts_nonchild[
                traversal.target_boxes[itgt_box]
            ]
            direct_by_itgt_box[itgt_box] += (nsources * ntargets * c_p2p)

        return direct_by_itgt_box

    def process_list2(self, traversal, m2l_cost):
        tree = traversal.tree
        ntarget_or_target_parent_boxes = len(traversal.target_or_target_parent_boxes)
        nm2l = np.zeros(ntarget_or_target_parent_boxes, dtype=np.float64)

        for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
            start, end = traversal.from_sep_siblings_starts[itgt_box:itgt_box+2]

            ilevel = tree.box_levels[tgt_ibox]
            nm2l[itgt_box] += m2l_cost[ilevel] * (end - start)

        return nm2l

    @staticmethod
    def aggregate(per_box_result):
        return np.sum(per_box_result)
