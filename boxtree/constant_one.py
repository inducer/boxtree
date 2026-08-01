"""
.. autoclass:: ConstantOneTreeIndependentDataForWrangler
.. autoclass:: ConstantOneExpansionWrangler
"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

from boxtree.fmm import ExpansionWranglerInterface, TreeIndependentDataForWrangler


if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as onp

    from arraycontext import Array, ArrayContext
    from pyopencl.algorithm import BuiltList
    from pytools.obj_array import ObjectArray1D

# {{{ constant one wrangler


class ConstantOneTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    """
    .. automethod:: __init__
    """


class ConstantOneExpansionWrangler(ExpansionWranglerInterface):
    """This implements the 'analytical routines' for a Green's function that is
    constant 1 everywhere. For 'charges' of 'ones', this should get every particle
    a copy of the particle count.
    """

    def _get_source_slice(self, ibox: int) -> slice:
        pstart = self.tree.box_source_starts[ibox]
        return slice(pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox: int) -> slice:
        pstart = self.tree.box_target_starts[ibox]
        return slice(pstart, pstart + self.tree.box_target_counts_nonchild[ibox])

    def multipole_expansion_zeros(self) -> onp.Array1D[np.floating[Any]]:
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    def local_expansion_zeros(self) -> onp.Array1D[np.floating[Any]]:
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    def output_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    @override
    def reorder_sources(self, source_array: Array) -> Array:
        return source_array[self.tree.user_source_ids]

    @override
    def reorder_potentials(self, potentials: Array) -> Array:
        return potentials[self.tree.sorted_target_ids]

    @override
    def multipole_expansions_view(
            self, mpole_exps: Array, level: int
        ) -> tuple[int, Array]:
        # FIXME
        raise NotImplementedError

    @override
    def local_expansions_view(
            self, local_exps: Array, level: int
        ) -> tuple[int, Array]:
        # FIXME
        raise NotImplementedError

    @override
    def form_multipoles(
            self,
            actx: ArrayContext,
            level_start_source_box_nrs: Array,
            source_boxes: Array,
            src_weight_vecs: Sequence[Array],
        ) -> Array:
        src_weights, = src_weight_vecs
        mpoles = self.multipole_expansion_zeros()

        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    @override
    def coarsen_multipoles(
            self,
            actx: ArrayContext,
            level_start_source_parent_box_nrs: Array,
            source_parent_boxes: Array,
            mpoles: Array) -> Array:
        tree = self.tree

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        mpoles[ibox] += mpoles[child]

        return mpoles

    @override
    def eval_direct(
            self,
            actx: ArrayContext,
            target_boxes: Array,
            neighbor_sources_starts: Array,
            neighbor_sources_lists: Array,
            src_weight_vecs: Sequence[Array]) -> Array:
        src_weights, = src_weight_vecs
        pot = self.output_zeros()

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            nsrcs = 0
            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            # print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum

        return pot

    @override
    def multipole_to_local(
            self,
            actx: ArrayContext,
            level_start_target_or_target_parent_box_nrs: Array,
            target_or_target_parent_boxes: Array,
            starts: Array,
            lists: Array,
            mpole_exps: Array) -> Array:
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            # print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]

            local_exps[tgt_ibox] += contrib

        return local_exps

    @override
    def eval_multipoles(
            self,
            actx: ArrayContext,
            target_boxes_by_source_level: ObjectArray1D[Array],
            from_sep_smaller_by_level: ObjectArray1D[BuiltList],
            mpole_exps: Array) -> Array:
        pot = self.output_zeros()

        for level, ssn in enumerate(from_sep_smaller_by_level):
            for itgt_box, tgt_ibox in enumerate(target_boxes_by_source_level[level]):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                contrib = 0
                start, end = ssn.starts[itgt_box:itgt_box+2]
                for src_ibox in ssn.lists[start:end]:
                    contrib += mpole_exps[src_ibox]

                pot[tgt_pslice] += contrib

        return pot

    @override
    def form_locals(
            self,
            actx: ArrayContext,
            level_start_target_or_target_parent_box_nrs: Array,
            target_or_target_parent_boxes: Array,
            starts: Array,
            lists: Array,
            src_weight_vecs: Sequence[Array]) -> Array:
        src_weights, = src_weight_vecs
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            # print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            nsrcs = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                contrib += np.sum(src_weights[src_pslice])

            local_exps[tgt_ibox] += contrib

        return local_exps

    @override
    def refine_locals(
            self,
            actx: ArrayContext,
            level_start_target_or_target_parent_box_nrs: Array,
            target_or_target_parent_boxes: Array,
            local_exps: Array) -> Array:
        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    @override
    def eval_locals(
            self,
            actx: ArrayContext,
            level_start_target_box_nrs: Array,
            target_boxes: Array,
            local_exps: Array) -> Array:
        pot = self.output_zeros()

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot

    @override
    def finalize_potentials(self, actx: ArrayContext, potentials: Array) -> Array:
        return potentials

# }}}

# vim: foldmethod=marker
