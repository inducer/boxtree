"""
.. autoclass:: ConstantOneTreeIndependentDataForWrangler
.. autoclass:: ConstantOneExpansionWrangler
"""

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

import numpy as np

from boxtree.fmm import ExpansionWranglerInterface, TreeIndependentDataForWrangler
from boxtree.timing import DummyTimingFuture


# {{{ constant one wrangler


class ConstantOneTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    """
    .. automethod:: __init__
    """


class ConstantOneExpansionWrangler(ExpansionWranglerInterface):
    """This implements the 'analytical routines' for a Green's function that is
    constant 1 everywhere. For 'charges' of 'ones', this should get every particle
    a copy of the particle count.

    Timing results returned by this wrangler contain the field *ops_elapsed*,
    which counts approximately the number of floating-point operations required.
    """

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_target_counts_nonchild[ibox])

    def multipole_expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    local_expansion_zeros = multipole_expansion_zeros

    def output_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def multipole_expansions_view(self, mpole_exps, level):
        # FIXME
        raise NotImplementedError

    def local_expansions_view(self, local_exps, level):
        # FIXME
        raise NotImplementedError

    @staticmethod
    def timing_future(ops):
        return DummyTimingFuture.from_op_count(ops)

    def form_multipoles(self, level_start_source_box_nrs, source_boxes,
            src_weight_vecs):
        src_weights, = src_weight_vecs
        mpoles = self.multipole_expansion_zeros()
        ops = 0

        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])
            ops += src_weights[pslice].size

        return mpoles, self.timing_future(ops)

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        tree = self.tree
        ops = 0

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
                        ops += 1

        return mpoles, self.timing_future(ops)

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weight_vecs):
        src_weights, = src_weight_vecs
        pot = self.output_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            nsrcs = 0
            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            #print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum
            ops += pot[tgt_pslice].size * nsrcs

        return pot, self.timing_future(ops)

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.local_expansion_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            #print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]
                ops += 1

            local_exps[tgt_ibox] += contrib

        return local_exps, self.timing_future(ops)

    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_nonsiblings_by_level,
            mpole_exps):
        pot = self.output_zeros()
        ops = 0

        for level, ssn in enumerate(from_sep_smaller_nonsiblings_by_level):
            for itgt_box, tgt_ibox in \
                    enumerate(target_boxes_by_source_level[level]):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                contrib = 0

                start, end = ssn.starts[itgt_box:itgt_box+2]
                for src_ibox in ssn.lists[start:end]:
                    contrib += mpole_exps[src_ibox]

                pot[tgt_pslice] += contrib
                ops += pot[tgt_pslice].size * (end - start)

        return pot, self.timing_future(ops)

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weight_vecs):
        src_weights, = src_weight_vecs
        local_exps = self.local_expansion_zeros()
        ops = 0

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            #print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            nsrcs = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                contrib += np.sum(src_weights[src_pslice])

            local_exps[tgt_ibox] += contrib
            ops += nsrcs

        return local_exps, self.timing_future(ops)

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        ops = 0

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]
                ops += 1

        return local_exps, self.timing_future(ops)

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.output_zeros()
        ops = 0

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]
            ops += pot[tgt_pslice].size

        return pot, self.timing_future(ops)

    def finalize_potentials(self, potentials, template_ary):
        return potentials

# }}}

# vim: foldmethod=marker:filetype=pyopencl
