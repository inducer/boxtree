from __future__ import division

"""Integration between boxtree and pyfmmlib."""

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




class Helmholtz2DExpansionWrangler:
    def __init__(self, tree, helmholtz_k, nterms):
        self.tree = tree
        self.helmholtz_k = helmholtz_k
        self.nterms = nterms

    def expansion_zeros(self):
        return np.zeros((self.tree.nboxes, 2*self.nterms+1), dtype=np.complex128)

    def potential_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.complex128)

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_target_counts[ibox])

    def _get_sources(self, pslice):
        # FIXME yuck!
        return np.array([
            self.tree.sources[idim][pslice]
            for idim in range(self.tree.dimensions)
            ], order="F")

    def _get_targets(self, pslice):
        # FIXME yuck!
        return np.array([
            self.tree.targets[idim][pslice]
            for idim in range(self.tree.dimensions)
            ], order="F")

    def reorder_src_weights(self, src_weights):
        return src_weights[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def form_multipoles(self, leaf_boxes, src_weights):
        rscale = 1 # FIXME

        from pyfmmlib import h2dformmp

        mpoles = self.expansion_zeros()
        for ibox in leaf_boxes:
            pslice = self._get_source_slice(ibox)

            if pslice.stop - pslice.start == 0:
                continue

            ier, mpoles[ibox] = h2dformmp(
                    self.helmholtz_k, rscale, self._get_sources(pslice),
                    src_weights[pslice],
                    self.tree.box_centers[:, ibox], self.nterms)
            if ier:
                raise RuntimeError("h2dformmp failed")

        return mpoles

    def coarsen_multipoles(self, parent_boxes, start_parent_box, end_parent_box,
            mpoles):
        tree = self.tree
        rscale = 1 # FIXME

        from pyfmmlib import h2dmpmp_vec

        for ibox in parent_boxes[start_parent_box:end_parent_box]:
            parent_center = tree.box_centers[:, ibox]
            for child in tree.box_child_ids[:, ibox]:
                if child:
                    child_center = tree.box_centers[:, child]

                    new_mp = h2dmpmp_vec(
                            self.helmholtz_k,
                            rscale, child_center, mpoles[child],
                            rscale, parent_center, self.nterms)

                    mpoles[ibox] += new_mp[:, 0]

    def eval_direct(self, leaf_boxes, neighbor_leaves_starts, neighbor_leaves_lists,
            src_weights):
        pot = self.potential_zeros()

        from pyfmmlib import hpotgrad2dall_vec

        for itgt_leaf, itgt_box in enumerate(leaf_boxes):
            tgt_pslice = self._get_target_slice(itgt_box)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tgt_result = np.zeros(tgt_pslice.stop - tgt_pslice.start, np.complex128)
            start, end = neighbor_leaves_starts[itgt_leaf:itgt_leaf+2]
            for isrc_box in neighbor_leaves_lists[start:end]:
                src_pslice = self._get_source_slice(isrc_box)

                if src_pslice.stop - src_pslice.start == 0:
                    continue

                tmp_pot, _, _ = hpotgrad2dall_vec(
                        ifgrad=False, ifhess=False,
                        sources=self._get_sources(src_pslice), charge=src_weights[src_pslice],
                        targets=self._get_targets(tgt_pslice), zk=self.helmholtz_k)

                tgt_result += tmp_pot

            pot[tgt_pslice] = tgt_result

        return pot


    def multipole_to_local(self, starts, lists, mpole_exps):
        tree = self.tree
        local_exps = self.expansion_zeros()

        rscale = 1

        from pyfmmlib import h2dmploc_vec

        for itgt_box in xrange(self.tree.nboxes):
            start, end = starts[itgt_box:itgt_box+2]
            tgt_center = tree.box_centers[:, itgt_box]

            #print itgt_box, "<-", lists[start:end]
            tgt_loc = 0

            for isrc_box in lists[start:end]:
                src_center = tree.box_centers[:, isrc_box]

                tgt_loc = tgt_loc + h2dmploc_vec(
                        self.helmholtz_k,
                        rscale, src_center, mpole_exps[isrc_box],
                        rscale, tgt_center, self.nterms)[:, 0]

            local_exps[itgt_box] += tgt_loc

        return local_exps

    def eval_multipoles(self, leaf_boxes, sep_smaller_nonsiblings_starts,
            sep_smaller_nonsiblings_lists, mpole_exps):
        pot = self.potential_zeros()

        rscale = 1

        from pyfmmlib import h2dmpeval_vec
        for itgt_leaf, itgt_box in enumerate(leaf_boxes):
            tgt_pslice = self._get_target_slice(itgt_box)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tgt_pot = 0
            start, end = sep_smaller_nonsiblings_starts[itgt_leaf:itgt_leaf+2]
            for isrc_box in sep_smaller_nonsiblings_lists[start:end]:

                tmp_pot, _, _ = h2dmpeval_vec(self.helmholtz_k, rscale, self.
                        tree.box_centers[:, isrc_box], mpole_exps[isrc_box],
                        self._get_targets(tgt_pslice),
                        ifgrad=False, ifhess=False)

                tgt_pot = tgt_pot + tmp_pot

            pot[tgt_pslice] += tgt_pot

        return pot

    def refine_locals(self, start_box, end_box, local_exps):
        rscale = 1 # FIXME

        from pyfmmlib import h2dlocloc_vec

        for tgt_ibox in xrange(start_box, end_box):
            tgt_center = self.tree.box_centers[:, tgt_ibox]
            src_ibox = self.tree.box_parent_ids[tgt_ibox]
            src_center = self.tree.box_centers[:, src_ibox]

            tmp_loc_exp = h2dlocloc_vec(
                        self.helmholtz_k,
                        rscale, src_center, local_exps[src_ibox],
                        rscale, tgt_center, self.nterms)[:, 0]

            local_exps[tgt_ibox] += tmp_loc_exp

        return local_exps

    def eval_locals(self, leaf_boxes, local_exps):
        pot = self.potential_zeros()
        rscale = 1 # FIXME

        from pyfmmlib import h2dtaeval_vec

        for ibox in leaf_boxes:
            tgt_pslice = self._get_target_slice(ibox)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tmp_pot, _, _ = h2dtaeval_vec(self.helmholtz_k, rscale,
                    self.tree.box_centers[:, ibox], local_exps[ibox],
                    self._get_targets(tgt_pslice), ifgrad=False, ifhess=False)

            pot[tgt_pslice] += tmp_pot

        return pot
