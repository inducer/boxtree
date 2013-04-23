from __future__ import division

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
import numpy.linalg as la
import pyopencl as cl

import pytest
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from boxtree.tools import make_particle_array, particle_array_to_host

import logging
logger = logging.getLogger(__name__)




# {{{ fmm interaction completeness test

class ConstantOneExpansionWrangler:
    """This implements the 'analytical routines' for a Green's function that is
    constant 1 everywhere. For 'charges' of 'ones', this should get every particle
    a copy of the particle count.
    """

    def __init__(self, tree):
        self.tree = tree

    def expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    def potential_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_target_counts[ibox])

    def reorder_src_weights(self, src_weights):
        return src_weights[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def form_multipoles(self, source_boxes, src_weights):
        mpoles = self.expansion_zeros()
        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    def coarsen_multipoles(self, parent_boxes, start_parent_box, end_parent_box,
            mpoles):
        tree = self.tree

        for ibox in parent_boxes[start_parent_box:end_parent_box]:
            for child in tree.box_child_ids[:, ibox]:
                if child:
                    mpoles[ibox] += mpoles[child]

    def eval_direct(self, target_boxes, neighbor_sources_starts, neighbor_sources_lists,
            src_weights):
        pot = self.potential_zeros()

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            #print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum

        return pot


    def multipole_to_local(self, target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            #print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]

            local_exps[tgt_ibox] += contrib

        return local_exps

    def eval_multipoles(self, target_boxes, sep_smaller_nonsiblings_starts,
            sep_smaller_nonsiblings_lists, mpole_exps):
        pot = self.potential_zeros()

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            contrib = 0
            start, end = sep_smaller_nonsiblings_starts[itgt_box:itgt_box+2]
            for src_ibox in sep_smaller_nonsiblings_lists[start:end]:
                contrib += mpole_exps[src_ibox]

            pot[tgt_pslice] += contrib

        return pot

    def form_locals(self, target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            #print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)

                contrib += np.sum(src_weights[src_pslice])

            local_exps[tgt_ibox] += contrib

        return local_exps

    def refine_locals(self, child_boxes, start_child_box, end_child_box, local_exps):
        for ibox in child_boxes[start_child_box:end_child_box]:
            local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(self, target_boxes, local_exps):
        pot = self.potential_zeros()

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot




@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize(("nsources", "ntargets"), [
    (10**6, None),
    (5 * 10**5, 4*10**5),
    ])
def test_fmm_completeness(ctx_getter, dims, nsources, ntargets):
    """Tests whether the built FMM traversal structures and driver completely
    capture all interactions.
    """

    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dtype = np.float64

    sources = make_particle_array(queue, nsources, dims, dtype, seed=15)

    if ntargets is None:
        # This says "same as sources" to the tree builder.
        targets = None
    else:
        targets = make_particle_array(
                queue, ntargets, dims, dtype, seed=16)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)
    trav = trav.get()
    tree = trav.tree

    #weights = np.random.randn(nsources)
    weights = np.ones(nsources)
    weights_sum = np.sum(weights)

    from boxtree.fmm import drive_fmm
    wrangler = ConstantOneExpansionWrangler(tree)

    if ntargets is None:
        # This check only works for targets == sources.
        assert (wrangler.reorder_potentials(
                wrangler.reorder_src_weights(weights)) == weights).all()

    pot = drive_fmm(trav, wrangler, weights)

    # {{{ build, evaluate matrix (and identify missing interactions)

    if 0:
        mat = np.zeros((ntargets, nsources), dtype)
        from pytools import ProgressBar

        logging.getLogger().setLevel(logging.WARNING)

        pb = ProgressBar("matrix", nsources)
        for i in xrange(nsources):
            unit_vec = np.zeros(nsources, dtype=dtype)
            unit_vec[i] = 1
            mat[:,i] = drive_fmm(trav, wrangler, unit_vec)
            pb.progress()
        pb.finished()

        logging.getLogger().setLevel(logging.INFO)

        missing_tgts, missing_srcs = np.where(mat == 0)

        if 0 and len(missing_tgts):
            import matplotlib.pyplot as pt

            from boxtree.visualization import TreePlotter
            plotter = TreePlotter(tree)
            plotter.draw_tree(fill=False, edgecolor="black")
            plotter.draw_box_numbers()
            plotter.set_bounding_box()

            tree_order_missing_tgts = \
                    tree.indices_to_tree_target_order(missing_tgts)
            tree_order_missing_srcs = \
                    tree.indices_to_tree_source_order(missing_srcs)

            src_boxes = [
                    tree.find_box_nr_for_source(i)
                    for i in tree_order_missing_srcs]
            tgt_boxes = [
                    tree.find_box_nr_for_target(i)
                    for i in tree_order_missing_tgts]
            print zip(src_boxes, tgt_boxes)

            pt.plot(
                    tree.targets[0][tree_order_missing_tgts],
                    tree.targets[1][tree_order_missing_tgts],
                    "rv")
            pt.plot(
                    tree.sources[0][tree_order_missing_srcs],
                    tree.sources[1][tree_order_missing_srcs],
                    "go")

            pt.show()

            pt.spy(mat)
            pt.show()

    # }}}

    rel_err = la.norm((pot - weights_sum) / nsources)
    good = rel_err < 1e-8
    if 1 and not good:
        import matplotlib.pyplot as pt
        pt.plot(pot-weights_sum)
        pt.show()

    assert good

# }}}

# {{{ test Helmholtz fmm with pyfmmlib

@pytest.mark.opencl
def test_pyfmmlib_fmm(ctx_getter):
    logging.basicConfig(level=logging.INFO)

    from pytest import importorskip
    importorskip("pyfmmlib")

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nsources = 3000
    ntargets = 1000
    dims = 2
    dtype = np.float64

    helmholtz_k = 2

    sources = make_particle_array(queue, nsources, dims, dtype, seed=15)
    targets = (
            make_particle_array(queue, ntargets, dims, dtype, seed=18)
            + np.array([2, 0]))

    sources_host = particle_array_to_host(sources)
    targets_host = particle_array_to_host(targets)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    trav = trav.get()

    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=20)

    weights = rng.uniform(queue, nsources, dtype=np.float64).get()
    #weights = np.ones(nsources)

    logger.info("computing direct (reference) result")

    from pyfmmlib import hpotgrad2dall_vec
    ref_pot, _, _ = hpotgrad2dall_vec(ifgrad=False, ifhess=False,
            sources=sources_host.T, charge=weights,
            targets=targets_host.T, zk=helmholtz_k)

    from boxtree.pyfmmlib_integration import Helmholtz2DExpansionWrangler
    wrangler = Helmholtz2DExpansionWrangler(trav.tree, helmholtz_k, nterms=10)

    from boxtree.fmm import drive_fmm
    pot = drive_fmm(trav, wrangler, weights)

    rel_err = la.norm(pot - ref_pot) / la.norm(ref_pot)
    logger.info("relative l2 error: %g" % rel_err)
    assert rel_err < 1e-5

# }}}




# You can test individual routines by typing
# $ python test_fmm.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker

