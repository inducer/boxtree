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

import pytools.test
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

    def form_multipoles(self, leaf_boxes, src_weights):
        mpoles = self.expansion_zeros()
        for ibox in leaf_boxes:
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

    def eval_direct(self, leaf_boxes, neighbor_leaves_starts, neighbor_leaves_lists,
            src_weights):
        pot = self.potential_zeros()

        for itgt_leaf, itgt_box in enumerate(leaf_boxes):
            tgt_pslice = self._get_target_slice(itgt_box)

            src_sum = 0
            start, end = neighbor_leaves_starts[itgt_leaf:itgt_leaf+2]
            for isrc_box in neighbor_leaves_lists[start:end]:
                src_pslice = self._get_source_slice(isrc_box)

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum

        return pot


    def multipole_to_local(self, starts, lists, mpole_exps):
        local_exps = self.expansion_zeros()

        for itgt_box in xrange(self.tree.nboxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            #print itgt_box, "<-", lists[start:end]
            for isrc_box in lists[start:end]:
                contrib += mpole_exps[isrc_box]

            local_exps[itgt_box] += contrib

        return local_exps

    def eval_multipoles(self, leaf_boxes, sep_smaller_nonsiblings_starts,
            sep_smaller_nonsiblings_lists, mpole_exps):
        pot = self.potential_zeros()

        for itgt_leaf, itgt_box in enumerate(leaf_boxes):
            tgt_pslice = self._get_target_slice(itgt_box)

            contrib = 0
            start, end = sep_smaller_nonsiblings_starts[itgt_leaf:itgt_leaf+2]
            for isrc_box in sep_smaller_nonsiblings_lists[start:end]:
                contrib += mpole_exps[isrc_box]

            pot[tgt_pslice] += contrib

        return pot

    def refine_locals(self, start_box, end_box, local_exps):
        for ibox in xrange(start_box, end_box):
            local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(self, leaf_boxes, local_exps):
        pot = self.potential_zeros()

        for ibox in leaf_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot




@pytools.test.mark_test.opencl
def test_fmm_completeness(ctx_getter):
    """Tests whether the built FMM traversal structures and driver completely
    capture all interactions.
    """

    import logging
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [
            2,
            3
            ]:
        for nsources, ntargets in [
                (10**6, None),
                (10**5, 3 * 10**5),
                ]:
            dtype = np.float64

            sources = make_particle_array(queue, nsources, dims, dtype, seed=15)

            if ntargets is None:
                # This says "same as sources" to the tree builder.
                targets = None
            else:
                targets = make_particle_array(
                        queue, ntargets, dims, dtype, seed=18)

            from boxtree import TreeBuilder
            tb = TreeBuilder(ctx)

            tree = tb(queue, sources, targets=targets,
                    max_particles_in_box=30, debug=True)

            from boxtree.traversal import FMMTraversalBuilder
            tg = FMMTraversalBuilder(ctx)
            trav = tg(queue, tree).get()

            weights = np.random.randn(nsources)
            #weights = np.ones(nparticles)
            weights_sum = np.sum(weights)

            from boxtree.fmm import drive_fmm
            wrangler = ConstantOneExpansionWrangler(trav.tree)

            if ntargets is None:
                # This check only works for targets == sources.
                assert (wrangler.reorder_potentials(
                        wrangler.reorder_src_weights(weights)) == weights).all()

            pot = drive_fmm(trav, wrangler, weights)

            # {{{ build, evaluate matrix (and identify missing interactions)

            if 0:
                mat = np.zeros((ntargets, nsources), dtype)
                from pytools import ProgressBar
                pb = ProgressBar("matrix", nsources)
                for i in xrange(nsources):
                    unit_vec = np.zeros(nsources, dtype=dtype)
                    unit_vec[i] = 1
                    mat[:,i] = drive_fmm(trav, wrangler, unit_vec)
                    pb.progress()
                pb.finished()

                missing_tgts, missing_srcs = np.where(mat == 0)

                if len(missing_tgts):
                    import matplotlib.pyplot as pt

                    from boxtree.visualization import TreePlotter
                    plotter = TreePlotter(tree)
                    plotter.draw_tree(fill=False, edgecolor="black")
                    plotter.draw_box_numbers()
                    plotter.set_bounding_box()

                    for tgt, src in zip(missing_tgts, missing_srcs):
                        pt.plot(
                                trav.tree.particles[0][tgt],
                                trav.tree.particles[1][tgt],
                                "ro")
                        pt.plot(
                                trav.tree.particles[0][src],
                                trav.tree.particles[1][src],
                                "go")

                    pt.show()

                #pt.spy(mat)
                #pt.show()

            # }}}

            assert la.norm((pot - weights_sum) / nsources) < 1e-8

# }}}

# {{{ test Helmholtz fmm with pyfmmlib

@pytools.test.mark_test.opencl
def test_pyfmmlib_fmm(ctx_getter):
    import logging
    logging.basicConfig(level=logging.INFO)

    from pytest import importorskip
    importorskip("pyfmmlib")

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nsources = 10**3
    ntargets = 10**3
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

    tree = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    trav = tg(queue, tree).get()

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

