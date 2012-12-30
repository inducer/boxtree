from __future__ import division

import numpy as np
import numpy.linalg as la
import sys
import pytools.test

import matplotlib.pyplot as pt

import pyopencl as cl
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




# {{{ basic tree build test

@pytools.test.mark_test.opencl
def test_tree(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    #for dims in [2, 3]:
    for dims in [2]:
        nparticles = 10**5
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        if do_plot:
            pt.plot(particles[0].get(), particles[1].get(), "x")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "%d boxes, testing..." % tree.nboxes

        starts = tree.box_particle_starts.get()
        pcounts = tree.box_particle_counts.get()
        sorted_particles = np.array([pi.get() for pi in tree.particles])
        centers = tree.box_centers.get()
        levels = tree.box_levels.get()

        unsorted_particles = np.array([pi.get() for pi in particles])
        assert (sorted_particles
                == unsorted_particles[:, tree.original_particle_ids.get()]).all()

        assert np.max(levels) + 1 == tree.nlevels

        root_extent = tree.root_extent

        all_good_so_far = True

        if do_plot:
            tree.plot(zorder=10)

        for ibox in xrange(tree.nboxes):
            lev = int(levels[ibox])
            box_size = root_extent / (1 << lev)
            el = extent_low = centers[:, ibox] - 0.5*box_size
            eh = extent_high = extent_low + box_size

            box_particle_nrs = np.arange(starts[ibox], starts[ibox]+pcounts[ibox],
                    dtype=np.intp)

            box_particles = sorted_particles[:,box_particle_nrs]
            good = (
                    (box_particles < extent_high[:, np.newaxis])
                    &
                    (extent_low[:, np.newaxis] <= box_particles)
                    )

            all_good_here = good.all()
            if do_plot and not all_good_here and all_good_so_far:
                pt.plot(
                        box_particles[0, np.where(~good)[1]],
                        box_particles[1, np.where(~good)[1]], "ro")

                pt.plot([el[0], eh[0], eh[0], el[0], el[0]],
                        [el[1], el[1], eh[1], eh[1], el[1]], "r-", lw=1)

            all_good_so_far = all_good_so_far and all_good_here

        if do_plot:
            pt.gca().set_aspect("equal", "datalim")
            pt.show()

        assert all_good_so_far

        print "done"

# }}}

# {{{ connectivity test

@pytools.test.mark_test.opencl
def test_tree_connectivity(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [2]:
        nparticles = 10**5
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "tree built"

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        trav = tg(queue, tree).get()

        print "traversal built"

        levels = tree.box_levels.get()
        parents = tree.box_parent_ids.get().T
        children = tree.box_child_ids.get().T
        centers = tree.box_centers.get().T

        # {{{ parent and child relations, levels match up

        for ibox in xrange(1, tree.nboxes):
            # /!\ Not testing box 0, has no parents
            parent = parents[ibox]

            assert levels[parent] + 1 == levels[ibox]
            assert ibox in children[parent], ibox

        # }}}

        if 0:
            from boxtree import TreePlotter
            plotter = TreePlotter(tree)
            plotter.draw_tree(fill=False, edgecolor="black")
            plotter.draw_box_numbers()
            plotter.set_bounding_box()
            pt.show()

        # {{{ neighbor_leaves (list 1) consists of leaves

        for ileaf, ibox in enumerate(trav.leaf_boxes):
            start, end = trav.neighbor_leaves_starts[ileaf:ileaf+2]
            nbl = trav.neighbor_leaves_lists[start:end]
            assert ibox in nbl
            for jbox in nbl:
                assert (0 == children[jbox]).all()

        print "list 1 tested"

        # }}}

        # {{{ separated siblings (list 2) are actually separated

        for ibox in xrange(tree.nboxes):
            start, end = trav.sep_siblings_starts[ibox:ibox+2]
            seps = trav.sep_siblings_lists[start:end]

            assert (levels[seps] == levels[ibox]).all()

            # three-ish box radii (half of size)
            mindist = 2.5 * 0.5 * 2**-int(levels[ibox]) * tree.root_extent

            icenter = centers[ibox]
            for jbox in seps:
                dist = la.norm(centers[jbox]-icenter)
                assert dist > mindist, (dist, mindist)

        # }}}

        # {{{ sep_{smaller,bigger}_nonsiblings are duals of each other

        # (technically, we only test one half of that)

        for ileaf, ibox in enumerate(trav.leaf_boxes):
            start, end = trav.sep_smaller_nonsiblings_starts[ileaf:ileaf+2]

            for jbox in trav.sep_smaller_nonsiblings_lists[start:end]:
                rstart, rend = trav.sep_bigger_nonsiblings_starts[jbox:jbox+2]

                assert ibox in trav.sep_bigger_nonsiblings_lists[rstart:rend], (ibox, jbox)

        print "list 3, 4 are duals"

        # }}}

        # {{{ sep_smaller_nonsiblings satisfies size assumption

        for ileaf, ibox in enumerate(trav.leaf_boxes):
            start, end = trav.sep_smaller_nonsiblings_starts[ileaf:ileaf+2]

            for jbox in trav.sep_smaller_nonsiblings_lists[start:end]:
                assert levels[ibox] < levels[jbox]

        print "list 3 satisfies size assumption"

        # }}}

        # {{{ sep_smaller_nonsiblings satisfies size assumption

        for  ibox in xrange(tree.nboxes):
            start, end = trav.sep_bigger_nonsiblings_starts[ibox:ibox+2]

            for jbox in trav.sep_bigger_nonsiblings_lists[start:end]:
                assert levels[ibox] > levels[jbox]

        print "list 4 satisfies size assumption"

        # }}}


# }}}

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

    def _get_particle_slice(self, ibox):
        pstart = self.tree.box_particle_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_particle_counts[ibox])

    def reorder_src_weights(self, src_weights):
        return src_weights[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def form_multipoles(self, leaf_boxes, src_weights):
        mpoles = self.expansion_zeros()
        for ibox in leaf_boxes:
            pslice = self._get_particle_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    def coarsen_multipoles(self, branch_boxes, start_branch_box, end_branch_box,
            mpoles):
        tree = self.tree

        for ibox in branch_boxes[start_branch_box:end_branch_box]:
            for child in tree.box_child_ids[:, ibox]:
                if child:
                    mpoles[ibox] += mpoles[child]

    def do_direct_eval(self, leaf_boxes, neighbor_leaves_starts, neighbor_leaves_lists,
            src_weights):
        pot = self.potential_zeros()

        for itgt_leaf, itgt_box in enumerate(leaf_boxes):
            tgt_pslice = self._get_particle_slice(itgt_box)

            src_sum = 0
            start, end = neighbor_leaves_starts[itgt_leaf:itgt_leaf+2]
            for isrc_box in neighbor_leaves_lists[start:end]:
                src_pslice = self._get_particle_slice(isrc_box)

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
            tgt_pslice = self._get_particle_slice(itgt_box)

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
            tgt_pslice = self._get_particle_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot




@pytools.test.mark_test.opencl
def test_fmm_completeness(ctx_getter, do_plot=False):
    """Tests whether the built FMM traversal structures and driver completely
    capture all interactions.
    """

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [2]:
        nparticles = 10**6
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        if do_plot:
            pt.plot(particles[0].get(), particles[1].get(), "x")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        tree = tb(queue, particles, max_particles_in_box=30, debug=True)

        print "tree built"

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        trav = tg(queue, tree).get()

        print "traversal built"

        weights = np.random.randn(nparticles)
        #weights = np.ones(nparticles)
        weights_sum = np.sum(weights)

        from boxtree.fmm import  drive_fmm
        wrangler = ConstantOneExpansionWrangler(trav.tree)

        assert (wrangler.reorder_potentials(
                wrangler.reorder_src_weights(weights)) == weights).all()

        pot = drive_fmm(trav, wrangler, weights)

        # {{{ build, evaluate matrix (and identify missing interactions)

        if 0:
            mat = np.zeros((nparticles, nparticles), dtype)
            from pytools import ProgressBar
            pb = ProgressBar("matrix", nparticles)
            for i in xrange(nparticles):
                unit_vec = np.zeros(nparticles, dtype=dtype)
                unit_vec[i] = 1
                mat[:,i] = drive_fmm(trav, wrangler, unit_vec)
                pb.progress()
            pb.finished()

            missing_tgts, missing_srcs = np.where(mat == 0)

            if len(missing_tgts):
                import matplotlib.pyplot as pt

                from boxtree import TreePlotter
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

        assert la.norm((pot - weights_sum) / nparticles) < 1e-8

# }}}

# {{{ visualization helper (not a test)

def plot_traversal(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    #for dims in [2, 3]:
    for dims in [2]:
        nparticles = 10**5
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        #if do_plot:
            #pt.plot(particles[0].get(), particles[1].get(), "x")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "done"

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        trav = tg(queue, tree).get()

        from boxtree import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black")
        #plotter.draw_box_numbers()
        plotter.set_bounding_box()

        from random import randrange, seed
        seed(7)

        # {{{ generic box drawing helper

        def draw_some_box_lists(starts, lists, key_to_box=None,
                count=5):
            actual_count = 0
            while actual_count < count:
                if key_to_box is not None:
                    key = randrange(len(key_to_box))
                    ibox = key_to_box[key]
                else:
                    key = ibox = randrange(tree.nboxes)

                start, end = starts[key:key+2]
                if start == end:
                    continue

                plotter.draw_box(ibox, facecolor='red')

                #print ibox, start, end, lists[start:end]
                for jbox in lists[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')

                actual_count += 1

        # }}}

        if 0:
            # colleagues
            draw_some_box_lists(
                    trav.colleagues_starts,
                    trav.colleagues_lists)
        elif 0:
            # near neighbors ("list 1")
            draw_some_box_lists(
                    trav.neighbor_leaves_starts,
                    trav.neighbor_leaves_lists,
                    key_to_box=trav.leaf_boxes)
        elif 0:
            # well-separated siblings (list 2)
            draw_some_box_lists(
                    trav.sep_siblings_starts,
                    trav.sep_siblings_lists)
        elif 0:
            # separated smaller non-siblings (list 3)
            draw_some_box_lists(
                    trav.sep_smaller_nonsiblings_starts,
                    trav.sep_smaller_nonsiblings_lists,
                    key_to_box=trav.leaf_boxes)
        elif 1:
            # separated bigger non-siblings (list 4)
            draw_some_box_lists(
                    trav.sep_bigger_nonsiblings_starts,
                    trav.sep_bigger_nonsiblings_lists)

        pt.show()









# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
