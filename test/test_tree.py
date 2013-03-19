from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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
import sys
import pytools.test

import pyopencl as cl
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def make_particle_array(queue, nparticles, dims, dtype, seed=15):
    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=seed)

    from pytools.obj_array import make_obj_array
    return make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])


# {{{ bounding box test

def test_bounding_box(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from boxtree import AXIS_NAMES
    from boxtree.bounding_box import BoundingBoxFinder

    bbf = BoundingBoxFinder(ctx)

    #for dtype in [np.float32, np.float64]:
    for dtype in [np.float64, np.float32]:
        for dims in [2, 3]:
            axis_names = AXIS_NAMES[:dims]

            for nparticles in [9, 4096, 10**5]:
                print dtype, dims, nparticles
                particles = make_particle_array(queue, nparticles, dims, dtype)

                bbox_min = [np.min(x.get()) for x in particles]
                bbox_max = [np.max(x.get()) for x in particles]

                bbox_cl = bbf(particles).get()

                bbox_min_cl = np.empty(dims, dtype)
                bbox_max_cl = np.empty(dims, dtype)

                for i, ax in enumerate(axis_names):
                    bbox_min_cl[i] = bbox_cl["min_"+ax]
                    bbox_max_cl[i] = bbox_cl["max_"+ax]

                assert (bbox_min == bbox_min_cl).all()
                assert (bbox_max == bbox_max_cl).all()

# }}}

# {{{ basic tree build test

def run_build_test(builder, queue, dims, dtype, nparticles, do_plot, max_particles_in_box=30, nboxes_guess=None):
    dtype = np.dtype(dtype)

    print 75*"-"
    print "%dD %s - %d particles - max %d per box - box count guess: %s" % (
            dims, dtype.type.__name__, nparticles, max_particles_in_box, nboxes_guess)
    print 75*"-"
    particles = make_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    queue.finish()
    print "building..."
    tree = builder(queue, particles,
            max_particles_in_box=max_particles_in_box, debug=True,
            nboxes_guess=nboxes_guess).get()
    print "%d boxes, testing..." % tree.nboxes

    sorted_particles = np.array(list(tree.sources))

    unsorted_particles = np.array([pi.get() for pi in particles])
    assert (sorted_particles
            == unsorted_particles[:, tree.user_source_ids]).all()

    all_good_so_far = True

    if do_plot:
        from boxtree import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()

    for ibox in xrange(tree.nboxes):
        extent_low, extent_high = tree.get_box_extent(ibox)

        assert (extent_low >= tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
        assert (extent_high <= tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

        start = tree.box_source_starts[ibox]

        box_particles = sorted_particles[:,start:start+tree.box_source_counts[ibox]]
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

            plotter.draw_box(ibox, edgecolor="red")

        if not all_good_here:
            print "BAD BOX", ibox

        all_good_so_far = all_good_so_far and all_good_here

    if do_plot:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

    assert all_good_so_far

    print "done"
@pytools.test.mark_test.opencl
def test_particle_tree(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    for dtype in [
            #np.float64,
            np.float32,
            ]:
        for dims in [2, 3]:
            # test single-box corner case
            run_build_test(builder, queue, dims,
                    dtype, 4, do_plot=False)

            # test bi-level corner case
            run_build_test(builder, queue, dims,
                    dtype, 50, do_plot=False)

            # exercise reallocation code
            run_build_test(builder, queue, dims, dtype, 10**5,
                    do_plot=False, nboxes_guess=5)

            # test many empty leaves corner case
            run_build_test(builder, queue, dims, dtype, 10**5,
                    do_plot=False, max_particles_in_box=5)

            # test vanilla tree build
            run_build_test(builder, queue, dims, dtype, 10**5,
                    do_plot=do_plot)



@pytools.test.mark_test.opencl
def test_source_target_tree(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [2, 3]:
        nsources = 2 * 10**5
        ntargets = 3 * 10**5
        dtype = np.float64

        sources = make_particle_array(queue, nsources, dims, dtype,
                seed=12)
        targets = make_particle_array(queue, ntargets, dims, dtype,
                seed=19)

        if do_plot:
            import matplotlib.pyplot as pt
            pt.plot(sources[0].get(), sources[1].get(), "rx")
            pt.plot(targets[0].get(), targets[1].get(), "g+")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, sources, targets=targets,
                max_particles_in_box=10, debug=True).get()
        print "%d boxes, testing..." % tree.nboxes

        sorted_sources = np.array(list(tree.sources))
        sorted_targets = np.array(list(tree.targets))

        unsorted_sources = np.array([pi.get() for pi in sources])
        unsorted_targets = np.array([pi.get() for pi in targets])
        assert (sorted_sources
                == unsorted_sources[:, tree.user_source_ids]).all()

        user_target_ids = np.empty(tree.ntargets, dtype=np.intp)
        user_target_ids[tree.sorted_target_ids] = np.arange(tree.ntargets, dtype=np.intp)
        assert (sorted_targets
                == unsorted_targets[:, user_target_ids]).all()

        all_good_so_far = True

        if do_plot:
            from boxtree import TreePlotter
            plotter = TreePlotter(tree)
            plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
            plotter.set_bounding_box()

        for ibox in xrange(tree.nboxes):
            extent_low, extent_high = tree.get_box_extent(ibox)

            assert (extent_low >= tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
            assert (extent_high <= tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

            src_start = tree.box_source_starts[ibox]
            tgt_start = tree.box_target_starts[ibox]

            for what, particles in [
                    ("sources", sorted_sources[:,src_start:src_start+tree.box_source_counts[ibox]]),
                    ("targets", sorted_targets[:,tgt_start:tgt_start+tree.box_target_counts[ibox]]),
                    ]:
                good = (
                        (particles < extent_high[:, np.newaxis])
                        &
                        (extent_low[:, np.newaxis] <= particles)
                        ).all(axis=0)

                all_good_here = good.all()
                if do_plot and not all_good_here:
                    pt.plot(
                            particles[0, np.where(~good)[0]],
                            particles[1, np.where(~good)[0]], "ro")

                    plotter.draw_box(ibox, edgecolor="red")
                    pt.show()

            if not all_good_here:
                print "BAD BOX %s %d" % (what, ibox)

            all_good_so_far = all_good_so_far and all_good_here

        if do_plot:
            pt.gca().set_aspect("equal", "datalim")
            pt.show()

        assert all_good_so_far

        print "done"


# }}}

# {{{ connectivity test

@pytools.test.mark_test.opencl
def test_tree_connectivity(ctx_getter):
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
            import matplotlib.pyplot as pt
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

# {{{ geometry query test

def test_geometry_query(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dims = 2
    nparticles = 10**5
    dtype = np.float64

    particles = make_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    print "building..."
    tree = tb(queue, particles, max_particles_in_box=30, debug=True)
    print "%d boxes, testing..." % tree.nboxes

    nballs = 10**4
    ball_centers = make_particle_array(queue, nballs, dims, dtype)
    ball_radii = cl.array.empty(queue, nballs, dtype).fill(0.1)

    from boxtree.geo_lookup import LeavesToBallsLookupBuilder
    lblb = LeavesToBallsLookupBuilder(ctx)

    lbl = lblb(queue, tree, ball_centers, ball_radii)

    # get data to host for test
    tree = tree.get()
    lbl = lbl.get()
    ball_centers = np.array([x.get() for x in ball_centers]).T
    ball_radii = ball_radii.get()

    from boxtree import box_flags_enum

    for ibox in xrange(tree.nboxes):
        # We only want leaves here.
        if tree.box_flags[ibox] & box_flags_enum.HAS_CHILDREN:
            continue

        box_center = tree.box_centers[:, ibox]
        ext_l, ext_h = tree.get_box_extent(ibox)
        box_rad = 0.5*(ext_h-ext_l)[0]

        linf_circle_dists = np.max(np.abs(ball_centers-box_center), axis=-1)
        near_circles, = np.where(linf_circle_dists - ball_radii < box_rad)

        start, end = lbl.balls_near_box_starts[ibox:ibox+2]
        #print sorted(lbl.balls_near_box_lists[start:end])
        #print sorted(near_circles)
        assert sorted(lbl.balls_near_box_lists[start:end]) == sorted(near_circles)

# }}}

# {{{ visualization helper (not a test)

def plot_traversal(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    #for dims in [2, 3]:
    for dims in [2]:
        nparticles = 10**4
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

                #print ibox, start, end, lists[start:end]
                for jbox in lists[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')

                plotter.draw_box(ibox, facecolor='red')

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
        elif 1:
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

        import matplotlib.pyplot as pt
        pt.show()

# }}}




# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
