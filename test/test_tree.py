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
from boxtree.tools import make_particle_array




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

# {{{ test basic tree build

def run_build_test(builder, queue, dims, dtype, nparticles, do_plot, max_particles_in_box=30, **kwargs):
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        tol = 1e-4
    elif dtype == np.float64:
        tol = 1e-12
    else:
        raise RuntimeError("unsupported dtype: %s" % dtype)

    print 75*"-"
    print "%dD %s - %d particles - max %d per box - %s" % (
            dims, dtype.type.__name__, nparticles, max_particles_in_box,
            " - ".join("%s: %s" % (k, v) for k, v in kwargs.iteritems()))
    print 75*"-"
    particles = make_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    queue.finish()
    print "building..."
    tree = builder(queue, particles,
            max_particles_in_box=max_particles_in_box, debug=True,
            **kwargs).get()
    print "%d boxes, testing..." % tree.nboxes

    sorted_particles = np.array(list(tree.sources))

    unsorted_particles = np.array([pi.get() for pi in particles])
    assert (sorted_particles
            == unsorted_particles[:, tree.user_source_ids]).all()

    all_good_so_far = True

    if do_plot:
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()

    from boxtree import box_flags_enum as bfe

    scaled_tol = tol*tree.root_extent
    for ibox in xrange(tree.nboxes):

        # Empty boxes exist in non-pruned trees--which themselves are undocumented.
        # These boxes will fail these tests.
        if not (tree.box_flags[ibox] & bfe.HAS_OWN_SRCNTGTS):
            continue

        extent_low, extent_high = tree.get_box_extent(ibox)
        if extent_low[0] == extent_low[1]:
            print "ZERO", ibox, tree.box_centers[:, ibox]
            1/0

        assert (extent_low >= tree.bounding_box[0] - scaled_tol).all(), (
                ibox, extent_low, tree.bounding_box[0])
        assert (extent_high <= tree.bounding_box[1] + scaled_tol).all(), (
                ibox, extent_high, tree.bounding_box[1])

        start = tree.box_source_starts[ibox]

        box_particles = sorted_particles[:,start:start+tree.box_source_counts[ibox]]
        good = (
                (box_particles < extent_high[:, np.newaxis] + scaled_tol)
                &
                (extent_low[:, np.newaxis] - scaled_tol <= box_particles)
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
            np.float64,
            np.float32,
            ]:
        for dims in [2, 3]:
            # test single-box corner case
            run_build_test(builder, queue, dims,
                    dtype, 4, do_plot=False)

            # test bi-level corner case
            run_build_test(builder, queue, dims,
                    dtype, 50, do_plot=False)

            # test unpruned tree build
            run_build_test(builder, queue, dims, dtype, 10**5,
                    do_plot=False, skip_prune=True)

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
            from boxtree.visualization import TreePlotter
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

# {{{ test sources-with-extent tree

@pytools.test.mark_test.opencl
def test_source_with_extent_tree(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [
            2,
            3
            ]:
        nsources = 3000
        ntargets = 30000
        dtype = np.float64

        sources = make_particle_array(queue, nsources, dims, dtype,
                seed=12)
        targets = make_particle_array(queue, ntargets, dims, dtype,
                seed=19)

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=13)
        source_radii = 2**rng.uniform(queue, nsources, dtype=dtype,
                a=-10, b=0)

        if do_plot:
            import matplotlib.pyplot as pt
            pt.plot(sources[0].get(), sources[1].get(), "rx")
            pt.plot(targets[0].get(), targets[1].get(), "g+")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, sources, targets=targets, source_radii=source_radii,
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
        if ntargets:
            assert (sorted_targets
                    == unsorted_targets[:, user_target_ids]).all()

        all_good_so_far = True

        if do_plot:
            from boxtree.visualization import TreePlotter
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





# You can test individual routines by typing
# $ python test_tree.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
