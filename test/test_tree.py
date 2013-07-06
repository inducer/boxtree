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
import sys
import pytest
import logging

import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from boxtree.tools import make_normal_particle_array

logger = logging.getLogger(__name__)


# {{{ bounding box test

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("nparticles", [9, 4096, 10**5])
def test_bounding_box(ctx_getter, dtype, dims, nparticles):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from boxtree.tools import AXIS_NAMES
    from boxtree.bounding_box import BoundingBoxFinder

    bbf = BoundingBoxFinder(ctx)

    axis_names = AXIS_NAMES[:dims]

    logger.info("%s - %s %s" % (dtype, dims, nparticles))

    particles = make_normal_particle_array(queue, nparticles, dims, dtype)

    bbox_min = [np.min(x.get()) for x in particles]
    bbox_max = [np.max(x.get()) for x in particles]

    bbox_cl, evt = bbf(particles, radii=None)
    bbox_cl = bbox_cl.get()

    bbox_min_cl = np.empty(dims, dtype)
    bbox_max_cl = np.empty(dims, dtype)

    for i, ax in enumerate(axis_names):
        bbox_min_cl[i] = bbox_cl["min_"+ax]
        bbox_max_cl[i] = bbox_cl["max_"+ax]

    assert (bbox_min == bbox_min_cl).all()
    assert (bbox_max == bbox_max_cl).all()

# }}}


# {{{ test basic (no source/target distinction) tree build

def run_build_test(builder, queue, dims, dtype, nparticles, do_plot,
        max_particles_in_box=30, **kwargs):
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        tol = 1e-4
    elif dtype == np.float64:
        tol = 1e-12
    else:
        raise RuntimeError("unsupported dtype: %s" % dtype)

    logger.info(75*"-")
    logger.info("%dD %s - %d particles - max %d per box - %s" % (
            dims, dtype.type.__name__, nparticles, max_particles_in_box,
            " - ".join("%s: %s" % (k, v) for k, v in kwargs.iteritems())))
    logger.info(75*"-")

    particles = make_normal_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    queue.finish()

    tree, _ = builder(queue, particles,
            max_particles_in_box=max_particles_in_box, debug=True,
            **kwargs)
    tree = tree.get()

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

        assert (extent_low >= tree.bounding_box[0] - scaled_tol).all(), (
                ibox, extent_low, tree.bounding_box[0])
        assert (extent_high <= tree.bounding_box[1] + scaled_tol).all(), (
                ibox, extent_high, tree.bounding_box[1])

        start = tree.box_source_starts[ibox]

        box_children = tree.box_child_ids[:, ibox]
        existing_children = box_children[box_children != 0]

        assert (tree.box_source_counts_nonchild[ibox]
                + np.sum(tree.box_source_counts_cumul[existing_children])
                == tree.box_source_counts_cumul[ibox])

        box_particles = sorted_particles[:,
                start:start+tree.box_source_counts_cumul[ibox]]
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


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("dims", [2, 3])
def test_particle_tree(ctx_getter, dtype, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

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


@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_source_target_tree(ctx_getter, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nsources = 2 * 10**5
    ntargets = 3 * 10**5
    dtype = np.float64

    sources = make_normal_particle_array(queue, nsources, dims, dtype,
            seed=12)
    targets = make_normal_particle_array(queue, ntargets, dims, dtype,
            seed=19)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(sources[0].get(), sources[1].get(), "rx")
        pt.plot(targets[0].get(), targets[1].get(), "g+")

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=10, debug=True)
    tree = tree.get()

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

        assert (extent_low >=
                tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
        assert (extent_high <=
                tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

        src_start = tree.box_source_starts[ibox]
        tgt_start = tree.box_target_starts[ibox]

        box_children = tree.box_child_ids[:, ibox]
        existing_children = box_children[box_children != 0]

        assert (tree.box_source_counts_nonchild[ibox]
                + np.sum(tree.box_source_counts_cumul[existing_children])
                == tree.box_source_counts_cumul[ibox])
        assert (tree.box_target_counts_nonchild[ibox]
                + np.sum(tree.box_target_counts_cumul[existing_children])
                == tree.box_target_counts_cumul[ibox])

        for what, particles in [
                ("sources", sorted_sources[:,
                    src_start:src_start+tree.box_source_counts_cumul[ibox]]),
                ("targets", sorted_targets[:,
                    tgt_start:tgt_start+tree.box_target_counts_cumul[ibox]]),
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
        assert all_good_so_far

    if do_plot:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

# }}}


# {{{ test sources/targets-with-extent tree

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_extent_tree(ctx_getter, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nsources = 100000
    ntargets = 200000
    dtype = np.float64
    npoint_sources_per_source = 16

    sources = make_normal_particle_array(queue, nsources, dims, dtype,
            seed=12)
    targets = make_normal_particle_array(queue, ntargets, dims, dtype,
            seed=19)

    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=13)
    source_radii = 2**rng.uniform(queue, nsources, dtype=dtype,
            a=-10, b=0)
    target_radii = 2**rng.uniform(queue, ntargets, dtype=dtype,
            a=-10, b=0)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    dev_tree, _ = tb(queue, sources, targets=targets,
            source_radii=source_radii, target_radii=target_radii,
            max_particles_in_box=10, debug=True)

    logger.info("transfer tree, check orderings")

    tree = dev_tree.get()

    sorted_sources = np.array(list(tree.sources))
    sorted_targets = np.array(list(tree.targets))
    sorted_source_radii = tree.source_radii
    sorted_target_radii = tree.target_radii

    unsorted_sources = np.array([pi.get() for pi in sources])
    unsorted_targets = np.array([pi.get() for pi in targets])
    unsorted_source_radii = source_radii.get()
    unsorted_target_radii = target_radii.get()
    assert (sorted_sources
            == unsorted_sources[:, tree.user_source_ids]).all()
    assert (sorted_source_radii
            == unsorted_source_radii[tree.user_source_ids]).all()

    # {{{ test box structure, stick-out criterion

    logger.info("test box structure, stick-out criterion")

    user_target_ids = np.empty(tree.ntargets, dtype=np.intp)
    user_target_ids[tree.sorted_target_ids] = np.arange(tree.ntargets, dtype=np.intp)
    if ntargets:
        assert (sorted_targets
                == unsorted_targets[:, user_target_ids]).all()
        assert (sorted_target_radii
                == unsorted_target_radii[user_target_ids]).all()

    all_good_so_far = True

    # {{{ check sources, targets

    for ibox in xrange(tree.nboxes):
        extent_low, extent_high = tree.get_box_extent(ibox)

        box_radius = np.max(extent_high-extent_low) * 0.5
        stick_out_dist = tree.stick_out_factor * box_radius

        assert (extent_low >=
                tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
        assert (extent_high <=
                tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

        box_children = tree.box_child_ids[:, ibox]
        existing_children = box_children[box_children != 0]

        assert (tree.box_source_counts_nonchild[ibox]
                + np.sum(tree.box_source_counts_cumul[existing_children])
                == tree.box_source_counts_cumul[ibox])
        assert (tree.box_target_counts_nonchild[ibox]
                + np.sum(tree.box_target_counts_cumul[existing_children])
                == tree.box_target_counts_cumul[ibox])

        for what, starts, counts, points, radii in [
                ("source", tree.box_source_starts, tree.box_source_counts_cumul,
                    sorted_sources, sorted_source_radii),
                ("target", tree.box_target_starts, tree.box_target_counts_cumul,
                    sorted_targets, sorted_target_radii),
                ]:
            bstart = starts[ibox]
            bslice = slice(bstart, bstart+counts[ibox])
            check_particles = points[:, bslice]
            check_radii = radii[bslice]

            good = (
                    (check_particles + check_radii
                        < extent_high[:, np.newaxis] + stick_out_dist)
                    &
                    (extent_low[:, np.newaxis] - stick_out_dist
                        <= check_particles - check_radii)
                    ).all(axis=0)

            all_good_here = good.all()

            if not all_good_here:
                print "BAD BOX %s %d level %d" % (what, ibox, tree.box_levels[ibox])

            all_good_so_far = all_good_so_far and all_good_here
            assert all_good_here

    # }}}

    assert all_good_so_far

    # }}}

    # {{{ create, link point sources

    logger.info("creating point sources")

    np.random.seed(20)

    from pytools.obj_array import make_obj_array
    point_sources = make_obj_array([
            cl.array.to_device(queue,
                unsorted_sources[i][:, np.newaxis]
                + unsorted_source_radii[:, np.newaxis]
                * np.random.uniform(
                    -1, 1, size=(nsources, npoint_sources_per_source))
                 )
            for i in range(dims)])

    point_source_starts = cl.array.arange(queue,
            0, (nsources+1)*npoint_sources_per_source, npoint_sources_per_source,
            dtype=tree.particle_id_dtype)

    from boxtree.tree import TreeWithLinkedPointSources
    dev_tree = TreeWithLinkedPointSources(queue, dev_tree,
            point_source_starts, point_sources,
            debug=True)

    # }}}

# }}}


# {{{ geometry query test

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_geometry_query(ctx_getter, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nparticles = 10**5
    dtype = np.float64

    particles = make_normal_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    tree, _ = tb(queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    ball_centers = make_normal_particle_array(queue, nballs, dims, dtype)
    ball_radii = cl.array.empty(queue, nballs, dtype).fill(0.1)

    from boxtree.geo_lookup import LeavesToBallsLookupBuilder
    lblb = LeavesToBallsLookupBuilder(ctx)

    lbl, _ = lblb(queue, tree, ball_centers, ball_radii)

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
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
