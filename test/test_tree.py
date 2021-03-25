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
def test_bounding_box(ctx_factory, dtype, dims, nparticles):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree.tools import AXIS_NAMES
    from boxtree.bounding_box import BoundingBoxFinder

    bbf = BoundingBoxFinder(ctx)

    axis_names = AXIS_NAMES[:dims]

    logger.info(f"{dtype} - {dims} {nparticles}")

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
        max_particles_in_box=None, max_leaf_refine_weight=None,
        refine_weights=None, **kwargs):
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        tol = 1e-4
    elif dtype == np.float64:
        tol = 1e-12
    else:
        raise RuntimeError("unsupported dtype: %s" % dtype)

    logger.info(75*"-")
    if max_particles_in_box is not None:
        logger.info("%dD %s - %d particles - max %d per box - %s" % (
            dims, dtype.type.__name__, nparticles, max_particles_in_box,
            " - ".join(f"{k}: {v}" for k, v in kwargs.items())))
    else:
        logger.info("%dD %s - %d particles - max leaf weight %d  - %s" % (
            dims, dtype.type.__name__, nparticles, max_leaf_refine_weight,
            " - ".join(f"{k}: {v}" for k, v in kwargs.items())))
    logger.info(75*"-")

    particles = make_normal_particle_array(queue, nparticles, dims, dtype)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    queue.finish()

    tree, _ = builder(queue, particles,
                      max_particles_in_box=max_particles_in_box,
                      refine_weights=refine_weights,
                      max_leaf_refine_weight=max_leaf_refine_weight,
                      debug=True, **kwargs)
    tree = tree.get(queue=queue)

    sorted_particles = np.array(list(tree.sources))

    unsorted_particles = np.array([pi.get() for pi in particles])
    assert (sorted_particles
            == unsorted_particles[:, tree.user_source_ids]).all()

    if refine_weights is not None:
        refine_weights_reordered = refine_weights.get()[tree.user_source_ids]

    all_good_so_far = True

    if do_plot:
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()

    from boxtree import box_flags_enum as bfe

    scaled_tol = tol*tree.root_extent
    for ibox in range(tree.nboxes):
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
                & (extent_low[:, np.newaxis] - scaled_tol <= box_particles))

        all_good_here = good.all()
        if do_plot and not all_good_here and all_good_so_far:
            pt.plot(
                    box_particles[0, np.where(~good)[1]],
                    box_particles[1, np.where(~good)[1]], "ro")

            plotter.draw_box(ibox, edgecolor="red")

        if not all_good_here:
            print("BAD BOX", ibox)

        if not (tree.box_flags[ibox] & bfe.HAS_CHILDREN):
            # Check that leaf particle density is as promised.
            nparticles_in_box = tree.box_source_counts_cumul[ibox]
            if max_particles_in_box is not None:
                if nparticles_in_box > max_particles_in_box:
                    print("too many particles ({} > {}); box {}".format(
                        nparticles_in_box, max_particles_in_box, ibox))
                    all_good_here = False
            else:
                assert refine_weights is not None
                box_weight = np.sum(
                    refine_weights_reordered[start:start+nparticles_in_box])
                if box_weight > max_leaf_refine_weight:
                    print("refine weight exceeded ({} > {}); box {}".format(
                        box_weight, max_leaf_refine_weight, ibox))
                    all_good_here = False

        all_good_so_far = all_good_so_far and all_good_here

    if do_plot:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

    assert all_good_so_far


def particle_tree_test_decorator(f):
    f = pytest.mark.opencl(f)
    f = pytest.mark.parametrize("dtype", [np.float64, np.float32])(f)
    f = pytest.mark.parametrize("dims", [2, 3])(f)

    return f


@particle_tree_test_decorator
def test_single_box_particle_tree(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims,
            dtype, 4, max_particles_in_box=30, do_plot=do_plot)


@particle_tree_test_decorator
def test_two_level_particle_tree(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims,
            dtype, 50, max_particles_in_box=30, do_plot=do_plot)


@particle_tree_test_decorator
def test_unpruned_particle_tree(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    # test unpruned tree build
    run_build_test(builder, queue, dims, dtype, 10**5,
            do_plot=do_plot, max_particles_in_box=30, skip_prune=True)


@particle_tree_test_decorator
def test_particle_tree_with_reallocations(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims, dtype, 10**5,
            max_particles_in_box=30, do_plot=do_plot, nboxes_guess=5)


@particle_tree_test_decorator
def test_particle_tree_with_many_empty_leaves(
        ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims, dtype, 10**5,
            max_particles_in_box=5, do_plot=do_plot)


@particle_tree_test_decorator
def test_vanilla_particle_tree(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims, dtype, 10**5,
            max_particles_in_box=30, do_plot=do_plot)


@particle_tree_test_decorator
def test_explicit_refine_weights_particle_tree(ctx_factory, dtype, dims,
            do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    nparticles = 10**5

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx, seed=10)
    refine_weights = rng.uniform(queue, nparticles, dtype=np.int32, a=1, b=10)

    run_build_test(builder, queue, dims, dtype, nparticles,
            refine_weights=refine_weights, max_leaf_refine_weight=100,
            do_plot=do_plot)


@particle_tree_test_decorator
def test_non_adaptive_particle_tree(ctx_factory, dtype, dims, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    builder = TreeBuilder(ctx)

    run_build_test(builder, queue, dims, dtype, 10**4,
            max_particles_in_box=30, do_plot=do_plot, kind="non-adaptive")

# }}}


# {{{ source/target tree

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_source_target_tree(ctx_factory, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
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
        pt.show()

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=10, debug=True)
    tree = tree.get(queue=queue)

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

    tol = 1e-15

    for ibox in range(tree.nboxes):
        extent_low, extent_high = tree.get_box_extent(ibox)

        assert (extent_low
                >= tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
        assert (extent_high
                <= tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

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
                    (particles < extent_high[:, np.newaxis] + tol)
                    & (extent_low[:, np.newaxis] - tol <= particles)
                    ).all(axis=0)

            all_good_here = good.all()

            if do_plot and not all_good_here:
                pt.plot(
                        particles[0, np.where(~good)[0]],
                        particles[1, np.where(~good)[0]], "ro")

                plotter.draw_box(ibox, edgecolor="red")
                pt.show()

        if not all_good_here:
            print("BAD BOX %s %d" % (what, ibox))

        all_good_so_far = all_good_so_far and all_good_here
        assert all_good_so_far

    if do_plot:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

# }}}


# {{{ test sources/targets-with-extent tree

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("extent_norm", ["linf", "l2"])
def test_extent_tree(ctx_factory, dims, extent_norm, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 100000
    ntargets = 200000
    dtype = np.float64
    npoint_sources_per_source = 16

    sources = make_normal_particle_array(queue, nsources, dims, dtype,
            seed=12)
    targets = make_normal_particle_array(queue, ntargets, dims, dtype,
            seed=19)

    refine_weights = cl.array.zeros(queue, nsources+ntargets, np.int32)
    refine_weights[:nsources] = 1

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=13)
    source_radii = 2**rng.uniform(queue, nsources, dtype=dtype,
            a=-10, b=0)
    target_radii = 2**rng.uniform(queue, ntargets, dtype=dtype,
            a=-10, b=0)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    dev_tree, _ = tb(queue, sources, targets=targets,
            source_radii=source_radii,
            target_radii=target_radii,
            extent_norm=extent_norm,

            refine_weights=refine_weights,
            max_leaf_refine_weight=20,

            #max_particles_in_box=10,

            # Set artificially small, to exercise the reallocation code.
            nboxes_guess=10,

            debug=True,
            stick_out_factor=0)

    logger.info("transfer tree, check orderings")

    tree = dev_tree.get(queue=queue)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(sources[0].get(), sources[1].get(), "rx")
        pt.plot(targets[0].get(), targets[1].get(), "g+")

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.draw_box_numbers()
        plotter.set_bounding_box()

        pt.gca().set_aspect("equal", "datalim")
        pt.show()

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

    assert np.sum(tree.box_source_counts_nonchild) == nsources
    assert np.sum(tree.box_target_counts_nonchild) == ntargets

    for ibox in range(tree.nboxes):
        kid_sum = sum(
                    tree.box_target_counts_cumul[ichild_box]
                    for ichild_box in tree.box_child_ids[:, ibox]
                    if ichild_box != 0)
        assert (
                tree.box_target_counts_cumul[ibox]
                == (
                    tree.box_target_counts_nonchild[ibox]
                    + kid_sum)), ibox

    for ibox in range(tree.nboxes):
        extent_low, extent_high = tree.get_box_extent(ibox)

        assert (extent_low
                >= tree.bounding_box[0] - 1e-12*tree.root_extent).all(), ibox
        assert (extent_high
                <= tree.bounding_box[1] + 1e-12*tree.root_extent).all(), ibox

        box_children = tree.box_child_ids[:, ibox]
        existing_children = box_children[box_children != 0]

        assert (tree.box_source_counts_nonchild[ibox]
                + np.sum(tree.box_source_counts_cumul[existing_children])
                == tree.box_source_counts_cumul[ibox])
        assert (tree.box_target_counts_nonchild[ibox]
                + np.sum(tree.box_target_counts_cumul[existing_children])
                == tree.box_target_counts_cumul[ibox])

    del existing_children
    del box_children

    for ibox in range(tree.nboxes):
        lev = int(tree.box_levels[ibox])
        box_radius = 0.5 * tree.root_extent / (1 << lev)
        box_center = tree.box_centers[:, ibox]
        extent_low = box_center - box_radius
        extent_high = box_center + box_radius

        stick_out_dist = tree.stick_out_factor * box_radius
        radius_with_stickout = (1 + tree.stick_out_factor) * box_radius

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

            if extent_norm == "linf":
                good = (
                        (check_particles + check_radii
                            < extent_high[:, np.newaxis] + stick_out_dist)
                        &  # noqa: W504
                        (extent_low[:, np.newaxis] - stick_out_dist
                            <= check_particles - check_radii)
                        ).all(axis=0)

            elif extent_norm == "l2":
                center_dists = np.sqrt(
                        np.sum(
                            (check_particles - box_center.reshape(-1, 1))**2,
                            axis=0))

                good = (
                        (center_dists + check_radii)**2
                        < dims * radius_with_stickout**2)

            else:
                raise ValueError("unexpected value of extent_norm")

            all_good_here = good.all()

            if not all_good_here:
                print("BAD BOX %s %d level %d"
                        % (what, ibox, tree.box_levels[ibox]))

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

    from boxtree.tree import link_point_sources
    dev_tree = link_point_sources(queue, dev_tree,
            point_source_starts, point_sources,
            debug=True)

    # }}}

# }}}


# {{{ leaves to balls query test

@pytest.mark.opencl
@pytest.mark.geo_lookup
@pytest.mark.parametrize("dims", [2, 3])
def test_leaves_to_balls_query(ctx_factory, dims, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
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

    from boxtree.area_query import LeavesToBallsLookupBuilder
    lblb = LeavesToBallsLookupBuilder(ctx)

    lbl, _ = lblb(queue, tree, ball_centers, ball_radii)

    # get data to host for test
    tree = tree.get(queue=queue)
    lbl = lbl.get(queue=queue)
    ball_centers = np.array([x.get() for x in ball_centers]).T
    ball_radii = ball_radii.get()

    assert len(lbl.balls_near_box_starts) == tree.nboxes + 1

    from boxtree import box_flags_enum

    for ibox in range(tree.nboxes):
        # We only want leaves here.
        if tree.box_flags[ibox] & box_flags_enum.HAS_CHILDREN:
            continue

        box_center = tree.box_centers[:, ibox]
        ext_l, ext_h = tree.get_box_extent(ibox)
        box_rad = 0.5*(ext_h-ext_l)[0]

        linf_circle_dists = np.max(np.abs(ball_centers-box_center), axis=-1)
        near_circles, = np.where(linf_circle_dists - ball_radii < box_rad)

        start, end = lbl.balls_near_box_starts[ibox:ibox+2]
        assert sorted(lbl.balls_near_box_lists[start:end]) == sorted(near_circles)

# }}}


# {{{ area query test

def run_area_query_test(ctx, queue, tree, ball_centers, ball_radii):
    """
    Performs an area query and checks that the result is as expected.
    """
    from boxtree.area_query import AreaQueryBuilder
    aqb = AreaQueryBuilder(ctx)

    area_query, _ = aqb(queue, tree, ball_centers, ball_radii)

    # Get data to host for test.
    tree = tree.get(queue=queue)
    area_query = area_query.get(queue=queue)
    ball_centers = np.array([x.get() for x in ball_centers]).T
    ball_radii = ball_radii.get()

    from boxtree import box_flags_enum
    leaf_boxes, = (tree.box_flags & box_flags_enum.HAS_CHILDREN == 0).nonzero()

    leaf_box_radii = np.empty(len(leaf_boxes))
    dims = len(tree.sources)
    leaf_box_centers = np.empty((len(leaf_boxes), dims))

    for idx, leaf_box in enumerate(leaf_boxes):
        box_center = tree.box_centers[:, leaf_box]
        ext_l, ext_h = tree.get_box_extent(leaf_box)
        leaf_box_radii[idx] = np.max(ext_h - ext_l) * 0.5
        leaf_box_centers[idx] = box_center

    for ball_nr, (ball_center, ball_radius) \
            in enumerate(zip(ball_centers, ball_radii)):
        linf_box_dists = np.max(np.abs(ball_center - leaf_box_centers), axis=-1)
        near_leaves_indices, \
            = np.where(linf_box_dists < ball_radius + leaf_box_radii)
        near_leaves = leaf_boxes[near_leaves_indices]

        start, end = area_query.leaves_near_ball_starts[ball_nr:ball_nr+2]
        found = area_query.leaves_near_ball_lists[start:end]
        actual = near_leaves
        assert set(found) == set(actual), (found, actual)


@pytest.mark.opencl
@pytest.mark.area_query
@pytest.mark.parametrize("dims", [2, 3])
def test_area_query(ctx_factory, dims, do_plot=False):
    ctx = ctx_factory()
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

    run_area_query_test(ctx, queue, tree, ball_centers, ball_radii)


@pytest.mark.opencl
@pytest.mark.area_query
@pytest.mark.parametrize("dims", [2, 3])
def test_area_query_balls_outside_bbox(ctx_factory, dims, do_plot=False):
    """
    The input to the area query includes balls whose centers are not within
    the tree bounding box.
    """
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nparticles = 10**4
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
    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx, seed=13)
    bbox_min = tree.bounding_box[0].min()
    bbox_max = tree.bounding_box[1].max()
    from pytools.obj_array import make_obj_array
    ball_centers = make_obj_array([
        rng.uniform(queue, nballs, dtype=dtype, a=bbox_min-1, b=bbox_max+1)
        for i in range(dims)])
    ball_radii = cl.array.empty(queue, nballs, dtype).fill(0.1)

    run_area_query_test(ctx, queue, tree, ball_centers, ball_radii)


@pytest.mark.opencl
@pytest.mark.area_query
@pytest.mark.parametrize("dims", [2, 3])
def test_area_query_elwise(ctx_factory, dims, do_plot=False):
    ctx = ctx_factory()
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

    from boxtree.area_query import (
        AreaQueryElementwiseTemplate, PeerListFinder)

    template = AreaQueryElementwiseTemplate(
        extra_args="""
            coord_t *ball_radii,
            %for ax in AXIS_NAMES[:dimensions]:
                coord_t *ball_${ax},
            %endfor
        """,
        ball_center_and_radius_expr="""
            %for ax in AXIS_NAMES[:dimensions]:
                ${ball_center}.${ax} = ball_${ax}[${i}];
            %endfor
            ${ball_radius} = ball_radii[${i}];
        """,
        leaf_found_op="")

    peer_lists, evt = PeerListFinder(ctx)(queue, tree)

    kernel = template.generate(
        ctx,
        dims,
        tree.coord_dtype,
        tree.box_id_dtype,
        peer_lists.peer_list_starts.dtype,
        tree.nlevels)

    evt = kernel(
        *template.unwrap_args(
            tree, peer_lists, ball_radii, *ball_centers),
        queue=queue,
        wait_for=[evt],
        range=slice(len(ball_radii)))

    cl.wait_for_events([evt])

# }}}


# {{{ level restriction test

@pytest.mark.opencl
@pytest.mark.parametrize("lookbehind", [0, 1])
@pytest.mark.parametrize("skip_prune", [True, False])
@pytest.mark.parametrize("dims", [2, 3])
def test_level_restriction(ctx_factory, dims, skip_prune, lookbehind, do_plot=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nparticles = 10**5
    dtype = np.float64

    from boxtree.tools import make_surface_particle_array
    particles = make_surface_particle_array(queue, nparticles, dims, dtype, seed=15)

    if do_plot:
        import matplotlib.pyplot as pt
        pt.plot(particles[0].get(), particles[1].get(), "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    tree_dev, _ = tb(queue, particles, kind="adaptive-level-restricted",
                     max_particles_in_box=30, debug=True,
                     skip_prune=skip_prune, lr_lookbehind=lookbehind,

                     # Artificially low to exercise reallocation code
                     nboxes_guess=10)

    def find_neighbors(leaf_box_centers, leaf_box_radii):
        # We use an area query with a ball that is slightly larger than
        # the size of a leaf box to find the neighboring leaves.
        #
        # Note that since this comes from an area query, the self box will be
        # included in the neighbor list.
        from boxtree.area_query import AreaQueryBuilder
        aqb = AreaQueryBuilder(ctx)

        ball_radii = cl.array.to_device(queue,
            np.min(leaf_box_radii) / 2 + leaf_box_radii)
        leaf_box_centers = [
            cl.array.to_device(queue, axis) for axis in leaf_box_centers]

        area_query, _ = aqb(queue, tree_dev, leaf_box_centers, ball_radii)
        area_query = area_query.get(queue=queue)
        return (area_query.leaves_near_ball_starts,
                area_query.leaves_near_ball_lists)

    # Get data to host for test.
    tree = tree_dev.get(queue=queue)

    # Find leaf boxes.
    from boxtree import box_flags_enum
    leaf_boxes, = (tree.box_flags & box_flags_enum.HAS_CHILDREN == 0).nonzero()

    leaf_box_radii = np.empty(len(leaf_boxes))
    leaf_box_centers = np.empty((dims, len(leaf_boxes)))

    for idx, leaf_box in enumerate(leaf_boxes):
        box_center = tree.box_centers[:, leaf_box]
        ext_l, ext_h = tree.get_box_extent(leaf_box)
        leaf_box_radii[idx] = np.max(ext_h - ext_l) * 0.5
        leaf_box_centers[:, idx] = box_center

    neighbor_starts, neighbor_and_self_lists = find_neighbors(
        leaf_box_centers, leaf_box_radii)

    # Check level restriction.
    for leaf_idx, leaf in enumerate(leaf_boxes):
        neighbors = neighbor_and_self_lists[
            neighbor_starts[leaf_idx]:neighbor_starts[leaf_idx+1]]
        neighbor_levels = np.array(tree.box_levels[neighbors], dtype=int)
        leaf_level = int(tree.box_levels[leaf])
        assert (np.abs(neighbor_levels - leaf_level) <= 1).all(), \
                (neighbor_levels, leaf_level)

# }}}


# {{{ space invader query test

@pytest.mark.opencl
@pytest.mark.geo_lookup
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_space_invader_query(ctx_factory, dims, dtype, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dtype = np.dtype(dtype)
    nparticles = 10**5

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

    from boxtree.area_query import (
        LeavesToBallsLookupBuilder, SpaceInvaderQueryBuilder)

    siqb = SpaceInvaderQueryBuilder(ctx)
    # We can use leaves-to-balls lookup to get the set of overlapping balls for
    # each box, and from there to compute the outer space invader distance.
    lblb = LeavesToBallsLookupBuilder(ctx)

    siq, _ = siqb(queue, tree, ball_centers, ball_radii)
    lbl, _ = lblb(queue, tree, ball_centers, ball_radii)

    # get data to host for test
    tree = tree.get(queue=queue)
    siq = siq.get(queue=queue)
    lbl = lbl.get(queue=queue)

    ball_centers = np.array([x.get() for x in ball_centers])
    ball_radii = ball_radii.get()

    # Find leaf boxes.
    from boxtree import box_flags_enum

    outer_space_invader_dist = np.zeros(tree.nboxes)

    for ibox in range(tree.nboxes):
        # We only want leaves here.
        if tree.box_flags[ibox] & box_flags_enum.HAS_CHILDREN:
            continue

        start, end = lbl.balls_near_box_starts[ibox:ibox + 2]
        space_invaders = lbl.balls_near_box_lists[start:end]
        if len(space_invaders) > 0:
            outer_space_invader_dist[ibox] = np.max(np.abs(
                    tree.box_centers[:, ibox].reshape((-1, 1))
                    - ball_centers[:, space_invaders]))

    assert np.allclose(siq, outer_space_invader_dist)

# }}}


# {{{ test_same_tree_with_zero_weight_particles

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_same_tree_with_zero_weight_particles(ctx_factory, dims):
    logging.basicConfig(level=logging.INFO)

    ntargets_values = [300, 400, 500]
    stick_out_factors = [0, 0.1, 0.3, 1]
    nsources = 20

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    trees = []

    for stick_out_factor in stick_out_factors:
        for ntargets in [40]:
            np.random.seed(10)
            sources = np.random.rand(dims, nsources)**2
            sources[:, 0] = -0.1
            sources[:, 1] = 1.1

            np.random.seed()
            targets = np.random.rand(dims, max(ntargets_values))[:, :ntargets].copy()
            target_radii = np.random.rand(max(ntargets_values))[:ntargets]

            sources = cl.array.to_device(queue, sources)
            targets = cl.array.to_device(queue, targets)

            refine_weights = cl.array.empty(queue, nsources + ntargets, np.int32)
            refine_weights[:nsources] = 1
            refine_weights[nsources:] = 0

            tree, _ = tb(queue, sources, targets=targets,
                    target_radii=target_radii,
                    stick_out_factor=stick_out_factor,
                    max_leaf_refine_weight=10,
                    refine_weights=refine_weights,
                    debug=True)
            tree = tree.get(queue=queue)
            trees.append(tree)

            print("TREE:", tree.nboxes)

    if 0:
        import matplotlib.pyplot as plt
        for tree in trees:
            plt.figure()
            tree.plot()

        plt.show()

# }}}


# {{{ test_max_levels_error

def test_max_levels_error(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    logging.basicConfig(level=logging.INFO)

    sources = [cl.array.zeros(queue, 11, float) for i in range(2)]
    from boxtree.tree_build import MaxLevelsExceeded
    with pytest.raises(MaxLevelsExceeded):
        tree, _ = tb(queue, sources, max_particles_in_box=10, debug=True)

# }}}


# You can test individual routines by typing
# $ python test_tree.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
