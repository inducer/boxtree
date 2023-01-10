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

import logging
import sys

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from boxtree.array_context import _acf  # noqa: F401
from boxtree.array_context import PytestPyOpenCLArrayContextFactory
from boxtree.tools import make_normal_particle_array


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ bounding box test

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("nparticles", [9, 4096, 10**5])
def test_bounding_box(actx_factory, dtype, dims, nparticles):
    actx = actx_factory()

    from boxtree.bounding_box import BoundingBoxFinder
    from boxtree.tools import AXIS_NAMES
    bbf = BoundingBoxFinder(actx.context)

    axis_names = AXIS_NAMES[:dims]
    logger.info("%s - %s %s", dtype, dims, nparticles)

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    bbox_min = [np.min(actx.to_numpy(x)) for x in particles]
    bbox_max = [np.max(actx.to_numpy(x)) for x in particles]

    bbox_cl, evt = bbf(particles, radii=None)
    bbox_cl = actx.to_numpy(bbox_cl)

    bbox_min_cl = np.empty(dims, dtype)
    bbox_max_cl = np.empty(dims, dtype)

    for i, ax in enumerate(axis_names):
        bbox_min_cl[i] = bbox_cl[f"min_{ax}"]
        bbox_max_cl[i] = bbox_cl[f"max_{ax}"]

    assert np.all(bbox_min == bbox_min_cl)
    assert np.all(bbox_max == bbox_max_cl)

# }}}


# {{{ test basic (no source/target distinction) tree build

def run_build_test(builder, actx, dims, dtype, nparticles, visualize,
        max_particles_in_box=None, max_leaf_refine_weight=None,
        refine_weights=None, **kwargs):
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        tol = 1e-4
    elif dtype == np.float64:
        tol = 1e-12
    else:
        raise RuntimeError("unsupported dtype: %s" % dtype)

    logger.info(75 * "-")

    if max_particles_in_box is not None:
        logger.info("%dD %s - %d particles - max %d per box - %s",
            dims, dtype.type.__name__, nparticles, max_particles_in_box,
            " - ".join(f"{k}: {v}" for k, v in kwargs.items()))
    else:
        logger.info("%dD %s - %d particles - max leaf weight %d  - %s",
            dims, dtype.type.__name__, nparticles, max_leaf_refine_weight,
            " - ".join(f"{k}: {v}" for k, v in kwargs.items()))

    logger.info(75 * "-")

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    actx.queue.finish()

    tree, _ = builder(actx.queue, particles,
                      max_particles_in_box=max_particles_in_box,
                      refine_weights=refine_weights,
                      max_leaf_refine_weight=max_leaf_refine_weight,
                      debug=True, **kwargs)
    tree = tree.get(queue=actx.queue)

    sorted_particles = np.array(list(tree.sources))

    unsorted_particles = np.array([actx.to_numpy(pi) for pi in particles])
    assert np.all(sorted_particles
            == unsorted_particles[:, tree.user_source_ids])

    if refine_weights is not None:
        refine_weights_reordered = (
                actx.to_numpy(refine_weights)[tree.user_source_ids])

    all_good_so_far = True

    if visualize:
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()

    from boxtree import box_flags_enum as bfe

    scaled_tol = tol*tree.root_extent
    for ibox in range(tree.nboxes):
        # Empty boxes exist in non-pruned trees--which themselves are undocumented.
        # These boxes will fail these tests.
        if not (tree.box_flags[ibox] & bfe.IS_SOURCE_OR_TARGET_BOX):
            continue

        extent_low, extent_high = tree.get_box_extent(ibox)

        assert np.all(extent_low >= tree.bounding_box[0] - scaled_tol), (
                ibox, extent_low, tree.bounding_box[0])
        assert np.all(extent_high <= tree.bounding_box[1] + scaled_tol), (
                ibox, extent_high, tree.bounding_box[1])

        center = tree.box_centers[:, ibox]

        for _, bbox_min, bbox_max in [
                (
                    "source",
                    tree.box_source_bounding_box_min[:, ibox],
                    tree.box_source_bounding_box_max[:, ibox]),
                (
                    "target",
                    tree.box_target_bounding_box_min[:, ibox],
                    tree.box_target_bounding_box_max[:, ibox]),
                ]:
            assert np.all(extent_low - scaled_tol <= bbox_min)
            assert np.all(bbox_min - scaled_tol <= center)

            assert np.all(bbox_max - scaled_tol <= extent_high)
            assert np.all(center - scaled_tol <= bbox_max)

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

        all_good_here = np.all(good)
        if visualize and not all_good_here and all_good_so_far:
            pt.plot(
                    box_particles[0, np.where(~good)[1]],
                    box_particles[1, np.where(~good)[1]], "ro")

            plotter.draw_box(ibox, edgecolor="red")

        if not all_good_here:
            print("BAD BOX", ibox)

        if not (tree.box_flags[ibox] & bfe.HAS_SOURCE_OR_TARGET_CHILD_BOXES):
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

    if visualize:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

    assert all_good_so_far


def particle_tree_test_decorator(f):
    f = pytest.mark.opencl(f)
    f = pytest.mark.parametrize("dtype", [np.float64, np.float32])(f)
    f = pytest.mark.parametrize("dims", [2, 3])(f)

    return f


@particle_tree_test_decorator
def test_single_box_particle_tree(actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims,
            dtype, 4, max_particles_in_box=30, visualize=visualize)


@particle_tree_test_decorator
def test_two_level_particle_tree(actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims,
            dtype, 50, max_particles_in_box=30, visualize=visualize)


@particle_tree_test_decorator
def test_unpruned_particle_tree(actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    # test unpruned tree build
    run_build_test(builder, actx, dims, dtype, 10**5,
            visualize=visualize, max_particles_in_box=30, skip_prune=True)


@particle_tree_test_decorator
def test_particle_tree_with_reallocations(
        actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims, dtype, 10**5,
            max_particles_in_box=30, visualize=visualize, nboxes_guess=5)


@particle_tree_test_decorator
def test_particle_tree_with_many_empty_leaves(
        actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims, dtype, 10**5,
            max_particles_in_box=5, visualize=visualize)


@particle_tree_test_decorator
def test_vanilla_particle_tree(actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims, dtype, 10**5,
            max_particles_in_box=30, visualize=visualize)


@particle_tree_test_decorator
def test_explicit_refine_weights_particle_tree(
        actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    nparticles = 10**5

    rng = np.random.default_rng(10)
    refine_weights = actx.from_numpy(
            rng.integers(1, 10, (nparticles,), dtype=np.int32)
            )

    run_build_test(builder, actx, dims, dtype, nparticles,
            refine_weights=refine_weights, max_leaf_refine_weight=100,
            visualize=visualize)


@particle_tree_test_decorator
def test_non_adaptive_particle_tree(actx_factory, dtype, dims, visualize=False):
    actx = actx_factory()

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)

    run_build_test(builder, actx, dims, dtype, 10**4,
            max_particles_in_box=30, visualize=visualize, kind="non-adaptive")

# }}}


# {{{ source/target tree

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
def test_source_target_tree(actx_factory, dims, visualize=False):
    actx = actx_factory()

    nsources = 2 * 10**5
    ntargets = 3 * 10**5
    dtype = np.float64

    sources = make_normal_particle_array(actx.queue, nsources, dims, dtype,
            seed=12)
    targets = make_normal_particle_array(actx.queue, ntargets, dims, dtype,
            seed=19)

    if visualize:
        import matplotlib.pyplot as pt
        np_sources, np_targets = actx.to_numpy(sources), actx.to_numpy(targets)
        pt.plot(np_sources[0], np_sources[1], "rx")
        pt.plot(np_targets[0], np_targets[1], "g+")
        pt.show()

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, sources, targets=targets,
            max_particles_in_box=10, debug=True)
    tree = tree.get(queue=actx.queue)

    sorted_sources = np.array(list(tree.sources))
    sorted_targets = np.array(list(tree.targets))

    unsorted_sources = np.array([actx.to_numpy(pi) for pi in sources])
    unsorted_targets = np.array([actx.to_numpy(pi) for pi in targets])
    assert np.all(sorted_sources
            == unsorted_sources[:, tree.user_source_ids])

    user_target_ids = np.empty(tree.ntargets, dtype=np.intp)
    user_target_ids[tree.sorted_target_ids] = np.arange(tree.ntargets, dtype=np.intp)
    assert np.all(sorted_targets
            == unsorted_targets[:, user_target_ids])

    all_good_so_far = True

    if visualize:
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()

    tol = 1e-15

    for ibox in range(tree.nboxes):
        extent_low, extent_high = tree.get_box_extent(ibox)

        assert np.all(extent_low
                >= tree.bounding_box[0] - 1e-12*tree.root_extent), ibox
        assert np.all(extent_high
                <= tree.bounding_box[1] + 1e-12*tree.root_extent), ibox

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
            good = np.all(
                    (particles < extent_high[:, np.newaxis] + tol)
                    & (extent_low[:, np.newaxis] - tol <= particles),
                    axis=0)

            all_good_here = np.all(good)

            if visualize and not all_good_here:
                pt.plot(
                        particles[0, np.where(~good)[0]],
                        particles[1, np.where(~good)[0]], "ro")

                plotter.draw_box(ibox, edgecolor="red")
                pt.show()

            if not all_good_here:
                print("BAD BOX %s %d" % (what, ibox))

            all_good_so_far = all_good_so_far and all_good_here

        assert all_good_so_far

    if visualize:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()

# }}}


# {{{ test sources/targets-with-extent tree

@pytest.mark.opencl
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("extent_norm", ["linf", "l2"])
def test_extent_tree(actx_factory, dims, extent_norm, visualize=False):
    actx = actx_factory()

    nsources = 100000
    ntargets = 200000
    dtype = np.float64
    npoint_sources_per_source = 16

    sources = make_normal_particle_array(actx.queue, nsources, dims, dtype,
            seed=12)
    targets = make_normal_particle_array(actx.queue, ntargets, dims, dtype,
            seed=19)

    refine_weights = actx.zeros(nsources + ntargets, np.int32)
    refine_weights[:nsources] = 1

    rng = np.random.default_rng(13)
    source_radii = actx.from_numpy(
            2**rng.uniform(-10, 0, (nsources,)).astype(dtype)
            )
    target_radii = actx.from_numpy(
            2**rng.uniform(-10, 0, (ntargets,)).astype(dtype)
            )

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    dev_tree, _ = tb(actx.queue, sources, targets=targets,
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

    tree = dev_tree.get(queue=actx.queue)

    if visualize:
        import matplotlib.pyplot as pt
        np_sources, np_targets = actx.to_numpy(sources), actx.to_numpy(targets)
        pt.plot(np_sources[0], np_sources[1], "rx")
        pt.plot(np_targets[0], np_targets[1], "g+")

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

    unsorted_sources = np.array([actx.to_numpy(pi) for pi in sources])
    unsorted_targets = np.array([actx.to_numpy(pi) for pi in targets])
    unsorted_source_radii = actx.to_numpy(source_radii)
    unsorted_target_radii = actx.to_numpy(target_radii)

    assert np.all(sorted_sources
            == unsorted_sources[:, tree.user_source_ids])
    assert np.all(sorted_source_radii
            == unsorted_source_radii[tree.user_source_ids])

    # {{{ test box structure, stick-out criterion

    logger.info("test box structure, stick-out criterion")

    user_target_ids = np.empty(tree.ntargets, dtype=np.intp)
    user_target_ids[tree.sorted_target_ids] = np.arange(tree.ntargets, dtype=np.intp)
    if ntargets:
        assert np.all(sorted_targets
                == unsorted_targets[:, user_target_ids])
        assert np.all(sorted_target_radii
                == unsorted_target_radii[user_target_ids])

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

        assert np.all(extent_low
                >= tree.bounding_box[0] - 1e-12*tree.root_extent), ibox
        assert np.all(extent_high
                <= tree.bounding_box[1] + 1e-12*tree.root_extent), ibox

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
                good = np.all(
                        (check_particles + check_radii
                            < extent_high[:, np.newaxis] + stick_out_dist)
                        &  # noqa: W504
                        (extent_low[:, np.newaxis] - stick_out_dist
                            <= check_particles - check_radii),
                        axis=0)

            elif extent_norm == "l2":
                center_dists = np.sqrt(
                        np.sum(
                            (check_particles - box_center.reshape(-1, 1))**2,
                            axis=0))

                good = (
                        (center_dists + check_radii)**2
                        < dims * radius_with_stickout**2)

            else:
                raise ValueError(f"unexpected value of extent_norm: '{extent_norm}'")

            all_good_here = np.all(good)

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

    from pytools.obj_array import make_obj_array
    point_sources = make_obj_array([
            actx.from_numpy(
                unsorted_sources[i][:, np.newaxis]
                + unsorted_source_radii[:, np.newaxis]
                * rng.uniform(-1, 1, size=(nsources, npoint_sources_per_source))
                )
            for i in range(dims)])

    point_source_starts = actx.from_numpy(
            np.arange(
                0,
                (nsources + 1) * npoint_sources_per_source,
                npoint_sources_per_source,
                dtype=tree.particle_id_dtype)
            )

    from boxtree.tree import link_point_sources
    dev_tree = link_point_sources(actx.queue, dev_tree,
            point_source_starts, point_sources,
            debug=True)

    # }}}

# }}}


# {{{ leaves to balls query test

@pytest.mark.opencl
@pytest.mark.geo_lookup
@pytest.mark.parametrize("dims", [2, 3])
def test_leaves_to_balls_query(actx_factory, dims, visualize=False):
    actx = actx_factory()

    nparticles = 10**5
    dtype = np.float64

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    ball_centers = make_normal_particle_array(actx.queue, nballs, dims, dtype)
    ball_radii = 0.1 + actx.zeros(nballs, dtype)

    from boxtree.area_query import LeavesToBallsLookupBuilder
    lblb = LeavesToBallsLookupBuilder(actx.context)

    lbl, _ = lblb(actx.queue, tree, ball_centers, ball_radii)

    # get data to host for test
    tree = tree.get(queue=actx.queue)
    lbl = lbl.get(queue=actx.queue)
    ball_centers = np.array([actx.to_numpy(x) for x in ball_centers]).T
    ball_radii = actx.to_numpy(ball_radii)

    assert len(lbl.balls_near_box_starts) == tree.nboxes + 1

    from boxtree import box_flags_enum

    for ibox in range(tree.nboxes):
        # We only want leaves here.
        if tree.box_flags[ibox] & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES:
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

def run_area_query_test(actx, tree, ball_centers, ball_radii):
    """
    Performs an area query and checks that the result is as expected.
    """
    from boxtree.area_query import AreaQueryBuilder
    aqb = AreaQueryBuilder(actx.context)

    area_query, _ = aqb(actx.queue, tree, ball_centers, ball_radii)

    # Get data to host for test.
    tree = tree.get(queue=actx.queue)
    area_query = area_query.get(queue=actx.queue)
    ball_centers = np.array([actx.to_numpy(x) for x in ball_centers]).T
    ball_radii = actx.to_numpy(ball_radii)

    from boxtree import box_flags_enum
    leaf_boxes, = (
            tree.box_flags & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES == 0
            ).nonzero()

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
def test_area_query(actx_factory, dims, visualize=False):
    actx = actx_factory()

    nparticles = 10**5
    dtype = np.float64

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    ball_centers = make_normal_particle_array(actx.queue, nballs, dims, dtype)
    ball_radii = 0.1 + actx.zeros(nballs, dtype)

    run_area_query_test(actx, tree, ball_centers, ball_radii)


@pytest.mark.opencl
@pytest.mark.area_query
@pytest.mark.parametrize("dims", [2, 3])
def test_area_query_balls_outside_bbox(actx_factory, dims, visualize=False):
    """
    The input to the area query includes balls whose centers are not within
    the tree bounding box.
    """
    actx = actx_factory()

    nparticles = 10**4
    dtype = np.float64

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    bbox_min = tree.bounding_box[0].min()
    bbox_max = tree.bounding_box[1].max()

    from pytools.obj_array import make_obj_array
    rng = np.random.default_rng(13)
    ball_centers = make_obj_array([
        actx.from_numpy(
            rng.uniform(bbox_min - 1, bbox_max + 1, nballs).astype(dtype))
        for i in range(dims)])
    ball_radii = 0.1 + actx.zeros(nballs, dtype)

    run_area_query_test(actx, tree, ball_centers, ball_radii)


@pytest.mark.opencl
@pytest.mark.area_query
@pytest.mark.parametrize("dims", [2, 3])
def test_area_query_elwise(actx_factory, dims, visualize=False):
    actx = actx_factory()

    nparticles = 10**5
    dtype = np.float64

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    ball_centers = make_normal_particle_array(actx.queue, nballs, dims, dtype)
    ball_radii = 0.1 + actx.zeros(nballs, dtype)

    from boxtree.area_query import AreaQueryElementwiseTemplate, PeerListFinder

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

    peer_lists, evt = PeerListFinder(actx.context)(actx.queue, tree)

    kernel = template.generate(
        actx.context,
        dims,
        tree.coord_dtype,
        tree.box_id_dtype,
        peer_lists.peer_list_starts.dtype,
        tree.nlevels)

    evt = kernel(
        *template.unwrap_args(
            tree, peer_lists, ball_radii, *ball_centers),
        queue=actx.queue,
        wait_for=[evt],
        range=slice(len(ball_radii)))

# }}}


# {{{ level restriction test

@pytest.mark.opencl
@pytest.mark.parametrize("lookbehind", [0, 1])
@pytest.mark.parametrize("skip_prune", [True, False])
@pytest.mark.parametrize("dims", [2, 3])
def test_level_restriction(
        actx_factory, dims, skip_prune, lookbehind, visualize=False):
    actx = actx_factory()

    nparticles = 10**5
    dtype = np.float64

    from boxtree.tools import make_surface_particle_array
    particles = make_surface_particle_array(
            actx.queue, nparticles, dims, dtype, seed=15)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree_dev, _ = tb(actx.queue, particles,
            kind="adaptive-level-restricted",
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
        aqb = AreaQueryBuilder(actx.context)

        ball_radii = actx.from_numpy(np.min(leaf_box_radii) / 2 + leaf_box_radii)
        leaf_box_centers = [actx.from_numpy(axis) for axis in leaf_box_centers]

        area_query, _ = aqb(actx.queue, tree_dev, leaf_box_centers, ball_radii)
        area_query = area_query.get(queue=actx.queue)
        return (area_query.leaves_near_ball_starts,
                area_query.leaves_near_ball_lists)

    # Get data to host for test.
    tree = tree_dev.get(queue=actx.queue)

    # Find leaf boxes.
    from boxtree import box_flags_enum
    leaf_boxes, = (
            tree.box_flags & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES == 0
            ).nonzero()

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
        assert np.all(np.abs(neighbor_levels - leaf_level) <= 1), \
                (neighbor_levels, leaf_level)

# }}}


# {{{ space invader query test

@pytest.mark.opencl
@pytest.mark.geo_lookup
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_space_invader_query(actx_factory, dims, dtype, visualize=False):
    actx = actx_factory()

    dtype = np.dtype(dtype)
    nparticles = 10**5

    particles = make_normal_particle_array(actx.queue, nparticles, dims, dtype)

    if visualize:
        import matplotlib.pyplot as pt
        np_particles = actx.to_numpy(particles)
        pt.plot(np_particles[0], np_particles[1], "x")

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    nballs = 10**4
    ball_centers = make_normal_particle_array(actx.queue, nballs, dims, dtype)
    ball_radii = 0.1 + actx.zeros(nballs, dtype)

    from boxtree.area_query import (
        LeavesToBallsLookupBuilder, SpaceInvaderQueryBuilder)

    siqb = SpaceInvaderQueryBuilder(actx.context)
    # We can use leaves-to-balls lookup to get the set of overlapping balls for
    # each box, and from there to compute the outer space invader distance.
    lblb = LeavesToBallsLookupBuilder(actx.context)

    siq, _ = siqb(actx.queue, tree, ball_centers, ball_radii)
    lbl, _ = lblb(actx.queue, tree, ball_centers, ball_radii)

    # get data to host for test
    tree = tree.get(queue=actx.queue)
    siq = siq.get(queue=actx.queue)
    lbl = lbl.get(queue=actx.queue)

    ball_centers = np.array([actx.to_numpy(x) for x in ball_centers])
    ball_radii = actx.to_numpy(ball_radii)

    # Find leaf boxes.
    from boxtree import box_flags_enum

    outer_space_invader_dist = np.zeros(tree.nboxes)

    for ibox in range(tree.nboxes):
        # We only want leaves here.
        if tree.box_flags[ibox] & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES:
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
def test_same_tree_with_zero_weight_particles(actx_factory, dims):
    actx = actx_factory()

    ntargets_values = [300, 400, 500]
    stick_out_factors = [0, 0.1, 0.3, 1]
    nsources = 20

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    trees = []

    rng = np.random.default_rng(10)
    for stick_out_factor in stick_out_factors:
        for ntargets in [40]:
            sources = rng.random((dims, nsources))**2
            sources[:, 0] = -0.1
            sources[:, 1] = 1.1

            targets = rng.random((dims, max(ntargets_values)))[:, :ntargets].copy()
            target_radii = rng.random(max(ntargets_values))[:ntargets]

            sources = actx.from_numpy(sources)
            targets = actx.from_numpy(targets)

            refine_weights = actx.empty(nsources + ntargets, np.int32)
            refine_weights[:nsources] = 1
            refine_weights[nsources:] = 0

            tree, _ = tb(actx.queue, sources, targets=targets,
                    target_radii=target_radii,
                    stick_out_factor=stick_out_factor,
                    max_leaf_refine_weight=10,
                    refine_weights=refine_weights,
                    debug=True)
            tree = tree.get(queue=actx.queue)
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

def test_max_levels_error(actx_factory):
    actx = actx_factory()

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    sources = [actx.zeros(11, np.float64) for i in range(2)]
    from boxtree.tree_build import MaxLevelsExceeded
    with pytest.raises(MaxLevelsExceeded):
        tree, _ = tb(actx.queue, sources, max_particles_in_box=10, debug=True)

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
