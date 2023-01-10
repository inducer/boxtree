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

import logging

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from boxtree.array_context import _acf  # noqa: F401
from boxtree.array_context import PytestPyOpenCLArrayContextFactory
from boxtree.tools import make_normal_particle_array


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ connectivity test

@pytest.mark.opencl
@pytest.mark.parametrize(("dims", "sources_are_targets"), [
    (2, True),
    (2, False),
    (3, True),
    (3, False),
    ])
def test_tree_connectivity(actx_factory, dims, sources_are_targets):
    actx = actx_factory()
    dtype = np.float64

    sources = make_normal_particle_array(actx.queue, 1 * 10**5, dims, dtype)
    if sources_are_targets:
        targets = None
    else:
        targets = make_normal_particle_array(actx.queue, 2 * 10**5, dims, dtype)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)
    tree, _ = tb(actx.queue, sources, max_particles_in_box=30,
            targets=targets, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(actx.context)
    trav, _ = tg(actx.queue, tree, debug=True)
    tree = tree.get(queue=actx.queue)
    trav = trav.get(queue=actx.queue)

    levels = tree.box_levels
    parents = tree.box_parent_ids.T
    children = tree.box_child_ids.T
    centers = tree.box_centers.T

    # {{{ parent and child relations, levels match up

    for ibox in range(1, tree.nboxes):
        # /!\ Not testing box 0, has no parents
        parent = parents[ibox]

        assert levels[parent] + 1 == levels[ibox]
        assert ibox in children[parent], ibox

    # }}}

    if 0:
        import matplotlib.pyplot as pt

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black")
        plotter.draw_box_numbers()
        plotter.set_bounding_box()
        pt.show()

    # {{{ neighbor_source_boxes (list 1) consists of source boxes

    for itgt_box, ibox in enumerate(trav.target_boxes):
        start, end = trav.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
        nbl = trav.neighbor_source_boxes_lists[start:end]

        if sources_are_targets:
            assert ibox in nbl

        for jbox in nbl:
            assert np.all(0 == children[jbox]), (ibox, jbox, children[jbox])

    logger.info("list 1 consists of source boxes")

    # }}}

    # {{{ separated siblings (list 2) are actually separated

    for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
        start, end = trav.from_sep_siblings_starts[itgt_box:itgt_box+2]
        seps = trav.from_sep_siblings_lists[start:end]

        assert np.all(levels[seps] == levels[tgt_ibox])

        # three-ish box radii (half of size)
        mindist = 2.5 * 0.5 * 2**-int(levels[tgt_ibox]) * tree.root_extent

        icenter = centers[tgt_ibox]
        for jbox in seps:
            dist = la.norm(centers[jbox]-icenter)
            assert dist > mindist, (dist, mindist)

    logger.info("separated siblings (list 2) are actually separated")

    # }}}

    if sources_are_targets:
        # {{{ from_sep_{smaller,bigger} are duals of each other

        assert np.all(trav.target_or_target_parent_boxes == np.arange(tree.nboxes))

        # {{{ list 4 <= list 3

        for level, ssn in enumerate(trav.from_sep_smaller_by_level):
            for itarget_box, ibox in \
                    enumerate(trav.target_boxes_sep_smaller_by_source_level[level]):
                start, end = ssn.starts[itarget_box:itarget_box+2]

                for jbox in ssn.lists[start:end]:
                    rstart, rend = trav.from_sep_bigger_starts[jbox:jbox+2]

                    assert ibox in trav.from_sep_bigger_lists[rstart:rend], \
                            (ibox, jbox)

        # }}}

        # {{{ list 4 <= list 3

        box_to_target_boxes_sep_smaller_by_source_level = []
        for level in range(trav.tree.nlevels):
            box_to_target_boxes_sep_smaller = np.empty(
                tree.nboxes, tree.box_id_dtype)
            box_to_target_boxes_sep_smaller.fill(-1)
            box_to_target_boxes_sep_smaller[
                trav.target_boxes_sep_smaller_by_source_level[level]
            ] = np.arange(
                len(trav.target_boxes_sep_smaller_by_source_level[level]),
                dtype=tree.box_id_dtype
            )
            box_to_target_boxes_sep_smaller_by_source_level.append(
                box_to_target_boxes_sep_smaller)

        assert np.all(trav.source_boxes == trav.target_boxes)
        assert np.all(trav.target_or_target_parent_boxes == np.arange(
                tree.nboxes, dtype=tree.box_id_dtype))

        for ibox in range(tree.nboxes):
            start, end = trav.from_sep_bigger_starts[ibox:ibox+2]

            for jbox in trav.from_sep_bigger_lists[start:end]:
                # In principle, entries of from_sep_bigger_lists are
                # source boxes. In this special case, source and target boxes
                # are the same thing (i.e. leaves--see assertion above), so we
                # may treat them as targets anyhow.

                good = False

                for level, ssn in enumerate(trav.from_sep_smaller_by_level):
                    jtgt_box = \
                        box_to_target_boxes_sep_smaller_by_source_level[level][jbox]
                    if jtgt_box == -1:
                        continue
                    rstart, rend = ssn.starts[jtgt_box:jtgt_box + 2]
                    good = good or ibox in ssn.lists[rstart:rend]

                if not good:
                    from boxtree.visualization import TreePlotter
                    plotter = TreePlotter(tree)
                    plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
                    plotter.set_bounding_box()

                    plotter.draw_box(ibox, facecolor="green", alpha=0.5)
                    plotter.draw_box(jbox, facecolor="red", alpha=0.5)

                    import matplotlib.pyplot as pt
                    pt.gca().set_aspect("equal")
                    pt.show()

                # This assertion failing means that ibox's list 4 contains a box
                # 'jbox' whose list 3 does not contain ibox.
                assert good, (ibox, jbox)

        # }}}

        logger.info("list 3, 4 are duals")

        # }}}

    # {{{ from_sep_smaller satisfies relative level assumption

    # for itarget_box, ibox in enumerate(trav.target_boxes):
    #    for ssn in trav.from_sep_smaller_by_level:
    for level, ssn in enumerate(trav.from_sep_smaller_by_level):
        for itarget_box, ibox in enumerate(
                trav.target_boxes_sep_smaller_by_source_level[level]):
            start, end = ssn.starts[itarget_box:itarget_box+2]

            for jbox in ssn.lists[start:end]:
                assert levels[ibox] < levels[jbox]

    logger.info("list 3 satisfies relative level assumption")

    # }}}

    # {{{ from_sep_bigger satisfies relative level assumption

    for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
        start, end = trav.from_sep_bigger_starts[itgt_box:itgt_box+2]

        for jbox in trav.from_sep_bigger_lists[start:end]:
            assert levels[tgt_ibox] > levels[jbox]

    logger.info("list 4 satisfies relative level assumption")

    # }}}

    # {{{ level_start_*_box_nrs lists make sense

    for name, ref_array in [
            ("level_start_source_box_nrs", trav.source_boxes),
            ("level_start_source_parent_box_nrs", trav.source_parent_boxes),
            ("level_start_target_box_nrs", trav.target_boxes),
            ("level_start_target_or_target_parent_box_nrs",
                trav.target_or_target_parent_boxes)
            ]:
        level_starts = getattr(trav, name)
        for lev in range(tree.nlevels):
            start, stop = level_starts[lev:lev+2]

            box_nrs = ref_array[start:stop]

            assert np.all(tree.box_levels[box_nrs] == lev), name

    # }}}

# }}}


# {{{ visualization helper

# Set 'visualize' kwarg to True to actually plot. Otherwise, this
# test simply ensures that interaction list plotting is still
# working.

def test_plot_traversal(actx_factory, well_sep_is_n_away=1, visualize=False):
    pytest.importorskip("matplotlib")
    actx = actx_factory()

    for dims in [2]:
        nparticles = 10**4
        dtype = np.float64

        from pytools.obj_array import make_obj_array
        rng = np.random.default_rng(15)
        particles = make_obj_array([
            actx.from_numpy(rng.normal(0.0, 1.0, (nparticles,)).astype(dtype))
            for i in range(dims)])

        from boxtree import TreeBuilder
        tb = TreeBuilder(actx.context)

        actx.queue.finish()
        tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=well_sep_is_n_away)
        trav, _ = tg(actx.queue, tree)

        tree = tree.get(queue=actx.queue)
        trav = trav.get(queue=actx.queue)

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black")
        #plotter.draw_box_numbers()
        plotter.set_bounding_box()

        from boxtree.visualization import draw_box_lists
        if well_sep_is_n_away == 1:
            draw_box_lists(plotter, trav, 380)
        elif well_sep_is_n_away == 2:
            draw_box_lists(plotter, trav, 320)

        if visualize:
            import matplotlib.pyplot as pt
            pt.gca().set_xticks([])
            pt.gca().set_yticks([])

            pt.show()

# }}}


# {{{ test_from_sep_siblings_translation_and_rotation_classes

@pytest.mark.parametrize("well_sep_is_n_away", (1, 2))
def test_from_sep_siblings_translation_and_rotation_classes(
        actx_factory, well_sep_is_n_away):
    actx = actx_factory()

    dims = 3
    nparticles = 10**4
    dtype = np.float64

    # {{{ build tree

    from pytools.obj_array import make_obj_array
    rng = np.random.default_rng(15)
    particles = make_obj_array([
        actx.from_numpy(rng.normal(0.0, 1.0, (nparticles,)).astype(dtype))
        for i in range(dims)])

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)

    actx.queue.finish()
    tree, _ = tb(actx.queue, particles, max_particles_in_box=30, debug=True)

    # }}}

    # {{{ build traversal

    from boxtree.rotation_classes import RotationClassesBuilder
    from boxtree.translation_classes import TranslationClassesBuilder
    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=well_sep_is_n_away)
    trav, _ = tg(actx.queue, tree)

    rb = RotationClassesBuilder(actx.context)
    result, _ = rb(actx.queue, trav, tree)

    tb = TranslationClassesBuilder(actx.context)
    result_tb, _ = tb(actx.queue, trav, tree)

    rot_classes = actx.to_numpy(
            result.from_sep_siblings_rotation_classes)
    rot_angles = actx.to_numpy(
            result.from_sep_siblings_rotation_class_to_angle)

    translation_classes = actx.to_numpy(
            result_tb.from_sep_siblings_translation_classes)
    distance_vectors = actx.to_numpy(
        result_tb.from_sep_siblings_translation_class_to_distance_vector)

    tree = tree.get(queue=actx.queue)
    trav = trav.get(queue=actx.queue)

    centers = tree.box_centers.T

    # }}}

    # For each entry of from_sep_siblings, compute the source-target translation
    # direction as a vector, and check that the from_sep_siblings rotation class
    # in the traversal corresponds to the angle with the z-axis of the
    # translation direction.

    for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
        start, end = trav.from_sep_siblings_starts[itgt_box:itgt_box+2]
        seps = trav.from_sep_siblings_lists[start:end]
        level_rot_classes = rot_classes[start:end]

        level_translation_classes = translation_classes[start:end]
        actual_translation_vecs = distance_vectors[:, level_translation_classes].T
        expected_translation_vecs = centers[tgt_ibox] - centers[seps]

        if len(expected_translation_vecs) > 0:
            assert np.allclose(actual_translation_vecs, expected_translation_vecs,
                               atol=1e-13, rtol=1e-13)

        theta = np.arctan2(
                la.norm(expected_translation_vecs[:, :dims - 1], axis=1),
                expected_translation_vecs[:, dims - 1])
        level_rot_angles = rot_angles[level_rot_classes]

        assert np.allclose(theta, level_rot_angles, atol=1e-13, rtol=1e-13)

# }}}


# You can test individual routines by typing
# $ python test_traversal.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
