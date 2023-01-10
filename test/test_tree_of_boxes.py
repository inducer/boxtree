__copyright__ = "Copyright (C) 2012 Andreas Kloeckner, Xiaoyu Wei"

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

from boxtree import (
    make_meshmode_mesh_from_leaves, make_tree_of_boxes_root,
    uniformly_refine_tree_of_boxes)
from boxtree.array_context import _acf  # noqa: F401
from boxtree.array_context import PytestPyOpenCLArrayContextFactory


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ make_global_leaf_quadrature

def make_global_leaf_quadrature(actx, tob, order):
    from meshmode.discretization.poly_element import (
        GaussLegendreTensorProductGroupFactory)
    group_factory = GaussLegendreTensorProductGroupFactory(order=order)

    mesh, _ = make_meshmode_mesh_from_leaves(tob)

    if 0:
        import matplotlib.pyplot as plt
        from meshmode.mesh import visualization as mvis
        mvis.draw_2d_mesh(mesh,
                          set_bounding_box=True,
                          draw_vertex_numbers=False,
                          draw_element_numbers=False)
        plt.plot(tob.box_centers[0][tob.leaf_boxes],
                 tob.box_centers[1][tob.leaf_boxes], "rx")
        plt.plot(mesh.vertices[0], mesh.vertices[1], "ro")
        plt.show()

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh, group_factory)

    lflevels = tob.box_levels[tob.leaf_boxes]
    lfmeasures = (tob.root_extent / (2**lflevels))**tob.dimensions

    from arraycontext import flatten
    weights = flatten(actx.thaw(discr.quad_weights()), actx)
    jacobians = actx.from_numpy(
        np.repeat(lfmeasures/(2**tob.dimensions), discr.groups[0].nunit_dofs)
        )
    q = weights * jacobians

    from pytools.obj_array import make_obj_array
    nodes = discr.nodes()
    x = make_obj_array([flatten(actx.thaw(axis), actx) for axis in nodes])

    return x, q

# }}}


# {{{ test_uniform_tree_of_boxes

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("nlevels", [1, 4])
def test_uniform_tree_of_boxes(actx_factory, dim, order, nlevels):
    actx = actx_factory()

    lower_bounds = np.random.rand(dim)
    radius = np.random.rand() + 0.1
    upper_bounds = lower_bounds + radius
    tob = make_tree_of_boxes_root((lower_bounds, upper_bounds))

    for _ in range(nlevels - 1):
        tob = uniformly_refine_tree_of_boxes(tob)

    _, q = make_global_leaf_quadrature(actx, tob, order)

    # integrates 1 exactly
    box_area = actx.np.sum(q)
    assert np.isclose(actx.to_numpy(box_area), radius**dim)

# }}}


# {{{ test_uniform_tree_of_boxes_convergence

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_uniform_tree_of_boxes_convergence(actx_factory, dim, order):
    actx = actx_factory()

    radius = np.pi
    lower_bounds = np.zeros(dim) - radius / 2
    upper_bounds = lower_bounds + radius
    tob = make_tree_of_boxes_root((lower_bounds, upper_bounds))

    min_level = 0
    max_level = 1

    for _ in range(min_level):
        tob = uniformly_refine_tree_of_boxes(tob)

    # integrate cos(0.1*x + 0.2*y + 0.3*z + e) over [-pi/2, pi/2]**dim
    qexact_table = {
        1: 20 * np.sin(np.pi/20) * np.cos(np.e),
        2: 50 * (np.sqrt(5) - 1) * np.sin(np.pi/20) * np.cos(np.e),
        3: 250/3 * (np.sqrt(10 - 2*np.sqrt(5)) - 2) * np.cos(np.e)
    }
    qexact = qexact_table[dim]

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for _ in range(min_level, max_level + 1):
        x, q = make_global_leaf_quadrature(actx, tob, order)
        x = np.array([actx.to_numpy(xx) for xx in x])
        q = actx.to_numpy(q)

        inner = np.ones_like(q) * np.e
        for iaxis in range(dim):
            inner += (iaxis + 1) * 0.1 * x[iaxis]
        f = np.cos(inner)
        qh = np.sum(f * q)
        err = abs(qexact - qh)

        if err < 1e-14:
            break  # eoc will be off after hitting machine epsilon

        # under uniform refinement, last box is always leaf
        eoc_rec.add_data_point(tob.get_box_size(-1), err)
        tob = uniformly_refine_tree_of_boxes(tob)

    if len(eoc_rec.history) > 1:
        # Gauss quadrature is exact up to degree 2q+1
        eps = 0.05
        assert eoc_rec.order_estimate() >= 2*order + 2 - eps
    else:
        print(err)
        assert err < 1e-14

# }}}


# {{{ test_tree_plot

def test_tree_plot():
    radius = np.pi
    dim = 2
    nlevels = 3
    lower_bounds = np.zeros(dim) - radius / 2
    upper_bounds = lower_bounds + radius
    tob = make_tree_of_boxes_root((lower_bounds, upper_bounds))

    for _ in range(nlevels - 1):
        tob = uniformly_refine_tree_of_boxes(tob)

    # test TreePlotter compatibility
    from boxtree.visualization import TreePlotter
    tp = TreePlotter(tob)
    tp.draw_tree()
    tp.set_bounding_box()

    # import matplotlib.pyplot as plt
    # plt.show()

# }}}


# {{{ test_traversal_from_tob


def test_traversal_from_tob(actx_factory):
    actx = actx_factory()

    radius = np.pi
    dim = 2
    nlevels = 3
    lower_bounds = np.zeros(dim) - radius/2
    upper_bounds = lower_bounds + radius
    tob = make_tree_of_boxes_root((lower_bounds, upper_bounds))

    for _ in range(nlevels):
        tob = uniformly_refine_tree_of_boxes(tob)

    from boxtree.tree_of_boxes import _sort_boxes_by_level
    tob = _sort_boxes_by_level(tob)

    from dataclasses import replace
    tob = replace(
        tob,
        box_centers=actx.from_numpy(tob.box_centers),
        root_extent=tob.root_extent,
        box_parent_ids=actx.from_numpy(tob.box_parent_ids),
        box_child_ids=actx.from_numpy(tob.box_child_ids),
        box_levels=actx.from_numpy(tob.box_levels),
        box_flags=actx.from_numpy(tob.box_flags),
        )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(actx.context)
    trav, _ = tg(actx.queue, tob)

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
