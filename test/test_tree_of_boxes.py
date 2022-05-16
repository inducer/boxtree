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

import sys
import pytest

import numpy as np
import pyopencl as cl

from arraycontext import pytest_generate_tests_for_array_contexts
from boxtree.array_context import (                                 # noqa: F401
        PytestPyOpenCLArrayContextFactory, _acf)

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


def make_global_leaf_quadrature(actx, tob, order):
    from meshmode.discretization.poly_element import \
        GaussLegendreTensorProductGroupFactory
    group_factory = GaussLegendreTensorProductGroupFactory(order=order)

    from boxtree.tree_build import make_mesh_from_leaves
    mesh = make_mesh_from_leaves(tob)

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh, group_factory)

    lflevels = tob.box_levels[tob.leaf_boxes()]
    lfmeasures = (tob.root_extent / (2**lflevels))**tob.dim

    from arraycontext import flatten
    weights = flatten(discr.quad_weights(), actx).with_queue(actx.queue)
    jacobians = cl.array.to_device(
        actx.queue,
        np.repeat(lfmeasures/(2**tob.dim), discr.groups[0].nunit_dofs))
    q = weights * jacobians

    from pytools.obj_array import make_obj_array
    nodes = discr.nodes()
    x = make_obj_array([flatten(coords, actx).with_queue(actx.queue)
                        for coords in nodes])

    return x, q


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("nlevels", [1, 4])
def test_uniform_tree_of_boxes(ctx_factory, dim, order, nlevels):
    from boxtree.tree_build import make_tob_root, uniformly_refined
    lower_bounds = np.random.rand(dim)
    radius = np.random.rand() + 0.1
    upper_bounds = lower_bounds + radius
    tob = make_tob_root(dim=dim, bbox=[lower_bounds, upper_bounds])

    for _ in range(nlevels - 1):
        tob = uniformly_refined(tob)

    from arraycontext import PyOpenCLArrayContext
    queue = cl.CommandQueue(ctx_factory())
    actx = PyOpenCLArrayContext(queue)

    x, q = make_global_leaf_quadrature(actx, tob, order)

    # integrates 1 exactly
    assert np.isclose(sum(q.get()), radius**dim)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_uniform_tree_of_boxes_convergence(ctx_factory, dim, order):
    from boxtree.tree_build import make_tob_root, uniformly_refined
    radius = np.pi
    lower_bounds = np.zeros(dim) - radius/2
    upper_bounds = lower_bounds + radius
    tob = make_tob_root(dim=dim, bbox=[lower_bounds, upper_bounds])

    min_level = 0
    max_level = 1

    for _ in range(min_level):
        tob = uniformly_refined(tob)

    # integrate cos(0.1*x + 0.2*y + 0.3*z + e) over [-pi/2, pi/2]**dim
    qexact_table = {
        1: 20 * np.sin(np.pi/20) * np.cos(np.e),
        2: 50 * (np.sqrt(5) - 1) * np.sin(np.pi/20) * np.cos(np.e),
        3: 250/3 * (np.sqrt(10 - 2*np.sqrt(5)) - 2) * np.cos(np.e)
    }
    qexact = qexact_table[dim]

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    from arraycontext import PyOpenCLArrayContext
    queue = cl.CommandQueue(ctx_factory())
    actx = PyOpenCLArrayContext(queue)

    for _ in range(min_level, max_level + 1):
        x, q = make_global_leaf_quadrature(actx, tob, order)
        x, q = (np.array([xx.get() for xx in x]), q.get())

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
        tob = uniformly_refined(tob)

    if len(eoc_rec.history) > 1:
        # Gauss quadrature is exact up to degree 2q+1
        eps = 0.01
        assert eoc_rec.order_estimate() >= 2*order + 2 - eps
    else:
        print(err)
        assert err < 1e-14


def test_traversal_from_tob(ctx_factory):
    from boxtree.tree_build import make_tob_root, uniformly_refined
    radius = np.pi
    dim = 2
    nlevels = 3
    lower_bounds = np.zeros(dim) - radius/2
    upper_bounds = lower_bounds + radius
    tob = make_tob_root(dim=dim, bbox=[lower_bounds, upper_bounds])

    for _ in range(nlevels):
        tob = uniformly_refined(tob)

    # FIXME
    tob._is_pruned = True
    tob.sources_have_extent = False
    tob.particle_id_dtype = np.int32
    tob.box_id_dtype = np.int32
    tob.box_level_dtype = np.int32
    tob.coord_dtype = np.float64
    tob.sources_are_targets = True
    tob.targets_have_extent = True
    tob.extent_norm = "linf"

    from boxtree.traversal import FMMTraversalBuilder
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tob)


# You can test individual routines by typing
# $ python test_tree.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
