import numpy as np
import pyopencl as cl
import time

import pytest
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from pymbolic import evaluate

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.opencl
@pytest.mark.parametrize(
    ("nsources", "ntargets", "dims", "dtype"), [
        (5000, 5000, 3, np.float64)
    ]
)
def test_cost_counter(ctx_factory, nsources, ntargets, dims, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ Generate sources, targets and target_radii

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = p_normal(queue, ntargets, dims, dtype, seed=18)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=22)
    target_radii = rng.uniform(
        queue, ntargets, a=0, b=0.05, dtype=dtype
    ).get()

    # }}}

    # {{{ Generate tree and traversal

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)
    tree, _ = tb(
        queue, sources, targets=targets, target_radii=target_radii,
        stick_out_factor=0.15, max_particles_in_box=30, debug=True
    )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx, well_sep_is_n_away=2)
    trav_dev, _ = tg(queue, tree, debug=True)
    trav = trav_dev.get(queue=queue)

    # }}}

    # {{{ Construct cost models

    from boxtree.cost import CLCostModel, PythonCostModel
    cl_cost_model = CLCostModel(queue, None)
    python_cost_model = PythonCostModel(None)

    constant_one_params = dict(
        c_l2l=1,
        c_l2p=1,
        c_m2l=1,
        c_m2m=1,
        c_m2p=1,
        c_p2l=1,
        c_p2m=1,
        c_p2p=1
    )
    for ilevel in range(trav.tree.nlevels):
        constant_one_params["p_fmm_lev%d" % ilevel] = 1

    from boxtree.cost import pde_aware_translation_cost_model
    xlat_cost = pde_aware_translation_cost_model(dims, trav.tree.nlevels)

    # }}}

    # {{{ Test process_direct

    queue.finish()
    start_time = time.time()

    cl_direct = cl_cost_model.process_direct(trav_dev, 5.0)

    queue.finish()
    logger.info("OpenCL time for process_direct: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_direct = python_cost_model.process_direct(trav, 5.0)

    logger.info("Python time for process_direct: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.equal(cl_direct.get(), python_direct).all()

    # }}}

    # {{{ Test aggregate

    start_time = time.time()

    cl_direct_aggregate = cl_cost_model.aggregate(cl_direct)

    queue.finish()
    logger.info("OpenCL time for aggregate: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_direct_aggregate = python_cost_model.aggregate(python_direct)

    logger.info("Python time for aggregate: {0}".format(
        str(time.time() - start_time)
    ))

    assert cl_direct_aggregate == python_direct_aggregate

    # }}}

    # {{{ Test process_list2

    nlevels = trav.tree.nlevels
    m2l_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        m2l_cost[ilevel] = evaluate(
            xlat_cost.m2l(ilevel, ilevel),
            context=constant_one_params
        )
    m2l_cost_dev = cl.array.to_device(queue, m2l_cost)

    queue.finish()
    start_time = time.time()

    cl_m2l_cost = cl_cost_model.process_list2(trav_dev, m2l_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_list2: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_m2l_cost = python_cost_model.process_list2(trav, m2l_cost)
    logger.info("Python time for process_list2: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.equal(cl_m2l_cost.get(), python_m2l_cost).all()

    # }}}

    # {{{ Test process_list 3

    m2p_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        m2p_cost[ilevel] = evaluate(
            xlat_cost.m2p(ilevel),
            context=constant_one_params
        )
    m2p_cost_dev = cl.array.to_device(queue, m2p_cost)

    queue.finish()
    start_time = time.time()

    cl_m2p_cost = cl_cost_model.process_list3(trav_dev, m2p_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_list3: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_m2p_cost = python_cost_model.process_list3(trav, m2p_cost)
    logger.info("Python time for process_list3: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.equal(cl_m2p_cost.get(), python_m2p_cost).all()

    # }}}


def main():
    nsouces = 100000
    ntargets = 100000
    ndims = 3
    dtype = np.float64
    ctx_factory = cl.create_some_context

    test_cost_counter(ctx_factory, nsouces, ntargets, ndims, dtype)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        main()
