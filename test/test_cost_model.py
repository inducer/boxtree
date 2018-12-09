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
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    # }}}

    # {{{ Construct cost models

    from boxtree.cost import CLCostModel, PythonCostModel
    cl_cost_model = CLCostModel(queue, None)
    python_cost_model = PythonCostModel(None)

    CONSTANT_ONE_PARAMS = dict(
        c_l2l=1,
        c_l2p=1,
        c_m2l=1,
        c_m2m=1,
        c_m2p=1,
        c_p2l=1,
        c_p2m=1,
        c_p2p=1
    )

    from boxtree.cost import pde_aware_translation_cost_model
    xlat_cost = pde_aware_translation_cost_model(dims, trav.tree.nlevels)

    # }}}

    # {{{ Test collect_direct_interaction_data

    start_time = time.time()
    cl_direct_interaction = \
        cl_cost_model.collect_direct_interaction_data(trav)
    logger.info("OpenCL time for collect_direct_interaction_data: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_direct_interaction = \
        python_cost_model.collect_direct_interaction_data(trav)
    logger.info("Python time for collect_direct_interaction_data: {0}".format(
        str(time.time() - start_time)
    ))

    for field in ["nlist1_srcs_by_itgt_box", "nlist3close_srcs_by_itgt_box",
                  "nlist4close_srcs_by_itgt_box"]:
        assert np.equal(
            cl_direct_interaction[field],
            python_direct_interaction[field]
        ).all()

    # }}}

    # {{{ Test count_direct

    start_time = time.time()

    cl_count_direct = evaluate(
        cl_cost_model.count_direct(xlat_cost, trav),
        context=CONSTANT_ONE_PARAMS
    )

    logger.info("OpenCL time for count_direct: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_count_direct = evaluate(
        python_cost_model.count_direct(xlat_cost, trav),
        context=CONSTANT_ONE_PARAMS
    )

    logger.info("Python time for count_direct: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.equal(cl_count_direct, python_count_direct).all()

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
