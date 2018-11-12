import numpy as np
import pyopencl as cl
import time

import pytest
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)

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
        stick_out_factor=0.15, max_particles_in_box=60, debug=True
    )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx, well_sep_is_n_away=2)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    # }}}

    from boxtree.cost import CLCostCounter, PythonCostCounter
    cl_cost_counter = CLCostCounter(queue)
    python_cost_counter = PythonCostCounter()

    start_time = time.time()
    cl_direct_interaction = cl_cost_counter.collect_direct_interaction_data(
        trav, trav.tree
    )
    logger.info("OpenCL time for collect_direct_interaction_data: {0}".format(
        str(time.time() - start_time))
    )

    start_time = time.time()
    python_direct_interaction = python_cost_counter.collect_direct_interaction_data(
        trav, trav.tree
    )
    logger.info("Python time for collect_direct_interaction_data: {0}".format(
        str(time.time() - start_time))
    )

    for field in ["nlist1_srcs_by_itgt_box", "nlist3close_srcs_by_itgt_box",
                  "nlist4close_srcs_by_itgt_box"]:
        assert np.equal(
            cl_direct_interaction[field],
            python_direct_interaction[field]
        ).all()


def main():
    nsouces = 5000
    ntargets = 5000
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
