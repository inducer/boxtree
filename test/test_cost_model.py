__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Matt Wala
Copyright (C) 2018 Hao Gao
"""

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
import pyopencl as cl
import time

import pytest
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from pymbolic import evaluate
from boxtree.cost import FMMCostModel, _PythonFMMCostModel
from boxtree.cost import make_pde_aware_translation_cost_model
import sys

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SUPPORTS_PROCESS_TIME = (sys.version_info >= (3, 3))


# {{{ test_compare_cl_and_py_cost_model

@pytest.mark.opencl
@pytest.mark.parametrize(
    ("nsources", "ntargets", "dims", "dtype"), [
        (50000, 50000, 3, np.float64)
    ]
)
def test_compare_cl_and_py_cost_model(ctx_factory, nsources, ntargets, dims, dtype):
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

    cl_cost_model = FMMCostModel(None)
    python_cost_model = _PythonFMMCostModel(None)

    constant_one_params = cl_cost_model.get_unit_calibration_params().copy()
    for ilevel in range(trav.tree.nlevels):
        constant_one_params["p_fmm_lev%d" % ilevel] = 10

    xlat_cost = make_pde_aware_translation_cost_model(dims, trav.tree.nlevels)

    # }}}

    # {{{ Test process_form_multipoles

    nlevels = trav.tree.nlevels
    p2m_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        p2m_cost[ilevel] = evaluate(
            xlat_cost.p2m(ilevel),
            context=constant_one_params
        )
    p2m_cost_dev = cl.array.to_device(queue, p2m_cost)

    queue.finish()
    start_time = time.time()

    cl_form_multipoles = cl_cost_model.process_form_multipoles(
        queue, trav_dev, p2m_cost_dev
    )

    queue.finish()
    logger.info("OpenCL time for process_form_multipoles: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_form_multipoles = python_cost_model.process_form_multipoles(
        queue, trav, p2m_cost
    )

    logger.info("Python time for process_form_multipoles: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_form_multipoles.get(), python_form_multipoles)

    # }}}

    # {{{ Test process_coarsen_multipoles

    m2m_cost = np.zeros(nlevels - 1, dtype=np.float64)
    for target_level in range(nlevels - 1):
        m2m_cost[target_level] = evaluate(
            xlat_cost.m2m(target_level + 1, target_level),
            context=constant_one_params
        )
    m2m_cost_dev = cl.array.to_device(queue, m2m_cost)

    queue.finish()
    start_time = time.time()
    cl_coarsen_multipoles = cl_cost_model.process_coarsen_multipoles(
        queue, trav_dev, m2m_cost_dev
    )

    queue.finish()
    logger.info("OpenCL time for coarsen_multipoles: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_coarsen_multipoles = python_cost_model.process_coarsen_multipoles(
        queue, trav, m2m_cost
    )

    logger.info("Python time for coarsen_multipoles: {}".format(
        str(time.time() - start_time)
    ))

    assert cl_coarsen_multipoles == python_coarsen_multipoles

    # }}}

    # {{{ Test process_direct

    queue.finish()
    start_time = time.time()

    cl_ndirect_sources_per_target_box = \
        cl_cost_model.get_ndirect_sources_per_target_box(queue, trav_dev)

    cl_direct = cl_cost_model.process_direct(
        queue, trav_dev, cl_ndirect_sources_per_target_box, 5.0
    )

    queue.finish()
    logger.info("OpenCL time for process_direct: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_ndirect_sources_per_target_box = \
        python_cost_model.get_ndirect_sources_per_target_box(queue, trav)

    python_direct = python_cost_model.process_direct(
        queue, trav, python_ndirect_sources_per_target_box, 5.0
    )

    logger.info("Python time for process_direct: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_direct.get(), python_direct)

    # }}}

    # {{{ Test aggregate_over_boxes

    start_time = time.time()

    cl_direct_aggregate = cl_cost_model.aggregate_over_boxes(cl_direct)

    queue.finish()
    logger.info("OpenCL time for aggregate_over_boxes: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_direct_aggregate = python_cost_model.aggregate_over_boxes(python_direct)

    logger.info("Python time for aggregate_over_boxes: {}".format(
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

    cl_m2l_cost = cl_cost_model.process_list2(queue, trav_dev, m2l_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_list2: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_m2l_cost = python_cost_model.process_list2(queue, trav, m2l_cost)
    logger.info("Python time for process_list2: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_m2l_cost.get(), python_m2l_cost)

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

    cl_m2p_cost = cl_cost_model.process_list3(queue, trav_dev, m2p_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_list3: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_m2p_cost = python_cost_model.process_list3(queue, trav, m2p_cost)
    logger.info("Python time for process_list3: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_m2p_cost.get(), python_m2p_cost)

    # }}}

    # {{{ Test process_list4

    p2l_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        p2l_cost[ilevel] = evaluate(
            xlat_cost.p2l(ilevel),
            context=constant_one_params
        )
    p2l_cost_dev = cl.array.to_device(queue, p2l_cost)

    queue.finish()
    start_time = time.time()

    cl_p2l_cost = cl_cost_model.process_list4(queue, trav_dev, p2l_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_list4: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_p2l_cost = python_cost_model.process_list4(queue, trav, p2l_cost)
    logger.info("Python time for process_list4: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_p2l_cost.get(), python_p2l_cost)

    # }}}

    # {{{ Test process_refine_locals

    l2l_cost = np.zeros(nlevels - 1, dtype=np.float64)
    for ilevel in range(nlevels - 1):
        l2l_cost[ilevel] = evaluate(
            xlat_cost.l2l(ilevel, ilevel + 1),
            context=constant_one_params
        )
    l2l_cost_dev = cl.array.to_device(queue, l2l_cost)

    queue.finish()
    start_time = time.time()

    cl_refine_locals_cost = cl_cost_model.process_refine_locals(
        queue, trav_dev, l2l_cost_dev
    )

    queue.finish()
    logger.info("OpenCL time for refine_locals: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_refine_locals_cost = python_cost_model.process_refine_locals(
        queue, trav, l2l_cost
    )
    logger.info("Python time for refine_locals: {}".format(
        str(time.time() - start_time)
    ))

    assert cl_refine_locals_cost == python_refine_locals_cost

    # }}}

    # {{{ Test process_eval_locals

    l2p_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        l2p_cost[ilevel] = evaluate(
            xlat_cost.l2p(ilevel),
            context=constant_one_params
        )
    l2p_cost_dev = cl.array.to_device(queue, l2p_cost)

    queue.finish()
    start_time = time.time()

    cl_l2p_cost = cl_cost_model.process_eval_locals(queue, trav_dev, l2p_cost_dev)

    queue.finish()
    logger.info("OpenCL time for process_eval_locals: {}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()
    python_l2p_cost = python_cost_model.process_eval_locals(queue, trav, l2p_cost)
    logger.info("Python time for process_eval_locals: {}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_l2p_cost.get(), python_l2p_cost)

    # }}}

# }}}


# {{{ test_estimate_calibration_params

@pytest.mark.opencl
def test_estimate_calibration_params(ctx_factory):
    from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

    nsources_list = [1000, 2000, 3000, 4000]
    ntargets_list = [1000, 2000, 3000, 4000]
    dims = 3
    dtype = np.float64

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    traversals = []
    traversals_dev = []
    level_to_orders = []
    timing_results = []

    def fmm_level_to_nterms(tree, ilevel):
        return 10

    for nsources, ntargets in zip(nsources_list, ntargets_list):
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

        traversals.append(trav)
        traversals_dev.append(trav_dev)

        # }}}

        wrangler = FMMLibExpansionWrangler(trav.tree, 0, fmm_level_to_nterms)
        level_to_orders.append(wrangler.level_nterms)

        timing_data = {}
        from boxtree.fmm import drive_fmm
        src_weights = np.random.rand(tree.nsources).astype(tree.coord_dtype)
        drive_fmm(trav, wrangler, (src_weights,), timing_data=timing_data)

        timing_results.append(timing_data)

    if SUPPORTS_PROCESS_TIME:
        time_field_name = "process_elapsed"
    else:
        time_field_name = "wall_elapsed"

    def test_params_sanity(test_params):
        param_names = ["c_p2m", "c_m2m", "c_p2p", "c_m2l", "c_m2p", "c_p2l", "c_l2l",
                       "c_l2p"]
        for name in param_names:
            assert isinstance(test_params[name], np.float64)

    def test_params_equal(test_params1, test_params2):
        param_names = ["c_p2m", "c_m2m", "c_p2p", "c_m2l", "c_m2p", "c_p2l", "c_l2l",
                       "c_l2p"]
        for name in param_names:
            assert test_params1[name] == test_params2[name]

    python_cost_model = _PythonFMMCostModel(make_pde_aware_translation_cost_model)

    python_model_results = []

    for icase in range(len(traversals)-1):
        traversal = traversals[icase]
        level_to_order = level_to_orders[icase]

        python_model_results.append(python_cost_model.cost_per_stage(
            queue, traversal, level_to_order,
            _PythonFMMCostModel.get_unit_calibration_params(),
        ))

    python_params = python_cost_model.estimate_calibration_params(
        python_model_results, timing_results[:-1], time_field_name=time_field_name
    )

    test_params_sanity(python_params)

    cl_cost_model = FMMCostModel(make_pde_aware_translation_cost_model)

    cl_model_results = []

    for icase in range(len(traversals_dev)-1):
        traversal = traversals_dev[icase]
        level_to_order = level_to_orders[icase]

        cl_model_results.append(cl_cost_model.cost_per_stage(
            queue, traversal, level_to_order,
            FMMCostModel.get_unit_calibration_params(),
        ))

    cl_params = cl_cost_model.estimate_calibration_params(
        cl_model_results, timing_results[:-1], time_field_name=time_field_name
    )

    test_params_sanity(cl_params)

    if SUPPORTS_PROCESS_TIME:
        test_params_equal(cl_params, python_params)

# }}}


# {{{ test_cost_model_op_counts_agree_with_constantone_wrangler

class OpCountingTranslationCostModel:
    """A translation cost model which assigns at cost of 1 to each operation."""

    def __init__(self, dim, nlevels):
        pass

    @staticmethod
    def direct():
        return 1

    @staticmethod
    def p2l(level):
        return 1

    l2p = p2l
    p2m = p2l
    m2p = p2l

    @staticmethod
    def m2m(src_level, tgt_level):
        return 1

    l2l = m2m
    m2l = m2m


@pytest.mark.opencl
@pytest.mark.parametrize(
    ("nsources", "ntargets", "dims", "dtype"), [
        (5000, 5000, 3, np.float64)
    ]
)
def test_cost_model_op_counts_agree_with_constantone_wrangler(
        ctx_factory, nsources, ntargets, dims, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(queue, nsources, dims, dtype, seed=16)
    targets = p_normal(queue, ntargets, dims, dtype, seed=19)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=20)
    target_radii = rng.uniform(queue, ntargets, a=0, b=0.04, dtype=dtype).get()

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

    from boxtree.tools import ConstantOneExpansionWrangler
    wrangler = ConstantOneExpansionWrangler(trav.tree)

    timing_data = {}
    from boxtree.fmm import drive_fmm
    src_weights = np.random.rand(tree.nsources).astype(tree.coord_dtype)
    drive_fmm(trav, wrangler, (src_weights,), timing_data=timing_data)

    cost_model = FMMCostModel(
        translation_cost_model_factory=OpCountingTranslationCostModel
    )

    level_to_order = np.array([1 for _ in range(tree.nlevels)])

    modeled_time = cost_model.cost_per_stage(
        queue, trav_dev, level_to_order,
        FMMCostModel.get_unit_calibration_params(),
    )

    mismatches = []
    for stage in timing_data:
        if timing_data[stage]["ops_elapsed"] != modeled_time[stage]:
            mismatches.append(
                    (stage, timing_data[stage]["ops_elapsed"], modeled_time[stage]))

    assert not mismatches, "\n".join(str(s) for s in mismatches)

    # {{{ Test per-box cost

    total_cost = 0.0
    for stage in timing_data:
        total_cost += timing_data[stage]["ops_elapsed"]

    per_box_cost = cost_model.cost_per_box(
        queue, trav_dev, level_to_order,
        FMMCostModel.get_unit_calibration_params(),
    )
    total_aggregate_cost = cost_model.aggregate_over_boxes(per_box_cost)

    assert total_cost == (
            total_aggregate_cost
            + modeled_time["coarsen_multipoles"]
            + modeled_time["refine_locals"]
    )

    # }}}

# }}}


# You can test individual routines by typing
# $ python test_cost_model.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
