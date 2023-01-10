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

import logging
import sys
import time

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from boxtree.array_context import _acf  # noqa: F401
from boxtree.array_context import PytestPyOpenCLArrayContextFactory
from boxtree.cost import (
    FMMCostModel, _PythonFMMCostModel, make_pde_aware_translation_cost_model)


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_compare_cl_and_py_cost_model

@pytest.mark.opencl
@pytest.mark.parametrize(
    ("nsources", "ntargets", "dims", "dtype"), [
        (50000, 50000, 3, np.float64)
    ]
)
def test_compare_cl_and_py_cost_model(actx_factory, nsources, ntargets, dims, dtype):
    actx = actx_factory()

    # {{{ Generate sources, targets and target_radii

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(actx.queue, nsources, dims, dtype, seed=15)
    targets = p_normal(actx.queue, ntargets, dims, dtype, seed=18)

    rng = np.random.default_rng(22)
    target_radii = rng.uniform(0.0, 0.05, (ntargets,)).astype(dtype)

    # }}}

    # {{{ Generate tree and traversal

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)
    tree, _ = tb(
        actx.queue, sources, targets=targets, target_radii=target_radii,
        stick_out_factor=0.15, max_particles_in_box=30, debug=True
    )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=2)
    trav_dev, _ = tg(actx.queue, tree, debug=True)
    trav = trav_dev.get(queue=actx.queue)

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

    from pymbolic import evaluate
    nlevels = trav.tree.nlevels
    p2m_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        p2m_cost[ilevel] = evaluate(
            xlat_cost.p2m(ilevel),
            context=constant_one_params
        )
    p2m_cost_dev = actx.from_numpy(p2m_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_form_multipoles = cl_cost_model.process_form_multipoles(
        actx.queue, trav_dev, p2m_cost_dev
    )

    actx.queue.finish()
    logger.info("OpenCL time for process_form_multipoles: %gs",
            time.time() - start_time)

    start_time = time.time()

    python_form_multipoles = python_cost_model.process_form_multipoles(
        actx.queue, trav, p2m_cost
    )

    logger.info("Python time for process_form_multipoles: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_form_multipoles), python_form_multipoles)

    # }}}

    # {{{ Test process_coarsen_multipoles

    m2m_cost = np.zeros(nlevels - 1, dtype=np.float64)
    for target_level in range(nlevels - 1):
        m2m_cost[target_level] = evaluate(
            xlat_cost.m2m(target_level + 1, target_level),
            context=constant_one_params
        )
    m2m_cost_dev = actx.from_numpy(m2m_cost)

    actx.queue.finish()
    start_time = time.time()
    cl_coarsen_multipoles = cl_cost_model.process_coarsen_multipoles(
        actx.queue, trav_dev, m2m_cost_dev
    )

    actx.queue.finish()
    logger.info("OpenCL time for coarsen_multipoles: %gs",
            time.time() - start_time)

    start_time = time.time()

    python_coarsen_multipoles = python_cost_model.process_coarsen_multipoles(
        actx.queue, trav, m2m_cost
    )

    logger.info("Python time for coarsen_multipoles: %gs",
            time.time() - start_time)

    assert cl_coarsen_multipoles == python_coarsen_multipoles

    # }}}

    # {{{ Test process_direct

    actx.queue.finish()
    start_time = time.time()

    cl_ndirect_sources_per_target_box = \
        cl_cost_model.get_ndirect_sources_per_target_box(actx.queue, trav_dev)

    cl_direct = cl_cost_model.process_direct(
        actx.queue, trav_dev, cl_ndirect_sources_per_target_box, 5.0
    )

    actx.queue.finish()
    logger.info("OpenCL time for process_direct: %gs",
            time.time() - start_time)

    start_time = time.time()

    python_ndirect_sources_per_target_box = \
        python_cost_model.get_ndirect_sources_per_target_box(actx.queue, trav)

    python_direct = python_cost_model.process_direct(
        actx.queue, trav, python_ndirect_sources_per_target_box, 5.0
    )

    logger.info("Python time for process_direct: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_direct), python_direct)

    # }}}

    # {{{ Test aggregate_over_boxes

    start_time = time.time()

    cl_direct_aggregate = cl_cost_model.aggregate_over_boxes(cl_direct)

    actx.queue.finish()
    logger.info("OpenCL time for aggregate_over_boxes: %gs",
            time.time() - start_time)

    start_time = time.time()

    python_direct_aggregate = python_cost_model.aggregate_over_boxes(python_direct)

    logger.info("Python time for aggregate_over_boxes: %gs",
            time.time() - start_time)

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
    m2l_cost_dev = actx.from_numpy(m2l_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_m2l_cost = cl_cost_model.process_list2(actx.queue, trav_dev, m2l_cost_dev)

    actx.queue.finish()
    logger.info("OpenCL time for process_list2: %gs",
            time.time() - start_time)

    start_time = time.time()
    python_m2l_cost = python_cost_model.process_list2(actx.queue, trav, m2l_cost)
    logger.info("Python time for process_list2: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_m2l_cost), python_m2l_cost)

    # }}}

    # {{{ Test process_list 3

    m2p_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        m2p_cost[ilevel] = evaluate(
            xlat_cost.m2p(ilevel),
            context=constant_one_params
        )
    m2p_cost_dev = actx.from_numpy(m2p_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_m2p_cost = cl_cost_model.process_list3(actx.queue, trav_dev, m2p_cost_dev)

    actx.queue.finish()
    logger.info("OpenCL time for process_list3: %gs",
            time.time() - start_time)

    start_time = time.time()
    python_m2p_cost = python_cost_model.process_list3(actx.queue, trav, m2p_cost)
    logger.info("Python time for process_list3: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_m2p_cost), python_m2p_cost)

    # }}}

    # {{{ Test process_list4

    p2l_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        p2l_cost[ilevel] = evaluate(
            xlat_cost.p2l(ilevel),
            context=constant_one_params
        )
    p2l_cost_dev = actx.from_numpy(p2l_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_p2l_cost = cl_cost_model.process_list4(actx.queue, trav_dev, p2l_cost_dev)

    actx.queue.finish()
    logger.info("OpenCL time for process_list4: %gs",
            time.time() - start_time)

    start_time = time.time()
    python_p2l_cost = python_cost_model.process_list4(actx.queue, trav, p2l_cost)
    logger.info("Python time for process_list4: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_p2l_cost), python_p2l_cost)

    # }}}

    # {{{ Test process_refine_locals

    l2l_cost = np.zeros(nlevels - 1, dtype=np.float64)
    for ilevel in range(nlevels - 1):
        l2l_cost[ilevel] = evaluate(
            xlat_cost.l2l(ilevel, ilevel + 1),
            context=constant_one_params
        )
    l2l_cost_dev = actx.from_numpy(l2l_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_refine_locals_cost = cl_cost_model.process_refine_locals(
        actx.queue, trav_dev, l2l_cost_dev
    )

    actx.queue.finish()
    logger.info("OpenCL time for refine_locals: %gs",
            time.time() - start_time)

    start_time = time.time()
    python_refine_locals_cost = python_cost_model.process_refine_locals(
        actx.queue, trav, l2l_cost
    )
    logger.info("Python time for refine_locals: %gs",
            time.time() - start_time)

    assert cl_refine_locals_cost == python_refine_locals_cost

    # }}}

    # {{{ Test process_eval_locals

    l2p_cost = np.zeros(nlevels, dtype=np.float64)
    for ilevel in range(nlevels):
        l2p_cost[ilevel] = evaluate(
            xlat_cost.l2p(ilevel),
            context=constant_one_params
        )
    l2p_cost_dev = actx.from_numpy(l2p_cost)

    actx.queue.finish()
    start_time = time.time()

    cl_l2p_cost = cl_cost_model.process_eval_locals(
            actx.queue, trav_dev, l2p_cost_dev)

    actx.queue.finish()
    logger.info("OpenCL time for process_eval_locals: %gs",
            time.time() - start_time)

    start_time = time.time()
    python_l2p_cost = python_cost_model.process_eval_locals(
            actx.queue, trav, l2p_cost)
    logger.info("Python time for process_eval_locals: %gs",
            time.time() - start_time)

    assert np.array_equal(actx.to_numpy(cl_l2p_cost), python_l2p_cost)

    # }}}

# }}}


# {{{ test_estimate_calibration_params

@pytest.mark.opencl
def test_estimate_calibration_params(actx_factory):
    from boxtree.pyfmmlib_integration import (
        FMMLibExpansionWrangler, FMMLibTreeIndependentDataForWrangler, Kernel)

    nsources_list = [1000, 2000, 3000, 4000]
    ntargets_list = [1000, 2000, 3000, 4000]
    dims = 3
    dtype = np.float64

    actx = actx_factory()

    traversals = []
    traversals_dev = []
    level_to_orders = []
    timing_results = []

    def fmm_level_to_order(tree, ilevel):
        return 10

    for nsources, ntargets in zip(nsources_list, ntargets_list):
        # {{{ Generate sources, targets and target_radii

        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(actx.queue, nsources, dims, dtype, seed=15)
        targets = p_normal(actx.queue, ntargets, dims, dtype, seed=18)

        rng = np.random.default_rng(22)
        target_radii = rng.uniform(0.0, 0.05, (ntargets,)).astype(dtype)

        # }}}

        # {{{ Generate tree and traversal

        from boxtree import TreeBuilder
        tb = TreeBuilder(actx.context)
        tree, _ = tb(
            actx.queue, sources, targets=targets, target_radii=target_radii,
            stick_out_factor=0.15, max_particles_in_box=30, debug=True
        )

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=2)
        trav_dev, _ = tg(actx.queue, tree, debug=True)
        trav = trav_dev.get(queue=actx.queue)

        traversals.append(trav)
        traversals_dev.append(trav_dev)

        # }}}

        tree_indep = FMMLibTreeIndependentDataForWrangler(
                trav.tree.dimensions, Kernel.LAPLACE)
        wrangler = FMMLibExpansionWrangler(tree_indep, trav,
                fmm_level_to_order=fmm_level_to_order)
        level_to_orders.append(wrangler.level_orders)

        timing_data = {}
        from boxtree.fmm import drive_fmm
        src_weights = np.random.rand(tree.nsources).astype(tree.coord_dtype)
        drive_fmm(wrangler, (src_weights,), timing_data=timing_data)

        timing_results.append(timing_data)

    time_field_name = "process_elapsed"

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
            actx.queue, traversal, level_to_order,
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
            actx.queue, traversal, level_to_order,
            FMMCostModel.get_unit_calibration_params(),
        ))

    cl_params = cl_cost_model.estimate_calibration_params(
        cl_model_results, timing_results[:-1], time_field_name=time_field_name
    )

    test_params_sanity(cl_params)
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
        actx_factory, nsources, ntargets, dims, dtype):
    actx = actx_factory()

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(actx.queue, nsources, dims, dtype, seed=16)
    targets = p_normal(actx.queue, ntargets, dims, dtype, seed=19)

    rng = np.random.default_rng(20)
    target_radii = rng.uniform(0, 0.04, (ntargets,)).astype(dtype)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx.context)
    tree, _ = tb(
        actx.queue, sources, targets=targets, target_radii=target_radii,
        stick_out_factor=0.15, max_particles_in_box=30, debug=True
    )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=2)
    trav_dev, _ = tg(actx.queue, tree, debug=True)
    trav = trav_dev.get(queue=actx.queue)

    from boxtree.constant_one import (
        ConstantOneExpansionWrangler, ConstantOneTreeIndependentDataForWrangler)
    tree_indep = ConstantOneTreeIndependentDataForWrangler()
    wrangler = ConstantOneExpansionWrangler(tree_indep, trav)

    timing_data = {}
    from boxtree.fmm import drive_fmm
    src_weights = np.random.rand(tree.nsources).astype(tree.coord_dtype)
    drive_fmm(wrangler, (src_weights,), timing_data=timing_data)

    cost_model = FMMCostModel(
        translation_cost_model_factory=OpCountingTranslationCostModel
    )

    level_to_order = np.array([1 for _ in range(tree.nlevels)])

    modeled_time = cost_model.cost_per_stage(
        actx.queue, trav_dev, level_to_order,
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
        actx.queue, trav_dev, level_to_order,
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
# $ python test_cost_model.py 'test_routine(_acf)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
