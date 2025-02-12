import logging
import os
import sys

import numpy as np

import pyopencl as cl


# Configure the root logger
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))

logger = logging.getLogger(__name__)

# Set the logger level of this module to INFO so that logging outputs of this module
# are shown
logger.setLevel(logging.INFO)

# `process_elapsed` in `ProcessTimer` is only supported for Python >= 3.3
SUPPORTS_PROCESS_TIME = (sys.version_info >= (3, 3))


def demo_cost_model():
    if not SUPPORTS_PROCESS_TIME:
        raise NotImplementedError(
            "Currently this script uses process time which only works on Python>=3.3"
        )

    from boxtree.pyfmmlib_integration import (
        FMMLibExpansionWrangler,
        FMMLibTreeIndependentDataForWrangler,
        Kernel,
    )

    rng = np.random.default_rng(seed=42)
    nsources_list = [1000, 2000, 3000, 4000, 5000]
    ntargets_list = [1000, 2000, 3000, 4000, 5000]
    dims = 3
    dtype = np.float64

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    traversals = []
    traversals_dev = []
    level_orders_list = []
    timing_results = []

    def fmm_level_to_order(tree, ilevel):
        return 10

    for nsources, ntargets in zip(nsources_list, ntargets_list, strict=True):
        # {{{ Generate sources, targets and target_radii

        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = p_normal(queue, ntargets, dims, dtype, seed=18)

        from pyopencl.clrandom import PhiloxGenerator

        clrng = PhiloxGenerator(queue.context, seed=22)
        target_radii = clrng.uniform(
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

        tree_indep = FMMLibTreeIndependentDataForWrangler(
                trav.tree.dimensions, Kernel.LAPLACE)
        wrangler = FMMLibExpansionWrangler(tree_indep, trav,
                fmm_level_to_order=fmm_level_to_order)
        level_orders_list.append(wrangler.level_orders)

        timing_data = {}
        from boxtree.fmm import drive_fmm
        src_weights = rng.random(size=tree.nsources, dtype=tree.coord_dtype)
        drive_fmm(wrangler, (src_weights,), timing_data=timing_data)

        timing_results.append(timing_data)

    time_field_name = "process_elapsed"

    from boxtree.cost import FMMCostModel, make_pde_aware_translation_cost_model
    cost_model = FMMCostModel(make_pde_aware_translation_cost_model)

    model_results = []
    for icase in range(len(traversals)-1):
        traversal = traversals_dev[icase]
        model_results.append(
            cost_model.cost_per_stage(
                queue, traversal, level_orders_list[icase],
                FMMCostModel.get_unit_calibration_params(),
            )
        )
    queue.finish()

    params = cost_model.estimate_calibration_params(
        model_results, timing_results[:-1], time_field_name=time_field_name
    )

    predicted_time = cost_model.cost_per_stage(
        queue, traversals_dev[-1], level_orders_list[-1], params,
    )
    queue.finish()

    for field in ["form_multipoles", "eval_direct", "multipole_to_local",
                  "eval_multipoles", "form_locals", "eval_locals",
                  "coarsen_multipoles", "refine_locals"]:
        measured = timing_results[-1][field]["process_elapsed"]
        pred_err = (
                (measured - predicted_time[field])
                / measured)
        logger.info("actual/predicted time for %s: %.3g/%.3g -> %g %% error",
                field,
                measured,
                predicted_time[field],
                abs(100*pred_err))


if __name__ == "__main__":
    demo_cost_model()
