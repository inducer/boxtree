import numpy as np
import pyopencl as cl
import sys

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def demo_cost_model():
    from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

    nsources_list = [1000, 2000, 3000, 4000, 5000]
    ntargets_list = [1000, 2000, 3000, 4000, 5000]
    dims = 3
    dtype = np.float64

    ctx = cl.create_some_context()
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
        drive_fmm(trav, wrangler, src_weights, timing_data=timing_data)

        timing_results.append(timing_data)

    assert sys.version_info >= (3, 0)
    wall_time = False

    from boxtree.cost import CLFMMCostModel
    from boxtree.cost import pde_aware_translation_cost_model
    cl_cost_model = CLFMMCostModel(queue, pde_aware_translation_cost_model)
    cl_params = cl_cost_model.estimate_calibration_params(
        traversals_dev[:-1], level_to_orders[:-1], timing_results[:-1],
        wall_time=wall_time
    )

    ndirect_sources_per_target_box = \
        cl_cost_model.get_ndirect_sources_per_target_box(traversals_dev[-1])

    cl_predicted_time = cl_cost_model(
        traversals_dev[-1], level_to_orders[-1], cl_params,
        ndirect_sources_per_target_box
    )

    for field in ["form_multipoles", "eval_direct", "multipole_to_local",
                  "eval_multipoles", "form_locals", "eval_locals",
                  "coarsen_multipoles", "refine_locals"]:
        logger.info("predicted time for {0}: {1}".format(
            field, str(cl_cost_model.aggregate(cl_predicted_time[field]))
        ))
        logger.info("actual time for {0}: {1}".format(
            field, str(timing_results[-1][field]["process_elapsed"])
        ))


if __name__ == '__main__':
    demo_cost_model()
