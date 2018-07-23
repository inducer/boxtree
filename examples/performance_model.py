from __future__ import division
import pyopencl as cl
import numpy as np
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import functools
from boxtree.fmm import PerformanceModel, PerformanceCounter
from boxtree.fmm import drive_fmm
from pyopencl.clrandom import PhiloxGenerator

context = cl.create_some_context()
queue = cl.CommandQueue(context)
dtype = np.float64
helmholtz_k = 0


def fmm_level_to_nterms(tree, level):
    return max(level, 3)


# {{{ Generate traversal objects for forming models and verification

traversals = []

for nsources, ntargets, dims in [(6000, 6000, 3),
                                 (9000, 9000, 3),
                                 (12000, 12000, 3),
                                 (15000, 15000, 3),
                                 (20000, 20000, 3)]:

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = p_normal(queue, ntargets, dims, dtype, seed=18)

    rng = PhiloxGenerator(context, seed=22)
    target_radii = rng.uniform(
        queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

    from boxtree import TreeBuilder
    tb = TreeBuilder(context)
    tree, _ = tb(queue, sources, targets=targets, target_radii=target_radii,
                 stick_out_factor=0.25, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(context, well_sep_is_n_away=2)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    traversals.append(trav)

# }}}

wrangler_factory = functools.partial(
    FMMLibExpansionWrangler, helmholtz_k=0, fmm_level_to_nterms=fmm_level_to_nterms)

ntraversals = len(traversals)
model = PerformanceModel(context, wrangler_factory, True)
for i in range(ntraversals - 1):
    model.time_performance(traversals[i])

eval_traversal = traversals[-1]
eval_wrangler = wrangler_factory(eval_traversal.tree)
dimensions = eval_traversal.tree.dimensions
eval_counter = PerformanceCounter(eval_traversal, eval_wrangler, True)

predict_timing = {}
wall_time = True

# {{{ Predict eval_direct

param = model.eval_direct_model(wall_time=wall_time)

direct_workload = np.sum(eval_counter.count_direct())
direct_nsource_boxes = eval_traversal.neighbor_source_boxes_starts[-1]

predict_timing["eval_direct"] = (
        direct_workload * param[0] + direct_nsource_boxes * param[1] + param[2])

# }}}

# {{{ Predict multipole_to_local

param = model.multipole_to_local_model(wall_time=wall_time)

m2l_workload = np.sum(eval_counter.count_m2l())

predict_timing["multipole_to_local"] = m2l_workload * param[0] + param[1]

# }}}

# {{{ Predict eval_multipoles

param = model.eval_multipoles_model(wall_time=wall_time)

m2p_workload = np.sum(eval_counter.count_m2p())

predict_timing["eval_multipoles"] = m2p_workload * param[0] + param[1]

# }}}

# {{{ Actual timing

true_timing = {}

rng = PhiloxGenerator(context)
source_weights = rng.uniform(
    queue, eval_traversal.tree.nsources, eval_traversal.tree.coord_dtype).get()

_ = drive_fmm(eval_traversal, eval_wrangler, source_weights, timing_data=true_timing)

# }}}


for field in ["eval_direct", "multipole_to_local", "eval_multipoles"]:
    wall_time_field = predict_timing[field]

    if wall_time:
        true_time_field = true_timing[field].wall_elapsed
    else:
        true_time_field = true_timing[field].process_elapsed

    diff = abs(wall_time_field - true_time_field)

    print(field + " error: " + str(diff / true_time_field))
