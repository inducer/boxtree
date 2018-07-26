from __future__ import division
import pyopencl as cl
import numpy as np
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import functools
from boxtree.distributed.perf_model import PerformanceModel, PerformanceCounter
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

test_cases = [
    (9000, 9000, 3),
    (9000, 9000, 3),
    (12000, 12000, 3),
    (12000, 12000, 3),
    (15000, 15000, 3),
    (15000, 15000, 3),
    (18000, 18000, 3),
    (18000, 18000, 3),
    (25000, 25000, 3)  # this last test case is for evaluation
]

for nsources, ntargets, dims in test_cases:

    from boxtree.distributed.perf_model import generate_random_traversal
    traversals.append(generate_random_traversal(
        context, nsources, ntargets, dims, dtype
    ))

# }}}

wrangler_factory = functools.partial(
    FMMLibExpansionWrangler, helmholtz_k=0, fmm_level_to_nterms=fmm_level_to_nterms)

ntraversals = len(traversals)
model = PerformanceModel(context, wrangler_factory, True, drive_fmm)
for i in range(ntraversals - 1):
    model.time_performance(traversals[i])

eval_traversal = traversals[-1]
eval_wrangler = wrangler_factory(eval_traversal.tree)
dimensions = eval_traversal.tree.dimensions
eval_counter = PerformanceCounter(eval_traversal, eval_wrangler, True)

wall_time = True

predict_timing = model.predict_time(eval_traversal, eval_counter,
                                    wall_time=wall_time)

# {{{ Actual timing

true_timing = {}

rng = PhiloxGenerator(context)
source_weights = rng.uniform(
    queue, eval_traversal.tree.nsources, eval_traversal.tree.coord_dtype).get()

_ = drive_fmm(eval_traversal, eval_wrangler, source_weights, timing_data=true_timing)

# }}}


for field in ["eval_direct", "multipole_to_local", "eval_multipoles", "form_locals",
              "eval_locals"]:
    predict_time_field = predict_timing[field]

    if wall_time:
        true_time_field = true_timing[field].wall_elapsed
    else:
        true_time_field = true_timing[field].process_elapsed

    diff = abs(predict_time_field - true_time_field)

    print(field + ": predict " + str(predict_time_field) + " actual " +
          str(true_time_field) + " error " + str(diff / true_time_field))
