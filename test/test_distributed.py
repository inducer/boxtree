import numpy as np
import sys
from mpi4py import MPI
from boxtree.distributed import (generate_local_tree, generate_local_travs,
                                 drive_dfmm, WorkloadWeight,
                                 DistributedFMMLibExpansionWranglerCodeContainer)
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import time

# import logging
# logging.basicConfig(level=logging.INFO)

# Parameters
dims = 2
nsources = 10000
ntargets = 10000
dtype = np.float64

# Get the current rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialization
trav = None
sources_weights = None
wrangler = None


ORDER = 3
HELMHOLTZ_K = 0


# Generate particles and run shared-memory parallelism on rank 0
if rank == 0:
    last_time = time.time()

    # Configure PyOpenCL
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    print(queue.context.devices)

    # Generate random particles and source weights
    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = (p_normal(queue, ntargets, dims, dtype, seed=18) +
               np.array([2, 0, 0])[:dims])

    from boxtree.tools import particle_array_to_host
    sources_host = particle_array_to_host(sources)
    targets_host = particle_array_to_host(targets)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=20)
    sources_weights = rng.uniform(queue, nsources, dtype=np.float64).get()

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=22)
    target_radii = rng.uniform(queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

    # Display sources and targets
    if "--display" in sys.argv:
        import matplotlib.pyplot as plt
        plt.plot(sources_host[:, 0], sources_host[:, 1], "bo")
        plt.plot(targets_host[:, 0], targets_host[:, 1], "ro")
        plt.show()

    now = time.time()
    print("Generate particles " + str(now - last_time))
    last_time = now

    # Calculate potentials using direct evaluation
    # distances = la.norm(sources_host.reshape(1, nsources, 2) - \
    #                    targets_host.reshape(ntargets, 1, 2),
    #                    ord=2, axis=2)
    # pot_naive = np.sum(-np.log(distances)*sources_weights, axis=1)

    # Build the tree and interaction lists
    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)
    tree, _ = tb(queue, sources, targets=targets, target_radii=target_radii,
                 stick_out_factor=0.25, max_particles_in_box=30, debug=True)

    now = time.time()
    print("Generate tree " + str(now - last_time))
    last_time = now

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    now = time.time()
    print("Generate traversal " + str(now - last_time))
    last_time = now

    # Get pyfmmlib expansion wrangler
    def fmm_level_to_nterms(tree, level):
        return ORDER

    wrangler = FMMLibExpansionWrangler(
        trav.tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)

    # Compute FMM using shared memory parallelism
    from boxtree.fmm import drive_fmm
    pot_fmm = drive_fmm(trav, wrangler, sources_weights) * 2 * np.pi

    now = time.time()
    print("Shared memory FMM " + str(now - last_time))
    # print(la.norm(pot_fmm - pot_naive, ord=2))

comm.barrier()
start_time = last_time = time.time()

# Compute FMM using distributed memory parallelism
workload_weight = WorkloadWeight(
    direct=15,
    m2l=ORDER*ORDER,
    m2p=ORDER*ORDER,
    p2l=ORDER*ORDER,
    multipole=ORDER*ORDER*5
)

local_tree, local_data, box_bounding_box = generate_local_tree(trav)
trav_local, trav_global = generate_local_travs(local_tree, box_bounding_box)

comm.barrier()
last_time = time.time()


def fmm_level_to_nterms(tree, level):
    return ORDER


from boxtree.distributed import queue
local_wrangler = (
    DistributedFMMLibExpansionWranglerCodeContainer()
    .get_wrangler(queue, local_tree, HELMHOLTZ_K, ORDER))

if rank == 0:
    global_wrangler = FMMLibExpansionWrangler(
        trav.tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)
else:
    global_wrangler = None

pot_dfmm = drive_dfmm(
    local_wrangler, trav_local, global_wrangler, trav_global, sources_weights,
    local_data
)

print("Distributed FMM " + str(time.time() - last_time))

if rank == 0:
    print("Total time " + str(time.time() - start_time))
    print((la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf) /
           la.norm(pot_fmm, ord=np.inf)))
