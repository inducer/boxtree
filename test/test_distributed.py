import numpy as np
import sys
from mpi4py import MPI
from boxtree.distributed import (DistributedFMMInfo,
    DistributedFMMLibExpansionWrangler)
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

# Parameters
dims = 3
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
HELMHOLTZ_K = 0


def fmm_level_to_nterms(tree, level):
    return max(level, 3)


# Generate particles and run shared-memory parallelism on rank 0
if rank == 0:
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

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    # Get pyfmmlib expansion wrangler
    wrangler = FMMLibExpansionWrangler(
        trav.tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)

    # Compute FMM using shared memory parallelism
    from boxtree.fmm import drive_fmm
    pot_fmm = drive_fmm(trav, wrangler, sources_weights) * 2 * np.pi

    # print(la.norm(pot_fmm - pot_naive, ord=2))

# Compute FMM using distributed memory parallelism
from boxtree.distributed import queue


def distributed_expansion_wrangler_factory(tree):
    return DistributedFMMLibExpansionWrangler(
        queue, tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)


distribued_fmm_info = DistributedFMMInfo(
    trav, distributed_expansion_wrangler_factory, comm=comm)
pot_dfmm = distribued_fmm_info.drive_dfmm(sources_weights)

if rank == 0:
    print((la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf) /
           la.norm(pot_fmm, ord=np.inf)))
