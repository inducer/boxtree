import numpy as np
import sys
from mpi4py import MPI

# Parameters
dims = 2
nsources = 300
ntargets = 100
dtype = np.float64

# Get the current rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialization
trav = None
sources_weights = None
wrangler = None

# Generate particles and run shared-memory parallelism on rank 0
if rank == 0:
    # Configure PyOpenCL
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Generate random particles and source weights
    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = p_normal(queue, ntargets, dims, dtype, seed=18) + np.array([2, 0, 0])[:dims]

    from boxtree.tools import particle_array_to_host
    sources_host = particle_array_to_host(sources)
    targets_host = particle_array_to_host(targets)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=20)
    sources_weights = rng.uniform(queue, nsources, dtype=np.float64).get()

    # Display sources and targets
    if "--display" in sys.argv:
        import matplotlib.pyplot as plt
        plt.plot(sources_host[:, 0], sources_host[:, 1], "bo")
        plt.plot(targets_host[:, 0], targets_host[:, 1], "ro")
        plt.show()

    # Calculate potentials using direct evaluation
    import numpy.linalg as la
    distances = la.norm(sources_host.reshape(1, nsources, 2) - \
                        targets_host.reshape(ntargets, 1, 2), 
                        ord=2, axis=2)
    pot_naive = np.sum(-np.log(distances)*sources_weights, axis=1)

    # Build the tree and interaction lists
    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)
    tree, _ = tb(queue, sources, targets=targets, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree, debug=True)
    trav = trav.get(queue=queue)

    # Get pyfmmlib expansion wrangler
    from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
    def fmm_level_to_nterms(tree, level):
        return 20
    wrangler = FMMLibExpansionWrangler(trav.tree, 0, fmm_level_to_nterms=fmm_level_to_nterms)

    # Compute FMM using shared memory parallelism
    from boxtree.fmm import drive_fmm
    pot_fmm = drive_fmm(trav, wrangler, sources_weights)* 2 * np.pi
    print(la.norm(pot_fmm - pot_naive, ord=2))

# Compute FMM using distributed memory parallelism
from boxtree.distributed import drive_dfmm
# Note: The drive_dfmm interface works as follows: 
# Rank 0 passes the correct trav, wrangler, and sources_weights
# All other ranks pass None to these arguments
pot_dfmm = drive_dfmm(trav, wrangler, sources_weights)
