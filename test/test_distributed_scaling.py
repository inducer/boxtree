import numpy as np
import sys
from mpi4py import MPI
from boxtree.distributed import generate_local_tree, generate_local_travs, drive_dfmm
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import time
import pyopencl as cl
import pyopencl.array

import logging
logging.basicConfig(level=logging.INFO)

# Global parameters
DIMS = 2
DTYPE = np.float64

ORDER = 3
HELMHOLTZ_K = 0


def build_global_traversal_and_weights(ctx, nsources, ntargets):
    queue = cl.CommandQueue(ctx)

    # Generate random particles and source weights
    from boxtree.tools import make_uniform_particle_array as p_uniform
    sources = p_uniform(queue, nsources, DIMS, DTYPE, seed=15)
    targets = p_uniform(queue, ntargets, DIMS, DTYPE, seed=18)

    from boxtree.tools import particle_array_to_host
    sources_host = particle_array_to_host(sources)
    targets_host = particle_array_to_host(targets)

    sources_weights = np.ones(nsources, DTYPE)

    # Build the tree and interaction lists
    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)
    tree, _ = tb(queue, sources, targets=targets,
                 max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    return (trav, sources_weights)


def get_dfmm_stats(ctx, comm, nsources, ntargets):
    if comm.Get_rank() == 0:
        trav, sources_weights = (
            build_global_traversal_and_weights(ctx, nsources, ntargets))
    else:
        trav, sources_weights = 2 * (None,)

    comm.barrier()

    # Compute FMM using distributed memory parallelism
    local_tree, local_src_weights, local_target, box_bounding_box = (
        generate_local_tree(trav, sources_weights))

    trav_local, trav_global = (
        generate_local_travs(local_tree, local_src_weights, box_bounding_box))

    def fmm_level_to_nterms(tree, level):
        return ORDER

    from boxtree.distributed import (
        DistributedFMMLibExpansionWranglerCodeContainer, queue)

    local_wrangler = (
        DistributedFMMLibExpansionWranglerCodeContainer()
        .get_wrangler(queue, local_tree, HELMHOLTZ_K, ORDER))

    if comm.Get_rank() == 0:
        global_wrangler = FMMLibExpansionWrangler(
            trav.tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)
    else:
        global_wrangler = None

    stats = {}

    _ = drive_dfmm(
        local_wrangler, trav_local, trav_global, local_src_weights, global_wrangler,
        local_target["mask"], local_target["scan"], local_target["size"],
        _stats=stats)

    return stats


def get_mpole_communication_data(ctx, comm, nsources, ntargets):
    stats = get_dfmm_stats(ctx, comm, nsources, ntargets)

    my_mpoles_sent = np.zeros(1, int)
    max_mpoles_sent = np.zeros(1, int)

    my_mpoles_sent[0] = sum(stats["mpoles_sent_per_round"])

    comm.barrier()
    comm.Reduce(my_mpoles_sent, max_mpoles_sent, op=MPI.MAX, root=0)

    if comm.Get_rank() == 0:
        nrounds = len(stats["mpoles_sent_per_round"])
        print(f"{comm.Get_size()} & {nrounds} & {max_mpoles_sent[0]} \\\\")


if __name__ == "__main__":
    ctx = cl._csc(interactive=False)
    get_mpole_communication_data(ctx, MPI.COMM_WORLD, 100 ** 2, 100 ** 2)
