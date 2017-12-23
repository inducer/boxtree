import numpy as np
from mpi4py import MPI
from boxtree.distributed import generate_local_tree, generate_local_travs, drive_dfmm


class ConstantOneExpansionWrangler(object):
    """This implements the 'analytical routines' for a Green's function that is
    constant 1 everywhere. For 'charges' of 'ones', this should get every particle
    a copy of the particle count.
    """

    def __init__(self, tree):
        self.tree = tree

    def multipole_expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    local_expansion_zeros = multipole_expansion_zeros

    def potential_zeros(self):
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_target_counts_nonchild[ibox])

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        mpoles = self.multipole_expansion_zeros()
        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        tree = self.tree

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        mpoles[ibox] += mpoles[child]

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights):
        pot = self.potential_zeros()

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            #print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum

        return pot

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0
            #print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]

            local_exps[tgt_ibox] += contrib

        return local_exps

    def eval_multipoles(self, level_start_target_box_nrs, target_boxes,
            from_sep_smaller_nonsiblings_by_level, mpole_exps):
        pot = self.potential_zeros()

        for ssn in from_sep_smaller_nonsiblings_by_level:
            for itgt_box, tgt_ibox in enumerate(target_boxes):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                contrib = 0

                start, end = ssn.starts[itgt_box:itgt_box+2]
                for src_ibox in ssn.lists[start:end]:
                    contrib += mpole_exps[src_ibox]

                pot[tgt_pslice] += contrib

        return pot

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            #print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)

                contrib += np.sum(src_weights[src_pslice])

            local_exps[tgt_ibox] += contrib

        return local_exps

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.potential_zeros()

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot

    def finalize_potentials(self, potentials):
        return potentials


# Parameters
dims = 2
nsources = 100000
ntargets = 100000
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
    # sources_weights = rng.uniform(queue, nsources, dtype=np.float64).get()
    sources_weights = np.ones((nsources,))

    # Build the tree and interaction lists
    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)
    tree, _ = tb(queue, sources, targets=targets, max_particles_in_box=30,
                 debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    d_trav, _ = tg(queue, tree, debug=True)
    trav = d_trav.get(queue=queue)

    wrangler = ConstantOneExpansionWrangler(trav.tree)

    # Compute FMM using shared memory parallelism
    from boxtree.fmm import drive_fmm
    pot_fmm = drive_fmm(trav, wrangler, sources_weights)

local_tree, local_src_weights, local_target, box_bounding_box = \
    generate_local_tree(trav, sources_weights)

trav_local, trav_global = generate_local_travs(local_tree, local_src_weights,
                                               box_bounding_box)

local_wrangler = ConstantOneExpansionWrangler(local_tree)

if rank == 0:
    global_wrangler = ConstantOneExpansionWrangler(trav.tree)
else:
    global_wrangler = None

pot_dfmm = drive_dfmm(
    local_wrangler, trav_local, trav_global, local_src_weights, global_wrangler,
    local_target["mask"], local_target["scan"], local_target["size"],
    _communicate_mpoles_via_allreduce=True
)

if rank == 0:
    assert(np.all(pot_fmm == nsources))
    assert(np.all(pot_dfmm == nsources))
