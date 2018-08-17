import numpy as np
import pyopencl as cl
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import logging
import os
import pytest

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logging.getLogger("boxtree.distributed").setLevel(logging.INFO)


def _test_against_shared(dims, nsources, ntargets, dtype):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize arguments for worker processes
    trav = None
    sources_weights = None
    helmholtz_k = 0

    # Configure PyOpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    def fmm_level_to_nterms(tree, level):
        return max(level, 3)

    # Generate particles and run shared-memory parallelism on rank 0
    if rank == 0:

        # Generate random particles and source weights
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = p_normal(queue, ntargets, dims, dtype, seed=18)

        from pyopencl.clrandom import PhiloxGenerator
        rng = PhiloxGenerator(queue.context, seed=20)
        sources_weights = rng.uniform(queue, nsources, dtype=np.float64).get()

        from pyopencl.clrandom import PhiloxGenerator
        rng = PhiloxGenerator(queue.context, seed=22)
        target_radii = rng.uniform(
            queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

        # Build the tree and interaction lists
        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)
        tree, _ = tb(queue, sources, targets=targets, target_radii=target_radii,
                     stick_out_factor=0.25, max_particles_in_box=30, debug=True)

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx, well_sep_is_n_away=2)
        d_trav, _ = tg(queue, tree, debug=True)
        trav = d_trav.get(queue=queue)

        # Get pyfmmlib expansion wrangler
        wrangler = FMMLibExpansionWrangler(
            trav.tree, helmholtz_k, fmm_level_to_nterms=fmm_level_to_nterms)

        # Compute FMM using shared memory parallelism
        from boxtree.fmm import drive_fmm
        pot_fmm = drive_fmm(trav, wrangler, sources_weights) * 2 * np.pi

    # Compute FMM using distributed memory parallelism

    def distributed_expansion_wrangler_factory(tree):
        from boxtree.distributed.calculation import \
                DistributedFMMLibExpansionWrangler

        return DistributedFMMLibExpansionWrangler(
            queue, tree, helmholtz_k, fmm_level_to_nterms=fmm_level_to_nterms)

    from boxtree.distributed import DistributedFMMInfo
    distribued_fmm_info = DistributedFMMInfo(
        queue, trav, distributed_expansion_wrangler_factory, comm=comm)
    pot_dfmm = distribued_fmm_info.drive_dfmm(sources_weights)

    if rank == 0:
        error = (la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf) /
                 la.norm(pot_fmm, ord=np.inf))
        print(error)
        assert error < 1e-14


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, dims, nsources, ntargets", [
    (4, 3, 10000, 10000)
])
def test_against_shared(num_processes, dims, nsources, ntargets):
    pytest.importorskip("mpi4py")

    newenv = os.environ.copy()
    newenv["PYTEST"] = "1"
    newenv["dims"] = str(dims)
    newenv["nsources"] = str(nsources)
    newenv["ntargets"] = str(ntargets)

    import subprocess
    import sys
    subprocess.run([
        "mpiexec", "-np", str(num_processes),
        "-x", "PYTEST", "-x", "dims", "-x", "nsources", "-x", "ntargets",
        # https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
        sys.executable, "-m", "mpi4py.run", __file__],
        env=newenv,
        check=True
    )


# {{{ Constantone expansion wrangler

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

    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_nonsiblings_by_level,
            mpole_exps):
        pot = self.potential_zeros()

        for level, ssn in enumerate(from_sep_smaller_nonsiblings_by_level):
            for itgt_box, tgt_ibox in \
                    enumerate(target_boxes_by_source_level[level]):
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

# }}}


def _test_constantone(dims, nsources, ntargets, dtype):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialization
    trav = None
    sources_weights = None

    # Configure PyOpenCL
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    print(queue.context.devices)

    if rank == 0:

        # Generate random particles
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = (p_normal(queue, ntargets, dims, dtype, seed=18) +
                   np.array([2, 0, 0])[:dims])

        # Constant one source weights
        sources_weights = np.ones((nsources,), dtype=dtype)

        # Build the global tree
        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)
        tree, _ = tb(queue, sources, targets=targets, max_particles_in_box=30,
                     debug=True)

        # Build global interaction lists
        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        d_trav, _ = tg(queue, tree, debug=True)
        trav = d_trav.get(queue=queue)

    def constantone_expansion_wrangler_factory(tree):
        return ConstantOneExpansionWrangler(tree)

    from boxtree.distributed import DistributedFMMInfo
    distributed_fmm_info = DistributedFMMInfo(
        queue, trav, constantone_expansion_wrangler_factory, comm=MPI.COMM_WORLD
    )

    pot_dfmm = distributed_fmm_info.drive_dfmm(
        sources_weights, _communicate_mpoles_via_allreduce=True
    )

    if rank == 0:
        assert (np.all(pot_dfmm == nsources))


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, dims, nsources, ntargets", [
    (4, 3, 10000, 10000)
])
def test_constantone(num_processes, dims, nsources, ntargets):
    pytest.importorskip("mpi4py")

    newenv = os.environ.copy()
    newenv["PYTEST"] = "2"
    newenv["dims"] = str(dims)
    newenv["nsources"] = str(nsources)
    newenv["ntargets"] = str(ntargets)

    import subprocess
    import sys
    subprocess.run([
        "mpiexec", "-np", str(num_processes),
        "-x", "PYTEST", "-x", "dims", "-x", "nsources", "-x", "ntargets",
        # https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
        sys.executable, "-m", "mpi4py.run", __file__],
        env=newenv,
        check=True
    )


if __name__ == "__main__":

    dtype = np.float64

    if "PYTEST" in os.environ:
        if os.environ["PYTEST"] == "1":
            # Run "test_against_shared" test case
            dims = int(os.environ["dims"])
            nsources = int(os.environ["nsources"])
            ntargets = int(os.environ["ntargets"])

            _test_against_shared(dims, nsources, ntargets, dtype)

        elif os.environ["PYTEST"] == "2":
            # Run "test_constantone" test case
            dims = int(os.environ["dims"])
            nsources = int(os.environ["nsources"])
            ntargets = int(os.environ["ntargets"])

            _test_constantone(dims, nsources, ntargets, dtype)

    else:
        import sys

        if len(sys.argv) > 1:

            # You can test individual routines by typing
            # $ python test_distributed.py 'test_constantone(4, 3, 10000, 10000)'
            exec(sys.argv[1])

        elif len(sys.argv) == 1:

            # Run against_shared test case with default parameter
            dims = 3
            nsources = 10000
            ntargets = 10000

            _test_against_shared(dims, nsources, ntargets, dtype)
