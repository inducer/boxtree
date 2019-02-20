import numpy as np
import pyopencl as cl
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from boxtree.tools import ConstantOneExpansionWrangler as \
    ConstantOneExpansionWranglerBase
import logging
import os
import pytest

# Note: Do not import mpi4py.MPI object at the module level, because OpenMPI does not
# support recursive invocations.

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logging.getLogger("boxtree.distributed").setLevel(logging.INFO)


def _test_against_shared(dims, nsources, ntargets, dtype):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize arguments for worker processes
    d_trav = None
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
        queue, d_trav, distributed_expansion_wrangler_factory, comm=comm)
    pot_dfmm = distribued_fmm_info.drive_dfmm(sources_weights)

    if rank == 0:
        error = (la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf)
                 / la.norm(pot_fmm, ord=np.inf))
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
    newenv["OMP_NUM_THREADS"] = "1"

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

class ConstantOneExpansionWrangler(ConstantOneExpansionWranglerBase):

    def __init__(self, tree):
        super(ConstantOneExpansionWrangler, self).__init__(tree)
        self.level_nterms = np.ones(tree.nlevels, dtype=np.int32)

# }}}


def _test_constantone(dims, nsources, ntargets, dtype):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialization
    d_trav = None
    sources_weights = None

    # Configure PyOpenCL
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    if rank == 0:

        # Generate random particles
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = (p_normal(queue, ntargets, dims, dtype, seed=18)
                   + np.array([2, 0, 0])[:dims])

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

    def constantone_expansion_wrangler_factory(tree):
        return ConstantOneExpansionWrangler(tree)

    from boxtree.distributed import DistributedFMMInfo
    distributed_fmm_info = DistributedFMMInfo(
        queue, d_trav, constantone_expansion_wrangler_factory, comm=MPI.COMM_WORLD
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
    newenv["OMP_NUM_THREADS"] = "1"

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
