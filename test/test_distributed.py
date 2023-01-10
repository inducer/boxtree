__copyright__ = "Copyright (C) 2021 Hao Gao"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import os
import sys

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from boxtree.array_context import PytestPyOpenCLArrayContextFactory  # noqa: F401
from boxtree.array_context import _acf
from boxtree.constant_one import (
    ConstantOneExpansionWrangler as ConstantOneExpansionWranglerBase,
    ConstantOneTreeIndependentDataForWrangler)
from boxtree.pyfmmlib_integration import (
    FMMLibExpansionWrangler, FMMLibTreeIndependentDataForWrangler, Kernel)


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

# NOTE: Do not import mpi4py.MPI object at the module level, because OpenMPI
# does not support recursive invocations.


def _cachedir():
    import tempfile
    return tempfile.mkdtemp(prefix="boxtree-pytest-")


# {{{ test_against_shared

def _test_against_shared(
        tmp_cache_basedir,
        dims, nsources, ntargets, dtype, communicate_mpoles_via_allreduce=False):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_cache_dir = os.path.join(tmp_cache_basedir, f"rank_{rank:03d}")

    # Initialize arguments for worker processes
    global_tree_host = None
    sources_weights = np.empty(0, dtype=dtype)
    helmholtz_k = 0

    def fmm_level_to_order(tree, level):
        return max(level, 3)

    from unittest.mock import patch
    with patch.dict(os.environ, {"XDG_CACHE_HOME": rank_cache_dir}):
        actx = _acf()

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(actx.context, well_sep_is_n_away=2)

        tree_indep = FMMLibTreeIndependentDataForWrangler(
            dims, Kernel.HELMHOLTZ if helmholtz_k else Kernel.LAPLACE)

        # Generate particles and run shared-memory parallelism on rank 0
        if rank == 0:
            # Generate random particles and source weights
            from boxtree.tools import make_normal_particle_array as p_normal
            sources = p_normal(actx.queue, nsources, dims, dtype, seed=15)
            targets = p_normal(actx.queue, ntargets, dims, dtype, seed=18)

            rng = np.random.default_rng(20)
            sources_weights = rng.uniform(0.0, 1.0, (nsources,))
            target_radii = rng.uniform(0.0, 0.05, (ntargets,))

            # Build the tree and interaction lists
            from boxtree import TreeBuilder
            tb = TreeBuilder(actx.context)
            global_tree_dev, _ = tb(
                actx.queue, sources, targets=targets, target_radii=target_radii,
                stick_out_factor=0.25, max_particles_in_box=30, debug=True)

            d_trav, _ = tg(actx.queue, global_tree_dev, debug=True)
            global_traversal_host = d_trav.get(queue=actx.queue)
            global_tree_host = global_traversal_host.tree

            # Get pyfmmlib expansion wrangler
            wrangler = FMMLibExpansionWrangler(
                    tree_indep, global_traversal_host,
                    fmm_level_to_order=fmm_level_to_order)

            # Compute FMM with one MPI rank
            from boxtree.fmm import drive_fmm
            pot_fmm = drive_fmm(wrangler, [sources_weights]) * 2 * np.pi

        # Compute FMM using the distributed implementation

        def wrangler_factory(local_traversal, global_traversal):
            from boxtree.distributed.calculation import (
                DistributedFMMLibExpansionWrangler)

            return DistributedFMMLibExpansionWrangler(
                actx.context, comm, tree_indep, local_traversal, global_traversal,
                fmm_level_to_order=fmm_level_to_order,
                communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)

        from boxtree.distributed import DistributedFMMRunner
        distribued_fmm_info = DistributedFMMRunner(
            actx.queue, global_tree_host, tg, wrangler_factory, comm=comm)

        timing_data = {}
        pot_dfmm = distribued_fmm_info.drive_dfmm(
                    [sources_weights], timing_data=timing_data)
        assert timing_data

    # Uncomment the following section to print the time taken of each stage
    """
    if rank == 1:
        from pytools import Table
        table = Table()
        table.add_row(["stage", "time (s)"])
        for stage in timing_data:
            table.add_row([stage, "%.2f" % timing_data[stage]["wall_elapsed"]])
        print(table)
    """

    if rank == 0:
        error = (la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf)
                 / la.norm(pot_fmm, ord=np.inf))
        print(error)
        assert error < 1e-14


@pytest.mark.mpi
@pytest.mark.parametrize(
    "num_processes, dims, nsources, ntargets, communicate_mpoles_via_allreduce", [
        (4, 3, 10000, 10000, True),
        (4, 3, 10000, 10000, False)
    ]
)
def test_against_shared(
        tmp_path, num_processes, dims, nsources, ntargets,
        communicate_mpoles_via_allreduce):
    pytest.importorskip("mpi4py")

    from boxtree.tools import run_mpi
    run_mpi(__file__, num_processes, {
        "PYTEST": "shared",
        "dims": dims,
        "nsources": nsources,
        "ntargets": ntargets,
        "OMP_NUM_THREADS": 1,
        "tmp_cache_basedir": tmp_path / "boxtree_distributed_test",
        "communicate_mpoles_via_allreduce": communicate_mpoles_via_allreduce
        })

# }}}


# {{{ test_constantone

def _test_constantone(tmp_cache_basedir, dims, nsources, ntargets, dtype):
    from boxtree.distributed.calculation import DistributedExpansionWrangler

    class ConstantOneExpansionWrangler(
            ConstantOneExpansionWranglerBase, DistributedExpansionWrangler):
        def __init__(
                self, queue, comm, tree_indep, local_traversal, global_traversal):
            DistributedExpansionWrangler.__init__(
                self, queue, comm, global_traversal, False,
                communicate_mpoles_via_allreduce=True)
            ConstantOneExpansionWranglerBase.__init__(
                self, tree_indep, local_traversal)
            self.level_orders = np.ones(local_traversal.tree.nlevels, dtype=np.int32)

        def reorder_sources(self, source_array):
            if self.comm.Get_rank() == 0:
                return source_array[self.global_traversal.tree.user_source_ids]
            else:
                return None

        def reorder_potentials(self, potentials):
            if self.comm.Get_rank() == 0:
                return potentials[self.global_traversal.tree.sorted_target_ids]
            else:
                return None

    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_cache_dir = os.path.join(tmp_cache_basedir, f"rank_{rank:03d}")

    # Initialization
    tree = None
    sources_weights = np.empty(0, dtype=dtype)

    from unittest.mock import patch
    with patch.dict(os.environ, {"XDG_CACHE_HOME": rank_cache_dir}):
        actx = _acf()

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(actx.context)

        if rank == 0:

            # Generate random particles
            from boxtree.tools import make_normal_particle_array as p_normal
            sources = p_normal(actx.queue, nsources, dims, dtype, seed=15)
            targets = (p_normal(actx.queue, ntargets, dims, dtype, seed=18)
                       + np.array([2, 0, 0])[:dims])

            # Constant one source weights
            sources_weights = np.ones((nsources,), dtype=dtype)

            # Build the global tree
            from boxtree import TreeBuilder
            tb = TreeBuilder(actx.context)
            tree, _ = tb(
                    actx.queue, sources, targets=targets, max_particles_in_box=30,
                    debug=True)
            tree = tree.get(actx.queue)

        tree_indep = ConstantOneTreeIndependentDataForWrangler()

        def wrangler_factory(local_traversal, global_traversal):
            return ConstantOneExpansionWrangler(
                    actx.queue, comm, tree_indep, local_traversal, global_traversal)

        from boxtree.distributed import DistributedFMMRunner
        distributed_fmm_info = DistributedFMMRunner(
            actx.queue, tree, tg, wrangler_factory, comm=MPI.COMM_WORLD)

        pot_dfmm = distributed_fmm_info.drive_dfmm([sources_weights])

    if rank == 0:
        assert (np.all(pot_dfmm == nsources))


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, dims, nsources, ntargets", [
    (4, 3, 10000, 10000)
])
def test_constantone(tmp_path, num_processes, dims, nsources, ntargets):
    pytest.importorskip("mpi4py")

    from boxtree.tools import run_mpi
    run_mpi(__file__, num_processes, {
        "PYTEST": "constantone",
        "dims": dims,
        "nsources": nsources,
        "ntargets": ntargets,
        "OMP_NUM_THREADS": 1,
        "tmp_cache_basedir": tmp_path / "boxtree_distributed_test",
        "communicate_mpoles_via_allreduce": False
        })

# }}}


if __name__ == "__main__":
    dtype = np.float64
    tmp_cache_basedir = os.environ.get("tmp_cache_basedir", _cachedir())

    if "PYTEST" in os.environ:
        dims = int(os.environ["dims"])
        nsources = int(os.environ["nsources"])
        ntargets = int(os.environ["ntargets"])
        communicate_mpoles_via_allreduce = (
                True if os.environ["communicate_mpoles_via_allreduce"] == "True"
                else False)

        if os.environ["PYTEST"] == "shared":
            _test_against_shared(
                tmp_cache_basedir,
                dims, nsources, ntargets, dtype,
                communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)
        elif os.environ["PYTEST"] == "constantone":
            _test_constantone(tmp_cache_basedir, dims, nsources, ntargets, dtype)
    else:
        if len(sys.argv) > 1:

            # You can test individual routines by typing
            # $ python test_distributed.py 'test_constantone(
            #       tmp_cache_basedir, 4, 3, 10000, 10000)'
            exec(sys.argv[1])

        elif len(sys.argv) == 1:

            # Run against_shared test case with default parameter
            dims = 3
            nsources = 10000
            ntargets = 10000

            _test_against_shared(tmp_cache_basedir, dims, nsources, ntargets, dtype)
