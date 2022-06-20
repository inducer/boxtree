__copyright__ = "Copyright (C) 2013 Andreas Kloeckner \
                 Copyright (C) 2018 Hao Gao"

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

__doc__ = """
High-level Point FMM Interface
------------------------------

To perform point-FMM, first construct a
:class:`boxtree.distributed.DistributedFMMRunner` object. The constructor will
distribute the necessary information from the root rank to all worker ranks. Then,
the :meth:`boxtree.distributed.DistributedFMMRunner.drive_dfmm` can be used for
launching FMM.

.. autoclass:: boxtree.distributed.DistributedFMMRunner


Distributed Algorithm Overview
------------------------------

1. Construct the global tree and traversal lists on the root rank and broadcast to
   all worker ranks.
2. Partition boxes into disjoint sets, where the number of sets is the number of MPI
   ranks. (See :ref:`partition-boxes`)
3. Each rank constructs the local tree and traversal lists independently, according
   to the partition. (See :ref:`construct-local-tree-traversal`)
4. Distribute source weights from the root rank to all worker ranks. (See
   :ref:`distributed-wrangler`)
5. Each rank independently forms multipole expansions from the leaf nodes of the
   local tree and propagates the partial multipole expansions upwards.
6. Communicate multipole expansions so that all ranks have the complete multipole
   expansions needed.
7. Each ranks indepedently forms local expansions, propagates the local expansions
   downwards, and evaluate potentials of target points in its partition. The
   calculated potentials are then assembled on the root rank.

For step 5-7, see :ref:`distributed-fmm-evaluation`.

Note that step 4-7 may be repeated multiple times with the same tree and traversal
object built from step 1-3. For example, when iteratively solving a PDE, step 4-7 is
executed for each iteration of the linear solver.

The next sections will cover the interfaces of these steps.

.. _partition-boxes:

Partition Boxes
---------------

.. autofunction:: boxtree.distributed.partition.partition_work

.. autoclass:: boxtree.distributed.partition.BoxMasks

.. autofunction:: boxtree.distributed.partition.get_box_masks

.. _construct-local-tree-traversal:

Construct Local Tree and Traversal
----------------------------------

.. autoclass:: boxtree.distributed.local_tree.LocalTree

.. autofunction:: boxtree.distributed.local_tree.generate_local_tree

.. autofunction:: boxtree.distributed.local_traversal.generate_local_travs

.. _distributed-wrangler:

Distributed Wrangler
--------------------

.. autoclass:: boxtree.distributed.calculation.DistributedExpansionWrangler

.. _distributed-fmm-evaluation:

Distributed FMM Evaluation
--------------------------

The distributed version of the FMM evaluation shares the same interface as the
shared-memory version. To evaluate FMM in a distributed manner, use a subclass
of :class:`boxtree.distributed.calculation.DistributedExpansionWrangler` in
:func:`boxtree.fmm.drive_fmm`.

"""

from mpi4py import MPI
import numpy as np
import pyopencl as cl
import pyopencl.array
from enum import IntEnum
import warnings
from boxtree.cost import FMMCostModel

__all__ = ["DistributedFMMRunner"]


class MPITags(IntEnum):
    DIST_WEIGHT = 1
    GATHER_POTENTIALS = 2
    REDUCE_POTENTIALS = 3
    REDUCE_INDICES = 4


def dtype_to_mpi(dtype):
    """ This function translates a numpy datatype into the corresponding type used in
    mpi4py.
    """
    if hasattr(MPI, "_typedict"):
        mpi_type = MPI._typedict[np.dtype(dtype).char]
    elif hasattr(MPI, "__TypeDict__"):
        mpi_type = MPI.__TypeDict__[np.dtype(dtype).char]
    else:
        raise RuntimeError("There is no dictionary to translate from Numpy dtype to "
                           "MPI type")
    return mpi_type


def construct_distributed_wrangler(
        queue, global_tree, traversal_builder, wrangler_factory,
        calibration_params, comm):
    """Helper function for constructing the distributed wrangler on each rank.

    Note: This function needs to be called collectively on all ranks.
    """

    mpi_rank = comm.Get_rank()

    # `tree_in_device_memory` is True if the global tree is in the device memory
    # `tree_in_device_memory` is False if the global tree is in the host memory
    #
    # Note that at this point, only the root rank has the valid global tree, so
    # we test `tree_in_device_memory` on the root rank and broadcast to all
    # worker ranks.
    tree_in_device_memory = None
    if mpi_rank == 0:
        tree_in_device_memory = isinstance(global_tree.targets[0], cl.array.Array)
    tree_in_device_memory = comm.bcast(tree_in_device_memory, root=0)

    # {{{ Broadcast the global tree

    global_tree_host = None
    if mpi_rank == 0:
        if tree_in_device_memory:
            global_tree_host = global_tree.get(queue)
        else:
            global_tree_host = global_tree

    global_tree_host = comm.bcast(global_tree_host, root=0)

    global_tree_dev = None
    if mpi_rank == 0 and tree_in_device_memory:
        global_tree_dev = global_tree
    else:
        global_tree_dev = global_tree_host.to_device(queue)
    global_tree_dev = global_tree_dev.with_queue(queue)

    global_trav_dev, _ = traversal_builder(queue, global_tree_dev)
    global_trav_host = global_trav_dev.get(queue)

    if tree_in_device_memory:
        global_trav = global_trav_dev
    else:
        global_trav = global_trav_host

    # }}}

    # {{{ Partition work

    cost_per_box = None

    if mpi_rank == 0:
        cost_model = FMMCostModel()

        if calibration_params is None:
            # Use default calibration parameters if not supplied
            # TODO: should replace the default calibration params with a more
            # accurate one
            warnings.warn("Calibration parameters for the cost model are not "
                        "supplied. The default one will be used.")
            calibration_params = \
                FMMCostModel.get_unit_calibration_params()

        # We need to construct a wrangler in order to access `level_orders`
        global_wrangler = wrangler_factory(global_trav, global_trav)

        cost_per_box = cost_model.cost_per_box(
            queue, global_trav_dev, global_wrangler.level_orders,
            calibration_params
        ).get()

    from boxtree.distributed.partition import partition_work
    responsible_boxes_list = partition_work(cost_per_box, global_trav_host, comm)

    # }}}

    # {{{ Compute local tree

    from boxtree.distributed.local_tree import generate_local_tree
    local_tree, src_idx, tgt_idx = generate_local_tree(
        queue, global_trav_host, responsible_boxes_list, comm)

    # }}}

    # {{ Gather source indices and target indices of each rank

    src_idx_all_ranks = comm.gather(src_idx, root=0)
    tgt_idx_all_ranks = comm.gather(tgt_idx, root=0)

    # }}}

    # {{{ Compute traversal object on each rank

    from boxtree.distributed.local_traversal import generate_local_travs
    local_trav_dev = generate_local_travs(queue, local_tree, traversal_builder)

    if not tree_in_device_memory:
        local_trav = local_trav_dev.get(queue=queue)
    else:
        local_trav = local_trav_dev.with_queue(None)

    # }}}

    wrangler = wrangler_factory(local_trav, global_trav)

    return wrangler, src_idx_all_ranks, tgt_idx_all_ranks


class DistributedFMMRunner:
    """Helper class for setting up and running distributed point FMM.

    .. automethod:: __init__
    .. automethod:: drive_dfmm
    """
    def __init__(self, queue, global_tree,
                 traversal_builder,
                 wrangler_factory,
                 calibration_params=None, comm=MPI.COMM_WORLD):
        """Construct a DistributedFMMRunner object.

        :arg global_tree: a :class:`boxtree.Tree` object. This tree could live in the
            host or the device memory, depending on the wrangler. This argument is
            only significant on the root rank.
        :arg traversal_builder: an object which, when called, takes a
            :class:`pyopencl.CommandQueue` object and a :class:`boxtree.Tree` object,
            and generates a :class:`boxtree.traversal.FMMTraversalInfo` object from
            the tree using the command queue.
        :arg wrangler_factory: an object which, when called, takes the local
            traversal and the global traversal objects and returns an
            :class:`boxtree.fmm.ExpansionWranglerInterface` object.
        :arg calibration_params: Calibration parameters for the cost model,
            if supplied. The cost model is used for estimating the execution time of
            each box, which is used for improving load balancing.
        :arg comm: MPI communicator.
        """
        self.wrangler, self.src_idx_all_ranks, self.tgt_idx_all_ranks = \
            construct_distributed_wrangler(
                queue, global_tree, traversal_builder, wrangler_factory,
                calibration_params, comm)

    def drive_dfmm(self, source_weights, timing_data=None):
        """Calculate potentials at target points.
        """
        from boxtree.fmm import drive_fmm
        return drive_fmm(
            self.wrangler, source_weights,
            timing_data=timing_data,
            global_src_idx_all_ranks=self.src_idx_all_ranks,
            global_tgt_idx_all_ranks=self.tgt_idx_all_ranks)
