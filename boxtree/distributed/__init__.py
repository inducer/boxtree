from __future__ import division

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

    .. automethod:: drive_dfmm

Distributed Algorithm Overview
------------------------------

1. Construct the global tree and traversal lists on the root rank and broadcast to
   all worker ranks.
2. Partition boxes into disjoint sets, where the number of sets is the number of MPI
   ranks. (See :ref:`partition-boxes`)
3. Each rank constructs the local tree and traversal lists independently, according
   to the partition. (See :ref:`construct-local-tree-traversal`)
4. Distribute source weights from the root rank to all worker ranks. (See
   :ref:`distribute-source-weights`)
5. Each rank independently forms multipole expansions from the leaf nodes of the
   local tree and propagates the partial multipole expansions upwards.
6. Communicate multipole expansions so that all ranks have the complete multipole
   expansions needed.
7. Each ranks indepedently forms local expansions, propagates the local expansions
   downwards, and evaluate potentials of target points in its partition. The
   calculated potentials are then assembled on the root rank.

For step 5-7, see :ref:`distributed-fmm-evaluation`.

Note that step 4-7 may be repeated multiple times with the same tree and traversal
object built from step 1-3. For example, when solving a PDE, step 4-7 is executed
for each GMRES iteration.

The next sections will cover the interfaces of these steps.

.. _partition-boxes:

Partition Boxes
---------------

.. autofunction:: boxtree.distributed.partition.partition_work

.. autofunction:: boxtree.distributed.partition.get_boxes_mask

.. _construct-local-tree-traversal:

Construct Local Tree and Traversal
----------------------------------

.. autofunction:: boxtree.distributed.local_tree.generate_local_tree

.. autofunction:: boxtree.distributed.local_traversal.generate_local_travs

.. _distribute-source-weights:

Distribute source weights
-------------------------

.. autofunction:: boxtree.distributed.calculation.DistributedExpansionWrangler\
.distribute_source_weights

.. _distributed-fmm-evaluation:

Distributed FMM Evaluation
--------------------------

The distributed version of the FMM evaluation shares the same interface as the
shared-memory version. To evaluate FMM in distributed manner, set ``comm`` to
a valid communicator in :func:`boxtree.fmm.drive_fmm`.

"""

from mpi4py import MPI
import numpy as np
from boxtree.cost import FMMCostModel

__all__ = ["DistributedFMMRunner"]

MPITags = dict(
    DIST_WEIGHT=1,
    GATHER_POTENTIALS=2,
    REDUCE_POTENTIALS=3,
    REDUCE_INDICES=4
)


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


class DistributedFMMRunner(object):
    def __init__(self, queue, global_tree_dev,
                 traversal_builder,
                 wrangler_factory,
                 calibration_params=None, comm=MPI.COMM_WORLD):
        """Constructor of the ``DistributedFMMRunner`` class.

        This constructor distributes the global tree from the root rank to each
        worker rank.

        :arg global_tree_dev: a :class:`boxtree.Tree` object in device memory.
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

        self.comm = comm
        mpi_rank = comm.Get_rank()

        # {{{ Broadcast global tree

        global_tree = None
        if mpi_rank == 0:
            global_tree = global_tree_dev.get(queue)
        global_tree = comm.bcast(global_tree, root=0)
        global_tree_dev = global_tree.to_device(queue).with_queue(queue)

        global_trav_dev, _ = traversal_builder(queue, global_tree_dev)
        global_trav = global_trav_dev.get(queue)

        # }}}

        # {{{ Partiton work

        cost_model = FMMCostModel()

        if calibration_params is None:
            # Use default calibration parameters if not supplied
            # TODO: should replace the default calibration params with a more
            # accurate one
            calibration_params = \
                FMMCostModel.get_unit_calibration_params()

        # We need to construct a wrangler in order to access `level_nterms`
        global_wrangler = wrangler_factory(global_trav, global_trav)

        cost_per_box = cost_model.cost_per_box(
            # Currently only pyfmmlib has `level_nterms` field.
            # See https://gitlab.tiker.net/inducer/boxtree/-/issues/25.
            queue, global_trav_dev, global_wrangler.level_nterms,
            calibration_params
        ).get()

        from boxtree.distributed.partition import partition_work
        responsible_boxes_list = partition_work(
            cost_per_box, global_trav, comm.Get_size())

        # It is assumed that, even if each rank computes `responsible_boxes_list`
        # independently, it should be the same across ranks.

        # }}}

        # {{{ Compute local tree

        from boxtree.distributed.local_tree import generate_local_tree
        self.local_tree, self.src_idx, self.tgt_idx = generate_local_tree(
            queue, global_trav, responsible_boxes_list[mpi_rank])

        # }}}

        # {{ Gather source indices and target indices of each rank

        self.src_idx_all_ranks = comm.gather(self.src_idx, root=0)
        self.tgt_idx_all_ranks = comm.gather(self.tgt_idx, root=0)

        # }}}

        # {{{ Compute traversal object on each rank

        from boxtree.distributed.local_traversal import generate_local_travs
        local_trav = generate_local_travs(
            queue, self.local_tree, traversal_builder,
            box_bounding_box={
                "min": global_trav.box_target_bounding_box_min,
                "max": global_trav.box_target_bounding_box_max
            }
        )

        # }}}

        self.wrangler = wrangler_factory(local_trav.get(None), global_trav)

    def drive_dfmm(self, source_weights, timing_data=None):
        """Calculate potentials at target points.
        """
        from boxtree.fmm import drive_fmm
        return drive_fmm(
            self.wrangler, source_weights,
            timing_data=timing_data,
            global_src_idx_all_ranks=self.src_idx_all_ranks,
            global_tgt_idx_all_ranks=self.tgt_idx_all_ranks)
