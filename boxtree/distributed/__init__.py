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

from mpi4py import MPI
import numpy as np
from boxtree.cost import FMMCostModel

__all__ = ['DistributedFMMRunner']

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
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(dtype).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(dtype).char]
    else:
        raise RuntimeError("There is no dictionary to translate from Numpy dtype to "
                           "MPI type")
    return mpi_type


class DistributedFMMRunner(object):
    """
    .. attribute:: global_wrangler

        An object implementing :class:`boxtree.fmm.ExpansionWranglerInterface`.
        *global_wrangler* contains reference to the global tree object on host memory
        and is used for distributing and collecting density/potential between the
        root and worker ranks.

    .. attribute:: local_wrangler

        An object implementing :class:`boxtree.fmm.ExpansionWranglerInterface`.
        *local_wrangler* contains reference to the local tree object on host memory
        and is used for local FMM operations.
    """

    def __init__(self, queue, global_tree_dev,
                 traversal_builder,
                 distributed_expansion_wrangler_factory,
                 calibration_params=None, comm=MPI.COMM_WORLD):
        """Constructor of the ``DistributedFMMRunner`` class.

        This constructor distributes the global tree from the root rank to each
        worker rank.

        :arg global_tree_dev: a :class:`boxtree.Tree` object in device memory.
        :arg traversal_builder: an object which, when called, takes a
            :class:`pyopencl.CommandQueue` object and a :class:`boxtree.Tree` object,
            and generates a :class:`boxtree.traversal.FMMTraversalInfo` object from
            the tree using the command queue.
        :arg distributed_expansion_wrangler_factory: an object which, when called,
            takes a :class:`boxtree.Tree` object and returns an object implementing
            :class:`boxtree.fmm.ExpansionWranglerInterface`.
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
        self.global_trav = global_trav_dev.get(queue)

        # }}}

        self.distributed_expansion_wrangler_factory = \
            distributed_expansion_wrangler_factory

        # {{{ Get global wrangler

        self.global_wrangler = distributed_expansion_wrangler_factory(global_tree)

        # }}}

        # {{{ Partiton work

        cost_model = FMMCostModel()

        if calibration_params is None:
            # Use default calibration parameters if not supplied
            # TODO: should replace the default calibration params with a more
            # accurate one
            calibration_params = \
                FMMCostModel.get_unit_calibration_params()

        cost_per_box = cost_model.cost_per_box(
            queue, global_trav_dev, self.global_wrangler.level_nterms,
            calibration_params
        ).get()

        from boxtree.distributed.partition import partition_work
        responsible_boxes_list = partition_work(
            cost_per_box, self.global_trav, comm.Get_size()
        )

        # It is assumed that, even if each rank computes `responsible_boxes_list`
        # independently, it should be the same across ranks.

        # }}}

        # {{{ Compute local tree

        from boxtree.distributed.partition import ResponsibleBoxesQuery
        responsible_box_query = ResponsibleBoxesQuery(queue, self.global_trav)

        from boxtree.distributed.local_tree import generate_local_tree
        self.local_tree, self.src_idx, self.tgt_idx = generate_local_tree(
            queue, global_tree, responsible_boxes_list, responsible_box_query
        )

        # }}}

        # {{ Gather source indices and target indices of each rank

        self.src_idx_all_ranks = comm.gather(self.src_idx, root=0)
        self.tgt_idx_all_ranks = comm.gather(self.tgt_idx, root=0)

        # }}}

        # {{{ Compute traversal object on each rank

        from boxtree.distributed.local_traversal import generate_local_travs
        self.local_trav = generate_local_travs(
            queue, self.local_tree, traversal_builder,
            box_bounding_box={
                "min": self.global_trav.box_target_bounding_box_min,
                "max": self.global_trav.box_target_bounding_box_max
            }
        )

        # }}}

        # {{{ Get local wrangler

        self.local_wrangler = self.distributed_expansion_wrangler_factory(
            self.local_tree)

        # }}}

    def drive_dfmm(self, source_weights, _communicate_mpoles_via_allreduce=False):
        """Calculate potentials at target points.
        """
        from boxtree.distributed.calculation import calculate_pot
        return calculate_pot(
            self.local_wrangler, self.global_wrangler, self.local_trav,
            source_weights, self.src_idx_all_ranks, self.tgt_idx_all_ranks,
            _communicate_mpoles_via_allreduce=_communicate_mpoles_via_allreduce
        )
