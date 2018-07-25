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
from boxtree.distributed.perf_model import PerformanceModel, PerformanceCounter

MPITags = dict(
    DIST_TREE=0,
    DIST_SOURCES=1,
    DIST_TARGETS=2,
    DIST_RADII=3,
    DIST_WEIGHT=4,
    GATHER_POTENTIALS=5,
    REDUCE_POTENTIALS=6,
    REDUCE_INDICES=7
)


def dtype_to_mpi(dtype):
    """ This function translates a numpy.dtype object into the corresponding type
    used in mpi4py.
    """
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(dtype).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(dtype).char]
    else:
        raise RuntimeError("There is no dictionary to translate from Numpy dtype to "
                           "MPI type")
    return mpi_type


class DistributedFMMInfo(object):

    def __init__(self, queue, global_trav, distributed_expansion_wrangler_factory,
                 comm=MPI.COMM_WORLD):

        self.global_trav = global_trav
        self.distributed_expansion_wrangler_factory = \
            distributed_expansion_wrangler_factory

        self.comm = comm
        current_rank = comm.Get_rank()

        # {{{ Get global wrangler

        if current_rank == 0:
            self.global_wrangler = distributed_expansion_wrangler_factory(
                self.global_trav.tree
            )
        else:
            self.global_wrangler = None

        # }}}

        # {{{ Broadcast well_sep_is_n_away

        if current_rank == 0:
            well_sep_is_n_away = global_trav.well_sep_is_n_away
        else:
            well_sep_is_n_away = None

        well_sep_is_n_away = comm.bcast(well_sep_is_n_away, root=0)

        # }}}

        # {{{ Get performance model and counter

        if current_rank == 0:
            from boxtree.fmm import drive_fmm
            model = PerformanceModel(
                queue.context,
                distributed_expansion_wrangler_factory,
                True, drive_fmm
            )
            model.time_random_traversals()

            counter = PerformanceCounter(global_trav, self.global_wrangler, True)

        # }}}

        # {{{ Partiton work

        if current_rank == 0:
            from boxtree.distributed.partition import partition_work
            responsible_boxes_list = partition_work(
                model, counter, global_trav, comm.Get_size()
            )
        else:
            responsible_boxes_list = None

        # }}}

        # {{{ Compute and distribute local tree

        if current_rank == 0:
            from boxtree.distributed.partition import ResponsibleBoxesQuery
            responsible_box_query = ResponsibleBoxesQuery(queue, global_trav)
        else:
            responsible_box_query = None

        from boxtree.distributed.local_tree import generate_local_tree
        self.local_tree, self.local_data, self.box_bounding_box = \
            generate_local_tree(queue, self.global_trav, responsible_boxes_list,
                                responsible_box_query)

        # }}}

        # {{{ Compute traversal object on each process

        from boxtree.distributed.local_traversal import generate_local_travs
        self.local_trav = generate_local_travs(
            queue, self.local_tree, self.box_bounding_box,
            well_sep_is_n_away=well_sep_is_n_away)

        # }}}

        # {{{ Get local wrangler

        """
        Note: The difference between "local wrangler" and "global wrangler" is that
        they reference different tree object. "local wrangler" uses local tree
        object on each worker process for FMM computation, whereas "global wrangler"
        is only valid on root process used for assembling results from worker
        processes.
        """

        self.local_wrangler = self.distributed_expansion_wrangler_factory(
            self.local_tree)

        # }}}

    def drive_dfmm(self, source_weights, _communicate_mpoles_via_allreduce=False):
        from boxtree.distributed.calculation import calculate_pot
        return calculate_pot(
            self.local_wrangler, self.global_wrangler, self.local_trav,
            source_weights, self.local_data,
            _communicate_mpoles_via_allreduce=_communicate_mpoles_via_allreduce
        )
