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

import numpy as np
import pyopencl as cl
from boxtree.distributed import MPITags
from mpi4py import MPI
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from boxtree.fmm import ExpansionWranglerInterface
from pytools import memoize_method
from pyopencl.tools import dtype_to_ctype
from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template

import logging
logger = logging.getLogger(__name__)


# {{{ Distributed FMM wrangler

class DistributedExpansionWrangler(ExpansionWranglerInterface):
    """Distributed expansion wrangler base class.

    This is an abstract class and should not be directly instantiated. Instead, it is
    expected that all distributed wranglers should be subclasses of this class.

    .. automethod:: __init__
    .. automethod:: distribute_source_weights
    .. automethod:: gather_potential_results
    .. automethod:: communicate_mpoles
    """
    def __init__(self, context, comm, global_traversal,
                 traversal_in_device_memory,
                 communicate_mpoles_via_allreduce=False):
        self.context = context
        self.comm = comm
        self.global_traversal = global_traversal
        self.traversal_in_device_memory = traversal_in_device_memory
        self.communicate_mpoles_via_allreduce = communicate_mpoles_via_allreduce

    def distribute_source_weights(self, src_weight_vecs, src_idx_all_ranks):
        mpi_rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        if mpi_rank == 0:
            distribute_weight_req = []
            local_src_weight_vecs = np.empty((mpi_size,), dtype=object)

            for irank in range(mpi_size):
                local_src_weight_vecs[irank] = [
                    source_weights[src_idx_all_ranks[irank]]
                    for source_weights in src_weight_vecs]

                if irank != 0:
                    distribute_weight_req.append(self.comm.isend(
                        local_src_weight_vecs[irank], dest=irank,
                        tag=MPITags.DIST_WEIGHT
                    ))

            MPI.Request.Waitall(distribute_weight_req)
            local_src_weight_vecs = local_src_weight_vecs[0]
        else:
            local_src_weight_vecs = self.comm.recv(source=0, tag=MPITags.DIST_WEIGHT)

        return local_src_weight_vecs

    def gather_potential_results(self, potentials, tgt_idx_all_ranks):
        mpi_rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        from boxtree.distributed import dtype_to_mpi
        potentials_mpi_type = dtype_to_mpi(potentials.dtype)

        gathered_potentials = None

        if mpi_rank == 0:
            # The root rank received calculated potentials from all worker ranks
            potentials_all_ranks = np.empty((mpi_size,), dtype=object)
            potentials_all_ranks[0] = potentials

            recv_reqs = []

            for irank in range(1, mpi_size):
                potentials_all_ranks[irank] = np.empty(
                    tgt_idx_all_ranks[irank].shape, dtype=potentials.dtype)

                recv_reqs.append(
                    self.comm.Irecv(
                        [potentials_all_ranks[irank], potentials_mpi_type],
                        source=irank, tag=MPITags.GATHER_POTENTIALS))

            MPI.Request.Waitall(recv_reqs)

            # Assemble potentials from worker ranks together on the root rank
            gathered_potentials = np.empty(
                self.global_traversal.tree.ntargets, dtype=potentials.dtype)

            for irank in range(mpi_size):
                gathered_potentials[tgt_idx_all_ranks[irank]] = (
                    potentials_all_ranks[irank])
        else:
            # Worker ranks send calculated potentials to the root rank
            self.comm.Send([potentials, potentials_mpi_type],
                           dest=0, tag=MPITags.GATHER_POTENTIALS)

        return gathered_potentials

    def _slice_mpoles(self, mpoles, slice_indices):
        if len(slice_indices) == 0:
            return np.empty((0,), dtype=mpoles.dtype)

        level_start_slice_indices = np.searchsorted(
            slice_indices, self.traversal.tree.level_start_box_nrs)
        mpoles_list = []

        for ilevel in range(self.traversal.tree.nlevels):
            start, stop = level_start_slice_indices[ilevel:ilevel+2]
            if stop > start:
                level_start_box_idx, mpoles_current_level = (
                    self.multipole_expansions_view(mpoles, ilevel))
                mpoles_list.append(
                    mpoles_current_level[
                        slice_indices[start:stop] - level_start_box_idx
                    ].reshape(-1)
                )

        return np.concatenate(mpoles_list)

    def _update_mpoles(self, mpoles, mpole_updates, slice_indices):
        if len(slice_indices) == 0:
            return

        level_start_slice_indices = np.searchsorted(
            slice_indices, self.traversal.tree.level_start_box_nrs)
        mpole_updates_start = 0

        for ilevel in range(self.traversal.tree.nlevels):
            start, stop = level_start_slice_indices[ilevel:ilevel+2]
            if stop > start:
                level_start_box_idx, mpoles_current_level = (
                    self.multipole_expansions_view(mpoles, ilevel))
                mpoles_shape = (stop - start,) + mpoles_current_level.shape[1:]

                from pytools import product
                mpole_updates_end = mpole_updates_start + product(mpoles_shape)

                mpoles_current_level[
                    slice_indices[start:stop] - level_start_box_idx
                ] += mpole_updates[
                    mpole_updates_start:mpole_updates_end
                ].reshape(mpoles_shape)

                mpole_updates_start = mpole_updates_end

    @memoize_method
    def find_boxes_used_by_subrange_kernel(self, box_id_dtype):
        return ElementwiseKernel(
            self.context,
            Template(r"""
                ${box_id_t} *contributing_boxes_list,
                int subrange_start,
                int subrange_end,
                ${box_id_t} *box_to_user_rank_starts,
                int *box_to_user_rank_lists,
                char *box_in_subrange
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
            ),
            Template(r"""
                ${box_id_t} ibox = contributing_boxes_list[i];
                ${box_id_t} iuser_start = box_to_user_rank_starts[ibox];
                ${box_id_t} iuser_end = box_to_user_rank_starts[ibox + 1];
                for(${box_id_t} iuser = iuser_start; iuser < iuser_end; iuser++) {
                    int useri = box_to_user_rank_lists[iuser];
                    if(subrange_start <= useri && useri < subrange_end) {
                        box_in_subrange[i] = 1;
                    }
                }
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            "find_boxes_used_by_subrange"
        )

    def find_boxes_used_by_subrange(
            self, subrange, box_to_user_rank_starts, box_to_user_rank_lists,
            contributing_boxes_list):
        """Test whether the multipole expansions of the contributing boxes are used
        by at least one box in a range.

        :arg subrange: the range is represented by ``(subrange[0], subrange[1])``.
        :arg box_to_user_rank_starts: a :class:`pyopencl.array.Array` object
            indicating the start and end index in *box_to_user_rank_lists* for each
            box in *contributing_boxes_list*.
        :arg box_to_user_rank_lists: a :class:`pyopencl.array.Array` object storing
            the users of each box in *contributing_boxes_list*.
        :returns: a :class:`pyopencl.array.Array` object with the same shape as
            *contributing_boxes_list*, where the i-th entry is 1 if
            ``contributing_boxes_list[i]`` is used by at least on box in the
            subrange specified.
        """
        box_in_subrange = cl.array.zeros(
            contributing_boxes_list.queue,
            contributing_boxes_list.shape[0],
            dtype=np.int8
        )
        knl = self.find_boxes_used_by_subrange_kernel(
                self.traversal.tree.box_id_dtype)

        knl(
            contributing_boxes_list,
            subrange[0],
            subrange[1],
            box_to_user_rank_starts,
            box_to_user_rank_lists,
            box_in_subrange
        )

        return box_in_subrange

    def communicate_mpoles(self, mpole_exps, return_stats=False):
        """Based on Algorithm 3: Reduce and Scatter in Lashuk et al. [1]_.

        The main idea is to mimic an allreduce as done on a hypercube network, but to
        decrease the bandwidth cost by sending only information that is relevant to
        the rank receiving the message.

        .. [1] Lashuk, Ilya, Aparna Chandramowlishwaran, Harper Langston,
            Tuan-Anh Nguyen, Rahul Sampath, Aashay Shringarpure, Richard Vuduc,
            Lexing Ying, Denis Zorin, and George Biros. “A massively parallel
            adaptive fast multipole method on heterogeneous architectures."
            Communications of the ACM 55, no. 5 (2012): 101-109.
        """
        mpi_rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()
        tree = self.traversal.tree

        if self.communicate_mpoles_via_allreduce:
            # Use MPI allreduce for communicating multipole expressions. It is slower
            # but might be helpful for debugging purposes.
            mpole_exps_all = np.zeros_like(mpole_exps)
            self.comm.Allreduce(mpole_exps, mpole_exps_all)
            mpole_exps[:] = mpole_exps_all
            return

        stats = {}

        # contributing_boxes:
        #
        # A mask of the the set of boxes that the current process contributes
        # to. This process contributes to a box when:
        #
        # (a) this process owns sources that contribute to the multipole expansion
        # in the box (via the upward pass) or
        # (b) this process has received a portion of the multipole expansion in this
        # box from another process.
        #
        # Initially, this set consists of the boxes satisfying condition (a), which
        # are precisely the boxes owned by this process and their ancestors.
        if self.traversal_in_device_memory:
            with cl.CommandQueue(self.context) as queue:
                contributing_boxes = tree.ancestor_mask.get(queue=queue)
                responsible_boxes_list = tree.responsible_boxes_list.get(queue=queue)
        else:
            contributing_boxes = tree.ancestor_mask.copy()
            responsible_boxes_list = tree.responsible_boxes_list
        contributing_boxes[responsible_boxes_list] = 1

        from boxtree.tools import AllReduceCommPattern
        comm_pattern = AllReduceCommPattern(mpi_rank, mpi_size)

        # Temporary buffers for receiving data
        mpole_exps_buf = np.empty(mpole_exps.shape, dtype=mpole_exps.dtype)
        boxes_list_buf = np.empty(tree.nboxes, dtype=tree.box_id_dtype)

        stats["bytes_sent_by_stage"] = []
        stats["bytes_recvd_by_stage"] = []

        if self.traversal_in_device_memory:
            box_to_user_rank_starts_dev = \
                tree.box_to_user_rank_starts.with_queue(None)
            box_to_user_rank_lists_dev = tree.box_to_user_rank_lists.with_queue(None)
        else:
            with cl.CommandQueue(self.context) as queue:
                box_to_user_rank_starts_dev = cl.array.to_device(
                    queue, tree.box_to_user_rank_starts).with_queue(None)
                box_to_user_rank_lists_dev = cl.array.to_device(
                    queue, tree.box_to_user_rank_lists).with_queue(None)

        while not comm_pattern.done():
            send_requests = []

            # Send data to other processors.
            if comm_pattern.sinks():
                # Compute the subset of boxes to be sent.
                message_subrange = comm_pattern.messages()

                contributing_boxes_list = np.nonzero(contributing_boxes)[0].astype(
                    tree.box_id_dtype
                )

                with cl.CommandQueue(self.context) as queue:
                    contributing_boxes_list_dev = cl.array.to_device(
                        queue, contributing_boxes_list)

                    box_in_subrange = self.find_boxes_used_by_subrange(
                        message_subrange,
                        box_to_user_rank_starts_dev, box_to_user_rank_lists_dev,
                        contributing_boxes_list_dev
                    )

                    box_in_subrange_host = box_in_subrange.get().astype(bool)

                relevant_boxes_list = contributing_boxes_list[
                    box_in_subrange_host
                ].astype(tree.box_id_dtype)

                """
                # Pure Python version for debugging purpose
                relevant_boxes_list = []
                for contrib_box in contributing_boxes_list:
                    iuser_start, iuser_end = tree.box_to_user_rank_starts[
                        contrib_box:contrib_box + 2
                    ]
                    for user_box in tree.box_to_user_rank_lists[
                            iuser_start:iuser_end]:
                        if subrange_start <= user_box < subrange_end:
                            relevant_boxes_list.append(contrib_box)
                            break
                """

                relevant_boxes_list = np.array(
                    relevant_boxes_list, dtype=tree.box_id_dtype
                )

                relevant_mpole_exps = self._slice_mpoles(
                    mpole_exps, relevant_boxes_list)

                # Send the box subset to the other processors.
                for sink in comm_pattern.sinks():
                    req = self.comm.Isend(relevant_mpole_exps, dest=sink,
                                          tag=MPITags.REDUCE_POTENTIALS)
                    send_requests.append(req)

                    req = self.comm.Isend(relevant_boxes_list, dest=sink,
                                          tag=MPITags.REDUCE_INDICES)
                    send_requests.append(req)

            # Receive data from other processors.
            for source in comm_pattern.sources():
                self.comm.Recv(mpole_exps_buf, source=source,
                               tag=MPITags.REDUCE_POTENTIALS)

                status = MPI.Status()
                self.comm.Recv(
                    boxes_list_buf, source=source, tag=MPITags.REDUCE_INDICES,
                    status=status)
                nboxes = status.Get_count() // boxes_list_buf.dtype.itemsize

                # Update data structures.
                self._update_mpoles(
                        mpole_exps, mpole_exps_buf, boxes_list_buf[:nboxes])

                contributing_boxes[boxes_list_buf[:nboxes]] = 1

            for req in send_requests:
                req.wait()

            comm_pattern.advance()

        if return_stats:
            return stats

    def finalize_potentials(self, potentials, template_ary):
        if self.comm.Get_rank() == 0:
            return super().finalize_potentials(potentials, template_ary)
        else:
            return None


class DistributedFMMLibExpansionWrangler(
        DistributedExpansionWrangler, FMMLibExpansionWrangler):
    def __init__(
            self, context, comm, tree_indep, local_traversal, global_traversal,
            fmm_level_to_order=None,
            communicate_mpoles_via_allreduce=False,
            **kwargs):
        DistributedExpansionWrangler.__init__(
            self, context, comm, global_traversal, False,
            communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)
        FMMLibExpansionWrangler.__init__(
            self, tree_indep, local_traversal,
            fmm_level_to_order=fmm_level_to_order, **kwargs)

    #TODO: use log_process like FMMLibExpansionWrangler?
    def reorder_sources(self, source_array):
        if self.comm.Get_rank() == 0:
            return source_array[..., self.global_traversal.tree.user_source_ids]
        else:
            return None

    def reorder_potentials(self, potentials):
        if self.comm.Get_rank() == 0:
            return potentials[self.global_traversal.tree.sorted_target_ids]
        else:
            return None

# }}}
