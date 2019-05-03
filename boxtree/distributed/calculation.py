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
from boxtree.distributed import dtype_to_mpi
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from pytools import memoize_method
from pyopencl.tools import dtype_to_ctype
from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template

import logging
logger = logging.getLogger(__name__)


# {{{ Distributed FMM wrangler

class DistributedFMMLibExpansionWrangler(FMMLibExpansionWrangler):

    def __init__(self, queue, tree, helmholtz_k, fmm_level_to_nterms=None):
        super(DistributedFMMLibExpansionWrangler, self).__init__(
            tree, helmholtz_k, fmm_level_to_nterms
        )

        self.queue = queue

    def slice_mpoles(self, mpoles, slice_indices):
        if len(slice_indices) == 0:
            return np.empty((0,), dtype=mpoles.dtype)

        level_start_slice_indices = np.searchsorted(
            slice_indices, self.tree.level_start_box_nrs)
        mpoles_list = []

        for ilevel in range(self.tree.nlevels):
            start, stop = level_start_slice_indices[ilevel:ilevel+2]
            if stop > start:
                level_start_box_idx, mpoles_current_level = \
                    self.multipole_expansions_view(mpoles, ilevel)
                mpoles_list.append(
                    mpoles_current_level[
                        slice_indices[start:stop] - level_start_box_idx
                    ].reshape(-1)
                )

        return np.concatenate(mpoles_list)

    def update_mpoles(self, mpoles, mpole_updates, slice_indices):
        if len(slice_indices) == 0:
            return

        level_start_slice_indices = np.searchsorted(
            slice_indices, self.tree.level_start_box_nrs)
        mpole_updates_start = 0

        for ilevel in range(self.tree.nlevels):
            start, stop = level_start_slice_indices[ilevel:ilevel+2]
            if stop > start:
                level_start_box_idx, mpoles_current_level = \
                    self.multipole_expansions_view(mpoles, ilevel)
                mpoles_shape = (stop - start,) + mpoles_current_level.shape[1:]

                from pytools import product
                mpole_updates_end = mpole_updates_start + product(mpoles_shape)

                mpoles_current_level[
                    slice_indices[start:stop] - level_start_box_idx
                ] += mpole_updates[
                    mpole_updates_start:mpole_updates_end
                ].reshape(mpoles_shape)

                mpole_updates_start = mpole_updates_end

    def empty_box_in_subrange_mask(self):
        return cl.array.empty(self.queue, self.tree.nboxes, dtype=np.int8)

    @memoize_method
    def find_boxes_used_by_subrange_kernel(self):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                ${box_id_t} *contributing_boxes_list,
                int subrange_start,
                int subrange_end,
                ${box_id_t} *box_to_user_starts,
                int *box_to_user_lists,
                char *box_in_subrange
            """).render(
                box_id_t=dtype_to_ctype(self.tree.box_id_dtype),
            ),
            Template(r"""
                ${box_id_t} ibox = contributing_boxes_list[i];
                ${box_id_t} iuser_start = box_to_user_starts[ibox];
                ${box_id_t} iuser_end = box_to_user_starts[ibox + 1];
                for(${box_id_t} iuser = iuser_start; iuser < iuser_end; iuser++) {
                    int useri = box_to_user_lists[iuser];
                    if(subrange_start <= useri && useri < subrange_end) {
                        box_in_subrange[i] = 1;
                    }
                }
            """).render(
                box_id_t=dtype_to_ctype(self.tree.box_id_dtype)
            ),
            "find_boxes_used_by_subrange"
        )

    def find_boxes_used_by_subrange(self, box_in_subrange, subrange,
                                    box_to_user_starts, box_to_user_lists,
                                    contributing_boxes_list):
        knl = self.find_boxes_used_by_subrange_kernel()

        knl(
            contributing_boxes_list,
            subrange[0],
            subrange[1],
            box_to_user_starts,
            box_to_user_lists,
            box_in_subrange
        )

# }}}


# {{{ Communicate mpoles

def communicate_mpoles(wrangler, comm, trav, mpole_exps, return_stats=False):
    """Based on Algorithm 3: Reduce and Scatter in [1].

    The main idea is to mimic a allreduce as done on a hypercube network, but to
    decrease the bandwidth cost by sending only information that is relevant to
    the processes receiving the message.

    This function needs to be called collectively by all processes in :arg comm.

    .. [1] Lashuk, Ilya, Aparna Chandramowlishwaran, Harper Langston,
       Tuan-Anh Nguyen, Rahul Sampath, Aashay Shringarpure, Richard Vuduc, Lexing
       Ying, Denis Zorin, and George Biros. “A massively parallel adaptive fast
       multipole method on heterogeneous architectures." Communications of the
       ACM 55, no. 5 (2012): 101-109.
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

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
    contributing_boxes = trav.tree.ancestor_mask.copy()
    contributing_boxes[trav.tree.responsible_boxes_list] = 1

    from boxtree.tools import AllReduceCommPattern
    comm_pattern = AllReduceCommPattern(rank, nprocs)

    # Temporary buffers for receiving data
    mpole_exps_buf = np.empty(mpole_exps.shape, dtype=mpole_exps.dtype)
    boxes_list_buf = np.empty(trav.tree.nboxes, dtype=trav.tree.box_id_dtype)

    stats["bytes_sent_by_stage"] = []
    stats["bytes_recvd_by_stage"] = []

    box_to_user_starts_dev = cl.array.to_device(
        wrangler.queue, trav.tree.box_to_user_starts
    )

    box_to_user_lists_dev = cl.array.to_device(
        wrangler.queue, trav.tree.box_to_user_lists
    )

    while not comm_pattern.done():
        send_requests = []

        # Send data to other processors.
        if comm_pattern.sinks():
            # Compute the subset of boxes to be sent.
            message_subrange = comm_pattern.messages()

            contributing_boxes_list = np.nonzero(contributing_boxes)[0].astype(
                trav.tree.box_id_dtype
            )

            contributing_boxes_list_dev = cl.array.to_device(
                wrangler.queue, contributing_boxes_list
            )

            box_in_subrange = cl.array.zeros(
                wrangler.queue, contributing_boxes_list.shape[0], dtype=np.int8
            )

            wrangler.find_boxes_used_by_subrange(
                box_in_subrange, message_subrange,
                box_to_user_starts_dev, box_to_user_lists_dev,
                contributing_boxes_list_dev
            )

            box_in_subrange_host = box_in_subrange.get().astype(bool)

            relevant_boxes_list = contributing_boxes_list[
                box_in_subrange_host
            ].astype(trav.tree.box_id_dtype)

            relevant_mpole_exps = wrangler.slice_mpoles(
                mpole_exps, relevant_boxes_list
            )

            # Send the box subset to the other processors.
            for sink in comm_pattern.sinks():
                req = comm.Isend(relevant_mpole_exps, dest=sink,
                                 tag=MPITags["REDUCE_POTENTIALS"])
                send_requests.append(req)

                req = comm.Isend(relevant_boxes_list, dest=sink,
                                 tag=MPITags["REDUCE_INDICES"])
                send_requests.append(req)

        # Receive data from other processors.
        for source in comm_pattern.sources():
            comm.Recv(mpole_exps_buf, source=source,
                      tag=MPITags["REDUCE_POTENTIALS"])

            status = MPI.Status()
            comm.Recv(boxes_list_buf, source=source, tag=MPITags["REDUCE_INDICES"],
                      status=status)
            nboxes = status.Get_count() // boxes_list_buf.dtype.itemsize

            # Update data structures.
            wrangler.update_mpoles(mpole_exps, mpole_exps_buf,
                                   boxes_list_buf[:nboxes])

            contributing_boxes[boxes_list_buf[:nboxes]] = 1

        for req in send_requests:
            req.wait()

        comm_pattern.advance()

    if return_stats:
        return stats

# }}}


# {{{ Distribute source weights

def distribute_source_weights(source_weights, local_data, comm=MPI.COMM_WORLD):
    """ This function transfers needed source_weights from root process to each
    worker process in communicator :arg comm.

    This function needs to be called by all processes in the :arg comm communicator.

    :param source_weights: Source weights in tree order on root, None on worker
        processes.
    :param local_data: Returned from *generate_local_tree*. None on worker processes.
    :return Source weights needed for the current process.
    """
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    if current_rank == 0:
        weight_req = []
        local_src_weights = np.empty((total_rank,), dtype=object)

        for irank in range(total_rank):
            local_src_weights[irank] = source_weights[local_data[irank].src_idx]

            if irank != 0:
                weight_req.append(
                    comm.isend(local_src_weights[irank], dest=irank,
                               tag=MPITags["DIST_WEIGHT"])
                )

        MPI.Request.Waitall(weight_req)

        local_src_weights = local_src_weights[0]
    else:
        local_src_weights = comm.recv(source=0, tag=MPITags["DIST_WEIGHT"])

    return local_src_weights

# }}}


# {{{ FMM driver for calculating potentials

def calculate_pot(local_wrangler, global_wrangler, local_trav, source_weights,
                  local_data, comm=MPI.COMM_WORLD,
                  _communicate_mpoles_via_allreduce=False):
    """ Calculate potentials for targets on distributed memory machines.

    This function needs to be called collectively by all process in :arg comm.

    :param local_wrangler: Expansion wranglers for each worker process for FMM.
    :param global_wrangler: Expansion wrangler on root process for assembling partial
        results from worker processes together. This argument differs from
        :arg local_wrangler by referening the global tree instead of local trees.
        This argument is None on worker processes.
    :param local_trav: Local traversal object returned from generate_local_travs.
    :param source_weights: Source weights for FMM. None on worker processes.
    :param local_data: LocalData object returned from generate_local_tree.
    :param comm: MPI communicator.
    :param _communicate_mpoles_via_allreduce: Use MPI allreduce for communicating
        multipole expressions. Using MPI allreduce is slower but might be helpful for
        debugging purpose.
    :return: On the root process, this function returns calculated potentials. On
        worker processes, this function returns None.
    """

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # {{{ Distribute source weights

    if current_rank == 0:
        # Convert src_weights to tree order
        source_weights = source_weights[global_wrangler.tree.user_source_ids]

    local_src_weights = distribute_source_weights(
        source_weights, local_data, comm=comm
    )

    # }}}

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps = local_wrangler.form_multipoles(
            local_trav.level_start_source_box_nrs,
            local_trav.source_boxes,
            local_src_weights)[0]

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    local_wrangler.coarsen_multipoles(
            local_trav.level_start_source_parent_box_nrs,
            local_trav.source_parent_boxes,
            mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ Communicate mpoles

    if _communicate_mpoles_via_allreduce:
        mpole_exps_all = np.zeros_like(mpole_exps)
        comm.Allreduce(mpole_exps, mpole_exps_all)
        mpole_exps = mpole_exps_all
    else:
        communicate_mpoles(local_wrangler, comm, local_trav, mpole_exps)

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    potentials = local_wrangler.eval_direct(
            local_trav.target_boxes,
            local_trav.neighbor_source_boxes_starts,
            local_trav.neighbor_source_boxes_lists,
            local_src_weights)[0]

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps = local_wrangler.multipole_to_local(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_trav.from_sep_siblings_starts,
            local_trav.from_sep_siblings_lists,
            mpole_exps)[0]

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + local_wrangler.eval_multipoles(
            local_trav.target_boxes_sep_smaller_by_source_level,
            local_trav.from_sep_smaller_by_level,
            mpole_exps)[0]

    # these potentials are called beta in [1]

    if local_trav.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                     "('list 3 close')")
        potentials = potentials + local_wrangler.eval_direct(
                local_trav.target_boxes,
                local_trav.from_sep_close_smaller_starts,
                local_trav.from_sep_close_smaller_lists,
                local_src_weights)[0]

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_exps = local_exps + local_wrangler.form_locals(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_trav.from_sep_bigger_starts,
            local_trav.from_sep_bigger_lists,
            local_src_weights)[0]

    if local_trav.from_sep_close_bigger_starts is not None:
        logger.debug("evaluate separated close bigger interactions directly "
                     "('list 4 close')")

        potentials = potentials + local_wrangler.eval_direct(
                local_trav.target_boxes,
                local_trav.from_sep_close_bigger_starts,
                local_trav.from_sep_close_bigger_lists,
                local_src_weights)[0]

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    local_wrangler.refine_locals(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")
    potentials = potentials + local_wrangler.eval_locals(
            local_trav.level_start_target_box_nrs,
            local_trav.target_boxes,
            local_exps)[0]

    # }}}

    # {{{ Worker processes send calculated potentials to the root process

    potentials_mpi_type = dtype_to_mpi(potentials.dtype)

    if current_rank == 0:
        potentials_all_ranks = np.empty((total_rank,), dtype=object)
        potentials_all_ranks[0] = potentials

        for irank in range(1, total_rank):
            potentials_all_ranks[irank] = np.empty(
                (local_data[irank].ntargets,), dtype=potentials.dtype)

            comm.Recv([potentials_all_ranks[irank], potentials_mpi_type],
                      source=irank, tag=MPITags["GATHER_POTENTIALS"])
    else:
        comm.Send([potentials, potentials_mpi_type],
                  dest=0, tag=MPITags["GATHER_POTENTIALS"])

    # }}}

    # {{{ Assemble potentials from worker processes together on the root process

    if current_rank == 0:
        potentials = np.empty((global_wrangler.tree.ntargets,),
                              dtype=potentials.dtype)

        for irank in range(total_rank):
            potentials[local_data[irank].tgt_idx] = potentials_all_ranks[irank]

        logger.debug("reorder potentials")
        result = global_wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = global_wrangler.finalize_potentials(result)

    # }}}

    if current_rank == 0:
        return result

# }}}
