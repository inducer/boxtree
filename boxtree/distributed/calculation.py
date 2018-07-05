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
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
import time
from boxtree.distributed import dtype_to_mpi
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from pytools import memoize_method
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_1  # noqa: F401

import logging
logger = logging.getLogger(__name__)


# {{{ distributed fmm wrangler

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
        knl = lp.make_kernel(
            [
                "{[ibox]: 0 <= ibox < nboxes}",
                "{[iuser]: iuser_start <= iuser < iuser_end}",
            ],
            """
            for ibox
                <> iuser_start = box_to_user_starts[ibox]
                <> iuser_end = box_to_user_starts[ibox + 1]
                for iuser
                    <> useri = box_to_user_lists[iuser]
                    <> in_subrange = subrange_start <= useri and useri < subrange_end
                    if in_subrange
                        box_in_subrange[ibox] = 1
                    end
                end
            end
            """,
            [
                lp.ValueArg("subrange_start, subrange_end", np.int32),
                lp.GlobalArg("box_to_user_lists", shape=None),
                "..."
            ])
        knl = lp.split_iname(knl, "ibox", 16, outer_tag="g.0", inner_tag="l.0")
        return knl

    def find_boxes_used_by_subrange(self, box_in_subrange, subrange,
                                    box_to_user_starts, box_to_user_lists):
        knl = self.find_boxes_used_by_subrange_kernel()
        knl(self.queue,
            subrange_start=subrange[0],
            subrange_end=subrange[1],
            box_to_user_starts=box_to_user_starts,
            box_to_user_lists=box_to_user_lists,
            box_in_subrange=box_in_subrange)

        box_in_subrange.finish()

# }}}


# {{{ communicate mpoles

def communicate_mpoles(wrangler, comm, trav, mpole_exps, return_stats=False):
    """Based on Algorithm 3: Reduce and Scatter in [1].

    The main idea is to mimic a allreduce as done on a hypercube network, but to
    decrease the bandwidth cost by sending only information that is relevant to
    the processes receiving the message.

    .. [1] Lashuk, Ilya, Aparna Chandramowlishwaran, Harper Langston,
       Tuan-Anh Nguyen, Rahul Sampath, Aashay Shringarpure, Richard Vuduc, Lexing
       Ying, Denis Zorin, and George Biros. “A massively parallel adaptive fast
       multipole method on heterogeneous architectures." Communications of the
       ACM 55, no. 5 (2012): 101-109.
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    stats = {}

    from time import time
    t_start = time()
    logger.debug("communicate multipoles: start")

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

    # Temporary buffer for holding the mask
    box_in_subrange = wrangler.empty_box_in_subrange_mask()

    stats["bytes_sent_by_stage"] = []
    stats["bytes_recvd_by_stage"] = []

    while not comm_pattern.done():
        send_requests = []

        # Send data to other processors.
        if comm_pattern.sinks():
            # Compute the subset of boxes to be sent.
            message_subrange = comm_pattern.messages()

            box_in_subrange.fill(0)

            wrangler.find_boxes_used_by_subrange(
                box_in_subrange, message_subrange,
                trav.tree.box_to_user_starts, trav.tree.box_to_user_lists)

            box_in_subrange_host = (
                box_in_subrange.map_to_host(flags=cl.map_flags.READ))

            with box_in_subrange_host.data:
                relevant_boxes_list = (
                    np.nonzero(box_in_subrange_host & contributing_boxes)
                    [0]
                    .astype(trav.tree.box_id_dtype))

            del box_in_subrange_host

            relevant_mpole_exps = wrangler.slice_mpoles(mpole_exps,
                relevant_boxes_list)

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

    stats["total_time"] = time() - t_start
    logger.info("communicate multipoles: done in %.2f s" % stats["total_time"])

    if return_stats:
        return stats

# }}}


def get_gen_local_weights_helper(queue, particle_dtype, weight_dtype):
    gen_local_source_weights_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        arguments=Template("""
            __global ${weight_t} *src_weights,
            __global ${particle_id_t} *particle_mask,
            __global ${particle_id_t} *particle_scan,
            __global ${weight_t} *local_weights
        """, strict_undefined=True).render(
            weight_t=dtype_to_ctype(weight_dtype),
            particle_id_t=dtype_to_ctype(particle_dtype)
        ),
        operation="""
            if(particle_mask[i]) {
                local_weights[particle_scan[i]] = src_weights[i];
            }
        """
    )

    def gen_local_weights(global_weights, source_mask, source_scan):
        local_nsources = source_scan[-1].get(queue)
        local_weights = cl.array.empty(queue, (local_nsources,),
                                       dtype=weight_dtype)
        gen_local_source_weights_knl(global_weights, source_mask, source_scan,
                                     local_weights)
        return local_weights.get(queue)

    return gen_local_weights


def distribute_source_weights(queue, source_weights, global_tree, local_data,
        comm=MPI.COMM_WORLD):
    """
    source_weights: source weights in tree order
    global_tree: complete tree structure on root, None otherwise.
    local_data: returned from *generate_local_tree*
    """
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    if current_rank == 0:
        weight_req = np.empty((total_rank,), dtype=object)
        local_src_weights = np.empty((total_rank,), dtype=object)

        # Generate local_weights
        for rank in range(total_rank):
            local_src_weights[rank] = source_weights[local_data[rank].src_idx]

            weight_req[rank] = comm.isend(local_src_weights[rank], dest=rank,
                                          tag=MPITags["DIST_WEIGHT"])

        for rank in range(1, total_rank):
            weight_req[rank].wait()
        local_src_weights = local_src_weights[0]
    else:
        local_src_weights = comm.recv(source=0, tag=MPITags["DIST_WEIGHT"])

    return local_src_weights


def calculate_pot(queue, wrangler, global_wrangler, local_trav, source_weights,
                  local_data, comm=MPI.COMM_WORLD,
                  _communicate_mpoles_via_allreduce=False):

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    if current_rank == 0:
        start_time = time.time()

    # {{{ Distribute source weights

    if current_rank == 0:
        global_tree = global_wrangler.tree
        # Convert src_weights to tree order
        source_weights = source_weights[global_tree.user_source_ids]
    else:
        global_tree = None

    local_src_weights = distribute_source_weights(
        queue, source_weights, global_tree, local_data, comm=comm)

    # }}}

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps = wrangler.form_multipoles(
            local_trav.level_start_source_box_nrs,
            local_trav.source_boxes,
            local_src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    wrangler.coarsen_multipoles(
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
        communicate_mpoles(wrangler, comm, local_trav, mpole_exps)

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    potentials = wrangler.eval_direct(
            local_trav.target_boxes,
            local_trav.neighbor_source_boxes_starts,
            local_trav.neighbor_source_boxes_lists,
            local_src_weights)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps = wrangler.multipole_to_local(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_trav.from_sep_siblings_starts,
            local_trav.from_sep_siblings_lists,
            mpole_exps)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            local_trav.target_boxes_sep_smaller_by_source_level,
            local_trav.from_sep_smaller_by_level,
            mpole_exps)

    # these potentials are called beta in [1]

    if local_trav.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                     "('list 3 close')")
        potentials = potentials + wrangler.eval_direct(
                local_trav.target_boxes,
                local_trav.from_sep_close_smaller_starts,
                local_trav.from_sep_close_smaller_lists,
                local_src_weights)

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_trav.from_sep_bigger_starts,
            local_trav.from_sep_bigger_lists,
            local_src_weights)

    if local_trav.from_sep_close_bigger_starts is not None:
        logger.debug("evaluate separated close bigger interactions directly "
                     "('list 4 close')")

        potentials = potentials + wrangler.eval_direct(
                local_trav.target_or_target_parent_boxes,
                local_trav.from_sep_close_bigger_starts,
                local_trav.from_sep_close_bigger_lists,
                local_src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    wrangler.refine_locals(
            local_trav.level_start_target_or_target_parent_box_nrs,
            local_trav.target_or_target_parent_boxes,
            local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")
    potentials = potentials + wrangler.eval_locals(
            local_trav.level_start_target_box_nrs,
            local_trav.target_boxes,
            local_exps)

    # }}}

    potentials_mpi_type = dtype_to_mpi(potentials.dtype)
    if current_rank == 0:
        potentials_all_ranks = np.empty((total_rank,), dtype=object)
        potentials_all_ranks[0] = potentials
        for i in range(1, total_rank):
            potentials_all_ranks[i] = np.empty(
                (local_data[i].ntargets,), dtype=potentials.dtype)
            comm.Recv([potentials_all_ranks[i], potentials_mpi_type],
                      source=i, tag=MPITags["GATHER_POTENTIALS"])
    else:
        comm.Send([potentials, potentials_mpi_type],
                  dest=0, tag=MPITags["GATHER_POTENTIALS"])

    if current_rank == 0:

        potentials = np.empty((global_wrangler.tree.ntargets,),
                              dtype=potentials.dtype)

        for irank in range(total_rank):
            potentials[local_data[irank].tgt_idx] = potentials_all_ranks[irank]

        logger.debug("reorder potentials")
        result = global_wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = global_wrangler.finalize_potentials(result)

        logger.info("Distributed FMM evaluation completes in {} sec.".format(
            str(time.time() - start_time)
        ))

        return result
