from __future__ import division
from mpi4py import MPI
import numpy as np
import loopy as lp
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import GenericScanKernel
from pytools import memoize_in, memoize_method
from boxtree import Tree


__copyright__ = "Copyright (C) 2012 Andreas Kloeckner \
                 Copyright (C) 2017 Hao Gao"

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
logger = logging.getLogger(__name__)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
print("Process %d of %d on %s with ctx %s.\n" % (
    MPI.COMM_WORLD.Get_rank(),
    MPI.COMM_WORLD.Get_size(),
    MPI.Get_processor_name(),
    queue.context.devices))


def tree_to_device(queue, tree, additional_fields_to_device=[]):
    field_to_device = [
        "box_centers", "box_child_ids", "box_flags", "box_levels",
        "box_parent_ids", "box_source_counts_cumul",
        "box_source_counts_nonchild", "box_source_starts",
        "box_target_counts_cumul", "box_target_counts_nonchild",
        "box_target_starts", "level_start_box_nrs_dev", "sources", "targets",
    ] + additional_fields_to_device
    d_tree = tree.copy()
    for field in field_to_device:
        current_obj = d_tree.__getattribute__(field)
        if current_obj.dtype == object:
            new_obj = np.empty_like(current_obj)
            for i in range(current_obj.shape[0]):
                new_obj[i] = cl.array.to_device(queue, current_obj[i])
            d_tree.__setattr__(field, new_obj)
        else:
            d_tree.__setattr__(
                field, cl.array.to_device(queue, current_obj))

    if tree.sources_have_extent:
        d_tree.source_radii = cl.array.to_device(queue, d_tree.source_radii)
    if tree.targets_have_extent:
        d_tree.target_radii = cl.array.to_device(queue, d_tree.target_radii)

    return d_tree


class LocalTree(Tree):
    """
    .. attribute:: box_to_user_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_user_lists

        ``int32 [*]``

        A :ref:`csr` array. For each box, the list of processes which own
        targets that *use* the multipole expansion at this box, via either List
        3 or (possibly downward propagated from an ancestor) List 2.
    """

    @property
    def nboxes(self):
        return self.box_source_starts.shape[0]

    @property
    def nsources(self):
        return self.sources[0].shape[0]

    @property
    def ntargets(self):
        return self.targets[0].shape[0]

    @classmethod
    def copy_from_global_tree(cls, global_tree, responsible_boxes_list,
                              ancestor_mask, box_to_user_starts,
                              box_to_user_lists):
        local_tree = global_tree.copy(
            responsible_boxes_list=responsible_boxes_list,
            ancestor_mask=ancestor_mask,
            box_to_user_starts=box_to_user_starts,
            box_to_user_lists=box_to_user_lists)
        local_tree.__class__ = cls
        return local_tree

    def to_device(self, queue):
        additional_fields_to_device = ["responsible_boxes_list", "ancestor_mask",
            "box_to_user_starts", "box_to_user_lists"]

        return tree_to_device(queue, self, additional_fields_to_device)


# {{{ parallel fmm wrangler

class DistributedFMMLibExpansionWranglerCodeContainer(object):

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

    def get_wrangler(self, queue, tree, helmholtz_k, fmm_order):
        return DistributedFMMLibExpansionWrangler(self, queue, tree, helmholtz_k,
                                               fmm_order)


from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler


class DistributedFMMLibExpansionWrangler(FMMLibExpansionWrangler):

    def __init__(self, code_container, queue, tree, helmholtz_k, fmm_order):
        """
        :arg fmm_order: Only supports single order for now
        """
        def fmm_level_to_nterms(tree, level):
            return fmm_order

        FMMLibExpansionWrangler.__init__(self, tree, helmholtz_k,
                                         fmm_level_to_nterms)
        self.queue = queue
        self.fmm_order = fmm_order
        self.code_container = code_container

    def slice_mpoles(self, mpoles, slice_indices):
        mpoles = mpoles.reshape((-1,) + self.expansion_shape(self.fmm_order))
        return mpoles[slice_indices, :].reshape((-1,))

    def update_mpoles(self, mpoles, mpole_updates, slice_indices):
        """
        :arg mpole_updates: The first *len(slice_indices)* entries should contain
            the values to add to *mpoles*
        """
        mpoles = mpoles.reshape((-1,) + self.expansion_shape(self.fmm_order))
        mpole_updates = mpole_updates.reshape(
            (-1,) + self.expansion_shape(self.fmm_order))
        mpoles[slice_indices, :] += mpole_updates[:len(slice_indices), :]

    def empty_box_in_subrange_mask(self):
        return cl.array.empty(self.queue, self.tree.nboxes, dtype=np.int8)

    def find_boxes_used_by_subrange(self, box_in_subrange, subrange,
                                    box_to_user_starts, box_to_user_lists):
        knl = self.code_container.find_boxes_used_by_subrange_kernel()
        knl(self.queue,
            subrange_start=subrange[0],
            subrange_end=subrange[1],
            box_to_user_starts=box_to_user_starts,
            box_to_user_lists=box_to_user_lists,
            box_in_subrange=box_in_subrange)

        box_in_subrange.finish()

# }}}


class MPITags():
    DIST_TREE = 0
    DIST_WEIGHT = 1
    GATHER_POTENTIALS = 2
    REDUCE_POTENTIALS = 3
    REDUCE_INDICES = 4


def partition_work(traversal, total_rank, queue):
    """ This function returns a pyopencl array of size total_rank*nboxes, where
    the (i,j) entry is 1 iff rank i is responsible for box j.
    """
    tree = traversal.tree

    workload = np.zeros((tree.nboxes,), dtype=np.float64)
    for i in range(traversal.target_boxes.shape[0]):
        box_idx = traversal.target_boxes[i]
        box_ntargets = tree.box_target_counts_nonchild[box_idx]

        # workload for list 1
        start = traversal.neighbor_source_boxes_starts[i]
        end = traversal.neighbor_source_boxes_starts[i + 1]
        list1 = traversal.neighbor_source_boxes_lists[start:end]
        particle_count = 0
        for j in range(list1.shape[0]):
            particle_count += tree.box_source_counts_nonchild[list1[j]]
        workload[box_idx] += box_ntargets * particle_count

        # workload for list 3 near
        if tree.targets_have_extent:
            start = traversal.from_sep_close_smaller_starts[i]
            end = traversal.from_sep_close_smaller_starts[i + 1]
            list3_near = traversal.from_sep_close_smaller_lists[start:end]
            particle_count = 0
            for j in range(list3_near.shape[0]):
                particle_count += tree.box_source_counts_nonchild[list3_near[j]]
            workload[box_idx] += box_ntargets * particle_count

    for i in range(tree.nboxes):
        # workload for multipole calculation
        workload[i] += tree.box_source_counts_nonchild[i] * 5

    total_workload = 0
    for i in range(tree.nboxes):
        total_workload += workload[i]

    dfs_order = np.empty((tree.nboxes,), dtype=tree.box_id_dtype)
    idx = 0
    stack = [0]
    while len(stack) > 0:
        box_id = stack.pop()
        dfs_order[idx] = box_id
        idx += 1
        for i in range(2**tree.dimensions):
            child_box_id = tree.box_child_ids[i][box_id]
            if child_box_id > 0:
                stack.append(child_box_id)

    responsible_boxes_mask = np.zeros((total_rank, tree.nboxes), dtype=np.int8)
    responsible_boxes_list = np.empty((total_rank,), dtype=object)

    rank = 0
    start = 0
    workload_count = 0
    for i in range(tree.nboxes):
        box_idx = dfs_order[i]
        responsible_boxes_mask[rank][box_idx] = 1
        workload_count += workload[box_idx]
        if (workload_count > (rank + 1)*total_workload/total_rank or
                i == tree.nboxes - 1):
            responsible_boxes_list[rank] = cl.array.to_device(
                queue, dfs_order[start:i+1])
            start = i + 1
            rank += 1

    responsible_boxes_mask = cl.array.to_device(queue, responsible_boxes_mask)
    return responsible_boxes_mask, responsible_boxes_list


def get_gen_local_tree_helper(queue, tree):
    d_tree = tree_to_device(queue, tree)

    particle_mask_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        arguments=Template("""
            __global char *responsible_boxes,
            __global ${particle_id_t} *box_particle_starts,
            __global ${particle_id_t} *box_particle_counts_nonchild,
            __global ${particle_id_t} *particle_mask
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
        operation=Template("""
            if(responsible_boxes[i]) {
                for(${particle_id_t} pid = box_particle_starts[i];
                    pid < box_particle_starts[i] + box_particle_counts_nonchild[i];
                    ++pid) {
                    particle_mask[pid] = 1;
                }
            }
        """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype))
    )

    mask_scan_knl = GenericScanKernel(
        queue.context, tree.particle_id_dtype,
        arguments=Template("""
            __global ${mask_t} *ary,
            __global ${mask_t} *scan
            """, strict_undefined=True).render(
            mask_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
        input_expr="ary[i]",
        scan_expr="a+b", neutral="0",
        output_statement="scan[i + 1] = item;"
    )

    fetch_local_paticles_arguments = Template("""
        __global const ${mask_t} *particle_mask,
        __global const ${mask_t} *particle_scan
        % for dim in range(ndims):
            , __global const ${coord_t} *particles_${dim}
        % endfor
        % for dim in range(ndims):
            , __global ${coord_t} *local_particles_${dim}
        % endfor
        % if particles_have_extent:
            , __global const ${coord_t} *particle_radii
            , __global ${coord_t} *local_particle_radii
        % endif
    """, strict_undefined=True)

    fetch_local_particles_prg = Template("""
        if(particle_mask[i]) {
            ${particle_id_t} des = particle_scan[i];
            % for dim in range(ndims):
                local_particles_${dim}[des] = particles_${dim}[i];
            % endfor
            % if particles_have_extent:
                local_particle_radii[des] = particle_radii[i];
            % endif
        }
    """, strict_undefined=True)

    fetch_local_src_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(tree.particle_id_dtype),
            coord_t=dtype_to_ctype(tree.coord_dtype),
            ndims=tree.dimensions,
            particles_have_extent=tree.sources_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
            ndims=tree.dimensions,
            particles_have_extent=tree.sources_have_extent
        )
    )

    fetch_local_tgt_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(tree.particle_id_dtype),
            coord_t=dtype_to_ctype(tree.coord_dtype),
            ndims=tree.dimensions,
            particles_have_extent=tree.targets_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
            ndims=tree.dimensions,
            particles_have_extent=tree.targets_have_extent
        )
    )

    generate_box_particle_starts = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *particle_scan,
            __global ${particle_id_t} *new_starts
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
        "new_starts[i] = particle_scan[old_starts[i]]",
        name="generate_box_particle_starts"
    )

    generate_box_particle_counts_nonchild = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global char *res_boxes,
            __global ${particle_id_t} *old_counts_nonchild,
            __global ${particle_id_t} *new_counts_nonchild
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
        "if(res_boxes[i]) new_counts_nonchild[i] = old_counts_nonchild[i];"
    )

    generate_box_particle_counts_cumul = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *old_counts_cumul,
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *new_counts_cumul,
            __global ${particle_id_t} *particle_scan
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
        """
        new_counts_cumul[i] =
            particle_scan[old_starts[i] + old_counts_cumul[i]] -
            particle_scan[old_starts[i]]
        """
    )

    def gen_local_tree_helper(src_box_mask, tgt_box_mask, local_tree):
        """ This helper function generates a copy of the tree but with subset of
            particles, and fetch the generated fields to *local_tree*.
        """
        nsources = tree.nsources

        # source particle mask
        src_particle_mask = cl.array.zeros(queue, (nsources,),
                                           dtype=tree.particle_id_dtype)
        particle_mask_knl(src_box_mask,
                          d_tree.box_source_starts,
                          d_tree.box_source_counts_nonchild,
                          src_particle_mask)

        # scan of source particle mask
        src_particle_scan = cl.array.empty(queue, (nsources + 1,),
                                           dtype=tree.particle_id_dtype)
        src_particle_scan[0] = 0
        mask_scan_knl(src_particle_mask, src_particle_scan)

        # local sources
        local_nsources = src_particle_scan[-1].get(queue)
        local_sources = np.empty((tree.dimensions,), dtype=object)
        for i in range(tree.dimensions):
            local_sources[i] = cl.array.empty(queue, (local_nsources,),
                                              dtype=tree.coord_dtype)
        assert(tree.sources_have_extent is False)
        fetch_local_src_knl(src_particle_mask, src_particle_scan,
                            *d_tree.sources.tolist(),
                            *local_sources.tolist())

        # box_source_starts
        local_box_source_starts = cl.array.empty(queue, (tree.nboxes,),
                                                 dtype=tree.particle_id_dtype)
        generate_box_particle_starts(d_tree.box_source_starts, src_particle_scan,
                                     local_box_source_starts)

        # box_source_counts_nonchild
        local_box_source_counts_nonchild = cl.array.zeros(
            queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
        generate_box_particle_counts_nonchild(src_box_mask,
                                              d_tree.box_source_counts_nonchild,
                                              local_box_source_counts_nonchild)

        # box_source_counts_cumul
        local_box_source_counts_cumul = cl.array.empty(
            queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
        generate_box_particle_counts_cumul(d_tree.box_source_counts_cumul,
                                           d_tree.box_source_starts,
                                           local_box_source_counts_cumul,
                                           src_particle_scan)

        ntargets = tree.ntargets
        # target particle mask
        tgt_particle_mask = cl.array.zeros(queue, (ntargets,),
                                           dtype=tree.particle_id_dtype)
        particle_mask_knl(tgt_box_mask,
                          d_tree.box_target_starts,
                          d_tree.box_target_counts_nonchild,
                          tgt_particle_mask)

        # scan of target particle mask
        tgt_particle_scan = cl.array.empty(queue, (ntargets + 1,),
                                           dtype=tree.particle_id_dtype)
        tgt_particle_scan[0] = 0
        mask_scan_knl(tgt_particle_mask, tgt_particle_scan)

        # local targets
        local_ntargets = tgt_particle_scan[-1].get(queue)
        local_targets = np.empty((tree.dimensions,), dtype=object)
        for i in range(tree.dimensions):
            local_targets[i] = cl.array.empty(queue, (local_ntargets,),
                                              dtype=tree.coord_dtype)
        if tree.targets_have_extent:
            local_target_radii = cl.array.empty(queue, (local_ntargets,),
                                                dtype=tree.coord_dtype)
            fetch_local_tgt_knl(tgt_particle_mask, tgt_particle_scan,
                                *d_tree.targets.tolist(), *local_targets.tolist(),
                                d_tree.target_radii, local_target_radii)
        else:
            fetch_local_tgt_knl(tgt_particle_mask, tgt_particle_scan,
                                *d_tree.targets.tolist(),
                                *local_targets.tolist())

        # box_target_starts
        local_box_target_starts = cl.array.empty(queue, (tree.nboxes,),
                                                 dtype=tree.particle_id_dtype)
        generate_box_particle_starts(d_tree.box_target_starts, tgt_particle_scan,
                                     local_box_target_starts)

        # box_target_counts_nonchild
        local_box_target_counts_nonchild = cl.array.zeros(
            queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
        generate_box_particle_counts_nonchild(tgt_box_mask,
                                              d_tree.box_target_counts_nonchild,
                                              local_box_target_counts_nonchild)

        # box_target_counts_cumul
        local_box_target_counts_cumul = cl.array.empty(
            queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
        generate_box_particle_counts_cumul(d_tree.box_target_counts_cumul,
                                           d_tree.box_target_starts,
                                           local_box_target_counts_cumul,
                                           tgt_particle_scan)

        # Fetch fields to local_tree
        for i in range(tree.dimensions):
            local_sources[i] = local_sources[i].get(queue=queue)
        local_tree.sources = local_sources
        for i in range(tree.dimensions):
            local_targets[i] = local_targets[i].get(queue=queue)
        local_tree.targets = local_targets
        if tree.targets_have_extent:
            local_tree.target_radii = local_target_radii.get(queue=queue)
        local_tree.box_source_starts = local_box_source_starts.get(queue=queue)
        local_tree.box_source_counts_nonchild = \
            local_box_source_counts_nonchild.get(queue=queue)
        local_tree.box_source_counts_cumul = \
            local_box_source_counts_cumul.get(queue=queue)
        local_tree.box_target_starts = local_box_target_starts.get(queue=queue)
        local_tree.box_target_counts_nonchild = \
            local_box_target_counts_nonchild.get(queue=queue)
        local_tree.box_target_counts_cumul = \
            local_box_target_counts_cumul.get(queue=queue)
        return (src_particle_mask, src_particle_scan, tgt_particle_mask,
                tgt_particle_scan)

    return gen_local_tree_helper


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


def generate_local_tree(traversal, src_weights, comm=MPI.COMM_WORLD):
    # {{{ kernel to mark if a box mpole is used by a process via an interaction list

    @memoize_in(generate_local_tree, "loopy_cache")
    def get_box_mpole_is_used_marker_kernel():
        knl = lp.make_kernel(
            [
                "{[irank] : 0 <= irank < total_rank}",
                "{[itgt_box] : 0 <= itgt_box < ntgt_boxes}",
                "{[isrc_box] : isrc_box_start <= isrc_box < isrc_box_end}",
            ],
            """
            for irank, itgt_box
                <> tgt_ibox = target_boxes[itgt_box]
                <> is_relevant = relevant_boxes_mask[irank, tgt_ibox]
                if is_relevant
                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_end = source_box_starts[itgt_box + 1]
                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box]
                        box_mpole_is_used[irank, src_ibox] = 1
                    end
                end
            end
            """,
            [
                lp.ValueArg("nboxes", np.int32),
                lp.GlobalArg("relevant_boxes_mask, box_mpole_is_used",
                              shape=("total_rank", "nboxes")),
                lp.GlobalArg("source_box_lists", shape=None),
                "..."
            ],
            default_offset=lp.auto)

        knl = lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0", inner_tag="l.0")
        return knl

    # }}}

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # {{{ Construct local tree for each rank on root
    local_target = {"mask": None, "scan": None, "size": None}
    if current_rank == 0:
        tree = traversal.tree
        local_tree = np.empty((total_rank,), dtype=object)
        local_target_mask = np.empty((total_rank,), dtype=object)
        local_target_scan = np.empty((total_rank,), dtype=object)
        local_ntargets = np.empty((total_rank,), dtype=tree.particle_id_dtype)
        local_target["mask"] = local_target_mask
        local_target["scan"] = local_target_scan
        local_target["size"] = local_ntargets

        d_box_parent_ids = cl.array.to_device(queue, tree.box_parent_ids)

        # {{{ Partition the work

        # Each rank is responsible for calculating the multiple expansion as well as
        # evaluating target potentials in *responsible_boxes*
        responsible_boxes_mask, responsible_boxes_list = \
            partition_work(traversal, total_rank, queue)

        # Calculate ancestors of responsible boxes
        ancestor_boxes = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                        dtype=np.int8)
        for rank in range(total_rank):
            ancestor_boxes_last = responsible_boxes_mask[rank, :].copy()
            mark_parent_knl = cl.elementwise.ElementwiseKernel(
                ctx,
                "__global char *current, __global char *parent, "
                "__global %s *box_parent_ids" % dtype_to_ctype(tree.box_id_dtype),
                "if(i != 0 && current[i]) parent[box_parent_ids[i]] = 1"
            )
            while ancestor_boxes_last.any():
                ancestor_boxes_new = cl.array.zeros(queue, (tree.nboxes,),
                                                    dtype=np.int8)
                mark_parent_knl(ancestor_boxes_last, ancestor_boxes_new,
                                d_box_parent_ids)
                ancestor_boxes_new = ancestor_boxes_new & (~ancestor_boxes[rank, :])
                ancestor_boxes[rank, :] = \
                    ancestor_boxes[rank, :] | ancestor_boxes_new
                ancestor_boxes_last = ancestor_boxes_new

        # In order to evaluate, each rank needs sources in boxes in
        # *src_boxes_mask*
        src_boxes_mask = responsible_boxes_mask.copy()

        # Add list 1 and list 4 to src_boxes_mask
        add_interaction_list_boxes = cl.elementwise.ElementwiseKernel(
            ctx,
            Template("""
                __global ${box_id_t} *box_list,
                __global char *responsible_boxes_mask,
                __global ${box_id_t} *interaction_boxes_starts,
                __global ${box_id_t} *interaction_boxes_lists,
                __global char *src_boxes_mask
            """, strict_undefined=True).render(
                box_id_t=dtype_to_ctype(tree.box_id_dtype)
            ),
            Template(r"""
                typedef ${box_id_t} box_id_t;
                box_id_t current_box = box_list[i];
                if(responsible_boxes_mask[current_box]) {
                    for(box_id_t box_idx = interaction_boxes_starts[i];
                        box_idx < interaction_boxes_starts[i + 1];
                        ++box_idx)
                        src_boxes_mask[interaction_boxes_lists[box_idx]] = 1;
                }
            """, strict_undefined=True).render(
                box_id_t=dtype_to_ctype(tree.box_id_dtype)
            ),
        )

        for rank in range(total_rank):
            # Add list 1 of responsible boxes
            d_target_boxes = cl.array.to_device(queue, traversal.target_boxes)
            d_neighbor_source_boxes_starts = cl.array.to_device(
                queue, traversal.neighbor_source_boxes_starts)
            d_neighbor_source_boxes_lists = cl.array.to_device(
                queue, traversal.neighbor_source_boxes_lists)
            add_interaction_list_boxes(
                d_target_boxes, responsible_boxes_mask[rank],
                d_neighbor_source_boxes_starts,
                d_neighbor_source_boxes_lists, src_boxes_mask[rank],
                range=range(0, traversal.target_boxes.shape[0]))

            # Add list 4 of responsible boxes or ancestor boxes
            d_target_or_target_parent_boxes = cl.array.to_device(
                queue, traversal.target_or_target_parent_boxes)
            d_from_sep_bigger_starts = cl.array.to_device(
                queue, traversal.from_sep_bigger_starts)
            d_from_sep_bigger_lists = cl.array.to_device(
                queue, traversal.from_sep_bigger_lists)
            add_interaction_list_boxes(
                d_target_or_target_parent_boxes,
                responsible_boxes_mask[rank] | ancestor_boxes[rank],
                d_from_sep_bigger_starts, d_from_sep_bigger_lists,
                src_boxes_mask[rank],
                range=range(0, traversal.target_or_target_parent_boxes.shape[0]))

            if tree.targets_have_extent:
                d_from_sep_close_bigger_starts = cl.array.to_device(
                    queue, traversal.from_sep_close_bigger_starts)
                d_from_sep_close_bigger_lists = cl.array.to_device(
                    queue, traversal.from_sep_close_bigger_lists)
                add_interaction_list_boxes(
                    d_target_or_target_parent_boxes,
                    responsible_boxes_mask[rank] | ancestor_boxes[rank],
                    d_from_sep_close_bigger_starts,
                    d_from_sep_close_bigger_lists,
                    src_boxes_mask[rank]
                )

                # Add list 3 direct
                d_from_sep_close_smaller_starts = cl.array.to_device(
                    queue, traversal.from_sep_close_smaller_starts)
                d_from_sep_close_smaller_lists = cl.array.to_device(
                    queue, traversal.from_sep_close_smaller_lists)

                add_interaction_list_boxes(
                    d_target_boxes,
                    responsible_boxes_mask[rank],
                    d_from_sep_close_smaller_starts,
                    d_from_sep_close_smaller_lists,
                    src_boxes_mask[rank]
                )

        # {{{ compute box_to_user

        logger.debug("computing box_to_user: start")

        box_mpole_is_used = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                           dtype=np.int8)
        knl = get_box_mpole_is_used_marker_kernel()

        # A mpole is used by process p if it is in the List 2 of either a box
        # owned by p or one of its ancestors.
        knl(queue,
            total_rank=total_rank,
            nboxes=tree.nboxes,
            target_boxes=traversal.target_or_target_parent_boxes,
            relevant_boxes_mask=responsible_boxes_mask | ancestor_boxes,
            source_box_starts=traversal.from_sep_siblings_starts,
            source_box_lists=traversal.from_sep_siblings_lists,
            box_mpole_is_used=box_mpole_is_used)

        box_mpole_is_used.finish()

        # A mpole is used by process p if it is in the List 3 of a box owned by p.
        for level in range(tree.nlevels):
            source_box_starts = traversal.from_sep_smaller_by_level[level].starts
            source_box_lists = traversal.from_sep_smaller_by_level[level].lists
            knl(queue,
                total_rank=total_rank,
                nboxes=tree.nboxes,
                target_boxes=traversal.target_boxes,
                relevant_boxes_mask=responsible_boxes_mask,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                box_mpole_is_used=box_mpole_is_used)

            box_mpole_is_used.finish()

        from boxtree.tools import MaskCompressorKernel
        matcompr = MaskCompressorKernel(ctx)
        (
            box_to_user_starts,
            box_to_user_lists,
            evt) = matcompr(queue, box_mpole_is_used.transpose(),
                            list_dtype=np.int32)

        cl.wait_for_events([evt])
        del box_mpole_is_used

        logger.debug("computing box_to_user: done")

        # }}}

        # Convert src_weights to tree order
        src_weights = src_weights[tree.user_source_ids]
        src_weights = cl.array.to_device(queue, src_weights)
        local_src_weights = np.empty((total_rank,), dtype=object)

        # request objects for non-blocking communication
        tree_req = np.empty((total_rank,), dtype=object)
        weight_req = np.empty((total_rank,), dtype=object)

        gen_local_tree_helper = get_gen_local_tree_helper(queue, tree)
        gen_local_weights_helper = get_gen_local_weights_helper(queue,
            tree.particle_id_dtype, src_weights.dtype)
        for rank in range(total_rank):
            local_tree[rank] = LocalTree.copy_from_global_tree(
                tree, responsible_boxes_list[rank].get(),
                ancestor_boxes[rank].get(),
                box_to_user_starts.get(),
                box_to_user_lists.get())

            local_tree[rank].user_source_ids = None
            local_tree[rank].sorted_target_ids = None

            (src_mask, src_scan, tgt_mask, tgt_scan) = \
                gen_local_tree_helper(src_boxes_mask[rank],
                                      responsible_boxes_mask[rank],
                                      local_tree[rank])

            local_src_weights[rank] = gen_local_weights_helper(
                src_weights, src_mask, src_scan)

            local_target_mask[rank] = tgt_mask
            local_target_scan[rank] = tgt_scan
            local_ntargets[rank] = tgt_scan[-1].get(queue)

            tree_req[rank] = comm.isend(local_tree[rank], dest=rank,
                                        tag=MPITags.DIST_TREE)
            weight_req[rank] = comm.isend(local_src_weights[rank], dest=rank,
                                          tag=MPITags.DIST_WEIGHT)

    # }}}

    # Recieve the local trav from root
    if current_rank == 0:
        for rank in range(1, total_rank):
            tree_req[rank].wait()
        local_tree = local_tree[0]
    else:
        local_tree = comm.recv(source=0, tag=MPITags.DIST_TREE)

    # Recieve source weights from root
    if current_rank == 0:
        for rank in range(1, total_rank):
            weight_req[rank].wait()
        local_src_weights = local_src_weights[0]
    else:
        local_src_weights = comm.recv(source=0, tag=MPITags.DIST_WEIGHT)

    rtv = (local_tree, local_src_weights, local_target)

    # Recieve box extent
    if current_rank == 0:
        box_target_bounding_box_min = traversal.box_target_bounding_box_min
        box_target_bounding_box_max = traversal.box_target_bounding_box_max
    else:
        box_target_bounding_box_min = np.empty(
            (local_tree.dimensions, local_tree.aligned_nboxes),
            dtype=local_tree.coord_dtype
        )
        box_target_bounding_box_max = np.empty(
            (local_tree.dimensions, local_tree.aligned_nboxes),
            dtype=local_tree.coord_dtype
        )
    comm.Bcast(box_target_bounding_box_min, root=0)
    comm.Bcast(box_target_bounding_box_max, root=0)
    box_bounding_box = {
        "min": box_target_bounding_box_min,
        "max": box_target_bounding_box_max
    }
    rtv += (box_bounding_box,)

    return rtv


def generate_local_travs(local_tree, local_src_weights, box_bounding_box=None,
                         comm=MPI.COMM_WORLD):
    d_tree = local_tree.to_device(queue)

    # Modify box flags for targets
    from boxtree import box_flags_enum
    box_flag_t = dtype_to_ctype(box_flags_enum.dtype)
    modify_target_flags_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *box_target_counts_nonchild,
            __global ${particle_id_t} *box_target_counts_cumul,
            __global ${box_flag_t} *box_flags
        """).render(particle_id_t=dtype_to_ctype(local_tree.particle_id_dtype),
                    box_flag_t=box_flag_t),
        Template("""
            box_flags[i] &= (~${HAS_OWN_TARGETS});
            box_flags[i] &= (~${HAS_CHILD_TARGETS});
            if(box_target_counts_nonchild[i]) box_flags[i] |= ${HAS_OWN_TARGETS};
            if(box_target_counts_nonchild[i] < box_target_counts_cumul[i])
                box_flags[i] |= ${HAS_CHILD_TARGETS};
        """).render(HAS_OWN_TARGETS=("(" + box_flag_t + ") " +
                                     str(box_flags_enum.HAS_OWN_TARGETS)),
                    HAS_CHILD_TARGETS=("(" + box_flag_t + ") " +
                                       str(box_flags_enum.HAS_CHILD_TARGETS)))
    )
    modify_target_flags_knl(d_tree.box_target_counts_nonchild,
                            d_tree.box_target_counts_cumul,
                            d_tree.box_flags)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(queue.context)
    d_trav_global, _ = tg(queue, d_tree, debug=True,
                          box_bounding_box=box_bounding_box)
    trav_global = d_trav_global.get(queue=queue)

    # Source flags
    d_tree.box_flags = d_tree.box_flags & 250
    modify_own_sources_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${box_id_t} *responsible_box_list,
            __global ${box_flag_t} *box_flags
        """).render(box_id_t=dtype_to_ctype(local_tree.box_id_dtype),
                    box_flag_t=box_flag_t),
        Template(r"""
            box_flags[responsible_box_list[i]] |= ${HAS_OWN_SOURCES};
        """).render(HAS_OWN_SOURCES=("(" + box_flag_t + ") " +
                                     str(box_flags_enum.HAS_OWN_SOURCES)))
        )
    modify_child_sources_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global char *ancestor_box_mask,
            __global ${box_flag_t} *box_flags
        """).render(box_flag_t=box_flag_t),
        Template("""
            if(ancestor_box_mask[i]) box_flags[i] |= ${HAS_CHILD_SOURCES};
        """).render(HAS_CHILD_SOURCES=("(" + box_flag_t + ") " +
                                       str(box_flags_enum.HAS_CHILD_SOURCES)))
    )
    modify_own_sources_knl(d_tree.responsible_boxes_list, d_tree.box_flags)
    modify_child_sources_knl(d_tree.ancestor_mask, d_tree.box_flags)

    d_trav_local, _ = tg(queue, d_tree, debug=True,
                         box_bounding_box=box_bounding_box)
    trav_local = d_trav_local.get(queue=queue)

    return trav_local, trav_global


# {{{ communicate mpoles

def communicate_mpoles(wrangler, comm, trav, mpole_exps, return_stats=False):
    """Based on Algorithm 3: Reduce and Scatter in [1].

    The main idea is to mimic a hypercube allreduce, but to reduce bandwidth by
    sending only necessary information.

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
                                 tag=MPITags.REDUCE_POTENTIALS)
                send_requests.append(req)

                req = comm.Isend(relevant_boxes_list, dest=sink,
                                 tag=MPITags.REDUCE_INDICES)
                send_requests.append(req)

        # Receive data from other processors.
        for source in comm_pattern.sources():
            comm.Recv(mpole_exps_buf, source=source, tag=MPITags.REDUCE_POTENTIALS)

            status = MPI.Status()
            comm.Recv(boxes_list_buf, source=source, tag=MPITags.REDUCE_INDICES,
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
    logger.debug("communicate multipoles: done in %.2f s" % stats["total_time"])

    if return_stats:
        return stats

# }}}


def drive_dfmm(wrangler, trav_local, trav_global, local_src_weights, global_wrangler,
               local_target_mask, local_target_scan, local_ntargets,
               comm=MPI.COMM_WORLD, _communicate_mpoles_via_allreduce=False):
    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # {{{ "Step 2.1:" Construct local multipoles

    import time
    logger.debug("construct local multipoles")

    mpole_exps = wrangler.form_multipoles(
            trav_local.level_start_source_box_nrs,
            trav_local.source_boxes,
            local_src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    wrangler.coarsen_multipoles(
            trav_local.level_start_source_parent_box_nrs,
            trav_local.source_parent_boxes,
            mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ Communicate mpoles

    last_time = time.time()

    if _communicate_mpoles_via_allreduce:
        mpole_exps_all = np.zeros_like(mpole_exps)
        comm.Allreduce(mpole_exps, mpole_exps_all)
        mpole_exps = mpole_exps_all
    else:
        communicate_mpoles(wrangler, comm, trav_local, mpole_exps)

    print("Communication: " + str(time.time()-last_time))

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")
    last_time = time.time()

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    potentials = wrangler.eval_direct(
            trav_global.target_boxes,
            trav_global.neighbor_source_boxes_starts,
            trav_global.neighbor_source_boxes_lists,
            local_src_weights)

    # these potentials are called alpha in [1]
    print("List 1: " + str(time.time()-last_time))

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps = wrangler.multipole_to_local(
            trav_global.level_start_target_or_target_parent_box_nrs,
            trav_global.target_or_target_parent_boxes,
            trav_global.from_sep_siblings_starts,
            trav_global.from_sep_siblings_lists,
            mpole_exps)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles
    last_time = time.time()

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            trav_global.level_start_target_box_nrs,
            trav_global.target_boxes,
            trav_global.from_sep_smaller_by_level,
            mpole_exps)

    # these potentials are called beta in [1]

    if trav_global.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                     "('list 3 close')")
        potentials = potentials + wrangler.eval_direct(
                trav_global.target_boxes,
                trav_global.from_sep_close_smaller_starts,
                trav_global.from_sep_close_smaller_lists,
                local_src_weights)

    print("List 3: " + str(time.time()-last_time))

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            trav_global.level_start_target_or_target_parent_box_nrs,
            trav_global.target_or_target_parent_boxes,
            trav_global.from_sep_bigger_starts,
            trav_global.from_sep_bigger_lists,
            local_src_weights)

    if trav_global.from_sep_close_bigger_starts is not None:
        logger.debug("evaluate separated close bigger interactions directly "
                     "('list 4 close')")

        potentials = potentials + wrangler.eval_direct(
                trav_global.target_or_target_parent_boxes,
                trav_global.from_sep_close_bigger_starts,
                trav_global.from_sep_close_bigger_lists,
                local_src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    wrangler.refine_locals(
            trav_global.level_start_target_or_target_parent_box_nrs,
            trav_global.target_or_target_parent_boxes,
            local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")
    potentials = potentials + wrangler.eval_locals(
            trav_global.level_start_target_box_nrs,
            trav_global.target_boxes,
            local_exps)

    # }}}

    potentials_mpi_type = MPI._typedict[potentials.dtype.char]
    if current_rank == 0:
        potentials_all_ranks = np.empty((total_rank,), dtype=object)
        potentials_all_ranks[0] = potentials
        for i in range(1, total_rank):
            potentials_all_ranks[i] = np.empty(
                (local_ntargets[i],), dtype=potentials.dtype)
            comm.Recv([potentials_all_ranks[i], potentials_mpi_type],
                      source=i, tag=MPITags.GATHER_POTENTIALS)
    else:
        comm.Send([potentials, potentials_mpi_type],
                  dest=0, tag=MPITags.GATHER_POTENTIALS)

    if current_rank == 0:
        d_potentials = cl.array.empty(queue, (global_wrangler.tree.ntargets,),
                                      dtype=potentials.dtype)
        fill_potentials_knl = cl.elementwise.ElementwiseKernel(
            ctx,
            Template("""
                __global ${particle_id_t} *particle_mask,
                __global ${particle_id_t} *particle_scan,
                __global ${potential_t} *local_potentials,
                __global ${potential_t} *potentials
            """).render(
                particle_id_t=dtype_to_ctype(global_wrangler.tree.particle_id_dtype),
                potential_t=dtype_to_ctype(potentials.dtype)),
            r"""
                if(particle_mask[i]) {
                    potentials[i] = local_potentials[particle_scan[i]];
                }
            """
        )

        for i in range(total_rank):
            local_potentials = cl.array.to_device(queue, potentials_all_ranks[i])
            fill_potentials_knl(
                local_target_mask[i], local_target_scan[i],
                local_potentials, d_potentials)

        potentials = d_potentials.get()

        logger.debug("reorder potentials")
        result = global_wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = global_wrangler.finalize_potentials(result)

        logger.info("fmm complete")

        return result
