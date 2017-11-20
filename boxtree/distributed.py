from __future__ import division
from mpi4py import MPI
import numpy as np
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import ExclusiveScanKernel
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


class LocalTree(Tree):

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
    def copy_from_global_tree(cls, global_tree, responsible_boxes):
        local_tree = global_tree.copy(responsible_boxes=responsible_boxes)
        local_tree.__class__ = cls
        return local_tree


def partition_work(tree, total_rank, queue):
    """ This function returns a list of total_rank elements, where element i is a
    pyopencl array of indices of process i's responsible boxes.
    """
    responsible_boxes = []
    num_boxes = tree.box_source_starts.shape[0]
    num_boxes_per_rank = num_boxes // total_rank
    extra_boxes = num_boxes - num_boxes_per_rank * total_rank
    start_idx = 0

    for current_rank in range(extra_boxes):
        end_idx = start_idx + num_boxes_per_rank + 1
        responsible_boxes.append(cl.array.arange(queue, start_idx, end_idx,
                                                 dtype=tree.box_id_dtype))
        start_idx = end_idx

    for current_rank in range(extra_boxes, total_rank):
        end_idx = start_idx + num_boxes_per_rank
        responsible_boxes.append(cl.array.arange(queue, start_idx, end_idx,
                                                 dtype=tree.box_id_dtype))
        start_idx = end_idx

    return responsible_boxes


gen_local_tree_tpl = Template(r"""
typedef ${dtype_to_ctype(tree.box_id_dtype)} box_id_t;
typedef ${dtype_to_ctype(tree.particle_id_dtype)} particle_id_t;
typedef ${dtype_to_ctype(mask_dtype)} mask_t;
typedef ${dtype_to_ctype(tree.coord_dtype)} coord_t;

__kernel void generate_particle_mask(
    __global const box_id_t *res_boxes,
    __global const particle_id_t *box_particle_starts,
    __global const particle_id_t *box_particle_counts_nonchild,
    const int total_num_res_boxes,
    __global mask_t *particle_mask)
{
    /*
     * generate_particle_mask takes the responsible box indices as input and generate
     * a mask for responsible particles.
     */
    int res_box_idx = get_global_id(0);

    if(res_box_idx < total_num_res_boxes) {
        box_id_t cur_box = res_boxes[res_box_idx];
        for(particle_id_t i = box_particle_starts[cur_box];
            i < box_particle_starts[cur_box] + box_particle_counts_nonchild[cur_box];
            i++) {
            particle_mask[i] = 1;
        }
    }
}

__kernel void generate_local_particles(
    const int total_num_particles,
    % for dim in range(ndims):
        __global const coord_t *particles_${dim},
    % endfor
    __global const mask_t *particle_mask,
    __global const mask_t *particle_scan
    % for dim in range(ndims):
        , __global coord_t *local_particles_${dim}
    % endfor
)
{
    /*
     * generate_local_particles generates an array of particles for which a process
     * is responsible for.
     */
    int particle_idx = get_global_id(0);

    if(particle_idx < total_num_particles && particle_mask[particle_idx])
    {
        particle_id_t des = particle_scan[particle_idx];
        % for dim in range(ndims):
            local_particles_${dim}[des] = particles_${dim}[particle_idx];
        % endfor
    }
}
""", strict_undefined=True)

traversal_preamble_tpl = Template(r"""
    #define HAS_CHILD_SOURCES ${HAS_CHILD_SOURCES}
    #define HAS_CHILD_TARGETS ${HAS_CHILD_TARGETS}
    #define HAS_OWN_SOURCES ${HAS_OWN_SOURCES}
    #define HAS_OWN_TARGETS ${HAS_OWN_TARGETS}
    typedef ${box_flag_t} box_flag_t;
    typedef ${box_id_t} box_id_t;
    typedef ${particle_id_t} particle_id_t;
    typedef ${box_level_t} box_level_t;
""", strict_undefined=True)

gen_traversal_tpl = Template(r"""
__kernel void generate_tree_flags(
    __global box_flag_t *tree_flags,
    __global const particle_id_t *box_source_counts_nonchild,
    __global const particle_id_t *box_source_counts_cumul,
    __global const particle_id_t *box_target_counts_nonchild,
    __global const particle_id_t *box_target_counts_cumul)
{
    box_id_t box_idx = get_global_id(0);
    box_flag_t flag = 0;
    if (box_source_counts_nonchild[box_idx])
        flag |= HAS_OWN_SOURCES;
    if (box_source_counts_cumul[box_idx] > box_source_counts_nonchild[box_idx])
        flag |= HAS_CHILD_SOURCES;
    if (box_target_counts_nonchild[box_idx])
        flag |= HAS_OWN_TARGETS;
    if (box_target_counts_cumul[box_idx] > box_target_counts_nonchild[box_idx])
        flag |= HAS_CHILD_TARGETS;
    tree_flags[box_idx] = flag;
}
""", strict_undefined=True)

SOURCES_PARENTS_AND_TARGETS_TEMPLATE = r"""//CL//
void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    box_flag_t flags = box_flags[box_id];

    if (flags & HAS_OWN_SOURCES)
    { APPEND_source_boxes(box_id); }

    if (flags & HAS_CHILD_SOURCES)
    { APPEND_source_parent_boxes(box_id); }

    %if not sources_are_targets:
        if (flags & HAS_OWN_TARGETS)
        { APPEND_target_boxes(box_id); }
    %endif
    if (flags & (HAS_CHILD_TARGETS | HAS_OWN_TARGETS))
    { APPEND_target_or_target_parent_boxes(box_id); }
}
"""


def drive_dfmm(traversal, expansion_wrangler, src_weights, comm=MPI.COMM_WORLD):

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # {{{ Construct local traversal on root

    if current_rank == 0:
        tree = traversal.tree
        ndims = tree.sources.shape[0]
        nboxes = tree.box_source_starts.shape[0]

        # Partition the work across all ranks by allocating responsible boxes
        responsible_boxes = partition_work(tree, total_rank, queue)

        # Compile the program
        mask_dtype = tree.particle_id_dtype
        gen_local_tree_prg = cl.Program(ctx, gen_local_tree_tpl.render(
            tree=tree,
            dtype_to_ctype=dtype_to_ctype,
            mask_dtype=mask_dtype,
            ndims=ndims)).build()

        # Construct mask scan kernel
        mask_scan_knl = ExclusiveScanKernel(
            ctx, mask_dtype,
            scan_expr="a+b", neutral="0",
        )

        def gen_local_particles(rank, particles, nparticles,
                                box_particle_starts,
                                box_particle_counts_nonchild,
                                box_particle_counts_cumul):
            """
            This helper function generates the sources/targets related fields for
            a local tree
            """
            # Put particle structures to device memory
            d_box_particle_starts = cl.array.to_device(queue, box_particle_starts)
            d_box_particle_counts_nonchild = cl.array.to_device(
                queue, box_particle_counts_nonchild)
            d_box_particle_counts_cumul = cl.array.to_device(
                queue, box_particle_counts_cumul)
            d_particles = np.empty((ndims,), dtype=object)
            for i in range(ndims):
                d_particles[i] = cl.array.to_device(queue, particles[i])

            # Generate the particle mask array
            d_particle_mask = cl.array.zeros(queue, (nparticles,), dtype=mask_dtype)
            num_responsible_boxes = responsible_boxes[rank].shape[0]
            gen_local_tree_prg.generate_particle_mask(
                queue, ((num_responsible_boxes + 127)//128,), (128,),
                responsible_boxes[rank].data,
                d_box_particle_starts.data,
                d_box_particle_counts_nonchild.data,
                np.int32(num_responsible_boxes),
                d_particle_mask.data,
                g_times_l=True)

            # Generate the scan of the particle mask array
            d_particle_scan = cl.array.empty(queue, (nparticles,),
                                             dtype=tree.particle_id_dtype)
            mask_scan_knl(d_particle_mask, d_particle_scan)

            # Generate particles for rank's local tree
            local_nparticles = d_particle_scan[-1].get(queue) + 1
            d_local_particles = np.empty((ndims,), dtype=object)
            for i in range(ndims):
                d_local_particles[i] = cl.array.empty(queue, (local_nparticles,),
                                                      dtype=tree.coord_dtype)

            d_paticles_list = d_particles.tolist()
            for i in range(ndims):
                d_paticles_list[i] = d_paticles_list[i].data
            d_local_particles_list = d_local_particles.tolist()
            for i in range(ndims):
                d_local_particles_list[i] = d_local_particles_list[i].data

            gen_local_tree_prg.generate_local_particles(
                queue, ((nparticles + 127) // 128,), (128,),
                np.int32(nparticles),
                *d_paticles_list,
                d_particle_mask.data,
                d_particle_scan.data,
                *d_local_particles_list,
                g_times_l=True)

            # Generate "box_particle_starts" of the local tree
            local_box_particle_starts = cl.array.empty(queue, (nboxes,),
                                                       dtype=tree.particle_id_dtype)
            generate_box_particle_starts = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${particle_id_t} *old_starts,
                    __global ${scan_t} *particle_scan,
                    __global ${particle_id_t} *new_starts
                """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
                            scan_t=dtype_to_ctype(mask_dtype)),
                "new_starts[i] = particle_scan[old_starts[i]]",
                name="generate_box_particle_starts"
            )

            generate_box_particle_starts(d_box_particle_starts, d_particle_scan,
                                         local_box_particle_starts)

            # Generate "box_particle_counts_nonchild" of the local tree
            local_box_particle_counts_nonchild = cl.array.zeros(
                queue, (nboxes,), dtype=tree.particle_id_dtype)

            generate_box_particle_counts_nonchild = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${box_id_t} *res_boxes,
                    __global ${particle_id_t} *old_counts_nonchild,
                    __global ${particle_id_t} *new_counts_nonchild
                """).render(box_id_t=dtype_to_ctype(tree.box_id_dtype),
                            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
                "new_counts_nonchild[res_boxes[i]] = "
                "old_counts_nonchild[res_boxes[i]]",
                name="generate_box_particle_counts_nonchild"
            )

            generate_box_particle_counts_nonchild(responsible_boxes[rank],
                                                  d_box_particle_counts_nonchild,
                                                  local_box_particle_counts_nonchild)

            # Generate "box_particle_counts_cumul"
            local_box_particle_counts_cumul = cl.array.empty(
                queue, (nboxes,), dtype=tree.particle_id_dtype)

            generate_box_particle_counts_cumul = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${particle_id_t} *old_counts_cumul,
                    __global ${particle_id_t} *old_starts,
                    __global ${particle_id_t} *new_counts_cumul,
                    __global ${mask_t} *particle_scan
                """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
                            mask_t=dtype_to_ctype(mask_dtype)),
                "new_counts_cumul[i] = "
                "particle_scan[old_starts[i] + old_counts_cumul[i]] - "
                "particle_scan[old_starts[i]]",
                name="generate_box_particle_counts_cumul"
            )

            generate_box_particle_counts_cumul(d_box_particle_counts_cumul,
                                               d_box_particle_starts,
                                               local_box_particle_counts_cumul,
                                               d_particle_scan)

            local_particles = np.empty((ndims,), dtype=object)
            for i in range(ndims):
                local_particles[i] = d_local_particles[i].get()
            local_box_particle_starts = local_box_particle_starts.get()
            local_box_particle_counts_nonchild = \
                local_box_particle_counts_nonchild.get()
            local_box_particle_counts_cumul = local_box_particle_counts_cumul.get()

            return (local_particles,
                    local_box_particle_starts,
                    local_box_particle_counts_nonchild,
                    local_box_particle_counts_cumul)

        local_trav = np.empty((total_rank,), dtype=object)
        # request object for non-blocking communication
        req = np.empty((total_rank,), dtype=object)

        for rank in range(total_rank):
            local_tree = LocalTree.copy_from_global_tree(
                tree, responsible_boxes[rank].get())

            (local_tree.sources,
             local_tree.box_source_starts,
             local_tree.box_source_counts_nonchild,
             local_tree.box_source_counts_cumul) = \
                gen_local_particles(rank, tree.sources, tree.nsources,
                                    tree.box_source_starts,
                                    tree.box_source_counts_nonchild,
                                    tree.box_source_counts_cumul)

            (local_tree.targets,
             local_tree.box_target_starts,
             local_tree.box_target_counts_nonchild,
             local_tree.box_target_counts_cumul) = \
                gen_local_particles(rank, tree.targets, tree.ntargets,
                                    tree.box_target_starts,
                                    tree.box_target_counts_nonchild,
                                    tree.box_source_counts_cumul)

            local_tree.user_source_ids = None
            local_tree.sorted_target_ids = None
            local_tree.box_flags = None

            local_trav[rank] = traversal.copy()
            local_trav[rank].tree = local_tree
            local_trav[rank].source_boxes = None
            local_trav[rank].target_boxes = None
            local_trav[rank].source_parent_boxes = None
            local_trav[rank].level_start_source_box_nrs = None
            local_trav[rank].level_start_source_parent_box_nrs = None
            local_trav[rank].target_or_target_parent_boxes = None
            local_trav[rank].level_start_target_box_nrs = None
            local_trav[rank].level_start_target_or_target_parent_box_nrs = None
            req[rank] = comm.isend(local_trav[rank], dest=rank)

    # }}}

    # Distribute the local trav to each rank
    if current_rank == 0:
        for rank in range(1, total_rank):
            req[rank].wait()
        local_trav = local_trav[0]
    else:
        local_trav = comm.recv(source=0)
    local_tree = local_trav.tree

    from boxtree.tree import box_flags_enum
    traversal_preamble = traversal_preamble_tpl.render(
        box_flag_t=dtype_to_ctype(box_flags_enum.dtype),
        box_id_t=dtype_to_ctype(local_tree.box_id_dtype),
        particle_id_t=dtype_to_ctype(local_tree.particle_id_dtype),
        box_level_t=dtype_to_ctype(local_tree.box_level_dtype),
        HAS_CHILD_SOURCES=box_flags_enum.HAS_CHILD_SOURCES,
        HAS_CHILD_TARGETS=box_flags_enum.HAS_CHILD_TARGETS,
        HAS_OWN_SOURCES=box_flags_enum.HAS_OWN_SOURCES,
        HAS_OWN_TARGETS=box_flags_enum.HAS_OWN_TARGETS
    )

    # {{{ Fetch local tree to device memory

    d_box_source_counts_nonchild = cl.array.to_device(
        queue, local_tree.box_source_counts_nonchild)
    d_box_source_counts_cumul = cl.array.to_device(
        queue, local_tree.box_source_counts_cumul)
    d_box_target_counts_nonchild = cl.array.to_device(
        queue, local_tree.box_target_counts_nonchild)
    d_box_target_counts_cumul = cl.array.to_device(
        queue, local_tree.box_target_counts_cumul)
    local_tree.box_flags = cl.array.empty(queue, (local_tree.nboxes,),
                                          box_flags_enum.dtype)
    d_level_start_box_nrs = cl.array.to_device(
        queue, local_tree.level_start_box_nrs)
    d_box_levels = cl.array.to_device(
        queue, local_tree.box_levels)

    # }}}

    gen_traversal_src = traversal_preamble + gen_traversal_tpl.render()
    gen_traversal_prg = cl.Program(ctx, gen_traversal_src).build()

    gen_traversal_prg.generate_tree_flags(
        queue, ((local_tree.nboxes + 127) // 128,), (128,),
        local_tree.box_flags.data,
        d_box_source_counts_nonchild.data,
        d_box_source_counts_cumul.data,
        d_box_target_counts_nonchild.data,
        d_box_target_counts_cumul.data,
        g_times_l=True)

    # {{{

    from pyopencl.algorithm import ListOfListsBuilder
    from pyopencl.tools import VectorArg

    sources_parents_and_targets_builder = ListOfListsBuilder(
        ctx,
        [("source_parent_boxes", local_tree.box_id_dtype),
         ("source_boxes", local_tree.box_id_dtype),
         ("target_or_target_parent_boxes", local_tree.box_id_dtype)] + (
            [("target_boxes", local_tree.box_id_dtype)]
            if not local_tree.sources_are_targets else []),
        traversal_preamble + Template(SOURCES_PARENTS_AND_TARGETS_TEMPLATE).render(),
        arg_decls=[VectorArg(box_flags_enum.dtype, "box_flags")],
        name_prefix="sources_parents_and_targets")

    result, evt = sources_parents_and_targets_builder(
        queue, local_tree.nboxes, local_tree.box_flags.data)

    local_trav.source_boxes = result["source_boxes"].lists
    if not tree.sources_are_targets:
        local_trav.target_boxes = result["target_boxes"].lists
    else:
        local_trav.target_boxes = local_trav.source_boxes
    local_trav.source_parent_boxes = result["source_parent_boxes"].lists
    local_trav.target_or_target_parent_boxes = \
        result["target_or_target_parent_boxes"].lists

    # }}}

    # {{{
    level_start_box_nrs_extractor = cl.elementwise.ElementwiseTemplate(
        arguments="""//CL//
        box_id_t *level_start_box_nrs,
        box_level_t *box_levels,
        box_id_t *box_list,
        box_id_t *list_level_start_box_nrs,
        """,

        operation=r"""//CL//
            // Kernel is ranged so that this is true:
            // assert(i > 0);

            box_id_t my_box_id = box_list[i];
            int my_level = box_levels[my_box_id];

            bool is_level_leading_box;
            if (i == 0)
                is_level_leading_box = true;
            else
            {
                box_id_t prev_box_id = box_list[i-1];
                box_id_t my_level_start = level_start_box_nrs[my_level];

                is_level_leading_box = (
                        prev_box_id < my_level_start
                        && my_level_start <= my_box_id);
            }

            if (is_level_leading_box)
                list_level_start_box_nrs[my_level] = i;
        """,
        name="extract_level_start_box_nrs").build(ctx,
            type_aliases=(
                ("box_id_t", local_tree.box_id_dtype),
                ("box_level_t", local_tree.box_level_dtype),
            )
        )

    def extract_level_start_box_nrs(box_list, wait_for):
        result = cl.array.empty(queue, local_tree.nlevels + 1,
                                local_tree.box_id_dtype) \
                                .fill(len(box_list))
        evt = level_start_box_nrs_extractor(
                d_level_start_box_nrs,
                d_box_levels,
                box_list,
                result,
                range=slice(0, len(box_list)),
                queue=queue, wait_for=wait_for)

        result = result.get()

        # Postprocess result for unoccupied levels
        prev_start = len(box_list)
        for ilev in range(tree.nlevels-1, -1, -1):
            result[ilev] = prev_start = min(result[ilev], prev_start)

        return result, evt

    local_trav.level_start_source_box_nrs, _ = \
        extract_level_start_box_nrs(local_trav.source_boxes, wait_for=[])
    local_trav.level_start_source_parent_box_nrs, _ = \
        extract_level_start_box_nrs(local_trav.source_parent_boxes, wait_for=[])
    local_trav.level_start_target_box_nrs, _ = \
        extract_level_start_box_nrs(local_trav.target_boxes, wait_for=[])
    local_trav.level_start_target_or_target_parent_box_nrs, _ = \
        extract_level_start_box_nrs(local_trav.target_or_target_parent_boxes,
                                    wait_for=[])

    # }}}
