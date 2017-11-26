from __future__ import division
from mpi4py import MPI
import numpy as np
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import GenericScanKernel
from boxtree import Tree
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

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
    def copy_from_global_tree(cls, global_tree, responsible_boxes_list,
                              ancestor_mask):
        local_tree = global_tree.copy(
            responsible_boxes_list=responsible_boxes_list,
            ancestor_mask=ancestor_mask)
        local_tree.__class__ = cls
        return local_tree

    def to_device(self, queue):
        field_to_device = [
            "box_centers", "box_child_ids", "box_flags", "box_levels",
            "box_parent_ids", "box_source_counts_cumul",
            "box_source_counts_nonchild", "box_source_starts",
            "box_target_counts_cumul", "box_target_counts_nonchild",
            "box_target_starts", "level_start_box_nrs_dev", "sources", "targets",
            "responsible_boxes_list", "ancestor_mask"
        ]
        d_tree = self.copy()
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
        return d_tree


class MPI_Tags():
    DIST_TREE = 0
    DIST_WEIGHT = 1


def partition_work(tree, total_rank, queue):
    """ This function returns a pyopencl array of size total_rank*nboxes, where
    the (i,j) entry is 1 iff rank i is responsible for box j.
    """
    responsible_boxes_mask = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                            dtype=np.int8)
    responsible_boxes_list = np.empty((total_rank,), dtype=object)
    nboxes_per_rank = tree.nboxes // total_rank
    extra_boxes = tree.nboxes - nboxes_per_rank * total_rank
    start_idx = 0

    for current_rank in range(extra_boxes):
        end_idx = start_idx + nboxes_per_rank + 1
        responsible_boxes_mask[current_rank, start_idx:end_idx] = 1
        responsible_boxes_list[current_rank] = cl.array.arange(
            queue, start_idx, end_idx, dtype=tree.box_id_dtype)
        start_idx = end_idx

    for current_rank in range(extra_boxes, total_rank):
        end_idx = start_idx + nboxes_per_rank
        responsible_boxes_mask[current_rank, start_idx:end_idx] = 1
        responsible_boxes_list[current_rank] = cl.array.arange(
            queue, start_idx, end_idx, dtype=tree.box_id_dtype)
        start_idx = end_idx

    return responsible_boxes_mask, responsible_boxes_list


gen_local_tree_tpl = Template(r"""
typedef ${dtype_to_ctype(tree.box_id_dtype)} box_id_t;
typedef ${dtype_to_ctype(tree.particle_id_dtype)} particle_id_t;
typedef ${dtype_to_ctype(mask_dtype)} mask_t;
typedef ${dtype_to_ctype(tree.coord_dtype)} coord_t;

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


def gen_local_particles(queue, particles, nparticles, tree,
                        responsible_boxes,
                        box_particle_starts,
                        box_particle_counts_nonchild,
                        box_particle_counts_cumul,
                        particle_weights=None):
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
    d_particles = np.empty((tree.dimensions,), dtype=object)
    for i in range(tree.dimensions):
        d_particles[i] = cl.array.to_device(queue, particles[i])

    # Generate the particle mask array
    d_particle_mask = cl.array.zeros(queue, (nparticles,),
                                     dtype=tree.particle_id_dtype)
    particle_mask_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        arguments=Template("""
            __global char *responsible_boxes,
            __global ${particle_id_t} *box_particle_starts,
            __global ${particle_id_t} *box_particle_counts_nonchild,
            __global ${particle_id_t} *particle_mask
        """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
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
    particle_mask_knl(responsible_boxes, d_box_particle_starts,
                      d_box_particle_counts_nonchild, d_particle_mask)

    # Generate the scan of the particle mask array
    mask_scan_knl = GenericScanKernel(
        queue.context, tree.particle_id_dtype,
        arguments=Template("""
            __global ${mask_t} *ary,
            __global ${mask_t} *scan
            """).render(mask_t=dtype_to_ctype(tree.particle_id_dtype)),
        input_expr="ary[i]",
        scan_expr="a+b", neutral="0",
        output_statement="scan[i + 1] = item;"
    )
    d_particle_scan = cl.array.empty(queue, (nparticles + 1,),
                                     dtype=tree.particle_id_dtype)
    d_particle_scan[0] = 0
    mask_scan_knl(d_particle_mask, d_particle_scan)

    # Generate particles for rank's local tree
    local_nparticles = d_particle_scan[-1].get(queue)
    d_local_particles = np.empty((tree.dimensions,), dtype=object)
    for i in range(tree.dimensions):
        d_local_particles[i] = cl.array.empty(queue, (local_nparticles,),
                                              dtype=tree.coord_dtype)

    d_paticles_list = d_particles.tolist()
    for i in range(tree.dimensions):
        d_paticles_list[i] = d_paticles_list[i].data
    d_local_particles_list = d_local_particles.tolist()
    for i in range(tree.dimensions):
        d_local_particles_list[i] = d_local_particles_list[i].data

    gen_local_tree_prg = cl.Program(
        queue.context,
        gen_local_tree_tpl.render(
            tree=tree,
            dtype_to_ctype=dtype_to_ctype,
            mask_dtype=tree.particle_id_dtype,
            ndims=tree.dimensions
        )
    ).build()

    gen_local_tree_prg.generate_local_particles(
        queue, ((nparticles + 127) // 128,), (128,),
        np.int32(nparticles),
        *d_paticles_list,
        d_particle_mask.data,
        d_particle_scan.data,
        *d_local_particles_list,
        g_times_l=True)

    # Generate "box_particle_starts" of the local tree
    local_box_particle_starts = cl.array.empty(queue, (tree.nboxes,),
                                               dtype=tree.particle_id_dtype)

    generate_box_particle_starts = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *particle_scan,
            __global ${particle_id_t} *new_starts
        """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
        "new_starts[i] = particle_scan[old_starts[i]]",
        name="generate_box_particle_starts"
    )

    generate_box_particle_starts(d_box_particle_starts, d_particle_scan,
                                 local_box_particle_starts)

    # Generate "box_particle_counts_nonchild" of the local tree
    local_box_particle_counts_nonchild = cl.array.zeros(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)

    generate_box_particle_counts_nonchild = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global char *res_boxes,
            __global ${particle_id_t} *old_counts_nonchild,
            __global ${particle_id_t} *new_counts_nonchild
        """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
        "if(res_boxes[i]) new_counts_nonchild[i] = old_counts_nonchild[i];"
    )

    generate_box_particle_counts_nonchild(responsible_boxes,
                                          d_box_particle_counts_nonchild,
                                          local_box_particle_counts_nonchild)

    # Generate "box_particle_counts_cumul"
    local_box_particle_counts_cumul = cl.array.empty(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)

    generate_box_particle_counts_cumul = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *old_counts_cumul,
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *new_counts_cumul,
            __global ${particle_id_t} *particle_scan
        """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
        """
        new_counts_cumul[i] =
            particle_scan[old_starts[i] + old_counts_cumul[i]] -
            particle_scan[old_starts[i]]
        """
    )

    generate_box_particle_counts_cumul(d_box_particle_counts_cumul,
                                       d_box_particle_starts,
                                       local_box_particle_counts_cumul,
                                       d_particle_scan)

    local_particles = np.empty((tree.dimensions,), dtype=object)
    for i in range(tree.dimensions):
        local_particles[i] = d_local_particles[i].get()
    local_box_particle_starts = local_box_particle_starts.get()
    local_box_particle_counts_nonchild = local_box_particle_counts_nonchild.get()
    local_box_particle_counts_cumul = local_box_particle_counts_cumul.get()

    # {{{ Generate source weights
    if particle_weights is not None:
        local_particle_weights = cl.array.empty(queue, (local_nparticles,),
            dtype=particle_weights.dtype)
        gen_local_source_weights_knl = cl.elementwise.ElementwiseKernel(
            queue.context,
            arguments=Template("""
                __global ${weight_t} *src_weights,
                __global ${particle_id_t} *particle_mask,
                __global ${particle_id_t} *particle_scan,
                __global ${weight_t} *local_weights
            """).render(
                weight_t=dtype_to_ctype(particle_weights.dtype),
                particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
            ),
            operation="""
                if(particle_mask[i]) {
                    local_weights[particle_scan[i]] = src_weights[i];
                }
            """
        )
        gen_local_source_weights_knl(particle_weights, d_particle_mask,
                                     d_particle_scan, local_particle_weights)

    # }}}

    if particle_weights is not None:
        return (local_particles,
                local_box_particle_starts,
                local_box_particle_counts_nonchild,
                local_box_particle_counts_cumul,
                local_particle_weights.get())
    else:
        return (local_particles,
                local_box_particle_starts,
                local_box_particle_counts_nonchild,
                local_box_particle_counts_cumul)


def drive_dfmm(traversal, src_weights, comm=MPI.COMM_WORLD):

    # Get MPI and pyopencl information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # {{{ Construct local tree for each rank on root

    if current_rank == 0:
        local_tree = np.empty((total_rank,), dtype=object)
        tree = traversal.tree

        d_box_parent_ids = cl.array.to_device(queue, tree.box_parent_ids)

        # {{{ Partition the work

        # Each rank is responsible for calculating the multiple expansion as well as
        # evaluating target potentials in *responsible_boxes*
        responsible_boxes_mask, responsible_boxes_list = \
            partition_work(tree, total_rank, queue)

        # In order to evaluate, each rank needs sources in boxes in
        # *src_boxes*
        src_boxes = responsible_boxes_mask.copy()

        # Add list 1 and list 4 of responsible boxes to src_boxes

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

        # }}}

        # Convert src_weights to tree order
        src_weights = src_weights[tree.user_source_ids]
        src_weights = cl.array.to_device(queue, src_weights)
        local_src_weights = np.empty((total_rank,), dtype=object)

        # request objects for non-blocking communication
        tree_req = np.empty((total_rank,), dtype=object)
        weight_req = np.empty((total_rank,), dtype=object)

        for rank in range(total_rank):
            local_tree[rank] = LocalTree.copy_from_global_tree(
                tree, responsible_boxes_list[rank].get(),
                ancestor_boxes[rank].get())

            (local_tree[rank].sources,
             local_tree[rank].box_source_starts,
             local_tree[rank].box_source_counts_nonchild,
             local_tree[rank].box_source_counts_cumul,
             local_src_weights[rank]) = \
                gen_local_particles(queue, tree.sources, tree.nsources, tree,
                                    src_boxes[rank],
                                    tree.box_source_starts,
                                    tree.box_source_counts_nonchild,
                                    tree.box_source_counts_cumul,
                                    src_weights)

            (local_tree[rank].targets,
             local_tree[rank].box_target_starts,
             local_tree[rank].box_target_counts_nonchild,
             local_tree[rank].box_target_counts_cumul) = \
                gen_local_particles(queue, tree.targets, tree.ntargets, tree,
                                    responsible_boxes_mask[rank],
                                    tree.box_target_starts,
                                    tree.box_target_counts_nonchild,
                                    tree.box_target_counts_cumul,
                                    None)

            local_tree[rank].source_radii = None
            local_tree[rank].target_radii = None
            local_tree[rank].user_source_ids = None
            local_tree[rank].sorted_target_ids = None

            tree_req[rank] = comm.isend(local_tree[rank], dest=rank,
                                        tag=MPI_Tags.DIST_TREE)
            weight_req[rank] = comm.isend(local_src_weights[rank], dest=rank,
                                          tag=MPI_Tags.DIST_WEIGHT)

    # }}}

    # Recieve the local trav from root
    if current_rank == 0:
        for rank in range(1, total_rank):
            tree_req[rank].wait()
        local_tree = local_tree[0]
    else:
        local_tree = comm.recv(source=0, tag=MPI_Tags.DIST_TREE)

    # Recieve source weights from root
    if current_rank == 0:
        for rank in range(1, total_rank):
            weight_req[rank].wait()
        local_src_weights = local_src_weights[0]
    else:
        local_src_weights = comm.recv(source=0, tag=MPI_Tags.DIST_WEIGHT)

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
    d_trav_global, _ = tg(queue, d_tree, debug=True)
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

    d_trav_local, _ = tg(queue, d_tree, debug=True)
    trav_local = d_trav_local.get(queue=queue)

    def fmm_level_to_nterms(tree, level):
        return 3
    wrangler = FMMLibExpansionWrangler(
        local_tree, 0, fmm_level_to_nterms=fmm_level_to_nterms)

    # {{{ "Step 2.1:" Construct local multipoles

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

    # {{{ Communicate mpole

    mpole_exps_all = np.empty_like(mpole_exps)
    comm.Allreduce(mpole_exps, mpole_exps_all)

    # }}}
