from __future__ import division
from mpi4py import MPI
import numpy as np
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import GenericScanKernel
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


class AllReduceCommPattern(object):
    """Describes a butterfly communication pattern for allreduce. Supports efficient
    allreduce between an arbitrary number of processes.
    """

    def __init__(self, rank, nprocs):
        """
        :arg rank: My rank
        :arg nprocs: Total number of processors
        """
        assert nprocs > 0
        self.rank = rank
        self.left = 0
        self.right = nprocs
        self.midpoint = nprocs // 2

    def sources(self):
        """Return the set of source nodes at this communication stage. The current
        process receives messages from these nodes.
        """
        if self.rank < self.midpoint:
            partner = self.midpoint + (self.rank - self.left)
            if self.rank == self.midpoint - 1 and partner == self.right:
                partners = set()
            elif self.rank == self.midpoint - 1 and partner == self.right - 2:
                partners = set([partner, partner + 1])
            else:
                partners = set([partner])
        else:
            partner = self.left + (self.rank - self.midpoint)
            if self.rank == self.right - 1 and partner == self.midpoint:
                partners = set()
            elif self.rank == self.right - 1 and partner == self.midpoint - 2:
                partners = set([partner, partner + 1])
            else:
                partners = set([partner])

        return partners

    def sinks(self):
        """Return the set of sink nodes at this communication stage. The current process
        sends a message to these nodes.
        """
        if self.rank < self.midpoint:
            partner = self.midpoint + (self.rank - self.left)
            if partner == self.right:
                partner -= 1
        else:
            partner = self.left + (self.rank - self.midpoint)
            if partner == self.midpoint:
                partner -= 1

        return set([partner])

    def messages(self):
        """Return the set of relevant messages to send to the sinks.
        """
        if self.rank < self.midpoint:
            return set(range(self.midpoint, self.right))
        else:
            return set(range(self.left, self.midpoint))

    def advance(self):
        """Advance to the next stage in the communication pattern.
        """
        if self.rank < self.midpoint:
            self.right = self.midpoint
            self.midpoint = (self.midpoint + self.left) // 2
        else:
            self.left = self.midpoint
            self.midpoint = (self.midpoint + self.right) // 2

    def done(self):
        """Return whether this node is finished communicating.
        """
        return self.left + 1 == self.right


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

        if self.sources_have_extent:
            d_tree.source_radii = cl.array.to_device(queue, d_tree.source_radii)
        if self.targets_have_extent:
            d_tree.target_radii = cl.array.to_device(queue, d_tree.target_radii)

        return d_tree


class MPITags():
    DIST_TREE = 0
    DIST_WEIGHT = 1
    GATHER_POTENTIALS = 2


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


def gen_local_particles(queue, particles, nparticles, tree,
                        responsible_boxes,
                        box_particle_starts,
                        box_particle_counts_nonchild,
                        box_particle_counts_cumul,
                        particle_radii=None,
                        particle_weights=None,
                        return_mask_scan=False):
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
    particle_mask_knl(responsible_boxes, d_box_particle_starts,
                      d_box_particle_counts_nonchild, d_particle_mask)

    # Generate the scan of the particle mask array
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
        d_paticles_list[i] = d_paticles_list[i]
    d_local_particles_list = d_local_particles.tolist()
    for i in range(tree.dimensions):
        d_local_particles_list[i] = d_local_particles_list[i]

    fetch_local_particles_knl = cl.elementwise.ElementwiseKernel(
        ctx,
        Template("""
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
        """, strict_undefined=True).render(
            mask_t=dtype_to_ctype(tree.particle_id_dtype),
            coord_t=dtype_to_ctype(tree.coord_dtype),
            ndims=tree.dimensions,
            particles_have_extent=(particle_radii is not None)
        ),
        Template("""
            if(particle_mask[i]) {
                ${particle_id_t} des = particle_scan[i];
                % for dim in range(ndims):
                    local_particles_${dim}[des] = particles_${dim}[i];
                % endfor
                % if particles_have_extent:
                    local_particle_radii[des] = particle_radii[i];
                % endif
            }
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
            ndims=tree.dimensions,
            particles_have_extent=(particle_radii is not None)
        )
    )

    if particle_radii is None:
        fetch_local_particles_knl(d_particle_mask, d_particle_scan,
                                  *d_paticles_list, *d_local_particles_list)
    else:
        d_particle_radii = cl.array.to_device(queue, particle_radii)
        d_local_particle_radii = cl.array.empty(queue, (local_nparticles,),
                                                dtype=tree.coord_dtype)
        fetch_local_particles_knl(d_particle_mask, d_particle_scan,
                                  *d_paticles_list, *d_local_particles_list,
                                  d_particle_radii, d_local_particle_radii)

    # Generate "box_particle_starts" of the local tree
    local_box_particle_starts = cl.array.empty(queue, (tree.nboxes,),
                                               dtype=tree.particle_id_dtype)

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
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
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
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)
        ),
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
            """, strict_undefined=True).render(
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

    rtv = (local_particles,
           local_box_particle_starts,
           local_box_particle_counts_nonchild,
           local_box_particle_counts_cumul)

    if particle_radii is not None:
        rtv = rtv + (d_local_particle_radii.get(),)

    if particle_weights is not None:
        rtv = rtv + (local_particle_weights.get(),)

    if return_mask_scan:
        rtv = rtv + (d_particle_mask, d_particle_scan, local_nparticles)

    return rtv


def generate_local_tree(traversal, src_weights, comm=MPI.COMM_WORLD):
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
            partition_work(tree, total_rank, queue)

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

        # }}}

        # Convert src_weights to tree order
        src_weights = src_weights[tree.user_source_ids]
        src_weights = cl.array.to_device(queue, src_weights)
        local_src_weights = np.empty((total_rank,), dtype=object)

        # request objects for non-blocking communication
        tree_req = np.empty((total_rank,), dtype=object)
        weight_req = np.empty((total_rank,), dtype=object)

        if tree.sources_have_extent:
            source_radii = tree.source_radii
        else:
            source_radii = None

        if tree.targets_have_extent:
            target_radii = tree.target_radii
        else:
            target_radii = None

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
                                    src_boxes_mask[rank],
                                    tree.box_source_starts,
                                    tree.box_source_counts_nonchild,
                                    tree.box_source_counts_cumul,
                                    source_radii, src_weights)

            (local_tree[rank].targets,
             local_tree[rank].box_target_starts,
             local_tree[rank].box_target_counts_nonchild,
             local_tree[rank].box_target_counts_cumul,
             local_tree[rank].target_radii,
             local_target_mask[rank],
             local_target_scan[rank],
             local_ntargets[rank]) = \
                gen_local_particles(queue, tree.targets, tree.ntargets, tree,
                                    responsible_boxes_mask[rank],
                                    tree.box_target_starts,
                                    tree.box_target_counts_nonchild,
                                    tree.box_target_counts_cumul,
                                    target_radii, None, return_mask_scan=True)

            local_tree[rank].user_source_ids = None
            local_tree[rank].sorted_target_ids = None

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

    return local_tree, local_src_weights, local_target


def generate_local_travs(local_tree, local_src_weights, comm=MPI.COMM_WORLD):
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

    return trav_local, trav_global


def drive_dfmm(wrangler, trav_local, trav_global, local_src_weights, global_wrangler,
               local_target_mask, local_target_scan, local_ntargets,
               comm=MPI.COMM_WORLD):
    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

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

    mpole_exps_all = np.zeros_like(mpole_exps)
    comm.Allreduce(mpole_exps, mpole_exps_all)

    mpole_exps = mpole_exps_all

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    potentials = wrangler.eval_direct(
            trav_global.target_boxes,
            trav_global.neighbor_source_boxes_starts,
            trav_global.neighbor_source_boxes_lists,
            local_src_weights)

    # these potentials are called alpha in [1]

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
