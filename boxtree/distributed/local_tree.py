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

import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from boxtree import Tree
from mpi4py import MPI
import time
import numpy as np
from pytools import memoize

import logging
logger = logging.getLogger(__name__)


@memoize
def particle_mask_kernel(context, particle_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        arguments=Template("""
            __global char *responsible_boxes,
            __global ${particle_id_t} *box_particle_starts,
            __global ${particle_id_t} *box_particle_counts_nonchild,
            __global ${particle_id_t} *particle_mask
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype)
        ),
        operation=Template("""
            if(responsible_boxes[i]) {
                for(${particle_id_t} pid = box_particle_starts[i];
                    pid < box_particle_starts[i] + box_particle_counts_nonchild[i];
                    ++pid) {
                    particle_mask[pid] = 1;
                }
            }
        """).render(particle_id_t=dtype_to_ctype(particle_id_dtype))
    )


@memoize
def mask_scan_kernel(context, particle_id_dtype):
    from pyopencl.scan import GenericScanKernel
    return GenericScanKernel(
        context, particle_id_dtype,
        arguments=Template("""
            __global ${mask_t} *ary,
            __global ${mask_t} *scan
            """, strict_undefined=True).render(
            mask_t=dtype_to_ctype(particle_id_dtype)
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


@memoize
def fetch_local_sources_kernel(
        context, particle_id_dtype, coord_dtype, dimensions, sources_have_extent):
    return cl.elementwise.ElementwiseKernel(
        context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(particle_id_dtype),
            coord_t=dtype_to_ctype(coord_dtype),
            ndims=dimensions,
            particles_have_extent=sources_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(particle_id_dtype),
            ndims=dimensions,
            particles_have_extent=sources_have_extent
        )
    )


@memoize
def fetch_local_targets_kernel(
        context, particle_id_dtype, coord_dtype, dimensions, targets_have_extent):
    return cl.elementwise.ElementwiseKernel(
        context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(particle_id_dtype),
            coord_t=dtype_to_ctype(coord_dtype),
            ndims=dimensions,
            particles_have_extent=targets_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(particle_id_dtype),
            ndims=dimensions,
            particles_have_extent=targets_have_extent
        )
    )


@memoize
def generate_box_particle_starts_kernel(context, particle_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *particle_scan,
            __global ${particle_id_t} *new_starts
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype)
        ),
        "new_starts[i] = particle_scan[old_starts[i]]",
        name="generate_box_particle_starts"
    )


@memoize
def generate_box_particle_counts_nonchild_kernel(context, particle_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global char *res_boxes,
            __global ${particle_id_t} *old_counts_nonchild,
            __global ${particle_id_t} *new_counts_nonchild
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype)
        ),
        "if(res_boxes[i]) new_counts_nonchild[i] = old_counts_nonchild[i];"
    )


@memoize
def generate_box_particle_counts_cumul_kernel(context, particle_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global ${particle_id_t} *old_counts_cumul,
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *new_counts_cumul,
            __global ${particle_id_t} *particle_scan
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype)
        ),
        """
        new_counts_cumul[i] =
            particle_scan[old_starts[i] + old_counts_cumul[i]] -
            particle_scan[old_starts[i]]
        """
    )


def fetch_local_particles(
        queue, global_tree, src_box_mask, tgt_box_mask, local_tree):
    """This helper function generates particles of the local tree, and reconstruct
    list of lists indexing accordingly.

    Specifically, this function generates the following fields for the local tree:
    sources, targets, target_radii, box_source_starts, box_source_counts_nonchild,
    box_source_counts_cumul, box_target_starts, box_target_counts_nonchild,
    box_target_counts_cumul.

    These generated fields are stored directly into *local_tree*.
    """
    global_tree_dev = global_tree.to_device(queue).with_queue(queue)
    nsources = global_tree.nsources

    # {{{ source particle mask

    src_particle_mask = cl.array.zeros(
        queue, (nsources,),
        dtype=global_tree.particle_id_dtype
    )

    particle_mask_kernel(queue.context, global_tree.particle_id_dtype)(
        src_box_mask,
        global_tree_dev.box_source_starts,
        global_tree_dev.box_source_counts_nonchild,
        src_particle_mask
    )

    # }}}

    # {{{ scan of source particle mask

    src_particle_scan = cl.array.empty(
        queue, (nsources + 1,),
        dtype=global_tree.particle_id_dtype
    )

    src_particle_scan[0] = 0
    mask_scan_kernel(queue.context, global_tree.particle_id_dtype)(
        src_particle_mask, src_particle_scan
    )

    # }}}

    # {{{ local sources

    local_nsources = src_particle_scan[-1].get(queue)

    local_sources = cl.array.empty(
        queue, (global_tree.dimensions, local_nsources),
        dtype=global_tree.coord_dtype
    )

    local_sources_list = [
        local_sources[idim, :]
        for idim in range(global_tree.dimensions)
    ]

    assert global_tree.sources_have_extent is False

    fetch_local_sources_kernel(
        queue.context,
        global_tree.particle_id_dtype,
        global_tree.coord_dtype,
        global_tree.dimensions,
        global_tree.sources_have_extent
    )(
        src_particle_mask, src_particle_scan,
        *global_tree_dev.sources.tolist(),
        *local_sources_list
    )

    # }}}

    # {{{ box_source_starts

    local_box_source_starts = cl.array.empty(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    generate_box_particle_starts_kernel(
        queue.context, global_tree.particle_id_dtype)(
            global_tree_dev.box_source_starts,
            src_particle_scan,
            local_box_source_starts
        )

    # }}}

    # {{{ box_source_counts_nonchild

    local_box_source_counts_nonchild = cl.array.zeros(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    generate_box_particle_counts_nonchild_kernel(
        queue.context, global_tree.particle_id_dtype)(
            src_box_mask,
            global_tree_dev.box_source_counts_nonchild,
            local_box_source_counts_nonchild
        )

    # }}}

    # {{{ box_source_counts_cumul

    local_box_source_counts_cumul = cl.array.empty(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    generate_box_particle_counts_cumul_kernel(
        queue.context, global_tree.particle_id_dtype)(
            global_tree_dev.box_source_counts_cumul,
            global_tree_dev.box_source_starts,
            local_box_source_counts_cumul,
            src_particle_scan
        )

    # }}}

    # {{{ target particle mask

    ntargets = global_tree.ntargets

    tgt_particle_mask = cl.array.zeros(
        queue, (ntargets,),
        dtype=global_tree.particle_id_dtype
    )

    particle_mask_kernel(queue.context, global_tree.particle_id_dtype)(
        tgt_box_mask,
        global_tree_dev.box_target_starts,
        global_tree_dev.box_target_counts_nonchild,
        tgt_particle_mask
    )

    # }}}

    # {{{ scan of target particle mask

    tgt_particle_scan = cl.array.empty(
        queue, (ntargets + 1,),
        dtype=global_tree.particle_id_dtype
    )

    tgt_particle_scan[0] = 0
    mask_scan_kernel(queue.context, global_tree.particle_id_dtype)(
        tgt_particle_mask, tgt_particle_scan
    )

    # }}}

    # {{{ local targets

    local_ntargets = tgt_particle_scan[-1].get(queue)

    local_targets = cl.array.empty(
        queue, (local_tree.dimensions, local_ntargets),
        dtype=local_tree.coord_dtype
    )

    local_targets_list = [
        local_targets[idim, :]
        for idim in range(local_tree.dimensions)
    ]

    if local_tree.targets_have_extent:
        local_target_radii = cl.array.empty(
            queue, (local_ntargets,),
            dtype=global_tree.coord_dtype
        )

        fetch_local_targets_kernel(
            queue.context,
            global_tree.particle_id_dtype,
            global_tree.coord_dtype,
            global_tree.dimensions,
            True
        )(
            tgt_particle_mask, tgt_particle_scan,
            *global_tree_dev.targets.tolist(),
            *local_targets_list,
            global_tree_dev.target_radii,
            local_target_radii
        )
    else:
        fetch_local_targets_kernel(
            queue.context,
            global_tree.particle_id_dtype,
            global_tree.coord_dtype,
            global_tree.dimensions,
            False
        )(
            tgt_particle_mask, tgt_particle_scan,
            *global_tree_dev.targets.tolist(),
            *local_targets_list
        )

    # {{{ box_target_starts

    local_box_target_starts = cl.array.empty(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    generate_box_particle_starts_kernel(
        queue.context, global_tree.particle_id_dtype)(
            global_tree_dev.box_target_starts,
            tgt_particle_scan,
            local_box_target_starts
        )

    # }}}

    # {{{ box_target_counts_nonchild

    local_box_target_counts_nonchild = cl.array.zeros(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype)

    generate_box_particle_counts_nonchild_kernel(
        queue.context, global_tree.particle_id_dtype)(
            tgt_box_mask,
            global_tree_dev.box_target_counts_nonchild,
            local_box_target_counts_nonchild
    )

    # }}}

    # {{{ box_target_counts_cumul

    local_box_target_counts_cumul = cl.array.empty(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    generate_box_particle_counts_cumul_kernel(
        queue.context, global_tree.particle_id_dtype)(
            global_tree_dev.box_target_counts_cumul,
            global_tree_dev.box_target_starts,
            local_box_target_counts_cumul,
            tgt_particle_scan
        )

    # }}}

    # {{{ Fetch fields to local_tree

    local_sources = local_sources.get(queue=queue)
    local_tree.sources = local_sources

    local_targets = local_targets.get(queue=queue)
    local_tree.targets = local_targets

    if global_tree.targets_have_extent:
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

    # }}}

    # {{{ src_idx and tgt_idx

    src_particle_mask = src_particle_mask.get(queue=queue).astype(bool)
    src_idx = np.arange(nsources)[src_particle_mask]

    tgt_particle_mask = tgt_particle_mask.get(queue=queue).astype(bool)
    tgt_idx = np.arange(ntargets)[tgt_particle_mask]

    # }}}

    return local_tree, src_idx, tgt_idx


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
        return self._nsources

    @property
    def ntargets(self):
        return self._ntargets

    @property
    def dimensions(self):
        return self._dimensions


def generate_local_tree(queue, global_traversal, responsible_boxes_list,
                        comm=MPI.COMM_WORLD):
    global_tree = global_traversal.tree

    # Get MPI information
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    start_time = time.time()

    from boxtree.distributed.partition import get_boxes_mask
    (responsible_boxes_mask, ancestor_boxes, src_boxes_mask, box_mpole_is_used) = \
        get_boxes_mask(queue, global_traversal, responsible_boxes_list[mpi_rank])

    local_tree = global_tree.copy(
        responsible_boxes_list=responsible_boxes_list[mpi_rank],
        ancestor_mask=ancestor_boxes.get(),
        box_to_user_starts=None,
        box_to_user_lists=None,
        _dimensions=None,
        _ntargets=None,
        _nsources=None,
    )

    local_tree.user_source_ids = None
    local_tree.sorted_target_ids = None

    local_tree, src_idx, tgt_idx = fetch_local_particles(
        queue,
        global_tree,
        src_boxes_mask,
        responsible_boxes_mask,
        local_tree,
    )

    local_tree._dimensions = local_tree.dimensions
    local_tree._ntargets = local_tree.targets[0].shape[0]
    local_tree._nsources = local_tree.sources[0].shape[0]

    local_tree.__class__ = LocalTree

    # {{{ compute the users of multipole expansions of each box on root rank

    box_mpole_is_used_all_ranks = None
    if mpi_rank == 0:
        box_mpole_is_used_all_ranks = np.empty(
            (mpi_size, global_tree.nboxes), dtype=box_mpole_is_used.dtype
        )
    comm.Gather(box_mpole_is_used.get(), box_mpole_is_used_all_ranks, root=0)

    box_to_user_starts = None
    box_to_user_lists = None

    if mpi_rank == 0:
        box_mpole_is_used_all_ranks = cl.array.to_device(
            queue, box_mpole_is_used_all_ranks
        )

        from boxtree.tools import MaskCompressorKernel
        matcompr = MaskCompressorKernel(queue.context)
        (box_to_user_starts, box_to_user_lists, evt) = \
            matcompr(queue, box_mpole_is_used_all_ranks.transpose(),
                     list_dtype=np.int32)

        cl.wait_for_events([evt])
        del box_mpole_is_used

        box_to_user_starts = box_to_user_starts.get()
        box_to_user_lists = box_to_user_lists.get()

        logger.debug("computing box_to_user: done")

    box_to_user_starts = comm.bcast(box_to_user_starts, root=0)
    box_to_user_lists = comm.bcast(box_to_user_lists, root=0)

    local_tree.box_to_user_starts = box_to_user_starts
    local_tree.box_to_user_lists = box_to_user_lists

    # }}}

    logger.info("Generate local tree on rank {} in {} sec.".format(
        mpi_rank, str(time.time() - start_time)
    ))

    return local_tree, src_idx, tgt_idx
