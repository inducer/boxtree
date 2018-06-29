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

from collections import namedtuple
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from boxtree import Tree
from mpi4py import MPI
import time
import numpy as np
from boxtree.distributed import MPITags

import logging
logger = logging.getLogger(__name__)

FetchLocalParticlesKernels = namedtuple(
    'FetchLocalParticlesKernels',
    [
        'particle_mask_knl',
        'mask_scan_knl',
        'fetch_local_src_knl',
        'fetch_local_tgt_knl',
        'generate_box_particle_starts',
        'generate_box_particle_counts_nonchild',
        'generate_box_particle_counts_cumul'
    ]
)


def get_fetch_local_particles_knls(context, global_tree):
    """
    This function compiles several PyOpenCL kernels helpful for fetching particles of
    local trees from global tree.

    :param context: The context to compile against.
    :param global_tree: The global tree from which local trees are generated.
    :return: A FetchLocalParticlesKernels object.
    """

    particle_mask_knl = cl.elementwise.ElementwiseKernel(
        context,
        arguments=Template("""
            __global char *responsible_boxes,
            __global ${particle_id_t} *box_particle_starts,
            __global ${particle_id_t} *box_particle_counts_nonchild,
            __global ${particle_id_t} *particle_mask
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype)
        ),
        operation=Template("""
            if(responsible_boxes[i]) {
                for(${particle_id_t} pid = box_particle_starts[i];
                    pid < box_particle_starts[i] + box_particle_counts_nonchild[i];
                    ++pid) {
                    particle_mask[pid] = 1;
                }
            }
        """).render(particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype))
    )

    from pyopencl.scan import GenericScanKernel
    mask_scan_knl = GenericScanKernel(
        context, global_tree.particle_id_dtype,
        arguments=Template("""
            __global ${mask_t} *ary,
            __global ${mask_t} *scan
            """, strict_undefined=True).render(
            mask_t=dtype_to_ctype(global_tree.particle_id_dtype)
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
        context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(global_tree.particle_id_dtype),
            coord_t=dtype_to_ctype(global_tree.coord_dtype),
            ndims=global_tree.dimensions,
            particles_have_extent=global_tree.sources_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype),
            ndims=global_tree.dimensions,
            particles_have_extent=global_tree.sources_have_extent
        )
    )

    fetch_local_tgt_knl = cl.elementwise.ElementwiseKernel(
        context,
        fetch_local_paticles_arguments.render(
            mask_t=dtype_to_ctype(global_tree.particle_id_dtype),
            coord_t=dtype_to_ctype(global_tree.coord_dtype),
            ndims=global_tree.dimensions,
            particles_have_extent=global_tree.targets_have_extent
        ),
        fetch_local_particles_prg.render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype),
            ndims=global_tree.dimensions,
            particles_have_extent=global_tree.targets_have_extent
        )
    )

    generate_box_particle_starts = cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *particle_scan,
            __global ${particle_id_t} *new_starts
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype)
        ),
        "new_starts[i] = particle_scan[old_starts[i]]",
        name="generate_box_particle_starts"
    )

    generate_box_particle_counts_nonchild = cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global char *res_boxes,
            __global ${particle_id_t} *old_counts_nonchild,
            __global ${particle_id_t} *new_counts_nonchild
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype)
        ),
        "if(res_boxes[i]) new_counts_nonchild[i] = old_counts_nonchild[i];"
    )

    generate_box_particle_counts_cumul = cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global ${particle_id_t} *old_counts_cumul,
            __global ${particle_id_t} *old_starts,
            __global ${particle_id_t} *new_counts_cumul,
            __global ${particle_id_t} *particle_scan
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(global_tree.particle_id_dtype)
        ),
        """
        new_counts_cumul[i] =
            particle_scan[old_starts[i] + old_counts_cumul[i]] -
            particle_scan[old_starts[i]]
        """
    )

    return FetchLocalParticlesKernels(
        particle_mask_knl=particle_mask_knl,
        mask_scan_knl=mask_scan_knl,
        fetch_local_src_knl=fetch_local_src_knl,
        fetch_local_tgt_knl=fetch_local_tgt_knl,
        generate_box_particle_starts=generate_box_particle_starts,
        generate_box_particle_counts_nonchild=generate_box_particle_counts_nonchild,
        generate_box_particle_counts_cumul=generate_box_particle_counts_cumul
    )


def fetch_local_particles(queue, global_tree, src_box_mask, tgt_box_mask, local_tree,
                          local_data, knls):
    """ This helper function fetches particles needed for worker processes, and
    reconstruct list of lists indexing.

    Specifically, this function generates the following fields for the local tree:
    sources, targets, target_radii, box_source_starts, box_source_counts_nonchild,
    box_source_counts_cumul, box_target_starts, box_target_counts_nonchild,
    box_target_counts_cumul.

    These generated fields are stored directly into :arg:local_tree.

    """
    global_tree_dev = global_tree.to_device(queue).with_queue(queue)
    nsources = global_tree.nsources

    # {{{ source particle mask

    src_particle_mask = cl.array.zeros(
        queue, (nsources,),
        dtype=global_tree.particle_id_dtype
    )

    knls.particle_mask_knl(
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
    knls.mask_scan_knl(src_particle_mask, src_particle_scan)

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

    assert(global_tree.sources_have_extent is False)

    knls.fetch_local_src_knl(
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

    knls.generate_box_particle_starts(
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

    knls.generate_box_particle_counts_nonchild(
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

    knls.generate_box_particle_counts_cumul(
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

    knls.particle_mask_knl(
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
    knls.mask_scan_knl(tgt_particle_mask, tgt_particle_scan)

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

        knls.fetch_local_tgt_knl(
            tgt_particle_mask, tgt_particle_scan,
            *global_tree_dev.targets.tolist(),
            *local_targets_list,
            global_tree_dev.target_radii,
            local_target_radii
        )

    else:

        knls.fetch_local_tgt_knl(
            tgt_particle_mask, tgt_particle_scan,
            *global_tree_dev.targets.tolist(),
            *local_targets_list
        )

    # {{{ box_target_starts

    local_box_target_starts = cl.array.empty(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype
    )

    knls.generate_box_particle_starts(
        global_tree_dev.box_target_starts,
        tgt_particle_scan,
        local_box_target_starts
    )

    # }}}

    # {{{ box_target_counts_nonchild

    local_box_target_counts_nonchild = cl.array.zeros(
        queue, (global_tree.nboxes,),
        dtype=global_tree.particle_id_dtype)

    knls.generate_box_particle_counts_nonchild(
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

    knls.generate_box_particle_counts_cumul(
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

    # {{{ Fetch fields to local_data

    local_data["src_mask"] = src_particle_mask
    local_data["src_scan"] = src_particle_scan
    local_data["nsources"] = local_nsources
    local_data["tgt_mask"] = tgt_particle_mask
    local_data["tgt_scan"] = tgt_particle_scan
    local_data["ntargets"] = local_ntargets
    local_data["tgt_box_mask"] = tgt_box_mask

    # }}}


class LocalTreeBuilder:

    def __init__(self, global_tree, queue):
        self.global_tree = global_tree
        self.knls = get_fetch_local_particles_knls(queue.context, global_tree)
        self.queue = queue

    def from_global_tree(self, responsible_boxes_list, responsible_boxes_mask,
                         src_boxes_mask, ancestor_mask):

        local_tree = self.global_tree.copy(
            responsible_boxes_list=responsible_boxes_list,
            ancestor_mask=ancestor_mask.get(),
            box_to_user_starts=None,
            box_to_user_lists=None,
            _dimensions=None,
            _ntargets=None,
            _nsources=None,
        )

        local_tree.user_source_ids = None
        local_tree.sorted_target_ids = None

        local_data = {
            "src_mask": None, "src_scan": None, "nsources": None,
            "tgt_mask": None, "tgt_scan": None, "ntargets": None
        }

        fetch_local_particles(
            self.queue,
            self.global_tree,
            src_boxes_mask,
            responsible_boxes_mask,
            local_tree,
            local_data,
            self.knls
        )

        local_tree._dimensions = local_tree.dimensions
        local_tree._ntargets = local_tree.targets[0].shape[0]
        local_tree._nsources = local_tree.sources[0].shape[0]

        local_tree.__class__ = LocalTree

        return local_tree, local_data


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


def generate_local_tree(queue, traversal, responsible_boxes_list,
                        responsible_box_query, comm=MPI.COMM_WORLD):

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    if current_rank == 0:
        start_time = time.time()

    if current_rank == 0:
        local_data = np.empty((total_rank,), dtype=object)
    else:
        local_data = None

    if current_rank == 0:
        tree = traversal.tree

        local_tree_builder = LocalTreeBuilder(tree, queue)

        box_mpole_is_used = cl.array.empty(
            queue, (total_rank, tree.nboxes,), dtype=np.int8
        )

        # request objects for non-blocking communication
        tree_req = []
        particles_req = []

        # buffer holding communication data so that it is not garbage collected
        local_tree = np.empty((total_rank,), dtype=object)
        local_targets = np.empty((total_rank,), dtype=object)
        local_sources = np.empty((total_rank,), dtype=object)
        local_target_radii = np.empty((total_rank,), dtype=object)

        for irank in range(total_rank):

            (responsible_boxes_mask, ancestor_boxes, src_boxes_mask,
             box_mpole_is_used[irank]) = \
                responsible_box_query.get_boxes_mask(responsible_boxes_list[irank])

            local_tree[irank], local_data[irank] = \
                local_tree_builder.from_global_tree(
                    responsible_boxes_list[irank], responsible_boxes_mask,
                    src_boxes_mask, ancestor_boxes
                )

            # master process does not need to communicate with itself
            if irank == 0:
                continue

            # {{{ Peel sources and targets off tree

            local_targets[irank] = local_tree[irank].targets
            local_tree[irank].targets = None

            local_sources[irank] = local_tree[irank].sources
            local_tree[irank].sources = None

            local_target_radii[irank] = local_tree[irank].target_radii
            local_tree[irank].target_radii = None

            # }}}

            # Send the local tree skeleton without sources and targets
            tree_req.append(comm.isend(
                local_tree[irank], dest=irank, tag=MPITags["DIST_TREE"]))

            # Send the sources and targets
            particles_req.append(comm.Isend(
                local_sources[irank], dest=irank, tag=MPITags["DIST_SOURCES"]))

            particles_req.append(comm.Isend(
                local_targets[irank], dest=irank, tag=MPITags["DIST_TARGETS"]))

            if tree.targets_have_extent:
                particles_req.append(comm.Isend(
                    local_target_radii[irank], dest=irank, tag=MPITags["DIST_RADII"])
                )

        from boxtree.tools import MaskCompressorKernel
        matcompr = MaskCompressorKernel(queue.context)
        (box_to_user_starts, box_to_user_lists, evt) = \
            matcompr(queue, box_mpole_is_used.transpose(),
                     list_dtype=np.int32)

        cl.wait_for_events([evt])
        del box_mpole_is_used

        box_to_user_starts = box_to_user_starts.get()
        box_to_user_lists = box_to_user_lists.get()

        logger.debug("computing box_to_user: done")

    # Receive the local tree from root
    if current_rank == 0:
        MPI.Request.Waitall(tree_req)
        local_tree = local_tree[0]
    else:
        local_tree = comm.recv(source=0, tag=MPITags["DIST_TREE"])

    # Receive sources and targets
    if current_rank == 0:
        MPI.Request.Waitall(particles_req)
    else:
        reqs = []

        local_tree.sources = np.empty(
            (local_tree.dimensions, local_tree.nsources),
            dtype=local_tree.coord_dtype
        )
        reqs.append(comm.Irecv(
            local_tree.sources, source=0, tag=MPITags["DIST_SOURCES"]))

        local_tree.targets = np.empty(
            (local_tree.dimensions, local_tree.ntargets),
            dtype=local_tree.coord_dtype
        )
        reqs.append(comm.Irecv(
            local_tree.targets, source=0, tag=MPITags["DIST_TARGETS"]))

        if local_tree.targets_have_extent:
            local_tree.target_radii = np.empty(
                (local_tree.ntargets,),
                dtype=local_tree.coord_dtype
            )
            reqs.append(comm.Irecv(
                local_tree.target_radii, source=0, tag=MPITags["DIST_RADII"]))

        MPI.Request.Waitall(reqs)

    # Receive box extent
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

    if current_rank != 0:
        box_to_user_starts = None
        box_to_user_lists = None

    box_to_user_starts = comm.bcast(box_to_user_starts, root=0)
    box_to_user_lists = comm.bcast(box_to_user_lists, root=0)

    local_tree.box_to_user_starts = box_to_user_starts
    local_tree.box_to_user_lists = box_to_user_lists

    if current_rank == 0:
        logger.info("Distribute local tree in {} sec.".format(
            str(time.time() - start_time))
        )

    return local_tree, local_data, box_bounding_box
