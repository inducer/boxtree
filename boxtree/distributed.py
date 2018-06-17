from __future__ import division
from mpi4py import MPI
import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_1  # noqa: F401
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import GenericScanKernel
from pytools import memoize_in, memoize_method
from boxtree import Tree
from collections import namedtuple
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import time

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
            box_to_user_lists=box_to_user_lists,
            _dimensions=None,
            _ntargets=None,
            _nsources=None,
            _particle_dtype=None,
            _radii_dtype=None
        )
        local_tree.__class__ = cls
        return local_tree

    def to_device(self, queue):
        additional_fields_to_device = ["responsible_boxes_list", "ancestor_mask",
                                       "box_to_user_starts", "box_to_user_lists"]

        return tree_to_device(queue, self, additional_fields_to_device)


# {{{ distributed fmm wrangler

class DistributedFMMLibExpansionWrangler(FMMLibExpansionWrangler):

    def __init__(self, tree, helmholtz_k, fmm_level_to_nterms=None):
        super(DistributedFMMLibExpansionWrangler, self).__init__(
            tree, helmholtz_k, fmm_level_to_nterms
        )

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
        return cl.array.empty(queue, self.tree.nboxes, dtype=np.int8)

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
        knl(queue,
            subrange_start=subrange[0],
            subrange_end=subrange[1],
            box_to_user_starts=box_to_user_starts,
            box_to_user_lists=box_to_user_lists,
            box_in_subrange=box_in_subrange)

        box_in_subrange.finish()

# }}}


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

WorkloadWeight = namedtuple('Workload', ['direct', 'm2l', 'm2p', 'p2l', 'multipole'])


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


def get_gen_local_tree_kernels(tree):
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

    return dict(
        particle_mask_knl=particle_mask_knl,
        mask_scan_knl=mask_scan_knl,
        fetch_local_src_knl=fetch_local_src_knl,
        fetch_local_tgt_knl=fetch_local_tgt_knl,
        generate_box_particle_starts=generate_box_particle_starts,
        generate_box_particle_counts_nonchild=generate_box_particle_counts_nonchild,
        generate_box_particle_counts_cumul=generate_box_particle_counts_cumul
    )


def gen_local_tree_helper(tree, src_box_mask, tgt_box_mask, local_tree,
                          local_data, knls):
    """ This helper function generates a copy of the tree but with subset of
        particles, and fetch the generated fields to *local_tree*.
    """
    d_tree = tree_to_device(queue, tree)
    nsources = tree.nsources

    # source particle mask
    src_particle_mask = cl.array.zeros(queue, (nsources,),
                                       dtype=tree.particle_id_dtype)
    knls["particle_mask_knl"](src_box_mask,
                              d_tree.box_source_starts,
                              d_tree.box_source_counts_nonchild,
                              src_particle_mask)

    # scan of source particle mask
    src_particle_scan = cl.array.empty(queue, (nsources + 1,),
                                        dtype=tree.particle_id_dtype)
    src_particle_scan[0] = 0
    knls["mask_scan_knl"](src_particle_mask, src_particle_scan)

    # local sources
    local_nsources = src_particle_scan[-1].get(queue)
    local_sources = cl.array.empty(
        queue, (tree.dimensions, local_nsources), dtype=tree.coord_dtype)
    local_sources_list = [local_sources[idim, :] for idim in range(tree.dimensions)]

    assert(tree.sources_have_extent is False)

    knls["fetch_local_src_knl"](src_particle_mask, src_particle_scan,
                                *d_tree.sources.tolist(),
                                *local_sources_list)

    # box_source_starts
    local_box_source_starts = cl.array.empty(queue, (tree.nboxes,),
                                                dtype=tree.particle_id_dtype)
    knls["generate_box_particle_starts"](d_tree.box_source_starts, src_particle_scan,
                                         local_box_source_starts)

    # box_source_counts_nonchild
    local_box_source_counts_nonchild = cl.array.zeros(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
    knls["generate_box_particle_counts_nonchild"](src_box_mask,
                                                  d_tree.box_source_counts_nonchild,
                                                  local_box_source_counts_nonchild)

    # box_source_counts_cumul
    local_box_source_counts_cumul = cl.array.empty(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
    knls["generate_box_particle_counts_cumul"](d_tree.box_source_counts_cumul,
                                               d_tree.box_source_starts,
                                               local_box_source_counts_cumul,
                                               src_particle_scan)

    ntargets = tree.ntargets
    # target particle mask
    tgt_particle_mask = cl.array.zeros(queue, (ntargets,),
                                        dtype=tree.particle_id_dtype)
    knls["particle_mask_knl"](tgt_box_mask,
                              d_tree.box_target_starts,
                              d_tree.box_target_counts_nonchild,
                              tgt_particle_mask)

    # scan of target particle mask
    tgt_particle_scan = cl.array.empty(queue, (ntargets + 1,),
                                        dtype=tree.particle_id_dtype)
    tgt_particle_scan[0] = 0
    knls["mask_scan_knl"](tgt_particle_mask, tgt_particle_scan)

    # local targets
    local_ntargets = tgt_particle_scan[-1].get(queue)

    local_targets = cl.array.empty(
        queue, (tree.dimensions, local_ntargets), dtype=tree.coord_dtype)
    local_targets_list = [local_targets[idim, :] for idim in range(tree.dimensions)]

    if tree.targets_have_extent:
        local_target_radii = cl.array.empty(queue, (local_ntargets,),
                                            dtype=tree.coord_dtype)
        knls["fetch_local_tgt_knl"](tgt_particle_mask, tgt_particle_scan,
                                    *d_tree.targets.tolist(),
                                    *local_targets_list,
                                    d_tree.target_radii, local_target_radii)
    else:
        knls["fetch_local_tgt_knl"](tgt_particle_mask, tgt_particle_scan,
                                    *d_tree.targets.tolist(),
                                    *local_targets_list)

    # box_target_starts
    local_box_target_starts = cl.array.empty(queue, (tree.nboxes,),
                                                dtype=tree.particle_id_dtype)
    knls["generate_box_particle_starts"](d_tree.box_target_starts, tgt_particle_scan,
                                         local_box_target_starts)

    # box_target_counts_nonchild
    local_box_target_counts_nonchild = cl.array.zeros(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
    knls["generate_box_particle_counts_nonchild"](tgt_box_mask,
                                                  d_tree.box_target_counts_nonchild,
                                                  local_box_target_counts_nonchild)

    # box_target_counts_cumul
    local_box_target_counts_cumul = cl.array.empty(
        queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
    knls["generate_box_particle_counts_cumul"](d_tree.box_target_counts_cumul,
                                               d_tree.box_target_starts,
                                               local_box_target_counts_cumul,
                                               tgt_particle_scan)

    # Fetch fields to local_tree
    local_sources = local_sources.get(queue=queue)
    local_tree.sources = local_sources

    local_targets = local_targets.get(queue=queue)
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

    # Fetch fields to local_data
    local_data["src_mask"] = src_particle_mask
    local_data["src_scan"] = src_particle_scan
    local_data["nsources"] = local_nsources
    local_data["tgt_mask"] = tgt_particle_mask
    local_data["tgt_scan"] = tgt_particle_scan
    local_data["ntargets"] = local_ntargets
    local_data["tgt_box_mask"] = tgt_box_mask


def generate_local_tree(traversal, comm=MPI.COMM_WORLD, workload_weight=None):

    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # Log OpenCL context information
    logger.info("Process %d of %d on %s with ctx %s." % (
        comm.Get_rank(),
        comm.Get_size(),
        MPI.Get_processor_name(),
        queue.context.devices)
    )

    if current_rank == 0:
        start_time = time.time()

    # {{{ Construct local tree for each rank on root

    if current_rank == 0:
        local_data = np.empty((total_rank,), dtype=object)
        for i in range(total_rank):
            local_data[i] = {
                "src_mask": None, "src_scan": None, "nsources": None,
                "tgt_mask": None, "tgt_scan": None, "ntargets": None
            }
    else:
        local_data = None

    knls = None

    if current_rank == 0:
        tree = traversal.tree

        local_tree = np.empty((total_rank,), dtype=object)
        local_targets = np.empty((total_rank,), dtype=object)
        local_sources = np.empty((total_rank,), dtype=object)
        local_target_radii = np.empty((total_rank,), dtype=object)

        # {{{ Partition the work

        # Each rank is responsible for calculating the multiple expansion as well as
        # evaluating target potentials in *responsible_boxes*
        if workload_weight is None:
            workload_weight = WorkloadWeight(
                direct=1,
                m2l=1,
                m2p=1,
                p2l=1,
                multipole=5
            )

        from boxtree.partition import partition_work
        responsible_boxes_list = partition_work(traversal, total_rank,
                                                workload_weight)

        responsible_boxes_mask = np.zeros((total_rank, tree.nboxes), dtype=np.int8)
        for irank in range(total_rank):
            responsible_boxes_mask[irank, responsible_boxes_list[irank]] = 1
        responsible_boxes_mask = cl.array.to_device(queue, responsible_boxes_mask)

        for irank in range(total_rank):
            responsible_boxes_list[irank] = cl.array.to_device(
                queue, responsible_boxes_list[irank])

        from boxtree.partition import ResponsibleBoxesQuery
        responsible_box_query = ResponsibleBoxesQuery(queue, traversal)

        # Calculate ancestors of responsible boxes
        ancestor_boxes = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                        dtype=np.int8)
        for irank in range(total_rank):
            ancestor_boxes[irank, :] = responsible_box_query.ancestor_boxes_mask(
                responsible_boxes_mask[irank, :])

        # In order to evaluate, each rank needs sources in boxes in
        # *src_boxes_mask*
        src_boxes_mask = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                        dtype=np.int8)

        for irank in range(total_rank):
            src_boxes_mask[irank, :] = responsible_box_query.src_boxes_mask(
                responsible_boxes_mask[irank, :], ancestor_boxes[irank, :]
            )

        # {{{ compute box_to_user

        logger.debug("computing box_to_user: start")

        box_mpole_is_used = cl.array.zeros(queue, (total_rank, tree.nboxes),
                                           dtype=np.int8)

        for irank in range(total_rank):
            box_mpole_is_used[irank, :] = \
                responsible_box_query.multipole_boxes_mask(
                    responsible_boxes_mask[irank, :], ancestor_boxes[irank, :]
                )

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

        # kernels for generating local trees
        knls = get_gen_local_tree_kernels(tree)

        # request objects for non-blocking communication
        tree_req = []
        particles_req = []

        for rank in range(total_rank):
            local_tree[rank] = LocalTree.copy_from_global_tree(
                tree, responsible_boxes_list[rank].get(),
                ancestor_boxes[rank].get(),
                box_to_user_starts.get(),
                box_to_user_lists.get())

            local_tree[rank].user_source_ids = None
            local_tree[rank].sorted_target_ids = None

            gen_local_tree_helper(tree,
                                  src_boxes_mask[rank],
                                  responsible_boxes_mask[rank],
                                  local_tree[rank],
                                  local_data[rank],
                                  knls)

            if rank == 0:
                # master process does not need to communicate with itself
                continue

            # {{{ Peel sources and targets off tree

            local_tree[rank]._dimensions = local_tree[rank].dimensions

            local_tree[rank]._ntargets = local_tree[rank].ntargets
            local_targets[rank] = local_tree[rank].targets
            local_tree[rank].targets = None

            local_tree[rank]._nsources = local_tree[rank].nsources
            local_sources[rank] = local_tree[rank].sources
            local_tree[rank].sources = None

            local_target_radii[rank] = local_tree[rank].target_radii
            local_tree[rank].target_radii = None

            local_tree[rank]._particle_dtype = tree.sources[0].dtype
            local_tree[rank]._radii_dtype = tree.target_radii.dtype

            # }}}

            # Send the local tree skeleton without sources and targets
            tree_req.append(comm.isend(
                local_tree[rank], dest=rank, tag=MPITags["DIST_TREE"]))

            # Send the sources and targets
            particles_req.append(comm.Isend(
                local_sources[rank], dest=rank, tag=MPITags["DIST_SOURCES"]))

            particles_req.append(comm.Isend(
                local_targets[rank], dest=rank, tag=MPITags["DIST_TARGETS"]))

            if tree.targets_have_extent:
                particles_req.append(comm.Isend(
                    local_target_radii[rank], dest=rank, tag=MPITags["DIST_RADII"]))

    # }}}

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
            (local_tree._dimensions, local_tree._nsources),
            dtype=local_tree._particle_dtype
        )
        reqs.append(comm.Irecv(
            local_tree.sources, source=0, tag=MPITags["DIST_SOURCES"]))

        local_tree.targets = np.empty(
            (local_tree._dimensions, local_tree._ntargets),
            dtype=local_tree._particle_dtype
        )
        reqs.append(comm.Irecv(
            local_tree.targets, source=0, tag=MPITags["DIST_TARGETS"]))

        if local_tree.targets_have_extent:
            local_tree.target_radii = np.empty(
                (local_tree._ntargets,),
                dtype=local_tree._radii_dtype
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

    if current_rank == 0:
        logger.info("Distribute local tree in {} sec.".format(
            str(time.time() - start_time))
        )

    return local_tree, local_data, box_bounding_box, knls


def generate_local_travs(
        local_tree, box_bounding_box=None, comm=MPI.COMM_WORLD,
        well_sep_is_n_away=1, from_sep_smaller_crit=None,
        merge_close_lists=False):

    start_time = time.time()

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

    # Generate local source flags
    local_box_flags = d_tree.box_flags & 250
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

    modify_own_sources_knl(d_tree.responsible_boxes_list, local_box_flags)
    modify_child_sources_knl(d_tree.ancestor_mask, local_box_flags)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(
        queue.context,
        well_sep_is_n_away=well_sep_is_n_away,
        from_sep_smaller_crit=from_sep_smaller_crit
    )

    d_local_trav, _ = tg(
        queue, d_tree, debug=True,
        box_bounding_box=box_bounding_box,
        local_box_flags=local_box_flags
    )

    if merge_close_lists and d_tree.targets_have_extent:
        d_local_trav = d_local_trav.merge_close_lists(queue)

    local_trav = d_local_trav.get(queue=queue)

    logger.info("Generate local traversal in {} sec.".format(
        str(time.time() - start_time))
    )

    return local_trav


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


def distribute_source_weights(source_weights, global_tree, local_data,
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
        source_weights = cl.array.to_device(queue, source_weights)
        gen_local_weights_helper = get_gen_local_weights_helper(
            queue, global_tree.particle_id_dtype, source_weights.dtype)
        for rank in range(total_rank):
            local_src_weights[rank] = gen_local_weights_helper(
                    source_weights,
                    local_data[rank]["src_mask"],
                    local_data[rank]["src_scan"]
            )
            weight_req[rank] = comm.isend(local_src_weights[rank], dest=rank,
                                          tag=MPITags["DIST_WEIGHT"])

        for rank in range(1, total_rank):
            weight_req[rank].wait()
        local_src_weights = local_src_weights[0]
    else:
        local_src_weights = comm.recv(source=0, tag=MPITags["DIST_WEIGHT"])

    return local_src_weights


def calculate_pot(wrangler, global_wrangler, local_trav, source_weights,
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
        source_weights, global_tree, local_data, comm=comm)

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
                (local_data[i]["ntargets"],), dtype=potentials.dtype)
            comm.Recv([potentials_all_ranks[i], potentials_mpi_type],
                      source=i, tag=MPITags["GATHER_POTENTIALS"])
    else:
        comm.Send([potentials, potentials_mpi_type],
                  dest=0, tag=MPITags["GATHER_POTENTIALS"])

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
                local_data[i]["tgt_mask"], local_data[i]["tgt_scan"],
                local_potentials, d_potentials)

        potentials = d_potentials.get()

        logger.debug("reorder potentials")
        result = global_wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = global_wrangler.finalize_potentials(result)

        logger.info("Distributed FMM evaluation completes in {} sec.".format(
            str(time.time() - start_time)
        ))

        return result


class DistributedFMMInfo(object):

    def __init__(self, global_trav, distributed_expansion_wrangler_factory,
                 comm=MPI.COMM_WORLD):
        self.global_trav = global_trav
        self.distributed_expansion_wrangler_factory = \
            distributed_expansion_wrangler_factory

        self.comm = comm
        current_rank = comm.Get_rank()

        if current_rank == 0:
            well_sep_is_n_away = global_trav.well_sep_is_n_away
        else:
            well_sep_is_n_away = None
        well_sep_is_n_away = comm.bcast(well_sep_is_n_away, root=0)

        self.local_tree, self.local_data, self.box_bounding_box, _ = \
            generate_local_tree(self.global_trav)
        self.local_trav = generate_local_travs(
            self.local_tree, self.box_bounding_box, comm=comm,
            well_sep_is_n_away=well_sep_is_n_away)
        self.local_wrangler = self.distributed_expansion_wrangler_factory(
            self.local_tree)
        if current_rank == 0:
            self.global_wrangler = self.distributed_expansion_wrangler_factory(
                self.global_trav.tree)
        else:
            self.global_wrangler = None

    def drive_dfmm(self, source_weights):
        return calculate_pot(
            self.local_wrangler, self.global_wrangler, self.local_trav,
            source_weights, self.local_data)
