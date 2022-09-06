__copyright__ = "Copyright (C) 2012 Andreas Kloeckner \
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
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from pytools import memoize_method
from dataclasses import dataclass


def get_box_ids_dfs_order(tree):
    """Helper function for getting box ids of a tree in depth-first order.

    :arg tree: A :class:`boxtree.Tree` object in the host memory. See
        :meth:`boxtree.Tree.get` for getting a tree object in host memory.
    :return: A numpy array of box ids in depth-first order.
    """
    # FIXME: optimize the performance with OpenCL
    dfs_order = np.empty((tree.nboxes,), dtype=tree.box_id_dtype)
    idx = 0
    stack = [0]
    while stack:
        box_id = stack.pop()
        dfs_order[idx] = box_id
        idx += 1
        for i in range(2**tree.dimensions):
            child_box_id = tree.box_child_ids[i][box_id]
            if child_box_id > 0:
                stack.append(child_box_id)
    return dfs_order


def partition_work(cost_per_box, traversal, comm):
    """This function assigns responsible boxes for each rank.

    If a rank is responsible for a box, it will calculate the multiple expansion of
    the box and evaluate target potentials in the box.

    :arg cost_per_box: The expected running time of each box. This argument is only
        significant on the root rank.
    :arg traversal: The global traversal object containing all particles. This
        argument is significant on all ranks.
    :arg comm: MPI communicator.
    :return: A numpy array containing the responsible boxes of the current rank.
    """
    tree = traversal.tree
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    if mpi_size > tree.nboxes:
        raise RuntimeError("Fail to partition work because the number of boxes is "
                           "less than the number of processes.")

    # transform tree from the level order to the morton dfs order
    # dfs_order[i] stores the level-order box index of dfs index i
    dfs_order = get_box_ids_dfs_order(tree)

    # partition all boxes in dfs order evenly according to workload on the root rank

    responsible_boxes_segments = None
    # contains: [start_index, end_index)
    responsible_boxes_current_rank = np.empty(2, dtype=tree.box_id_dtype)

    # FIXME: Right now, the responsible boxes assigned to all ranks are computed
    # centrally on the root rank to avoid inconsistency risks of floating point
    # operations. We could improve the efficiency by letting each rank compute the
    # costs of a subset of boxes, and use MPI_Scan to aggregate the results.
    if mpi_rank == 0:
        total_workload = np.sum(cost_per_box)

        # second axis: [start_index, end_index)
        responsible_boxes_segments = np.empty((mpi_size, 2), dtype=tree.box_id_dtype)
        segment_idx = 0
        start = 0
        workload_count = 0
        for box_idx_dfs_order in range(tree.nboxes):
            if segment_idx + 1 == mpi_size:
                responsible_boxes_segments[segment_idx, :] = [start, tree.nboxes]
                break

            box_idx = dfs_order[box_idx_dfs_order]
            workload_count += cost_per_box[box_idx]
            if (workload_count > (segment_idx + 1) * total_workload / mpi_size
                    or box_idx_dfs_order == tree.nboxes - 1):
                # record "end of rank segment"
                responsible_boxes_segments[segment_idx, :] = (
                    [start, box_idx_dfs_order + 1])
                start = box_idx_dfs_order + 1
                segment_idx += 1

    comm.Scatter(responsible_boxes_segments, responsible_boxes_current_rank, root=0)

    return dfs_order[
        responsible_boxes_current_rank[0]:responsible_boxes_current_rank[1]]


class GetBoxMasksCodeContainer:
    def __init__(self, cl_context, box_id_dtype):
        self.cl_context = cl_context
        self.box_id_dtype = box_id_dtype

    @memoize_method
    def add_interaction_list_boxes_kernel(self):
        """Given a ``responsible_boxes_mask`` and an interaction list, mark source
        boxes for target boxes in ``responsible_boxes_mask`` in a new separate mask.
        """
        return cl.elementwise.ElementwiseKernel(
            self.cl_context,
            Template("""
                __global ${box_id_t} *box_list,
                __global char *responsible_boxes_mask,
                __global ${box_id_t} *interaction_boxes_starts,
                __global ${box_id_t} *interaction_boxes_lists,
                __global char *src_boxes_mask
            """, strict_undefined=True).render(
                box_id_t=dtype_to_ctype(self.box_id_dtype)
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
                box_id_t=dtype_to_ctype(self.box_id_dtype)
            ),
        )

    @memoize_method
    def add_parent_boxes_kernel(self):
        return cl.elementwise.ElementwiseKernel(
            self.cl_context,
            "__global char *current, __global char *parent, "
            "__global %s *box_parent_ids" % dtype_to_ctype(self.box_id_dtype),
            "if(i != 0 && current[i]) parent[box_parent_ids[i]] = 1"
        )


def get_ancestor_boxes_mask(queue, code, traversal, responsible_boxes_mask):
    """Query the ancestors of responsible boxes.

    :arg responsible_boxes_mask: A :class:`pyopencl.array.Array` object of shape
        ``(tree.nboxes,)`` whose i-th entry is 1 if ``i`` is a responsible box.
    :return: A :class:`pyopencl.array.Array` object of shape ``(tree.nboxes,)`` whose
        i-th entry is 1 if ``i`` is an ancestor of the responsible boxes specified by
        *responsible_boxes_mask*.
    """
    ancestor_boxes = cl.array.zeros(queue, (traversal.tree.nboxes,), dtype=np.int8)
    ancestor_boxes_last = responsible_boxes_mask.copy()

    while ancestor_boxes_last.any():
        ancestor_boxes_new = cl.array.zeros(
            queue, (traversal.tree.nboxes,), dtype=np.int8)
        code.add_parent_boxes_kernel()(
            ancestor_boxes_last, ancestor_boxes_new, traversal.tree.box_parent_ids)
        ancestor_boxes_new = ancestor_boxes_new & (~ancestor_boxes)
        ancestor_boxes = ancestor_boxes | ancestor_boxes_new
        ancestor_boxes_last = ancestor_boxes_new

    return ancestor_boxes


def get_point_src_boxes_mask(
        queue, code, traversal, responsible_boxes_mask, ancestor_boxes_mask):
    """Query the boxes whose sources are needed in order to evaluate potentials
    of boxes represented by *responsible_boxes_mask*.

    :arg responsible_boxes_mask: A :class:`pyopencl.array.Array` object of shape
        ``(tree.nboxes,)`` whose i-th entry is 1 if ``i`` is a responsible box.
    :param ancestor_boxes_mask: A :class:`pyopencl.array.Array` object of shape
        ``(tree.nboxes,)`` whose i-th entry is 1 if ``i`` is either a responsible box
        or an ancestor of the responsible boxes.
    :return: A :class:`pyopencl.array.Array` object of shape ``(tree.nboxes,)`` whose
        i-th entry is 1 if souces of box ``i`` are needed for evaluating the
        potentials of targets in boxes represented by *responsible_boxes_mask*.
    """

    src_boxes_mask = responsible_boxes_mask.copy()

    # Add list 1 of responsible boxes
    code.add_interaction_list_boxes_kernel()(
        traversal.target_boxes, responsible_boxes_mask,
        traversal.neighbor_source_boxes_starts,
        traversal.neighbor_source_boxes_lists, src_boxes_mask,
        queue=queue)

    # Add list 4 of responsible boxes or ancestor boxes
    code.add_interaction_list_boxes_kernel()(
        traversal.target_or_target_parent_boxes,
        responsible_boxes_mask | ancestor_boxes_mask,
        traversal.from_sep_bigger_starts, traversal.from_sep_bigger_lists,
        src_boxes_mask,
        queue=queue)

    if traversal.tree.targets_have_extent:
        # Add list 3 close of responsible boxes
        if traversal.from_sep_close_smaller_starts is not None:
            code.add_interaction_list_boxes_kernel()(
                traversal.target_boxes,
                responsible_boxes_mask,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_boxes_mask,
                queue=queue
            )

        # Add list 4 close of responsible boxes
        if traversal.from_sep_close_bigger_starts is not None:
            code.add_interaction_list_boxes_kernel()(
                traversal.target_boxes,
                responsible_boxes_mask | ancestor_boxes_mask,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_boxes_mask,
                queue=queue
            )

    return src_boxes_mask


def get_multipole_src_boxes_mask(
        queue, code, traversal, responsible_boxes_mask, ancestor_boxes_mask):
    """Query the boxes whose multipoles are used in order to evaluate
    potentials of targets in boxes represented by *responsible_boxes_mask*.

    :arg responsible_boxes_mask: A :class:`pyopencl.array.Array` object of shape
        ``(tree.nboxes,)`` whose i-th entry is 1 if ``i`` is a responsible box.
    :arg ancestor_boxes_mask: A :class:`pyopencl.array.Array` object of shape
        ``(tree.nboxes,)`` whose i-th entry is 1 if ``i`` is either a responsible box
        or an ancestor of the responsible boxes.
    :return: A :class:`pyopencl.array.Array` object of shape ``(tree.nboxes,)``
        whose i-th entry is 1 if multipoles of box ``i`` are needed for evaluating
        the potentials of targets in boxes represented by *responsible_boxes_mask*.
    """

    multipole_boxes_mask = cl.array.zeros(
        queue, (traversal.tree.nboxes,), dtype=np.int8
    )

    # A mpole is used by process p if it is in the List 2 of either a box
    # owned by p or one of its ancestors.
    code.add_interaction_list_boxes_kernel()(
        traversal.target_or_target_parent_boxes,
        responsible_boxes_mask | ancestor_boxes_mask,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        multipole_boxes_mask,
        queue=queue
    )
    multipole_boxes_mask.finish()

    # A mpole is used by process p if it is in the List 3 of a box owned by p.
    for ilevel in range(traversal.tree.nlevels):
        code.add_interaction_list_boxes_kernel()(
            traversal.target_boxes_sep_smaller_by_source_level[ilevel],
            responsible_boxes_mask,
            traversal.from_sep_smaller_by_level[ilevel].starts,
            traversal.from_sep_smaller_by_level[ilevel].lists,
            multipole_boxes_mask,
            queue=queue
        )

        multipole_boxes_mask.finish()

    return multipole_boxes_mask


@dataclass
class BoxMasks:
    """
    Box masks needed for the distributed calculation. Each of these masks is a
    PyOpenCL array with length ``tree.nboxes``, whose `i`-th entry is 1 if box `i` is
    set.

    .. attribute:: responsible_boxes

        Current process will evaluate target potentials and multipole expansions in
        these boxes. Sources and targets in these boxes are needed.

    .. attribute:: ancestor_boxes

        Ancestors of the responsible boxes.

    .. attribute:: point_src_boxes

        Current process needs sources but not targets in these boxes.

    .. attribute:: multipole_src_boxes

        Current process needs multipole expressions in these boxes.
    """
    responsible_boxes: cl.array.Array
    ancestor_boxes: cl.array.Array
    point_src_boxes: cl.array.Array
    multipole_src_boxes: cl.array.Array


def get_box_masks(queue, traversal, responsible_boxes_list):
    """Given the responsible boxes for a rank, this helper function calculates the
    relevant masks.

    :arg responsible_boxes_list: A numpy array of responsible box indices.

    :returns: A :class:`BoxMasks` object of the relevant masks.
    """
    code = GetBoxMasksCodeContainer(queue.context, traversal.tree.box_id_dtype)

    # FIXME: It is wasteful to copy the whole traversal object into device memory
    # here because
    # 1) Not all fields are needed.
    # 2) For sumpy wrangler, a device traversal object is already available.
    traversal = traversal.to_device(queue)

    responsible_boxes_mask = np.zeros((traversal.tree.nboxes,), dtype=np.int8)
    responsible_boxes_mask[responsible_boxes_list] = 1
    responsible_boxes_mask = cl.array.to_device(queue, responsible_boxes_mask)

    ancestor_boxes_mask = get_ancestor_boxes_mask(
        queue, code, traversal, responsible_boxes_mask)

    point_src_boxes_mask = get_point_src_boxes_mask(
        queue, code, traversal, responsible_boxes_mask, ancestor_boxes_mask)

    multipole_src_boxes_mask = get_multipole_src_boxes_mask(
        queue, code, traversal, responsible_boxes_mask, ancestor_boxes_mask)

    return BoxMasks(
        responsible_boxes_mask, ancestor_boxes_mask, point_src_boxes_mask,
        multipole_src_boxes_mask)
