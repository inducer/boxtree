from __future__ import division

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
from pytools import memoize


def partition_work(boxes_time, traversal, total_rank):
    """This function assigns responsible boxes for each rank.

    Each process is responsible for calculating the multiple expansions as well as
    evaluating target potentials in *responsible_boxes*.

    :arg boxes_time: The expected running time of each box.
    :arg traversal: The traversal object built on root containing all particles.
    :arg total_rank: The total number of ranks.
    :return: A numpy array of shape ``(total_rank,)``, where the i-th element is an
        numpy array containing the responsible boxes of process i.
    """
    tree = traversal.tree

    if total_rank > tree.nboxes:
        raise RuntimeError("Fail to partition work because the number of boxes is "
                           "less than the number of processes.")

    total_workload = 0
    for i in range(tree.nboxes):
        total_workload += boxes_time[i]

    # transform tree from level order to dfs order
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

    # partition all boxes in dfs order evenly according to workload
    responsible_boxes_list = np.empty((total_rank,), dtype=object)
    rank = 0
    start = 0
    workload_count = 0
    for i in range(tree.nboxes):
        if rank + 1 == total_rank:
            responsible_boxes_list[rank] = dfs_order[start:tree.nboxes]
            break

        box_idx = dfs_order[i]
        workload_count += boxes_time[box_idx]
        if (workload_count > (rank + 1)*total_workload/total_rank
                or i == tree.nboxes - 1):
            responsible_boxes_list[rank] = dfs_order[start:i+1]
            start = i + 1
            rank += 1

    return responsible_boxes_list


@memoize
def mark_parent_kernel(context, box_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        "__global char *current, __global char *parent, "
        "__global %s *box_parent_ids" % dtype_to_ctype(box_id_dtype),
        "if(i != 0 && current[i]) parent[box_parent_ids[i]] = 1"
    )


# helper kernel for adding boxes from interaction list 1 and 4
@memoize
def add_interaction_list_boxes_kernel(context, box_id_dtype):
    return cl.elementwise.ElementwiseKernel(
        context,
        Template("""
            __global ${box_id_t} *box_list,
            __global char *responsible_boxes_mask,
            __global ${box_id_t} *interaction_boxes_starts,
            __global ${box_id_t} *interaction_boxes_lists,
            __global char *src_boxes_mask
        """, strict_undefined=True).render(
            box_id_t=dtype_to_ctype(box_id_dtype)
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
            box_id_t=dtype_to_ctype(box_id_dtype)
        ),
    )


def get_ancestor_boxes_mask(queue, traversal, responsible_boxes_mask):
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
            queue, (traversal.tree.nboxes,), dtype=np.int8
        )
        mark_parent_kernel(queue.context, traversal.tree.box_id_dtype)(
            ancestor_boxes_last, ancestor_boxes_new, traversal.tree.box_parent_ids
        )
        ancestor_boxes_new = ancestor_boxes_new & (~ancestor_boxes)
        ancestor_boxes = ancestor_boxes | ancestor_boxes_new
        ancestor_boxes_last = ancestor_boxes_new

    return ancestor_boxes


def get_src_boxes_mask(
        queue, traversal, responsible_boxes_mask, ancestor_boxes_mask):
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
    add_interaction_list_boxes_kernel(queue.context, traversal.tree.box_id_dtype)(
        traversal.target_boxes, responsible_boxes_mask,
        traversal.neighbor_source_boxes_starts,
        traversal.neighbor_source_boxes_lists, src_boxes_mask,
        range=range(0, traversal.target_boxes.shape[0]),
        queue=queue
    )

    # Add list 4 of responsible boxes or ancestor boxes
    add_interaction_list_boxes_kernel(queue.context, traversal.tree.box_id_dtype)(
        traversal.target_or_target_parent_boxes,
        responsible_boxes_mask | ancestor_boxes_mask,
        traversal.from_sep_bigger_starts, traversal.from_sep_bigger_lists,
        src_boxes_mask,
        range=range(0, traversal.target_or_target_parent_boxes.shape[0]),
        queue=queue
    )

    if traversal.tree.targets_have_extent:
        # Add list 3 close of responsible boxes
        if traversal.from_sep_close_smaller_starts is not None:
            add_interaction_list_boxes_kernel(
                    queue.context, traversal.tree.box_id_dtype)(
                traversal.target_boxes,
                responsible_boxes_mask,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_boxes_mask,
                queue=queue
            )

        # Add list 4 close of responsible boxes
        if traversal.from_sep_close_bigger_starts is not None:
            add_interaction_list_boxes_kernel(
                    queue.context, traversal.tree.box_id_dtype)(
                traversal.target_boxes,
                responsible_boxes_mask | ancestor_boxes_mask,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_boxes_mask,
                queue=queue
            )

    return src_boxes_mask


def get_multipole_boxes_mask(
        queue, traversal, responsible_boxes_mask, ancestor_boxes_mask):
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
    add_interaction_list_boxes_kernel(queue.context, traversal.tree.box_id_dtype)(
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
        add_interaction_list_boxes_kernel(
                queue.context, traversal.tree.box_id_dtype)(
            traversal.target_boxes_sep_smaller_by_source_level[ilevel],
            responsible_boxes_mask,
            traversal.from_sep_smaller_by_level[ilevel].starts,
            traversal.from_sep_smaller_by_level[ilevel].lists,
            multipole_boxes_mask,
            queue=queue
        )

        multipole_boxes_mask.finish()

    return multipole_boxes_mask


def get_boxes_mask(queue, traversal, responsible_boxes_list):
    """Given the responsible boxes for a rank, this helper function calculates the
    following four masks:

    * responsible_box_mask: Current process will evaluate target potentials and
      multipole expansions in these boxes. Sources and targets in these boxes
      are needed.
    * ancestor_boxes_mask: The the ancestor of the responsible boxes.
    * src_boxes_mask: Current process needs sources but not targets in these boxes.
    * multipole_boxes_mask: Current process needs multipole expressions in these
      boxes.

    :arg responsible_boxes_list: A numpy array of responsible box indices.

    :returns: responsible_box_mask, ancestor_boxes_mask, src_boxes_mask and
        multipole_boxes_mask, as described above.
    """
    traversal = traversal.to_device(queue)

    responsible_boxes_mask = np.zeros((traversal.tree.nboxes,), dtype=np.int8)
    responsible_boxes_mask[responsible_boxes_list] = 1
    responsible_boxes_mask = cl.array.to_device(queue, responsible_boxes_mask)

    ancestor_boxes_mask = get_ancestor_boxes_mask(
        queue, traversal, responsible_boxes_mask
    )

    src_boxes_mask = get_src_boxes_mask(
        queue, traversal, responsible_boxes_mask, ancestor_boxes_mask
    )

    multipole_boxes_mask = get_multipole_boxes_mask(
        queue, traversal, responsible_boxes_mask, ancestor_boxes_mask
    )

    return (responsible_boxes_mask, ancestor_boxes_mask, src_boxes_mask,
            multipole_boxes_mask)
