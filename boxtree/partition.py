import numpy as np

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


def partition_work(traversal, total_rank, workload_weight):
    """ This function assigns responsible boxes of each process.

    :arg traversal: The traversal object built on root containing all particles.
    :arg total_rank: The total number of processes.
    :arg workload_weight: Workload coefficients of various operations (e.g. direct
        evaluations, multipole-to-local, etc.) used for load balacing.
    :return: A numpy array of shape (total_rank,), where the ith element is an numpy
        array containing the responsible boxes of process i.
    """
    tree = traversal.tree

    # store the workload of each box
    workload = np.zeros((tree.nboxes,), dtype=np.float64)

    # add workload of list 1
    for itarget_box, box_idx in enumerate(traversal.target_boxes):
        box_ntargets = tree.box_target_counts_nonchild[box_idx]
        start = traversal.neighbor_source_boxes_starts[itarget_box]
        end = traversal.neighbor_source_boxes_starts[itarget_box + 1]
        list1 = traversal.neighbor_source_boxes_lists[start:end]
        particle_count = 0
        for ibox in list1:
            particle_count += tree.box_source_counts_nonchild[ibox]
        workload[box_idx] += box_ntargets * particle_count * workload_weight.direct

    # add workload of list 2
    for itarget_or_target_parent_boxes, box_idx in enumerate(
            traversal.target_or_target_parent_boxes):
        start = traversal.from_sep_siblings_starts[itarget_or_target_parent_boxes]
        end = traversal.from_sep_siblings_starts[itarget_or_target_parent_boxes + 1]
        workload[box_idx] += (end - start) * workload_weight.m2l

    for ilevel in range(tree.nlevels):
        # add workload of list 3 far
        for itarget_box, box_idx in enumerate(
                traversal.target_boxes_sep_smaller_by_source_level[ilevel]):
            box_ntargets = tree.box_target_counts_nonchild[box_idx]
            start = traversal.from_sep_smaller_by_level[ilevel].starts[itarget_box]
            end = traversal.from_sep_smaller_by_level[ilevel].starts[
                                                                itarget_box + 1]
            workload[box_idx] += (end - start) * box_ntargets

        # add workload of list 3 near
        if tree.targets_have_extent and \
                traversal.from_sep_close_smaller_starts is not None:
            for itarget_box, box_idx in enumerate(traversal.target_boxes):
                box_ntargets = tree.box_target_counts_nonchild[box_idx]
                start = traversal.from_sep_close_smaller_starts[itarget_box]
                end = traversal.from_sep_close_smaller_starts[itarget_box + 1]
                particle_count = 0
                for near_box_id in traversal.from_sep_close_smaller_lists[start:end]:
                    particle_count += tree.box_source_counts_nonchild[near_box_id]
                workload[box_idx] += (
                    box_ntargets * particle_count * workload_weight.direct)

    # add workload of list 4
    for itarget_or_target_parent_boxes, box_idx in enumerate(
            traversal.target_or_target_parent_boxes):
        start = traversal.from_sep_bigger_starts[itarget_or_target_parent_boxes]
        end = traversal.from_sep_bigger_starts[itarget_or_target_parent_boxes + 1]
        particle_count = 0
        for far_box_id in traversal.from_sep_bigger_lists[start:end]:
            particle_count += tree.box_source_counts_nonchild[far_box_id]
        workload[box_idx] += particle_count * workload_weight.p2l

        if tree.targets_have_extent and \
                traversal.from_sep_close_bigger_starts is not None:
            box_ntargets = tree.box_target_counts_nonchild[box_idx]
            start = traversal.from_sep_close_bigger_starts[
                        itarget_or_target_parent_boxes]
            end = traversal.from_sep_close_bigger_starts[
                        itarget_or_target_parent_boxes + 1]
            particle_count = 0
            for direct_box_id in traversal.from_sep_close_bigger_lists[start:end]:
                particle_count += tree.box_source_counts_nonchild[direct_box_id]
            workload[box_idx] += (
                    box_ntargets * particle_count * workload_weight.direct)

    for i in range(tree.nboxes):
        # add workload of multipole calculation
        workload[i] += tree.box_source_counts_nonchild[i] * workload_weight.multipole

    total_workload = 0
    for i in range(tree.nboxes):
        total_workload += workload[i]

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
        box_idx = dfs_order[i]
        workload_count += workload[box_idx]
        if (workload_count > (rank + 1)*total_workload/total_rank or
                i == tree.nboxes - 1):
            responsible_boxes_list[rank] = dfs_order[start:i+1]
            start = i + 1
            rank += 1

    return responsible_boxes_list
