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

import time
import logging

logger = logging.getLogger(__name__)


def generate_local_travs(
        queue, local_tree, traversal_builder, merge_close_lists=False):
    """Generate local traversal from local tree.

    :arg queue: a :class:`pyopencl.CommandQueue` object.
    :arg local_tree: the local tree of class
        `boxtree.tools.ImmutableHostDeviceArray` on which the local traversal
        object will be constructed.
    :arg traversal_builder: a function, taken a :class:`pyopencl.CommandQueue` and
        a tree, returns the traversal object based on the tree.

    :return: generated local traversal object in device memory
    """
    start_time = time.time()

    local_tree.with_queue(queue)

    # We need `source_boxes_mask` and `source_parent_boxes_mask` here to restrict the
    # multipole formation and upward propagation within the rank's responsible boxes
    # region. Had there not been such restrictions, some sources might be distributed
    # to more than 1 rank and counted multiple times.
    local_trav, _ = traversal_builder(
        queue, local_tree.to_device(queue),
        source_boxes_mask=local_tree.responsible_boxes_mask.device,
        source_parent_boxes_mask=local_tree.ancestor_mask.device
    )

    if merge_close_lists and local_tree.targets_have_extent:
        local_trav = local_trav.merge_close_lists(queue)

    logger.info("Generate local traversal in {} sec.".format(
        str(time.time() - start_time))
    )

    return local_trav
