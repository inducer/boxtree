from __future__ import division

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

from mpi4py import MPI
import numpy as np

def drive_dfmm(traversal, expansion_wrangler, src_weights):
    
    #  {{{ Get MPI information

    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # }}}

    # {{{ Distribute tree parameters

    if current_rank == 0:
        tree = traversal.tree
        # TODO: distribute more parameters of the tree
        parameters = {"sources_are_targets": tree.sources_are_targets,
                      "sources_have_extent": tree.sources_have_extent,
                      "nsources":tree.nsources,
                      "nboxes":tree.box_source_starts.shape[0], 
                      "dimensions":tree.sources.shape[0], 
                      "coord_dtype":tree.coord_dtype,
                      "box_id_dtype":tree.box_id_dtype}
    else:
        parameters = None
    parameters = comm.bcast(parameters, root=0)
    
    # }}}

    # {{{ Fill tree parameters to the locally essentail tree

    from boxtree import Tree
    letree = Tree()
    # TODO: add more parameters to the locally essential tree
    letree.sources_are_targets = parameters["sources_are_targets"]

    # }}}

    # {{{ Construct locally essential tree mask for each rank

    # Problem: Current implementation divides all boxes with targets evenly across all 
    # ranks. This scheme is subject to significant load imbalance. A better way to do 
    # this is to assign a weight to each box according to its interaction list, and then 
    # divides boxes evenly by the total weights.

    if current_rank == 0:
        # mask[i][j] is true iff box j is in the locally essential tree of rank i
        mask = np.zeros((total_rank, parameters["nboxes"]), dtype=bool)
        target_boxes = traversal.target_boxes
        num_boxes_per_rank = (len(target_boxes) + total_rank - 1) // total_rank
        
        for i in range(total_rank):
            # Get the start and end box index for rank i 
            box_start_idx = num_boxes_per_rank * i
            if current_rank == total_rank - 1:
                box_end_idx = len(target_boxes)
            else:
                box_end_idx = num_boxes_per_rank * (i + 1)

            # Mark all ancestors of boxes of rank i
            new_mask = np.zeros(parameters["nboxes"], dtype=bool)
            new_mask[target_boxes[box_start_idx:box_end_idx]] = True
            while np.count_nonzero(new_mask) != 0:
                np.logical_or(mask[i, :], new_mask, out=mask[i, :])
                new_mask_idx = np.nonzero(new_mask)
                new_mask_parent_idx = tree.box_parent_ids[new_mask_idx]
                new_mask[:] = False
                new_mask[new_mask_parent_idx] = True
                new_mask = np.logical_and(new_mask, np.logical_not(mask[i, :]), 
                                          out=new_mask)

    # }}}
