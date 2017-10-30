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
    
    # {{{ Get MPI information

    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # }}}

    # {{{ Broadcast traversal object without particles

    if current_rank == 0:
        local_traversal = traversal.copy()
        local_tree = local_traversal.tree
        local_tree.sources = None
        if local_tree.sources_have_extent == True:
            local_tree.source_radii = None
        local_tree.targets = None
        if local_tree.targets_have_extent == True:
            local_tree.target_radii = None
        local_tree.user_source_ids = None
        local_tree.sorted_target_ids = None
    else:
        local_traversal = None

    comm.bcast(local_traversal, root=0)

    # }}}

    # {{{ Generate an array which contains responsible box indices

    num_boxes = local_traversal.tree.box_source_starts.shape[0]
    num_responsible_boxes_per_rank = (num_boxes + total_rank - 1) // total_rank
    if current_rank == total_rank - 1:
        responsible_boxes = np.arange(num_responsible_boxes_per_rank * current_rank,
                                      num_boxes, dtype=box_id_dtype)
    else:
        responsible_boxes = np.arange(num_responsible_boxes_per_rank * current_rank,
            num_responsible_boxes_per_rank * (current_rank + 1), dtype=box_id_dtype)

    # }}}
