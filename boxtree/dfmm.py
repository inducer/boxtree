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

    # {{{ Distribute problem parameters

    if current_rank == 0:
        tree = traversal.tree
        parameters = {"nsources":tree.nsources, 
                      "dimensions":tree.sources.shape[0], 
                      "coord_dtype":tree.coord_dtype}
    else:
        parameters = None
    parameters = comm.bcast(parameters, root=0)
    
    # }}}

    # {{{ Distribute source particles
    
    num_sources_per_rank = (parameters["nsources"] + total_rank - 1) // total_rank
    sources = []

    for i in range(parameters["dimensions"]):
        # Prepare send buffer
        if current_rank == 0:
            sendbuf = np.empty((num_sources_per_rank * total_rank,), 
                               dtype=parameters['coord_dtype'])
            sendbuf[:parameters["nsources"]] = tree.sources[i]
        else:
            sendbuf = None

        # Prepare receive buffer
        recvbuf = np.empty((num_sources_per_rank,), dtype=parameters['coord_dtype'])

        # Scatter send buffer
        comm.Scatter(sendbuf, recvbuf, root=0)

        # Trim the receive buffer for the last rank
        if current_rank == total_rank - 1:
            num_sources_current_rank = parameters["nsources"] - \
                num_sources_per_rank * (total_rank - 1)
            sources.append(recvbuf[:num_sources_current_rank])
        else:
            sources.append(recvbuf)
    
    # }}}
