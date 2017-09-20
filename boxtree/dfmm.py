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

import numpy as np
import hpx

@hpx.create_action()
def main(sources, num_particles_per_block):
    
    # {{{ Distribute source array
    
    num_particles = sources.shape[0]
    import math
    num_block = math.ceil(num_particles / num_particles_per_block)
    d_sources = hpx.GlobalMemory.alloc_cyclic(num_block, 
        (num_particles_per_block, 2), sources.dtype)
    finished_copy = hpx.And(num_block)
    for i in range(num_block):
        d_sources[i].set(sources[i*num_particles_per_block : (i+1)*num_particles_per_block],
                         sync='async', rsync_lco=finished_copy)
    finished_copy.wait()

    # }}}

    # WIP: this is a placeholder
    potentials = np.empty((num_particles,), dtype=float) 
    hpx.exit(array=potentials)

def ddrive_fmm(traversal, expansion_wrangler, src_weights, num_particles_per_block=10000, 
               hpx_options=[]):
    """Distributed implementation of top-level driver routine for a fast 
    multipole calculation.

    :arg traversal: A :class:`boxtree.traversal.FMMTraversalInfo` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.
    :arg hpx_options: Options for HPX runtime. Pass directly to hpx.init.

    Returns the potentials computed by *expansion_wrangler*.
    """
    wrangler = expansion_wrangler
    logger.info("start fmm")
    
    logger.debug("reorder source weights")
    src_weights = wrangler.reorder_sources(src_weights)

    logger.debug("start hpx runtime")
    hpx.init(argv=hpx_options)

    # launch the main action
    sources = np.stack([wrangler.tree.sources[0], wrangler.tree.sources[1]], 
                       axis=-1)
    num_particles = sources.shape[0]
    potentials = hpx.run(main, sources, num_particles_per_block, 
                         shape=(num_particles,), dtype=float)

    logger.debug("finalize hpx runtime")
    hpx.finalize()

    return potentials
    
