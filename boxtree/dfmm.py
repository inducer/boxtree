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
import pyopencl as cl
from mako.template import Template
from pyopencl.tools import dtype_to_ctype
from pyopencl.scan import ExclusiveScanKernel


def partition_work(tree, total_rank, queue):
    """ This function returns a list of total_rank elements, where element i is a
    pyopencl array of indices of process i's responsible boxes. 
    """
    responsible_boxes = []
    num_boxes = tree.box_source_starts.shape[0]
    num_boxes_per_rank = num_boxes // total_rank
    extra_boxes = num_boxes - num_boxes_per_rank * total_rank
    start_idx = 0
    
    for current_rank in range(extra_boxes):
        end_idx = start_idx + num_boxes_per_rank + 1
        responsible_boxes.append(cl.array.arange(queue, start_idx, end_idx, 
                                                 dtype=tree.box_id_dtype))
        start_idx = end_idx

    for current_rank in range(extra_boxes, num_boxes):
        end_idx = start_idx + num_boxes_per_rank
        responsible_boxes.append(cl.array.arange(queue, start_idx, end_idx, 
                                                 dtype=tree.box_id_dtype))
        start_idx = end_idx

    return responsible_boxes


gen_local_tree_tpl = Template(r"""
typedef ${dtype_to_ctype(tree.box_id_dtype)} box_id_t;
typedef ${dtype_to_ctype(tree.particle_id_dtype)} particle_id_t;
typedef ${dtype_to_ctype(mask_dtype)} mask_t;
typedef ${dtype_to_ctype(tree.coord_dtype)} coord_t;

__kernel void generate_particle_mask(
    __global const box_id_t *res_boxes, 
    __global const particle_id_t *box_particle_starts,
    __global const particle_id_t *box_particle_counts_nonchild,
    const int total_num_res_boxes,
    __global mask_t *particle_mask) 
{
    /* 
     * generate_particle_mask takes the responsible box indices as input and generate 
     * a mask for responsible particles.
     */
    int res_box_idx = get_global_id(0);
    
    if(res_box_idx < total_num_res_boxes) {
        box_id_t cur_box = res_boxes[res_box_idx];
        for(particle_id_t i = box_particle_starts[cur_box];
            i < box_particle_starts[cur_box] + box_particle_counts_nonchild[cur_box];
            i++) {
            particle_mask[i] = 1;
        }
    }
}

__kernel void generate_local_particles(
    const int total_num_particles,
    % for dim in range(ndims):
        __global const coord_t *particles_${dim},
    % endfor
    __global const mask_t *particle_mask,
    __global const mask_t *particle_scan
    % for dim in range(ndims):
        , __global coord_t *local_particles_${dim}
    % endfor
)
{
    /* 
     * generate_local_particles generates an array of particles for which a process 
     * is responsible for.
     */
    int particle_idx = get_global_id(0);

    if(particle_idx < total_num_particles && particle_mask[particle_idx])
    {
        particle_id_t des = particle_scan[particle_idx];
        % for dim in range(ndims):
            local_particles_${dim}[des] = particles_${dim}[particle_idx];
        % endfor
    }
}
""", strict_undefined=True)

def drive_dfmm(traversal, expansion_wrangler, src_weights, comm=MPI.COMM_WORLD):
    
    # Get MPI information
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    if current_rank == 0:
        tree = traversal.tree
        ndims = tree.sources.shape[0]
        nboxes = tree.box_source_starts.shape[0]

        # Partition the work across all ranks by allocating responsible boxes
        responsible_boxes = partition_work(tree, total_rank, queue)

        # Compile the program
        mask_dtype = tree.particle_id_dtype
        gen_local_tree_prg = cl.Program(ctx, gen_local_tree_tpl.render(
            tree=tree,
            dtype_to_ctype=dtype_to_ctype,
            mask_dtype=mask_dtype,
            ndims=ndims)).build()

        # Construct mask scan kernel
        mask_scan_knl = ExclusiveScanKernel(
            ctx, mask_dtype,
            scan_expr="a+b", neutral="0",
        )

        def gen_local_tree(rank, particles, nparticles,
                           box_particle_starts, 
                           box_particle_counts_nonchild,
                           box_particle_counts_cumul):
            """
            This helper function generates the sources/targets related fields for
            a local tree
            """
            # Put particle structures to device memory
            d_box_particle_starts = cl.array.to_device(queue, box_particle_starts)
            d_box_particle_counts_nonchild = cl.array.to_device(
                queue, box_particle_counts_nonchild)
            d_box_particle_counts_cumul = cl.array.to_device(queue,
                box_particle_counts_cumul)
            d_particles = np.empty((ndims,), dtype=object)
            for i in range(ndims):
                d_particles[i] = cl.array.to_device(queue, particles[i])

            # Generate the particle mask array
            d_particle_mask = cl.array.zeros(queue, (nparticles,), dtype=mask_dtype)
            num_responsible_boxes = responsible_boxes[rank].shape[0]
            gen_local_tree_prg.generate_particle_mask(
                queue, ((num_responsible_boxes + 127)//128,), (128,),
                responsible_boxes[rank].data,
                d_box_particle_starts.data,
                d_box_particle_counts_nonchild.data,
                np.int32(num_responsible_boxes),
                d_particle_mask.data,
                g_times_l=True)

            # Generate the scan of the particle mask array
            d_particle_scan = cl.array.empty(queue, (nparticles,), 
                                             dtype=tree.particle_id_dtype)
            mask_scan_knl(d_particle_mask, d_particle_scan)

            # Generate particles for rank's local tree
            local_nparticles = d_particle_scan[-1].get(queue) + 1
            d_local_particles = np.empty((ndims,), dtype=object)
            for i in range(ndims):
                d_local_particles[i] = cl.array.empty(queue, (local_nparticles,),
                                                      dtype=tree.coord_dtype)
            
            d_paticles_list = d_particles.tolist()
            for i in range(ndims):
                d_paticles_list[i] = d_paticles_list[i].data
            d_local_particles_list = d_local_particles.tolist()
            for i in range(ndims):
                d_local_particles_list[i] = d_local_particles_list[i].data

            gen_local_tree_prg.generate_local_particles(
                queue, ((nparticles + 127) // 128,), (128,),
                np.int32(nparticles),
                *d_paticles_list,
                d_particle_mask.data,
                d_particle_scan.data,
                *d_local_particles_list,
                g_times_l=True)

            # Generate "box_particle_starts" of the local tree
            l_box_particle_starts = cl.array.empty(queue, (nboxes,), 
                                                   dtype=tree.particle_id_dtype)
            generate_box_particle_starts = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${particle_id_t} *old_starts, 
                    __global ${scan_t} *particle_scan, 
                    __global ${particle_id_t} *new_starts
                """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
                            scan_t=dtype_to_ctype(mask_dtype)),
                "new_starts[i] = particle_scan[old_starts[i]]",
                name="generate_box_particle_starts"
            )

            generate_box_particle_starts(d_box_particle_starts, d_particle_scan, 
                                         l_box_particle_starts)

            # Generate "box_particle_counts_nonchild" of the local tree
            l_box_particle_counts_nonchild = cl.array.zeros(queue, (nboxes,), 
                dtype=tree.particle_id_dtype)
            
            generate_box_particle_counts_nonchild = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${box_id_t} *res_boxes,
                    __global ${particle_id_t} *old_counts_nonchild,
                    __global ${particle_id_t} *new_counts_nonchild
                """).render(box_id_t=dtype_to_ctype(tree.box_id_dtype),
                            particle_id_t=dtype_to_ctype(tree.particle_id_dtype)),
                "new_counts_nonchild[res_boxes[i]] = "
                "old_counts_nonchild[res_boxes[i]]",
                name="generate_box_particle_counts_nonchild"
            )
            
            generate_box_particle_counts_nonchild(responsible_boxes[rank], 
                                                  d_box_particle_counts_nonchild,
                                                  l_box_particle_counts_nonchild)

            # Generate "box_particle_counts_cumul"
            l_box_particle_counts_cumul = cl.array.empty(queue, (nboxes,),
                dtype=tree.particle_id_dtype)

            generate_box_particle_counts_cumul = cl.elementwise.ElementwiseKernel(
                queue.context,
                Template("""
                    __global ${particle_id_t} *old_counts_cumul,
                    __global ${particle_id_t} *old_starts,
                    __global ${particle_id_t} *new_counts_cumul,
                    __global ${mask_t} *particle_scan
                """).render(particle_id_t=dtype_to_ctype(tree.particle_id_dtype),
                            mask_t=dtype_to_ctype(mask_dtype)),
                "new_counts_cumul[i] = "
                "particle_scan[old_starts[i] + old_counts_cumul[i]] - "
                "particle_scan[old_starts[i]]",
                name="generate_box_particle_counts_cumul"
            )

            generate_box_particle_counts_cumul(d_box_particle_counts_cumul,
                                               d_box_particle_starts,
                                               l_box_particle_counts_cumul,
                                               d_particle_scan)

            return d_local_particles


        for rank in range(total_rank):
            d_local_sources = gen_local_tree(rank, tree.sources, tree.nsources,
                                             tree.box_source_starts,
                                             tree.box_source_counts_nonchild,
                                             tree.box_source_counts_cumul)
            d_local_targets = gen_local_tree(rank, tree.targets, tree.ntargets,
                                             tree.box_target_starts,
                                             tree.box_target_counts_nonchild,
                                             tree.box_source_counts_cumul)
