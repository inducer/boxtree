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
from pyopencl.scan import GenericScanKernel


def partition_work(tree, total_rank, queue):
    # This function returns a list of total_rank elements, where element i is a
    # pyopencl array of indices of process i's responsible boxes. 
    responsible_boxes = []
    num_boxes = tree.box_source_starts.shape[0]
    num_boxes_per_rank = (num_boxes + total_rank - 1) // total_rank
    for current_rank in range(total_rank):
        if current_rank == total_rank - 1:
            responsible_boxes.append(cl.array.arange(
                queue,
                num_boxes_per_rank * current_rank,
                num_boxes, 
                dtype=tree.box_id_dtype))
        else:
            responsible_boxes.append(cl.array.arange(
                queue,
                num_boxes_per_rank * current_rank,
                num_boxes_per_rank * (current_rank + 1), 
                dtype=tree.box_id_dtype))
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
    /* generate_particle_mask takes the responsible box indices as input and generate 
     * a mask for responsible particles.
     */
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    int num_res_boxes = (total_num_res_boxes + gsize - 1) / gsize;
    box_id_t res_boxes_start = num_res_boxes * gid;
    
    for(box_id_t cur_box = res_boxes_start;
        cur_box < res_boxes_start + num_res_boxes && cur_box < total_num_res_boxes;
        cur_box ++) 
    {
        for(particle_id_t i = box_particle_starts[cur_box];
            i < box_particle_starts[cur_box] + box_particle_counts_nonchild[cur_box];
            i++) 
        {
            particle_mask[i] = 1;
        }
    }
}

__kernel void generate_local_particles(
    const int total_num_particles,
    const int total_num_local_particles,
    __global const coord_t *particles,
    __global const mask_t *particle_mask,
    __global const mask_t *particle_scan,
    __global coord_t *local_particles)
{
    /* generate_local_particles generates an array of particles for which a process 
     * is responsible for.
     */
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    int num_particles = (total_num_particles + gsize - 1) / gsize;
    particle_id_t start = num_particles * gid;

    for(particle_id_t i = start;
        i < start + num_particles && i < total_num_particles;
        i++) 
    {
        if(particle_mask[i]) 
        {
            particle_id_t des = particle_scan[i];
            % for dim in range(ndims):
                local_particles[total_num_local_particles * ${dim} + des]
                = particles[total_num_particles * ${dim} + i];
            % endfor
        }
    }
}

""", strict_undefined=True)


def drive_dfmm(traversal, expansion_wrangler, src_weights):
    
    # {{{ Get MPI information

    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    # }}}
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    if current_rank == 0:
        tree = traversal.tree
        ndims = tree.sources.shape[0]

        # Partition the work across all ranks by allocating responsible boxes
        responsible_boxes = partition_work(tree, total_rank, queue)

        # Convert tree structures to device memory
        d_box_source_starts = cl.array.to_device(queue, tree.box_source_starts)
        d_box_source_counts_nonchild = cl.array.to_device(queue, 
            tree.box_source_counts_nonchild)

        # Compile the program
        mask_dtype = tree.particle_id_dtype
        gen_local_tree_prg = cl.Program(ctx, gen_local_tree_tpl.render(
            tree=tree,
            dtype_to_ctype=dtype_to_ctype,
            mask_dtype=mask_dtype,
            ndims=ndims)).build()

        # Construct mask scan kernel
        arg_tpl = Template(r"__global ${mask_t} *ary, __global ${mask_t} *out")
        mask_scan_knl = GenericScanKernel(
            ctx, mask_dtype,
            arguments=arg_tpl.render(mask_t=dtype_to_ctype(mask_dtype)),
            input_expr="ary[i]",
            scan_expr="a+b", neutral="0",
            output_statement="out[i] = item;")

        for rank in range(total_rank):
            # Generate the particle mask array
            d_source_mask = cl.array.zeros(queue, (tree.nsources,), 
                                           dtype=mask_dtype)
            gen_local_tree_prg.generate_particle_mask(
                queue, 
                (2048,),
                None,
                responsible_boxes[rank].data,
                d_box_source_starts.data,
                d_box_source_counts_nonchild.data,
                np.int32(responsible_boxes[rank].shape[0]),
                d_source_mask.data
                )

            # Generate the scan of the particle mask array
            d_source_scan = cl.array.empty(queue, (tree.nsources,),
                                           dtype=tree.particle_id_dtype)
            mask_scan_knl(d_source_mask, d_source_scan)

            local_nsources = d_source_scan[-1].get(queue)
            local_sources = cl.array.empty(queue, (ndims, local_nsources),
                                           dtype=tree.coord_dtype)
