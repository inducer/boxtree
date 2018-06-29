from __future__ import division

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
from pyopencl.tools import dtype_to_ctype
import pyopencl as cl
from mako.template import Template

import logging
logger = logging.getLogger(__name__)


def generate_local_travs(
        queue, local_tree, box_bounding_box=None,
        well_sep_is_n_away=1, from_sep_smaller_crit=None,
        merge_close_lists=False):

    start_time = time.time()

    d_tree = local_tree.to_device(queue).with_queue(queue)

    # Modify box flags for targets
    from boxtree import box_flags_enum
    box_flag_t = dtype_to_ctype(box_flags_enum.dtype)
    modify_target_flags_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${particle_id_t} *box_target_counts_nonchild,
            __global ${particle_id_t} *box_target_counts_cumul,
            __global ${box_flag_t} *box_flags
        """).render(particle_id_t=dtype_to_ctype(local_tree.particle_id_dtype),
                    box_flag_t=box_flag_t),
        Template("""
            box_flags[i] &= (~${HAS_OWN_TARGETS});
            box_flags[i] &= (~${HAS_CHILD_TARGETS});
            if(box_target_counts_nonchild[i]) box_flags[i] |= ${HAS_OWN_TARGETS};
            if(box_target_counts_nonchild[i] < box_target_counts_cumul[i])
                box_flags[i] |= ${HAS_CHILD_TARGETS};
        """).render(HAS_OWN_TARGETS=("(" + box_flag_t + ") " +
                                     str(box_flags_enum.HAS_OWN_TARGETS)),
                    HAS_CHILD_TARGETS=("(" + box_flag_t + ") " +
                                       str(box_flags_enum.HAS_CHILD_TARGETS)))
    )

    modify_target_flags_knl(d_tree.box_target_counts_nonchild,
                            d_tree.box_target_counts_cumul,
                            d_tree.box_flags)

    # Generate local source flags
    local_box_flags = d_tree.box_flags & 250
    modify_own_sources_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global ${box_id_t} *responsible_box_list,
            __global ${box_flag_t} *box_flags
        """).render(box_id_t=dtype_to_ctype(local_tree.box_id_dtype),
                    box_flag_t=box_flag_t),
        Template(r"""
            box_flags[responsible_box_list[i]] |= ${HAS_OWN_SOURCES};
        """).render(HAS_OWN_SOURCES=("(" + box_flag_t + ") " +
                                     str(box_flags_enum.HAS_OWN_SOURCES)))
        )

    modify_child_sources_knl = cl.elementwise.ElementwiseKernel(
        queue.context,
        Template("""
            __global char *ancestor_box_mask,
            __global ${box_flag_t} *box_flags
        """).render(box_flag_t=box_flag_t),
        Template("""
            if(ancestor_box_mask[i]) box_flags[i] |= ${HAS_CHILD_SOURCES};
        """).render(HAS_CHILD_SOURCES=("(" + box_flag_t + ") " +
                                       str(box_flags_enum.HAS_CHILD_SOURCES)))
    )

    modify_own_sources_knl(d_tree.responsible_boxes_list, local_box_flags)
    modify_child_sources_knl(d_tree.ancestor_mask, local_box_flags)

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(
        queue.context,
        well_sep_is_n_away=well_sep_is_n_away,
        from_sep_smaller_crit=from_sep_smaller_crit
    )

    d_local_trav, _ = tg(
        queue, d_tree, debug=True,
        box_bounding_box=box_bounding_box,
        local_box_flags=local_box_flags
    )

    if merge_close_lists and d_tree.targets_have_extent:
        d_local_trav = d_local_trav.merge_close_lists(queue)

    local_trav = d_local_trav.get(queue=queue)

    logger.info("Generate local traversal in {} sec.".format(
        str(time.time() - start_time))
    )

    return local_trav
