__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from functools import partial
import numpy as np
from pyopencl.elementwise import ElementwiseTemplate
from pyopencl.scan import ScanTemplate
from mako.template import Template
from pytools import Record, memoize, log_process
from boxtree.tools import (get_type_moniker, get_coord_vec_dtype,
        coord_vec_subscript_code)

import logging
logger = logging.getLogger(__name__)


# TODO:
# - Add *restrict where applicable.
# - Split up the arrays so that there is one array per box level. This avoids
#   having to reallocate the middle of an array.
# - Use level-relative box numbering in parent_box_ids, child_box_ids. This
#   avoids having to renumber these arrays after reallocation.

# -----------------------------------------------------------------------------
# CONTROL FLOW
# ------------
#
# Since this file mostly fills in the blanks in the tree build
# implementation, control flow here can be a bit hard to see.
#
# - Everything starts and ends in the driver in tree_build.py
#
# - The first thing that happens is that data types get built and
#   kernels get compiled. Most of the file consists of type and
#   code generators for these kernels.
#
# - We start with a reduction that determines the bounding box of all
#   particles.
#
# - The level loop is in the driver, which alternates between scans and local
#   post processing, according to the algorithm described below.
#
# - Once the level loop finishes, a "box info" kernel is run
#   that extracts flags for each box.
#
# - As a last step, empty leaf boxes are eliminated. This is done by a
#   scan kernel that computes indices, and by an elementwise kernel
#   that compresses arrays and maps them to new box IDs, if applicable.
#
# -----------------------------------------------------------------------------
#
# HOW DOES THE LEVEL LOOP WORK?
# -----------------------------
#
# This code sorts particles into an nD-tree of boxes.  It does this by doing two
# successive (parallel) scans and a postprocessing step.
#
# The following information is being pushed around by the scans, which
# proceed over particles:
#
# - a cumulative count ("pcnt") and weight ("pwt") of particles in each subbox
#   ("morton_nr") , should the current box need to be subdivided.
#
# - the "split_box_id". This is an array that, for each box, answers the
#   question, "After I am subdivided, what is end of the range of boxes
#   that my particles get pushed into?" The split_box_id is not meaningful
#   unless the box is about to be subdivided.
#
# Using this data, the stages of the algorithm proceed as follows:
#
# 1. Count the number of particles in each subbox. This stage uses a segmented
#    (per-box) scan to fill "pcnt" and "pwt". This information is kept
#    per-particle ("morton_bin_counts") and per-box ("box_morton_bin_counts").
#
# 2. Using a scan over the boxes, segmented by level, make a decision whether to
#    refine each box, and compute the split_box_id. This stage also computes the
#    total number of new boxes needed. If a box knows it needs to be subdivided,
#    it asks for 2**d new boxes at the next level.
#
# 3. Realize the splitting determined in #2. This part consists of splitting the
#    boxes (done in the "box splitter kernel") and renumbering the particles so
#    that particles in the same box have are numbered contiguously (done in the
#    "particle renumberer kernel").
#
# HOW DOES LEVEL RESTRICTION WORK?
# --------------------------------
#
# This requires some post-processing in the level loop described above: as an
# additional step, the "level restrict" kernel gets run at the end of the level
# loop. The job of the level restrict kernel is to mark boxes on higher levels
# to be split based on looking at the levels of their neighbor boxes. The
# splitting is then realized by the next iteration of the level loop,
# simultaneously with the creation of the next level.
#
# -----------------------------------------------------------------------------


class _KernelInfo(Record):
    pass


# {{{ data types

refine_weight_dtype = np.dtype(np.int32)


@memoize(use_kwargs=True)
def make_morton_bin_count_type(device, dimensions, particle_id_dtype,
        srcntgts_have_extent):
    fields = []

    # Non-child srcntgts are sorted *before* all the child srcntgts.
    if srcntgts_have_extent:
        fields.append(("nonchild_srcntgts", particle_id_dtype))

    from boxtree.tools import padded_bin
    for mnr in range(2**dimensions):
        fields.append(("pcnt%s" % padded_bin(mnr, dimensions), particle_id_dtype))
    # Morton bin weight totals
    for mnr in range(2**dimensions):
        fields.append(("pwt%s" % padded_bin(mnr, dimensions), refine_weight_dtype))

    dtype = np.dtype(fields)

    name_suffix = ""
    if srcntgts_have_extent:
        name_suffix = "_ext"

    name = "boxtree_morton_bin_count_%dd_p%s%s_t" % (
            dimensions,
            get_type_moniker(particle_id_dtype),
            name_suffix)

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

# }}}


# {{{ preamble

TYPE_DECL_PREAMBLE_TPL = Template(r"""//CL//
    typedef ${dtype_to_ctype(morton_bin_count_dtype)} morton_counts_t;
    typedef morton_counts_t scan_t;
    typedef ${dtype_to_ctype(refine_weight_dtype)} refine_weight_t;
    typedef ${dtype_to_ctype(bbox_dtype)} bbox_t;
    typedef ${dtype_to_ctype(coord_dtype)} coord_t;
    typedef ${dtype_to_ctype(coord_vec_dtype)} coord_vec_t;
    typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
    typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;
    typedef ${dtype_to_ctype(box_level_dtype)} box_level_t;

    // morton_nr == -1 is defined to mean that the srcntgt is
    // remaining at the present level and will not be sorted
    // into a child box.
    typedef ${dtype_to_ctype(morton_nr_dtype)} morton_nr_t;
    """, strict_undefined=True)

GENERIC_PREAMBLE_TPL = Template(r"""//CL//

    // Use this as dbg_printf(("oh snap: %d\n", stuff)); Note the double
    // parentheses.
    //
    // Watch out: 64-bit values on Intel CL must be printed with %ld, or
    // subsequent values will print as 0. Things may crash. And you'll be very
    // confused.

    %if enable_printf:
        #define dbg_printf(ARGS) printf ARGS
    %else:
        #define dbg_printf(ARGS) /* */
    %endif

    %if enable_assert:
        #define dbg_assert(CONDITION) \
            { \
                if (!(CONDITION)) \
                    printf("*** ASSERTION VIOLATED: %s, line %d: %s\n", \
                        __func__, __LINE__, #CONDITION); \
            }
    %else:
        #define dbg_assert(CONDITION) ((void) 0)
    %endif

""", strict_undefined=True)

# }}}

# BEGIN KERNELS IN THE LEVEL LOOP

# {{{ morton scan

MORTON_NR_SCAN_PREAMBLE_TPL = Template(r"""//CL//

    // {{{ neutral element

    scan_t scan_t_neutral()
    {
        scan_t result;

        %if srcntgts_have_extent:
            result.nonchild_srcntgts = 0;
        %endif

        %for mnr in range(2**dimensions):
            result.pcnt${padded_bin(mnr, dimensions)} = 0;
        %endfor
        %for mnr in range(2**dimensions):
            result.pwt${padded_bin(mnr, dimensions)} = 0;
        %endfor
        return result;
    }

    // }}}

    inline int my_add_sat(int a, int b)
    {
        long result = (long) a + b;
        return (result > INT_MAX) ? INT_MAX : result;
    }

    // {{{ scan 'add' operation
    scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary)
    {
        if (!across_seg_boundary)
        {
            %if srcntgts_have_extent:
                b.nonchild_srcntgts += a.nonchild_srcntgts;
            %endif

            %for mnr in range(2**dimensions):
                <% field = "pcnt"+padded_bin(mnr, dimensions) %>
                b.${field} = a.${field} + b.${field};
            %endfor
            %for mnr in range(2**dimensions):
                <% field = "pwt"+padded_bin(mnr, dimensions) %>
                // XXX: The use of add_sat() seems to be causing trouble
                // with multiple compilers. For d=3:
                // 1. POCL will miscompile and either give wrong
                //    results or crash.
                // 2. Intel will use a large amount of memory.
                // Versions tested: POCL 0.13, Intel OpenCL 16.1
                b.${field} = my_add_sat(a.${field}, b.${field});
            %endfor
        }

        return b;
    }

    // }}}

    // {{{ scan data type init from particle

    scan_t scan_t_from_particle(
        const int i,
        const int particle_level,
        bbox_t const *bbox,
        global morton_nr_t *morton_nrs, // output/side effect
        global particle_id_t *user_srcntgt_ids,
        global refine_weight_t *refine_weights
        %for ax in axis_names:
            , global const coord_t *${ax}
        %endfor
        %if srcntgts_have_extent:
            , global const coord_t *srcntgt_radii
            , const coord_t stick_out_factor
        %endif
    )
    {
        particle_id_t user_srcntgt_id = user_srcntgt_ids[i];

        // The next level is 1 + the current level of the particle.
        // This should be 0.5 when next level = 1. (Level 0 is the root.)
        coord_t next_level_box_size_factor =
            ((coord_t) 1) / ((coord_t) (1U << (1 + particle_level)));

        %if srcntgts_have_extent:
            bool stop_srcntgt_descent = false;
            coord_t srcntgt_radius = srcntgt_radii[user_srcntgt_id];
        %endif

        %if not srcntgts_have_extent:
            // This argument is only supplied with srcntgts_have_extent.
            #define stick_out_factor 0.
        %endif

        const coord_t one_half = ((coord_t) 1) / 2;
        const coord_t box_radius_factor =
            // AMD CPU seems to like to miscompile this--change with care.
            // (last seen on 13.4-2)
            (1. + stick_out_factor)
            * one_half; // convert diameter to radius

        %if not srcntgts_have_extent:
            #undef stick_out_factor
        %endif

        %for ax in axis_names:
            // Most FMMs are isotropic, i.e. global_extent_{x,y,z} are all the same.
            // Nonetheless, the gain from exploiting this assumption seems so
            // minimal that doing so here didn't seem worthwhile in the
            // srcntgts_extent_norm == "linf" case.

            coord_t global_min_${ax} = bbox->min_${ax};
            coord_t global_extent_${ax} = bbox->max_${ax} - global_min_${ax};
            coord_t srcntgt_${ax} = ${ax}[user_srcntgt_id];

            // Note that the upper bound of the global bounding box is computed
            // to be slightly larger than the highest found coordinate, so that
            // 1.0 is never reached as a scaled coordinate at the highest
            // level, and it isn't either by the fact that boxes are
            // [)-half-open in subsequent levels.

            // So (1 << (1 + particle_level)) is 2 when building level 1.
            // Because the floating point factor is strictly less than 1, 2 is
            // never reached, so when building level 1, the result is either
            // 0 or 1.
            // After that, we just add one (less significant) bit per level.

            unsigned ${ax}_bits = (unsigned) (
                ((srcntgt_${ax} - global_min_${ax}) / global_extent_${ax})
                * (1U << (1 + particle_level)));

            // Need to compute center to compare excess with stick_out_factor.
            // Unused if no stickout, relying on compiler to eliminate this.
            const coord_t next_level_box_center_${ax} =
                global_min_${ax}
                + global_extent_${ax}
                * (${ax}_bits + one_half)
                * next_level_box_size_factor;

        %endfor

        %if srcntgts_extent_norm == "linf":
            %for ax in axis_names:
                const coord_t next_level_box_stick_out_radius_${ax} =
                    box_radius_factor
                    * global_extent_${ax}
                    * next_level_box_size_factor;

                // stop descent here if particle sticks out of next-level box
                stop_srcntgt_descent = stop_srcntgt_descent ||
                    (srcntgt_${ax} + srcntgt_radius >=
                        next_level_box_center_${ax}
                        + next_level_box_stick_out_radius_${ax});
                stop_srcntgt_descent = stop_srcntgt_descent ||
                    (srcntgt_${ax} - srcntgt_radius <
                        next_level_box_center_${ax}
                        - next_level_box_stick_out_radius_${ax});
            %endfor

        %elif srcntgts_extent_norm == "l2":

            coord_t next_level_box_stick_out_radius =
                box_radius_factor
                * global_extent_x  /* assume isotropy */
                * next_level_box_size_factor;

            coord_t next_level_box_center_to_srcntgt_bdry_l2_dist =
                sqrt(
                %for ax in axis_names:
                    +   (srcntgt_${ax} - next_level_box_center_${ax})
                      * (srcntgt_${ax} - next_level_box_center_${ax})
                %endfor
                ) + srcntgt_radius;

            // stop descent here if particle sticks out of next-level box
            stop_srcntgt_descent = stop_srcntgt_descent ||
                (
                next_level_box_center_to_srcntgt_bdry_l2_dist
                * next_level_box_center_to_srcntgt_bdry_l2_dist
                    >= ${dimensions}
                        * next_level_box_stick_out_radius
                        * next_level_box_stick_out_radius);

        %elif srcntgts_extent_norm is None:
            // nothing to do

        %else:
            <%
                raise ValueError("unexpected value of 'srcntgts_extent_norm': %s"
                    % srcntgts_extent_norm)
            %>
        %endif

        // Pick off the lowest-order bit for each axis, put it in its place.
        int level_morton_number = 0
        %for iax, ax in enumerate(axis_names):
            | (${ax}_bits & 1U) << (${dimensions-1-iax})
        %endfor
            ;

        %if srcntgts_have_extent:
            if (stop_srcntgt_descent)
            {
                level_morton_number = -1;
            }
        %endif

        scan_t result;
        %if srcntgts_have_extent:
            result.nonchild_srcntgts = (level_morton_number == -1);
        %endif
        %for mnr in range(2**dimensions):
            <% field = "pcnt"+padded_bin(mnr, dimensions) %>
            result.${field} = (level_morton_number == ${mnr});
        %endfor
        %for mnr in range(2**dimensions):
            <% field = "pwt"+padded_bin(mnr, dimensions) %>
            result.${field} = (level_morton_number == ${mnr}) ?
                    refine_weights[user_srcntgt_id] : 0;
        %endfor
        morton_nrs[i] = level_morton_number;

        return result;
    }

    // }}}

""", strict_undefined=True)

# }}}

# {{{ morton scan output

MORTON_NR_SCAN_OUTPUT_STMT_TPL = Template(r"""//CL//
    {
        particle_id_t my_id_in_my_box = -1
        %if srcntgts_have_extent:
            + item.nonchild_srcntgts
        %endif
        %for mnr in range(2**dimensions):
            + item.pcnt${padded_bin(mnr, dimensions)}
        %endfor
            ;
        dbg_printf(("my_id_in_my_box:%d\n", my_id_in_my_box));
        morton_bin_counts[i] = item;

        box_id_t current_box_id = srcntgt_box_ids[i];

        particle_id_t box_srcntgt_count = box_srcntgt_counts_cumul[current_box_id];

        // Am I the last particle in my current box?
        // If so, populate particle count.

        if (my_id_in_my_box+1 == box_srcntgt_count)
        {
            dbg_printf(("store box %d cbi:%d\n", i, current_box_id));
            dbg_printf(("   store_sums: %d %d %d %d\n",
                item.pcnt00, item.pcnt01, item.pcnt10, item.pcnt11));
            box_morton_bin_counts[current_box_id] = item;
        }
    }
""", strict_undefined=True)

# }}}

# {{{ split box id scan

SPLIT_BOX_ID_SCAN_TPL = ScanTemplate(
    name_prefix="split_box_id_scan",
    arguments=r"""//CL:mako//
        /* input */
        box_id_t *srcntgt_box_ids,
        particle_id_t *box_srcntgt_counts_cumul,
        morton_counts_t *box_morton_bin_counts,
        refine_weight_t *refine_weights,
        refine_weight_t max_leaf_refine_weight,
        box_level_t *box_levels,
        box_id_t *level_start_box_ids,
        box_id_t *level_used_box_counts,
        int *box_force_split,
        box_level_t last_level,

        /* output */
        int *box_has_children,
        box_id_t *split_box_ids,
        int *have_oversize_split_box,
        """,
    preamble=r"""//CL:mako//
        scan_t count_new_boxes_needed(
            box_id_t box_id,
            box_level_t level,
            box_level_t last_level,
            refine_weight_t max_leaf_refine_weight,
            __global particle_id_t *box_srcntgt_counts_cumul,
            __global morton_counts_t *box_morton_bin_counts,
            __global box_id_t *level_start_box_ids,
            __global box_id_t *level_used_box_counts,
            %if level_restrict:
                __global int *box_force_split,
            %endif
            __global int *have_oversize_split_box, // output/side effect
            __global int *box_has_children // output/side effect
            )
        {
            scan_t result = 0;

            // First box at my level? Start counting at the number of boxes
            // used at the child level.
            if (box_id == level_start_box_ids[level])
            {
                result += level_start_box_ids[level + 1];
                result += level_used_box_counts[level + 1];
            }

            %if srcntgts_have_extent:
                const particle_id_t nonchild_srcntgts_in_box =
                    box_morton_bin_counts[box_id].nonchild_srcntgts;
            %else:
                const particle_id_t nonchild_srcntgts_in_box = 0;
            %endif

            // Get box refine weight.
            refine_weight_t box_refine_weight = 0;
            %for mnr in range(2**dimensions):
                box_refine_weight = add_sat(box_refine_weight,
                    box_morton_bin_counts[box_id].pwt${padded_bin(mnr, dimensions)});
            %endfor

            // Add 2**d to make enough room for a split of the current box

            if ((
                level + 1 == last_level
                &&
                %if adaptive:
                    /* box overfull? */
                    box_refine_weight
                        > max_leaf_refine_weight
                %else:
                    /* box non-empty? */
                    /* Note: Refine weights are allowed to be 0,
                       so check # of particles directly. */
                    box_srcntgt_counts_cumul[box_id] - nonchild_srcntgts_in_box
                        >= 0
                %endif
                )
                %if level_restrict:
                    || box_force_split[box_id]
                %endif
                )
            {
                result += ${2**dimensions};
                box_has_children[box_id] = 1;

                // Check if the box is oversized. This drives the level loop.
                refine_weight_t max_subbox_refine_weight = 0;
                %for mnr in range(2**dimensions):
                    max_subbox_refine_weight = max(max_subbox_refine_weight,
                        box_morton_bin_counts[box_id]
                        .pwt${padded_bin(mnr, dimensions)});
                %endfor
                if (max_subbox_refine_weight > max_leaf_refine_weight)
                {
                    *have_oversize_split_box = 1;
                }
            }

            return result;
        }
        """,
    input_expr=r"""//CL:mako//
            count_new_boxes_needed(
                i,
                box_levels[i],
                last_level,
                max_leaf_refine_weight,
                box_srcntgt_counts_cumul,
                box_morton_bin_counts,
                level_start_box_ids,
                level_used_box_counts,
                %if level_restrict:
                    box_force_split,
                %endif
                have_oversize_split_box,
                box_has_children
            )""",
    scan_expr="across_seg_boundary ? b : a + b",
    neutral="0",
    is_segment_start_expr="i == 0 || box_levels[i] != box_levels[i-1]",
    output_statement=r"""//CL//
        dbg_assert(item >= 0);

        split_box_ids[i] = item;

        """)

# }}}

# {{{ box splitter kernel

BOX_SPLITTER_KERNEL_TPL = Template(r"""//CL//
    box_id_t ibox = i;

    bool do_split_box =
       (box_has_children[ibox] && box_levels[ibox] + 1 == level)
       %if level_restrict:
           || box_force_split[ibox]
       %endif
       ;

    if (!do_split_box)
    {
        PYOPENCL_ELWISE_CONTINUE;
    }

    // {{{ Set up child box data structure.

    morton_counts_t box_morton_bin_count = box_morton_bin_counts[ibox];

    %for mnr in range(2**dimensions):
    {
        box_id_t new_box_id = split_box_ids[ibox] - ${2**dimensions} + ${mnr};

        // Parent / child / level info
        box_parent_ids[new_box_id] = ibox;
        box_child_ids_mnr_${mnr}[ibox] = new_box_id;
        box_level_t new_level = box_levels[ibox] + 1;
        box_levels[new_box_id] = new_level;

        // Box particle counts / starting particle number
        particle_id_t new_count =
            box_morton_bin_count.pcnt${padded_bin(mnr, dimensions)};
        box_srcntgt_counts_cumul[new_box_id] = new_count;

        // Only set the starting particle number / start flags if
        // the new box has particles to begin with.
        if (new_count > 0)
        {
            particle_id_t new_box_start = box_srcntgt_starts[ibox]
            %if srcntgts_have_extent:
                + box_morton_bin_count.nonchild_srcntgts
            %endif
            %for sub_mnr in range(mnr):
                + box_morton_bin_count.pcnt${padded_bin(sub_mnr, dimensions)}
            %endfor
            ;

            box_start_flags[new_box_start] = 1;
            box_srcntgt_starts[new_box_id] = new_box_start;
        }

        // Compute box center.
        coord_t radius = (root_extent * 1 / (coord_t) (1 << (1 + new_level)));

        %for idim, ax in enumerate(axis_names):
        {
            <% has_bit = mnr & 2**(dimensions-1-idim) %>
            box_centers_${ax}[new_box_id] = box_centers_${ax}[ibox]
                ${"+" if has_bit else "-"} radius;
        }
        %endfor
    }
    %endfor

    // }}}
""", strict_undefined=True)

# }}}

# {{{ post-split particle renumbering

PARTICLE_RENUMBERER_PREAMBLE_TPL = Template(r"""//CL//
    <%
      def get_count_for_branch(known_bits):
          if len(known_bits) == dimensions:
              return "counts.pcnt%s" % known_bits

          dim = len(known_bits)
          boundary_morton_nr = known_bits + "1" + (dimensions-dim-1)*"0"

          return ("((morton_nr < %s) ? %s : %s)" % (
              int(boundary_morton_nr, 2),
              get_count_for_branch(known_bits+"0"),
              get_count_for_branch(known_bits+"1")))
    %>

    particle_id_t get_count(morton_counts_t counts, int morton_nr)
    {
        %if srcntgts_have_extent:
            if (morton_nr == -1)
                return counts.nonchild_srcntgts;
        %endif
        return ${get_count_for_branch("")};
    }

""", strict_undefined=True)


PARTICLE_RENUMBERER_KERNEL_TPL = Template(r"""//CL//
    box_id_t ibox = srcntgt_box_ids[i];
    dbg_assert(ibox >= 0);

    dbg_printf(("postproc %d:\n", i));
    dbg_printf(("   my box id: %d\n", ibox));

    bool do_split_box = (box_has_children[ibox] && box_levels[ibox] + 1 == level)
       %if level_restrict:
           || box_force_split[ibox]
       %endif
       ;

    if (!do_split_box)
    {
        // Not splitting? Copy over existing particle info.
        new_user_srcntgt_ids[i] = user_srcntgt_ids[i];
        new_srcntgt_box_ids[i] = ibox;

        PYOPENCL_ELWISE_CONTINUE;
    }

    morton_nr_t my_morton_nr = morton_nrs[i];
    // printf("   my morton nr: %d\n", my_morton_nr);

    morton_counts_t my_box_morton_bin_counts = box_morton_bin_counts[ibox];

    morton_counts_t my_morton_bin_counts = morton_bin_counts[i];
    particle_id_t my_count = get_count(my_morton_bin_counts, my_morton_nr);

    // {{{ compute this srcntgt's new index

    particle_id_t my_box_start = box_srcntgt_starts[ibox];
    particle_id_t tgt_particle_idx = my_box_start + my_count-1;
    %if srcntgts_have_extent:
        tgt_particle_idx +=
            (my_morton_nr >= 0)
                ? my_box_morton_bin_counts.nonchild_srcntgts
                : 0;
    %endif
    %for mnr in range(2**dimensions):
        <% bin_nmr = padded_bin(mnr, dimensions) %>
        tgt_particle_idx +=
            (my_morton_nr > ${mnr})
                ? my_box_morton_bin_counts.pcnt${bin_nmr}
                : 0;
    %endfor

    dbg_assert(tgt_particle_idx < n);
    dbg_printf(("   moving %ld -> %d "
        "(ibox %d, my_box_start %d, my_count %d)\n",
        i, tgt_particle_idx,
        ibox, my_box_start, my_count));

    new_user_srcntgt_ids[tgt_particle_idx] = user_srcntgt_ids[i];

    // }}}

    // {{{ compute this srcntgt's new box id

    box_id_t new_box_id = split_box_ids[ibox] - ${2**dimensions} + my_morton_nr;

    %if srcntgts_have_extent:
        if (my_morton_nr == -1)
        {
            new_box_id = ibox;
        }
    %endif

    dbg_printf(("   new_box_id: %d\n", new_box_id));
    dbg_assert(new_box_id >= 0);

    new_srcntgt_box_ids[tgt_particle_idx] = new_box_id;

    // }}}
""", strict_undefined=True)

# }}}

# {{{ level restrict kernel

from boxtree.traversal import TRAVERSAL_PREAMBLE_MAKO_DEFS

LEVEL_RESTRICT_TPL = Template(
    TRAVERSAL_PREAMBLE_MAKO_DEFS + r"""//CL:mako//
    <%def name="my_load_center(name, box_id)">
        ## This differs from load_center() because in this kernel box centers
        ## live in one array per axis.
        coord_vec_t ${name};
        %for i in range(dimensions):
            ${name}.${AXIS_NAMES[i]} = box_centers_${AXIS_NAMES[i]}[${box_id}];
        %endfor
    </%def>

    #define NLEVELS (${max_levels})

    box_id_t box_id = i;

    // Skip unless this box is a leaf.
    if (box_has_children[box_id])
    {
        PYOPENCL_ELWISE_CONTINUE;
    }

    ${walk_init(0)}

    // Descend the tree searching for neighboring leaves.
    while (continue_walk)
    {
        box_id_t child_box_id;
        // Look for the child in the appropriate array.
    %for morton_nr in range(2**dimensions):
        if (walk_morton_nr == ${morton_nr})
        {
            child_box_id = box_child_ids_mnr_${morton_nr}[walk_parent_box_id];
        }
    %endfor

        if (child_box_id)
        {
            int child_level = walk_stack_size + 1;

            // Check adjacency.
            bool is_adjacent;

            if (child_box_id == box_id)
            {
                // Skip considering self.
                is_adjacent = false;
            }
            else
            {
                ${my_load_center("box_center", "box_id")}
                ${my_load_center("child_center", "child_box_id")}
                is_adjacent = is_adjacent_or_overlapping(
                    root_extent, child_center, child_level, box_center, level);
            }

            if (is_adjacent)
            {
                // Invariant: When new leaves get added,
                // they are never more than 2 levels deeper than
                // all their adjacent leaves.
                //
                // Hence in we only need to look at boxes up to
                // (level + 2) deep.

                if (box_has_children[child_box_id])
                {
                    if (child_level <= 1 + level)
                    {
                        ${walk_push("child_box_id")}
                        continue;
                    }
                }
                else
                {
                    // We are looking at a neighboring leaf box.
                    // Check if my box must be split to enforce level
                    // restriction.
                    if (child_level == 2 + level || (
                        child_level == 1 + level &&
                        box_force_split[child_box_id]))
                    {
                        box_force_split[box_id] = 1;
                        atomic_or(have_upper_level_split_box, 1);
                        continue_walk = false;
                    }
                }
            }
        }
        ${walk_advance()}
    }
""", strict_undefined=True)


def build_level_restrict_kernel(context, preamble_with_dtype_decls,
            dimensions, axis_names, box_id_dtype, coord_dtype,
            box_level_dtype, max_levels):
    from boxtree.tools import VectorArg, ScalarArg

    arguments = (
        [
            # input
            ScalarArg(box_level_dtype, "level"),  # [1]
            ScalarArg(coord_dtype, "root_extent"),  # [1]
            VectorArg(np.int32, "box_has_children"),  # [nboxes]

            # input/output
            VectorArg(np.int32, "box_force_split"),  # [nboxes]

            # output
            VectorArg(np.int32, "have_upper_level_split_box"),  # [1]
        ]
        # input, length depends on dim
        + [VectorArg(box_id_dtype, f"box_child_ids_mnr_{mnr}")
             for mnr in range(2**dimensions)]  # [nboxes]
        + [VectorArg(coord_dtype, f"box_centers_{ax}")
             for ax in axis_names]  # [nboxes]
        )

    render_vars = dict(
        AXIS_NAMES=axis_names,
        dimensions=dimensions,
        max_levels=max_levels,
        # Entries below are needed by HELPER_FUNCTION_TEMPLATE
        # and/or TRAVERSAL_PREAMBLE_MAKO_DEFS:
        debug=False,
        targets_have_extent=False,
        sources_have_extent=False,
        get_coord_vec_dtype=get_coord_vec_dtype,
        cvec_sub=partial(coord_vec_subscript_code, dimensions),
        )

    from boxtree.traversal import HELPER_FUNCTION_TEMPLATE
    from pyopencl.elementwise import ElementwiseKernel

    return ElementwiseKernel(
            context,
            arguments=arguments,
            operation=LEVEL_RESTRICT_TPL.render(**render_vars),
            name="level_restrict",
            preamble=(
                str(preamble_with_dtype_decls)
                + Template(r"""
                    #define LEVEL_TO_RAD(level) \
                        (root_extent * 1 / (coord_t) (1 << (level + 1)))
                    """
                    + HELPER_FUNCTION_TEMPLATE)
                .render(**render_vars)))

# }}}

# END KERNELS IN THE LEVEL LOOP


# {{{ nonchild srcntgt count extraction

EXTRACT_NONCHILD_SRCNTGT_COUNT_TPL = ElementwiseTemplate(
    arguments="""//CL//
        /* input */
        morton_counts_t *box_morton_bin_counts,
        particle_id_t *box_srcntgt_counts_cumul,
        box_id_t highest_possibly_split_box_nr,

        /* output */
        particle_id_t *box_srcntgt_counts_nonchild,
        """,
    operation=r"""//CL//
        if (i >= highest_possibly_split_box_nr)
        {
            // box_morton_bin_counts gets written in morton scan output.
            // Therefore, newly created boxes in the last level don't
            // have it initialized.

            box_srcntgt_counts_nonchild[i] = 0;
        }
        else if (box_srcntgt_counts_cumul[i] == 0)
        {
            // If boxes are empty, box_morton_bin_counts never gets initialized.
            box_srcntgt_counts_nonchild[i] = 0;
        }
        else
            box_srcntgt_counts_nonchild[i] =
                box_morton_bin_counts[i].nonchild_srcntgts;
        """,
    name="extract_nonchild_srcntgt_count")

# }}}

# {{{ source and target index finding

SOURCE_AND_TARGET_INDEX_FINDER = ElementwiseTemplate(
    arguments="""//CL:mako//
        /* input */
        particle_id_t *user_srcntgt_ids,
        particle_id_t nsources,
        box_id_t *srcntgt_box_ids,
        box_id_t *box_parent_ids,

        particle_id_t *box_srcntgt_starts,
        particle_id_t *box_srcntgt_counts_cumul,
        particle_id_t *source_numbers,

        %if srcntgts_have_extent:
            particle_id_t *box_srcntgt_counts_nonchild,
        %endif

        /* output */
        particle_id_t *user_source_ids,
        particle_id_t *srcntgt_target_ids,
        particle_id_t *sorted_target_ids,

        particle_id_t *box_source_starts,
        particle_id_t *box_source_counts_cumul,
        particle_id_t *box_target_starts,
        particle_id_t *box_target_counts_cumul,

        %if srcntgts_have_extent:
            particle_id_t *box_source_counts_nonchild,
            particle_id_t *box_target_counts_nonchild,
        %endif
        """,
    operation=r"""//CL:mako//
        ## Splitting sources and targets makes no sense when they're the same.
        <% assert not sources_are_targets %>

        particle_id_t sorted_srcntgt_id = i;
        particle_id_t source_nr = source_numbers[i];
        particle_id_t target_nr = i - source_nr;

        box_id_t box_id = srcntgt_box_ids[sorted_srcntgt_id];

        particle_id_t box_start = box_srcntgt_starts[box_id];
        particle_id_t box_count = box_srcntgt_counts_cumul[box_id];

        particle_id_t user_srcntgt_id
            = user_srcntgt_ids[sorted_srcntgt_id];

        bool is_source = user_srcntgt_id < nsources;

        // {{{ write start and end of box in terms of sources and targets

        // first particle for this or the parents' boxes? update starts
        {
            particle_id_t walk_box_start = box_start;
            box_id_t walk_box_id = box_id;

            while (sorted_srcntgt_id == walk_box_start)
            {
                box_source_starts[walk_box_id] = source_nr;
                box_target_starts[walk_box_id] = target_nr;

                box_id_t new_box_id = box_parent_ids[walk_box_id];
                if (new_box_id == walk_box_id)
                {
                    // don't loop at root
                    dbg_assert(walk_box_id == 0);
                    break;
                }

                walk_box_id = new_box_id;
                walk_box_start = box_srcntgt_starts[walk_box_id];
            }
        }

        %if srcntgts_have_extent:
            // last non-child particle?

            // (Can't be "first child particle", because then the box might
            // not have any child particles!)

            particle_id_t box_nonchild_count = box_srcntgt_counts_nonchild[box_id];

            if (sorted_srcntgt_id + 1 == box_start + box_nonchild_count)
            {
                particle_id_t box_start_source_nr = source_numbers[box_start];
                particle_id_t box_start_target_nr = box_start - box_start_source_nr;

                box_source_counts_nonchild[box_id] =
                    source_nr + (particle_id_t) is_source
                    - box_start_source_nr;

                box_target_counts_nonchild[box_id] =
                    target_nr + 1 - (particle_id_t) is_source
                    - box_start_target_nr;
            }
        %endif

        // {{{ last particle for this or the parents' boxes? update counts

        {
            particle_id_t walk_box_start = box_start;
            particle_id_t walk_box_count = box_count;
            box_id_t walk_box_id = box_id;

            while (sorted_srcntgt_id + 1 == walk_box_start + walk_box_count)
            {
                particle_id_t box_start_source_nr =
                    source_numbers[walk_box_start];
                particle_id_t box_start_target_nr =
                    walk_box_start - box_start_source_nr;

                box_source_counts_cumul[walk_box_id] =
                    source_nr + (particle_id_t) is_source
                    - box_start_source_nr;

                box_target_counts_cumul[walk_box_id] =
                    target_nr + 1 - (particle_id_t) is_source
                    - box_start_target_nr;

                box_id_t new_box_id = box_parent_ids[walk_box_id];
                if (new_box_id == walk_box_id)
                {
                    // don't loop at root
                    dbg_assert(walk_box_id == 0);
                    break;
                }

                walk_box_id = new_box_id;

                walk_box_start = box_srcntgt_starts[walk_box_id];
                walk_box_count = box_srcntgt_counts_cumul[walk_box_id];
            }
        }

        // }}}

        // }}}

        if (is_source)
        {
            user_source_ids[source_nr] = user_srcntgt_id;
        }
        else
        {
            particle_id_t user_target_id = user_srcntgt_id - nsources;

            srcntgt_target_ids[target_nr] = user_srcntgt_id;
            sorted_target_ids[user_target_id] = target_nr;
        }

        """,
    name="find_source_and_target_indices")

# }}}

# {{{ source/target permuter

SRCNTGT_PERMUTER_TPL = ElementwiseTemplate(
    arguments="""//CL:mako//
        particle_id_t *from_ids
        %for ax in axis_names:
            , coord_t *${ax}
        %endfor
        %for ax in axis_names:
            , coord_t *sorted_${ax}
        %endfor
        """,
    operation=r"""//CL:mako//
        particle_id_t from_idx = from_ids[i];
        %for ax in axis_names:
            sorted_${ax}[i] = ${ax}[from_idx];
        %endfor
        """,
    name="permute_srcntgt")

# }}}

# {{{ box info kernel

BOX_INFO_KERNEL_TPL = ElementwiseTemplate(
    arguments="""//CL:mako//
        /* input */
        box_id_t *box_parent_ids,
        particle_id_t *box_srcntgt_counts_cumul,
        particle_id_t *box_source_counts_cumul,
        particle_id_t *box_target_counts_cumul,
        int *box_has_children,
        box_level_t *box_levels,
        box_level_t nlevels,

        /* output if not srcntgts_have_extent, input+output otherwise */
        particle_id_t *box_source_counts_nonchild,
        particle_id_t *box_target_counts_nonchild,

        /* output */
        box_flags_t *box_flags, /* [nboxes] */
        """,
    operation=r"""//CL:mako//
        const coord_t one_half = ((coord_t) 1) / 2;

        box_id_t box_id = i;

        /* Note that srcntgt_counts is a cumulative count over all children,
         * up to the point below where it is set to zero for non-leaves.
         *
         * box_srcntgt_counts_cumul is zero (here) exactly for empty leaves
         * because it gets initialized to zero and never gets set to another
         * value.
         */

        particle_id_t particle_count = box_srcntgt_counts_cumul[box_id];

        /* Up until this point, last-level (=>leaf) boxes may have their
         * non-child counts set to zero. (see
         * EXTRACT_NONCHILD_SRCNTGT_COUNT_TPL)
         * This is fixed below.
         *
         * In addition, upper-level leaves may have only part of their
         * particles declared non-child (i.e. "not fit for downward propagation")
         * Here, we reclassify all particles in leaves as non-child.
         */

        %if srcntgts_have_extent:
            const particle_id_t nonchild_source_count =
                box_source_counts_nonchild[box_id];
            const particle_id_t nonchild_target_count =
                box_target_counts_nonchild[box_id];
        %else:
            const particle_id_t nonchild_source_count = 0;
            const particle_id_t nonchild_target_count = 0;
        %endif

        const particle_id_t nonchild_srcntgt_count
            = nonchild_source_count + nonchild_target_count;

        box_flags_t my_box_flags = 0;

        dbg_assert(particle_count >= nonchild_srcntgt_count);

        if (box_has_children[box_id])
        {
            // This box has children, it is not a leaf.

            my_box_flags |= BOX_HAS_CHILDREN;

            %if sources_are_targets:
                if (particle_count - nonchild_srcntgt_count)
                    my_box_flags |= BOX_HAS_CHILD_SOURCES | BOX_HAS_CHILD_TARGETS;
            %else:
                particle_id_t source_count = box_source_counts_cumul[box_id];
                particle_id_t target_count = box_target_counts_cumul[box_id];

                dbg_assert(source_count >= nonchild_source_count);
                dbg_assert(target_count >= nonchild_target_count);

                if (source_count - nonchild_source_count)
                    my_box_flags |= BOX_HAS_CHILD_SOURCES;
                if (target_count - nonchild_target_count)
                    my_box_flags |= BOX_HAS_CHILD_TARGETS;
            %endif

            if (nonchild_source_count)
                my_box_flags |= BOX_HAS_OWN_SOURCES;
            if (nonchild_target_count)
                my_box_flags |= BOX_HAS_OWN_TARGETS;
        }
        else
        {
            // This box is a leaf, i.e. it has no children.

            %if sources_are_targets:
                if (particle_count)
                    my_box_flags |= BOX_HAS_OWN_SOURCES | BOX_HAS_OWN_TARGETS;

                box_source_counts_nonchild[box_id] = particle_count;
                dbg_assert(box_source_counts_nonchild == box_target_counts_nonchild);
            %else:
                particle_id_t my_source_count = box_source_counts_cumul[box_id];
                particle_id_t my_target_count = particle_count - my_source_count;

                if (my_source_count)
                    my_box_flags |= BOX_HAS_OWN_SOURCES;
                if (my_target_count)
                    my_box_flags |= BOX_HAS_OWN_TARGETS;

                box_source_counts_nonchild[box_id] = my_source_count;
                box_target_counts_nonchild[box_id] = my_target_count;
            %endif
        }

        box_flags[box_id] = my_box_flags;
    """)

# }}}

# {{{ box extents

BOX_EXTENTS_FINDER_TEMPLATE = ElementwiseTemplate(
    arguments="""//CL:mako//
    box_id_t aligned_nboxes,
    box_id_t *box_child_ids,
    coord_t *box_centers,
    particle_id_t *box_particle_starts,
    particle_id_t *box_particle_counts_nonchild

    %for iaxis in range(dimensions):
        , const coord_t *particle_${AXIS_NAMES[iaxis]}
    %endfor
    ,
    const coord_t *particle_radii,
    int enable_radii,

    coord_t *box_particle_bounding_box_min,
    coord_t *box_particle_bounding_box_max,
    """,

    operation=TRAVERSAL_PREAMBLE_MAKO_DEFS + r"""//CL:mako//
        box_id_t ibox = i;

        ${load_center("box_center", "ibox")}

        <% axis_names = AXIS_NAMES[:dimensions] %>

        // incorporate own particles
        %for iaxis, ax in enumerate(axis_names):
            coord_t min_particle_${ax} =
                ${coord_vec_subscript_code("box_center", iaxis)};
            coord_t max_particle_${ax} =
                ${coord_vec_subscript_code("box_center", iaxis)};
        %endfor

        particle_id_t start = box_particle_starts[ibox];
        particle_id_t stop = start + box_particle_counts_nonchild[ibox];

        for (particle_id_t iparticle = start; iparticle < stop; ++iparticle)
        {
            coord_t particle_rad = 0;
            %if srcntgts_have_extent:
                // If only one has extent, then the radius array for the other
                // may well be a null pointer.
                if (enable_radii)
                    particle_rad = particle_radii[iparticle];
            %endif

            %for iaxis, ax in enumerate(axis_names):
                coord_t particle_coord_${ax} = particle_${ax}[iparticle];

                min_particle_${ax} = min(
                    min_particle_${ax},
                    particle_coord_${ax} - particle_rad);
                max_particle_${ax} = max(
                    max_particle_${ax},
                    particle_coord_${ax} + particle_rad);
            %endfor
        }

        // incorporate child boxes
        for (int morton_nr = 0; morton_nr < ${2**dimensions}; ++morton_nr)
        {
            box_id_t child_id = box_child_ids[
                    morton_nr * aligned_nboxes + ibox];

            if (child_id == 0)
                continue;

            %for iaxis, ax in enumerate(axis_names):
                min_particle_${ax} = min(
                    min_particle_${ax},
                    box_particle_bounding_box_min[
                        ${iaxis} * aligned_nboxes + child_id]);
                max_particle_${ax} = max(
                    max_particle_${ax},
                    box_particle_bounding_box_max[
                        ${iaxis} * aligned_nboxes + child_id]);
            %endfor
        }

        // write result
        %for iaxis, ax in enumerate(axis_names):
            box_particle_bounding_box_min[
                ${iaxis} * aligned_nboxes + ibox] = min_particle_${ax};
            box_particle_bounding_box_max[
                ${iaxis} * aligned_nboxes + ibox] = max_particle_${ax};
        %endfor
    """,
    name="find_box_extents")

# }}}

# {{{ kernel creation top-level


@log_process(logger)
def get_tree_build_kernel_info(context, dimensions, coord_dtype,
        particle_id_dtype, box_id_dtype,
        sources_are_targets, srcntgts_extent_norm,
        morton_nr_dtype, box_level_dtype, kind):
    """
    :arg srcntgts_extent_norm: one of ``None``, ``"l2"`` or ``"linf"``
    """

    level_restrict = (kind == "adaptive-level-restricted")
    adaptive = not (kind == "non-adaptive")

    # {{{ preparation

    if np.iinfo(box_id_dtype).min == 0:
        from warnings import warn
        warn("Careful with unsigned types for box_id_dtype. Some CL implementations "
                "(notably Intel 2012) mis-implemnet unsigned operations, leading to "
                "incorrect results.", stacklevel=4)

    from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
    coord_vec_dtype = get_coord_vec_dtype(coord_dtype, dimensions)

    particle_id_dtype = np.dtype(particle_id_dtype)
    box_id_dtype = np.dtype(box_id_dtype)

    dev = context.devices[0]
    morton_bin_count_dtype, _ = make_morton_bin_count_type(
            dev, dimensions, particle_id_dtype,
            srcntgts_have_extent=srcntgts_extent_norm is not None)

    from boxtree.bounding_box import make_bounding_box_dtype
    bbox_dtype, bbox_type_decl = make_bounding_box_dtype(
            dev, dimensions, coord_dtype)

    from boxtree.tools import AXIS_NAMES
    axis_names = AXIS_NAMES[:dimensions]

    from boxtree.tools import padded_bin
    from boxtree.tree import box_flags_enum
    codegen_args = dict(
            dimensions=dimensions,
            axis_names=axis_names,
            padded_bin=padded_bin,
            coord_dtype=coord_dtype,
            coord_vec_dtype=coord_vec_dtype,
            bbox_dtype=bbox_dtype,
            refine_weight_dtype=refine_weight_dtype,
            particle_id_dtype=particle_id_dtype,
            morton_bin_count_dtype=morton_bin_count_dtype,
            morton_nr_dtype=morton_nr_dtype,
            box_id_dtype=box_id_dtype,
            box_level_dtype=box_level_dtype,
            dtype_to_ctype=dtype_to_ctype,
            AXIS_NAMES=AXIS_NAMES,
            box_flags_enum=box_flags_enum,

            adaptive=adaptive,
            level_restrict=level_restrict,

            sources_are_targets=sources_are_targets,
            srcntgts_have_extent=srcntgts_extent_norm is not None,
            srcntgts_extent_norm=srcntgts_extent_norm,

            enable_assert=False,
            enable_printf=False,
            )

    # }}}

    generic_preamble = str(GENERIC_PREAMBLE_TPL.render(**codegen_args))

    preamble_with_dtype_decls = (
            dtype_to_c_struct(dev, bbox_dtype)
            + dtype_to_c_struct(dev, morton_bin_count_dtype)
            + str(TYPE_DECL_PREAMBLE_TPL.render(**codegen_args))
            + generic_preamble
            )

    # BEGIN KERNELS IN LEVEL LOOP

    # {{{ scan

    scan_preamble = (
            preamble_with_dtype_decls
            + str(MORTON_NR_SCAN_PREAMBLE_TPL.render(**codegen_args))
            )

    from boxtree.tools import VectorArg, ScalarArg
    common_arguments = (
            [
                # box-local morton bin counts for each particle at the current level
                # only valid from scan -> split'n'sort

                VectorArg(morton_bin_count_dtype, "morton_bin_counts"),
                # [nsrcntgts]

                # (local) morton nrs for each particle at the current level
                # only valid from scan -> split'n'sort
                VectorArg(morton_nr_dtype, "morton_nrs"),  # [nsrcntgts]

                # segment flags
                # invariant to sorting once set
                # (particles are only reordered within a box)
                VectorArg(np.uint8, "box_start_flags"),   # [nsrcntgts]

                VectorArg(box_id_dtype, "srcntgt_box_ids"),  # [nsrcntgts]
                VectorArg(box_id_dtype, "split_box_ids"),  # [nboxes]

                # per-box morton bin counts
                VectorArg(morton_bin_count_dtype, "box_morton_bin_counts"),
                # [nboxes]

                VectorArg(refine_weight_dtype, "refine_weights"),
                # [nsrcntgts]

                ScalarArg(refine_weight_dtype, "max_leaf_refine_weight"),

                # particle# at which each box starts
                VectorArg(particle_id_dtype, "box_srcntgt_starts"),  # [nboxes]

                # number of particles in each box
                VectorArg(particle_id_dtype, "box_srcntgt_counts_cumul"),  # [nboxes]

                # pointer to parent box
                VectorArg(box_id_dtype, "box_parent_ids"),  # [nboxes]

                # level number
                VectorArg(box_level_dtype, "box_levels"),  # [nboxes]

                ScalarArg(np.int32, "level"),
                ScalarArg(bbox_dtype, "bbox"),

                VectorArg(particle_id_dtype, "user_srcntgt_ids"),  # [nsrcntgts]
                ]

            # particle coordinates
            + [VectorArg(coord_dtype, ax) for ax in axis_names]

            + ([VectorArg(coord_dtype, "srcntgt_radii")]
                if srcntgts_extent_norm is not None else [])
            )

    morton_count_scan_arguments = list(common_arguments)

    if srcntgts_extent_norm is not None:
        morton_count_scan_arguments += [
            (ScalarArg(coord_dtype, "stick_out_factor"))
        ]

    from pyopencl.scan import GenericScanKernel
    morton_count_scan = GenericScanKernel(
            context, morton_bin_count_dtype,
            arguments=morton_count_scan_arguments,
            input_expr=(
                "scan_t_from_particle(%s)"
                % ", ".join([
                    "i", "box_levels[srcntgt_box_ids[i]]", "&bbox", "morton_nrs",
                    "user_srcntgt_ids",
                    "refine_weights",
                    ]
                    + ["%s" % ax for ax in axis_names]
                    + (["srcntgt_radii, stick_out_factor"]
                       if srcntgts_extent_norm is not None else []))),
            scan_expr="scan_t_add(a, b, across_seg_boundary)",
            neutral="scan_t_neutral()",
            is_segment_start_expr="box_start_flags[i]",
            output_statement=MORTON_NR_SCAN_OUTPUT_STMT_TPL.render(**codegen_args),
            preamble=scan_preamble,
            name_prefix="morton_scan")

    # }}}

    # {{{ split_box_id scan

    from pyopencl.scan import GenericScanKernel
    split_box_id_scan = SPLIT_BOX_ID_SCAN_TPL.build(
            context,
            type_aliases=(
                ("scan_t", box_id_dtype),
                ("index_t", particle_id_dtype),
                ("particle_id_t", particle_id_dtype),
                ("box_id_t", box_id_dtype),
                ("morton_counts_t", morton_bin_count_dtype),
                ("box_level_t", box_level_dtype),
                ("refine_weight_t", refine_weight_dtype),
                ),
            var_values=(
                ("dimensions", dimensions),
                ("srcntgts_have_extent", srcntgts_extent_norm is not None),
                ("srcntgts_extent_norm", srcntgts_extent_norm),
                ("adaptive", adaptive),
                ("padded_bin", padded_bin),
                ("level_restrict", level_restrict),
                ),
            more_preamble=generic_preamble)

    # }}}

    # {{{ box splitter

    # Work around a bug in Mako < 0.7.3
    # FIXME: Is this needed?
    box_s_codegen_args = codegen_args.copy()
    box_s_codegen_args.update(
        dim=None,
        boundary_morton_nr=None)

    box_splitter_kernel_source = BOX_SPLITTER_KERNEL_TPL.render(**box_s_codegen_args)

    from pyopencl.elementwise import ElementwiseKernel
    box_splitter_kernel = ElementwiseKernel(
            context,
            common_arguments
            + [
                VectorArg(np.int32, "box_has_children"),
                VectorArg(np.int32, "box_force_split"),
                ScalarArg(coord_dtype, "root_extent"),
                ]
            + [VectorArg(box_id_dtype, f"box_child_ids_mnr_{mnr}")
                          for mnr in range(2**dimensions)]
            + [VectorArg(coord_dtype, f"box_centers_{ax}")
                          for ax in axis_names],
            str(box_splitter_kernel_source),
            name="box_splitter",
            preamble=preamble_with_dtype_decls
            )

    # }}}

    # {{{ particle renumberer

    # Work around a bug in Mako < 0.7.3
    # FIXME: Copied from above. It may not be necessary?
    part_rn_codegen_args = codegen_args.copy()
    part_rn_codegen_args.update(
            dim=None,
            boundary_morton_nr=None)

    particle_renumberer_preamble = \
            PARTICLE_RENUMBERER_PREAMBLE_TPL.render(**part_rn_codegen_args)

    particle_renumberer_kernel_source = \
            PARTICLE_RENUMBERER_KERNEL_TPL.render(**codegen_args)

    from pyopencl.elementwise import ElementwiseKernel
    particle_renumberer_kernel = ElementwiseKernel(
            context,
            common_arguments
            + [
                VectorArg(np.int32, "box_has_children"),
                VectorArg(np.int32, "box_force_split"),
                VectorArg(particle_id_dtype, "new_user_srcntgt_ids"),
                VectorArg(box_id_dtype, "new_srcntgt_box_ids"),
                ],
            str(particle_renumberer_kernel_source), name="renumber_particles",
            preamble=(
                preamble_with_dtype_decls
                + str(particle_renumberer_preamble))
            )

    # }}}

    # {{{ level restrict propagator

    if level_restrict:
        # At compile time the level restrict kernel requires fixing a
        # "max_levels" constant for traversing the tree. This constant cannot be
        # known at this point, hence we return a kernel builder.

        level_restrict_kernel_builder = partial(build_level_restrict_kernel,
            context, preamble_with_dtype_decls, dimensions, axis_names, box_id_dtype,
            coord_dtype, box_level_dtype)
    else:
        level_restrict_kernel_builder = None

    # }}}

    # END KERNELS IN LEVEL LOOP

    if srcntgts_extent_norm is not None:
        extract_nonchild_srcntgt_count_kernel = \
                EXTRACT_NONCHILD_SRCNTGT_COUNT_TPL.build(
                        context,
                        type_aliases=(
                            ("particle_id_t", particle_id_dtype),
                            ("box_id_t", box_id_dtype),
                            ("morton_counts_t", morton_bin_count_dtype),
                            ),
                        var_values=(),
                        more_preamble=generic_preamble)

    else:
        extract_nonchild_srcntgt_count_kernel = None

    # {{{ find-prune-indices

    # FIXME: Turn me into a scan template

    from boxtree.tools import VectorArg
    find_prune_indices_kernel = GenericScanKernel(
            context, box_id_dtype,
            arguments=[
                # input
                VectorArg(particle_id_dtype, "box_srcntgt_counts_cumul"),
                # output
                VectorArg(box_id_dtype, "src_box_id"),
                VectorArg(box_id_dtype, "dst_box_id"),
                VectorArg(box_id_dtype, "nboxes_post_prune"),
                ],
            input_expr="box_srcntgt_counts_cumul[i] != 0",
            preamble=box_flags_enum.get_c_defines(),
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (box_srcntgt_counts_cumul[i])
                {
                    dst_box_id[i] = item - 1;
                    src_box_id[item - 1] = i;
                }
                if (i+1 == N) *nboxes_post_prune = item;
                """,
            name_prefix="find_prune_indices_scan")

    # }}}

    # {{{ find new level box counts

    find_level_box_counts_kernel = GenericScanKernel(
        context, box_id_dtype,
        arguments=[
            # input
            VectorArg(box_level_dtype, "box_levels"),  # [nboxes]
            # output
            VectorArg(box_id_dtype, "level_box_counts"),  # [nlevels]
            ],
        input_expr="1",
        is_segment_start_expr="i == 0 || box_levels[i] != box_levels[i - 1]",
        scan_expr="across_seg_boundary ? b : a + b",
        neutral="0",
        output_statement=r"""//CL//
        if (i + 1 == N || box_levels[i] != box_levels[i + 1])
        {
            level_box_counts[box_levels[i]] = item;
        }
        """,
        name_prefix="find_level_box_counts_scan")

    # }}}

    # {{{ particle permuter

    # used if there is only one source/target array
    srcntgt_permuter = SRCNTGT_PERMUTER_TPL.build(
            context,
            type_aliases=(
                ("particle_id_t", particle_id_dtype),
                ("box_id_t", box_id_dtype),
                ("coord_t", coord_dtype),
                ),
            var_values=(
                ("axis_names", axis_names),
                ),
            more_preamble=generic_preamble)

    # }}}

    # {{{ source-and-target splitter

    # These kernels are only needed if there are separate sources and
    # targets.  But they're only built on-demand anyway, so that's not
    # really a loss.

    # FIXME: make me a scan template
    from pyopencl.scan import GenericScanKernel
    source_counter = GenericScanKernel(
            context, box_id_dtype,
            arguments=[
                # input
                VectorArg(particle_id_dtype, "user_srcntgt_ids"),
                ScalarArg(particle_id_dtype, "nsources"),
                # output
                VectorArg(particle_id_dtype, "source_numbers"),
                ],
            input_expr="(user_srcntgt_ids[i] < nsources) ? 1 : 0",
            scan_expr="a+b", neutral="0",
            output_statement="source_numbers[i] = prev_item;",
            name_prefix="source_counter")

    if not sources_are_targets:
        source_and_target_index_finder = SOURCE_AND_TARGET_INDEX_FINDER.build(
                context,
                type_aliases=(
                    ("particle_id_t", particle_id_dtype),
                    ("box_id_t", box_id_dtype),
                    ),
                var_values=(
                    ("srcntgts_have_extent", srcntgts_extent_norm is not None),
                    ("sources_are_targets", sources_are_targets),
                    ),
                more_preamble=generic_preamble)
    else:
        source_and_target_index_finder = None

    # }}}

    # {{{ box-info

    type_aliases = (
            ("box_id_t", box_id_dtype),
            ("particle_id_t", particle_id_dtype),
            ("bbox_t", bbox_dtype),
            ("coord_t", coord_dtype),
            ("morton_nr_t", morton_nr_dtype),
            ("coord_vec_t", coord_vec_dtype),
            ("box_flags_t", box_flags_enum.dtype),
            ("box_level_t", box_level_dtype),
            )
    codegen_args_tuples = tuple(codegen_args.items())
    box_info_kernel = BOX_INFO_KERNEL_TPL.build(
            context,
            type_aliases,
            var_values=codegen_args_tuples,
            more_preamble=box_flags_enum.get_c_defines() + generic_preamble,
            )

    # }}}

    # {{{ box extent

    box_extents_finder_kernel = BOX_EXTENTS_FINDER_TEMPLATE.build(context,
        type_aliases=(
            ("box_id_t", box_id_dtype),
            ("coord_t", coord_dtype),
            ("coord_vec_t", get_coord_vec_dtype(coord_dtype, dimensions)),
            ("particle_id_t", particle_id_dtype),
            ),
        var_values=(
            ("coord_vec_subscript_code",
                partial(coord_vec_subscript_code, dimensions)),
            ("dimensions", dimensions),
            ("AXIS_NAMES", AXIS_NAMES),
            ("srcntgts_have_extent", srcntgts_extent_norm is not None),
            ),
    )

    # }}}

    return _KernelInfo(
            particle_id_dtype=particle_id_dtype,
            box_id_dtype=box_id_dtype,
            morton_bin_count_dtype=morton_bin_count_dtype,

            morton_count_scan=morton_count_scan,
            split_box_id_scan=split_box_id_scan,
            box_splitter_kernel=box_splitter_kernel,
            particle_renumberer_kernel=particle_renumberer_kernel,
            level_restrict=level_restrict,
            level_restrict_kernel_builder=level_restrict_kernel_builder,

            extract_nonchild_srcntgt_count_kernel=(
                extract_nonchild_srcntgt_count_kernel),
            find_prune_indices_kernel=find_prune_indices_kernel,
            find_level_box_counts_kernel=find_level_box_counts_kernel,
            srcntgt_permuter=srcntgt_permuter,
            source_counter=source_counter,
            source_and_target_index_finder=source_and_target_index_finder,
            box_info_kernel=box_info_kernel,
            box_extents_finder_kernel=box_extents_finder_kernel,
            )

# }}}


# {{{ point source linking kernels

# scan over (non-point) source ids in tree order
POINT_SOURCE_LINKING_SOURCE_SCAN_TPL = ScanTemplate(
    arguments=r"""//CL:mako//
        /* input */
        particle_id_t *point_source_starts,
        particle_id_t *user_source_ids,

        /* output */
        particle_id_t *tree_order_point_source_starts,
        particle_id_t *tree_order_point_source_counts,
        particle_id_t *npoint_sources
        """,
    input_expr="""
        point_source_starts[user_source_ids[i]+1]
        - point_source_starts[user_source_ids[i]]
        """,
    scan_expr="a + b",
    neutral="0",
    output_statement="""//CL//
        tree_order_point_source_starts[i] = prev_item;
        tree_order_point_source_counts[i] = item - prev_item;

        // Am I the last particle overall? If so, write point source count
        if (i+1 == N)
            *npoint_sources = item;
        """)

POINT_SOURCE_LINKING_USER_POINT_SOURCE_ID_SCAN_TPL = ScanTemplate(
    arguments=r"""//CL:mako//
        char *source_boundaries,
        particle_id_t *user_point_source_ids
        """,
    input_expr="user_point_source_ids[i]",
    scan_expr="across_seg_boundary ? b : a + b",
    neutral="0",
    is_segment_start_expr="source_boundaries[i]",
    output_statement="user_point_source_ids[i] = item;")

POINT_SOURCE_LINKING_BOX_POINT_SOURCES = ElementwiseTemplate(
    arguments="""//CL//
        particle_id_t *box_point_source_starts,
        particle_id_t *box_point_source_counts_nonchild,
        particle_id_t *box_point_source_counts_cumul,

        particle_id_t *box_source_starts,
        particle_id_t *box_source_counts_nonchild,
        particle_id_t *box_source_counts_cumul,

        particle_id_t *tree_order_point_source_starts,
        particle_id_t *tree_order_point_source_counts,
        """,
    operation=r"""//CL:mako//
        box_id_t ibox = i;
        particle_id_t s_start = box_source_starts[ibox];
        particle_id_t ps_start = tree_order_point_source_starts[s_start];

        box_point_source_starts[ibox] = ps_start;

        %for count_type in ["nonchild", "cumul"]:
            {
                particle_id_t s_count = box_source_counts_${count_type}[ibox];
                if (s_count == 0)
                {
                    box_point_source_counts_${count_type}[ibox] = 0;
                }
                else
                {
                    particle_id_t last_s_in_box = s_start+s_count-1;
                    particle_id_t beyond_last_ps_in_box =
                        tree_order_point_source_starts[last_s_in_box]
                        + tree_order_point_source_counts[last_s_in_box];
                    box_point_source_counts_${count_type}[ibox] = \
                            beyond_last_ps_in_box - ps_start;
                }
            }
        %endfor
        """,
    name="box_point_sources")

# }}}

# {{{ target filtering

TREE_ORDER_TARGET_FILTER_SCAN_TPL = ScanTemplate(
    arguments=r"""//CL:mako//
        /* input */
        unsigned char *tree_order_flags,

        /* output */
        particle_id_t *filtered_from_unfiltered_target_index,
        particle_id_t *unfiltered_from_filtered_target_index,
        particle_id_t *nfiltered_targets
        """,
    input_expr="tree_order_flags[i] ? 1 : 0",
    scan_expr="a + b",
    neutral="0",
    output_statement="""//CL//
        filtered_from_unfiltered_target_index[i] = prev_item;
        if (item != prev_item)
            unfiltered_from_filtered_target_index[prev_item] = i;

        // Am I the last particle overall? If so, write count
        if (i+1 == N)
            *nfiltered_targets = item;
        """)

TREE_ORDER_TARGET_FILTER_INDEX_TPL = ElementwiseTemplate(
    arguments="""//CL//
        /* input */
        particle_id_t *box_target_starts,
        particle_id_t *box_target_counts_nonchild,
        particle_id_t *filtered_from_unfiltered_target_index,
        particle_id_t ntargets,
        particle_id_t nfiltered_targets,

        /* output */
        particle_id_t *box_target_starts_filtered,
        particle_id_t *box_target_counts_nonchild_filtered,
        """,
    operation=r"""//CL//
        particle_id_t unfiltered_start = box_target_starts[i];
        particle_id_t unfiltered_count = box_target_counts_nonchild[i];

        particle_id_t filtered_start =
            filtered_from_unfiltered_target_index[unfiltered_start];
        box_target_starts_filtered[i] = filtered_start;

        if (unfiltered_count > 0)
        {
            particle_id_t unfiltered_post_last =
                unfiltered_start + unfiltered_count;

            particle_id_t filtered_post_last;
            if (unfiltered_post_last < ntargets)
            {
                filtered_post_last =
                    filtered_from_unfiltered_target_index[unfiltered_post_last];
            }
            else
            {
                // The above access would be out of bounds in this case.
                filtered_post_last = nfiltered_targets;
            }

            box_target_counts_nonchild_filtered[i] =
                filtered_post_last - filtered_start;
        }
        else
            box_target_counts_nonchild_filtered[i] = 0;
        """,
    name="tree_order_target_filter_index_finder")

# }}}

# vim: foldmethod=marker
