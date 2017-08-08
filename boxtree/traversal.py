from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import numpy as np
from pytools import Record, memoize_method, memoize_in
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.cltypes  # noqa
from pyopencl.elementwise import ElementwiseTemplate
from mako.template import Template
from boxtree.tools import AXIS_NAMES, DeviceDataRecord

import logging
logger = logging.getLogger(__name__)


# {{{ preamble

# This 'walk' mechanism walks over 'child' boxes in the tree.

TRAVERSAL_PREAMBLE_MAKO_DEFS = r"""//CL:mako//
<%def name="walk_init(start_box_id)">
    box_id_t walk_box_stack[NLEVELS];
    int walk_morton_nr_stack[NLEVELS];

    // start at root
    int walk_stack_size = 0;
    box_id_t walk_parent_box_id = ${start_box_id};
    int walk_morton_nr = 0;
    bool continue_walk = true;
</%def>

<%def name="walk_get_box_id()">
    box_id_t walk_box_id = box_child_ids[
        walk_morton_nr * aligned_nboxes + walk_parent_box_id];
</%def>

<%def name="walk_advance()">
    while (true)
    {
        ++walk_morton_nr;
        if (walk_morton_nr < ${2**dimensions})
            break;

        // Ran out of children, pull the next guy off the stack
        // and advance him.

        continue_walk = (
            // Stack empty? Abort.
            walk_stack_size > 0
            );

        if (continue_walk)
        {
            --walk_stack_size;
            dbg_printf(("    ascend\n"));
            walk_parent_box_id = walk_box_stack[walk_stack_size];
            walk_morton_nr = walk_morton_nr_stack[walk_stack_size];
        }
        else
        {
            dbg_printf(("done\n"));
            break;
        }
    }
</%def>

<%def name="walk_push(new_box)">
    walk_box_stack[walk_stack_size] = walk_parent_box_id;
    walk_morton_nr_stack[walk_stack_size] = walk_morton_nr;
    ++walk_stack_size;

    %if debug:
    if (walk_stack_size >= NLEVELS)
    {
        dbg_printf(("  ** ERROR: overran levels stack\n"));
        return;
    }
    %endif

    walk_parent_box_id = ${new_box};
    walk_morton_nr = 0;
</%def>

<%def name="load_center(name, box_id, declare=True)">
    %if declare:
        coord_vec_t ${name};
    %endif
    %for i in range(dimensions):
        ${name}.${AXIS_NAMES[i]} = box_centers[aligned_nboxes * ${i} + ${box_id}];
    %endfor
</%def>

<%def name="check_l_infty_ball_overlap(
        is_overlapping, box_id, ball_radius, ball_center)">
    {
        ${load_center("box_center", box_id)}
        int box_level = box_levels[${box_id}];

        coord_t size_sum = LEVEL_TO_RAD(box_level) + ${ball_radius};

        coord_t max_dist = 0;
        %for i in range(dimensions):
            max_dist = fmax(max_dist,
                fabs(${ball_center}.s${i} - box_center.s${i}));
        %endfor

        ${is_overlapping} = max_dist <= size_sum;
    }
</%def>
"""


TRAVERSAL_PREAMBLE_TYPEDEFS_AND_DEFINES = r"""//CL//
${box_flags_enum.get_c_defines()}
${box_flags_enum.get_c_typedef()}

typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
%if particle_id_dtype is not None:
    typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;
%endif
## Convert to dict first, as this may be passed as a tuple-of-tuples.
<% vec_types_dict = dict(vec_types) %>
typedef ${dtype_to_ctype(coord_dtype)} coord_t;
typedef ${dtype_to_ctype(vec_types_dict[coord_dtype, dimensions])} coord_vec_t;

#define COORD_T_MACH_EPS ((coord_t) ${ repr(np.finfo(coord_dtype).eps) })

#define NLEVELS ${max_levels}

#define LEVEL_TO_RAD(level) \
        (root_extent * 1 / (coord_t) (1 << (level + 1)))

%if 0:
    #define dbg_printf(ARGS) printf ARGS
%else:
    #define dbg_printf(ARGS) /* */
%endif
"""


TRAVERSAL_PREAMBLE_TEMPLATE = (
    TRAVERSAL_PREAMBLE_MAKO_DEFS +
    TRAVERSAL_PREAMBLE_TYPEDEFS_AND_DEFINES)

# }}}

# {{{ adjacency test

HELPER_FUNCTION_TEMPLATE = r"""//CL//

/*
These adjacency tests check the l^\infty distance between centers to check whether
two boxes are adjacent or overlapping.

Rather than a 'small floating point number', these adjacency test routines use the
smaller of the source/target box radii as the floating point tolerance, which
calls the following configuration 'adjacent' even though it actually is not:

    +---------+     +---------+
    |         |     |         |
    |         |     |         |
    |    o    |     |    o<--->
    |         |  r  |       r |
    |         |<--->|         |
    +---------+     +---------+

This is generically OK since one would expect the distance between the edge of
a large box and the edge of a smaller box to be a integer multiple of the
smaller box's diameter (which is twice its radius, our tolerance).
*/


inline bool is_adjacent_or_overlapping_with_neighborhood(
    coord_t root_extent,
    coord_vec_t target_center, int target_level,
    coord_t target_box_neighborhood_size,
    coord_vec_t source_center, int source_level)
{
    // This checks if the source box overlaps the target box
    // including a neighborhood of target_box_neighborhood_size boxes
    // of the same size as the target box.

    coord_t target_rad = LEVEL_TO_RAD(target_level);
    coord_t source_rad = LEVEL_TO_RAD(source_level);
    coord_t rad_sum = (
        (2*(target_box_neighborhood_size-1) + 1) * target_rad
        + source_rad);
    coord_t slack = rad_sum + fmin(target_rad, source_rad);

    coord_t l_inf_dist = 0;
    %for i in range(dimensions):
        l_inf_dist = fmax(
            l_inf_dist,
            fabs(target_center.s${i} - source_center.s${i}));
    %endfor

    return l_inf_dist <= slack;
}

inline bool is_adjacent_or_overlapping(
    coord_t root_extent,
    // note: order does not matter
    coord_vec_t target_center, int target_level,
    coord_vec_t source_center, int source_level)
{
    return is_adjacent_or_overlapping_with_neighborhood(
        root_extent,
        target_center, target_level,
        1,
        source_center, source_level);
}

"""

# }}}

# {{{ sources and their parents, targets

SOURCES_PARENTS_AND_TARGETS_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    box_flags_t flags = box_flags[box_id];

    if (flags & BOX_HAS_OWN_SOURCES)
    { APPEND_source_boxes(box_id); }

    if (flags & BOX_HAS_CHILD_SOURCES)
    { APPEND_source_parent_boxes(box_id); }

    %if not sources_are_targets:
        if (flags & BOX_HAS_OWN_TARGETS)
        { APPEND_target_boxes(box_id); }
    %endif
    if (flags & (BOX_HAS_CHILD_TARGETS | BOX_HAS_OWN_TARGETS))
    { APPEND_target_or_target_parent_boxes(box_id); }
}
"""

# }}}

# {{{ level start box nrs

LEVEL_START_BOX_NR_EXTRACTOR_TEMPLATE = ElementwiseTemplate(
    arguments="""//CL//
    box_id_t *level_start_box_nrs,
    box_level_t *box_levels,
    box_id_t *box_list,
    box_id_t *list_level_start_box_nrs,
    """,

    operation=r"""//CL//
        // Kernel is ranged so that this is true:
        // assert(i > 0);

        box_id_t my_box_id = box_list[i];
        int my_level = box_levels[my_box_id];

        bool is_level_leading_box;
        if (i == 0)
            is_level_leading_box = true;
        else
        {
            box_id_t prev_box_id = box_list[i-1];
            box_id_t my_level_start = level_start_box_nrs[my_level];

            is_level_leading_box = (
                    prev_box_id < my_level_start
                    && my_level_start <= my_box_id);
        }

        if (is_level_leading_box)
            list_level_start_box_nrs[my_level] = i;
    """,
    name="extract_level_start_box_nrs")

# }}}

# {{{ same-level non-well-separated boxes (generalization of "colleagues")

SAME_LEVEL_NON_WELL_SEP_BOXES_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    ${load_center("center", "box_id")}

    if (box_id == 0)
    {
        // The root has no boxes on the same level, nws or not.
        return;
    }

    int level = box_levels[box_id];

    dbg_printf(("box id: %d level: %d\n", box_id, level));

    // To find this box's same-level nws boxes, start at the top of the tree, descend
    // into adjacent (or overlapping) parents.
    ${walk_init(0)}

    while (continue_walk)
    {
        ${walk_get_box_id()}

        dbg_printf(("  level: %d walk parent box id: %d morton: %d child id: %d\n",
            walk_stack_size, walk_parent_box_id, walk_morton_nr, walk_box_id));

        if (walk_box_id)
        {
            ${load_center("walk_center", "walk_box_id")}

            bool a_or_o = is_adjacent_or_overlapping_with_neighborhood(
                    root_extent,
                    center, level,
                    ${well_sep_is_n_away},
                    walk_center, box_levels[walk_box_id]);

            if (a_or_o)
            {
                // walk_box_id lives on level walk_stack_size+1.
                if (walk_stack_size+1 == level && walk_box_id != box_id)
                {
                    dbg_printf(("    found same-lev nws\n"));
                    APPEND_same_level_non_well_sep_boxes(walk_box_id);
                }
                else
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    dbg_printf(("    descend\n"));
                    ${walk_push("walk_box_id")}

                    continue;
                }
            }
            else
            {
                dbg_printf(("    not adjacent\n"));
            }
        }

        ${walk_advance()}
    }
}

"""

# }}}

# {{{ neighbor source boxes ("list 1")

NEIGBHOR_SOURCE_BOXES_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t target_box_number)
{
    // /!\ target_box_number is *not* a box_id, despite the type.
    // It's the number of the source box we're currently processing.

    box_id_t box_id = target_boxes[target_box_number];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    dbg_printf(("box id: %d level: %d\n", box_id, level));

    // root box is not part of walk, check it up front.
    // Also no need to check for overlap-iness. The root box
    // overlaps *everybody*.

    {
        box_flags_t root_flags = box_flags[0];
        if (root_flags & BOX_HAS_OWN_SOURCES)
        {
            APPEND_neighbor_source_boxes(0);
        }
    }

    // To find this box's adjacent boxes, start at the top of the tree, descend
    // into adjacent (or overlapping) parents.
    ${walk_init(0)}

    while (continue_walk)
    {
        ${walk_get_box_id()}

        dbg_printf(("  walk parent box id: %d morton: %d child id: %d level: %d\n",
            walk_parent_box_id, walk_morton_nr, walk_box_id, walk_stack_size));

        if (walk_box_id)
        {
            ${load_center("walk_center", "walk_box_id")}

            bool a_or_o = is_adjacent_or_overlapping(
                root_extent,
                center, level,
                walk_center, box_levels[walk_box_id]);

            if (a_or_o)
            {
                box_flags_t flags = box_flags[walk_box_id];
                /* walk_box_id == box_id is ok */
                if (flags & BOX_HAS_OWN_SOURCES)
                {
                    dbg_printf(("    neighbor source box\n"));

                    APPEND_neighbor_source_boxes(walk_box_id);
                }

                if (flags & BOX_HAS_CHILD_SOURCES)
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    dbg_printf(("    descend\n"));

                    ${walk_push("walk_box_id")}

                    continue;
                }
            }
            else
            {
                dbg_printf(("    not adjacent\n"));
            }
        }

        ${walk_advance()}
    }
}

"""

# }}}

# {{{ from well-separated siblings ("list 2")

FROM_SEP_SIBLINGS_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t itarget_or_target_parent_box)
{
    box_id_t box_id = target_or_target_parent_boxes[itarget_or_target_parent_box];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    box_id_t parent = box_parent_ids[box_id];
    if (parent == box_id)
        return;

    box_id_t parent_slnf_start = same_level_non_well_sep_boxes_starts[parent];
    box_id_t parent_slnf_stop = same_level_non_well_sep_boxes_starts[parent+1];

    // /!\ i is not a box_id, it's an index into same_level_non_well_sep_boxes_list.
    for (box_id_t i = parent_slnf_start; i < parent_slnf_stop; ++i)
    {
        box_id_t parent_nf = same_level_non_well_sep_boxes_lists[i];

        for (int morton_nr = 0; morton_nr < ${2**dimensions}; ++morton_nr)
        {
            box_id_t sib_box_id = box_child_ids[
                    morton_nr * aligned_nboxes + parent_nf];

            ${load_center("sib_center", "sib_box_id")}

            bool sep = !is_adjacent_or_overlapping_with_neighborhood(
                root_extent,
                center, level,
                ${well_sep_is_n_away},
                sib_center, box_levels[sib_box_id]);

            if (sep)
            {
                APPEND_from_sep_siblings(sib_box_id);
            }
        }
    }
}
"""

# }}}

# {{{ from separated smaller ("list 3")

FROM_SEP_SMALLER_TEMPLATE = r"""//CL//

inline bool meets_sep_smaller_criterion(
    coord_t root_extent,
    coord_vec_t target_center, int target_level,
    coord_vec_t source_center, int source_level,
    coord_t stick_out_factor)
{
    coord_t target_rad = LEVEL_TO_RAD(target_level);
    coord_t source_rad = LEVEL_TO_RAD(source_level);
    coord_t max_allowed_center_l_inf_dist = (
        3 * target_rad
        + (1 + stick_out_factor) * source_rad);

    coord_t l_inf_dist = 0;
    %for i in range(dimensions):
        l_inf_dist = fmax(
            l_inf_dist,
            fabs(target_center.s${i} - source_center.s${i}));
    %endfor

    return l_inf_dist >= max_allowed_center_l_inf_dist * (1 - 8 * COORD_T_MACH_EPS);
}


void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t target_box_number)
{
    // /!\ target_box_number is *not* a box_id, despite the type.
    // It's the number of the target box we're currently processing.

    box_id_t box_id = target_boxes[target_box_number];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    box_id_t slnws_start = same_level_non_well_sep_boxes_starts[box_id];
    box_id_t slnws_stop = same_level_non_well_sep_boxes_starts[box_id+1];

    // /!\ i is not a box_id, it's an index into same_level_non_well_sep_boxes_lists.
    for (box_id_t i = slnws_start; i < slnws_stop; ++i)
    {
        box_id_t same_lev_nws_box = same_level_non_well_sep_boxes_lists[i];

        if (same_lev_nws_box == box_id)
            continue;

        // Colleagues (same-level NWS boxes) for 1-away are always adjacent, so
        // we always want to descend into them. For 2-away, we may already
        // satisfy the criteria for being in list 3 and therefore may never
        // need to descend. Hence include the start box in the search here
        // if we're in the two-or-more-away case.
        ${walk_init("same_lev_nws_box")}

        while (continue_walk)
        {
            // Loop invariant:
            // walk_parent_box_id is, at first, always adjacent to box_id.
            //
            // This is true at the first level because colleagues are adjacent
            // by definition, and is kept true throughout the walk by only descending
            // into adjacent boxes.
            //
            // As we descend, we may find a child of an adjacent box that is
            // non-adjacent to box_id.
            //
            // If neither sources nor targets have extent, then that
            // nonadjacent child box is added to box_id's from_sep_smaller ("list 3
            // far") and that's it.
            //
            // If they have extent, then while they may be separated, the
            // intersection of box_id's and the child box's stick-out region
            // may be non-empty, and we thus need to add that child to
            // from_sep_close_smaller ("list 3 close") for the interaction to be
            // done by direct evaluation. We also need to descend into that
            // child.

            ${walk_get_box_id()}

            dbg_printf(("  walk parent box id: %d morton: %d child id: %d\n",
                walk_parent_box_id, walk_morton_nr, walk_box_id));

            box_flags_t child_box_flags = box_flags[walk_box_id];

            if (walk_box_id &&
                    (child_box_flags &
                            (BOX_HAS_OWN_SOURCES | BOX_HAS_CHILD_SOURCES)))
            {
                ${load_center("walk_center", "walk_box_id")}

                int walk_level = box_levels[walk_box_id];

                bool in_list_1 = is_adjacent_or_overlapping(root_extent,
                    center, level, walk_center, walk_level);

                if (in_list_1)
                {
                    if (child_box_flags & BOX_HAS_CHILD_SOURCES)
                    {
                        // We want to descend into this box. Put the current state
                        // on the stack.

                        if (walk_level <= from_sep_smaller_source_level
                                || from_sep_smaller_source_level == -1)
                        {
                            ${walk_push("walk_box_id")}
                            continue;
                        }
                        // otherwise there's no point to descending further.
                    }
                }
                else
                {
                    %if sources_have_extent or targets_have_extent:
                        const bool meets_crit =
                            meets_sep_smaller_criterion(root_extent,
                                center, level,
                                walk_center, walk_level,
                                stick_out_factor);
                    %else:
                        const bool meets_crit = true;
                    %endif

                    // We're no longer *immediately* adjacent to our target
                    // box, but our stick-out regions might still have a
                    // non-empty intersection.

                    if (meets_crit)
                    {
                        if (from_sep_smaller_source_level == walk_level)
                            APPEND_from_sep_smaller(walk_box_id);
                    }
                    else
                    {
                    %if sources_have_extent or targets_have_extent:
                        // from_sep_smaller_source_level == -1 means "only build
                        // list 3 close", with sources on any level.
                        // This kernel will be run once per source level to
                        // generate per-level list 3, and once
                        // (not per level) to generate list 3 close.

                        if (
                               (child_box_flags & BOX_HAS_OWN_SOURCES)
                               && (from_sep_smaller_source_level == -1))
                            APPEND_from_sep_close_smaller(walk_box_id);

                        if (child_box_flags & BOX_HAS_CHILD_SOURCES)
                        {
                            ${walk_push("walk_box_id")}
                            continue;
                        }
                    %endif
                    }
                }
            }
            ${walk_advance()}
        }
    }
}
"""

# }}}

# {{{ from separated bigger ("list 4")

# List 4 consists of lists that 'missed the boat' on entering the downward
# propagation through list 2. That is, they are non-well-separated from the
# target box itself or a box in its chain of parents.
#
# To be in list 4, a box must have its own sources. In the no-extents case,
# this will happen only if that box is a leaf, but for the with-extents case,
# any box can have sources.
#
# (Yes, you read that right--same-level non-well separated boxes *can* be in
# list 4, although only for 2+-away. They *could* also use list 3, but that
# would be less efficient because it would not make use of the downward
# propagation.)
#
# For a box not well-separated from the target box or one of its parents, we
# check whether the box is adjacent to our target box (in its list 1).  If so,
# we don't need to consider it (because the interaction to this box will be
# mediated by list 1).
#
# Case I: Sources or targets do not have extent
#
# In this case and once non-membership in list 1 has been verified, list 4
# membership is simply a matter of deciding whether the source box's
# contribution should enter the downward propagation at this target box or
# whether it has already entered it at a parent of the target box.
#
# It suffices to check this for the immediate parent because the check has to
# be monotone: Child boxes are subsets of parent boxes, and therefore any
# minimum distance requirement satisfied by the parent will also be satisfied
# by the child. Thus, if the source box is in the target box's parent's list 4,
# then it entered downward propagation with it or another ancestor.
#
# Case II: Sources or targets have extent
#
# The with-extents case is conceptually similar to the no-extents case, however
# there is an extra 'separation requirement' based on the extents that, if not
# satisfied, may prevent a source box from entering the downward propagation
# at a given box. If we once again assume monotonicity of this 'separation
# requirement' check, then simply verifying whether or not the interaction from
# the source box would be *allowed* to enter the downward propagation at the
# parent suffices to determine whether the target box may be responsible for
# entering the source interaction into the downward propagation.
#
# In cases where the source box is not yet part of the downward propagation
# received from the parent and also not eligible for entering downward
# propagation at this box (noting that this can only happen in the with-extents
# case), the interaction is added to the (non-downward-propagating) 'list 4
# close' (from_sep_close_bigger).


FROM_SEP_BIGGER_TEMPLATE = r"""//CL//

inline bool meets_sep_bigger_criterion(
    coord_t root_extent,
    coord_vec_t target_center, int target_level,
    coord_vec_t source_center, int source_level,
    coord_t stick_out_factor)
{
    coord_t target_rad = LEVEL_TO_RAD(target_level);
    coord_t source_rad = LEVEL_TO_RAD(source_level);
    coord_t max_allowed_center_l_inf_dist = (
        3 * (1 + stick_out_factor) * target_rad
        +  source_rad);

    coord_t l_inf_dist = 0;
    %for i in range(dimensions):
        l_inf_dist = fmax(
            l_inf_dist,
            fabs(target_center.s${i} - source_center.s${i}));
    %endfor

    return l_inf_dist >= max_allowed_center_l_inf_dist * (1 - 8 * COORD_T_MACH_EPS);
}


void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t itarget_or_target_parent_box)
{
    box_id_t tgt_ibox = target_or_target_parent_boxes[itarget_or_target_parent_box];
    ${load_center("tgt_box_center", "tgt_ibox")}

    int tgt_box_level = box_levels[tgt_ibox];
    // The root box has no parents, so no list 4.
    if (tgt_box_level == 0)
        return;

    box_id_t parent_box_id = box_parent_ids[tgt_ibox];
    const int parent_level = tgt_box_level - 1;
    ${load_center("parent_center", "parent_box_id")}

    box_flags_t tgt_box_flags = box_flags[tgt_ibox];

    %if well_sep_is_n_away == 1:
        // In a 1-away FMM, tgt_ibox's colleagues are by default uninteresting
        // (i.e. not in list 4) because they're adjacent. So in this case, we
        // may directly jump to the parent level.

        int walk_level = tgt_box_level - 1;
        box_id_t current_parent_box_id = parent_box_id;
    %else:
        // In a 2+-away FMM, tgt_ibox's same-level well-separated boxes *may*
        // be sufficiently separated from tgt_ibox to be in its list 4.

        int walk_level = tgt_box_level;
        box_id_t current_parent_box_id = tgt_ibox;
    %endif

    /*
    Look for same-level non-well-separated boxes of parents that are
    non-adjacent to tgt_ibox.
    Walk up the tree from tgt_ibox.

    Box 0 (== level 0) doesn't have any slnws boxes, so we can stop the
    search for such slnws boxes there.
    */
    for (; walk_level != 0;
            // {{{ advance
            --walk_level,
            current_parent_box_id = box_parent_ids[current_parent_box_id]
            // }}}
            )
    {
        box_id_t slnws_start =
            same_level_non_well_sep_boxes_starts[current_parent_box_id];
        box_id_t slnws_stop =
            same_level_non_well_sep_boxes_starts[current_parent_box_id+1];

        // /!\ i is not a box id, it's an index into
        // same_level_non_well_sep_boxes_lists.
        for (box_id_t i = slnws_start; i < slnws_stop; ++i)
        {
            box_id_t slnws_box_id = same_level_non_well_sep_boxes_lists[i];

            if (box_flags[slnws_box_id] & BOX_HAS_OWN_SOURCES)
            {
                ${load_center("slnws_center", "slnws_box_id")}

                bool in_list_1 = is_adjacent_or_overlapping(root_extent,
                    tgt_box_center, tgt_box_level,
                    slnws_center, walk_level);

                if (!in_list_1)
                {
                    %if sources_have_extent or targets_have_extent:
                        /*
                        With-extent list 4 separation criterion.
                        Needs to be monotone.  (see main comment narrative
                        above for what that means) If you change this, also
                        change the equivalent check for the parent, below.
                        */
                        const bool tgt_meets_with_ext_sep_criterion =
                            meets_sep_bigger_criterion(root_extent,
                                tgt_box_center, tgt_box_level,
                                slnws_center, walk_level,
                                stick_out_factor);

                    if (!tgt_meets_with_ext_sep_criterion)
                    {
                        /*
                        slnws_box_id failed the separation criterion (i.e.  is
                        too close to the target box) for list 4 proper. Stick
                        it in list 4 close.
                        */

                        if (tgt_box_flags & BOX_HAS_OWN_TARGETS)
                        {
                            APPEND_from_sep_close_bigger(slnws_box_id);
                        }
                    }
                    else
                    %endif
                    {
                        bool in_parent_list_1 =
                            is_adjacent_or_overlapping(root_extent,
                                parent_center, parent_level,
                                slnws_center, walk_level);

                        bool would_be_in_parent_list_4_not_considering_stickout = (
                                !in_parent_list_1
                                %if well_sep_is_n_away > 1:
                                    /*
                                    From-sep-bigger boxes can only be in the
                                    parent's from-sep-bigger list if they're
                                    actually bigger (or equal) to the parent
                                    box size.

                                    For 1-away, that's guaranteed at this
                                    point, because we only start ascending the
                                    tree at the parent's level, so any box we
                                    find here is naturally big enough. For
                                    2-away, we start looking at the target
                                    box's level, so slnws_box_id may actually
                                    be too small (at too deep a level) to be in
                                    the parent's from-sep-bigger list.
                                    */

                                    && walk_level < tgt_box_level
                                %endif
                                );

                        if (would_be_in_parent_list_4_not_considering_stickout)
                        {
                            /*
                            Our immediate parent box was already far enough
                            away to (hypothetically) let the interaction into
                            its downward propagation--so this happened either
                            there or at a more distant ancestor. We'll get the
                            interaction that way. Nothing to do, unless the box
                            was too close to the parent and ended up in the
                            parent's from_sep_close_bigger. If that's the case,
                            we'll simply let it enter the downward propagation
                            here.

                            With-extent list 4 separation criterion.
                            Needs to be monotone.  (see main comment narrative
                            above for what that means) If you change this, also
                            change the equivalent check for the target box, above.
                            */

                            %if sources_have_extent or targets_have_extent:
                                const bool parent_meets_with_ext_sep_criterion =
                                    meets_sep_bigger_criterion(root_extent,
                                        parent_center, parent_level,
                                        slnws_center, walk_level,
                                        stick_out_factor);

                                if (!parent_meets_with_ext_sep_criterion)
                                {
                                    APPEND_from_sep_bigger(slnws_box_id);
                                }
                            %endif
                        }
                        else
                        {
                            /*
                            We're the first box down the chain to be far enough
                            away to let the interaction into our local downward
                            propagation.
                            */
                            APPEND_from_sep_bigger(slnws_box_id);
                        }
                    }
                }
            }
        }
    }
}
"""

# }}}


# {{{ traversal info (output)

class FMMTraversalInfo(DeviceDataRecord):
    """Interaction lists needed for a fast-multipole-like linear-time gather of
    particle interactions.

    Terminology (largely) follows this article:

        Carrier, J., Greengard, L. and Rokhlin, V. "A Fast
        Adaptive Multipole Algorithm for Particle Simulations." SIAM Journal on
        Scientific and Statistical Computing 9, no. 4 (July 1988): 669-686.
        `DOI: 10.1137/0909044 <http://dx.doi.org/10.1137/0909044>`_.

    Unless otherwise indicated, all bulk data in this data structure is stored
    in a :class:`pyopencl.array.Array`. See also :meth:`get`.

    .. attribute:: tree

        An instance of :class:`boxtree.Tree`.

    .. attribute:: well_sep_is_n_away

        The distance (measured in target box diameters in the :math:`l^\infty`
        norm) from the edge of the target box at which the 'well-separated'
        (i.e. M2L-handled) 'far-field' starts.

    .. ------------------------------------------------------------------------
    .. rubric:: Basic box lists for iteration
    .. ------------------------------------------------------------------------

    .. attribute:: source_boxes

        ``box_id_t [*]``

        List of boxes having sources.

    .. attribute:: target_boxes

        ``box_id_t [*]``

        List of boxes having targets.
        If :attr:`boxtree.Tree.sources_are_targets`,
        then ``target_boxes is source_boxes``.

    .. attribute:: source_parent_boxes

        ``box_id_t [*]``

        List of boxes that are (directly or indirectly) a parent
        of one of the :attr:`source_boxes`. These boxes may have sources of their
        own.

    .. attribute:: level_start_source_box_nrs

        ``box_id_t [nlevels+1]``

        Indices into :attr:`source_boxes` indicating where
        each level starts and ends.

    .. attribute:: level_start_source_parent_box_nrs

        ``box_id_t [nlevels+1]``

        Indices into :attr:`source_parent_boxes` indicating where
        each level starts and ends.

    .. attribute:: target_or_target_parent_boxes

        ``box_id_t [*]``

        List of boxes that are one of the :attr:`target_boxes`
        or their (direct or indirect) parents.

    .. attribute:: ntarget_or_target_parent_boxes

        Number of :attr:`target_or_target_parent_boxes`.

    .. attribute:: level_start_target_box_nrs

        ``box_id_t [nlevels+1]``

        Indices into :attr:`target_boxes` indicating where
        each level starts and ends.

    .. attribute:: level_start_target_or_target_parent_box_nrs

        ``box_id_t [nlevels+1]``

        Indices into :attr:`target_or_target_parent_boxes` indicating where
        each level starts and ends.

    .. ------------------------------------------------------------------------
    .. rubric:: Same-level non-well-separated boxes
    .. ------------------------------------------------------------------------

    Boxes considered to be within the 'non-well-separated area' according to
    :attr:`well_sep_is_n_away` that are on the same level as their reference
    box. See :ref:`csr`.

    This is a generalization of the "colleagues" concept from the Carrier paper
    to the case in which :attr:`well_sep_is_n_away` is not 1.

    .. attribute:: same_level_non_well_sep_boxes_starts

        ``box_id_t [nboxes+1]``

    .. attribute:: same_level_non_well_sep_boxes_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Neighbor Sources ("List 1")
    .. ------------------------------------------------------------------------

    List of source boxes immediately adjacent to each target box. Indexed like
    :attr:`target_boxes`. See :ref:`csr`. (Note: This list contains global box
    numbers, not indices into :attr:`source_boxes`.)

    .. attribute:: neighbor_source_boxes_starts

        ``box_id_t [ntarget_boxes+1]``

    .. attribute:: neighbor_source_boxes_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Siblings ("List 2")
    .. ------------------------------------------------------------------------

    Well-separated boxes on the same level.  Indexed like
    :attr:`target_or_target_parent_boxes`. See :ref:`csr`.

    .. attribute:: from_sep_siblings_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]``

    .. attribute:: from_sep_siblings_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Smaller Boxes ("List 3")
    .. ------------------------------------------------------------------------

    Smaller source boxes separated from the target box by their own size.

    If :attr:`boxtree.Tree.targets_have_extent`, then
    :attr:`from_sep_close_smaller_starts` will be non-*None*. It records
    interactions between boxes that would ordinarily be handled
    through "List 3", but must be evaluated specially/directly
    because of :ref:`extent`.

    Indexed like :attr:`target_or_target_parent_boxes`.  See :ref:`csr`.

    .. attribute:: from_sep_smaller_by_level

        A list of :attr:`boxtree.Tree.nlevels` (corresponding to the levels on
        which each listed source box resides) objects, each of which has
        attributes *count*, *starts* and *lists*, which form a CSR list of List
        3 source boxes.

        *starts* has shape/type ``box_id_t [ntarget_boxes+1]``. *lists* is of type
        ``box_id_t``.  (Note: This list contains global box numbers, not
        indices into :attr:`source_boxes`.)

    .. attribute:: from_sep_close_smaller_starts

        ``box_id_t [ntargets+1]`` (or *None*)

    .. attribute:: from_sep_close_smaller_lists

        ``box_id_t [*]`` (or *None*)

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Bigger Boxes ("List 4")
    .. ------------------------------------------------------------------------

    Bigger source boxes separated from the target box by the (smaller) target
    box's size.
    (Note: This list contains global box numbers, not indices into
    :attr:`source_boxes`.)

    If :attr:`boxtree.Tree.sources_have_extent` or
    :attr:`boxtree.Tree.targets_have_extent`, then
    :attr:`from_sep_close_bigger_starts` will be non-*None*. It records
    interactions between boxes that would ordinarily be handled through "List
    4", but must be evaluated specially/directly because of :ref:`extent`.

    Indexed like :attr:`target_or_target_parent_boxes`. See :ref:`csr`.

    .. attribute:: from_sep_bigger_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]``

    .. attribute:: from_sep_bigger_lists

        ``box_id_t [*]``

    .. attribute:: from_sep_close_bigger_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]`` (or *None*)

    .. attribute:: from_sep_close_bigger_lists

        ``box_id_t [*]`` (or *None*)
    """

    # {{{ "close" list merging -> "unified list 1"

    def merge_close_lists(self, queue, debug=False):
        """Return a new :class:`FMMTraversalInfo` instance with the contents of
        :attr:`from_sep_close_smaller_starts` and
        :attr:`from_sep_close_bigger_starts` merged into
        :attr:`neighbor_source_boxes_starts` and these two attributes set to
        *None*.
        """

        from boxtree.tools import reverse_index_array
        target_or_target_parent_boxes_from_all_boxes = reverse_index_array(
                self.target_or_target_parent_boxes, target_size=self.tree.nboxes,
                queue=queue)
        target_or_target_parent_boxes_from_tgt_boxes = cl.array.take(
                target_or_target_parent_boxes_from_all_boxes,
                self.target_boxes, queue=queue)

        del target_or_target_parent_boxes_from_all_boxes

        @memoize_in(self, "merge_close_lists_kernel")
        def get_new_nb_sources_knl(write_counts):
            from pyopencl.elementwise import ElementwiseTemplate
            return ElementwiseTemplate("""//CL:mako//
                /* input: */
                box_id_t *target_or_target_parent_boxes_from_tgt_boxes,
                box_id_t *neighbor_source_boxes_starts,
                box_id_t *from_sep_close_smaller_starts,
                box_id_t *from_sep_close_bigger_starts,

                %if not write_counts:
                    box_id_t *neighbor_source_boxes_lists,
                    box_id_t *from_sep_close_smaller_lists,
                    box_id_t *from_sep_close_bigger_lists,

                    box_id_t *new_neighbor_source_boxes_starts,
                %endif

                /* output: */

                %if write_counts:
                    box_id_t *new_neighbor_source_boxes_counts,
                %else:
                    box_id_t *new_neighbor_source_boxes_lists,
                %endif
                """,
                """//CL:mako//
                box_id_t itgt_box = i;
                box_id_t itarget_or_target_parent_box =
                    target_or_target_parent_boxes_from_tgt_boxes[itgt_box];

                box_id_t neighbor_source_boxes_start =
                    neighbor_source_boxes_starts[itgt_box];
                box_id_t neighbor_source_boxes_count =
                    neighbor_source_boxes_starts[itgt_box + 1]
                    - neighbor_source_boxes_start;

                box_id_t from_sep_close_smaller_start =
                    from_sep_close_smaller_starts[itgt_box];
                box_id_t from_sep_close_smaller_count =
                    from_sep_close_smaller_starts[itgt_box + 1]
                    - from_sep_close_smaller_start;

                box_id_t from_sep_close_bigger_start =
                    from_sep_close_bigger_starts[itarget_or_target_parent_box];
                box_id_t from_sep_close_bigger_count =
                    from_sep_close_bigger_starts[itarget_or_target_parent_box + 1]
                    - from_sep_close_bigger_start;

                %if write_counts:
                    if (itgt_box == 0)
                        new_neighbor_source_boxes_counts[0] = 0;

                    new_neighbor_source_boxes_counts[itgt_box + 1] =
                        neighbor_source_boxes_count
                        + from_sep_close_smaller_count
                        + from_sep_close_bigger_count
                        ;
                %else:

                    box_id_t cur_idx = new_neighbor_source_boxes_starts[itgt_box];

                    #define COPY_FROM(NAME) \
                        for (box_id_t i = 0; i < NAME##_count; ++i) \
                            new_neighbor_source_boxes_lists[cur_idx++] = \
                                NAME##_lists[NAME##_start+i];

                    COPY_FROM(neighbor_source_boxes)
                    COPY_FROM(from_sep_close_smaller)
                    COPY_FROM(from_sep_close_bigger)

                %endif
                """).build(
                        queue.context,
                        type_aliases=(
                            ("box_id_t", self.tree.box_id_dtype),
                            ),
                        var_values=(
                            ("write_counts", write_counts),
                            )
                        )

        ntarget_boxes = len(self.target_boxes)
        new_neighbor_source_boxes_counts = cl.array.empty(
                queue, ntarget_boxes+1, self.tree.box_id_dtype)
        get_new_nb_sources_knl(True)(
            # input:
            target_or_target_parent_boxes_from_tgt_boxes,
            self.neighbor_source_boxes_starts,
            self.from_sep_close_smaller_starts,
            self.from_sep_close_bigger_starts,

            # output:
            new_neighbor_source_boxes_counts,
            range=slice(ntarget_boxes),
            queue=queue)

        new_neighbor_source_boxes_starts = cl.array.cumsum(
                new_neighbor_source_boxes_counts)
        del new_neighbor_source_boxes_counts

        new_neighbor_source_boxes_lists = cl.array.empty(
                queue,
                int(new_neighbor_source_boxes_starts[ntarget_boxes].get()),
                self.tree.box_id_dtype)

        new_neighbor_source_boxes_lists.fill(999999999)

        get_new_nb_sources_knl(False)(
            # input:
            target_or_target_parent_boxes_from_tgt_boxes,

            self.neighbor_source_boxes_starts,
            self.from_sep_close_smaller_starts,
            self.from_sep_close_bigger_starts,
            self.neighbor_source_boxes_lists,
            self.from_sep_close_smaller_lists,
            self.from_sep_close_bigger_lists,

            new_neighbor_source_boxes_starts,

            # output:
            new_neighbor_source_boxes_lists,
            range=slice(ntarget_boxes),
            queue=queue)

        return self.copy(
            neighbor_source_boxes_starts=new_neighbor_source_boxes_starts,
            neighbor_source_boxes_lists=new_neighbor_source_boxes_lists,
            from_sep_close_smaller_starts=None,
            from_sep_close_smaller_lists=None,
            from_sep_close_bigger_starts=None,
            from_sep_close_bigger_lists=None)

    # }}}

    # {{{ debugging aids

    def get_box_list(self, what, index):
        starts = getattr(self, what+"_starts")
        lists = getattr(self, what+"_lists")
        start, stop = starts[index:index+2]
        return lists[start:stop]

    # }}}

    @property
    def ntarget_or_target_parent_boxes(self):
        return len(self.target_or_target_parent_boxes)

# }}}


class _KernelInfo(Record):
    pass


class FMMTraversalBuilder:
    def __init__(self, context, well_sep_is_n_away=1):
        self.context = context
        self.well_sep_is_n_away = well_sep_is_n_away

    # {{{ kernel builder

    @memoize_method
    def get_kernel_info(self, dimensions, particle_id_dtype, box_id_dtype,
            coord_dtype, box_level_dtype, max_levels,
            sources_are_targets, sources_have_extent, targets_have_extent):

        logger.info("traversal build kernels: start build")

        debug = False

        from pyopencl.tools import dtype_to_ctype
        from boxtree.tree import box_flags_enum
        render_vars = dict(
                np=np,
                dimensions=dimensions,
                dtype_to_ctype=dtype_to_ctype,
                particle_id_dtype=particle_id_dtype,
                box_id_dtype=box_id_dtype,
                box_flags_enum=box_flags_enum,
                coord_dtype=coord_dtype,
                vec_types=cl.cltypes.vec_types,
                max_levels=max_levels,
                AXIS_NAMES=AXIS_NAMES,
                debug=debug,
                sources_are_targets=sources_are_targets,
                sources_have_extent=sources_have_extent,
                targets_have_extent=targets_have_extent,
                well_sep_is_n_away=self.well_sep_is_n_away,
                )
        from pyopencl.algorithm import ListOfListsBuilder
        from pyopencl.tools import VectorArg, ScalarArg

        result = {}

        # {{{ source boxes, their parents, target boxes

        src = Template(
                TRAVERSAL_PREAMBLE_TEMPLATE
                + SOURCES_PARENTS_AND_TARGETS_TEMPLATE,
                strict_undefined=True).render(**render_vars)

        result["sources_parents_and_targets_builder"] = \
                ListOfListsBuilder(self.context,
                        [
                            ("source_parent_boxes", box_id_dtype),
                            ("source_boxes", box_id_dtype),
                            ("target_or_target_parent_boxes", box_id_dtype)
                            ] + (
                                [("target_boxes", box_id_dtype)]
                                if not sources_are_targets
                                else []),
                        str(src),
                        arg_decls=[
                            VectorArg(box_flags_enum.dtype, "box_flags"),
                            ],
                        debug=debug,
                        name_prefix="sources_parents_and_targets")

        result["level_start_box_nrs_extractor"] = \
                LEVEL_START_BOX_NR_EXTRACTOR_TEMPLATE.build(self.context,
                    type_aliases=(
                        ("box_id_t", box_id_dtype),
                        ("box_level_t", box_level_dtype),
                        ),
                    )

        # }}}

        # {{{ build list N builders

        base_args = [
                VectorArg(coord_dtype, "box_centers"),
                ScalarArg(coord_dtype, "root_extent"),
                VectorArg(np.uint8, "box_levels"),
                ScalarArg(box_id_dtype, "aligned_nboxes"),
                VectorArg(box_id_dtype, "box_child_ids"),
                VectorArg(box_flags_enum.dtype, "box_flags"),
                ]

        for list_name, template, extra_args, extra_lists in [
                ("same_level_non_well_sep_boxes",
                    SAME_LEVEL_NON_WELL_SEP_BOXES_TEMPLATE, [], []),
                ("neighbor_source_boxes", NEIGBHOR_SOURCE_BOXES_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_boxes"),
                            ], []),
                ("from_sep_siblings", FROM_SEP_SIBLINGS_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_or_target_parent_boxes"),
                            VectorArg(box_id_dtype, "box_parent_ids"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_starts"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_lists"),
                            ], []),
                ("from_sep_smaller", FROM_SEP_SMALLER_TEMPLATE,
                        [
                            ScalarArg(coord_dtype, "stick_out_factor"),
                            VectorArg(box_id_dtype, "target_boxes"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_starts"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_lists"),
                            ScalarArg(box_id_dtype, "from_sep_smaller_source_level"),
                            ],
                            ["from_sep_close_smaller"]
                            if sources_have_extent or targets_have_extent
                            else []),
                ("from_sep_bigger", FROM_SEP_BIGGER_TEMPLATE,
                        [
                            ScalarArg(coord_dtype, "stick_out_factor"),
                            VectorArg(box_id_dtype, "target_or_target_parent_boxes"),
                            VectorArg(box_id_dtype, "box_parent_ids"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_starts"),
                            VectorArg(box_id_dtype,
                                "same_level_non_well_sep_boxes_lists"),
                            #ScalarArg(box_id_dtype, "from_sep_bigger_source_level"),
                            ],
                            ["from_sep_close_bigger"]
                            if sources_have_extent or targets_have_extent
                            else []),
                ]:
            src = Template(
                    TRAVERSAL_PREAMBLE_TEMPLATE
                    + HELPER_FUNCTION_TEMPLATE
                    + template,
                    strict_undefined=True).render(**render_vars)

            result[list_name+"_builder"] = ListOfListsBuilder(self.context,
                    [(list_name, box_id_dtype)]
                    + [(extra_list_name, box_id_dtype)
                        for extra_list_name in extra_lists],
                    str(src),
                    arg_decls=base_args + extra_args,
                    debug=debug, name_prefix=list_name,
                    complex_kernel=True)

        # }}}

        logger.info("traversal build kernels: done")

        return _KernelInfo(**result)

    # }}}

    # {{{ driver

    def __call__(self, queue, tree, wait_for=None, debug=False):
        """
        :arg queue: A :class:`pyopencl.CommandQueue` instance.
        :arg tree: A :class:`boxtree.Tree` instance.
        :arg wait_for: may either be *None* or a list of :class:`pyopencl.Event`
            instances for whose completion this command waits before starting
            exeuction.
        :return: A tuple *(trav, event)*, where *trav* is a new instance of
            :class:`FMMTraversalInfo` and *event* is a :class:`pyopencl.Event`
            for dependency management.
        """

        if not tree._is_pruned:
            raise ValueError("tree must be pruned for traversal generation")

        if tree.sources_have_extent:
            # YAGNI
            raise NotImplementedError(
                    "trees with source extent are not supported for "
                    "traversal generation")

        # Generated code shouldn't depend on the *exact* number of tree levels.
        # So round up to the next multiple of 5.
        from pytools import div_ceil
        max_levels = div_ceil(tree.nlevels, 5) * 5

        knl_info = self.get_kernel_info(
                tree.dimensions, tree.particle_id_dtype, tree.box_id_dtype,
                tree.coord_dtype, tree.box_level_dtype, max_levels,
                tree.sources_are_targets,
                tree.sources_have_extent, tree.targets_have_extent)

        def fin_debug(s):
            if debug:
                queue.finish()

            logger.debug(s)

        logger.info("start building traversal")

        # {{{ source boxes, their parents, and target boxes

        fin_debug("building list of source boxes, their parents, and target boxes")

        result, evt = knl_info.sources_parents_and_targets_builder(
                queue, tree.nboxes, tree.box_flags.data, wait_for=wait_for)
        wait_for = [evt]

        source_parent_boxes = result["source_parent_boxes"].lists
        source_boxes = result["source_boxes"].lists
        target_or_target_parent_boxes = result["target_or_target_parent_boxes"].lists

        if not tree.sources_are_targets:
            target_boxes = result["target_boxes"].lists
        else:
            target_boxes = source_boxes

        # }}}

        # {{{ figure out level starts in *_parent_boxes

        def extract_level_start_box_nrs(box_list, wait_for):
            result = cl.array.empty(queue,
                    tree.nlevels+1, tree.box_id_dtype) \
                            .fill(len(box_list))
            evt = knl_info.level_start_box_nrs_extractor(
                    tree.level_start_box_nrs_dev,
                    tree.box_levels,
                    box_list,
                    result,
                    range=slice(0, len(box_list)),
                    queue=queue, wait_for=wait_for)

            result = result.get()

            # Postprocess result for unoccupied levels
            prev_start = len(box_list)
            for ilev in range(tree.nlevels-1, -1, -1):
                result[ilev] = prev_start = \
                        min(result[ilev], prev_start)

            return result, evt

        fin_debug("finding level starts in source boxes array")
        level_start_source_box_nrs, evt_s = \
                extract_level_start_box_nrs(
                        source_boxes, wait_for=wait_for)

        fin_debug("finding level starts in source parent boxes array")
        level_start_source_parent_box_nrs, evt_sp = \
                extract_level_start_box_nrs(
                        source_parent_boxes, wait_for=wait_for)

        fin_debug("finding level starts in target boxes array")
        level_start_target_box_nrs, evt_t = \
                extract_level_start_box_nrs(
                        target_boxes, wait_for=wait_for)

        fin_debug("finding level starts in target or target parent boxes array")
        level_start_target_or_target_parent_box_nrs, evt_tp = \
                extract_level_start_box_nrs(
                        target_or_target_parent_boxes, wait_for=wait_for)

        wait_for = [evt_s, evt_sp, evt_t, evt_tp]

        # }}}

        # {{{ same-level near-field

        # If well_sep_is_n_away is 1, this agrees with the definition of
        # 'colleagues' from the classical FMM literature.

        fin_debug("finding same-level near-field boxes")

        result, evt = knl_info.same_level_non_well_sep_boxes_builder(
                queue, tree.nboxes,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                wait_for=wait_for)
        wait_for = [evt]
        same_level_non_well_sep_boxes = result["same_level_non_well_sep_boxes"]

        # }}}

        # {{{ neighbor source boxes ("list 1")

        fin_debug("finding neighbor source boxes ('list 1')")

        result, evt = knl_info.neighbor_source_boxes_builder(
                queue, len(target_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                target_boxes.data, wait_for=wait_for)

        wait_for = [evt]
        neighbor_source_boxes = result["neighbor_source_boxes"]

        # }}}

        # {{{ well-separated siblings ("list 2")

        fin_debug("finding well-separated siblings ('list 2')")

        result, evt = knl_info.from_sep_siblings_builder(
                queue, len(target_or_target_parent_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                target_or_target_parent_boxes.data, tree.box_parent_ids.data,
                same_level_non_well_sep_boxes.starts.data,
                same_level_non_well_sep_boxes.lists.data,
                wait_for=wait_for)
        wait_for = [evt]
        from_sep_siblings = result["from_sep_siblings"]

        # }}}

        with_extent = tree.sources_have_extent or tree.targets_have_extent

        # {{{ separated smaller ("list 3")

        fin_debug("finding separated smaller ('list 3')")

        from_sep_smaller_base_args = (
                queue, len(target_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                tree.stick_out_factor, target_boxes.data,
                same_level_non_well_sep_boxes.starts.data,
                same_level_non_well_sep_boxes.lists.data,
                )

        from_sep_smaller_wait_for = []
        from_sep_smaller_by_level = []

        for ilevel in range(tree.nlevels):
            fin_debug("finding separated smaller ('list 3 level %d')" % ilevel)

            result, evt = knl_info.from_sep_smaller_builder(
                    *(from_sep_smaller_base_args + (ilevel,)),
                    omit_lists=("from_sep_close_smaller",) if with_extent else (),
                    wait_for=wait_for)

            from_sep_smaller_by_level.append(result["from_sep_smaller"])
            from_sep_smaller_wait_for.append(evt)

        if with_extent:
            fin_debug("finding separated smaller close ('list 3 close')")
            result, evt = knl_info.from_sep_smaller_builder(
                    *(from_sep_smaller_base_args + (-1,)),
                    omit_lists=("from_sep_smaller",),
                    wait_for=wait_for)
            from_sep_close_smaller_starts = result["from_sep_close_smaller"].starts
            from_sep_close_smaller_lists = result["from_sep_close_smaller"].lists

            from_sep_smaller_wait_for.append(evt)
        else:
            from_sep_close_smaller_starts = None
            from_sep_close_smaller_lists = None

        # }}}

        wait_for = from_sep_smaller_wait_for
        del from_sep_smaller_wait_for

        # {{{ separated bigger ("list 4")

        fin_debug("finding separated bigger ('list 4')")

        result, evt = knl_info.from_sep_bigger_builder(
                queue, len(target_or_target_parent_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                tree.stick_out_factor, target_or_target_parent_boxes.data,
                tree.box_parent_ids.data,
                same_level_non_well_sep_boxes.starts.data,
                same_level_non_well_sep_boxes.lists.data,
                wait_for=wait_for)

        wait_for = [evt]
        from_sep_bigger = result["from_sep_bigger"]

        if with_extent:
            from_sep_close_bigger_starts = result["from_sep_close_bigger"].starts
            from_sep_close_bigger_lists = result["from_sep_close_bigger"].lists
        else:
            from_sep_close_bigger_starts = None
            from_sep_close_bigger_lists = None

        # }}}

        if self.well_sep_is_n_away == 1:
            colleagues_starts = same_level_non_well_sep_boxes.starts
            colleagues_lists = same_level_non_well_sep_boxes.lists
        else:
            colleagues_starts = None
            colleagues_lists = None

        evt, = wait_for

        logger.info("traversal built")

        return FMMTraversalInfo(
                tree=tree,
                well_sep_is_n_away=self.well_sep_is_n_away,

                source_boxes=source_boxes,
                target_boxes=target_boxes,

                level_start_source_box_nrs=level_start_source_box_nrs,
                level_start_target_box_nrs=level_start_target_box_nrs,

                source_parent_boxes=source_parent_boxes,
                level_start_source_parent_box_nrs=level_start_source_parent_box_nrs,

                target_or_target_parent_boxes=target_or_target_parent_boxes,
                level_start_target_or_target_parent_box_nrs=(
                    level_start_target_or_target_parent_box_nrs),

                same_level_non_well_sep_boxes_starts=(
                    same_level_non_well_sep_boxes.starts),
                same_level_non_well_sep_boxes_lists=(
                    same_level_non_well_sep_boxes.lists),
                # Deprecated, but we'll keep these alive for the time being.
                colleagues_starts=colleagues_starts,
                colleagues_lists=colleagues_lists,

                neighbor_source_boxes_starts=neighbor_source_boxes.starts,
                neighbor_source_boxes_lists=neighbor_source_boxes.lists,

                from_sep_siblings_starts=from_sep_siblings.starts,
                from_sep_siblings_lists=from_sep_siblings.lists,

                from_sep_smaller_by_level=from_sep_smaller_by_level,

                from_sep_close_smaller_starts=from_sep_close_smaller_starts,
                from_sep_close_smaller_lists=from_sep_close_smaller_lists,

                from_sep_bigger_starts=from_sep_bigger.starts,
                from_sep_bigger_lists=from_sep_bigger.lists,

                from_sep_close_bigger_starts=from_sep_close_bigger_starts,
                from_sep_close_bigger_lists=from_sep_close_bigger_lists,
                ).with_queue(None), evt

    # }}}

# vim: filetype=pyopencl:fdm=marker
