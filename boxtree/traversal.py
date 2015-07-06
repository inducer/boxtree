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
from pytools import Record, memoize_method, memoize_method_nested
import pyopencl as cl
import pyopencl.array  # noqa
from pyopencl.elementwise import ElementwiseTemplate
from mako.template import Template
from boxtree.tools import AXIS_NAMES, DeviceDataRecord

import logging
logger = logging.getLogger(__name__)


# {{{ preamble

TRAVERSAL_PREAMBLE_TEMPLATE = r"""//CL//
${box_flags_enum.get_c_defines()}
${box_flags_enum.get_c_typedef()}

typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
%if particle_id_dtype is not None:
    typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;
%endif
typedef ${dtype_to_ctype(coord_dtype)} coord_t;
typedef ${dtype_to_ctype(vec_types[coord_dtype, dimensions])} coord_vec_t;

#define NLEVELS ${max_levels}
#define STICK_OUT_FACTOR ((coord_t) ${stick_out_factor})

<%def name="load_center(name, box_id)">
    coord_vec_t ${name};
    %for i in range(dimensions):
        ${name}.${AXIS_NAMES[i]} = box_centers[aligned_nboxes * ${i} + ${box_id}];
    %endfor
</%def>

#define LEVEL_TO_RAD(level) \
        (root_extent * 1 / (coord_t) (1 << (level + 1)))

%if 0:
    #define dbg_printf(ARGS) printf ARGS
%else:
    #define dbg_printf(ARGS) /* */
%endif

<%def name="walk_init(start_box_id)">
    box_id_t box_stack[NLEVELS];
    int morton_nr_stack[NLEVELS];

    // start at root
    int walk_level = 0;
    box_id_t walk_box_id = ${start_box_id};
    int walk_morton_nr = 0;
    bool continue_walk = true;
</%def>

<%def name="walk_advance()">
    while (true)
    {
        ++walk_morton_nr;
        if (walk_morton_nr < ${2**dimensions})
            break;

        // Ran out of children, pull the next guy off the stack
        // and advance him.

        continue_walk = walk_level > 0;
        if (continue_walk)
        {
            --walk_level;
            dbg_printf(("    ascend\n"));
            walk_box_id = box_stack[walk_level];
            walk_morton_nr = morton_nr_stack[walk_level];
        }
        else
        {
            dbg_printf(("done\n"));
            break;
        }
    }
</%def>

<%def name="walk_push(new_box)">
    box_stack[walk_level] = walk_box_id;
    morton_nr_stack[walk_level] = walk_morton_nr;
    ++walk_level;

    %if debug:
    if (walk_level >= NLEVELS)
    {
        dbg_printf(("  ** ERROR: overran levels stack\n"));
        return;
    }
    %endif

    walk_box_id = ${new_box};
    walk_morton_nr = 0;
</%def>

"""

# }}}

# {{{ adjacency test

HELPER_FUNCTION_TEMPLATE = r"""//CL//

inline bool is_adjacent_or_overlapping(
    coord_t root_extent,
    // target and source order only matter if include_stick_out is true.
    coord_vec_t target_center, int target_level,
    coord_vec_t source_center, int source_level,
    // this is expected to be constant so that the inliner will kill the if.
    const bool include_stick_out
    )
{
    // This checks if the two boxes overlap
    // with an amount of 'slack' corresponding to half the
    // width of the smaller of the two boxes.
    // (Without the 'slack', there wouldn't be any
    // overlap.)

    coord_t target_rad = LEVEL_TO_RAD(target_level);
    coord_t source_rad = LEVEL_TO_RAD(source_level);
    coord_t rad_sum = target_rad + source_rad;
    coord_t slack = rad_sum + fmin(target_rad, source_rad);

    if (include_stick_out)
    {
        slack += STICK_OUT_FACTOR * (
            0
            %if targets_have_extent:
                + target_rad
            %endif
            %if sources_have_extent:
                + source_rad
            %endif
            );
    }

    coord_t max_dist = 0;
    %for i in range(dimensions):
        max_dist = fmax(max_dist, fabs(target_center.s${i} - source_center.s${i}));
    %endfor

    return max_dist <= slack;
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
        box_id_t prev_box_id = box_list[i-1];

        int my_level = box_levels[my_box_id];
        box_id_t my_level_start = level_start_box_nrs[my_level];

        if (prev_box_id < my_level_start && my_level_start <= my_box_id)
            list_level_start_box_nrs[my_level] = i;
    """,
    name="extract_level_start_box_nrs")

# }}}

# {{{ colleagues

COLLEAGUES_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    ${load_center("center", "box_id")}

    if (box_id == 0)
    {
        // The root has no colleagues.
        return;
    }

    int level = box_levels[box_id];

    dbg_printf(("box id: %d level: %d\n", box_id, level));

    // To find this box's colleagues, start at the top of the tree, descend
    // into adjacent (or overlapping) parents.
    ${walk_init(0)}

    while (continue_walk)
    {
        box_id_t child_box_id = box_child_ids[
                walk_morton_nr * aligned_nboxes + walk_box_id];
        dbg_printf(("  level: %d walk box id: %d morton: %d child id: %d\n",
            walk_level, walk_box_id, walk_morton_nr, child_box_id));

        if (child_box_id)
        {
            ${load_center("child_center", "child_box_id")}

            bool a_or_o = is_adjacent_or_overlapping(root_extent,
                center, level, child_center, box_levels[child_box_id], false);

            if (a_or_o)
            {
                // child_box_id lives on walk_level+1.
                if (walk_level+1 == level  && child_box_id != box_id)
                {
                    dbg_printf(("    colleague\n"));
                    APPEND_colleagues(child_box_id);
                }
                else
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    dbg_printf(("    descend\n"));
                    ${walk_push("child_box_id")}

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

    // To find this box's colleagues, start at the top of the tree, descend
    // into adjacent (or overlapping) parents.
    ${walk_init(0)}

    while (continue_walk)
    {
        box_id_t child_box_id = box_child_ids[
                walk_morton_nr * aligned_nboxes + walk_box_id];

        dbg_printf(("  walk box id: %d morton: %d child id: %d level: %d\n",
            walk_box_id, walk_morton_nr, child_box_id, walk_level));

        if (child_box_id)
        {
            ${load_center("child_center", "child_box_id")}

            bool a_or_o = is_adjacent_or_overlapping(root_extent,
                center, level, child_center, box_levels[child_box_id], false);

            if (a_or_o)
            {
                box_flags_t flags = box_flags[child_box_id];
                /* child_box_id == box_id is ok */
                if (flags & BOX_HAS_OWN_SOURCES)
                {
                    dbg_printf(("    neighbor source box\n"));

                    APPEND_neighbor_source_boxes(child_box_id);
                }

                if (flags & BOX_HAS_CHILD_SOURCES)
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    dbg_printf(("    descend\n"));

                    ${walk_push("child_box_id")}

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

# {{{ well-separated siblings ("list 2")

SEP_SIBLINGS_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t itarget_or_target_parent_box)
{
    box_id_t box_id = target_or_target_parent_boxes[itarget_or_target_parent_box];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    box_id_t parent = box_parent_ids[box_id];
    if (parent == box_id)
        return;

    box_id_t parent_coll_start = colleagues_starts[parent];
    box_id_t parent_coll_stop = colleagues_starts[parent+1];

    // /!\ i is not a box_id, it's an index into colleagues_list.
    for (box_id_t i = parent_coll_start; i < parent_coll_stop; ++i)
    {
        box_id_t parent_colleague = colleagues_list[i];

        for (int morton_nr = 0; morton_nr < ${2**dimensions}; ++morton_nr)
        {
            box_id_t sib_box_id = box_child_ids[
                    morton_nr * aligned_nboxes + parent_colleague];

            ${load_center("sib_center", "sib_box_id")}

            bool sep = !is_adjacent_or_overlapping(root_extent,
                center, level, sib_center, box_levels[sib_box_id], false);

            if (sep)
            {
                APPEND_sep_siblings(sib_box_id);
            }
        }
    }
}
"""

# }}}

# {{{ separated smaller ("list 3")

SEP_SMALLER_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t target_box_number)
{
    // /!\ target_box_number is *not* a box_id, despite the type.
    // It's the number of the target box we're currently processing.

    box_id_t box_id = target_boxes[target_box_number];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    box_id_t coll_start = colleagues_starts[box_id];
    box_id_t coll_stop = colleagues_starts[box_id+1];

    // /!\ i is not a box_id, it's an index into colleagues_list.
    for (box_id_t i = coll_start; i < coll_stop; ++i)
    {
        box_id_t colleague = colleagues_list[i];

        ${walk_init("colleague")}

        while (continue_walk)
        {
            // Loop invariant: walk_box_id is, at first, always adjacent to box_id.
            // This is true at the first level because colleagues are by adjacent
            // by definition, and is kept true throughout the walk by only descending
            // into adjacent boxes.
            //
            // As we descend, we may find a child of an adjacent box that is
            // non-adjacent to box_id.
            //
            // If neither sources nor targets have extent, then that
            // nonadjacent child box is added to box_id's sep_smaller ("list 3
            // far") and that's it.
            //
            // If they have extent, then while they may be separated, the
            // intersection of box_id's and the child box's stick-out region
            // may be non-empty, and we thus need to add that child to
            // sep_close_smaller ("list 3 close") for the interaction to be
            // done by direct evaluation. We also need to descend into that
            // child.

            box_id_t child_box_id = box_child_ids[
                    walk_morton_nr * aligned_nboxes + walk_box_id];

            dbg_printf(("  walk box id: %d morton: %d child id: %d\n",
                walk_box_id, walk_morton_nr, child_box_id));

            box_flags_t child_box_flags = box_flags[child_box_id];

            if (child_box_id &&
                    (child_box_flags &
                            (BOX_HAS_OWN_SOURCES | BOX_HAS_CHILD_SOURCES)))
            {
                ${load_center("child_center", "child_box_id")}

                bool a_or_o = is_adjacent_or_overlapping(root_extent,
                    center, level, child_center, box_levels[child_box_id], false);

                if (a_or_o)
                {
                    if (child_box_flags & BOX_HAS_CHILD_SOURCES)
                    {
                        // We want to descend into this box. Put the current state
                        // on the stack.

                        ${walk_push("child_box_id")}
                        continue;
                    }
                }
                else
                {
                    %if sources_have_extent or targets_have_extent:
                        const bool a_or_o_with_stick_out =
                            is_adjacent_or_overlapping(root_extent,
                                center, level, child_center,
                                box_levels[child_box_id], true);
                    %else:
                        const bool a_or_o_with_stick_out = false;
                    %endif

                    // We're no longer *immediately* adjacent to our target
                    // box, but our stick-out regions might still have a
                    // non-empty intersection.

                    if (!a_or_o_with_stick_out)
                    {
                        APPEND_sep_smaller(child_box_id);
                    }
                    else
                    {
                    %if sources_have_extent or targets_have_extent:
                        if (child_box_flags & BOX_HAS_OWN_SOURCES)
                        {
                            APPEND_sep_close_smaller(child_box_id);
                        }

                        if (child_box_flags & BOX_HAS_CHILD_SOURCES)
                        {
                            ${walk_push("child_box_id")}
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

# {{{ separated bigger ("list 4")

# "Normal" case: Sources/targets without extent
# ---------------------------------------------
#
# List 4 interactions for box "B" are about a parent P's colleague A not
# adjacent to B.
#
# -------|----------|----------|
# Case   |    1     |    2     |
#        | adj to A | adj to A |
# -------|----------|----------|
#        |          |          |
# A---P  |    X !   |    X !   |
#     |  |          |          |
#     o  |    X     |    X     |
#     |  |          |          |
#     o  |    X     |    X     |
#     |  |          |          |
#     o  |    X     |    O     |
#     |  |          |          |
#     B  |    O !   |    O !   |
#
# Note that once a parent is no longer adjacent, its children won't be either.
#
# (X: yes, O:no, exclamation marks denote that this *must* be the case. Entries
# without exclamation mark are choices for this case)
#
# Case 1: A->B interaction enters the downward propagation at B, i.e. A is in
#    B's "sep_bigger". (list 4)
#
# Case 2: A->B interaction entered the downward propagation at B's parent, i.e.
#    A is not in B's "sep_bigger". (list 4)

# Sources/targets with extent
# ---------------------------
#
# List 4 interactions for box "B" are about a parent P's colleague A not
# adjacent to B.
#
# -------|----------|----------|----------|
# Case   |    1     |    2     |    3     |
#        | so   adj | so   adj | so   adj |
# -------|----------|----------|----------|
#        |          |          |          |
# A---P  | X!    X! | X!    X! | X!    X! |
#     |  |          |          |          |
#     o  | X     ?  | X     ?  | X     ?  |
#     |  |          |          |          |
#     o  | X     ?  | X     ?  | X     ?  |
#     |  |          |          |          |
#     o  | X     ?  | X     ?  | O     O  |
#     |  |          |          |          |
#     B  | X     O! | O     O! | O     O! |
#
# "so": adjacent or overlapping when stick-out is taken into account (to A)
# "adj": adjacent to A without stick-out
#
# Note that once a parent is no longer "adj" or "so", its children won't be
# either.  Also note that "adj" => "so". (And there by "not so" => "not adj".)
#
# (X: yes, O:no, ?: doesn't matter, exclamation marks denote that this *must*
# be the case. Entries without exclamation mark are choices for this case)
#
# Case 1: A->B interaction must be processed by direct eval because of "so",
#    i.e. it is in B's "sep_close_bigger".
#
# Case 2: A->B interaction enters downward the propagation at B,
#    i.e. it is in B's "sep_bigger".
#
# Case 3: A->B interaction enters downward the propagation at B's parent,
#    i.e. A is not in B's "sep*bigger"

SEP_BIGGER_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t itarget_or_target_parent_box)
{
    box_id_t tgt_ibox = target_or_target_parent_boxes[itarget_or_target_parent_box];
    ${load_center("center", "tgt_ibox")}

    int box_level = box_levels[tgt_ibox];
    // The root box has no parents, so no list 4.
    if (box_level == 0)
        return;

    box_id_t parent_box_id = box_parent_ids[tgt_ibox];
    ${load_center("parent_center", "parent_box_id")}

    box_id_t current_parent_box_id = parent_box_id;
    int walk_level = box_level - 1;

    box_flags_t tgt_box_flags = box_flags[tgt_ibox];

    // Look for colleagues of parents that are non-adjacent to tgt_ibox.
    // Walk up the tree from tgt_ibox.

    // Box 0 (== level 0) doesn't have any colleagues, so we can stop the
    // search for such colleagues there.
    for (int walk_level = box_level - 1; walk_level != 0;
            // {{{ advance
            --walk_level,
            current_parent_box_id = box_parent_ids[current_parent_box_id]
            // }}}
            )
    {
        box_id_t coll_start = colleagues_starts[current_parent_box_id];
        box_id_t coll_stop = colleagues_starts[current_parent_box_id+1];

        // /!\ i is not a box id, it's an index into colleagues_list.
        for (box_id_t i = coll_start; i < coll_stop; ++i)
        {
            box_id_t colleague_box_id = colleagues_list[i];

            if (box_flags[colleague_box_id] & BOX_HAS_OWN_SOURCES)
            {
                ${load_center("colleague_center", "colleague_box_id")}
                bool a_or_o = is_adjacent_or_overlapping(root_extent,
                    center, box_level, colleague_center, walk_level, false);

                if (!a_or_o)
                {
                    // Found one.

                    %if sources_have_extent or targets_have_extent:
                        const bool a_or_o_with_stick_out =
                            is_adjacent_or_overlapping(root_extent,
                                center, box_level, colleague_center,
                                walk_level, true);

                    if (a_or_o_with_stick_out)
                    {
                        // "Case 1" above: colleague_box_id is too close and
                        // overlaps our stick_out region. We're obliged to do
                        // the interaction directly.

                        if (tgt_box_flags & BOX_HAS_OWN_TARGETS)
                        {
                            APPEND_sep_close_bigger(colleague_box_id);
                        }
                    }
                    else
                    %endif
                    {
                        bool parent_a_or_o_with_stick_out =
                            is_adjacent_or_overlapping(root_extent,
                                parent_center, box_level-1, colleague_center,
                                walk_level, true);

                        if (parent_a_or_o_with_stick_out)
                        {
                            // "Case 2" above: We're the first box down the chain
                            // to be far enough away to let the interaction into
                            // our local downward subtree.
                            APPEND_sep_bigger(colleague_box_id);
                        }
                        else
                        {
                            // "Case 2" above: A parent box was already far
                            // enough away to let the interaction into its
                            // local downward subtree. We'll get the interaction
                            // that way. Nothing to do.
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

    Terminology follows this article:

        Carrier, J., Greengard, L. and Rokhlin, V. "A Fast
        Adaptive Multipole Algorithm for Particle Simulations." SIAM Journal on
        Scientific and Statistical Computing 9, no. 4 (July 1988): 669-686.
        `DOI: 10.1137/0909044 <http://dx.doi.org/10.1137/0909044>`_.

    Unless otherwise indicated, all bulk data in this data structure is stored
    in a :class:`pyopencl.array.Array`. See also :meth:`get`.

    .. attribute:: tree

        An instance of :class:`boxtree.Tree`.

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

    .. attribute:: level_start_target_or_target_parent_box_nrs

        ``box_id_t [nlevels+1]``

        Indices into :attr:`target_or_target_parent_boxes` indicating where
        each level starts and ends.

    .. ------------------------------------------------------------------------
    .. rubric:: Colleagues
    .. ------------------------------------------------------------------------

    Immediately adjacent boxes on the same level. See :ref:`csr`.

    .. attribute:: colleagues_starts

        ``box_id_t [nboxes+1]``

    .. attribute:: colleagues_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Neighbor Sources ("List 1")
    .. ------------------------------------------------------------------------

    List of source boxes immediately adjacent to each target box. Indexed like
    :attr:`target_boxes`. See :ref:`csr`.

    .. attribute:: neighbor_source_boxes_starts

        ``box_id_t [ntarget_boxes+1]``

    .. attribute:: neighbor_source_boxes_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Siblings ("List 2")
    .. ------------------------------------------------------------------------

    Well-separated boxes on the same level.  Indexed like
    :attr:`target_or_target_parent_boxes`. See :ref:`csr`.

    .. attribute:: sep_siblings_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]``

    .. attribute:: sep_siblings_lists

        ``box_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Smaller Boxes ("List 3")
    .. ------------------------------------------------------------------------

    Smaller source boxes separated from the target box by their own size.

    If :attr:`boxtree.Tree.targets_have_extent`, then
    :attr:`sep_close_smaller_starts` will be non-*None*. It records
    interactions between boxes that would ordinarily be handled
    through "List 3", but must be evaluated specially/directly
    because of :ref:`extent`.

    Indexed like :attr:`target_or_target_parent_boxes`.  See :ref:`csr`.

    .. attribute:: sep_smaller_starts

        ``box_id_t [ntargets+1]``

    .. attribute:: sep_smaller_lists

        ``box_id_t [*]``

    .. attribute:: sep_close_smaller_starts

        ``box_id_t [ntargets+1]`` (or *None*)

    .. attribute:: sep_close_smaller_lists

        ``box_id_t [*]`` (or *None*)

    .. ------------------------------------------------------------------------
    .. rubric:: Separated Bigger Boxes ("List 4")
    .. ------------------------------------------------------------------------

    Bigger source boxes separated from the target box by the (smaller) target
    box's size.

    If :attr:`boxtree.Tree.sources_have_extent`, then
    :attr:`sep_close_bigger_starts` will be non-*None*. It records
    interactions between boxes that would ordinarily be handled
    through "List 4", but must be evaluated specially/directly
    because of :ref:`extent`.

    Indexed like :attr:`target_or_target_parent_boxes`. See :ref:`csr`.

    .. attribute:: sep_bigger_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]``

    .. attribute:: sep_bigger_lists

        ``box_id_t [*]``

    .. attribute:: sep_close_bigger_starts

        ``box_id_t [ntarget_or_target_parent_boxes+1]`` (or *None*)

    .. attribute:: sep_close_bigger_lists

        ``box_id_t [*]`` (or *None*)
    """

    # {{{ "close" list merging -> "unified list 1"

    def merge_close_lists(self, queue, debug=False):
        """Return a new :class:`FMMTraversalInfo` instance with the contents of
        :attr:`sep_close_smaller_starts` and :attr:`sep_close_bigger_starts`
        merged into :attr:`neighbor_source_boxes_starts` and these two
        attributes set to *None*.
        """

        from boxtree.tools import reverse_index_array
        target_or_target_parent_boxes_from_all_boxes = reverse_index_array(
                self.target_or_target_parent_boxes, target_size=self.tree.nboxes,
                queue=queue)
        target_or_target_parent_boxes_from_tgt_boxes = cl.array.take(
                target_or_target_parent_boxes_from_all_boxes,
                self.target_boxes, queue=queue)

        del target_or_target_parent_boxes_from_all_boxes

        @memoize_method_nested
        def get_new_nb_sources_knl(write_counts):
            from pyopencl.elementwise import ElementwiseTemplate
            return ElementwiseTemplate("""//CL:mako//
                /* input: */
                box_id_t *target_or_target_parent_boxes_from_tgt_boxes,
                box_id_t *neighbor_source_boxes_starts,
                box_id_t *sep_close_smaller_starts,
                box_id_t *sep_close_bigger_starts,

                %if not write_counts:
                    box_id_t *neighbor_source_boxes_lists,
                    box_id_t *sep_close_smaller_lists,
                    box_id_t *sep_close_bigger_lists,

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

                box_id_t sep_close_smaller_start =
                    sep_close_smaller_starts[itgt_box];
                box_id_t sep_close_smaller_count =
                    sep_close_smaller_starts[itgt_box + 1]
                    - sep_close_smaller_start;

                box_id_t sep_close_bigger_start =
                    sep_close_bigger_starts[itarget_or_target_parent_box];
                box_id_t sep_close_bigger_count =
                    sep_close_bigger_starts[itarget_or_target_parent_box + 1]
                    - sep_close_bigger_start;

                %if write_counts:
                    if (itgt_box == 0)
                        new_neighbor_source_boxes_counts[0] = 0;

                    new_neighbor_source_boxes_counts[itgt_box + 1] =
                        neighbor_source_boxes_count
                        + sep_close_smaller_count
                        + sep_close_bigger_count
                        ;
                %else:

                    box_id_t cur_idx = new_neighbor_source_boxes_starts[itgt_box];

                    #define COPY_FROM(NAME) \
                        for (box_id_t i = 0; i < NAME##_count; ++i) \
                            new_neighbor_source_boxes_lists[cur_idx++] = \
                                NAME##_lists[NAME##_start+i];

                    COPY_FROM(neighbor_source_boxes)
                    COPY_FROM(sep_close_smaller)
                    COPY_FROM(sep_close_bigger)

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
            self.sep_close_smaller_starts,
            self.sep_close_bigger_starts,

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
            self.sep_close_smaller_starts,
            self.sep_close_bigger_starts,
            self.neighbor_source_boxes_lists,
            self.sep_close_smaller_lists,
            self.sep_close_bigger_lists,

            new_neighbor_source_boxes_starts,

            # output:
            new_neighbor_source_boxes_lists,
            range=slice(ntarget_boxes),
            queue=queue)

        return self.copy(
            neighbor_source_boxes_starts=new_neighbor_source_boxes_starts,
            neighbor_source_boxes_lists=new_neighbor_source_boxes_lists,
            sep_close_smaller_starts=None,
            sep_close_smaller_lists=None,
            sep_close_bigger_starts=None,
            sep_close_bigger_lists=None)

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
    def __init__(self, context):
        self.context = context

    # {{{ kernel builder

    @memoize_method
    def get_kernel_info(self, dimensions, particle_id_dtype, box_id_dtype,
            coord_dtype, box_level_dtype, max_levels,
            sources_are_targets, sources_have_extent, targets_have_extent,
            stick_out_factor):

        logging.info("building traversal build kernels")

        debug = False

        from pyopencl.tools import dtype_to_ctype
        from boxtree.tree import box_flags_enum
        render_vars = dict(
                dimensions=dimensions,
                dtype_to_ctype=dtype_to_ctype,
                particle_id_dtype=particle_id_dtype,
                box_id_dtype=box_id_dtype,
                box_flags_enum=box_flags_enum,
                coord_dtype=coord_dtype,
                vec_types=cl.array.vec.types,
                max_levels=max_levels,
                AXIS_NAMES=AXIS_NAMES,
                debug=debug,
                sources_are_targets=sources_are_targets,
                sources_have_extent=sources_have_extent,
                targets_have_extent=targets_have_extent,
                stick_out_factor=stick_out_factor,
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
                ("colleagues", COLLEAGUES_TEMPLATE, [], []),
                ("neighbor_source_boxes", NEIGBHOR_SOURCE_BOXES_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_boxes"),
                            ], []),
                ("sep_siblings", SEP_SIBLINGS_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_or_target_parent_boxes"),
                            VectorArg(box_id_dtype, "box_parent_ids"),
                            VectorArg(box_id_dtype, "colleagues_starts"),
                            VectorArg(box_id_dtype, "colleagues_list"),
                            ], []),
                ("sep_smaller", SEP_SMALLER_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_boxes"),
                            VectorArg(box_id_dtype, "colleagues_starts"),
                            VectorArg(box_id_dtype, "colleagues_list"),
                            ],
                            ["sep_close_smaller"]
                            if sources_have_extent or targets_have_extent
                            else []),
                ("sep_bigger", SEP_BIGGER_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "target_or_target_parent_boxes"),
                            VectorArg(box_id_dtype, "box_parent_ids"),
                            VectorArg(box_id_dtype, "colleagues_starts"),
                            VectorArg(box_id_dtype, "colleagues_list"),
                            ],
                            ["sep_close_bigger"]
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

        logging.info("traversal build kernels built")

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

        # Generated code shouldn't depend on tje *exact* number of tree levels.
        # So round up to the next multiple of 5.
        from pytools import div_ceil
        max_levels = div_ceil(tree.nlevels, 5) * 5

        knl_info = self.get_kernel_info(
                tree.dimensions, tree.particle_id_dtype, tree.box_id_dtype,
                tree.coord_dtype, tree.box_level_dtype, max_levels,
                tree.sources_are_targets,
                tree.sources_have_extent, tree.targets_have_extent,
                tree.stick_out_factor)

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
                    range=slice(1, len(box_list)),
                    queue=queue, wait_for=wait_for)

            result = result.get()

            # We skipped box 0 above. This is always true, whether
            # box 0 (=level 0) is a leaf or a parent.
            result[0] = 0

            # Postprocess result for unoccupied levels
            prev_start = len(box_list)
            for ilev in range(tree.nlevels-1, -1, -1):
                result[ilev] = prev_start = \
                        min(result[ilev], prev_start)

            return result, evt

        fin_debug("finding level starts in source parent boxes array")
        level_start_source_parent_box_nrs, evt_s = \
                extract_level_start_box_nrs(
                        source_parent_boxes, wait_for=wait_for)

        fin_debug("finding level starts in target or target parent boxes array")
        level_start_target_or_target_parent_box_nrs, evt_t = \
                extract_level_start_box_nrs(
                        target_or_target_parent_boxes, wait_for=wait_for)

        wait_for = [evt_s, evt_t]

        # }}}

        # {{{ colleagues

        fin_debug("finding colleagues")

        result, evt = knl_info.colleagues_builder(
                queue, tree.nboxes,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                wait_for=wait_for)
        wait_for = [evt]
        colleagues = result["colleagues"]

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

        result, evt = knl_info.sep_siblings_builder(
                queue, len(target_or_target_parent_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                target_or_target_parent_boxes.data, tree.box_parent_ids.data,
                colleagues.starts.data, colleagues.lists.data, wait_for=wait_for)
        wait_for = [evt]
        sep_siblings = result["sep_siblings"]

        # }}}

        # {{{ separated smaller ("list 3")

        fin_debug("finding separated smaller ('list 3')")

        result, evt = knl_info.sep_smaller_builder(
                queue, len(target_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                target_boxes.data,
                colleagues.starts.data, colleagues.lists.data,
                wait_for=wait_for)
        wait_for = [evt]
        sep_smaller = result["sep_smaller"]

        if tree.sources_have_extent or tree.targets_have_extent:
            sep_close_smaller_starts = result["sep_close_smaller"].starts
            sep_close_smaller_lists = result["sep_close_smaller"].lists
        else:
            sep_close_smaller_starts = None
            sep_close_smaller_lists = None

        # }}}

        # {{{ separated bigger ("list 4")

        fin_debug("finding separated bigger ('list 4')")

        result, evt = knl_info.sep_bigger_builder(
                queue, len(target_or_target_parent_boxes),
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_flags.data,
                target_or_target_parent_boxes.data, tree.box_parent_ids.data,
                colleagues.starts.data, colleagues.lists.data, wait_for=wait_for)
        wait_for = [evt]
        sep_bigger = result["sep_bigger"]

        if tree.sources_have_extent or tree.targets_have_extent:
            sep_close_bigger_starts = result["sep_close_bigger"].starts
            sep_close_bigger_lists = result["sep_close_bigger"].lists
        else:
            sep_close_bigger_starts = None
            sep_close_bigger_lists = None

        # }}}

        evt, = wait_for

        logger.info("traversal built")

        return FMMTraversalInfo(
                tree=tree,

                source_boxes=source_boxes,
                target_boxes=target_boxes,

                source_parent_boxes=source_parent_boxes,
                level_start_source_parent_box_nrs=level_start_source_parent_box_nrs,

                target_or_target_parent_boxes=target_or_target_parent_boxes,
                level_start_target_or_target_parent_box_nrs=(
                    level_start_target_or_target_parent_box_nrs),

                colleagues_starts=colleagues.starts,
                colleagues_lists=colleagues.lists,

                neighbor_source_boxes_starts=neighbor_source_boxes.starts,
                neighbor_source_boxes_lists=neighbor_source_boxes.lists,

                sep_siblings_starts=sep_siblings.starts,
                sep_siblings_lists=sep_siblings.lists,

                sep_smaller_starts=sep_smaller.starts,
                sep_smaller_lists=sep_smaller.lists,

                sep_close_smaller_starts=sep_close_smaller_starts,
                sep_close_smaller_lists=sep_close_smaller_lists,

                sep_bigger_starts=sep_bigger.starts,
                sep_bigger_lists=sep_bigger.lists,

                sep_close_bigger_starts=sep_close_bigger_starts,
                sep_close_bigger_lists=sep_close_bigger_lists,
                ).with_queue(None), evt

    # }}}

# vim: filetype=pyopencl:fdm=marker
