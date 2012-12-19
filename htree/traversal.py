from __future__ import division
import numpy as np
from pytools import memoize, memoize_method, Record
import pyopencl as cl
import pyopencl.array
from mako.template import Template
from htree import AXIS_NAMES




# {{{ preamble

PREAMBLE_TEMPLATE = r"""//CL//

typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;
typedef ${dtype_to_ctype(coord_dtype)} coord_t;
typedef ${dtype_to_ctype(vec_types[coord_dtype, dimensions])} coord_vec_t;

#define NLEVELS ${max_levels}

<%def name="load_center(name, box_id)">
    coord_vec_t ${name};
    %for i in range(dimensions):
        ${name}.${AXIS_NAMES[i]} = box_centers[aligned_nboxes * ${i} + ${box_id}];
    %endfor
</%def>

#define LEVEL_TO_SIZE(level) \
        (root_extent * 1 / (coord_t) (1 << level))

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

<%def name="walk_push()">
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
</%def>

"""

# }}}

# {{{ adjacency test

ADJACENCY_TEST_TEMPLATE = r"""//CL//

bool is_adjacent_or_overlapping(
    USER_ARG_DECL coord_vec_t center, int level, box_id_t other_box_id)
{
    ${load_center("other_center", "other_box_id")}
    int other_level = box_levels[other_box_id];

    // This checks if the two boxes overlap
    // with an amount of 'slack' corresponding to half the
    // width of the smaller of the two boxes.
    // (Without the 'slack', there wouldn't be any
    // overlap.)

    coord_t size_sum = 0.5 * (LEVEL_TO_SIZE(level) + LEVEL_TO_SIZE(other_level));
    coord_t slack = size_sum + 0.5 * LEVEL_TO_SIZE(max(level, other_level));

    coord_t max_dist = 0;
    %for i in range(dimensions):
        max_dist = fmax(max_dist, fabs(center.s${i} - other_center.s${i}));
    %endfor

    return max_dist <= slack;
}

bool is_well_separated(
    USER_ARG_DECL coord_vec_t center, int level, box_id_t other_box_id)
{
    ${load_center("other_center", "other_box_id")}
    int other_level = box_levels[other_box_id];

    coord_t max_half_box = 0.5 * LEVEL_TO_SIZE(min(level, other_level));
    coord_t max_diag = sqrt(${dimensions}) * max_half_box;

    coord_t required_dist = 3 * max_diag;

    coord_t l_inf_dist = 0;
    %for i in range(dimensions):
        l_inf_dist = fmax(l_inf_dist, fabs(center.s${i} - other_center.s${i}));
    %endfor

    return l_inf_dist >= required_dist;
}

"""

# }}}

# {{{ leaves and branches

LEAVES_AND_BRANCHES_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    unsigned char box_type = box_types[box_id];
    if (box_type == BOX_LEAF)
    { APPEND_leaves(box_id); }
    else if (box_type == BOX_BRANCH)
    { APPEND_branches(box_id); }
}
"""

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
    // into adjacent (or overlapping) branches.
    ${walk_init(0)}

    while (continue_walk)
    {
        box_id_t child_box_id = box_child_ids[walk_morton_nr * aligned_nboxes + walk_box_id];
        dbg_printf(("  level: %d walk box id: %d morton: %d child id: %d\n",
            walk_level, walk_box_id, walk_morton_nr, child_box_id));

        if (child_box_id)
        {
            bool a_or_o = is_adjacent_or_overlapping(
                USER_ARGS center, level, child_box_id);

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
                    ${walk_push()}

                    walk_box_id = child_box_id;
                    walk_morton_nr = 0;
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

# {{{ neighbor leaves ("list 1")

NEIGBHOR_LEAVES_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t leaf_number)
{
    // /!\ leaf_number is *not* a box_id, despite the type.
    // It's the number of the leaf we're currently processing.

    box_id_t box_id = leaf_boxes[leaf_number];

    ${load_center("center", "box_id")}

    int level = box_levels[box_id];

    dbg_printf(("box id: %d level: %d\n", box_id, level));

    // To find this box's colleagues, start at the top of the tree, descend
    // into adjacent (or overlapping) branches.
    ${walk_init(0)}

    while (continue_walk)
    {
        box_id_t child_box_id = box_child_ids[walk_morton_nr * aligned_nboxes + walk_box_id];
        dbg_printf(("  walk box id: %d morton: %d child id: %d level: %d\n",
            walk_box_id, walk_morton_nr, child_box_id, walk_level));

        if (child_box_id)
        {
            bool a_or_o = is_adjacent_or_overlapping(
                USER_ARGS center, level, child_box_id);

            if (a_or_o)
            {
                if (box_types[child_box_id] == BOX_LEAF && child_box_id != box_id)
                {
                    dbg_printf(("    neighbor leaf\n"));
                    APPEND_neighbor_leaves(child_box_id);
                }
                else
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    dbg_printf(("    descend\n"));
                    ${walk_push()}

                    walk_box_id = child_box_id;
                    walk_morton_nr = 0;
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

WELL_SEP_SIBLINGS_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
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

            bool ws = is_well_separated(
                USER_ARGS center, level, sib_box_id);

            if (ws)
            {
                APPEND_well_sep_siblings(sib_box_id);
            }
        }
    }
}

"""

# }}}

# {{{ separated smaller non-siblings ("list 3", also used for "list 4")

SEP_SMALLER_NONSIBLINGS_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t leaf_number)
{
    // /!\ leaf_number is *not* a box_id, despite the type.
    // It's the number of the leaf we're currently processing.
    box_id_t box_id = leaf_boxes[leaf_number];

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
            // Loop invariant: walk_box_id is always adjacent to box_id.
            // This is true at the first level because colleagues are by adjacent
            // by definition, and is kept true throughout the walk by only descending
            // into adjacent boxes.

            box_id_t child_box_id = box_child_ids[walk_morton_nr * aligned_nboxes + walk_box_id];
            dbg_printf(("  walk box id: %d morton: %d child id: %d level: %d\n",
                walk_box_id, walk_morton_nr, child_box_id, walk_level));

            if (child_box_id)
            {
                bool a_or_o = is_adjacent_or_overlapping(
                    USER_ARGS center, level, child_box_id);

                if (a_or_o)
                {
                    if (box_types[child_box_id] != BOX_LEAF)
                    {
                        // We want to descend into this box. Put the current state
                        // on the stack.

                        dbg_printf(("    descend\n"));
                        ${walk_push()}

                        walk_box_id = child_box_id;
                        walk_morton_nr = 0;
                        continue;
                    }
                }
                else
                {
                    APPEND_origin_box(box_id);
                    APPEND_sep_smaller_nonsiblings(child_box_id);
                }
            }

            ${walk_advance()}
        }
    }
}

"""

# }}}




class TraversalInfo(Record):
    pass



# {{{ driver

class _KernelInfo(Record):
    pass

class FMMTraversalGenerator:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernel_info(self, dimensions, particle_id_dtype, box_id_dtype,
            coord_dtype, max_levels):

        debug = False

        from pyopencl.tools import dtype_to_ctype
        render_vars = dict(
                dimensions=dimensions,
                dtype_to_ctype=dtype_to_ctype,
                particle_id_dtype=particle_id_dtype,
                box_id_dtype=box_id_dtype,
                coord_dtype=coord_dtype,
                vec_types=cl.array.vec.types,
                max_levels=max_levels,
                AXIS_NAMES=AXIS_NAMES,
                debug=debug,
                )
        from pyopencl.algorithm import ListOfListsBuilder
        from pyopencl.tools import VectorArg, ScalarArg
        from htree import box_type_enum

        # {{{ leaves and branches

        src = Template(
                box_type_enum.get_c_defines()
                + PREAMBLE_TEMPLATE
                + LEAVES_AND_BRANCHES_TEMPLATE,
                strict_undefined=True).render(**render_vars)

        leaves_and_branches_builder = ListOfListsBuilder(self.context,
                [
                    ("leaves", box_id_dtype),
                    ("branches", box_id_dtype),
                    ],
                str(src),
                arg_decls=[
                    VectorArg(np.uint8, "box_types"),
                    ], debug=debug, name_prefix="leaves_and_branches")

        # }}}

        # {{{ complex lists

        base_args = [
                VectorArg(coord_dtype, "box_centers"),
                ScalarArg(coord_dtype, "root_extent"),
                VectorArg(np.uint8, "box_levels"),
                ScalarArg(box_id_dtype, "aligned_nboxes"),
                VectorArg(box_id_dtype, "box_child_ids"),
                VectorArg(np.uint8, "box_types"),
                ]

        builders = {}
        for list_name, template, extra_args in [
                ("colleagues", COLLEAGUES_TEMPLATE, []),
                ("neighbor_leaves", NEIGBHOR_LEAVES_TEMPLATE,
                        [VectorArg(box_id_dtype, "leaf_boxes")]),
                ("well_sep_siblings", WELL_SEP_SIBLINGS_TEMPLATE,
                        [
                            VectorArg(box_id_dtype, "box_parent_ids"),
                            VectorArg(box_id_dtype, "colleagues_starts"),
                            VectorArg(box_id_dtype, "colleagues_list"),
                            ]),
                ]:
            src = Template(
                    box_type_enum.get_c_defines()
                    + PREAMBLE_TEMPLATE
                    + ADJACENCY_TEST_TEMPLATE
                    + template,
                    strict_undefined=True).render(**render_vars)

            builders[list_name+"_builder"] = ListOfListsBuilder(self.context,
                    [(list_name, box_id_dtype) ],
                    str(src),
                    arg_decls=base_args + extra_args,
                    debug=debug, name_prefix=list_name,
                    complex_kernel=True)

        # }}}

        # {{{ separated smaller non-siblings

        src = Template(
                box_type_enum.get_c_defines()
                + PREAMBLE_TEMPLATE
                + ADJACENCY_TEST_TEMPLATE
                + SEP_SMALLER_NONSIBLINGS_TEMPLATE,
                strict_undefined=True).render(**render_vars)

        builders["sep_smaller_nonsiblings_builder"] = ListOfListsBuilder(self.context,
                [
                    ("sep_smaller_nonsiblings", box_id_dtype),
                    ("origin_box", box_id_dtype)
                    ],
                str(src),
                arg_decls=base_args + [
                    VectorArg(box_id_dtype, "leaf_boxes"),
                    VectorArg(box_id_dtype, "colleagues_starts"),
                    VectorArg(box_id_dtype, "colleagues_list"),
                    ],
                debug=debug, name_prefix="sep_smaller_nonsiblings",
                complex_kernel=True,
                count_sharing={
                    # /!\ This makes a promise that APPEND_origin_box will
                    # always occur *before* APPEND_sep_smaller_nonsiblings.
                    "sep_smaller_nonsiblings": "origin_box"
                    })

        # }}}

        return _KernelInfo(
                leaves_and_branches_builder=leaves_and_branches_builder,
                **builders)

    def __call__(self, queue, tree):
        from pytools import div_ceil
        max_levels = div_ceil(tree.nlevels, 10) * 10

        knl_info = self.get_kernel_info(
                tree.dimensions, tree.particle_id_dtype, tree.box_id_dtype,
                tree.coord_dtype, max_levels)

        result = knl_info.leaves_and_branches_builder(
                queue, tree.nboxes, tree.box_types.data)

        leaf_box_count = result["leaves"].count
        leaf_boxes = result["leaves"].list
        branch_box_count = result["leaves"].count
        branch_boxes = result["leaves"].list

        colleagues = knl_info.colleagues_builder(
                queue, tree.nboxes,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_types.data) \
                        ["colleagues"]

        # "list 1"
        neighbor_leaves = knl_info.neighbor_leaves_builder(
                queue, leaf_box_count,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_types.data,
                leaf_boxes.data)["neighbor_leaves"]

        # "list 2"
        well_sep_siblings = knl_info.well_sep_siblings_builder(
                queue, tree.nboxes,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_types.data,
                tree.box_parent_ids.data,
                colleagues.starts.data, colleagues.list.data)["well_sep_siblings"]

        # "list 3"
        result = knl_info.sep_smaller_nonsiblings_builder(
                queue, leaf_box_count,
                tree.box_centers.data, tree.root_extent, tree.box_levels.data,
                tree.aligned_nboxes, tree.box_child_ids.data, tree.box_types.data,
                leaf_boxes.data,
                colleagues.starts.data, colleagues.list.data)

        sep_smaller_nonsiblings = result["sep_smaller_nonsiblings"]

        # TODO: "list 4"

        return TraversalInfo(
                leaf_box_count=leaf_box_count,
                leaf_boxes=leaf_boxes,
                branch_box_count=branch_box_count,
                branch_boxes=branch_boxes,

                colleagues_starts=colleagues.starts,
                colleagues_list=colleagues.list,

                neighbor_leaves_starts=neighbor_leaves.starts,
                neighbor_leaves_list=neighbor_leaves.list,

                well_sep_siblings_starts=well_sep_siblings.starts,
                well_sep_siblings_list=well_sep_siblings.list,

                sep_smaller_nonsiblings_starts=sep_smaller_nonsiblings.starts,
                sep_smaller_nonsiblings_list=sep_smaller_nonsiblings.list,
                )

# }}}




# vim: filetype=pyopencl:fdm=marker
