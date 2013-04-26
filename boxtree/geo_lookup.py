from __future__ import division

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




from pytools import memoize_method, Record
import numpy as np
import pyopencl as cl
import pyopencl.array
from mako.template import Template
from boxtree.tools import AXIS_NAMES, FromDeviceGettableRecord

import logging
logger = logging.getLogger(__name__)




# {{{ output

class LeavesToBallsLookup(FromDeviceGettableRecord):
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance used to build this lookup.

    .. attribute:: balls_near_box_starts

        Indices into :attr:`balls_near_box_lists`.
        ``balls_near_box_lists[balls_near_box_starts[ibox]:balls_near_box_starts[ibox]+1]``
        results in a list of balls that overlap leaf box *ibox*.

        .. note:: Only leaf boxes have non-empty entries in this table. Nonetheless,
            this list is indexed by the global box index.

    .. attribute:: balls_near_box_lists
    """

# }}}

# {{{ kernel templates

BALLS_TO_LEAVES_TEMPLATE = r"""//CL//
typedef ${dtype_to_ctype(ball_id_dtype)} ball_id_t;

void generate(LIST_ARG_DECL USER_ARG_DECL ball_id_t ball_nr)
{
    coord_vec_t ball_center;
    %for i in range(dimensions):
        ball_center.${AXIS_NAMES[i]} = ball_${AXIS_NAMES[i]}[ball_nr];
    %endfor

    coord_t ball_radius = ball_radii[ball_nr];

    // To find overlapping leaves, start at the top of the tree, descend
    // into overlapping boxes.
    ${walk_init(0)}

    while (continue_walk)
    {
        box_id_t child_box_id = box_child_ids[walk_morton_nr * aligned_nboxes + walk_box_id];
        dbg_printf(("  walk box id: %d morton: %d child id: %d level: %d\n",
            walk_box_id, walk_morton_nr, child_box_id, walk_level));

        if (child_box_id)
        {
            bool is_overlapping;

            {
                ${load_center("child_center", "child_box_id")}
                int child_level = box_levels[child_box_id];

                coord_t size_sum = LEVEL_TO_RAD(child_level) + ball_radius;

                coord_t max_dist = 0;
                %for i in range(dimensions):
                    max_dist = fmax(max_dist, fabs(ball_center.s${i} - child_center.s${i}));
                %endfor

                is_overlapping = max_dist <= size_sum;
            }

            if (is_overlapping)
            {
                if (!(box_flags[child_box_id] & BOX_HAS_CHILDREN))
                {
                    APPEND_ball_numbers(ball_nr);
                    APPEND_overlapping_leaves(child_box_id);
                }
                else
                {
                    // We want to descend into this box. Put the current state
                    // on the stack.

                    ${walk_push("child_box_id")}
                    continue;
                }
            }
        }

        ${walk_advance()}
    }
}
"""

class _KernelInfo(Record):
    pass

class LeavesToBallsLookupBuilder(object):
    """Given a set of :math:`l^\infty` "balls", this class helps build a
    look-up table from leaf boxes to balls that overlap with each leaf box.
    """
    def __init__(self, context):
        self.context = context

        from pyopencl.algorithm import KeyValueSorter
        self.key_value_sorter = KeyValueSorter(context)

    @memoize_method
    def get_balls_to_leaves_kernel(self, dimensions, coord_dtype, box_id_dtype,
            ball_id_dtype, max_levels, stick_out_factor):
        from pyopencl.tools import dtype_to_ctype
        from boxtree import box_flags_enum
        render_vars = dict(
                dimensions=dimensions,
                dtype_to_ctype=dtype_to_ctype,
                box_id_dtype=box_id_dtype,
                particle_id_dtype=None,
                ball_id_dtype=ball_id_dtype,
                coord_dtype=coord_dtype,
                vec_types=cl.array.vec.types,
                max_levels=max_levels,
                AXIS_NAMES=AXIS_NAMES,
                box_flags_enum=box_flags_enum,
                debug=False,
                stick_out_factor=stick_out_factor,
                )

        logger.info("start building leaves-to-balls lookup kernel")

        from boxtree.traversal import TRAVERSAL_PREAMBLE_TEMPLATE

        src = Template(
                TRAVERSAL_PREAMBLE_TEMPLATE
                + BALLS_TO_LEAVES_TEMPLATE,
                strict_undefined=True).render(**render_vars)


        from pyopencl.tools import VectorArg, ScalarArg
        from pyopencl.algorithm import ListOfListsBuilder
        result = ListOfListsBuilder(self.context,
                [
                    ("ball_numbers", ball_id_dtype),
                    ("overlapping_leaves", box_id_dtype),
                    ],
                str(src),
                arg_decls=[
                    VectorArg(box_flags_enum.dtype, "box_flags"),
                    VectorArg(coord_dtype, "box_centers"),
                    VectorArg(box_id_dtype, "box_child_ids"),
                    VectorArg(np.uint8, "box_levels"),
                    ScalarArg(coord_dtype, "root_extent"),
                    ScalarArg(box_id_dtype, "aligned_nboxes"),
                    VectorArg(coord_dtype, "ball_radii"),
                    ] + [VectorArg(coord_dtype, "ball_"+ax) for ax in AXIS_NAMES[:dimensions]],
                name_prefix="circles_to_balls",
                count_sharing={
                    # /!\ This makes a promise that APPEND_ball_numbers will
                    # always occur *before* APPEND_overlapping_leaves.
                    "overlapping_leaves": "ball_numbers"
                    },
                complex_kernel=True)

        logger.info("done building leaves-to-balls lookup kernel")

        return result

    def __call__(self, queue, tree, ball_centers, ball_radii, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`
        :arg tree: a :class:`boxtree.Tree`.
        :arg ball_centers: an object array of coordinate
            :class:`pyopencl.array.Array` instances.
            Their *dtype* must match *tree*'s
            :attr:`boxtree.Tree.coord_dtype`.
        :arg ball_radii: a
            :class:`pyopencl.array.Array`
            of positive numbers.
            Its *dtype* must match *tree*'s
            :attr:`boxtree.Tree.coord_dtype`.
        :arg wait_for: may either be *None* or a list of :class:`pyopencl.Event`
            instances for whose completion this command waits before starting
            exeuction.
        :returns: a tuple *(lbl, event)*, where *lbl* is an instance of
            :class:`LeavesToBallsLookup`, and *event* is a :class:`pyopencl.Event` 
            for dependency management.
        """

        from pytools import single_valued
        if single_valued(bc.dtype for bc in ball_centers) != tree.coord_dtype:
            raise TypeError("ball_centers dtype must match tree.coord_dtype")
        if ball_radii.dtype != tree.coord_dtype:
            raise TypeError("ball_radii dtype must match tree.coord_dtype")

        ball_id_dtype = tree.particle_id_dtype # ?

        from pytools import div_ceil
        max_levels = div_ceil(tree.nlevels, 10) * 10

        b2l_knl = self.get_balls_to_leaves_kernel(
                tree.dimensions, tree.coord_dtype,
                tree.box_id_dtype, ball_id_dtype,
                max_levels, tree.stick_out_factor)

        logger.info("leaves-to-balls lookup: prepare ball list")

        nballs = len(ball_radii)
        result, evt = b2l_knl(
                queue, nballs,
                tree.box_flags.data, tree.box_centers.data,
                tree.box_child_ids.data, tree.box_levels.data,
                tree.root_extent, tree.aligned_nboxes,
                ball_radii.data, *tuple(bc.data for bc in ball_centers),
                wait_for=wait_for)
        wait_for = [evt]

        logger.info("leaves-to-balls lookup: key-value sort")

        balls_near_box_starts, balls_near_box_lists, evt \
                = self.key_value_sorter(
                        queue,
                        # keys
                        result["overlapping_leaves"].lists,
                        # values
                        result["ball_numbers"].lists,
                        tree.nboxes, starts_dtype=tree.box_id_dtype,
                        wait_for=wait_for)

        logger.info("leaves-to-balls lookup: built")

        return LeavesToBallsLookup(
                tree=tree,
                balls_near_box_starts=balls_near_box_starts,
                balls_near_box_lists=balls_near_box_lists), evt

# }}}

# vim: filetype=pyopencl:fdm=marker
