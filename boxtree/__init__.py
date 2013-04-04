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
from pytools import memoize, memoize_method, Record
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseTemplate
from pyopencl.scan import ScanTemplate
from mako.template import Template
from functools import partial
from boxtree.tools import FromDeviceGettableRecord, get_type_moniker

# {{{ module documentation

__doc__ = """
This module sorts particles into an adaptively refined quad/octree.
See :mod:`boxtree.traversal` for computing fast multipole interaction
lists on this tree structure. Note that while traversal generation
builds on the result of particle sorting (described here),
it is completely distinct in the software sense.

The tree builder can be run in three modes:

* one where no distinction is made between sources and targets. In this mode,
  all participants in the interaction are called 'particles'.
  (*targets is None* in the call to :meth:`TreeBuilder.__call__`)

* one where a distinction between sources and targets is made.
  (*targets is not None* in the call to :meth:`TreeBuilder.__call__`)

* one where a distinction between sources and targets is made,
  and where sources are considered to have an extent, given by
  a radius.
  (``targets is not None`` and ``source_radii is not None`` in the
  call to :meth:`TreeBuilder.__call__`)

.. _particle-orderings:

Particle Orderings
------------------

There are four particle orderings:

* **user source order**
* **tree source order** (tree/box-sorted)
* **user target order**
* **tree target order** (tree/box-sorted)

:attr:`Tree.user_source_ids` helps translate source arrays into
tree order for processing. :attr:`Tree.sorted_target_ids`
helps translate potentials back into user target order for output.

If each 'original' source above is linked to a number of point sources,
the point sources have their own orderings:

* **user point source order**
* **tree point source order** (tree/box-sorted)

:attr:`TreeWithLinkedPointSources.user_point_source_ids` helps translate point
source arrays into tree order for processing.

Sources with extent
-------------------

By default, source particles are considered to be points. If *source_radii* is
passed to :class:`TreeBuilder.__call__`, however, this is no longer true. Each
source then has an :math:`l^\infty` 'radius'. Each such source is typically
then later linked to a set of related point sources contained within the
extent. (See :meth:`Tree.link_point_sources` for more.)

Note that targets with extent are not supported.
"""


# TODO:
# - Allow for (groups of?) sources stuck in tree
# - Add *restrict where applicable.

# -----------------------------------------------------------------------------
# CONTROL FLOW
# ------------
#
# Since this file mostly fills in the blanks in the outer parallel 'scan'
# implementation, control flow here can be a bit hard to see.
#
# - Everything starts and ends in the 'driver' bit at the end.
#
# - The first thing that happens is that data types get built and
#   kernels get compiled. Most of the file consists of type and
#   code generators for these kernels.
#
# - We start with a reduction that determines the bounding box of all
#   particles.
#
# - The level loop is in the driver below, which alternates between
#   scans and local post processing ("split and sort"), according to
#   the algorithm described below.
#
# - Once the level loop finishes, a "box info" kernel is run
#   that extracts some more information for each box. (center, level, ...)
#
# - As a last step, empty leaf boxes are eliminated. This is done by a
#   scan kernel that computes indices, and by an elementwise kernel
#   that compresses arrays and maps them to new box IDs, if applicable.
#
# -----------------------------------------------------------------------------
#
# HOW DOES THE PRIMARY SCAN WORK?
# -------------------------------
#
# This code sorts particles into an nD-tree of boxes. It does this by doing a
# (parallel) scan over particles and a (local, i.e. independent for each particle)
# postprocessing step for each level.
#
# The following information is being pushed around by the scan, which
# proceeds over particles:
#
# - a cumulative count ("counts") of particles in each subbox ("morton_nr") at
#   the current level, should the current box need to be subdivided.
#
# - the "split_box_id". The very first entry here gets intialized to
#   the number of boxes present at the previous level. If a box knows it needs to
#   be subdivided, its first particle asks for 2**d new boxes. This gets scanned
#   over by summing globally (unsegmented-ly). The splits are then realized in
#   the post-processing step.
#
# -----------------------------------------------------------------------------

# }}}




AXIS_NAMES = ("x", "y", "z", "w")




class _KernelInfo(Record):
    pass

# {{{ data types

@memoize
def make_morton_bin_count_type(device, dimensions, particle_id_dtype, sources_have_extent):
    fields = []

    # Non-child sources are sorted *before* all the child sources.
    if sources_have_extent:
        fields.append(("nonchild_sources", particle_id_dtype))

    from boxtree.tools import padded_bin
    for mnr in range(2**dimensions):
        fields.append(("pcnt%s" % padded_bin(mnr, dimensions), particle_id_dtype))

    dtype = np.dtype(fields)

    name_suffix = ""
    if sources_have_extent:
        name_suffix = "_ncs"

    name = "boxtree_morton_bin_count_%dd_p%s%s_t" % (
            dimensions,
            get_type_moniker(particle_id_dtype),
            name_suffix)

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

# }}}

# {{{ box flags

class box_flags_enum:
    """Constants for box types."""

    dtype = np.dtype(np.uint8)

    HAS_OWN_SOURCES = 1 << 0
    HAS_OWN_TARGETS = 1 << 1
    HAS_OWN_SRCNTGTS = (HAS_OWN_SOURCES | HAS_OWN_TARGETS)
    HAS_CHILD_SOURCES = 1 << 2
    HAS_CHILD_TARGETS = 1 << 3
    HAS_CHILDREN = (HAS_CHILD_SOURCES | HAS_CHILD_TARGETS)

    @classmethod
    def get_c_defines(cls):
        """Return a string with C defines corresponding to these constants.
        """

        return "\n".join(
                "#define BOX_%s %d"
                % (name, getattr(cls, name))
                for name in sorted(dir(box_flags_enum))
                if name[0].isupper()) + "\n\n"


    @classmethod
    def get_c_typedef(cls):
        """Returns a typedef to define box_flags_t."""

        from pyopencl.tools import dtype_to_ctype
        return "\n\ntypedef %s box_flags_t;\n\n" % dtype_to_ctype(cls.dtype)

# }}}

# {{{ preamble

TYPE_DECL_PREAMBLE_TPL = Template(r"""//CL//
    typedef ${dtype_to_ctype(morton_bin_count_dtype)} morton_counts_t;
    typedef morton_counts_t scan_t;
    typedef ${dtype_to_ctype(bbox_dtype)} bbox_t;
    typedef ${dtype_to_ctype(coord_dtype)} coord_t;
    typedef ${dtype_to_ctype(coord_vec_dtype)} coord_vec_t;
    typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
    typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;

    // morton_nr == -1 is defined to mean that the srcntgt is
    // remaining at the present level and will not be sorted
    // into a child box.
    typedef ${dtype_to_ctype(morton_nr_dtype)} morton_nr_t;
    """, strict_undefined=True)

GENERIC_PREAMBLE_TPL = Template(r"""//CL//
    #define STICK_OUT_FACTOR ((coord_t) ${stick_out_factor})

    // Use this as dbg_printf(("oh snap: %d\n", stuff));
    // Note the double parentheses.
    //
    // Watch out: 64-bit values on Intel CL must be printed with %ld, or
    //    subsequent values will print as 0. And you'll be very confused.

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

# {{{ scan primitive code template

SCAN_PREAMBLE_TPL = Template(r"""//CL//

    // {{{ neutral element

    scan_t scan_t_neutral()
    {
        scan_t result;
        %if sources_have_extent:
            result.nonchild_sources = 0;
        %endif
        %for mnr in range(2**dimensions):
            result.pcnt${padded_bin(mnr, dimensions)} = 0;
        %endfor
        return result;
    }

    // }}}

    // {{{ scan 'add' operation
    scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary)
    {
        if (!across_seg_boundary)
        {
            %if sources_have_extent:
                b.nonchild_sources += a.nonchild_sources;
            %endif

            %for mnr in range(2**dimensions):
                <% field = "pcnt"+padded_bin(mnr, dimensions) %>
                b.${field} = a.${field} + b.${field};
            %endfor
        }

        return b;
    }

    // }}}

    // {{{ scan data type init from particle

    scan_t scan_t_from_particle(
        const int i,
        const int level,
        bbox_t const *bbox,
        global morton_nr_t *morton_nrs, // output/side effect
        global particle_id_t *user_srcntgt_ids
        %for ax in axis_names:
            , global const coord_t *${ax}
        %endfor
        %if sources_have_extent:
            , global const coord_t *source_radii
        %endif
    )
    {
        particle_id_t user_srcntgt_id = user_srcntgt_ids[i];

        // Recall that 'level' is the level currently being built, e.g. 1 at the root.
        // This should be 0.5 at level 1. (Level 0 is the root.)
        coord_t next_level_box_size_factor = ((coord_t) 1) / ((coord_t) (1U << level));

        %if sources_have_extent:
            bool stop_srcntgt_descent = false;
            coord_t source_radius = source_radii[i];
        %endif

        %for ax in axis_names:
            // Most FMMs are isotropic, i.e. global_extent_{x,y,z} are all the same.
            // Nonetheless, the gain from exploiting this assumption seems so
            // minimal that doing so here didn't seem worthwhile.

            coord_t global_min_${ax} = bbox->min_${ax};
            coord_t global_extent_${ax} = bbox->max_${ax} - global_min_${ax};
            coord_t srcntgt_${ax} = ${ax}[user_srcntgt_id];

            // Note that the upper bound of the global bounding box is computed
            // to be slightly larger than the highest found coordinate, so that
            // 1.0 is never reached as a scaled coordinate at the highest
            // level, and it isn't either by the fact that boxes are
            // [)-half-open in subsequent levels.

            // So (1 << level) is 2 when building level 1.  Because the
            // floating point factor is strictly less than 1, 2 is never
            // reached, so when building level 1, the result is either 0 or 1.
            // After that, we just add one (less significant) bit per level.

            unsigned ${ax}_bits = (unsigned) (
                ((srcntgt_${ax} - global_min_${ax}) / global_extent_${ax})
                * (1U << level));

            %if sources_have_extent:
                // Need to compute center to compare excess with STICK_OUT_FACTOR.
                coord_t next_level_box_center_${ax} =
                    global_min_${ax}
                    + global_extent_${ax} * (${ax}_bits + (coord_t) 0.5) * next_level_box_size_factor;

                coord_t next_level_box_stick_out_radius_${ax} =
                    (coord_t) (
                        0.5 // convert diameter to radius
                        * (1 + STICK_OUT_FACTOR))
                    * global_extent_${ax} * next_level_box_size_factor;

                stop_srcntgt_descent = stop_srcntgt_descent ||
                    (srcntgt_${ax} + source_radius >=
                        next_level_box_center_${ax}
                        + next_level_box_stick_out_radius_${ax});
                stop_srcntgt_descent = stop_srcntgt_descent ||
                    (srcntgt_${ax} - source_radius <
                        next_level_box_center_${ax}
                        - next_level_box_stick_out_radius_${ax});
            %endif
        %endfor

        // Pick off the lowest-order bit for each axis, put it in its place.
        int level_morton_number = 0
        %for iax, ax in enumerate(axis_names):
            | (${ax}_bits & 1U) << (${dimensions-1-iax})
        %endfor
            ;

        %if sources_have_extent:
            if (stop_srcntgt_descent)
                level_morton_number = -1;
        %endif

        scan_t result;
        %if sources_have_extent:
            result.nonchild_sources = (level_morton_number == -1);
        %endif
        %for mnr in range(2**dimensions):
            <% field = "pcnt"+padded_bin(mnr, dimensions) %>
            result.${field} = (level_morton_number == ${mnr});
        %endfor
        morton_nrs[i] = level_morton_number;

        return result;
    }

    // }}}

""", strict_undefined=True)

# }}}

# {{{ scan output code template

SCAN_OUTPUT_STMT_TPL = Template(r"""//CL//
    {
        particle_id_t my_id_in_my_box = -1
        %if sources_have_extent:
            + item.nonchild_sources
        %endif
        %for mnr in range(2**dimensions):
            + item.pcnt${padded_bin(mnr, dimensions)}
        %endfor
            ;
        dbg_printf(("my_id_in_my_box:%d\n", my_id_in_my_box));
        morton_bin_counts[i] = item;

        box_id_t current_box_id = srcntgt_box_ids[i];
        particle_id_t box_srcntgt_count = box_srcntgt_counts[current_box_id];

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
        arguments=r"""//CL:mako//
            /* input */
            box_id_t *srcntgt_box_ids,
            particle_id_t *box_srcntgt_starts,
            particle_id_t *box_srcntgt_counts,
            particle_id_t max_particles_in_box,
            morton_counts_t *box_morton_bin_counts,
            box_level_t *box_levels,
            box_level_t level,

            /* input/output */
            box_id_t *nboxes,

            /* output */
            box_id_t *split_box_ids,
            """,
        preamble=r"""//CL:mako//
            scan_t count_new_boxes_needed(
                particle_id_t i,
                box_id_t box_id,
                __global box_id_t *nboxes,
                __global particle_id_t *box_srcntgt_starts,
                __global particle_id_t *box_srcntgt_counts,
                __global morton_counts_t *box_morton_bin_counts,
                particle_id_t max_particles_in_box,
                __global box_level_t *box_levels,
                box_level_t level
                )
            {
                scan_t result = 0;

                // First particle? Start counting at (the previous level's) nboxes.
                if (i == 0)
                    result += *nboxes;

                %if sources_have_extent:
                    const particle_id_t nonchild_sources_in_box =
                        box_morton_bin_counts[box_id].nonchild_sources;
                %else:
                    const particle_id_t nonchild_sources_in_box = 0;
                %endif

                particle_id_t first_particle_in_my_box =
                    box_srcntgt_starts[box_id];

                // Add 2**d to make enough room for a split of the current box
                // This will be the split_box_id for *all* particles in this box,
                // including non-child sources.

                if (i == first_particle_in_my_box
                    %if sources_have_extent:
                        // Only last-level boxes get to produce new boxes.
                        // If sources have extent, then prior-level boxes
                        // will keep asking for more boxes to be allocated.
                        // Prevent that.

                        &&
                        box_levels[box_id] + 1 == level
                    %endif
                    &&
                    /* box overfull? */
                    box_srcntgt_counts[box_id] - nonchild_sources_in_box
                        > max_particles_in_box)
                {
                    result += ${2**dimensions};
                }

                return result;
            }
            """,
        input_expr="""count_new_boxes_needed(
                i, srcntgt_box_ids[i], nboxes,
                box_srcntgt_starts, box_srcntgt_counts, box_morton_bin_counts,
                max_particles_in_box, box_levels, level
                )""",
        scan_expr="a + b",
        neutral="0",
        output_statement="""//CL//
            dbg_assert(item >= 0);

            split_box_ids[i] = item;

            // Am I the last particle overall? If so, write box count
            if (i+1 == N)
                *nboxes = item;
            """)

# }}}

# {{{ split-and-sort kernel

SPLIT_AND_SORT_PREAMBLE_TPL = Template(r"""//CL//
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
        %if sources_have_extent:
            if (morton_nr == -1)
                return counts.nonchild_sources;
        %endif
        return ${get_count_for_branch("")};
    }

""", strict_undefined=True)

SPLIT_AND_SORT_KERNEL_TPL =  Template(r"""//CL//
    box_id_t my_box_id = srcntgt_box_ids[i];
    dbg_assert(my_box_id >= 0);
    dbg_assert(my_box_id < nboxes);

    dbg_printf(("postproc %d:\n", i));
    dbg_printf(("   my box id: %d\n", my_box_id));

    particle_id_t box_srcntgt_count = box_srcntgt_counts[my_box_id];

    %if sources_have_extent:
        const particle_id_t nonchild_source_count =
            box_morton_bin_counts[my_box_id].nonchild_sources;

    %else:
        const particle_id_t nonchild_source_count = 0;
    %endif

    bool do_split_box =
        box_srcntgt_count - nonchild_source_count
        > max_particles_in_box;

    %if sources_have_extent:
        ## Only do split-box processing for srcntgts that were touched
        ## on the immediately preceding level.
        ##
        ## If sources have no extent, then subsequent levels
        ## will never decide to split boxes that were kept unsplit on prior
        ## levels either. If sources do
        ## have an extent, this could happen. Prevent running the
        ## split code for such particles.

        int box_level = box_levels[my_box_id];
        do_split_box = do_split_box && box_level + 1 == level;
    %endif

    if (do_split_box)
    {
        morton_nr_t my_morton_nr = morton_nrs[i];
        dbg_printf(("   my morton nr: %d\n", my_morton_nr));

        morton_counts_t my_box_morton_bin_counts = box_morton_bin_counts[my_box_id];

        morton_counts_t my_morton_bin_counts = morton_bin_counts[i];
        particle_id_t my_count = get_count(my_morton_bin_counts, my_morton_nr);

        // {{{ compute this srcntgt's new index

        particle_id_t my_box_start = box_srcntgt_starts[my_box_id];
        particle_id_t tgt_particle_idx = my_box_start + my_count-1;
        %if sources_have_extent:
            tgt_particle_idx +=
                (my_morton_nr >= 0)
                    ? my_box_morton_bin_counts.nonchild_sources
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
        dbg_printf(("   moving %ld -> %d (my_box_id %d, my_box_start %d, my_count %d)\n",
            i, tgt_particle_idx,
            my_box_id, my_box_start, my_count));

        new_user_srcntgt_ids[tgt_particle_idx] = user_srcntgt_ids[i];

        // }}}

        // {{{ compute this srcntgt's new box id

        box_id_t new_box_id = split_box_ids[i] - ${2**dimensions} + my_morton_nr;

        %if sources_have_extent:
            if (my_morton_nr == -1)
                new_box_id = my_box_id;
        %endif

        dbg_printf(("   new_box_id: %d\n", new_box_id));
        dbg_assert(new_box_id >= 0);

        new_srcntgt_box_ids[tgt_particle_idx] = new_box_id;

        // }}}

        // {{{ set up child box data structure

        %for mnr in range(2**dimensions):
          /* Am I the last particle in my Morton bin? */
            %if mnr > 0:
                else
            %endif
            if (${mnr} == my_morton_nr
                && my_box_morton_bin_counts.pcnt${padded_bin(mnr, dimensions)} == my_count)
            {
                dbg_printf(("   ## splitting\n"));

                particle_id_t new_box_start = my_box_start
                %if sources_have_extent:
                    + my_box_morton_bin_counts.nonchild_sources
                %endif
                %for sub_mnr in range(mnr):
                    + my_box_morton_bin_counts.pcnt${padded_bin(sub_mnr, dimensions)}
                %endfor
                    ;

                dbg_printf(("   new_box_start: %d\n", new_box_start));

                box_start_flags[new_box_start] = 1;
                box_srcntgt_starts[new_box_id] = new_box_start;
                box_parent_ids[new_box_id] = my_box_id;
                box_morton_nrs[new_box_id] = my_morton_nr;

                particle_id_t new_count =
                    my_box_morton_bin_counts.pcnt${padded_bin(mnr, dimensions)};
                box_srcntgt_counts[new_box_id] = new_count;
                box_levels[new_box_id] = level;

                if (new_count > max_particles_in_box)
                {
                    *have_oversize_split_box = 1;
                }

                dbg_printf(("   box pcount: %d\n", box_srcntgt_counts[new_box_id]));
            }
        %endfor

        // }}}
    }
    else
    {
        // Not splitting? Copy over existing particle info.
        new_user_srcntgt_ids[i] = user_srcntgt_ids[i];
        new_srcntgt_box_ids[i] = my_box_id;
    }
""", strict_undefined=True)

# }}}

# END KERNELS IN THE LEVEL LOOP

# {{{ nonchild source count extraction

EXTRACT_NONCHILD_SOURCE_COUNT_TPL = ElementwiseTemplate(
    arguments="""//CL//
        morton_counts_t *box_morton_bin_counts,
        particle_id_t *box_nonchild_source_counts,
        """,
    operation=r"""//CL//
        box_nonchild_source_counts[i] = box_morton_bin_counts[i].nonchild_sources;
        """,
    name="extract_nonchild_source_count")

# }}}

# {{{ source/target permuter

SRCNTGT_PERMUTER_TPL = ElementwiseTemplate(
    arguments="""//CL:mako//
        particle_id_t *source_ids
        %for ax in axis_names:
            , coord_t *${ax}
        %endfor
        %for ax in axis_names:
            , coord_t *sorted_${ax}
        %endfor
        """,
    operation=r"""//CL:mako//
        particle_id_t src_idx = source_ids[i];
        %for ax in axis_names:
            sorted_${ax}[i] = ${ax}[src_idx];
        %endfor
        """,
    name="permute_srcntgt")

# }}}

# {{{ source and target index finding

SOURCE_AND_TARGET_INDEX_FINDER = ElementwiseTemplate(
    arguments="""//CL//
        particle_id_t *user_srcntgt_ids,
        particle_id_t nsources,
        box_id_t *srcntgt_box_ids,

        particle_id_t *box_srcntgt_starts,
        particle_id_t *box_srcntgt_counts,
        particle_id_t *source_numbers,

        particle_id_t *user_source_ids,
        particle_id_t *srcntgt_target_ids,
        particle_id_t *sorted_target_ids,

        particle_id_t *box_source_starts,
        particle_id_t *box_source_counts,
        particle_id_t *box_target_starts,
        particle_id_t *box_target_counts,
        """,
    operation=r"""//CL//
        particle_id_t sorted_srcntgt_id = i;
        particle_id_t source_nr = source_numbers[i];
        particle_id_t target_nr = i - source_nr;

        box_id_t box_id = srcntgt_box_ids[sorted_srcntgt_id];

        particle_id_t box_start = box_srcntgt_starts[box_id];
        particle_id_t box_count = box_srcntgt_counts[box_id];

        int srcntgt_id_in_box = i - box_start;

        particle_id_t user_srcntgt_id
            = user_srcntgt_ids[sorted_srcntgt_id];

        bool is_source = user_srcntgt_id < nsources;

        // {{{ write start and end of box in terms of sources and targets

        // first particle?
        if (sorted_srcntgt_id == box_start)
        {
            box_source_starts[box_id] = source_nr;
            box_target_starts[box_id] = target_nr;
        }

        // last particle?
        if (sorted_srcntgt_id + 1 == box_start + box_count)
        {
            particle_id_t box_start_source_nr = source_numbers[box_start];
            particle_id_t box_start_target_nr = box_start - box_start_source_nr;

            box_source_counts[box_id] =
                source_nr + (particle_id_t) is_source
                - box_start_source_nr;

            box_target_counts[box_id] =
                target_nr + 1 - (particle_id_t) is_source
                - box_start_target_nr;
        }

        // }}}

        if (is_source)
        {
            particle_id_t my_box_source_start
                = source_numbers[box_start];
            int source_id_in_box
                = source_nr - my_box_source_start;

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

# {{{ box info kernel

BOX_INFO_KERNEL_TPL =  ElementwiseTemplate(
    arguments="""//CL:mako//
        /* input */
        box_id_t *box_parent_ids,
        morton_nr_t *box_morton_nrs,
        bbox_t bbox,
        box_id_t aligned_nboxes,
        particle_id_t *box_srcntgt_counts,
        particle_id_t *box_source_counts,
        particle_id_t max_particles_in_box,

        /* output */
        box_id_t *box_child_ids, /* [2**dimensions, aligned_nboxes] */
        coord_t *box_centers, /* [dimensions, aligned_nboxes] */
        box_flags_t *box_flags, /* [nboxes] */

        /* more input input */
        %if sources_have_extent:
            particle_id_t *box_nonchild_source_counts,
        %endif
        """,
    operation=r"""//CL:mako//
        box_id_t box_id = i;

        /* Note that srcntgt_counts is a cumulative count over all children,
         * up to the point below where it is set to zero for non-leaves.
         *
         * box_srcntgt_counts is zero exactly for empty leaves because it gets
         * initialized to zero and never gets set to another value. If you
         * check above, most box info is only ever initialized *if* there's a
         * particle in the box, because the sort/build is a repeated scan over
         * *particles* (not boxes). Thus, no particle -> no work done.
         */

        particle_id_t particle_count = box_srcntgt_counts[box_id];

        %if sources_have_extent:
            const particle_id_t nonchild_source_count =
                box_nonchild_source_counts[box_id];
        %else:
            const particle_id_t nonchild_source_count = 0;
        %endif

        box_flags_t my_box_flags = 0;

        if (particle_count == 0)
        {
            // Lots of stuff uninitialized for empty leaves, prevent
            // damage by quitting now.

            // Also, those should have gotten pruned by this point,
            // unless skip_prune is True.

            box_flags[box_id] = 0; // no children, no sources
            PYOPENCL_ELWISE_CONTINUE;
        }
        else if (particle_count - nonchild_source_count > max_particles_in_box)
        {
            // This box has children, i.e. it's not a leaf.
            my_box_flags |= BOX_HAS_CHILDREN;

            // The only srcntgts allowed here are of the 'non-child source'
            // variety.
            box_srcntgt_counts[box_id] = particle_count = nonchild_source_count;

            dbg_assert(particle_count >= nonchild_source_count);

            %if sources_are_targets:
                if (particle_count - nonchild_source_count)
                    my_box_flags |= BOX_HAS_CHILD_SOURCES | BOX_HAS_CHILD_TARGETS;
            %else:
                particle_id_t source_count = box_source_counts[box_id];

                dbg_assert(source_count >= nonchild_source_count);

                particle_id_t child_source_count =
                     source_count - nonchild_source_count;
                particle_id_t child_target_count = particle_count - source_count;

                if (child_source_count)
                    my_box_flags |= BOX_HAS_CHILD_SOURCES;
                if (child_target_count)
                    my_box_flags |= BOX_HAS_CHILD_TARGETS;
            %endif

            if (nonchild_source_count)
                my_box_flags |= BOX_HAS_OWN_SOURCES;
        }
        else
        {
            // This box is a leaf, i.e. it has no children.

            dbg_assert(nonchild_source_count == 0);

            %if sources_are_targets:
                if (particle_count)
                    my_box_flags |= BOX_HAS_OWN_SOURCES | BOX_HAS_OWN_TARGETS;
            %else:
                particle_id_t my_source_count = box_source_counts[box_id];
                particle_id_t my_target_count = particle_count - my_source_count;

                if (my_source_count)
                    my_box_flags |= BOX_HAS_OWN_SOURCES;
                if (my_target_count)
                    my_box_flags |= BOX_HAS_OWN_TARGETS;
            %endif
        }

        box_flags[box_id] = my_box_flags;

        box_id_t parent_id = box_parent_ids[box_id];
        morton_nr_t morton_nr = box_morton_nrs[box_id];
        box_child_ids[parent_id + aligned_nboxes*morton_nr] = box_id;

        /* walk up to root to find center */
        coord_vec_t center = 0;

        box_id_t walk_parent_id = parent_id;
        box_id_t current_box_id = box_id;
        morton_nr_t walk_morton_nr = morton_nr;
        while (walk_parent_id != current_box_id)
        {
            %for idim in range(dimensions):
                center.s${idim} = 0.5*(
                    center.s${idim}
                    - 0.5 + (bool) (walk_morton_nr & ${2**(dimensions-1-idim)}));
            %endfor

            current_box_id = walk_parent_id;
            walk_parent_id = box_parent_ids[walk_parent_id];
            walk_morton_nr = box_morton_nrs[current_box_id];
        }

        coord_t extent = bbox.max_x - bbox.min_x;
        %for idim in range(dimensions):
        {
            box_centers[box_id + aligned_nboxes*${idim}] =
                bbox.min_${AXIS_NAMES[idim]} + extent*(0.5+center.s${idim});
        }
        %endfor
    """)

# }}}

# {{{ tree data structure (output)

class Tree(FromDeviceGettableRecord):
    """A quad/octree consisting of particles sorted into a hierarchy of boxes.
    Optionally, particles may be designated 'sources' and 'targets'. They
    may also be assigned radii which restrict the minimum size of the box
    into which they may be sorted.

    Instances of this class are not constructed directly. They are returned
    by :meth:`TreeBuilder.__call__`.

    **Flags**

    .. attribute:: sources_have_extent

        whether this tree has sources in non-leaf boxes

    **Data types**

    .. attribute:: particle_id_dtype
    .. attribute:: box_id_dtype
    .. attribute:: coord_dtype

    **Counts and sizes**

    .. attribute:: root_extent

        the root box size, a scalar

    .. attribute:: stick_out_factor

        See argument *stick_out_factor* of :meth:`Tree.__call__`.

    .. attribute:: nlevels

        the number of levels

    .. attribute:: bounding_box

        a tuple *(bbox_min, bbox_max)* of
        :mod:`numpy` vectors giving the (built) extent
        of the tree. Note that this may be slightly larger
        than what is required to contain all particles.

    .. attribute:: level_starts

        ``box_id_t [nlevels+1]``
        A :class:`numpy.ndarray` of box ids
        indicating the ID at which each level starts. Levels
        are contiguous in box ID space. To determine
        how many boxes there are in each level, check
        access the start of the next level. This array is
        built so that this works even for the last level.

    .. attribute:: level_starts_dev

        ``particle_id_t [nlevels+1``
        The same array as :attr:`level_starts`
        as a :class:`pyopencl.array.Array`.

    **Per-particle arrays**

    .. attribute:: sources

        ``coord_t [dimensions][nsources]``
        (an object array of coordinate arrays)
        Stored in :ref:`tree source order <particle-orderings>`.
        May be the same array as :attr:`targets`.

    .. attribute:: source_radii

        ``coord_t [nsources]``
        :math:`l^\infty` radii of the *sources*.
        Available if :attr:`sources_have_extent` is *True*.

    .. attribute:: targets

        ``coord_t [dimensions][nsources]``
        (an object array of coordinate arrays)
        Stored in :ref:`tree target order <particle-orderings>`. May be the same array as :attr:`sources`.

    .. attribute:: user_source_ids

        ``particle_id_t [nsources]``
        Fetching *from* these indices will reorder the sources
        from user source order into :ref:`tree source order <particle-orderings>`.

    .. attribute:: sorted_target_ids

        ``particle_id_t [ntargets]``
        Fetching *from* these indices will reorder the targets
        from :ref:`tree target order <particle-orderings>` into user target order.

    **Per-box arrays**

    .. attribute:: box_source_starts

        ``particle_id_t [nboxes]`` May be the same array as :attr:`box_target_starts`.

    .. attribute:: box_source_counts

        ``particle_id_t [nboxes]`` May be the same array as :attr:`box_target_counts`.

    .. attribute:: box_target_starts

        ``particle_id_t [nboxes]`` May be the same array as :attr:`box_source_starts`.

    .. attribute:: box_target_counts

        ``particle_id_t [nboxes]`` May be the same array as :attr:`box_source_counts`.

    .. attribute:: box_parent_ids

        ``box_id_t [nboxes]``
        Box 0 (the root) has 0 as its parent.

    .. attribute:: box_child_ids

        ``box_id_t [2**dimensions, aligned_nboxes]`` (C order)
        "0" is used as a 'no child' marker, as the root box can never
        occur as any box's child.

    .. attribute:: box_centers

        ``coord_t [dimensions, aligned_nboxes]`` (C order)

    .. attribute:: box_levels

        ``uint8 [nboxes]``

    .. attribute:: box_flags

        :attr:`box_flags_enum.dtype` ``[nboxes]``
        A combination of the :class:`box_flags_enum` constants.

    """

    @property
    def dimensions(self):
        return len(self.sources)

    @property
    def nboxes(self):
        # box_flags is created after the level loop and therefore
        # reflects the right number of boxes.
        return len(self.box_flags)

    @property
    def nsources(self):
        return len(self.user_source_ids)

    @property
    def ntargets(self):
        return len(self.sorted_target_ids)

    @property
    def nlevels(self):
        return len(self.level_starts) - 1

    @property
    def aligned_nboxes(self):
        return self.box_child_ids.shape[-1]

    def plot(self, **kwargs):
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(self)
        plotter.draw_tree(fill=False, edgecolor="black", **kwargs)

    def get_box_extent(self, ibox):
        lev = int(self.box_levels[ibox])
        box_size = self.root_extent / (1 << lev)
        extent_low = self.box_centers[:, ibox] - 0.5*box_size
        extent_high = extent_low + box_size
        return extent_low, extent_high

    def link_point_sources(self, point_source_starts, point_sources):
        """Build a :class:`TreeWithLinkedPointSources`.

        Requires that :attr:`sources_have_extent` is *True*.

        :arg point_source_starts: `point_source_starts[isrc]` and
            `point_source_starts[isrc+1]` together indicate a ranges of point
            particle indices in *point_sources* which will be linked to the
            original (extent-having) source number *isrc*. *isrc* is in :ref:`user
            point source order <particle-orderings>`.

            All the particles linked to *isrc* shoud fall within the :math:`l^\infty`
            'circle' around particle number *isrc* with the radius drawn from
            :attr:`source_radii`.

        :arg point_sources: an object array of (XYZ) point coordinate arrays.
        """

        raise NotImplementedError

class TreeWithLinkedPointSources(Tree):
    """A :class:`Tree` after processing by :meth:`Tree.link_point_sources`.
    The sources of the original tree are linked with
    extent are expanded into point sources which are linked to the
    extent-having sources in the original tree. (In an FMM context, they may
    stand in for the 'underlying' source for the purpose of the far-field
    calculation.) Has all the same attributes as :class:`Tree`.
    :attr:`Tree.sources_have_extent` is always *True* for instances of this
    type. In addition, the following attributes are available.

    Instances of this class are not constructed directly. They are returned
    by :meth:`Tree.link_point_sources`.

    .. attribute:: npoint_sources

    .. attribute:: point_source_starts

        ``particle_id_t [nsources]``
        The array
        ``point_sources[:][point_source_starts[isrc]:point_source_starts[isrc]+point_source_counts[isrc]]``
        contains the locations of point sources corresponding to
        the 'original' source with index *isrc*. (Note that this
        expression will not entirely work because :attr:`point_sources`
        is an object array.)

        This array is stored in :ref:`tree point source order <particle-orderings>`,
        unlike the parameter to :meth:`Tree.link_point_sources`.

    .. attribute:: point_source_counts

        ``particle_id_t [nsources]`` (See :attr:`point_source_starts`.)

    .. attribute:: point_sources

        ``coord_t [dimensions][npoint_sources]``
        (an object array of coordinate arrays)
        Stored in :ref:`tree point source order <particle-orderings>`.

    .. attribute:: user_point_source_ids

        ``particle_id_t [nsources]``
        Fetching *from* these indices will reorder the sources
        from user point source order into :ref:`tree point source order <particle-orderings>`.

    .. attribute:: box_point_source_starts

        ``particle_id_t [nboxes]``

    .. attribute:: box_point_source_counts

        ``particle_id_t [nboxes]``
    """

# }}}

# {{{ driver

class TreeBuilder(object):
    def __init__(self, context):
        """
        :arg context: A :class:`pyopencl.Context`.
        """

        self.context = context

        from boxtree.bounding_box import BoundingBoxFinder
        self.bbox_finder = BoundingBoxFinder(self.context)

        # This is used to map box IDs and compress box lists in empty leaf
        # pruning.

        from boxtree.tools import GappyCopyAndMapKernel
        self.gappy_copy_and_map = GappyCopyAndMapKernel(self.context)

    # {{{ kernel creation

    morton_nr_dtype = np.dtype(np.int8)
    box_level_dtype = np.dtype(np.uint8)

    @memoize_method
    def get_kernel_info(self, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype,
            sources_are_targets, sources_have_extent,
            stick_out_factor):

        # {{{ preparation

        if np.iinfo(box_id_dtype).min == 0:
            from warnings import warn
            warn("Careful with unsigned types for box_id_dtype. Some CL implementations "
                    "(notably Intel 2012) mis-implemnet unsigned operations, leading to "
                    "incorrect results.", stacklevel=4)

        from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
        coord_vec_dtype = cl.array.vec.types[coord_dtype, dimensions]

        particle_id_dtype = np.dtype(particle_id_dtype)
        box_id_dtype = np.dtype(box_id_dtype)

        dev = self.context.devices[0]
        morton_bin_count_dtype, _ = make_morton_bin_count_type(
                dev, dimensions, particle_id_dtype,
                sources_have_extent)

        from boxtree.bounding_box import make_bounding_box_dtype
        bbox_dtype, bbox_type_decl = make_bounding_box_dtype(
                dev, dimensions, coord_dtype)

        axis_names = AXIS_NAMES[:dimensions]

        from boxtree.tools import padded_bin
        codegen_args = dict(
                dimensions=dimensions,
                axis_names=axis_names,
                padded_bin=padded_bin,
                coord_dtype=coord_dtype,
                coord_vec_dtype=coord_vec_dtype,
                bbox_dtype=bbox_dtype,
                particle_id_dtype=particle_id_dtype,
                morton_bin_count_dtype=morton_bin_count_dtype,
                morton_nr_dtype=self.morton_nr_dtype,
                box_id_dtype=box_id_dtype,
                dtype_to_ctype=dtype_to_ctype,
                AXIS_NAMES=AXIS_NAMES,
                box_flags_enum=box_flags_enum,

                sources_are_targets=sources_are_targets,
                sources_have_extent=sources_have_extent,

                stick_out_factor=stick_out_factor,

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
                + str(SCAN_PREAMBLE_TPL.render(**codegen_args))
                )

        from pyopencl.tools import VectorArg, ScalarArg
        common_arguments = (
                [
                    # box-local morton bin counts for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(morton_bin_count_dtype, "morton_bin_counts"), # [nsrcntgts]

                    # (local) morton nrs for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(self.morton_nr_dtype, "morton_nrs"), # [nsrcntgts]

                    # segment flags
                    # invariant to sorting once set
                    # (particles are only reordered within a box)
                    VectorArg(np.uint8, "box_start_flags"), # [nsrcntgts]

                    VectorArg(box_id_dtype, "srcntgt_box_ids"), # [nsrcntgts]
                    VectorArg(box_id_dtype, "split_box_ids"), # [nsrcntgts]

                    # per-box morton bin counts
                    VectorArg(morton_bin_count_dtype, "box_morton_bin_counts"), # [nsrcntgts]

                    # particle# at which each box starts
                    VectorArg(particle_id_dtype, "box_srcntgt_starts"), # [nboxes]

                    # number of particles in each box
                    VectorArg(particle_id_dtype,"box_srcntgt_counts"), # [nboxes]

                    # pointer to parent box
                    VectorArg(box_id_dtype, "box_parent_ids"), # [nboxes]

                    # morton nr identifier {quadr,oct}ant of parent in which this box was created
                    VectorArg(self.morton_nr_dtype, "box_morton_nrs"), # [nboxes]

                    # number of boxes total
                    VectorArg(box_id_dtype, "nboxes"), # [1]

                    ScalarArg(np.int32, "level"),
                    ScalarArg(particle_id_dtype, "max_particles_in_box"),
                    ScalarArg(bbox_dtype, "bbox"),

                    VectorArg(particle_id_dtype, "user_srcntgt_ids"), # [nsrcntgts]
                    ]

                # particle coordinates
                + [VectorArg(coord_dtype, ax) for ax in axis_names]

                + ([VectorArg(coord_dtype, "source_radii")]
                    if sources_have_extent else [])
                )

        from pyopencl.scan import GenericScanKernel
        morton_count_scan = GenericScanKernel(
                self.context, morton_bin_count_dtype,
                arguments=common_arguments,
                input_expr="scan_t_from_particle(%s)"
                    % ", ".join([
                        "i", "level", "&bbox", "morton_nrs",
                        "user_srcntgt_ids",
                        ]
                        + ["%s" % ax for ax in axis_names]
                        + (["source_radii"] if sources_have_extent else [])),
                scan_expr="scan_t_add(a, b, across_seg_boundary)",
                neutral="scan_t_neutral()",
                is_segment_start_expr="box_start_flags[i]",
                output_statement=SCAN_OUTPUT_STMT_TPL.render(**codegen_args),
                preamble=scan_preamble)

        # }}}

        # {{{ split_box_id scan

        from pyopencl.scan import GenericScanKernel
        split_box_id_scan = SPLIT_BOX_ID_SCAN_TPL.build(
                self.context,
                type_aliases=(
                    ("scan_t", box_id_dtype),
                    ("index_t", particle_id_dtype),
                    ("particle_id_t", particle_id_dtype),
                    ("box_id_t", box_id_dtype),
                    ("morton_counts_t", morton_bin_count_dtype),
                    ("box_level_t", self.box_level_dtype),
                    ),
                var_values=(
                    ("dimensions", dimensions),
                    ("sources_have_extent", sources_have_extent),
                    ),
                more_preamble=generic_preamble)

        # }}}

        # {{{ split-and-sort

        split_and_sort_kernel_source = SPLIT_AND_SORT_KERNEL_TPL.render(**codegen_args)

        from pyopencl.elementwise import ElementwiseKernel
        split_and_sort_kernel = ElementwiseKernel(
                self.context,
                common_arguments
                + [
                    VectorArg(particle_id_dtype, "new_user_srcntgt_ids"),
                    VectorArg(np.int32, "have_oversize_split_box"),
                    VectorArg(box_id_dtype, "new_srcntgt_box_ids"),
                    VectorArg(self.box_level_dtype, "box_levels"),
                    ],
                str(split_and_sort_kernel_source), name="split_and_sort",
                preamble=(
                    preamble_with_dtype_decls
                    + str(SPLIT_AND_SORT_PREAMBLE_TPL.render(**codegen_args)))
                )

        # }}}

        # END KERNELS IN LEVEL LOOP

        if sources_have_extent:
            extract_nonchild_source_count_kernel = \
                    EXTRACT_NONCHILD_SOURCE_COUNT_TPL.build(
                            self.context,
                            type_aliases=(
                                ("particle_id_t", particle_id_dtype),
                                ("morton_counts_t", morton_bin_count_dtype),
                                ),
                            var_values=(),
                            more_preamble=generic_preamble)

        else:
            extract_nonchild_source_count_kernel = None

        # {{{ find-prune-indices

        # FIXME: Turn me into a scan template

        from pyopencl.tools import VectorArg
        find_prune_indices_kernel = GenericScanKernel(
                self.context, box_id_dtype,
                arguments=[
                    # input
                    VectorArg(particle_id_dtype, "box_srcntgt_counts"),
                    # output
                    VectorArg(box_id_dtype, "to_box_id"),
                    VectorArg(box_id_dtype, "from_box_id"),
                    VectorArg(box_id_dtype, "nboxes_post_prune"),
                    ],
                input_expr="box_srcntgt_counts[i] == 0 ? 1 : 0",
                preamble=box_flags_enum.get_c_defines(),
                scan_expr="a+b", neutral="0",
                output_statement="""
                    to_box_id[i] = i-prev_item;
                    if (box_srcntgt_counts[i])
                        from_box_id[i-prev_item] = i;
                    if (i+1 == N) *nboxes_post_prune = N-item;
                    """)

        # }}}

        # {{{ particle permuter

        # used if there is only one source/target array
        srcntgt_permuter = SRCNTGT_PERMUTER_TPL.build(
                self.context,
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
                self.context, box_id_dtype,
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

        source_and_target_index_finder = SOURCE_AND_TARGET_INDEX_FINDER.build(
                self.context,
                type_aliases=(
                    ("particle_id_t", particle_id_dtype),
                    ("box_id_t", box_id_dtype),
                    ),
                var_values=())

        # }}}

        # {{{ box-info

        type_aliases = (
                ("box_id_t", box_id_dtype),
                ("particle_id_t", particle_id_dtype),
                ("bbox_t", bbox_dtype),
                ("coord_t", coord_dtype),
                ("morton_nr_t", self.morton_nr_dtype),
                ("coord_vec_t", coord_vec_dtype),
                ("box_flags_t", box_flags_enum.dtype),
                )
        codegen_args_tuples = tuple(codegen_args.iteritems())
        box_info_kernel = BOX_INFO_KERNEL_TPL.build(
                self.context,
                type_aliases,
                var_values=codegen_args_tuples,
                more_preamble=box_flags_enum.get_c_defines() + generic_preamble,
                )

        # }}}

        return _KernelInfo(
                particle_id_dtype=particle_id_dtype,
                box_id_dtype=box_id_dtype,
                morton_bin_count_dtype=morton_bin_count_dtype,

                morton_count_scan=morton_count_scan,
                split_box_id_scan=split_box_id_scan,
                split_and_sort_kernel=split_and_sort_kernel,

                extract_nonchild_source_count_kernel=extract_nonchild_source_count_kernel,
                find_prune_indices_kernel=find_prune_indices_kernel,
                srcntgt_permuter=srcntgt_permuter,
                source_counter=source_counter,
                source_and_target_index_finder=source_and_target_index_finder,
                box_info_kernel=box_info_kernel,
                )

    # }}}

    # {{{ run control

    def __call__(self, queue, particles, max_particles_in_box,
            allocator=None, debug=False, targets=None,
            source_radii=None, stick_out_factor=0.25, **kwargs):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` instance
        :arg particles: an object array of (XYZ) point coordinate arrays.
        :arg targets: an object array of (XYZ) point coordinate arrays or ``None``.
            If ``None``, *particles* act as targets, too.
            Must have the same (inner) dtype as *particles*.
        :arg source_radii: If not *None*, a :class:`pyopencl.array.Array` of the
            same dtype as *particles*. The array specifies radii
            of :math:`l^\infty` 'circles' centered at *particles* that contain
            the entire extent of each source. Specifying this parameter
            implies that the return value of this call has
            :attr:`Tree.sources_have_extent` set to *True*.

            If this is given, *targets* must also be given, i.e. sources and
            targets must be separate.
        :arg stick_out_factor: The fraction of the box diameter by which the
            :math:`l^\infty` circles given by *source_radii* may stick out
            the box in which they are contained.
        :arg kwargs: Used internally for debugging.

        :returns: an instance of :class:`Tree`
        """

        # {{{ input processing

        dimensions = len(particles)
        axis_names = AXIS_NAMES[:dimensions]
        if source_radii is not None and targets is None:
            raise ValueError("must specify targets when specifying "
                    "source_radii")

        sources_are_targets = targets is not None
        sources_have_extent = source_radii is not None

        from pytools import single_valued
        particle_id_dtype = np.int32
        box_id_dtype = np.int32
        coord_dtype = single_valued(coord.dtype for coord in particles)

        if source_radii is not None:
            if source_radii.dtype != coord_dtype:
                raise TypeError("dtypes of coordinate arrays and "
                        "source_radii must agree")

        # }}}

        empty = partial(cl.array.empty, queue, allocator=allocator)
        zeros = partial(cl.array.zeros, queue, allocator=allocator)

        knl_info = self.get_kernel_info(dimensions, coord_dtype,
                particle_id_dtype, box_id_dtype,
                sources_are_targets, sources_have_extent,
                stick_out_factor)

        # {{{ combine sources and targets into one array, if necessary

        if targets is None:
            # Targets weren't specified. Sources are also targets. Let's
            # call them "srcntgts".

            srcntgts = particles
            nsrcntgts = single_valued(len(coord) for coord in srcntgts)

            assert source_radii is None
        else:
            # Here, we mash sources and targets into one array to give us one
            # big array of "srcntgts". In this case, a "srcntgt" is either a
            # source or a target, but not really both, as above. How will we be
            # able to tell which it was? Easy: We'll compare its 'user' id with
            # nsources. If it's >=, it's a target, otherwise it's a source.

            target_coord_dtype = single_valued(tgt_i.dtype for tgt_i in targets)

            if target_coord_dtype != coord_dtype:
                raise TypeError("sources and targets must have same coordinate "
                        "dtype")

            nsources = single_valued(len(coord) for coord in particles)
            ntargets = single_valued(len(coord) for coord in targets)
            nsrcntgts = nsources + ntargets

            def combine_srcntgt_arrays(ary1, ary2=None):
                if ary2 is None:
                    result = zeros(nsrcntgts, ary1.dtype)
                else:
                    result = empty(nsrcntgts, ary1.dtype)

                cl.enqueue_copy(queue, result.data, ary1.data)
                if ary2 is not None and ary2.nbytes:
                    cl.enqueue_copy(queue, result.data, ary2.data,
                            dest_offset=ary1.nbytes)

                return result

            from pytools.obj_array import make_obj_array
            srcntgts = make_obj_array([
                combine_srcntgt_arrays(src_i, tgt_i)
                for src_i, tgt_i in zip(particles, targets)
                ])

            if source_radii is not None:
                source_radii = combine_srcntgt_arrays(source_radii)

        del particles

        user_srcntgt_ids = cl.array.arange(queue, nsrcntgts, dtype=particle_id_dtype,
                allocator=allocator)

        # }}}

        # {{{ find and process bounding box

        bbox = self.bbox_finder(srcntgts).get()

        root_extent = max(
                bbox["max_"+ax] - bbox["min_"+ax]
                for ax in axis_names) * (1+1e-4)

        # make bbox square and slightly larger at the top, to ensure scaled
        # coordinates are always < 1
        bbox_min = np.empty(dimensions, coord_dtype)
        for i, ax in enumerate(axis_names):
            bbox_min[i] = bbox["min_"+ax]

        bbox_max = bbox_min + root_extent
        for i, ax in enumerate(axis_names):
            bbox["max_"+ax] = bbox_max[i]

        # }}}

        from pytools import div_ceil

        # {{{ allocate data

        # box-local morton bin counts for each particle at the current level
        # only valid from scan -> split'n'sort
        morton_bin_counts = empty(nsrcntgts, dtype=knl_info.morton_bin_count_dtype)

        # (local) morton nrs for each particle at the current level
        # only valid from scan -> split'n'sort
        morton_nrs = empty(nsrcntgts, dtype=self.morton_nr_dtype)

        # 0/1 segment flags
        # invariant to sorting once set
        # (particles are only reordered within a box)
        # valid throughout computation
        box_start_flags = zeros(nsrcntgts, dtype=np.int8)
        srcntgt_box_ids = zeros(nsrcntgts, dtype=box_id_dtype)
        split_box_ids = zeros(nsrcntgts, dtype=box_id_dtype)

        # number of boxes total, and a guess
        nboxes_dev = empty((), dtype=box_id_dtype)
        nboxes_dev.fill(1)

        # /!\ If you're allocating an array here that depends on nboxes_guess,
        # you *must* also write reallocation code down below for the case when
        # nboxes_guess was too low.

        # Outside nboxes_guess feeding is solely for debugging purposes,
        # to test the reallocation code.
        nboxes_guess = kwargs.get("nboxes_guess")
        if nboxes_guess  is None:
            nboxes_guess = div_ceil(nsrcntgts, max_particles_in_box) * 2**dimensions

        # per-box morton bin counts
        box_morton_bin_counts = empty(nboxes_guess,
                dtype=knl_info.morton_bin_count_dtype)

        # particle# at which each box starts
        box_srcntgt_starts = zeros(nboxes_guess, dtype=particle_id_dtype)

        # pointer to parent box
        box_parent_ids = zeros(nboxes_guess, dtype=box_id_dtype)

        # morton nr identifier {quadr,oct}ant of parent in which this box was created
        box_morton_nrs = zeros(nboxes_guess, dtype=self.morton_nr_dtype)

        # box -> level map
        box_levels = zeros(nboxes_guess, self.box_level_dtype)

        # number of particles in each box
        # needs to be globally initialized because empty boxes never get touched
        box_srcntgt_counts = zeros(nboxes_guess, dtype=particle_id_dtype)

        # Initalize box 0 to contain all particles
        cl.enqueue_copy(queue, box_srcntgt_counts.data,
                box_srcntgt_counts.dtype.type(nsrcntgts))

        # set parent of root box to itself
        cl.enqueue_copy(queue, box_parent_ids.data, box_parent_ids.dtype.type(0))

        # }}}

        # {{{ level loop

        from pytools.obj_array import make_obj_array
        have_oversize_split_box = zeros((), np.int32)

        # Level 0 starts at 0 and always contains box 0 and nothing else.
        # Level 1 therefore starts at 1.
        level_starts = [0, 1]

        from time import time
        start_time = time()
        if nsrcntgts > max_particles_in_box:
            level = 1
        else:
            level = 0

        # INVARIANTS -- Upon entry to this loop:
        #
        # - level is the level being built.
        # - the last entry of level_starts is the beginning of the level to be built

        # This while condition prevents entering the loop in case there's just a
        # single box, by how 'level' is set above. Read this as 'while True' with
        # an edge case.

        while level:
            if debug:
                # More invariants:
                assert level == len(level_starts) - 1

            if level > np.iinfo(self.box_level_dtype).max:
                raise RuntimeError("level count exceeded maximum")

            common_args = ((morton_bin_counts, morton_nrs,
                    box_start_flags, srcntgt_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    box_srcntgt_starts, box_srcntgt_counts,
                    box_parent_ids, box_morton_nrs,
                    nboxes_dev,
                    level, max_particles_in_box, bbox,
                    user_srcntgt_ids)
                    + tuple(srcntgts)
                    + ((source_radii,) if sources_have_extent else ())
                    )

            # writes: box_morton_bin_counts, morton_nrs
            knl_info.morton_count_scan(*common_args, queue=queue, size=nsrcntgts)

            # writes: nboxes_dev, split_box_ids
            knl_info.split_box_id_scan(
                    srcntgt_box_ids,
                    box_srcntgt_starts,
                    box_srcntgt_counts,
                    max_particles_in_box,
                    box_morton_bin_counts,
                    box_levels,
                    level,

                    # input/output:
                    nboxes_dev,

                    # output:
                    split_box_ids,
                    queue=queue, size=nsrcntgts)

            nboxes_new = int(nboxes_dev.get())

            # Assumption: Everything between here and the top of the loop must
            # be repeatable, so that in an out-of-memory situation, we can just
            # rerun this bit of the code after reallocating and a minimal reset
            # procedure.

            # {{{ reallocate and retry if nboxes_guess was too small

            if nboxes_new > nboxes_guess:
                while nboxes_guess < nboxes_new:
                    nboxes_guess *= 2

                from boxtree.tools import realloc_array
                my_realloc = partial(realloc_array, new_shape=nboxes_guess,
                        zero_fill=False, queue=queue)
                my_realloc_zeros = partial(realloc_array, new_shape=nboxes_guess,
                        zero_fill=True, queue=queue)

                box_morton_bin_counts = my_realloc(box_morton_bin_counts)
                box_srcntgt_starts = my_realloc_zeros(box_srcntgt_starts)
                box_parent_ids = my_realloc_zeros(box_parent_ids)
                box_morton_nrs = my_realloc_zeros(box_morton_nrs)
                box_levels = my_realloc_zeros(box_levels)
                box_srcntgt_counts = my_realloc_zeros(box_srcntgt_counts)

                del my_realloc
                del my_realloc_zeros

                # reset nboxes_dev to previous value
                nboxes_dev.fill(level_starts[-1])

                # retry
                if debug:
                    print "nboxes_guess exceeded: enlarged allocations, restarting level"

                continue

            # }}}

            if debug:
                print "LEVEL %d -> %d boxes" % (level, nboxes_new)

            assert level_starts[-1] != nboxes_new or sources_have_extent

            if level_starts[-1] == nboxes_new:
                # We haven't created new boxes in this level loop trip.  Unless
                # sources have extent, this should never happen.  (I.e., we
                # should've never entered this loop trip.)
                #
                # If sources have extent, this can happen if boxes were
                # in-principle overfull, but couldn't subdivide because of
                # extent restrictions.

                assert sources_have_extent

                level -= 1

                break

            level_starts.append(nboxes_new)
            del nboxes_new

            new_user_srcntgt_ids = cl.array.empty_like(user_srcntgt_ids)
            new_srcntgt_box_ids = cl.array.empty_like(srcntgt_box_ids)
            split_and_sort_args = (
                    common_args
                    + (new_user_srcntgt_ids, have_oversize_split_box,
                        new_srcntgt_box_ids, box_levels))
            knl_info.split_and_sort_kernel(*split_and_sort_args)

            if debug:
                level_bl_chunk = box_levels.get()[level_starts[-2]:level_starts[-1]]
                assert ((level_bl_chunk == level) | (level_bl_chunk == 0)).all()
                del level_bl_chunk

            if debug:
                assert (box_srcntgt_starts.get() < nsrcntgts).all()

            user_srcntgt_ids = new_user_srcntgt_ids
            del new_user_srcntgt_ids
            srcntgt_box_ids = new_srcntgt_box_ids
            del new_srcntgt_box_ids

            if not int(have_oversize_split_box.get()):
                break

            level += 1

            have_oversize_split_box.fill(0)

        end_time = time()
        if debug:
            elapsed = end_time-start_time
            npasses = level+1
            print "elapsed time: %g s (%g s/particle/pass)" % (
                    elapsed, elapsed/(npasses*nsrcntgts))
            del npasses

        nboxes = int(nboxes_dev.get())

        # }}}

        # {{{ extract number of non-child sources from box morton counts

        if sources_have_extent:
            box_nonchild_source_counts = empty(nboxes, particle_id_dtype)
            knl_info.extract_nonchild_source_count_kernel(
                    box_morton_bin_counts,
                    box_nonchild_source_counts,
                    range=slice(nboxes))

        # }}}

        del morton_nrs
        del box_morton_bin_counts

        # {{{ prune empty leaf boxes

        is_pruned = not kwargs.get("skip_prune")
        if is_pruned:
            # What is the original index of this box?
            from_box_id = empty(nboxes, box_id_dtype)

            # Where should I put this box?
            to_box_id = empty(nboxes, box_id_dtype)

            nboxes_post_prune_dev = empty((), dtype=box_id_dtype)
            knl_info.find_prune_indices_kernel(
                    box_srcntgt_counts, to_box_id, from_box_id, nboxes_post_prune_dev,
                    size=nboxes)

            nboxes_post_prune = int(nboxes_post_prune_dev.get())

            if debug:
                print "%d empty leaves" % (nboxes-nboxes_post_prune)

            prune_empty = partial(self.gappy_copy_and_map,
                    queue, allocator, nboxes_post_prune, from_box_id)

            box_srcntgt_starts = prune_empty(box_srcntgt_starts)
            box_srcntgt_counts = prune_empty(box_srcntgt_counts)

            if debug:
                assert (box_srcntgt_counts.get() > 0).all()

            srcntgt_box_ids = cl.array.take(to_box_id, srcntgt_box_ids)

            box_parent_ids = prune_empty(box_parent_ids, map_values=to_box_id)
            box_morton_nrs = prune_empty(box_morton_nrs)
            box_levels = prune_empty(box_levels)
            if sources_have_extent:
                box_nonchild_source_counts = prune_empty(
                        box_nonchild_source_counts)

            # Remap level_starts to new box IDs.
            # FIXME: It would be better to do this on the device.
            level_starts = list(to_box_id.get()[np.array(level_starts[:-1], box_id_dtype)])
            level_starts = level_starts + [nboxes_post_prune]
        else:
            nboxes_post_prune = nboxes

        level_starts = np.array(level_starts, box_id_dtype)

        # }}}

        del nboxes

        # {{{ compute source/target particle indices and counts in each box

        # {{{ helper: turn a "to" index list into a "from" index list

        # (= 'transpose'/invert a permutation)

        def reverse_particle_index_array(orig_indices):
            n = len(orig_indices)
            result = empty(n, particle_id_dtype)
            cl.array.multi_put(
                    [cl.array.arange(queue, n, dtype=particle_id_dtype,
                        allocator=allocator)],
                    orig_indices,
                    out=[result],
                    queue=queue)

            return result

        # }}}

        if targets is None:
            user_source_ids = user_srcntgt_ids
            sorted_target_ids = reverse_particle_index_array(
                    user_srcntgt_ids)

            box_source_starts = box_target_starts = box_srcntgt_starts
            box_source_counts = box_target_counts = box_srcntgt_counts
        else:
            source_numbers = empty(nsrcntgts, particle_id_dtype)
            knl_info.source_counter(user_srcntgt_ids, nsources,
                    source_numbers, queue=queue, allocator=allocator)

            user_source_ids = empty(nsources, particle_id_dtype)
            # srcntgt_target_ids is temporary until particle permutation is done
            srcntgt_target_ids = empty(ntargets, particle_id_dtype)
            sorted_target_ids = empty(ntargets, particle_id_dtype)

            # need to use zeros because parent boxes won't be initialized
            box_source_starts = zeros(nboxes_post_prune, particle_id_dtype)
            box_source_counts = zeros(nboxes_post_prune, particle_id_dtype)
            box_target_starts = zeros(nboxes_post_prune, particle_id_dtype)
            box_target_counts = zeros(nboxes_post_prune, particle_id_dtype)

            knl_info.source_and_target_index_finder(
                    # input:
                    user_srcntgt_ids, nsources, srcntgt_box_ids,
                    box_srcntgt_starts, box_srcntgt_counts,
                    source_numbers,

                    # output:
                    user_source_ids, srcntgt_target_ids, sorted_target_ids,
                    box_source_starts, box_source_counts,
                    box_target_starts, box_target_counts,
                    queue=queue, range=slice(nsrcntgts))

            if debug:
                usi_host = user_source_ids.get()
                assert (usi_host < nsources).all()
                assert (0 <= usi_host).all()
                del usi_host

                sti_host = srcntgt_target_ids.get()
                assert (sti_host < nsources+ntargets).all()
                assert (nsources <= sti_host).all()
                del sti_host

                counts = box_srcntgt_counts.get()
                is_leaf = counts <= max_particles_in_box
                assert (box_source_counts.get()[is_leaf] + box_target_counts.get()[is_leaf]
                        == box_srcntgt_counts.get()[is_leaf]).all()
                del counts
                del is_leaf

            del source_numbers

        del box_srcntgt_starts

        # }}}

        # {{{ permute and source/target-split (if necessary) particle array

        if targets is None:
            sources = targets = make_obj_array([
                cl.array.empty_like(pt) for pt in srcntgts])
            knl_info.srcntgt_permuter(
                    user_srcntgt_ids,
                    *(tuple(srcntgts) + tuple(sources)))

            assert source_radii is None

        else:
            sources = make_obj_array([
                empty(nsources, coord_dtype) for i in xrange(dimensions)])
            knl_info.srcntgt_permuter(
                    user_source_ids,
                    *(tuple(srcntgts) + tuple(sources)),
                    queue=queue, range=slice(nsources))

            targets = make_obj_array([
                empty(ntargets, coord_dtype) for i in xrange(dimensions)])
            knl_info.srcntgt_permuter(
                    srcntgt_target_ids,
                    *(tuple(srcntgts) + tuple(targets)),
                    queue=queue, range=slice(ntargets))

            if source_radii is not None:
                source_radii = cl.array.take(
                        source_radii, user_source_ids, queue=queue)

            del srcntgt_target_ids

        # }}}

        del srcntgts

        # {{{ compute box info

        # A number of arrays below are nominally 2-dimensional and stored with
        # the box index as the fastest-moving index. To make sure that accesses
        # remain aligned, we round up the number of boxes used for indexing.
        aligned_nboxes = div_ceil(nboxes_post_prune, 32)*32

        box_child_ids = zeros((2**dimensions, aligned_nboxes), box_id_dtype)
        box_centers = empty((dimensions, aligned_nboxes), coord_dtype)
        box_flags = empty(nboxes_post_prune, box_flags_enum.dtype)

        knl_info.box_info_kernel(
                # input:
                box_parent_ids, box_morton_nrs, bbox, aligned_nboxes,
                box_srcntgt_counts, box_source_counts,
                max_particles_in_box,

                # output:
                box_child_ids, box_centers, box_flags,
                *(
                    (box_nonchild_source_counts,) if sources_have_extent else ()),
                range=slice(nboxes_post_prune))

        # }}}

        nlevels = len(level_starts) - 1
        assert level + 1 == nlevels, (level+1, nlevels)
        if debug:
            max_level = np.max(box_levels.get())

            assert max_level + 1 == nlevels

        del nlevels

        # {{{ build output

        return Tree(
                # If you change this, also change the documentation
                # of what's in the tree, above.

                particle_id_dtype=knl_info.particle_id_dtype,
                box_id_dtype=knl_info.box_id_dtype,
                coord_dtype=coord_dtype,

                root_extent=root_extent,
                stick_out_factor=stick_out_factor,

                bounding_box=(bbox_min, bbox_max),
                level_starts=level_starts,
                level_starts_dev=cl.array.to_device(queue, level_starts,
                    allocator=allocator),

                sources=sources,
                targets=targets,

                box_source_starts=box_source_starts,
                box_source_counts=box_source_counts,
                box_target_starts=box_target_starts,
                box_target_counts=box_target_counts,

                box_parent_ids=box_parent_ids,
                box_child_ids=box_child_ids,
                box_centers=box_centers,
                box_levels=box_levels,
                box_flags=box_flags,

                user_source_ids=user_source_ids,
                sorted_target_ids=sorted_target_ids,

                _is_pruned=is_pruned,
                )

        # }}}

    # }}}

# }}}




# vim: filetype=pyopencl:fdm=marker
