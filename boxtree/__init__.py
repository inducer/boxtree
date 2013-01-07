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
from mako.template import Template
from functools import partial
from boxtree.tools import FromDeviceGettableRecord, get_type_moniker

__doc__ = """
This tree builder can be run in two modes:

* one where no distinction is made between sources and targets. In this mode,
  all participants in the interaction are called 'particles'.

* one where a distinction is made.

Particle Orderings
------------------

There are four particle orderings:

* **user source order**
* **tree (sorted) source order**
* **user target order**
* **tree (sorted) target order**

:attr:`Tree.user_source_ids` helps translate source arrays into
tree order for processing. :attr:`Tree.sorted_target_ids`
helps translate potentials back into user target order for output.
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




AXIS_NAMES = ("x", "y", "z", "w")

# {{{ bounding box finding

@memoize
def make_bounding_box_dtype(device, dimensions, coord_dtype):
    fields = []
    for i in range(dimensions):
        fields.append(("min_%s" % AXIS_NAMES[i], coord_dtype))
        fields.append(("max_%s" % AXIS_NAMES[i], coord_dtype))

    dtype = np.dtype(fields)

    name = "boxtree_bbox_%dd_t" % dimensions

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    return dtype, c_decl




BBOX_CODE_TEMPLATE = Template(r"""//CL//
    ${bbox_struct_decl}

    typedef boxtree_bbox_${dimensions}d_t bbox_t;
    typedef ${coord_ctype} coord_t;

    bbox_t bbox_neutral()
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${coord_dtype_3ltr}_MAX;
            result.max_${ax} = -${coord_dtype_3ltr}_MAX;
        %endfor
        return result;
    }

    bbox_t bbox_from_particle(${", ".join("coord_t %s" % ax for ax in axis_names)})
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${ax};
            result.max_${ax} = ${ax};
        %endfor
        return result;
    }

    bbox_t agg_bbox(bbox_t a, bbox_t b)
    {
        %for ax in axis_names:
            a.min_${ax} = min(a.min_${ax}, b.min_${ax});
            a.max_${ax} = max(a.max_${ax}, b.max_${ax});
        %endfor
        return a;
    }
""", strict_undefined=True)

class BoundingBoxFinder:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernel(self, dimensions, coord_dtype):
        from pyopencl.tools import dtype_to_ctype
        bbox_dtype, bbox_cdecl = make_bounding_box_dtype(
                self.context.devices[0], dimensions, coord_dtype)

        if coord_dtype == np.float64:
            coord_dtype_3ltr = "DBL"
        elif coord_dtype == np.float32:
            coord_dtype_3ltr = "FLT"
        else:
            raise TypeError("unknown coord_dtype")

        axis_names = AXIS_NAMES[:dimensions]

        coord_ctype = dtype_to_ctype(coord_dtype)

        preamble = BBOX_CODE_TEMPLATE.render(
                axis_names=axis_names,
                dimensions=dimensions,
                coord_ctype=dtype_to_ctype(coord_dtype),
                coord_dtype_3ltr=coord_dtype_3ltr,
                bbox_struct_decl=bbox_cdecl
                )

        from pyopencl.reduction import ReductionKernel
        return ReductionKernel(self.context, bbox_dtype,
                neutral="bbox_neutral()",
                reduce_expr="agg_bbox(a, b)",
                map_expr="bbox_from_particle(%s)" % ", ".join(
                    "%s[i]" % ax for ax in axis_names),
                arguments=", ".join(
                    "__global %s *%s" % (coord_ctype, ax) for ax in axis_names),
                preamble=preamble,
                name="bounding_box")

    def __call__(self, particles):
        dimensions = len(particles)

        from pytools import single_valued
        coord_dtype = single_valued(coord.dtype for coord in particles)

        return self.get_kernel(dimensions, coord_dtype)(*particles)

# }}}

class _KernelInfo(Record):
    pass

def padded_bin(i, l):
    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s

# {{{ data types

@memoize
def make_morton_bin_count_type(device, dimensions, particle_id_dtype):
    fields = []
    for mnr in range(2**dimensions):
        fields.append(('c%s' % padded_bin(mnr, dimensions), particle_id_dtype))

    dtype = np.dtype(fields)

    name = "boxtree_morton_bin_count_%dd_p%s_t" % (
            dimensions,
            get_type_moniker(particle_id_dtype)
            )

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

@memoize
def make_scan_type(device, dimensions, particle_id_dtype, box_id_dtype):
    morton_dtype, _ = make_morton_bin_count_type(device, dimensions, particle_id_dtype)
    dtype = np.dtype([
            ('counts', morton_dtype),
            ('split_box_id', box_id_dtype), # sum-scanned
            ('morton_nr', np.uint8),
            ])

    name = "boxtree_tree_scan_%dd_p%s_b%s_t" % (
            dimensions,
            get_type_moniker(particle_id_dtype),
            get_type_moniker(box_id_dtype)
            )

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

# }}}

class box_flags_enum:
    """Constants for box types."""

    dtype = np.dtype(np.uint8)

    HAS_SOURCES = 1
    HAS_CHILDREN = 2

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


# {{{ preamble

PREAMBLE_TPL = Template(r"""//CL//
    ${bbox_type_decl}
    ${morton_bin_count_type_decl}
    ${tree_scan_type_decl}

    typedef ${dtype_to_ctype(morton_bin_count_dtype)} morton_t;
    typedef ${dtype_to_ctype(scan_dtype)} scan_t;
    typedef boxtree_bbox_${dimensions}d_t bbox_t;
    typedef ${dtype_to_ctype(coord_dtype)} coord_t;
    typedef ${dtype_to_ctype(coord_vec_dtype)} coord_vec_t;
    typedef ${dtype_to_ctype(box_id_dtype)} box_id_t;
    typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;
    typedef ${dtype_to_ctype(morton_nr_dtype)} morton_nr_t;

    <%
      def get_count_for_branch(known_bits):
          if len(known_bits) == dimensions:
              return "counts.c%s" % known_bits

          dim = len(known_bits)
          boundary_morton_nr = known_bits + "1" + (dimensions-dim-1)*"0"

          return ("((morton_nr < %s) ? %s : %s)" % (
              int(boundary_morton_nr, 2),
              get_count_for_branch(known_bits+"0"),
              get_count_for_branch(known_bits+"1")))
    %>

    particle_id_t get_count(morton_t counts, int morton_nr)
    {
        return ${get_count_for_branch("")};
    }

    #ifdef DEBUG
        #define dbg_printf(ARGS) printf ARGS
    #else
        #define dbg_printf(ARGS) /* */
    #endif

""", strict_undefined=True)

# }}}

# {{{ scan primitive code template

SCAN_PREAMBLE_TPL = Template(r"""//CL//
    scan_t scan_t_neutral()
    {
        scan_t result;
        %for mnr in range(2**dimensions):
            result.counts.c${padded_bin(mnr, dimensions)} = 0;
        %endfor
        result.split_box_id = 0;
        return result;
    }

    scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary)
    {
        if (!across_seg_boundary)
        {
            %for mnr in range(2**dimensions):
                <% field = "counts.c"+padded_bin(mnr, dimensions) %>
                b.${field} = a.${field} + b.${field};
            %endfor
        }

        // split_box_id must use a non-segmented scan to globally
        // assign box numbers.
        b.split_box_id = a.split_box_id + b.split_box_id;

        // b.morton_nr gets propagated
        return b;
    }

    scan_t scan_t_from_particle(
        int i,
        int level,
        box_id_t box_id,
        box_id_t nboxes,
        particle_id_t box_start,
        particle_id_t box_srcntgt_count,
        particle_id_t max_particles_in_box,
        bbox_t *bbox
        %for ax in axis_names:
            , coord_t ${ax}
        %endfor
    )
    {
        // Note that the upper bound must be slightly larger than the highest
        // found coordinate, so that 1.0 is never reached as a scaled
        // coordinate.

        %for ax in axis_names:
            unsigned ${ax}_bits = (unsigned) (
                ((${ax}-bbox->min_${ax})/(bbox->max_${ax}-bbox->min_${ax}))
                * (1U << (level+1)));
        %endfor

        unsigned level_morton_number = 0
        %for iax, ax in enumerate(axis_names):
            | (${ax}_bits & 1U) << (${dimensions-1-iax})
        %endfor
            ;

        scan_t result;
        %for mnr in range(2**dimensions):
            <% field = "counts.c"+padded_bin(mnr, dimensions) %>
            result.${field} = (level_morton_number == ${mnr});
        %endfor
        result.morton_nr = level_morton_number;

        // split_box_id is not very meaningful now, but when scanned over
        // by addition, will yield new, unused ids for boxes that are created by
        // subdividing the current box (if it is over-full).

        result.split_box_id = 0;
        if (i == 0)
        {
            // Particle number zero brings in the box count from the
            // previous level.

            result.split_box_id = nboxes;
        }
        if (i == box_start
            && box_srcntgt_count > max_particles_in_box)
        {
            // If this box is overfull, put in a 'request' for 2**d sub-box
            // IDs. Sub-boxes will have to subtract from the total to find
            // their id. These requested box IDs are then scanned over by
            // a global sum.

            result.split_box_id += ${2**dimensions};
        }

        return result;
    }

""", strict_undefined=True)

# }}}

# {{{ scan output code template

SCAN_OUTPUT_STMT_TPL = Template(r"""//CL//
    {
        particle_id_t my_id_in_my_box = -1
        %for mnr in range(2**dimensions):
            + item.counts.c${padded_bin(mnr, dimensions)}
        %endfor
            ;
        dbg_printf(("my_id_in_my_box:%d\n", my_id_in_my_box));
        morton_bin_counts[i] = item.counts;
        morton_nrs[i] = item.morton_nr;

        box_id_t current_box_id = srcntgt_box_ids[i];
        particle_id_t box_srcntgt_count = box_srcntgt_counts[current_box_id];

        split_box_ids[i] = item.split_box_id;

        // Am I the last particle in my current box?
        // If so, populate particle count.

        if (my_id_in_my_box+1 == box_srcntgt_count)
        {
            dbg_printf(("store box %d cbi:%d\n", i, current_box_id));
            dbg_printf(("   store_sums: %d %d %d %d\n", item.counts.c00, item.counts.c01, item.counts.c10, item.counts.c11));
            box_morton_bin_counts[current_box_id] = item.counts;
        }

        // Am I the last particle overall? If so, write box count
        if (i+1 == N)
            *nboxes = item.split_box_id;
    }
""", strict_undefined=True)

# }}}

# {{{ split-and-sort kernel

SPLIT_AND_SORT_KERNEL_TPL =  Template(r"""//CL//
    morton_t my_morton_bin_counts = morton_bin_counts[i];
    box_id_t my_box_id = srcntgt_box_ids[i];

    dbg_printf(("postproc %d:\n", i));
    dbg_printf(("   my_sums: %d %d %d %d\n",
        my_morton_bin_counts.c00, my_morton_bin_counts.c01,
        my_morton_bin_counts.c10, my_morton_bin_counts.c11));
    dbg_printf(("   my box id: %d\n", my_box_id));

    particle_id_t box_srcntgt_count = box_srcntgt_counts[my_box_id];

    /* Is this box being split? */
    if (box_srcntgt_count > max_particles_in_box)
    {
        morton_nr_t my_morton_nr = morton_nrs[i];
        dbg_printf(("   my morton nr: %d\n", my_morton_nr));

        box_id_t new_box_id = split_box_ids[i] - ${2**dimensions} + my_morton_nr;
        dbg_printf(("   new_box_id: %d\n", new_box_id));

        morton_t my_box_morton_bin_counts = box_morton_bin_counts[my_box_id];
        /*
        dbg_printf(("   box_sums: %d %d %d %d\n", my_box_morton_bin_counts.c00, my_box_morton_bin_counts.c01, my_box_morton_bin_counts.c10, my_box_morton_bin_counts.c11));
        */

        particle_id_t my_count = get_count(my_morton_bin_counts, my_morton_nr);

        particle_id_t my_box_start = box_srcntgt_starts[my_box_id];
        particle_id_t tgt_particle_idx = my_box_start + my_count-1;
        %for mnr in range(2**dimensions):
            <% bin_nmr = padded_bin(mnr, dimensions) %>
            tgt_particle_idx +=
                (my_morton_nr > ${mnr})
                    ? my_box_morton_bin_counts.c${bin_nmr}
                    : 0;
        %endfor

        dbg_printf(("   moving %d -> %d\n", i, tgt_particle_idx));

        new_user_srcntgt_ids[tgt_particle_idx] = user_srcntgt_ids[i];
        new_srcntgt_box_ids[tgt_particle_idx] = new_box_id;

        %for mnr in range(2**dimensions):
          /* Am I the last particle in my Morton bin? */
            %if mnr > 0:
                else
            %endif
            if (${mnr} == my_morton_nr
                && my_box_morton_bin_counts.c${padded_bin(mnr, dimensions)} == my_count)
            {
                dbg_printf(("   ## splitting\n"));

                particle_id_t new_box_start = my_box_start
                %for sub_mnr in range(mnr):
                    + my_box_morton_bin_counts.c${padded_bin(sub_mnr, dimensions)}
                %endfor
                    ;

                dbg_printf(("   new_box_start: %d\n", new_box_start));

                box_start_flags[new_box_start] = 1;
                box_srcntgt_starts[new_box_id] = new_box_start;
                box_parent_ids[new_box_id] = my_box_id;
                box_morton_nrs[new_box_id] = my_morton_nr;

                particle_id_t new_count =
                    my_box_morton_bin_counts.c${padded_bin(mnr, dimensions)};
                box_srcntgt_counts[new_box_id] = new_count;
                if (new_count > max_particles_in_box)
                    *have_oversize_box = 1;

                dbg_printf(("   box pcount: %d\n", box_srcntgt_counts[new_box_id]));
            }
        %endfor
    }
    else
    {
        // Not splitting? Copy over existing particle info.
        new_user_srcntgt_ids[i] = user_srcntgt_ids[i];
        new_srcntgt_box_ids[i] = my_box_id;
    }
""", strict_undefined=True)

# }}}

# {{{ box info kernel

BOX_INFO_KERNEL_TPL =  ElementwiseTemplate(
    arguments="""//CL//
        /* input */
        box_id_t *box_parent_ids,
        morton_nr_t *box_morton_nrs,
        bbox_t bbox,
        box_id_t aligned_nboxes,
        particle_id_t *box_srcntgt_counts,
        particle_id_t max_particles_in_box,
        /* output */
        box_id_t *box_child_ids, /* [2**dimensions, aligned_nboxes] */
        coord_t *box_centers, /* [dimensions, aligned_nboxes] */
        unsigned char *box_levels, /* [nboxes] */
        box_flags_t *box_flags, /* [nboxes] */
        """,
    operation=r"""//CL:mako//
        box_id_t box_id = i;

        particle_id_t p_count = box_srcntgt_counts[box_id];
        if (p_count == 0)
        {
            // Lots of stuff uninitialized for these guys, prevent
            // damage by quitting now.

            // Also, those should have gotten pruned by this point.

            box_flags[box_id] = 0; // no children, no sources
            return;
        }
        else if (p_count > max_particles_in_box)
        {
            box_flags[box_id] = BOX_HAS_CHILDREN;
            box_srcntgt_counts[box_id] = 0;
        }
        else
            box_flags[box_id] = BOX_HAS_SOURCES;

        box_id_t parent_id = box_parent_ids[box_id];
        morton_nr_t morton_nr = box_morton_nrs[box_id];
        box_child_ids[parent_id + aligned_nboxes*morton_nr] = box_id;

        /* walk up to root to find center and level */
        coord_vec_t center = 0;
        int level = 0;

        box_id_t walk_parent_id = parent_id;
        box_id_t current_box_id = box_id;
        morton_nr_t walk_morton_nr = morton_nr;
        while (walk_parent_id != current_box_id)
        {
            ++level;

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

        box_levels[box_id] = level;

        /* box_srcntgt_counts is zero for empty leaves because it gets initialized
         * to zero and never gets set. If you check above, most box info is only
         * ever initialized *if* there's a particle in the box, because the sort/build
         * is a repeated scan over *particles* (not boxes). Thus, no particle -> no
         * work done.
         */

    """)

# }}}

# {{{ gappy copy kernel (for empty leaf pruning)

# This is used to map box IDs and compress box lists in empty leaf
# pruning.

GAPPY_COPY_TPL =  Template(r"""//CL//

    typedef ${dtype_to_ctype(dtype)} value_t;

    value_t val = input_ary[from_indices[i]];

    %if map_values:
        val = value_map[val];
    %endif

    output_ary[i] = val;

""", strict_undefined=True)

# }}}

# {{{ tree data structure (output)

class Tree(FromDeviceGettableRecord):
    """
    **Data types**

    .. attribute:: particle_id_dtype
    .. attribute:: box_id_dtype
    .. attribute:: coord_dtype

    **Counts and sizes**

    .. attribute:: root_extent

        the root box size, a scalar

    .. attribute:: nlevels

        the number of levels

    .. attribute:: bounding_box

        a tuple *(bbox_min, bbox_max)* of
        :mod:`numpy` vectors giving the (built) extent
        of the tree. Note that this may be slightly larger
        than what is required to contain all particles.

    .. attribute:: level_starts

        `box_id_t [nlevels+1]`
        A :class:`numpy.ndarray` of box ids
        indicating the ID at which each level starts. Levels
        are contiguous in box ID space. To determine
        how many boxes there are in each level, check
        access the start of the next level. This array is
        built so that this works even for the last level.

    .. attribute:: level_starts_dev

        `particle_id_t [nlevels+1`
        The same array as :attr:`level_starts`
        as a :class:`pyopencl.array.Array`.

    **Per-particle arrays**

    .. attribute:: sources

        `coord_t [dimensions][nsources]`
        (an object array of coordinate arrays)
        Stored in tree order. May be the same array as :attr:`targets`.

    .. attribute:: targets

        `coord_t [dimensions][nsources]`
        (an object array of coordinate arrays)
        Stored in tree order. May be the same array as :attr:`sources`.

    .. attribute:: user_source_ids

        `particle_id_t [nsources]`
        Fetching *from* these indices will reorder the sources
        from user order into tree order.

    .. attribute:: sorted_target_ids

        `particle_id_t [ntargets]`
        Fetching *from* these indices will reorder the targets
        from tree order into user order.

    **Per-box arrays**

    .. attribute:: box_source_starts

        `particle_id_t [nboxes]` May be the same array as :attr:`box_target_starts`.

    .. attribute:: box_source_counts

        `particle_id_t [nboxes]` May be the same array as :attr:`box_target_counts`.

    .. attribute:: box_target_starts

        `particle_id_t [nboxes]` May be the same array as :attr:`box_source_starts`.

    .. attribute:: box_target_counts

        `particle_id_t [nboxes]` May be the same array as :attr:`box_source_counts`.

    .. attribute:: box_parent_ids

        `box_id_t [nboxes]`
        Box 0 (the root) has 0 as its parent.

    .. attribute:: box_child_ids

        `box_id_t [2**dimensions, aligned_nboxes]` (C order)
        "0" is used as a 'no child' marker, as the root box can never
        occur as any box's child.

    .. attribute:: box_centers

        `coord_t` [dimensions, aligned_nboxes] (C order)

    .. attribute:: box_levels

        `uint8 [nboxes]`
    .. attribute:: box_flags

        :attr:`box_flags_enum.dtype` `[nboxes]`
        A combination of the :class:`box_flags_enum` constants.
    """

    @property
    def dimensions(self):
        return len(self.sources)

    @property
    def nboxes(self):
        return self.box_levels.shape[0]

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
        plotter = TreePlotter(self)
        plotter.draw_tree(fill=False, edgecolor="black", **kwargs)

    def get_box_extent(self, ibox):
        lev = int(self.box_levels[ibox])
        box_size = self.root_extent / (1 << lev)
        extent_low = self.box_centers[:, ibox] - 0.5*box_size
        extent_high = extent_low + box_size
        return extent_low, extent_high


# }}}

# {{{ visualization

class TreePlotter:
    """Assumes that the tree has data living on the host. See :meth:`Tree.get`."""
    def __init__(self, tree):
        self.tree = tree

    def draw_tree(self, **kwargs):
        if self.tree.dimensions != 2:
            raise NotImplementedError("can only plot 2D trees for now")

        for ibox in xrange(self.tree.nboxes):
            self.draw_box(ibox, **kwargs)

    def set_bounding_box(self):
        import matplotlib.pyplot as pt
        bbox_min, bbox_max = self.tree.bounding_box
        pt.xlim(bbox_min[0], bbox_max[0])
        pt.ylim(bbox_min[1], bbox_max[1])

    def draw_box(self, ibox, **kwargs):
        """
        :arg kwargs: keyword arguments to pass on to :class:`matplotlib.patches.PathPatch`,
            e.g. `facecolor='red', edgecolor='yellow', alpha=0.5`
        """

        el, eh = self.tree.get_box_extent(ibox)

        import matplotlib.pyplot as pt
        import matplotlib.patches as mpatches
        from matplotlib.path import Path

        pathdata = [
            (Path.MOVETO, (el[0], el[1])),
            (Path.LINETO, (eh[0], el[1])),
            (Path.LINETO, (eh[0], eh[1])),
            (Path.LINETO, (el[0], eh[1])),
            (Path.CLOSEPOLY, (el[0], el[1])),
            ]

        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        pt.gca().add_patch(patch)

    def draw_box_numbers(self):
        import matplotlib.pyplot as pt
        for ibox in xrange(self.tree.nboxes):
            x, y = self.centers[:, ibox]
            lev = int(self.levels[ibox])
            pt.text(x, y, str(ibox), fontsize=20*1.15**(-lev),
                    ha="center", va="center",
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))


# }}}

# {{{ driver

def _realloc_array(ary, new_shape, zero_fill):
    new_ary = cl.array.empty(ary.queue, shape=new_shape, dtype=ary.dtype,
            allocator=ary.allocator)
    if zero_fill:
        new_ary.fill(0)
    cl.enqueue_copy(ary.queue, new_ary.data, ary.data, byte_count=ary.nbytes)
    return new_ary




class TreeBuilder(object):
    def __init__(self, context):
        self.context = context

    # {{{ kernel creation

    @memoize_method
    def get_bbox_finder(self):
        return BoundingBoxFinder(self.context)

    @memoize_method
    def get_gappy_copy_and_map_kernel(self, dtype, src_index_dtype, map_values=False):
        from pyopencl.tools import VectorArg
        from pyopencl.elementwise import ElementwiseKernel

        args = [
                VectorArg(dtype, "input_ary"),
                VectorArg(dtype, "output_ary"),
                VectorArg(src_index_dtype, "from_indices")
                ]

        if map_values:
            args.append(VectorArg(dtype, "value_map"))

        from pyopencl.tools import dtype_to_ctype
        src = GAPPY_COPY_TPL.render(
                dtype=dtype,
                dtype_to_ctype=dtype_to_ctype,
                map_values=map_values)

        return ElementwiseKernel(self.context,
                args, str(src), name="gappy_copy_and_map")

    morton_nr_dtype = np.dtype(np.uint8)

    @memoize_method
    def get_kernel_info(self, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype):
        if np.iinfo(box_id_dtype).min == 0:
            from warnings import warn
            warn("Careful with signed types for box_id_dtype. Some CL implementations "
                    "(notably Intel 2012) mis-implemnet signed operations, leading to "
                    "incorrect results.", stacklevel=4)

        from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
        coord_vec_dtype = cl.array.vec.types[coord_dtype, dimensions]

        particle_id_dtype = np.dtype(particle_id_dtype)
        box_id_dtype = np.dtype(box_id_dtype)

        dev = self.context.devices[0]
        scan_dtype, scan_type_decl = make_scan_type(dev,
                dimensions, particle_id_dtype, box_id_dtype)
        morton_bin_count_dtype, _ = scan_dtype.fields["counts"]
        bbox_dtype, bbox_type_decl = make_bounding_box_dtype(
                dev, dimensions, coord_dtype)

        axis_names = AXIS_NAMES[:dimensions]

        codegen_args = dict(
                dimensions=dimensions,
                axis_names=axis_names,
                padded_bin=padded_bin,
                coord_dtype=coord_dtype,
                coord_vec_dtype=coord_vec_dtype,
                morton_bin_count_type_decl=dtype_to_c_struct(
                    dev, morton_bin_count_dtype),
                tree_scan_type_decl=scan_type_decl,
                bbox_type_decl=dtype_to_c_struct(dev, bbox_dtype),
                particle_id_dtype=particle_id_dtype,
                morton_bin_count_dtype=morton_bin_count_dtype,
                scan_dtype=scan_dtype,
                morton_nr_dtype=self.morton_nr_dtype,
                box_id_dtype=box_id_dtype,
                dtype_to_ctype=dtype_to_ctype,
                AXIS_NAMES=AXIS_NAMES,
                box_flags_enum=box_flags_enum
                )

        preamble = PREAMBLE_TPL.render(**codegen_args)

        # {{{ scan

        scan_preamble = preamble + SCAN_PREAMBLE_TPL.render(**codegen_args)

        from pyopencl.tools import VectorArg, ScalarArg
        scan_knl_arguments = (
                [
                    # box-local morton bin counts for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(morton_bin_count_dtype, "morton_bin_counts"), # [nsrcntgts]

                    # (local) morton nrs for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(np.uint8, "morton_nrs"), # [nsrcntgts]

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
                + [VectorArg(coord_dtype, ax) for ax in axis_names]
                )

        from pyopencl.scan import GenericScanKernel
        scan_kernel = GenericScanKernel(
                self.context, scan_dtype,
                arguments=scan_knl_arguments,
                input_expr="scan_t_from_particle(%s)"
                    % ", ".join([
                        "i", "level", "srcntgt_box_ids[i]", "*nboxes",
                        "box_srcntgt_starts[srcntgt_box_ids[i]]",
                        "box_srcntgt_counts[srcntgt_box_ids[i]]",
                        "max_particles_in_box",
                        "&bbox"
                        ]
                        +["%s[user_srcntgt_ids[i]]" % ax for ax in axis_names]),
                scan_expr="scan_t_add(a, b, across_seg_boundary)",
                neutral="scan_t_neutral()",
                is_segment_start_expr="box_start_flags[i]",
                output_statement=SCAN_OUTPUT_STMT_TPL.render(**codegen_args),
                preamble=scan_preamble)

        # }}}

        # {{{ split-and-sort

        split_and_sort_kernel_source = SPLIT_AND_SORT_KERNEL_TPL.render(**codegen_args)

        from pyopencl.elementwise import ElementwiseKernel
        split_and_sort_kernel = ElementwiseKernel(
                self.context,
                scan_knl_arguments
                + [
                    VectorArg(particle_id_dtype, "new_user_srcntgt_ids"),
                    VectorArg(np.int32, "have_oversize_box"),
                    VectorArg(box_id_dtype, "new_srcntgt_box_ids"),
                    ],
                str(split_and_sort_kernel_source), name="split_and_sort",
                preamble=str(preamble))

        # }}}

        # {{{ box-info

        type_values = (
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
                type_values, var_values=codegen_args_tuples,
                more_preamble=box_flags_enum.get_c_defines(),
                declare_types=("bbox_t",))

        # }}}

        # {{{ source-and-target splitter

        # These kernels are only needed if there are separate sources and
        # targets.  But they're only built on-demand anyway, so that's not
        # really a loss.

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

        from pyopencl.elementwise import ElementwiseTemplate
        source_and_target_index_finder = ElementwiseTemplate(
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
                name="find_source_and_target_indices").build(
                            self.context,
                            type_values=(
                                ("particle_id_t", particle_id_dtype),
                                ("box_id_t", box_id_dtype),
                                ),
                            var_values=())

        # }}}

        # {{{ particle permuter

        # used if there is only one source/target array
        srcntgt_permuter = ElementwiseTemplate(
                arguments=[
                    VectorArg(particle_id_dtype, "source_ids")
                    ]
                    + [VectorArg(coord_dtype, ax) for ax in axis_names]
                    + [VectorArg(coord_dtype, "sorted_"+ax) for ax in axis_names],
                operation=r"""//CL:mako//
                    particle_id_t src_idx = source_ids[i];
                    %for ax in axis_names:
                        sorted_${ax}[i] = ${ax}[src_idx];
                    %endfor
                    """,
                name="permute_srcntgt").build(
                            self.context,
                            type_values=(
                                ("particle_id_t", particle_id_dtype),
                                ("box_id_t", box_id_dtype),
                                ),
                            var_values=(
                                ("axis_names", axis_names),
                                ))

        # }}}

        # {{{ find-prune-indices

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
                    from_box_id[i-prev_item] = i;
                    if (i+1 == N) *nboxes_post_prune = N-item;
                    """)

        # }}}

        return _KernelInfo(
                particle_id_dtype=particle_id_dtype,
                box_id_dtype=box_id_dtype,
                scan_kernel=scan_kernel,
                morton_bin_count_dtype=morton_bin_count_dtype,
                split_and_sort_kernel=split_and_sort_kernel,
                box_info_kernel=box_info_kernel,
                find_prune_indices_kernel=find_prune_indices_kernel,
                source_counter=source_counter,
                source_and_target_index_finder=source_and_target_index_finder,
                srcntgt_permuter=srcntgt_permuter,
                )

    # }}}

    def _gappy_copy_and_map(self, queue, allocator, new_size,
            src_indices, ary, map_values=None):
        """Compresses box info arrays after empty leaf pruning and, optionally,
        maps old box IDs to new box IDs (if the array being operated on contains
        box IDs).
        """

        assert len(ary) >= new_size

        result = cl.array.empty(queue, new_size, ary.dtype, allocator=allocator)

        kernel = self.get_gappy_copy_and_map_kernel(ary.dtype, src_indices.dtype,
                map_values=map_values is not None)

        args = (ary, result, src_indices)
        if map_values is not None:
            args += (map_values,)

        kernel(*args, queue=queue, range=slice(new_size))

        return result

    # {{{ run control

    def __call__(self, queue, particles, max_particles_in_box, nboxes_guess=None,
            allocator=None, debug=False, targets=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` instance
        :arg particles: an object array of (XYZ) point coordinate arrays.
        :arg targets: an object array of (XYZ) point coordinate arrays or `None`.
            If `None`, *particles* act as targets, too.
        :returns: an instance of :class:`Tree`
        """
        dimensions = len(particles)

        axis_names = AXIS_NAMES[:dimensions]

        empty = partial(cl.array.empty, queue, allocator=allocator)
        zeros = partial(cl.array.zeros, queue, allocator=allocator)

        # {{{ get kernel info

        from pytools import single_valued
        coord_dtype = single_valued(coord.dtype for coord in particles)
        particle_id_dtype = np.uint32
        box_id_dtype = np.int32
        knl_info = self.get_kernel_info(dimensions, coord_dtype, particle_id_dtype, box_id_dtype)

        # }}}

        # {{{ combine sources and targets into one array, if necessary

        if targets is None:
            # Targets weren't specified. Sources are also targets. Let's
            # call them "srcntgts".

            srcntgts = particles
            nsrcntgts = single_valued(len(coord) for coord in srcntgts)
        else:
            # Here, we mash sources and targets into one array to give us one
            # big array of "srcntgts". In this case, a "srcntgt" is either a
            # source or a target, but not really both, as above. How will we be
            # able to tell which it was? Easy: We'll compare its 'user' id with
            # nsources. If its >=, it's a target, and a source otherwise.

            target_coord_dtype = single_valued(tgt_i.dtype for tgt_i in targets)

            if target_coord_dtype != coord_dtype:
                raise TypeError("sources and targets must have same coordinate "
                        "dtype")

            nsources = single_valued(len(coord) for coord in particles)
            ntargets = single_valued(len(coord) for coord in targets)
            nsrcntgts = nsources + ntargets

            def combine_coord_arrays(ary1, ary2):
                result = empty(nsrcntgts, coord_dtype)
                cl.enqueue_copy(queue, result.data, ary1.data)
                cl.enqueue_copy(queue, result.data, ary2.data,
                        dest_offset=ary1.nbytes)
                return result

            from pytools.obj_array import make_obj_array
            srcntgts = make_obj_array([
                combine_coord_arrays(src_i, tgt_i)
                for src_i, tgt_i in zip(particles, targets)
                ])

        del particles

        user_srcntgt_ids = cl.array.arange(queue, nsrcntgts, dtype=particle_id_dtype,
                allocator=allocator)

        # }}}

        # {{{ find and process bounding box

        bbox = self.get_bbox_finder()(srcntgts).get()

        root_extent = max(
                bbox["max_"+ax] - bbox["min_"+ax]
                for ax in axis_names) * (1+1e-4)

        # make bbox square and slightly larger at the top, to ensure scaled
        # coordinates are alwyas < 1
        bbox_min = np.empty(dimensions, coord_dtype)
        for i, ax in enumerate(axis_names):
            bbox_min[i] = bbox["min_"+ax]
            bbox["max_"+ax] = bbox["min_"+ax] + root_extent
        bbox_max = bbox_min + root_extent

        # }}}

        # {{{ allocate data

        morton_bin_counts = empty(nsrcntgts, dtype=knl_info.morton_bin_count_dtype)
        morton_nrs = empty(nsrcntgts, dtype=np.uint8)
        box_start_flags = zeros(nsrcntgts, dtype=np.int8)
        srcntgt_box_ids = zeros(nsrcntgts, dtype=box_id_dtype)
        split_box_ids = zeros(nsrcntgts, dtype=box_id_dtype)

        from pytools import div_ceil
        nboxes_guess = div_ceil(nsrcntgts, max_particles_in_box) * 2**dimensions

        box_morton_bin_counts = empty(nboxes_guess,
                dtype=knl_info.morton_bin_count_dtype)
        box_srcntgt_starts = zeros(nboxes_guess, dtype=particle_id_dtype)
        box_parent_ids = zeros(nboxes_guess, dtype=box_id_dtype)
        box_morton_nrs = zeros(nboxes_guess, dtype=self.morton_nr_dtype)
        box_srcntgt_counts = zeros(nboxes_guess, dtype=particle_id_dtype)

        # Initalize box 0 to contain all particles
        cl.enqueue_copy(queue, box_srcntgt_counts.data,
                box_srcntgt_counts.dtype.type(nsrcntgts))

        nboxes_dev = empty((), dtype=box_id_dtype)
        nboxes_dev.fill(1)

        # set parent of root box to itself
        cl.enqueue_copy(queue, box_parent_ids.data, box_parent_ids.dtype.type(0))

        # }}}

        # {{{ level loop

        from pytools.obj_array import make_obj_array
        have_oversize_box = zeros((), np.int32)

        # Level 0 starts at 0 and always contains box 0 and nothing else.
        # Level 1 therefore starts at 1.
        level_starts = [0, 1]

        from time import time
        start_time = time()
        level = 0
        while True:
            args = ((morton_bin_counts, morton_nrs,
                    box_start_flags, srcntgt_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    box_srcntgt_starts, box_srcntgt_counts,
                    box_parent_ids, box_morton_nrs,
                    nboxes_dev,
                    level, max_particles_in_box, bbox,
                    user_srcntgt_ids)
                    + tuple(srcntgts))

            # writes: nboxes_dev, box_morton_bin_counts, split_box_ids
            knl_info.scan_kernel(*args)

            nboxes_new = nboxes_dev.get()

            if nboxes_new > nboxes_guess:
                while nboxes_guess < nboxes_new:
                    nboxes_guess *= 2

                my_realloc= partial(_realloc_array, new_shape=nboxes_guess,
                        zero_fill=False)
                my_realloc_zeros = partial(_realloc_array, new_shape=nboxes_guess,
                        zero_fill=True)

                box_morton_bin_counts = my_realloc(box_morton_bin_counts)
                box_srcntgt_starts = my_realloc_zeros(box_srcntgt_starts)
                box_parent_ids = my_realloc_zeros(box_parent_ids)
                box_morton_nrs = my_realloc_zeros(box_morton_nrs)
                box_srcntgt_counts = my_realloc_zeros(box_srcntgt_counts)

                del my_realloc
                del my_realloc_zeros

                # resta
                nboxes_dev.fill(level_starts[-1])

                # retry
                if debug:
                    print "nboxes_guess exceeded: enlarged allocations, restarting level"

                continue

            level_starts.append(int(nboxes_new))
            del nboxes_new

            if debug:
                print "LEVEL %d -> %d boxes" % (len(level_starts)-2, level_starts[-1])

            new_user_srcntgt_ids = cl.array.empty_like(user_srcntgt_ids)
            new_srcntgt_box_ids = cl.array.empty_like(srcntgt_box_ids)
            split_and_sort_args = (
                    args
                    + (new_user_srcntgt_ids,
                        have_oversize_box, new_srcntgt_box_ids))
            knl_info.split_and_sort_kernel(*split_and_sort_args)

            user_srcntgt_ids = new_user_srcntgt_ids
            del new_user_srcntgt_ids
            srcntgt_box_ids = new_srcntgt_box_ids
            del new_srcntgt_box_ids

            if not int(have_oversize_box.get()):
                break

            level += 1

            have_oversize_box.fill(0)

        end_time = time()
        if debug:
            elapsed = end_time-start_time
            npasses = level+1
            print "elapsed time: %g s (%g s/particle/pass)" % (
                    elapsed, elapsed/(npasses*nsrcntgts))

        nboxes = int(nboxes_dev.get())

        # }}}

        # {{{ prune empty leaf boxes

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

        prune_empty = partial(self._gappy_copy_and_map,
                queue, allocator, nboxes_post_prune, from_box_id)

        box_srcntgt_starts = prune_empty(box_srcntgt_starts)
        box_srcntgt_counts = prune_empty(box_srcntgt_counts)

        srcntgt_box_ids = cl.array.take(to_box_id, srcntgt_box_ids)

        box_parent_ids = prune_empty(box_parent_ids, map_values=to_box_id)
        box_morton_nrs = prune_empty(box_morton_nrs)

        # Remap level_starts to new box IDs.
        # FIXME: It would be better to do this on the device.
        level_starts = list(to_box_id.get()[np.array(level_starts[:-1], box_id_dtype)])
        level_starts = np.array(level_starts + [nboxes_post_prune], box_id_dtype)

        # }}}

        del nboxes

        # {{{ update particle indices and box info for source/target split

        # {{{ turn a "to" index list into a "from" index list

        # (= 'transpose/invert a permutation)

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

        # {{{ permute and s/t-split (if necessary) particle array

        if targets is None:
            sources = targets = make_obj_array([
                cl.array.empty_like(pt) for pt in srcntgts])
            knl_info.srcntgt_permuter(
                    user_srcntgt_ids,
                    *(tuple(srcntgts) + tuple(sources)))
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
        box_levels = empty(nboxes_post_prune, np.uint8)
        box_flags = empty(nboxes_post_prune, box_flags_enum.dtype)

        knl_info.box_info_kernel(
                # input:
                box_parent_ids, box_morton_nrs, bbox, aligned_nboxes,
                box_srcntgt_counts, max_particles_in_box,

                # output:
                box_child_ids, box_centers, box_levels, box_flags,

                range=slice(nboxes_post_prune))

        # }}}

        nlevels = len(level_starts) - 1
        assert level + 2 == nlevels
        if debug:
            assert np.max(box_levels.get()) + 1 == nlevels

        del nlevels

        return Tree(
                # If you change this, also change the documentation
                # of what's in the tree, above.

                particle_id_dtype=knl_info.particle_id_dtype,
                box_id_dtype=knl_info.box_id_dtype,
                coord_dtype=coord_dtype,

                root_extent=root_extent,

                # +2 because we stop one level before the end and we
                # did not count level 0.
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
                )

    # }}}

# }}}




# vim: filetype=pyopencl:fdm=marker
