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

__doc__ = """
Terminology: Sources/targets vs particles
-----------------------------------------

If 'targets' is not specified, 'particles' are both sources and
targets.

Orderings
---------

If there are only particles, then there are two particle orderings:
'user order', and 'tree order'. :attr:`Tree.original_particle_ids`
and :attr:`Tree.reordered_particle_ids` allow conversion between
the two.
"""


# TODO:
# - Distinguish sources and targets
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
# - the current box number. At the start of the scan, this is correct only
#   for the first particle in each box. As the scan proceeds, this gets
#   propagated throughout the rest of the box. (by "max" as the associative,
#   commutative operator)
#
# - the "subdivided box number". The very first entry here gets intialized to
#   the number of boxes present at the previous level. If a box knows it needs to
#   be subdivided, its first particle asks for 2**d new boxes. This gets scanned
#   over by summing globally (unsegmented-ly). The splits are then realized in
#   the post-processing step.
#
# -----------------------------------------------------------------------------
#




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

    name = "boxtree_morton_bin_count_%dd_t" % dimensions
    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    # FIXME: build id_type into name
    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

@memoize
def make_scan_type(device, dimensions, particle_id_dtype, box_id_dtype):
    morton_dtype, _ = make_morton_bin_count_type(device, dimensions, particle_id_dtype)
    dtype = np.dtype([
            ('counts', morton_dtype),
            ('current_box_id', box_id_dtype), # max-scanned
            ('subdivided_box_id', box_id_dtype), # sum-scanned
            ('morton_nr', np.uint8),
            ])

    name = "boxtree_tree_scan_%dd_t" % dimensions
    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    # FIXME: build id_types into name
    dtype = get_or_register_dtype(name, dtype)
    return dtype, c_decl

# }}}

class box_type_enum:
    """Constants for box types."""

    BRANCH = 0

    # these only occur if particles have refinement restrictions
    NONEMPTY_BRANCH = 1

    LEAF = 2

    # these are pruned and will not occur in output
    EMPTY_LEAF = 3

    @classmethod
    def get_c_defines(cls):
        """Return a string with C defines corresponding to these constants."""

        return "\n".join(
                "#define BOX_%s %d"
                % (name, getattr(cls, name))
                for name in sorted(dir(box_type_enum))
                if name[0].isupper()) + "\n\n"


# {{{ preamble

PREAMBLE_TPL = Template(r"""//CL//
    ${bbox_type_decl}
    ${morton_bin_count_type_decl}
    ${tree_scan_type_decl}

    typedef boxtree_morton_bin_count_${dimensions}d_t morton_t;
    typedef boxtree_tree_scan_${dimensions}d_t scan_t;
    typedef boxtree_bbox_${dimensions}d_t bbox_t;
    typedef ${coord_ctype} coord_t;
    typedef ${coord_ctype}${dimensions} coord_vec_t;
    typedef ${box_id_ctype} box_id_t;
    typedef ${particle_id_ctype} particle_id_t;
    typedef ${morton_nr_ctype} morton_nr_t;

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
        result.current_box_id = 0;
        result.subdivided_box_id = 0;
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
            b.current_box_id = max(a.current_box_id, b.current_box_id);
        }

        // subdivided_box_id must use a non-segmented scan to globally
        // assign box numbers.
        b.subdivided_box_id = a.subdivided_box_id + b.subdivided_box_id;

        // b.morton_nr gets propagated
        return b;
    }

    scan_t scan_t_from_particle(
        int i,
        int level,
        box_id_t box_id,
        box_id_t box_count,
        particle_id_t box_start,
        particle_id_t box_particle_count,
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

        // current_box_id is only valid if the current box starts at this
        // particle, but that's ok.  We'll max-scan it so that by output time
        // every particle knows its (by then possibly former) box id.

        result.current_box_id = box_id;

        // subdivided_box_id is not very meaningful now, but when scanned over
        // by addition, will yield new, unused ids for boxes that are created by
        // subdividing the current box (if it is over-full).

        result.subdivided_box_id = 0;
        if (i == 0)
        {
            // Particle number zero brings in the box count from the
            // previous level.

            result.subdivided_box_id = box_count;
        }
        if (i == box_start
            && box_particle_count > max_particles_in_box)
        {
            // If this box is overfull, put in a 'request' for 2**d sub-box
            // IDs. Sub-boxes will have to subtract from the total to find
            // their id. These requested box IDs are then scanned over by
            // a global sum.

            result.subdivided_box_id += ${2**dimensions};
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

        particle_id_t box_particle_count = box_particle_counts[item.current_box_id];

        unsplit_box_ids[i] = item.current_box_id;
        split_box_ids[i] = item.subdivided_box_id;

        // Am I the last particle in my current box?
        // If so, populate particle count.

        if (my_id_in_my_box+1 == box_particle_count)
        {
            dbg_printf(("store box %d cbi:%d\n", i, item.current_box_id));
            dbg_printf(("   store_sums: %d %d %d %d\n", item.counts.c00, item.counts.c01, item.counts.c10, item.counts.c11));
            box_morton_bin_counts[item.current_box_id] = item.counts;
        }

        // Am I the last particle overall? If so, write box count
        if (i+1 == N)
            *box_count = item.subdivided_box_id;
    }
""", strict_undefined=True)

# }}}

# {{{ split-and-sort kernel

SPLIT_AND_SORT_KERNEL_TPL =  Template(r"""//CL//
    morton_t my_morton_bin_counts = morton_bin_counts[i];
    box_id_t my_box_id = unsplit_box_ids[i];

    dbg_printf(("postproc %d:\n", i));
    dbg_printf(("   my_sums: %d %d %d %d\n",
        my_morton_bin_counts.c00, my_morton_bin_counts.c01,
        my_morton_bin_counts.c10, my_morton_bin_counts.c11));
    dbg_printf(("   my box id: %d\n", my_box_id));

    particle_id_t box_particle_count = box_particle_counts[my_box_id];

    /* Is this box being split? */
    if (box_particle_count > max_particles_in_box)
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

        particle_id_t my_box_start = box_particle_starts[my_box_id];
        particle_id_t tgt_particle_idx = my_box_start + my_count-1;
        %for mnr in range(2**dimensions):
            <% bin_nmr = padded_bin(mnr, dimensions) %>
            tgt_particle_idx +=
                (my_morton_nr > ${mnr})
                    ? my_box_morton_bin_counts.c${bin_nmr}
                    : 0;
        %endfor

        dbg_printf(("   moving %d -> %d\n", i, tgt_particle_idx));
        %for ax in axis_names:
            sorted_${ax}[tgt_particle_idx] = ${ax}[i];
        %endfor
        new_original_particle_ids[tgt_particle_idx] = original_particle_ids[i];

        box_ids[tgt_particle_idx] = new_box_id;

        %for mnr in range(2**dimensions):
          /* Am I the last particle in my Morton bin? */
            %if mnr:
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
                box_particle_starts[new_box_id] = new_box_start;
                box_parent_ids[new_box_id] = my_box_id;
                box_morton_nrs[new_box_id] = my_morton_nr;

                particle_id_t new_count =
                    my_box_morton_bin_counts.c${padded_bin(mnr, dimensions)};
                box_particle_counts[new_box_id] = new_count;
                if (new_count > max_particles_in_box)
                    *have_oversize_box = 1;

                dbg_printf(("   box pcount: %d\n", box_particle_counts[new_box_id]));
            }
        %endfor
    }
    else
    {
        // Not splitting? Copy over existing particle info.
        %for ax in axis_names:
            sorted_${ax}[i] = ${ax}[i];
        %endfor
        new_original_particle_ids[i] = original_particle_ids[i];
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
        particle_id_t *box_particle_counts,
        particle_id_t max_particles_in_box,
        /* output */
        box_id_t *box_child_ids, /* [2**dimensions, aligned_nboxes] */
        coord_t *box_centers, /* [dimensions, aligned_nboxes] */
        unsigned char *box_levels, /* [nboxes] */
        unsigned char *box_types, /* [nboxes] */
        """,
    operation=r"""//CL:mako//
        box_id_t box_id = i;

        particle_id_t p_count = box_particle_counts[box_id];
        if (p_count == 0)
        {
            // Lots of stuff uninitialized for these guys, prevent
            // damage by quitting now.

            // Also, those should have gotten pruned by this point.

            box_types[box_id] = BOX_EMPTY_LEAF;
            return;
        }
        else if (p_count > max_particles_in_box)
        {
            box_types[box_id] = BOX_BRANCH;
            box_particle_counts[box_id] = 0;
        }
        else
            box_types[box_id] = BOX_LEAF;

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

        /* box_particle_counts is zero for empty leaves because it gets initialized
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

class Tree(Record):
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

    .. attribute:: particles

        `coord_t [dimensions][nparticles]`
        (an object array of coordinate arrays)
        Stored in tree order.

    .. attribute:: original_particle_ids

        `particle_id_t [nparticles]`
        Fetching *from* these indices will reorder the particles
        from user order into tree order.

    .. attribute:: reordered_particle_ids

        `particle_id_t [nparticles]`
        Fetching *from* these indices will reorder the particles
        from tree order into user order.

    **Per-box arrays**

    .. attribute:: box_particle_starts

        `particle_id_t [nboxes]`

    .. attribute:: box_particle_counts

        `particle_id_t [nboxes]`

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
    .. attribute:: box_types

        `uint 8[nboxes]`
        One of the :class:`box_type_enum` constants.
    """

    @property
    def dimensions(self):
        return len(self.particles)

    @property
    def nboxes(self):
        return self.box_levels.shape[0]

    @property
    def nparticles(self):
        return len(self.original_particle_ids)

    @property
    def nlevels(self):
        return len(self.level_starts) - 1

    @property
    def aligned_nboxes(self):
        return self.box_child_ids.shape[-1]

    def plot(self, **kwargs):
        plotter = TreePlotter(self)
        plotter.draw_tree(fill=False, edgecolor="black", **kwargs)

    def get(self):
        """Return a copy of `self` in which all data lives on the host."""

        result = {}
        for field_name in self.__class__.fields:
            try:
                attr = getattr(self, field_name)
            except AttributeError:
                pass
            else:
                if isinstance(attr, cl.array.Array):
                    result[field_name] = attr.get()
                elif isinstance(attr, np.ndarray) and attr.dtype == object:
                    from pytools.obj_array import with_object_array_or_scalar
                    result[field_name] = with_object_array_or_scalar(
                            lambda x: x.get(), attr)

        return self.copy(**result)


# }}}

# {{{ visualization

class TreePlotter:
    def __init__(self, tree):
        self.tree = tree
        self.centers = tree.box_centers.get()
        self.levels = tree.box_levels.get()
        self.root_extent = tree.root_extent

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

        lev = int(self.levels[ibox])
        box_size = self.root_extent / (1 << lev)
        el = self.centers[:, ibox] - 0.5*box_size
        eh = el + box_size

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

class TreeBuilder(object):
    def __init__(self, context):
        self.context = context

    # {{{ kernel creation

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
        coord_ctype = dtype_to_ctype(coord_dtype)
        coord_vec_dtype = cl.array.vec.types[coord_dtype, dimensions]

        particle_id_dtype = np.dtype(particle_id_dtype)
        particle_id_ctype = dtype_to_ctype(particle_id_dtype)

        box_id_dtype = np.dtype(box_id_dtype)
        box_id_ctype = dtype_to_ctype(box_id_dtype)

        morton_nr_ctype = dtype_to_ctype(self.morton_nr_dtype)

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
                coord_ctype=coord_ctype,
                morton_bin_count_type_decl=dtype_to_c_struct(
                    dev, morton_bin_count_dtype),
                tree_scan_type_decl=scan_type_decl,
                bbox_type_decl=dtype_to_c_struct(dev, bbox_dtype),
                particle_id_ctype=particle_id_ctype,
                morton_nr_ctype=morton_nr_ctype,
                box_id_ctype=box_id_ctype,
                AXIS_NAMES=AXIS_NAMES,
                box_type_enum=box_type_enum
                )

        preamble = PREAMBLE_TPL.render(**codegen_args)

        # {{{ scan

        scan_preamble = preamble + SCAN_PREAMBLE_TPL.render(**codegen_args)

        from pyopencl.tools import VectorArg, ScalarArg
        scan_knl_arguments = (
                [
                    # box-local morton bin counts for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(morton_bin_count_dtype, "morton_bin_counts"), # [nparticles]

                    # (local) morton nrs for each particle at the current level
                    # only valid from scan -> split'n'sort
                    VectorArg(np.uint8, "morton_nrs"), # [nparticles]

                    # segment flags
                    # invariant to sorting once set
                    # (particles are only reordered within a box)
                    VectorArg(np.uint8, "box_start_flags"), # [nparticles]

                    VectorArg(box_id_dtype, "box_ids"), # [nparticles]
                    VectorArg(box_id_dtype, "unsplit_box_ids"), # [nparticles]
                    VectorArg(box_id_dtype, "split_box_ids"), # [nparticles]

                    # per-box morton bin counts
                    VectorArg(morton_bin_count_dtype, "box_morton_bin_counts"), # [nparticles]

                    # particle# at which each box starts
                    VectorArg(particle_id_dtype, "box_particle_starts"), # [nboxes]

                    # number of particles in each box
                    VectorArg(particle_id_dtype,"box_particle_counts"), # [nboxes]

                    # pointer to parent box
                    VectorArg(box_id_dtype, "box_parent_ids"), # [nboxes]

                    # morton nr identifier {quadr,oct}ant of parent in which this box was created
                    VectorArg(self.morton_nr_dtype, "box_morton_nrs"), # [nboxes]

                    # number of boxes total
                    VectorArg(box_id_dtype, "box_count"), # [1]

                    ScalarArg(np.int32, "level"),
                    ScalarArg(particle_id_dtype, "max_particles_in_box"),
                    ScalarArg(bbox_dtype, "bbox"),
                    ]
                + [VectorArg(coord_dtype, ax) for ax in axis_names]
                )

        from pyopencl.scan import GenericScanKernel
        scan_kernel = GenericScanKernel(
                self.context, scan_dtype,
                arguments=scan_knl_arguments,
                input_expr="scan_t_from_particle(%s)"
                    % ", ".join([
                        "i", "level", "box_ids[i]", "*box_count",
                        "box_particle_starts[box_ids[i]]",
                        "box_particle_counts[box_ids[i]]",
                        "max_particles_in_box",
                        "&bbox"
                        ]
                        +["%s[i]" % ax for ax in axis_names]),
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
                + [VectorArg(coord_dtype, "sorted_"+ax) for ax in axis_names]
                + [
                    VectorArg(particle_id_dtype, "original_particle_ids"),
                    VectorArg(particle_id_dtype, "new_original_particle_ids"),
                    VectorArg(np.int32, "have_oversize_box"),
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
                )
        codegen_args_tuples = tuple(codegen_args.iteritems())
        box_info_kernel = BOX_INFO_KERNEL_TPL.build(
                self.context,
                type_values, var_values=codegen_args_tuples,
                more_preamble=box_type_enum.get_c_defines(),
                declare_types=("bbox_t",))

        # }}}

        # {{{ find-prune-indices

        from pyopencl.tools import VectorArg
        find_prune_indices_kernel = GenericScanKernel(
                self.context, box_id_dtype,
                arguments=[
                    # input
                    VectorArg(particle_id_dtype, "box_particle_counts"),
                    # output
                    VectorArg(box_id_dtype, "to_box_id"),
                    VectorArg(box_id_dtype, "from_box_id"),
                    VectorArg(box_id_dtype, "nboxes_post_prune"),
                    ],
                input_expr="box_particle_counts[i] == 0 ? 1 : 0",
                preamble=box_type_enum.get_c_defines(),
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
                )

    # }}}

    def gappy_copy_and_map(self, queue, allocator, new_size,
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
        :returns: an instance of :class:`Tree`
        """
        dimensions = len(particles)

        bbox = self.get_bbox_finder()(particles).get()

        axis_names = AXIS_NAMES[:dimensions]

        # {{{ get kernel info

        from pytools import single_valued
        coord_dtype = single_valued(coord.dtype for coord in particles)
        particle_id_dtype = np.uint32
        box_id_dtype = np.int32
        knl_info = self.get_kernel_info(dimensions, coord_dtype, particle_id_dtype, box_id_dtype)

        # }}}

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

        nparticles = single_valued(len(coord) for coord in particles)

        from functools import partial
        empty = partial(cl.array.empty, queue, allocator=allocator)
        zeros = partial(cl.array.zeros, queue, allocator=allocator)

        morton_bin_counts = empty(nparticles, dtype=knl_info.morton_bin_count_dtype)
        morton_nrs = empty(nparticles, dtype=np.uint8)
        box_start_flags = zeros(nparticles, dtype=np.int8)
        box_ids = zeros(nparticles, dtype=box_id_dtype)
        unsplit_box_ids = zeros(nparticles, dtype=box_id_dtype)
        split_box_ids = zeros(nparticles, dtype=box_id_dtype)

        from pytools import div_ceil
        nboxes_guess = div_ceil(nparticles, max_particles_in_box) * 2**dimensions

        box_morton_bin_counts = empty(nboxes_guess,
                dtype=knl_info.morton_bin_count_dtype)
        box_particle_starts = zeros(nboxes_guess, dtype=particle_id_dtype)
        box_parent_ids = zeros(nboxes_guess, dtype=box_id_dtype)
        box_morton_nrs = zeros(nboxes_guess, dtype=self.morton_nr_dtype)
        box_particle_counts = zeros(nboxes_guess, dtype=particle_id_dtype)

        original_particle_ids = cl.array.arange(queue, nparticles, dtype=particle_id_dtype,
                allocator=allocator)

        # Initalize box 0 to contain all particles
        cl.enqueue_copy(queue, box_particle_counts.data,
                box_particle_counts.dtype.type(nparticles))

        nboxes_dev = empty((), dtype=box_id_dtype)
        nboxes_dev.fill(1)

        # set parent of root box to itself
        cl.enqueue_copy(queue, box_parent_ids.data, box_parent_ids.dtype.type(0))

        from pytools.obj_array import make_obj_array

        have_oversize_box = zeros((), np.int32)

        # Level 0 starts at 0 and always contains box 0 and nothing else.
        # Level 1 therefore starts at 1.
        level_starts = [0, 1]

        from time import time
        start_time = time()
        level = 0
        while True:
            if debug:
                print "LEV"
            args = ((morton_bin_counts, morton_nrs,
                    box_start_flags, box_ids, unsplit_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    box_particle_starts, box_particle_counts,
                    box_parent_ids, box_morton_nrs,
                    nboxes_dev,
                    level, max_particles_in_box, bbox)
                    + tuple(particles))
            knl_info.scan_kernel(*args)

            nboxes = nboxes_dev.get()
            level_starts.append(int(nboxes))

            if nboxes > nboxes_guess:
                # FIXME
                raise NotImplementedError("Initial guess for box count was "
                        "too low. Should resize temp arrays.")

            #print "split_box_ids", split_box_ids.get()[:nparticles]

            sorted_particles = make_obj_array([
                cl.array.empty_like(pt) for pt in particles])

            new_original_particle_ids = cl.array.empty_like(original_particle_ids)
            split_and_sort_args = (
                    args
                    + tuple(sorted_particles)
                    + (original_particle_ids, new_original_particle_ids,
                        have_oversize_box))
            knl_info.split_and_sort_kernel(*split_and_sort_args)

            if 0:
                print "--------------LEVL"
                print "nboxes_dev", nboxes_dev.get()
                print "box_ids", box_ids.get()[:nparticles]
                print "starts", box_particle_starts.get()[:nboxes]
                print "counts", box_particle_counts.get()[:nboxes]

            original_particle_ids = new_original_particle_ids
            particles = sorted_particles

            if not int(have_oversize_box.get()):
                break

            level += 1

            have_oversize_box.fill(0)

        end_time = time()
        if debug:
            elapsed = end_time-start_time
            npasses = level+1
            print "elapsed time: %g s (%g s/particle/pass)" % (
                    elapsed, elapsed/(npasses*nparticles))

        nboxes = int(nboxes)

        # {{{ prune empty leaf boxes

        # What is the original index of this box?
        from_box_id = empty(nboxes, box_id_dtype)

        # Where should I put this box?
        to_box_id = empty(nboxes, box_id_dtype)

        nboxes_post_prune_dev = empty((), dtype=box_id_dtype)
        knl_info.find_prune_indices_kernel(
                box_particle_counts, to_box_id, from_box_id, nboxes_post_prune_dev,
                size=nboxes)

        nboxes_post_prune = int(nboxes_post_prune_dev.get())

        print "%d empty leaves" % (nboxes-nboxes_post_prune)

        from functools import partial
        prune_empty = partial(self.gappy_copy_and_map,
                queue, allocator, nboxes_post_prune, from_box_id)

        box_particle_starts = prune_empty(box_particle_starts)
        box_particle_counts = prune_empty(box_particle_counts)
        box_parent_ids = prune_empty(box_parent_ids, map_values=to_box_id)
        box_morton_nrs = prune_empty(box_morton_nrs)

        # FIXME: It would be better to do this on the device.
        level_starts = list(to_box_id.get()[np.array(level_starts[:-1], box_id_dtype)])
        level_starts = np.array(level_starts + [nboxes_post_prune], box_id_dtype)

        # }}}

        del nboxes

        # {{{ compute box info

        # A number of arrays below are nominally 2-dimensional and stored with
        # the box index as the fastest-moving index. To make sure that accesses
        # remain aligned, we round up the number of boxes used for indexing.
        aligned_nboxes = div_ceil(nboxes_post_prune, 32)*32

        box_child_ids = zeros((2**dimensions, aligned_nboxes), box_id_dtype)
        box_centers = empty((dimensions, aligned_nboxes), coord_dtype)
        box_levels = empty(nboxes_post_prune, np.uint8)
        box_types = empty(nboxes_post_prune, np.uint8)

        knl_info.box_info_kernel(
                # input:
                box_parent_ids, box_morton_nrs, bbox, aligned_nboxes,
                box_particle_counts, max_particles_in_box,

                # output:
                box_child_ids, box_centers, box_levels, box_types,

                range=slice(nboxes_post_prune))

        # }}}

        assert level + 2 == len(level_starts) - 1 # == number of levels

        # {{{ compute reordered particle ids

        reordered_particle_ids = empty(nparticles, particle_id_dtype)
        cl.array.multi_put(
                [cl.array.arange(queue, nparticles, dtype=particle_id_dtype,
                    allocator=allocator)],
                original_particle_ids,
                out=[reordered_particle_ids],
                queue=queue)

        # }}}

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

                particles=particles,
                box_particle_starts=box_particle_starts,
                box_particle_counts=box_particle_counts,
                box_parent_ids=box_parent_ids,
                box_child_ids=box_child_ids,
                box_centers=box_centers,
                box_levels=box_levels,
                box_types=box_types,

                original_particle_ids=original_particle_ids,
                reordered_particle_ids=reordered_particle_ids,
                )

    # }}}

# }}}




# vim: filetype=pyopencl:fdm=marker
