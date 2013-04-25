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




import pyopencl as cl
import numpy as np
from boxtree.tools import FromDeviceGettableRecord

import logging
logger = logging.getLogger(__name__)



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
    def get_flag_names_and_values(cls):
        return [(name, getattr(cls, name))
                for name in sorted(dir(cls))
                if name[0].isupper()]

    @classmethod
    def get_c_defines(cls):
        """Return a string with C defines corresponding to these constants.
        """

        return "\n".join(
                "#define BOX_%s %d" % (flag_name, value)
                for flag_name, value in cls.get_flag_names_and_values())

    @classmethod
    def stringify_value(cls, val):
        return " ".join(
                flag_name for flag_name, flag_value in cls.get_flag_names_and_values()
                if val & flag_value)

    @classmethod
    def get_c_typedef(cls):
        """Returns a typedef to define box_flags_t."""

        from pyopencl.tools import dtype_to_ctype
        return "\n\ntypedef %s box_flags_t;\n\n" % dtype_to_ctype(cls.dtype)

# }}}

# {{{ tree data structure

class Tree(FromDeviceGettableRecord):
    """A quad/octree consisting of particles sorted into a hierarchy of boxes.
    Optionally, particles may be designated 'sources' and 'targets'. They
    may also be assigned radii which restrict the minimum size of the box
    into which they may be sorted.

    Instances of this class are not constructed directly. They are returned
    by :meth:`TreeBuilder.__call__`.

    **Flags**

    .. attribute:: sources_are_targets

        ``bool`` whether sources and targets are the same

    .. attribute:: sources_have_extent

        ``bool`` whether this tree has sources in non-leaf boxes

    .. attribute:: target_have_extent

        ``bool`` whether this tree has targets in non-leaf boxes

    **Data types**

    .. attribute:: particle_id_dtype
    .. attribute:: box_id_dtype
    .. attribute:: coord_dtype
    .. attribute:: box_level_dtype

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

    .. attribute:: level_start_box_nrs

        ``box_id_t [nlevels+1]``
        A :class:`numpy.ndarray` of box ids
        indicating the ID at which each level starts. Levels
        are contiguous in box ID space. To determine
        how many boxes there are in each level, check
        access the start of the next level. This array is
        built so that this works even for the last level.

    .. attribute:: level_start_box_nrs_dev

        ``particle_id_t [nlevels+1``
        The same array as :attr:`level_start_box_nrs`
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

        ``box_level_t [nboxes]``

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
        return len(self.level_start_box_nrs) - 1

    @property
    def aligned_nboxes(self):
        return self.box_child_ids.shape[-1]

    def plot(self, **kwargs):
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(self)
        plotter.draw_tree(fill=False, edgecolor="black", **kwargs)
        plotter.set_bounding_box()

    def get_box_extent(self, ibox):
        lev = int(self.box_levels[ibox])
        box_size = self.root_extent / (1 << lev)
        extent_low = self.box_centers[:, ibox] - 0.5*box_size
        extent_high = extent_low + box_size
        return extent_low, extent_high

    # {{{ point source linking

    def link_point_sources(self, queue, point_source_starts, point_sources,
            debug=False):
        """Build a :class:`TreeWithLinkedPointSources`.

        Requires that :attr:`sources_have_extent` is *True*.

        :arg queue: a :class:`pyopencl.CommandQueue` instance
        :arg point_source_starts: `point_source_starts[isrc]` and
            `point_source_starts[isrc+1]` together indicate a ranges of point
            particle indices in *point_sources* which will be linked to the
            original (extent-having) source number *isrc*. *isrc* is in :ref:`user
            source order <particle-orderings>`.

            All the particles linked to *isrc* shoud fall within the :math:`l^\infty`
            'circle' around particle number *isrc* with the radius drawn from
            :attr:`source_radii`.

        :arg point_sources: an object array of (XYZ) point coordinate arrays.
        """

        # The whole point of this routine is that all point sources within
        # a box are reordered to be contiguous.

        logger.info("point source linking: start")

        if not self.sources_have_extent:
            raise ValueError("only allowed on trees whose sources have extent")

        npoint_sources_dev = cl.array.empty(queue, (), self.particle_id_dtype)

        # {{{ compute tree_order_point_source_{starts, counts}

        # Scan over lengths of point source lists in tree order to determine
        # indices of point source starts for each source.

        tree_order_point_source_starts = cl.array.empty(
                queue, self.nsources, self.particle_id_dtype)
        tree_order_point_source_counts = cl.array.empty(
                queue, self.nsources, self.particle_id_dtype)

        from boxtree.tree_build_kernels import POINT_SOURCE_LINKING_SOURCE_SCAN_TPL
        knl = POINT_SOURCE_LINKING_SOURCE_SCAN_TPL.build(
            queue.context,
            type_aliases=(
                ("scan_t", self.particle_id_dtype),
                ("index_t", self.particle_id_dtype),
                ("particle_id_t", self.particle_id_dtype),
                ),
            )

        logger.debug("point source linking: tree order source scan")

        knl(point_source_starts, self.user_source_ids,
                tree_order_point_source_starts, tree_order_point_source_counts,
                npoint_sources_dev, size=self.nsources, queue=queue)

        # }}}

        npoint_sources = int(npoint_sources_dev.get())

        # {{{ compute user_point_source_ids

        # A list of point source starts, indexed in tree order,
        # but giving point source indices in user order.
        tree_order_index_user_point_source_starts = cl.array.take(
                point_source_starts, self.user_source_ids)

        user_point_source_ids = cl.array.empty(
                queue, npoint_sources, self.particle_id_dtype)
        user_point_source_ids.fill(1)
        cl.array.multi_put([tree_order_index_user_point_source_starts ],
                dest_indices=tree_order_point_source_starts,
                out=[user_point_source_ids])

        if debug:
            ups_host = user_point_source_ids.get()
            assert (ups_host >= 0).all()
            assert (ups_host < npoint_sources).all()

        source_boundaries = cl.array.zeros(queue, npoint_sources, np.int8)

        # FIXME: Should be a scalar, in principle.
        ones = cl.array.empty(queue, self.nsources, np.int8)
        ones.fill(1)

        cl.array.multi_put(
                [ones],
                dest_indices=tree_order_point_source_starts,
                out=[source_boundaries])

        from boxtree.tree_build_kernels import POINT_SOURCE_LINKING_USER_POINT_SOURCE_ID_SCAN_TPL

        logger.debug("point source linking: point source id scan")

        knl = POINT_SOURCE_LINKING_USER_POINT_SOURCE_ID_SCAN_TPL.build(
            queue.context,
            type_aliases=(
                ("scan_t", self.particle_id_dtype),
                ("index_t", self.particle_id_dtype),
                ("particle_id_t", self.particle_id_dtype),
                ),
            )
        knl(source_boundaries, user_point_source_ids,
                size=npoint_sources, queue=queue)

        if debug:
            ups_host = user_point_source_ids.get()
            assert (ups_host >= 0).all()
            assert (ups_host < npoint_sources).all()

        # }}}

        from pytools.obj_array import make_obj_array
        tree_order_point_sources = make_obj_array([
            cl.array.take(point_sources[i], user_point_source_ids)
            for i in range(self.dimensions)
            ])

        # {{{ compute box point source metadata

        from boxtree.tree_build_kernels import POINT_SOURCE_LINKING_BOX_POINT_SOURCES

        knl = POINT_SOURCE_LINKING_BOX_POINT_SOURCES.build(
            queue.context,
            type_aliases=(
                ("particle_id_t", self.particle_id_dtype),
                ),
            )

        logger.debug("point source linking: box point sources")

        box_point_source_starts = cl.array.empty(
                queue, self.nboxes, self.particle_id_dtype)
        box_point_source_counts = cl.array.empty(
                queue, self.nboxes, self.particle_id_dtype)

        knl(
                box_point_source_starts, box_point_source_counts,
                self.box_source_starts, self.box_source_counts,
                tree_order_point_source_starts,
                tree_order_point_source_counts,
                range=slice(self.nboxes), queue=queue)

        # }}}

        logger.info("point source linking: complete")

        tree_attrs = {}
        for attr_name in self.__class__.fields:
            try:
                tree_attrs[attr_name] = getattr(self, attr_name)
            except AttributeError:
                pass

        return TreeWithLinkedPointSources(
                npoint_sources=npoint_sources,
                point_source_starts=tree_order_point_source_starts,
                point_source_counts=tree_order_point_source_counts,
                point_sources=tree_order_point_sources,
                user_point_source_ids=user_point_source_ids,
                box_point_source_starts=box_point_source_starts,
                box_point_source_counts=box_point_source_counts,

                **tree_attrs)

    # }}}

    # {{{ debugging aids

    # these assume numpy arrays (i.e. post-.get()), for now

    def _reverse_index_lookup(self, ary, new_key_size):
        result = np.empty(new_key_size, ary.dtype)
        result.fill(-1)
        result[ary] = np.arange(len(ary), dtype=ary.dtype)
        return result

    def indices_to_tree_source_order(self, user_indices):
        # user_source_ids : tree order source indices -> user order source indices
        # tree_source_ids : user order source indices -> tree order source indices

        tree_source_ids = self._reverse_index_lookup(self.user_source_ids, self.nsources)
        return tree_source_ids[user_indices]

    def indices_to_tree_target_order(self, user_indices):
        # sorted_target_ids : user order target indices -> tree order target indices

        return self.sorted_target_ids[user_indices]

    def find_box_nr_for_target(self, itarget):
        """
        :arg itarget: target number in tree order
        """
        crit = (
                (self.box_target_starts <= itarget)
                &
                (itarget < self.box_target_starts + self.box_target_counts))

        return int(np.where(crit)[0])

    def find_box_nr_for_source(self, isource):
        """
        :arg isource: source number in tree order
        """
        crit = (
                (self.box_source_starts <= isource)
                &
                (isource < self.box_source_starts + self.box_source_counts))

        return int(np.where(crit)[0])

    # }}}


# }}}

# {{{ tree with linked point sources

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

# vim: filetype=pyopencl:fdm=marker
