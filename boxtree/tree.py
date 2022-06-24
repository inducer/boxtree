"""
.. _tree-kinds:

Supported tree kinds
--------------------

The following tree kinds are supported:

- *Nonadaptive* trees have all leaves on the same (last) level.

- *Adaptive* trees differ from nonadaptive trees in that they may have leaves on
  more than one level. Adaptive trees have the option of being
  *level-restricted*: in a level-restricted tree, neighboring leaves differ by
  at most one level.

All trees returned by the tree builder are pruned so that empty leaves have been
removed. If a level-restricted tree is requested, the tree gets constructed in
such a way that the version of the tree before pruning is also level-restricted.

Tree data structure
-------------------

.. currentmodule:: boxtree

.. autoclass:: box_flags_enum

.. autoclass:: TreeOfBoxes

.. autoclass:: Tree

.. currentmodule:: boxtree.tree

Tree with linked point sources
------------------------------

.. autoclass:: TreeWithLinkedPointSources

.. autofunction:: link_point_sources

Filtering the lists of targets
------------------------------

.. currentmodule:: boxtree.tree

Data structures
^^^^^^^^^^^^^^^

.. autoclass:: FilteredTargetListsInUserOrder
.. autoclass:: FilteredTargetListsInTreeOrder

Tools
^^^^^

.. autoclass:: ParticleListFilter
"""

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

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from arraycontext import Array

from cgen import Enum
from pytools import memoize_method

from boxtree.array_context import PyOpenCLArrayContext, dataclass_array_container


logger = logging.getLogger(__name__)


# {{{ box flags

class box_flags_enum(Enum):  # noqa
    """Constants for box flags bit field.

    .. rubric:: Flags for particle-based trees

    .. attribute:: dtype

    .. attribute:: IS_SOURCE_BOX
    .. attribute:: IS_TARGET_BOX
    .. attribute:: IS_SOURCE_OR_TARGET_BOX
    .. attribute:: HAS_SOURCE_CHILD_BOXES
    .. attribute:: HAS_TARGET_CHILD_BOXES
    .. attribute:: HAS_SOURCE_OR_TARGET_CHILD_BOXES
    .. attribute:: IS_LEAF_BOX

    .. warning ::

        :attr:`IS_LEAF_BOX` is only used for :class:`TreeOfBoxes` for the moment.
    """

    c_name = "box_flags_t"
    dtype = np.dtype(np.uint8)
    c_value_prefix = "BOX_"

    IS_SOURCE_BOX = 1 << 0
    IS_TARGET_BOX = 1 << 1
    IS_SOURCE_OR_TARGET_BOX = (IS_SOURCE_BOX | IS_TARGET_BOX)
    HAS_SOURCE_CHILD_BOXES = 1 << 2
    HAS_TARGET_CHILD_BOXES = 1 << 3
    HAS_SOURCE_OR_TARGET_CHILD_BOXES = (
            HAS_SOURCE_CHILD_BOXES | HAS_TARGET_CHILD_BOXES)

    # FIXME: Only used for TreeOfBoxes for now
    IS_LEAF_BOX = 1 << 4

    # Deprecated alias, do not use.
    HAS_CHILDREN = HAS_SOURCE_OR_TARGET_CHILD_BOXES

# }}}


# {{{ tree of boxes

@dataclass_array_container
@dataclass(frozen=True)
class TreeOfBoxes:
    """A quad/octree tree of pure boxes, excluding their contents (e.g.
    particles).  It is a lightweight tree handled with :mod:`numpy`, intended
    for mesh adaptivity. One may generate a :class:`meshmode.mesh.Mesh` object
    consisting of leaf boxes using :func:`make_meshmode_mesh_from_leaves`.

    .. attribute:: dimensions

    .. attribute:: nlevels

    .. attribute:: nboxes

    .. attribute:: root_extent

        (Scalar) extent of the root box.

    .. attribute:: box_centers

        mod:`numpy` array of shape ``(dim, nboxes)`` of the centers of the boxes.

    .. attribute:: box_parent_ids

        :mod:`numpy` vector of parent box ids.

    .. attribute:: box_child_ids

        (2**dim)-by-nboxes :mod:`numpy` array of children box ids.

    .. attribute:: box_levels

        :mod:`numpy` vector of box levels in non-decreasing order.

    .. attribute:: bounding_box

        A :class:`tuple` ``(bbox_min, bbox_max)`` of :mod:`numpy` vectors
        giving the (built) extent of the tree. Note that this may be slightly
        larger than what is required to contain all particles, if any.

    .. attribute:: box_flags

        :attr:`box_flags_enum.dtype` ``[nboxes]``

        A bitwise combination of :class:`box_flags_enum` constants.

    .. attribute:: level_start_box_nrs

        ``box_id_t [nlevels+1]``

        An array of box ids indicating the ID at which each level starts. Levels
        are contiguous in box ID space. To determine how many boxes there are
        in each level, access the start of the next level. This array is
        built so that this works even for the last level.

    .. attribute:: box_id_dtype
    .. attribute:: box_level_dtype
    .. attribute:: coord_dtype

        See :class:`Tree` documentation.

    .. attribute:: leaf_boxes

        Array of leaf boxes.

    .. attribute:: sources_have_extent
    .. attribute:: targets_have_extent
    .. attribute:: extent_norm
    .. attribute:: stick_out_factor

        See :class:`Tree` documentation.

    .. automethod:: __init__
    """

    root_extent: Array
    box_centers: Array

    box_parent_ids: Array
    box_child_ids: Array
    box_levels: Array

    box_flags: Array | None
    level_start_box_nrs: Array | None

    # FIXME: these should be properties and take values from box_parent_ids, etc
    box_id_dtype: np.dtype
    box_level_dtype: np.dtype
    coord_dtype: np.dtype

    sources_have_extent: bool
    targets_have_extent: bool
    extent_norm: str
    stick_out_factor: float

    _is_pruned: bool

    @property
    def dimensions(self):
        return self.box_centers.shape[0]

    @property
    def nboxes(self):
        return self.box_centers.shape[1]

    @property
    def aligned_nboxes(self):
        return self.box_child_ids.shape[-1]

    @property
    def nlevels(self):
        return max(self.box_levels) + 1

    @property
    def leaf_boxes(self):
        boxes = np.arange(self.nboxes)
        return boxes[self.box_flags & box_flags_enum.IS_LEAF_BOX != 0]

    @cached_property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        lows = self.box_centers[:, 0] - 0.5 * self.root_extent
        highs = lows + self.root_extent
        return lows, highs

    # {{{ dummy interface for TreePlotter

    def get_box_size(self, ibox):
        lev = self.box_levels[ibox]
        box_size = self.root_extent * 0.5**lev
        return box_size

    def get_box_extent(self, ibox):
        box_size = self.get_box_size(ibox)
        extent_low = self.box_centers[:, ibox] - 0.5*box_size
        extent_high = extent_low + box_size
        return extent_low, extent_high

    # }}}

# }}}


# {{{ tree with particles

@dataclass_array_container
@dataclass(frozen=True)
class Tree(TreeOfBoxes):
    r"""A quad/octree consisting of particles sorted into a hierarchy of boxes.

    Optionally, particles may be designated 'sources' and 'targets'. They
    may also be assigned radii which restrict the minimum size of the box
    into which they may be sorted.

    Instances of this class are not constructed directly. They are returned
    by :meth:`TreeBuilder.__call__`.

    Inherits from :class:`TreeOfBoxes`.

    .. rubric:: Flags

    .. attribute:: sources_are_targets

        ``bool``

        Whether sources and targets are the same

    .. attribute:: sources_have_extent

        ``bool``

        Whether this tree has sources in non-leaf boxes

    .. attribute:: targets_have_extent

        ``bool``

        Whether this tree has targets in non-leaf boxes

    .. ------------------------------------------------------------------------
    .. rubric:: Data types
    .. ------------------------------------------------------------------------

    .. attribute:: particle_id_dtype
    .. attribute:: box_id_dtype
    .. attribute:: coord_dtype
    .. attribute:: box_level_dtype

    .. ------------------------------------------------------------------------
    .. rubric:: Counts and sizes
    .. ------------------------------------------------------------------------

    .. attribute:: stick_out_factor

        A scalar used for calculating how much particles with extent may
        overextend their containing box.

        Each box in the tree can be thought of as being surrounded by a
        fictitious box whose :math:`l^\infty` radius is `1 + stick_out_factor`
        larger. Particles with extent are allowed to extend inside (a) the
        fictitious box or (b) a disk surrounding the fictitious box, depending on
        :attr:`extent_norm`.

    .. attribute:: extent_norm

        One of ``None``, ``"l2"`` or ``"linf"``. If *None*, particles do not have
        extent. If not *None*, indicates the norm with which extent-bearing particles
        are determined to lie 'inside' a box, taking into account the box's
        :attr:`stick_out_factor`.

        This image illustrates the difference in semantics:

        .. image:: images/linf-l2.png

        In the figure, the box has (:math:`\ell^\infty`) radius :math:`R`, the
        particle has radius :math:`r`, and :attr:`stick_out_factor` is denoted
        :math:`\alpha`.

    .. attribute:: nlevels

    .. attribute:: nboxes

    .. attribute:: nsources

    .. attribute:: ntargets

    .. attribute:: level_start_box_nrs

        ``box_id_t [nlevels+1]``

        An array of box ids indicating the ID at which each level starts. Levels
        are contiguous in box ID space. To determine how many boxes there are
        in each level, access the start of the next level. This array is
        built so that this works even for the last level.

    .. ------------------------------------------------------------------------
    .. rubric:: Per-particle arrays
    .. ------------------------------------------------------------------------

    .. attribute:: sources

        ``coord_t [dimensions][nsources]``
        (an object array of coordinate arrays)

        Stored in :ref:`tree source order <particle-orderings>`.
        May be the same array as :attr:`targets`.

    .. attribute:: source_radii

        ``coord_t [nsources]``
        :math:`l^\infty` radii of the :attr:`sources`.

        Available if :attr:`sources_have_extent` is *True*.

    .. attribute:: targets

        ``coord_t [dimensions][ntargets]``
        (an object array of coordinate arrays)

        Stored in :ref:`tree target order <particle-orderings>`. May be the
        same array as :attr:`sources`.

    .. attribute:: target_radii

        ``coord_t [ntargets]``

        :math:`l^\infty` radii of the :attr:`targets`.
        Available if :attr:`targets_have_extent` is *True*.

    .. ------------------------------------------------------------------------
    .. rubric:: Tree/user order indices
    .. ------------------------------------------------------------------------

    See :ref:`particle-orderings`.

    .. attribute:: user_source_ids

        ``particle_id_t [nsources]``

        Fetching *from* these indices will reorder the sources
        from user source order into :ref:`tree source order <particle-orderings>`.

    .. attribute:: sorted_target_ids

        ``particle_id_t [ntargets]``

        Fetching *from* these indices will reorder the targets
        from :ref:`tree target order <particle-orderings>` into user target order.

    .. ------------------------------------------------------------------------
    .. rubric:: Box properties
    .. ------------------------------------------------------------------------

    .. attribute:: box_source_starts

        ``particle_id_t [nboxes]``

        List of sources in each box. Records start indices in :attr:`sources`
        for each box.
        Use together with :attr:`box_source_counts_nonchild`
        or :attr:`box_source_counts_cumul`.
        May be the same array as :attr:`box_target_starts`.

    .. attribute:: box_source_counts_nonchild

        ``particle_id_t [nboxes]``

        List of sources in each box. Records number of sources from :attr:`sources`
        in each box (excluding those belonging to child boxes).
        Use together with :attr:`box_source_starts`.
        May be the same array as :attr:`box_target_counts_nonchild`.

    .. attribute:: box_source_counts_cumul

        ``particle_id_t [nboxes]``

        List of sources in each box. Records number of sources from :attr:`sources`
        in each box and its children.
        Use together with :attr:`box_source_starts`.
        May be the same array as :attr:`box_target_counts_cumul`.

    .. attribute:: box_target_starts

        ``particle_id_t [nboxes]``

        List of targets in each box. Records start indices in :attr:`targets`
        for each box.
        Use together with :attr:`box_target_counts_nonchild`
        or :attr:`box_target_counts_cumul`.
        May be the same array as :attr:`box_source_starts`.

    .. attribute:: box_target_counts_nonchild

        ``particle_id_t [nboxes]``

        List of targets in each box. Records number of targets from :attr:`targets`
        in each box (excluding those belonging to child boxes).
        Use together with :attr:`box_target_starts`.
        May be the same array as :attr:`box_source_counts_nonchild`.

    .. attribute:: box_target_counts_cumul

        ``particle_id_t [nboxes]``

        List of targets in each box. Records number of targets from :attr:`targets`
        in each box and its children.
        Use together with :attr:`box_target_starts`.
        May be the same array as :attr:`box_source_counts_cumul`.

    .. attribute:: box_parent_ids

        ``box_id_t [nboxes]``

        Box 0 (the root) has 0 as its parent.

    .. attribute:: box_child_ids

        ``box_id_t [2**dimensions, aligned_nboxes]`` (C order, 'structure of arrays')

        "0" is used as a 'no child' marker, as the root box can never
        occur as any box's child.

    .. attribute:: box_centers

        ``coord_t [dimensions, aligned_nboxes]`` (C order, 'structure of arrays')

    .. attribute:: box_levels

        :attr:`box_level_dtype` ``box_level_t [nboxes]``

    .. ------------------------------------------------------------------------
    .. rubric:: Particle-adaptive box extents
    .. ------------------------------------------------------------------------

    These attributes capture the maximum extent of particles (including the
    particle's extents) inside of the box.  If the box is empty, both *min* and *max*
    will reflect the box center.  The purpose of this information is to reduce the
    cost of some interactions through knowledge that some boxes are partially empty.
    (See the *from_sep_smaller_crit* argument to the constructor of
    :class:`boxtree.traversal.FMMTraversalBuilder` for an example.)

    .. note::

        To obtain the overall, non-adaptive box extent, use
        :attr:`boxtree.Tree.box_centers` along with :attr:`boxtree.Tree.box_levels`.

    If they are not available, the corresponding attributes will be *None*.

    .. attribute:: box_source_bounding_box_min

        ``coordt_t [dimensions, aligned_nboxes]``

    .. attribute:: box_source_bounding_box_max

        ``coordt_t [dimensions, aligned_nboxes]``

    .. attribute:: box_target_bounding_box_min

        ``coordt_t [dimensions, aligned_nboxes]``

    .. attribute:: box_target_bounding_box_max

        ``coordt_t [dimensions, aligned_nboxes]``
    """

    # flags
    sources_are_targets: bool

    # data types
    particle_id_dtype: np.dtype

    # per-particle arrays
    sources: Array
    source_radii: Array
    targets: Array
    target_radii: Array

    # FIXME: this needs to be init=True to overwrite the cached property in
    # the base class. That fails because `x[:, 0] - c` tries to do arithmetic
    # on a non-contiguous array and is not supported by pyopencl
    bounding_box: tuple[Array, Array] = field(init=True)

    # tree / user order indices
    user_source_ids: Array
    sorted_target_ids: Array

    # box properties
    box_source_starts: Array
    box_source_counts_nonchild: Array
    box_source_counts_cumul: Array
    box_target_starts: Array
    box_target_counts_nonchild: Array
    box_target_counts_cumul: Array

    # particle-adaptive box extents
    box_source_bounding_box_min: Array
    box_source_bounding_box_max: Array
    box_target_bounding_box_min: Array
    box_target_bounding_box_max: Array

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
        return len(self.sources[0])

    @property
    def ntargets(self):
        return len(self.targets[0])

    @property
    def nlevels(self):
        return len(self.level_start_box_nrs) - 1

    # {{{ dummy interface for TreePlotter

    def plot(self, **kwargs):
        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(self)
        plotter.draw_tree(**kwargs)
        plotter.set_bounding_box()

    def get_box_extent(self, ibox):
        lev = int(self.box_levels[ibox])
        box_size = self.root_extent / (1 << lev)
        extent_low = self.box_centers[:, ibox] - 0.5*box_size
        extent_high = extent_low + box_size
        return extent_low, extent_high

    # }}}

    # {{{ debugging aids

    # these assume numpy arrays for now

    def _reverse_index_lookup(self, ary, new_key_size):
        result = np.empty(new_key_size, ary.dtype)
        result.fill(-1)
        result[ary] = np.arange(len(ary), dtype=ary.dtype)
        return result

    def indices_to_tree_source_order(self, user_indices):
        # user_source_ids : tree order source indices -> user order source indices
        # tree_source_ids : user order source indices -> tree order source indices

        tree_source_ids = self._reverse_index_lookup(
                self.user_source_ids, self.nsources)
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
                & (itarget
                    < self.box_target_starts + self.box_target_counts_nonchild))

        return int(np.where(crit)[0])

    def find_box_nr_for_source(self, isource):
        """
        :arg isource: source number in tree order
        """
        crit = (
                (self.box_source_starts <= isource)
                & (isource
                    < self.box_source_starts + self.box_source_counts_nonchild))

        return int(np.where(crit)[0])

    # }}}

# }}}


# {{{ tree with linked point sources

@dataclass_array_container
@dataclass(frozen=True)
class TreeWithLinkedPointSources(Tree):
    """In this :class:`boxtree.Tree` subclass, the sources of the original tree are
    linked with extent are expanded into point sources which are linked to the
    extent-having sources in the original tree. (In an FMM context, they may
    stand in for the 'underlying' source for the purpose of the far-field
    calculation.) Has all the same attributes as :class:`boxtree.Tree`.
    :attr:`boxtree.Tree.sources_have_extent` is always *True* for instances of this
    type. In addition, the following attributes are available.

    .. attribute:: npoint_sources

    .. attribute:: point_source_starts

        ``particle_id_t [nsources]``

        The array
        ``point_sources[:][point_source_starts[isrc]:
        point_source_starts[isrc]+point_source_counts[isrc]]``
        contains the locations of point sources corresponding to
        the 'original' source with index *isrc*. (Note that this
        expression will not entirely work because :attr:`point_sources`
        is an object array.)

        This array is stored in :ref:`tree point source order <particle-orderings>`,
        unlike the parameter to
        :meth:`boxtree.tree.TreeWithLinkedPointSources.__init__`

    .. attribute:: point_source_counts

        ``particle_id_t [nsources]`` (See :attr:`point_source_starts`.)

    .. attribute:: point_sources

        ``coord_t [dimensions][npoint_sources]``
        (an object array of coordinate arrays)

        Stored in :ref:`tree point source order <particle-orderings>`.

    .. attribute:: user_point_source_ids

        ``particle_id_t [nsources]``

        Fetching *from* these indices will reorder the sources
        from user point source order into
        :ref:`tree point source order <particle-orderings>`.

    .. attribute:: box_point_source_starts

        ``particle_id_t [nboxes]``

    .. attribute:: box_point_source_counts_nonchild

        ``particle_id_t [nboxes]``

    .. attribute:: box_point_source_counts_cumul

        ``particle_id_t [nboxes]``

    .. method:: __init__

        This constructor is not intended to be called by users directly.
        Call :func:`link_point_sources` instead.
    """

    npoint_sources: int
    point_source_starts: Array
    point_source_counts: Array
    point_sources: Array
    user_point_source_ids: Array
    box_point_source_starts: Array
    box_point_source_counts_nonchild: Array
    box_point_source_counts_cumul: Array


def link_point_sources(
        actx: PyOpenCLArrayContext, tree: Tree,
        point_source_starts: Array, point_sources: Array, *,
        debug: bool = False):
    r"""
    *Construction:* Requires that :attr:`boxtree.Tree.sources_have_extent` is *True*
    on *tree*.

    :arg point_source_starts: ``point_source_starts[isrc]`` and
        ``point_source_starts[isrc+1]`` together indicate a ranges of point
        particle indices in *point_sources* which will be linked to the
        original (extent-having) source number *isrc*. *isrc* is in :ref:`user
        source order <particle-orderings>`.

        All the particles linked to *isrc* should fall within the :math:`l^\infty`
        'circle' around particle number *isrc* with the radius drawn from
        :attr:`boxtree.Tree.source_radii`.

    :arg point_sources: an object array of (XYZ) point coordinate arrays.
    """

    # The whole point of this routine is that all point sources within
    # a box are reordered to be contiguous.

    logger.info("point source linking: start")

    if not tree.sources_have_extent:
        raise ValueError("only allowed on trees whose sources have extent")

    npoint_sources_dev = actx.np.zeros((), tree.particle_id_dtype)

    # {{{ compute tree_order_point_source_{starts, counts}

    # Scan over lengths of point source lists in tree order to determine
    # indices of point source starts for each source.

    tree_order_point_source_starts = actx.np.zeros(
            tree.nsources, tree.particle_id_dtype)
    tree_order_point_source_counts = actx.np.zeros(
            tree.nsources, tree.particle_id_dtype)

    from boxtree.tree_build_kernels import POINT_SOURCE_LINKING_SOURCE_SCAN_TPL
    knl = POINT_SOURCE_LINKING_SOURCE_SCAN_TPL.build(
        actx.queue.context,
        type_aliases=(
            ("scan_t", tree.particle_id_dtype),
            ("index_t", tree.particle_id_dtype),
            ("particle_id_t", tree.particle_id_dtype),
            ),
        )

    logger.debug("point source linking: tree order source scan")

    knl(point_source_starts, tree.user_source_ids,
            tree_order_point_source_starts, tree_order_point_source_counts,
            npoint_sources_dev, size=tree.nsources, queue=actx.queue)

    # }}}

    npoint_sources = int(actx.to_numpy(npoint_sources_dev))

    # {{{ compute user_point_source_ids

    # A list of point source starts, indexed in tree order,
    # but giving point source indices in user order.
    tree_order_index_user_point_source_starts = (
            point_source_starts[tree.user_source_ids])

    user_point_source_ids = actx.np.zeros(npoint_sources, tree.particle_id_dtype)
    user_point_source_ids.fill(1)

    import pyopencl.array as cl_array
    cl_array.multi_put(
            [tree_order_index_user_point_source_starts],
            dest_indices=tree_order_point_source_starts,
            out=[user_point_source_ids])

    if debug:
        ups_host = actx.to_numpy(user_point_source_ids)
        assert np.all(ups_host >= 0)
        assert np.all(ups_host < npoint_sources)

    source_boundaries = actx.zeros(npoint_sources, np.int8)

    # FIXME: Should be a scalar, in principle.
    ones = actx.np.zeros(tree.nsources, np.int8)
    ones.fill(1)

    cl_array.multi_put(
            [ones],
            dest_indices=tree_order_point_source_starts,
            out=[source_boundaries])

    from boxtree.tree_build_kernels import (
        POINT_SOURCE_LINKING_USER_POINT_SOURCE_ID_SCAN_TPL,
    )

    logger.debug("point source linking: point source id scan")

    knl = POINT_SOURCE_LINKING_USER_POINT_SOURCE_ID_SCAN_TPL.build(
        actx.queue.context,
        type_aliases=(
            ("scan_t", tree.particle_id_dtype),
            ("index_t", tree.particle_id_dtype),
            ("particle_id_t", tree.particle_id_dtype),
            ),
        )
    knl(source_boundaries, user_point_source_ids,
            size=npoint_sources, queue=actx.queue)

    if debug:
        ups_host = actx.to_numpy(user_point_source_ids)
        assert np.all(ups_host >= 0)
        assert np.all(ups_host < npoint_sources)

    # }}}

    from pytools.obj_array import make_obj_array
    tree_order_point_sources = make_obj_array([
        cl_array.take(point_sources[i], user_point_source_ids, queue=actx.queue)
        for i in range(tree.dimensions)
        ])

    # {{{ compute box point source metadata

    from boxtree.tree_build_kernels import POINT_SOURCE_LINKING_BOX_POINT_SOURCES

    knl = POINT_SOURCE_LINKING_BOX_POINT_SOURCES.build(
        actx.queue.context,
        type_aliases=(
            ("particle_id_t", tree.particle_id_dtype),
            ("box_id_t", tree.box_id_dtype),
            ),
        )

    logger.debug("point source linking: box point sources")

    box_point_source_starts = actx.np.zeros(tree.nboxes, tree.particle_id_dtype)
    box_point_source_counts_cumul = actx.np.zeros(tree.nboxes, tree.particle_id_dtype)
    box_point_source_counts_nonchild = actx.np.zeros(
            tree.nboxes, tree.particle_id_dtype)

    knl(
            box_point_source_starts, box_point_source_counts_nonchild,
            box_point_source_counts_cumul,

            tree.box_source_starts, tree.box_source_counts_nonchild,
            tree.box_source_counts_cumul,

            tree_order_point_source_starts,
            tree_order_point_source_counts,
            range=slice(tree.nboxes), queue=actx.queue)

    # }}}

    logger.info("point source linking: complete")

    from dataclasses import fields
    tree_attrs = {}
    for f in fields(tree):
        try:  # noqa: SIM105
            tree_attrs[f.name] = getattr(tree, f.name)
        except AttributeError:
            pass

    tree_with_point_sources = TreeWithLinkedPointSources(
            npoint_sources=npoint_sources,
            point_source_starts=tree_order_point_source_starts,
            point_source_counts=tree_order_point_source_counts,
            point_sources=tree_order_point_sources,
            user_point_source_ids=user_point_source_ids,
            box_point_source_starts=box_point_source_starts,
            box_point_source_counts_nonchild=box_point_source_counts_nonchild,
            box_point_source_counts_cumul=box_point_source_counts_cumul,

            **tree_attrs)

    return actx.freeze(tree_with_point_sources)


# }}}


# {{{ particle list filter

@dataclass_array_container
@dataclass(frozen=True)
class FilteredTargetListsInUserOrder:
    """Use :meth:`ParticleListFilter.filter_target_lists_in_user_order` to create
    instances of this class.

    This class represents subsets of the list of targets in each box (as given
    by :attr:`boxtree.Tree.box_target_starts` and
    :attr:`boxtree.Tree.box_target_counts_cumul`). This subset is specified by
    an array of *flags* in user target order.

    The list consists of target numbers in user target order.
    See also :class:`FilteredTargetListsInTreeOrder`.

    .. attribute:: nfiltered_targets

    .. attribute:: target_starts

        ``particle_id_t [nboxes+1]``

        Filtered list of targets in each box. Records start indices in
        :attr:`boxtree.Tree.targets` for each box.  Use together with
        :attr:`target_lists`. The lists for each box are
        contiguous, so that ``target_starts[ibox+1]`` records the
        end of the target list for *ibox*.

    .. attribute:: target_lists

        ``particle_id_t [nboxes]``

        Filtered list of targets in each box. Records number of targets from
        :attr:`boxtree.Tree.targets` in each box (excluding those belonging to
        child boxes).  Use together with :attr:`target_starts`.

        Target numbers are stored in user order, as the class name suggests.
    """

    nfiltered_targets: int
    target_starts: Array
    target_lists: Array


@dataclass_array_container
@dataclass(frozen=True)
class FilteredTargetListsInTreeOrder:
    """Use :meth:`ParticleListFilter.filter_target_lists_in_tree_order` to create
    instances of this class.

    This class represents subsets of the list of targets in each box (as given by
    :attr:`boxtree.Tree.box_target_starts` and
    :attr:`boxtree.Tree.box_target_counts_cumul`).This subset is
    specified by an array of *flags* in user target order.

    Unlike :class:`FilteredTargetListsInUserOrder`, this does not create a
    CSR-like list of targets, but instead creates a new numbering of targets
    that only counts the filtered targets. This allows all targets in a box to
    be placed consecutively, which is intended to help traversal performance.

    .. attribute:: nfiltered_targets

    .. attribute:: box_target_starts

        ``particle_id_t [nboxes]``

        Filtered list of targets in each box, like
        :attr:`boxtree.Tree.box_target_starts`.  Records start indices in
        :attr:`targets` for each box.  Use together with
        :attr:`box_target_counts_nonchild`.

    .. attribute:: box_target_counts_nonchild

        ``particle_id_t [nboxes]``

        Filtered list of targets in each box, like
        :attr:`boxtree.Tree.box_target_counts_nonchild`.
        Records number of sources from :attr:`targets` in each box
        (excluding those belonging to child boxes).
        Use together with :attr:`box_target_starts`.

    .. attribute:: targets

        ``coord_t [dimensions][nfiltered_targets]``
        (an object array of coordinate arrays)

    .. attribute:: unfiltered_from_filtered_target_indices

        Storing *to* these indices will reorder the targets
        from *filtered* tree target order into 'regular'
        :ref:`tree target order <particle-orderings>`.
    """

    nfiltered_targets: int
    box_target_starts: Array
    box_target_counts_nonchild: Array
    targets: Array
    unfiltered_from_filtered_target_indices: Array


class ParticleListFilter:
    """
    .. automethod:: filter_target_lists_in_tree_order
    .. automethod:: filter_target_lists_in_user_order
    """

    def __init__(self, array_context: PyOpenCLArrayContext):
        self._setup_actx = array_context

    @property
    def context(self):
        return self._setup_actx.queue.context

    @memoize_method
    def get_filter_target_lists_in_user_order_kernel(self, particle_id_dtype,
            user_order_flags_dtype):
        from mako.template import Template

        from pyopencl.algorithm import ListOfListsBuilder
        from pyopencl.tools import dtype_to_ctype

        from boxtree.tools import VectorArg

        builder = ListOfListsBuilder(self.context,
            [("filt_tgt_list", particle_id_dtype)], Template("""//CL//
            typedef ${dtype_to_ctype(particle_id_dtype)} particle_id_t;

            void generate(LIST_ARG_DECL USER_ARG_DECL index_type i)
            {
                particle_id_t b_t_start = box_target_starts[i];
                particle_id_t b_t_count = box_target_counts_nonchild[i];

                for (particle_id_t j = b_t_start; j < b_t_start+b_t_count; ++j)
                {
                    particle_id_t user_target_id = user_target_ids[j];
                    if (user_order_flags[user_target_id])
                    {
                        APPEND_filt_tgt_list(user_target_id);
                    }
                }
            }
            """, strict_undefined=True).render(
                dtype_to_ctype=dtype_to_ctype,
                particle_id_dtype=particle_id_dtype
                ), arg_decls=[
                    VectorArg(user_order_flags_dtype, "user_order_flags"),
                    VectorArg(particle_id_dtype, "user_target_ids"),
                    VectorArg(particle_id_dtype, "box_target_starts"),
                    VectorArg(particle_id_dtype, "box_target_counts_nonchild"),
                ])

        return builder

    def filter_target_lists_in_user_order(self, actx, tree, flags):
        """
        :arg flags: an array of length :attr:`boxtree.Tree.ntargets` of
            :class:`numpy.int8` objects, which indicate by being zero that the
            corresponding target (in user target order) is not part of the
            filtered list, or by being nonzero that it is.

        :returns: A :class:`FilteredTargetListsInUserOrder`
        """
        user_order_flags = flags
        del flags

        user_target_ids = actx.np.zeros(tree.ntargets, tree.sorted_target_ids.dtype)
        user_target_ids[tree.sorted_target_ids] = actx.from_numpy(
                np.arange(tree.ntargets, dtype=user_target_ids.dtype)
                )

        kernel = self.get_filter_target_lists_in_user_order_kernel(
                tree.particle_id_dtype, user_order_flags.dtype)

        result, _evt = kernel(actx.queue, tree.nboxes,
                user_order_flags,
                user_target_ids,
                tree.box_target_starts,
                tree.box_target_counts_nonchild)

        target_lists = FilteredTargetListsInUserOrder(
                nfiltered_targets=result["filt_tgt_list"].count,
                target_starts=result["filt_tgt_list"].starts,
                target_lists=result["filt_tgt_list"].lists,
                )

        return actx.freeze(target_lists)

    @memoize_method
    def get_filter_target_lists_in_tree_order_kernels(self, particle_id_dtype):
        from boxtree.tree_build_kernels import (
            TREE_ORDER_TARGET_FILTER_INDEX_TPL,
            TREE_ORDER_TARGET_FILTER_SCAN_TPL,
        )

        scan_knl = TREE_ORDER_TARGET_FILTER_SCAN_TPL.build(
            self.context,
            type_aliases=(
                ("scan_t", particle_id_dtype),
                ("particle_id_t", particle_id_dtype),
                ),
            )

        index_knl = TREE_ORDER_TARGET_FILTER_INDEX_TPL.build(
            self.context,
            type_aliases=(
                ("particle_id_t", particle_id_dtype),
                ),
            )

        return scan_knl, index_knl

    def filter_target_lists_in_tree_order(self, actx, tree, flags):
        """
        :arg flags: an array of length :attr:`boxtree.Tree.ntargets` of
            :class:`numpy.int8` objects, which indicate by being zero that the
            corresponding target (in user target order) is not part of the
            filtered list, or by being nonzero that it is.
        :returns: A :class:`FilteredTargetListsInTreeOrder`
        """

        tree_order_flags = actx.np.zeros(tree.ntargets, np.int8)
        tree_order_flags[tree.sorted_target_ids] = flags

        filtered_from_unfiltered_target_indices = actx.np.zeros(
                tree.ntargets, tree.particle_id_dtype)
        unfiltered_from_filtered_target_indices = actx.np.zeros(
                tree.ntargets, tree.particle_id_dtype)

        nfiltered_targets = actx.np.zeros(1, tree.particle_id_dtype)

        scan_knl, index_knl = self.get_filter_target_lists_in_tree_order_kernels(
                tree.particle_id_dtype)

        scan_knl(tree_order_flags,
                filtered_from_unfiltered_target_indices,
                unfiltered_from_filtered_target_indices,
                nfiltered_targets,
                queue=actx.queue)

        nfiltered_targets = int(actx.to_numpy(nfiltered_targets).item())
        unfiltered_from_filtered_target_indices = \
                unfiltered_from_filtered_target_indices[:nfiltered_targets]

        from pytools.obj_array import make_obj_array
        filtered_targets = make_obj_array([
            actx.thaw(targets_i)[unfiltered_from_filtered_target_indices]
            for targets_i in tree.targets
            ])

        box_target_starts_filtered = actx.np.zeros_like(tree.box_target_starts)
        box_target_counts_nonchild_filtered = (
                actx.np.zeros_like(tree.box_target_counts_nonchild))

        index_knl(
                # input
                tree.box_target_starts,
                tree.box_target_counts_nonchild,
                filtered_from_unfiltered_target_indices,
                tree.ntargets,
                nfiltered_targets,

                # output
                box_target_starts_filtered,
                box_target_counts_nonchild_filtered,

                queue=actx.queue)

        target_lists = FilteredTargetListsInTreeOrder(
                nfiltered_targets=nfiltered_targets,
                box_target_starts=box_target_starts_filtered,
                box_target_counts_nonchild=box_target_counts_nonchild_filtered,
                unfiltered_from_filtered_target_indices=(
                    unfiltered_from_filtered_target_indices),
                targets=filtered_targets,
                )

        return actx.freeze(target_lists)

# }}}

# vim: filetype=pyopencl:fdm=marker
