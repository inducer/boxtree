"""
.. currentmodule:: boxtree

.. _tree-of-boxes:

Manipulating Trees of Boxes
---------------------------

These functions manipulate instances of :class:`TreeOfBoxes`.

.. note::

    These functions currently keep their bulk data in :class:`numpy.ndarray`
    instances.  This contrasts with the particle-based tree (:class:`Tree`),
    which operates on data in :class:`pyopencl.array.Array` instances).  Along
    with the rest of :mod:`boxtree`, this will migrate to :mod:`arraycontext`
    in the future.

.. autofunction:: make_tree_of_boxes_root
.. autofunction:: refine_tree_of_boxes
.. autofunction:: uniformly_refine_tree_of_boxes
.. autofunction:: coarsen_tree_of_boxes
.. autofunction:: refine_and_coarsen_tree_of_boxes
.. autofunction:: make_meshmode_mesh_from_leaves
"""

__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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

import sys
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from boxtree.tree import TreeOfBoxes, box_flags_enum


if TYPE_CHECKING or getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    from meshmode.mesh import Mesh


# {{{ utils for tree of boxes

def _compute_tob_box_flags(box_child_ids: np.ndarray) -> np.ndarray:
    nboxes = box_child_ids.shape[1]
    # For the time being, we will work with the assumption that each box
    # in the tree is both a source and a target box.
    box_flags = np.full(
            nboxes,
            box_flags_enum.IS_SOURCE_BOX | box_flags_enum.IS_TARGET_BOX,
            dtype=box_flags_enum.dtype)

    box_is_leaf = np.all(box_child_ids == 0, axis=0)
    box_flags[box_is_leaf] = box_flags[box_is_leaf] | box_flags_enum.IS_LEAF_BOX

    box_flags[~box_is_leaf] = box_flags[~box_is_leaf] | (
            box_flags_enum.HAS_SOURCE_CHILD_BOXES
            | box_flags_enum.HAS_TARGET_CHILD_BOXES)

    return box_flags


def _resized_array(arr: np.ndarray, new_size: int) -> np.ndarray:
    """Return a resized copy of the array. The new_size is a scalar which is
    applied to the last dimension.
    """
    old_size = arr.shape[-1]
    prefix = (slice(None), ) * (arr.ndim - 1)
    if old_size >= new_size:
        return arr[prefix + (slice(new_size), )].copy()
    else:
        new_shape = list(arr.shape)
        new_shape[-1] = new_size
        new_arr = np.zeros(new_shape, arr.dtype)
        new_arr[prefix + (slice(old_size), )] = arr
        return new_arr


def _vec_of_signs(dim: int, i: int) -> np.ndarray:
    """The sign vector is obtained by converting i to a dim-bit binary.
    """
    # e.g. bin(10) = '0b1010'
    binary_digits = [int(bd) for bd in bin(i)[2:]]
    n = len(binary_digits)
    assert n <= dim
    return np.array([0]*(dim-n) + binary_digits) * 2 - 1

# }}}


# {{{ refine/coarsen a tree of boxes

def refine_tree_of_boxes(tob: TreeOfBoxes, refine_flags: np.ndarray) -> TreeOfBoxes:
    """Make a refined copy of `tob` where boxes flagged with `refine_flags` are
    refined.
    """
    return refine_and_coarsen_tree_of_boxes(tob, refine_flags, None)


def uniformly_refine_tree_of_boxes(tob: TreeOfBoxes) -> TreeOfBoxes:
    """Make a uniformly refined copy of `tob`.
    """
    refine_flags = np.zeros(tob.nboxes, bool)
    refine_flags[tob.box_flags & box_flags_enum.IS_LEAF_BOX != 0] = 1
    return refine_tree_of_boxes(tob, refine_flags)


def coarsen_tree_of_boxes(
        tob: TreeOfBoxes, coarsen_flags: np.ndarray,
        error_on_ignored_flags: bool = True
        ) -> TreeOfBoxes:
    """Make a coarsened copy of `tob` where boxes flagged with `coarsen_flags`
    are coarsened.
    """
    return refine_and_coarsen_tree_of_boxes(
        tob, None, coarsen_flags,
        error_on_ignored_flags=error_on_ignored_flags)


def _apply_refine_flags_without_sorting(refine_flags, tob):
    box_is_leaf = tob.box_flags & box_flags_enum.IS_LEAF_BOX != 0

    if refine_flags[~box_is_leaf].any():
        raise ValueError("attempting to split non-leaf")

    refine_parents, = np.where(refine_flags)
    if len(refine_parents) == 0:
        return tob

    dim = tob.dimensions
    nchildren = 2**dim
    n_new_boxes = len(refine_parents) * nchildren
    nboxes_new = tob.nboxes + n_new_boxes

    child_box_starts = (
            tob.nboxes
            + nchildren * np.arange(len(refine_parents)))

    refine_parents_per_child = np.empty(
            (nchildren, len(refine_parents)), np.intp)
    refine_parents_per_child[:] = refine_parents.reshape(-1)
    refine_parents_per_child = refine_parents_per_child.reshape(-1)

    box_parents = _resized_array(tob.box_parent_ids, nboxes_new)
    box_centers = _resized_array(tob.box_centers, nboxes_new)
    box_children = _resized_array(tob.box_child_ids, nboxes_new)
    box_levels = _resized_array(tob.box_levels, nboxes_new)

    # new boxes are appended at the end, so applying coarsen_flags wrt the
    # original tree is still meaningful after this
    box_parents[tob.nboxes:] = refine_parents_per_child
    box_levels[tob.nboxes:] = tob.box_levels[box_parents[tob.nboxes:]] + 1
    box_children[:, refine_parents] = (
        child_box_starts + np.arange(nchildren).reshape(-1, 1))

    for i in range(2**dim):
        children_i = box_children[i, refine_parents]
        offsets = (
                tob.root_extent * _vec_of_signs(dim, i).reshape(-1, 1)
                * (1/2**(1+box_levels[children_i])))
        box_centers[:, children_i] = (
                box_centers[:, refine_parents] + offsets)

    return TreeOfBoxes(
        box_centers=box_centers,
        root_extent=tob.root_extent,
        box_parent_ids=box_parents,
        box_child_ids=box_children,
        box_levels=box_levels,

        box_flags=_compute_tob_box_flags(box_children),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
        )


def _apply_coarsen_flags(coarsen_flags, tob, error_on_ignored_flags=True):
    box_is_leaf = tob.box_flags & box_flags_enum.IS_LEAF_BOX != 0
    if coarsen_flags[~box_is_leaf].any():
        raise ValueError("attempting to coarsen non-leaf")
    coarsen_sources, = np.where(coarsen_flags)
    if coarsen_sources.size == 0:
        return tob

    coarsen_parents = tob.box_parent_ids[coarsen_sources]
    coarsen_peers = tob.box_child_ids[:, coarsen_parents].reshape(-1)
    coarsen_peer_is_leaf = box_is_leaf[coarsen_peers]
    coarsen_exec_flags = np.all(coarsen_peer_is_leaf, axis=0)

    # when a leaf box marked for coarsening has non-leaf peers
    coarsen_flags_ignored = (coarsen_exec_flags != coarsen_flags)
    if np.any(coarsen_flags_ignored):
        msg = (f"{np.sum(coarsen_flags_ignored)} out of "
               f"{np.sum(coarsen_flags)} coarsening flags ignored "
               "to prevent removing non-leaf boxes")
        if error_on_ignored_flags:
            raise RuntimeError(msg)
        else:
            import warnings
            warnings.warn(msg, stacklevel=3)

    # deleted boxes are marked as:
    # level = inf
    # parent = -1
    coarsen_parents = coarsen_parents[coarsen_exec_flags]
    coarsen_peers = coarsen_peers[:, coarsen_exec_flags]
    box_parents = tob.box_parent_ids.copy()
    box_parents[coarsen_peers] = -1
    box_children = tob.box_child_ids.copy()
    box_children[:, coarsen_parents] = 0
    box_levels = tob.box_levels.copy()
    box_levels[coarsen_peers] = np.inf

    return TreeOfBoxes(
        box_centers=tob.box_centers,
        root_extent=tob.root_extent,
        box_parent_ids=box_parents,
        box_child_ids=box_children,
        box_levels=box_levels,

        box_flags=_compute_tob_box_flags(box_children),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
        )


def _sort_boxes_by_level(tob, queue=None):
    if not np.any(np.diff(tob.box_levels) < 0):
        return tob

    # reorder boxes to into non-decreasing levels
    neworder = np.argsort(tob.box_levels)
    box_centers = tob.box_centers[:, neworder]
    box_parent_ids = tob.box_parent_ids[neworder]
    box_child_ids = tob.box_child_ids[:, neworder]
    box_levels = tob.box_levels[neworder]

    return TreeOfBoxes(
        box_centers=box_centers,
        root_extent=tob.root_extent,
        box_parent_ids=box_parent_ids,
        box_child_ids=box_child_ids,
        box_levels=box_levels,

        box_flags=_compute_tob_box_flags(box_child_ids),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
        )


def _sort_and_prune_deleted_boxes(tob):
    tob = _sort_boxes_by_level(tob)
    n_stale_boxes = np.sum(tob.box_levels == np.inf)
    newn = tob.nboxes - n_stale_boxes

    return TreeOfBoxes(
        root_extent=tob.root_extent,
        box_parent_ids=tob.box_parent_ids[:newn],
        box_child_ids=tob.box_child_ids[:, :newn],
        box_levels=tob.box_levels[:newn],
        box_centers=tob.box_centers[:, :newn],

        box_flags=_compute_tob_box_flags(tob.box_child_ids[:, :newn]),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
        )


def refine_and_coarsen_tree_of_boxes(
        tob: TreeOfBoxes,
        refine_flags: Optional[np.ndarray] = None,
        coarsen_flags: Optional[np.ndarray] = None, *,
        error_on_ignored_flags: bool = True,
        ) -> TreeOfBoxes:
    """Make a refined/coarsened copy. When children of the same parent box
    are marked differently, the refinement flag takes priority.

    Both refinement and coarsening flags can only be set of leaves.
    To prevent drastic mesh change, coarsening is only executed when a leaf
    box is marked for coarsening, and its parent's children are all leaf
    boxes (so that change in the number of boxes is bounded per box flagged).
    Please note that the above behavior may be subject to change in the future.

    :arg refine_flags: a boolean array of size `nboxes`.
    :arg coarsen_flags: a boolean array of size `nboxes`.
    :arg error_on_ignored_flags: if true, an exception is raised when enforcing
        level restriction requires ignoring some coarsening flags.
    :returns: a processed copy of the tree.
    """
    if refine_flags is None:
        refine_flags = np.zeros(tob.nboxes, dtype=bool)
    if coarsen_flags is None:
        coarsen_flags = np.zeros(tob.nboxes, dtype=bool)

    if (refine_flags & coarsen_flags).any():
        raise ValueError("some boxes are simultaneously marked "
                         "to refine and coarsen")

    tob = _apply_refine_flags_without_sorting(refine_flags, tob)
    coarsen_flags = _resized_array(coarsen_flags, tob.nboxes)
    tob = _apply_coarsen_flags(coarsen_flags, tob, error_on_ignored_flags)
    return _sort_and_prune_deleted_boxes(tob)

# }}}


# {{{ make_tree_of_boxes_root

def make_tree_of_boxes_root(
        bbox: Tuple[np.ndarray, np.ndarray], *,
        box_id_dtype: Any = None,
        box_level_dtype: Any = None,
        coord_dtype: Any = None,
        ) -> TreeOfBoxes:
    """
    Make the minimal tree of boxes, consisting of a single root box filling
    *bbox*.

    .. note::

        *bbox* is expected to be square (with tolerances as accepted by
        :func:`numpy.allclose`).

    :arg bbox: a :class:`tuple` of ``(lower_bounds, upper_bounds)`` for the
        bounding box.
    """
    assert len(bbox) == 2

    from pytools import single_valued
    dim = single_valued([len(bbox[0]), len(bbox[1])])

    if box_id_dtype is None:
        box_id_dtype = np.int32
    box_id_dtype = np.dtype(box_id_dtype)

    if box_level_dtype is None:
        box_level_dtype = np.int32
    box_level_dtype = np.dtype(box_level_dtype)

    if coord_dtype is None:
        coord_dtype = bbox[0].dtype
    coord_dtype = np.dtype(coord_dtype)

    box_centers = np.array(
        [(bbox[0][iaxis] + bbox[1][iaxis]) * 0.5 for iaxis in range(dim)],
        dtype=coord_dtype,
        ).reshape(dim, 1)
    root_extent = single_valued(
        np.array(
            [(bbox[1][iaxis] - bbox[0][iaxis]) for iaxis in range(dim)],
            dtype=coord_dtype),
        equality_pred=np.allclose)

    box_parent_ids = np.array([0], dtype=box_id_dtype)
    box_parent_ids[0] = -1  # root has no parent

    box_child_ids = np.array([0] * 2**dim, box_id_dtype).reshape(2**dim, 1)

    return TreeOfBoxes(
            box_centers=box_centers,
            root_extent=root_extent,
            box_parent_ids=box_parent_ids,
            box_child_ids=box_child_ids,
            box_levels=np.array([0], box_level_dtype),

            box_flags=_compute_tob_box_flags(box_child_ids),
            level_start_box_nrs=np.array([0], dtype=box_level_dtype),

            box_id_dtype=box_id_dtype,
            box_level_dtype=box_level_dtype,
            coord_dtype=coord_dtype,
            sources_have_extent=False,
            targets_have_extent=False,
            extent_norm="linf",
            stick_out_factor=0,
            _is_pruned=True,
            )

# }}}


# {{{ make_meshmode_mesh_from_leaves

def make_meshmode_mesh_from_leaves(tob: TreeOfBoxes) -> Tuple["Mesh", np.ndarray]:
    """Make a :class:`~meshmode.mesh.Mesh` from the leaf boxes of the tree
    of boxes *tob*.

    :returns: A tuple of the mesh and a vector of the element number -> box number
        mapping.
    """
    dim = tob.dimensions
    lfboxes = tob.leaf_boxes
    lfcenters = tob.box_centers[:, lfboxes]
    lflevels = tob.box_levels[lfboxes]
    lfradii = tob.root_extent / 2 / (2**lflevels)

    # use tensor product nodes ordering
    import modepy.nodes as nd
    cell_nodes_1d = np.array([-1, 1])
    cell_nodes = nd.tensor_product_nodes(dim, cell_nodes_1d)

    lfvertices = (
        np.repeat(lfcenters, 2**dim, axis=1)
        + np.repeat(lfradii, 2**dim) * np.tile(cell_nodes, (1, len(lfboxes)))
    )

    # FIXME: purge redundant vertices
    from meshmode.mesh import Mesh, TensorProductElementGroup
    from meshmode.mesh.generation import make_group_from_vertices

    vertex_indices = np.arange(
        len(lfboxes) * 2**dim, dtype=np.int32).reshape([-1, 2**dim])
    group = make_group_from_vertices(
        lfvertices, vertex_indices, 1,
        group_cls=TensorProductElementGroup,
        unit_nodes=None)

    return Mesh(vertices=lfvertices, groups=[group]), tob.leaf_boxes

# }}}

# vim: foldmethod=marker
