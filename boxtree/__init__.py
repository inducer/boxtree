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

from boxtree.tree import Tree, TreeWithLinkedPointSources, box_flags_enum
from boxtree.tree_build import TreeBuilder

__all__ = [
    "Tree", "TreeWithLinkedPointSources",
    "TreeBuilder", "box_flags_enum"]

__doc__ = r"""
:mod:`boxtree` can do three main things:

* it can sort particles into an adaptively refined quad/octree,
  see :class:`boxtree.Tree` and :class:`boxtree.TreeBuilder`.

* it can compute fast-multipole-like interaction lists on this tree structure,
  see :mod:`boxtree.traversal`. Note that while this traversal generation
  builds on the result of particle sorting,
  it is completely distinct in the software sense.

* It can compute geometric lookup structures based on a :class:`boxtree.Tree`,
  see :mod:`boxtree.area_query`.

Tree modes
----------

:mod:`boxtree` can operate in three 'modes':

* one where no distinction is made between sources and targets. In this mode,
  all participants in the interaction are called 'particles'.
  (``targets is None`` in the call to :meth:`boxtree.TreeBuilder.__call__`)

* one where a distinction between sources and targets is made.
  (``targets is not None`` in the call to :meth:`boxtree.TreeBuilder.__call__`)

* one where a distinction between sources and targets is made,
  and where sources and/or targets are considered to have an extent, given by an
  :math:`l^\infty` radius.
  (``targets is not None`` and ``source_radii is not None or target_radii is
  not None`` in the call to :meth:`boxtree.TreeBuilder.__call__`)

  If sources have an extent, it is possible to 'link' each source with a number
  of point sources. For this case, it is important to internalize this bit of
  terminology: A *source* may consist of multiple *point sources*.

.. _extent:

Sources and targets with extent
-------------------------------

:attr:`Tree.source_radii` and :attr:`Tree.target_radii` specify the
radii of of :math:`l^\infty` 'circles' (that is, squares) centered at
:attr:`Tree.sources` and :attr:`Tree.targets` that contain the entire
extent of that source or target.

:mod:`boxtree.traversal` guarantees that, in generating traversals, all
interactions to targets within the source extent and from sources within the
target extent will invoke special (usually direct, non-multipole) evaluation.

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

:attr:`boxtree.tree.TreeWithLinkedPointSources.user_point_source_ids` helps
translate point source arrays into tree order for processing.

.. _csr:

CSR-like interaction list storage
---------------------------------

Many list-like data structures in :mod:`boxtree` consists of
two arrays, one whose name ends in ``_starts``, and another whose
name ends in ``_lists``. For example,
suppose we would like to find the colleagues of box #17 using
:attr:`boxtree.traversal.FMMTraversalInfo.colleagues_starts`
and
:attr:`boxtree.traversal.FMMTraversalInfo.colleagues_lists`.

The following snippet of code achieves this::

    ibox = 17
    start, end = colleagues_starts[ibox:ibox+2]
    ibox_colleagues = colleagues_lists[start:end]

This indexing scheme has the following properties:

* If the underlying indexing array (say the list of all boxes) has *n* entries,
  then the ``_starts`` array has *n+1* entries. The very last entry determines
  the length of the last list.

* The lists in ``_lists`` are stored contiguously. The start of the next list
  is automatically the end of the previous one.
"""

# vim: filetype=pyopencl:fdm=marker
