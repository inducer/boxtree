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

from boxtree.tree import Tree, box_flags_enum
from boxtree.tree_build import TreeBuilder

__all__ = ["Tree", "TreeBuilder", "box_flags_enum"]

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

  In this mode, it is possible to 'link' each source with a number of point
  sources. It is important to internalize this bit of terminology here:
  A *source* may consist of multiple *point sources*.

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

# vim: filetype=pyopencl:fdm=marker
