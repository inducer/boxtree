boxtree: Quad/Octrees, FMM Traversals, Geometric Queries
========================================================

.. image:: https://gitlab.tiker.net/inducer/boxtree/badges/master/pipeline.svg
   :target: https://gitlab.tiker.net/inducer/boxtree/commits/master
.. image:: https://badge.fury.io/py/boxtree.png
    :target: http://pypi.python.org/pypi/boxtree

boxtree is a package that, given some point locations in two or three
dimensions, sorts them into an adaptive quad/octree of boxes, efficiently, in
parallel, using `PyOpenCL <http://mathema.tician.de/software/pyopencl>`_.

It can also generate traversal lists needed for adaptive fast multipole methods
and related algorithms and tree-based look-up tables for geometric proximity.

boxtree is under the MIT license.

Resources:

* `documentation <http://documen.tician.de/boxtree>`_
* `wiki home page <http://wiki.tiker.net/BoxTree>`_
* `source code via git <https://github.com/inducer/boxtree>`_
