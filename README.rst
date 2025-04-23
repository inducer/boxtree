boxtree: Quad/Octrees, FMM Traversals, Geometric Queries
========================================================

.. image:: https://gitlab.tiker.net/inducer/boxtree/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/boxtree/commits/main
.. image:: https://github.com/inducer/boxtree/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/boxtree/actions/workflows/ci.yml
.. image:: https://badge.fury.io/py/boxtree.svg
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/boxtree/
.. image:: https://zenodo.org/badge/7193697.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/badge/latestdoi/7193697

``boxtree`` is a package that, given some point locations in two or three
dimensions, sorts them into an adaptive quad/octree of boxes, efficiently, in
parallel, using `PyOpenCL <https://mathema.tician.de/software/pyopencl>`__.

It can also generate traversal lists needed for adaptive fast multipole methods
and related algorithms and tree-based look-up tables for geometric proximity.

``boxtree`` is under the MIT license.

Resources:

* `Documentation <https://documen.tician.de/boxtree>`__
* `PyPI package <https://pypi.org/project/boxtree>`__
* `Source Code (GitHub) <https://github.com/inducer/boxtree>`__
