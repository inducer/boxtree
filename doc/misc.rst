Installation
============

This command should install the latest release of :mod:`boxtree` directly from PyPI::

    pip install boxtree

You may need to run this with :command:`sudo` if you are not in a virtual environment
(not recommended). If you don't already have `pip <https://pypi.org/project/pip>`__,
run this beforehand::

    python -m ensurepip

For a more manual installation, download the source, unpack it, and run::

    pip install .

This should also install all the required dependencies (see ``pyproject.toml``
for a complete list). The main one is PyOpenCL, which has extensive installation
instructions on the `PyOpenCL Wiki <https://wiki.tiker.net/PyOpenCL/Installation>`__.

For development, you may want to install in `editable mode
<https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`__::

    pip install --no-build-isolation --editable .[test]

User-visible Changes
====================

.. note::

    You can get snapshots of in-development versions from
    :mod:`boxtree`'s `git repository <https://github.com/inducer/boxtree>`_.

Version 2024.1
--------------

* Use :mod:`arraycontext` as the main array abstraction (over :mod:`pyopencl`
  only at the moment). This changed the API of many functions and classes,
  since most of them now take an :class:`~arraycontext.ArrayContext` instead
  of a :class:`pyopencl.Context`.
* Remove (temporarily) cost model support. This removed the *timing_data*
  parameter and return values from the FMM driver.
* Removed *DeviceDataRecord* in favour of array containers from :mod:`arraycontext`.

Version 2019.1
--------------

* Faster M2Ls in the FMMLIB backend using precomputed rotation matrices.  This
  change adds an optional *rotation_data* parameter to the FMMLIB geometry wrangler
  constructor.

Version 2018.2
--------------

* Changed index style of the *from_sep_close_bigger_starts* interaction list.

Version 2018.1
--------------

* Added *timing_data* parameter to FMM driver.

Version 2013.1
--------------

* Initial release.

.. _license:

License
=======

Boxtree is licensed to you under the MIT/X Consortium license:

Copyright (c) 2012-13 Andreas Klöckner and contributors.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Acknowledgments
===============

Work on meshmode was supported in part by

* the US National Science Foundation under grant numbers DMS-1418961,
  DMS-1654756, SHF-1911019, and OAC-1931577.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.

The views and opinions expressed herein do not necessarily reflect those of the
funding agencies.

Cross-References to Other Documentation
=======================================

.. currentmodule:: numpy

.. class:: int8

    See :class:`numpy.generic`.

.. class:: int32

    See :class:`numpy.generic`.
