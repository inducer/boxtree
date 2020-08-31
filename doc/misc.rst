Installation
============

This command should install :mod:`boxtree`::

    pip install boxtree

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, download the source, unpack it,
and say::

    python setup.py install

In addition, you need to have PyOpenCL installed. See the
`PyOpenCL Wiki <http://wiki.tiker.net/PyOpenCL/Installation>`_
for instructions.

User-visible Changes
====================

Version 2019.1
--------------

.. note::

    This version is currently under development. You can get snapshots from
    boxtree's `git repository <https://github.com/inducer/boxtree>`_

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

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <http://wiki.tiker.net/BoxTree/FrequentlyAskedQuestions>`_.

Acknowledgments
===============

Andreas Klöckner's work on :mod:`pytential` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.

Cross-References to Other Documentation
=======================================

.. currentmodule:: numpy

.. class:: int8

    See :class:`numpy.generic`.

.. class:: int32

    See :class:`numpy.generic`.
