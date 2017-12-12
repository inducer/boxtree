from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Matt Wala"

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
import pytest
import sys

import numpy as np

import pyopencl as cl
import pyopencl.array  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7, 8, 16, 17])
def test_allreduce_comm_pattern(p):
    from boxtree.distributed import AllReduceCommPattern

    # This models the parallel allreduce communication pattern.

    # processor -> communication pattern of the processor
    patterns = [AllReduceCommPattern(i, p) for i in range(p)]
    # processor -> list of data items on the processor
    data = [[i] for i in range(p)]
    from copy import deepcopy

    while not all(pat.done() for pat in patterns):
        new_data = deepcopy(data)

        for i in range(p):
            if patterns[i].done():
                for pat in patterns:
                    if not pat.done():
                        assert i not in pat.sources() | pat.sinks()
                continue

            # Check sources / sinks match up
            for s in patterns[i].sinks():
                assert i in patterns[s].sources()

            for s in patterns[i].sources():
                assert i in patterns[s].sinks()

            # Send / recv data
            for s in patterns[i].sinks():
                new_data[s].extend(data[i])

        for pat in patterns:
            if not pat.done():
                pat.advance()
        data = new_data

    for item in data:
        assert len(item) == p
        assert set(item) == set(range(p))


def test_matrix_compressor(ctx_getter):
    cl_context = ctx_getter()

    from boxtree.tools import MatrixCompressorKernel
    matcompr = MatrixCompressorKernel(cl_context)

    n = 40
    m = 10

    np.random.seed(15)

    arr = (np.random.rand(n, m) > 0.5).astype(np.int8)

    with cl.CommandQueue(cl_context) as q:
        d_arr = cl.array.to_device(q, arr)
        arr_starts, arr_lists, evt = matcompr(q, d_arr)
        cl.wait_for_events([evt])
        arr_starts = arr_starts.get(q)
        arr_lists = arr_lists.get(q)

    for i in range(n):
        items = arr_lists[arr_starts[i]:arr_starts[i+1]]
        assert set(items) == set(arr[i].nonzero()[0])


# You can test individual routines by typing
# $ python test_tree.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        import py.test
        py.test.cmdline.main([__file__])

# vim: fdm=marker
