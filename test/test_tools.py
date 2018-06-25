from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner \
                 Copyright (C) 2017 Matt Wala \
                 Copyright (C) 2018 Hao Gao"

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
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7, 8, 16, 17])
def test_allreduce_comm_pattern(p):
    from boxtree.tools import AllReduceCommPattern

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


@pytest.mark.parametrize("order", "CF")
def test_masked_matrix_compression(ctx_getter, order):
    cl_context = ctx_getter()

    from boxtree.tools import MaskCompressorKernel
    matcompr = MaskCompressorKernel(cl_context)

    n = 40
    m = 10

    np.random.seed(15)

    arr = (np.random.rand(n, m) > 0.5).astype(np.int8).copy(order=order)

    with cl.CommandQueue(cl_context) as q:
        d_arr = cl.array.Array(q, (n, m), arr.dtype, order=order)
        d_arr[:] = arr
        arr_starts, arr_lists, evt = matcompr(q, d_arr)
        cl.wait_for_events([evt])
        arr_starts = arr_starts.get(q)
        arr_lists = arr_lists.get(q)

    for i in range(n):
        items = arr_lists[arr_starts[i]:arr_starts[i+1]]
        assert set(items) == set(arr[i].nonzero()[0])


def test_masked_list_compression(ctx_getter):
    cl_context = ctx_getter()

    from boxtree.tools import MaskCompressorKernel
    listcompr = MaskCompressorKernel(cl_context)

    n = 20

    np.random.seed(15)

    arr = (np.random.rand(n) > 0.5).astype(np.int8)

    with cl.CommandQueue(cl_context) as q:
        d_arr = cl.array.to_device(q, arr)
        arr_list, evt = listcompr(q, d_arr)
        cl.wait_for_events([evt])
        arr_list = arr_list.get(q)

    assert set(arr_list) == set(arr.nonzero()[0])


def test_device_record():
    from boxtree.tools import DeviceDataRecord

    array = np.arange(60).reshape((3, 4, 5))

    obj_array = np.empty((3,), dtype=object)
    for i in range(3):
        obj_array[i] = np.arange((i + 1) * 40).reshape(5, i + 1, 8)

    record = DeviceDataRecord(
        array=array,
        obj_array=obj_array
    )

    ctx = cl.create_some_context()

    with cl.CommandQueue(ctx) as queue:
        record_dev = record.to_device(queue)
        record_host = record_dev.get(queue)

        assert np.array_equal(record_host.array, record.array)

        for i in range(3):
            assert np.array_equal(record_host.obj_array[i], record.obj_array[i])


# You can test individual routines by typing
# $ python test_tools.py 'test_routine'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
