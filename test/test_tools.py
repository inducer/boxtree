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

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from boxtree.array_context import _acf  # noqa: F401
from boxtree.array_context import PytestPyOpenCLArrayContextFactory
from boxtree.tools import (  # noqa: F401
    make_normal_particle_array as p_normal, make_surface_particle_array as p_surface,
    make_uniform_particle_array as p_uniform)


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_allreduce_comm_pattern

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

# }}}


# {{{ test_masked_matrix_compression

@pytest.mark.parametrize("order", ["C", "F"])
def test_masked_matrix_compression(actx_factory, order):
    actx = actx_factory()

    from boxtree.tools import MaskCompressorKernel
    matcompr = MaskCompressorKernel(actx.context)

    n = 40
    m = 10

    rng = np.random.default_rng(15)
    arr = (rng.random((n, m)) > 0.5).astype(np.int8).copy(order=order)
    d_arr = actx.from_numpy(arr)

    arr_starts, arr_lists, evt = matcompr(actx.queue, d_arr)
    arr_starts = actx.to_numpy(arr_starts)
    arr_lists = actx.to_numpy(arr_lists)

    for i in range(n):
        items = arr_lists[arr_starts[i]:arr_starts[i+1]]
        assert set(items) == set(arr[i].nonzero()[0])

# }}}


# {{{ test_masked_list_compression

def test_masked_list_compression(actx_factory):
    actx = actx_factory()

    from boxtree.tools import MaskCompressorKernel
    listcompr = MaskCompressorKernel(actx.context)

    n = 20

    np.random.seed(15)

    arr = (np.random.rand(n) > 0.5).astype(np.int8)
    d_arr = actx.from_numpy(arr)

    arr_list, evt = listcompr(actx.queue, d_arr)
    arr_list = actx.to_numpy(arr_list)

    assert set(arr_list) == set(arr.nonzero()[0])

# }}}


# {{{ test_device_record

def test_device_record(actx_factory):
    actx = actx_factory()

    from boxtree.tools import DeviceDataRecord
    array = np.arange(60).reshape((3, 4, 5))

    obj_array = np.empty((3,), dtype=object)
    for i in range(3):
        obj_array[i] = np.arange((i + 1) * 40).reshape(5, i + 1, 8)

    record = DeviceDataRecord(
        array=array,
        obj_array=obj_array
    )

    record_dev = record.to_device(actx.queue)
    record_host = record_dev.get(actx.queue)

    assert np.array_equal(record_host.array, record.array)

    for i in range(3):
        assert np.array_equal(record_host.obj_array[i], record.obj_array[i])

# }}}


# {{{ test_particle_array

@pytest.mark.parametrize("array_factory", (p_normal, p_surface, p_uniform))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_particle_array(actx_factory, array_factory, dim, dtype):
    actx = actx_factory()

    particles = array_factory(actx.queue, 1000, dim, dtype)
    assert len(particles) == dim
    assert all(len(particles[0]) == len(axis) for axis in particles)

# }}}


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
