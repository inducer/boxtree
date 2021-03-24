import numpy as np
import pyopencl as cl

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner \
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


import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


from boxtree.tools import (  # noqa: F401
        make_normal_particle_array as p_normal,
        make_surface_particle_array as p_surface,
        make_uniform_particle_array as p_uniform)


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


@pytest.mark.parametrize("array_factory", (p_normal, p_surface, p_uniform))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_particle_array(ctx_factory, array_factory, dim, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    particles = array_factory(queue, 1000, dim, dtype)
    assert len(particles) == dim
    assert all(len(particles[0]) == len(axis) for axis in particles)


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
