from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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




import numpy as np
from pytools import Record, memoize_method
import pyopencl as cl
import pyopencl.array
from mako.template import Template





AXIS_NAMES = ("x", "y", "z", "w")




def padded_bin(i, l):
    """Format *i* as binary number, pad it to length *l*."""

    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s



def realloc_array(ary, new_shape, zero_fill, queue):
    new_ary = cl.array.empty(queue, shape=new_shape, dtype=ary.dtype,
            allocator=ary.allocator)
    if zero_fill:
        new_ary.fill(0)
    cl.enqueue_copy(queue, new_ary.data, ary.data, byte_count=ary.nbytes)
    return new_ary


def make_particle_array(queue, nparticles, dims, dtype, seed=15):
    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=seed)

    from pytools.obj_array import make_obj_array
    return make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])




def particle_array_to_host(parray):
    return np.array([x.get() for x in parray], order="F").T




# {{{ host/device data storage

class FromDeviceGettableRecord(Record):
    """A record of array-type data. Some of this data may live in
    :class:`pyopencl.array.Array` objects. :meth:`get` can then be
    called to convert all these device arrays into :mod:`numpy.ndarray`
    instances on the host.
    """

    def get(self):
        """Return a copy of `self` in which all data lives on the host."""

        result = {}
        for field_name in self.__class__.fields:
            try:
                attr = getattr(self, field_name)
            except AttributeError:
                pass
            else:
                if isinstance(attr, np.ndarray) and attr.dtype == object:
                    from pytools.obj_array import with_object_array_or_scalar
                    result[field_name] = with_object_array_or_scalar(
                            lambda x: x.get(), attr)
                else:
                    try:
                        get_meth = attr.get
                    except AttributeError:
                        continue

                    result[field_name] = get_meth()

        return self.copy(**result)

# }}}

# {{{ type mangling

def get_type_moniker(dtype):
    return "%s%d" % (dtype.kind, dtype.itemsize)

# }}}

# {{{ gappy-copy-and-map kernel

GAPPY_COPY_TPL =  Template(r"""//CL//

    typedef ${dtype_to_ctype(dtype)} value_t;

    value_t val = input_ary[from_indices[i]];

    // Optionally, noodle values through a lookup table.
    %if map_values:
        val = value_map[val];
    %endif

    output_ary[i] = val;

""", strict_undefined=True)


class GappyCopyAndMapKernel:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def _get_kernel(self, dtype, src_index_dtype, map_values=False):
        from pyopencl.tools import VectorArg

        args = [
                VectorArg(dtype, "input_ary"),
                VectorArg(dtype, "output_ary"),
                VectorArg(src_index_dtype, "from_indices")
                ]

        if map_values:
            args.append(VectorArg(dtype, "value_map"))

        from pyopencl.tools import dtype_to_ctype
        src = GAPPY_COPY_TPL.render(
                dtype=dtype,
                dtype_to_ctype=dtype_to_ctype,
                map_values=map_values)

        from pyopencl.elementwise import ElementwiseKernel
        return ElementwiseKernel(self.context,
                args, str(src), name="gappy_copy_and_map")

    def __call__(self, queue, allocator, new_size,
            src_indices, ary, map_values=None):
        """Compresses box info arrays after empty leaf pruning and, optionally,
        maps old box IDs to new box IDs (if the array being operated on contains
        box IDs).
        """

        assert len(ary) >= new_size

        result = cl.array.empty(queue, new_size, ary.dtype, allocator=allocator)

        kernel = self._get_kernel(ary.dtype, src_indices.dtype,
                map_values=map_values is not None)

        args = (ary, result, src_indices)
        if map_values is not None:
            args += (map_values,)

        kernel(*args, queue=queue, range=slice(new_size))

        return result

# }}}

# vim: foldmethod=marker:filetype=pyopencl
