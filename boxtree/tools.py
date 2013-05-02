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
from pyopencl.tools import first_arg_dependent_memoize_nested
from mako.template import Template
from pytools.obj_array import make_obj_array





AXIS_NAMES = ("x", "y", "z", "w")




def padded_bin(i, l):
    """Format *i* as binary number, pad it to length *l*."""

    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s



def realloc_array(ary, new_shape, zero_fill, queue, wait_for):
    new_ary = cl.array.empty(queue, shape=new_shape, dtype=ary.dtype,
            allocator=ary.allocator)
    if zero_fill:
        new_ary.fill(0, wait_for=wait_for)
        wait_for = new_ary.events

    evt = cl.enqueue_copy(queue, new_ary.data, ary.data, byte_count=ary.nbytes,
            wait_for=wait_for)

    return new_ary, evt



def reverse_index_array(indices, target_size=None, result_fill_value=None, queue=None):
    """For an array of *indices*, return a new array *result* that satisfies
    ``result[indices] == arange(len(indices))

    :arg target_n: The length of the result, or *None* if the result is to
        have the same length as *indices*.
    :arg result_fill_value: If not *None*, fill *result* with this value
        prior to storing reversed indices.
    """

    queue = queue or indices.queue

    if target_size is None:
        target_size = len(indices)

    result = cl.array.empty(queue, target_size, indices.dtype)

    if result_fill_value is not None:
        result.fill(result_fill_value)

    cl.array.multi_put(
            [cl.array.arange(queue, len(indices), dtype=indices.dtype,
                allocator=indices.allocator)],
            indices,
            out=[result],
            queue=queue)

    return result

# {{{ particle distribution generators

def make_normal_particle_array(queue, nparticles, dims, dtype, seed=15):
    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=seed)

    return make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])

def make_surface_particle_array(queue, nparticles, dims, dtype, seed=15):
    import loopy as lp

    if dims == 2:
        @first_arg_dependent_memoize_nested
        def get_2d_knl(context, dtype):
            knl = lp.make_kernel(context.devices[0],
                "{[i]: 0<=i<n}",
                """
                    <> phi = 2*M_PI/n * i
                    x[i] = 0.5* (3*cos(phi) + 2*sin(3*phi))
                    y[i] = 0.5* (1*sin(phi) + 1.5*sin(2*phi))
                    """,
                [
                    lp.GlobalArg("x", dtype, shape="n"),
                    lp.GlobalArg("y", dtype, shape="n"),
                    lp.ValueArg("n", np.int32),
                    ])

            knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

            return lp.CompiledKernel(context, knl)

        evt, result = get_2d_knl(queue.context, dtype)(queue, n=nparticles)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    elif dims == 3:
        n = int(nparticles**0.5)

        @first_arg_dependent_memoize_nested
        def get_3d_knl(context, dtype):
            knl = lp.make_kernel(context.devices[0],
                "{[i,j]: 0<=i,j<n}",
                """
                    <> phi = 2*M_PI/n * i
                    <> theta = 2*M_PI/n * j
                    x[i,j] = 5*cos(phi) * (3 + cos(theta))
                    y[i,j] = 5*sin(phi) * (3 + cos(theta))
                    z[i,j] = 5*sin(theta)
                    """,
                [
                    lp.GlobalArg("x", dtype, shape="n,n"),
                    lp.GlobalArg("y", dtype, shape="n,n"),
                    lp.GlobalArg("z", dtype, shape="n,n"),
                    lp.ValueArg("n", np.int32),
                    ])

            knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")

            return lp.CompiledKernel(context, knl)

        evt, result = get_3d_knl(queue.context, dtype)(queue, n=n)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    else:
        raise NotImplementedError


def make_uniform_particle_array(queue, nparticles, dims, dtype, seed=15):
    import loopy as lp

    if dims == 2:
        n = int(nparticles**0.5)

        @first_arg_dependent_memoize_nested
        def get_2d_knl(context, dtype):
            knl = lp.make_kernel(context.devices[0],
                "{[i,j]: 0<=i,j<n}",
                """
                    <> xx = 4*i/(n-1)
                    <> yy = 4*j/(n-1)
                    <float64> angle = 0.3
                    <> s = sin(angle)
                    <> c = cos(angle)
                    x[i,j] = c*xx + s*yy - 2
                    y[i,j] = -s*xx + c*yy - 2
                    """,
                [
                    lp.GlobalArg("x", dtype, shape="n,n"),
                    lp.GlobalArg("y", dtype, shape="n,n"),
                    lp.ValueArg("n", np.int32),
                    ], assumptions="n>0")

            knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")

            return lp.CompiledKernel(context, knl)

        evt, result = get_2d_knl(queue.context, dtype)(queue, n=n)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    elif dims == 3:
        n = int(nparticles**(1/3))

        @first_arg_dependent_memoize_nested
        def get_3d_knl(context, dtype):
            knl = lp.make_kernel(context.devices[0],
                "{[i,j,k]: 0<=i,j,k<n}",
                """
                    <> xx = i/(n-1)
                    <> yy = j/(n-1)
                    <> zz = k/(n-1)

                    <float64> phi = 0.3
                    <> s1 = sin(phi)
                    <> c1 = cos(phi)

                    <> xxx = c1*xx + s1*yy
                    <> yyy = -s1*xx + c1*yy
                    <> zzz = zz

                    <float64> theta = 0.7
                    <> s2 = sin(theta)
                    <> c2 = cos(theta)

                    x[i,j,k] = 4 * (c2*xxx + s2*zzz) - 2
                    y[i,j,k] = 4 * yyy - 2
                    z[i,j,k] = 4 * (-s2*xxx + c2*zzz) - 2
                    """,
                [
                    lp.GlobalArg("x", dtype, shape="n,n,n"),
                    lp.GlobalArg("y", dtype, shape="n,n,n"),
                    lp.GlobalArg("z", dtype, shape="n,n,n"),
                    lp.ValueArg("n", np.int32),
                    ], assumptions="n>0")

            knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "k", 16, outer_tag="g.0", inner_tag="l.0")

            return lp.CompiledKernel(context, knl)

        evt, result = get_3d_knl(queue.context, n=n)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    else:
        raise NotImplementedError

def make_rotated_uniform_particle_array(queue, nparticles, dims, dtype, seed=15):
    raise NotImplementedError

# }}}




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
        """Return a copy of `self` in which all data lives on the host, i.e.
        all :class:`pyopencl.array.Array` objects are replaced by corresponding
        :class:`numpy.ndarray` instances on the host.
        """

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
            src_indices, ary, map_values=None, wait_for=None):
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

        evt = kernel(*args, queue=queue, range=slice(new_size), wait_for=wait_for)

        return result, evt

# }}}

# vim: foldmethod=marker:filetype=pyopencl
