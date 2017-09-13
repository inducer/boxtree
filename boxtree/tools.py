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
import pyopencl.array  # noqa
from pyopencl.tools import dtype_to_c_struct
from mako.template import Template
from pytools.obj_array import make_obj_array


AXIS_NAMES = ("x", "y", "z", "w")


def padded_bin(i, l):
    """Format *i* as binary number, pad it to length *l*."""

    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s


# NOTE: Order of positional args should match GappyCopyAndMapKernel.__call__()
def realloc_array(queue, allocator, new_shape, ary, zero_fill=False, wait_for=[]):
    if zero_fill:
        array_maker = cl.array.zeros
    else:
        array_maker = cl.array.empty

    new_ary = array_maker(queue, shape=new_shape, dtype=ary.dtype,
                          allocator=allocator)

    evt = cl.enqueue_copy(queue, new_ary.data, ary.data, byte_count=ary.nbytes,
                          wait_for=wait_for + new_ary.events)

    return new_ary, evt


def reverse_index_array(indices, target_size=None, result_fill_value=None,
        queue=None):
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
    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=seed)

    return make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])


def make_surface_particle_array(queue, nparticles, dims, dtype, seed=15):
    import loopy as lp

    if dims == 2:
        def get_2d_knl(dtype):
            knl = lp.make_kernel(
                "{[i]: 0<=i<n}",
                """
                    <> phi = 2*M_PI/n * i
                    x[i] = 0.5* (3*cos(phi) + 2*sin(3*phi))
                    y[i] = 0.5* (1*sin(phi) + 1.5*sin(2*phi))
                    """,
                [
                    lp.GlobalArg("x,y", dtype, shape=lp.auto),
                    lp.ValueArg("n", np.int32),
                    ],
                name="make_surface_dist")

            knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_2d_knl(dtype)(queue, n=nparticles)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    elif dims == 3:
        n = int(nparticles**0.5)

        def get_3d_knl(dtype):
            knl = lp.make_kernel(
                "{[i,j]: 0<=i,j<n}",
                """
                    <> phi = 2*M_PI/n * i
                    <> theta = 2*M_PI/n * j
                    x[i,j] = 5*cos(phi) * (3 + cos(theta))
                    y[i,j] = 5*sin(phi) * (3 + cos(theta))
                    z[i,j] = 5*sin(theta)
                    """,
                [
                    lp.GlobalArg("x,y,z,", dtype, shape=lp.auto),
                    lp.ValueArg("n", np.int32),
                    ])

            knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_3d_knl(dtype)(queue, n=n)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    else:
        raise NotImplementedError


def make_uniform_particle_array(queue, nparticles, dims, dtype, seed=15):
    import loopy as lp

    if dims == 2:
        n = int(nparticles**0.5)

        def get_2d_knl(dtype):
            knl = lp.make_kernel(
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
                    lp.GlobalArg("x,y", dtype, shape=lp.auto),
                    lp.ValueArg("n", np.int32),
                    ], assumptions="n>0")

            knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_2d_knl(dtype)(queue, n=n)

        result = [x.ravel() for x in result]

        return make_obj_array(result)
    elif dims == 3:
        n = int(nparticles**(1/3))

        def get_3d_knl(dtype):
            knl = lp.make_kernel(
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
                    lp.GlobalArg("x,y,z", dtype, shape=lp.auto),
                    lp.ValueArg("n", np.int32),
                    ], assumptions="n>0")

            knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "k", 16, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_3d_knl(dtype)(queue, n=n)

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

class DeviceDataRecord(Record):
    """A record of array-type data. Some of this data may live in
    :class:`pyopencl.array.Array` objects. :meth:`get` can then be
    called to convert all these device arrays into :mod:`numpy.ndarray`
    instances on the host.
    """

    def _transform_arrays(self, f):
        result = {}

        def transform_val(val):
            from pyopencl.algorithm import BuiltList
            if isinstance(val, np.ndarray) and val.dtype == object:
                from pytools.obj_array import with_object_array_or_scalar
                return with_object_array_or_scalar(f, val)
            elif isinstance(val, list):
                return [transform_val(i) for i in val]
            elif isinstance(val, BuiltList):
                return BuiltList(
                        count=val.count,
                        starts=f(val.starts),
                        lists=f(val.lists))
            else:
                return f(val)

        for field_name in self.__class__.fields:
            try:
                attr = getattr(self, field_name)
            except AttributeError:
                pass
            else:
                result[field_name] = transform_val(attr)

        return self.copy(**result)

    def get(self, queue, **kwargs):
        """Return a copy of `self` in which all data lives on the host, i.e.
        all :class:`pyopencl.array.Array` objects are replaced by corresponding
        :class:`numpy.ndarray` instances on the host.
        """

        def try_get(attr):
            try:
                get_meth = attr.get
            except AttributeError:
                return attr

            return get_meth(queue=queue, **kwargs)

        return self._transform_arrays(try_get)

    def with_queue(self, queue):
        """Return a copy of `self` in
        all :class:`pyopencl.array.Array` objects are assigned to
        :class:`pyopencl.CommandQueue` *queue*.
        """

        def try_with_queue(attr):
            if isinstance(attr, cl.array.Array):
                attr.finish()

            try:
                wq_meth = attr.with_queue
            except AttributeError:
                return attr

            ary = wq_meth(queue)
            return ary

        return self._transform_arrays(try_with_queue)

# }}}


# {{{ type mangling

def get_type_moniker(dtype):
    return "%s%d" % (dtype.kind, dtype.itemsize)

# }}}


# {{{ gappy-copy-and-map kernel

GAPPY_COPY_TPL = Template(r"""//CL//

    typedef ${dtype_to_ctype(dtype)} value_t;

    %if from_indices:
        value_t val = input_ary[from_indices[i]];
    %else:
        value_t val = input_ary[i];
    %endif

    // Optionally, noodle values through a lookup table.
    %if map_values:
        val = value_map[val];
    %endif

    %if to_indices:
        output_ary[to_indices[i]] = val;
    %else:
        output_ary[i] = val;
    %endif

""", strict_undefined=True)


class GappyCopyAndMapKernel:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def _get_kernel(self, dtype, src_index_dtype, dst_index_dtype,
                    have_src_indices, have_dst_indices, map_values):
        from pyopencl.tools import VectorArg

        args = [
                VectorArg(dtype, "input_ary", with_offset=True),
                VectorArg(dtype, "output_ary", with_offset=True),
               ]

        if have_src_indices:
            args.append(VectorArg(src_index_dtype, "from_indices", with_offset=True))

        if have_dst_indices:
            args.append(VectorArg(dst_index_dtype, "to_indices", with_offset=True))

        if map_values:
            args.append(VectorArg(dtype, "value_map", with_offset=True))

        from pyopencl.tools import dtype_to_ctype
        src = GAPPY_COPY_TPL.render(
                dtype=dtype,
                dtype_to_ctype=dtype_to_ctype,
                from_dtype=src_index_dtype,
                to_dtype=dst_index_dtype,
                from_indices=have_src_indices,
                to_indices=have_dst_indices,
                map_values=map_values)

        from pyopencl.elementwise import ElementwiseKernel
        return ElementwiseKernel(self.context,
                args, str(src),
                preamble=dtype_to_c_struct(self.context.devices[0], dtype),
                name="gappy_copy_and_map")

    # NOTE: Order of positional args should match realloc_array()
    def __call__(self, queue, allocator, new_shape, ary, src_indices=None,
                 dst_indices=None, map_values=None, zero_fill=False,
                 wait_for=None, range=None, debug=False):
        """Compresses box info arrays after empty leaf pruning and, optionally,
        maps old box IDs to new box IDs (if the array being operated on contains
        box IDs).
        """

        have_src_indices = src_indices is not None
        have_dst_indices = dst_indices is not None
        have_map_values = map_values is not None

        if not (have_src_indices or have_dst_indices):
            raise ValueError("must specify at least one of src or dest indices")

        if range is None:
            if have_src_indices and have_dst_indices:
                raise ValueError(
                    "must supply range when passing both src and dest indices")
            elif have_src_indices:
                range = slice(src_indices.shape[0])
                if debug:
                    assert int(cl.array.max(src_indices).get()) < len(ary)
            elif have_dst_indices:
                range = slice(dst_indices.shape[0])
                if debug:
                    assert int(cl.array.max(dst_indices).get()) < new_shape

        if zero_fill:
            array_maker = cl.array.zeros
        else:
            array_maker = cl.array.empty

        result = array_maker(queue, new_shape, ary.dtype, allocator=allocator)

        kernel = self._get_kernel(ary.dtype,
                                  src_indices.dtype if have_src_indices else None,
                                  dst_indices.dtype if have_dst_indices else None,
                                  have_src_indices,
                                  have_dst_indices,
                                  have_map_values)

        args = (ary, result)
        args += (src_indices,) if have_src_indices else ()
        args += (dst_indices,) if have_dst_indices else ()
        args += (map_values,) if have_map_values else ()

        evt = kernel(*args, queue=queue, range=range, wait_for=wait_for)

        return result, evt

# }}}


# {{{ map values through table

from pyopencl.elementwise import ElementwiseTemplate


MAP_VALUES_TPL = ElementwiseTemplate(
    arguments="""//CL//
        dst_value_t *dst,
        src_value_t *src,
        dst_value_t *map_values
        """,
    operation=r"""//CL//
        dst[i] = map_values[src[i]];
        """,
    name="map_values")


class MapValuesKernel(object):

    def __init__(self, context):
        self.context = context

    @memoize_method
    def _get_kernel(self, dst_dtype, src_dtype):
        type_aliases = (
            ("src_value_t", src_dtype),
            ("dst_value_t", dst_dtype)
            )

        return MAP_VALUES_TPL.build(self.context, type_aliases)

    def __call__(self, map_values, src, dst=None):
        """
        Map the entries of the array `src` through the table `map_values`.
        """
        if dst is None:
            dst = src

        kernel = self._get_kernel(dst.dtype, src.dtype)
        evt = kernel(dst, src, map_values)

        return dst, evt

# }}}


# {{{ binary search

from mako.template import Template


BINARY_SEARCH_TEMPLATE = Template("""
inline size_t bsearch(__global ${idx_t} *starts, size_t len, ${idx_t} val)
{
    size_t l_idx = 0, r_idx = len - 1, my_idx;
    for (;;)
    {
        my_idx = (l_idx + r_idx) / 2;

        if (starts[my_idx] <= val && val < starts[my_idx + 1])
        {
            return my_idx;
        }

        if (starts[my_idx] > val)
        {
            r_idx = my_idx - 1;
        }
        else
        {
            l_idx = my_idx + 1;
        }
    }
}
""")


class InlineBinarySearch(object):

    def __init__(self, idx_t):
        self.idx_t = idx_t

    @memoize_method
    def __str__(self):
        return BINARY_SEARCH_TEMPLATE.render(idx_t=self.idx_t)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
