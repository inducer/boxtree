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

from typing import Any, Dict

import numpy as np
from pytools import Record, memoize_method
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.cltypes as cltypes
from pyopencl.tools import dtype_to_c_struct, ScalarArg, VectorArg as _VectorArg
from mako.template import Template
from pytools.obj_array import make_obj_array
import loopy as lp

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa

from functools import partial
import sys


# Use offsets in VectorArg by default.
VectorArg = partial(_VectorArg, with_offset=True)

AXIS_NAMES = ("x", "y", "z", "w")


def padded_bin(i, nbits):
    """Format *i* as binary number, pad it to length *nbits*."""
    return bin(i)[2:].rjust(nbits, "0")


# NOTE: Order of positional args should match GappyCopyAndMapKernel.__call__()
def realloc_array(queue, allocator, new_shape, ary, zero_fill=False, wait_for=None):
    if wait_for is None:
        wait_for = []

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
    if dims == 2:
        def get_2d_knl(dtype):
            knl = lp.make_kernel(
                "{[i]: 0<=i<n}",
                """
                    for i
                        <> phi = 2*M_PI/n * i
                        x[i] = 0.5* (3*cos(phi) + 2*sin(3*phi))
                        y[i] = 0.5* (1*sin(phi) + 1.5*sin(2*phi))
                    end
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
                    for i,j
                        <> phi = 2*M_PI/n * i
                        <> theta = 2*M_PI/n * j
                        x[i,j] = 5*cos(phi) * (3 + cos(theta))
                        y[i,j] = 5*sin(phi) * (3 + cos(theta))
                        z[i,j] = 5*sin(theta)
                    end
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
    if dims == 2:
        n = int(nparticles**0.5)

        def get_2d_knl(dtype):
            knl = lp.make_kernel(
                "{[i,j]: 0<=i,j<n}",
                """
                    for i,j
                        <> xx = 4*i/(n-1)
                        <> yy = 4*j/(n-1)
                        <float64> angle = 0.3
                        <> s = sin(angle)
                        <> c = cos(angle)
                        x[i,j] = c*xx + s*yy - 2
                        y[i,j] = -s*xx + c*yy - 2
                    end
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
                    for i,j,k
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
                    end
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

    def _transform_arrays(self, f, exclude_fields=frozenset()):
        result = {}

        def transform_val(val):
            from pyopencl.algorithm import BuiltList
            if isinstance(val, np.ndarray) and val.dtype == object:
                from pytools.obj_array import obj_array_vectorize
                return obj_array_vectorize(f, val)
            elif isinstance(val, list):
                return [transform_val(i) for i in val]
            elif isinstance(val, BuiltList):
                transformed_list = {}
                for field in val.__dict__:
                    if field != "count" and not field.startswith("_"):
                        transformed_list[field] = f(getattr(val, field))
                return BuiltList(count=val.count, **transformed_list)
            else:
                return f(val)

        for field_name in self.__class__.fields:
            if field_name in exclude_fields:
                continue

            try:
                attr = getattr(self, field_name)
            except AttributeError:
                pass
            else:
                result[field_name] = transform_val(attr)

        return self.copy(**result)

    def get(self, queue, **kwargs):
        """Return a copy of `self` in which all data lives on the host, i.e.
        all :class:`pyopencl.array.Array` and `ImmutableHostDeviceArray` objects are
        replaced by corresponding :class:`numpy.ndarray` instances on the host.
        """
        def try_get(attr):
            if isinstance(attr, ImmutableHostDeviceArray):
                return attr.host

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

    def to_device(self, queue, exclude_fields=frozenset()):
        """Return a copy of `self` in all :class:`numpy.ndarray` arrays are
        transferred to device memory as :class:`pyopencl.array.Array` objects.

        :arg exclude_fields: a :class:`frozenset` containing fields excluding from
            transferring to the device memory.
        """

        def _to_device(attr):
            if isinstance(attr, np.ndarray):
                return cl.array.to_device(queue, attr).with_queue(None)
            elif isinstance(attr, ImmutableHostDeviceArray):
                return attr.device
            elif isinstance(attr, DeviceDataRecord):
                return attr.to_device(queue)
            else:
                return attr

        return self._transform_arrays(_to_device, exclude_fields=exclude_fields)

    def to_host_device_array(self, queue, exclude_fields=frozenset()):
        """Return a copy of `self` where all device and host arrays are transformed
        to `ImmutableHostDeviceArray` objects.

        :arg exclude_fields: a :class:`frozenset` containing fields excluding from
            transformed to `ImmutableHostDeviceArray`.
        """
        def _to_host_device_array(attr):
            if isinstance(attr, (np.ndarray, cl.array.Array)):
                return ImmutableHostDeviceArray(queue, attr)
            elif isinstance(attr, DeviceDataRecord):
                return attr.to_host_device_array(queue)
            else:
                return attr

        return self._transform_arrays(
            _to_host_device_array, exclude_fields=exclude_fields
        )

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
        from boxtree.tools import VectorArg

        args = [
                VectorArg(dtype, "input_ary"),
                VectorArg(dtype, "output_ary"),
               ]

        if have_src_indices:
            args.append(VectorArg(src_index_dtype, "from_indices"))

        if have_dst_indices:
            args.append(VectorArg(dst_index_dtype, "to_indices"))

        if map_values:
            args.append(VectorArg(dtype, "value_map"))

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


class MapValuesKernel:

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
/*
 * Returns the largest value of i such that arr[i] <= val, or (size_t) -1 if val
 * is less than all values.
 */
inline size_t bsearch(
    __global const ${elem_t} *arr,
    size_t len,
    const ${elem_t} val)
{
    if (val < arr[0])
    {
        return -1;
    }

    size_t l = 0, r = len, i;

    while (1)
    {
        i = l + (r - l) / 2;

        if (arr[i] <= val && (i == len - 1 || val < arr[i + 1]))
        {
            return i;
        }

        if (arr[i] <= val)
        {
            l = i;
        }
        else
        {
            r = i;
        }
    }
}
""")


class InlineBinarySearch:

    def __init__(self, elem_type_name):
        self.render_vars = {"elem_t": elem_type_name}

    @memoize_method
    def __str__(self):
        return BINARY_SEARCH_TEMPLATE.render(**self.render_vars)

# }}}


# {{{ compress a masked array into a list / list of lists


MASK_LIST_COMPRESSOR_BODY = r"""
void generate(LIST_ARG_DECL USER_ARG_DECL index_type i)
{
    if (mask[i])
    {
        APPEND_output(i);
    }
}
"""


MASK_MATRIX_COMPRESSOR_BODY = r"""
void generate(LIST_ARG_DECL USER_ARG_DECL index_type i)
{
    for (int j = 0; j < ncols; ++j)
    {
        if (mask[outer_stride * i + j * inner_stride])
        {
            APPEND_output(j);
        }
    }
}
"""


class MaskCompressorKernel:
    """
    .. automethod:: __call__
    """
    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_list_compressor_kernel(self, mask_dtype, list_dtype):
        from pyopencl.algorithm import ListOfListsBuilder

        return ListOfListsBuilder(
                self.context,
                [("output", list_dtype)],
                MASK_LIST_COMPRESSOR_BODY,
                [
                    _VectorArg(mask_dtype, "mask"),
                ],
                name_prefix="compress_list")

    @memoize_method
    def get_matrix_compressor_kernel(self, mask_dtype, list_dtype):
        from pyopencl.algorithm import ListOfListsBuilder

        return ListOfListsBuilder(
                self.context,
                [("output", list_dtype)],
                MASK_MATRIX_COMPRESSOR_BODY,
                [
                    ScalarArg(np.int32, "ncols"),
                    ScalarArg(np.int32, "outer_stride"),
                    ScalarArg(np.int32, "inner_stride"),
                    _VectorArg(mask_dtype, "mask"),
                ],
                name_prefix="compress_matrix")

    def __call__(self, queue, mask, list_dtype=None):
        """Convert a mask to a list in :ref:`csr` format.

        :arg mask: Either a 1D or 2D array.
            * If *mask* is 1D, it should represent a masked list, where
              *mask[i]* is true if and only if *i* is in the list.
            * If *mask* is 2D, it should represent a list of masked lists,
              so that *mask[i,j]* is true if and only if *j* is in list *i*.

        :arg list_dtype: The dtype for the output list(s). Defaults to the mask
            dtype.

        :returns: The return value depends on the type of the input.
            * If mask* is 1D, returns a tuple *(list, evt)*.
            * If *mask* is 2D, returns a tuple *(starts, lists, event)*, as a
              :ref:`csr` list.
        """
        if list_dtype is None:
            list_dtype = mask.dtype

        if len(mask.shape) == 1:
            knl = self.get_list_compressor_kernel(mask.dtype, list_dtype)
            result, evt = knl(queue, mask.shape[0], mask.data)
            return (result["output"].lists, evt)
        elif len(mask.shape) == 2:
            # FIXME: This is efficient for small column sizes but may not be
            # for larger ones since the work is partitioned by row.
            knl = self.get_matrix_compressor_kernel(mask.dtype, list_dtype)
            size = mask.dtype.itemsize
            assert size > 0
            result, evt = knl(queue, mask.shape[0], mask.shape[1],
                              mask.strides[0] // size, mask.strides[1] // size,
                              mask.data)
            return (result["output"].starts, result["output"].lists, evt)
        else:
            raise ValueError("unsupported dimensionality")

# }}}


# {{{ Communication pattern for partial multipole expansions

class AllReduceCommPattern:
    """Describes a tree-like communication pattern for exchanging and reducing
    multipole expansions. Supports an arbitrary number of processes.

    A user must instantiate a version of this with identical *size* and varying
    *rank* on each rank. During each stage, each rank sends its contribution to
    the reduction results on ranks returned by :meth:`sinks` and listens for
    contributions from :meth:`source`. :meth:`messages` can be used for determining
    array indices whose partial results need to be sent during the current stage.
    Then, all ranks call :meth:`advance` and use :meth:`done` to check whether the
    communication is complete. In the use case of multipole communication, the
    reduction result is a vector of multipole expansions to which all ranks add
    contribution. These contributions are communicated sparsely via arrays of box
    indices and expansions.

    .. automethod:: __init__
    .. automethod:: sources
    .. automethod:: sinks
    .. automethod:: messages
    .. automethod:: advance
    .. automethod:: done
    """

    def __init__(self, rank, size):
        """
        :arg rank: Current rank.
        :arg size: Total number of ranks.
        """
        assert 0 <= rank < size
        self.rank = rank
        self.left = 0
        self.right = size
        self.midpoint = size // 2

    def sources(self):
        """Return the set of source nodes at the current communication stage. The
        current rank receives messages from these ranks.
        """
        if self.rank < self.midpoint:
            partner = self.midpoint + (self.rank - self.left)
            if self.rank == self.midpoint - 1 and partner == self.right:
                partners = set()
            elif self.rank == self.midpoint - 1 and partner == self.right - 2:
                partners = {partner, partner + 1}
            else:
                partners = {partner}
        else:
            partner = self.left + (self.rank - self.midpoint)
            if self.rank == self.right - 1 and partner == self.midpoint:
                partners = set()
            elif self.rank == self.right - 1 and partner == self.midpoint - 2:
                partners = {partner, partner + 1}
            else:
                partners = {partner}

        return partners

    def sinks(self):
        """Return the set of sink nodes at this communication stage. The current rank
        sends a message to these ranks.
        """
        if self.rank < self.midpoint:
            partner = self.midpoint + (self.rank - self.left)
            if partner == self.right:
                partner -= 1
        else:
            partner = self.left + (self.rank - self.midpoint)
            if partner == self.midpoint:
                partner -= 1

        return {partner}

    def messages(self):
        """Return a range of ranks, such that the partial results of array indices
        used by these ranks are sent to the sinks.  This is returned as a
        [start, end) pair. By design, it is a consecutive range.
        """
        if self.rank < self.midpoint:
            return (self.midpoint, self.right)
        else:
            return (self.left, self.midpoint)

    def advance(self):
        """Advance to the next stage in the communication pattern.
        """
        if self.done():
            raise RuntimeError("finished communicating")

        if self.rank < self.midpoint:
            self.right = self.midpoint
            self.midpoint = (self.midpoint + self.left) // 2
        else:
            self.left = self.midpoint
            self.midpoint = (self.midpoint + self.right) // 2

    def done(self):
        """Return whether the current rank is finished communicating.
        """
        return self.left + 1 == self.right

# }}}


# {{{ MPI launcher

def run_mpi(script: str, num_processes: int, env: Dict[str, Any]) -> None:
    """Launch MPI processes.

    This function forks another process and uses ``mpiexec`` to launch
    *num_processes* MPI processes running *script*.

    :arg script: the Python script to run.
    :arg num_processes: the number of MPI process to launch.
    :arg env: a :class:`dict` of environment variables.
    """
    import os
    env = {key: str(value) for key, value in env.items()}
    env = {**os.environ, **env}

    import subprocess
    from mpi4py import MPI

    # Using "-m mpi4py" is necessary for avoiding deadlocks on exception cleanup
    # See https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html for details.

    mpi_library_name = MPI.Get_library_version()
    if mpi_library_name.startswith("Open MPI"):
        command = ["mpiexec", "-np", str(num_processes), "--oversubscribe"]
        for env_variable_name in env:
            command.extend(["-x", env_variable_name])
        command.extend([sys.executable, "-m", "mpi4py", script])
    else:
        command = [
            "mpiexec", "-np", str(num_processes), sys.executable,
            "-m", "mpi4py", script
            ]

    subprocess.run(command, env=env, check=True)

# }}}


# {{{ HostDeviceArray

class ImmutableHostDeviceArray:
    """Interface for arrays on both host and device.

    .. note:: This interface assumes the array is immutable. The behavior of
    modifying the content of either the host array or the device array is undefined.

    @TODO: Once available, replace this implementation with PyOpenCL's in-house
    implementation.
    """
    def __init__(self, queue, array):
        self.queue = queue
        self.shape = array.shape
        self.host_array = None
        self.device_array = None

        if isinstance(array, np.ndarray):
            self.host_array = array
        elif isinstance(array, cl.array.Array):
            self.device_array = array

    def with_queue(self, queue):
        self.queue = queue

    @property
    def svm_capable(self):
        svm_capabilities = \
            self.queue.device.get_info(cl.device_info.SVM_CAPABILITIES)
        if svm_capabilities & cl.device_svm_capabilities.FINE_GRAIN_BUFFER != 0:
            return True
        else:
            return False

    @property
    def host(self):
        if self.host_array is None:
            self.host_array = self.device_array.get(self.queue)
        return self.host_array

    @property
    def device(self):
        if self.device_array is None:
            # @TODO: Use SVM
            self.device_array = cl.array.to_device(self.queue, self.host_array)

        self.device_array.with_queue(self.queue)
        return self.device_array

# }}}


# {{{ coord_vec tools

def get_coord_vec_dtype(
        coord_dtype: np.dtype[Any], dimensions: int) -> np.dtype[Any]:
    if dimensions == 1:
        return coord_dtype
    else:
        return cltypes.vec_types[coord_dtype, dimensions]


def coord_vec_subscript_code(dimensions: int, vec_name: str, iaxis: int) -> str:
    assert 0 <= iaxis < dimensions
    if dimensions == 1:
        # a coord_vec_t is just a scalar
        return vec_name
    else:
        return f"{vec_name}.s{iaxis}"

# }}}

# vim: foldmethod=marker
