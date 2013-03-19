from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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




import pyopencl as cl
from boxtree.tools import get_type_moniker
from pytools import memoize, memoize_method
from mako.template import Template
import numpy as np




# {{{ bounding box finding

@memoize
def make_bounding_box_dtype(device, dimensions, coord_dtype):
    from boxtree import AXIS_NAMES
    fields = []
    for i in range(dimensions):
        fields.append(("min_%s" % AXIS_NAMES[i], coord_dtype))
        fields.append(("max_%s" % AXIS_NAMES[i], coord_dtype))

    dtype = np.dtype(fields)

    name = "boxtree_bbox_%dd_%s_t" % (dimensions, get_type_moniker(coord_dtype))

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    return dtype, c_decl




BBOX_CODE_TEMPLATE = Template(r"""//CL//
    ${bbox_struct_decl}

    typedef ${dtype_to_ctype(bbox_dtype)} bbox_t;
    typedef ${dtype_to_ctype(coord_dtype)} coord_t;

    bbox_t bbox_neutral()
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${coord_dtype_3ltr}_MAX;
            result.max_${ax} = -${coord_dtype_3ltr}_MAX;
        %endfor
        return result;
    }

    bbox_t bbox_from_particle(${", ".join("coord_t %s" % ax for ax in axis_names)})
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${ax};
            result.max_${ax} = ${ax};
        %endfor
        return result;
    }

    bbox_t agg_bbox(bbox_t a, bbox_t b)
    {
        %for ax in axis_names:
            a.min_${ax} = min(a.min_${ax}, b.min_${ax});
            a.max_${ax} = max(a.max_${ax}, b.max_${ax});
        %endfor
        return a;
    }
""", strict_undefined=True)

class BoundingBoxFinder:
    def __init__(self, context):
        self.context = context

        for dev in context.devices:
            if (dev.vendor == "Intel(R) Corporation"
                    and dev.version == "OpenCL 1.2 (Build 56860)"):
                raise RuntimeError("bounding box finder does not work "
                        "properly with this CL runtime.")

    @memoize_method
    def get_kernel(self, dimensions, coord_dtype):
        from pyopencl.tools import dtype_to_ctype
        bbox_dtype, bbox_cdecl = make_bounding_box_dtype(
                self.context.devices[0], dimensions, coord_dtype)

        if coord_dtype == np.float64:
            coord_dtype_3ltr = "DBL"
        elif coord_dtype == np.float32:
            coord_dtype_3ltr = "FLT"
        else:
            raise TypeError("unknown coord_dtype")

        from boxtree import AXIS_NAMES
        axis_names = AXIS_NAMES[:dimensions]

        coord_ctype = dtype_to_ctype(coord_dtype)

        preamble = BBOX_CODE_TEMPLATE.render(
                axis_names=axis_names,
                dimensions=dimensions,
                coord_dtype=coord_dtype,
                coord_dtype_3ltr=coord_dtype_3ltr,
                bbox_struct_decl=bbox_cdecl,
                dtype_to_ctype=dtype_to_ctype,
                bbox_dtype=bbox_dtype,
                )

        from pyopencl.reduction import ReductionKernel
        return ReductionKernel(self.context, bbox_dtype,
                neutral="bbox_neutral()",
                reduce_expr="agg_bbox(a, b)",
                map_expr="bbox_from_particle(%s)" % ", ".join(
                    "%s[i]" % ax for ax in axis_names),
                arguments=", ".join(
                    "__global %s *%s" % (coord_ctype, ax) for ax in axis_names),
                preamble=preamble,
                name="bounding_box")

    def __call__(self, particles):
        dimensions = len(particles)

        from pytools import single_valued
        coord_dtype = single_valued(coord.dtype for coord in particles)

        return self.get_kernel(dimensions, coord_dtype)(*particles)

# }}}
