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


import pyopencl as cl  # noqa
from boxtree.tools import get_type_moniker
from pytools import memoize, memoize_method
from pyopencl.reduction import ReductionTemplate
import numpy as np


@memoize
def make_bounding_box_dtype(device, dimensions, coord_dtype):
    from boxtree.tools import AXIS_NAMES
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


BBOX_REDUCTION_TPL = ReductionTemplate(
    preamble=r"""//CL:mako//
        <%
            if coord_dtype == np.float64:
                coord_dtype_3ltr = "DBL"
            elif coord_dtype == np.float32:
                coord_dtype_3ltr = "FLT"
            else:
                raise TypeError("unknown coord_dtype")
        %>

        bbox_t bbox_neutral()
        {
            bbox_t result;
            %for ax in axis_names:
                result.min_${ax} = ${coord_dtype_3ltr}_MAX;
                result.max_${ax} = -${coord_dtype_3ltr}_MAX;
            %endfor
            return result;
        }

        bbox_t bbox_from_particle(
            %for ax in axis_names:
                coord_t ${ax},
            %endfor
            coord_t radius
            )
        {
            bbox_t result;
            %for ax in axis_names:
                result.min_${ax} = ${ax} - radius;
                result.max_${ax} = ${ax} + radius;
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

        """,
    arguments=r"""//CL:mako//
        %for ax in axis_names:
            coord_t *${ax},
        %endfor
        %if have_radii:
            coord_t *radii,
        %endif
        """,
    neutral="bbox_neutral()",
    reduce_expr="agg_bbox(a, b)",
    map_expr=r"""//CL:mako//
        bbox_from_particle(
            %for ax in axis_names:
                ${ax}[i],
            %endfor
            %if have_radii:
                radii[i]
            %else:
                0
            %endif
            )
            """,
    name_prefix="bounding_box")


class BoundingBoxFinder:
    def __init__(self, context):
        self.context = context

        for dev in context.devices:
            if (dev.vendor == "Intel(R) Corporation"
                    and dev.version == "OpenCL 1.2 (Build 56860)"):
                raise RuntimeError("bounding box finder does not work "
                        "properly with this CL runtime.")

    @memoize_method
    def get_kernel(self, dimensions, coord_dtype, have_radii):
        bbox_dtype, bbox_cdecl = make_bounding_box_dtype(
                self.context.devices[0], dimensions, coord_dtype)

        from boxtree.tools import AXIS_NAMES
        return BBOX_REDUCTION_TPL.build(
                self.context,
                type_aliases=(
                    ("reduction_t", bbox_dtype),
                    ("bbox_t", bbox_dtype),
                    ("coord_t", coord_dtype),
                    ),
                var_values=(
                    ("axis_names", AXIS_NAMES[:dimensions]),
                    ("dimensions", dimensions),
                    ("coord_dtype", coord_dtype),
                    ("have_radii", have_radii),
                    ("np", np),
                    )
                )

    def __call__(self, particles, radii, wait_for=None):
        dimensions = len(particles)

        from pytools import single_valued
        coord_dtype = single_valued(coord.dtype for coord in particles)

        if radii is None:
            radii_tuple = ()
        else:
            radii_tuple = (radii,)

        knl = self.get_kernel(dimensions, coord_dtype,
                # have_radii:
                radii is not None)
        return knl(*(tuple(particles) + radii_tuple),
                wait_for=wait_for, return_event=True)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
