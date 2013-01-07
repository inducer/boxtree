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
from pytools import Record
import pyopencl as cl
import pyopencl.array



class FromDeviceGettableRecord(Record):
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

