"""
.. autoclass:: TimingResult

.. autoclass:: TimingFuture
"""

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


from collections.abc import Mapping


# {{{ timing result

class TimingResult(Mapping):
    """Interface for returned timing data.

    This supports accessing timing results via a mapping interface, along with
    combining results via :meth:`merge`.

    .. automethod:: merge
    """

    def __init__(self, *args, **kwargs):
        """See constructor for :class:`dict`."""
        self._mapping = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def merge(self, other):
        """Merge this result with another by adding together common fields."""
        result = {}

        for key in self:
            val = self.get(key)
            other_val = other.get(key)

            if val is None or other_val is None:
                continue

            result[key] = val + other_val

        return type(self)(result)

# }}}


# {{{ timing future

class TimingFuture:
    """Returns timing data for a potentially asynchronous operation.

    .. automethod:: result
    .. automethod:: done
    """

    def result(self):
        """Return a :class:`TimingResult`. May block."""
        raise NotImplementedError

    def done(self):
        """Return *True* if the operation is complete."""
        raise NotImplementedError

# }}}


# {{{ timing recorder

class TimingRecorder:

    def __init__(self):
        from collections import defaultdict
        self.futures = defaultdict(list)

    def add(self, description, future):
        self.futures[description].append(future)

    def summarize(self):
        result = {}

        for description, futures_list in self.futures.items():
            futures = iter(futures_list)

            timing_result = next(futures).result()
            for future in futures:
                timing_result = timing_result.merge(future.result())

            result[description] = timing_result

        return result

# }}}


# {{{ time recording tool

class DummyTimingFuture(TimingFuture):
    @classmethod
    def from_timer(cls, timer):
        return cls(wall_elapsed=timer.wall_elapsed,
                   process_elapsed=timer.process_elapsed)

    @classmethod
    def from_op_count(cls, op_count):
        return cls(ops_elapsed=op_count)

    def __init__(self, *args, **kwargs):
        self._result = TimingResult(*args, **kwargs)

    def result(self):
        return self._result

    def done(self):
        return True


def return_timing_data(wrapped):
    """A decorator for recording timing data for a function call.

    The decorated function returns a tuple (*retval*, *timing_future*)
    where *retval* is the original return value and *timing_future*
    supports the timing data future interface in :mod:`boxtree.fmm`.
    """

    from pytools import ProcessTimer

    def wrapper(*args, **kwargs):
        timer = ProcessTimer()
        retval = wrapped(*args, **kwargs)
        timer.done()

        future = DummyTimingFuture.from_timer(timer)
        return (retval, future)

    from functools import update_wrapper
    new_wrapper = update_wrapper(wrapper, wrapped)

    return new_wrapper

# }}}


# vim: foldmethod=marker
