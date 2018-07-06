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

import logging
logger = logging.getLogger(__name__)

from pytools import ProcessLogger, Record


def drive_fmm(traversal, expansion_wrangler, src_weights, timing_data=None):
    """Top-level driver routine for a fast multipole calculation.

    In part, this is intended as a template for custom FMMs, in the sense that
    you may copy and paste its
    `source code <https://github.com/inducer/boxtree/blob/master/boxtree/fmm.py>`_
    as a starting point.

    Nonetheless, many common applications (such as point-to-point FMMs) can be
    covered by supplying the right *expansion_wrangler* to this routine.

    :arg traversal: A :class:`boxtree.traversal.FMMTraversalInfo` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.
    :arg timing_data: Either *None*, or a :class:`dict` that is populated with
        timing information for the stages of the algorithm (in the form of
        instances of :class:`TimingResult`), if such information is available.

    Returns the potentials computed by *expansion_wrangler*.

    """
    wrangler = expansion_wrangler

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "qbx fmm")
    recorder = TimingRecorder()

    src_weights = wrangler.reorder_sources(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    mpole_exps = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights,
            timing_data=recorder.next())

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps,
            timing_data=recorder.next())

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    potentials = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights,
            timing_data=recorder.next())

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    local_exps = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps,
            timing_data=recorder.next())

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps,
            timing_data=recorder.next())

    # these potentials are called beta in [1]

    if traversal.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                "('list 3 close')")

        potentials = potentials + wrangler.eval_direct(
                traversal.target_boxes,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_weights,
                timing_data=recorder.next())

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    local_exps = local_exps + wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weights,
            timing_data=recorder.next())

    if traversal.from_sep_close_bigger_starts is not None:
        potentials = potentials + wrangler.eval_direct(
                traversal.target_or_target_parent_boxes,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_weights,
                timing_data=recorder.next())

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps,
            timing_data=recorder.next())

    # }}}

    # {{{ "Stage 8:" evaluate locals

    potentials = potentials + wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps,
            timing_data=recorder.next())

    # }}}

    result = wrangler.reorder_potentials(potentials)

    result = wrangler.finalize_potentials(result)

    fmm_proc.done()

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result


# {{{ expansion wrangler interface

class ExpansionWranglerInterface:
    """Abstract expansion handling interface for use with :func:`drive_fmm`.

    See this
    `test code <https://github.com/inducer/boxtree/blob/master/test/test_fmm.py>`_
    for a very simple sample implementation.

    Will usually hold a reference (and thereby be specific to) a
    :class:`boxtree.Tree` instance.

    This interface supports collecting timing data. If timing data is requested,
    the *timing_data* argument is a :class:`TimingDataWaiter` whose fields can
    """

    def multipole_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """

    def local_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """

    def output_zeros(self):
        """Return a potentials array (which must support addition) capable of
        holding a potential value for each target in the tree. Note that
        :func:`drive_fmm` makes no assumptions about *potential* other than
        that it supports addition--it may consist of potentials, gradients of
        the potential, or arbitrary other per-target output data.
        """

    def reorder_sources(self, source_array):
        """Return a copy of *source_array* in
        :ref:`tree source order <particle-orderings>`.
        *source_array* is in user source order.
        """

    def reorder_potentials(self, potentials):
        """Return a copy of *potentials* in
        :ref:`user target order <particle-orderings>`.
        *source_weights* is in tree target order.
        """

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights,
            timing_data=None):
        """Return an expansions array (compatible with
        :meth:`multipole_expansion_zeros`)
        containing multipole expansions in *source_boxes* due to sources
        with *src_weights*.
        All other expansions must be zero.
        """

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles, timing_data=None):
        """For each box in *source_parent_boxes*,
        gather (and translate) the box's children's multipole expansions in
        *mpole* and add the resulting expansion into the box's multipole
        expansion in *mpole*.

        :returns: *mpoles*
        """

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights, timing_data=None):
        """For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weights*, which use :ref:`csr` and are
        indexed like *target_boxes*.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps, timing_data=None):
        """For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.  *starts* and *lists* use :ref:`csr`, and
        *starts* is indexed like *target_or_target_parent_boxes*.

        :returns: a new (local) expansion array, see
            :meth:`local_expansion_zeros`.
        """

    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_by_level, mpole_exps,
            timing_data=None):
        """For a level *i*, each box in *target_boxes_by_source_level[i]*, evaluate
        the multipole expansion in *mpole_exps* in the nearby boxes given in
        *from_sep_smaller_by_level*, and return a new potential array.
        *starts* and *lists* in *from_sep_smaller_by_level[i]* use :ref:`csr`
        and *starts* is indexed like *target_boxes_by_source_level[i]*.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights,
            timing_data=None):
        """For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.  *starts* and *lists*
        use :ref:`csr` and *starts* is indexed like
        *target_or_target_parent_boxes*.

        :returns: a new local expansion array, see
            :meth:`local_expansion_zeros`.
        """

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps, timing_data=None):
        """For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: *local_exps*
        """

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps,
            timing_data=None):
        """For each box in *target_boxes*, evaluate the local expansion in
        *local_exps* and return a new potential array.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    def finalize_potentials(self, potentials):
        """
        Postprocess the reordered potentials. This is where global scaling
        factors could be applied. This is distinct from :meth:`reorder_potentials`
        because some derived FMMs (notably the QBX FMM) do their own reordering.
        """

# }}}


# {{{ timing result

class TimingResult(Record):
    """
    .. automethod:: __add__

    .. attribute:: wall_elapsed
    .. attribute:: process_elapsed
    """

    def __init__(self, wall_elapsed, process_elapsed):
        Record.__init__(self,
                wall_elapsed=wall_elapsed,
                process_elapsed=process_elapsed)

    def __add__(self, other):
        wall_elapsed = self.wall_elapsed + other.wall_elapsed
        process_elapsed = self.process_elapsed + other.process_elapsed
        return TimingResult(wall_elapsed, process_elapsed)

# }}}


# {{{ timing waiter

class TimingWaiter(object):
    """Obtains timing data through a supplied callback function.

    Attributes that can be set::

    .. attribute:: description

        A string, the description of the timing data.

    .. attribute:: callback

        Returns a :class:`TimingResult`.
    """

    def __init__(self):
        self.description = None
        self.callback = None
        self._result = None

    @property
    def empty(self):
        return not self.callback

    @property
    def result(self):
        if self._result is None:
            self.wait()

        return self._result

    def wait(self):
        if self.empty:
            return

        callback_result = self.callback()
        self._result = TimingResult(
                callback_result.wall_elapsed,
                callback_result.process_elapsed)

# }}}


# {{{ timing recorder

class TimingRecorder(object):

    def __init__(self):
        self.records = []

    def next(self):
        self.records.append(TimingWaiter())
        return self.records[-1]

    def summarize(self):
        result = {}

        for record in self.records:
            if record.empty:
                continue

            description = record.description

            if description in result:
                result[description] += record.result
            else:
                result[description] = record.result

        return result

# }}}


# vim: filetype=pyopencl:fdm=marker
