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


def drive_fmm(traversal, expansion_wrangler, src_weights):
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

    Returns the potentials computed by *expansion_wrangler*.
    """
    wrangler = expansion_wrangler

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    logger.info("start fmm")

    logger.debug("reorder source weights")

    src_weights = wrangler.reorder_sources(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    potentials = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps)

    # these potentials are called beta in [1]

    if traversal.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                "('list 3 close')")

        potentials = potentials + wrangler.eval_direct(
                traversal.target_boxes,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_weights)

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weights)

    if traversal.from_sep_close_bigger_starts is not None:
        logger.debug("evaluate separated close bigger interactions directly "
                "('list 4 close')")

        potentials = potentials + wrangler.eval_direct(
                traversal.target_or_target_parent_boxes,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")

    potentials = potentials + wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps)

    # }}}

    logger.debug("reorder potentials")
    result = wrangler.reorder_potentials(potentials)

    logger.debug("finalize potentials")
    result = wrangler.finalize_potentials(result)

    logger.info("fmm complete")

    return result


# {{{ expansion wrangler interface

class ExpansionWranglerInterface:
    """Abstract expansion handling interface for use with :func:`drive_fmm`.

    See this
    `test code <https://github.com/inducer/boxtree/blob/master/test/test_fmm.py>`_
    for a very simple sample implementation.

    Will usually hold a reference (and thereby be specific to) a
    :class:`boxtree.Tree` instance.
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

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        """Return an expansions array (compatible with
        :meth:`multipole_expansion_zeros`)
        containing multipole expansions in *source_boxes* due to sources
        with *src_weights*.
        All other expansions must be zero.
        """

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        """For each box in *source_parent_boxes*,
        gather (and translate) the box's children's multipole expansions in
        *mpole* and add the resulting expansion into the box's multipole
        expansion in *mpole*.

        :returns: *mpoles*
        """

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights):
        """For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weights*, which use :ref:`csr` and are
        indexed like *target_boxes*.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        """For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.  *starts* and *lists* use :ref:`csr`, and
        *starts* is indexed like *target_or_target_parent_boxes*.

        :returns: a new (local) expansion array, see
            :meth:`local_expansion_zeros`.
        """

    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_by_level, mpole_exps):
        """For a level *i*, each box in *target_boxes_by_source_level[i]*, evaluate
        the multipole expansion in *mpole_exps* in the nearby boxes given in
        *from_sep_smaller_by_level*, and return a new potential array.
        *starts* and *lists* in *from_sep_smaller_by_level[i]* use :ref:`csr`
        and *starts* is indexed like *target_boxes_by_source_level[i]*.

        :returns: a new potential array, see :meth:`output_zeros`.
        """

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        """For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.  *starts* and *lists*
        use :ref:`csr` and *starts* is indexed like
        *target_or_target_parent_boxes*.

        :returns: a new local expansion array, see
            :meth:`local_expansion_zeros`.
        """
        pass

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        """For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: *local_exps*
        """

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
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


# vim: filetype=pyopencl:fdm=marker
