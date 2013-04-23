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
    tree = traversal.tree

    wrangler = expansion_wrangler

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    logger.info("start fmm")

    logger.debug("reorder source weights")

    src_weights = wrangler.reorder_src_weights(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")

    mpole_exps = wrangler.form_multipoles(
            traversal.source_boxes,
            src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")

    for lev in xrange(tree.nlevels-1, -1, -1):
        start_parent_box, end_parent_box = \
                traversal.level_start_source_parent_box_nrs[lev:lev+2]
        wrangler.coarsen_multipoles(
                traversal.source_parent_boxes, start_parent_box, end_parent_box,
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
            traversal.target_or_target_parent_boxes,
            traversal.sep_siblings_starts,
            traversal.sep_siblings_lists,
            mpole_exps)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller nonsiblings' mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller nonsiblings' mpoles ('list 3') at particles")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            traversal.target_boxes,
            traversal.sep_smaller_nonsiblings_starts,
            traversal.sep_smaller_nonsiblings_lists,
            mpole_exps)

    # these potentials are called beta in [1]

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger nonsiblings' mpoles ("list 4")

    logger.debug("form locals for separated bigger nonsiblings' mpoles ('list 4')")

    local_exps = local_exps + wrangler.form_locals(
            traversal.target_or_target_parent_boxes,
            traversal.sep_bigger_nonsiblings_starts,
            traversal.sep_bigger_nonsiblings_lists,
            src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    for lev in xrange(1, tree.nlevels):
        start_box, end_box = traversal.level_start_target_or_target_parent_box_nrs[lev:lev+2]
        wrangler.refine_locals(
                traversal.target_or_target_parent_boxes,
                start_box, end_box, local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")

    potentials = potentials + wrangler.eval_locals(
            traversal.target_boxes,
            local_exps)

    # }}}

    logger.debug("reorder potentials")
    result = wrangler.reorder_potentials(potentials)

    logger.info("fmm complete")

    return result




# vim: filetype=pyopencl:fdm=marker
