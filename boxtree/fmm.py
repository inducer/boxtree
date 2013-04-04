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





def drive_fmm(traversal, expansion_wrangler, src_weights):
    tree = traversal.tree

    wrangler = expansion_wrangler

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    src_weights = wrangler.reorder_src_weights(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    mpole_exps = wrangler.form_multipoles(
            traversal.leaf_boxes,
            src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    for lev in xrange(tree.nlevels-1, -1, -1):
        start_parent_box, end_parent_box = traversal.level_start_parent_box_nrs[lev:lev+2]
        wrangler.coarsen_multipoles(
                traversal.parent_boxes, start_parent_box, end_parent_box,
                mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct calculation on neighbor leaves ("list 1")

    potentials = wrangler.eval_direct(
            traversal.leaf_boxes,
            traversal.neighbor_leaves_starts,
            traversal.neighbor_leaves_lists,
            src_weights)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    local_exps = wrangler.multipole_to_local(
            traversal.sep_siblings_starts,
            traversal.sep_siblings_lists,
            mpole_exps)

    # sib_local_exps is called Gamma in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller nonsiblings' mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its contribution
    # *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            traversal.leaf_boxes,
            traversal.sep_smaller_nonsiblings_starts,
            traversal.sep_smaller_nonsiblings_lists,
            mpole_exps)

    # these potentials are called beta in [1]

    # }}}

    # {{{ "Stage 6:" translate separated bigger nonsiblings' mpoles ("list 4") to local

    local_exps = local_exps + wrangler.multipole_to_local(
            traversal.sep_bigger_nonsiblings_starts,
            traversal.sep_bigger_nonsiblings_lists,
            mpole_exps)

    # bigger_nonsib_local_exps is called Delta in [1]

    # }}}

    # {{{ "Stage 7:" propagate sib_local_exps downward

    for lev in xrange(1, tree.nlevels):
        start_box, end_box = tree.level_start_box_nrs[lev:lev+2]
        wrangler.refine_locals(start_box, end_box, local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    potentials = potentials + wrangler.eval_locals(
            traversal.leaf_boxes,
            local_exps)

    # }}}

    return wrangler.reorder_potentials(potentials)




# vim: filetype=pyopencl:fdm=marker
