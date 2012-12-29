from __future__ import division




def drive_fmm(traversal, expansion_wrangler, src_weights):
    tree = traversal.tree

    wrangler = expansion_wrangler

    # Following this article:
    # [1]  Carrier, J., Leslie Greengard, and Vladimir Rokhlin. “A Fast
    # Adaptive Multipole Algorithm for Particle Simulations.” SIAM Journal on
    # Scientific and Statistical Computing 9, no. 4 (July 1988): 669–686.
    # doi:10.1137/0909044.

    # Step/stage numbers refer to the paper above.

    # {{{ "Step 2.1:" Construct local multipoles

    mpole_exps = wrangler.form_multipoles(
            traversal.leaf_boxes,

            tree.box_starts,
            tree.box_particle_counts,
            tree.original_particle_ids,

            tree.box_centers,

            src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    for lev in xrange(tree.nlevels-1, -1, -1):
        start, end = traversal.branch_box_level_starts[lev:lev+2]
        wrangler.coarsen_multipoles(
                traversal.branch_boxes, start, end,
                tree.box_child_ids, tree.box_centers,
                mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct calculation on neighbor leaves ("list 1")

    potentials = wrangler.do_direct_eval(
            traversal.leaf_boxes,
            traversal.neighbor_leaves_starts,
            traversal.neighbor_leaves_lists,
            src_weights)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' (list 2) mpoles to local

    sib_local_exps = wrangler.multipole_to_local(
            traversal.sep_siblings_starts,
            traversal.sep_siblings_lists,
            mpole_exps)

    # sib_local_exps is called Gamma in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller nonsiblings' mpoles ("list 3") at particles

    potentials = potentials + wrangler.eval_multipoles(
            traversal.leaf_boxes,
            traversal.sep_smaller_nonsiblings_starts,
            traversal.sep_smaller_nonsiblings_lists,
            mpole_exps)

    # these potentials are called beta in [1]

    # }}}

    # {{{ "Stage 6:" translate separated bigger nonsiblings' mpoles ("list 4") to local

    bigger_sib_local_exps = wrangler.add_to_local(
            traversal.sep_bigger_nonsiblings_starts,
            traversal.sep_bigger_nonsiblings_lists,

            tree.box_starts,
            tree.box_particle_counts,
            tree.original_particle_ids,
            src_weights,

            tree.box_centers)

    # bigger_sib_local_exps is called Delta in [1]

    # }}}

    # {{{ "Stage 7:" propagate sib_local_exps downward

    for lev in xrange(1, tree.nlevels):
        start, end = traversal.level_starts[lev:lev+2]
        wrangler.refine_locals(
                start, end,
                tree.box_parent_ids,
                tree.box_centers,
                sib_local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    local_exps = sib_local_exps
    local_exps += bigger_sib_local_exps
    del sib_local_exps
    del bigger_sib_local_exps

    potentials = potentials + wrangler.add_eval_locals(
            traversal.leaf_boxes,
            local_exps)

    # }}}

    return potentials




# vim: filetype=pyopencl:fdm=marker
