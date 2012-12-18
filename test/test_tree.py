from __future__ import division

import numpy as np
import sys
import pytools.test

import matplotlib.pyplot as pt

import pyopencl as cl
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests





@pytools.test.mark_test.opencl
def test_tree(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    #for dims in [2, 3]:
    for dims in [2]:
        nparticles = 10**5
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        if do_plot:
            pt.plot(particles[0].get(), particles[1].get(), "x")

        from htree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "%d boxes, testing..." % tree.nboxes

        starts = tree.box_starts.get()
        pcounts = tree.box_particle_counts.get()
        sorted_particles = np.array([pi.get() for pi in tree.particles])
        centers = tree.box_centers.get()
        levels = tree.box_levels.get()

        unsorted_particles = np.array([pi.get() for pi in particles])
        assert (sorted_particles
                == unsorted_particles[:, tree.original_particle_ids.get()]).all()

        assert np.max(levels) + 1 == tree.nlevels

        root_extent = tree.root_extent

        all_good_so_far = True

        if do_plot:
            tree.plot(zorder=10)

        for ibox in xrange(tree.nboxes):
            lev = int(levels[ibox])
            box_size = root_extent / (1 << lev)
            el = extent_low = centers[:, ibox] - 0.5*box_size
            eh = extent_high = extent_low + box_size

            box_particle_nrs = np.arange(starts[ibox], starts[ibox]+pcounts[ibox],
                    dtype=np.intp)

            box_particles = sorted_particles[:,box_particle_nrs]
            good = (
                    (box_particles < extent_high[:, np.newaxis])
                    &
                    (extent_low[:, np.newaxis] <= box_particles)
                    )

            all_good_here = good.all()
            if do_plot and not all_good_here and all_good_so_far:
                pt.plot(
                        box_particles[0, np.where(~good)[1]],
                        box_particles[1, np.where(~good)[1]], "ro")

                pt.plot([el[0], eh[0], eh[0], el[0], el[0]],
                        [el[1], el[1], eh[1], eh[1], el[1]], "r-", lw=1)

            all_good_so_far = all_good_so_far and all_good_here

        if do_plot:
            pt.gca().set_aspect("equal", "datalim")
            pt.show()

        assert all_good_so_far

        print "done"

@pytools.test.mark_test.opencl
def test_tree_connectivity(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [2]:
        nparticles = 10**3
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        if do_plot:
            pt.plot(particles[0].get(), particles[1].get(), "x")

        from htree import TreeBuilder
        tb = TreeBuilder(ctx)

        tree = tb(queue, particles, max_particles_in_box=30, debug=True)


        levels = tree.box_levels.get()
        parents = tree.box_parent_ids.get().T
        children = tree.box_child_ids.get().T

        for ibox in xrange(1, tree.nboxes):
            parent = parents[ibox]

            assert levels[parent] + 1 == levels[ibox]
            assert ibox in children[parent], ibox




@pytools.test.mark_test.opencl
def test_traversal(ctx_getter, do_plot=False):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    #for dims in [2, 3]:
    for dims in [2]:
        nparticles = 10**4
        dtype = np.float64

        from pyopencl.clrandom import RanluxGenerator
        rng = RanluxGenerator(queue, seed=15)

        from pytools.obj_array import make_obj_array
        particles = make_obj_array([
            rng.normal(queue, nparticles, dtype=dtype)
            for i in range(dims)])

        if do_plot:
            pt.plot(particles[0].get(), particles[1].get(), "x")

        from htree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "done"

        from htree.traversal import FMMTraversalGenerator
        tg = FMMTraversalGenerator(ctx)
        traversal = tg(queue, tree)

        coll_starts = traversal.colleagues_starts.get()
        coll_list = traversal.colleagues_list.get()
        neigh_starts = traversal.neighbor_leaves_starts.get()
        neigh_list = traversal.neighbor_leaves_list.get()
        wss_starts = traversal.well_sep_siblings_starts.get()
        wss_list = traversal.well_sep_siblings_list.get()
        leaves = traversal.leaf_boxes.get()

        from htree import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black")
        plotter.draw_box_numbers()

        from random import randrange, seed
        seed(5)

        if 0:
            # colleagues

            for i in xrange(5):
                ibox = randrange(tree.nboxes)
                plotter.draw_box(ibox, facecolor='red')

                start, end = coll_starts[ibox:ibox+2]

                for jbox in coll_list[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')
        elif 1:
            # near neighbors ("list 1")

            for i in xrange(20):
                ileaf = randrange(len(leaves))
                ibox = leaves[ileaf]
                plotter.draw_box(ibox, facecolor='red')

                start, end = neigh_starts[ileaf:ileaf+2]

                for jbox in neigh_list[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')
        else:
            # well-separated siblings (list 2)

            for i in xrange(1):
                ibox = randrange(tree.nboxes)
                plotter.draw_box(ibox, facecolor='red')

                start, end = wss_starts[ibox:ibox+2]
                print start, end

                for jbox in wss_list[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')
        pt.show()









# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
