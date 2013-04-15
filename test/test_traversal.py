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
import numpy.linalg as la
import pyopencl as cl

import pytools.test
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from boxtree.tools import make_particle_array

import logging
logger = logging.getLogger(__name__)


# {{{ connectivity test

@pytools.test.mark_test.opencl
def test_tree_connectivity(ctx_getter):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    for dims in [2]:
        nparticles = 10**5
        dtype = np.float64

        particles = make_particle_array(queue, nparticles, dims, dtype)

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        tree = tb(queue, particles, max_particles_in_box=30, debug=True)

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        trav = tg(queue, tree).get()
        tree = tree.get()

        levels = tree.box_levels
        parents = tree.box_parent_ids.T
        children = tree.box_child_ids.T
        centers = tree.box_centers.T

        # {{{ parent and child relations, levels match up

        for ibox in xrange(1, tree.nboxes):
            # /!\ Not testing box 0, has no parents
            parent = parents[ibox]

            assert levels[parent] + 1 == levels[ibox]
            assert ibox in children[parent], ibox

        # }}}

        if 0:
            import matplotlib.pyplot as pt
            from boxtree.visualization import TreePlotter
            plotter = TreePlotter(tree)
            plotter.draw_tree(fill=False, edgecolor="black")
            plotter.draw_box_numbers()
            plotter.set_bounding_box()
            pt.show()

        # {{{ neighbor_source_boxes (list 1) consists of source boxes

        for isrcbox, ibox in enumerate(trav.source_boxes):
            start, end = trav.neighbor_source_boxes_starts[isrcbox:isrcbox+2]
            nbl = trav.neighbor_source_boxes_lists[start:end]

            assert ibox in nbl
            for jbox in nbl:
                assert (0 == children[jbox]).all(), (ibox, jbox, children[jbox])

        print "list 1 tested"

        # }}}

        # {{{ separated siblings (list 2) are actually separated

        for ibox in xrange(tree.nboxes):
            start, end = trav.sep_siblings_starts[ibox:ibox+2]
            seps = trav.sep_siblings_lists[start:end]

            assert (levels[seps] == levels[ibox]).all()

            # three-ish box radii (half of size)
            mindist = 2.5 * 0.5 * 2**-int(levels[ibox]) * tree.root_extent

            icenter = centers[ibox]
            for jbox in seps:
                dist = la.norm(centers[jbox]-icenter)
                assert dist > mindist, (dist, mindist)

        # }}}

        # {{{ sep_{smaller,bigger}_nonsiblings are duals of each other

        # (technically, we only test one half of that)

        for isource_box, ibox in enumerate(trav.source_boxes):
            start, end = trav.sep_smaller_nonsiblings_starts[isource_box:isource_box+2]

            for jbox in trav.sep_smaller_nonsiblings_lists[start:end]:
                rstart, rend = trav.sep_bigger_nonsiblings_starts[jbox:jbox+2]

                assert ibox in trav.sep_bigger_nonsiblings_lists[rstart:rend], (ibox, jbox)

        print "list 3, 4 are duals"

        # }}}

        # {{{ sep_smaller_nonsiblings satisfies size assumption

        for isource_box, ibox in enumerate(trav.source_boxes):
            start, end = trav.sep_smaller_nonsiblings_starts[isource_box:isource_box+2]

            for jbox in trav.sep_smaller_nonsiblings_lists[start:end]:
                assert levels[ibox] < levels[jbox]

        print "list 3 satisfies size assumption"

        # }}}

        # {{{ sep_smaller_nonsiblings satisfies size assumption

        for  ibox in xrange(tree.nboxes):
            start, end = trav.sep_bigger_nonsiblings_starts[ibox:ibox+2]

            for jbox in trav.sep_bigger_nonsiblings_lists[start:end]:
                assert levels[ibox] > levels[jbox]

        print "list 4 satisfies size assumption"

        # }}}

# }}}

# {{{ visualization helper (not a test)

def plot_traversal(ctx_getter, do_plot=False):
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

        #if do_plot:
            #pt.plot(particles[0].get(), particles[1].get(), "x")

        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)

        queue.finish()
        print "building..."
        tree = tb(queue, particles, max_particles_in_box=30, debug=True)
        print "done"

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx)
        trav = tg(queue, tree).get()

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree)
        plotter.draw_tree(fill=False, edgecolor="black")
        #plotter.draw_box_numbers()
        plotter.set_bounding_box()

        from random import randrange, seed
        seed(7)

        # {{{ generic box drawing helper

        def draw_some_box_lists(starts, lists, key_to_box=None,
                count=5):
            actual_count = 0
            while actual_count < count:
                if key_to_box is not None:
                    key = randrange(len(key_to_box))
                    ibox = key_to_box[key]
                else:
                    key = ibox = randrange(tree.nboxes)

                start, end = starts[key:key+2]
                if start == end:
                    continue

                #print ibox, start, end, lists[start:end]
                for jbox in lists[start:end]:
                    plotter.draw_box(jbox, facecolor='yellow')

                plotter.draw_box(ibox, facecolor='red')

                actual_count += 1

        # }}}

        if 0:
            # colleagues
            draw_some_box_lists(
                    trav.colleagues_starts,
                    trav.colleagues_lists)
        elif 0:
            # near neighbors ("list 1")
            draw_some_box_lists(
                    trav.neighbor_leaves_starts,
                    trav.neighbor_leaves_lists,
                    key_to_box=trav.source_boxes)
        elif 0:
            # well-separated siblings (list 2)
            draw_some_box_lists(
                    trav.sep_siblings_starts,
                    trav.sep_siblings_lists)
        elif 1:
            # separated smaller non-siblings (list 3)
            draw_some_box_lists(
                    trav.sep_smaller_nonsiblings_starts,
                    trav.sep_smaller_nonsiblings_lists,
                    key_to_box=trav.source_boxes)
        elif 1:
            # separated bigger non-siblings (list 4)
            draw_some_box_lists(
                    trav.sep_bigger_nonsiblings_starts,
                    trav.sep_bigger_nonsiblings_lists)

        import matplotlib.pyplot as pt
        pt.show()

# }}}



# You can test individual routines by typing
# $ python test_traversal.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker

