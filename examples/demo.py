# STARTEXAMPLE
import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 2
nparticles = 10**4

# -----------------------------------------------------------------------------
# generate some random particle positions
# -----------------------------------------------------------------------------
from pyopencl.clrandom import RanluxGenerator
rng = RanluxGenerator(queue, seed=15)

from pytools.obj_array import make_obj_array
particles = make_obj_array([
    rng.normal(queue, nparticles, dtype=np.float64)
    for i in range(dims)])

# -----------------------------------------------------------------------------
# build tree and traversals (lists)
# -----------------------------------------------------------------------------
from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=30)

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# ENDEXAMPLE

# -----------------------------------------------------------------------------
# plot the tree
# -----------------------------------------------------------------------------

import matplotlib.pyplot as pt

pt.plot(particles[0].get(), particles[1].get(), "x")

from boxtree.visualization import TreePlotter
plotter = TreePlotter(tree.get(queue=queue))
plotter.draw_tree(fill=False, edgecolor="black")
plotter.draw_box_numbers()
plotter.set_bounding_box()
pt.gca().set_aspect("equal")
pt.savefig("tree.png")
