# STARTEXAMPLE
import pyopencl as cl
import numpy as np
import logging
logging.basicConfig(level="INFO")

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 2
nparticles = 500

# -----------------------------------------------------------------------------
# generate some random particle positions
# -----------------------------------------------------------------------------
from pyopencl.clrandom import PhiloxGenerator
rng = PhiloxGenerator(ctx, seed=15)

from pytools.obj_array import make_obj_array
particles = make_obj_array([
    rng.normal(queue, nparticles, dtype=np.float64)
    for i in range(dims)])

# -----------------------------------------------------------------------------
# build tree and traversals (lists)
# -----------------------------------------------------------------------------
from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=5)

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# ENDEXAMPLE

# -----------------------------------------------------------------------------
# plot the tree
# -----------------------------------------------------------------------------

import matplotlib.pyplot as pt

pt.plot(particles[0].get(), particles[1].get(), "+")

from boxtree.visualization import TreePlotter
plotter = TreePlotter(tree.get(queue=queue))
plotter.draw_tree(fill=False, edgecolor="black")
#plotter.draw_box_numbers()
plotter.set_bounding_box()
pt.gca().set_aspect("equal")
pt.tight_layout()
pt.tick_params(
    axis="x",          # changes apply to the x-axis
    which="both",      # both major and minor ticks are affected
    bottom="off",      # ticks along the bottom edge are off
    top="off",         # ticks along the top edge are off
    labelbottom="off")
pt.tick_params(
    axis="y",
    which="both",
    left="off",
    top="off",
    labelleft="off")
pt.savefig("tree.pdf")
