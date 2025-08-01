# STARTEXAMPLE
import logging

import numpy as np

import pyopencl as cl
from pytools import obj_array

from boxtree import TreeBuilder
from boxtree.array_context import PyOpenCLArrayContext
from boxtree.traversal import FMMTraversalBuilder
from boxtree.visualization import TreePlotter


logging.basicConfig(level="INFO")

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

dims = 2
nparticles = 500

# -----------------------------------------------------------------------------
# generate some random particle positions
# -----------------------------------------------------------------------------
rng = np.random.default_rng(seed=15)

particles = obj_array.new_1d([
    actx.from_numpy(rng.normal(size=nparticles))
    for i in range(dims)])

# -----------------------------------------------------------------------------
# build tree and traversals (lists)
# -----------------------------------------------------------------------------
tb = TreeBuilder(actx)
tree, _ = tb(actx, particles, max_particles_in_box=5)

tg = FMMTraversalBuilder(actx)
trav, _ = tg(actx, tree)

# ENDEXAMPLE

# -----------------------------------------------------------------------------
# plot the tree
# -----------------------------------------------------------------------------
import matplotlib.pyplot as pt


particles = actx.to_numpy(particles)
tree = actx.to_numpy(tree)

pt.plot(particles[0], particles[1], "+")
plotter = TreePlotter(tree)

plotter.draw_tree(fill=False, edgecolor="black")
# plotter.draw_box_numbers()
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
