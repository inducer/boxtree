import pyopencl as cl
import boxtree.tree_interactive_build

import matplotlib.pyplot as pt

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

tree = boxtree.tree_interactive_build.BoxTree()
tree.generate_uniform_boxtree(queue, nlevels=4)

# call get() before plotting
from boxtree.visualization import BoxTreePlotter
plt = BoxTreePlotter(tree.get(queue))
plt.draw_tree()
plt.set_bounding_box()
plt.draw_box_numbers()

pt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
pt.tick_params(
    axis='y',
    which='both',
    left='off',
    top='off',
    labelleft='off')

pt.show()
