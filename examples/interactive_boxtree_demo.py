import pyopencl as cl
import boxtree.tree_interactive_build

import matplotlib.pyplot as pt
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

tree = boxtree.tree_interactive_build.BoxTree()
tree.generate_uniform_boxtree(queue, nlevels=3, root_extent=2,
        # root_vertex=(0,0,0))
        root_vertex=(0,0))

# the default quad formula uses cell centers and cell measures
from modepy import GaussLegendreQuadrature
n_q_points = 5
quadrature_formula = GaussLegendreQuadrature(n_q_points - 1)
print(quadrature_formula.nodes, quadrature_formula.weights)
quad = boxtree.tree_interactive_build.QuadratureOnBoxTree(tree,
        quadrature_formula)
cell_centers = quad.get_cell_centers(queue)
cell_measures = quad.get_cell_measures(queue)
q_points = quad.get_q_points(queue)
q_weights = quad.get_q_weights(queue)

# print(q_points)
# print(q_weights, np.sum(q_weights.get()))

# print(q_points - cell_centers)
# print(q_weights - cell_measures)

# call get() before plotting
from boxtree.visualization import BoxTreePlotter
plt = BoxTreePlotter(tree.get(queue))
plt.draw_tree()
plt.set_bounding_box()
plt.draw_box_numbers()

qx = q_points[0].get(queue)
qy = q_points[1].get(queue)

pt.plot(qx, qy, '*')

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
