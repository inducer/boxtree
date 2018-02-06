from __future__ import absolute_import
import pyopencl as cl
import numpy as np
from six.moves import range

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 2
nparticles = 10**5

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
tg1 = FMMTraversalBuilder(ctx)
trav1, _ = tg1(queue, tree)

tg2 = FMMTraversalBuilder(ctx, compress_list_3=True)
trav2, _ = tg2(queue, tree)

trav1 = trav1.get(queue)
trav2 = trav2.get(queue)

for i in range(tree.nlevels):
    count1 = trav1.from_sep_smaller_by_level[i].count
    starts1 = trav1.from_sep_smaller_by_level[i].starts
    lists1 = trav1.from_sep_smaller_by_level[i].lists

    count2 = trav2.from_sep_smaller_by_level[i].count
    starts2 = trav2.from_sep_smaller_by_level[i].starts
    lists2 = trav2.from_sep_smaller_by_level[i].lists
    nonempty_indices = trav2.from_sep_smaller_by_level[i].nonempty_indices
    num_nonempty_lists = trav2.from_sep_smaller_by_level[i].num_nonempty_lists

    assert count1 == count2

    true_nonempty_starts = []
    true_nonempty_indices = []
    for j in range(trav1.target_boxes.shape[0]):
        if starts1[j] != starts1[j + 1]:
            true_nonempty_starts.append(starts1[j])
            true_nonempty_indices.append(j)
    true_num_nonempty_lists = len(true_nonempty_indices)
    true_nonempty_indices = np.array(true_nonempty_indices)
    true_nonempty_starts.append(starts1[-1])
    true_nonempty_starts = np.array(true_nonempty_starts)

    assert np.all(starts2 == true_nonempty_starts)
    assert np.all(lists1 == lists2)
    assert np.all(nonempty_indices == true_nonempty_indices)
    assert num_nonempty_lists == true_num_nonempty_lists
