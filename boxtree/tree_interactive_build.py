from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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
from itertools import product

import loopy as lp
import pyopencl as cl
import pyopencl.array  # noqa
from pytools.obj_array import make_obj_array
from boxtree.tools import DeviceDataRecord

# {{{ Data structure for a tree of boxes

class BoxTree(DeviceDataRecord):
    r"""A quad/oct-tree tree consisting of a hierarchy of boxes.
    The intended use for this data structure is to generate particles
    as quadrature points in each leaf box. Those particles can then
    be used to build a Tree object and then generate the Traversal for
    FMM.

    This class is designed such that it is easy to adaptively modify
    the tree structure (refine/coarsen). Unlike flattened Tree class,
    it handles itself and does not rely on external tree builders.

    .. ------------------------------------------------------------------------
    .. rubric:: Data types
    .. ------------------------------------------------------------------------

    .. attribute:: box_id_dtype
    .. attribute:: coord_dtype
    .. attribute:: box_level_dtype

    .. ------------------------------------------------------------------------
    .. rubric:: Counts and sizes
    .. ------------------------------------------------------------------------

    .. attribute:: root_extent

        the root box size, a scalar

    .. attribute:: nlevels

    .. attribute:: nboxes

        Can be larger than the actual number of boxes, since the box ids of
        purged boxes during coarsening are not reused.

    .. attribute:: nboxes_level

        ``size_t [nlevels]``

        Can be larger than the actual number of boxes at each level due to
        the same reason for nboxes.

    .. attribute:: n_active_boxes

    .. ------------------------------------------------------------------------
    .. rubric:: Box properties
    .. ------------------------------------------------------------------------

    .. attribute:: box_centers

        ``coord_t [dimensions, nboxes]``

        (C order, 'structure of arrays')

    .. attribute:: box_levels

        :attr:`box_level_dtype` ``box_level_t [nboxes]``

    .. attribute:: box_is_active

        :attr:`bool` ``bool [nboxes]``

        FIXME: pyopencl cannot map 'bool'. Resort to use an int for now.

    .. ------------------------------------------------------------------------
    .. rubric:: Structural properties
    .. ------------------------------------------------------------------------

    .. attribute:: level_boxes

        ``numpy.ndarray [nlevels]``
        ``box_id_t [nlevels][nboxes_level[level]]``

        A :class:`numpy.ndarray` of box ids at each level. It acts as an
        inverse lookup table of box_levels. The outer layer is an object
        array to account for variable lengths at each level.

    .. attribute:: box_parent_ids

        ``box_id_t [nboxes]``

        Box 0 (the root) has 0 as its parent.

    .. attribute:: box_child_ids

        ``box_id_t [2**dimensions, nboxes]``

        (C order, 'structure of arrays')

        "0" is used as a 'no child' marker, as the root box can never
        occur as any box's child. Boxes with no child are called active
        boxes.

    .. attribute:: active_boxes

        ``box_id_t [n_active_boxes]``
    """
    def get_copy_kwargs(self, **kwargs):
        # cl arrays
        for f in self.__class__.fields:
            if f not in kwargs:
                try:
                    kwargs[f] = getattr(self, f)
                except AttributeError:
                    pass

        # others
        kwargs.update({
            "size_t":self.size_t,
            "box_id_dtype":self.box_id_dtype,
            "box_level_dtype":self.box_level_dtype,
            "coord_dtype":self.coord_dtype,
            "root_extent":self.root_extent,
            })

        return kwargs

    def generate_uniform_boxtree(self, queue,
            root_vertex=np.zeros(2),
            root_extent=1,
            nlevels=1,
            box_id_dtype=np.int32,
            box_level_dtype=np.int32,
            coord_dtype=np.float64):
        """A plain boxtree with uniform levels (a complete tree).
        The root box is given its vertex with the smallest coordinates
        and its extent.
        """
        self.size_t = np.int32
        self.box_id_dtype = box_id_dtype
        self.box_level_dtype = box_level_dtype
        self.coord_dtype = coord_dtype

        dim = len(root_vertex)
        self.root_extent = root_extent

        self.nboxes_level = cl.array.to_device(queue,
                np.array(
                    [1 << (dim * l) for l in range(nlevels)],
                    dtype=self.size_t))
        self.register_fields({"nboxes_level":self.nboxes_level})

        nboxes = self.size_t(cl.array.sum(self.nboxes_level).get())

        self.box_levels = cl.array.zeros(queue, nboxes, box_level_dtype)
        level_start = 0
        for l in range(nlevels):
            offset = self.size_t(self.nboxes_level[l].get())
            self.box_levels[level_start:level_start + offset] = l
            level_start += offset
        self.register_fields({"box_levels":self.box_levels})

        self.box_centers = cl.array.zeros(queue, (dim, nboxes), coord_dtype)
        ibox = 0
        for l in range(nlevels):
            dx = self.root_extent / (1 << l)
            for cid in product(range(1 << l), repeat=dim):
                for d in range(dim):
                    self.box_centers[d, ibox] = cid[d] * dx + (
                            dx / 2 + root_vertex[d])
                ibox += 1
        self.register_fields({"box_centers":self.box_centers})

        n_active_boxes = self.size_t(self.nboxes_level[nlevels - 1].get())
        self.active_boxes = cl.array.to_device(queue,
                np.array(
                    range(nboxes - n_active_boxes, nboxes),
                    dtype=self.box_id_dtype))
        self.register_fields({"active_boxes":self.active_boxes})

        # FIXME: map bool in pyopencl
        #   pyopencl/compyte/dtypes.py", line 107, in dtype_to_ctype
        #     raise ValueError("unable to map dtype '%s'" % dtype)
        #     ValueError: unable to map dtype 'bool'
        self.box_is_active = cl.array.zeros(queue, nboxes, np.int8)
        self.box_is_active[nboxes - n_active_boxes:] = 1
        self.register_fields({"box_is_active":self.box_is_active})

        self.level_boxes = make_obj_array([
            cl.array.zeros(queue, 1 << (dim * l), self.box_id_dtype)
            for l in range(nlevels)])
        ibox = 0
        for l in range(nlevels):
            for b in range(len(self.level_boxes[l])):
                self.level_boxes[l][b] = ibox
                ibox += 1
        self.register_fields({"level_boxes":self.level_boxes})

        self.box_parent_ids = cl.array.zeros(queue, nboxes, self.box_id_dtype)
        self.box_child_ids = cl.array.zeros(queue, (1 << dim, nboxes),
                self.box_id_dtype)
        for l in range(nlevels):
            if l == 0:
                self.box_parent_ids[0] = 0
            else:
                multi_index_bases = tuple(
                        1 << ((dim - 1 - d) * (l-1)) for d in range(dim))
                for ilb, multi_ind in zip(range(len(self.level_boxes[l])),
                        product(range(1 << l), repeat=dim)):
                    ibox = self.box_id_dtype(self.level_boxes[l][ilb].get())
                    parent_multi_ind = tuple(ind // 2 for ind in multi_ind)
                    parent_level_id = np.sum([ind * base for ind, base
                        in zip(parent_multi_ind, multi_index_bases)])
                    self.box_parent_ids[ibox] = self.level_boxes[
                            l-1][parent_level_id]

                    child_multi_index_bases = tuple(
                            1 << (dim - 1 - d) for d in range(dim))
                    child_multi_ind = tuple(ind - pind * 2 for ind, pind
                            in zip(multi_ind, parent_multi_ind))
                    child_id = np.sum([ind * base for ind, base
                        in zip(child_multi_ind, child_multi_index_bases)])
                    self.box_child_ids[child_id][self.box_id_dtype(
                        self.level_boxes[l-1][parent_level_id].get())
                            ] = ibox
        self.register_fields({
            "box_parent_ids":self.box_parent_ids,
            "box_child_ids":self.box_child_ids})

    @property
    def dimensions(self):
        return len(self.box_centers)

    @property
    def nboxes(self):
        return len(self.box_levels)

    @property
    def nlevels(self):
        return len(self.level_boxes)

    @property
    def n_active_boxes(self):
        return len(self.active_boxes)

    def plot(self, **kwargs):
        from boxtree.visualization import BoxTreePlotter
        plotter = BoxTreePlotter(self)
        plotter.draw_tree(**kwargs)
        plotter.set_bounding_box()

    def get_box_extent(self, ibox):
        if isinstance(ibox, cl.array.Array):
            ibox = self.box_id_dtype(ibox.get())
            lev = self.box_level_dtype(self.box_levels[ibox].get())
        else:
            lev = self.box_level_dtype(self.box_levels[ibox])
        box_size = self.root_extent / (1 << lev)
        extent_low = np.zeros(self.dimensions, self.coord_dtype)
        for d in range(self.dimensions):
            if isinstance(self.box_centers[0], cl.array.Array):
                extent_low[d] = self.box_centers[
                        d, ibox].get() - 0.5 * box_size
            else:
                extent_low[d] = self.box_centers[d, ibox] - 0.5 * box_size

        extent_high = extent_low + box_size
        return extent_low, extent_high

# }}} End Data structure for a tree of boxes

# {{{ Discretize a BoxTree using quadrature

class QuadratureOnBoxTree(object):
    """Tensor-product quadrature rules applied to each active box.

    This class only holds data on template quadrature rules, while it
    responds to queries for:
        - quadrature nodes
        - quadrature weights
        - active box centers
        - active box measures (areas/volumes)

    The information is generated on the fly and is responsive to
    changes made to the underlying BoxTree object.

    .. attribute:: quadrature_formula

        The 1D template quadrature formula defined on [-1, 1].

    .. attribute:: boxtree

        The underlying BoxTree object, assumed to reside on the device,
        i.e., its get() method was not called.
    """

    def __init__(self, boxtree, quadrature_formula=None):
        """If no quadrature_formula is supplied, the trivial one is used
        sum[ f(centers) * measures ]
        """
        if not isinstance(boxtree, BoxTree):
            raise RuntimeError("Invalid boxtree")

        self.boxtree = boxtree

        if quadrature_formula is None:
            from modepy import GaussLegendreQuadrature
            self.quadrature_formula = GaussLegendreQuadrature(0)
        else:
            self.quadrature_formula = quadrature_formula

    def get_q_points(self):
        """Return quadrature points.
        """
        q_nodes = cl.array.to_device(queue, self.quadrature_formula.nodes)

        def get_knl(dim):
            quad_vars = ["iq%d" % i for i in range(dim)]
            knl = lp.make_kernel(
                "{[iabox, iaxis, QUAD_VARS]: 0<=iabox<n_active_boxes and "
                " 0<=iaxis<DIM and ".replace("DIM", str(dim))
                + " and ".join(
                    ["0<=" + qvar + "<n_q_nodes" for qvar in quad_vars])
                + "}",
                """
                <> box_id = active_boxes[iabox]
                <> box_level = box_levels[box_id]
                <> box_relative_extent = root_extent / 2 ** (box_level+1)
                result[iaxis, QUAD_VARS, iabox] = box_centers[iaxis] + (
                    if(iaxis==0, q_nodes[iq0],
                        if(iaxis==1, q_nodes[iq1], q_nodes[iq2])
                        ) * box_relative_extent)
                """
                .replace("QUAD_VARS", ", ".join(quad_vars)),
                [
                    lp.GlobalArg("result", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("active_boxes", self.boxtree.box_id_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("box_levels", self.boxtree.box_level_dtype,
                        shape="nboxes"),
                    lp.GlobalArg("q_nodes", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.ValueArg("n_active_boxes", self.boxtree.size_t),
                    lp.ValueArg("n_q_nodes", self.boxtree.size_t),
                    lp.ValueArg("nboxes", self.boxtree.size_t),
                    ],
                name="get_active_box_quad_points")

            knl = lp.split_iname(knl,
                    "iabox", 128, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_knl(self.boxtree.dimensions)(queue,
                active_boxes=self.boxtree.active_boxes,
                box_levels=self.boxtree.box_levels,
                q_weights=q_weights,
                n_active_boxes=self.boxtree.n_active_boxes,
                n_q_nodes=len(q_weights),
                nboxes=self.boxtree.nboxes)

        assert len(result) == 1
        # TODO: If the ordering is not the same as dealii, reverse the order of quad vars
        return result[0].ravel()

        pass

    def get_q_weights(self):
        """Return quadrature weights.
        """
        q_weights = cl.array.to_device(queue, self.quadrature_formula.weights)

        def get_knl(dim):
            quad_vars = ["iq%d" % i for i in range(dim)]
            knl = lp.make_kernel(
                "{[iabox, QUAD_VARS]: 0<=iabox<n_active_boxes and "
                + " and ".join(
                    ["0<=" + qvar + "<n_q_nodes" for qvar in quad_vars])
                + "}",
                """
                <> box_id = active_boxes[iabox]
                <> box_level = box_levels[box_id]
                <> box_relative_measure = root_measure / (
                                             2 ** ((box_level+1) * DIM))
                result[QUAD_VARS, iabox] = QUAD_WEIGHT * box_relative_measure
                """.replace("DIM", str(dim))
                .replace("QUAD_VARS", ", ".join(quad_vars))
                .replace("QUAD_WEIGHT", " * ".join(
                    ["q_weights[" + qvar + "]" for qvar in quad_vars]))
                ,
                [
                    lp.GlobalArg("result", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("active_boxes", self.boxtree.box_id_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("box_levels", self.boxtree.box_level_dtype,
                        shape="nboxes"),
                    lp.GlobalArg("q_weights", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.ValueArg("n_active_boxes", self.boxtree.size_t),
                    lp.ValueArg("n_q_nodes", self.boxtree.size_t),
                    lp.ValueArg("nboxes", self.boxtree.size_t),
                    ],
                name="get_active_box_quad_weights")

            knl = lp.split_iname(knl,
                    "iabox", 128, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_knl(self.boxtree.dimensions)(queue,
                active_boxes=self.boxtree.active_boxes,
                box_levels=self.boxtree.box_levels,
                q_weights=q_weights,
                n_active_boxes=self.boxtree.n_active_boxes,
                n_q_nodes=len(q_weights),
                nboxes=self.boxtree.nboxes)

        assert len(result) == 1
        # TODO: If the ordering is not the same as dealii, reverse the order of quad vars
        return result[0].ravel()

    def get_cell_centers(self, queue):
        """Return centers of active boxes.
        """
        def get_knl():
            knl = lp.make_kernel(
                "{[iabox, iaxis]: 0<=iabox<n_active_boxes and "
                                 "0<=iaxis<dims}",
                """
                <> box_id = active_boxes[iabox]
                result[iaxis, iabox] = box_centers[iaxis, box_id]
                """,
                [
                    lp.GlobalArg("result", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("active_boxes", self.boxtree.box_id_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("box_centers", self.boxtree.coord_dtype,
                        shape="(dims, nboxes)"),
                    lp.ValueArg("n_active_boxes", self.boxtree.size_t),
                    lp.ValueArg("dims", self.boxtree.size_t),
                    lp.ValueArg("nboxes", self.boxtree.size_t),
                    ],
                name="get_active_box_centers")

            knl = lp.split_iname(knl,
                    "iabox", 128, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_knl()(queue,
                active_boxes=self.boxtree.active_boxes,
                box_centers=self.boxtree.box_centers,
                n_active_boxes=self.boxtree.n_active_boxes,
                dims=self.boxtree.dimensions,
                nboxes=self.boxtree.nboxes)

        assert len(result) == 1
        return result[0]

    def get_cell_measures(self, queue):
        """Return measures of active boxes.
        This method does not call BoxTree.get_box_extent() for performance
        reasons.
        """
        def get_knl():
            knl = lp.make_kernel(
                "{[iabox]: 0<=iabox<n_active_boxes}",
                """
                <> box_id = active_boxes[iabox]
                <> box_level = box_levels[box_id]
                result[iabox] = root_measure / (2 ** (box_level * dims))
                """,
                [
                    lp.GlobalArg("result", self.boxtree.coord_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("active_boxes", self.boxtree.box_id_dtype,
                        shape=lp.auto),
                    lp.GlobalArg("box_levels", self.boxtree.box_level_dtype,
                        shape="nboxes"),
                    lp.ValueArg("root_measure", self.boxtree.coord_dtype),
                    lp.ValueArg("n_active_boxes", self.boxtree.size_t),
                    lp.ValueArg("dims", self.boxtree.size_t),
                    lp.ValueArg("nboxes", self.boxtree.size_t),
                    ],
                name="get_active_box_measures")

            knl = lp.split_iname(knl,
                    "iabox", 128, outer_tag="g.0", inner_tag="l.0")

            return knl

        evt, result = get_knl()(queue,
                active_boxes=self.boxtree.active_boxes,
                box_levels=self.boxtree.box_levels,
                root_measure=(
                    self.boxtree.root_extent ** self.boxtree.dimensions),
                n_active_boxes=self.boxtree.n_active_boxes,
                dims=self.boxtree.dimensions,
                nboxes=self.boxtree.nboxes)

        assert len(result) == 1
        return result[0]


# }}} End  Discretize a BoxTree using quadrature
