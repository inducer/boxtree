from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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




class TreePlotter:
    """Assumes that the tree has data living on the host.
    See :meth:`boxtree.Tree.get`.
    """

    def __init__(self, tree):
        self.tree = tree

    def draw_tree(self, **kwargs):
        if self.tree.dimensions != 2:
            raise NotImplementedError("can only plot 2D trees for now")

        for ibox in xrange(self.tree.nboxes):
            self.draw_box(ibox, **kwargs)

    def set_bounding_box(self):
        import matplotlib.pyplot as pt
        bbox_min, bbox_max = self.tree.bounding_box
        pt.xlim(bbox_min[0], bbox_max[0])
        pt.ylim(bbox_min[1], bbox_max[1])

    def draw_box(self, ibox, **kwargs):
        """
        :arg kwargs: keyword arguments to pass on to :class:`matplotlib.patches.PathPatch`,
            e.g. `facecolor='red', edgecolor='yellow', alpha=0.5`
        """

        el, eh = self.tree.get_box_extent(ibox)

        import matplotlib.pyplot as pt
        import matplotlib.patches as mpatches
        from matplotlib.path import Path

        pathdata = [
            (Path.MOVETO, (el[0], el[1])),
            (Path.LINETO, (eh[0], el[1])),
            (Path.LINETO, (eh[0], eh[1])),
            (Path.LINETO, (el[0], eh[1])),
            (Path.CLOSEPOLY, (el[0], el[1])),
            ]

        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        pt.gca().add_patch(patch)

    def draw_box_numbers(self):
        import matplotlib.pyplot as pt

        tree = self.tree

        for ibox in xrange(tree.nboxes):
            x, y = tree.box_centers[:, ibox]
            lev = int(tree.box_levels[ibox])
            pt.text(x, y, str(ibox), fontsize=20*1.15**(-lev),
                    ha="center", va="center",
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))




# vim: filetype=pyopencl:fdm=marker
