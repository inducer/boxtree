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

import numpy as np


# {{{ utilities

def int_to_roman(inp):
    """
    Convert an integer to Roman numerals.
    """
    # stolen from
    # https://code.activestate.com/recipes/81611-roman-numerals/

    if not isinstance(inp, int):
        raise TypeError("expected integer, got %s" % type(inp))
    if inp == 0:
        return "Z"
    if not 0 < inp < 4000:
        raise ValueError("Argument must be between 1 and 3999 (got %d)" % inp)
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")
    result = ""
    for i in range(len(ints)):
        count = int(inp / ints[i])
        result += nums[i] * count
        inp -= ints[i] * count
    return result

# }}}


# {{{ tree plotting

class TreePlotter:
    """Assumes that the tree has data living on the host.
    See :meth:`boxtree.Tree.get`.
    """

    def __init__(self, tree):
        self.tree = tree

    def draw_tree(self, **kwargs):
        if self.tree.dimensions != 2:
            raise NotImplementedError("can only plot 2D trees for now")

        fill = kwargs.pop("fill", False)
        edgecolor = kwargs.pop("edgecolor", "black")
        kwargs["fill"] = fill
        kwargs["edgecolor"] = edgecolor

        for ibox in range(self.tree.nboxes):
            self.draw_box(ibox, **kwargs)

    def set_bounding_box(self):
        import matplotlib.pyplot as pt
        bbox_min, bbox_max = self.tree.bounding_box
        pt.xlim(bbox_min[0], bbox_max[0])
        pt.ylim(bbox_min[1], bbox_max[1])

        pt.gca().set_aspect("equal")

    def draw_box(self, ibox, **kwargs):
        """
        :arg kwargs: keyword arguments to pass on to
            :class:`matplotlib.patches.PathPatch`,
            e.g. `facecolor="red", edgecolor="yellow", alpha=0.5`
        """

        el, eh = self.tree.get_box_extent(ibox)

        shrink_factor = kwargs.pop("shrink_factor", 0)
        if shrink_factor:
            center = 0.5*(el+eh)
            el += (center-el)*shrink_factor
            eh += (center-eh)*shrink_factor

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

        for ibox in range(tree.nboxes):
            x, y = tree.box_centers[:, ibox]
            lev = int(tree.box_levels[ibox])
            pt.text(x, y, str(ibox), fontsize=20*1.15**(-lev),
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.5, lw=0))

    def get_tikz_for_tree(self):
        if self.tree.dimensions != 2:
            raise NotImplementedError("can only plot 2D trees for now")

        lines = []

        lines.append(r"\def\nboxes{%d}" % self.tree.nboxes)
        lines.append(r"\def\lastboxnr{%d}" % (self.tree.nboxes-1))
        for ibox in range(self.tree.nboxes):
            el, eh = self.tree.get_box_extent(ibox)

            c = self.tree.box_centers[:, ibox]

            lines.append(
                    r"\coordinate (boxl%d) at (%r, %r);"
                    % (ibox, float(el[0]), float(el[1])))
            lines.append(
                    r"\coordinate (boxh%d) at (%r, %r);"
                    % (ibox, float(eh[0]), float(eh[1])))
            lines.append(
                    r"\coordinate (boxc%d) at (%r, %r);"
                    % (ibox, float(c[0]), float(c[1])))
            lines.append(
                    r"\def\boxsize%s{%r}"
                    % (int_to_roman(ibox), float(eh[0]-el[0])))
            lines.append(
                    r"\def\boxlevel%s{%r}"
                    % (int_to_roman(ibox), self.tree.box_levels[ibox]))

        lines.append(
                r"\def\boxpath#1{(boxl#1) rectangle (boxh#1)}")
        lines.append(
                r"\def\drawboxes{"
                r"\foreach \ibox in {0,...,\lastboxnr}{"
                r"\draw \boxpath{\ibox};"
                r"}}")
        lines.append(
                r"\def\drawboxnrs{"
                r"\foreach \ibox in {0,...,\lastboxnr}{"
                r"\node [font=\tiny] at (boxc\ibox) {\ibox};"
                r"}}")
        return "\n".join(lines)

# }}}


# {{{ traversal plotting

def _draw_box_list(tree_plotter, ibox, starts, lists, key_to_box=None, **kwargs):
    default_facecolor = "blue"

    if key_to_box is not None:
        ind, = np.where(key_to_box == ibox)
        if len(ind):
            key, = ind
        else:
            # indicate empty list
            actual_kwargs = {
                    "edgecolor": getattr(kwargs, "facecolor", default_facecolor),
                    "fill": False,
                    "alpha": 0.5,
                    "shrink_factor": -0.1+0.1*np.random.rand(),
                    }
            tree_plotter.draw_box(ibox, **actual_kwargs)
            return
    else:
        key = ibox

    start, end = starts[key:key+2]
    if start == end:
        return

    actual_kwargs = {
            "facecolor": default_facecolor,
            "linewidth": 0,
            "alpha": 0.5,
            "shrink_factor": 0.1 + np.random.rand()*0.2,
            }
    actual_kwargs.update(kwargs)
    print(actual_kwargs["facecolor"], ibox, lists[start:end])
    for jbox in lists[start:end]:
        tree_plotter.draw_box(jbox, **actual_kwargs)


def draw_same_level_non_well_sep_boxes(tree_plotter, traversal, ibox):
    tree_plotter.draw_box(ibox, facecolor="red",
            alpha=0.5)

    # same-level non-well-sep
    _draw_box_list(tree_plotter, ibox,
            traversal.same_level_non_well_sep_boxes_starts,
            traversal.same_level_non_well_sep_boxes_lists,
            facecolor="green")


def draw_box_lists(tree_plotter, traversal, ibox):
    tree_plotter.draw_box(ibox, facecolor="red",
            alpha=0.5)

    # from near neighbors ("list 1")
    _draw_box_list(tree_plotter, ibox,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            key_to_box=traversal.target_boxes,
            facecolor="green")

    # from well-separated siblings (list 2)
    _draw_box_list(tree_plotter, ibox,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            key_to_box=traversal.target_or_target_parent_boxes,
            facecolor="blue")

    # from separated smaller (list 3)
    for ilev in range(tree_plotter.tree.nlevels):
        _draw_box_list(tree_plotter, ibox,
                traversal.from_sep_smaller_by_level[ilev].starts,
                traversal.from_sep_smaller_by_level[ilev].lists,
                key_to_box=traversal.target_boxes_sep_smaller_by_source_level[ilev],
                facecolor="orange")

    # list 3 close
    if traversal.from_sep_close_smaller_starts is not None:
        _draw_box_list(tree_plotter, ibox,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                key_to_box=traversal.target_boxes,
                facecolor="orange", hatch=".")

    # from separated bigger (list 4)
    _draw_box_list(tree_plotter, ibox,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            key_to_box=traversal.target_or_target_parent_boxes,
            facecolor="purple")

    # list 4 close
    if traversal.from_sep_close_bigger_starts is not None:
        _draw_box_list(tree_plotter, ibox,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                key_to_box=traversal.target_boxes,
                facecolor="purple", hatch=".")

# }}}

# vim: filetype=pyopencl:fdm=marker
