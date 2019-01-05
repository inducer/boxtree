from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Matt Wala
Copyright (C) 2018 Hao Gao
"""

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
import pyopencl as cl
import pyopencl.array  # noqa: F401
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from functools import partial
from pymbolic import var, evaluate
from pytools import memoize_method
import sys

if sys.version_info >= (3, 0):
    Template = partial(Template, strict_undefined=True)
else:
    Template = partial(Template, strict_undefined=True, disable_unicode=True)

if sys.version_info >= (3, 4):
    from abc import ABC, abstractmethod
else:
    from abc import ABCMeta, abstractmethod
    ABC = ABCMeta('ABC', (), {})


class TranslationCostModel:
    """Provides modeled costs for individual translations or evaluations."""

    def __init__(self, ncoeffs_fmm_by_level, uses_point_and_shoot):
        self.ncoeffs_fmm_by_level = ncoeffs_fmm_by_level
        self.uses_point_and_shoot = uses_point_and_shoot

    @staticmethod
    def direct():
        return var("c_p2p")

    def p2l(self, level):
        return var("c_p2l") * self.ncoeffs_fmm_by_level[level]

    def l2p(self, level):
        return var("c_l2p") * self.ncoeffs_fmm_by_level[level]

    def p2m(self, level):
        return var("c_p2m") * self.ncoeffs_fmm_by_level[level]

    def m2p(self, level):
        return var("c_m2p") * self.ncoeffs_fmm_by_level[level]

    def m2m(self, src_level, tgt_level):
        return var("c_m2m") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def l2l(self, src_level, tgt_level):
        return var("c_l2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2l(self, src_level, tgt_level):
        return var("c_m2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def e2e_cost(self, nsource_coeffs, ntarget_coeffs):
        if self.uses_point_and_shoot:
            return (
                    # Rotate the coordinate system to be z axis aligned.
                    nsource_coeffs ** (3 / 2)
                    # Translate the expansion along the z axis.
                    + nsource_coeffs ** (1 / 2) * ntarget_coeffs
                    # Rotate the coordinate system back.
                    + ntarget_coeffs ** (3 / 2))

        return nsource_coeffs * ntarget_coeffs


# {{{ translation cost model factories

def pde_aware_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation operators that make use of the
    knowledge that the potential satisfies a PDE.
    """
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])
    ncoeffs_fmm = (p_fmm + 1) ** (dim - 1)

    if dim == 3:
        uses_point_and_shoot = True
    else:
        uses_point_and_shoot = False

    return TranslationCostModel(
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=uses_point_and_shoot
    )


def taylor_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation based on Taylor expansions
    in Cartesian coordinates.
    """
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])
    ncoeffs_fmm = (p_fmm + 1) ** dim

    return TranslationCostModel(
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=False
    )

# }}}


class CostModel(ABC):
    def __init__(self,
                 translation_cost_model_factory=pde_aware_translation_cost_model,
                 calibration_params=None):
        """
        :arg translation_cost_model_factory: a function, which takes tree dimension
            and the number of tree levels as arguments, returns an object of
            :class:`TranslationCostModel`.
        """
        self.translation_cost_model_factory = translation_cost_model_factory
        if calibration_params is None:
            calibration_params = dict()
        self.calibration_params = calibration_params

    @abstractmethod
    def process_form_multipoles(self, traversal, p2m_cost):
        """Cost for forming multipole expansions of each box.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg p2m_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) representing the cost of forming the multipole
            expansion of one source at each level.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (nsource_boxes,), with each entry represents the cost of the box.
        """
        pass

    @abstractmethod
    def process_coarsen_multipoles(self, traversal, m2m_cost):
        """Cost for upward propagation.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg m2m_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels, nlevels), where the (i,j) entry represents the
            multipole-to-multipole cost from source level i to target level j.
        :return: a :class:`float`, the overall cost of upward propagation.

        Note: This method returns a number instead of an array.
        """
        pass

    @abstractmethod
    def process_direct(self, traversal, c_p2p):
        """Direct evaluation cost of each target box of *traversal*.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg c_p2p: calibration constant.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (ntarget_boxes,), with each entry represents the cost of the box.
        """
        pass

    @abstractmethod
    def process_list2(self, traversal, m2l_cost):
        """
        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg m2l_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) representing the translation cost of each level.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (ntarget_or_target_parent_boxes,), with each entry representing the cost
            of multipole-to-local translations to this box.
        """
        pass

    @abstractmethod
    def process_list3(self, traversal, m2p_cost):
        """
        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg m2p_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) where the ith entry represents the evaluation cost
            from multipole expansion at level i to a point.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (nboxes,), with each entry representing the cost of evaluating all
            targets inside this box from multipole expansions of list-3 boxes.
        """
        pass

    @abstractmethod
    def process_list4(self, traversal, p2l_cost):
        """
        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg p2l_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) where the ith entry represents the translation cost
            from a point to the local expansion at level i.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (ntarget_or_target_parent_boxes,), with each entry representing the cost
            of point-to-local translations to this box.
        """
        pass

    @abstractmethod
    def process_eval_locals(self, traversal, l2p_cost,
                            box_target_counts_nonchild=None):
        """
        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg l2p_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) where the ith entry represents the cost of evaluating
            the potential of a target in a box of level i using the box's local
            expansion.
        :arg box_target_counts_nonchild: if None, use
            traversal.tree.box_target_counts_nonchild.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            (ntarget_boxes,), the cost of evaluating the potentials of all targets
            inside this box from its local expansion.
        """
        pass

    @abstractmethod
    def process_refine_locals(self, traversal, l2l_cost):
        """Cost of downward propagation.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg l2l_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels-1,), where the ith entry represents the cost of
            tranlating local expansion from level i to level i+1.
        :return: a :class:`float`, the overall cost of downward propagation.

        Note: this method returns a number instead of an array.
        """
        pass

    @staticmethod
    @abstractmethod
    def aggregate(per_box_result):
        """Sum all entries of *per_box_result* into a number.

        :arg per_box_result: an object of :class:`numpy.ndarray` or
            :class:`pyopencl.array.Array`, the result to be sumed.
        :return: a :class:`float`, the result of the sum.
        """
        pass

    def translation_cost_from_model(self, nlevels, xlat_cost, context):
        """Evaluate translation cost from model. The result of this function can be
        used for process_* methods in this class.

        :arg nlevels: the number of tree levels.
        :arg xlat_cost: a :class:`TranslationCostModel`.
        :arg context: a :class:`dict` of parameters passed as context when
            evaluating symbolic expressions in *xlat_cost*.
        :return: a :class:`dict`, the translation cost of each step in FMM.
        """
        return {
            "p2m_cost": np.array([
                evaluate(xlat_cost.p2m(ilevel), context=context)
                for ilevel in range(nlevels)
            ], dtype=np.float64),
            "c_p2p": evaluate(xlat_cost.direct(), context=context),
            "m2l_cost": np.array([
                evaluate(xlat_cost.m2l(ilevel, ilevel), context=context)
                for ilevel in range(nlevels)
            ], dtype=np.float64),
            "m2p_cost": np.array([
                evaluate(xlat_cost.m2p(ilevel), context=context)
                for ilevel in range(nlevels)
            ], dtype=np.float64),
            "p2l_cost": np.array([
                evaluate(xlat_cost.p2l(ilevel), context=context)
                for ilevel in range(nlevels)
            ], dtype=np.float64),
            "l2p_cost": np.array([
                evaluate(xlat_cost.l2p(ilevel), context=context)
                for ilevel in range(nlevels)
            ], dtype=np.float64)
        }

    def estimate_calibration_params(self, traversals, level_to_orders,
                                    timing_results, wall_time=False):
        """
        :arg traversals: a :class:`list` of
            :class:`boxtree.traversal.FMMTraversalInfo` objects. Note each traversal
            object can reside on host or device depending on cost counting
            implemenation instead of the expansion wrangler used for executing.
        :arg level_to_orders: a :class:`list` of the same length as *traversals*.
            Each entry is a :class:`numpy.ndarray` representing the expansion order
            of different levels.
        :arg timing_results: a :class:`list` of the same length as *traversals*.
            Each entry is a :class:`dict` filled with timing data returned by
            *boxtree.fmm.drive_fmm*
        :arg wall_time: a :class:`bool`, whether to use wall time or processor time.
        :return: a :class:`dict` of calibration parameters.
        """
        nresults = len(traversals)
        assert len(level_to_orders) == nresults
        assert len(timing_results) == nresults

        _FMM_STAGE_TO_CALIBRATION_PARAMETER = {
            "form_multipoles": "c_p2m",
            "coarsen_multipoles": "c_m2m",
            "eval_direct": "c_p2p",
            "multipole_to_local": "c_m2l",
            "eval_multipoles": "c_m2p",
            "form_locals": "c_p2l",
            "refine_locals": "c_l2l",
            "eval_locals": "c_l2p"
        }

        params = set(_FMM_STAGE_TO_CALIBRATION_PARAMETER.values())

        uncalibrated_times = {}
        actual_times = {}

        for param in params:
            uncalibrated_times[param] = np.zeros(nresults)
            actual_times[param] = np.zeros(nresults)

        for icase, traversal in enumerate(traversals):
            tree = traversal.tree

            xlat_cost = self.translation_cost_model_factory(
                tree.dimensions, tree.nlevels
            )

            training_ctx = dict(
                c_l2l=1,
                c_l2p=1,
                c_m2l=1,
                c_m2m=1,
                c_m2p=1,
                c_p2l=1,
                c_p2m=1,
                c_p2p=1
            )
            for ilevel in range(tree.nlevels):
                training_ctx["p_fmm_lev%d" % ilevel] = level_to_orders[icase][ilevel]

            translation_cost = self.translation_cost_from_model(
                tree.nlevels, xlat_cost, training_ctx
            )

            uncalibrated_times["c_p2p"][icase] = self.aggregate(
                self.process_direct(traversal, translation_cost["c_p2p"])
            )

            uncalibrated_times["c_m2l"][icase] = self.aggregate(
                self.process_list2(traversal, translation_cost["m2l_cost"])
            )

            uncalibrated_times["c_m2p"][icase] = self.aggregate(
                self.process_list3(traversal, translation_cost["m2p_cost"])
            )

            uncalibrated_times["c_p2l"][icase] = self.aggregate(
                self.process_list4(traversal, translation_cost["p2l_cost"])
            )

            uncalibrated_times["c_l2p"][icase] = self.aggregate(
                self.process_eval_locals(traversal, translation_cost["l2p_cost"])
            )

        if wall_time:
            field = "wall_elapsed"
        else:
            field = "process_elapsed"

        for icase, timing_result in enumerate(timing_results):
            for param, time in timing_result.items():
                calibration_param = (
                    _FMM_STAGE_TO_CALIBRATION_PARAMETER[param])
                actual_times[calibration_param][icase] = time[field]

        result = {}

        for param in params:
            uncalibrated = uncalibrated_times[param]
            actual = actual_times[param]

            if np.allclose(uncalibrated, 0):
                result[param] = float("NaN")
                continue

            result[param] = (
                    actual.dot(uncalibrated) / uncalibrated.dot(uncalibrated))

        return result

    def __call__(self, traversal, level_to_order, params):
        """Predict cost of a new traversal object.

        :arg traversal: a :class:`boxtree.traversal.FMMTraversalInfo` object.
        :arg level_to_order: a :class:`numpy.ndarray` of shape
            (traversal.tree.nlevels,) representing the expansion orders
            of different levels.
        :arg params: the calibration parameters returned by
            *estimate_calibration_params*.
        :return: a :class:`dict`, the cost of fmm stages.
        """
        tree = traversal.tree
        result = {}

        for ilevel in range(tree.nlevels):
            params["p_fmm_lev%d" % ilevel] = level_to_order[ilevel]

        xlat_cost = self.translation_cost_model_factory(
            tree.dimensions, tree.nlevels
        )

        translation_cost = self.translation_cost_from_model(
            tree.nlevels, xlat_cost, params
        )

        result["form_multipoles"] = self.process_form_multipoles(
            traversal, translation_cost["p2m_cost"]
        )

        result["eval_direct"] = self.process_direct(
            traversal, translation_cost["c_p2p"]
        )

        result["multipole_to_local"] = self.process_list2(
            traversal, translation_cost["m2l_cost"]
        )

        result["eval_multipoles"] = self.process_list3(
            traversal, translation_cost["m2p_cost"]
        )

        result["form_locals"] = self.process_list4(
            traversal, translation_cost["p2l_cost"]
        )

        result["eval_locals"] = self.process_eval_locals(
            traversal, translation_cost["l2p_cost"]
        )

        return result


class CLCostModel(CostModel):
    """
    Note: For methods in this class, argument *traversal* should live on device
        memory.
    """
    def __init__(self, queue,
                 translation_cost_model_factory=pde_aware_translation_cost_model,
                 calibration_params=None):
        self.queue = queue
        super(CLCostModel, self).__init__(
            translation_cost_model_factory, calibration_params
        )

    # {{{ form multipoles

    @memoize_method
    def process_form_multipoles_knl(self, box_id_dtype, particle_id_dtype,
                                    box_level_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                double *np2m,
                ${box_id_t} *source_boxes,
                ${particle_id_t} *box_source_counts_nonchild,
                ${box_level_t} *box_levels,
                double *p2m_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} box_idx = source_boxes[i];
                ${particle_id_t} nsources = box_source_counts_nonchild[box_idx];
                ${box_level_t} ilevel = box_levels[box_idx];
                np2m[i] = nsources * p2m_cost[ilevel];
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            name="process_form_multipoles"
        )

    def process_form_multipoles(self, traversal, p2m_cost):
        tree = traversal.tree
        np2m = cl.array.zeros(
            self.queue, len(traversal.source_boxes), dtype=np.float64
        )

        process_form_multipoles_knl = self.process_form_multipoles_knl(
            tree.box_id_dtype, tree.particle_id_dtype, tree.box_level_dtype
        )

        process_form_multipoles_knl(
            np2m,
            traversal.source_boxes,
            tree.box_source_counts_nonchild,
            tree.box_levels,
            p2m_cost
        )

        return np2m

    # }}}

    # {{{ propagate multipoles upward

    def process_coarsen_multipoles_knl(self, ndimensions, box_id_dtype,
                                       box_level_dtype, nlevels):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                ${box_id_t} *source_parent_boxes,
                ${box_level_t} *box_levels,
                double *m2m_cost,
                double *nm2m,
                % for i in range(2**ndimensions):
                    % if i == 2**ndimensions - 1:
                        ${box_id_t} *box_child_ids_${i}
                    % else:
                        ${box_id_t} *box_child_ids_${i},
                    % endif
                % endfor
            """).render(
                ndimensions=ndimensions,
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} box_idx = source_parent_boxes[i];
                ${box_level_t} target_level = box_levels[box_idx];
                if(target_level <= 1) {
                    nm2m[i] = 0.0;
                } else {
                    ${box_level_t} source_level = target_level + 1;
                    int nchild = 0;
                    % for i in range(2**ndimensions):
                        if(box_child_ids_${i}[box_idx])
                            nchild += 1;
                    % endfor
                    nm2m[i] = nchild * m2m_cost[
                        source_level * ${nlevels} + target_level
                    ];
                }
            """).render(
                ndimensions=ndimensions,
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype),
                nlevels=nlevels
            ),
            name="process_coarsen_multipoles"
        )

    def process_coarsen_multipoles(self, traversal, m2m_cost):
        tree = traversal.tree
        nm2m = cl.array.zeros(
            self.queue, len(traversal.source_parent_boxes), dtype=np.float64
        )

        process_coarsen_multipoles_knl = self.process_coarsen_multipoles_knl(
            tree.dimensions, tree.box_id_dtype, tree.box_level_dtype, tree.nlevels
        )

        process_coarsen_multipoles_knl(
            traversal.source_parent_boxes,
            tree.box_levels,
            m2m_cost,
            nm2m,
            *tree.box_child_ids,
            queue=self.queue
        )

        return self.aggregate(nm2m)

    # }}}

    # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

    @memoize_method
    def process_direct_knl(self, particle_id_dtype, box_id_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template("""
                double *direct_by_itgt_box,
                ${box_id_t} *source_boxes_starts,
                ${box_id_t} *source_boxes_lists,
                ${particle_id_t} *box_source_counts_nonchild,
                ${particle_id_t} *box_target_counts_nonchild,
                ${box_id_t} *target_boxes,
                double c_p2p
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            Template(r"""
                ${particle_id_t} nsources = 0;
                ${box_id_t} source_boxes_start_idx = source_boxes_starts[i];
                ${box_id_t} source_boxes_end_idx = source_boxes_starts[i + 1];

                for(${box_id_t} cur_source_boxes_idx = source_boxes_start_idx;
                    cur_source_boxes_idx < source_boxes_end_idx;
                    cur_source_boxes_idx++)
                {
                    ${box_id_t} cur_source_box = source_boxes_lists[
                        cur_source_boxes_idx
                    ];
                    nsources += box_source_counts_nonchild[cur_source_box];
                }

                ${particle_id_t} ntargets = box_target_counts_nonchild[
                    target_boxes[i]
                ];

                direct_by_itgt_box[i] += (nsources * ntargets * c_p2p);
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            name="process_direct"
        )

    def process_direct(self, traversal, c_p2p):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)
        particle_id_dtype = tree.particle_id_dtype
        box_id_dtype = tree.box_id_dtype

        count_direct_interaction_knl = self.process_direct_knl(
            particle_id_dtype, box_id_dtype
        )

        direct_by_itgt_box_dev = cl.array.zeros(
            self.queue, (ntarget_boxes,), dtype=np.float64
        )

        # List 1
        count_direct_interaction_knl(
            direct_by_itgt_box_dev,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            traversal.tree.box_source_counts_nonchild,
            traversal.tree.box_target_counts_nonchild,
            traversal.target_boxes,
            c_p2p
        )

        # List 3 close
        if traversal.from_sep_close_smaller_starts is not None:
            count_direct_interaction_knl(
                direct_by_itgt_box_dev,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                traversal.tree.box_source_counts_nonchild,
                traversal.tree.box_target_counts_nonchild,
                traversal.target_boxes,
                c_p2p
            )

        # List 4 close
        if traversal.from_sep_close_bigger_starts is not None:
            count_direct_interaction_knl(
                direct_by_itgt_box_dev,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                traversal.tree.box_source_counts_nonchild,
                traversal.tree.box_target_counts_nonchild,
                traversal.target_boxes,
                c_p2p
            )

        return direct_by_itgt_box_dev

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    @memoize_method
    def process_list2_knl(self, box_id_dtype, box_level_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                double *nm2l,
                ${box_id_t} *target_or_target_parent_boxes,
                ${box_id_t} *from_sep_siblings_starts,
                ${box_level_t} *box_levels,
                double *m2l_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} start = from_sep_siblings_starts[i];
                ${box_id_t} end = from_sep_siblings_starts[i+1];
                ${box_level_t} ilevel = box_levels[target_or_target_parent_boxes[i]];

                nm2l[i] += (end - start) * m2l_cost[ilevel];
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            name="process_list2"
        )

    def process_list2(self, traversal, m2l_cost):
        tree = traversal.tree
        box_id_dtype = tree.box_id_dtype
        box_level_dtype = tree.box_level_dtype

        ntarget_or_target_parent_boxes = len(traversal.target_or_target_parent_boxes)
        nm2l = cl.array.zeros(
            self.queue, (ntarget_or_target_parent_boxes,), dtype=np.float64
        )

        process_list2_knl = self.process_list2_knl(box_id_dtype, box_level_dtype)
        process_list2_knl(
            nm2l,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            tree.box_levels,
            m2l_cost
        )

        return nm2l

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    @memoize_method
    def process_list3_knl(self, box_id_dtype, particle_id_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                ${box_id_t} *target_boxes_sep_smaller,
                ${box_id_t} *sep_smaller_start,
                ${particle_id_t} *box_target_counts_nonchild,
                double m2p_cost_current_level,
                double *nm2p
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            Template(r"""
                ${box_id_t} target_box = target_boxes_sep_smaller[i];
                ${box_id_t} start = sep_smaller_start[i];
                ${box_id_t} end = sep_smaller_start[i+1];
                ${particle_id_t} ntargets = box_target_counts_nonchild[target_box];
                nm2p[target_box] += (
                    ntargets * (end - start) * m2p_cost_current_level
                );
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            name="process_list3"
        )

    def process_list3(self, traversal, m2p_cost):
        tree = traversal.tree
        nm2p = cl.array.zeros(self.queue, tree.nboxes, dtype=np.float64)

        process_list3_knl = self.process_list3_knl(
            tree.box_id_dtype, tree.particle_id_dtype
        )

        for ilevel, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):
            process_list3_knl(
                traversal.target_boxes_sep_smaller_by_source_level[ilevel],
                sep_smaller_list.starts,
                tree.box_target_counts_nonchild,
                m2p_cost[ilevel].get().reshape(-1)[0],
                nm2p,
                queue=self.queue
            )
            self.queue.finish()

        return nm2p

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    @memoize_method
    def process_list4_knl(self, box_id_dtype, particle_id_dtype, box_level_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                double *nm2p,
                ${box_id_t} *from_sep_bigger_starts,
                ${box_id_t} *from_sep_bigger_lists,
                ${particle_id_t} *box_source_counts_nonchild,
                ${box_level_t} *box_levels,
                double *p2l_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} start = from_sep_bigger_starts[i];
                ${box_id_t} end = from_sep_bigger_starts[i+1];
                for(${box_id_t} idx=start; idx < end; idx++) {
                    ${box_id_t} src_ibox = from_sep_bigger_lists[idx];
                    ${particle_id_t} nsources = box_source_counts_nonchild[src_ibox];
                    ${box_level_t} ilevel = box_levels[src_ibox];
                    nm2p[i] += nsources * p2l_cost[ilevel];
                }
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            name="process_list4"
        )

    def process_list4(self, traversal, p2l_cost):
        tree = traversal.tree
        target_or_target_parent_boxes = traversal.target_or_target_parent_boxes
        nm2p = cl.array.zeros(
            self.queue, len(target_or_target_parent_boxes), dtype=np.float64
        )

        process_list4_knl = self.process_list4_knl(
            tree.box_id_dtype, tree.particle_id_dtype, tree.box_level_dtype
        )

        process_list4_knl(
            nm2p,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            tree.box_source_counts_nonchild,
            tree.box_levels,
            p2l_cost
        )

        return nm2p

    # }}}

    # {{{ evaluate local expansions at targets

    @memoize_method
    def process_eval_locals_knl(self, box_id_dtype, particle_id_dtype,
                                box_level_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                double *neval_locals,
                ${box_id_t} *target_boxes,
                ${particle_id_t} *box_target_counts_nonchild,
                ${box_level_t} *box_levels,
                double *l2p_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            Template(r"""
                ${box_id_t} box_idx = target_boxes[i];
                ${particle_id_t} ntargets = box_target_counts_nonchild[box_idx];
                ${box_level_t} ilevel = box_levels[box_idx];
                neval_locals[i] += ntargets * l2p_cost[ilevel];
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_level_t=dtype_to_ctype(box_level_dtype)
            ),
            name="process_eval_locals"
        )

    def process_eval_locals(self, traversal, l2p_cost,
                            box_target_counts_nonchild=None):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)
        neval_locals = cl.array.zeros(self.queue, ntarget_boxes, dtype=np.float64)
        if box_target_counts_nonchild is None:
            box_target_counts_nonchild = traversal.tree.box_target_counts_nonchild

        process_eval_locals_knl = self.process_eval_locals_knl(
            tree.box_id_dtype, tree.particle_id_dtype, tree.box_level_dtype
        )

        process_eval_locals_knl(
            neval_locals,
            traversal.target_boxes,
            box_target_counts_nonchild,
            tree.box_levels,
            l2p_cost
        )

        return neval_locals

    # }}}

    # {{{ propogate locals downward

    @memoize_method
    def process_refine_locals_knl(self, box_id_dtype):
        from pyopencl.reduction import ReductionKernel
        return ReductionKernel(
            self.queue.context,
            np.float64,
            neutral="0.0",
            reduce_expr="a+b",
            map_expr=r"""
                (level_start_target_or_target_parent_box_nrs[i + 1]
                 - level_start_target_or_target_parent_box_nrs[i])
                 * l2l_cost[i - 1]
            """,
            arguments=Template(r"""
                ${box_id_t} *level_start_target_or_target_parent_box_nrs,
                double *l2l_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            name="process_refine_locals"
        )

    def process_refine_locals(self, traversal, l2l_cost):
        tree = traversal.tree
        process_refine_locals_knl = self.process_refine_locals_knl(tree.box_id_dtype)

        level_start_target_or_target_parent_box_nrs = cl.array.to_device(
            self.queue, traversal.level_start_target_or_target_parent_box_nrs
        )

        cost = process_refine_locals_knl(
            level_start_target_or_target_parent_box_nrs,
            l2l_cost,
            range=range(1, tree.nlevels)
        ).get()

        return cost

    # }}}

    @staticmethod
    def aggregate(per_box_result):
        return cl.array.sum(per_box_result).get().reshape(-1)[0]

    def translation_cost_from_model(self, nlevels, xlat_cost, context):
        translation_costs = super(CLCostModel, self).translation_cost_from_model(
            nlevels, xlat_cost, context
        )

        for name in translation_costs:
            if not isinstance(translation_costs[name], np.ndarray):
                continue
            translation_costs[name] = cl.array.to_device(
                self.queue, translation_costs[name]
            )

        return translation_costs


class PythonCostModel(CostModel):
    def process_form_multipoles(self, traversal, p2m_cost):
        tree = traversal.tree
        np2m = np.zeros(len(traversal.source_boxes), dtype=np.float64)

        for ilevel in range(tree.nlevels):
            start, stop = traversal.level_start_source_box_nrs[ilevel:ilevel + 2]
            for isrc_box, src_ibox in enumerate(
                    traversal.source_boxes[start:stop], start):
                nsources = tree.box_source_counts_nonchild[src_ibox]
                np2m[isrc_box] = nsources * p2m_cost[ilevel]

        return np2m

    def process_direct(self, traversal, c_p2p):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)

        # target box index -> nsources
        direct_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.float64)

        for itgt_box in range(ntarget_boxes):
            nsources = 0

            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources += tree.box_source_counts_nonchild[src_ibox]

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                    traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2]
                )
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                    traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2]
                )
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            ntargets = tree.box_target_counts_nonchild[
                traversal.target_boxes[itgt_box]
            ]
            direct_by_itgt_box[itgt_box] += (nsources * ntargets * c_p2p)

        return direct_by_itgt_box

    def process_list2(self, traversal, m2l_cost):
        tree = traversal.tree
        ntarget_or_target_parent_boxes = len(traversal.target_or_target_parent_boxes)
        nm2l = np.zeros(ntarget_or_target_parent_boxes, dtype=np.float64)

        for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
            start, end = traversal.from_sep_siblings_starts[itgt_box:itgt_box+2]

            ilevel = tree.box_levels[tgt_ibox]
            nm2l[itgt_box] += m2l_cost[ilevel] * (end - start)

        return nm2l

    def process_list3(self, traversal, m2p_cost):
        tree = traversal.tree
        nm2p = np.zeros(tree.nboxes, dtype=np.float64)

        for ilevel, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):
            for itgt_box, tgt_ibox in enumerate(
                    traversal.target_boxes_sep_smaller_by_source_level[ilevel]):
                ntargets = tree.box_target_counts_nonchild[tgt_ibox]
                start, end = sep_smaller_list.starts[itgt_box:itgt_box + 2]
                nm2p[tgt_ibox] += ntargets * (end - start) * m2p_cost[ilevel]

        return nm2p

    def process_list4(self, traversal, p2l_cost):
        tree = traversal.tree
        target_or_target_parent_boxes = traversal.target_or_target_parent_boxes
        nm2p = np.zeros(len(target_or_target_parent_boxes), dtype=np.float64)

        for itgt_box in range(len(target_or_target_parent_boxes)):
            start, end = traversal.from_sep_bigger_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.from_sep_bigger_lists[start:end]:
                nsources = tree.box_source_counts_nonchild[src_ibox]
                ilevel = tree.box_levels[src_ibox]
                nm2p[itgt_box] += nsources * p2l_cost[ilevel]

        return nm2p

    def process_eval_locals(self, traversal, l2p_cost,
                            box_target_counts_nonchild=None):
        tree = traversal.tree
        ntarget_boxes = len(traversal.target_boxes)
        neval_locals = np.zeros(ntarget_boxes, dtype=np.float64)
        if box_target_counts_nonchild is None:
            box_target_counts_nonchild = tree.box_target_counts_nonchild

        for target_lev in range(tree.nlevels):
            start, stop = traversal.level_start_target_box_nrs[
                    target_lev:target_lev+2]
            for itgt_box, tgt_ibox in enumerate(
                    traversal.target_boxes[start:stop], start):
                neval_locals[itgt_box] += (box_target_counts_nonchild[tgt_ibox]
                                           * l2p_cost[target_lev])

        return neval_locals

    def process_coarsen_multipoles(self, traversal, m2m_cost):
        tree = traversal.tree
        result = 0.0

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            cost = m2m_cost[source_level, target_level]

            nmultipoles = 0
            start, stop = traversal.level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in traversal.source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        nmultipoles += 1

            result += cost * nmultipoles

        return result

    def process_refine_locals(self, traversal, l2l_cost):
        tree = traversal.tree
        result = 0.0

        for target_lev in range(1, tree.nlevels):
            start, stop = traversal.level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            source_lev = target_lev - 1
            result += (stop-start) * l2l_cost[source_lev]

        return result

    @staticmethod
    def aggregate(per_box_result):
        return np.sum(per_box_result)
