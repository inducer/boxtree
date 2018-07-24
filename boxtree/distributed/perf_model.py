from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner \
                 Copyright (C) 2018 Hao Gao"

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

import pyopencl as cl
import numpy as np
from collections import namedtuple
from pyopencl.clrandom import PhiloxGenerator


def generate_random_traversal(context, nsources, ntargets, dims, dtype):
    with cl.CommandQueue(context) as queue:
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = p_normal(queue, ntargets, dims, dtype, seed=18)

        rng = PhiloxGenerator(context, seed=22)
        target_radii = rng.uniform(
            queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

        from boxtree import TreeBuilder
        tb = TreeBuilder(context)
        tree, _ = tb(queue, sources, targets=targets, target_radii=target_radii,
                     stick_out_factor=0.25, max_particles_in_box=30, debug=True)

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(context, well_sep_is_n_away=2)
        d_trav, _ = tg(queue, tree, debug=True)
        trav = d_trav.get(queue=queue)

        return trav


FMMParameters = namedtuple(
    "FMMParameters",
    ['ncoeffs_fmm_by_level',
     'translation_source_power',
     'translation_target_power',
     'translation_max_power']
)


class PerformanceCounter:

    def __init__(self, traversal, wrangler, uses_pde_expansions):
        self.traversal = traversal
        self.wrangler = wrangler
        self.uses_pde_expansions = uses_pde_expansions

        self.parameters = self.get_fmm_parameters(
            traversal.tree.dimensions,
            uses_pde_expansions,
            wrangler.level_nterms
        )

    @staticmethod
    def xlat_cost(p_source, p_target, parameters):
        """
        :param p_source: A numpy array of numbers of source terms
        :return: The same shape as *p_source*
        """
        return (
                p_source ** parameters.translation_source_power
                * p_target ** parameters.translation_target_power
                * np.maximum(p_source, p_target) ** parameters.translation_max_power
        )

    @staticmethod
    def get_fmm_parameters(dimensions, use_pde_expansions, level_nterms):
        if use_pde_expansions:
            ncoeffs_fmm_by_level = level_nterms ** (dimensions - 1)

            if dimensions == 2:
                translation_source_power = 1
                translation_target_power = 1
                translation_max_power = 0
            elif dimensions == 3:
                # Based on a reading of FMMlib, i.e. a point-and-shoot FMM.
                translation_source_power = 0
                translation_target_power = 0
                translation_max_power = 3
            else:
                raise ValueError("Don't know how to estimate expansion complexities "
                                 "for dimension %d" % dimensions)

        else:
            ncoeffs_fmm_by_level = level_nterms ** dimensions

            translation_source_power = dimensions
            translation_target_power = dimensions
            translation_max_power = 0

        return FMMParameters(
            ncoeffs_fmm_by_level=ncoeffs_fmm_by_level,
            translation_source_power=translation_source_power,
            translation_target_power=translation_target_power,
            translation_max_power=translation_max_power
        )

    def count_nsources_by_level(self):
        """
        :return: A numpy array of share (tree.nlevels,) such that the ith index
            documents the number of sources on level i.
        """
        tree = self.traversal.tree

        nsources_by_level = np.empty((tree.nlevels,), dtype=np.int32)

        for ilevel in range(tree.nlevels):
            start_ibox = tree.level_start_box_nrs[ilevel]
            end_ibox = tree.level_start_box_nrs[ilevel + 1]
            count = 0

            for ibox in range(start_ibox, end_ibox):
                count += tree.box_source_counts_nonchild[ibox]

            nsources_by_level[ilevel] = count

        return nsources_by_level

    def count_nters_fmm_total(self):
        """
        :return: total number of terms formed across all levels during form_multipole
        """
        nsources_by_level = self.count_nsources_by_level()

        ncoeffs_fmm_by_level = self.parameters.ncoeffs_fmm_by_level

        nterms_fmm_total = np.sum(nsources_by_level * ncoeffs_fmm_by_level)

        return nterms_fmm_total

    def count_direct(self, use_global_idx=False):
        """
        :return: If *use_global_idx* is True, return a numpy array of shape
            (tree.nboxes,) such that the ith entry represents the workload from
            direct evaluation on box i. If *use_global_idx* is False, return a numpy
            array of shape (ntarget_boxes,) such that the ith entry represents the
            workload on *target_boxes* i.
        """
        traversal = self.traversal
        tree = traversal.tree

        if use_global_idx:
            direct_workload = np.zeros((tree.nboxes,), dtype=np.int64)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            direct_workload = np.zeros((ntarget_boxes,), dtype=np.int64)

        for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
            ntargets = tree.box_target_counts_nonchild[tgt_ibox]
            nsources = 0

            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]

            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                    traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                    traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            count = nsources * ntargets

            if use_global_idx:
                direct_workload[tgt_ibox] = count
            else:
                direct_workload[itgt_box] = count

        return direct_workload

    def count_m2l(self, use_global_idx=False):
        """
        :return: If *use_global_idx* is True, return a numpy array of shape
            (tree.nboxes,) such that the ith entry represents the workload from
            multipole to local expansion on box i. If *use_global_idx* is False,
            return a numpy array of shape (ntarget_or_target_parent_boxes,) such that
            the ith entry represents the workload on *target_or_target_parent_boxes*
            i.
        """
        trav = self.traversal
        wrangler = self.wrangler
        parameters = self.parameters

        ntarget_or_target_parent_boxes = len(trav.target_or_target_parent_boxes)

        if use_global_idx:
            nm2l = np.zeros((trav.tree.nboxes,), dtype=np.intp)
        else:
            nm2l = np.zeros((ntarget_or_target_parent_boxes,), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
            start, end = trav.from_sep_siblings_starts[itgt_box:itgt_box+2]
            from_sep_siblings_level = trav.tree.box_levels[
                trav.from_sep_siblings_lists[start:end]
            ]

            if start == end:
                continue

            tgt_box_level = trav.tree.box_levels[tgt_ibox]

            from_sep_siblings_nterms = wrangler.level_nterms[from_sep_siblings_level]
            tgt_box_nterms = wrangler.level_nterms[tgt_box_level]

            from_sep_siblings_costs = self.xlat_cost(
                from_sep_siblings_nterms, tgt_box_nterms, parameters)

            if use_global_idx:
                nm2l[tgt_ibox] += np.sum(from_sep_siblings_costs)
            else:
                nm2l[itgt_box] += np.sum(from_sep_siblings_costs)

        return nm2l

    def count_m2p(self, use_global_idx=False):
        trav = self.traversal
        tree = trav.tree

        if use_global_idx:
            nm2p = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            nm2p = np.zeros((len(trav.target_boxes),), dtype=np.intp)

        for ilevel, sep_smaller_list in enumerate(trav.from_sep_smaller_by_level):
            ncoeffs_fmm_cur_level = self.parameters.ncoeffs_fmm_by_level[ilevel]
            tgt_box_list = trav.target_boxes_sep_smaller_by_source_level[ilevel]

            for itgt_box, tgt_ibox in enumerate(tgt_box_list):
                ntargets = tree.box_target_counts_nonchild[tgt_ibox]

                start, end = sep_smaller_list.starts[itgt_box:itgt_box + 2]

                workload = (end - start) * ntargets * ncoeffs_fmm_cur_level

                if use_global_idx:
                    nm2p[tgt_ibox] += workload
                else:
                    nm2p[sep_smaller_list.nonempty_indices[itgt_box]] += workload

        return nm2p

    def count_p2l(self, use_global_idx=False):
        trav = self.traversal
        tree = trav.tree
        parameters = self.parameters

        if use_global_idx:
            np2l = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            np2l = np.zeros(len(trav.target_or_target_parent_boxes), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
            tgt_box_level = trav.tree.box_levels[tgt_ibox]
            ncoeffs = parameters.ncoeffs_fmm_by_level[tgt_box_level]

            start, end = trav.from_sep_bigger_starts[itgt_box:itgt_box + 2]

            np2l_sources = 0
            for src_ibox in trav.from_sep_bigger_lists[start:end]:
                np2l_sources += tree.box_source_counts_nonchild[src_ibox]

            if use_global_idx:
                np2l[tgt_ibox] = np2l_sources * ncoeffs
            else:
                np2l[itgt_box] = np2l_sources * ncoeffs

        return np2l

    def count_eval_part(self, use_global_idx=False):
        trav = self.traversal
        tree = trav.tree
        parameters = self.parameters

        if use_global_idx:
            neval_part = np.zeros(tree.nboxes, dtype=np.intp)
        else:
            neval_part = np.zeros(len(trav.target_boxes), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(trav.target_boxes):
            ntargets = tree.box_target_counts_nonchild[tgt_ibox]
            tgt_box_level = trav.tree.box_levels[tgt_ibox]
            ncoeffs_fmm = parameters.ncoeffs_fmm_by_level[tgt_box_level]

            if use_global_idx:
                neval_part[tgt_ibox] = ntargets * ncoeffs_fmm
            else:
                neval_part[itgt_box] = ntargets * ncoeffs_fmm

        return neval_part


class PerformanceModel:

    def __init__(self, cl_context, wrangler_factory, uses_pde_expansions, drive_fmm):
        self.cl_context = cl_context
        self.wrangler_factory = wrangler_factory
        self.uses_pde_expansions = uses_pde_expansions
        self.drive_fmm = drive_fmm

        self.time_result = []

        from pyopencl.clrandom import PhiloxGenerator
        self.rng = PhiloxGenerator(cl_context)

    def time_performance(self, traversal):
        wrangler = self.wrangler_factory(traversal.tree)

        counter = PerformanceCounter(traversal, wrangler, self.uses_pde_expansions)

        # Record useful metadata for assembling performance data
        timing_data = {
            "nterms_fmm_total": counter.count_nters_fmm_total(),
            "direct_workload": np.sum(counter.count_direct()),
            "direct_nsource_boxes": traversal.neighbor_source_boxes_starts[-1],
            "m2l_workload": np.sum(counter.count_m2l()),
            "m2p_workload": np.sum(counter.count_m2p()),
            "p2l_workload": np.sum(counter.count_p2l()),
            "eval_part_workload": np.sum(counter.count_eval_part())
        }

        # Generate random source weights
        with cl.CommandQueue(self.cl_context) as queue:
            source_weights = self.rng.uniform(
                queue,
                traversal.tree.nsources,
                traversal.tree.coord_dtype
            ).get()

        # Time a FMM run
        self.drive_fmm(traversal, wrangler, source_weights, timing_data=timing_data)

        self.time_result.append(timing_data)

    def form_multipoles_model(self, wall_time=True):
        return self.linear_regression(
            "form_multipoles", ["nterms_fmm_total"],
            wall_time=wall_time)

    def eval_direct_model(self, wall_time=True):
        return self.linear_regression(
            "eval_direct",
            ["direct_workload", "direct_nsource_boxes"],
            wall_time=wall_time)

    def multipole_to_local_model(self, wall_time=True):
        return self.linear_regression(
            "multipole_to_local", ["m2l_workload"],
            wall_time=wall_time
        )

    def eval_multipoles_model(self, wall_time=True):
        return self.linear_regression(
            "eval_multipoles", ["m2p_workload"],
            wall_time=wall_time
        )

    def form_locals_model(self, wall_time=True):
        return self.linear_regression(
            "form_locals", ["p2l_workload"],
            wall_time=wall_time
        )

    def eval_locals_model(self, wall_time=True):
        return self.linear_regression(
            "eval_locals", ["eval_part_workload"],
            wall_time=wall_time
        )

    def linear_regression(self, y_name, x_name, wall_time=True):
        """
            :arg y_name: Name of the depedent variable
            :arg x_name: A list of names of independent variables
        """
        nresult = len(self.time_result)
        nvariables = len(x_name)

        if nresult < 1:
            raise RuntimeError("Please run FMM at lease once using time_performance"
                               "before forming models.")
        elif nresult == 1:
            result = self.time_result[0]

            if wall_time:
                dependent_value = result[y_name].wall_elapsed
            else:
                dependent_value = result[y_name].process_elapsed

            independent_value = result[x_name[0]]
            coeff = dependent_value / independent_value

            return (coeff,) + tuple(0.0 for _ in range(nvariables - 1))
        else:
            dependent_value = np.empty((nresult,), dtype=float)
            coeff_matrix = np.empty((nresult, nvariables + 1), dtype=float)

            for iresult, result in enumerate(self.time_result):
                if wall_time:
                    dependent_value[iresult] = result[y_name].wall_elapsed
                else:
                    dependent_value[iresult] = result[y_name].process_elapsed

                for icol, variable_name in enumerate(x_name):
                    coeff_matrix[iresult, icol] = result[variable_name]

            coeff_matrix[:, -1] = 1

            from numpy.linalg import lstsq
            coeff = lstsq(coeff_matrix, dependent_value, rcond=-1)[0]

            return coeff

    def time_random_traversals(self):
        context = self.cl_context
        dtype = np.float64

        traversals = []

        for nsources, ntargets, dims in [(9000, 9000, 3),
                                         (12000, 12000, 3),
                                         (15000, 15000, 3),
                                         (18000, 18000, 3),
                                         (21000, 21000, 3)]:
            generated_traversal = generate_random_traversal(
                context, nsources, ntargets, dims, dtype
            )

            traversals.append(generated_traversal)

        for trav in traversals:
            self.time_performance(trav)

    def predict_time(self, eval_traversal, eval_counter, wall_time=True):
        predict_timing = {}

        # {{{ Predict eval_direct

        param = self.eval_direct_model(wall_time=wall_time)

        direct_workload = np.sum(eval_counter.count_direct())
        direct_nsource_boxes = eval_traversal.neighbor_source_boxes_starts[-1]

        predict_timing["eval_direct"] = (
            direct_workload * param[0] + direct_nsource_boxes * param[1] + param[2])

        # }}}

        # {{{ Predict multipole_to_local

        param = self.multipole_to_local_model(wall_time=wall_time)

        m2l_workload = np.sum(eval_counter.count_m2l())

        predict_timing["multipole_to_local"] = m2l_workload * param[0] + param[1]

        # }}}

        # {{{ Predict eval_multipoles

        param = self.eval_multipoles_model(wall_time=wall_time)

        m2p_workload = np.sum(eval_counter.count_m2p())

        predict_timing["eval_multipoles"] = m2p_workload * param[0] + param[1]

        # }}}

        # {{{ Predict form_locals

        param = self.form_locals_model(wall_time=wall_time)

        p2l_workload = np.sum(eval_counter.count_p2l())

        predict_timing["form_locals"] = p2l_workload * param[0] + param[1]

        # }}}

        # {{{

        param = self.eval_locals_model(wall_time=wall_time)

        eval_part_workload = np.sum(eval_counter.count_eval_part())

        predict_timing["eval_locals"] = eval_part_workload * param[0] + param[1]

        # }}}

        return predict_timing
