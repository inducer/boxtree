from __future__ import division

"""Integration between boxtree and pyfmmlib."""

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
from pytools import memoize_method


__doc__ = """Integrates :mod:`boxtree` with
`pyfmmlib <http://pypi.python.org/pypi/pyfmmlib>`_.
"""


class FMMLibExpansionWrangler(object):
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using pyfmmlib.
    """

    # {{{ constructor

    def __init__(self, tree, helmholtz_k, nterms, ifgrad=False,
            dipole_vec=None, dipoles_already_reordered=False):
        self.tree = tree

        if helmholtz_k == 0:
            self.eqn_letter = "l"
            self.kernel_kwargs = {}
        else:
            self.eqn_letter = "h"
            self.kernel_kwargs = {"zk": helmholtz_k}

        self.nterms = nterms
        self.dtype = np.complex128

        self.ifgrad = ifgrad

        self.dim = tree.dimensions

        common_extra_kwargs = {}
        if self.dim == 3 and self.eqn_letter == "h":
            nquad = max(6, int(2.5*nterms))
            from pyfmmlib import legewhts
            xnodes, weights = legewhts(nquad, ifwhts=1)

            common_extra_kwargs = {
                    "xnodes": xnodes,
                    "wts": weights,
                    }

        self.common_extra_kwargs = common_extra_kwargs

        if dipole_vec is not None:
            assert dipole_vec.shape == (self.dim, self.tree.nsources)

            if not dipoles_already_reordered:
                dipole_vec = self.reorder_sources(dipole_vec)

            self.dipole_vec = dipole_vec.copy(order="F")
            self.dp_suffix = "_dp"
        else:
            self.dipole_vec = None
            self.dp_suffix = ""

    # }}}

    # {{{ overridable target lists for the benefit of the QBX FMM

    def box_target_starts(self):
        return self.tree.box_target_starts

    def box_target_counts_nonchild(self):
        return self.tree.box_target_counts_nonchild

    def targets(self):
        return self.tree.targets

    # }}}

    # {{{ routine getters

    def get_routine(self, name, suffix=""):
        import pyfmmlib
        return getattr(pyfmmlib, "%s%s%s" % (
            self.eqn_letter,
            name % self.dim,
            suffix))

    def get_vec_routine(self, name):
        return self.get_routine(name, "_vec")

    def get_translation_routine(self, name, vec_suffix="_vec"):
        suffix = ""
        if self.dim == 3:
            suffix = "quadu"
        suffix += vec_suffix

        rout = self.get_routine(name, suffix)

        if self.dim == 2:
            return rout
        else:

            def wrapper(*args, **kwargs):
                kwargs.update(self.common_extra_kwargs)
                val, ier = rout(*args, **kwargs)
                if (ier != 0).any():
                    raise RuntimeError("%s failed with nonzero ier" % name)

                return val

            # Doesn't work in in Py2
            # from functools import update_wrapper
            # update_wrapper(wrapper, rout)
            return wrapper

    def get_direct_eval_routine(self):
        if self.dim == 2:
            rout = self.get_vec_routine("potgrad%ddall" + self.dp_suffix)

            def wrapper(*args, **kwargs):
                kwargs["ifgrad"] = self.ifgrad
                kwargs["ifhess"] = False
                pot, grad, hess = rout(*args, **kwargs)

                if not self.ifgrad:
                    grad = 0

                return pot, grad

            # Doesn't work in in Py2
            # from functools import update_wrapper
            # update_wrapper(wrapper, rout)
            return wrapper

        elif self.dim == 3:
            rout = self.get_vec_routine("potfld%ddall" + self.dp_suffix)

            def wrapper(*args, **kwargs):
                kwargs["iffld"] = self.ifgrad
                pot, fld = rout(*args, **kwargs)
                if self.ifgrad:
                    grad = -fld
                else:
                    grad = 0
                return pot, grad

            # Doesn't work in in Py2
            # from functools import update_wrapper
            # update_wrapper(wrapper, rout)
            return wrapper
        else:
            raise ValueError("unsupported dimensionality")

    def get_expn_eval_routine(self, expn_kind):
        name = "%%dd%seval" % expn_kind
        rout = self.get_routine(name, "_vec")

        if self.dim == 2:
            def wrapper(*args, **kwargs):
                kwargs["ifgrad"] = self.ifgrad
                kwargs["ifhess"] = False

                pot, grad, hess = rout(*args, **kwargs)
                if not self.ifgrad:
                    grad = 0

                return pot, grad

            # Doesn't work in in Py2
            # from functools import update_wrapper
            # update_wrapper(wrapper, rout)
            return wrapper

        elif self.dim == 3:
            def wrapper(*args, **kwargs):
                kwargs["iffld"] = self.ifgrad
                pot, fld, ier = rout(*args, **kwargs)

                if (ier != 0).any():
                    raise RuntimeError("%s failed with nonzero ier" % name)

                if self.ifgrad:
                    grad = -fld
                else:
                    grad = 0

                return pot, grad

            # Doesn't work in in Py2
            # from functools import update_wrapper
            # update_wrapper(wrapper, rout)
            return wrapper
        else:
            raise ValueError("unsupported dimensionality")

    # }}}

    # {{{ data vector utilities

    def _expansions_level_starts(self, order_to_size):
        result = [0]
        for lev in range(self.tree.nlevels):
            lev_nboxes = (
                    self.tree.level_start_box_nrs[lev+1]
                    - self.tree.level_start_box_nrs[lev])

            expn_size = order_to_size(self.level_orders[lev])
            result.append(
                    result[-1]
                    + expn_size * lev_nboxes)

        return result

    def expansion_shape(self, nterms):
        if self.dim == 2 and self.eqn_letter == "l":
            return (self.nterms+1,)
        elif self.dim == 2 and self.eqn_letter == "h":
            return (2*self.nterms+1,)
        elif self.dim == 3:
            # This is the transpose of the Fortran format, to
            # minimize mismatch between C and Fortran orders.
            return (2*self.nterms+1, self.nterms+1,)
        else:
            raise ValueError("unsupported dimensionality")

    # @memoize_method
    # def multipole_expansions_level_starts(self):
    #     from pytools import product
    #     return self._expansions_level_starts(
    #             lambda nterms: product(self.expansion_shape(nterms)))

    # @memoize_method
    # def local_expansions_level_starts(self):
    #     from pytools import product
    #     return self._expansions_level_starts(
    #             lambda nterms: product(self.expansion_shape(nterms)))

    def multipole_expansions_view(self, mpole_exps, level):
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        # expn_start, expn_stop = \
        #         self.multipole_expansions_level_starts()[level:level+2]
        # return (box_start,
        #         mpole_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

        return box_start, mpole_exps[box_start:box_stop]

    def local_expansions_view(self, local_exps, level):
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        # expn_start, expn_stop = \
        #         self.local_expansions_level_starts()[level:level+2]
        # return (box_start,
        #         local_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

        return box_start, local_exps[box_start:box_stop]

    def multipole_expansion_zeros(self):
        return np.zeros(
                (self.tree.nboxes,) + self.expansion_shape(self.nterms),
                dtype=self.dtype)

    local_expansion_zeros = multipole_expansion_zeros

    def output_zeros(self):
        if self.ifgrad:
            from pytools import make_obj_array
            return make_obj_array([
                    np.zeros(self.tree.ntargets, self.dtype)
                    for i in range(1 + self.dim)])
        else:
            return np.zeros(self.tree.ntargets, self.dtype)

    def add_potgrad_onto_output(self, output, output_slice, pot, grad):
        if self.ifgrad:
            output[0, output_slice] += pot
            output[1:, output_slice] += grad
        else:
            output[output_slice] += pot

    # }}}

    # {{{ source/target particle wrangling

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(
                pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.box_target_starts()[ibox]
        return slice(
                pstart, pstart + self.box_target_counts_nonchild()[ibox])

    @memoize_method
    def _get_single_sources_array(self):
        return np.array([
            self.tree.sources[idim]
            for idim in range(self.dim)
            ], order="F")

    def _get_sources(self, pslice):
        return self._get_single_sources_array()[:, pslice]

    @memoize_method
    def _get_single_targets_array(self):
        return np.array([
            self.targets()[idim]
            for idim in range(self.dim)
            ], order="F")

    def _get_targets(self, pslice):
        return self._get_single_targets_array()[:, pslice]

    # }}}

    def reorder_sources(self, source_array):
        return source_array[..., self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def get_source_kwargs(self, src_weights, pslice):
        if self.dipole_vec is None:
            return {
                    "charge": src_weights[pslice],
                    }
        else:
            return {
                    "dipstr": src_weights[pslice],
                    "dipvec": self.dipole_vec[:, pslice],
                    }

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        rscale = 1  # FIXME

        formmp = self.get_routine("%ddformmp" + self.dp_suffix)

        mpoles = self.multipole_expansion_zeros()
        for src_ibox in source_boxes:
            pslice = self._get_source_slice(src_ibox)

            if pslice.stop - pslice.start == 0:
                continue

            kwargs = {}
            kwargs.update(self.kernel_kwargs)
            kwargs.update(self.get_source_kwargs(src_weights, pslice))

            ier, mpole = formmp(
                    rscale=rscale,
                    source=self._get_sources(pslice),
                    center=self.tree.box_centers[:, src_ibox],
                    nterms=self.nterms,
                    **kwargs)

            if ier:
                raise RuntimeError("formmp failed")

            mpoles[src_ibox] = mpole.T

        return mpoles

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        tree = self.tree
        rscale = 1  # FIXME

        mpmp = self.get_translation_routine("%ddmpmp")

        # 2 is the last relevant source_level.
        # 1 is the last relevant target_level.
        # (Nobody needs a multipole on level 0, i.e. for the root box.)
        for source_level in range(tree.nlevels-1, 1, -1):
            start, stop = level_start_source_parent_box_nrs[
                            source_level:source_level+2]
            target_level = source_level - 1

            for ibox in source_parent_boxes[start:stop]:
                parent_center = tree.box_centers[:, ibox]
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        child_center = tree.box_centers[:, child]

                        kwargs = {}
                        if self.dim == 3 and self.eqn_letter == "h":
                            kwargs["radius"] = tree.root_extent * 2**(-target_level)

                        kwargs.update(self.kernel_kwargs)

                        new_mp = mpmp(
                                rscale1=rscale,
                                center1=child_center,
                                expn1=mpoles[child].T,

                                rscale2=rscale,
                                center2=parent_center,
                                nterms2=self.nterms,

                                **kwargs)

                        mpoles[ibox] += new_mp[..., 0].T

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights):
        output = self.output_zeros()

        ev = self.get_direct_eval_routine()

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            #tgt_result = np.zeros(tgt_pslice.stop - tgt_pslice.start, self.dtype)
            tgt_pot_result = 0
            tgt_grad_result = 0

            start, end = neighbor_sources_starts[itgt_box:itgt_box+2]
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)

                if src_pslice.stop - src_pslice.start == 0:
                    continue

                kwargs = {}
                kwargs.update(self.kernel_kwargs)
                kwargs.update(self.get_source_kwargs(src_weights, src_pslice))

                tmp_pot, tmp_grad = ev(
                        sources=self._get_sources(src_pslice),
                        targets=self._get_targets(tgt_pslice),
                        **kwargs)

                tgt_pot_result += tmp_pot
                tgt_grad_result += tmp_grad

            self.add_potgrad_onto_output(
                    output, tgt_pslice, tgt_pot_result, tgt_grad_result)

        return output

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        tree = self.tree
        local_exps = self.local_expansion_zeros()

        mploc = self.get_translation_routine("%ddmploc", vec_suffix="_imany")

        for lev in range(self.tree.nlevels):
            lstart, lstop = level_start_target_or_target_parent_box_nrs[lev:lev+2]
            if lstart == lstop:
                continue

            starts_on_lvl = starts[lstart:lstop+1]

            ntgt_boxes = lstop-lstart
            itgt_box_vec = np.arange(ntgt_boxes)
            tgt_ibox_vec = target_or_target_parent_boxes[lstart:lstop]

            nsrc_boxes_per_tgt_box = (
                    starts[lstart + itgt_box_vec+1] - starts[lstart + itgt_box_vec])

            nsrc_boxes = np.sum(nsrc_boxes_per_tgt_box)

            src_boxes_starts = np.empty(ntgt_boxes+1, dtype=np.int32)
            src_boxes_starts[0] = 0
            src_boxes_starts[1:] = np.cumsum(nsrc_boxes_per_tgt_box)

            # FIXME
            rscale1 = np.ones(nsrc_boxes)
            rscale1_offsets = np.arange(nsrc_boxes)

            kwargs = {}
            if self.dim == 3 and self.eqn_letter == "h":
                kwargs["radius"] = (
                        tree.root_extent * 2**(-lev)
                        * np.ones(ntgt_boxes))

            # FIXME
            rscale2 = np.ones(ntgt_boxes, np.float64)

            # These get max'd/added onto: pass initialized versions.
            if self.dim == 3:
                ier = np.zeros(ntgt_boxes, dtype=np.int32)
                kwargs["ier"] = ier

            expn2 = np.zeros(
                    (ntgt_boxes,) + self.expansion_shape(self.nterms),
                    dtype=self.dtype)

            kwargs.update(self.kernel_kwargs)

            expn2 = mploc(
                    rscale1=rscale1,
                    rscale1_offsets=rscale1_offsets,
                    rscale1_starts=src_boxes_starts,

                    center1=tree.box_centers,
                    center1_offsets=lists,
                    center1_starts=starts_on_lvl,

                    expn1=mpole_exps.T,
                    expn1_offsets=lists,
                    expn1_starts=starts_on_lvl,

                    rscale2=rscale2,
                    # FIXME: wrong layout, will copy
                    center2=tree.box_centers[:, tgt_ibox_vec],
                    expn2=expn2.T,
                    **kwargs).T

            local_exps[tgt_ibox_vec] += expn2

        return local_exps

    def eval_multipoles(self, level_start_target_box_nrs, target_boxes,
            sep_smaller_nonsiblings_by_level, mpole_exps):
        output = self.output_zeros()

        rscale = 1

        mpeval = self.get_expn_eval_routine("mp")

        for ssn in sep_smaller_nonsiblings_by_level:
            for itgt_box, tgt_ibox in enumerate(target_boxes):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                if tgt_pslice.stop - tgt_pslice.start == 0:
                    continue

                tgt_pot = 0
                tgt_grad = 0
                start, end = ssn.starts[itgt_box:itgt_box+2]
                for src_ibox in ssn.lists[start:end]:

                    tmp_pot, tmp_grad = mpeval(
                            rscale=rscale,
                            center=self.tree.box_centers[:, src_ibox],
                            expn=mpole_exps[src_ibox].T,
                            ztarg=self._get_targets(tgt_pslice),
                            **self.kernel_kwargs)

                    tgt_pot = tgt_pot + tmp_pot
                    tgt_grad = tgt_grad + tmp_grad

                self.add_potgrad_onto_output(
                        output, tgt_pslice, tgt_pot, tgt_grad)

        return output

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        rscale = 1  # FIXME
        local_exps = self.local_expansion_zeros()

        formta = self.get_routine("%ddformta" + self.dp_suffix)

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0

            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                tgt_center = self.tree.box_centers[:, tgt_ibox]

                if src_pslice.stop - src_pslice.start == 0:
                    continue

                kwargs = {}
                kwargs.update(self.kernel_kwargs)
                kwargs.update(self.get_source_kwargs(src_weights, src_pslice))

                ier, mpole = formta(
                        rscale=rscale,
                        source=self._get_sources(src_pslice),
                        center=tgt_center,
                        nterms=self.nterms,
                        **kwargs)
                if ier:
                    raise RuntimeError("formta failed")

                contrib = contrib + mpole.T

            local_exps[tgt_ibox] = contrib

        return local_exps

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        rscale = 1  # FIXME

        locloc = self.get_translation_routine("%ddlocloc")

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]

            for tgt_ibox in target_or_target_parent_boxes[start:stop]:
                tgt_center = self.tree.box_centers[:, tgt_ibox]
                src_ibox = self.tree.box_parent_ids[tgt_ibox]
                src_center = self.tree.box_centers[:, src_ibox]

                kwargs = {}
                if self.dim == 3 and self.eqn_letter == "h":
                    kwargs["radius"] = self.tree.root_extent * 2**(-target_lev)

                kwargs.update(self.kernel_kwargs)
                tmp_loc_exp = locloc(
                            rscale1=rscale,
                            center1=src_center,
                            expn1=local_exps[src_ibox].T,

                            rscale2=rscale,
                            center2=tgt_center,
                            nterms2=self.nterms,

                            **kwargs)[..., 0]

                local_exps[tgt_ibox] += tmp_loc_exp.T

        return local_exps

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        output = self.output_zeros()
        rscale = 1  # FIXME

        taeval = self.get_expn_eval_routine("ta")

        for tgt_ibox in target_boxes:
            tgt_pslice = self._get_target_slice(tgt_ibox)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tmp_pot, tmp_grad = taeval(
                    rscale=rscale,
                    center=self.tree.box_centers[:, tgt_ibox],
                    expn=local_exps[tgt_ibox].T,
                    ztarg=self._get_targets(tgt_pslice),

                    **self.kernel_kwargs)

            self.add_potgrad_onto_output(
                    output, tgt_pslice, tmp_pot, tmp_grad)

        return output

# vim: foldmethod=marker
