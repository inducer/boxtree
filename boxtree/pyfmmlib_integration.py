"""
Integrates :mod:`boxtree` with
`pyfmmlib <http://pypi.python.org/pypi/pyfmmlib>`_.

.. autoclass:: FMMLibTreeIndependentDataForWrangler
.. autoclass:: FMMLibExpansionWrangler

Internal bits
^^^^^^^^^^^^^

.. autoclass:: FMMLibRotationDataInterface
.. autoclass:: FMMLibRotationData
.. autoclass:: FMMLibRotationDataNotSuppliedWarning
"""

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


import logging
logger = logging.getLogger(__name__)
import enum

import numpy as np

from pytools import memoize_method, log_process
from boxtree.timing import return_timing_data
from boxtree.fmm import TreeIndependentDataForWrangler, ExpansionWranglerInterface


# {{{ rotation data interface

class FMMLibRotationDataInterface:
    """Abstract interface for additional, optional data for precomputation of
    rotation matrices passed to the expansion wrangler.

    .. automethod:: m2l_rotation_lists

    .. automethod:: m2l_rotation_angles

    """

    def m2l_rotation_lists(self):
        """Return a :mod:`numpy` array mapping entries of List 2 to rotation classes.
        """
        raise NotImplementedError

    def m2l_rotation_angles(self):
        """Return a :mod:`numpy` array mapping List 2 rotation classes to
        rotation angles.
        """
        raise NotImplementedError


class FMMLibRotationData(FMMLibRotationDataInterface):
    """An implementation of the :class:`FMMLibRotationDataInterface`.

    .. automethod:: __init__
    """

    def __init__(self, queue, trav):
        self.queue = queue
        self.trav = trav
        self.tree = trav.tree

    @property
    @memoize_method
    def rotation_classes_builder(self):
        from boxtree.rotation_classes import RotationClassesBuilder
        return RotationClassesBuilder(self.queue.context)

    @memoize_method
    def build_rotation_classes_lists(self):
        trav = self.trav.to_device(self.queue)
        tree = self.tree.to_device(self.queue)
        return self.rotation_classes_builder(self.queue, trav, tree)[0]

    @memoize_method
    def m2l_rotation_lists(self):
        return (self
                .build_rotation_classes_lists()
                .from_sep_siblings_rotation_classes
                .get(self.queue))

    @memoize_method
    def m2l_rotation_angles(self):
        return (self
                .build_rotation_classes_lists()
                .from_sep_siblings_rotation_class_to_angle
                .get(self.queue))


class FMMLibRotationDataNotSuppliedWarning(UserWarning):
    pass

# }}}


@enum.unique
class Kernel(enum.Enum):
    LAPLACE = enum.auto()
    HELMHOLTZ = enum.auto()


# {{{ tree-independent data for wrangler

class FMMLibTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    """
    .. automethod:: __init__
    """

    def __init__(self, dim, kernel, ifgrad=False):
        self.dim = dim
        self.ifgrad = ifgrad
        self.kernel = kernel

        if kernel == Kernel.LAPLACE:
            self.eqn_letter = "l"
        elif kernel == Kernel.HELMHOLTZ:
            self.eqn_letter = "h"
        else:
            raise ValueError(kernel)

        self.dtype = np.complex128

    # {{{ routine getters

    def get_routine(self, name, suffix=""):
        import pyfmmlib
        return getattr(pyfmmlib, "{}{}{}".format(
            self.eqn_letter,
            name % self.dim,
            suffix))

    def get_vec_routine(self, name):
        return self.get_routine(name, "_vec")

    def get_translation_routine(self, wrangler, name, vec_suffix="_vec"):
        suffix = ""
        if self.dim == 3:
            suffix = "quadu"
        suffix += vec_suffix

        rout = self.get_routine(name, suffix)

        if self.dim == 2:
            def wrapper(*args, **kwargs):
                # not used
                kwargs.pop("level_for_projection", None)

                return rout(*args, **kwargs)
        else:

            def wrapper(*args, **kwargs):
                kwargs.pop("level_for_projection", None)
                nterms2 = kwargs["nterms2"]
                kwargs.update(wrangler.projection_quad_extra_kwargs(order=nterms2))

                val, ier = rout(*args, **kwargs)
                if (ier != 0).any():
                    raise RuntimeError("%s failed with nonzero ier" % name)

                return val

        # Doesn't work in in Py2
        # from functools import update_wrapper
        # update_wrapper(wrapper, rout)
        return wrapper

    def get_direct_eval_routine(self, use_dipoles):
        if self.dim == 2:
            rout = self.get_vec_routine(
                    "potgrad%ddall" + ("_dp" if use_dipoles else ""))

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
            rout = self.get_vec_routine(
                    "potfld%ddall" + ("_dp" if use_dipoles else ""))

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

# }}}


# {{{ wrangler

class FMMLibExpansionWrangler(ExpansionWranglerInterface):
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using pyfmmlib.

    Timing results returned by this wrangler contains the values *wall_elapsed*
    and (optionally, if supported) *process_elapsed*, which measure wall time
    and process time in seconds, respectively.
    """

    # {{{ constructor

    def __init__(self, tree_indep, traversal, *,
            helmholtz_k=None, fmm_level_to_order=None,
            dipole_vec=None, dipoles_already_reordered=False, order=None,
            optimized_m2l_precomputation_memory_cutoff_bytes=10**8,
            rotation_data=None):
        """
        :arg fmm_level_to_order: A callable that, upon being passed the tree
            and the tree level as an integer, returns the order for the multipole and
            local expansions on that level.
        :arg rotation_data: Either *None* or an instance of the
            :class:`FMMLibRotationDataInterface`. In three dimensions, passing
            *rotation_data* enables optimized M2L (List 2) translations.
            In two dimensions, this does nothing.
        :arg optimized_m2l_precomputation_memory_cutoff_bytes: When using
            optimized List 2 translations, an upper bound in bytes on the
            amount of storage to use for a precomputed rotation matrix.
        """

        if order is not None and fmm_level_to_order is not None:
            raise TypeError("may specify either fmm_level_to_order or order, "
                    "but not both")

        if order is not None:
            from warnings import warn
            warn("Passing order is deprecated. Pass fmm_level_to_order instead.",
                    DeprecationWarning, stacklevel=2)

            def fmm_level_to_order(tree, level):  # noqa pylint:disable=function-redefined
                return order

        super().__init__(tree_indep, traversal)

        if tree_indep.kernel == Kernel.LAPLACE:
            self.kernel_kwargs = {}
            self.rscale_factor = 1

            if helmholtz_k:
                raise ValueError(
                        "helmholtz_k must be zero or unspecified for Laplace")

            helmholtz_k = 0

        elif tree_indep.kernel == Kernel.HELMHOLTZ:
            self.kernel_kwargs = {"zk": helmholtz_k}

            if not helmholtz_k:
                raise ValueError(
                        "helmholtz_k must be specified and nonzero")

            self.rscale_factor = abs(helmholtz_k)

        else:
            raise ValueError(tree_indep.kernel)

        self.helmholtz_k = helmholtz_k

        tree = traversal.tree

        if tree_indep.dim != tree.dimensions:
            raise ValueError(f"Kernel dim ({tree_indep.dim}) "
                    f"does not match tree dim ({tree.dimensions})")

        self.level_orders = np.array([
            fmm_level_to_order(tree, lev) for lev in range(tree.nlevels)
            ], dtype=np.int32)

        if tree_indep.kernel == Kernel.HELMHOLTZ:
            logger.info("expansion orders by level used in Helmholtz FMM: %s",
                    self.level_orders)

        self.rotation_data = rotation_data
        self.rotmat_cutoff_bytes = optimized_m2l_precomputation_memory_cutoff_bytes

        if self.dim == 3:
            if rotation_data is None:
                from warnings import warn
                warn(
                        "List 2 (multipole-to-local) translations will be "
                        "unoptimized. Supply a rotation_data argument to "
                        "FMMLibExpansionWrangler for optimized List 2.",
                        FMMLibRotationDataNotSuppliedWarning,
                        stacklevel=2)

            self.supports_optimized_m2l = rotation_data is not None
        else:
            self.supports_optimized_m2l = False

        # FIXME: dipole_vec shouldn't be stored here! Otherwise, we'll recompute
        # bunches of tree-dependent stuff for every new dipole vector.

        # It's not super bad because the dipole vectors are typically geometry
        # normals and thus change about at the same time as the tree... but there's
        # still no reason for them to be here.
        self.use_dipoles = dipole_vec is not None
        if self.use_dipoles:
            assert dipole_vec.shape == (self.dim, self.tree.nsources)

            if not dipoles_already_reordered:
                dipole_vec = self.reorder_sources(dipole_vec)

            self.dipole_vec = dipole_vec.copy(order="F")
        else:
            self.dipole_vec = None

    # }}}

    @property
    def dim(self):
        return self.tree.dimensions

    def level_to_rscale(self, level):
        result = self.tree.root_extent * 2 ** -level * self.rscale_factor
        if abs(result) > 1:
            result = 1
        if self.dim == 3 and self.tree_indep.eqn_letter == "l":
            # Laplace 3D uses the opposite convention compared to
            # all other cases.
            # https://gitlab.tiker.net/inducer/boxtree/merge_requests/81
            result = 1 / result
        return result

    @memoize_method
    def projection_quad_extra_kwargs(self, level=None, order=None):
        if level is None and order is None:
            raise TypeError("must pass exactly one of level or order")
        if level is not None and order is not None:
            raise TypeError("must pass exactly one of level or order")
        if level is not None:
            order = self.level_orders[level]

        common_extra_kwargs = {}

        if self.dim == 3 and self.tree_indep.eqn_letter == "h":
            nquad = max(6, int(2.5*order))
            from pyfmmlib import legewhts
            xnodes, weights = legewhts(nquad, ifwhts=1)

            common_extra_kwargs = {
                    "xnodes": xnodes,
                    "wts": weights,
                    }

        return common_extra_kwargs

    # {{{ overridable target lists for the benefit of the QBX FMM

    def box_target_starts(self):
        return self.tree.box_target_starts

    def box_target_counts_nonchild(self):
        return self.tree.box_target_counts_nonchild

    def targets(self):
        return self.tree.targets

    # }}}

    # {{{ level starts

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

    @memoize_method
    def multipole_expansions_level_starts(self):
        from pytools import product
        return self._expansions_level_starts(
                lambda order: product(
                    self.expansion_shape(order)))

    @memoize_method
    def local_expansions_level_starts(self):
        from pytools import product
        return self._expansions_level_starts(
                lambda order: product(
                    self.expansion_shape(order)))

    # }}}

    # {{{ views into arrays of expansions

    def multipole_expansions_view(self, mpole_exps, level):
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        expn_start, expn_stop = \
                self.multipole_expansions_level_starts()[level:level+2]
        return (box_start,
                mpole_exps[expn_start:expn_stop].reshape(
                    box_stop-box_start,
                    *self.expansion_shape(self.level_orders[level])))

    def local_expansions_view(self, local_exps, level):
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        expn_start, expn_stop = \
                self.local_expansions_level_starts()[level:level+2]
        return (box_start,
                local_exps[expn_start:expn_stop].reshape(
                    box_stop-box_start,
                    *self.expansion_shape(self.level_orders[level])))

    # }}}

    def get_source_kwargs(self, src_weights, pslice):
        if self.dipole_vec is None:
            return {
                    "charge": src_weights[pslice],
                    }
        else:
            if self.tree_indep.eqn_letter == "l" and self.dim == 2:
                return {
                        "dipstr": -src_weights[pslice] * (
                            self.dipole_vec[0, pslice]
                            + 1j * self.dipole_vec[1, pslice])
                        }
            else:
                return {
                        "dipstr": src_weights[pslice],
                        "dipvec": self.dipole_vec[:, pslice],
                        }

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

    @memoize_method
    def _get_single_box_centers_array(self):
        return np.array([
            self.tree.box_centers[idim]
            for idim in range(self.dim)
            ], order="F")

    # }}}

    # {{{ precompute rotation matrices for optimized m2l

    @memoize_method
    def m2l_rotation_matrices(self):
        # Returns a tuple (rotmatf, rotmatb, rotmat_order), consisting of the
        # forward rotation matrices, backward rotation matrices, and the
        # translation order of the matrices. rotmat_order is -1 if not
        # supported.

        rotmatf = None
        rotmatb = None
        rotmat_order = -1

        if not self.supports_optimized_m2l:
            return (rotmatf, rotmatb, rotmat_order)

        m2l_rotation_angles = self.rotation_data.m2l_rotation_angles()

        if len(m2l_rotation_angles) == 0:
            # The pyfmmlib wrapper may or may not complain if you give it a
            # zero-length array.
            return (rotmatf, rotmatb, rotmat_order)

        def mem_estimate(order):
            # Rotation matrix memory cost estimate.
            return (8
                    * (order + 1)**2
                    * (2*order + 1)
                    * len(m2l_rotation_angles))

        # Find the largest order we can use. Because the memory cost of the
        # matrices could be large, only precompute them if the cost estimate
        # for the order does not exceed the cutoff.
        for order in sorted(self.level_orders, reverse=True):
            if mem_estimate(order) < self.rotmat_cutoff_bytes:
                rotmat_order = order
                break

        if rotmat_order == -1:
            return (rotmatf, rotmatb, rotmat_order)

        # Compute the rotation matrices.
        from pyfmmlib import rotviarecur3p_init_vec as rotmat_builder

        ier, rotmatf = (
                rotmat_builder(rotmat_order, m2l_rotation_angles))
        assert (0 == ier).all()

        ier, rotmatb = (
                rotmat_builder(rotmat_order, -m2l_rotation_angles))
        assert (0 == ier).all()

        return (rotmatf, rotmatb, rotmat_order)

    # }}}

    # {{{ data vector utilities

    def expansion_shape(self, order):
        if self.dim == 2 and self.tree_indep.eqn_letter == "l":
            return (order+1,)
        elif self.dim == 2 and self.tree_indep.eqn_letter == "h":
            return (2*order+1,)
        elif self.dim == 3:
            # This is the transpose of the Fortran format, to
            # minimize mismatch between C and Fortran orders.
            return (2*order+1, order+1,)
        else:
            raise ValueError("unsupported dimensionality")

    def multipole_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """

        return np.zeros(
                self.multipole_expansions_level_starts()[-1],
                dtype=self.tree_indep.dtype)

    def local_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """
        return np.zeros(
                self.local_expansions_level_starts()[-1],
                dtype=self.tree_indep.dtype)

    def output_zeros(self):
        """Return a potentials array (which must support addition) capable of
        holding a potential value for each target in the tree. Note that
        :func:`drive_fmm` makes no assumptions about *potential* other than
        that it supports addition--it may consist of potentials, gradients of
        the potential, or arbitrary other per-target output data.
        """

        if self.tree_indep.ifgrad:
            from pytools.obj_array import make_obj_array
            return make_obj_array([
                    np.zeros(self.tree.ntargets, self.tree_indep.dtype)
                    for i in range(1 + self.dim)])
        else:
            return np.zeros(self.tree.ntargets, self.tree_indep.dtype)

    def add_potgrad_onto_output(self, output, output_slice, pot, grad):
        if self.tree_indep.ifgrad:
            output[0, output_slice] += pot
            output[1:, output_slice] += grad
        else:
            output[output_slice] += pot

    # }}}

    @log_process(logger)
    def reorder_sources(self, source_array):
        return source_array[..., self.tree.user_source_ids]

    @log_process(logger)
    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    @log_process(logger)
    @return_timing_data
    def form_multipoles(self, level_start_source_box_nrs, source_boxes,
            src_weight_vecs):
        src_weights, = src_weight_vecs
        formmp = self.tree_indep.get_routine(
                "%ddformmp" + ("_dp" if self.use_dipoles else ""))

        mpoles = self.multipole_expansion_zeros()
        for lev in range(self.tree.nlevels):
            start, stop = level_start_source_box_nrs[lev:lev+2]
            if start == stop:
                continue

            level_start_ibox, mpoles_view = self.multipole_expansions_view(
                    mpoles, lev)

            rscale = self.level_to_rscale(lev)

            for src_ibox in source_boxes[start:stop]:
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
                        nterms=self.level_orders[lev],
                        **kwargs)

                if ier:
                    raise RuntimeError("formmp failed")

                mpoles_view[src_ibox-level_start_ibox] = mpole.T

        return mpoles

    @log_process(logger)
    @return_timing_data
    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        tree = self.tree

        mpmp = self.tree_indep.get_translation_routine(self, "%ddmpmp")

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                            target_level:target_level+2]

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpoles, source_level)
            target_level_start_ibox, target_mpoles_view = \
                    self.multipole_expansions_view(mpoles, target_level)

            source_rscale = self.level_to_rscale(source_level)
            target_rscale = self.level_to_rscale(target_level)

            for ibox in source_parent_boxes[start:stop]:
                parent_center = tree.box_centers[:, ibox]
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        child_center = tree.box_centers[:, child]

                        kwargs = {}
                        if self.dim == 3 and self.tree_indep.eqn_letter == "h":
                            kwargs["radius"] = tree.root_extent * 2**(-target_level)

                        kwargs.update(self.kernel_kwargs)

                        new_mp = mpmp(
                                rscale1=source_rscale,
                                center1=child_center,
                                expn1=source_mpoles_view[
                                    child - source_level_start_ibox].T,

                                rscale2=target_rscale,
                                center2=parent_center,
                                nterms2=self.level_orders[target_level],

                                **kwargs)

                        target_mpoles_view[
                                ibox - target_level_start_ibox] += new_mp[..., 0].T

        return mpoles

    @log_process(logger)
    @return_timing_data
    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weight_vecs):
        src_weights, = src_weight_vecs
        output = self.output_zeros()

        ev = self.tree_indep.get_direct_eval_routine(self.use_dipoles)

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            # tgt_result = np.zeros(
            #         tgt_pslice.stop - tgt_pslice.start, self.tree_indep.dtype)
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

    @log_process(logger)
    @return_timing_data
    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        tree = self.tree
        local_exps = self.local_expansion_zeros()

        # Precomputed rotation matrices (matrices of larger order can be used
        # for translations of smaller order)
        rotmatf, rotmatb, rotmat_order = self.m2l_rotation_matrices()

        for lev in range(self.tree.nlevels):
            lstart, lstop = level_start_target_or_target_parent_box_nrs[lev:lev+2]
            if lstart == lstop:
                continue

            starts_on_lvl = starts[lstart:lstop+1]

            mploc = self.tree_indep.get_translation_routine(
                    self, "%ddmploc", vec_suffix="_imany")

            kwargs = {}

            # {{{ set up optimized m2l, if applicable

            if self.level_orders[lev] <= rotmat_order:
                m2l_rotation_lists = self.rotation_data.m2l_rotation_lists()
                assert len(m2l_rotation_lists) == len(lists)

                mploc = self.tree_indep.get_translation_routine(
                        self, "%ddmploc", vec_suffix="2_trunc_imany")

                kwargs["ldm"] = rotmat_order
                kwargs["nterms"] = self.level_orders[lev]
                kwargs["nterms1"] = self.level_orders[lev]

                kwargs["rotmatf"] = rotmatf
                kwargs["rotmatf_offsets"] = m2l_rotation_lists
                kwargs["rotmatf_starts"] = starts_on_lvl

                kwargs["rotmatb"] = rotmatb
                kwargs["rotmatb_offsets"] = m2l_rotation_lists
                kwargs["rotmatb_starts"] = starts_on_lvl

            # }}}

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, lev)
            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            ntgt_boxes = lstop-lstart
            itgt_box_vec = np.arange(ntgt_boxes)
            tgt_ibox_vec = target_or_target_parent_boxes[lstart:lstop]

            nsrc_boxes_per_tgt_box = (
                    starts[lstart + itgt_box_vec+1] - starts[lstart + itgt_box_vec])

            nsrc_boxes = np.sum(nsrc_boxes_per_tgt_box)

            src_boxes_starts = np.empty(ntgt_boxes+1, dtype=np.int32)
            src_boxes_starts[0] = 0
            src_boxes_starts[1:] = np.cumsum(nsrc_boxes_per_tgt_box)

            rscale = self.level_to_rscale(lev)

            rscale1 = np.ones(nsrc_boxes) * rscale
            rscale1_offsets = np.arange(nsrc_boxes)

            if self.dim == 3 and self.tree_indep.eqn_letter == "h":
                kwargs["radius"] = (
                        tree.root_extent * 2**(-lev)
                        * np.ones(ntgt_boxes))

            rscale2 = np.ones(ntgt_boxes, np.float64) * rscale

            # These get max'd/added onto: pass initialized versions.
            if self.dim == 3:
                ier = np.zeros(ntgt_boxes, dtype=np.int32)
                kwargs["ier"] = ier

            expn2 = np.zeros(
                    (ntgt_boxes,) + self.expansion_shape(self.level_orders[lev]),
                    dtype=self.tree_indep.dtype)

            kwargs.update(self.kernel_kwargs)

            expn2 = mploc(
                    rscale1=rscale1,
                    rscale1_offsets=rscale1_offsets,
                    rscale1_starts=src_boxes_starts,

                    center1=tree.box_centers,
                    center1_offsets=lists,
                    center1_starts=starts_on_lvl,

                    expn1=source_mpoles_view.T,
                    expn1_offsets=lists - source_level_start_ibox,
                    expn1_starts=starts_on_lvl,

                    rscale2=rscale2,
                    # FIXME: wrong layout, will copy
                    center2=tree.box_centers[:, tgt_ibox_vec],
                    expn2=expn2.T,

                    nterms2=self.level_orders[lev],

                    **kwargs).T

            target_local_exps_view[tgt_ibox_vec - target_level_start_ibox] += expn2

        return local_exps

    @log_process(logger)
    @return_timing_data
    def eval_multipoles(self,
            target_boxes_by_source_level, sep_smaller_nonsiblings_by_level,
            mpole_exps):
        output = self.output_zeros()

        mpeval = self.tree_indep.get_expn_eval_routine("mp")

        for isrc_level, ssn in enumerate(sep_smaller_nonsiblings_by_level):
            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, isrc_level)

            rscale = self.level_to_rscale(isrc_level)

            for itgt_box, tgt_ibox in \
                    enumerate(target_boxes_by_source_level[isrc_level]):
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
                            expn=source_mpoles_view[
                                src_ibox - source_level_start_ibox].T,
                            ztarg=self._get_targets(tgt_pslice),
                            **self.kernel_kwargs)

                    tgt_pot = tgt_pot + tmp_pot
                    tgt_grad = tgt_grad + tmp_grad

                self.add_potgrad_onto_output(
                        output, tgt_pslice, tgt_pot, tgt_grad)

        return output

    @log_process(logger)
    @return_timing_data
    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weight_vecs):
        src_weights, = src_weight_vecs
        local_exps = self.local_expansion_zeros()

        formta = self.tree_indep.get_routine(
                "%ddformta" + ("_dp" if self.use_dipoles else ""), suffix="_imany")

        sources = self._get_single_sources_array()
        # sources_starts / sources_lists is a CSR list mapping box centers to
        # lists of starting indices into the sources array. To get the starting
        # source indices we have to look at box_source_starts.
        sources_offsets = self.tree.box_source_starts[lists]

        # nsources_starts / nsources_lists is a CSR list mapping box centers to
        # lists of indices into nsources, each of which represents a source
        # count.
        nsources = self.tree.box_source_counts_nonchild
        nsources_offsets = lists

        # centers is indexed into by values of centers_offsets, which is a list
        # mapping box indices to box center indices.
        centers = self._get_single_box_centers_array()

        source_kwargs = self.get_source_kwargs(src_weights, slice(None))

        for lev in range(self.tree.nlevels):
            lev_start, lev_stop = \
                    level_start_target_or_target_parent_box_nrs[lev:lev+2]

            if lev_start == lev_stop:
                continue

            target_box_start, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            centers_offsets = target_or_target_parent_boxes[lev_start:lev_stop]

            rscale = self.level_to_rscale(lev)

            sources_starts = starts[lev_start:1 + lev_stop]
            nsources_starts = sources_starts

            kwargs = {}
            kwargs.update(self.kernel_kwargs)
            for key, val in source_kwargs.items():
                kwargs[key] = val
                # Add CSR lists mapping box centers to lists of starting positions
                # in the array of source strengths.
                # Since the source strengths have the same order as the sources,
                # these lists are the same as those for starting position in the
                # sources array.
                kwargs[key + "_starts"] = sources_starts
                kwargs[key + "_offsets"] = sources_offsets

            ier, expn = formta(
                    rscale=rscale,
                    sources=sources,
                    sources_offsets=sources_offsets,
                    sources_starts=sources_starts,
                    nsources=nsources,
                    nsources_starts=nsources_starts,
                    nsources_offsets=nsources_offsets,
                    centers=centers,
                    centers_offsets=centers_offsets,
                    nterms=self.level_orders[lev],
                    **kwargs)

            if ier.any():
                raise RuntimeError("formta failed")

            target_local_exps_view[
                    target_or_target_parent_boxes[lev_start:lev_stop]
                    - target_box_start] = expn.T

        return local_exps

    @log_process(logger)
    @return_timing_data
    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):

        locloc = self.tree_indep.get_translation_routine(self, "%ddlocloc")

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]

            source_lev = target_lev - 1

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, source_lev)
            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, target_lev)
            source_rscale = self.level_to_rscale(source_lev)
            target_rscale = self.level_to_rscale(target_lev)

            for tgt_ibox in target_or_target_parent_boxes[start:stop]:
                tgt_center = self.tree.box_centers[:, tgt_ibox]
                src_ibox = self.tree.box_parent_ids[tgt_ibox]
                src_center = self.tree.box_centers[:, src_ibox]

                kwargs = {}
                if self.dim == 3 and self.tree_indep.eqn_letter == "h":
                    kwargs["radius"] = self.tree.root_extent * 2**(-target_lev)

                kwargs.update(self.kernel_kwargs)
                tmp_loc_exp = locloc(
                            rscale1=source_rscale,
                            center1=src_center,
                            expn1=source_local_exps_view[
                                src_ibox - source_level_start_ibox].T,

                            rscale2=target_rscale,
                            center2=tgt_center,
                            nterms2=self.level_orders[target_lev],

                            **kwargs)[..., 0]

                target_local_exps_view[
                        tgt_ibox - target_level_start_ibox] += tmp_loc_exp.T

        return local_exps

    @log_process(logger)
    @return_timing_data
    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        output = self.output_zeros()
        taeval = self.tree_indep.get_expn_eval_routine("ta")

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            rscale = self.level_to_rscale(lev)

            for tgt_ibox in target_boxes[start:stop]:
                tgt_pslice = self._get_target_slice(tgt_ibox)

                if tgt_pslice.stop - tgt_pslice.start == 0:
                    continue

                tmp_pot, tmp_grad = taeval(
                        rscale=rscale,
                        center=self.tree.box_centers[:, tgt_ibox],
                        expn=source_local_exps_view[
                            tgt_ibox - source_level_start_ibox].T,
                        ztarg=self._get_targets(tgt_pslice),

                        **self.kernel_kwargs)

                self.add_potgrad_onto_output(
                        output, tgt_pslice, tmp_pot, tmp_grad)

        return output

    @log_process(logger)
    def finalize_potentials(self, potential, template_ary):
        if self.tree_indep.eqn_letter == "l" and self.dim == 2:
            scale_factor = -1/(2*np.pi)
        elif self.tree_indep.eqn_letter == "h" and self.dim == 2:
            scale_factor = 1
        elif self.tree_indep.eqn_letter in ["l", "h"] and self.dim == 3:
            scale_factor = 1/(4*np.pi)
        else:
            raise NotImplementedError(
                    "scale factor for pyfmmlib %s for %d dimensions" % (
                        self.tree_indep.eqn_letter,
                        self.dim))

        if self.tree_indep.eqn_letter == "l" and self.dim == 2:
            potential = potential.real

        return potential * scale_factor

# }}}


# vim: foldmethod=marker
