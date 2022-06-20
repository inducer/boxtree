"""
.. autofunction:: drive_fmm

.. autoclass:: TreeIndependentDataForWrangler
.. autoclass:: ExpansionWranglerInterface
"""

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

from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)
from boxtree.tree import Tree
from boxtree.traversal import FMMTraversalInfo


from pytools import ProcessLogger


# {{{ expansion wrangler interface

class TreeIndependentDataForWrangler:
    """An object that can be used to store information for efficient
    wrangler execution that depends on the kernel but not the tree and/or
    the traversal.

    Examples of such data include generated code for carrying out
    translations.

    .. note::

        Instances of this type should not hold a reference (and thereby be
        specific to) a :class:`boxtree.Tree` instance. Their purpose is to
        host caches for generated translation code that is reusable across
        trees. It is OK for these instances to be specific to a given kernel
        (or set of kernels).
    """


class ExpansionWranglerInterface(ABC):
    """Abstract expansion handling interface for use with :func:`drive_fmm`.

    See this
    `test code <https://github.com/inducer/boxtree/blob/master/test/test_fmm.py>`_
    for a very simple sample implementation.

    .. note::

        Wranglers may hold a reference (and thereby be specific to) a
        :class:`boxtree.Tree` instance.
        :class:`TreeIndependentDataForWrangler` exists to hold data that
        is more broadly reusable.

    Functions that support returning timing data return a value supporting the
    :class:`~boxtree.timing.TimingFuture` interface.

    .. versionchanged:: 2018.1

        Changed (a subset of) functions to return timing data.

    .. attribute:: tree_indep

        An instance of (a typically wrangler-dependent subclass of)
        :class:`TreeIndependentDataForWrangler`.

    .. attribute:: traversal

        An instance of :class:`~boxtree.traversal.FMMTraversalInfo`.

    .. autoattribute:: tree

    .. rubric:: Particle ordering

    .. automethod:: reorder_sources
    .. automethod:: reorder_potentials

    .. rubric:: Views into arrays of expansions

    .. automethod:: multipole_expansions_view
    .. automethod:: local_expansions_view

    .. rubric:: Translations

    .. automethod:: form_multipoles
    .. automethod:: coarsen_multipoles
    .. automethod:: eval_direct
    .. automethod:: multipole_to_local
    .. automethod:: eval_multipoles
    .. automethod:: form_locals
    .. automethod:: refine_locals
    .. automethod:: eval_locals
    .. automethod:: finalize_potentials
    """

    def __init__(self, tree_indep: TreeIndependentDataForWrangler,
            traversal: FMMTraversalInfo):
        self.tree_indep = tree_indep
        self.traversal = traversal

    @property
    def tree(self) -> Tree:
        return self.traversal.tree

    @abstractmethod
    def reorder_sources(self, source_array):
        """Return a copy of *source_array* in
        :ref:`tree source order <particle-orderings>`.
        *source_array* is in user source order.
        """

    @abstractmethod
    def reorder_potentials(self, potentials):
        """Return a copy of *potentials* in
        :ref:`user target order <particle-orderings>`.
        *source_weights* is in tree target order.
        """

    # {{{ views into arrays of expansions

    # Included here for the benefit of the distributed-memory FMM

    @abstractmethod
    def multipole_expansions_view(self, mpole_exps, level):
        pass

    @abstractmethod
    def local_expansions_view(self, local_exps, level):
        pass

    # }}}

    # {{{ translations

    @abstractmethod
    def form_multipoles(self,
            level_start_source_box_nrs, source_boxes,
            src_weight_vecs):
        """Return an expansions array
        containing multipole expansions in *source_boxes* due to sources
        with *src_weight_vecs*.
        All other expansions must be zero.

        :return: A pair (*mpoles*, *timing_future*).
        """

    @abstractmethod
    def coarsen_multipoles(self,
            level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        """For each box in *source_parent_boxes*,
        gather (and translate) the box's children's multipole expansions in
        *mpole* and add the resulting expansion into the box's multipole
        expansion in *mpole*.

        :returns: A pair (*mpoles*, *timing_future*).
        """

    @abstractmethod
    def eval_direct(self,
            target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weight_vecs):
        """For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weight_vecs*, which use :ref:`csr` and are
        indexed like *target_boxes*.

        :returns: A pair (*pot*, *timing_future*), where *pot* is a
            a new potential array.
        """

    @abstractmethod
    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        """For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.  *starts* and *lists* use :ref:`csr`, and
        *starts* is indexed like *target_or_target_parent_boxes*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is
            a new (local) expansion array.
        """

    @abstractmethod
    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_by_level, mpole_exps):
        """For a level *i*, each box in *target_boxes_by_source_level[i]*, evaluate
        the multipole expansion in *mpole_exps* in the nearby boxes given in
        *from_sep_smaller_by_level*, and return a new potential array.
        *starts* and *lists* in *from_sep_smaller_by_level[i]* use :ref:`csr`
        and *starts* is indexed like *target_boxes_by_source_level[i]*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new potential
            array.
        """

    @abstractmethod
    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weight_vecs):
        """For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.  *starts* and *lists*
        use :ref:`csr` and *starts* is indexed like
        *target_or_target_parent_boxes*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new
            local expansion array.
        """

    @abstractmethod
    def refine_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        """For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: A pair (*local_exps*, *timing_future*).
        """

    @abstractmethod
    def eval_locals(self,
            level_start_target_box_nrs, target_boxes, local_exps):
        """For each box in *target_boxes*, evaluate the local expansion in
        *local_exps* and return a new potential array.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new potential
            array.
        """

    # }}}

    @abstractmethod
    def finalize_potentials(self, potentials, template_ary):
        """
        Postprocess the reordered potentials. This is where global scaling
        factors could be applied. This is distinct from :meth:`reorder_potentials`
        because some derived FMMs (notably the QBX FMM) do their own reordering.

        :arg template_ary: If the array type used inside of the FMM
            is different from the array type used by the user (e.g.
            :class:`boxtree.pyfmmlib_integration.FMMLibExpansionWrangler`
            uses :class:`numpy.ndarray` internally, this array can be used
            to help convert the output back to the user's array
            type (typically :class:`pyopencl.array.Array`).
        """

    def distribute_source_weights(self, src_weight_vecs, src_idx_all_ranks):
        """Used by the distributed implementation for transferring needed source
        weights from root rank to each worker rank in the communicator.

        This method needs to be called collectively by all ranks in the communicator.

        :arg src_weight_vecs: a sequence of :class:`numpy.ndarray`, each with length
            ``nsources``, representing the weights of sources on the root rank.
            *None* on worker ranks.
        :arg src_idx_all_ranks: a :class:`list` of length ``nranks``, including the
            root rank, where the i-th entry is a :class:`numpy.ndarray` of indices,
            of which *src_weight_vecs* to be sent from the root rank to rank *i*.
            Each entry can be generated by :func:`.generate_local_tree`. *None* on
            worker ranks.

        :return: Received source weights of the current rank, including the root
            rank.
        """
        return src_weight_vecs

    def gather_potential_results(self, potentials, tgt_idx_all_ranks):
        """Used by the distributed implementation for gathering calculated potentials
        from all worker ranks in the communicator to the root rank.

        This method needs to be called collectively by all ranks in the communicator.

        :arg potentials: Calculated potentials on each rank. This argument is
            significant on all ranks, including the root rank.
        :arg tgt_idx_all_ranks: a :class:`list` of length ``nranks``, where the
            i-th entry is a :class:`numpy.ndarray` of the global potential indices
            of potentials from rank *i*. This argument is only significant on the
            root rank.

        :return: Gathered potentials on the root rank. *None* on worker ranks.
        """
        return potentials

    def communicate_mpoles(self, mpole_exps, return_stats=False):
        """Used by the distributed implementation for forming the complete multipole
        expansions from the partial multipole expansions.

        This function accepts partial multipole expansions in the argument
        *mpole_exps*, and modifies *mpole_exps* in place with the communicated and
        reduced multipole expansions.

        This function needs to be called collectively by all ranks in the
        communicator.

        :returns: Statistics of the communication if *return_stats* is True. *None*
            otherwise.
        """
        pass

# }}}


def drive_fmm(wrangler: ExpansionWranglerInterface, src_weight_vecs,
              timing_data=None,
              global_src_idx_all_ranks=None, global_tgt_idx_all_ranks=None):
    """Top-level driver routine for a fast multipole calculation.

    In part, this is intended as a template for custom FMMs, in the sense that
    you may copy and paste its
    `source code <https://github.com/inducer/boxtree/blob/master/boxtree/fmm.py>`_
    as a starting point.

    Nonetheless, many common applications (such as point-to-point FMMs) can be
    covered by supplying the right *expansion_wrangler* to this routine.

    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`. For distributed implementation, this
        wrangler should be a subclass of
        :class:`boxtree.distributed.calculation.DistributedExpansionWrangler`.
    :arg src_weight_vecs: A sequence of source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*. For distributed
        implementation, this argument is only significant on the root rank, but
        worker ranks still need to supply a dummy vector.
    :arg timing_data: Either *None*, or a :class:`dict` that is populated with
        timing information for the stages of the algorithm (in the form of
        :class:`~boxtree.timing.TimingResult`), if such information is available.
    :arg global_src_idx_all_ranks: Only used in the distributed implementation. A
        :class:`list` of length ``nranks``, where the i-th entry is a
        :class:`numpy.ndarray` representing the global indices of sources in the
        local tree on rank *i*. Each entry can be returned from
        *generate_local_tree*. This argument is only significant on the root rank.
    :arg global_tgt_idx_all_ranks: Only used in the distributed implementation. A
        :class:`list` of length ``nranks``, where the i-th entry is a
        :class:`numpy.ndarray` representing the global indices of targets in the
        local tree on rank *i*. Each entry can be returned from
        *generate_local_tree*. This argument is only significant on the root rank.

    :return: the potentials computed by *expansion_wrangler*. For the distributed
        implementation, the potentials are gathered and returned on the root rank;
        this function returns *None* on the worker ranks.
    """

    traversal = wrangler.traversal

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "fmm")
    from boxtree.timing import TimingRecorder
    recorder = TimingRecorder()

    src_weight_vecs = [wrangler.reorder_sources(weight) for
        weight in src_weight_vecs]

    src_weight_vecs = wrangler.distribute_source_weights(
        src_weight_vecs, global_src_idx_all_ranks)

    # {{{ "Step 2.1:" Construct local multipoles

    mpole_exps, timing_future = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weight_vecs)

    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    mpole_exps, timing_future = wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps)

    recorder.add("coarsen_multipoles", timing_future)

    # mpole_exps is called Phi in [1]

    # }}}

    wrangler.communicate_mpoles(mpole_exps)

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    potentials, timing_future = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weight_vecs)

    recorder.add("eval_direct", timing_future)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    local_exps, timing_future = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps)

    recorder.add("multipole_to_local", timing_future)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps)

    recorder.add("eval_multipoles", timing_future)

    potentials = potentials + mpole_result

    # these potentials are called beta in [1]

    if traversal.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                "('list 3 close')")

        direct_result, timing_future = wrangler.eval_direct(
                traversal.target_boxes,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_weight_vecs)

        recorder.add("eval_direct", timing_future)

        potentials = potentials + direct_result

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    local_result, timing_future = wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weight_vecs)

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    if traversal.from_sep_close_bigger_starts is not None:
        direct_result, timing_future = wrangler.eval_direct(
                traversal.target_boxes,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_weight_vecs)

        recorder.add("eval_direct", timing_future)

        potentials = potentials + direct_result

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    local_exps, timing_future = wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    local_result, timing_future = wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps)

    recorder.add("eval_locals", timing_future)

    potentials = potentials + local_result

    # }}}

    potentials = wrangler.gather_potential_results(
                    potentials, global_tgt_idx_all_ranks)

    result = wrangler.reorder_potentials(potentials)

    result = wrangler.finalize_potentials(result, template_ary=src_weight_vecs[0])

    fmm_proc.done()

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result


# vim: filetype=pyopencl:fdm=marker
