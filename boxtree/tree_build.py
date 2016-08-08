from __future__ import division, absolute_import

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

from six.moves import range, zip

import numpy as np
from pytools import memoize_method
import pyopencl as cl
import pyopencl.array  # noqa
from functools import partial
from boxtree.tree import Tree

import logging
logger = logging.getLogger(__name__)


class TreeBuilder(object):
    def __init__(self, context):
        """
        :arg context: A :class:`pyopencl.Context`.
        """

        self.context = context

        from boxtree.bounding_box import BoundingBoxFinder
        self.bbox_finder = BoundingBoxFinder(self.context)

        # This is used to map box IDs and compress box lists in empty leaf
        # pruning.

        from boxtree.tools import GappyCopyAndMapKernel
        self.gappy_copy_and_map = GappyCopyAndMapKernel(self.context)

    morton_nr_dtype = np.dtype(np.int8)
    box_level_dtype = np.dtype(np.uint8)

    @memoize_method
    def get_kernel_info(self, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype,
            sources_are_targets, srcntgts_have_extent,
            stick_out_factor, adaptive):

        from boxtree.tree_build_kernels import get_tree_build_kernel_info
        return get_tree_build_kernel_info(self.context, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype,
            sources_are_targets, srcntgts_have_extent,
            stick_out_factor, self.morton_nr_dtype, self.box_level_dtype,
            adaptive=adaptive)

    # {{{ run control

    def __call__(self, queue, particles, max_particles_in_box=None,
            allocator=None, debug=False, targets=None,
            source_radii=None, target_radii=None, stick_out_factor=0.25,
            refine_weights=None, max_leaf_refine_weight=None,
            wait_for=None, non_adaptive=False,
            **kwargs):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` instance
        :arg particles: an object array of (XYZ) point coordinate arrays.
        :arg targets: an object array of (XYZ) point coordinate arrays or ``None``.
            If ``None``, *particles* act as targets, too.
            Must have the same (inner) dtype as *particles*.
        :arg source_radii: If not *None*, a :class:`pyopencl.array.Array` of the
            same dtype as *particles*.
            If this is given, *targets* must also be given, i.e. sources and
            targets must be separate. See :ref:`extent`.

        :arg target_radii: Like *source_radii*, but for targets.
        :arg stick_out_factor: See :attr:`Tree.stick_out_factor` and :ref:`extent`.
        :arg refine_weights: If not *None*, a :class:`pyopencl.array.Array` of the
            type :class:`numpy.float32`. A box will be split if it has a cumulative
            refine_weight greater than 1.
        :arg wait_for: may either be *None* or a list of :class:`pyopencl.Event`
            instances for whose completion this command waits before starting
            exeuction.
        :arg non_adaptive: If *True*, return a tree in which all leaf boxes are
            on the same (last) level. The tree is pruned, in the sense that empty
            boxes have been eliminated.
        :arg kwargs: Used internally for debugging.

        :returns: a tuple ``(tree, event)``, where *tree* is an instance of
            :class:`Tree`, and *event* is a :class:`pyopencl.Event` for dependency
            management.
        """

        # {{{ input processing

        # we'll modify this below, so copy it
        if wait_for is None:
            wait_for = []
        else:
            wait_for = list(wait_for)

        dimensions = len(particles)

        from boxtree.tools import AXIS_NAMES
        axis_names = AXIS_NAMES[:dimensions]

        sources_are_targets = targets is None
        sources_have_extent = source_radii is not None
        targets_have_extent = target_radii is not None
        srcntgts_have_extent = sources_have_extent or targets_have_extent

        if srcntgts_have_extent and targets is None:
            raise ValueError("must specify targets when specifying "
                    "any kind of radii")

        from pytools import single_valued
        particle_id_dtype = np.int32
        box_id_dtype = np.int32
        coord_dtype = single_valued(coord.dtype for coord in particles)

        if targets is None:
            nsrcntgts = single_valued(len(coord) for coord in particles)
        else:
            nsources = single_valued(len(coord) for coord in particles)
            ntargets = single_valued(len(coord) for coord in targets)
            nsrcntgts = nsources + ntargets

        if source_radii is not None:
            if source_radii.shape != (nsources,):
                raise ValueError("source_radii has an invalid shape")

            if source_radii.dtype != coord_dtype:
                raise TypeError("dtypes of coordinate arrays and "
                        "source_radii must agree")

        if target_radii is not None:
            if target_radii.shape != (ntargets,):
                raise ValueError("target_radii has an invalid shape")

            if target_radii.dtype != coord_dtype:
                raise TypeError("dtypes of coordinate arrays and "
                        "target_radii must agree")

        # }}}

        empty = partial(cl.array.empty, queue, allocator=allocator)

        def zeros(shape, dtype):
            result = (cl.array.empty(queue, shape, dtype, allocator=allocator)
                    .fill(0))
            event, = result.events
            return result, event

        knl_info = self.get_kernel_info(dimensions, coord_dtype,
                particle_id_dtype, box_id_dtype,
                sources_are_targets, srcntgts_have_extent,
                stick_out_factor, adaptive=not non_adaptive)

        # {{{ combine sources and targets into one array, if necessary

        prep_events = []

        if targets is None:
            # Targets weren't specified. Sources are also targets. Let's
            # call them "srcntgts".

            srcntgts = particles

            assert source_radii is None
            assert target_radii is None

            srcntgt_radii = None

        else:
            # Here, we mash sources and targets into one array to give us one
            # big array of "srcntgts". In this case, a "srcntgt" is either a
            # source or a target, but not really both, as above. How will we be
            # able to tell which it was? Easy: We'll compare its 'user' id with
            # nsources. If it's >=, it's a target, otherwise it's a source.

            target_coord_dtype = single_valued(tgt_i.dtype for tgt_i in targets)

            if target_coord_dtype != coord_dtype:
                raise TypeError("sources and targets must have same coordinate "
                        "dtype")

            def combine_srcntgt_arrays(ary1, ary2=None):
                if ary2 is None:
                    dtype = ary1.dtype
                else:
                    dtype = ary2.dtype

                result = empty(nsrcntgts, dtype)
                if (ary1 is None) or (ary2 is None):
                    result.fill(0)

                if ary1 is not None and ary1.nbytes:
                    result[:len(ary1)] = ary1

                if ary2 is not None and ary2.nbytes:
                    result[nsources:] = ary2

                return result

            from pytools.obj_array import make_obj_array
            srcntgts = make_obj_array([
                combine_srcntgt_arrays(src_i, tgt_i)
                for src_i, tgt_i in zip(particles, targets)
                ])

            if srcntgts_have_extent:
                srcntgt_radii = combine_srcntgt_arrays(source_radii, target_radii)
            else:
                srcntgt_radii = None

        del source_radii
        del target_radii

        del particles

        user_srcntgt_ids = cl.array.arange(queue, nsrcntgts, dtype=particle_id_dtype,
                allocator=allocator)

        evt, = user_srcntgt_ids.events
        wait_for.append(evt)
        del evt

        # }}}

        # {{{ process refine_weights

        from boxtree.tree_build_kernels import refine_weight_dtype

        specified_max_particles_in_box = max_particles_in_box is not None
        specified_refine_weights = refine_weights is not None and \
            max_leaf_refine_weight is not None

        if specified_max_particles_in_box and specified_refine_weights:
            raise ValueError("may only specify one of max_particles_in_box and "
                    "refine_weights/max_leaf_refine_weight")
        elif not specified_max_particles_in_box and not specified_refine_weights:
            raise ValueError("must specify either max_particles_in_box or "
                    "refine_weights/max_leaf_refine_weight")
        elif specified_max_particles_in_box:
            refine_weights = (
                cl.array.empty(
                    queue, nsrcntgts, refine_weight_dtype, allocator=allocator)
                .fill(1))
            event, = refine_weights.events
            prep_events.append(event)
            max_leaf_refine_weight = max_particles_in_box
        elif specified_refine_weights:
            if refine_weights.dtype != refine_weight_dtype:
                raise TypeError("refine_weights must have dtype '%s'"
                        % refine_weight_dtype)

        if max_leaf_refine_weight < cl.array.max(refine_weights).get():
            raise ValueError(
                    "entries of refine_weights cannot exceed max_leaf_refine_weight")
        if 0 > cl.array.min(refine_weights).get():
            raise ValueError("all entries of refine_weights must be nonnegative")
        if max_leaf_refine_weight <= 0:
            raise ValueError("max_leaf_refine_weight must be positive")

        total_refine_weight = cl.array.sum(
                refine_weights, dtype=np.dtype(np.int64)).get()

        del max_particles_in_box
        del specified_max_particles_in_box
        del specified_refine_weights

        # }}}

        # {{{ find and process bounding box

        bbox, _ = self.bbox_finder(srcntgts, srcntgt_radii, wait_for=wait_for)
        bbox = bbox.get()

        root_extent = max(
                bbox["max_"+ax] - bbox["min_"+ax]
                for ax in axis_names) * (1+1e-4)

        # make bbox square and slightly larger at the top, to ensure scaled
        # coordinates are always < 1
        bbox_min = np.empty(dimensions, coord_dtype)
        for i, ax in enumerate(axis_names):
            bbox_min[i] = bbox["min_"+ax]

        bbox_max = bbox_min + root_extent
        for i, ax in enumerate(axis_names):
            bbox["max_"+ax] = bbox_max[i]

        # }}}

        from pytools import div_ceil

        # {{{ allocate data

        logger.debug("allocating memory")

        # box-local morton bin counts for each particle at the current level
        # only valid from scan -> split'n'sort
        morton_bin_counts = empty(nsrcntgts, dtype=knl_info.morton_bin_count_dtype)

        # (local) morton nrs for each particle at the current level
        # only valid from scan -> split'n'sort
        morton_nrs = empty(nsrcntgts, dtype=self.morton_nr_dtype)

        # 0/1 segment flags
        # invariant to sorting once set
        # (particles are only reordered within a box)
        # valid throughout computation
        box_start_flags, evt = zeros(nsrcntgts, dtype=np.int8)
        prep_events.append(evt)
        srcntgt_box_ids, evt = zeros(nsrcntgts, dtype=box_id_dtype)
        prep_events.append(evt)
        split_box_ids, evt = zeros(nsrcntgts, dtype=box_id_dtype)
        prep_events.append(evt)

        # number of boxes total, and a guess
        nboxes_dev = empty((), dtype=box_id_dtype)
        nboxes_dev.fill(1)

        # /!\ If you're allocating an array here that depends on nboxes_guess,
        # you *must* also write reallocation code down below for the case when
        # nboxes_guess was too low.

        # Outside nboxes_guess feeding is solely for debugging purposes,
        # to test the reallocation code.
        nboxes_guess = kwargs.get("nboxes_guess")
        if nboxes_guess is None:
            nboxes_guess = 2**dimensions * (
                    (max_leaf_refine_weight + total_refine_weight - 1)
                    // max_leaf_refine_weight)

        assert nboxes_guess > 0

        # per-box morton bin counts
        box_morton_bin_counts = empty(nboxes_guess,
                dtype=knl_info.morton_bin_count_dtype)

        # particle# at which each box starts
        box_srcntgt_starts, evt = zeros(nboxes_guess, dtype=particle_id_dtype)
        prep_events.append(evt)

        # pointer to parent box
        box_parent_ids, evt = zeros(nboxes_guess, dtype=box_id_dtype)
        prep_events.append(evt)

        # morton nr identifier {quadr,oct}ant of parent in which this box was created
        box_morton_nrs, evt = zeros(nboxes_guess, dtype=self.morton_nr_dtype)
        prep_events.append(evt)

        # box -> level map
        box_levels, evt = zeros(nboxes_guess, self.box_level_dtype)
        prep_events.append(evt)

        # number of particles in each box
        # needs to be globally initialized because empty boxes never get touched
        box_srcntgt_counts_cumul, evt = zeros(nboxes_guess, dtype=particle_id_dtype)
        prep_events.append(evt)

        # Initalize box 0 to contain all particles
        box_srcntgt_counts_cumul[0].fill(
                nsrcntgts, queue=queue, wait_for=[evt])

        # box -> whether the box has a child
        box_has_children, evt = zeros(nboxes_guess, dtype=np.dtype(np.int32))
        prep_events.append(evt)

        # set parent of root box to itself
        evt = cl.enqueue_copy(
                queue, box_parent_ids.data, np.zeros((), dtype=box_parent_ids.dtype))
        prep_events.append(evt)

        # }}}

        def fin_debug(s):
            if debug:
                queue.finish()

            logger.debug(s)

        from pytools.obj_array import make_obj_array
        have_oversize_split_box, evt = zeros((), np.int32)
        prep_events.append(evt)

        wait_for = prep_events

        # {{{ level loop

        # Level 0 starts at 0 and always contains box 0 and nothing else.
        # Level 1 therefore starts at 1.
        level_start_box_nrs = [0, 1]

        from time import time
        start_time = time()
        if total_refine_weight > max_leaf_refine_weight:
            level = 1
        else:
            level = 0

        # INVARIANTS -- Upon entry to this loop:
        #
        # - level is the level being built.
        # - the last entry of level_start_box_nrs is the beginning of the level
        #   to be built

        # This while condition prevents entering the loop in case there's just a
        # single box, by how 'level' is set above. Read this as 'while True' with
        # an edge case.

        logger.debug("entering level loop with %s srcntgts" % nsrcntgts)

        while level:
            if debug:
                # More invariants:
                assert level == len(level_start_box_nrs) - 1

            if level > np.iinfo(self.box_level_dtype).max:
                raise RuntimeError("level count exceeded maximum")

            common_args = ((morton_bin_counts, morton_nrs,
                    box_start_flags, srcntgt_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    refine_weights,
                    max_leaf_refine_weight,
                    box_srcntgt_starts, box_srcntgt_counts_cumul,
                    box_parent_ids, box_morton_nrs,
                    nboxes_dev,
                    level, bbox,
                    user_srcntgt_ids)
                    + tuple(srcntgts)
                    + ((srcntgt_radii,) if srcntgts_have_extent else ())
                    )

            fin_debug("morton count scan")

            # writes: box_morton_bin_counts, morton_nrs
            evt = knl_info.morton_count_scan(
                    *common_args, queue=queue, size=nsrcntgts,
                    wait_for=wait_for)
            wait_for = [evt]

            fin_debug("split box id scan")

            # writes: nboxes_dev, split_box_ids
            evt = knl_info.split_box_id_scan(
                    srcntgt_box_ids,
                    box_srcntgt_starts,
                    box_srcntgt_counts_cumul,
                    box_morton_bin_counts,
                    refine_weights,
                    max_leaf_refine_weight,
                    box_levels,
                    level,

                    # input/output:
                    nboxes_dev,

                    # output:
                    box_has_children,
                    split_box_ids,
                    queue=queue, size=nsrcntgts, wait_for=wait_for)
            wait_for = [evt]

            nboxes_new = int(nboxes_dev.get())

            # Assumption: Everything between here and the top of the loop must
            # be repeatable, so that in an out-of-memory situation, we can just
            # rerun this bit of the code after reallocating and a minimal reset
            # procedure.

            # {{{ reallocate and retry if nboxes_guess was too small

            if nboxes_new > nboxes_guess:
                fin_debug("starting nboxes_guess increase")

                while nboxes_guess < nboxes_new:
                    nboxes_guess *= 2

                from boxtree.tools import realloc_array
                my_realloc = partial(realloc_array, new_shape=nboxes_guess,
                        zero_fill=False, queue=queue, wait_for=wait_for)
                my_realloc_zeros = partial(realloc_array, new_shape=nboxes_guess,
                        zero_fill=True, queue=queue, wait_for=wait_for)

                resize_events = []
                box_morton_bin_counts, evt = my_realloc(box_morton_bin_counts)
                resize_events.append(evt)

                box_srcntgt_starts, evt = my_realloc_zeros(box_srcntgt_starts)
                resize_events.append(evt)
                box_parent_ids, evt = my_realloc_zeros(box_parent_ids)
                resize_events.append(evt)
                box_morton_nrs, evt = my_realloc_zeros(box_morton_nrs)
                resize_events.append(evt)
                box_levels, evt = my_realloc_zeros(box_levels)
                resize_events.append(evt)
                box_srcntgt_counts_cumul, evt = \
                        my_realloc_zeros(box_srcntgt_counts_cumul)
                resize_events.append(evt)
                box_has_children, evt = my_realloc_zeros(box_has_children)
                resize_events.append(evt)

                del my_realloc
                del my_realloc_zeros

                # reset nboxes_dev to previous value
                nboxes_dev.fill(level_start_box_nrs[-1])
                resize_events.append(evt)

                wait_for = resize_events

                # retry
                logger.info("nboxes_guess exceeded: "
                        "enlarged allocations, restarting level")

                continue

            # }}}

            logger.info("LEVEL %d -> %d boxes" % (level, nboxes_new))

            assert level_start_box_nrs[-1] != nboxes_new or srcntgts_have_extent

            if level_start_box_nrs[-1] == nboxes_new:
                # We haven't created new boxes in this level loop trip.  Unless
                # srcntgts have extent, this should never happen.  (I.e., we
                # should've never entered this loop trip.)
                #
                # If srcntgts have extent, this can happen if boxes were
                # in-principle overfull, but couldn't subdivide because of
                # extent restrictions.

                assert srcntgts_have_extent

                level -= 1

                logger.debug("no new boxes created this loop trip")
                break

            level_start_box_nrs.append(nboxes_new)
            del nboxes_new

            new_user_srcntgt_ids = cl.array.empty_like(user_srcntgt_ids)
            new_srcntgt_box_ids = cl.array.empty_like(srcntgt_box_ids)
            split_and_sort_args = (
                    common_args
                    + (new_user_srcntgt_ids, have_oversize_split_box,
                        new_srcntgt_box_ids, box_levels,
                        box_has_children))
            fin_debug("split and sort")

            evt = knl_info.split_and_sort_kernel(*split_and_sort_args,
                    wait_for=wait_for)
            wait_for = [evt]

            if debug:
                level_bl_chunk = box_levels.get()[
                        level_start_box_nrs[-2]:level_start_box_nrs[-1]]
                assert ((level_bl_chunk == level) | (level_bl_chunk == 0)).all()
                del level_bl_chunk

            if debug:
                assert (box_srcntgt_starts.get() < nsrcntgts).all()

            user_srcntgt_ids = new_user_srcntgt_ids
            del new_user_srcntgt_ids
            srcntgt_box_ids = new_srcntgt_box_ids
            del new_srcntgt_box_ids

            if not int(have_oversize_split_box.get()):
                logger.debug("no overfull boxes left")
                break

            level += 1

            have_oversize_split_box.fill(0)

        end_time = time()
        elapsed = end_time-start_time
        npasses = level+1
        logger.info("elapsed time: %g s (%g s/particle/pass)" % (
                elapsed, elapsed/(npasses*nsrcntgts)))
        del npasses

        nboxes = int(nboxes_dev.get())

        # }}}

        # {{{ extract number of non-child srcntgts from box morton counts

        if srcntgts_have_extent:
            box_srcntgt_counts_nonchild = empty(nboxes, particle_id_dtype)
            fin_debug("extract non-child srcntgt count")

            assert len(level_start_box_nrs) >= 2
            highest_possibly_split_box_nr = level_start_box_nrs[-2]

            evt = knl_info.extract_nonchild_srcntgt_count_kernel(
                    # input
                    box_morton_bin_counts,
                    box_srcntgt_counts_cumul,
                    highest_possibly_split_box_nr,

                    # output
                    box_srcntgt_counts_nonchild,

                    range=slice(nboxes), wait_for=wait_for)
            wait_for = [evt]

            del highest_possibly_split_box_nr

            if debug:
                assert (box_srcntgt_counts_nonchild.get()
                        <= box_srcntgt_counts_cumul.get()[:nboxes]).all()

        # }}}

        del morton_nrs
        del box_morton_bin_counts

        # {{{ prune empty leaf boxes

        is_pruned = not kwargs.get("skip_prune")
        if is_pruned:

            # What is the original index of this box?
            from_box_id = empty(nboxes, box_id_dtype)

            # Where should I put this box?
            to_box_id = empty(nboxes, box_id_dtype)

            fin_debug("find prune indices")

            nboxes_post_prune_dev = empty((), dtype=box_id_dtype)
            evt = knl_info.find_prune_indices_kernel(
                    box_srcntgt_counts_cumul,
                    to_box_id, from_box_id, nboxes_post_prune_dev,
                    size=nboxes, wait_for=wait_for)
            wait_for = [evt]

            fin_debug("prune copy")

            nboxes_post_prune = int(nboxes_post_prune_dev.get())

            logger.info("%d empty leaves" % (nboxes-nboxes_post_prune))

            prune_events = []

            prune_empty = partial(self.gappy_copy_and_map,
                    queue, allocator, nboxes_post_prune, from_box_id)

            box_srcntgt_starts, evt = prune_empty(box_srcntgt_starts)
            prune_events.append(evt)

            box_srcntgt_counts_cumul, evt = prune_empty(box_srcntgt_counts_cumul)
            prune_events.append(evt)

            if debug:
                assert (box_srcntgt_counts_cumul.get() > 0).all()

            srcntgt_box_ids = cl.array.take(to_box_id, srcntgt_box_ids)

            box_parent_ids, evt = prune_empty(box_parent_ids, map_values=to_box_id)
            prune_events.append(evt)
            box_morton_nrs, evt = prune_empty(box_morton_nrs)
            prune_events.append(evt)
            box_levels, evt = prune_empty(box_levels)
            prune_events.append(evt)
            if srcntgts_have_extent:
                box_srcntgt_counts_nonchild, evt = prune_empty(
                        box_srcntgt_counts_nonchild)
                prune_events.append(evt)

            box_has_children, evt = prune_empty(box_has_children)
            prune_events.append(evt)

            # Remap level_start_box_nrs to new box IDs.
            # FIXME: It would be better to do this on the device.
            level_start_box_nrs = list(
                    to_box_id.get()
                    [np.array(level_start_box_nrs[:-1], box_id_dtype)])
            level_start_box_nrs = level_start_box_nrs + [nboxes_post_prune]

            wait_for = prune_events
        else:
            logger.info("skipping empty-leaf pruning")
            nboxes_post_prune = nboxes

        level_start_box_nrs = np.array(level_start_box_nrs, box_id_dtype)

        # }}}

        del nboxes

        # {{{ compute source/target particle indices and counts in each box

        if targets is None:
            from boxtree.tools import reverse_index_array
            user_source_ids = user_srcntgt_ids
            sorted_target_ids = reverse_index_array(user_srcntgt_ids)

            box_source_starts = box_target_starts = box_srcntgt_starts
            box_source_counts_cumul = box_target_counts_cumul = \
                    box_srcntgt_counts_cumul
            if srcntgts_have_extent:
                box_source_counts_nonchild = box_target_counts_nonchild = \
                        box_srcntgt_counts_nonchild
        else:
            source_numbers = empty(nsrcntgts, particle_id_dtype)

            fin_debug("source counter")
            evt = knl_info.source_counter(user_srcntgt_ids, nsources,
                    source_numbers, queue=queue, allocator=allocator,
                    wait_for=wait_for)
            wait_for = [evt]

            user_source_ids = empty(nsources, particle_id_dtype)
            # srcntgt_target_ids is temporary until particle permutation is done
            srcntgt_target_ids = empty(ntargets, particle_id_dtype)
            sorted_target_ids = empty(ntargets, particle_id_dtype)

            # need to use zeros because parent boxes won't be initialized
            box_source_starts, evt = zeros(nboxes_post_prune, particle_id_dtype)
            wait_for.append(evt)
            box_source_counts_cumul, evt = zeros(
                    nboxes_post_prune, particle_id_dtype)
            wait_for.append(evt)
            box_target_starts, evt = zeros(
                    nboxes_post_prune, particle_id_dtype)
            wait_for.append(evt)
            box_target_counts_cumul, evt = zeros(
                    nboxes_post_prune, particle_id_dtype)
            wait_for.append(evt)

            if srcntgts_have_extent:
                box_source_counts_nonchild, evt = zeros(
                        nboxes_post_prune, particle_id_dtype)
                wait_for.append(evt)
                box_target_counts_nonchild, evt = zeros(
                        nboxes_post_prune, particle_id_dtype)
                wait_for.append(evt)

            fin_debug("source and target index finder")
            evt = knl_info.source_and_target_index_finder(*(
                # input:
                (
                    user_srcntgt_ids, nsources, srcntgt_box_ids,
                    box_parent_ids,

                    box_srcntgt_starts, box_srcntgt_counts_cumul,
                    source_numbers,
                )
                + ((box_srcntgt_counts_nonchild,)
                    if srcntgts_have_extent else ())

                # output:
                + (
                    user_source_ids, srcntgt_target_ids, sorted_target_ids,
                    box_source_starts, box_source_counts_cumul,
                    box_target_starts, box_target_counts_cumul,
                    )
                + ((
                    box_source_counts_nonchild,
                    box_target_counts_nonchild,
                    ) if srcntgts_have_extent else ())
                ),
                queue=queue, range=slice(nsrcntgts),
                wait_for=wait_for)
            wait_for = [evt]

            if srcntgts_have_extent:
                if debug:
                    assert (
                            box_srcntgt_counts_nonchild.get()
                            ==
                            (box_source_counts_nonchild
                            + box_target_counts_nonchild).get()).all()

            if debug:
                usi_host = user_source_ids.get()
                assert (usi_host < nsources).all()
                assert (0 <= usi_host).all()
                del usi_host

                sti_host = srcntgt_target_ids.get()
                assert (sti_host < nsources+ntargets).all()
                assert (nsources <= sti_host).all()
                del sti_host

                assert (box_source_counts_cumul.get()
                        + box_target_counts_cumul.get()
                        == box_srcntgt_counts_cumul.get()).all()

            del source_numbers

        del box_srcntgt_starts
        if srcntgts_have_extent:
            del box_srcntgt_counts_nonchild

        # }}}

        # {{{ permute and source/target-split (if necessary) particle array

        if targets is None:
            sources = targets = make_obj_array([
                cl.array.empty_like(pt) for pt in srcntgts])

            fin_debug("srcntgt permuter (particles)")
            evt = knl_info.srcntgt_permuter(
                    user_srcntgt_ids,
                    *(tuple(srcntgts) + tuple(sources)),
                    wait_for=wait_for)
            wait_for = [evt]

            assert srcntgt_radii is None

        else:
            sources = make_obj_array([
                empty(nsources, coord_dtype) for i in range(dimensions)])
            fin_debug("srcntgt permuter (sources)")
            evt = knl_info.srcntgt_permuter(
                    user_source_ids,
                    *(tuple(srcntgts) + tuple(sources)),
                    queue=queue, range=slice(nsources),
                    wait_for=wait_for)
            wait_for = [evt]

            targets = make_obj_array([
                empty(ntargets, coord_dtype) for i in range(dimensions)])
            fin_debug("srcntgt permuter (targets)")
            evt = knl_info.srcntgt_permuter(
                    srcntgt_target_ids,
                    *(tuple(srcntgts) + tuple(targets)),
                    queue=queue, range=slice(ntargets),
                    wait_for=wait_for)
            wait_for = [evt]

            if srcntgt_radii is not None:
                fin_debug("srcntgt permuter (source radii)")
                source_radii = cl.array.take(
                        srcntgt_radii, user_source_ids, queue=queue,
                        wait_for=wait_for)

                fin_debug("srcntgt permuter (target radii)")
                target_radii = cl.array.take(
                        srcntgt_radii, srcntgt_target_ids, queue=queue,
                        wait_for=wait_for)

                wait_for = source_radii.events + target_radii.events

            del srcntgt_target_ids

        del srcntgt_radii

        # }}}

        del srcntgts

        nlevels = len(level_start_box_nrs) - 1
        assert level + 1 == nlevels, (level+1, nlevels)
        if debug:
            max_level = np.max(box_levels.get())

            assert max_level + 1 == nlevels

        # {{{ compute box info

        # A number of arrays below are nominally 2-dimensional and stored with
        # the box index as the fastest-moving index. To make sure that accesses
        # remain aligned, we round up the number of boxes used for indexing.
        aligned_nboxes = div_ceil(nboxes_post_prune, 32)*32

        box_child_ids, evt = zeros((2**dimensions, aligned_nboxes), box_id_dtype)
        wait_for.append(evt)
        box_centers = empty((dimensions, aligned_nboxes), coord_dtype)

        from boxtree.tree import box_flags_enum
        box_flags = empty(nboxes_post_prune, box_flags_enum.dtype)

        if not srcntgts_have_extent:
            # If srcntgts_have_extent, then non-child counts have already been
            # computed, and we have nothing to do here. But if not, then
            # we must fill these non-child counts. This amounts to copying
            # the cumulative counts and setting them to zero for non-leaves.

            # {{{ make sure box_{source,target}_counts_nonchild are not defined
            # (before we overwrite them)

            try:
                box_source_counts_nonchild
            except NameError:
                pass
            else:
                assert False

            try:
                box_target_counts_nonchild
            except NameError:
                pass
            else:
                assert False

            # }}}

            box_source_counts_nonchild, evt = zeros(
                    nboxes_post_prune, particle_id_dtype)
            wait_for.append(evt)

            if sources_are_targets:
                box_target_counts_nonchild = box_source_counts_nonchild
            else:
                box_target_counts_nonchild, evt = zeros(
                        nboxes_post_prune, particle_id_dtype)
                wait_for.append(evt)

        fin_debug("compute box info")
        evt = knl_info.box_info_kernel(
                *(
                    # input:
                    box_parent_ids, box_morton_nrs, bbox, aligned_nboxes,

                    box_srcntgt_counts_cumul,
                    box_source_counts_cumul, box_target_counts_cumul,
                    box_has_children, box_levels, nlevels,

                    # output if srcntgts_have_extent, input+output otherwise
                    box_source_counts_nonchild, box_target_counts_nonchild,

                    # output:
                    box_child_ids, box_centers, box_flags,
                ),
                range=slice(nboxes_post_prune),
                wait_for=wait_for)

        # }}}

        del box_has_children

        # {{{ build output

        extra_tree_attrs = {}

        if sources_have_extent:
            extra_tree_attrs.update(source_radii=source_radii)
        if targets_have_extent:
            extra_tree_attrs.update(target_radii=target_radii)

        logger.info("tree build complete")

        return Tree(
                # If you change this, also change the documentation
                # of what's in the tree, above.

                sources_are_targets=sources_are_targets,
                sources_have_extent=sources_have_extent,
                targets_have_extent=targets_have_extent,

                particle_id_dtype=knl_info.particle_id_dtype,
                box_id_dtype=knl_info.box_id_dtype,
                coord_dtype=coord_dtype,
                box_level_dtype=self.box_level_dtype,

                root_extent=root_extent,
                stick_out_factor=stick_out_factor,

                bounding_box=(bbox_min, bbox_max),
                level_start_box_nrs=level_start_box_nrs,
                level_start_box_nrs_dev=cl.array.to_device(
                    queue, level_start_box_nrs,
                    allocator=allocator),

                sources=sources,
                targets=targets,

                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                box_source_counts_cumul=box_source_counts_cumul,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                box_target_counts_cumul=box_target_counts_cumul,

                box_parent_ids=box_parent_ids,
                box_child_ids=box_child_ids,
                box_centers=box_centers,
                box_levels=box_levels,
                box_flags=box_flags,

                user_source_ids=user_source_ids,
                sorted_target_ids=sorted_target_ids,

                _is_pruned=is_pruned,

                **extra_tree_attrs
                ).with_queue(None), evt

        # }}}

    # }}}

# vim: foldmethod=marker:filetype=pyopencl
