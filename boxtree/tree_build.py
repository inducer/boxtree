"""
.. currentmodule:: boxtree

Building Trees
--------------

.. autoclass:: TreeBuilder
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


import numpy as np
from pytools import memoize_method
import pyopencl as cl
import pyopencl.array  # noqa
from functools import partial
from boxtree.tree import Tree
from pytools import ProcessLogger, DebugProcessLogger

import logging
logger = logging.getLogger(__name__)


class MaxLevelsExceeded(RuntimeError):
    pass


class TreeBuilder:
    """
    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, context):
        """
        :arg context: A :class:`pyopencl.Context`.
        """

        self.context = context

        from boxtree.bounding_box import BoundingBoxFinder
        self.bbox_finder = BoundingBoxFinder(self.context)

        # This is used to map box IDs and compress box lists in empty leaf
        # pruning.

        from boxtree.tools import GappyCopyAndMapKernel, MapValuesKernel
        self.gappy_copy_and_map = GappyCopyAndMapKernel(self.context)
        self.map_values_kernel = MapValuesKernel(self.context)

    morton_nr_dtype = np.dtype(np.int8)
    box_level_dtype = np.dtype(np.uint8)
    ROOT_EXTENT_STRETCH_FACTOR = 1e-4

    @memoize_method
    def get_kernel_info(self, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype,
            sources_are_targets, srcntgts_extent_norm,
            kind):

        from boxtree.tree_build_kernels import get_tree_build_kernel_info
        return get_tree_build_kernel_info(self.context, dimensions, coord_dtype,
            particle_id_dtype, box_id_dtype,
            sources_are_targets, srcntgts_extent_norm,
            self.morton_nr_dtype, self.box_level_dtype,
            kind=kind)

    # {{{ run control

    def __call__(self, queue, particles, kind="adaptive",
            max_particles_in_box=None, allocator=None, debug=False,
            targets=None, source_radii=None, target_radii=None,
            stick_out_factor=None, refine_weights=None,
            max_leaf_refine_weight=None, wait_for=None,
            extent_norm=None, bbox=None,
            **kwargs):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` instance
        :arg particles: an object array of (XYZ) point coordinate arrays.
        :arg kind: One of the following strings:

            - 'adaptive'
            - 'adaptive-level-restricted'
            - 'non-adaptive'

            'adaptive' requests an adaptive tree without level restriction.  See
            :ref:`tree-kinds` for further explanation.

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
            type :class:`numpy.int32`. A box will be split if it has a cumulative
            refine_weight greater than *max_leaf_refine_weight*. If this is given,
            *max_leaf_refine_weight* must also be given and *max_particles_in_box*
            must be *None*.
        :arg max_leaf_refine_weight: If not *None*, specifies the maximum weight
            of a leaf box.
        :arg max_particles_in_box: If not *None*, specifies the maximum number
            of particles in a leaf box. If this is given, both
            *refine_weights* and *max_leaf_refine_weight* must be *None*.
        :arg wait_for: may either be *None* or a list of :class:`pyopencl.Event`
            instances for whose completion this command waits before starting
            execution.
        :arg extent_norm: ``"l2"`` or ``"linf"``. Indicates the norm with respect
            to which particle stick-out is measured. See :attr:`Tree.extent_norm`.
        :arg bbox: Bounding box of either type:
            1. A dim-by-2 array, with each row to be [min, max] coordinates
            in its corresponding axis direction.
            2. (Internal use only) of the same type as returned by
            *boxtree.bounding_box.make_bounding_box_dtype*.
            When given, this bounding box is used for tree
            building. Otherwise, the bounding box is determined from particles
            in such a way that it is square and is slightly larger at the top (so
            that scaled coordinates are always < 1).
            When supplied, the bounding box must be square and have all the
            particles in its closure.
        :arg kwargs: Used internally for debugging.

        :returns: a tuple ``(tree, event)``, where *tree* is an instance of
            :class:`Tree`, and *event* is a :class:`pyopencl.Event` for dependency
            management.
        """

        # {{{ input processing

        if kind not in ["adaptive", "adaptive-level-restricted", "non-adaptive"]:
            raise ValueError(f"unknown tree kind '{kind}'")

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

        if extent_norm is None:
            extent_norm = "linf"

        if extent_norm not in ["linf", "l2"]:
            raise ValueError("unexpected value of 'extent_norm': %s"
                    % extent_norm)

        srcntgts_extent_norm = extent_norm
        srcntgts_have_extent = sources_have_extent or targets_have_extent
        if not srcntgts_have_extent:
            srcntgts_extent_norm = None

        del extent_norm

        if srcntgts_extent_norm and targets is None:
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

        if sources_have_extent or targets_have_extent:
            if stick_out_factor is None:
                raise ValueError("if sources or targets have extent, "
                        "stick_out_factor must be explicitly specified")
        else:
            stick_out_factor = 0

        # }}}

        empty = partial(cl.array.empty, queue, allocator=allocator)

        def zeros(shape, dtype):
            result = cl.array.zeros(queue, shape, dtype, allocator=allocator)
            if result.events:
                event, = result.events
            else:
                from numbers import Number
                if isinstance(shape, Number):
                    shape = (shape,)
                from pytools import product
                assert product(shape) == 0
                event = cl.enqueue_marker(queue)

            return result, event

        knl_info = self.get_kernel_info(dimensions, coord_dtype,
                particle_id_dtype, box_id_dtype,
                sources_are_targets, srcntgts_extent_norm,
                kind=kind)

        logger.debug("tree build: start")

        # {{{ combine sources and targets into one array, if necessary

        prep_events = []

        if targets is None:
            # Targets weren't specified. Sources are also targets. Let's
            # call them "srcntgts".

            if isinstance(particles, np.ndarray) and particles.dtype.char == "O":
                srcntgts = particles
            else:
                from pytools.obj_array import make_obj_array
                srcntgts = make_obj_array([
                    p.with_queue(queue).copy() for p in particles
                    ])

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

        if bbox is None:
            bbox, _ = self.bbox_finder(srcntgts, srcntgt_radii, wait_for=wait_for)
            bbox = bbox.get()

            root_extent = max(
                bbox["max_"+ax] - bbox["min_"+ax]
                for ax in axis_names) * (1+TreeBuilder.ROOT_EXTENT_STRETCH_FACTOR)

            # make bbox square and slightly larger at the top, to ensure scaled
            # coordinates are always < 1
            bbox_min = np.empty(dimensions, coord_dtype)
            for i, ax in enumerate(axis_names):
                bbox_min[i] = bbox["min_"+ax]

            bbox_max = bbox_min + root_extent
            for i, ax in enumerate(axis_names):
                bbox["max_"+ax] = bbox_max[i]
        else:
            # Validate that bbox is a superset of particle-derived bbox
            bbox_auto, _ = self.bbox_finder(
                    srcntgts, srcntgt_radii, wait_for=wait_for)
            bbox_auto = bbox_auto.get()

            # Convert unstructured numpy array to bbox_type
            if isinstance(bbox, np.ndarray):
                if len(bbox) == dimensions:
                    bbox_bak = bbox.copy()
                    bbox = np.empty(1, bbox_auto.dtype)
                    for i, ax in enumerate(axis_names):
                        bbox["min_"+ax] = bbox_bak[i][0]
                        bbox["max_"+ax] = bbox_bak[i][1]
                else:
                    assert len(bbox) == 1
            else:
                raise NotImplementedError("Unsupported bounding box type: "
                        + str(type(bbox)))

            # bbox must cover bbox_auto
            bbox_min = np.empty(dimensions, coord_dtype)
            bbox_max = np.empty(dimensions, coord_dtype)

            for i, ax in enumerate(axis_names):
                bbox_min[i] = bbox["min_" + ax]
                bbox_max[i] = bbox["max_" + ax]
                assert bbox_min[i] < bbox_max[i]
                assert bbox_min[i] <= bbox_auto["min_" + ax]
                assert bbox_max[i] >= bbox_auto["max_" + ax]

            # bbox must be a square
            bbox_exts = bbox_max - bbox_min
            for ext in bbox_exts:
                assert abs(ext - bbox_exts[0]) < 1e-15

            root_extent = bbox_exts[0]

        # }}}

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

        # Outside nboxes_guess feeding is solely for debugging purposes,
        # to test the reallocation code.
        nboxes_guess = kwargs.get("nboxes_guess")
        if nboxes_guess is None:
            nboxes_guess = 2**dimensions * (
                    (max_leaf_refine_weight + total_refine_weight - 1)
                    // max_leaf_refine_weight)

        assert nboxes_guess > 0

        # /!\ IMPORTANT
        #
        # If you're allocating an array here that depends on nboxes_guess, or if
        # your array contains box numbers, you have to write code for the
        # following down below as well:
        #
        # * You *must* write reallocation code to handle box renumbering and
        #   reallocation triggered at the top of the level loop.
        #
        # * If your array persists after the level loop, you *must* write code
        #   to handle box renumbering and reallocation triggered by the box
        #   pruning step.

        split_box_ids, evt = zeros(nboxes_guess, dtype=box_id_dtype)
        prep_events.append(evt)

        # per-box morton bin counts
        box_morton_bin_counts, evt = zeros(nboxes_guess,
                                      dtype=knl_info.morton_bin_count_dtype)
        prep_events.append(evt)

        # particle# at which each box starts
        box_srcntgt_starts, evt = zeros(nboxes_guess, dtype=particle_id_dtype)
        prep_events.append(evt)

        # pointer to parent box
        box_parent_ids, evt = zeros(nboxes_guess, dtype=box_id_dtype)
        prep_events.append(evt)

        # pointer to child box, by morton number
        box_child_ids, evts = zip(
            *(zeros(nboxes_guess, dtype=box_id_dtype) for d in range(2**dimensions)))
        prep_events.extend(evts)

        # box centers, by dimension
        box_centers, evts = zip(
            *(zeros(nboxes_guess, dtype=coord_dtype) for d in range(dimensions)))
        prep_events.extend(evts)

        # Initialize box_centers[0] to contain the root box's center
        for d, (ax, evt) in enumerate(zip(axis_names, evts)):
            center_ax = bbox["min_"+ax] + (bbox["max_"+ax] - bbox["min_"+ax]) / 2
            box_centers[d][0].fill(center_ax, wait_for=[evt])

        # box -> level map
        box_levels, evt = zeros(nboxes_guess, self.box_level_dtype)
        prep_events.append(evt)

        # number of particles in each box
        # needs to be globally initialized because empty boxes never get touched
        box_srcntgt_counts_cumul, evt = zeros(nboxes_guess, dtype=particle_id_dtype)
        prep_events.append(evt)

        # Initialize box 0 to contain all particles
        box_srcntgt_counts_cumul[0].fill(
                nsrcntgts, queue=queue, wait_for=[evt])

        # box -> whether the box has a child. FIXME: use smaller integer type
        box_has_children, evt = zeros(nboxes_guess, dtype=np.dtype(np.int32))
        prep_events.append(evt)

        # box -> whether the box needs a splitting to enforce level restriction.
        # FIXME: use smaller integer type
        force_split_box, evt = zeros(nboxes_guess
                                     if knl_info.level_restrict
                                     else 0, dtype=np.dtype(np.int32))
        prep_events.append(evt)

        # set parent of root box to itself
        evt = cl.enqueue_copy(
                queue, box_parent_ids.data, np.zeros((), dtype=box_parent_ids.dtype))
        prep_events.append(evt)

        # 2*(num bits in the significand)
        # https://gitlab.tiker.net/inducer/boxtree/issues/23
        nlevels_max = 2*(np.finfo(coord_dtype).nmant + 1)
        assert nlevels_max <= np.iinfo(self.box_level_dtype).max

        # level -> starting box on level
        level_start_box_nrs_dev, evt = zeros(nlevels_max, dtype=box_id_dtype)
        prep_events.append(evt)

        # level -> number of used boxes on level
        level_used_box_counts_dev, evt = zeros(nlevels_max, dtype=box_id_dtype)
        prep_events.append(evt)

        # }}}

        def fin_debug(s):
            if debug:
                queue.finish()

            logger.debug(s)

        from pytools.obj_array import make_obj_array
        have_oversize_split_box, evt = zeros((), np.int32)
        prep_events.append(evt)

        # True if and only if the level restrict kernel found a box to split in
        # order to enforce level restriction.
        have_upper_level_split_box, evt = zeros((), np.int32)
        prep_events.append(evt)

        wait_for = prep_events

        from pytools import div_ceil

        # {{{ level loop

        # Level 0 starts at 0 and always contains box 0 and nothing else.
        # Level 1 therefore starts at 1.
        level_start_box_nrs = [0, 1]
        level_start_box_nrs_dev[0] = 0
        level_start_box_nrs_dev[1] = 1
        wait_for.extend(level_start_box_nrs_dev.events)

        # This counts the number of boxes that have been used per level. Note
        # that this could be fewer than the actual number of boxes allocated to
        # the level (in the case of building a level restricted tree, more boxes
        # are pre-allocated for a level than used since we may decide to split
        # parent level boxes later).
        level_used_box_counts = [1]
        level_used_box_counts_dev[0] = 1
        wait_for.extend(level_used_box_counts_dev.events)

        # level -> number of leaf boxes on level. Initially the root node is a
        # leaf.
        level_leaf_counts = np.array([1])

        tree_build_proc = ProcessLogger(logger, "tree build")

        if total_refine_weight > max_leaf_refine_weight:
            level = 1
        else:
            level = 0

        # INVARIANTS -- Upon entry to this loop:
        #
        # - level is the level being built.
        # - the last entry of level_start_box_nrs is the beginning of the level
        #   to be built
        # - the last entry of level_used_box_counts is the number of boxes that
        #   are used (not just allocated) at the previous level

        # This while condition prevents entering the loop in case there's just a
        # single box, by how 'level' is set above. Read this as 'while True' with
        # an edge case.

        level_loop_proc = DebugProcessLogger(logger, "tree build level loop")

        # When doing level restriction, the level loop may need to be entered
        # one more time after creating all the levels (see fixme note below
        # regarding this). This flag is set to True when that happens.
        final_level_restrict_iteration = False

        while level:
            if debug:
                # More invariants:
                assert level == len(level_start_box_nrs) - 1
                assert level == len(level_used_box_counts)
                assert level == len(level_leaf_counts)

            if level + 1 >= nlevels_max:  # level is zero-based
                raise MaxLevelsExceeded("Level count exceeded number of significant "
                        "bits in coordinate dtype. That means that a large number "
                        "of particles was indistinguishable up to floating point "
                        "precision (because they ended up in the same box).")

            common_args = ((morton_bin_counts, morton_nrs,
                    box_start_flags,
                    srcntgt_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    refine_weights,
                    max_leaf_refine_weight,
                    box_srcntgt_starts, box_srcntgt_counts_cumul,
                    box_parent_ids, box_levels,
                    level, bbox,
                    user_srcntgt_ids)
                    + tuple(srcntgts)
                    + ((srcntgt_radii,) if srcntgts_have_extent else ())
                    )

            fin_debug("morton count scan")

            morton_count_args = common_args
            if srcntgts_have_extent:
                morton_count_args += (stick_out_factor,)

            # writes: box_morton_bin_counts
            evt = knl_info.morton_count_scan(
                    *morton_count_args, queue=queue, size=nsrcntgts,
                    wait_for=wait_for)
            wait_for = [evt]

            fin_debug("split box id scan")

            # writes: box_has_children, split_box_ids
            evt = knl_info.split_box_id_scan(
                    srcntgt_box_ids,
                    box_srcntgt_counts_cumul,
                    box_morton_bin_counts,
                    refine_weights,
                    max_leaf_refine_weight,
                    box_levels,
                    level_start_box_nrs_dev,
                    level_used_box_counts_dev,
                    force_split_box,
                    level,

                    # output:
                    box_has_children,
                    split_box_ids,
                    have_oversize_split_box,

                    queue=queue,
                    size=level_start_box_nrs[level],
                    wait_for=wait_for)
            wait_for = [evt]

            # {{{ compute new level_used_box_counts, level_leaf_counts

            # The last split_box_id on each level tells us how many boxes are
            # needed at the next level.
            new_level_used_box_counts = [1]
            for level_start_box_id in level_start_box_nrs[1:]:
                last_box_on_prev_level = level_start_box_id - 1
                new_level_used_box_counts.append(
                    # FIXME: Get this all at once.
                    int(split_box_ids[last_box_on_prev_level].get())
                    - level_start_box_id)

            # New leaf count =
            #   old leaf count
            #   + nr. new boxes from splitting parent's leaves
            #   - nr. new boxes from splitting current level's leaves / 2**d
            level_used_box_counts_diff = (new_level_used_box_counts
                    - np.append(level_used_box_counts, [0]))
            new_level_leaf_counts = (level_leaf_counts
                    + level_used_box_counts_diff[:-1]
                    - level_used_box_counts_diff[1:] // 2 ** dimensions)
            new_level_leaf_counts = np.append(
                    new_level_leaf_counts,
                    [level_used_box_counts_diff[-1]])
            del level_used_box_counts_diff

            # }}}

            # Assumption: Everything between here and the top of the loop must
            # be repeatable, so that in an out-of-memory situation, we can just
            # rerun this bit of the code after reallocating and a minimal reset
            # procedure.

            # The algorithm for deciding on level sizes is as follows:
            # 1. Compute the minimal necessary size of each level, including the
            #    new level being created.
            # 2. If level restricting, add padding to the new level being created.
            # 3. Check if there is enough existing space for each level.
            # 4. If any level does not have sufficient space, reallocate all levels:
            #    4a. Compute new sizes of upper levels
            #    4b. If level restricting, add padding to all levels.

            curr_upper_level_lengths = np.diff(level_start_box_nrs)
            minimal_upper_level_lengths = np.max(
                [new_level_used_box_counts[:-1], curr_upper_level_lengths], axis=0)
            minimal_new_level_length = new_level_used_box_counts[-1]

            # Allocate extra space at the end of the current level for higher
            # level leaves that may be split later.
            #
            # If there are no further levels to split (i.e.
            # have_oversize_split_box = 0), then we do not need to allocate any
            # extra space, since no new leaves can be created at the bottom
            # level.
            if knl_info.level_restrict and have_oversize_split_box.get():
                # Currently undocumented.
                lr_lookbehind_levels = kwargs.get("lr_lookbehind", 1)
                minimal_new_level_length += sum(
                    2**(lev*dimensions) * new_level_leaf_counts[level - lev]
                    for lev in range(1, 1 + min(level, lr_lookbehind_levels)))

            nboxes_minimal = \
                    sum(minimal_upper_level_lengths) + minimal_new_level_length

            needs_renumbering = \
                    (curr_upper_level_lengths < minimal_upper_level_lengths).any()

            # {{{ prepare for reallocation/renumbering

            if needs_renumbering:
                assert knl_info.level_restrict

                # {{{ compute new level_start_box_nrs

                # Represents the amount of padding needed for upper levels.
                upper_level_padding = np.zeros(level, dtype=int)

                # Recompute the level padding.
                for ulevel in range(level):
                    upper_level_padding[ulevel] = sum(
                        2**(lev*dimensions) * new_level_leaf_counts[ulevel - lev]
                        for lev in range(
                            1, 1 + min(ulevel, lr_lookbehind_levels)))

                new_upper_level_unused_box_counts = np.max(
                    [upper_level_padding,
                    minimal_upper_level_lengths - new_level_used_box_counts[:-1]],
                    axis=0)

                new_level_start_box_nrs = np.empty(level + 1, dtype=int)
                new_level_start_box_nrs[0] = 0
                new_level_start_box_nrs[1:] = np.cumsum(
                    new_level_used_box_counts[:-1]
                    + new_upper_level_unused_box_counts)

                assert not (level_start_box_nrs == new_level_start_box_nrs).all()

                # }}}

                # {{{ set up reallocators

                old_box_count = level_start_box_nrs[-1]
                # Where should I put this box?
                dst_box_id = cl.array.empty(queue,
                        shape=old_box_count, dtype=box_id_dtype)

                for level_start, new_level_start, level_len in zip(
                        level_start_box_nrs, new_level_start_box_nrs,
                        curr_upper_level_lengths):
                    dst_box_id[level_start:level_start + level_len] = \
                            cl.array.arange(queue,
                                            new_level_start,
                                            new_level_start + level_len,
                                            dtype=box_id_dtype)

                wait_for.extend(dst_box_id.events)

                realloc_array = partial(self.gappy_copy_and_map,
                        dst_indices=dst_box_id, range=slice(old_box_count),
                        debug=debug)
                realloc_and_renumber_array = partial(self.gappy_copy_and_map,
                        dst_indices=dst_box_id, map_values=dst_box_id,
                        range=slice(old_box_count), debug=debug)
                renumber_array = partial(self.map_values_kernel, dst_box_id)

                # }}}

                # Update level_start_box_nrs. This will be the
                # level_start_box_nrs for the reallocated data.

                level_start_box_nrs = list(new_level_start_box_nrs)
                level_start_box_nrs_dev[:level + 1] = \
                    np.array(new_level_start_box_nrs, dtype=box_id_dtype)
                level_start_box_nrs_updated = True
                wait_for.extend(level_start_box_nrs_dev.events)

                nboxes_new = level_start_box_nrs[-1] + minimal_new_level_length

                del new_level_start_box_nrs
            else:
                from boxtree.tools import realloc_array
                realloc_and_renumber_array = realloc_array
                renumber_array = None
                level_start_box_nrs_updated = False
                nboxes_new = nboxes_minimal

            del nboxes_minimal

            # }}}

            # {{{ reallocate and/or renumber boxes if necessary

            if level_start_box_nrs_updated or nboxes_new > nboxes_guess:
                fin_debug("starting nboxes_guess increase")

                while nboxes_guess < nboxes_new:
                    nboxes_guess *= 2

                def my_realloc_nocopy(ary, shape=nboxes_guess):
                    return cl.array.empty(queue, allocator=allocator,
                            shape=shape, dtype=ary.dtype)

                def my_realloc_zeros_nocopy(ary, shape=nboxes_guess):
                    result = cl.array.zeros(queue, allocator=allocator,
                            shape=shape, dtype=ary.dtype)
                    return result, result.events[0]

                my_realloc = partial(realloc_array,
                        queue, allocator, nboxes_guess, wait_for=wait_for)
                my_realloc_zeros = partial(realloc_array,
                        queue, allocator, nboxes_guess, zero_fill=True,
                        wait_for=wait_for)
                my_realloc_zeros_and_renumber = partial(realloc_and_renumber_array,
                        queue, allocator, nboxes_guess, zero_fill=True,
                        wait_for=wait_for)

                resize_events = []

                split_box_ids = my_realloc_nocopy(split_box_ids)

                # *Most*, but not *all* of the values in this array are
                # rewritten when the morton scan is redone. Specifically,
                # only the box morton bin counts of boxes on the level
                # currently being processed are written-but we need to
                # retain the box morton bin counts from the higher levels.
                box_morton_bin_counts, evt = my_realloc_zeros(
                        box_morton_bin_counts)
                resize_events.append(evt)

                # force_split_box is unused unless level restriction is enabled.
                if knl_info.level_restrict:
                    force_split_box, evt = my_realloc_zeros(force_split_box)
                    resize_events.append(evt)

                box_srcntgt_starts, evt = my_realloc_zeros(box_srcntgt_starts)
                resize_events.append(evt)

                box_srcntgt_counts_cumul, evt = \
                        my_realloc_zeros(box_srcntgt_counts_cumul)
                resize_events.append(evt)

                box_has_children, evt = my_realloc_zeros(box_has_children)
                resize_events.append(evt)

                box_centers, evts = zip(
                    *(my_realloc(ary) for ary in box_centers))
                resize_events.extend(evts)

                box_child_ids, evts = zip(
                    *(my_realloc_zeros_and_renumber(ary)
                      for ary in box_child_ids))
                resize_events.extend(evts)

                box_parent_ids, evt = my_realloc_zeros_and_renumber(box_parent_ids)
                resize_events.append(evt)

                if not level_start_box_nrs_updated:
                    box_levels, evt = my_realloc(box_levels)
                    resize_events.append(evt)
                else:
                    box_levels, evt = my_realloc_zeros_nocopy(box_levels)
                    cl.wait_for_events([evt])
                    for box_level, (level_start, level_end) in enumerate(zip(
                            level_start_box_nrs, level_start_box_nrs[1:])):
                        box_levels[level_start:level_end].fill(box_level)
                    resize_events.extend(box_levels.events)

                if level_start_box_nrs_updated:
                    srcntgt_box_ids, evt = renumber_array(srcntgt_box_ids)
                    resize_events.append(evt)

                del my_realloc_zeros
                del my_realloc_nocopy
                del my_realloc_zeros_nocopy
                del renumber_array

                # Can't del on Py2.7 - these are used in generator expressions
                # above, which are nested scopes
                my_realloc = None
                my_realloc_zeros_and_renumber = None

                # retry
                logger.info("nboxes_guess exceeded: "
                            "enlarged allocations, restarting level")

                continue

            # }}}

            logger.debug("LEVEL %d -> %d boxes" % (level, nboxes_new))

            assert (
                level_start_box_nrs[-1] != nboxes_new
                or srcntgts_have_extent
                or final_level_restrict_iteration)

            if level_start_box_nrs[-1] == nboxes_new:
                # We haven't created new boxes in this level loop trip.
                #
                # If srcntgts have extent, this can happen if boxes were
                # in-principle overfull, but couldn't subdivide because of
                # extent restrictions.
                if srcntgts_have_extent and not final_level_restrict_iteration:
                    level -= 1
                    break
                assert final_level_restrict_iteration

            # {{{ update level_start_box_nrs, level_used_box_counts

            level_start_box_nrs.append(nboxes_new)
            level_start_box_nrs_dev[level + 1].fill(nboxes_new)
            wait_for.extend(level_start_box_nrs_dev.events)

            level_used_box_counts = new_level_used_box_counts
            level_used_box_counts_dev[:level + 1] = \
                    np.array(level_used_box_counts, dtype=box_id_dtype)
            wait_for.extend(level_used_box_counts_dev.events)

            level_leaf_counts = new_level_leaf_counts
            if debug:
                for level_start, level_nboxes, leaf_count in zip(
                        level_start_box_nrs,
                        level_used_box_counts,
                        level_leaf_counts):
                    if level_nboxes == 0:
                        assert leaf_count == 0
                        continue
                    nleaves_actual = level_nboxes - int(
                        cl.array.sum(box_has_children[
                            level_start:level_start + level_nboxes]).get())
                    assert leaf_count == nleaves_actual

            # Can't del in Py2.7 - see note below
            new_level_leaf_counts = None

            # }}}

            del nboxes_new
            del new_level_used_box_counts

            # {{{ split boxes

            box_splitter_args = (
                common_args
                + (box_has_children, force_split_box, root_extent)
                + box_child_ids
                + box_centers)

            evt = knl_info.box_splitter_kernel(*box_splitter_args,
                    range=slice(level_start_box_nrs[-1]),
                    wait_for=wait_for)

            wait_for = [evt]

            fin_debug("box splitter")

            # Mark the levels of boxes added for padding (these were not updated
            # by the box splitter kernel).
            last_used_box = level_start_box_nrs[-2] + level_used_box_counts[-1]
            box_levels[last_used_box:level_start_box_nrs[-1]].fill(level)

            wait_for.extend(box_levels.events)

            if debug:
                box_levels.finish()
                level_bl_chunk = box_levels.get()[
                        level_start_box_nrs[-2]:level_start_box_nrs[-1]]
                assert (level_bl_chunk == level).all()
                del level_bl_chunk

            if debug:
                assert (box_srcntgt_starts.get() < nsrcntgts).all()

            # }}}

            # {{{ renumber particles within split boxes

            new_user_srcntgt_ids = cl.array.empty_like(user_srcntgt_ids)
            new_srcntgt_box_ids = cl.array.empty_like(srcntgt_box_ids)

            particle_renumberer_args = (
                common_args
                + (box_has_children, force_split_box,
                   new_user_srcntgt_ids, new_srcntgt_box_ids))

            evt = knl_info.particle_renumberer_kernel(*particle_renumberer_args,
                    range=slice(nsrcntgts), wait_for=wait_for)

            wait_for = [evt]

            fin_debug("particle renumbering")

            user_srcntgt_ids = new_user_srcntgt_ids
            del new_user_srcntgt_ids
            srcntgt_box_ids = new_srcntgt_box_ids
            del new_srcntgt_box_ids

            # }}}

            # {{{ enforce level restriction on upper levels

            if final_level_restrict_iteration:
                # Roll back level update.
                #
                # FIXME: The extra iteration at the end to split boxes should
                # not be necessary. Instead, all the work for the final box
                # split should be done in the last iteration of the level
                # loop. Currently the main issue that forces the extra iteration
                # to be there is the need to use the box renumbering and
                # reallocation code. In order to fix this issue, the box
                # numbering and reallocation code needs to be accessible after
                # the final level restriction is done.
                assert int(have_oversize_split_box.get()) == 0
                assert level_used_box_counts[-1] == 0
                del level_used_box_counts[-1]
                del level_start_box_nrs[-1]
                level -= 1
                break

            if knl_info.level_restrict:
                # Avoid generating too many kernels.
                LEVEL_STEP = 10  # noqa
                if level % LEVEL_STEP == 1:
                    level_restrict_kernel = knl_info.level_restrict_kernel_builder(
                            LEVEL_STEP * div_ceil(level, LEVEL_STEP))

                # Upward pass - check if leaf boxes at higher levels need
                # further splitting.
                assert len(force_split_box) > 0
                force_split_box.fill(0)
                wait_for.extend(force_split_box.events)

                did_upper_level_split = False

                if debug:
                    boxes_split = []

                for upper_level, upper_level_start, upper_level_box_count in zip(
                        # We just built level. Our parent level doesn't need to
                        # be rechecked for splitting because the smallest boxes
                        # in the tree (ours) already have a 2-to-1 ratio with
                        # that. Start checking at the level above our parent.
                        range(level - 2, 0, -1),
                        # At this point, the last entry in level_start_box_nrs
                        # already refers to (level + 1).
                        level_start_box_nrs[-4::-1],
                        level_used_box_counts[-3::-1]):

                    upper_level_slice = slice(
                        upper_level_start, upper_level_start + upper_level_box_count)

                    have_upper_level_split_box.fill(0)
                    wait_for.extend(have_upper_level_split_box.events)

                    # writes: force_split_box, have_upper_level_split_box
                    evt = level_restrict_kernel(
                        upper_level,
                        root_extent,
                        box_has_children,
                        force_split_box,
                        have_upper_level_split_box,
                        *(box_child_ids + box_centers),
                        slice=upper_level_slice,
                        wait_for=wait_for)

                    wait_for = [evt]

                    if debug:
                        force_split_box.finish()
                        boxes_split.append(int(cl.array.sum(
                            force_split_box[upper_level_slice]).get()))

                    if int(have_upper_level_split_box.get()) == 0:
                        break

                    did_upper_level_split = True

                if debug:
                    total_boxes_split = sum(boxes_split)
                    logger.debug("level restriction: {total_boxes_split} boxes split"
                                 .format(total_boxes_split=total_boxes_split))
                    from itertools import count
                    for level_, nboxes_split in zip(
                            count(level - 2, step=-1), boxes_split[:-1]):
                        logger.debug("level {level}: {nboxes_split} boxes split"
                            .format(level=level_, nboxes_split=nboxes_split))
                    del boxes_split

                if int(have_oversize_split_box.get()) == 0 and did_upper_level_split:
                    # We are in the situation where there are boxes left to
                    # split on upper levels, and the level loop is done creating
                    # lower levels.
                    #
                    # We re-run the level loop one more time to finish creating
                    # the upper level boxes.
                    final_level_restrict_iteration = True
                    level += 1
                    continue

            # }}}

            if not int(have_oversize_split_box.get()):
                logger.debug("no boxes left to split")
                break

            level += 1
            have_oversize_split_box.fill(0)

            # {{{ check that nonchild part of box_morton_bin_counts is consistent

            if debug and 0:
                h_box_morton_bin_counts = box_morton_bin_counts.get()
                h_box_srcntgt_counts_cumul = box_srcntgt_counts_cumul.get()
                h_box_child_ids = tuple(bci.get() for bci in box_child_ids)

                has_mismatch = False
                for ibox in range(level_start_box_nrs[-1]):
                    is_leaf = all(bci[ibox] == 0 for bci in h_box_child_ids)
                    if is_leaf:
                        # nonchild count only found in box_info kernel
                        continue

                    if h_box_srcntgt_counts_cumul[ibox] == 0:
                        # empty boxes don't have box_morton_bin_counts written
                        continue

                    kid_sum = sum(
                            h_box_srcntgt_counts_cumul[bci[ibox]]
                            for bci in h_box_child_ids
                            if bci[ibox] != 0)

                    if (
                            h_box_srcntgt_counts_cumul[ibox]
                            != (h_box_morton_bin_counts[ibox]["nonchild_srcntgts"]
                                + kid_sum)):
                        print("MISMATCH", level, ibox)
                        has_mismatch = True

                assert not has_mismatch
                print("LEVEL %d OK" % level)

                # Cannot delete in Py 2.7: referred to from nested scope.
                h_box_srcntgt_counts_cumul = None

                del h_box_morton_bin_counts
                del h_box_child_ids

            # }}}

        nboxes = level_start_box_nrs[-1]

        npasses = level+1
        level_loop_proc.done("%d levels, %d boxes", level, nboxes)
        del npasses

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
                h_box_srcntgt_counts_nonchild = box_srcntgt_counts_nonchild.get()
                h_box_srcntgt_counts_cumul = box_srcntgt_counts_cumul.get()

                assert (h_box_srcntgt_counts_nonchild
                        <= h_box_srcntgt_counts_cumul[:nboxes]).all()

                del h_box_srcntgt_counts_nonchild

                # Cannot delete in Py 2.7: referred to from nested scope.
                h_box_srcntgt_counts_cumul = None

        # }}}

        del morton_nrs
        del box_morton_bin_counts

        # {{{ prune empty/unused leaf boxes

        prune_empty_leaves = not kwargs.get("skip_prune")

        if prune_empty_leaves:
            # What is the original index of this box?
            src_box_id = empty(nboxes, box_id_dtype)

            # Where should I put this box?
            #
            # Initialize to all zeros, because pruned boxes should be mapped to
            # zero (e.g. when pruning child_box_ids).
            dst_box_id, evt = zeros(nboxes, box_id_dtype)
            wait_for.append(evt)

            fin_debug("find prune indices")

            nboxes_post_prune_dev = empty((), dtype=box_id_dtype)
            evt = knl_info.find_prune_indices_kernel(
                    box_srcntgt_counts_cumul,
                    src_box_id, dst_box_id, nboxes_post_prune_dev,
                    size=nboxes, wait_for=wait_for)
            wait_for = [evt]
            nboxes_post_prune = int(nboxes_post_prune_dev.get())
            logger.debug("{} boxes after pruning "
                        "({} empty leaves and/or unused boxes removed)"
                    .format(nboxes_post_prune, nboxes - nboxes_post_prune))
            should_prune = True
        elif knl_info.level_restrict:
            # Remove unused boxes from the tree.
            src_box_id = empty(nboxes, box_id_dtype)
            dst_box_id = empty(nboxes, box_id_dtype)

            new_level_start_box_nrs = np.empty_like(level_start_box_nrs)
            new_level_start_box_nrs[0] = 0
            new_level_start_box_nrs[1:] = np.cumsum(level_used_box_counts)
            for level_start, new_level_start, level_used_box_count in zip(
                    level_start_box_nrs, new_level_start_box_nrs,
                    level_used_box_counts):
                def make_slice(start, offset=level_used_box_count):
                    return slice(start, start + offset)

                def make_arange(start, offset=level_used_box_count):
                    return cl.array.arange(
                            queue, start, start + offset, dtype=box_id_dtype)

                src_box_id[make_slice(new_level_start)] = make_arange(level_start)
                dst_box_id[make_slice(level_start)] = make_arange(new_level_start)
            wait_for.extend(src_box_id.events + dst_box_id.events)

            nboxes_post_prune = new_level_start_box_nrs[-1]

            logger.info("{} boxes after pruning ({} unused boxes removed)"
                    .format(nboxes_post_prune, nboxes - nboxes_post_prune))
            should_prune = True
        else:
            should_prune = False

        if should_prune:
            prune_events = []

            prune_empty = partial(self.gappy_copy_and_map,
                    queue, allocator, nboxes_post_prune,
                    src_indices=src_box_id,
                    range=slice(nboxes_post_prune), debug=debug)

            box_srcntgt_starts, evt = prune_empty(box_srcntgt_starts)
            prune_events.append(evt)

            box_srcntgt_counts_cumul, evt = prune_empty(box_srcntgt_counts_cumul)
            prune_events.append(evt)

            if debug and prune_empty_leaves:
                assert (box_srcntgt_counts_cumul.get() > 0).all()

            srcntgt_box_ids, evt = self.map_values_kernel(
                    dst_box_id, srcntgt_box_ids)
            prune_events.append(evt)

            box_parent_ids, evt = prune_empty(box_parent_ids, map_values=dst_box_id)
            prune_events.append(evt)

            box_levels, evt = prune_empty(box_levels)
            prune_events.append(evt)

            if srcntgts_have_extent:
                box_srcntgt_counts_nonchild, evt = prune_empty(
                        box_srcntgt_counts_nonchild)
                prune_events.append(evt)

            box_has_children, evt = prune_empty(box_has_children)
            prune_events.append(evt)

            box_child_ids, evts = zip(
                *(prune_empty(ary, map_values=dst_box_id)
                  for ary in box_child_ids))
            prune_events.extend(evts)

            box_centers, evts = zip(
                *(prune_empty(ary) for ary in box_centers))
            prune_events.extend(evts)

            # Update box counts and level start box indices.
            box_levels.finish()

            evt = knl_info.find_level_box_counts_kernel(
                box_levels, level_used_box_counts_dev)
            cl.wait_for_events([evt])

            nlevels = len(level_used_box_counts)
            level_used_box_counts = level_used_box_counts_dev[:nlevels].get()

            level_start_box_nrs = [0]
            level_start_box_nrs.extend(np.cumsum(level_used_box_counts))

            level_start_box_nrs_dev[:nlevels + 1] = np.array(
                level_start_box_nrs, dtype=box_id_dtype)
            prune_events.extend(level_start_box_nrs_dev.events)

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
                            == (box_source_counts_nonchild
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

        assert nlevels == len(level_used_box_counts)
        assert level + 1 == nlevels, (level+1, nlevels)
        if debug:
            max_level = np.max(box_levels.get())
            assert max_level + 1 == nlevels

        # {{{ gather box child ids, box centers

        # A number of arrays below are nominally 2-dimensional and stored with
        # the box index as the fastest-moving index. To make sure that accesses
        # remain aligned, we round up the number of boxes used for indexing.
        aligned_nboxes = div_ceil(nboxes_post_prune, 32)*32

        box_child_ids_new, evt = zeros((2**dimensions, aligned_nboxes), box_id_dtype)
        wait_for.append(evt)
        box_centers_new = empty((dimensions, aligned_nboxes), coord_dtype)

        for mnr, child_row in enumerate(box_child_ids):
            box_child_ids_new[mnr, :nboxes_post_prune] = \
                    child_row[:nboxes_post_prune]
        wait_for.extend(box_child_ids_new.events)

        for dim, center_row in enumerate(box_centers):
            box_centers_new[dim, :nboxes_post_prune] = center_row[:nboxes_post_prune]
        wait_for.extend(box_centers_new.events)

        cl.wait_for_events(wait_for)

        box_centers = box_centers_new
        box_child_ids = box_child_ids_new

        del box_centers_new
        del box_child_ids_new

        # }}}

        # {{{ compute box flags

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
                raise AssertionError

            try:
                box_target_counts_nonchild
            except NameError:
                pass
            else:
                raise AssertionError

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
                    box_parent_ids, box_srcntgt_counts_cumul,
                    box_source_counts_cumul, box_target_counts_cumul,
                    box_has_children, box_levels, nlevels,

                    # output if srcntgts_have_extent, input+output otherwise
                    box_source_counts_nonchild, box_target_counts_nonchild,

                    # output:
                    box_flags,
                ),
                range=slice(nboxes_post_prune),
                wait_for=wait_for)

        # }}}

        del box_has_children
        wait_for = [evt]

        # {{{ compute box bounding box

        fin_debug("finding box extents")

        box_source_bounding_box_min = cl.array.empty(
                queue, (dimensions, aligned_nboxes),
                dtype=coord_dtype)
        box_source_bounding_box_max = cl.array.empty(
                queue, (dimensions, aligned_nboxes),
                dtype=coord_dtype)

        if sources_are_targets:
            box_target_bounding_box_min = box_source_bounding_box_min
            box_target_bounding_box_max = box_source_bounding_box_max
        else:
            box_target_bounding_box_min = cl.array.empty(
                    queue, (dimensions, aligned_nboxes),
                    dtype=coord_dtype)
            box_target_bounding_box_max = cl.array.empty(
                    queue, (dimensions, aligned_nboxes),
                    dtype=coord_dtype)

        bogus_radii_array = cl.array.empty(queue, 1, dtype=coord_dtype)

        # nlevels-1 is the highest valid level index
        for level in range(nlevels-1, -1, -1):
            start, stop = level_start_box_nrs[level:level+2]

            for (skip, enable_radii, box_bounding_box_min, box_bounding_box_max,
                    pstarts, pcounts, particle_radii, particles) in [
                    (
                        # never skip
                        False,

                        sources_have_extent,
                        box_source_bounding_box_min,
                        box_source_bounding_box_max,
                        box_source_starts,
                        box_source_counts_nonchild,
                        source_radii if sources_have_extent else bogus_radii_array,
                        sources),
                    (
                        # skip the 'target' round if sources and targets
                        # are the same.
                        sources_are_targets,

                        targets_have_extent,
                        box_target_bounding_box_min,
                        box_target_bounding_box_max,
                        box_target_starts,
                        box_target_counts_nonchild,
                        target_radii if targets_have_extent else bogus_radii_array,
                        targets),
                    ]:

                if skip:
                    continue

                args = (
                        (
                            aligned_nboxes,
                            box_child_ids,
                            box_centers,
                            pstarts, pcounts,)
                        + tuple(particles)
                        + (
                            particle_radii,
                            enable_radii,

                            box_bounding_box_min,
                            box_bounding_box_max))

                evt = knl_info.box_extents_finder_kernel(
                        *args,

                        range=slice(start, stop),
                        queue=queue, wait_for=wait_for)

            wait_for = [evt]

        del bogus_radii_array

        # }}}

        # {{{ build output

        extra_tree_attrs = {}

        if sources_have_extent:
            extra_tree_attrs.update(source_radii=source_radii)
        if targets_have_extent:
            extra_tree_attrs.update(target_radii=target_radii)

        tree_build_proc.done(
                "%d levels, %d boxes, %d particles, box extent norm: %s, "
                "max_leaf_refine_weight: %d",
                nlevels, len(box_parent_ids), nsrcntgts, srcntgts_extent_norm,
                max_leaf_refine_weight)

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
                extent_norm=srcntgts_extent_norm,

                bounding_box=(bbox_min, bbox_max),
                level_start_box_nrs=level_start_box_nrs,
                level_start_box_nrs_dev=level_start_box_nrs_dev,

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

                box_source_bounding_box_min=box_source_bounding_box_min,
                box_source_bounding_box_max=box_source_bounding_box_max,
                box_target_bounding_box_min=box_target_bounding_box_min,
                box_target_bounding_box_max=box_target_bounding_box_max,

                _is_pruned=prune_empty_leaves,

                **extra_tree_attrs
                ).with_queue(None), evt

        # }}}

    # }}}

# vim: foldmethod=marker:filetype=pyopencl
