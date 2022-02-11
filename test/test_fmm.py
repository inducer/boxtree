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
import numpy.linalg as la
import pyopencl as cl
import warnings

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from boxtree.pyfmmlib_integration import FMMLibRotationDataNotSuppliedWarning
from boxtree.tools import (  # noqa: F401
        make_normal_particle_array as p_normal,
        make_surface_particle_array as p_surface,
        make_uniform_particle_array as p_uniform,
        particle_array_to_host,
        ConstantOneExpansionWrangler)

import logging
logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", FMMLibRotationDataNotSuppliedWarning)


# {{{ ref fmmlib pot computation

def get_fmmlib_ref_pot(wrangler, weights, sources_host, targets_host,
        helmholtz_k, dipole_vec=None):
    dims = sources_host.shape[0]
    eqn_letter = "h" if helmholtz_k else "l"
    use_dipoles = dipole_vec is not None

    import pyfmmlib
    fmmlib_routine = getattr(
            pyfmmlib,
            "%spot%s%ddall%s_vec" % (
                eqn_letter,
                "fld" if dims == 3 else "grad",
                dims,
                "_dp" if use_dipoles else ""))

    kwargs = {}
    if dims == 3:
        kwargs["iffld"] = False
    else:
        kwargs["ifgrad"] = False
        kwargs["ifhess"] = False

    if use_dipoles:
        if helmholtz_k == 0 and dims == 2:
            kwargs["dipstr"] = (
                    -weights  # pylint:disable=invalid-unary-operand-type
                    * (dipole_vec[0] + 1j * dipole_vec[1]))
        else:
            kwargs["dipstr"] = weights
            kwargs["dipvec"] = dipole_vec
    else:
        kwargs["charge"] = weights
    if helmholtz_k:
        kwargs["zk"] = helmholtz_k

    return wrangler.finalize_potentials(
            fmmlib_routine(
                sources=sources_host, targets=targets_host,
                **kwargs)[0]
            )

# }}}


# {{{ fmm interaction completeness test

class ConstantOneExpansionWranglerWithFilteredTargetsInTreeOrder(
        ConstantOneExpansionWrangler):
    def __init__(self, tree, filtered_targets):
        ConstantOneExpansionWrangler.__init__(self, tree)
        self.filtered_targets = filtered_targets

    def output_zeros(self):
        return np.zeros(self.filtered_targets.nfiltered_targets, dtype=np.float64)

    def _get_target_slice(self, ibox):
        pstart = self.filtered_targets.box_target_starts[ibox]
        return slice(
                pstart, pstart
                + self.filtered_targets.box_target_counts_nonchild[ibox])

    def reorder_potentials(self, potentials):
        tree_order_all_potentials = np.zeros(self.tree.ntargets, potentials.dtype)
        tree_order_all_potentials[
                self.filtered_targets.unfiltered_from_filtered_target_indices] \
                = potentials

        return tree_order_all_potentials[self.tree.sorted_target_ids]


class ConstantOneExpansionWranglerWithFilteredTargetsInUserOrder(
        ConstantOneExpansionWrangler):
    def __init__(self, tree, filtered_targets):
        ConstantOneExpansionWrangler.__init__(self, tree)
        self.filtered_targets = filtered_targets

    def _get_target_slice(self, ibox):
        user_target_ids = self.filtered_targets.target_lists[
                self.filtered_targets.target_starts[ibox]:
                self.filtered_targets.target_starts[ibox+1]]
        return self.tree.sorted_target_ids[user_target_ids]


@pytest.mark.parametrize("well_sep_is_n_away", [1, 2])
@pytest.mark.parametrize(("dims", "nsources_req", "ntargets_req",
        "who_has_extent", "source_gen", "target_gen", "filter_kind",
        "extent_norm", "from_sep_smaller_crit"),
        [
            (2, 10**5, None, "", p_normal, p_normal, None, "linf", "static_linf"),
            (2, 5 * 10**4, 4*10**4, "", p_normal, p_normal, None, "linf", "static_linf"),  # noqa: E501
            (2, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "linf", "static_linf"),  # noqa: E501

            (3, 10**5, None, "", p_normal, p_normal, None, "linf", "static_linf"),
            (3, 5 * 10**5, 4*10**4, "", p_normal, p_normal, None, "linf", "static_linf"),  # noqa: E501
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "linf", "static_linf"),  # noqa: E501

            (2, 10**5, None, "", p_normal, p_normal, "user", "linf", "static_linf"),
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, "user", "linf", "static_linf"),  # noqa: E501
            (2, 10**5, None, "", p_normal, p_normal, "tree", "linf", "static_linf"),
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, "tree", "linf", "static_linf"),  # noqa: E501

            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "linf", "static_linf"),  # noqa: E501
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "linf", "precise_linf"),  # noqa: E501
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "l2", "precise_linf"),  # noqa: E501
            (3, 5 * 10**5, 4*10**4, "t", p_normal, p_normal, None, "l2", "static_l2"),  # noqa: E501

            ])
def test_fmm_completeness(ctx_factory, dims, nsources_req, ntargets_req,
         who_has_extent, source_gen, target_gen, filter_kind, well_sep_is_n_away,
         extent_norm, from_sep_smaller_crit):
    """Tests whether the built FMM traversal structures and driver completely
    capture all interactions.
    """

    sources_have_extent = "s" in who_has_extent
    targets_have_extent = "t" in who_has_extent

    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dtype = np.float64

    try:
        sources = source_gen(queue, nsources_req, dims, dtype, seed=15)
        nsources = len(sources[0])

        if ntargets_req is None:
            # This says "same as sources" to the tree builder.
            targets = None
            ntargets = ntargets_req
        else:
            targets = target_gen(queue, ntargets_req, dims, dtype, seed=16)
            ntargets = len(targets[0])
    except ImportError:
        pytest.skip("loo.py not available, but needed for particle array "
                "generation")

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=12)
    if sources_have_extent:
        source_radii = 2**rng.uniform(queue, nsources, dtype=dtype,
                a=-10, b=0)
    else:
        source_radii = None

    if targets_have_extent:
        target_radii = 2**rng.uniform(queue, ntargets, dtype=dtype,
                a=-10, b=0)
    else:
        target_radii = None

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30,
            source_radii=source_radii, target_radii=target_radii,
            debug=True, stick_out_factor=0.25, extent_norm=extent_norm)
    if 0:
        tree.get(queue).plot()
        import matplotlib.pyplot as pt
        pt.show()

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx,
            well_sep_is_n_away=well_sep_is_n_away,
            from_sep_smaller_crit=from_sep_smaller_crit)
    trav, _ = tbuild(queue, tree, debug=True)

    if who_has_extent:
        pre_merge_trav = trav
        trav = trav.merge_close_lists(queue)

    #weights = np.random.randn(nsources)
    weights = np.ones(nsources)
    weights_sum = np.sum(weights)

    host_trav = trav.get(queue=queue)
    host_tree = host_trav.tree

    if who_has_extent:
        pre_merge_host_trav = pre_merge_trav.get(queue=queue)

    from boxtree.tree import ParticleListFilter
    plfilt = ParticleListFilter(ctx)

    if filter_kind:
        flags = rng.uniform(queue, ntargets or nsources, np.int32, a=0, b=2) \
                .astype(np.int8)
        if filter_kind == "user":
            filtered_targets = plfilt.filter_target_lists_in_user_order(
                    queue, tree, flags)
            wrangler = ConstantOneExpansionWranglerWithFilteredTargetsInUserOrder(
                    host_tree, filtered_targets.get(queue=queue))
        elif filter_kind == "tree":
            filtered_targets = plfilt.filter_target_lists_in_tree_order(
                    queue, tree, flags)
            wrangler = ConstantOneExpansionWranglerWithFilteredTargetsInTreeOrder(
                    host_tree, filtered_targets.get(queue=queue))
        else:
            raise ValueError("unsupported value of 'filter_kind'")
    else:
        wrangler = ConstantOneExpansionWrangler(host_tree)
        flags = cl.array.empty(queue, ntargets or nsources, dtype=np.int8)
        flags.fill(1)

    if ntargets is None and not filter_kind:
        # This check only works for targets == sources.
        assert (wrangler.reorder_potentials(
                wrangler.reorder_sources(weights)) == weights).all()

    from boxtree.fmm import drive_fmm
    pot = drive_fmm(host_trav, wrangler, (weights,))

    if filter_kind:
        pot = pot[flags.get() > 0]

    rel_err = la.norm((pot - weights_sum) / nsources)
    good = rel_err < 1e-8

    # {{{ build, evaluate matrix (and identify incorrect interactions)

    if 0 and not good:
        mat = np.zeros((ntargets, nsources), dtype)
        from pytools import ProgressBar

        logging.getLogger().setLevel(logging.WARNING)

        pb = ProgressBar("matrix", nsources)
        for i in range(nsources):
            unit_vec = np.zeros(nsources, dtype=dtype)
            unit_vec[i] = 1
            mat[:, i] = drive_fmm(host_trav, wrangler, (unit_vec,))
            pb.progress()
        pb.finished()

        logging.getLogger().setLevel(logging.INFO)

        import matplotlib.pyplot as pt

        if 0:
            pt.imshow(mat)
            pt.colorbar()
            pt.show()

        incorrect_tgts, incorrect_srcs = np.where(mat != 1)

        if 1 and len(incorrect_tgts):
            from boxtree.visualization import TreePlotter
            plotter = TreePlotter(host_tree)
            plotter.draw_tree(fill=False, edgecolor="black")
            plotter.draw_box_numbers()
            plotter.set_bounding_box()

            tree_order_incorrect_tgts = \
                    host_tree.indices_to_tree_target_order(incorrect_tgts)
            tree_order_incorrect_srcs = \
                    host_tree.indices_to_tree_source_order(incorrect_srcs)

            src_boxes = [
                    host_tree.find_box_nr_for_source(i)
                    for i in tree_order_incorrect_srcs]
            tgt_boxes = [
                    host_tree.find_box_nr_for_target(i)
                    for i in tree_order_incorrect_tgts]
            print(src_boxes)
            print(tgt_boxes)

            # plot all sources/targets
            if 0:
                pt.plot(
                        host_tree.targets[0],
                        host_tree.targets[1],
                        "v", alpha=0.9)
                pt.plot(
                        host_tree.sources[0],
                        host_tree.sources[1],
                        "gx", alpha=0.9)

            # plot offending sources/targets
            if 0:
                pt.plot(
                        host_tree.targets[0][tree_order_incorrect_tgts],
                        host_tree.targets[1][tree_order_incorrect_tgts],
                        "rv")
                pt.plot(
                        host_tree.sources[0][tree_order_incorrect_srcs],
                        host_tree.sources[1][tree_order_incorrect_srcs],
                        "go")
            pt.gca().set_aspect("equal")

            from boxtree.visualization import draw_box_lists
            draw_box_lists(
                    plotter,
                    pre_merge_host_trav if who_has_extent else host_trav,
                    22)
            # from boxtree.visualization import draw_same_level_non_well_sep_boxes
            # draw_same_level_non_well_sep_boxes(plotter, host_trav, 2)

            pt.show()

    # }}}

    if 0 and not good:
        import matplotlib.pyplot as pt
        pt.plot(pot-weights_sum)
        pt.show()

    if 0 and not good:
        import matplotlib.pyplot as pt
        filt_targets = [
                host_tree.targets[0][flags.get() > 0],
                host_tree.targets[1][flags.get() > 0],
                ]
        host_tree.plot()
        bad = np.abs(pot - weights_sum) >= 1e-3
        bad_targets = [
                filt_targets[0][bad],
                filt_targets[1][bad],
                ]
        print(bad_targets[0].shape)
        pt.plot(filt_targets[0], filt_targets[1], "x")
        pt.plot(bad_targets[0], bad_targets[1], "v")
        pt.show()

    assert good

# }}}


# {{{ test fmmlib integration

@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("use_dipoles", [True, False])
@pytest.mark.parametrize("helmholtz_k", [0, 2])
def test_pyfmmlib_fmm(ctx_factory, dims, use_dipoles, helmholtz_k):
    logging.basicConfig(level=logging.INFO)

    from pytest import importorskip
    importorskip("pyfmmlib")

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 3000
    ntargets = 1000
    dtype = np.float64

    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = (
            p_normal(queue, ntargets, dims, dtype, seed=18)
            + np.array([2, 0, 0])[:dims])

    sources_host = particle_array_to_host(sources)
    targets_host = particle_array_to_host(targets)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    trav = trav.get(queue=queue)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=20)

    weights = rng.uniform(queue, nsources, dtype=np.float64).get()
    #weights = np.ones(nsources)

    if use_dipoles:
        np.random.seed(13)
        dipole_vec = np.random.randn(dims, nsources)
    else:
        dipole_vec = None

    if dims == 2 and helmholtz_k == 0:
        base_nterms = 20
    else:
        base_nterms = 10

    def fmm_level_to_nterms(tree, lev):
        result = base_nterms

        if lev < 3 and helmholtz_k:
            # exercise order-varies-by-level capability
            result += 5

        if use_dipoles:
            result += 1

        return result

    from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
    wrangler = FMMLibExpansionWrangler(
            trav.tree, helmholtz_k,
            fmm_level_to_nterms=fmm_level_to_nterms,
            dipole_vec=dipole_vec)

    from boxtree.fmm import drive_fmm

    timing_data = {}
    pot = drive_fmm(trav, wrangler, (weights,), timing_data=timing_data)
    print(timing_data)
    assert timing_data

    # {{{ ref fmmlib computation

    logger.info("computing direct (reference) result")

    ref_pot = get_fmmlib_ref_pot(wrangler, weights, sources_host.T,
            targets_host.T, helmholtz_k, dipole_vec)

    rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)
    logger.info("relative l2 error vs fmmlib direct: %g" % rel_err)
    assert rel_err < 1e-5, rel_err

    # }}}

    # {{{ check against sumpy

    try:
        import sumpy  # noqa
    except ImportError:
        have_sumpy = False
        from warnings import warn
        warn("sumpy unavailable: cannot compute independent reference "
                "values for pyfmmlib")
    else:
        have_sumpy = True

    if have_sumpy:
        from sumpy.kernel import (  # pylint:disable=import-error
                LaplaceKernel, HelmholtzKernel, DirectionalSourceDerivative)
        from sumpy.p2p import P2P  # pylint:disable=import-error

        sumpy_extra_kwargs = {}
        if helmholtz_k:
            knl = HelmholtzKernel(dims)
            sumpy_extra_kwargs["k"] = helmholtz_k
        else:
            knl = LaplaceKernel(dims)

        if use_dipoles:
            knl = DirectionalSourceDerivative(knl)
            sumpy_extra_kwargs["src_derivative_dir"] = dipole_vec

        p2p = P2P(ctx,
                [knl],
                exclude_self=False)

        evt, (sumpy_ref_pot,) = p2p(
                queue, targets, sources, (weights,),
                out_host=True, **sumpy_extra_kwargs)

        sumpy_rel_err = (
                la.norm(pot - sumpy_ref_pot, np.inf)
                / la.norm(sumpy_ref_pot, np.inf))

        logger.info("relative l2 error vs sumpy direct: %g" % sumpy_rel_err)
        assert sumpy_rel_err < 1e-5, sumpy_rel_err

    # }}}

# }}}


# {{{ test fmmlib numerical stability

@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("helmholtz_k", [0, 2])
@pytest.mark.parametrize("order", [35])
def test_pyfmmlib_numerical_stability(ctx_factory, dims, helmholtz_k, order):
    logging.basicConfig(level=logging.INFO)

    from pytest import importorskip
    importorskip("pyfmmlib")

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 30
    dtype = np.float64

    # The input particles are arranged with geometrically increasing/decreasing
    # spacing along a line, to build a deep tree that stress-tests the
    # translations.
    particle_line = np.array([2**-i for i in range(nsources//2)], dtype=dtype)
    particle_line = np.hstack([particle_line, 3 - particle_line])
    zero = np.zeros(nsources, dtype=dtype)

    sources = np.vstack([
            particle_line,
            zero,
            zero])[:dims]

    targets = sources * (1 + 1e-3)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=2, debug=True)

    assert tree.nlevels >= 15

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    trav = trav.get(queue=queue)
    weights = np.ones_like(sources[0])

    from boxtree.pyfmmlib_integration import (
            FMMLibExpansionWrangler, FMMLibRotationData)

    def fmm_level_to_nterms(tree, lev):
        return order

    wrangler = FMMLibExpansionWrangler(
            trav.tree, helmholtz_k,
            fmm_level_to_nterms=fmm_level_to_nterms,
            rotation_data=FMMLibRotationData(queue, trav))

    from boxtree.fmm import drive_fmm

    pot = drive_fmm(trav, wrangler, (weights,))
    assert not np.isnan(pot).any()

    # {{{ ref fmmlib computation

    logger.info("computing direct (reference) result")

    ref_pot = get_fmmlib_ref_pot(wrangler, weights, sources, targets,
            helmholtz_k)

    rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)
    logger.info("relative l2 error vs fmmlib direct: %g" % rel_err)

    if dims == 2:
        error_bound = (1/2) ** (1 + order)
    else:
        error_bound = (3/4) ** (1 + order)

    assert rel_err < error_bound, rel_err

    # }}}

# }}}


# {{{ test particle count thresholding in traversal generation

@pytest.mark.parametrize("enable_extents", [True, False])
def test_interaction_list_particle_count_thresholding(ctx_factory, enable_extents):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    logging.basicConfig(level=logging.INFO)

    dims = 2
    nsources = 1000
    ntargets = 1000
    dtype = np.float

    max_particles_in_box = 30
    # Ensure that we have underfilled boxes.
    from_sep_smaller_min_nsources_cumul = 1 + max_particles_in_box

    from boxtree.fmm import drive_fmm
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = p_normal(queue, ntargets, dims, dtype, seed=15)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=12)

    if enable_extents:
        target_radii = 2**rng.uniform(queue, ntargets, dtype=dtype, a=-10, b=0)
    else:
        target_radii = None

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=max_particles_in_box,
            target_radii=target_radii,
            debug=True, stick_out_factor=0.25)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True,
            _from_sep_smaller_min_nsources_cumul=from_sep_smaller_min_nsources_cumul)

    weights = np.ones(nsources)
    weights_sum = np.sum(weights)

    host_trav = trav.get(queue=queue)
    host_tree = host_trav.tree

    wrangler = ConstantOneExpansionWrangler(host_tree)

    pot = drive_fmm(host_trav, wrangler, (weights,))

    assert (pot == weights_sum).all()

# }}}


# {{{ test fmm with float32 dtype

@pytest.mark.parametrize("enable_extents", [True, False])
def test_fmm_float32(ctx_factory, enable_extents):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from pyopencl.characterize import has_struct_arg_count_bug
    if has_struct_arg_count_bug(queue.device):
        pytest.xfail("won't work on devices with the struct arg count issue")

    logging.basicConfig(level=logging.INFO)

    dims = 2
    nsources = 1000
    ntargets = 1000
    dtype = np.float32

    from boxtree.fmm import drive_fmm
    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = p_normal(queue, ntargets, dims, dtype, seed=15)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=12)

    if enable_extents:
        target_radii = 2**rng.uniform(queue, ntargets, dtype=dtype, a=-10, b=0)
    else:
        target_radii = None

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30,
            target_radii=target_radii,
            debug=True, stick_out_factor=0.25)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    weights = np.ones(nsources)
    weights_sum = np.sum(weights)

    host_trav = trav.get(queue=queue)
    host_tree = host_trav.tree

    wrangler = ConstantOneExpansionWrangler(host_tree)

    pot = drive_fmm(host_trav, wrangler, (weights,))

    assert (pot == weights_sum).all()

# }}}


# {{{ test with fmm optimized 3d m2l

@pytest.mark.parametrize("well_sep_is_n_away", (1, 2))
@pytest.mark.parametrize("helmholtz_k", (0, 2))
@pytest.mark.parametrize("nsrcntgts", (20, 10000))
def test_fmm_with_optimized_3d_m2l(ctx_factory, nsrcntgts, helmholtz_k,
                                   well_sep_is_n_away):
    logging.basicConfig(level=logging.INFO)

    from pytest import importorskip
    importorskip("pyfmmlib")

    dims = 3

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = ntargets = nsrcntgts // 2
    dtype = np.float64

    sources = p_normal(queue, nsources, dims, dtype, seed=15)
    targets = (
            p_normal(queue, ntargets, dims, dtype, seed=18)
            + np.array([2, 0, 0])[:dims])

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    trav = trav.get(queue=queue)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(queue.context, seed=20)

    weights = rng.uniform(queue, nsources, dtype=np.float64).get()

    base_nterms = 10

    def fmm_level_to_nterms(tree, lev):
        result = base_nterms

        if lev < 3 and helmholtz_k:
            # exercise order-varies-by-level capability
            result += 5

        return result

    from boxtree.pyfmmlib_integration import (
            FMMLibExpansionWrangler, FMMLibRotationData)

    baseline_wrangler = FMMLibExpansionWrangler(
            trav.tree, helmholtz_k,
            fmm_level_to_nterms=fmm_level_to_nterms)

    optimized_wrangler = FMMLibExpansionWrangler(
            trav.tree, helmholtz_k,
            fmm_level_to_nterms=fmm_level_to_nterms,
            rotation_data=FMMLibRotationData(queue, trav))

    from boxtree.fmm import drive_fmm

    baseline_timing_data = {}
    baseline_pot = drive_fmm(
            trav, baseline_wrangler, (weights,), timing_data=baseline_timing_data)

    optimized_timing_data = {}
    optimized_pot = drive_fmm(
            trav, optimized_wrangler, (weights,), timing_data=optimized_timing_data)

    baseline_time = baseline_timing_data["multipole_to_local"]["process_elapsed"]
    if baseline_time is not None:
        print("Baseline M2L time : %#.4g s" % baseline_time)

    opt_time = optimized_timing_data["multipole_to_local"]["process_elapsed"]
    if opt_time is not None:
        print("Optimized M2L time: %#.4g s" % opt_time)

    assert np.allclose(baseline_pot, optimized_pot, atol=1e-13, rtol=1e-13)

# }}}


# You can test individual routines by typing
# $ python test_fmm.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
