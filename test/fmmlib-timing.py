from __future__ import division, absolute_import, print_function

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

from six.moves import range

import numpy as np
import numpy.linalg as la
import pyopencl as cl

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from boxtree.tools import (  # noqa: F401
        make_normal_particle_array as p_normal,
        make_surface_particle_array as p_surface,
        make_uniform_particle_array as p_uniform,
        particle_array_to_host,
        ConstantOneExpansionWrangler)

import logging
logger = logging.getLogger(__name__)


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
            dipole_vec=dipole_vec,
            from_sep_siblings_rotation_classes=trav.from_sep_siblings_rotation_classes,
            from_sep_siblings_rotation_angles=trav.from_sep_siblings_rotation_class_to_angle)

    from boxtree.fmm import drive_fmm

    timing_data = {}
    pot = drive_fmm(trav, wrangler, weights, timing_data=timing_data)
    print("M2L timing", timing_data["multipole_to_local"]["wall_elapsed"])
    assert timing_data

    # {{{ ref fmmlib computation

    logger.info("computing direct (reference) result")

    import pyfmmlib
    fmmlib_routine = getattr(
            pyfmmlib,
            "%spot%s%ddall%s_vec" % (
                wrangler.eqn_letter,
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

    ref_pot = wrangler.finalize_potentials(
            fmmlib_routine(
                sources=sources_host.T, targets=targets_host.T,
                **kwargs)[0]
            )

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
                queue, targets, sources, [weights],
                out_host=True, **sumpy_extra_kwargs)

        sumpy_rel_err = (
                la.norm(pot - sumpy_ref_pot, np.inf)
                / la.norm(sumpy_ref_pot, np.inf))

        logger.info("relative l2 error vs sumpy direct: %g" % sumpy_rel_err)
        assert sumpy_rel_err < 1e-5, sumpy_rel_err

    # }}}

# }}}


# You can test individual routines by typing
# $ python test_fmm.py 'test_routine(cl.create_some_context)'

if __name__ == "__main__":
    test_pyfmmlib_fmm(cl._csc, 3, True, 0)

    """
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
    """

# vim: fdm=marker
