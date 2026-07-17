"""Nonlinear-in-θ PSFs must not run away on the exact core.

A badly-conditioned nonlinear PSF (a saturating tanh MLP, an alignment
gain → ∞) can fool the Gauss–Newton EIV/phi merit ‖f‖ = ‖ψᵀPr‖: it drives
the merit to zero by collapsing its *sensitivities* ψ → 0 rather than by
fitting the data — a spurious root (observed: max|F| ~ 5e3, |θ| ~ 1e18).

The fix keeps ``inner="auto"`` on Gauss–Newton for nonlinear PSFs (so
well-conditioned ones keep the EIV instrument — see ``test_parametric_
exact_psf``), but ``gn_minimize`` detects the sensitivity collapse via the
symmetric Gram ‖½(G+Gᵀ)‖ crashing by orders of magnitude, returns the last
*healthy* iterate, and flags ``diverged`` so ``solve_force_od`` falls back
to the frozen-precision L-BFGS (the true-NLL descent).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import OverdampedLangevinInference
from SFI.langevin import OverdampedProcess
from SFI.statefunc import make_sf
from SFI.bases import X
from SFI.inference.parametric_core.solve import _is_linear_basis


def _double_well_data(seed=1):
    def U(x):
        return 0.25 * (x[0] ** 2 - 1.0) ** 2 + 0.5 * x[1] ** 2

    F = make_sf(jax.grad(lambda x: -U(x)), dim=2, rank=1)
    proc = OverdampedProcess(F, D=0.5 * jnp.eye(2))
    proc.initialize(jnp.zeros(2))
    return proc.simulate(dt=0.02, Nsteps=8000, key=random.PRNGKey(seed), prerun=100)


def _mlp_potential(H=8):
    mlp_U = (
        X(dim=2).rank_to_features()
        .dense(H, weight="W1", bias="b1").elementwisemap(jnp.tanh)
        .dense(H, weight="W2", bias="b2").elementwisemap(jnp.tanh)
        .dense(1, weight="W3", bias=None)
    )
    return -(mlp_U.d_x()), H


def _init(mlp, H, seed=123):
    theta0, key = {}, random.PRNGKey(seed)
    for name, shape in [("W1", (2, H)), ("b1", (H,)),
                        ("W2", (H, H)), ("b2", (H,)), ("W3", (H, 1))]:
        key, sub = random.split(key)
        theta0[name] = (jnp.sqrt(2.0 / sum(shape)) * random.normal(sub, shape)
                        if name.startswith("W") else jnp.zeros(shape))
    return theta0


def test_nonlinear_psf_is_detected():
    mlp, H = _mlp_potential()
    assert not _is_linear_basis(mlp)


def test_auto_falls_back_to_lbfgs_on_collapse_and_stays_sane():
    coll = _double_well_data()
    mlp, H = _mlp_potential()
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(mlp, _init(mlp, H), inner="auto", inner_maxiter=40, max_outer=2)

    info = inf.metadata["force_parametric_info"]
    # auto starts on GN (nonlinear PSF), detects the collapse, falls back
    assert info["inner"] == "lbfgs", info
    theta = np.asarray(inf.force_coefficients_full)
    assert np.isfinite(theta).all()
    assert np.abs(theta).max() < 1e3, np.abs(theta).max()   # no |θ|→∞ runaway

    pts = jnp.asarray(np.random.default_rng(5).normal(size=(64, 2)) * 0.8)
    Fmag = float(np.abs(np.asarray(inf.force_inferred(pts))).max())
    assert Fmag < 1e2, Fmag                        # not the ~5e3 garbage field


def test_explicit_gn_keeps_last_healthy_iterate_not_garbage():
    # Force the unsafe path explicitly; the collapse guard must keep the
    # returned parameters finite and bounded — the last healthy iterate,
    # never the |θ|~1e18 spurious root — and surface diverged in info.
    coll = _double_well_data()
    mlp, H = _mlp_potential()
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(mlp, _init(mlp, H), inner="gn", inner_maxiter=40, max_outer=3)
    theta = np.asarray(inf.force_coefficients_full)
    assert np.isfinite(theta).all()
    assert np.abs(theta).max() < 1e3, np.abs(theta).max()   # healthy iterate
    assert inf.metadata["force_parametric_info"].get("diverged") is True


def test_linear_basis_still_uses_gn():
    coll = _double_well_data()
    B = X(dim=2)                                   # linear-in-θ
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(B, inner="auto")
    assert inf.metadata["force_parametric_info"]["inner"] == "gn"
    c = np.asarray(inf.force_coefficients_full)
    assert np.isfinite(c).all()
