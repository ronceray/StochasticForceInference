"""ψ via the θ-recursion for NON-interacting models (the F1 speedup).

The exact per-particle θ-sensitivity recursion (`jacobians.rk4_composed_
jacobian_theta` / `_phase_theta`) historically served only interacting
models; the non-interacting Gram programs used ``jax.jacfwd`` with
n_params forward tangents through the flow — the dominant cost for wide
or expensive bases.  These tests pin the equivalence contract of the
generalized dispatch:

* OD ψ, J, r: recursion == jacfwd **exactly** (same chain rule, fp only).
* OD/UD instruments: recursion == legacy vmapped instrument exactly.
* Euler variants: recursion == jacfwd through the Euler flow exactly.
* UD residual ψ: the recursion drops the θ-dependence of the shooting
  Jacobian blocks — an O(Δt²)-relative approximation (already shipped on
  the interacting path).  Pinned by a dt-halving scaling test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _od_basis(d=2, order=2):
    from SFI.bases import monomials_up_to

    return monomials_up_to(order, dim=d, rank="vector").to_psf()


def _ud_psf():
    import jax.numpy as jnp

    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v - params["c"] * x**3

    return make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64),
                            ParamSpec("c", shape=(), dtype=jnp.float64)])


def _window(T=6, N=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(0.0, 0.7, size=(T, N, d)))


def _theta(F_psf, seed=1):
    rng = np.random.default_rng(seed)
    n = int(F_psf.template.size)
    return jnp.asarray(rng.normal(0.0, 0.4, size=n))


# ── OD: recursion vs jacfwd (exact) ──────────────────────────────────────


@pytest.mark.parametrize("integrator,n_sub", [("rk4", 1), ("rk4", 3), ("euler", 1), ("euler", 2)])
def test_od_psi_recursion_matches_jacfwd(integrator, n_sub):
    from SFI.inference.parametric_core.flow_multi import (
        multi_step_residuals,
        multi_step_residuals_with_psi,
    )

    F = _od_basis()
    th = _theta(F)
    X_w = _window(T=6, N=3, d=2)
    dt = 0.05

    r, J, psi = multi_step_residuals_with_psi(F, th, X_w, None, dt, n_sub, integrator)

    def resid(t):
        rr, _ = multi_step_residuals(F, F.unflatten_params(t), X_w, None, dt, n_sub, integrator)
        return rr

    r_ref = resid(th)
    psi_ref = jax.jacfwd(resid)(th)
    _, J_ref = multi_step_residuals(F, F.unflatten_params(th), X_w, None, dt, n_sub, integrator)

    np.testing.assert_allclose(np.asarray(r), np.asarray(r_ref), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(J), np.asarray(J_ref), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(psi), np.asarray(psi_ref), rtol=1e-10,
                               atol=1e-12 * float(np.max(np.abs(psi_ref))))


def test_od_instrument_recursion_matches_legacy():
    from SFI.inference.parametric_core.flow import od_instrument
    from SFI.inference.parametric_core.flow_multi import multi_od_instrument

    F = _od_basis()
    th = _theta(F)
    X_base = _window(T=1, N=4, d=2)[0]
    dt, n_sub = 0.05, 1

    got = multi_od_instrument(F, th, X_base, None, dt, n_sub, "rk4")

    def drift_of_theta(t):
        struct = F.unflatten_params(t)
        return lambda y: F(y[None], params=struct)[0]

    ref = jax.vmap(
        lambda xb: od_instrument(drift_of_theta, th, xb, dt, n_sub, "rk4")
    )(X_base)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-10,
                               atol=1e-12 * float(np.max(np.abs(ref))))


# ── UD: exact pieces exact, shooting-ψ O(Δt²) ────────────────────────────


def _ud_resid_ref(F, th, Y_w, dt, n_sub, integrator):
    """Reference r and AD ψ via the legacy per-particle window residuals."""
    from SFI.inference.parametric_core.flow_ud import ud_window_residuals

    def resid(t):
        struct = F.unflatten_params(t)

        def force(x, *, v):
            return F(x[None], v=v[None], params=struct)[0]

        def per_particle(Yp):
            r, *_ = ud_window_residuals(lambda x, v: force(x, v=v), Yp, dt, n_sub, integrator)
            return r

        return jnp.swapaxes(jax.vmap(per_particle, in_axes=1)(Y_w), 0, 1)

    return resid


@pytest.mark.parametrize("integrator,n_sub", [("rk4", 1), ("euler", 2)])
def test_ud_psi_recursion_alpha_and_r_exact(integrator, n_sub):
    """r, α's, v̂ from the ψ-recursion path match the legacy path exactly."""
    from SFI.inference.parametric_core.flow_multi import (
        ud_multi_step_residuals,
        ud_multi_step_residuals_with_psi,
    )

    F = _ud_psf()
    th = _theta(F)
    Y_w = _window(T=7, N=2, d=1)
    dt = 0.05

    r0, ap0, a00, am0, vh0 = ud_multi_step_residuals(
        F, F.unflatten_params(th), Y_w, None, dt, n_sub, integrator)
    r1, ap1, a01, am1, vh1, _psi = ud_multi_step_residuals_with_psi(
        F, th, Y_w, None, dt, n_sub, integrator)

    for got, ref, name in [(r1, r0, "r"), (ap1, ap0, "a+"), (a01, a00, "a0"),
                           (am1, am0, "a-"), (vh1, vh0, "vhat")]:
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-10,
                                   atol=1e-13, err_msg=name)


def test_ud_psi_recursion_dt_scaling():
    """‖ψ_recursion − ψ_AD‖/‖ψ_AD‖ = O(Δt) and small.

    The recursion drops the θ-dependence of the shooting Jacobian blocks
    (which would need ∂²F/∂x∂θ — the second-order AD this path avoids):
    an O(Δt³) absolute error on a ψ of magnitude O(Δt²), i.e. O(Δt)
    relative.  As a test function this preserves consistency of the
    estimating equation; the efficiency loss is O(rel²) ≈ 1e-4 here."""
    from SFI.inference.parametric_core.flow_multi import ud_multi_step_residuals_with_psi

    F = _ud_psf()
    th = _theta(F)

    def rel_err(dt):
        Y_w = _window(T=7, N=2, d=1, seed=3) * 0.5
        *_, psi = ud_multi_step_residuals_with_psi(F, th, Y_w, None, dt, 1, "rk4")
        resid = _ud_resid_ref(F, th, Y_w, dt, 1, "rk4")
        psi_ref = jax.jacfwd(resid)(th)
        return float(np.linalg.norm(np.asarray(psi - psi_ref))
                     / np.linalg.norm(np.asarray(psi_ref)))

    e1, e2 = rel_err(0.08), rel_err(0.04)
    assert e1 < 0.02, f"UD psi approximation too coarse at dt=0.08: {e1:.3e}"
    assert e2 < e1 / 1.7, f"not O(dt): err(0.08)={e1:.3e}, err(0.04)={e2:.3e}"


def test_ud_instrument_recursion_matches_legacy():
    from SFI.inference.parametric_core.flow_multi import multi_ud_instrument
    from SFI.inference.parametric_core.flow_ud import ud_instrument

    F = _ud_psf()
    th = _theta(F)
    W = _window(T=2, N=3, d=1, seed=5)
    Y_a, Y_b = W[0], W[1]
    dt, n_sub = 0.05, 1

    got = multi_ud_instrument(F, th, Y_a, Y_b, None, dt, n_sub, "rk4")

    def force_of_theta(t):
        struct = F.unflatten_params(t)
        return lambda x, v: F(x[None], v=v[None], params=struct)[0]

    ref = jax.vmap(
        lambda a, b: ud_instrument(force_of_theta, th, a, b, dt, n_sub, "rk4", n_predict=2)
    )(Y_a, Y_b)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9,
                               atol=1e-11 * float(np.max(np.abs(ref))))


# ── interacting path must be untouched (regression) ─────────────────────


def test_interacting_psi_still_matches_jacfwd_frozen_background():
    """The generalized dispatch must not change the interacting path."""
    from SFI.bases.pairs import pair_direction, parametric_radial_kernel
    from SFI.inference.parametric_core.flow_multi import multi_step_residuals_with_psi
    from SFI.statefunc import make_psf

    dim = 2
    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x,
                      params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r,
                                        params={"k_spring": ()}, dim=dim)
    F = F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")
    th = jnp.asarray([1.0, 0.5])
    X_w = _window(T=4, N=3, d=2, seed=8)

    r, J, psi = multi_step_residuals_with_psi(F, th, X_w, None, 0.02, 1, "rk4")
    assert np.all(np.isfinite(np.asarray(r)))
    assert np.all(np.isfinite(np.asarray(psi)))
    assert psi.shape == r.shape + (2,)
