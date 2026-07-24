"""Per-interval (non-uniform) dt in the parametric-core kernels.

Stage-1 scope: the shared kernels (``flow_multi``, ``precompute``,
``covariance``) accept a per-interval dt vector alongside the historical
scalar, with

* closed-form anchors — OD OU per-interval ``J = e^{−k·dt_k}`` and
  Lyapunov ``Q = D(1−e^{−2k·dt_k})/k``; UD free-particle α-propagators
  ``(α₋, α₀) = (dt₂/dt₁, −(1+dt₂/dt₁))·I`` and exact process blocks
  ``A = (2/3)D·dt₂²(dt₁+dt₂)``, ``C = (1/3)D·dt₂²·dt₃`` (which reduce to
  the classical ``(4/3, 1/3)Δt³D`` at uniform spacing);
* a dense-AD reference — for a *linear* underdamped model the frozen
  Jacobian blocks are state-independent, so the α's are the exact
  derivatives of the shooting map and must match ``jax.jacobian`` of a
  hand-composed two-dt replica;
* scalar ↔ constant-vector equivalence at 1e-12 on every touched
  function (the uniform limit is the constant array).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.inference.parametric_core.covariance import (
    build_od_blocks,
    build_ud_blocks,
    build_ud_blocks_exact,
)
from SFI.inference.parametric_core.precompute import (
    od_point_tensors,
    ud_point_tensors,
)


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _ou_psf():
    from SFI.bases import monomials_up_to

    return monomials_up_to(1, dim=1, rank="vector").to_psf()


def _ud_lin_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    return make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def _alternating_dt(n, lo=0.014, hi=0.026):
    return jnp.where(jnp.arange(n) % 2 == 0, lo, hi)


# ── OD closed forms per interval ─────────────────────────────────────────


def test_od_point_tensors_alternating_dt_closed_forms():
    # k, D exactly representable in float32: the PSF template is created at
    # the session dtype, so θ is rounded through it — non-representable
    # values would shift the closed form by ~4e-8·dt.
    k, D = 1.25, 0.5
    F = _ou_psf()
    theta = jnp.asarray([0.0, -k])                 # F(x) = −k x
    X = jnp.asarray(np.linspace(-1, 1, 9))[:, None, None]
    dts = _alternating_dt(8)
    pt = od_point_tensors(F, theta, X, None, dts, 8, "rk4",
                          with_psi=False, D_lyap=jnp.asarray([[D]]))
    J = np.asarray(pt["J"]).reshape(8)
    Q = np.asarray(pt["Q"]).reshape(8)
    dts = np.asarray(dts)
    np.testing.assert_allclose(J, np.exp(-k * dts), rtol=1e-9)
    np.testing.assert_allclose(Q, D * (1 - np.exp(-2 * k * dts)) / k,
                               rtol=1e-8)
    # the two interval lengths must give two distinct J values
    assert abs(J[0] - J[1]) > 1e-3


# ── UD two-dt α-propagators ──────────────────────────────────────────────


def test_ud_free_particle_alphas_closed_form_uneven():
    """Free particle: v̂ = ΔY/dt₁ exactly, so r = Y₊ − Y₀ − (dt₂/dt₁)(Y₀ − Y₋)
    and the propagators are α₋ = (dt₂/dt₁)·I, α₀ = −(1 + dt₂/dt₁)·I."""
    from SFI.inference.parametric_core.flow_multi import (
        ud_multi_step_residuals_with_psi,
    )

    psf = _ud_lin_psf()
    theta0 = jnp.asarray([0.0, 0.0])
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.normal(0, 0.5, size=(6, 1, 1)))
    dts = jnp.asarray([0.03, 0.07, 0.05, 0.04, 0.06])
    r, ap, a0, am, vhat, psi = ud_multi_step_residuals_with_psi(
        psf, theta0, Y, None, dts, 4, "rk4")[:6]
    for n in range(4):
        rho = float(dts[n + 1] / dts[n])
        np.testing.assert_allclose(float(ap[n, 0, 0, 0]), 1.0, rtol=1e-12)
        np.testing.assert_allclose(float(a0[n, 0, 0, 0]), -(1.0 + rho),
                                   rtol=1e-12)
        np.testing.assert_allclose(float(am[n, 0, 0, 0]), rho, rtol=1e-12)
        # and the residual itself is the 3-point free-flight misfit
        r_ref = float(Y[n + 2, 0, 0] - Y[n + 1, 0, 0]
                      - rho * (Y[n + 1, 0, 0] - Y[n, 0, 0]))
        np.testing.assert_allclose(float(r[n, 0, 0]), r_ref, rtol=1e-12)


def test_ud_two_dt_alphas_match_dense_jacobian():
    """Linear damped model: the frozen Jacobian blocks are state-independent,
    so the α's are the *exact* derivatives of the two-dt shooting map — they
    must match dense autodiff of a hand-composed replica."""
    from SFI.inference.parametric_core.flow_multi import (
        ud_multi_step_residuals_with_psi,
    )
    from SFI.inference.parametric_core.jacobians import (
        rk4_composed_jacobian_phase,
    )

    psf = _ud_lin_psf()
    theta = jnp.asarray([1.1, 0.7])                # k = −1.1?  sign via psf
    struct = psf.unflatten_params(theta)
    dFdx, dFdv = psf.d_x(), psf.d_v()
    n_sub = 4

    def replica(yp, yc, yn, dt1, dt2):             # frames (1, d)
        V0 = (yc - yp) / dt1
        Phi_x0, Phi_v0, _, Jxv, _, Jvv = rk4_composed_jacobian_phase(
            psf, dFdx, dFdv, yp, V0, struct, None, None, dt1, n_sub, "rk4")
        dV = jnp.linalg.solve(Jxv, (yc - Phi_x0)[..., None]).squeeze(-1)
        v_hat = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv, dV)
        Phi_x_out = rk4_composed_jacobian_phase(
            psf, dFdx, dFdv, yc, v_hat, struct, None, None, dt2, n_sub,
            "rk4")[0]
        return yn - Phi_x_out

    rng = np.random.default_rng(1)
    Y = jnp.asarray(rng.normal(0, 0.4, size=(5, 1, 1)))
    dts = jnp.asarray([0.03, 0.07, 0.045, 0.06])
    r, ap, a0, am = ud_multi_step_residuals_with_psi(
        psf, theta, Y, None, dts, n_sub, "rk4")[:4]

    for n in range(3):
        args = (Y[n], Y[n + 1], Y[n + 2], dts[n], dts[n + 1])
        r_ref = replica(*args)
        np.testing.assert_allclose(np.asarray(r[n]), np.asarray(r_ref),
                                   rtol=1e-12, atol=1e-14)
        for slot, alpha in ((0, am), (1, a0), (2, ap)):
            jac = jax.jacobian(replica, argnums=slot)(*args)
            # (1, d, 1, d) → (d, d)
            jac = np.asarray(jac)[0, :, 0, :]
            np.testing.assert_allclose(np.asarray(alpha[n, 0]), jac,
                                       rtol=1e-9, atol=1e-12)


# ── UD exact process blocks, uneven closed form ──────────────────────────


def test_ud_exact_blocks_free_particle_uneven_closed_form():
    """Free particle, per-interval Lyapunov tensors: the exact blocks obey
    A_n = (2/3)·D·dt₂²(dt₁+dt₂) and C_n = (1/3)·D·dt₂²·dt₃ — reducing to the
    classical (4/3, 1/3)Δt³D at uniform spacing.  RK4 integrates the
    polynomial free-particle Lyapunov solution exactly."""
    psf = _ud_lin_psf()
    theta0 = jnp.asarray([0.0, 0.0])
    D = 0.7
    rng = np.random.default_rng(2)
    Y = jnp.asarray(rng.normal(0, 0.5, size=(7, 1, 1)))
    dts = jnp.asarray([0.03, 0.07, 0.05, 0.04, 0.06, 0.055])
    pt = ud_point_tensors(psf, theta0, Y, None, dts, 4, "rk4",
                          with_psi=False, D_lyap=jnp.asarray([[D]]))
    ex = build_ud_blocks_exact(pt["ap"][:, 0], pt["a0"][:, 0], pt["am"][:, 0],
                               {k: v[:, 0] for k, v in pt["qing"].items()},
                               jnp.zeros((1, 1)), jitter=0.0)
    dts = np.asarray(dts)
    n_res = 5
    A = np.asarray(ex.A).reshape(n_res)
    C = np.asarray(ex.offdiag[0]).reshape(n_res - 1)
    E = np.asarray(ex.offdiag[1]).reshape(n_res - 2)
    for n in range(n_res):
        dt1, dt2 = dts[n], dts[n + 1]
        np.testing.assert_allclose(
            A[n], (2.0 / 3.0) * D * dt2**2 * (dt1 + dt2), rtol=1e-11)
    for n in range(n_res - 1):
        dt2, dt3 = dts[n + 1], dts[n + 2]
        np.testing.assert_allclose(
            C[n], (1.0 / 3.0) * D * dt2**2 * dt3, rtol=1e-11)
    np.testing.assert_allclose(E, 0.0, atol=1e-18)   # Λ = 0 → no lag-2


# ── scalar ↔ constant-vector equivalence ─────────────────────────────────


def test_od_point_tensors_scalar_vs_constant_vector():
    k, D, dt = 0.9, 0.4, 0.02
    F = _ou_psf()
    theta = jnp.asarray([0.1, -k])
    rng = np.random.default_rng(3)
    X = jnp.asarray(rng.normal(0, 1, size=(10, 3, 1)))
    kw = dict(with_psi=True, with_instrument=True,
              D_lyap=jnp.asarray([[D]]), conv_lambda=jnp.asarray([[0.01]]))
    a = od_point_tensors(F, theta, X, None, dt, 2, "rk4", **kw)
    b = od_point_tensors(F, theta, X, None, jnp.full(9, dt), 2, "rk4", **kw)
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]),
                                   rtol=1e-12, atol=1e-15, err_msg=key)


def test_ud_point_tensors_scalar_vs_constant_vector():
    dt = 0.05
    psf = _ud_lin_psf()
    theta = jnp.asarray([1.0, 0.8])
    rng = np.random.default_rng(4)
    Y = jnp.asarray(rng.normal(0, 0.5, size=(10, 2, 1)))
    kw = dict(with_psi=True, with_instrument=True, D_lyap=jnp.asarray([[0.7]]))
    a = ud_point_tensors(psf, theta, Y, None, dt, 2, "rk4", **kw)
    b = ud_point_tensors(psf, theta, Y, None, jnp.full(9, dt), 2, "rk4", **kw)
    assert set(a) == set(b)
    for key in a:
        if key == "qing":
            for qk in a[key]:
                np.testing.assert_allclose(
                    np.asarray(a[key][qk]), np.asarray(b[key][qk]),
                    rtol=1e-12, atol=1e-15, err_msg=f"qing/{qk}")
        else:
            np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]),
                                       rtol=1e-12, atol=1e-15, err_msg=key)


def test_build_blocks_scalar_vs_constant_vector():
    rng = np.random.default_rng(5)
    d, n_res, dt = 2, 7, 0.03
    J = jnp.asarray(rng.normal(0, 0.1, size=(n_res, d, d))
                    + np.eye(d)[None])
    D = jnp.asarray([[0.5, 0.1], [0.1, 0.4]])
    Lam = jnp.asarray([[1e-3, 0.0], [0.0, 2e-3]])
    a = build_od_blocks(J, D, Lam, dt, jitter=1e-9)
    b = build_od_blocks(J, D, Lam, jnp.full(n_res, dt), jitter=1e-9)
    np.testing.assert_allclose(np.asarray(a.A), np.asarray(b.A), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(a.offdiag[0]),
                               np.asarray(b.offdiag[0]), rtol=1e-14)

    ap = jnp.broadcast_to(jnp.eye(d), (n_res, d, d))
    a0 = jnp.asarray(rng.normal(-2, 0.1, size=(n_res, d, d)))
    am = jnp.asarray(rng.normal(1, 0.1, size=(n_res, d, d)))
    ua = build_ud_blocks(ap, a0, am, D, Lam, dt, jitter=1e-9)
    ub = build_ud_blocks(ap, a0, am, D, Lam, jnp.full(n_res, dt), jitter=1e-9)
    np.testing.assert_allclose(np.asarray(ua.A), np.asarray(ub.A), rtol=1e-14)
    for lag in (0, 1):
        np.testing.assert_allclose(np.asarray(ua.offdiag[lag]),
                                   np.asarray(ub.offdiag[lag]), rtol=1e-14)
