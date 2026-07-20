"""Solver-entry guards and numerical policy (F5, F8, F12).

* F5 — the parametric solvers run in float64 internally regardless of the
  session dtype (the block Cholesky factors and the Gram accumulation over
  ~1e5-1e6 residuals are unreliable in float32; the historical behavior was
  to inherit the session dtype silently).
* F8 — non-uniform dt (an absolute time vector or per-step dt) must
  solve end-to-end with per-interval flow steps.
* F12 — a rank-deficient Gram (collinear basis) must yield finite
  covariance via a stable pseudo-inverse, not inf/NaN from a raw inverse.

NOTE: this module deliberately does NOT use the x64 fixture — F5 is about
what happens in a float32 session.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _ou(T=2500, dt=0.02, d=1, seed=2, noise=0.0):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, d))
    X[0] = 0.4
    for t in range(T - 1):
        X[t + 1] = X[t] - X[t] * dt + np.sqrt(2 * 0.4 * dt) * rng.normal(size=d)
    if noise:
        X = X + noise * rng.standard_normal(X.shape)
    return X.astype(np.float64)


def _coll_from(X, dt, t=None):
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    kw = {"dt": float(dt)} if t is None else {"t": jnp.asarray(t)}
    ds = TrajectoryDataset.from_arrays(X=jnp.asarray(X[:, None, :]), **kw)
    return TrajectoryCollection.from_dataset(ds)


def _basis(d=1, order=1):
    from SFI.bases import monomials_up_to

    return monomials_up_to(order, dim=d, rank="vector")


# ── F5: float64 inside the solve, whatever the session says ─────────────


def test_solve_computes_in_f64_from_f32_session():
    """Compute in float64, report in the session dtype.

    In a float32 session the solve must (a) return float32 arrays so the
    downstream world stays dtype-consistent, and (b) produce the SAME
    numbers (to f32 rounding) as an x64-session solve on the identical
    f32-rounded data — proving the arithmetic ran in float64.
    """
    from SFI.inference.parametric_core.solve import solve_force_od

    prev = jax.config.jax_enable_x64
    X = _ou(seed=21)
    try:
        jax.config.update("jax_enable_x64", False)
        res32 = solve_force_od(_coll_from(np.float32(X), 0.02), _basis(),
                               max_outer=3)
        assert np.asarray(res32.theta).dtype == np.float32, (
            "f32 session must get f32 results back")
        assert np.asarray(res32.theta_cov).dtype == np.float32

        jax.config.update("jax_enable_x64", True)
        res64 = solve_force_od(_coll_from(np.float32(X).astype(np.float64), 0.02),
                               _basis(), max_outer=3)
    finally:
        jax.config.update("jax_enable_x64", prev)

    # identical inputs (f32-rounded data), identical f64 arithmetic,
    # then one rounding to f32 on output
    np.testing.assert_allclose(np.asarray(res32.theta, dtype=np.float64),
                               np.asarray(res64.theta), rtol=2e-6)
    np.testing.assert_allclose(np.asarray(res32.D, dtype=np.float64),
                               np.asarray(res64.D), rtol=2e-6)


def test_session_dtype_restored_after_solve():
    from SFI.inference.parametric_core.solve import solve_force_od

    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        solve_force_od(_coll_from(np.float32(_ou(T=800)), 0.02), _basis(),
                       max_outer=1, profile_maxiter=1)
        assert jax.config.jax_enable_x64 is False, "solver leaked x64 config"
        assert jnp.zeros(1).dtype == jnp.float32, "session dtype changed"
    finally:
        jax.config.update("jax_enable_x64", prev)


# ── F8: non-uniform dt solves end-to-end ─────────────────────────────────


@pytest.mark.parametrize("solver", ["od", "ud"])
def test_non_uniform_dt_exact_solves(solver):
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        X = _ou(T=600)
        t = np.cumsum(0.02 + 0.005 * (np.arange(600) % 2))  # alternating dt
        coll = _coll_from(X, None, t=t)
        if solver == "od":
            from SFI.inference.parametric_core.solve import solve_force_od as sv

            F = _basis()
        else:
            from SFI.inference.parametric_core.solve import solve_force_ud as sv
            from SFI.statefunc.factory import make_psf
            from SFI.statefunc.params import ParamSpec

            def f(x, *, v, params, mask=None, extras=None):
                return -params["k"] * x - params["gamma"] * v

            F = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                         params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                                 ParamSpec("gamma", shape=(), dtype=jnp.float64)])
        res = sv(coll, F, max_outer=2)
        assert bool(jnp.all(jnp.isfinite(res.theta)))
    finally:
        jax.config.update("jax_enable_x64", prev)


# ── F12: rank-deficient Gram → finite covariance ─────────────────────────


def test_collinear_basis_finite_covariance():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from SFI.inference.parametric_core.solve import solve_force_od

        X = _ou(T=2500, seed=5)
        coll = _coll_from(X, 0.02)
        collinear = _basis(order=1) + _basis(order=1)  # duplicated features
        res = solve_force_od(coll, collinear, max_outer=3)
        assert np.all(np.isfinite(np.asarray(res.theta))), "theta not finite"
        assert np.all(np.isfinite(np.asarray(res.theta_cov))), (
            "theta_cov must be finite for a singular Gram (stable pinv)")
    finally:
        jax.config.update("jax_enable_x64", prev)
