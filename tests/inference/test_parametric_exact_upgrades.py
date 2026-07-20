"""Stage-3 upgrades of the exact core: Lyapunov Q, convexity, Huber.

* Lyapunov-exact process covariance: matches the closed form
  ``Q = D (1 − e^{−2kΔt})/k`` on the OU flow at integrator accuracy,
  where the endpoint trapezoid errs at O((kΔt)²); the UD lifted blocks
  reproduce the classical (4/3, 1/3)·Δt³ D structure at leading order.
* Convexity correction: removes the ``½∇²Φ:Λ`` residual-mean bias on a
  deliberately curved drift.
* Huber score: equals the plain score on tame data; bounds the influence
  of gross outliers.
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


def _lin_psf(k=1.3, d=1):
    from SFI.bases import monomials_up_to

    return monomials_up_to(1, dim=d, rank="vector").to_psf()


# ── Lyapunov Q ───────────────────────────────────────────────────────────


def test_od_lyapunov_q_matches_ou_closed_form():
    from SFI.inference.parametric_core.precompute import od_point_tensors

    k, D, dt = 1.3, 0.4, 0.3
    F = _lin_psf()
    theta = jnp.asarray([0.0, -k])                 # F(x) = −k x
    X = jnp.asarray(np.linspace(-1, 1, 7))[:, None, None]
    pt = od_point_tensors(F, theta, X, None, dt, 8, "rk4",
                          with_psi=False, D_lyap=jnp.asarray([[D]]))
    Q_exact = 2 * D * (1 - np.exp(-2 * k * dt)) / (2 * k)
    np.testing.assert_allclose(np.asarray(pt["Q"]).ravel(),
                               Q_exact, rtol=1e-5)
    # and the trapezoid is measurably wrong at this kΔt
    Q_trap = dt * (np.exp(-2 * k * dt) * D + D)
    assert abs(Q_trap - Q_exact) / Q_exact > 0.03


def test_ud_lyapunov_blocks_recover_leading_order():
    """For free inertial motion the exact-linearized UD blocks must equal
    the classical (4/3, 1/3)·Δt³ D at machine precision."""
    from SFI.inference.parametric_core.covariance import (
        build_ud_blocks,
        build_ud_blocks_exact,
    )
    from SFI.inference.parametric_core.precompute import ud_point_tensors
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    psf = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                   params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                           ParamSpec("gamma", shape=(), dtype=jnp.float64)])
    D, dt = 0.7, 0.05
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.normal(0, 0.5, size=(9, 1, 1)))

    # free motion: k = γ = 0 → exact blocks must equal 4/3, 1/3 exactly
    th0 = jnp.asarray([0.0, 0.0])
    pt = ud_point_tensors(psf, th0, Y, None, dt, 4, "rk4",
                          with_psi=False, D_lyap=jnp.asarray([[D]]))
    ex = build_ud_blocks_exact(pt["ap"][:, 0], pt["a0"][:, 0], pt["am"][:, 0],
                               {k: v[:, 0] for k, v in pt["qing"].items()},
                               jnp.zeros((1, 1)), jitter=0.0)
    np.testing.assert_allclose(np.asarray(ex.A).ravel(),
                               (4 / 3) * dt**3 * D, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(ex.offdiag[0]).ravel(),
                               (1 / 3) * dt**3 * D, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(ex.offdiag[1]).ravel(), 0.0,
                               atol=1e-18)

    # damped case: exact blocks deviate from leading order at O(γΔt)
    th = jnp.asarray([1.0, 0.8])
    pt2 = ud_point_tensors(psf, -th, Y, None, dt, 4, "rk4",
                           with_psi=False, D_lyap=jnp.asarray([[D]]))
    ex2 = build_ud_blocks_exact(pt2["ap"][:, 0], pt2["a0"][:, 0],
                                pt2["am"][:, 0],
                                {k: v[:, 0] for k, v in pt2["qing"].items()},
                                jnp.zeros((1, 1)), jitter=0.0)
    lead = build_ud_blocks(pt2["ap"][:, 0], pt2["a0"][:, 0], pt2["am"][:, 0],
                           D, jnp.zeros((1, 1)), dt, jitter=0.0)
    rel = abs(float(ex2.A[0, 0, 0]) - float(lead.A[0, 0, 0])) / float(lead.A[0, 0, 0])
    assert 1e-4 < rel < 0.2, f"expected O(γΔt) deviation, got {rel:.2e}"


# ── convexity correction ─────────────────────────────────────────────────


def test_convexity_correction_removes_noise_curvature_bias():
    """Drift F(x) = −x³ has strong curvature; with noise Λ on the flow's
    argument, E[y' − Φ(y)] = −½Φ''Λ + O(Λ²).  The corrected residual mean
    must be an order of magnitude closer to zero."""
    from SFI.inference.parametric_core.precompute import od_point_tensors
    from SFI.bases import monomials_up_to

    F = monomials_up_to(3, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.0, 0.0, 0.0, -1.0])     # F = −x³
    dt, lam = 0.05, 0.1**2
    rng = np.random.default_rng(1)

    # noiseless latent points x, noisy observations y = x + η; the latent
    # step is deterministic (D = 0) so the residual isolates the bias.
    x = np.full(160000, 1.5)
    eta = rng.normal(0, np.sqrt(lam), size=x.shape)
    y = x + eta
    from SFI.integrate.rk4 import ode_flow

    def phi(z):
        return ode_flow(lambda u: -u**3, jnp.atleast_1d(z), dt, 4)[0]

    x_next = np.asarray(jax.vmap(phi)(jnp.asarray(x)))
    eta_next = rng.normal(0, np.sqrt(lam), size=x.shape)
    # batch the independent pairs along the particle axis: block (2, M, 1)
    Y = jnp.asarray(np.stack([y, x_next + eta_next], axis=0)[:, :, None])

    r_p = od_point_tensors(F, theta, Y, None, dt, 4, "rk4",
                           with_psi=False)["r"]
    r_c = od_point_tensors(F, theta, Y, None, dt, 4, "rk4", with_psi=False,
                           conv_lambda=jnp.asarray([[lam]]))["r"]
    bias_plain = abs(float(jnp.mean(r_p)))
    bias_corr = abs(float(jnp.mean(r_c)))
    # bias = ½|Φ''|λ = ½·0.271·0.01 ≈ 1.35e-3 (Φ'' of the exact cubic-decay
    # flow at x=1.5, dt=0.05); statistical error of the mean ≈ 3e-4
    assert bias_plain > 1.0e-3, f"toy not in the biased regime: {bias_plain:.2e}"
    assert bias_corr < 0.45 * bias_plain, (
        f"correction ineffective: {bias_plain:.3e} → {bias_corr:.3e}")


# ── Huber score ──────────────────────────────────────────────────────────


def test_huber_matches_plain_on_tame_data_and_bounds_outliers():
    from SFI.inference.parametric_core.runner import make_exact_runs_od
    from SFI.inference.parametric_core.gram import unpack_gram
    from SFI.bases import monomials_up_to
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    rng = np.random.default_rng(3)
    T, dt, k, D = 3000, 0.02, 1.0, 0.4
    X = np.zeros((T, 1))
    for t in range(T - 1):
        X[t + 1] = X[t] - k * X[t] * dt + np.sqrt(2 * D * dt) * rng.normal()

    def coll_of(Xa):
        ds = TrajectoryDataset.from_arrays(X=jnp.asarray(Xa[:, None, :]), dt=dt)
        return TrajectoryCollection.from_dataset(ds)

    basis = monomials_up_to(1, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.0, -k])
    Dm, Lm = jnp.asarray([[D]]), jnp.asarray([[1e-8]])

    kw = dict(dt=dt, n_substeps=1, integrator="rk4", w=0.0,
              lyapunov=False, convexity=False)
    G0, f0, _, _ = unpack_gram(
        make_exact_runs_od(coll_of(X), basis, **kw).gram(theta, Dm, Lm), 2)
    Gh, fh, _, _ = unpack_gram(
        make_exact_runs_od(coll_of(X), basis, huber_c=4.0, **kw)
        .gram(theta, Dm, Lm), 2)
    # tame data: essentially no clipping at c = 4
    np.testing.assert_allclose(np.asarray(fh), np.asarray(f0), rtol=1e-2,
                               atol=1e-3 * float(jnp.linalg.norm(f0)))

    # inject gross outliers: 1% of points displaced by 30 std
    Xo = X.copy()
    idx = rng.choice(T, size=T // 100, replace=False)
    Xo[idx] += 30 * np.sqrt(2 * D * dt)
    fo_plain = unpack_gram(
        make_exact_runs_od(coll_of(Xo), basis, **kw).gram(theta, Dm, Lm), 2)[1]
    fo_hub = unpack_gram(
        make_exact_runs_od(coll_of(Xo), basis, huber_c=1.345, **kw)
        .gram(theta, Dm, Lm), 2)[1]
    # the Huberized score moves far less under contamination
    d_plain = float(jnp.linalg.norm(fo_plain - f0))
    d_hub = float(jnp.linalg.norm(fo_hub - f0))
    assert d_hub < 0.35 * d_plain, (d_plain, d_hub)
