"""Tests for the Tikhonov condition-cap guard in the parametric core."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_condition_cap_ridge_zero_when_well_conditioned():
    """A well-conditioned Gram needs no damping → λ = 0."""
    from SFI.inference.parametric_core.driver import condition_cap_ridge

    G = jnp.diag(jnp.array([1.0, 0.5, 0.25]))  # cond = 4
    lam = float(condition_cap_ridge(G, gram_cond_max=1e10))
    assert lam == 0.0


def test_condition_cap_ridge_caps_condition_number():
    """An ill-conditioned Gram is damped so cond(G+λI) ≤ gram_cond_max."""
    from SFI.inference.parametric_core.driver import condition_cap_ridge

    s_max, s_min = 1.0, 1e-14          # cond = 1e14
    G = jnp.diag(jnp.array([s_max, 1e-3, s_min]))
    cond_max = 1e8
    lam = float(condition_cap_ridge(G, gram_cond_max=cond_max))

    assert lam > 0.0
    # λ = σ_max / cond_max  → regularised condition number capped
    np.testing.assert_allclose(lam, s_max / cond_max, rtol=1e-10)
    reg_cond = (s_max + lam) / (s_min + lam)
    assert reg_cond <= cond_max * (1 + 1e-6)


def test_condition_cap_ridge_handles_nonfinite():
    """Non-finite / zero Gram returns a finite non-negative λ (no NaN)."""
    from SFI.inference.parametric_core.driver import condition_cap_ridge

    lam = float(condition_cap_ridge(jnp.zeros((3, 3)), gram_cond_max=1e10))
    assert np.isfinite(lam) and lam >= 0.0


def _make_ou_1d(T=3000, dt=0.02, seed=3):
    rng = np.random.default_rng(seed)
    k, D = 1.0, 0.5
    s = np.sqrt(2 * D)
    X = np.zeros((T, 1))
    X[0] = rng.normal(0, 0.5, size=1)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k * X[t]) * dt + s * rng.normal(size=1) * np.sqrt(dt)
    return X, k


def _degenerate_psf():
    """Two perfectly collinear features → rank-deficient Gram (cond = ∞)."""
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        c = params["c"]                 # (2,)
        return (c[0] + c[1]) * x        # both params act through the same direction

    return make_psf(f, dim=1, rank=1, n_features=1,
                    params=[ParamSpec("c", shape=(2,), dtype=jnp.float64)])


def test_force_core_tikhonov_bounds_rank_deficient_gram():
    """A rank-deficient Gram activates the condition-cap ridge (λ>0) and the
    estimate stays finite and bounded, while the identifiable combination
    (c0+c1 = −k) is still recovered."""
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.inference.parametric_core.solve import solve_force_od

    X, k = _make_ou_1d(T=3000, dt=0.02, seed=3)
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X[:, None, :]), dt=0.02)
    coll = TrajectoryCollection.from_dataset(ds)

    res = solve_force_od(coll, _degenerate_psf(), n_substeps=4, max_outer=3)

    assert np.all(np.isfinite(np.asarray(res.theta))), "θ not finite"
    assert np.linalg.norm(np.asarray(res.theta)) < 50.0, "θ not bounded by ridge"
    assert res.info["ridge_lambda"] > 0.0, "condition-cap ridge did not activate"
    c0, c1 = np.asarray(res.theta)
    assert abs((c0 + c1) - (-k)) < 0.25 * k, f"identifiable c0+c1={c0+c1:.3f} ≠ -k"


def test_gn_line_search_bar_uses_merit_objective():
    """The acceptance bar must come from the same objective as the trial merit.

    When ``nll_fn`` carries a constant offset against the Gram program's own
    nll (different window/conditioning normalisation — the case for the
    windowed conditional NLL), comparing trial-merit against gram-nll can
    reject every step and freeze θ at the init while reporting rel=0
    "convergence"."""
    import jax.numpy as jnp
    import numpy as np
    from SFI.inference.parametric_core.driver import gn_minimize

    theta_star = jnp.array([1.0, -2.0])
    OFFSET = 1e4   # >> the total NLL improvement (= ½‖θ*‖² = 2.5)

    def gram_fn(th, D, Se):
        G = jnp.eye(2)
        f = th - theta_star
        nll = 0.5 * jnp.dot(f, f)
        return G, f, nll

    def nll_fn(th, D, Se):
        f = th - theta_star
        return 0.5 * float(jnp.dot(f, f)) + OFFSET

    theta, _, _, info = gn_minimize(
        jnp.zeros(2), gram_fn, jnp.eye(1), jnp.zeros((1, 1)),
        nll_fn=nll_fn, max_iter=5)
    np.testing.assert_allclose(np.asarray(theta), np.asarray(theta_star), atol=1e-8)
