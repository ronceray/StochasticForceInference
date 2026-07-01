"""Tests for SFI.inference.parametric_core.objective — integrate Programs (loss + gram)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_ou_data(T=800, dt=0.02, d=2, seed=42):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    D = np.diag(np.array([0.5, 0.3], dtype=np.float64)[:d])
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ (rng.normal(size=d) * np.sqrt(dt))
    return X, k, D


def _make_collection(Y, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _make_linear_drift_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x

    return make_psf(
        f, dim=d, rank=1, n_features=1,
        params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)],
    )


def _setup(d=2, dt=0.02, seed=1):
    X, k, D = _make_ou_data(T=600, dt=dt, d=d, seed=seed)
    coll = _make_collection(X, dt)
    F_psf = _make_linear_drift_psf(d)
    theta = jnp.asarray(F_psf.flatten_params({"A": -jnp.asarray(k)}), dtype=jnp.float64)
    D_mat = jnp.asarray(D)
    Se = 1e-4 * jnp.eye(d)
    return coll, F_psf, theta, D_mat, Se, dt


def test_gram_program_runs_and_is_symmetric_psd():
    from SFI.inference.parametric_core.objective import ODGramProgram, unpack_gram
    from SFI.integrate.api import make_parametric_integrator

    coll, F_psf, theta, D, Se, dt = _setup()
    n_params = int(F_psf.template.size)
    prog = ODGramProgram(F_psf, dt=dt, n_substeps=4)
    _, run = make_parametric_integrator(
        coll, prog, reduce="sum", reduce_over_particles=True, weight_by_dt=False,
    )
    G, f, _H, nll = unpack_gram(run((theta, D, Se)), n_params)
    G = np.asarray(G)
    assert G.shape == (n_params, n_params) and f.shape == (n_params,)
    assert np.all(np.isfinite(G)) and np.all(np.isfinite(np.asarray(f)))
    # G is symmetric up to the local-window precision truncation (solve symmetrises)
    assert np.linalg.norm(G - G.T) / np.linalg.norm(G) < 1e-5
    assert np.linalg.eigvalsh(0.5 * (G + G.T)).min() > -1e-8


def test_loss_program_ad_gradient_matches_finite_difference():
    """AD-gradient of the summed windowed loss == central finite differences."""
    from SFI.inference.parametric_core.objective import ODLossProgram
    from SFI.integrate.api import make_parametric_integrator

    coll, F_psf, theta, D, Se, dt = _setup()
    prog = ODLossProgram(F_psf, dt=dt, n_substeps=4)
    _, run = make_parametric_integrator(
        coll, prog, reduce="sum", reduce_over_particles=True, weight_by_dt=False,
    )

    def total_loss(theta_live):
        return jnp.sum(run((theta_live, theta, D, Se)))

    g_ad = np.asarray(jax.grad(total_loss)(theta))

    eps = 1e-5
    g_fd = np.zeros_like(g_ad)
    for i in range(theta.shape[0]):
        e = jnp.zeros_like(theta).at[i].set(eps)
        g_fd[i] = float((total_loss(theta + e) - total_loss(theta - e)) / (2 * eps))

    np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-6)


def test_loss_gradient_approximates_gram_score():
    """grad(loss)|_{θ_live=θ} ≈ gram f (both are the windowed GN score)."""
    from SFI.inference.parametric_core.objective import (
        ODGramProgram, ODLossProgram, unpack_gram,
    )
    from SFI.integrate.api import make_parametric_integrator

    coll, F_psf, theta, D, Se, dt = _setup()
    n_params = int(F_psf.template.size)

    gprog = ODGramProgram(F_psf, dt=dt, n_substeps=4)
    _, grun = make_parametric_integrator(
        coll, gprog, reduce="sum", reduce_over_particles=True, weight_by_dt=False,
    )
    _, f_gram, _H, _ = unpack_gram(grun((theta, D, Se)), n_params)

    lprog = ODLossProgram(F_psf, dt=dt, n_substeps=4)
    _, lrun = make_parametric_integrator(
        coll, lprog, reduce="sum", reduce_over_particles=True, weight_by_dt=False,
    )
    g = np.asarray(jax.grad(lambda th: jnp.sum(lrun((th, theta, D, Se))))(theta))

    f_gram = np.asarray(f_gram)
    rel = np.linalg.norm(g - f_gram) / (np.linalg.norm(f_gram) + 1e-12)
    assert rel < 1e-3, f"loss grad vs gram score disagree: rel={rel:.2e}\n g={g}\n f={f_gram}"
