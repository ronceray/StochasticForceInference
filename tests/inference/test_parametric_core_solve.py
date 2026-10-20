"""End-to-end tests for SFI.inference.parametric_core.solve (OD force)."""

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


def _make_ou_data(T=2000, dt=0.02, d=2, seed=42, noise=0.0):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    D = np.diag(np.array([0.5, 0.3], dtype=np.float64)[:d])
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ (rng.normal(size=d) * np.sqrt(dt))
    Y = X + noise * rng.standard_normal(X.shape) if noise else X
    return Y, k, D


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


def test_solve_force_od_recovers_linear_drift():
    from SFI.inference.parametric_core.solve import solve_force_od

    # Long single trajectory; tolerance reflects single-realization OU drift
    # sampling error (matches existing window/loss backends to <1% — see parity test).
    Y, k, D_true = _make_ou_data(T=20000, dt=0.01, d=2, seed=7)
    coll = _make_collection(Y, dt=0.01)
    F_psf = _make_linear_drift_psf(2)

    res = solve_force_od(coll, F_psf, n_substeps=4, max_outer=6)
    A_hat = np.asarray(F_psf.unflatten_params(res.theta)["A"])

    rel = np.linalg.norm(A_hat - (-k)) / np.linalg.norm(k)
    assert rel < 0.10, f"A not recovered (rel={rel:.3e}):\n{A_hat}\nvs\n{-k}"
    # D recovered (matrix) tightly
    D_hat = np.asarray(res.D)
    assert np.linalg.norm(D_hat - D_true) / np.linalg.norm(D_true) < 0.05


def test_solve_force_od_profiles_Lambda_with_measurement_noise():
    """The profiled (D, Λ) path co-estimates drift, diffusion, and noise."""
    from SFI.inference.parametric_core.solve import solve_force_od

    sigma = 0.03
    Y, k, D_true = _make_ou_data(T=20000, dt=0.01, d=2, seed=5, noise=sigma)
    coll = _make_collection(Y, dt=0.01)
    F_psf = _make_linear_drift_psf(2)

    res = solve_force_od(coll, F_psf, n_substeps=4, max_outer=6)
    A_hat = np.asarray(F_psf.unflatten_params(res.theta)["A"])
    rel = np.linalg.norm(A_hat - (-k)) / np.linalg.norm(k)
    assert rel < 0.12, f"A not recovered (rel={rel:.3e}):\n{A_hat}\nvs {-k}"

    # measurement-noise variance recovered to the right order
    Lambda_hat = float(np.trace(np.asarray(res.Lambda))) / 2
    assert 0.4 * sigma**2 < Lambda_hat < 2.5 * sigma**2, f"Λ off: {Lambda_hat:.2e} vs {sigma**2:.2e}"
    # Λ is PSD
    assert np.linalg.eigvalsh(np.asarray(res.Lambda)).min() > -1e-9


def test_solve_force_od_gn_matches_lbfgs_linear():
    """The direct-GN inner solver reaches the same GLS fixed point as the
    L-BFGS path on a linear-in-θ OD problem (same θ̂ and D̂)."""
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, D_true = _make_ou_data(T=12000, dt=0.01, d=2, seed=9)
    coll = _make_collection(Y, dt=0.01)

    th_lbfgs = np.asarray(
        solve_force_od(coll, _make_linear_drift_psf(2), n_substeps=4, max_outer=6, inner="lbfgs", eiv=False).theta)
    res_gn = solve_force_od(coll, _make_linear_drift_psf(2), n_substeps=4, max_outer=6, inner="gn", eiv=False)
    th_gn = np.asarray(res_gn.theta)

    rel = np.linalg.norm(th_gn - th_lbfgs) / (np.linalg.norm(th_lbfgs) + 1e-30)
    assert rel < 2e-2, f"GN vs L-BFGS θ disagree: rel={rel:.3e}\n gn={th_gn}\n lbfgs={th_lbfgs}"
    assert np.linalg.norm(np.asarray(res_gn.D) - D_true) / np.linalg.norm(D_true) < 0.06
    assert res_gn.info.get("inner") == "gn"


def test_solve_force_od_auto_inner_uses_gn_for_basis():
    """``inner="auto"`` routes a linear Basis through the GN path."""
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, _ = _make_ou_data(T=8000, dt=0.01, d=2, seed=4)
    coll = _make_collection(Y, dt=0.01)
    basis = monomials_up_to(1, dim=2, rank="vector")

    res = solve_force_od(coll, basis, n_substeps=4, max_outer=6)  # inner="auto" default
    assert res.info.get("inner") == "gn"
    assert np.all(np.isfinite(np.asarray(res.theta)))


def test_solve_force_od_auto_n_substeps_removed():
    """``n_substeps="auto"`` (research path) was removed in v2.0: clear error,
    and the fixed substep count is recorded in ``info``."""
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, _ = _make_ou_data(T=2000, dt=0.02, d=2, seed=6)
    coll = _make_collection(Y, dt=0.02)

    with pytest.raises(ValueError, match="auto"):
        solve_force_od(coll, _make_linear_drift_psf(2), n_substeps="auto")

    res = solve_force_od(coll, _make_linear_drift_psf(2), n_substeps=2, max_outer=3)
    assert res.info["n_substeps"] == 2


def test_infer_force_routes_to_minimal_engine():
    """The public ``infer_force`` is the minimal engine: same θ̂ as a direct
    ``solve_force_od`` call, and the unified result surface is populated."""
    from SFI.inference.parametric_core.solve import solve_force_od
    from SFI.inference.overdamped import OverdampedLangevinInference

    Y, k, _ = _make_ou_data(T=3000, dt=0.02, d=2, seed=11)
    coll = _make_collection(Y, dt=0.02)

    res = solve_force_od(coll, _make_linear_drift_psf(2))
    theta_core = np.asarray(res.theta).ravel()

    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_make_linear_drift_psf(2))
    theta_pub = np.asarray(inf.force_coefficients_full).ravel()

    rel = np.linalg.norm(theta_pub - theta_core) / (np.linalg.norm(theta_core) + 1e-30)
    assert rel < 1e-10, f"public infer_force ≠ solve_force_od: rel={rel:.3e}"
    assert inf.metadata["force_method"] == "parametric"
    for attr in ("force_inferred", "force_G", "force_G_pinv", "Lambda",
                 "diffusion_average", "A_inv"):
        assert hasattr(inf, attr), f"missing {attr}"
