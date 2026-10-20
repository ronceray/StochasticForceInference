# TODO: review this file
# tests/inference/test_overdamped_inference.py
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.statefunc.factory import make_basis
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


def _make_ou_collection(
    T=600, dt=0.01, d=2, k_diag=(0.7, 1.1), D_diag=(0.2, 0.2), seed=0
):
    """
    Simulate a diagonal OU: x_{t+1} = x_t + dt*(-K x_t) + sqrt(2D dt) * N(0,I).
    Returns a TrajectoryCollection with a single dataset (N=1 particle).
    """
    rng = np.random.default_rng(seed)
    k = np.array(k_diag, dtype=np.float32)
    D = np.array(D_diag, dtype=np.float32)
    x = np.zeros((T, 1, d), dtype=np.float32)
    for t in range(T - 1):
        eta = rng.normal(size=(1, d)).astype(np.float32)
        x[t + 1, 0] = x[t, 0] + dt * (-k * x[t, 0]) + np.sqrt(2.0 * D * dt) * eta[0]
    ds = TrajectoryDataset.from_arrays(X=jnp.array(x), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds), k, D


def _coord_linear_vector_basis(dim: int):
    """
    Vector basis with features last: B(x)[i,m,a] = 1_{m=a} * x[i,a].
    One feature per coordinate, so F(x) = sum_a theta[a] * e_a * x_a.
    """

    def f(x, **kw):
        I = jnp.eye(dim, dtype=x.dtype)  # (m,a)
        return jnp.einsum("m,ma->ma", x, I)  # (i,m,a)

    return make_basis(f, dim=dim, rank=1, n_features=dim)


def _diag_rank2_basis(dim: int):
    """
    Rank-2 diffusion basis with features last.
    B0(x): identity (constant)
    B1(x): diag(x^2)
    """

    def f(x, **kw):
        I = jnp.eye(dim, dtype=x.dtype)
        B0 = jnp.einsum("mn->imna", I)[: x.shape[-2]]  # broadcast to (i,m,n)
        B1 = jnp.einsum("im,mn->imn", x**2, I)  # diag(x^2)
        # stack features last: (i,m,n,a) with a in {0,1}
        return jnp.stack([B0, B1], axis=-1)

    return make_basis(f, dim=dim, rank=2, n_features=2)


def test_diffusion_constant_smoketest_and_psd():
    coll, k, D = _make_ou_collection(T=500, dt=0.01, d=2, D_diag=(0.15, 0.25), seed=1)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")  # explicit; avoids auto branch
    # shapes and symmetry
    assert inf.diffusion_average.shape == (2, 2)
    np.testing.assert_allclose(
        inf.diffusion_average, inf.diffusion_average.T, rtol=1e-6, atol=1e-6
    )
    # PSD check (numerical slack)
    eig = np.linalg.eigvalsh(np.array(inf.diffusion_average))
    assert np.all(eig > -1e-6)


def test_force_linear_ito_recovers_ou_diagonal_to_reasonable_error():
    d = 2
    T = 4000
    dt = 0.01
    k_true = jnp.array([0.8, 1.2], dtype=jnp.float32)
    D_true = (0.20, 0.20)
    coll, _, _ = _make_ou_collection(
        T=T, dt=dt, d=d, k_diag=tuple(k_true.tolist()), D_diag=D_true, seed=11
    )

    basis = _coord_linear_vector_basis(d)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="rectangle")

    # Coefficients should be close to -k on each coord
    theta = np.array(inf.force_coefficients_full)  # length d
    assert theta.shape == (d,)
    # generous tolerance to keep CI fast
    np.testing.assert_allclose(theta, -np.array(k_true), rtol=2e-1, atol=2e-2)

    # Field-level check on a subsample
    Xs = coll.datasets[0].X[:: max(1, T // 50), 0]  # (S, d)
    F_est = jnp.vstack([inf.force_inferred(x[None, :])[0] for x in Xs])
    F_true = -k_true * Xs
    num = jnp.sum(jnp.einsum("sd,sd->s", F_est - F_true, F_est - F_true))
    den = jnp.sum(jnp.einsum("sd,sd->s", F_true, F_true)) + 1e-8
    rel = float(num / den)
    assert rel < 0.5  # not perfect, but clearly informative


def test_force_linear_strato_path_runs_and_returns_finite_moments():
    d = 2
    coll, _, _ = _make_ou_collection(T=700, dt=0.01, d=d, seed=7)
    basis = _coord_linear_vector_basis(d)

    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="noisy")
    inf.infer_force_linear(basis, M_mode="Strato", G_mode="trapeze")

    assert np.isfinite(np.asarray(inf.force_G_full)).all()
    assert np.isfinite(np.asarray(inf.force_moments)).all()
    assert inf.force_G_full.shape == (d, d)
    assert inf.force_moments.shape == (d,)


def test_compare_to_exact_smoketest():
    d = 2
    k = jnp.array([0.6, 1.0], dtype=jnp.float32)
    D = jnp.array([[0.2, 0.0], [0.0, 0.2]], dtype=jnp.float32)
    coll, _, _ = _make_ou_collection(
        T=800, dt=0.01, d=d, k_diag=tuple(k.tolist()), D_diag=(0.2, 0.2), seed=9
    )

    basis = _coord_linear_vector_basis(d)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="rectangle")

    def F_exact(x, **kw):
        return -k * x

    # diffusion_exact can be a constant array
    inf.compare_to_exact(force_exact=F_exact, diffusion_exact=D, maxpoints=300)
    assert hasattr(inf, "NMSE_force") and np.isfinite(np.asarray(inf.NMSE_force))


def test_infer_diffusion_linear_no_arg_defaults_to_symmetric_basis():
    """infer_diffusion_linear() with no basis defaults to symmetric_matrix_basis(d)."""
    d = 2
    coll, _, _ = _make_ou_collection(T=600, dt=0.01, d=d, seed=42)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_diffusion_linear()
    assert hasattr(inf, "diffusion_inferred")
    assert hasattr(inf, "diffusion_coefficients")
    # d(d+1)/2 = 3 free parameters for a 2×2 symmetric matrix
    assert inf.diffusion_coefficients.shape == (d * (d + 1) // 2,)



def test_diffusion_constant_method_dispatch_respected():
    """An explicit estimator must be used, not silently replaced by auto.

    Regression: compute_diffusion_constant() used to overwrite ``method``
    with the auto selection unconditionally, so "MSD" could never run.
    On noisy data, MSD is inflated by Lambda/dt relative to noisy,
    so the two must record their names and differ numerically.
    """
    coll, k, D = _make_ou_collection(T=2000, dt=0.01, d=2, D_diag=(0.2, 0.3), seed=3)
    coll = coll.degrade(noise=0.05, seed=7)

    inf_msd = OverdampedLangevinInference(coll)
    inf_msd.compute_diffusion_constant(method="MSD")
    assert inf_msd.metadata["diffusion_constant_method"] == "MSD"

    inf_vest = OverdampedLangevinInference(coll)
    inf_vest.compute_diffusion_constant(method="noisy")
    assert inf_vest.metadata["diffusion_constant_method"] == "noisy"

    d_msd = float(np.trace(np.asarray(inf_msd.diffusion_average))) / 2
    d_vest = float(np.trace(np.asarray(inf_vest.diffusion_average))) / 2
    # MSD carries the +Lambda/dt noise bias: sigma^2/dt = 0.05^2/0.01 = 0.25
    assert d_msd > d_vest + 0.1, (d_msd, d_vest)
