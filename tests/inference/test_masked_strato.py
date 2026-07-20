# TODO: review this file
# tests/inference/test_masked_strato.py
"""
Regression tests for Stratonovich gradient correction with masked data.

These tests verify that:
1. ``_apply_fill`` with ``zerostop`` preserves Jacobians for active entries
   while zeroing masked entries (no ``stop_gradient`` on the whole array).
2. ``DerivativeNode`` correctly flattens masks when vmapping over batch+particle.
3. Ito and Strato modes produce consistent force coefficients with masked
   multi-particle trajectories.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases.monomials import monomials_up_to
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simulate_ou(T, N, d, k, D, dt, seed=0):
    """Euler–Maruyama for a diagonal OU:  dX = -k X dt + sqrt(2D) dW."""
    rng = np.random.default_rng(seed)
    k = np.asarray(k, dtype=np.float32)
    D = np.asarray(D, dtype=np.float32)
    X = np.zeros((T, N, d), dtype=np.float32)
    for t in range(T - 1):
        eta = rng.normal(size=(N, d)).astype(np.float32)
        X[t + 1] = X[t] + dt * (-k * X[t]) + np.sqrt(2.0 * D * dt) * eta
    return X


def _random_mask(T, N, frac_masked=0.2, seed=1):
    """Per-particle mask with *frac_masked* entries knocked out per particle."""
    rng = np.random.default_rng(seed)
    mask = np.ones((T, N), dtype=bool)
    n_drop = int(frac_masked * T)
    for n in range(N):
        idx = rng.choice(T, size=n_drop, replace=False)
        mask[idx, n] = False
    return mask


# ---------------------------------------------------------------------------
# Test: _apply_fill preserves Jacobian for active entries
# ---------------------------------------------------------------------------
def test_apply_fill_zerostop_preserves_jacobian():
    """jnp.where in _apply_fill should NOT block jacfwd for active entries."""
    from SFI.statefunc.nodes.leaf import _apply_fill

    # Convention: arr has shape (batch..., spatial_dim), mask has shape (batch...)
    x = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1) — 3 samples, d=1
    mask = jnp.array([True, False, True])  # (3,)

    def f(x_in):
        x_safe = _apply_fill(x_in, mask, "zerostop")
        return (x_safe ** 2).ravel()  # (3,)

    J = jax.jacfwd(f)(x)  # (3, 3, 1)
    J = J[:, :, 0]        # (3, 3)
    # Active (0, 2): gradient = 2*x; masked (1): gradient = 0
    np.testing.assert_allclose(J[0, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(J[1, :], 0.0, atol=1e-5)  # masked row
    np.testing.assert_allclose(J[2, 2], 6.0, atol=1e-5)


def test_apply_fill_nanstop_preserves_jacobian():
    """Same test for nanstop policy."""
    from SFI.statefunc.nodes.leaf import _apply_fill

    x = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
    mask = jnp.array([True, False, True])  # (3,)

    def f(x_in):
        x_safe = _apply_fill(x_in, mask, "nanstop")
        # Re-mask output to avoid NaN in the result
        return jnp.where(mask, (x_safe ** 2).ravel(), 0.0)

    J = jax.jacfwd(f)(x)  # (3, 3, 1)
    J = J[:, :, 0]        # (3, 3)
    np.testing.assert_allclose(J[0, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(J[1, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(J[2, 2], 6.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: basis.d_x() Jacobian non-zero with mask
# ---------------------------------------------------------------------------
def test_basis_dx_nonzero_with_mask():
    """basis.d_x()(X, mask=mask) must return non-zero Jacobian for active particles."""
    basis = monomials_up_to(order=1, dim=1, rank="vector")
    dBdx = basis.d_x()

    X = jnp.array([[1.0], [2.0], [3.0]])          # (N=3, d=1)
    mask_all = jnp.array([True, True, True])       # all active
    mask_partial = jnp.array([True, False, True])  # particle 1 masked

    G_all = dBdx(X, mask=mask_all)
    G_partial = dBdx(X, mask=mask_partial)

    # Active particles should have non-zero gradient (feature 1 = x, so dB/dx = 1)
    assert not jnp.allclose(G_all, 0), "All-active Jacobian should be non-zero"
    assert not jnp.allclose(G_partial, 0), "Partially-masked Jacobian should be non-zero"

    # Active entries should match
    np.testing.assert_allclose(
        np.array(G_partial[0]), np.array(G_all[0]), atol=1e-6,
        err_msg="Active particle 0 Jacobian changed by mask"
    )
    np.testing.assert_allclose(
        np.array(G_partial[2]), np.array(G_all[2]), atol=1e-6,
        err_msg="Active particle 2 Jacobian changed by mask"
    )

    # Masked entry should be zero
    np.testing.assert_allclose(
        np.array(G_partial[1]), 0.0, atol=1e-6,
        err_msg="Masked particle 1 Jacobian should be zero"
    )


# ---------------------------------------------------------------------------
# Test: basis.d_x() with batched input (K, N, d) and mask (K, N)
# ---------------------------------------------------------------------------
def test_basis_dx_batched_with_mask():
    """Batched d_x() with (K,N,d) input and (K,N) mask must work."""
    basis = monomials_up_to(order=1, dim=1, rank="vector")
    dBdx = basis.d_x()

    K, N, d = 4, 3, 1
    X = jnp.ones((K, N, d), dtype=jnp.float32) * jnp.arange(K)[:, None, None]
    mask = jnp.ones((K, N), dtype=bool)
    mask = mask.at[1, 0].set(False)  # knock out one particle-time
    mask = mask.at[3, 2].set(False)

    G = dBdx(X, mask=mask)
    assert G.shape[0] == K and G.shape[1] == N, f"Unexpected shape {G.shape}"

    # Active particles: feature 1 (x) has d_x = 1
    # Feature 0 (constant 1) has d_x = 0
    # Masked particles: all zero
    G_np = np.array(G)
    assert not np.allclose(G_np, 0), "Batched Jacobian should be non-zero"
    # Check masked entries are zero
    np.testing.assert_allclose(G_np[1, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(G_np[3, 2], 0.0, atol=1e-6)
    # Check active entries are non-zero (feature 1)
    assert G_np[0, 0, ..., -1] != 0, "Active particle should have non-zero grad"
    assert G_np[2, 1, ..., -1] != 0, "Active particle should have non-zero grad"


# ---------------------------------------------------------------------------
# Test: Ito / Strato consistency with N=1, no mask (baseline)
# ---------------------------------------------------------------------------
def test_ito_strato_consistency_nomask():
    """Ito and Strato should give similar coefficients for 1D OU without mask."""
    T, N, d, dt = 2000, 1, 1, 0.01
    k, D = 1.0, 1.0
    X = _simulate_ou(T, N, d, k, D, dt, seed=10)

    ds = TrajectoryDataset.from_arrays(X=jnp.array(X), dt=dt)
    basis = monomials_up_to(order=1, dim=d, rank="vector")

    results = {}
    for mode in ("Ito", "Strato"):
        coll = TrajectoryCollection.from_dataset(ds)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="noisy")
        inf.infer_force_linear(basis, M_mode=mode)
        results[mode] = np.array(inf.force_coefficients_full)

    # x-coefficient should be near -k = -1.0 for both
    for mode, C in results.items():
        assert abs(C[-1] + k) < 0.3, f"{mode}: x-coeff {C[-1]:.3f} too far from -{k}"

    # Ito and Strato should be close to each other
    diff = abs(results["Ito"][-1] - results["Strato"][-1])
    assert diff < 0.2, f"Ito/Strato gap too large: {diff:.3f}"


# ---------------------------------------------------------------------------
# Test: Strato D_grad_b nonzero with mask (the main regression test)
# ---------------------------------------------------------------------------
def test_strato_grad_nonzero_with_mask():
    """Strato gradient correction D_grad_b must be non-zero even when mask is present."""
    T, N, d, dt = 1500, 3, 1, 0.01
    k, D = 1.0, 1.0
    X = _simulate_ou(T, N, d, k, D, dt, seed=20)
    mask = _random_mask(T, N, frac_masked=0.2, seed=21)

    ds = TrajectoryDataset.from_arrays(
        X=jnp.array(X), dt=dt, mask=jnp.array(mask)
    )
    basis = monomials_up_to(order=1, dim=d, rank="vector")

    coll = TrajectoryCollection.from_dataset(ds)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="noisy")
    inf.infer_force_linear(basis, M_mode="Strato")

    dgb = np.array(inf._force_D_grad_b_average)
    assert not np.allclose(dgb, 0), (
        f"D_grad_b is zero with mask — fill_policy may be blocking Jacobian: {dgb}"
    )
    # The x-feature gradient correction should be O(D) ≈ 1/dt * something
    assert abs(dgb[-1]) > 1.0, f"D_grad_b[-1] suspiciously small: {dgb[-1]}"


# ---------------------------------------------------------------------------
# Test: Ito / Strato consistency with masked multi-particle data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 3])
def test_ito_strato_consistency_masked(N):
    """Ito and Strato should give consistent coefficients with masked data."""
    T, d, dt = 2500, 1, 0.01
    k, D = 1.0, 1.0
    X = _simulate_ou(T, N, d, k, D, dt, seed=30 + N)
    mask = _random_mask(T, N, frac_masked=0.15, seed=31 + N)

    ds = TrajectoryDataset.from_arrays(
        X=jnp.array(X), dt=dt, mask=jnp.array(mask)
    )
    basis = monomials_up_to(order=1, dim=d, rank="vector")

    results = {}
    for mode in ("Ito", "Strato"):
        coll = TrajectoryCollection.from_dataset(ds)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="noisy")
        inf.infer_force_linear(basis, M_mode=mode)
        results[mode] = np.array(inf.force_coefficients_full)

    C_ito = results["Ito"][-1]
    C_strato = results["Strato"][-1]

    # Both should recover k ≈ -1.0 within statistical noise
    assert abs(C_ito + k) < 0.5, f"Ito x-coeff {C_ito:.3f} too far from -{k}"
    assert abs(C_strato + k) < 0.5, f"Strato x-coeff {C_strato:.3f} too far from -{k}"


# ---------------------------------------------------------------------------
# Test: 2D OU with mask — Strato nonzero gradient correction
# ---------------------------------------------------------------------------
def test_strato_2d_masked():
    """2D OU with mask: Strato gradient correction should be non-zero and finite."""
    T, N, d, dt = 1000, 2, 2, 0.01
    k = (0.8, 1.2)
    D = (0.3, 0.3)
    X = _simulate_ou(T, N, d, k, D, dt, seed=50)
    mask = _random_mask(T, N, frac_masked=0.1, seed=51)

    ds = TrajectoryDataset.from_arrays(
        X=jnp.array(X), dt=dt, mask=jnp.array(mask)
    )
    basis = monomials_up_to(order=1, dim=d, rank="vector")

    coll = TrajectoryCollection.from_dataset(ds)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="noisy")
    inf.infer_force_linear(basis, M_mode="Strato")

    dgb = np.array(inf._force_D_grad_b_average)
    assert np.isfinite(dgb).all(), f"D_grad_b has non-finite entries: {dgb}"
    assert not np.allclose(dgb, 0), f"D_grad_b all zero: {dgb}"

    C = np.array(inf.force_coefficients_full)
    assert np.isfinite(C).all(), f"Force coefficients non-finite: {C}"


# ---------------------------------------------------------------------------
# Test: Ito / Strato consistency with interacting particles
# ---------------------------------------------------------------------------
def _simulate_interacting(T, N, d, k_ext, k_int, D, dt, seed=0):
    """Euler–Maruyama for harmonic trap + pairwise spring."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, N, d), dtype=np.float32)
    for t in range(T - 1):
        F_ext = -k_ext * X[t]
        F_int = np.zeros_like(X[t])
        for i in range(N):
            for j in range(N):
                if i != j:
                    F_int[i] -= k_int * (X[t, i] - X[t, j])
        F = F_ext + F_int
        eta = rng.normal(size=(N, d)).astype(np.float32)
        X[t + 1] = X[t] + dt * F + np.sqrt(2.0 * D * dt) * eta
    return X


def _interacting_basis(d):
    """Build external + pairwise-spring basis."""
    from SFI.statefunc.factory import make_interactor

    def spring_pair(x):
        dr = x[1] - x[0]
        return dr[..., None]

    B_ext = monomials_up_to(order=1, dim=d, rank="vector")
    B_int = make_interactor(spring_pair, dim=d, rank=1, K=2, n_features=1).dispatch_pairs()
    return B_ext & B_int


def test_interacting_strato_grad_nonzero_masked():
    """Strato gradient correction must be non-zero for interacting particles + mask."""
    N, d, dt = 3, 1, 0.005
    T = 6000
    k_ext, k_int, D_val = 1.0, 0.3, 0.5

    X = _simulate_interacting(T, N, d, k_ext, k_int, D_val, dt, seed=60)
    mask = _random_mask(T, N, frac_masked=0.15, seed=61)

    ds = TrajectoryDataset.from_arrays(
        X=jnp.array(X), dt=dt, mask=jnp.array(mask)
    )
    basis = _interacting_basis(d)

    coll = TrajectoryCollection.from_dataset(ds)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="noisy")
    inf.infer_force_linear(basis, M_mode="Strato")

    dgb = np.array(inf._force_D_grad_b_average)
    assert np.isfinite(dgb).all(), f"D_grad_b has non-finite entries: {dgb}"
    assert not np.allclose(dgb, 0), f"D_grad_b all zero for interacting: {dgb}"

    C = np.array(inf.force_coefficients_full)
    assert np.isfinite(C).all(), f"Force coefficients non-finite: {C}"


@pytest.mark.parametrize("d", [1, 2])
def test_interacting_ito_strato_consistency_masked(d):
    """Ito and Strato should be consistent for interacting particles with mask."""
    N, dt = 3, 0.005
    T = 8000
    k_ext, k_int, D_val = 1.0, 0.3, 0.5

    X = _simulate_interacting(T, N, d, k_ext, k_int, D_val, dt, seed=70 + d)
    mask = _random_mask(T, N, frac_masked=0.15, seed=71 + d)

    ds = TrajectoryDataset.from_arrays(
        X=jnp.array(X), dt=dt, mask=jnp.array(mask)
    )
    basis = _interacting_basis(d)

    results = {}
    for mode in ("Ito", "Strato"):
        coll = TrajectoryCollection.from_dataset(ds)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="noisy")
        inf.infer_force_linear(basis, M_mode=mode)
        results[mode] = np.array(inf.force_coefficients_full)

    C_ito = results["Ito"]
    C_strato = results["Strato"]

    assert np.isfinite(C_ito).all(), f"Ito coefficients non-finite: {C_ito}"
    assert np.isfinite(C_strato).all(), f"Strato coefficients non-finite: {C_strato}"

    # Ito and Strato should give reasonably close results
    diff = np.max(np.abs(C_ito - C_strato))
    assert diff < 0.5, (
        f"Ito/Strato gap too large for interacting d={d}: {diff:.3f}\n"
        f"  Ito={C_ito}\n  Strato={C_strato}"
    )