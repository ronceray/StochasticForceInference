"""Tests for SFI.inference.parametric_core.covariance — banded residual covariance."""

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


def _random_J(n_res, d, seed=0):
    rng = np.random.default_rng(seed)
    # near-identity flow Jacobians
    return jnp.array(np.eye(d)[None] + 0.1 * rng.standard_normal((n_res, d, d)))


def test_measurement_only_covariance_matches_explicit_linear_map():
    """With D=0, r_k = ε_{k+1} − J_k ε_k exactly; assembled Σ == L Λ Lᵀ.

    This is a non-circular check: L is the explicit (n_res·d)×((n_res+1)·d)
    measurement-noise map, built independently of the block formulas.
    """
    from SFI.inference.parametric_core.covariance import build_od_blocks, assemble_dense

    d, n_res = 2, 5
    J = _random_J(n_res, d, seed=1)
    Lambda = jnp.array([[0.04, 0.01], [0.01, 0.03]])
    D = jnp.zeros((d, d))
    dt = 0.05

    blocks = build_od_blocks(J, D, Lambda, dt, jitter=0.0)
    Sigma = assemble_dense(blocks)  # (n_res*d, n_res*d)

    # Explicit map r = L ε, ε = (ε_0, ..., ε_{n_res}) each (d,)
    n_eps = n_res + 1
    L = np.zeros((n_res * d, n_eps * d))
    Jn = np.asarray(J)
    for k in range(n_res):
        # r_k = ε_{k+1} − J_k ε_k
        L[k * d:(k + 1) * d, (k + 1) * d:(k + 2) * d] = np.eye(d)
        L[k * d:(k + 1) * d, k * d:(k + 1) * d] = -Jn[k]
    Se_block = np.kron(np.eye(n_eps), np.asarray(Lambda))
    Sigma_ref = L @ Se_block @ L.T

    np.testing.assert_allclose(np.asarray(Sigma), Sigma_ref, atol=1e-12)


def test_process_only_is_block_diagonal():
    """With Λ=0, residuals are uncorrelated: off-diagonal blocks vanish,
    diagonal = dt(J D Jᵀ + D)."""
    from SFI.inference.parametric_core.covariance import build_od_blocks

    d, n_res = 2, 4
    J = _random_J(n_res, d, seed=2)
    D = jnp.array([[0.5, 0.1], [0.1, 0.3]])
    Lambda = jnp.zeros((d, d))
    dt = 0.05

    blocks = build_od_blocks(J, D, Lambda, dt, jitter=0.0)

    np.testing.assert_allclose(np.asarray(blocks.offdiag[0]), 0.0, atol=1e-14)
    A_expect = dt * (jnp.einsum("kij,jl,kml->kim", J, D, J) + D[None])
    np.testing.assert_allclose(np.asarray(blocks.A), np.asarray(A_expect), atol=1e-12)


def test_assembled_covariance_is_symmetric_psd():
    from SFI.inference.parametric_core.covariance import build_od_blocks, assemble_dense

    d, n_res = 2, 6
    J = _random_J(n_res, d, seed=3)
    D = jnp.array([[0.5, 0.1], [0.1, 0.3]])
    Lambda = jnp.array([[0.04, 0.0], [0.0, 0.02]])
    blocks = build_od_blocks(J, D, Lambda, 0.05, jitter=1e-9)
    Sigma = np.asarray(assemble_dense(blocks))

    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)
    eig = np.linalg.eigvalsh(Sigma)
    assert eig.min() > 0, f"Sigma not PD: min eig {eig.min()}"


def test_isotropic_matrix_D_matches_scalar_times_identity():
    """Matrix path with D = c·I equals what the documented scalar form gives."""
    from SFI.inference.parametric_core.covariance import build_od_blocks

    d, n_res = 3, 4
    J = _random_J(n_res, d, seed=4)
    c = 0.7
    D = c * jnp.eye(d)
    Lambda = 0.02 * jnp.eye(d)
    dt = 0.03
    blocks = build_od_blocks(J, D, Lambda, dt, jitter=0.0)

    A_expect = dt * (c * jnp.einsum("kij,kmj->kim", J, J) + c * jnp.eye(d)[None]) \
        + 0.02 * jnp.einsum("kij,kmj->kim", J, J) + 0.02 * jnp.eye(d)[None]
    np.testing.assert_allclose(np.asarray(blocks.A), np.asarray(A_expect), atol=1e-12)


def test_od_per_step_D_diagonal_blocks():
    """build_od_blocks accepts per-step D (n_res,d,d): A_k = dt(J_k D_k J_kᵀ + D_k) + ..."""
    from SFI.inference.parametric_core.covariance import build_od_blocks

    d, n_res = 2, 4
    J = _random_J(n_res, d, seed=21)
    rng = np.random.default_rng(22)
    Dk = jnp.array(np.stack([np.diag(0.3 + 0.4 * rng.random(d)) for _ in range(n_res)]))  # (n_res,d,d)
    Lambda = 0.01 * jnp.eye(d)
    dt = 0.05

    blocks = build_od_blocks(J, Dk, Lambda, dt, jitter=0.0)
    JDJ = jnp.einsum("kij,kjl,kml->kim", J, Dk, J)
    JSJ = jnp.einsum("kij,jl,kml->kim", J, Lambda, J)
    A_exp = dt * (JDJ + Dk) + JSJ + Lambda[None]
    np.testing.assert_allclose(np.asarray(blocks.A), np.asarray(A_exp), atol=1e-12)


def _random_alphas(n_res, d, seed=0):
    rng = np.random.default_rng(seed)
    ap = jnp.broadcast_to(jnp.eye(d), (n_res, d, d))  # alpha_plus = I by construction
    a0 = jnp.array(np.eye(d)[None] + 0.1 * rng.standard_normal((n_res, d, d)))
    am = jnp.array(0.1 * rng.standard_normal((n_res, d, d)))
    return ap, a0, am


def test_ud_measurement_only_covariance_matches_explicit_map():
    """With D=0, r_n = α₋ε_n + α₀ε_{n+1} + α₊ε_{n+2}; assembled pentadiagonal
    Σ == L Λ Lᵀ from the explicit map (non-circular)."""
    from SFI.inference.parametric_core.covariance import build_ud_blocks, assemble_dense

    d, n_res = 2, 6
    ap, a0, am = _random_alphas(n_res, d, seed=1)
    Lambda = jnp.array([[0.04, 0.01], [0.01, 0.03]])
    blocks = build_ud_blocks(ap, a0, am, jnp.zeros((d, d)), Lambda, 0.1, jitter=0.0)
    assert blocks.bandwidth == 2
    Sigma = np.asarray(assemble_dense(blocks))

    n_eps = n_res + 2
    L = np.zeros((n_res * d, n_eps * d))
    amn, a0n, apn = np.asarray(am), np.asarray(a0), np.asarray(ap)
    for n in range(n_res):
        L[n * d:(n + 1) * d, n * d:(n + 1) * d] = amn[n]          # ε_n
        L[n * d:(n + 1) * d, (n + 1) * d:(n + 2) * d] = a0n[n]    # ε_{n+1}
        L[n * d:(n + 1) * d, (n + 2) * d:(n + 3) * d] = apn[n]    # ε_{n+2}
    Se = np.kron(np.eye(n_eps), np.asarray(Lambda))
    np.testing.assert_allclose(Sigma, L @ Se @ L.T, atol=1e-12)


def test_ud_process_only_blocks():
    """With Λ=0: A=(4/3)Δt³D, lag-1=(1/3)Δt³D, lag-2=0."""
    from SFI.inference.parametric_core.covariance import build_ud_blocks

    d, n_res, dt = 2, 5, 0.1
    ap, a0, am = _random_alphas(n_res, d, seed=2)
    D = jnp.array([[0.5, 0.1], [0.1, 0.3]])
    blocks = build_ud_blocks(ap, a0, am, D, jnp.zeros((d, d)), dt, jitter=0.0)
    A_exp = np.broadcast_to(np.asarray((4.0 / 3.0) * dt**3 * D), (n_res, d, d))
    C_exp = np.broadcast_to(np.asarray((1.0 / 3.0) * dt**3 * D), (n_res - 1, d, d))
    np.testing.assert_allclose(np.asarray(blocks.A), A_exp, atol=1e-12)
    np.testing.assert_allclose(np.asarray(blocks.offdiag[0]), C_exp, atol=1e-12)
    np.testing.assert_allclose(np.asarray(blocks.offdiag[1]), 0.0, atol=1e-12)


def test_ud_per_step_D_process_blocks():
    """build_ud_blocks accepts per-step D: A_n=(4/3)Δt³ D_sym[n], lag-1=0.5(C[n]+C[n+1])."""
    from SFI.inference.parametric_core.covariance import build_ud_blocks

    d, n_res, dt = 2, 5, 0.1
    ap, a0, am = _random_alphas(n_res, d, seed=31)
    rng = np.random.default_rng(32)
    Dk = jnp.array(np.stack([np.diag(0.3 + 0.5 * rng.random(d)) for _ in range(n_res)]))
    blocks = build_ud_blocks(ap, a0, am, Dk, jnp.zeros((d, d)), dt, jitter=0.0)

    Dsym = 0.5 * (Dk + jnp.swapaxes(Dk, -1, -2))
    A_exp = (4.0 / 3.0) * dt**3 * Dsym
    Cproc = (1.0 / 3.0) * dt**3 * Dsym
    C_exp = 0.5 * (Cproc[:-1] + Cproc[1:])
    np.testing.assert_allclose(np.asarray(blocks.A), np.asarray(A_exp), atol=1e-12)
    np.testing.assert_allclose(np.asarray(blocks.offdiag[0]), np.asarray(C_exp), atol=1e-12)


def test_ud_jitter_does_not_mask_D_at_small_dt():
    """The diagonal regulariser must scale with the block magnitude, not be an
    absolute floor: at small Δt the process variance is (4/3)Δt³D ~ 1e-9, so an
    absolute jitter=1e-7 would dominate it and make D unidentifiable.  The
    diagonal block must instead still reflect the process variance.
    """
    from SFI.inference.parametric_core.covariance import build_ud_blocks

    d, n_res, dt = 1, 4, 1e-3
    ap = jnp.broadcast_to(jnp.eye(d), (n_res, d, d))
    a0 = jnp.broadcast_to(jnp.eye(d), (n_res, d, d))
    am = jnp.zeros((n_res, d, d))
    D = jnp.eye(d)                       # truth D = 1
    Lambda = jnp.zeros((d, d))        # clean data
    jitter = 1e-7

    blocks = build_ud_blocks(ap, a0, am, D, Lambda, dt, jitter=jitter)
    A00 = float(blocks.A[0, 0, 0])
    proc = (4.0 / 3.0) * dt**3 * 1.0     # ≈ 1.33e-9
    assert abs(A00 - proc) / proc < 0.05, (
        f"A00={A00:.3e} dominated by jitter floor (process var={proc:.3e}); "
        "jitter must be relative to the block scale"
    )


def test_od_relative_jitter_is_small_fraction_of_block():
    """Overdamped: relative jitter is a tiny fraction of the diagonal block, so
    the block still equals the (no-jitter) value to high relative accuracy."""
    from SFI.inference.parametric_core.covariance import build_od_blocks

    d, n_res = 2, 4
    J = _random_J(n_res, d, seed=7)
    D = jnp.array([[0.5, 0.1], [0.1, 0.3]])
    Lambda = 0.02 * jnp.eye(d)
    dt = 0.05
    A0 = build_od_blocks(J, D, Lambda, dt, jitter=0.0).A
    Aj = build_od_blocks(J, D, Lambda, dt, jitter=1e-7).A
    # relative jitter perturbs the block by < 1e-5 of its magnitude
    rel = np.abs(np.asarray(Aj - A0)) / (np.abs(np.asarray(A0)) + 1e-300)
    assert rel.max() < 1e-5, f"jitter perturbs block by {rel.max():.2e} (should be ~jitter)"


def test_valid_mask_decouples_masked_residuals():
    from SFI.inference.parametric_core.covariance import build_od_blocks

    d, n_res = 2, 5
    J = _random_J(n_res, d, seed=5)
    D = jnp.array([[0.5, 0.0], [0.0, 0.3]])
    Lambda = 0.02 * jnp.eye(d)
    jitter = 1e-7
    mask = jnp.array([True, True, False, True, True])

    blocks = build_od_blocks(J, D, Lambda, 0.05, jitter=jitter, valid_mask=mask)

    # masked diagonal -> neutral (1+jitter) I
    np.testing.assert_allclose(
        np.asarray(blocks.A[2]), (1.0 + jitter) * np.eye(d), atol=1e-12
    )
    # off-diagonal blocks touching index 2 (pairs (1,2) and (2,3)) are zeroed
    np.testing.assert_allclose(np.asarray(blocks.offdiag[0][1]), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(blocks.offdiag[0][2]), 0.0, atol=1e-14)
