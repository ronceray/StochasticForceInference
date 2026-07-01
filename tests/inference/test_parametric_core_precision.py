"""Tests for SFI.inference.parametric_core.precision — windowed center-row kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _random_band1_blocks(n_res, d, seed=0):
    """Random SPD bandwidth-1 covariance blocks."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n_res, d, d))
    for k in range(n_res):
        M = rng.standard_normal((d, d))
        A[k] = M @ M.T + (d + 1.0) * np.eye(d)  # well-conditioned SPD
    C = 0.05 * rng.standard_normal((n_res - 1, d, d))  # small off-diagonals
    return jnp.array(A), (jnp.array(C),)


def test_center_gram_contribution_matches_dense_precision():
    """G_w, f_w from the kernel equal ψ_cᵀ P_{c,:}ψ and ψ_cᵀ P_{c,:}r (dense P)."""
    from SFI.inference.parametric_core.precision import center_gram_contribution
    from SFI.inference.parametric_core.covariance import CovarianceBlocks, assemble_dense

    d, n_res, n_params = 2, 5, 3
    A, offdiag = _random_band1_blocks(n_res, d, seed=1)
    rng = np.random.default_rng(2)
    r = jnp.array(rng.standard_normal((n_res, d)))
    psi = jnp.array(rng.standard_normal((n_res, d, n_params)))
    c = 2  # interior center

    G_w, f_w, _H, _ = center_gram_contribution(A, offdiag, r, psi, c, jitter_chol=0.0)

    Sigma = np.asarray(assemble_dense(CovarianceBlocks(A, offdiag, 1)))
    P = np.linalg.inv(Sigma)
    Prow_c = P[c * d:(c + 1) * d, :]  # (d, n_res*d)
    Pr_c = Prow_c @ np.asarray(r).reshape(-1)          # (d,)
    Ppsi_c = Prow_c @ np.asarray(psi).reshape(n_res * d, n_params)  # (d, n_params)
    psi_c = np.asarray(psi[c])
    G_ref = psi_c.T @ Ppsi_c
    f_ref = psi_c.T @ Pr_c

    np.testing.assert_allclose(np.asarray(G_w), G_ref, atol=1e-10)
    np.testing.assert_allclose(np.asarray(f_w), f_ref, atol=1e-10)


def test_band1_local_window_center_row_is_exact():
    """For tridiagonal Σ the precision is exactly tridiagonal, so the center
    row from a W=5 local window equals the global full-inverse row."""
    from SFI.inference.parametric_core.precision import center_gram_contribution
    from SFI.inference.parametric_core.covariance import CovarianceBlocks, assemble_dense

    d, n_res, n_params = 1, 9, 1
    A, offdiag = _random_band1_blocks(n_res, d, seed=7)
    rng = np.random.default_rng(8)
    r = jnp.array(rng.standard_normal((n_res, d)))
    psi = jnp.ones((n_res, d, n_params))

    # global precision row at center index 4
    Sigma = np.asarray(assemble_dense(CovarianceBlocks(A, offdiag, 1)))
    P = np.linalg.inv(Sigma)
    Pr_global = (P[4 * d:(4 + 1) * d, :] @ np.asarray(r).reshape(-1))

    # local window: residuals 2..6 (W=5), center is local index 2
    sl = slice(2, 7)
    A_w = A[sl]
    C_w = (offdiag[0][2:6],)
    r_w = r[sl]
    psi_w = psi[sl]
    _, f_w, _H, _ = center_gram_contribution(A_w, C_w, r_w, psi_w, c=2, jitter_chol=0.0)
    # f_w = psi_c^T (P r)_c ; psi_c = 1 so f_w == (P r)_c
    np.testing.assert_allclose(np.asarray(f_w).ravel(), Pr_global, atol=1e-9)


def test_center_nll_contribution_equals_dense_nll():
    """center_nll_contribution = ½ r_cᵀ (P r)_c − ½ log det P_{cc} (dense P)."""
    from SFI.inference.parametric_core.precision import center_nll_contribution
    from SFI.inference.parametric_core.covariance import CovarianceBlocks, assemble_dense

    d, n_res = 2, 5
    A, offdiag = _random_band1_blocks(n_res, d, seed=9)
    rng = np.random.default_rng(10)
    r = jnp.array(rng.standard_normal((n_res, d)))
    c = 2

    nll = center_nll_contribution(A, offdiag, r, c, jitter_chol=0.0)

    Sigma = np.asarray(assemble_dense(CovarianceBlocks(A, offdiag, 1)))
    P = np.linalg.inv(Sigma)
    rc = np.asarray(r[c])
    Pr_c = (P[c * d:(c + 1) * d, :] @ np.asarray(r).reshape(-1))
    _, logdet = np.linalg.slogdet(P[c * d:(c + 1) * d, c * d:(c + 1) * d])
    ref = 0.5 * rc @ Pr_c - 0.5 * logdet
    np.testing.assert_allclose(float(nll), ref, atol=1e-9)


def test_conditional_nll_telescopes_to_exact():
    """Σ_c NLL(r_c | r_{<c}) over the full past == exact Gaussian NLL.

    Conditioning each residual on ALL its predecessors is the chain rule,
    so the conditional contributions sum to the exact block-Cholesky NLL.
    A fixed window (a few past points) approximates this — and crucially is
    non-degenerate in D, Λ (unlike the leave-one-out center block)."""
    from SFI.inference.parametric_core.precision import center_conditional_nll_contribution
    from SFI.inference.parametric_core.covariance import CovarianceBlocks, assemble_dense

    d, n_res = 2, 5
    A, (C,) = _random_band1_blocks(n_res, d, seed=11)
    rng = np.random.default_rng(12)
    r = jnp.array(rng.standard_normal((n_res, d)))

    total = 0.0
    for c in range(n_res):
        total = total + float(
            center_conditional_nll_contribution(A, (C,), r, c, n_cond=c, jitter_chol=0.0)
        )

    # exact Gaussian NLL (no 2π constant) of the full block-tridiagonal Σ
    Sigma = np.asarray(assemble_dense(CovarianceBlocks(A, (C,), 1)))
    rf = np.asarray(r).reshape(-1)
    _, logdet = np.linalg.slogdet(Sigma)
    ref = 0.5 * rf @ np.linalg.solve(Sigma, rf) + 0.5 * logdet
    np.testing.assert_allclose(total, ref, atol=1e-8)


def test_conditional_nll_is_not_degenerate_as_D_shrinks():
    """The conditional log-det penalises a vanishing covariance (no −∞ blow-up)."""
    from SFI.inference.parametric_core.precision import center_conditional_nll_contribution

    d, n_res = 1, 4
    rng = np.random.default_rng(13)
    r = jnp.array(rng.standard_normal((n_res, d)))
    vals = []
    for scale in (1.0, 0.1, 0.01):
        A = jnp.array(np.stack([scale * np.eye(d)] * n_res))
        C = jnp.zeros((n_res - 1, d, d))
        v = sum(float(center_conditional_nll_contribution(A, (C,), r, c, n_cond=min(c, 2), jitter_chol=0.0))
                for c in range(n_res))
        vals.append(v)
    # shrinking the (here diagonal) covariance must INCREASE the NLL, not send it to −∞
    assert vals[2] > vals[1] > vals[0], vals


def test_center_loss_contribution_equals_local_quadratic():
    """center_loss_contribution = ½ r_cᵀ P_cc r_c + r_cᵀ P_{c,c+1} r_{c+1} (frozen P)."""
    from SFI.inference.parametric_core.precision import center_loss_contribution
    from SFI.inference.parametric_core.covariance import CovarianceBlocks, assemble_dense

    d, n_res = 2, 5
    A, offdiag = _random_band1_blocks(n_res, d, seed=3)
    rng = np.random.default_rng(4)
    r = jnp.array(rng.standard_normal((n_res, d)))
    c = 2

    loss = center_loss_contribution(A, offdiag, r, c, jitter_chol=0.0, bandwidth=1)

    Sigma = np.asarray(assemble_dense(CovarianceBlocks(A, offdiag, 1)))
    P = np.linalg.inv(Sigma)
    rc = np.asarray(r[c])
    rcp1 = np.asarray(r[c + 1])
    P_cc = P[c * d:(c + 1) * d, c * d:(c + 1) * d]
    P_c_cp1 = P[c * d:(c + 1) * d, (c + 1) * d:(c + 2) * d]
    ref = 0.5 * rc @ P_cc @ rc + rc @ P_c_cp1 @ rcp1
    np.testing.assert_allclose(float(loss), ref, atol=1e-10)
