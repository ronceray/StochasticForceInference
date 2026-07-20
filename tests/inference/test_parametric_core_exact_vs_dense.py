"""Pins the exact banded core against dense GLS references.

Three relations on one small OD problem, at fixed (θ, D, Λ):

1. **banded == dense** — the composition Phase A (per-point r, J, ψ via
   the θ-recursions) → covariance blocks → banded whitening reproduces
   the dense full-segment GLS objects exactly.
2. **runner == dense** — the ``ExactRuns`` Gram program reproduces the
   dense GLS ``(G, f, nll)``, invariant under chunking (carry
   threading).
3. **runner IV == whitening reference** — the ``w=1`` instrument path
   equals the backward innovations pairing over the instrument-valid
   centers (invalid residuals are excluded from the estimating
   equation).
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


T, D_DT = 46, 0.02


def _data():
    rng = np.random.default_rng(3)
    X = np.zeros((T, 1))
    X[0] = 0.5
    for t in range(T - 1):
        X[t + 1] = X[t] - 1.3 * X[t] * D_DT + np.sqrt(2 * 0.4 * D_DT) * rng.normal()
    X += 0.03 * rng.standard_normal(X.shape)          # measurement noise
    return jnp.asarray(X[:, None, :])                 # (T, N=1, d=1)


def _model():
    from SFI.bases import monomials_up_to

    F = monomials_up_to(2, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.1, -1.2, 0.05])
    D = jnp.asarray([[0.4]])
    Lam = jnp.asarray([[0.03**2]])
    return F, theta, D, Lam


def _point_tensors(with_instrument=False):
    from SFI.inference.parametric_core.precompute import od_point_tensors

    X = _data()
    F, theta, D, Lam = _model()
    out = od_point_tensors(F, theta, X, None, D_DT, 1, "rk4",
                           with_psi=True, with_instrument=with_instrument)
    return X, F, theta, D, Lam, out


def _dense_pieces(J, D, Lam):
    from SFI.inference.parametric_core.covariance import (
        CovarianceBlocks,
        assemble_dense,
        build_od_blocks,
    )

    blocks = build_od_blocks(J, D, Lam, D_DT, jitter=0.0)
    S = np.asarray(assemble_dense(CovarianceBlocks(blocks.A, blocks.offdiag, 1)))
    return blocks, S


def test_banded_matches_dense_gls():
    _, _, _, D, Lam, pt = _point_tensors()
    r = pt["r"][:, 0]                                  # (R, d)
    J = pt["J"][:, 0]
    psi = pt["psi"][:, 0]
    from SFI.inference.parametric_core.banded import banded_nll_gram

    blocks, S = _dense_pieces(J, D, Lam)
    Sinv = np.linalg.inv(S)
    rf = np.asarray(r).reshape(-1)
    Pf = np.asarray(psi).reshape(rf.size, -1)

    out = banded_nll_gram(blocks.A, blocks.offdiag[0], None, r, psi, jitter=0.0)
    np.testing.assert_allclose(np.asarray(out["G"]), Pf.T @ Sinv @ Pf, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out["f"]), Pf.T @ Sinv @ rf, rtol=1e-9)
    np.testing.assert_allclose(
        float(out["nll"]), 0.5 * rf @ Sinv @ rf + 0.5 * np.linalg.slogdet(S)[1],
        rtol=1e-10)



def _coll_from(X):
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    ds = TrajectoryDataset.from_arrays(X=X, dt=D_DT)
    return TrajectoryCollection.from_dataset(ds)


@pytest.mark.parametrize("chunk_pts", [1000, 13])
def test_runner_gram_matches_dense(chunk_pts):
    """ExactRuns == dense GLS, invariant under chunking (carry threading)."""
    from SFI.inference.parametric_core.runner import make_exact_runs_od
    from SFI.inference.parametric_core.gram import unpack_gram

    X = _data()
    F, theta, D, Lam = _model()
    runs = make_exact_runs_od(_coll_from(X), F, dt=D_DT, n_substeps=1,
                              integrator="rk4", w=0.0, jitter=0.0,
                              jitter_chol=0.0, chunk_pts=chunk_pts,
                              lyapunov=False, convexity=False)
    G, f, H, nll = unpack_gram(runs.gram(theta, D, Lam), int(F.template.size))

    _, _, _, _, _, pt = _point_tensors()
    _, S = _dense_pieces(pt["J"][:, 0], D, Lam)
    Sinv = np.linalg.inv(S)
    rf = np.asarray(pt["r"][:, 0]).reshape(-1)
    Pf = np.asarray(pt["psi"][:, 0]).reshape(rf.size, -1)

    np.testing.assert_allclose(np.asarray(G), Pf.T @ Sinv @ Pf, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(f), Pf.T @ Sinv @ rf, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(H), np.asarray(G), rtol=1e-9)
    np.testing.assert_allclose(
        float(nll), 0.5 * rf @ Sinv @ rf + 0.5 * np.linalg.slogdet(S)[1],
        rtol=1e-10)


def test_runner_iv_matches_banded_reference():
    """ExactRuns w=1 == the backward-IV pairing over all valid centers."""
    from SFI.inference.parametric_core.banded import whiten_segment
    from SFI.inference.parametric_core.covariance import build_od_blocks
    from SFI.inference.parametric_core.gram import unpack_gram
    from SFI.inference.parametric_core.runner import make_exact_runs_od

    X, F, theta, D, Lam, pt = _point_tensors(with_instrument=True)
    runs = make_exact_runs_od(_coll_from(X), F, dt=D_DT, n_substeps=1,
                              integrator="rk4", w=1.0, jitter=0.0,
                              jitter_chol=0.0, chunk_pts=1000,
                              lyapunov=False, convexity=False)
    G, f, H, _ = unpack_gram(runs.gram(theta, D, Lam), int(F.template.size))

    r = pt["r"][:, 0]
    psi = pt["psi"][:, 0]
    inst = pt["psi_inst"][:, 0]
    # instrument-invalid residuals (dataset front) are EXCLUDED from the IV
    # estimating equation (zero left row), not ψ-fallback — the η-dirty
    # fallback leaked measurement noise into the score at gap boundaries.
    left = jnp.where(pt["inst_valid"][:, None, None], inst, 0.0)
    blocks = build_od_blocks(pt["J"][:, 0], D, Lam, D_DT, jitter=0.0)
    d = r.shape[-1]
    c1 = jnp.concatenate([blocks.offdiag[0], jnp.zeros((1, d, d), r.dtype)])
    e_t, cols_t, raw_t, *_ = whiten_segment(
        blocks.A, c1, None, r, psi, raw_cols=left, jitter=0.0, reverse=True)

    np.testing.assert_allclose(
        np.asarray(G), np.einsum("kdn,kdm->nm", np.asarray(raw_t),
                                 np.asarray(cols_t)), rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(f), np.einsum("kdn,kd->n", np.asarray(raw_t),
                                 np.asarray(e_t)), rtol=1e-9)

