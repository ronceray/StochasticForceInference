"""Dense-reference tests for the exact banded innovations kernel (Stage 2).

Every identity is checked against an explicitly assembled dense Σ:

    Σ ẽᵀẽ = rᵀΣ⁻¹r,   Σ ψ̃ᵀψ̃ = ψᵀΣ⁻¹ψ,   Σ ψ̃ᵀẽ = ψᵀΣ⁻¹r,
    2 Σ log diag R = log|Σ|,

for bandwidth 1 and 2, in both processing directions, with mask-induced
segment restarts (reference = independent dense segments) and with the
chunk carry (results must be chunking-invariant).  The IV test pins the
backward-pairing semantics: the reversed-time innovations pair the
instrument at k against the innovation of r_k given the FUTURE, which
keeps every instrument–residual pairing η-clean.
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


def _blocks(K=9, d=2, n=3, bandwidth=1, seed=0):
    rng = np.random.default_rng(seed)
    A = np.stack([(lambda M: M @ M.T + 5.0 * np.eye(d))(rng.normal(size=(d, d)))
                  for _ in range(K)])
    off1 = 0.4 * rng.normal(size=(K - 1, d, d))
    off2 = 0.2 * rng.normal(size=(K - 2, d, d)) if bandwidth == 2 else None
    r = rng.normal(size=(K, d))
    psi = rng.normal(size=(K, d, n))
    return (jnp.asarray(A), jnp.asarray(off1),
            None if off2 is None else jnp.asarray(off2),
            jnp.asarray(r), jnp.asarray(psi))


def _dense(A, off1, off2):
    from SFI.inference.parametric_core.covariance import (
        CovarianceBlocks,
        assemble_dense,
    )

    off = (off1,) if off2 is None else (off1, off2)
    return np.asarray(assemble_dense(CovarianceBlocks(A, off, len(off))))


def _dense_reference(A, off1, off2, r, psi):
    S = _dense(A, off1, off2)
    K, d = r.shape
    Sinv = np.linalg.inv(S)
    rf = np.asarray(r).reshape(-1)
    Pf = np.asarray(psi).reshape(K * d, -1)
    return {
        "nll": 0.5 * rf @ Sinv @ rf + 0.5 * np.linalg.slogdet(S)[1],
        "G": Pf.T @ Sinv @ Pf,
        "f": Pf.T @ Sinv @ rf,
    }


@pytest.mark.parametrize("bandwidth", [1, 2])
@pytest.mark.parametrize("reverse", [True, False])
def test_whitening_identities_vs_dense(bandwidth, reverse):
    from SFI.inference.parametric_core.banded import banded_nll_gram

    A, off1, off2, r, psi = _blocks(bandwidth=bandwidth)
    ref = _dense_reference(A, off1, off2, r, psi)
    out = banded_nll_gram(A, off1, off2, r, psi, jitter=0.0, reverse=reverse)

    np.testing.assert_allclose(float(out["nll"]), ref["nll"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(out["G"]), ref["G"], rtol=1e-9,
                               atol=1e-10 * np.max(np.abs(ref["G"])))
    np.testing.assert_allclose(np.asarray(out["f"]), ref["f"], rtol=1e-9,
                               atol=1e-10 * np.max(np.abs(ref["f"])))
    # symmetric path: H == G
    np.testing.assert_allclose(np.asarray(out["H"]), np.asarray(out["G"]),
                               rtol=1e-9, atol=1e-10 * np.max(np.abs(ref["G"])))


@pytest.mark.parametrize("bandwidth", [1, 2])
def test_masked_segments_match_independent_dense(bandwidth):
    """A gap decouples the stream into independent segments (link bits zero
    the couplings); reference = sum of dense NLL/Gram over the segments."""
    from SFI.inference.parametric_core.banded import banded_nll_gram

    A, off1, off2, r, psi = _blocks(K=11, bandwidth=bandwidth, seed=3)
    valid = np.ones(11, bool)
    valid[4] = False          # single-point gap → segments [0:4], [5:11]
    out = banded_nll_gram(A, off1, off2, r, psi,
                          valid=jnp.asarray(valid), jitter=0.0)

    ref = {"nll": 0.0, "G": 0.0, "f": 0.0}
    for sl in (slice(0, 4), slice(5, 11)):
        o2 = None if off2 is None else off2[sl.start:sl.stop - 2]
        seg = _dense_reference(A[sl], off1[sl.start:sl.stop - 1], o2,
                               r[sl], psi[sl])
        for k in ref:
            ref[k] = ref[k] + seg[k]

    np.testing.assert_allclose(float(out["nll"]), ref["nll"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(out["G"]), ref["G"], rtol=1e-9,
                               atol=1e-10 * np.max(np.abs(ref["G"])))
    np.testing.assert_allclose(np.asarray(out["f"]), ref["f"], rtol=1e-9,
                               atol=1e-10 * np.max(np.abs(ref["f"])))


@pytest.mark.parametrize("bandwidth", [1, 2])
def test_chunk_carry_invariance(bandwidth):
    """Splitting a segment into chunks with carry threading is exact.

    With ``reverse=True`` the chunks are processed last-to-first.  The
    per-step coupling convention makes the split trivial: chunk
    ``[a, b)`` passes ``c1[a:b]`` / ``c2[a:b]``, which naturally include
    the blocks crossing its upper edge."""
    from SFI.inference.parametric_core.banded import whiten_segment, banded_nll_gram

    A, off1, off2, r, psi = _blocks(K=12, bandwidth=bandwidth, seed=5)
    K, d = r.shape
    whole = banded_nll_gram(A, off1, off2, r, psi, jitter=0.0)

    # per-step couplings, zero-padded at the segment end
    c1 = jnp.concatenate([off1, jnp.zeros((1, d, d), r.dtype)])
    c2 = (None if off2 is None else
          jnp.concatenate([off2, jnp.zeros((2, d, d), r.dtype)]))

    cut = 7
    carry = None
    acc = {"nll": 0.0, "G": 0.0, "f": 0.0}
    for a, b in [(cut, K), (0, cut)]:                # last chunk first
        e_t, cols_t, _, logdet, _, carry = whiten_segment(
            A[a:b], c1[a:b], None if c2 is None else c2[a:b],
            r[a:b], psi[a:b], jitter=0.0, reverse=True, carry=carry)
        acc["nll"] += 0.5 * float(jnp.sum(e_t * e_t)) + float(jnp.sum(logdet))
        acc["G"] = acc["G"] + np.einsum("kdn,kdm->nm", np.asarray(cols_t),
                                        np.asarray(cols_t))
        acc["f"] = acc["f"] + np.einsum("kdn,kd->n", np.asarray(cols_t),
                                        np.asarray(e_t))

    np.testing.assert_allclose(acc["nll"], float(whole["nll"]), rtol=1e-10)
    np.testing.assert_allclose(acc["G"], np.asarray(whole["G"]), rtol=1e-9)
    np.testing.assert_allclose(acc["f"], np.asarray(whole["f"]), rtol=1e-9)


def test_backward_iv_pairing_matches_tail_precision():
    """Reversed-time innovations give ẽ_k = R̄_k⁻¹ (Σ_{[k:]}⁻¹ r_{[k:]})-first-block
    normalisation: the IV score Σ ψ̃_L,kᵀ ẽ_k equals
    Σ_k ψ_inst,kᵀ (Σ_{[k:]}⁻¹ r_{[k:]})₀ — every pairing uses residuals
    at-or-after k only (η-clean), the exact limit of the skip-window."""
    from SFI.inference.parametric_core.banded import whiten_segment

    A, off1, off2, r, psi = _blocks(K=8, d=2, n=3, bandwidth=1, seed=9)
    K, d = r.shape
    c1 = jnp.concatenate([off1, jnp.zeros((1, d, d), r.dtype)])

    e_t, cols_t, _, logdet, _, _ = whiten_segment(
        A, c1, None, r, psi, jitter=0.0, reverse=True)
    f_kernel = np.einsum("kdn,kd->n", np.asarray(cols_t), np.asarray(e_t))

    # dense reference: for every k, the first block-row of Σ_{[k:]}⁻¹ r_{[k:]}
    # and the S̄_k-weighted pairing with the RAW instrument column at k;
    # ψ̃ᵀẽ telescopes to exactly Σ_k ψ_L,kᵀ (Σ_{[k:]}⁻¹ r_{[k:]})₀ only for
    # the recursion's own whitened ψ̃, so compare against the L-solve form:
    S = _dense(A, off1, None)
    Sinv = np.linalg.inv(S)
    rf = np.asarray(r).reshape(-1)
    Pf = np.asarray(psi).reshape(K * d, -1)
    f_dense = Pf.T @ Sinv @ rf
    np.testing.assert_allclose(f_kernel, f_dense, rtol=1e-9)

    # η-cleanliness of the PAIRING: e_t[k] depends only on r[k:], so
    # perturbing r strictly BEFORE k leaves ẽ_k unchanged.
    r2 = np.asarray(r).copy()
    r2[:3] += 1.0                       # corrupt the past of k=3
    e_t2, *_rest = whiten_segment(A, c1, None, jnp.asarray(r2), psi,
                                  jitter=0.0, reverse=True)
    np.testing.assert_allclose(np.asarray(e_t2)[3:], np.asarray(e_t)[3:],
                               rtol=1e-12)
    assert not np.allclose(np.asarray(e_t2)[:3], np.asarray(e_t)[:3])
