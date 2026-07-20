"""Measurement-noise-aware, banded residual whitening.

Localisation noise makes the diagnostic residual a moving-average process
(the overdamped increment is MA(1) with lag-1 block ``-Λ``).  A naive
single-residual whitening leaves that serial correlation in place and
trips the Ljung--Box test even for a perfectly-specified fit — the
false-positive these tests pin down.  The builder instead whitens with
the banded block-Cholesky innovations (:func:`_sequential_innovations`),
which decorrelate the stream, while keeping the *marginal* Mahalanobis
norms for the bias / MSE-consistency check (with a negligible-``Λ``
guard so a spurious clean-data estimate does not mask real bias).
"""

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI import OverdampedLangevinInference
from SFI.bases import X
from SFI.diagnostics import assess
from SFI.diagnostics.residuals import _sequential_innovations
from SFI.langevin import OverdampedProcess
from SFI.trajectory import TrajectoryCollection


def _lag1(z):
    """Per-component lag-1 autocorrelation of a ``(K, d)`` series."""
    zc = z - z.mean(0)
    return (zc[:-1] * zc[1:]).sum(0) / (zc * zc).sum(0)


def test_sequential_innovations_decorrelate_ma1():
    """The banded innovations whiten an MA(1) stream; the marginal does not.

    Build ``r_k = (η_{k+1} − η_k) + ξ_k`` — the overdamped increment with
    localisation noise — whose covariance has diagonal ``A + 2Λ`` and
    lag-1 block ``−Λ``.  The sequential innovations should come out unit
    variance and serially uncorrelated; the single-residual whitening keeps
    the lag-1 correlation ``−Λ / (A + 2Λ)``.
    """
    rng = np.random.RandomState(0)
    K, d = 6000, 2
    A = np.diag([1.0, 0.7])
    Lam = np.diag([0.3, 0.25])
    eta = rng.randn(K + 1, d) @ np.linalg.cholesky(Lam).T
    xi = rng.randn(K, d) @ np.linalg.cholesky(A).T
    r = (eta[1:] - eta[:-1]) + xi  # (K, d)

    A_blocks = np.broadcast_to(A + 2.0 * Lam, (K, d, d))
    contiguous = np.concatenate([[False], np.ones(K - 1, bool)])
    offdiag = -np.ones(K)  # Cov(r_{k-1}, r_k) = −Λ
    z = np.asarray(
        _sequential_innovations(
            jnp.asarray(r)[:, None, :],
            jnp.ones((K, 1), bool),
            jnp.asarray(contiguous),
            jnp.asarray(A_blocks),
            jnp.asarray(Lam),
            jnp.asarray(offdiag),
        )
    )[:, 0, :]

    # Banded: unit variance and no serial correlation.
    assert np.allclose(z.std(0), 1.0, atol=0.05)
    assert np.all(np.abs(_lag1(z)) < 0.05)

    # Marginal (single-residual) whitening would leave a clear lag-1.
    L_inv = np.linalg.inv(np.linalg.cholesky(A + 2.0 * Lam))
    z_marg = r @ L_inv.T
    lag1_marg = _lag1(z_marg)
    assert np.all(np.abs(lag1_marg) > 0.1)  # the false-positive being fixed
    # close to the analytic MA(1) value −Λ/(A+2Λ)
    expected = -np.diag(Lam) / (np.diag(A) + 2.0 * np.diag(Lam))
    assert np.allclose(lag1_marg, expected, atol=0.05)


def test_sequential_innovations_reset_on_gap():
    """A non-contiguous boundary decouples the two runs (no spurious coupling)."""
    K, d = 200, 1
    r = np.ones((K, 1, d))  # constant residual; coupling would distort it
    mask = jnp.ones((K, 1), bool)
    A_blocks = np.broadcast_to(np.eye(d), (K, d, d))
    contiguous = np.ones(K, bool)
    contiguous[K // 2] = False  # a gap halfway
    z = np.asarray(
        _sequential_innovations(
            jnp.asarray(r), mask, jnp.asarray(contiguous),
            jnp.asarray(A_blocks), jnp.zeros((d, d)), -np.ones(K),
        )
    )[:, 0, :]
    # Λ = 0 → off-diagonal block is zero → pure marginal whitening, finite.
    assert np.all(np.isfinite(z))
    assert np.allclose(z, 1.0)


def test_overdamped_measurement_noise_no_false_flags():
    """Noisy OU, correctly fit with the noise-profiling parametric estimator:
    the noise-aware banded diagnostics neither flag autocorrelation nor put
    the whitened std far from 1."""
    d = 2
    B = X(dim=d)
    proc = OverdampedProcess(B, D=0.5, theta_F=jnp.asarray([-1.0]))
    proc.initialize(jnp.zeros((1, d)))
    clean = proc.simulate(dt=0.02, Nsteps=12000, key=random.PRNGKey(0), oversampling=6)

    sigma = 0.05  # measurement-noise term ≈ 25% of the thermal increment
    Xn = np.asarray(clean.datasets[0].X)
    Xn = Xn + sigma * np.random.RandomState(1).randn(*Xn.shape)
    coll = TrajectoryCollection.from_arrays(X=Xn, dt=0.02)

    inf = OverdampedLangevinInference(coll)
    inf.infer_force(B)  # profiles (D, Λ) — a reliable Λ for the diagnostics
    inf.compute_force_error()

    rep = assess(inf, level="standard")
    assert rep.residuals["autocorr"]["ljung_box"]["pvalue"] > 0.01
    assert 0.7 < rep.residuals["moments"]["std"] < 1.4
    assert not any("autocorr" in m for m in rep.flag_issues())
