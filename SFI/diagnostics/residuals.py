"""Per-backend residual builders.

Each builder takes a fitted inference object and returns a
:class:`ResidualBundle` containing pooled standardized residuals
:math:`z = \\Sigma^{-1/2} r` ready to feed into the statistical tests.

Measurement-noise-aware, banded whitening
-----------------------------------------
Both residuals carry two correlation sources that a single-residual
whitening ignores:

* **Measurement noise** :math:`\\Sigma_\\eta`.  The diagnostic residual
  covariance is :math:`C = \\text{(thermal)} + c\\,\\Sigma_\\eta`, not the
  thermal part alone.  The estimator's profiled :math:`\\Sigma_\\eta`
  (``inferer.Lambda``) is folded into ``C`` so that a *well-recovered but
  noisy* fit still whitens to unit variance instead of tripping every
  flag.  On clean data :math:`\\Sigma_\\eta\\approx 0` and this reduces to
  the thermal whitening.
* **Serial correlation.**  Localisation error is shared between
  neighbouring residuals, so the residual series is a moving-average
  process (overdamped increment → MA(1) with lag-1 block
  :math:`-\\Sigma_\\eta`; the kept underdamped acceleration series → MA(1)
  with lag-1 block :math:`\\Sigma_\\eta/\\Delta t^4`).  A *banded*
  whitening — the sequential block-Cholesky innovations of the
  tridiagonal residual covariance (:func:`_sequential_innovations`) —
  decorrelates the stream, exactly paralleling the parametric core's
  banded precision.  On clean data the off-diagonal block vanishes and
  the innovations coincide with the marginal whitening.

The whitened stream ``z`` (moments / normality / autocorrelation) uses
the banded innovations; the per-row Mahalanobis norms ``z_squared_norms``
(the chi-square / MSE-consistency *bias* check) keep the **marginal**
noise-aware form, which faithfully preserves a slowly-varying force bias
that the innovations would partly difference out.

Residual conventions
--------------------
**Overdamped**:

.. math::

    r_{t,n} = X_{t+1,n} - X_{t,n} - F(X_{t,n})\\,\\Delta t,
    \\qquad C_{t} = 2\\,\\bar D\\,\\Delta t + 2\\,\\Sigma_\\eta,

with lag-1 covariance :math:`-\\Sigma_\\eta`. For the linear path the
thermal part is the exact ML residual; for the parametric path it is an
approximation that is nevertheless consistent (whitened residuals should
have unit variance and no autocorrelation if the model is well
specified).

**Underdamped**: symmetric acceleration
:math:`\\hat a_t = (X_{t+1} - 2X_t + X_{t-1})/\\Delta t^2`,

.. math::

    r_t = \\hat a_t - F(\\hat x_t, \\hat v_t),
    \\qquad C_t = \\tfrac23\\,\\frac{2\\bar D}{\\Delta t}
                  + \\frac{6\\,\\Sigma_\\eta}{\\Delta t^4}.

For both regimes residuals are pooled across time, particles, and
spatial components, applying the dataset's ``dynamic_mask`` (for
overdamped) or its 1-step erosion (for underdamped, which needs three
consecutive valid observations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass
class ResidualBundle:
    """Standardised residuals + metadata.

    Attributes
    ----------
    z : np.ndarray
        Whitened residuals, shape ``(K,)``. Pooled across time,
        particles and spatial components after masking.
    z_components : np.ndarray
        Whitened residuals organised by spatial component, shape
        ``(K_per_component, d)``. Used for per-axis statistics.
    z_squared_norms : np.ndarray
        Per-row squared Mahalanobis norm
        :math:`r_t^\\top \\Sigma_t^{-1} r_t`, shape ``(K_per_row,)``.
        Used for the diffusion / "chi-square" check.
    force_quadratic_form : np.ndarray
        Per-row quadratic form
        :math:`F^\\top A^{-1} F` evaluated on the same valid samples
        used to build ``z``. Pre-computing it here avoids a second
        evaluation of ``F`` in the MSE-consistency check downstream.
    mean_dt : float
        Average step size used in the residual construction.
    n_obs : int
        Number of valid (un-masked) observations used to build ``z``.
    d : int
        Spatial dimension.
    regime : str
        ``"OD"`` or ``"UD"``.
    backend : str
        Coarse tag of the inference path (``"linear"``, ``"parametric"``,
        ``"nonlinear"``). For diagnostic display only.
    n_particles : int
        Maximum number of particles in any dataset.
    nmse_excess_factor : float
        Conversion factor from the chi-square excess to the force NMSE in
        :func:`mse_consistency`.  ``1.0`` for the overdamped increment
        residual; :data:`KAPPA_UD` for the underdamped acceleration
        residual (see that constant for the derivation).
    whitened : list of (np.ndarray, np.ndarray)
        Per-dataset ``(z_full, mask)`` pairs with ``z_full`` of shape
        ``(K, N, d)`` (time-major) and ``mask`` of shape ``(K, N)``.
        Kept so that autocorrelation can be measured strictly along
        time, per particle and per component — pooling the flattened
        ``z`` stream would mix particles and components at short lags.
    """

    z: np.ndarray
    z_components: np.ndarray
    z_squared_norms: np.ndarray
    force_quadratic_form: np.ndarray
    mean_dt: float
    n_obs: int
    d: int
    regime: str
    backend: str
    n_particles: int
    nmse_excess_factor: float = 1.0
    whitened: list = field(default_factory=list)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _inv_sqrt_psd(S: jnp.ndarray) -> jnp.ndarray:
    """Batched symmetric inverse square root ``S^{-1/2}`` (eigen-clamped).

    ``S`` has shape ``(..., d, d)`` and is symmetrised and floored before
    inversion so a marginally-indefinite conditional covariance (high
    measurement noise) stays well-posed.
    """
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2))
    w, U = jnp.linalg.eigh(S)
    floor = jnp.maximum(jnp.max(w, axis=-1, keepdims=True) * 1e-12, 1e-30)
    w = jnp.clip(w, floor, None)
    return jnp.einsum("...am,...m,...bm->...ab", U, 1.0 / jnp.sqrt(w), U)


@jax.jit
def _sequential_innovations(r, mask, contiguous, A_blocks, Lambda, offdiag_coef):
    r"""Whiten a tridiagonal (MA(1)) residual stream by block-Cholesky innovations.

    The residual series has diagonal covariance blocks ``A_blocks[k]`` and
    lag-1 blocks ``C_k = Cov(r_{k-1}, r_k) = offdiag_coef[k]·Λ``.  The
    LDL\\ :sup:`T` factorisation of the resulting block-tridiagonal
    covariance gives the innovations recursion (the exact whitening, the
    diagnostic twin of the parametric core's banded precision):

    .. math::

        M_k = C_k\\,S_{k-1}^{-1},\\quad
        w_k = r_k - M_k w_{k-1},\\quad
        S_k = A_k - M_k C_k,\\quad
        z_k = S_k^{-1/2} w_k,

    so ``z`` has unit covariance *and no serial correlation* — unlike the
    marginal whitening ``A_k^{-1/2} r_k``, which leaves the measurement-noise
    off-diagonal in place.  The recursion **resets** (drops to the marginal
    form ``z_k = A_k^{-1/2} r_k``) at the start of each contiguous run: where
    ``contiguous[k]`` is ``False`` (the kept index is not the immediate
    successor of the previous one) or either endpoint is masked, so gaps in
    the trajectory do not couple unrelated residuals.

    Parameters
    ----------
    r : ``(K, N, d)`` time-major residuals.
    mask : ``(K, N)`` bool validity.
    contiguous : ``(K,)`` bool — ``True`` where index ``k`` is the sampling
        successor of ``k-1`` (so the lag-1 block applies).
    A_blocks : ``(K, d, d)`` diagonal covariance blocks (= the marginal,
        noise-aware ``C``).
    Lambda : ``(d, d)`` measurement-noise covariance ``Λ``.
    offdiag_coef : ``(K,)`` lag-1 scalar coefficient (overdamped ``-1``;
        kept underdamped ``1/Δt^4``).

    Returns
    -------
    z : ``(K, N, d)`` whitened innovations (zero on masked rows).
    """
    K, N, d = r.shape
    I_d = jnp.eye(d, dtype=r.dtype)

    def body(carry, x):
        w_prev, S_prev, valid_prev = carry
        r_k, A_k, coef_k, contig_k, valid_k = x
        C = coef_k * Lambda  # (d, d) symmetric lag-1 block
        couple = contig_k & valid_prev & valid_k  # (N,)
        S_safe = jnp.where(valid_prev[:, None, None], S_prev, I_d[None])
        # M_k = C S_prev^{-1} = (S_prev^{-1} C)^T  (C, S_prev symmetric)
        M = jnp.swapaxes(jnp.linalg.solve(S_safe, jnp.broadcast_to(C, (N, d, d))), -1, -2)
        M = jnp.where(couple[:, None, None], M, 0.0)
        w_k = r_k - jnp.einsum("nij,nj->ni", M, w_prev)
        S_k = A_k[None] - jnp.einsum("nij,jk->nik", M, C)  # A_k − M C
        z_k = jnp.einsum("nij,nj->ni", _inv_sqrt_psd(S_k), w_k)
        # Carry the innovation covariance forward; reset masked rows so the
        # next step decouples from them.
        w_out = jnp.where(valid_k[:, None], w_k, 0.0)
        S_out = jnp.where(valid_k[:, None, None], S_k, I_d[None])
        z_out = jnp.where(valid_k[:, None], z_k, 0.0)
        return (w_out, S_out, valid_k), z_out

    init = (
        jnp.zeros((N, d), r.dtype),
        jnp.broadcast_to(I_d, (N, d, d)),
        jnp.zeros((N,), bool),
    )
    _, z = lax.scan(body, init, (r, A_blocks, offdiag_coef, contiguous, mask))
    return z


def _coerce_F_value(value, K: int, N: int, d: int) -> jnp.ndarray:
    """Reshape an SF / Basis / callable output to ``(K, N, d)``.

    The OD / UD ``force_inferred`` callable accepts batched inputs
    of shape ``(M, d)`` and returns ``(M, d)``; we always feed it the
    flattened ``(K * N, d)`` form and reshape back.
    """
    arr = jnp.asarray(value)
    if arr.shape == (K * N, d):
        return arr.reshape(K, N, d)
    if arr.shape == (K, N, d):
        return arr
    raise ValueError(f"Force callable returned shape {arr.shape}; expected ({K * N}, {d}) or ({K}, {N}, {d}).")


def _measurement_noise(inferer, d: int) -> jnp.ndarray:
    """PSD measurement-noise covariance Λ (``inferer.Lambda``), or zero.

    The parametric estimator profiles Λ natively and the diffusion
    estimators expose it as ``Lambda``; on clean data Λ ≈ 0 so the
    noise-aware whitening reduces to the thermal case.  The estimate can
    be marginally non-PSD on clean data, which the ``6Λ/Δt⁴`` weighting
    would amplify — so we clamp to the PSD cone.
    """
    Lam = getattr(inferer, "Lambda", None)
    if Lam is None:
        return jnp.zeros((d, d))
    Lam = jnp.asarray(Lam)
    if Lam.shape != (d, d):
        return jnp.zeros((d, d))
    w, U = jnp.linalg.eigh(0.5 * (Lam + Lam.T))
    return (U * jnp.maximum(w, 0.0)) @ U.T


def _backend_tag(inferer) -> str:
    if hasattr(inferer, "metadata") and isinstance(inferer.metadata, dict):
        return str(inferer.metadata.get("force_method", "linear"))
    return "linear"


# Continuous-limit noise factor for the underdamped acceleration residual.
#
# The underdamped diagnostic residual is the symmetric finite-difference
# acceleration  â(t) = (x_{t+1} - 2 x_t + x_{t-1}) / dt²  minus the fitted
# force F(x̂, v̂) — the same quantity the underdamped force estimator fits
# (the symmetric ULI kinematics in SFI.inference.underdamped: _A_sym_uli /
# _V_sym_uli / _X_sym_uli).
#
# For dx = v dt, dv = F dt + sqrt(2D) dW the position is C¹, so the noise
# part of â is the second difference of the integrated velocity noise.
# Writing N_t = sqrt(2D) ∫ B over one sampling cell (B the velocity
# Brownian motion), the adjacent-cell autocovariance integral gives
#
#     Var(x_{t+1} - 2 x_t + x_{t-1}) = (4/3) D dt³ = (2/3) (2D) dt³,
#
# hence  Var(â_noise) = (2/3) (2D) / dt = KAPPA_UD · A / dt   with A = 2D.
#
# This factor is exact for continuously sampled data (the physical case
# for experimental trajectories) and is the limit that finely oversampled
# simulations converge to.  The thermal residual is a clean MA(1) process in
# time (lag-1 ≈ 1/4, lag ≥ 2 ≈ 0), so the builder keeps every second valid
# time index to remove that thermal lag-1.  The leftover measurement-noise
# correlation (a lag-1 block Λ/Δt⁴ in the kept series) is removed by the
# banded innovations whitening (_sequential_innovations).
KAPPA_UD = 2.0 / 3.0


def _process_chunk(
    *,
    F_at: jnp.ndarray,  # (K, N, d)
    r: jnp.ndarray,  # (K, N, d) raw residual
    dt: jnp.ndarray,  # (K,) physical step (pooled into mean_dt)
    mask: jnp.ndarray,  # (K, N) bool
    A: jnp.ndarray,  # (d, d) = 2 D
    A_inv: jnp.ndarray,  # (d, d)
    contiguous: np.ndarray,  # (K,) bool sampling-successor flag
    offdiag_coef: jnp.ndarray,  # (K,) lag-1 coefficient on Λ
    var_scale: jnp.ndarray | None = None,  # (K,) thermal coefficient on A
    Lambda: jnp.ndarray | None = None,  # (d, d) measurement-noise covariance Λ
    noise_scale: jnp.ndarray | None = None,  # (K,) coefficient on Λ
):
    """Whiten residuals, compute Mahalanobis norms and ``F^T A^{-1} F``,
    and return only the masked-valid rows pooled along ``(K, N)``.

    Two whitenings are produced from the same residual covariance
    ``C = var_scale·A + noise_scale·Λ`` (diagonal blocks) with lag-1 block
    ``offdiag_coef·Λ``:

    * **Banded innovations** ``z`` (returned for the moments / normality /
      autocorrelation tests) — the sequential block-Cholesky whitening
      (:func:`_sequential_innovations`) of the tridiagonal covariance.  It
      has unit variance *and no serial correlation*, so it does not trip the
      Ljung--Box test on measurement-noise-correlated residuals.
    * **Marginal Mahalanobis norms** ``sqn = rᵀ C⁻¹ r`` (returned for the
      MSE-consistency / chi-square *bias* check) — the single-residual form,
      kept because the banded innovations would partly difference out a
      slowly-varying force bias that this check is meant to detect.

    The **measurement-noise term** ``noise_scale · Λ`` makes both
    noise-aware: for the underdamped acceleration residual it scales as
    ``6 Λ / Δt⁴`` and otherwise overwhelms the thermal term
    ``(2/3)(2D)/Δt`` even when the force is well recovered.  When ``Λ`` is
    zero (clean data, where the estimator profiles ``Λ ≈ 0``) the
    off-diagonal block vanishes and both reduce to the thermal whitening
    ``(var_scale·A)^{-1/2} r``.  ``var_scale`` defaults to ``dt`` (the
    overdamped Euler scale).
    """
    scale = dt if var_scale is None else var_scale
    Lam = jnp.zeros_like(A) if Lambda is None else jnp.asarray(Lambda)
    if Lambda is not None and noise_scale is not None:
        # Drop a spuriously-small Λ: if its residual contribution is a tiny
        # fraction of the thermal scale the data is *effectively clean*, and
        # folding it in would deflate the chi-square / MSE-consistency bias
        # signal — masking genuine misspecification.  Real underdamped noise is
        # amplified as 6Λ/Δt⁴, far above this floor, so it is never dropped.
        thermal = jnp.mean(scale) * jnp.trace(A)
        noise = jnp.mean(noise_scale) * jnp.trace(Lam)
        Lam = jnp.where(noise < 0.2 * thermal, jnp.zeros_like(Lam), Lam)
    C = scale[:, None, None] * A[None]  # (K, d, d) diagonal blocks
    if Lambda is not None and noise_scale is not None:
        C = C + noise_scale[:, None, None] * Lam[None]

    # Marginal Mahalanobis norms (bias / chi-square channel): rᵀ C⁻¹ r.
    w, U = jnp.linalg.eigh(C)  # (K, d), (K, d, d)
    C_inv = jnp.einsum("kam,km,kbm->kab", U, 1.0 / w, U)
    sqn = jnp.einsum("kni,kij,knj->kn", r, C_inv, r)  # (K, N)
    F2 = jnp.einsum("kjm,mp,kjp->kj", F_at, A_inv, F_at)  # (K, N)

    # Banded innovations (serial-decorrelation channel) used for z / z_full.
    z = _sequential_innovations(
        jnp.asarray(r),
        jnp.asarray(mask),
        jnp.asarray(contiguous),
        C,
        jnp.asarray(Lam),
        jnp.asarray(offdiag_coef),
    )  # (K, N, d)

    m_np = np.asarray(mask)
    z_full = np.asarray(z)  # (K, N, d) time-major, pre-masking
    z_np = z_full[m_np]  # (Kv, d)
    sqn_np = np.asarray(sqn)[m_np]  # (Kv,)
    F2_np = np.asarray(F2)[m_np]  # (Kv,)
    # Per-valid-row dt (broadcast scalar dt over particles)
    K, N = m_np.shape
    dt_kn = np.broadcast_to(np.asarray(dt)[:, None], (K, N))[m_np]
    return z_np, sqn_np, F2_np, dt_kn, z_full, m_np


# --------------------------------------------------------------------- #
# Overdamped builder
# --------------------------------------------------------------------- #
def build_overdamped_residuals(inferer, data=None) -> ResidualBundle:
    """Build standardised Euler--Maruyama residuals for an OD inferer.

    Routes data access through ``TrajectoryDataset.make_batch_producer`` —
    the same low-level streaming layer used by ``SFI.integrate`` — so
    multi-particle, masked, and multi-dataset trajectories are handled
    transparently.

    Works for any overdamped inference path (linear, parametric, nonlinear)
    as long as ``inferer.force_inferred`` is callable and
    ``inferer.A_inv`` is available.
    """
    if not hasattr(inferer, "force_inferred") or inferer.force_inferred is None:
        raise RuntimeError("inferer.force_inferred is missing; run a force-inference method first.")
    if not hasattr(inferer, "A_inv"):
        raise RuntimeError("inferer.A_inv is missing; run compute_diffusion_constant() first.")

    F = inferer.force_inferred
    A = jnp.asarray(getattr(inferer, "A", 2.0 * inferer.diffusion_average))
    A_inv = jnp.asarray(inferer.A_inv)
    d = int(A.shape[0])
    Lambda = _measurement_noise(inferer, d)  # Λ; increment noise = 2 Λ

    require = {"X", "X_plus"}
    z_chunks: list[np.ndarray] = []
    sqnorm_chunks: list[np.ndarray] = []
    F2_chunks: list[np.ndarray] = []
    dt_chunks: list[np.ndarray] = []
    whitened_chunks: list = []
    n_particles_max = 0
    backend = _backend_tag(inferer)

    collection = data if data is not None else inferer.data
    for ds_idx, ds in enumerate(collection.datasets):
        t_idx = ds.valid_indices(require)
        if t_idx.size == 0:
            continue
        d_ds = int(ds.d)
        if d_ds != d:
            raise ValueError(f"Dataset dimension {d_ds} does not match A_inv dimension {d}.")
        n_particles_max = max(n_particles_max, int(ds.N))

        producer = ds.make_batch_producer(
            require,
            include_mask=True,
            include_dt=True,
            force_dt_keys={"dt"},
        )
        row = producer(t_idx)
        X = row["X"]  # (K, N, d)
        Xp = row["X_plus"]  # (K, N, d)
        dt = row["dt"]  # (K,)
        mask = row["mask_out"]  # (K, N) bool
        K, N, _ = X.shape
        extras = ds.build_extras(t_idx, dataset_index=ds_idx)

        # Sampling adjacency: the lag-1 measurement-noise block (−Λ) only
        # couples residuals at consecutive frames; reset across any gap.
        t_arr = np.asarray(t_idx)
        contiguous = np.zeros(K, dtype=bool)
        contiguous[1:] = np.diff(t_arr) == 1
        offdiag_coef = -np.ones(K)  # Cov(r_{k-1}, r_k) = −Λ

        if N > 1:
            # Multiparticle: each frame has N interacting particles — compute
            # the force on each frame separately to preserve particle structure.
            # Per-particle extras (N, ...) reach each particle on the (N,) batch.
            F_frames = [np.asarray(F(np.asarray(X[t]), extras=extras)) for t in range(K)]
            F_at = np.stack(F_frames, axis=0)  # (K, N, d)
        else:
            F_at = _coerce_F_value(F(X.reshape(K * N, d), extras=extras), K, N, d)
        r = (Xp - X) - F_at * dt[:, None, None]
        z_v, sqn_v, F2_v, dt_v, z_full, mask_full = _process_chunk(
            F_at=F_at,
            r=r,
            dt=dt,
            mask=mask,
            A=A,
            A_inv=A_inv,
            contiguous=contiguous,
            offdiag_coef=jnp.asarray(offdiag_coef),
            var_scale=jnp.asarray(dt),
            Lambda=Lambda,
            noise_scale=2.0 * jnp.ones_like(jnp.asarray(dt)),
        )
        z_chunks.append(z_v)
        sqnorm_chunks.append(sqn_v)
        F2_chunks.append(F2_v)
        dt_chunks.append(dt_v)
        whitened_chunks.append((z_full, mask_full))

    return _assemble_bundle(
        z_chunks,
        sqnorm_chunks,
        F2_chunks,
        dt_chunks,
        whitened_chunks,
        d=d,
        regime="OD",
        backend=backend,
        n_particles=n_particles_max,
    )


# --------------------------------------------------------------------- #
# Underdamped builder
# --------------------------------------------------------------------- #
def build_underdamped_residuals(inferer, data=None) -> ResidualBundle:
    """Build standardised innovations for a UD inferer from the symmetric
    acceleration residual.

    Uses the symmetric ULI kinematics that the underdamped force estimator
    itself fits (see ``SFI.inference.underdamped``):

    .. math::

       \\hat x = \\tfrac13(X_{t-1}+X_t+X_{t+1}), \\quad
       \\hat v = \\frac{X_{t+1}-X_{t-1}}{2\\Delta t}, \\quad
       \\hat a = \\frac{X_{t+1}-2X_t+X_{t-1}}{\\Delta t^2},

    and forms the residual :math:`r_t = \\hat a - F(\\hat x, \\hat v)`.
    Its thermal noise covariance is :math:`\\tfrac23 A/\\Delta t` (see
    :data:`KAPPA_UD`); with measurement noise the diagonal block gains
    :math:`6\\Sigma_\\eta/\\Delta t^4`.  The thermal residual is MA(1), so
    only every second valid index is kept (removing the thermal lag-1);
    the residual measurement-noise correlation (lag-1 block
    :math:`\\Sigma_\\eta/\\Delta t^4`) is removed by the banded innovations
    whitening, leaving a serially independent stream.

    Like :func:`build_overdamped_residuals`, all data access uses
    ``TrajectoryDataset.make_batch_producer`` so masking and
    multi-dataset / multi-particle pooling are handled by the same
    streaming layer that powers ``SFI.integrate``.
    """
    if not hasattr(inferer, "force_inferred") or inferer.force_inferred is None:
        raise RuntimeError("inferer.force_inferred is missing; run a force-inference method first.")
    if not hasattr(inferer, "A_inv"):
        raise RuntimeError("inferer.A_inv is missing; run compute_diffusion_constant() first.")

    F = inferer.force_inferred
    A = jnp.asarray(getattr(inferer, "A", 2.0 * inferer.diffusion_average))
    A_inv = jnp.asarray(inferer.A_inv)
    d = int(A.shape[0])
    Lambda = _measurement_noise(inferer, d)  # Λ; acceleration noise = 6 Λ / Δt⁴

    # Symmetric 3-point stencil: X_{t-1}, X_t, X_{t+1}.
    require = {"X_minus", "X", "X_plus"}
    z_chunks: list[np.ndarray] = []
    sqnorm_chunks: list[np.ndarray] = []
    F2_chunks: list[np.ndarray] = []
    dt_chunks: list[np.ndarray] = []
    whitened_chunks: list = []
    n_particles_max = 0
    backend = _backend_tag(inferer)

    collection = data if data is not None else inferer.data
    for ds_idx, ds in enumerate(collection.datasets):
        t_idx = ds.valid_indices(require)
        # Adjacent acceleration residuals share two of three positions, so the
        # *thermal* series is MA(1).  Keeping every second valid index removes
        # that thermal lag-1 correlation (lag ≥ 2 thermal ≈ 0).  Under
        # measurement noise the kept series still carries a lag-1 block
        # Λ/Δt⁴ (the original lag-2), which the banded whitening below
        # decorrelates; on clean data that block vanishes.
        t_idx = t_idx[::2]
        if t_idx.size == 0:
            continue
        d_ds = int(ds.d)
        if d_ds != d:
            raise ValueError(f"Dataset dimension {d_ds} does not match A_inv dimension {d}.")
        n_particles_max = max(n_particles_max, int(ds.N))

        producer = ds.make_batch_producer(
            require,
            include_mask=True,
            include_dt=True,
            force_dt_keys={"dt"},
        )
        row = producer(t_idx)
        Xm = row["X_minus"]  # (K, N, d) at t-1
        X = row["X"]  # (K, N, d) at t
        Xp = row["X_plus"]  # (K, N, d) at t+1
        dt0 = row["dt"]  # (K,)  step t -> t+1
        mask = row["mask_out"]  # (K, N) bool — already AND'd
        K, N, _ = X.shape

        # Strided sampling adjacency: the lag-1 block (Λ/Δt⁴) only couples
        # kept residuals two original frames apart; reset across any gap.
        t_arr = np.asarray(t_idx)
        contiguous = np.zeros(K, dtype=bool)
        contiguous[1:] = np.diff(t_arr) == 2

        # Symmetric ULI kinematics (match SFI.inference.underdamped):
        #   x̂ = (X₋ + X + X₊)/3, v̂ = (X₊ − X₋)/(2 dt), â = (X₊ − 2X + X₋)/dt²
        dt_b = dt0[:, None, None]
        x_hat = (Xm + X + Xp) / 3.0
        v_hat = (Xp - Xm) / (2.0 * dt_b)
        a_hat = (Xp - 2.0 * X + Xm) / (dt_b * dt_b)

        extras = ds.build_extras(t_idx, dataset_index=ds_idx)
        if N > 1:
            # Preserve the particle axis so bases with per-particle extras
            # (e.g. per-agent home ranges) see each particle's own value.
            F_at = _coerce_F_value(F(x_hat, v=v_hat, extras=extras), K, N, d)
        else:
            F_at = _coerce_F_value(
                F(x_hat.reshape(K * N, d), v=v_hat.reshape(K * N, d), extras=extras),
                K,
                N,
                d,
            )
        # Residual in acceleration units; thermal Cov = (2/3) A / dt
        # (KAPPA_UD), measurement-noise Cov = 6 Λ / dt⁴ (second difference of
        # i.i.d. localisation errors: variance (1+4+1) Λ / dt⁴).
        r = a_hat - F_at
        var_scale = KAPPA_UD / dt0  # (K,)
        noise_scale = 6.0 / (dt0 ** 4)  # (K,)
        offdiag_coef = 1.0 / (dt0 ** 4)  # Cov(r_{k-1}, r_k) = Λ/Δt⁴
        z_v, sqn_v, F2_v, dt_pool, z_full, mask_full = _process_chunk(
            F_at=F_at,
            r=r,
            dt=dt0,
            mask=mask,
            A=A,
            A_inv=A_inv,
            contiguous=contiguous,
            offdiag_coef=offdiag_coef,
            var_scale=var_scale,
            Lambda=Lambda,
            noise_scale=noise_scale,
        )
        z_chunks.append(z_v)
        sqnorm_chunks.append(sqn_v)
        F2_chunks.append(F2_v)
        dt_chunks.append(dt_pool)
        whitened_chunks.append((z_full, mask_full))

    return _assemble_bundle(
        z_chunks,
        sqnorm_chunks,
        F2_chunks,
        dt_chunks,
        whitened_chunks,
        d=d,
        regime="UD",
        backend=backend,
        n_particles=n_particles_max,
        nmse_excess_factor=KAPPA_UD,
    )


def _assemble_bundle(
    z_chunks,
    sqnorm_chunks,
    F2_chunks,
    dt_chunks,
    whitened_chunks,
    *,
    d: int,
    regime: str,
    backend: str,
    n_particles: int,
    nmse_excess_factor: float = 1.0,
) -> ResidualBundle:
    if z_chunks:
        z_components = np.concatenate(z_chunks, axis=0)
        sqn_pooled = np.concatenate(sqnorm_chunks, axis=0)
        F2_pooled = np.concatenate(F2_chunks, axis=0)
        dt_pooled = np.concatenate(dt_chunks, axis=0)
        mean_dt = float(np.mean(dt_pooled)) if dt_pooled.size else float("nan")
    else:
        z_components = np.zeros((0, d))
        sqn_pooled = np.zeros((0,))
        F2_pooled = np.zeros((0,))
        mean_dt = float("nan")
    z_pooled = z_components.reshape(-1)
    return ResidualBundle(
        z=z_pooled,
        z_components=z_components,
        z_squared_norms=sqn_pooled,
        force_quadratic_form=F2_pooled,
        mean_dt=mean_dt,
        n_obs=int(z_components.shape[0]),
        d=d,
        regime=regime,
        backend=backend,
        n_particles=n_particles,
        nmse_excess_factor=nmse_excess_factor,
        whitened=whitened_chunks,
    )


# --------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------- #
def build_residuals(inferer, data=None) -> ResidualBundle:
    """Dispatch to the OD / UD residual builder based on the engine class.

    ``data`` (optional) evaluates the residuals on an independent
    :class:`~SFI.trajectory.TrajectoryCollection` instead of the
    training data — the held-out path used by ``holdout_score``.
    """
    cls = type(inferer).__name__
    if "Underdamped" in cls:
        return build_underdamped_residuals(inferer, data=data)
    if "Overdamped" in cls:
        return build_overdamped_residuals(inferer, data=data)
    # Fallback: try by attribute
    if hasattr(inferer, "_force_inference_underdamped"):
        return build_underdamped_residuals(inferer, data=data)
    return build_overdamped_residuals(inferer, data=data)
