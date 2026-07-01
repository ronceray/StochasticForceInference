# SFI/inference/parametric_core/objective.py
"""
Integrate-engine programs for the overdamped parametric core.

Each program declares the sliding-window stream ``X_window:{W}`` and
returns one value **per (window, particle)** ``(K, N)``; the engine owns
chunking, masking (``mask_out``), JIT, and the reduction over both window
centers and particles (``reduce_over_particles=True``).  Single particle
is just ``N=1`` — there is no separate multiparticle path (cf.
``infer_force_linear``).

Per particle the residuals/Jacobians come from :func:`flow_multi.
multi_step_residuals` (per-particle for non-interacting forces, full-flow
+ frozen-background Jacobian for interacting ones); the covariance and
precision kernels are applied per particle.

* :class:`ODLossProgram`    — frozen-precision quadratic (minimised over θ).
* :class:`ODCondNLLProgram` — windowed conditional NLL (profiling / joint).
* :class:`ODDiffNLLProgram` — conditional NLL for state-dependent D(x; θ_D).
* :class:`ODGramProgram`    — GN Gram/score/NLL for the parameter covariance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .covariance import build_od_blocks
from .flow_multi import multi_od_instrument, multi_step_residuals, multi_step_residuals_with_psi
from .precision import (
    center_conditional_nll_contribution,
    center_gram_contribution,
    center_loss_contribution,
)

__all__ = ["ODLossProgram", "ODGramProgram", "ODCondNLLProgram", "ODDiffNLLProgram", "unpack_gram"]

_BANDWIDTH = 1


def unpack_gram(packed, n_params):
    """Unpack a flat ``(G, f, H, nll)`` vector from :class:`ODGramProgram`.

    ``H = ψ_leftᵀPψ_left`` is the estimating-function variance (the sandwich
    meat for the IV path; ``H = G`` on the symmetric path).
    """
    n2 = n_params * n_params
    G = packed[:n2].reshape(n_params, n_params)
    f = packed[n2:n2 + n_params]
    H = packed[n2 + n_params:2 * n2 + n_params].reshape(n_params, n_params)
    return G, f, H, packed[-1]


def _center_index(W_res):
    return (W_res - 1) // 2


def _per_particle_loss(r, J, D, Lambda, dt, jitter, jitter_chol):
    """``½ rᵀ P r`` (center decomposition) per particle → ``(N,)``."""
    c = _center_index(r.shape[0])

    def _p(rp, Jp):
        blocks = build_od_blocks(Jp, D, Lambda, dt, jitter=jitter)
        return center_loss_contribution(blocks.A, blocks.offdiag, rp, c, jitter_chol, _BANDWIDTH)

    return jax.vmap(_p, in_axes=(1, 1))(r, J)


def _per_particle_condnll(r, J, D, Lambda, dt, jitter, jitter_chol, n_cond):
    """Windowed conditional NLL per particle (last residual | n_cond past) → ``(N,)``."""
    def _p(rp, Jp):
        blocks = build_od_blocks(Jp, D, Lambda, dt, jitter=jitter)
        return center_conditional_nll_contribution(blocks.A, blocks.offdiag, rp, n_cond, n_cond, jitter_chol)

    return jax.vmap(_p, in_axes=(1, 1))(r, J)


class _ODProgramBase:
    def __init__(self, F_psf, *, dt, n_substeps, jitter=1e-7, jitter_chol=1e-10,
                 extra_radius=1, integrator="rk4"):
        self._F = F_psf
        self._dt = dt
        self._n_sub = n_substeps
        self._jitter = jitter
        self._jitter_chol = jitter_chol
        self._integrator = integrator
        self._n_params = int(F_psf.template.size)
        W_pos = 2 * _BANDWIDTH + 2 * extra_radius + 2
        self._stream_key = f"X_window:{W_pos}"

    def require(self):
        req = {self._stream_key}
        if getattr(self._F, "required_extras", None):
            req.add("extras")
        return req

    def _resid(self, theta, X_w, extras):
        return multi_step_residuals(
            self._F, self._F.unflatten_params(theta), X_w, extras,
            self._dt, self._n_sub, self._integrator)


class ODLossProgram(_ODProgramBase):
    """Per-(window, particle) frozen-precision quadratic ``½ rᵀ P r``."""

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta_live, theta_frozen, D, Lambda = params
        X_w = streams[self._stream_key]            # (K, W, N, d)
        extras = streams.get("extras")

        def _window(xw):                            # (W, N, d)
            r, _ = self._resid(theta_live, xw, extras)
            _, J = self._resid(jax.lax.stop_gradient(theta_frozen), xw, extras)
            J = jax.lax.stop_gradient(J)
            return _per_particle_loss(r, J, D, Lambda, self._dt, self._jitter, self._jitter_chol)

        return jax.vmap(_window)(X_w)               # (K, N)


class ODCondNLLProgram(_ODProgramBase):
    """Per-(window, particle) conditional NLL with (θ, D, Λ) live."""

    def __init__(self, F_psf, *, n_cond=3, **kw):
        super().__init__(F_psf, **kw)
        self._n_cond = int(n_cond)
        self._stream_key = f"X_window:{n_cond + 2}"

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta, D, Lambda = params
        X_w = streams[self._stream_key]
        extras = streams.get("extras")

        def _window(xw):
            r, J = self._resid(theta, xw, extras)
            return _per_particle_condnll(
                r, J, D, Lambda, self._dt, self._jitter, self._jitter_chol, self._n_cond)

        return jax.vmap(_window)(X_w)


class ODDiffNLLProgram(_ODProgramBase):
    """Conditional NLL for state-dependent diffusion ``D(x; θ_D)`` (force fixed)."""

    def __init__(self, F_psf, theta_F, D_psf, *, n_cond=3, **kw):
        super().__init__(F_psf, **kw)
        self._theta_F = theta_F
        self._D_psf = D_psf
        self._n_cond = int(n_cond)
        self._stream_key = f"X_window:{n_cond + 2}"

    def require(self):
        req = {self._stream_key}
        if getattr(self._F, "required_extras", None) or getattr(self._D_psf, "required_extras", None):
            req.add("extras")
        return req

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta_D, Lambda = params
        X_w = streams[self._stream_key]
        extras = streams.get("extras")
        D_struct = self._D_psf.unflatten_params(theta_D)

        def _D(x):
            return self._D_psf(x[None], params=D_struct, extras=extras)[0]

        def _window(xw):                            # (W, N, d)
            r, J = self._resid(self._theta_F, xw, extras)   # (W-1,N,d),(W-1,N,d,d)
            n_cond = self._n_cond

            def _p(rp, Jp, Xp):                     # (W-1,d),(W-1,d,d),(W,d)
                Dk = jax.vmap(_D)(Xp[:-1])          # (W-1, d, d) state-dependent
                blocks = build_od_blocks(Jp, Dk, Lambda, self._dt, jitter=self._jitter)
                return center_conditional_nll_contribution(
                    blocks.A, blocks.offdiag, rp, n_cond, n_cond, self._jitter_chol)

            return jax.vmap(_p, in_axes=(1, 1, 1))(r, J, xw)  # (N,)

        return jax.vmap(_window)(X_w)


class ODGramProgram(_ODProgramBase):
    """Per-(window, particle) GN Gram/score/NLL packed as ``(G, f, nll)``.

    The estimating equation is ``⟨ψ_left P r⟩ = 0`` with regressor
    ``ψ_right = ∂r/∂θ``.  With ``w = 0`` (default) the left factor is the
    regressor — the symmetric MLE Gram, byte-identical to before.  With
    ``w > 0`` the *center* left factor is blended toward an η-clean instrument,

        ψ_left,c = (1 − w) ψ_right,c + w ψ_inst,c,

    which removes the measurement-noise errors-in-variables bias (``w → 1`` is
    the pure instrument).  ``w`` is a host-side scalar set from the EIV ratio.
    """

    def __init__(self, F_psf, *, w=0.0, extra_radius=1, **kw):
        super().__init__(F_psf, extra_radius=extra_radius, **kw)
        self._w = float(w)
        if self._w > 0.0:
            # The η-clean instrument reserves 1 front position OUTSIDE the
            # residual block the tridiagonal precision couples in (paired against
            # the first block residual); widen the window to hold it.  The w=0
            # (MLE) pass keeps the efficient center window.
            self._stream_key = f"X_window:{2 * _BANDWIDTH + 2 * extra_radius + 2 + 1}"

    def estimate_bytes_per_sample(self, sample_row):
        n = self._n_params
        shape = sample_row[self._stream_key].shape          # (W, N, d)
        N = int(shape[-2])
        out = N * (2 * n * n + n + 1) * 8
        if getattr(self._F, "particles_input", False):
            # Interacting working set: per-stage dF/dθ jacfwd transient over
            # the pair graph (~N² pair terms × n tangents), 4 RK4 stages ×
            # n_substeps, per window residual.  Without this the output-only
            # estimate lets the whole trajectory into one chunk and the
            # working set blows up (FINDINGS #7).
            W = int(shape[-3])
            out += W * 4 * self._n_sub * N * N * max(n, 8) * 8
        return int(out)

    def batch_call(self, *, params, **streams):
        theta_flat, D, Lambda = params
        X_w = streams[self._stream_key]
        extras = streams.get("extras")
        w = self._w
        interacting = getattr(self._F, "particles_input", False)

        # When the instrument is active (w>0) the base is the reserved front
        # position, OUTSIDE the tridiagonal-coupled block (η-clean of the whole
        # block), paired against the first block residual.
        skip = w > 0.0

        def _window(xw):
            xw_res = xw[1:] if skip else xw         # block (skip the front position)

            if interacting:
                # Frozen-background ψ_right (same-particle θ-recursion): the
                # full-flow jacfwd holds n_params tangents alive across the
                # whole window graph — O(N²·n_params) per window — and is the
                # interacting-path memory blow-up (FINDINGS #7).
                r, J, psi = multi_step_residuals_with_psi(
                    self._F, theta_flat, xw_res, extras,
                    self._dt, self._n_sub, self._integrator)
            else:
                def resid(th):
                    r, _ = self._resid(th, xw_res, extras)
                    return r                        # (R, N, d)

                r = resid(theta_flat)
                psi = jax.jacfwd(resid)(theta_flat)  # (R, N, d, n_params) = ψ_right
                _, J = self._resid(theta_flat, xw_res, extras)
            c = 0 if skip else _center_index(r.shape[0])

            if w > 0.0:
                # instrument base: the reserved front X[0], η-clean of the whole
                # block.  Flow-propagated trapeze to the first block residual;
                # interacting models use the same N-body flow as the residual.
                psi_inst = multi_od_instrument(
                    self._F, theta_flat, xw[0], extras,
                    self._dt, self._n_sub, self._integrator)  # (N, d, n_params)

            def _p(rp, psip, Jp, inst):
                blocks = build_od_blocks(Jp, D, Lambda, self._dt, jitter=self._jitter)
                psi_left = psip.at[c].set((1.0 - w) * psip[c] + w * inst)
                G, f, H, nll = center_gram_contribution(
                    blocks.A, blocks.offdiag, rp, psi_left, c, self._jitter_chol, psi_right_w=psip)
                return jnp.concatenate([G.ravel(), f, H.ravel(), nll[None]])

            def _p_sym(rp, psip, Jp):
                blocks = build_od_blocks(Jp, D, Lambda, self._dt, jitter=self._jitter)
                G, f, H, nll = center_gram_contribution(blocks.A, blocks.offdiag, rp, psip, c, self._jitter_chol)
                return jnp.concatenate([G.ravel(), f, H.ravel(), nll[None]])

            if w > 0.0:
                return jax.vmap(_p, in_axes=(1, 1, 1, 0))(r, psi, J, psi_inst)
            return jax.vmap(_p_sym, in_axes=(1, 1, 1))(r, psi, J)  # (N, n_params²+n_params+1)

        return jax.vmap(_window)(X_w)               # (K, N, packed)
