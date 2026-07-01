# SFI/inference/parametric_core/objective_ud.py
"""
Integrate-engine programs for the underdamped parametric core (bandwidth 2).

Mirror of :mod:`objective` (overdamped): each program returns one value
**per (window, particle)** ``(K, N)`` and runs with
``reduce_over_particles=True`` (single particle = N=1).  Per particle the
3-point shooting residual + α-propagators come from
:func:`flow_multi.ud_multi_step_residuals`; the pentadiagonal covariance
(:func:`covariance.build_ud_blocks`) and the bandwidth-generic precision
kernels are applied per particle.

The force is a velocity-dependent PSF (``needs_v=True``); the unobserved
velocity is resolved internally by shooting.  Interacting underdamped
multiparticle (multi-particle phase-space shooting) is a follow-up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .covariance import build_ud_blocks
from .flow_multi import multi_ud_instrument, ud_multi_step_residuals, ud_multi_step_residuals_with_psi
from .objective import unpack_gram  # shared packing
from .precision import (
    center_conditional_nll_contribution,
    center_gram_contribution,
    center_loss_contribution,
)

__all__ = ["UDLossProgram", "UDCondNLLProgram", "UDGramProgram", "UDDiffNLLProgram", "unpack_gram"]

_BANDWIDTH = 2


def _ud_force_struct(F_psf, theta):
    return F_psf.unflatten_params(theta)


class _UDProgramBase:
    def __init__(self, F_psf, *, dt, n_substeps, jitter=1e-7, jitter_chol=1e-10, integrator="rk4"):
        self._F = F_psf
        self._dt = dt
        self._n_sub = n_substeps
        self._jitter = jitter
        self._jitter_chol = jitter_chol
        self._integrator = integrator
        self._n_params = int(F_psf.template.size)

    def require(self):
        req = {self._stream_key}
        if getattr(self._F, "required_extras", None):
            req.add("extras")
        return req

    def _resid(self, theta, Y_w, extras):
        return ud_multi_step_residuals(
            self._F, self._F.unflatten_params(theta), Y_w, extras,
            self._dt, self._n_sub, self._integrator)


class UDLossProgram(_UDProgramBase):
    """Per-(window, particle) frozen-precision quadratic ``½ rᵀ P r`` (bandwidth 2)."""

    def __init__(self, F_psf, *, w_res=7, **kw):
        super().__init__(F_psf, **kw)
        self._center = w_res // 2
        self._stream_key = f"X_window:{w_res + 2}"

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta_live, theta_frozen, D, Lambda = params
        Y_w = streams[self._stream_key]
        extras = streams.get("extras")
        c = self._center

        def _window(yw):
            r, _, _, _, _ = self._resid(theta_live, yw, extras)
            _, ap, a0, am, _ = self._resid(jax.lax.stop_gradient(theta_frozen), yw, extras)
            ap, a0, am = (jax.lax.stop_gradient(x) for x in (ap, a0, am))

            def _p(rp, app, a0p, amp):
                blocks = build_ud_blocks(app, a0p, amp, D, Lambda, self._dt, jitter=self._jitter)
                return center_loss_contribution(blocks.A, blocks.offdiag, rp, c, self._jitter_chol, _BANDWIDTH)

            return jax.vmap(_p, in_axes=(1, 1, 1, 1))(r, ap, a0, am)

        return jax.vmap(_window)(Y_w)


class UDCondNLLProgram(_UDProgramBase):
    """Per-(window, particle) conditional NLL — profiling / joint."""

    def __init__(self, F_psf, *, n_cond=4, **kw):
        super().__init__(F_psf, **kw)
        self._n_cond = int(n_cond)
        self._stream_key = f"X_window:{n_cond + 3}"

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta, D, Lambda = params
        Y_w = streams[self._stream_key]
        extras = streams.get("extras")
        n_cond = self._n_cond

        def _window(yw):
            r, ap, a0, am, _ = self._resid(theta, yw, extras)

            def _p(rp, app, a0p, amp):
                blocks = build_ud_blocks(app, a0p, amp, D, Lambda, self._dt, jitter=self._jitter)
                return center_conditional_nll_contribution(
                    blocks.A, blocks.offdiag, rp, n_cond, n_cond, self._jitter_chol)

            return jax.vmap(_p, in_axes=(1, 1, 1, 1))(r, ap, a0, am)

        return jax.vmap(_window)(Y_w)


class UDDiffNLLProgram(_UDProgramBase):
    """Conditional NLL for state/velocity-dependent diffusion ``D(x, v; θ_D)`` (force fixed)."""

    def __init__(self, F_psf, theta_F, D_psf, *, n_cond=4, **kw):
        super().__init__(F_psf, **kw)
        self._theta_F = theta_F
        self._D_psf = D_psf
        self._n_cond = int(n_cond)
        self._stream_key = f"X_window:{n_cond + 3}"

    def estimate_bytes_per_sample(self, sample_row):
        return int(sample_row[self._stream_key].shape[-2]) * 8

    def batch_call(self, *, params, **streams):
        theta_D, Lambda = params
        Y_w = streams[self._stream_key]
        extras = streams.get("extras")
        n_cond = self._n_cond
        D_struct = self._D_psf.unflatten_params(theta_D)

        def _D(x, v):
            return self._D_psf(x[None], v=v[None], params=D_struct, extras=extras)[0]

        def _window(yw):                            # (W, N, d)
            r, ap, a0, am, vhat = self._resid(self._theta_F, yw, extras)
            Y_base = yw[1:-1]                        # (W-2, N, d)

            def _p(rp, app, a0p, amp, vhatp, Ybp):
                Dk = jax.vmap(_D)(Ybp, vhatp)        # (W-2, d, d)
                blocks = build_ud_blocks(app, a0p, amp, Dk, Lambda, self._dt, jitter=self._jitter)
                return center_conditional_nll_contribution(
                    blocks.A, blocks.offdiag, rp, n_cond, n_cond, self._jitter_chol)

            return jax.vmap(_p, in_axes=(1, 1, 1, 1, 1, 1))(r, ap, a0, am, vhat, Y_base)

        return jax.vmap(_window)(Y_w)


class UDGramProgram(_UDProgramBase):
    """Per-(window, particle) GN Gram/score/NLL packed as ``(G, f, nll)``.

    ``w`` is the EIV instrument blend weight (0 = symmetric MLE).  The
    underdamped η-clean instrument is wired in below when ``w > 0``.
    """

    def __init__(self, F_psf, *, w_res=7, w=0.0, **kw):
        super().__init__(F_psf, **kw)
        self._center = w_res // 2
        self._w = float(w)
        # The η-clean instrument (w>0) reserves 2 extra front positions for its
        # base (outside the residual block the pentadiagonal precision couples
        # in), so widen the window to keep the block — and hence the precision
        # context / variance — unchanged.  The w=0 (MLE) pass uses the narrow window.
        extra = 2 if self._w > 0.0 else 0
        self._stream_key = f"X_window:{w_res + 2 + extra}"

    def estimate_bytes_per_sample(self, sample_row):
        n = self._n_params
        shape = sample_row[self._stream_key].shape          # (W, N, d)
        N = int(shape[-2])
        out = N * (2 * n * n + n + 1) * 8
        if getattr(self._F, "particles_input", False):
            # Interacting working set (see ODGramProgram): per-stage dF/dθ
            # transient over the pair graph, two phase flows per residual.
            W = int(shape[-3])
            out += W * 8 * self._n_sub * N * N * max(n, 8) * 8
        return int(out)

    def batch_call(self, *, params, **streams):
        theta_flat, D, Lambda = params
        Y_w = streams[self._stream_key]
        extras = streams.get("extras")
        w = self._w
        interacting = getattr(self._F, "particles_input", False)
        # When the instrument is active (w>0) the window is [2 front positions |
        # residual block]: the instrument base pair (the front positions) is then
        # OUTSIDE the whole block's measurement-noise support, so it cannot overlap
        # any residual the pentadiagonal precision couples into the moment (the
        # legacy first-row IV), and is paired against the first block residual.
        # The w=0 (MLE) pass uses the narrow center window.
        skip = w > 0.0
        # center index: first block residual when the instrument is active; middle
        # residual otherwise (derived from the actual window width so it is correct
        # for both the narrow MLE window and the wider instrument one).
        W_pos = Y_w.shape[1]
        c = 0 if skip else (W_pos - 3) // 2

        def _window(yw):
            yw_res = yw[2:] if skip else yw         # block positions (skip the front pair)

            if interacting:
                # Frozen-background ψ_right (per-particle phase θ-recursion) —
                # see ODGramProgram; the full-flow jacfwd is the interacting
                # memory blow-up.
                r, ap, a0, am, _, psi = ud_multi_step_residuals_with_psi(
                    self._F, theta_flat, yw_res, extras,
                    self._dt, self._n_sub, self._integrator)
            else:
                def resid(th):
                    r, *_ = self._resid(th, yw_res, extras)
                    return r                        # (R, N, d) = ψ_right base
                r = resid(theta_flat)
                psi = jax.jacfwd(resid)(theta_flat)  # (R, N, d, n_params) = ψ_right
                _, ap, a0, am, _ = self._resid(theta_flat, yw_res, extras)

            if w > 0.0:
                # Instrument base pair: the reserved front pair (yw[0], yw[1]),
                # η-clean of the whole block.  Flow-propagated to the c-th residual;
                # interacting models use the same N-body flow as the residual.
                psi_inst = multi_ud_instrument(
                    self._F, theta_flat, yw[0], yw[1], extras,
                    self._dt, self._n_sub, self._integrator)  # (N, d, n_params)

            def _p(rp, psip, app, a0p, amp, inst):
                blocks = build_ud_blocks(app, a0p, amp, D, Lambda, self._dt, jitter=self._jitter)
                psi_left = psip.at[c].set((1.0 - w) * psip[c] + w * inst)
                G, f, H, nll = center_gram_contribution(
                    blocks.A, blocks.offdiag, rp, psi_left, c, self._jitter_chol, psi_right_w=psip)
                return jnp.concatenate([G.ravel(), f, H.ravel(), nll[None]])

            def _p_sym(rp, psip, app, a0p, amp):
                blocks = build_ud_blocks(app, a0p, amp, D, Lambda, self._dt, jitter=self._jitter)
                G, f, H, nll = center_gram_contribution(blocks.A, blocks.offdiag, rp, psip, c, self._jitter_chol)
                return jnp.concatenate([G.ravel(), f, H.ravel(), nll[None]])

            if w > 0.0:
                return jax.vmap(_p, in_axes=(1, 1, 1, 1, 1, 0))(r, psi, ap, a0, am, psi_inst)
            return jax.vmap(_p_sym, in_axes=(1, 1, 1, 1, 1))(r, psi, ap, a0, am)

        return jax.vmap(_window)(Y_w)
