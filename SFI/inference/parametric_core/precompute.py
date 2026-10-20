# SFI/inference/parametric_core/precompute.py
"""
Per-point flow tensors on contiguous position blocks (Stage-2 Phase A).

The exact banded core (:mod:`banded`) consumes per-point quantities —
each computed ONCE per trajectory point:

* overdamped: residuals ``r_k = X_{k+1} − X_k − Φ(X_k;θ)``, flow
  Jacobians ``J_k``, regressor ``ψ_k = ∂r_k/∂θ`` and (optionally) the
  η-clean instrument stream ``ψ_inst,k`` (base ``X_{k−1}``, lag-1);
* underdamped: 3-point shooting residuals ``r_n``, the α-propagators,
  shooting velocities ``v̂_n``, ``ψ_n`` from the phase θ-recursion, and
  the instrument stream (base pair ``(Y_{n−3}, Y_{n−2})``,
  ``n_predict = 2``).

All heavy evaluation goes through the Stage-1 θ-recursions
(:mod:`flow_multi`), so the basis is evaluated with (2+d)-wide stages —
never with n_params forward tangents.

Alignment conventions (segment of ``P`` consecutive positions):

* OD: arrays have length ``P−1``; residual ``k`` sits between positions
  ``k`` and ``k+1``.  ``psi_inst[k]`` uses base ``X_{k−1}`` and is
  invalid at ``k = 0`` (``inst_valid``).
* UD: arrays have length ``P−2``; residual ``n`` is centred on interior
  position ``n+1`` (needs ``Y_n, Y_{n+1}, Y_{n+2}``).  ``psi_inst[n]``
  uses the pair ``(Y_{n−2}, Y_{n−1})`` (positions ``n−2, n−1`` of the
  block), invalid for ``n < 2``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .flow import flow_displacement
from .frame_extras import frame_window, split_frames, with_frame
from .flow_multi import (
    _jpb,
    multi_od_instrument,
    multi_ud_instrument,
    ud_multi_step_residuals_with_psi,
)

__all__ = ["od_point_tensors", "ud_point_tensors"]


def od_point_tensors(F_psf, theta_flat, X_block, extras, dt, n_sub,
                     integrator="rk4", *, with_psi=True, with_instrument=False,
                     D_lyap=None, conv_lambda=None, with_stages=False):
    r"""Per-point OD tensors on a block of ``P`` consecutive positions.

    Returns a dict with ``r (P−1, N, d)``, ``J (P−1, N, d, d)`` and,
    on request, ``psi (P−1, N, d, n)``, ``psi_inst (P−1, N, d, n)`` +
    ``inst_valid (P−1,)`` (front entry False — no η-clean base).

    ``D_lyap``: also return the Lyapunov-exact process covariance
    ``Q (P−1, N, d, d)`` (replaces the trapezoid in the blocks).

    ``conv_lambda``: measurement-noise covariance ``Λ̂ (d, d)`` for the
    convexity correction — the residual mean bias of the flow evaluated
    at a noisy argument, ``E[Φ(x+η)] − Φ(x) = ½∇²Φ:Λ + O(Λ²)``, removed
    by central second differences of the primal flow at ``±√λ_i ê_i``
    (eigenpairs of Λ̂).  Only applied for non-interacting models: for
    interacting ones a simultaneous frame shift would overcount
    cross-particle second derivatives (the per-particle noises are
    independent), so the correction is skipped there.

    ``dt`` may be a scalar or a ``(P−1,)`` vector of per-interval steps
    (``dt[k]`` spans ``k → k+1``).

    ``with_stages``: additionally return the per-interval stage Jacobians
    ``stages (P−1, n_sub, S, N, d, d)`` — the fixed-θ cache from which
    the Lyapunov ``Q`` is recomputable for any ``D`` via
    :func:`jacobians.lyapunov_from_stages` (mutually exclusive with
    ``D_lyap``: the cache defers the ``D`` choice).

    Per-frame extras (:class:`frame_extras.FrameSeries` values, leading
    axis ``P``) are read at each interval's left frame: residual ``k``
    flows with frame ``k``'s values.
    """
    per_step = jnp.ndim(dt) > 0
    static, frames = split_frames(extras)
    in_axes = (0, 0, 0 if per_step else None, 0)
    fr_res = frame_window(frames, 0, -1)               # interval k ← frame k

    def jpb_at(fk):
        return _jpb(F_psf, theta_flat, with_frame(static, fk), dt, n_sub,
                    integrator, D_lyap=D_lyap, return_stages=with_stages)

    if with_stages:
        if D_lyap is not None:
            raise ValueError("with_stages caches the D-independent stage "
                             "Jacobians; do not combine with D_lyap")

        def _step_s(X_start, X_end, dt_k, fk):
            J, Phi, B, st = jpb_at(fk)(X_start, dt_k)
            return X_end - X_start - Phi, J, -B, st

        r, J, psi, stages = jax.vmap(_step_s, in_axes=in_axes)(
            X_block[:-1], X_block[1:], dt, fr_res)
        out = {"r": r, "J": J, "stages": stages}
    elif D_lyap is not None:
        def _step(X_start, X_end, dt_k, fk):
            J, Phi, B, Q = jpb_at(fk)(X_start, dt_k)
            return X_end - X_start - Phi, J, -B, Q

        r, J, psi, Q = jax.vmap(_step, in_axes=in_axes)(
            X_block[:-1], X_block[1:], dt, fr_res)
        out = {"r": r, "J": J, "Q": Q}
    else:
        def _step(X_start, X_end, dt_k, fk):
            J, Phi, B = jpb_at(fk)(X_start, dt_k)
            return X_end - X_start - Phi, J, -B

        r, J, psi = jax.vmap(_step, in_axes=in_axes)(
            X_block[:-1], X_block[1:], dt, fr_res)
        out = {"r": r, "J": J}

    if conv_lambda is not None and not getattr(F_psf, "particles_input", False):
        struct = F_psf.unflatten_params(theta_flat)

        def _drift(x):
            return F_psf(x[None], params=struct, extras=extras)[0]

        def _phi_frame(Xf):                          # (P−1, N, d) batched flow
            if per_step or frames is not None:
                def _one(Xr, dt_k, fk):
                    def _drift_frame(X):
                        return F_psf(X, params=struct,
                                     extras=with_frame(static, fk))
                    return flow_displacement(_drift_frame, Xr, dt_k, n_sub,
                                             integrator)
                return jax.vmap(_one, in_axes=(0, 0 if per_step else None,
                                               0))(Xf, dt, fr_res)
            flat = Xf.reshape(-1, Xf.shape[-1])
            disp = jax.vmap(lambda y: flow_displacement(
                _drift, y, dt, n_sub, integrator))(flat)
            return disp.reshape(Xf.shape)

        lam, U = jnp.linalg.eigh(0.5 * (conv_lambda + conv_lambda.T))
        s = jnp.sqrt(jnp.clip(lam, 0.0, None))[None, :] * U    # columns s_i
        X0 = X_block[:-1]
        phi0 = _phi_frame(X0)
        corr = jnp.zeros_like(phi0)
        for i in range(s.shape[-1]):
            si = s[:, i]
            corr = corr + 0.5 * (_phi_frame(X0 + si) + _phi_frame(X0 - si)
                                 - 2.0 * phi0)
        # E[r_plain] = −½∇²Φ:Λ (the flow's convexity against the argument
        # noise), so the correction is ADDED to re-center the residual.
        out["r"] = out["r"] + corr

    if with_psi:
        out["psi"] = psi
    if with_instrument:
        if per_step or frames is not None:
            # base X_j serves residual k = j+1: sensitivities over the
            # residual's interval dt[j+1] with its frame values (j+1),
            # propagation over the base's own interval dt[j] with frame j.
            dt_res_s, dt_base_s = (dt[1:], dt[:-1]) if per_step else (dt, dt)
            ax = 0 if per_step else None
            inst = jax.vmap(
                lambda xb, dt_res, dt_base, f_res, f_base: multi_od_instrument(
                    F_psf, theta_flat, xb, with_frame(static, f_res), dt_res,
                    n_sub, integrator, dt_prop=dt_base,
                    extras_prop=with_frame(static, f_base)),
                in_axes=(0, ax, ax, 0, 0),
            )(X_block[:-2], dt_res_s, dt_base_s,
              frame_window(frames, 1, -1), frame_window(frames, 0, -2))
        else:
            inst = jax.vmap(
                lambda xb: multi_od_instrument(F_psf, theta_flat, xb, extras,
                                               dt, n_sub, integrator)
            )(X_block[:-2])                            # bases X_0 … X_{P−3}
        # residual k pairs with base X_{k−1}: shift by one, front invalid
        pad = jnp.zeros_like(inst[:1])
        out["psi_inst"] = jnp.concatenate([pad, inst], axis=0)
        out["inst_valid"] = jnp.arange(r.shape[0]) >= 1
    return out


def ud_point_tensors(F_psf, theta_flat, Y_block, extras, dt, n_sub,
                     integrator="rk4", *, with_psi=True, with_instrument=False,
                     D_lyap=None, with_stages=False):
    r"""Per-point UD tensors on a block of ``P`` consecutive positions.

    Returns a dict with ``r (P−2, N, d)``, ``ap/a0/am (P−2, N, d, d)``,
    ``vhat (P−2, N, d)`` and, on request, ``psi (P−2, N, d, n)``,
    ``psi_inst`` + ``inst_valid`` (first two entries False).
    ``D_lyap``: additionally return ``qing``, the Lyapunov-exact process
    ingredients consumed by :func:`covariance.build_ud_blocks_exact`.

    ``dt`` may be a scalar or a ``(P−1,)`` vector of per-interval steps
    (``dt[k]`` spans ``k → k+1``): residual ``n`` shoots over ``dt[n]``
    and predicts over ``dt[n+1]``.

    ``with_stages``: additionally return ``cache`` — the lifted stage
    matrices of both interval flows plus the shooting propagators, from
    which the ``qing`` tensors are recomputable for any ``D`` via
    :func:`jacobians.lyapunov_from_stages` (mutually exclusive with
    ``D_lyap``).
    """
    if with_stages and D_lyap is not None:
        raise ValueError("with_stages caches the D-independent stage "
                         "matrices; do not combine with D_lyap")
    res = ud_multi_step_residuals_with_psi(
        F_psf, theta_flat, Y_block, extras, dt, n_sub, integrator,
        D_lyap=D_lyap, with_stages=with_stages)
    r, ap, a0, am, vhat, psi = res[:6]
    out = {"r": r, "ap": ap, "a0": a0, "am": am, "vhat": vhat}
    if D_lyap is not None:
        out["qing"] = res[6]
    if with_stages:
        out["cache"] = res[-1]
    if with_psi:
        out["psi"] = psi
    if with_instrument:
        static, frames = split_frames(extras)
        if frames is not None:
            # per-frame extras: the four intervals j … j+3 below read
            # their left frames j … j+3
            P = Y_block.shape[0]
            fr_w = tuple(frame_window(frames, i, P - 4 + i) for i in range(4))
            dt_w = (jnp.stack([dt[:-3], dt[1:-2], dt[2:-1], dt[3:]], axis=1)
                    if jnp.ndim(dt) > 0 else dt)
            inst = jax.vmap(
                lambda ya, yb, dtw, fw: multi_ud_instrument(
                    F_psf, theta_flat, ya, yb,
                    tuple(with_frame(static, f) for f in fw), dtw, n_sub,
                    integrator),
                in_axes=(0, 0, 0 if jnp.ndim(dt) > 0 else None, 0),
            )(Y_block[:-4], Y_block[1:-3], dt_w, fr_w)
        elif jnp.ndim(dt) > 0:
            # base pair (Y_j, Y_{j+1}) serves residual n = j+2: shoot over
            # dt[j], predict over the two physical intervals dt[j+1],
            # dt[j+2], evaluate over the residual's prediction interval
            # dt[j+3] (see multi_ud_instrument).
            dt_w = jnp.stack([dt[:-3], dt[1:-2], dt[2:-1], dt[3:]], axis=1)
            inst = jax.vmap(
                lambda ya, yb, dtw: multi_ud_instrument(
                    F_psf, theta_flat, ya, yb, extras, dtw, n_sub, integrator)
            )(Y_block[:-4], Y_block[1:-3], dt_w)
        else:
            inst = jax.vmap(
                lambda ya, yb: multi_ud_instrument(F_psf, theta_flat, ya, yb,
                                                   extras, dt, n_sub, integrator)
            )(Y_block[:-4], Y_block[1:-3])             # pairs (Y_j, Y_{j+1})
        # residual n uses pair (Y_{n−2}, Y_{n−1}) → j = n−2, valid n ≥ 2
        pad = jnp.zeros_like(inst[:1])
        out["psi_inst"] = jnp.concatenate([pad, pad, inst], axis=0)
        out["inst_valid"] = jnp.arange(r.shape[0]) >= 2
    return out
