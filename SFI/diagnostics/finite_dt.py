"""Finite sampling-interval scale of a fitted drift — the O(Δt) bias guard.

The linear estimators are exact in the limit Δt → 0; at finite Δt every
stencil carries biases of relative size Δt‖∇F‖, the drift gradient per
sampling step (the sampling interval in units of the local relaxation
time).  This module measures that scale on the data with the fitted force
and turns it into a verdict, so that a coarse-sampling regime is reported
rather than silently absorbed into the coefficients.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["finite_dt_scale", "sample_data_points"]

#: Verdict thresholds on the RMS per-step drift gradient Δt‖∇F̂‖.
FINITE_DT_FINE = 0.05
FINITE_DT_COARSE = 0.2

_HINTS = {
    "fine": "",
    "coarse": (
        "the O(Δt) biases of the linear estimators are a few to tens of per cent; "
        "confirm with the parametric estimator (infer_force, RK4 flow) or with a "
        "stride-2 subsample of the data"
    ),
    "unreliable": (
        "the sampling interval is comparable to the local relaxation time; the linear "
        "estimators are biased at O(1) — use the parametric estimator (infer_force)"
    ),
}


def sample_data_points(coll, *, needs_v: bool, n_points: int, seed: int = 0, dataset=None):
    """Uniform subsample of the valid data points of a collection.

    Returns ``(x, v, dt)``: ``x`` of shape ``(M, d)``, ``v`` the central
    finite-difference velocities at the same points when ``needs_v`` (else
    ``None``) and ``dt`` the median sampling interval; ``x`` is ``None``
    when no valid point exists.  ``dataset`` restricts the sample to one
    dataset of the collection.
    """
    xs, vs, dts = [], [], []
    ks = range(len(coll.datasets)) if dataset is None else [int(dataset)]
    for k in ks:
        t, X, mask = coll.to_arrays(dataset=k, as_numpy=True, include_mask=True)
        X = np.asarray(X, dtype=np.float64)
        valid = np.isfinite(X).all(axis=-1)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        if needs_v:
            V = np.asarray(coll.velocity_array(dataset=k, scheme="central", as_numpy=True), dtype=np.float64)
            valid &= np.isfinite(V).all(axis=-1)
            valid[0] = False          # one-sided end stencils of the central velocity
            valid[-1] = False
            vs.append(V[valid])
        xs.append(X[valid])
        dts.append(np.diff(np.asarray(t, dtype=np.float64)))
    x_all = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0))
    if x_all.shape[0] == 0:
        return None, None, float("nan")
    dt = float(np.median(np.concatenate(dts))) if dts and any(a.size for a in dts) else float("nan")
    rng = np.random.default_rng(seed)
    m = int(x_all.shape[0])
    idx = np.sort(rng.choice(m, size=min(int(n_points), m), replace=False)) if m > n_points else np.arange(m)
    x_sub = jnp.asarray(x_all[idx])
    v_sub = jnp.asarray(np.concatenate(vs, axis=0)[idx]) if needs_v else None
    return x_sub, v_sub, dt


def _verdict(scale: float) -> str:
    if not np.isfinite(scale):
        return "unavailable"
    if scale < FINITE_DT_FINE:
        return "fine"
    if scale < FINITE_DT_COARSE:
        return "coarse"
    return "unreliable"


def finite_dt_scale(inferer, *, n_points: int = 100_000, seed: int = 0, chunk: int = 4096) -> dict:
    r"""Per-step drift-gradient scale :math:`\Delta t\,\|\nabla \hat F\|` of a fitted force.

    Evaluates the Jacobian of ``inferer.force_inferred`` on a random subsample
    of the data points (with respect to the position for an overdamped fit,
    to the velocity for an underdamped one, where the relevant small
    parameter is :math:`\Delta t\,\gamma`) and reports its root-mean-square
    Frobenius norm times the sampling interval, together with its 99th
    percentile (the local worst case) and a verdict:

    * ``"fine"`` — below 0.05: the linear estimators are in their regime;
    * ``"coarse"`` — 0.05 to 0.2: O(Δt) biases of a few to tens of per cent;
      confirm with the parametric estimator or a stride-2 subsample;
    * ``"unreliable"`` — above 0.2: use the parametric estimator.

    Parameters
    ----------
    inferer :
        A fitted ``OverdampedLangevinInference`` or
        ``UnderdampedLangevinInference`` with a callable ``force_inferred``.
    n_points : int
        Maximum number of data points sampled (uniformly, without
        replacement) for the Jacobian average.
    seed : int
        Seed of the subsample.
    chunk : int
        Points per JIT-compiled Jacobian batch.

    Returns
    -------
    dict
        ``available`` (bool), ``grad_phi_dt`` (RMS scale), ``grad_p99_dt``
        (99th percentile of the per-point scale), ``grad_rms`` (RMS gradient
        without the Δt factor), ``dt`` (median sampling interval),
        ``n_points``, ``variable`` (``"x"`` or ``"v"``), ``verdict`` and
        ``hint``; when the scale cannot be computed (interacting bases,
        extras-dependent fields) ``available`` is False and ``reason`` says why.

    Notes
    -----
    Bases evaluated on particle sets (``particles_input``) are not
    supported: their Jacobian is the same-particle block, not the flat
    Jacobian taken here.  The Jacobian is exact automatic differentiation
    of the fitted field, so any basis that the statefunc layer can
    differentiate is accepted.
    """
    F = getattr(inferer, "force_inferred", None)
    if F is None:
        raise RuntimeError("inferer.force_inferred is missing; run a force-inference method first.")
    basis = getattr(inferer, "force_basis", None)
    if getattr(basis, "particles_input", False) or getattr(F, "particles_input", False):
        return dict(available=False, reason="interacting (particles_input) bases are not supported",
                    verdict="unavailable")
    needs_v = bool(getattr(F, "needs_v", False))
    x_sub, v_sub, dt = sample_data_points(inferer.data, needs_v=needs_v, n_points=n_points, seed=seed)
    if x_sub is None:
        return dict(available=False, reason="no valid data points", verdict="unavailable")

    if needs_v:
        def _jac(x, v):
            return jax.jacfwd(lambda vv: F(x[None], v=vv[None])[0])(v)
        jac = jax.jit(jax.vmap(_jac))
    else:
        def _jac(x):
            return jax.jacfwd(lambda xx: F(xx[None])[0])(x)
        jac = jax.jit(jax.vmap(_jac))

    norms = []
    try:
        for start in range(0, int(x_sub.shape[0]), int(chunk)):
            sl = slice(start, start + int(chunk))
            J = jac(x_sub[sl], v_sub[sl]) if needs_v else jac(x_sub[sl])
            norms.append(np.asarray(jnp.sqrt(jnp.sum(J ** 2, axis=(-2, -1)))))
    except Exception as exc:  # extras-dependent fields, unsupported graphs
        logger.info("finite_dt_scale unavailable: %s", exc)
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}", verdict="unavailable")
    g = np.concatenate(norms)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return dict(available=False, reason="non-finite Jacobians", verdict="unavailable")
    grad_rms = float(np.sqrt(np.mean(g ** 2)))
    scale = grad_rms * dt
    verdict = _verdict(scale)
    out = dict(
        available=True,
        grad_phi_dt=scale,
        grad_p99_dt=float(np.percentile(g, 99) * dt),
        grad_rms=grad_rms,
        dt=dt,
        n_points=int(g.size),
        variable="v" if needs_v else "x",
        verdict=verdict,
        hint=_HINTS.get(verdict, ""),
    )
    logger.info("finite-dt scale dt*|grad F| = %.3g (%s)", scale, verdict)
    return out
