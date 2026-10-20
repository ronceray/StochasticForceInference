"""Positivity check of a fitted diffusion field on the data.

A linear diffusion fit is not constrained to be positive semi-definite: a
quadratic dictionary with cross terms, or a dictionary the data does not
pin down, can go negative on part of the support.  The simulators clip
negative eigenvalues to zero (``sqrtm_psd``), so the generated noise
differs from the fitted field there without any error.  This module
measures the fraction of data points where the field is not PSD, so that
generative use of a fitted diffusion is preceded by an explicit check.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from .finite_dt import sample_data_points

logger = logging.getLogger(__name__)

__all__ = ["psd_fraction"]


def psd_fraction(D_sf, coll, *, n_points: int = 10_000, seed: int = 0, dataset=None, rtol: float = 1e-6,
                 chunk: int = 4096) -> dict:
    r"""Fraction of data points where a diffusion field is not positive semi-definite.

    Evaluates ``D_sf`` on a uniform subsample of the valid data points (with
    the central finite-difference velocity when the field needs ``v``) and
    counts the points whose smallest eigenvalue is below ``-rtol`` times
    the largest eigenvalue of the same point (rounding tolerance).

    Parameters
    ----------
    D_sf :
        A rank-2 state function (a fitted ``diffusion_inferred`` or any
        ``SF``); interacting (``particles_input``) fields are not supported.
    coll : TrajectoryCollection
        The data; ``dataset`` restricts the sample to one dataset.
    n_points, seed :
        Subsample size and seed.
    rtol : float
        Relative eigenvalue tolerance.

    Returns
    -------
    dict
        ``available``, ``fraction`` (of non-PSD points), ``n_negative``,
        ``n_points``, ``min_eig`` (most negative eigenvalue found) and
        ``min_eig_rel`` (relative to the point's largest eigenvalue); with
        ``reason`` when unavailable.
    """
    if getattr(D_sf, "particles_input", False):
        return dict(available=False, reason="interacting (particles_input) fields are not supported")
    needs_v = bool(getattr(D_sf, "needs_v", False))
    x, v, _ = sample_data_points(coll, needs_v=needs_v, n_points=n_points, seed=seed, dataset=dataset)
    if x is None:
        return dict(available=False, reason="no valid data points")
    mins, rels = [], []
    try:
        for start in range(0, int(x.shape[0]), int(chunk)):
            sl = slice(start, start + int(chunk))
            D = jnp.asarray(D_sf(x[sl], v=v[sl]) if needs_v else D_sf(x[sl]))
            if D.ndim == 4 and D.shape[-1] == 1:      # single-feature basis kept its feature axis
                D = D[..., 0]
            D = 0.5 * (D + jnp.swapaxes(D, -1, -2))
            w = np.asarray(jnp.linalg.eigvalsh(D), dtype=np.float64)     # ascending
            mins.append(w[:, 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                rels.append(w[:, 0] / np.maximum(np.abs(w[:, -1]), np.finfo(np.float64).tiny))
    except Exception as exc:
        logger.info("psd_fraction unavailable: %s", exc)
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    mn = np.concatenate(mins)
    rel = np.concatenate(rels)
    ok = np.isfinite(mn) & np.isfinite(rel)
    mn, rel = mn[ok], rel[ok]
    if mn.size == 0:
        return dict(available=False, reason="non-finite diffusion values")
    neg = rel < -float(rtol)
    out = dict(
        available=True,
        fraction=float(np.mean(neg)),
        n_negative=int(np.sum(neg)),
        n_points=int(mn.size),
        min_eig=float(np.min(mn)),
        min_eig_rel=float(np.min(rel)),
    )
    logger.info("diffusion PSD check: %.3g%% of %d points non-PSD (min eig %.3g)",
                100 * out["fraction"], out["n_points"], out["min_eig"])
    return out
