# SFI/inference/parametric_core/driver.py
"""
IRLS driver for the parametric core.

Minimises the windowed objective by alternating:

1. **freeze** the precision ``P`` at the current ``θ`` (and the current
   ``D, Λ``) and minimise the resulting quadratic
   ``Q(θ) = ½ Σ rₙ(θ)ᵀ P rₙ(θ)`` over ``θ`` with L-BFGS — the inner solve;
2. **reprofile** ``D, Λ`` from the residuals at the updated ``θ``.

For linear-in-θ models this converges in ~1 outer iteration (it is the
GLS fixed point); for nonlinear-in-θ it is ordinary IRLS.  The inner
objective and its gradient are supplied by the caller (built from the
integrate-engine loss program), so the driver is agnostic to dynamics and
bandwidth.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as _sp_minimize

from SFI.utils.maths import default_float_dtype

logger = logging.getLogger(__name__)

__all__ = ["irls_minimize", "gn_minimize", "condition_cap_ridge"]


def condition_cap_ridge(G, gram_cond_max=1e10):
    r"""Tikhonov ridge that caps the condition number of the Gram ``G``.

    Mirrors the existing Gauss–Newton driver: if ``cond(G) > gram_cond_max``
    return ``λ = σ_max / gram_cond_max`` (so the regularised matrix
    ``G + λI`` has condition number ``≤ gram_cond_max``), else ``0``.  Adding
    ``½λ‖θ‖²`` to the inner objective then bounds ``θ`` along the
    ill-conditioned / unconstrained directions instead of letting L-BFGS
    drive them to ``±∞`` — graceful degradation under high measurement noise.

    Returns a finite, non-negative scalar (``0`` for a non-finite / zero
    Gram, where no meaningful curvature scale exists).
    """
    G = jnp.asarray(G)
    svals = jnp.linalg.svdvals(G)
    s_max = svals[0]
    s_min = svals[-1]
    finite = jnp.isfinite(s_max) & (s_max > 0)
    over = s_max > gram_cond_max * s_min
    lam = jnp.where(finite & over, s_max / gram_cond_max, 0.0)
    return lam


def irls_minimize(
    theta0,
    build_inner,
    D0,
    Lambda0,
    *,
    profile_fn=None,
    max_outer=5,
    reprofile_iters=None,
    inner_maxiter=80,
    inner_ftol=1e-12,
    tol=1e-7,
    label="",
):
    r"""Run the IRLS outer loop.

    Parameters
    ----------
    theta0 : ``(n_params,)``
    build_inner : callable
        ``build_inner(theta_frozen, D, Lambda) -> inner(theta_live) ->
        (value, grad)``.  Built once per outer iteration so the frozen
        precision is hoisted out of the inner L-BFGS sweep.
    D0, Lambda0 : ``(d, d)`` initial noise matrices.
    profile_fn : callable or None
        ``profile_fn(theta, D, Lambda) -> (D, Lambda)``; called after
        the inner solve at the first ``reprofile_iters`` outer iterations
        (warm-started from the current noise).  ``None`` holds ``(D, Λ)``
        fixed.
    max_outer, reprofile_iters, inner_maxiter, inner_ftol, tol : see module docstring.

    Returns
    -------
    theta_best, D_best, Lambda_best, info
    """
    dtype = default_float_dtype()
    theta = jnp.asarray(theta0, dtype=dtype)
    D = jnp.asarray(D0, dtype=dtype)
    Lambda = jnp.asarray(Lambda0, dtype=dtype)
    if reprofile_iters is None:
        reprofile_iters = max_outer

    best_loss = float("inf")
    best = (theta, D, Lambda)
    converged = False
    outer = 0

    for outer in range(max_outer):
        theta_prev = theta
        inner = build_inner(jax.lax.stop_gradient(theta), D, Lambda)

        def _obj(x):
            v, g = inner(jnp.asarray(x, dtype=dtype))
            return float(v), np.asarray(g, dtype=np.float64)

        res = _sp_minimize(
            _obj, np.asarray(theta, dtype=np.float64), jac=True, method="L-BFGS-B",
            options={"maxiter": inner_maxiter, "ftol": inner_ftol, "gtol": inner_ftol},
        )
        theta = jnp.asarray(res.x, dtype=dtype)
        loss = float(res.fun)

        if profile_fn is not None and outer < reprofile_iters:
            D, Lambda = profile_fn(theta, D, Lambda)

        if loss < best_loss and bool(jnp.all(jnp.isfinite(theta))):
            best_loss = loss
            best = (theta, D, Lambda)

        rel = float(jnp.linalg.norm(theta - theta_prev) / (jnp.linalg.norm(theta_prev) + 1e-30))
        logger.info("[core-irls] %s outer %d: loss=%.6g  rel=%.3g", label, outer, loss, rel)
        if rel < tol:
            converged = True
            break

    info = {"outer_iterations": outer + 1, "converged": converged, "loss": best_loss}
    return best[0], best[1], best[2], info


def gn_minimize(
    theta0,
    gram_fn,
    D0,
    Lambda0,
    *,
    profile_fn=None,
    nll_fn=None,
    max_iter=8,
    reprofile_iters=None,
    gram_cond_max=1e10,
    line_search_alphas=(1.0, 0.5, 0.25, 0.1),
    tol=1e-7,
    label="",
    merit="nll",
):
    r"""Direct Gauss–Newton outer loop on the windowed Gram.

    The linear-in-θ fast path: the windowed estimator already returns the GN
    normal-equation pieces ``(G, f, nll)`` — with ``G = ψᵀPψ`` the GN Hessian,
    ``f = ψᵀPr`` the score (``ψ = ∂r/∂θ``, so ``f`` is the **+gradient** of
    ``½ rᵀP r``) — so the step is the closed-form ``δθ = −(G+λI+μ·diagG)⁻¹ f``.
    No nested ``value_and_grad`` through the flow, so for a linear-in-θ force it
    reaches the GLS fixed point in ~1–2 iterations instead of ~80 L-BFGS
    objective evaluations.

    Mirrors the legacy ``parametric_gn_iterate`` driver: Tikhonov condition cap
    (:func:`condition_cap_ridge`), Levenberg–Marquardt damping with line-search
    retry, ``(D, Λ)`` reprofiling at early iterations, and best-iterate
    tracking.

    Parameters
    ----------
    theta0 : ``(n_params,)``
    gram_fn : callable ``(theta, D, Lambda) -> (G, f, nll)``.
    D0, Lambda0 : ``(d, d)`` initial noise matrices.
    profile_fn : callable or None  ``(theta, D, Λ) -> (D, Λ)``; called after
        iter 0 for the first ``reprofile_iters`` iterations.
    nll_fn : callable or None  scalar objective for the line search; falls back
        to the Gram's own ``nll`` when ``None``.
    merit : {"nll", "phi"}  line-search / best-iterate merit.  ``"nll"`` (the
        symmetric MLE) descends the windowed NLL.  ``"phi"`` is the
        estimating-equation residual ``‖f‖ = ‖⟨ψ_left P r⟩‖`` — the correct
        merit when ``G`` is asymmetric (the EIV instrument path), where the IV
        root does **not** minimise the NLL.  The condition guard is taken on the
        symmetric part ``½(G+Gᵀ)`` so it stays valid for asymmetric ``G``.
    max_iter, gram_cond_max, line_search_alphas, tol : GN budget / guards.

    Returns
    -------
    theta_best, D_best, Lambda_best, info  (same contract as
    :func:`irls_minimize`).
    """
    dtype = default_float_dtype()
    theta = jnp.asarray(theta0, dtype=dtype)
    D = jnp.asarray(D0, dtype=dtype)
    Lambda = jnp.asarray(Lambda0, dtype=dtype)
    n_params = int(theta.shape[0])
    I_n = jnp.eye(n_params, dtype=dtype)
    if reprofile_iters is None:
        reprofile_iters = max_iter

    def _merit(th, D_, Se_):
        if merit == "phi":
            return float(jnp.linalg.norm(gram_fn(th, D_, Se_)[1]))
        if nll_fn is not None:
            return float(nll_fn(th, D_, Se_))
        return float(gram_fn(th, D_, Se_)[2])

    best_nll = float("inf")
    best = (theta, D, Lambda)
    converged = False
    lm = 0.0
    outer = 0

    for outer in range(max_iter):
        if profile_fn is not None and 0 < outer < reprofile_iters + 1:
            D, Lambda = profile_fn(theta, D, Lambda)

        G, f, nll = gram_fn(theta, D, Lambda)
        # The acceptance bar must be the SAME objective as the line-search trial
        # (_merit): when nll_fn is given, the Gram program's own nll carries an
        # arbitrary offset against it (different window/conditioning
        # normalisation), and comparing trial-merit against gram-nll can reject
        # every step — freezing θ at the init and reporting rel=0 "convergence".
        if merit == "phi":
            nll_val = float(jnp.linalg.norm(f))
        elif nll_fn is not None:
            nll_val = float(nll_fn(theta, D, Lambda))
        else:
            nll_val = float(nll)
        if nll_val < best_nll and bool(jnp.all(jnp.isfinite(theta))):
            best_nll = nll_val
            best = (theta, D, Lambda)

        # Tikhonov condition cap on the symmetric part (valid for asymmetric G).
        lam = float(condition_cap_ridge(0.5 * (G + G.T), gram_cond_max))
        G_reg = G + lam * I_n

        theta_prev = theta
        accepted = False
        retries = 0
        while not accepted and retries <= 8:
            G_damped = G_reg + lm * jnp.diag(jnp.diag(G_reg)) if lm > 0 else G_reg
            delta = -jnp.linalg.solve(G_damped, f)
            if not bool(jnp.all(jnp.isfinite(delta))):
                retries += 1
                lm = max(lm * 4.0, 1e-3)
                continue
            for alpha in line_search_alphas:
                theta_trial = theta + alpha * delta
                if not bool(jnp.all(jnp.isfinite(theta_trial))):
                    continue
                nll_trial = _merit(theta_trial, D, Lambda)
                if nll_trial < nll_val + 1e-8:
                    theta = theta_trial
                    accepted = True
                    if nll_trial < best_nll:
                        best_nll = nll_trial
                        best = (theta, D, Lambda)
                    break
            if not accepted:
                retries += 1
                lm = max(lm * 4.0, 1e-3)

        if accepted:
            lm *= 0.25
        rel = float(jnp.linalg.norm(theta - theta_prev) / (jnp.linalg.norm(theta_prev) + 1e-30))
        logger.info("[core-gn] %s iter %d: nll=%.6g rel=%.3g lam=%.2e accepted=%s",
                    label, outer, nll_val, rel, lam, accepted)
        if not accepted or rel < tol:
            converged = rel < tol
            break

    info = {"outer_iterations": outer + 1, "converged": converged, "loss": best_nll}
    return best[0], best[1], best[2], info
