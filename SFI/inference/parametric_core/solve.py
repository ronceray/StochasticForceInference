# SFI/inference/parametric_core/solve.py
"""
Orchestration for the overdamped parametric-core force solve.

Ties the pieces together — **everything through the ``SFI.integrate``
engine** (chunking, masking, JIT); no hand-rolled trajectory passes:

    flow → banded covariance → windowed precision/NLL  (integrate programs)
    + (D, Λ) profiled by the windowed conditional NLL
    + IRLS driver
    → θ̂, D̂, Σ̂_η, Gram (for covariance / sparsity).

``(D, Λ)`` are full symmetric matrices throughout.  Two noise/objective
roles share one program:

* ``ODLossProgram``    — frozen-precision quadratic, minimised over θ (fast
  inner solve);
* ``ODCondNLLProgram`` — windowed *conditional* NLL (local Schur log-det,
  non-degenerate in D, Λ), minimised over the noise to profile ``(D, Λ)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as _sp_minimize

from SFI.integrate.api import make_parametric_integrator
from SFI.utils.maths import default_float_dtype

from .driver import condition_cap_ridge, gn_minimize, irls_minimize
from .objective import (
    ODCondNLLProgram,
    ODDiffNLLProgram,
    ODGramProgram,
    ODLossProgram,
    unpack_gram,
)
from .objective_ud import UDCondNLLProgram, UDDiffNLLProgram, UDGramProgram, UDLossProgram

__all__ = [
    "ForceSolveResult", "DiffusionSolveResult",
    "solve_force_od", "solve_force_ud", "solve_diffusion_od", "solve_diffusion_ud",
]


@dataclass
class ForceSolveResult:
    theta: jnp.ndarray          # (n_params,) fitted parameters
    D: jnp.ndarray              # (d, d) diffusion
    Lambda: jnp.ndarray      # (d, d) measurement-noise covariance
    G: jnp.ndarray              # (n_params, n_params) GN Gram (symmetrised)
    f: jnp.ndarray              # (n_params,) score at optimum
    theta_cov: jnp.ndarray      # (n_params, n_params) parameter covariance:
                                # the sandwich G⁻¹ H G⁻ᵀ with H = ψ_leftᵀPψ_left
                                # (= G⁻¹ on the symmetric path, where H = G)
    info: dict


@dataclass
class DiffusionSolveResult:
    theta_D: jnp.ndarray        # (n_D,) fitted diffusion parameters
    info: dict


def _is_linear_basis(F):
    """True when ``F`` is a (linear-in-θ) ``Basis`` rather than a ``PSF``.

    A ``Basis`` exposes ``to_psf`` and has no ``template`` (the PSF attribute);
    its force is ``F = Σ_a θ_a b_a(x)``, so the windowed Gram is the GLS fixed
    point and the direct Gauss–Newton path applies.  A user-supplied ``PSF`` is
    treated as potentially nonlinear-in-θ (L-BFGS default) unless the caller
    forces ``inner="gn"``.
    """
    return hasattr(F, "to_psf") and not hasattr(F, "template")


def _resolve_inner(inner, is_linear):
    """Resolve ``inner="auto"`` to ``"gn"`` (linear) or ``"lbfgs"`` (nonlinear)."""
    if inner == "auto":
        return "gn" if is_linear else "lbfgs"
    if inner not in ("gn", "lbfgs"):
        raise ValueError(f"inner must be 'auto', 'gn', or 'lbfgs'; got {inner!r}")
    return inner


def _resolve_eiv_auto(eiv, F_psf):
    """Resolve ``eiv="auto"`` (the default) from the force model.

    ``"auto"`` → ``True`` for every model, including interacting ones
    (``particles_input=True``).  Interacting models briefly defaulted to
    ``False``: the original instrument evaluated the force on isolated
    single-particle frames, which zeroed (or crashed) every pair-feature
    column and made the IV Gram structurally singular on the interaction
    parameters — the noise-independent NMSE plateau on the aligning-ABP
    port.  The instrument now uses the same N-body flow as the residual
    (:func:`flow_multi.multi_od_instrument` / ``multi_ud_instrument``), so
    the consistent estimator is the default everywhere.  Explicit
    ``True``/``False``/float always wins.
    """
    if eiv == "auto":
        return True
    return eiv


def _resolve_w(eiv):
    """Resolve ``eiv`` to a scalar instrument blend weight ``w ∈ [0, 1]``.

    ``False`` → ``0`` (plain MLE); ``True`` → ``1`` (pure η-clean instrument);
    a float passes through (clamped to ``[0, 1]``) — the manual bias-variance blend.
    """
    if eiv is False:
        return 0.0
    if eiv is True:
        return 1.0
    if isinstance(eiv, (int, float)) and not isinstance(eiv, bool):
        return float(min(max(eiv, 0.0), 1.0))
    raise ValueError(f"eiv must be 'auto', bool, or float in [0, 1]; got {eiv!r}")


def _as_psf(F):
    """Accept a PSF directly or convert a Basis via ``to_psf()``."""
    if hasattr(F, "to_psf") and not hasattr(F, "template"):
        return F.to_psf()
    return F


def _chol_vec_to_mat(z, d):
    """Unconstrained ``z`` (lower-tri, log-diagonal) → SPD matrix ``L Lᵀ``."""
    ii, jj = np.tril_indices(d)
    L = jnp.zeros((d, d), dtype=z.dtype).at[ii, jj].set(z)
    di = np.diag_indices(d)
    L = L.at[di].set(jnp.exp(jnp.diagonal(L)))
    return L @ L.T


def _mat_to_chol_vec(M, d):
    """SPD matrix → unconstrained ``z`` (inverse of :func:`_chol_vec_to_mat`)."""
    L = jnp.linalg.cholesky(M + 1e-10 * jnp.eye(d, dtype=M.dtype))
    ii, jj = np.tril_indices(d)
    z = L[ii, jj]
    diag_pos = np.where(ii == jj)[0]
    return z.at[diag_pos].set(jnp.log(jnp.clip(jnp.diagonal(L), 1e-12, None)))


def _collection_dt(collection):
    dt = collection.datasets[0].dt
    return float(jnp.asarray(dt).reshape(-1)[0])


def _lbfgs(obj_and_grad, x0, dtype, maxiter, *, ftol=1e-12, gtol=1e-12):
    """scipy L-BFGS-B over a flat real vector with a jax value-and-grad.

    Infeasible iterates (NaN/inf objective or gradient — e.g. a line-search
    trial step proposing a non-PSD diffusion) are mapped to a finite penalty
    so the Fortran line search backtracks instead of aborting at the
    incumbent point (which silently returns ``x0``).  The penalty is kept on
    the scale of the objective: an astronomically large value (e.g. 1e15)
    makes the line search's quadratic interpolation collapse the trial step
    to ~0, where the f-decrease falls below ``ftol`` and scipy reports
    convergence at the initial point.
    """
    state: dict[str, float | None] = {"f_ref": None}

    def _obj(x):
        v, g = obj_and_grad(jnp.asarray(x, dtype=dtype))
        v = float(v)
        g = np.asarray(g, dtype=np.float64)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            f_ref = state["f_ref"]
            penalty = 1e15 if f_ref is None else f_ref + 1e3 * (1.0 + abs(f_ref))
            return penalty, np.zeros_like(g)
        if state["f_ref"] is None or v < state["f_ref"]:
            state["f_ref"] = v
        return v, g

    res = _sp_minimize(
        _obj, np.asarray(x0, dtype=np.float64), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    return jnp.asarray(res.x, dtype=dtype), float(res.fun)


# (D, Λ) profiling tolerances: each profile evaluation is a full trajectory
# pass, and the noise matrices only set the precision weighting of the θ solve,
# so they need far less accuracy than the θ optimum itself (which keeps 1e-12).
_PROFILE_FTOL = 1e-9
_PROFILE_GTOL = 1e-8


def _spd_floor(M, dtype, rel=1e-6, abs_floor=1e-12):
    """Symmetrise a moment estimate and floor its eigenvalues to SPD.

    Moment estimators fluctuate below zero when the true matrix is (near)
    zero — e.g. Λ on clean data — and the Cholesky reparameterisation of
    the profile needs a strictly positive matrix to start from.
    """
    M = jnp.asarray(M, dtype=dtype)
    M = 0.5 * (M + M.T)
    w, V = jnp.linalg.eigh(M)
    floor = jnp.maximum(rel * jnp.max(jnp.abs(w)), abs_floor)
    w = jnp.clip(w, floor, None)
    return (V * w) @ V.T


def _moment_init(collection, *, dynamics, d, dtype, drop_se):
    """Closed-form moment (D, Λ) initialisation — replaces the θ=0 profile.

    One trajectory pass per matrix, no optimizer: the noise-robust pair
    'noisy' (Vestergaard) D + increment-anticorrelation Λ (overdamped) or
    ULI 'noisy' D + ULI Λ (underdamped) — the same estimators behind
    ``compute_diffusion_constant``.  The windowed conditional NLL then
    refines ``(D, Λ)`` once at the fitted θ (the reprofile step), so the
    init only needs to be the right order of magnitude.
    """
    from SFI.integrate.api import integrate
    from SFI.integrate.integrand import Integrand, Term, TimeOperand

    if dynamics == "od":
        from SFI.inference.overdamped import _D_noisy, _Lambda
        D_fn, L_fn = _D_noisy, _Lambda
    else:
        from SFI.inference.underdamped import _D_noisy_uli, _Lambda_uli
        D_fn, L_fn = _D_noisy_uli, _Lambda_uli

    def _mean(fn, alias):
        op = TimeOperand(fn, alias=alias)
        prog = Integrand(times=[op], terms=[Term(eq="imn->imn", ops=(op.alias,))])
        return integrate(collection, prog, reduce="mean")

    D0 = _spd_floor(_mean(D_fn, "D_mom"), dtype)
    if drop_se:
        return D0, jnp.zeros((d, d), dtype=dtype)
    return D0, _spd_floor(_mean(L_fn, "Lambda_mom"), dtype)


def _init_theta_D(D_psf, collection, *, dynamics, dtype, n_D):
    """Default θ_D init: project the moment-estimated constant D̂ onto the model.

    θ_D = 0 means D(x) ≡ 0 — a singular covariance where the conditional NLL
    (and its gradient) is non-finite, so L-BFGS cannot leave the start point.
    Solve min_θ ‖D(x;θ) − D̂‖² at sampled trajectory points instead (exact for
    linear-in-θ models, first-order otherwise).  Falls back to zeros if the
    model cannot be evaluated standalone (e.g. needs extras).
    """
    try:
        X = jnp.asarray(collection.datasets[0].X)
        if X.ndim == 3:
            X = X[:, 0, :]
        d = X.shape[-1]
        D_const, _ = _moment_init(collection, dynamics=dynamics, d=d, dtype=dtype,
                                  drop_se=True)
        dt = _collection_dt(collection)
        idx = np.linspace(0, X.shape[0] - 2, min(64, X.shape[0] - 1)).astype(int)
        Xs = X[idx]
        Vs = (X[idx + 1] - X[idx]) / dt if dynamics == "ud" else None
        cols = []
        for j in range(n_D):
            th = D_psf.unflatten_params(jnp.zeros(n_D, dtype=dtype).at[j].set(1.0))
            if Vs is None:
                Bj = jax.vmap(lambda x: D_psf(x[None], params=th)[0])(Xs)
            else:
                Bj = jax.vmap(lambda x, v: D_psf(x[None], v=v[None], params=th)[0])(Xs, Vs)
            cols.append(Bj.reshape(-1))
        A = jnp.stack(cols, axis=1)
        b = jnp.broadcast_to(D_const, (Xs.shape[0],) + D_const.shape).reshape(-1)
        th0 = jnp.linalg.lstsq(A, b)[0]
        if bool(jnp.all(jnp.isfinite(th0))):
            return jnp.asarray(th0, dtype=dtype)
    except Exception:
        pass
    return jnp.zeros(n_D, dtype=dtype)


def _integrator(collection, program, *, reduce_over_particles=False):
    _, run = make_parametric_integrator(
        collection, program, reduce="sum",
        reduce_over_particles=reduce_over_particles, weight_by_dt=False,
    )
    return run


def _init_theta(theta0, F_psf, n_params, dtype):
    if theta0 is None:
        return jnp.zeros(n_params, dtype=dtype)
    if isinstance(theta0, dict):
        return jnp.asarray(F_psf.flatten_params(theta0), dtype=dtype)
    return jnp.asarray(theta0, dtype=dtype).reshape(-1)


def _resolve_n_substeps(n_substeps):
    """Validate the fixed substep count (``"auto"`` was removed in v2.0)."""
    if n_substeps == "auto":
        raise ValueError(
            "n_substeps='auto' was removed in v2.0; pass an int (default 1)."
        )
    return int(n_substeps)


def _orchestrate(build_runs, n_sub, theta, D, Lambda, *,
                 d, n_params, dtype, max_outer, inner_maxiter, profile_maxiter,
                 label, inner, noise_init, gram_cond_max=1e10, eiv=False):
    """Shared solve + (D,Λ) profiling + Gram.

    Dynamics-agnostic.  ``build_runs(n_substeps) -> (loss_run, nll_run,
    gram_run)`` constructs the integrate-engine runners at the fixed substep
    count.

    The inner solve is ``inner="gn"`` (direct Gauss–Newton on the windowed
    Gram — the linear-in-θ fast path) or ``inner="lbfgs"`` (frozen-precision
    L-BFGS IRLS — the nonlinear-in-θ path).
    """
    nsym = d * (d + 1) // 2
    fixed_noise = D is not None and Lambda is not None
    w = 0.0  # EIV instrument blend weight (resolved once the noise scale is known)
    is_od = label.startswith("od")
    # Drop Λ profiling for the *explicit overdamped MLE* (eiv=False) only — see
    # _profile.  Set once `w` (and thus the MLE intent) is known; NOT for an
    # eiv=True solve that merely got downgraded to w=0 on the L-BFGS path (still
    # noise-aware), and NOT for UD (velocity-EIV is a distinct, intended effect).
    drop_se_mle = False

    def _machinery(runs):
        """Build the (profiler, loss-vg, gram-fn, nll-scalar) for one set of runs.

        Each ``@jax.jit`` compiles lazily on first call, so the GN path never
        compiles the (grad-through-flow) loss program and the L-BFGS path never
        compiles the Gram program beyond the final covariance.
        """
        loss_run, nll_run, gram_run = runs

        @jax.jit
        def _profile_vg(z, th):
            return jax.value_and_grad(
                lambda zz: jnp.sum(nll_run((
                    th, _chol_vec_to_mat(zz[:nsym], d), _chol_vec_to_mat(zz[nsym:], d),
                )))
            )(z)

        @jax.jit
        def _profile_D_vg(zD, th):
            # D-only profiler (Λ held at 0) — for the w=0 / MLE path.
            return jax.value_and_grad(
                lambda zz: jnp.sum(nll_run((
                    th, _chol_vec_to_mat(zz, d), jnp.zeros((d, d), dtype=dtype),
                )))
            )(zD)

        @jax.jit
        def _loss_vg(theta_live, theta_frozen, D_, Se_):
            return jax.value_and_grad(lambda tl: jnp.sum(loss_run((tl, theta_frozen, D_, Se_))))(theta_live)

        def _profile(th, D_cur, Se_cur):
            # The explicit overdamped MLE (eiv=False) is the *no-measurement-noise*
            # estimator: profiling Λ into its symmetric precision is unidentifiable on
            # stiff/large-Δt clean data (the conditional NLL cannot separate process noise
            # ∝Δt from a Δt-independent Λ), and the spurious estimate collapses the Gram
            # (θ→0; OU Δt=0.4 NMSE 5e12, Lorenz 3.74).  So profile D only and hold Λ=0 —
            # matching legacy's symmetric path, which carries no Λ term.  Every other
            # solve (η-clean skip w>0, eiv=True downgraded to w=0 on L-BFGS, all UD) keeps
            # the joint (D,Λ) profile.
            if not drop_se_mle:
                z0 = jnp.concatenate([_mat_to_chol_vec(D_cur, d), _mat_to_chol_vec(Se_cur, d)])
                z, _ = _lbfgs(lambda z: _profile_vg(z, th), z0, dtype, profile_maxiter,
                              ftol=_PROFILE_FTOL, gtol=_PROFILE_GTOL)
                return _chol_vec_to_mat(z[:nsym], d), _chol_vec_to_mat(z[nsym:], d)
            zD0 = _mat_to_chol_vec(D_cur, d)
            zD, _ = _lbfgs(lambda z: _profile_D_vg(z, th), zD0, dtype, profile_maxiter,
                           ftol=_PROFILE_FTOL, gtol=_PROFILE_GTOL)
            return _chol_vec_to_mat(zD, d), jnp.zeros((d, d), dtype=dtype)

        def _gram_fn(th, D_, Se_):
            # Symmetrise only the MLE Gram; the EIV path (w>0) is an asymmetric
            # estimating equation and must keep ψ_left ≠ ψ_right intact.
            G, f, _, nll = unpack_gram(gram_run((th, D_, Se_)), n_params)
            return (G if w > 0.0 else 0.5 * (G + G.T)), f, nll

        def _nll_scalar(th, D_, Se_):
            return float(jnp.sum(nll_run((th, D_, Se_))))

        return _profile, _loss_vg, _gram_fn, _nll_scalar

    def _solve_pass(mach, theta, D0, Lambda0, prof):
        """One inner solve given prebuilt machinery; returns (θ, D, Λ, info)."""
        _profile, _loss_vg, _gram_fn, _nll_scalar = mach

        if inner == "gn":
            # GN caps cond(G) internally via condition_cap_ridge (gram_cond_max).
            # reprofile_iters=1: solve → one warm (D, Λ) profile at the fitted θ →
            # re-solve.  For linear-in-θ this is the IRLS fixed point; profiling at
            # every outer iteration only repeats the dominant-cost L-BFGS for ~no
            # change in θ̂.
            theta_h, D_h, Se_h, sinfo = gn_minimize(
                theta, _gram_fn, D0, Lambda0, profile_fn=prof, nll_fn=_nll_scalar,
                max_iter=max_outer, reprofile_iters=1,
                gram_cond_max=gram_cond_max, label=label,
                merit=("phi" if w > 0.0 else "nll"),
            )
            return theta_h, D_h, Se_h, {**sinfo, "ridge_lambda": 0.0}

        # L-BFGS IRLS: Tikhonov ½λ‖θ‖² (λ from the Gram at the pass start) bounds
        # θ along ill-conditioned directions instead of letting L-BFGS run to ±∞.
        lam = 0.0
        if gram_cond_max and np.isfinite(gram_cond_max):
            G0, _, _ = _gram_fn(theta, D0, Lambda0)
            lam = float(condition_cap_ridge(G0, gram_cond_max))

        def build_inner(theta_frozen, D_, Se_):
            def base(theta_live):
                return _loss_vg(theta_live, theta_frozen, D_, Se_)

            if lam <= 0.0:
                return base

            def innerf(theta_live):
                v, g = base(theta_live)
                return v + 0.5 * lam * jnp.dot(theta_live, theta_live), g + lam * theta_live

            return innerf

        theta_h, D_h, Se_h, sinfo = irls_minimize(
            theta, build_inner, D0, Lambda0, profile_fn=prof, reprofile_iters=1,
            max_outer=max_outer, inner_maxiter=inner_maxiter, label=label,
        )
        return theta_h, D_h, Se_h, {**sinfo, "ridge_lambda": lam}

    # ── EIV blend weight ──────────────────────────────────────────────────────
    # The instrument weight is known up front, so the first solve already uses the
    # η-clean instrument — no wasted (and, under noise, explosive) symmetric cold
    # pass.  The asymmetric estimating equation has no loss potential ⇒ Gauss–Newton
    # only; on the L-BFGS path (nonlinear-in-θ PSF) fall back to the MLE.
    w = _resolve_w(eiv)
    # Explicit MLE (the user asked for w=0 up front, not a downgrade): OD-only Λ drop.
    drop_se_mle = is_od and (w == 0.0) and not fixed_noise
    if w > 0.0 and inner != "gn":
        w = 0.0

    runs = build_runs(n_sub, w=w)
    mach = _machinery(runs)
    if fixed_noise:
        D0 = jnp.asarray(D, dtype=dtype)
        Lambda0 = jnp.asarray(Lambda, dtype=dtype)
        prof = None
    else:
        # Closed-form moment (D, Λ) init — one trajectory pass, no optimizer.
        # The windowed conditional NLL refines it once at the fitted θ (reprofile).
        D0, Lambda0 = noise_init(drop_se_mle)
        prof = mach[0]

    theta, D_hat, Lambda_hat, sinfo = _solve_pass(mach, theta, D0, Lambda0, prof)
    info = {**sinfo, "n_substeps": n_sub, "inner": inner, "eiv_w": w}

    _, _, gram_run = runs
    G_raw, f, H, _ = unpack_gram(gram_run((theta, D_hat, Lambda_hat)), n_params)
    # Parameter covariance: sandwich G⁻¹ H G⁻ᵀ.  On the symmetric path
    # H = G and this is the usual inverse information; on the IV path
    # (asymmetric G) the sandwich is the correct asymptotic covariance —
    # G⁻¹ alone would mis-state the error bars under measurement noise.
    Ginv = jnp.linalg.inv(G_raw + 1e-300 * jnp.eye(n_params, dtype=dtype))
    theta_cov = Ginv @ H @ Ginv.T
    theta_cov = 0.5 * (theta_cov + theta_cov.T)
    G = 0.5 * (G_raw + G_raw.T)
    return ForceSolveResult(theta=theta, D=D_hat, Lambda=Lambda_hat, G=G, f=f,
                            theta_cov=theta_cov, info=info)


def solve_force_od(
    collection,
    F,
    *,
    theta0=None,
    D=None,
    Lambda=None,
    n_substeps=1,
    integrator="rk4",
    max_outer=5,
    inner_maxiter=80,
    n_cond=3,
    profile_maxiter=20,
    inner="auto",
    eiv="auto",
    extra_radius=1,
):
    r"""Solve the overdamped parametric force problem.

    Parameters
    ----------
    collection : TrajectoryCollection
    F : PSF or Basis  (Basis is converted with ``to_psf()``).
    theta0 : dict, array, or None  initial parameters (default zeros).
    D, Lambda : ``(d, d)`` or None.  When given, held fixed; otherwise
        profiled: moment-estimator init (Vestergaard/ULI + Λ), then one
        windowed-conditional-NLL refinement at the fitted θ.
    n_substeps : int  RK4/Euler micro-steps per Δt (default 1 — the
        single-step minimal estimator).
    integrator : {"rk4", "euler"}  Flow predictor.  Default ``"rk4"`` (a single
        RK4 step — the minimal estimator).  A single ``"euler"`` step is also
        available, but note it cannot carry the force into the position update
        over one step, so the underdamped skip-trick (``eiv``) is unavailable
        for ``integrator="euler", n_substeps=1`` (auto-disabled with a warning).
    max_outer, inner_maxiter : solve budget.
    n_cond : int  past residuals conditioned on in the windowed NLL
        (window = ``n_cond + 2`` points).
    inner : {"auto", "gn", "lbfgs"}  inner solver.  ``"auto"`` → direct
        Gauss–Newton for a linear ``Basis`` (the fast path), L-BFGS for a
        (possibly nonlinear-in-θ) ``PSF``.
    eiv : {"auto", True, False, float}  measurement-noise errors-in-variables
        instrument.  ``"auto"`` (default) resolves to ``True`` for all
        models, interacting ones included (the multi-particle instrument
        uses the same N-body flow as the residual).  ``True`` uses the pure
        η-clean *skip* instrument as the left factor (``w = 1``) — the
        consistent estimator under measurement noise; ``False`` is the plain
        MLE (``w = 0``); a float in ``[0, 1]`` fixes the blend
        ``ψ_left = (1−w)ψ_right + w ψ_inst``.  Active only on the
        Gauss–Newton path (linear ``Basis``); the L-BFGS/PSF path falls
        back to the MLE.
    extra_radius : int  precision-window padding beyond the covariance
        bandwidth (default 1).  Raise to 2–3 in the noise-dominated
        regime β = Λ/(2DΔt) ≫ 1, where the precision of the residual
        process decays slowly (rate λ = 2|ρ|/(1+√(1−4ρ²)) with
        |ρ| = β/(2β+1)) and the default window under-resolves it.

    Returns
    -------
    ForceSolveResult
    """
    F_psf = _as_psf(F)
    inner = _resolve_inner(inner, _is_linear_basis(F))
    eiv = _resolve_eiv_auto(eiv, F_psf)
    dtype = default_float_dtype()
    n_params = int(F_psf.template.size)
    dt0 = _collection_dt(collection)
    d = collection.datasets[0].X.shape[-1]
    theta = _init_theta(theta0, F_psf, n_params, dtype)
    n_sub = _resolve_n_substeps(n_substeps)

    mp = dict(reduce_over_particles=True)

    def build_runs(n_sub, w=0.0):
        # The η-clean instrument (w>0) widens the Gram window; the w=0 cold/clean
        # pass keeps the efficient center MLE window (ODGramProgram decides from w).
        loss_run = _integrator(
            collection,
            ODLossProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator, extra_radius=extra_radius),
            **mp,
        )
        nll_run = _integrator(
            collection,
            ODCondNLLProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator, n_cond=n_cond),
            **mp,
        )
        gram_run = _integrator(
            collection,
            ODGramProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator, w=w, extra_radius=extra_radius),
            **mp,
        )
        return loss_run, nll_run, gram_run

    return _orchestrate(
        build_runs, n_sub, theta, D, Lambda,
        d=d, n_params=n_params, dtype=dtype, max_outer=max_outer,
        inner_maxiter=inner_maxiter, profile_maxiter=profile_maxiter,
        label="od-force", inner=inner,
        noise_init=lambda drop_se: _moment_init(
            collection, dynamics="od", d=d, dtype=dtype, drop_se=drop_se),
        eiv=eiv,
    )


def solve_diffusion_od(
    collection,
    F_psf,
    theta_F,
    D_basis,
    *,
    Lambda,
    theta_D0=None,
    n_substeps=1,
    integrator="rk4",
    n_cond=3,
    maxiter=100,
):
    r"""Infer state-dependent diffusion ``D(x; θ_D)`` with the force fixed.

    Minimises the windowed conditional NLL over ``θ_D`` (the log-det term
    makes the diffusion level identifiable), reusing the same objective and
    integrate engine as the force solve.  ``D_basis`` is a rank-2 Basis or
    PSF; ``Lambda`` (from the force inference) is held fixed.

    Returns
    -------
    DiffusionSolveResult  (``theta_D``)
    """
    F_psf = _as_psf(F_psf)
    D_psf = _as_psf(D_basis)
    dtype = default_float_dtype()
    n_D = int(D_psf.template.size)
    dt0 = _collection_dt(collection)
    theta_F = jnp.asarray(theta_F, dtype=dtype).reshape(-1)
    Lambda = jnp.asarray(Lambda, dtype=dtype)

    if theta_D0 is None:
        theta_D = _init_theta_D(D_psf, collection, dynamics="od", dtype=dtype, n_D=n_D)
    elif isinstance(theta_D0, dict):
        theta_D = jnp.asarray(D_psf.flatten_params(theta_D0), dtype=dtype)
    else:
        theta_D = jnp.asarray(theta_D0, dtype=dtype).reshape(-1)

    prog = ODDiffNLLProgram(
        F_psf, theta_F, D_psf, dt=dt0, n_substeps=n_substeps, integrator=integrator, n_cond=n_cond)
    nll_run = _integrator(collection, prog, reduce_over_particles=True)

    @jax.jit
    def _vg(td):
        return jax.value_and_grad(lambda t: jnp.sum(nll_run((t, Lambda))))(td)

    theta_D, nll_val = _lbfgs(_vg, theta_D, dtype, maxiter)
    return DiffusionSolveResult(theta_D=theta_D, info={"nll": nll_val, "n_D": n_D})


def solve_diffusion_ud(
    collection,
    F_psf,
    theta_F,
    D_basis,
    *,
    Lambda,
    theta_D0=None,
    n_substeps=1,
    integrator="rk4",
    n_cond=4,
    maxiter=100,
):
    r"""Infer state- and velocity-dependent diffusion ``D(x, v; θ_D)`` (UD, force fixed).

    Mirror of :func:`solve_diffusion_od` on the pentadiagonal underdamped
    program: ``D`` is evaluated at the shooting velocity ``(Yₙ, v̂ₙ)``.
    ``D_basis`` is a rank-2 ``needs_v=True`` Basis/PSF.

    Returns
    -------
    DiffusionSolveResult  (``theta_D``)
    """
    F_psf = _as_psf(F_psf)
    D_psf = _as_psf(D_basis)
    dtype = default_float_dtype()
    n_D = int(D_psf.template.size)
    dt0 = _collection_dt(collection)
    theta_F = jnp.asarray(theta_F, dtype=dtype).reshape(-1)
    Lambda = jnp.asarray(Lambda, dtype=dtype)

    if theta_D0 is None:
        theta_D = _init_theta_D(D_psf, collection, dynamics="ud", dtype=dtype, n_D=n_D)
    elif isinstance(theta_D0, dict):
        theta_D = jnp.asarray(D_psf.flatten_params(theta_D0), dtype=dtype)
    else:
        theta_D = jnp.asarray(theta_D0, dtype=dtype).reshape(-1)

    prog = UDDiffNLLProgram(
        F_psf, theta_F, D_psf, dt=dt0, n_substeps=n_substeps, integrator=integrator, n_cond=n_cond)
    nll_run = _integrator(collection, prog, reduce_over_particles=True)

    @jax.jit
    def _vg(td):
        return jax.value_and_grad(lambda t: jnp.sum(nll_run((t, Lambda))))(td)

    theta_D, nll_val = _lbfgs(_vg, theta_D, dtype, maxiter)
    return DiffusionSolveResult(theta_D=theta_D, info={"nll": nll_val, "n_D": n_D})


def solve_force_ud(
    collection,
    F,
    *,
    theta0=None,
    D=None,
    Lambda=None,
    n_substeps=1,
    integrator="rk4",
    max_outer=5,
    inner_maxiter=80,
    n_cond=4,
    profile_maxiter=20,
    inner="auto",
    eiv="auto",
):
    r"""Solve the underdamped parametric force problem.

    Same orchestration as :func:`solve_force_od`, on the bandwidth-2
    (pentadiagonal) underdamped programs: the unobserved velocity is
    resolved by shooting, residuals are 3-point, and the process noise
    enters at ``Δt³``.  ``F`` is a velocity-dependent PSF (``needs_v=True``).
    ``inner`` behaves as in :func:`solve_force_od`.
    ``eiv`` behaves as in :func:`solve_force_od`; the underdamped instrument
    is built from the clean lagged position pair (the velocity-EIV is stronger
    here, since the shooting velocity divides the position noise by ``Δt``).

    Returns
    -------
    ForceSolveResult
    """
    F_psf = _as_psf(F)
    inner = _resolve_inner(inner, _is_linear_basis(F))
    eiv = _resolve_eiv_auto(eiv, F_psf)
    dtype = default_float_dtype()
    n_params = int(F_psf.template.size)
    dt0 = _collection_dt(collection)
    d = collection.datasets[0].X.shape[-1]
    theta = _init_theta(theta0, F_psf, n_params, dtype)
    n_sub = _resolve_n_substeps(n_substeps)

    # Degeneracy guard (underdamped only): a single Euler phase-space step updates
    # position as x + v·dt — independent of the force θ — so the skip-trick EIV
    # instrument ∂Φˣ/∂θ is identically zero and the asymmetric Gram is rank-zero
    # (θ collapses to 0 / explodes).  Disable the instrument (fall back to the MLE)
    # with a warning instead of returning a silent singular fit.  RK4 (the default)
    # or n_substeps≥2 carry θ into the position, so the instrument is well-posed.
    if integrator == "euler" and n_sub == 1 and _resolve_w(eiv) > 0.0:
        import warnings
        warnings.warn(
            "Underdamped skip-trick (eiv) is unavailable for a single Euler step "
            "(integrator='euler', n_substeps=1): the Euler position update does not "
            "depend on the force, so the instrument is degenerate.  Falling back to "
            "eiv=False.  Use integrator='rk4' (the default) or n_substeps>=2 to keep "
            "the skip-trick.",
            RuntimeWarning, stacklevel=2,
        )
        eiv = False

    mp = dict(reduce_over_particles=True)

    def build_runs(n_sub, w=0.0):
        # The η-clean instrument (w>0) widens the Gram window; the w=0 cold/clean
        # pass keeps the efficient center MLE window (UDGramProgram decides from w).
        loss_run = _integrator(
            collection,
            UDLossProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator),
            **mp,
        )
        nll_run = _integrator(
            collection,
            UDCondNLLProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator, n_cond=n_cond),
            **mp,
        )
        gram_run = _integrator(
            collection,
            UDGramProgram(F_psf, dt=dt0, n_substeps=n_sub, integrator=integrator, w=w),
            **mp,
        )
        return loss_run, nll_run, gram_run

    return _orchestrate(
        build_runs, n_sub, theta, D, Lambda,
        d=d, n_params=n_params, dtype=dtype, max_outer=max_outer,
        inner_maxiter=inner_maxiter, profile_maxiter=profile_maxiter,
        label="ud-force", inner=inner,
        noise_init=lambda drop_se: _moment_init(
            collection, dynamics="ud", d=d, dtype=dtype, drop_se=drop_se),
        eiv=eiv,
    )
