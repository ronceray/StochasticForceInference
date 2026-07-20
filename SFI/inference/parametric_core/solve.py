# SFI/inference/parametric_core/solve.py
"""
Orchestration for the parametric-core force and diffusion solves.

Everything runs on the exact banded-innovations core
(:mod:`runner` / :mod:`banded`): per-point flow tensors → block-banded
covariance → reverse-time block-LDLᵀ innovations whitening, giving the
exact Gaussian likelihood, its Gauss–Newton Gram/score, and the
errors-in-variables instrument in one pass.

    flow → banded covariance → exact innovations Gram/NLL  (runner)
    + (D, Λ) profiled on cached fixed-θ tensors
    + Gauss–Newton / frozen-precision-IRLS driver
    → θ̂, D̂, Σ̂_η, Gram (for covariance / sparsity).

``(D, Λ)`` are full symmetric matrices throughout.  Non-uniform (per-step)
dt, state-dependent diffusion, nonlinear-in-θ PSFs, masks, and interacting
multi-particle models are all supported.  (The v2.0 overlapping-window
approximate-precision core was removed after one deprecation release.)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as _sp_minimize

from SFI.utils.maths import default_float_dtype

from .driver import condition_cap_ridge, gn_minimize, irls_minimize
from .gram import unpack_gram

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
    its force is ``F = Σ_a θ_a b_a(x)``, so the whitened Gram is the GLS fixed
    point and the direct Gauss–Newton path applies.  A user-supplied ``PSF`` is
    treated as potentially nonlinear-in-θ (damped Gauss–Newton by default).
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

    Caveat (coarse dt, (near-)clean data): ``"auto"`` does *not* inspect the
    noise level, so the EIV instrument stays on even when there is no
    measurement noise.  At coarse ``dt_eff`` on a nonlinear system the (D, Λ)
    profile can then fit a *spurious* Λ (dt-independent measurement noise is not
    identifiable against the process increment structure ∝Δt — see the
    ``eiv=False`` branch in :func:`_orchestrate`), inflating/skewing D̂ and the
    force fit.  For coarse-``dt_eff`` clean overdamped data pass ``eiv=False``
    (and ``n_substeps>=2``).
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


def _dt_steps(ds):
    """``(T−1,)`` per-interval steps of one dataset (host-side).

    Delegates to :func:`runner._dataset_dt_stream` so the precedence
    (``t`` before ``dt``, matching the runtime streams), the trim of the
    unused trailing entry of length-``T`` per-step arrays, and the
    scalar broadcast (every dataset weighs by its step count) are all
    consistent with what the solvers actually integrate over.
    """
    from .runner import _dataset_dt_stream

    return _dataset_dt_stream(ds, int(np.asarray(ds.X).shape[0]))


def _collection_dt_info(collection):
    """``(dt0, uniform)`` — the representative Δt and whether sampling is
    uniform across all steps and datasets (spread ≤ 1e-9·|dt0|)."""
    flat = np.concatenate([_dt_steps(ds) for ds in collection.datasets])
    dt0 = float(flat[0])
    uniform = float(np.max(np.abs(flat - dt0))) <= 1e-9 * abs(dt0)
    return dt0, uniform


def _as_x64_collection(collection):
    """Collection view with float64 positions/times (call inside x64 scope).

    Only the numerically critical inputs (``X``, ``dt``, ``t``) are cast;
    boolean masks are untouched and extras are left as-is (they promote
    where they mix with float64).  Returns the input unchanged when
    nothing needs casting.
    """
    import dataclasses

    def _cast(x):
        if x is None:
            return x
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.floating) and arr.dtype != jnp.float64:
            return arr.astype(jnp.float64)
        return x

    new_datasets, changed = [], False
    for ds in collection.datasets:
        X, dt, t = _cast(ds.X), _cast(ds.dt), _cast(ds.t)
        if X is ds.X and dt is ds.dt and t is ds.t:
            new_datasets.append(ds)
        else:
            changed = True
            new_datasets.append(dataclasses.replace(ds, X=X, dt=dt, t=t))
    if not changed:
        return collection
    return type(collection)(new_datasets, collection.weights)


def _cast_result_floats(res, dtype):
    """Cast the float arrays of a solve-result dataclass to ``dtype``."""
    import dataclasses

    def _cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    kw = {}
    for f in dataclasses.fields(res):
        v = getattr(res, f.name)
        kw[f.name] = ({k: _cast(vv) for k, vv in v.items()}
                      if isinstance(v, dict) else _cast(v))
    return dataclasses.replace(res, **kw)


def _run_in_x64(solver):
    """Compute a parametric solve in float64, report in the session dtype.

    The banded Cholesky factors and the Gram accumulation over ~1e5-1e6
    residuals are unreliable in float32 (the relative jitter floor is
    below float32 epsilon), so the solve enters a scoped
    ``jax.enable_x64()`` and upcasts the collection's float arrays.  The
    returned arrays are cast back to the *session* default float dtype:
    a float32 session keeps a numerically consistent float32 world
    downstream (mixed-dtype ``lax`` calls reject float64 there), while
    the arithmetic that produced the numbers was float64 throughout.
    The session-level dtype configuration is untouched.
    """
    import functools

    @functools.wraps(solver)
    def wrapper(collection, *args, **kwargs):
        out_dtype = default_float_dtype()
        with jax.enable_x64():
            res = solver(_as_x64_collection(collection), *args, **kwargs)
        if out_dtype == jnp.float64:
            return res
        return _cast_result_floats(res, out_dtype)

    return wrapper


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
    ``compute_diffusion_constant``.  The exact banded NLL then
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
        from .runner import _dataset_dt_stream

        dt_arr = jnp.asarray(_dataset_dt_stream(collection.datasets[0],
                                                int(X.shape[0])))
        idx = np.linspace(0, X.shape[0] - 2, min(64, X.shape[0] - 1)).astype(int)
        Xs = X[idx]
        Vs = ((X[idx + 1] - X[idx]) / dt_arr[idx][:, None]
              if dynamics == "ud" else None)
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


def _resolve_core(_core):
    """Validate the internal precision-core selector.

    The parametric estimators run exclusively on the banded-innovations
    **exact** core (:mod:`banded` / :mod:`runner`).
    """
    import os

    core = _core or os.environ.get("SFI_PARAMETRIC_CORE", "exact")
    if core != "exact":
        raise ValueError(f"unknown parametric core {core!r} (the only "
                         "core is 'exact')")
    return core


def _warn_inert_kwargs(**kw):
    """``extra_radius`` / ``n_cond`` are deprecated no-ops (the banded
    likelihood has no window truncation to tune); a non-default value
    warns that it is ignored."""
    changed = [k for k, (val, default) in kw.items() if val != default]
    if changed:
        import warnings

        verb = "is" if len(changed) == 1 else "are"
        warnings.warn(
            f"{', '.join(changed)} {verb} deprecated and {'has' if len(changed) == 1 else 'have'} "
            "no effect on the parametric estimator (its banded likelihood "
            "has no window truncation to tune)",
            DeprecationWarning, stacklevel=3)


def _exact_build_runs(collection, F_psf, *, dynamics, integrator,
                      chunk_target_bytes, uniform=True):
    """``build_runs(n_sub, w)`` for the exact banded core (GN path only).

    The runner derives per-interval Δt streams per dataset (``ds.t`` /
    ``ds.dt``), so uniform and non-uniform sampling share one path.  The
    Stage-3 block upgrades (Lyapunov-exact process covariance;
    convexity-corrected residual mean) default ON and can be disabled for
    A/B with ``SFI_EXACT_UPGRADES=0``.
    """
    import os

    from .runner import make_exact_runs_od, make_exact_runs_ud

    make = make_exact_runs_od if dynamics == "od" else make_exact_runs_ud
    upgrades = os.environ.get("SFI_EXACT_UPGRADES", "1") != "0"

    if dynamics == "ud" and not uniform and not upgrades:
        # the leading-order (4/3, 1/3)Δt³ UD blocks assume equal adjacent
        # intervals; the Lyapunov-exact blocks integrate each interval and
        # are correct under non-uniform sampling.
        raise ValueError(
            "non-uniform dt on underdamped dynamics requires the "
            "Lyapunov-exact process covariance; do not set "
            "SFI_EXACT_UPGRADES=0 (or lyapunov=False) for irregular UD data")

    d = collection.datasets[0].X.shape[-1]
    nsym = d * (d + 1) // 2

    def build_runs(n_sub, w=0.0):
        runs = make(collection, F_psf, dt=None, n_substeps=n_sub,
                    integrator=integrator, w=w,
                    chunk_target_bytes=chunk_target_bytes,
                    lyapunov=upgrades,
                    convexity=("auto" if upgrades else False))

        def gram_run(params):
            th, D_, Se_ = params
            return runs.gram(th, D_, Se_)

        def nll_run(params):
            th, D_, Se_ = params
            return runs.nll(th, D_, Se_)

        def loss_run(params):
            # frozen-precision IRLS quadratic (quasi-score semantics: the
            # covariance blocks are held at θ_frozen so the fixed point is
            # the same estimating-equation root as the GN path)
            tl, tf, D_, Se_ = params
            return runs.loss(tl, tf, D_, Se_)

        def profile_builder(th, D_cur, Se_cur, drop_se):
            # cache the Phase-A tensors once at the fitted θ; the convexity
            # correction (active on the w>0 path) is frozen at the profile
            # entry Λ — the final Gram re-evaluates it at the profiled Λ.
            conv = Se_cur if runs._convexity else None
            runs.prepare(th, conv_lambda=conv)
            if drop_se:
                def make_mats(z):
                    return (_chol_vec_to_mat(z, d),
                            jnp.zeros((d, d), dtype=z.dtype))
            else:
                def make_mats(z):
                    return (_chol_vec_to_mat(z[:nsym], d),
                            _chol_vec_to_mat(z[nsym:], d))
            return runs.profile_nll_vg(make_mats)

        return loss_run, nll_run, gram_run, profile_builder

    return build_runs


def _orchestrate(build_runs, n_sub, theta, D, Lambda, *,
                 d, n_params, dtype, max_outer, inner_maxiter, profile_maxiter,
                 label, inner, noise_init, gram_cond_max=1e10, eiv=False):
    """Shared solve + (D,Λ) profiling + Gram.

    Dynamics-agnostic.  ``build_runs(n_substeps) -> (loss_run, nll_run,
    gram_run)`` constructs the integrate-engine runners at the fixed substep
    count.

    The inner solve is ``inner="gn"`` (Gauss–Newton on the whitened Gram —
    direct for linear-in-θ, damped for nonlinear-in-θ) or ``inner="lbfgs"``
    (frozen-precision L-BFGS IRLS — the explicit overdamped route).
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
        loss_run, nll_run, gram_run = runs[0], runs[1], runs[2]
        # optional 4th element (exact core): a fixed-θ profile builder that
        # caches the Phase-A tensors once and returns a vg(z) evaluating the
        # cached NLL — zero basis evaluations per profile iteration.
        profile_builder = runs[3] if len(runs) > 3 else None

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
            if profile_builder is not None:
                vg = profile_builder(th, D_cur, Se_cur, drop_se_mle)
            if not drop_se_mle:
                z0 = jnp.concatenate([_mat_to_chol_vec(D_cur, d), _mat_to_chol_vec(Se_cur, d)])
                vg_full = vg if profile_builder is not None \
                    else (lambda z: _profile_vg(z, th))
                z, _ = _lbfgs(vg_full, z0, dtype, profile_maxiter,
                              ftol=_PROFILE_FTOL, gtol=_PROFILE_GTOL)
                return _chol_vec_to_mat(z[:nsym], d), _chol_vec_to_mat(z[nsym:], d)
            zD0 = _mat_to_chol_vec(D_cur, d)
            vg_D = vg if profile_builder is not None \
                else (lambda z: _profile_D_vg(z, th))
            zD, _ = _lbfgs(vg_D, zD0, dtype, profile_maxiter,
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
        # The exact banded NLL refines it once at the fitted θ (reprofile).
        D0, Lambda0 = noise_init(drop_se_mle)
        prof = mach[0]

    theta, D_hat, Lambda_hat, sinfo = _solve_pass(mach, theta, D0, Lambda0, prof)
    info = {**sinfo, "n_substeps": n_sub, "inner": inner, "eiv_w": w}

    gram_run = runs[2]
    G_raw, f, H, _ = unpack_gram(gram_run((theta, D_hat, Lambda_hat)), n_params)
    # Parameter covariance: sandwich G⁻¹ H G⁻ᵀ.  On the symmetric path
    # H = G and this is the usual inverse information; on the IV path
    # (asymmetric G) the sandwich is the correct asymptotic covariance —
    # G⁻¹ alone would mis-state the error bars under measurement noise.
    # stable_pinv keeps the covariance finite for a rank-deficient Gram
    # (collinear basis) where a raw inverse returns garbage.
    from SFI.utils.maths import stable_pinv

    Ginv = stable_pinv(G_raw)
    theta_cov = Ginv @ H @ Ginv.T
    theta_cov = 0.5 * (theta_cov + theta_cov.T)
    G = 0.5 * (G_raw + G_raw.T)
    return ForceSolveResult(theta=theta, D=D_hat, Lambda=Lambda_hat, G=G, f=f,
                            theta_cov=theta_cov, info=info)


@_run_in_x64
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
    profile_maxiter=None,
    inner="auto",
    eiv="auto",
    extra_radius=1,
    chunk_target_bytes=None,
    _core=None,
):
    r"""Solve the overdamped parametric force problem.

    Runs on the exact banded-innovations core (block-banded likelihood
    whitened exactly; per-step non-uniform dt supported).

    Parameters
    ----------
    collection : TrajectoryCollection
    F : PSF or Basis  (Basis is converted with ``to_psf()``).
    theta0 : dict, array, or None  initial parameters (default zeros).
    D, Lambda : ``(d, d)`` or None.  When given, held fixed; otherwise
        profiled: moment-estimator init (Vestergaard/ULI + Λ), then one
        exact banded-NLL refinement at the fitted θ (cached fixed-θ
        tensors — zero basis evaluations per profile iteration).
    n_substeps : int  RK4/Euler micro-steps per Δt (default 1 — the
        single-step minimal estimator; with per-step dt the substep is
        ``h_k = dt_k / n_substeps``).
    integrator : {"rk4", "euler"}  Flow predictor.  Default ``"rk4"`` (a single
        RK4 step — the minimal estimator).  A single ``"euler"`` step is also
        available, but note it cannot carry the force into the position update
        over one step, so the underdamped skip-trick (``eiv``) is unavailable
        for ``integrator="euler", n_substeps=1`` (auto-disabled with a warning).
    max_outer, inner_maxiter : solve budget.
    n_cond : int  deprecated, no effect (``DeprecationWarning`` when
        set).
    inner : {"auto", "gn", "lbfgs"}  inner solver.  ``"auto"`` →
        Gauss–Newton (direct for a linear ``Basis``, damped for a
        nonlinear ``PSF``); ``"lbfgs"`` forces the frozen-precision
        quadratic route.
    eiv : {"auto", True, False, float}  measurement-noise errors-in-variables
        instrument.  ``"auto"`` (default) resolves to ``True`` for all
        models, interacting ones included (the multi-particle instrument
        uses the same N-body flow as the residual).  ``True`` uses the pure
        η-clean *skip* instrument as the left factor (``w = 1``) — the
        consistent estimator under measurement noise; ``False`` is the plain
        MLE (``w = 0``); a float in ``[0, 1]`` fixes the blend
        ``ψ_left = (1−w)ψ_right + w ψ_inst``.  Active on the Gauss–Newton
        paths (including damped-GN PSFs); the explicit ``inner="lbfgs"``
        route falls back to the MLE.  Note ``"auto"`` does not inspect
        the noise level: for (near-)clean data at coarse effective
        sampling pass ``eiv=False`` (see :func:`_resolve_eiv_auto`).
    extra_radius : int  deprecated, no effect (``DeprecationWarning``
        when set).
    chunk_target_bytes : int or None
        Working-set budget per integration chunk (the engines pass
        ``max_memory_gb`` down through this); ``None`` keeps the engine
        default.

    Returns
    -------
    ForceSolveResult
    """
    F_psf = _as_psf(F)
    _resolve_core(_core)                 # validate the core selector
    inner_arg = inner
    inner = _resolve_inner(inner, _is_linear_basis(F))
    if inner_arg == "auto" and inner == "lbfgs":
        # the exact Gram is PSF-capable (θ-recursion with an AD fallback
        # for genuine θ-dependence), so nonlinear models default to the
        # damped Gauss–Newton path — which also keeps the EIV instrument
        # active for PSFs.  ``gn_minimize`` detects the sensitivity-collapse
        # runaway (‖ψᵀPψ‖ → 0) that a badly-conditioned nonlinear model can
        # fall into on the EIV/phi merit, and the block below falls back to
        # L-BFGS when it fires — so the default is EIV-when-safe, robust
        # otherwise.
        inner = "gn"
    eiv = _resolve_eiv_auto(eiv, F_psf)
    _warn_inert_kwargs(extra_radius=(extra_radius, 1),
                       n_cond=(n_cond, 3))
    if profile_maxiter is None:
        # Cached profile iterations are cheap, and the exact NLL is
        # stiff (its log-det spans the whole sequence): the historical
        # 20-iteration budget could stall far from the optimum on
        # badly-scaled (D, Λ) landscapes (measured: the flocking bench
        # needs ~150).  Explicit values always win.
        profile_maxiter = 200
    dtype = default_float_dtype()
    n_params = int(F_psf.template.size)
    d = collection.datasets[0].X.shape[-1]
    theta = _init_theta(theta0, F_psf, n_params, dtype)
    n_sub = _resolve_n_substeps(n_substeps)

    build_runs = _exact_build_runs(
        collection, F_psf, dynamics="od", integrator=integrator,
        chunk_target_bytes=chunk_target_bytes,
        uniform=_collection_dt_info(collection)[1])

    def _run(inner_):
        return _orchestrate(
            build_runs, n_sub, theta, D, Lambda,
            d=d, n_params=n_params, dtype=dtype, max_outer=max_outer,
            inner_maxiter=inner_maxiter, profile_maxiter=profile_maxiter,
            label="od-force", inner=inner_,
            noise_init=lambda drop_se: _moment_init(
                collection, dynamics="od", d=d, dtype=dtype, drop_se=drop_se),
            eiv=eiv,
        )

    out = _run(inner)
    # Auto-routed Gauss–Newton whose sensitivities collapsed on the EIV/phi
    # merit (a badly-conditioned nonlinear PSF: saturating MLP, alignment
    # gain → ∞) — gn_minimize flags ``diverged``.  Fall back to the bounded
    # frozen-precision L-BFGS (the true-NLL descent) instead of returning the
    # spurious-root fit.  An explicit inner="gn" is honoured as given; the
    # well-conditioned nonlinear PSFs that GN handles never trip the guard.
    if inner_arg == "auto" and inner == "gn" and out.info.get("diverged"):
        import warnings

        warnings.warn(
            "Gauss-Newton diverged on this nonlinear model (its sensitivities "
            "collapsed toward a spurious |theta|->inf root on the EIV "
            "instrument); falling back to the frozen-precision L-BFGS solver. "
            "This drops the instrument to plain MLE — pass eiv=False to "
            "silence, or inner='gn' to force GN.",
            RuntimeWarning, stacklevel=3)
        out = _run("lbfgs")
    return out


@_run_in_x64
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
    chunk_target_bytes=None,
    _core=None,
):
    r"""Infer state-dependent diffusion ``D(x; θ_D)`` with the force fixed.

    Minimises the conditional NLL over ``θ_D`` (the log-det term makes the
    diffusion level identifiable), reusing the same objective and integrate
    engine as the force solve.  ``D_basis`` is a rank-2 Basis or PSF;
    ``Lambda`` (from the force inference) is held fixed.

    The objective is the exact banded NLL on cached fixed-θ tensors — zero
    force/basis evaluations per L-BFGS iteration, per-step (non-uniform)
    dt supported (``n_cond`` is deprecated and ignored).

    Returns
    -------
    DiffusionSolveResult  (``theta_D``)
    """
    F_psf = _as_psf(F_psf)
    D_psf = _as_psf(D_basis)
    dtype = default_float_dtype()
    n_D = int(D_psf.template.size)
    _resolve_core(_core)                 # validate the core selector
    theta_F = jnp.asarray(theta_F, dtype=dtype).reshape(-1)
    Lambda = jnp.asarray(Lambda, dtype=dtype)

    if theta_D0 is None:
        theta_D = _init_theta_D(D_psf, collection, dynamics="od", dtype=dtype, n_D=n_D)
    elif isinstance(theta_D0, dict):
        theta_D = jnp.asarray(D_psf.flatten_params(theta_D0), dtype=dtype)
    else:
        theta_D = jnp.asarray(theta_D0, dtype=dtype).reshape(-1)

    import os

    from .runner import make_exact_runs_od

    upgrades = os.environ.get("SFI_EXACT_UPGRADES", "1") != "0"
    runs = make_exact_runs_od(
        collection, F_psf, dt=None, n_substeps=n_substeps,
        integrator=integrator, w=0.0,
        chunk_target_bytes=chunk_target_bytes,
        lyapunov=upgrades, convexity=False)
    runs.prepare(theta_F, with_base=True)
    vg = runs.diffusion_nll_vg(D_psf, Lambda)
    theta_D, nll_val = _lbfgs(vg, theta_D, dtype, maxiter)
    return DiffusionSolveResult(theta_D=theta_D,
                                info={"nll": nll_val, "n_D": n_D})


@_run_in_x64
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
    chunk_target_bytes=None,
    _core=None,
):
    r"""Infer state- and velocity-dependent diffusion ``D(x, v; θ_D)`` (UD, force fixed).

    Mirror of :func:`solve_diffusion_od` on the pentadiagonal underdamped
    program: ``D`` is evaluated at the shooting velocity ``(Yₙ, v̂ₙ)``.
    ``D_basis`` is a rank-2 ``needs_v=True`` Basis/PSF.  Behaves as in
    :func:`solve_diffusion_od`; with the
    Lyapunov upgrade the per-point ``D`` enters the exact per-interval
    process covariance instead of the leading-order ``(4/3, 1/3)Δt³``
    blocks (required for non-uniform dt).

    Returns
    -------
    DiffusionSolveResult  (``theta_D``)
    """
    F_psf = _as_psf(F_psf)
    D_psf = _as_psf(D_basis)
    dtype = default_float_dtype()
    n_D = int(D_psf.template.size)
    _resolve_core(_core)                 # validate the core selector
    dt_uniform = _collection_dt_info(collection)[1]
    theta_F = jnp.asarray(theta_F, dtype=dtype).reshape(-1)
    Lambda = jnp.asarray(Lambda, dtype=dtype)

    if theta_D0 is None:
        theta_D = _init_theta_D(D_psf, collection, dynamics="ud", dtype=dtype, n_D=n_D)
    elif isinstance(theta_D0, dict):
        theta_D = jnp.asarray(D_psf.flatten_params(theta_D0), dtype=dtype)
    else:
        theta_D = jnp.asarray(theta_D0, dtype=dtype).reshape(-1)

    import os

    from .runner import make_exact_runs_ud

    upgrades = os.environ.get("SFI_EXACT_UPGRADES", "1") != "0"
    if not dt_uniform and not upgrades:
        raise ValueError(
            "non-uniform dt on underdamped dynamics requires the "
            "Lyapunov-exact process covariance; do not set "
            "SFI_EXACT_UPGRADES=0 for irregular UD data")
    runs = make_exact_runs_ud(
        collection, F_psf, dt=None, n_substeps=n_substeps,
        integrator=integrator, w=0.0,
        chunk_target_bytes=chunk_target_bytes,
        lyapunov=upgrades, convexity=False)
    runs.prepare(theta_F, with_base=True)
    vg = runs.diffusion_nll_vg(D_psf, Lambda)
    theta_D, nll_val = _lbfgs(vg, theta_D, dtype, maxiter)
    return DiffusionSolveResult(theta_D=theta_D,
                                info={"nll": nll_val, "n_D": n_D})


@_run_in_x64
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
    profile_maxiter=None,
    inner="auto",
    eiv="auto",
    chunk_target_bytes=None,
    _core=None,
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
    _resolve_core(_core)                 # validate the core selector
    inner_arg = inner
    inner = _resolve_inner(inner, _is_linear_basis(F))
    if inner == "lbfgs":
        if inner_arg != "auto":
            # The underdamped frozen-precision quadratic is ill-behaved
            # under exact whitening: the shooting velocity re-solved at
            # θ_live lets over-damped models shrink the whitened residuals
            # monotonically (no tame IRLS fixed point near the quasi-score
            # root).  The damped Gauss–Newton path solves the same
            # estimating equation self-consistently and handles
            # nonlinear-in-θ PSFs.
            import warnings

            warnings.warn(
                "inner='lbfgs' is unavailable for underdamped solves on the "
                "exact core (the frozen-precision objective is unbounded "
                "along the damping direction); using the damped Gauss-Newton "
                "path.", RuntimeWarning, stacklevel=2)
        inner = "gn"
    eiv = _resolve_eiv_auto(eiv, F_psf)
    _warn_inert_kwargs(n_cond=(n_cond, 4))
    if profile_maxiter is None:
        profile_maxiter = 200            # see solve_force_od
    dtype = default_float_dtype()
    n_params = int(F_psf.template.size)
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

    build_runs = _exact_build_runs(
        collection, F_psf, dynamics="ud", integrator=integrator,
        chunk_target_bytes=chunk_target_bytes,
        uniform=_collection_dt_info(collection)[1])

    return _orchestrate(
        build_runs, n_sub, theta, D, Lambda,
        d=d, n_params=n_params, dtype=dtype, max_outer=max_outer,
        inner_maxiter=inner_maxiter, profile_maxiter=profile_maxiter,
        label="ud-force", inner=inner,
        noise_init=lambda drop_se: _moment_init(
            collection, dynamics="ud", d=d, dtype=dtype, drop_se=drop_se),
        eiv=eiv,
    )
