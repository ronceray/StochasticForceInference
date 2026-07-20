"""Overdamped-vs-underdamped dynamics classifier.

Reads raw trajectory data and decides whether the dynamics are overdamped
(OD, first-order Langevin) or underdamped (UD, second-order / inertial),
robust to high localization noise and coarse sampling.  Entry point:
:func:`classify_dynamics`.

The discriminator is the lag-resolved displacement covariance
``C_k(dt) = <Delta x_t . Delta x_{t+k}>`` with ``Delta x_t = x_{t+1} - x_t``:

* White localization noise (variance ``sigma^2`` per axis) contributes ``+2
  sigma^2`` to ``C0``, ``-sigma^2`` to ``C1`` and **exactly 0 to ``C_{k>=2}``**
  — so lag >= 2 is measurement-noise-immune by construction.
* A force field cannot fake inertia: for OD the apparent kinetic energy
  ``C0/dt^2`` diverges as ``2D/dt`` while the lag-2 velocity correlation stays
  finite, so the noise-corrected ratio ``rho2 = C2 / (C0 + 2 C1)`` (the
  combination ``C0 + 2 C1`` cancels white localization noise exactly) tends to
  0; for UD the displacement is ballistic (``Delta x ~ v dt``) so ``rho2`` rises
  to a positive plateau.  Scanning ``dt`` (via
  :meth:`TrajectoryCollection.degrade`) removes the force confound; using lag 2
  removes the noise.

The pipeline (see :func:`classify_dynamics`) layers a model-free
``dt``-scaling test (:func:`_scaling_statistics`), a parametric covariance fit
(:func:`_fit_dynamics`) and an overdamped-fit residual-autocorrelation
cross-check (:func:`_cross_check`) into an OD / UD / inconclusive verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from SFI.diagnostics.report import _to_jsonable
from SFI.integrate import Integrand, Term, TimeOperand, integrate, timeop

# --------------------------------------------------------------------------- #
# Lag-0/1/2 displacement-covariance time-operators.
#
# Each returns a per-particle (..., N, d, d) tensor, mirroring the diffusion /
# measurement-noise estimators in SFI.inference.overdamped (``_Lambda`` etc.).
# The dataset stream layer supplies the increments:
#   dX        = X[t+1] - X[t]      (offset 0..+1)
#   dX_minus  = X[t]   - X[t-1]    (offset -1..0)
#   dX_plus   = X[t+2] - X[t+1]    (offset +1..+2)
# so <dX . dX_minus> is the lag-1 covariance and <dX_minus . dX_plus> is the
# lag-2 covariance (displacements one step apart).
# --------------------------------------------------------------------------- #


@timeop(name="dyn_cov_lag0", batch_safe=True)
def _cov_lag0(**streams):
    """Lag-0 displacement covariance ``dX (x) dX``."""
    dX = streams["dX"]
    return _outer(dX, dX)


@timeop(name="dyn_cov_lag1", batch_safe=True)
def _cov_lag1(**streams):
    """Symmetrised lag-1 covariance ``1/2 (dX (x) dX^- + dX^- (x) dX)``."""
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    return 0.5 * (_outer(dX, dXm) + _outer(dXm, dX))


@timeop(name="dyn_cov_lag2", batch_safe=True)
def _cov_lag2(**streams):
    """Symmetrised lag-2 covariance ``1/2 (dX^- (x) dX^+ + dX^+ (x) dX^-)``."""
    dXm = streams["dX_minus"]
    dXp = streams["dX_plus"]
    return 0.5 * (_outer(dXm, dXp) + _outer(dXp, dXm))


_cov_lag0._requires = frozenset({"dX"})  # type: ignore[attr-defined]
_cov_lag1._requires = frozenset({"dX", "dX_minus"})  # type: ignore[attr-defined]
_cov_lag2._requires = frozenset({"dX_minus", "dX_plus"})  # type: ignore[attr-defined]


def _outer(a, b):
    """Outer product over the trailing spatial axis: ``a_m b_n``."""
    return jnp.einsum("...m,...n->...mn", a, b)


# --------------------------------------------------------------------------- #
# Covariance backbone
# --------------------------------------------------------------------------- #
def _integrate_tensor(coll, op, alias):
    """Time- and particle-averaged ``(d, d)`` value of a covariance time-op."""
    operand = TimeOperand(op, alias=alias)
    prog = Integrand(times=[operand], terms=[Term(eq="imn->imn", ops=(alias,))])
    return np.asarray(integrate(coll, prog, reduce="mean"))


def _mean_dt(coll):
    """Representative scalar step (uniform within a degraded collection)."""
    dts = []
    for ds in coll.datasets:
        dt = getattr(ds, "dt", None)
        if dt is None:
            t = getattr(ds, "t", None)
            if t is None:
                continue
            dt = np.diff(np.asarray(t))
        dts.append(float(np.mean(np.asarray(dt, dtype=float))))
    if not dts:
        raise ValueError("Cannot determine dt: provide dt or t on the trajectories.")
    return float(np.mean(dts))


def _n_eff(coll, require):
    """Number of valid (masked) increment samples available for ``require``."""
    n = 0
    for ds in coll.datasets:
        idx = ds.valid_indices(frozenset(require))
        n += int(np.asarray(idx).size) * int(ds.N)
    return n


def _increment_covariances(coll):
    """Lag-0/1/2 displacement covariances of ``coll`` at its native ``dt``.

    Parameters
    ----------
    coll : TrajectoryCollection
        Trajectories sharing a (uniform) time step.

    Returns
    -------
    dict
        ``C0``, ``C1``, ``C2`` : ``(d, d)`` displacement covariance tensors;
        ``dt`` : representative scalar step; ``n_eff`` : valid sample count.
    """
    C0 = _integrate_tensor(coll, _cov_lag0, "c0")
    C1 = _integrate_tensor(coll, _cov_lag1, "c1")
    C2 = _integrate_tensor(coll, _cov_lag2, "c2")
    return {
        "C0": C0,
        "C1": C1,
        "C2": C2,
        "dt": _mean_dt(coll),
        "n_eff": _n_eff(coll, {"dX_minus", "dX_plus"}),
    }


def _covariances_vs_dt(coll, strides):
    """Lag-0/1/2 covariances across a set of coarse-graining strides.

    Each stride ``s`` coarse-grains the trajectories via
    :meth:`TrajectoryCollection.degrade` (effective step ``s * dt``), then the
    lag-0/1/2 covariances are recomputed.  This is the dt-scan that separates
    the overdamped force confound (vanishes as ``dt -> 0``) from genuine
    inertia (saturates).

    Parameters
    ----------
    coll : TrajectoryCollection
    strides : sequence of int
        Positive coarse-graining factors (``1`` keeps the native step).

    Returns
    -------
    dict
        ``strides``, ``dt`` : ``(S,)`` arrays; ``C0``, ``C1``, ``C2`` :
        ``(S, d, d)`` covariance tensors; ``n_eff`` : ``(S,)`` sample counts.
    """
    strides = [int(s) for s in strides]
    dts, C0s, C1s, C2s, n_effs = [], [], [], [], []
    for s in strides:
        sub = coll if s == 1 else coll.degrade(downsample=s)
        cov = _increment_covariances(sub)
        dts.append(cov["dt"])
        C0s.append(cov["C0"])
        C1s.append(cov["C1"])
        C2s.append(cov["C2"])
        n_effs.append(cov["n_eff"])
    return {
        "strides": np.asarray(strides),
        "dt": np.asarray(dts),
        "C0": np.stack(C0s),
        "C1": np.stack(C1s),
        "C2": np.stack(C2s),
        "n_eff": np.asarray(n_effs),
    }


# --------------------------------------------------------------------------- #
# Parametric forward model for the lag-0/1/2 increment covariances
# --------------------------------------------------------------------------- #
def _dynamics_model(h, sigma2, D, V, gamma):
    r"""Predicted scalar (per-axis) lag-0/1/2 increment covariances.

    Three physical contributions to ``c_k(h) = Cov(dx_t, dx_{t+k})`` for step
    ``h``:

    * **Localization noise** ``sigma^2``: ``+2 sigma^2`` at lag 0, ``-sigma^2``
      at lag 1, ``0`` at lag 2 (white, dt-independent).
    * **Overdamped diffusion** ``D``: ``+2 D h`` at lag 0, ``0`` elsewhere.
    * **Inertia** — an Ornstein--Uhlenbeck velocity with variance ``V = <v^2>``
      and relaxation rate ``gamma`` (so ``tau_v = 1/gamma``).  Integrating
      ``<v(0) v(t)> = V e^{-gamma|t|}`` over the sampling cells gives

      .. math::

         c_0^{inert} &= \tfrac{2V}{\gamma^2}(\gamma h - 1 + e^{-\gamma h}), \\
         c_m^{inert} &= \tfrac{2V}{\gamma^2}(\cosh\gamma h - 1)\,
                        e^{-m\,\gamma h} \quad (m \ge 1).

      As ``gamma h -> 0`` this is ballistic (``c_k -> V h^2``); the lag-2 term
      decays as ``e^{-2 gamma h}``.

    Parameters
    ----------
    h : float or array
        Sampling step(s).
    sigma2, D, V, gamma : float
        Localization-noise variance, diffusion constant, velocity variance,
        and velocity relaxation rate (``gamma > 0``).

    Returns
    -------
    (c0, c1, c2) : arrays shaped like ``h``.
    """
    h = np.asarray(h, dtype=float)
    gh = gamma * h
    inv_g2 = 1.0 / (gamma * gamma)
    em1 = -np.expm1(-gh)  # = 1 - exp(-gamma h), in [0, 1]; overflow-free
    # (cosh(gamma h) - 1) exp(-gamma h) = (1 - exp(-gamma h))^2 / 2, so:
    g0 = 2.0 * inv_g2 * (gh + np.expm1(-gh))  # = (2/g^2)(gamma h - 1 + e^{-gamma h})
    g1 = inv_g2 * em1 * em1
    g2 = np.exp(-gh) * inv_g2 * em1 * em1
    c0 = 2.0 * sigma2 + 2.0 * D * h + V * g0
    c1 = -sigma2 + V * g1
    c2 = V * g2 + np.zeros_like(h)
    return c0, c1, c2


# --------------------------------------------------------------------------- #
# Parametric fit of {D, sigma^2, V, gamma} to the covariance scan
# --------------------------------------------------------------------------- #
def _scalar_lags(scan):
    """Isotropic per-axis scalar lags ``c_k = Tr(C_k) / d`` from a scan."""
    d = scan["C0"].shape[-1]
    c0 = np.einsum("sii->s", scan["C0"]) / d
    c1 = np.einsum("sii->s", scan["C1"]) / d
    c2 = np.einsum("sii->s", scan["C2"]) / d
    return c0, c1, c2


def _aicc(ssr, m, k):
    """Small-sample-corrected AIC for a Gaussian least-squares fit."""
    if m <= 0 or ssr <= 0:
        ssr = max(ssr, 1e-300)
    aic = m * np.log(ssr / m) + 2.0 * k
    denom = m - k - 1
    if denom > 0:
        aic += 2.0 * k * (k + 1) / denom
    return aic


def _lsq_stderr(res, m, n):
    """Parameter standard errors from a ``least_squares`` result."""
    J = np.asarray(res.jac, dtype=float)
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JTJ)
    dof = max(m - n, 1)
    s2 = 2.0 * res.cost / dof
    return np.sqrt(np.clip(np.diag(cov) * s2, 0.0, None))


def _fit_dynamics(scan):
    """Fit the diffusion + inertia + localization model to a covariance scan.

    Fits the four-parameter model ``{sigma^2, D, V, gamma}`` and the nested
    overdamped null ``V = 0`` (``{sigma^2, D}``), and compares them with the
    small-sample-corrected AIC.

    Returns
    -------
    dict
        ``params`` (``sigma2, D, V, gamma``), ``tau_v = 1/gamma``, ``stderr``
        (per parameter), ``V_z`` (``V / stderr_V``), ``aicc_full``,
        ``aicc_od``, ``delta_aicc`` (``aicc_od - aicc_full``; ``> 0`` favours
        the inertial model), and the raw ``ssr_full`` / ``ssr_od``.
    """
    h = np.asarray(scan["dt"], dtype=float)
    c0, c1, c2 = _scalar_lags(scan)
    n_eff = np.asarray(scan["n_eff"], dtype=float)
    w = np.sqrt(np.clip(n_eff, 1.0, None))

    data = np.concatenate([c0, c1, c2])
    wvec = np.concatenate([w, w, w])
    scale = max(float(np.median(np.abs(c0))), 1e-12)

    def _residuals(sigma2, D, V, gamma):
        m0, m1, m2 = _dynamics_model(h, sigma2, D, V, gamma)
        model = np.concatenate([m0, m1, m2])
        return wvec * (model - data) / scale

    # --- initial guesses from the lag structure ---------------------------- #
    gamma_lo = 1e-3 / float(h.max())
    gamma_hi = 1e3 / float(h.min())
    V0 = max(c2[0] / h[0] ** 2, 1e-6)
    s20 = max(-c1[0] + V0 * h[0] ** 2, 1e-9)
    g_init = min(max(1.0 / float(np.median(h)), gamma_lo * 1.01), gamma_hi * 0.99)
    g0_tail = (2.0 / g_init**2) * (g_init * h[-1] - 1.0 + np.exp(-g_init * h[-1]))
    D0 = max((c0[-1] - 2.0 * s20 - V0 * g0_tail) / (2.0 * h[-1]), 1e-6)

    res_full = least_squares(
        lambda p: _residuals(*p),
        x0=[s20, D0, V0, g_init],
        bounds=([0.0, 0.0, 0.0, gamma_lo], [np.inf, np.inf, np.inf, gamma_hi]),
        method="trf",
        max_nfev=5000,
    )
    sigma2, D, V, gamma = (float(x) for x in res_full.x)

    res_od = least_squares(
        lambda p: _residuals(p[0], p[1], 0.0, 1.0),
        x0=[s20, D0],
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        method="trf",
        max_nfev=5000,
    )

    m = data.size
    ssr_full = 2.0 * float(res_full.cost)
    ssr_od = 2.0 * float(res_od.cost)
    aicc_full = _aicc(ssr_full, m, 4)
    aicc_od = _aicc(ssr_od, m, 2)
    stderr = _lsq_stderr(res_full, m, 4)
    V_stderr = float(stderr[2]) if stderr[2] > 0 else 0.0
    V_z = V / V_stderr if V_stderr > 0 else np.inf

    return {
        "params": {"sigma2": sigma2, "D": D, "V": V, "gamma": gamma},
        "tau_v": 1.0 / gamma,
        "stderr": {"sigma2": float(stderr[0]), "D": float(stderr[1]), "V": V_stderr, "gamma": float(stderr[3])},
        "V_z": V_z,
        "ssr_full": ssr_full,
        "ssr_od": ssr_od,
        "aicc_full": aicc_full,
        "aicc_od": aicc_od,
        "delta_aicc": aicc_od - aicc_full,
    }


# --------------------------------------------------------------------------- #
# Model-free scaling statistics (Layer 1) and verdict
# --------------------------------------------------------------------------- #
def _scaling_statistics(scan):
    r"""Noise-immune lag-2 persistence and apparent-KE scaling across ``dt``.

    The Vestergaard combination ``c0 + 2 c1`` cancels white localization noise
    exactly (``2 sigma^2 + 2(-sigma^2) = 0``), leaving the noise-free variance.
    From it:

    * ``rho2 = c2 / (c0 + 2 c1)`` — noise-immune normalized lag-2 correlation;
      ``-> 0`` for overdamped (a force only adds an ``O(h^2)`` piece to a
      variance that grows like ``h``) and to a positive plateau for inertia.
    * ``K = (c0 + 2 c1) / h^2`` — noise-corrected apparent kinetic energy;
      log-log slope ``beta -> -1`` (overdamped, ``K ~ 2D/h``) vs ``-> 0``
      (underdamped, ``K -> const``) at fine sampling.

    Returns
    -------
    dict
        ``rho2``, ``K`` : ``(S,)`` arrays; ``noise_free_var`` : ``(S,)``;
        ``beta`` : log-log slope of ``K`` vs ``dt`` (OLS over positive ``K``).
    """
    h = np.asarray(scan["dt"], dtype=float)
    c0, c1, c2 = _scalar_lags(scan)
    noise_free_var = c0 + 2.0 * c1
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = np.where(np.abs(noise_free_var) > 0, c2 / noise_free_var, 0.0)
        K = noise_free_var / h**2
    good = K > 0
    beta = float(np.polyfit(np.log(h[good]), np.log(K[good]), 1)[0]) if good.sum() >= 2 else float("nan")
    return {"rho2": rho2, "K": K, "noise_free_var": noise_free_var, "beta": beta}


#: Verdict thresholds (documented; the full report exposes the raw statistics).
_AICC_MARGIN = 2.0  # AICc gain required to prefer the inertial model
_V_Z_MIN = 3.0  # significance of the fitted velocity variance V
_RHO2_UD = 0.15  # lag-2 persistence (at finest dt) confirming resolved inertia
_RHO2_OD = 0.05  # lag-2 persistence below which the data looks overdamped


def _decide_verdict(fit, scaling, scan):
    """Combine the parametric fit and the scaling test into a verdict.

    Returns ``"UD"`` when the inertial model is decisively preferred *and* the
    momentum is resolved (significant, persistent lag-2 correlation at the
    finest step); ``"OD"`` when there is no inertial signal; ``"inconclusive"``
    in the marginal band — typically coarse sampling (``gamma * dt >~ 1``),
    where the data is on the boundary of being effectively overdamped.
    """
    h = np.asarray(scan["dt"], dtype=float)
    rho2_fine = float(scaling["rho2"][int(np.argmin(h))])
    ud_preferred = (fit["delta_aicc"] > _AICC_MARGIN) and (fit["V_z"] > _V_Z_MIN)

    if ud_preferred and rho2_fine > _RHO2_UD:
        return "UD"
    if (not ud_preferred) or rho2_fine < _RHO2_OD:
        return "OD"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# Layer 3: overdamped-fit residual-autocorrelation cross-check
# --------------------------------------------------------------------------- #
def _cross_check(data, *, force_order: int = 3):
    """Corroborate the verdict via an overdamped fit's residual autocorrelation.

    Fits a flexible (cubic by default) overdamped force and runs the existing
    Ljung--Box residual-autocorrelation test from
    :func:`SFI.diagnostics.assess`.  An overdamped model cannot capture
    momentum, so on underdamped data its residuals stay serially correlated and
    the test fires.  This is *corroboration only*: a low p-value flags that an
    overdamped model leaves structure — usually inertia, but a force too
    complex for the cubic basis would also trip it — so the verdict itself rests
    on the noise-immune scaling test and the parametric fit, not on this.

    Best-effort: returns ``ljung_box_pvalue = None`` (with an ``error``) if the
    quick fit fails for any reason.
    """
    try:
        from SFI.bases.constants import unit_vector_basis
        from SFI.bases.monomials import monomials_up_to
        from SFI.diagnostics.assess import assess
        from SFI.inference.overdamped import OverdampedLangevinInference

        d = int(data.datasets[0].d)
        inf = OverdampedLangevinInference(data)
        inf.compute_diffusion_constant(method="auto")
        basis = monomials_up_to(order=force_order, dim=d, include_constant=True, include_x=True, include_v=False)
        inf.infer_force_linear(basis * unit_vector_basis(d))
        inf.compute_force_error()
        report = assess(inf, level="standard")
        lb = report.residuals.get("autocorr", {}).get("ljung_box", {})
        return {"ljung_box_pvalue": lb.get("pvalue"), "ljung_box_stat": lb.get("statistic")}
    except Exception as exc:  # pragma: no cover - corroboration only
        return {"ljung_box_pvalue": None, "error": repr(exc)}


# --------------------------------------------------------------------------- #
# Public report + entry point
# --------------------------------------------------------------------------- #
def _num_frames(ds):
    T = getattr(ds, "T", None)
    if T is None:
        T = np.asarray(ds.X).shape[0]
    return int(T)


def _default_strides(coll, max_strides=6, min_samples=200):
    """Geometric strides ``1, 2, 4, ...`` capped so the coarsest keeps stats."""
    T = min(_num_frames(ds) for ds in coll.datasets)
    strides, s = [], 1
    while len(strides) < max_strides and (T // s) >= min_samples:
        strides.append(s)
        s *= 2
    return strides or [1]


@dataclass
class DynamicsOrderReport:
    """Result of :func:`classify_dynamics`.

    Attributes
    ----------
    verdict : str
        ``"OD"``, ``"UD"`` or ``"inconclusive"``.
    fit : dict
        Parametric fit output (``params`` = ``sigma2, D, V, gamma``, ``tau_v``,
        ``stderr``, ``V_z``, ``delta_aicc`` and the raw SSR / AICc values).
    scaling : dict
        Model-free statistics (``rho2``, ``K``, ``beta``).
    scan : dict
        Lag-0/1/2 covariances across the dt scan (``strides``, ``dt``, ``C0``,
        ``C1``, ``C2``, ``n_eff``).
    cross_check : dict or None
        Overdamped-fit Ljung--Box result, or ``None`` if disabled.
    meta : dict
        Sampling summary (``d``, ``n_datasets``, ``strides``, ``dt_min/max``,
        ``gamma_dt_min``).
    """

    verdict: str
    fit: dict = field(default_factory=dict)
    scaling: dict = field(default_factory=dict)
    scan: dict = field(default_factory=dict)
    cross_check: Optional[dict] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serialisable representation of the report."""
        return _to_jsonable(
            {
                "verdict": self.verdict,
                "fit": self.fit,
                "scaling": self.scaling,
                "scan": self.scan,
                "cross_check": self.cross_check,
                "meta": self.meta,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self) -> None:
        """Print a human-readable summary of the classification."""
        p = self.fit.get("params", {})
        sc = self.scaling
        meta = self.meta
        print("\n=== SFI dynamics-order classification ===")
        print(f"verdict : {self.verdict}")
        if meta:
            print(
                f"sampling: d={meta.get('d', '?')}, "
                f"dt in [{meta.get('dt_min', float('nan')):.4g}, "
                f"{meta.get('dt_max', float('nan')):.4g}] over "
                f"{len(meta.get('strides', []))} strides"
            )
        if p:
            sigma = float(np.sqrt(max(p.get("sigma2", 0.0), 0.0)))
            print("\n-- parametric fit (diffusion + inertia + localization) --")
            print(f"  tau_v = 1/gamma = {self.fit.get('tau_v', float('nan')):.4g}   (momentum relaxation time)")
            print(
                f"  <v^2> = {p.get('V', float('nan')):.4g}   "
                f"D = {p.get('D', float('nan')):.4g}   sigma_loc = {sigma:.4g}"
            )
            print(
                f"  AICc(OD) - AICc(UD) = {self.fit.get('delta_aicc', float('nan')):+.2f}   "
                f"(>0 favours inertia)   V/stderr = {self.fit.get('V_z', float('nan')):.2f}"
            )
        if sc:
            rho2 = np.asarray(sc.get("rho2", []))
            rfine = float(rho2[0]) if rho2.size else float("nan")
            print("\n-- model-free scaling (noise-immune at lag>=2) --")
            print(f"  rho2(finest dt) = {rfine:.3f}   (OD -> 0, UD -> positive)")
            print(f"  apparent-KE log-log slope = {sc.get('beta', float('nan')):+.2f}   (OD -> -1, UD -> 0)")
        if self.cross_check and self.cross_check.get("ljung_box_pvalue") is not None:
            print("\n-- cross-check (OD-fit residual autocorrelation) --")
            print(
                f"  Ljung-Box p = {self.cross_check['ljung_box_pvalue']:.2e}   "
                "(small => overdamped model leaves structure)"
            )
        if self.verdict == "inconclusive":
            gdt = meta.get("gamma_dt_min")
            tail = f" (gamma*dt_min ~ {gdt:.2f}); sample finer than tau_v to decide." if gdt else "."
            print("\n  Note: momentum only marginally resolved" + tail)
        print()


def classify_dynamics(data, *, strides=None, cross_check: bool = True) -> DynamicsOrderReport:
    """Classify trajectory data as overdamped, underdamped, or inconclusive.

    Robust to high localization noise (lag >= 2 covariances are noise-immune)
    and reports *inconclusive* at coarse sampling where the momentum is
    unresolved (``gamma * dt >~ 1``), rather than guessing.

    Parameters
    ----------
    data : TrajectoryCollection
        Raw trajectories (positions).  Velocities are *not* required — the test
        reconstructs the dynamics order from position increments alone.
    strides : sequence of int, optional
        Coarse-graining factors for the dt scan.  Default: a geometric set
        ``1, 2, 4, ...`` capped so the coarsest still has enough samples.
    cross_check : bool
        Run the Layer-3 overdamped-fit residual-autocorrelation corroboration
        (default ``True``).

    Returns
    -------
    DynamicsOrderReport

    Notes
    -----
    Assumes *white* localization noise; spatially uniform, isotropic pooling of
    components.  Strong memory (a generalized Langevin / viscoelastic bath) can
    also produce velocity persistence without inertia — out of scope; the test
    detects trajectory smoothness, which inertia produces.

    Examples
    --------
    >>> report = classify_dynamics(collection)        # doctest: +SKIP
    >>> report.verdict                                # doctest: +SKIP
    'UD'
    """
    if strides is None:
        strides = _default_strides(data)
    strides = [int(s) for s in strides]

    scan = _covariances_vs_dt(data, strides)
    fit = _fit_dynamics(scan)
    scaling = _scaling_statistics(scan)
    verdict = _decide_verdict(fit, scaling, scan)
    cross = _cross_check(data) if cross_check else None

    dt = np.asarray(scan["dt"], dtype=float)
    meta = {
        "d": int(data.datasets[0].d),
        "n_datasets": len(data.datasets),
        "strides": [int(s) for s in scan["strides"]],
        "dt_min": float(dt.min()),
        "dt_max": float(dt.max()),
        "gamma_dt_min": float(fit["params"]["gamma"] * dt.min()),
    }
    return DynamicsOrderReport(
        verdict=verdict, fit=fit, scaling=scaling, scan=scan, cross_check=cross, meta=meta
    )
