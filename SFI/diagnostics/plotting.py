"""Diagnostic plotting helpers.

Three panels covering the common visual checks of a fitted inference
object:

* :func:`plot_qq` — normal Q--Q plot of pooled whitened residuals;
* :func:`plot_residual_histogram` — histogram of pooled residuals
  overlaid with the standard normal density;
* :func:`plot_residual_acf` — autocorrelation of the residuals (and of
  their squares) with the ``±1.96/√n`` band;
* :func:`plot_summary` — a 1×3 figure combining the three panels.

All helpers accept either a fitted inference object (calling
:func:`~SFI.diagnostics.assess.assess` internally) or a pre-computed
:class:`~SFI.diagnostics.report.DiagnosticsReport`.

Plots use the SFI palette (:data:`SFI.utils.plotting.SFI_COLORS`) but do
*not* call ``apply_style()`` — applying the gallery style is the caller's
responsibility (see ``GALLERY_STYLE_GUIDE.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from SFI.utils.plotting import SFI_COLORS

from .assess import assess
from .report import DiagnosticsReport


def _coerce_report(obj: Any, *, level: str = "standard") -> DiagnosticsReport:
    """Return a :class:`DiagnosticsReport` from either a report or inferer."""
    if isinstance(obj, DiagnosticsReport):
        return obj
    return assess(obj, level=level)


def _get_residuals(report: DiagnosticsReport) -> np.ndarray:
    """Return pooled standardised residuals as a 1-D array."""
    qq = report.residuals.get("normality", {}).get("qq", {})
    sample = qq.get("sample_quantiles")
    if sample is not None:
        return np.asarray(sample).ravel()
    raise ValueError(
        "Diagnostics report does not contain raw residual samples. "
        "Run assess() with level='standard' (the default) before plotting."
    )


def plot_qq(report_or_inferer, *, ax=None, level: str = "standard"):
    """Normal Q--Q plot of pooled whitened residuals."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    rep = _coerce_report(report_or_inferer, level=level)
    sample = np.sort(_get_residuals(rep))
    n = sample.size
    if n == 0:
        raise ValueError("No residuals to plot.")
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = norm.ppf(probs)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    lo = float(min(theo.min(), sample.min()))
    hi = float(max(theo.max(), sample.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=SFI_COLORS["exact"], lw=1.2, label="$y = x$")

    if n > 5000:  # subsample to keep the figure light
        idx = np.linspace(0, n - 1, 5000).astype(int)
        ax.scatter(theo[idx], sample[idx], s=6, alpha=0.6, color=SFI_COLORS["data"], label=f"residuals (n={n})")
    else:
        ax.scatter(theo, sample, s=8, alpha=0.7, color=SFI_COLORS["data"], label=f"residuals (n={n})")

    ax.set_xlabel("theoretical quantile (N(0,1))")
    ax.set_ylabel("sample quantile")
    ax.set_title("Q--Q plot")
    ax.legend(frameon=False, loc="best", fontsize=8)
    return ax


def plot_residual_histogram(report_or_inferer, *, ax=None, bins: int = 60, level: str = "standard"):
    """Histogram of pooled residuals overlaid with N(0,1) density."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    rep = _coerce_report(report_or_inferer, level=level)
    sample = _get_residuals(rep)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.hist(sample, bins=bins, density=True, alpha=0.65, color=SFI_COLORS["data"], label=f"residuals (n={sample.size})")
    grid = np.linspace(sample.min(), sample.max(), 256)
    ax.plot(grid, norm.pdf(grid), "--", color=SFI_COLORS["exact"], lw=1.4, label="N(0,1)")
    ax.set_xlabel("standardised residual $z$")
    ax.set_ylabel("density")
    ax.set_title("Residual histogram")
    ax.legend(frameon=False, loc="best", fontsize=8)
    return ax


def plot_residual_acf(report_or_inferer, *, ax=None, level: str = "standard"):
    """Autocorrelation of the residuals (and of $z^2$).

    Reads the ACF computed by
    :func:`~SFI.diagnostics.residual_tests.autocorrelation_tests`
    (stored on the report) rather than recomputing it.
    """
    import matplotlib.pyplot as plt

    rep = _coerce_report(report_or_inferer, level=level)
    ac = rep.residuals.get("autocorr", {})
    acf = np.asarray(ac.get("acf", []), dtype=np.float64)
    acf2 = np.asarray(ac.get("acf_squared", []), dtype=np.float64)
    n_eff = int(ac.get("n_eff", 0))
    if acf.size == 0:
        raise ValueError("Report has no autocorrelation data; run assess(level='standard').")

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    lags = np.arange(1, acf.size + 1)
    band = 1.96 / np.sqrt(max(n_eff, 1))
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.fill_between(
        np.arange(0, acf.size + 1), -band, band,
        color=SFI_COLORS["exact"], alpha=0.15, label="95% band",
    )
    ax.vlines(lags, 0.0, acf, color=SFI_COLORS["data"], lw=1.5, label="ACF($z$)")
    if acf2.size == acf.size:
        ax.plot(lags, acf2, "o-", color=SFI_COLORS["highlight"], ms=3, lw=1.0, label="ACF($z^2$)")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Residual ACF")
    ax.legend(frameon=False, loc="best", fontsize=8)
    return ax


def plot_summary(report_or_inferer, *, level: str = "standard", figsize=(13.0, 4.0)):
    """1×3 summary figure: Q--Q plot, residual histogram, and ACF.

    Returns the matplotlib :class:`Figure`.
    """
    import matplotlib.pyplot as plt

    rep = _coerce_report(report_or_inferer, level=level)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_qq(rep, ax=axes[0])
    plot_residual_histogram(rep, ax=axes[1])
    plot_residual_acf(rep, ax=axes[2])
    fig.set_layout_engine("tight")
    return fig


def plot_dynamics_order(report, *, axes=None):
    """Visualise an OD-vs-UD classification (:func:`classify_dynamics`).

    Two panels versus the sampling step ``dt``:

    * **lag-2 persistence** ``rho2 = C2/(C0+2C1)`` (noise-immune): tends to 0
      for overdamped data, to a positive plateau for inertia;
    * **apparent kinetic energy** ``K = (C0+2C1)/dt^2`` on log-log axes, with
      reference slopes ``-1`` (overdamped, ``K ~ 2D/dt``) and ``0``
      (underdamped); the fitted log-log slope ``beta`` is annotated.

    The parametric-fit prediction is overlaid on both panels.  Accepts a
    :class:`~SFI.diagnostics.dynamics_order.DynamicsOrderReport`.
    """
    import matplotlib.pyplot as plt

    from .dynamics_order import _dynamics_model

    dt = np.asarray(report.scan["dt"], dtype=float)
    rho2 = np.asarray(report.scaling["rho2"], dtype=float)
    K = np.asarray(report.scaling["K"], dtype=float)
    beta = float(report.scaling.get("beta", float("nan")))
    p = report.fit.get("params", {})
    verdict = report.verdict

    hg = np.geomspace(dt.min(), dt.max(), 100)
    m0, m1, m2 = _dynamics_model(hg, p.get("sigma2", 0.0), p.get("D", 0.0), p.get("V", 0.0), p.get("gamma", 1.0))
    nf = m0 + 2.0 * m1
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2_fit = np.where(np.abs(nf) > 0, m2 / nf, np.nan)
        K_fit = nf / hg**2

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    else:
        fig = np.ravel(axes)[0].figure
    ax_rho, ax_k = np.ravel(axes)[:2]

    ax_rho.semilogx(dt, rho2, "o", color=SFI_COLORS["data"], label="data")
    ax_rho.semilogx(hg, rho2_fit, "-", color=SFI_COLORS["inferred"], label="fit")
    ax_rho.axhline(0.0, color="0.5", lw=0.8)
    ax_rho.set_xlabel(r"sampling step $\Delta t$")
    ax_rho.set_ylabel(r"$\rho_2 = C_2 / (C_0 + 2 C_1)$")
    ax_rho.set_title(f"lag-2 persistence — verdict: {verdict}")
    ax_rho.legend(frameon=False, loc="best", fontsize=8)

    good = K > 0
    anchor = float(K[good][0]) if good.any() else 1.0
    ax_k.loglog(dt[good], K[good], "o", color=SFI_COLORS["data"], label="data")
    ax_k.loglog(hg, K_fit, "-", color=SFI_COLORS["inferred"], label="fit")
    ax_k.loglog(hg, anchor * (dt[0] / hg), ":", color=SFI_COLORS["exact"], lw=1.0, label="OD ref (slope $-1$)")
    ax_k.loglog(hg, np.full_like(hg, anchor), "--", color=SFI_COLORS["highlight"], lw=1.0, label="UD ref (slope $0$)")
    ax_k.set_xlabel(r"sampling step $\Delta t$")
    ax_k.set_ylabel(r"apparent KE $\tilde K = (C_0 + 2 C_1)/\Delta t^2$")
    ax_k.set_title(rf"scaling slope $\beta = {beta:.2f}$  (OD$\to-1$, UD$\to0$)")
    ax_k.legend(frameon=False, loc="best", fontsize=8)

    fig.set_layout_engine("tight")
    return fig


__all__ = [
    "plot_qq",
    "plot_residual_histogram",
    "plot_residual_acf",
    "plot_summary",
    "plot_dynamics_order",
]
