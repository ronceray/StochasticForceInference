# TODO: review this file
"""
Shared helpers for SFI gallery examples.

Provides:
- ``STYLE_PATH`` — path to the shared matplotlib style
- ``SFI_COLORS`` — named color palette for consistent plots
- ``apply_style()`` — load the gallery mplstyle
- ``stamp_fig()`` — add a discreet generation timestamp to a figure
- ``run_degradation_sweep()`` — benchmark loop over noise / downsampling
- ``plot_nmse_vs_param()`` — standard benchmark line plot
- ``plot_pareto_front()`` — Pareto front with IC thresholds
- ``plot_recovery_bar()`` — bar chart of recovered / spurious coefficients
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if "__file__" in dir():
    STYLE_PATH = str(Path(__file__).resolve().parent.parent / "sfi_gallery.mplstyle")
else:
    STYLE_PATH = ""

# Named palette matching the mplstyle prop_cycle
# Adjusted for readability on both dark (#131416) and light backgrounds
SFI_COLORS = dict(
    data="#3B9EFF",       # bright blue
    inferred="#FFC20A",   # gold
    exact="#FF7A1A",      # bright orange
    bootstrap="#5D3A9B",  # purple
    highlight="#40B0A6",  # teal
    error="#FF2D6F",      # bright pink
    secondary="#1A85FF",  # lighter blue
    tertiary="#D35FB7",   # pink/magenta
)


def stamp_fig(fig=None):
    """Add a discreet generation timestamp to the bottom-right of a figure.

    Idempotent: a figure is only stamped once (tracked via a private attribute).
    """
    if fig is None:
        fig = plt.gcf()
    if getattr(fig, "_sfi_stamped", False):
        return
    fig._sfi_stamped = True
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99, 0.005, now,
        fontsize=5, color="#808080", alpha=0.4,
        ha="right", va="bottom",
        transform=fig.transFigure,
    )


def stamp_output():
    """Print a discreet generation timestamp to stdout.

    Call once per script so the timestamp appears in terminal output
    blocks on gallery pages and in notebook cells.
    """
    print(f"[Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}]")


def _enable_auto_stamp():
    """Monkey-patch ``Figure.savefig`` to auto-stamp before saving.

    Called once by ``apply_style()``.  Safe to call multiple times (no-op
    after the first).
    """
    if getattr(matplotlib.figure.Figure, "_sfi_auto_stamp_installed", False):
        return
    _orig_savefig = matplotlib.figure.Figure.savefig

    def _stamped_savefig(self, *args, **kwargs):
        stamp_fig(self)
        return _orig_savefig(self, *args, **kwargs)

    matplotlib.figure.Figure.savefig = _stamped_savefig
    matplotlib.figure.Figure._sfi_auto_stamp_installed = True


def apply_style():
    """Load the SFI gallery matplotlib style and enable auto-stamping."""
    plt.style.use(STYLE_PATH)
    _enable_auto_stamp()
    # Silence warnings whose text embeds the absolute demo source path — these
    # would otherwise leak the local filesystem layout into captured gallery
    # output. constrained_layout still governs the layout; only the (harmless)
    # warnings are muted.
    warnings.filterwarnings(
        "ignore",
        message="The figure layout has changed to tight",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message="Very short trajectory")


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------


def run_degradation_sweep(
    proc,
    coll_clean,
    basis,
    *,
    noise_levels: Sequence[float] = (0.0,),
    downsample_factors: Sequence[int] = (1,),
    motion_blur: int = 0,
    M_mode: str = "auto",
    G_mode: str = "trapeze",
    diffusion_method: str = "auto",
    sparsity_criterion: str | None = "PASTIS",
    true_support: list[int] | None = None,
    true_coeffs=None,
    maxpoints: int = 2000,
    seed: int = 0,
    verbosity: int = 0,
):
    """
    Run inference over a grid of (noise, downsample) degradation levels.

    Returns a list of dicts, each with keys:
        noise, downsample, NMSE_force, NMSE_diffusion, D_est,
        n_selected, overlap (if true_support given), coeffs, support
    """
    from SFI.inference.overdamped import OverdampedLangevinInference

    results = []
    for noise in noise_levels:
        for ds_factor in downsample_factors:
            coll_deg = coll_clean.degrade(
                downsample=ds_factor,
                noise=noise if noise > 0 else None,
                motion_blur=min(motion_blur, max(ds_factor - 1, 0)),
                seed=seed,
            )
            inf = OverdampedLangevinInference(coll_deg, verbosity=verbosity)
            inf.compute_diffusion_constant(method=diffusion_method)
            inf.infer_force_linear(basis, M_mode=M_mode, G_mode=G_mode)
            inf.compute_force_error()
            inf.compare_to_exact(
                model_exact=proc,
                maxpoints=maxpoints,
            )

            row = dict(
                noise=noise,
                downsample=ds_factor,
                NMSE_force=float(inf.NMSE_force),
                NMSE_diffusion=float(inf.NMSE_diffusion),
                D_est=float(inf.diffusion_average.mean()),
                n_coeffs=int(basis.n_features),
            )

            if sparsity_criterion is not None:
                inf.sparsify_force(criterion=sparsity_criterion)
                inf.compare_to_exact(
                    model_exact=proc,
                    maxpoints=maxpoints,
                )
                k, support, score, coeffs = (
                    inf.force_sparsity_result.select_by_ic(
                        sparsity_criterion, p_param=1e-3
                    )
                )
                row["NMSE_force_sparse"] = float(inf.NMSE_force)
                row["n_selected"] = int(k)
                row["support"] = [int(s) for s in support]
                row["coeffs_sparse"] = np.array(coeffs)

                if true_support is not None:
                    from SFI.inference.sparse import overlap_metrics

                    om = overlap_metrics(true_support, [int(s) for s in support])
                    row.update(om)

            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Standard benchmark plots
# ---------------------------------------------------------------------------


def plot_nmse_vs_param(
    results,
    param: str = "noise",
    *,
    include_sparse: bool = True,
    ax=None,
    log_y: bool = True,
):
    """
    Plot NMSE_force vs a degradation parameter.

    Parameters
    ----------
    results : list of dicts from ``run_degradation_sweep``
    param : key to use as x-axis (``"noise"`` or ``"downsample"``)
    """
    if ax is None:
        _, ax = plt.subplots()

    x = [r[param] for r in results]
    y_full = [r["NMSE_force"] for r in results]
    ax.plot(x, y_full, "o-", color=SFI_COLORS["data"], label="full model")

    if include_sparse and "NMSE_force_sparse" in results[0]:
        y_sp = [r["NMSE_force_sparse"] for r in results]
        ax.plot(x, y_sp, "s--", color=SFI_COLORS["inferred"], label="sparse (PASTIS)")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("NMSE (force)")
    ax.legend()
    ax.set_title(f"Inference accuracy vs {param}")
    return ax


# The Pareto-front and recovery-bar plotters are maintained in the library;
# re-export them here so existing gallery imports keep working and there is a
# single source of truth (the local copies had drifted).
from SFI.utils.plotting import plot_pareto_front, plot_recovery_bar  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Kernel CI plot
# ---------------------------------------------------------------------------


def plot_kernel_ci(
    ax,
    r_eval,
    true_profile,
    ci,
    *,
    scale=None,
    label_true="True",
    label_inferred="Learned",
    ci_label=None,
    ci_alpha=0.2,
    margin=0.3,
):
    """Plot a 1D kernel profile with confidence band.

    Parameters
    ----------
    ax : matplotlib Axes
    r_eval : array, shape (R,)
        Radial grid.
    true_profile : array, shape (R,), or None
        True kernel profile.  If *None*, only the inferred curve and CI
        band are drawn.
    ci : dict
        Output of :func:`SFI.inference.kernel_predict_ci`
        (keys ``mean``, ``std``, ``lower``, ``upper``).
    scale : float, optional
        Normalisation divisor applied to all profiles (default:
        ``max(|true_profile|)`` if available, else 1).
    label_true, label_inferred : str
        Legend labels.
    ci_label : str or None
        Legend label for the CI band.  ``None`` suppresses a legend
        entry.
    ci_alpha : float
        Opacity of the CI fill.
    margin : float
        Fractional y-axis padding beyond the data range.
    """
    r = np.asarray(r_eval)
    mean = np.asarray(ci["mean"])
    lower = np.asarray(ci["lower"])
    upper = np.asarray(ci["upper"])

    if scale is None:
        scale = float(np.max(np.abs(true_profile))) if true_profile is not None else 1.0
    if scale == 0.0:
        scale = 1.0

    if true_profile is not None:
        ax.plot(r, np.asarray(true_profile) / scale, "--", lw=2,
                color=SFI_COLORS["exact"], label=label_true)
    ax.plot(r, mean / scale, lw=2,
            color=SFI_COLORS["inferred"], label=label_inferred)
    ax.fill_between(
        r, lower / scale, upper / scale,
        color=SFI_COLORS["inferred"], alpha=ci_alpha,
        label=ci_label,
    )
    ax.axhline(0, color="#808080", lw=0.5)

    # Auto y-limits
    all_vals = np.concatenate([mean, lower, upper]) / scale
    if true_profile is not None:
        all_vals = np.concatenate([all_vals, np.asarray(true_profile) / scale])
    ylo, yhi = float(all_vals.min()), float(all_vals.max())
    pad = margin * max(yhi - ylo, 0.1)
    ax.set_ylim(ylo - pad, yhi + pad)

    return ax
