r"""
Diagnostics — should we trust a fit?
====================================

An inference always returns numbers; the question is whether the model
behind them is consistent with the data.  :mod:`SFI.diagnostics`
answers it without ground truth: it recomputes the standardised
residuals of a fit and tests whether they look like independent
standard Gaussian noise, and it checks that the sampling interval
is fine enough for the estimator used.

We fit one double-well recording in three ways — the right model, a
model with a **missing term**, and the right model on data with
**unmodelled measurement noise** — and read the three reports.  Only at
the end do we compare with the true force.

.. rubric:: Tags

diagnostics · overdamped · nonlinear · 1D · synthetic
"""

# sphinx_gallery_tags = ["diagnostics", "overdamped", "nonlinear", "1D", "synthetic"]
# sphinx_gallery_thumbnail_number = 3

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

apply_style()
# sphinx_gallery_end_ignore
stamp_output()

# %%
# The data: a particle hopping between two wells
# ----------------------------------------------
#
# :math:`F(x) = x - x^3` has stable points at :math:`x = \pm 1`, and the
# particle hops between them; we record 8000 frames at
# :math:`\Delta t = 0.05`.  A second copy of the recording carries
# Gaussian localization noise of standard deviation 0.05, about 40% of the
# typical one-step displacement.

import SFI
from SFI.bases import monomials_up_to, unit_axes, x_components
from SFI.langevin import OverdampedProcess

(x,) = x_components(1)
(ex,) = unit_axes(1)
proc = OverdampedProcess((x - x * x * x) * ex, D=0.15)
proc.initialize(jnp.array([0.5]))
coll = proc.simulate(dt=0.05, Nsteps=8000, key=random.PRNGKey(2), prerun=200, oversampling=10)
coll_noisy = coll.degrade(noise=0.05, seed=3)

# %%
# Three fits
# ----------
#
# All three use the fast linear estimators in their clean-data setting
# (``"WeakNoise"`` diffusion, ``preset="clean"`` force), which assume no
# measurement noise:
#
# * **A** — the cubic basis :math:`\{1, x, x^2, x^3\}`, which contains the
#   true force;
# * **B** — the linear basis :math:`\{1, x\}`, which misses the cubic term;
# * **C** — the cubic basis again, on the noisy copy.

B_cubic = monomials_up_to(order=3, dim=1, rank="vector")
B_linear = monomials_up_to(order=1, dim=1, rank="vector")


def clean_fit(data, basis):
    inf = SFI.OverdampedLangevinInference(data)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis, preset="clean")
    inf.compute_force_error()
    return inf


fits = {"A": clean_fit(coll, B_cubic),
        "B": clean_fit(coll, B_linear),
        "C": clean_fit(coll_noisy, B_cubic)}

# %%
# Run the diagnostics
# -------------------
#
# :meth:`~SFI.OverdampedLangevinInference.diagnose` (equivalently
# :func:`SFI.diagnostics.assess`) returns a report; ``print_summary``
# lists the tests and, under ``-- Flags --``, every failed check with a
# one-line hint at the likely cure.  We read the three reports one by one
# below; the figure first gives an overview, with two of the panels of
# :func:`~SFI.diagnostics.plot_summary` for each fit: the Q–Q plot of the
# residuals against :math:`\mathcal N(0,1)` (points on the diagonal) and
# their autocorrelation (bars inside the 95% band).

reports = {name: inf.diagnose() for name, inf in fits.items()}

# sphinx_gallery_start_ignore
from SFI.diagnostics import plot_qq, plot_residual_acf

titles = {"A": "A: cubic basis", "B": "B: missing $x^3$ term", "C": "C: unmodelled noise"}


def diag_grid(reps, titles):
    fig, axes = plt.subplots(2, len(reps), figsize=(3.6 * len(reps), 5.8))
    for j, (name, rep) in enumerate(reps.items()):
        plot_qq(rep, ax=axes[0, j])
        plot_residual_acf(rep, ax=axes[1, j])
        for ln in axes[1, j].get_lines():          # keep ACF(z) only, drop ACF(z²)
            if ln.get_marker() == "o":
                ln.remove()
        axes[0, j].set_title(titles[name])
        axes[1, j].set_title("")
        axes[1, j].set_ylim(-0.065, 0.065)
        res = rep.residuals
        p_lb = res["autocorr"]["ljung_box"]["pvalue"]
        axes[0, j].text(0.97, 0.05, f"std = {res['moments']['std']:.2f}", transform=axes[0, j].transAxes,
                        ha="right", fontsize=9, color="#D0D0D0")
        axes[1, j].text(0.97, 0.9, f"Ljung–Box p = {p_lb:.1g}", transform=axes[1, j].transAxes,
                        ha="right", fontsize=9, color=SFI_COLORS["error"] if p_lb < 0.01 else "#D0D0D0")
        for ax in axes[:, j]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if j:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")
    axes[0, 0].legend(["$y = x$", "residuals"], loc="upper left", fontsize=8, frameon=False)
    axes[1, 0].legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    return fig


fig = diag_grid(reports, titles)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Reading the reports
# -------------------
#
# **A passes** every residual test: Gaussian, uncorrelated, and a χ²
# excess consistent with zero (the test is calibrated for the finite
# :math:`\Delta t`, which makes the residual standard deviation slightly
# below 1).

print(f"Fit A: D = {float(fits['A'].diffusion_average[0, 0]):.3f}")
reports["A"].print_summary()

# %%
# **B fails the autocorrelation test.**  The missing cubic force varies
# slowly as the particle wanders in and out of the wells, so it leaves a
# correlated signal in the residuals and the Ljung–Box test flags it —
# "widen the basis".  Its marginal residuals still look Gaussian: only
# the autocorrelation test sees structure the basis cannot express.

print(f"Fit B: D = {float(fits['B'].diffusion_average[0, 0]):.3f}")
reports["B"].print_summary()

# %%
# **C fails three tests at once.**  Localization noise corrupts both the
# increments and the positions at which the force is evaluated.  The
# clean diffusion estimator counts the noise as extra diffusion
# (:math:`\hat D` comes out about 50% high), so the whitened residuals are
# too narrow — Q–Q slope below 1, and the normality test fails; the
# residuals are correlated over a few frames (Ljung–Box); and their χ²
# excess is far above what the predicted error allows.

print(f"Fit C: D = {float(fits['C'].diffusion_average[0, 0]):.3f}")
reports["C"].print_summary()

# %%
# The hint points at the cure: the parametric estimator, which models the
# noise.

inf_C = SFI.OverdampedLangevinInference(coll_noisy)
inf_C.infer_force(B_cubic)      # profiles diffusion D and noise covariance Λ
inf_C.compute_force_error()
print(f"estimated noise amplitude: {np.sqrt(float(inf_C.Lambda[0, 0])):.3f}  (true 0.05)")
reports["C'"] = inf_C.diagnose()
reports["C'"].print_summary()

# %%
# The noise-aware refit C' recovers the noise amplitude and its residuals
# pass every test.
#
# The caution every fit carries
# -----------------------------
#
# All reports, C' included, end with a ``finite_dt`` caution: near the
# wells :math:`\Delta t\,|F'(x)|` reaches 0.3, a regime where the linear
# estimators can carry :math:`O(\Delta t)` biases of a few per cent.  This
# flag is about the sampling, not the fit, so it does not go away; the
# hint is to *confirm* the linear result with the parametric estimator,
# whose flow-based residual is accurate at coarser :math:`\Delta t`.

inf_A_par = SFI.OverdampedLangevinInference(coll)
inf_A_par.infer_force(B_cubic)
inf_A_par.compute_force_error()

labels = ["1", "x", "x²", "x³"]
exact = [0.0, 1.0, 0.0, -1.0]
fA, fP = fits["A"], inf_A_par
print("term   linear (A)        parametric        exact")
for i, lab in enumerate(labels):
    print(f"{lab:4s}  {fA.force_coefficients[i]:+.3f} ± {fA.force_coefficients_stderr[i]:.3f}"
          f"   {fP.force_coefficients[i]:+.3f} ± {fP.force_coefficients_stderr[i]:.3f}   {exact[i]:+.1f}")

# %%
# The two estimators agree well within the error bars: at this sampling
# the :math:`O(\Delta t)` bias is below the statistical error, and fit A
# stands.
#
# Behind the scenes: the forces
# -----------------------------
#
# This recording is synthetic, so we can now look at what the diagnostics
# were detecting.  B cannot bend its straight line into the double well;
# C's force is too steep, because each noise kick is undone at the next
# frame, which mimics an extra restoring force; A and C' lie on the exact
# force within their 95% bands.

for name, f in (("C", fits["C"]), ("C'", inf_C)):
    c = np.asarray(f.force_coefficients)
    print(f"fit {name:2s}: x {c[1]:+.2f}, x³ {c[3]:+.2f}   (exact +1, -1)")

# sphinx_gallery_start_ignore
xg = jnp.linspace(-1.8, 1.8, 200)[:, None]
xg_np = np.asarray(xg)[:, 0]
_, Xd, _ = coll.to_arrays(dataset=0)
x_data = np.asarray(Xd)[:, 0, 0]
fig, ax = plt.subplots(figsize=(7.5, 4))
axh = ax.twinx()
axh.hist(x_data, bins=60, range=(-1.8, 1.8), density=True, alpha=0.15, color=SFI_COLORS["data"])
axh.set_yticks([])
ax.set_zorder(axh.get_zorder() + 1)
ax.patch.set_visible(False)
for f, lab, c, ls in [(fits["B"].force_inferred, "B: missing term", SFI_COLORS["secondary"], "-"),
                      (fits["C"].force_inferred, "C: unmodelled noise", SFI_COLORS["error"], "-"),
                      (fits["A"].force_inferred, "A: cubic basis", SFI_COLORS["inferred"], "-"),
                      (inf_C.force_inferred, "C': noise-aware refit", SFI_COLORS["highlight"], "-")]:
    ci = f.predict_ci(xg)
    ax.fill_between(xg_np, np.asarray(ci["lower"])[:, 0], np.asarray(ci["upper"])[:, 0],
                    color=c, alpha=0.25, lw=0)
    ax.plot(xg_np, np.asarray(ci["mean"])[:, 0], color=c, lw=2, ls=ls, label=lab)
ax.plot(xg_np, xg_np - xg_np ** 3, "--", color=SFI_COLORS["exact"], lw=2, label="exact $x - x^3$")
ax.axhline(0, color="#808080", lw=0.5)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-4.5, 4.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")
ax.set_title("Inferred forces with 95% bands (data histogram behind)")
h, l = ax.get_legend_handles_labels()
order = [2, 0, 1, 3, 4]                      # A, B, C, C', exact
ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=8, loc="lower left")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Takeaways
# ---------
#
# * Run ``inf.diagnose()`` after every fit.  Clean residual tests are
#   necessary, not sufficient: they see missing *structure* and unmodelled
#   noise, not a smooth bias that leaves the residuals white.
# * Correlated residuals point to missing dynamics (B) — or, together
#   with too-narrow residuals and a χ² excess, to measurement noise (C).
#   Each flag's hint names the cure.
# * A ``finite_dt`` caution is about the sampling: confirm with the
#   parametric estimator.
#
# The :ref:`Diagnostics user guide <diagnostics_user_guide>` lists every
# test; :doc:`/gallery/experimental_workflow_demo` applies them to a
# recording with no ground truth.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
from scipy.stats import norm

for name, c in (("C", SFI_COLORS["error"]), ("A", SFI_COLORS["data"])):
    s = np.sort(np.asarray(reports[name].residuals["normality"]["qq"]["sample_quantiles"]).ravel())
    q = norm.ppf((np.arange(1, s.size + 1) - 0.5) / s.size)
    ax_t.scatter(q[::20], s[::20], s=8, color=c, alpha=0.8)
ax_t.plot([-4, 4], [-4, 4], "--", color=SFI_COLORS["exact"], lw=1.5)
ax_t.set_xlim(-4, 4); ax_t.set_ylim(-4, 4)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
