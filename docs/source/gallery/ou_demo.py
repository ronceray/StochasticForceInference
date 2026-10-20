r"""
Getting started: end-to-end inference (Ornstein–Uhlenbeck)
===========================================================

The recommended **starting point**: the complete Stochastic Force
Inference (SFI) workflow on the simplest non-trivial system, a particle
in a harmonic trap.  Its position obeys the overdamped Langevin
(Ornstein–Uhlenbeck) equation

.. math::

   \mathrm{d}x = \underbrace{-k\,x}_{F(x)}\,\mathrm{d}t
              + \sqrt{2D}\;\mathrm{d}W ,

a restoring force :math:`F(x) = -k\,x` plus white noise of strength
:math:`D`.  We simulate a trajectory, then pretend not to know
:math:`F` or :math:`D` and recover them from the data alone:

#. estimate :math:`D` and fit :math:`F` on a deliberately over-complete
   set of candidate terms, **with error bars**;
#. let SFI **select the minimal model**;
#. **validate** the fit as one would on real data — residual tests and
   a bootstrapped trajectory;
#. save the model and load it back.

Real recordings follow the same path: load your trajectories instead of
simulating them (:doc:`/gallery/experimental_workflow_demo`).

.. rubric:: Tags

synthetic · overdamped · linear · 1D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D"]
# sphinx_gallery_thumbnail_number = 7

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import plot_pareto_front

apply_style()
plt.rcParams["savefig.facecolor"] = "auto"   # the movie's dark background survives in the mp4
BG = "#131416"                               # page background (Furo dark)
# sphinx_gallery_end_ignore
stamp_output()

# %%
# The data
# --------
#
# We build the exact model with
# :class:`~SFI.langevin.OverdampedProcess` — the force ``-k * X(dim=1)``
# is :math:`-k\,x`, with :func:`~SFI.bases.X` the coordinate — and record
# 10 000 frames, i.e. 200 relaxation times :math:`1/k`.

import SFI
from SFI.bases import X
from SFI.langevin import OverdampedProcess

k_true, D_true = 1.0, 0.5   # trap stiffness and diffusion constant
dt = 0.02                   # time between recorded frames

proc = OverdampedProcess(F=-k_true * X(dim=1), D=D_true)
proc.initialize(jnp.array([0.0]))
coll = proc.simulate(dt=dt, Nsteps=10_000, key=random.PRNGKey(137), oversampling=10)
print(f"{coll.T} frames, dt = {dt}")

# %%
# ``coll`` is a :class:`~SFI.trajectory.TrajectoryCollection`, the data
# container every SFI tool consumes — whether it comes from a simulation
# or from a recording (:meth:`~SFI.trajectory.TrajectoryCollection.from_arrays`,
# :meth:`~SFI.trajectory.TrajectoryCollection.load`).

# sphinx_gallery_start_ignore
t_axis, Xd, _ = coll.to_arrays(dataset=0)
t_axis, x_data = np.asarray(t_axis), np.asarray(Xd)[:, 0, 0]
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.plot(t_axis[:2500], x_data[:2500], lw=0.6, color=SFI_COLORS["data"])
ax.axhline(0, ls=":", lw=0.5, color="#808080")
ax.set_xlabel("time")
ax.set_ylabel("$x$")
ax.set_title("First 50 time units of the trajectory")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infer the diffusion and the force
# ---------------------------------
#
# In a real experiment we would not know that the force is linear, so we
# offer SFI a generic library of candidate terms, the monomials
# :math:`\{1, x, x^2\}`, and fit their coefficients by linear regression
# on the increments.
#
# The data are clean (no measurement noise), so we select the clean-data
# estimators: ``"WeakNoise"`` for :math:`D` (it differences successive
# displacements, which cancels the drift) and ``preset="clean"`` for the
# force.  On real recordings keep the defaults (``"auto"``), which detect
# localization noise and switch to noise-robust estimators.

from SFI.bases import monomials_up_to

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")

B = monomials_up_to(order=2, dim=1, rank="vector")   # candidate terms 1, x, x²
inf.infer_force_linear(B, preset="clean")
inf.compute_force_error()
inf.print_report()

# %%
# The report gives each coefficient with its standard error, and a
# *predicted* normalized mean-squared error (NMSE) of the whole force —
# SFI's own estimate of its accuracy, available without ground truth.
# The inferred force carries its parameter covariance, so pointwise
# confidence bands are one call away:

xg = jnp.linspace(-2.5, 2.5, 200)[:, None]
band_full = inf.force_inferred.predict_ci(xg)          # 95% band: "lower", "upper"

# sphinx_gallery_start_ignore
coef_full = np.asarray(inf.force_coefficients_full)
err_full = np.asarray(inf.force_coefficients_stderr)
xg_np = np.asarray(xg)[:, 0]


def _force_panel(ax, band, label):
    axh = ax.twinx()
    axh.hist(x_data, bins=60, range=(-2.5, 2.5), density=True, alpha=0.15,
             color=SFI_COLORS["data"])
    axh.set_yticks([])
    axh.spines["right"].set_visible(False)
    ax.set_zorder(axh.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.fill_between(xg_np, np.asarray(band["lower"])[:, 0], np.asarray(band["upper"])[:, 0],
                    color=SFI_COLORS["inferred"], alpha=0.3, lw=0, label="95% band")
    ax.plot(xg_np, np.asarray(band["mean"])[:, 0], color=SFI_COLORS["inferred"], lw=2, label=label)
    ax.plot(xg_np, -k_true * xg_np, "--", color=SFI_COLORS["exact"], lw=1.5, label="exact $-kx$")
    ax.axhline(0, color="#808080", lw=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$F(x)$")
    ax.legend(loc="upper right", fontsize=8)


term_labels = ["$1$", "$x$", "$x^2$"]
coef_exact = np.array([0.0, -k_true, 0.0])
fig, (axf, axc) = plt.subplots(1, 2, figsize=(9, 3.4), width_ratios=[1.5, 1])
_force_panel(axf, band_full, "inferred (3 terms)")
axf.set_title("Force, full basis (data histogram behind)")
pos = np.arange(3)
axc.errorbar(pos, coef_full, yerr=1.96 * err_full, fmt="o", ms=7, capsize=5,
             color=SFI_COLORS["inferred"], ecolor="#B0B0B0", label="inferred ± 95%")
axc.scatter(pos, coef_exact, marker="_", s=600, lw=2.5, color=SFI_COLORS["exact"],
            zorder=3, label="exact")
axc.axhline(0, color="#808080", lw=0.5)
axc.set_xticks(pos, term_labels)
axc.set_xlim(-0.6, 2.6)
axc.set_ylabel("coefficient")
axc.set_title("Coefficients")
axc.legend(loc="upper center", fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The 95% error bars cover the truth: the linear coefficient is
# compatible with :math:`-k`, the constant and quadratic terms with zero.
#
# Which terms does the data support?
# ----------------------------------
#
# Rather than read the bars by eye, let SFI choose.
# :meth:`~SFI.OverdampedLangevinInference.sparsify_force` searches over
# subsets of the basis and scores each with an information criterion —
# by default **PASTIS**, which admits a spurious term only with small
# probability.

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()
band_sparse = inf.force_inferred.predict_ci(xg)

# %%
# The **Pareto front** (left) shows the decision: the information gained
# by the best model of each size.  The linear term is a huge jump; the
# other two add next to nothing, and PASTIS (marked) stops at one term.
# Dropping the spurious terms also tightens the force estimate — compare
# the band on the right with the full-basis one above.

# sphinx_gallery_start_ignore
fig, (axp, axf) = plt.subplots(1, 2, figsize=(9, 3.4), width_ratios=[1, 1.5])
plot_pareto_front(inf.force_sparsity_result, criteria=("PASTIS",), ax=axp)
axp.set_xticks(range(len(inf.force_sparsity_result.best_info_by_k)))
axp.set_title("Pareto front")
_force_panel(axf, band_sparse, "inferred (PASTIS: 1 term)")
axf.set_title("Force after model selection")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Is the fit consistent with the data?
# ------------------------------------
#
# With real data there is no exact model to compare against, so the
# first check is statistical.  :meth:`~SFI.OverdampedLangevinInference.diagnose`
# whitens the residuals,
# :math:`z_t = (\Delta x_t - \hat F(x_t)\,\Delta t)/\sqrt{2\hat D\,\Delta t}`,
# tests whether they look like independent :math:`\mathcal N(0,1)` draws,
# and compares the χ² excess of the residuals with the predicted NMSE.

report = inf.diagnose()
report.print_summary()

# sphinx_gallery_start_ignore
from SFI.diagnostics import plot_summary

fig = plot_summary(report, figsize=(11, 3.4))
fig.axes[0].set_title("Q–Q plot")
for ax, ncol in ((fig.axes[1], 1), (fig.axes[2], 3)):   # legends above the data
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=ncol)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Mean ≈ 0, standard deviation ≈ 1, no autocorrelation (which would
# point to missing dynamics) and Gaussian tails: the residuals show no
# structure the model misses, and their χ² excess is far from the
# :math:`|z| > 5` alarm level.  That last test catches gross force
# biases; at fine sampling its bound on the NMSE is loose, so the
# predicted NMSE remains the accuracy to quote.  The
# :ref:`Diagnostics user guide <diagnostics_user_guide>` lists every test;
# :doc:`/gallery/diagnostics_demo` shows them catching a wrong model.
#
# Does the model reproduce the data? Bootstrap
# ---------------------------------------------
#
# A dynamical check: simulate a *new* trajectory from the inferred model
# — the fitted force and diffusion are a ready-made simulator — and
# compare it with the recording.  If the fit is faithful the two are
# statistically indistinguishable.  In the movie the recording (top) and
# a bootstrap (bottom) scroll side by side; on the right each bead rides
# in the inferred trap :math:`\hat U(x) = -\int \hat F\,\mathrm{d}x`.

coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(random.PRNGKey(0))

# sphinx_gallery_start_ignore
_, Xb, _ = coll_boot.to_arrays(dataset=0)
x_boot = np.asarray(Xb)[:, 0, 0]
# inferred trap potential U(x) = -∫ F dx, for the movie
F_grid = np.asarray(band_sparse["mean"])[:, 0]
U_grid = -np.concatenate([[0.0], np.cumsum(0.5 * (F_grid[1:] + F_grid[:-1]) * np.diff(xg_np))])
U_grid -= U_grid.min()

WIN = 6.0                       # visible time window (relaxation times)
STEP = 2                        # data frames per movie frame
N_FRAMES = 200
i0 = int(WIN / dt)
fig_mov = plt.figure(figsize=(6.6, 3.3), layout="none", facecolor=BG)
gs = fig_mov.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.12, wspace=0.04,
                          left=0.08, right=0.985, bottom=0.14, top=0.92)
rows = []
for r, (xs, col, name) in enumerate([(x_data, SFI_COLORS["data"], "recorded data"),
                                     (x_boot, SFI_COLORS["bootstrap"], "bootstrap (inferred model)")]):
    ax_tr = fig_mov.add_subplot(gs[r, 0])
    ax_w = fig_mov.add_subplot(gs[r, 1], sharey=ax_tr)
    ax_w.plot(U_grid, xg_np, color=SFI_COLORS["inferred"], lw=2)
    ax_w.set_xlim(-0.1, 1.2 * U_grid.max())
    ax_w.tick_params(labelleft=False, left=False)
    ax_w.set_xticks([])
    ax_tr.set_ylim(-2.5, 2.5)
    ax_tr.set_ylabel("$x$")
    ax_tr.axhline(0, ls=":", lw=0.5, color="#808080")
    ax_tr.text(0.01, 0.95, name, transform=ax_tr.transAxes, va="top", color=col, fontsize=9,
               zorder=10, bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=2))
    if r == 0:
        ax_tr.tick_params(labelbottom=False)
        ax_w.set_title(r"inferred trap $\hat U(x)$", fontsize=9)
    else:
        ax_tr.set_xlabel("time")
    (line,) = ax_tr.plot([], [], color=col, lw=0.9)
    bead = ax_w.scatter([0], [0], s=110, marker="o", color=col, edgecolor="#D0D0D0", lw=0.8, zorder=5)
    rows.append((ax_tr, line, bead, xs))


def _movie_frame(n):
    i = i0 + STEP * n
    for ax_tr, line, bead, xs in rows:
        line.set_data(t_axis[i - i0:i + 1], xs[i - i0:i + 1])
        ax_tr.set_xlim(t_axis[i] - WIN, t_axis[i] + 0.15)
        bead.set_offsets([[np.interp(xs[i], xg_np, U_grid), xs[i]]])
    return []


movie = FuncAnimation(fig_mov, _movie_frame, frames=N_FRAMES, interval=70)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The sharper test uses statistics the fit never looked at.  SFI only
# uses **one-step increments**; the decay of correlations over many
# steps — the autocorrelation :math:`C(\tau)=\langle x(t)\,x(t+\tau)\rangle`
# and the mean-squared displacement
# :math:`\mathrm{MSD}(\tau)=\langle[x(t+\tau)-x(t)]^2\rangle` — is a
# genuine *prediction* of the inferred model.  Twenty bootstrap
# trajectories of the same length as the data give the spread these
# statistics would have if the inferred model were true; the data should
# fall inside it.

n_boot = 20
boots = [proc_boot.simulate(dt=dt, Nsteps=coll.T, key=random.PRNGKey(s)) for s in range(1, n_boot + 1)]

# sphinx_gallery_start_ignore
def acf(x, n_lags):
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    c = np.fft.irfft(f * np.conj(f))[:n_lags]
    return c / c[0]


def msd(x, n_lags):
    return np.array([np.mean((x[k:] - x[:-k]) ** 2) if k else 0.0 for k in range(n_lags)])


n_lags = int(5.0 / dt)
lags = np.arange(n_lags) * dt
xb_all = [np.asarray(b.to_arrays(dataset=0)[1])[:, 0, 0] for b in boots]
acf_b = np.array([acf(x, n_lags) for x in xb_all])
msd_b = np.array([msd(x, n_lags) for x in xb_all])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, stat_d, stat_b, exact, ylab, title in [
    (axes[0], acf(x_data, n_lags), acf_b, np.exp(-k_true * lags), r"$C(\tau)/C(0)$", "Autocorrelation"),
    (axes[1], msd(x_data, n_lags), msd_b, 2 * D_true / k_true * (1 - np.exp(-k_true * lags)),
     r"MSD$(\tau)$", "Mean-squared displacement"),
]:
    ax.fill_between(lags, stat_b.min(axis=0), stat_b.max(axis=0), color=SFI_COLORS["bootstrap"],
                    alpha=0.5, lw=0, label=f"bootstrap range ({n_boot} runs)")
    ax.plot(lags, stat_d, color=SFI_COLORS["data"], lw=2, label="data")
    ax.plot(lags, exact, ":", color="#D0D0D0", lw=1.2, label="exact OU")
    ax.set_xlabel(r"lag $\tau$")
    ax.set_ylabel(ylab)
    ax.set_title(title)
axes[0].axhline(0, ls=":", lw=0.5, color="#808080")
axes[0].legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The data stay inside the bootstrap range at all lags: the short-time
# force and diffusion reproduce the long-time behaviour.
#
# Save and load
# -------------
#
# :func:`~SFI.inference.save_results` writes a lightweight summary
# (coefficients, errors, metadata); :func:`~SFI.inference.save_model`
# writes the fitted force itself, reloaded against a *template* that
# supplies its structure.

import os
import tempfile

from SFI.inference import load_model, save_model, save_results

tmp = tempfile.mkdtemp()
save_results(inf, os.path.join(tmp, "ou_inference"))
save_model(inf.force_inferred, os.path.join(tmp, "ou_force"))
force_reloaded = load_model(os.path.join(tmp, "ou_force"), template=inf.force_inferred)

max_diff = float(jnp.max(jnp.abs(force_reloaded(xg) - inf.force_inferred(xg))))
print(f"Reloaded model matches the original: max |ΔF| = {max_diff:.1e}")

# %%
# Next steps
# ----------
#
# - **Your own data**: :doc:`/start_here` routes you to the right
#   estimator; :doc:`/gallery/experimental_workflow_demo` is the template
#   for a noisy recording, validated without ground truth.
# - **Noisy or coarsely sampled recordings**: the parametric estimators,
#   :doc:`/inference/noise_and_sampling`.
# - **Building models** (custom bases, interactions, masking):
#   :doc:`/bases/user_guide` and :doc:`/statefunc/user_guide`.
# - **Loading data** (CSV/Parquet/HDF5, several experiments):
#   :doc:`/trajectory/user_guide`.
# - **Inference in depth** (overdamped vs underdamped, diffusion
#   estimators, error analysis): :doc:`/inference/user_guide`.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
hist, edges = np.histogram(x_data, bins=50, range=(-2.5, 2.5), density=True)
ax_t.fill_between(0.5 * (edges[1:] + edges[:-1]), 0, 2.0 * hist, step="mid",
                  color=SFI_COLORS["data"], alpha=0.3, lw=0)
ax_t.plot(xg_np, U_grid, color=SFI_COLORS["inferred"], lw=3)
ax_t.set_xlim(-2.2, 2.2)
ax_t.set_ylim(-0.05, 2.4)
ax_t.set_xticks([]); ax_t.set_yticks([])
fig_thumb.canvas.draw()                      # fix the layout before placing beads in display units
size = 260                                   # bead marker area (pt²): rest it on the curve
r_px = (np.sqrt(size) / 2 + 1) * fig_thumb.dpi / 72
for xb, c in ((-1.0, SFI_COLORS["data"]), (0.75, SFI_COLORS["bootstrap"])):
    p0, p1 = ax_t.transData.transform([(xb, np.interp(xb, xg_np, U_grid)),
                                       (xb + 0.01, np.interp(xb + 0.01, xg_np, U_grid))])
    tang = (p1 - p0) / np.hypot(*(p1 - p0))
    normal = np.array([-tang[1], tang[0]]) * np.sign(tang[0])
    xc_, yc_ = ax_t.transData.inverted().transform(p0 + r_px * normal)
    ax_t.scatter([xc_], [yc_], s=size, marker="o", color=c, edgecolor="#D0D0D0", lw=1, zorder=5)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
