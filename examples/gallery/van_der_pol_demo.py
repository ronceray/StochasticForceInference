r"""
Van der Pol oscillator — underdamped inference
===============================================

Infer the force field of a stochastic Van der Pol oscillator from
position-only data.  The system has inertia, but only positions are
observed — as is typical of experimental tracking — so the velocity is
reconstructed from the positions by the underdamped estimators.

We fit an over-complete polynomial basis in :math:`(x, v)`, let PASTIS
find the three true terms, validate with a bootstrap and the residual
diagnostics, and then degrade the recording the way a real experiment
would: localisation noise on every position, and a camera ten times
slower.  The parametric estimator still finds the three terms with
honest error bars; a frame-rate sweep shows where and why it breaks.

.. rubric:: Tags

synthetic · underdamped · nonlinear · sparsity · 1D
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "nonlinear", "sparsity", "1D"]
# sphinx_gallery_thumbnail_number = 9

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings

import numpy as np
from jax import random
from matplotlib.ticker import MaxNLocator

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

apply_style()


def ratio_panel(ax, fits, labels, truth, width=0.3):
    """Selected coefficients / truth with ±2σ bars; one marker series per fit.

    ``fits`` is a list of ``(name, inf, color)``; terms selected outside the
    truth are marked with a cross at 0.
    """
    true_idx = [labels.index(k) for k in truth]
    sel = [set(int(i) for i in np.asarray(f[1].force_support)) for f in fits]
    shown = sorted(set(true_idx).union(*sel))
    offsets = np.linspace(-width / 2, width / 2, len(fits)) if len(fits) > 1 else [0.0]
    ax.axhline(1.0, color="#808080", lw=0.8, ls="--")
    for (name, f, color), off in zip(fits, offsets):
        sup = [int(i) for i in np.asarray(f.force_support)]
        val = dict(zip(sup, np.asarray(f.force_coefficients)))
        err = dict(zip(sup, np.asarray(f.force_coefficients_stderr)))
        first = True
        for j, i in enumerate(shown):
            if i in true_idx:
                t = truth[labels[i]]
                ax.errorbar(j + off, val.get(i, 0.0) / t, yerr=2 * err.get(i, 0.0) / abs(t),
                            fmt="o", ms=6, capsize=4, color=color, ecolor=color,
                            label=name if first else None)
                first = False
            elif i in val:
                ax.plot(j + off, 0.0, "x", ms=8, color=SFI_COLORS["error"])
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([labels[i] for i in shown], fontsize=9)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_ylabel("inferred / true")
# sphinx_gallery_end_ignore
stamp_output()

# %%
# System definition
# ------------------
#
# The stochastic Van der Pol oscillator,
#
# .. math::
#
#    \ddot{x} = \mu (1 - x^2)\,\dot{x} - x + \sqrt{2D}\,\xi(t),
#
# with :math:`\mu = 3` (a strongly nonlinear relaxation oscillation,
# period :math:`\approx 8.9`) and :math:`D = 0.05`.  We record
# 8000 positions at :math:`\Delta t = 0.02` — about eighteen cycles.  The
# friction :math:`-\partial_v F = \mu(x^2 - 1)` reaches about 10 at the
# turning points, so the simulator takes 40 internal steps per frame to
# keep the synthetic data accurate.

from SFI.bases import named_scalar, unit_axes, v_components, x_components
from SFI.langevin import UnderdampedProcess
from SFI.utils import plotting

mu = 3.0
D0 = 0.05
dt = 0.02
Nsteps = 8000
oversampling = 40

# Ground-truth force F(x, v) = mu (1 - x²) v - x, written compositionally.
(_x,) = x_components(1)
(_v,) = v_components(1)
(_ex,) = unit_axes(1)
mu_p = named_scalar("mu", default=mu)
F_exact = (mu_p * (1 - _x * _x) * _v - _x) * _ex

proc = UnderdampedProcess(F_exact, D=jnp.array([[D0]]))
proc.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(0),
                     prerun=200, oversampling=oversampling)

t, X, _ = coll.to_arrays(dataset=0)   # t:(T,), X:(T, N, d)
x = X[:, 0, 0]

# %%
# Phase portrait
# ---------------
#
# The limit cycle in the :math:`(x, \dot{x})` plane.  The velocity shown
# here is a plain central difference of the positions — the inference
# itself works from the same position record.

# sphinx_gallery_start_ignore
v_fd = coll.velocity_array(dataset=0, scheme="central")[:, 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
sc = ax.scatter(x, v_fd, c=t, s=0.4, cmap="viridis", alpha=0.7, rasterized=True)
ax.set_xlabel("x")
ax.set_ylabel("v (central difference)")
ax.set_title("Phase portrait (time-coloured)")
plt.colorbar(sc, ax=ax, label="t", shrink=0.8)

ax = axes[1]
plotting.timeseries(coll, dims=[0], ax=ax, lw=0.6, color=SFI_COLORS["data"])
ax.set_xlim(0, 60)
ax.set_title("Position time series (first 60 time units)")
fig.suptitle("Van der Pol oscillator — trajectory data", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Force inference and model selection
# ------------------------------------
#
# Monomials in :math:`(x, v)` up to degree 3 give 10 candidate terms; the
# truth uses three (:math:`x`, :math:`v`, :math:`x^2 v`).  The data are
# clean, so we use the ``"clean"`` preset of the linear underdamped
# estimator (the automatic choice would also guard against measurement
# noise) and the ``WeakNoise`` diffusion estimator, suited to a strong
# drift; PASTIS then selects the model.

from SFI import UnderdampedLangevinInference
from SFI.bases import monomials_up_to
from SFI.utils.plotting import plot_pareto_front

B = monomials_up_to(order=3, dim=1, include_v=True, rank="vector")
labels = list(B.labels)
truth = {"x0·e0": -1.0, "v0·e0": mu, "(x0^2·v0)·e0": -mu}

inf = UnderdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B, preset="clean")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, data_exact=coll, maxpoints=2000)
inf.print_report()

# The selected coefficients against the truth, in units of their standard error.
c_lin, cov_lin = inf.coeff_block(slice(None))        # full-basis order
for lbl, tv in truth.items():
    i = labels.index(lbl)
    err = np.sqrt(cov_lin[i, i])
    print(f"{lbl:>13} = {c_lin[i]:+.3f} ± {err:.3f}   true {tv:+.0f}   "
          f"off by {100 * (c_lin[i] / tv - 1):+.1f}% = {(c_lin[i] - tv) / err:+.1f}σ")

# %%
# Left: the Pareto front — information gain rises steeply up to three
# terms and then flattens, so PASTIS stops at three.  Right: each selected
# coefficient divided by its true value, with ±2σ error bars.  All three
# overshoot the truth by a few per cent — several standard errors (printed
# above): this is the :math:`O(\Delta t)` bias of the linear estimator,
# which the next sections diagnose and remove.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
axes[0].set_title(f"Pareto front — PASTIS keeps {len(np.asarray(inf.force_support))} / {len(labels)}")
ratio_panel(axes[1], [("linear, clean data", inf, SFI_COLORS["inferred"])], labels, truth)
axes[1].set_title("Selected coefficients (±2σ)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Force validation
# -----------------
#
# Exact vs inferred force along the trajectory (every frame, so that the
# arrays follow the time series).

F_true, F_pred = inf.force_comparison_arrays(model_exact=proc, maxpoints=Nsteps)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plt.sca(axes[0])
plotting.comparison_scatter(F_true, F_pred, maxpoints=4000, alpha=0.15,
                            color=SFI_COLORS["data"])
axes[0].set_xlabel(r"exact force $F$")
axes[0].set_ylabel(r"inferred force $\hat F$")
axes[0].xaxis.set_major_locator(MaxNLocator(5))   # comparison_scatter keeps only a 0 tick
axes[0].yaxis.set_major_locator(MaxNLocator(5))

T2 = min(600, len(F_true))
tt = np.arange(T2) * dt
axes[1].plot(tt, F_true[:T2], lw=2.2, label="Exact", color=SFI_COLORS["exact"])
axes[1].plot(tt, F_pred[:T2], lw=0.9, label="Inferred", color=SFI_COLORS["inferred"])
axes[1].set_xlabel("time")
axes[1].set_ylabel("F")
axes[1].set_title("Force along the trajectory")
axes[1].legend()
fig.suptitle("Underdamped force inference — validation", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Bootstrapped trajectory
# -------------------------
#
# A trajectory simulated from the inferred sparse model should trace the
# same limit cycle.

coll_boot, _ = inf.simulate_bootstrapped_trajectory(
    random.PRNGKey(1), oversampling=oversampling, simulate=True
)
_, X_boot, _ = coll_boot.to_arrays(dataset=0)
x_boot = X_boot[:, 0, 0]
v_fd_boot = coll_boot.velocity_array(dataset=0, scheme="central")[:, 0, 0]

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(x, v_fd, lw=0.3, alpha=0.6, label="Data", color=SFI_COLORS["data"])
ax.plot(x_boot, v_fd_boot, lw=0.3, alpha=0.5, label="Bootstrap",
        color=SFI_COLORS["inferred"])
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title("Phase portrait: data vs bootstrap")
for line in ax.legend().get_lines():          # legend lines thicker than the traces
    line.set(linewidth=2, alpha=1)

ax = axes[1]
plotting.timeseries(coll, dims=[0], ax=ax, lw=0.6, label="Data",
                    color=SFI_COLORS["data"])
plotting.timeseries(coll_boot, dims=[0], ax=ax, lw=0.6, alpha=0.8,
                    label="Bootstrap", color=SFI_COLORS["inferred"])
ax.set_xlim(0, 60)
ax.set_ylim(-2.4, 3.0)                         # headroom for the legend
ax.set_title("Time series comparison")
for line in ax.legend(loc="upper center", ncol=2).get_lines():
    line.set(linewidth=2, alpha=1)
fig.suptitle("Generative validation — bootstrapped trajectory", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Diagnostics
# -----------
#
# The underdamped residual compares the second-difference acceleration
# :math:`(x_{t+1} - 2x_t + x_{t-1})/\Delta t^2` — the same kinematics the
# estimator fits — to the inferred force, standardised by the noise of a
# second difference of integrated white noise.  Adjacent accelerations
# overlap, so every second sample is used.
#
# The flags point at the coefficient bias seen above.  ``finite_dt``:
# :math:`\Delta t\,|\partial_v F|` reaches ~0.1–0.2 along the fast
# branches of the cycle, where the linear estimator carries an
# :math:`O(\Delta t)` bias of several per cent.  The residual
# autocorrelation flag is its other face: with this much data, the
# systematic error of the fitted force along the cycle shows up as
# correlated residuals.  The parametric estimator, used next, integrates
# the dynamics across each sampling interval and removes it.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Van der Pol — diagnostics")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Robustness: localisation noise
# ------------------------------
#
# Real tracking adds a localisation error to every position.  We blur the
# positions with :math:`\sigma = 0.05` — 2.5% of the oscillation
# amplitude, barely visible on the trajectory.  For inertial data this is
# severe: a naive finite-difference velocity carries an error
# :math:`\sigma/(\sqrt{2}\,\Delta t)`, and the second-difference
# acceleration :math:`\sqrt{6}\,\sigma/\Delta t^2`.

sigma = 0.05
coll_noisy = coll.degrade(noise=sigma, seed=3)
print(f"naive velocity error      sigma/(sqrt(2) dt) = {sigma / (np.sqrt(2) * dt):.2f}"
      f"   (speeds up to {np.abs(v_fd).max():.1f})")
print(f"naive acceleration error  sqrt(6) sigma/dt^2 = {np.sqrt(6) * sigma / dt**2:.0f}"
      f"   (|F| up to {np.abs(F_true).max():.1f})")
F_rms = np.sqrt(np.mean(F_true**2))
print(f"force signal per step     rms(F) dt^2 = {F_rms * dt**2:.1e} at dt = {dt}, "
      f"{F_rms * (10 * dt)**2:.1e} at dt = {10 * dt:.1f}   (sigma = {sigma})")

# %%
# Each frame-to-frame step carries a force signal of order
# :math:`F\Delta t^2` against a position error :math:`\sigma`.  At the
# full frame rate the first is about :math:`2\times 10^{-3}`, the second
# :math:`5\times 10^{-2}` (printed above): every step is noise-dominated,
# and the acceleration — hence the force — is lost.  Keeping one frame in
# ten multiplies :math:`F\Delta t^2` by a hundred, to about four times
# :math:`\sigma`, and leaves :math:`\sigma` unchanged.
# ``degrade(downsample=10)`` does that: the same noisy recording, filmed
# with a camera ten times slower (800 frames, :math:`\Delta t = 0.2`,
# about 45 per cycle and only a handful per fast jump).

stride = 10
coll_coarse = coll_noisy.degrade(downsample=stride)
dt_coarse = stride * dt

# sphinx_gallery_start_ignore
v_noisy = coll_noisy.velocity_array(scheme="central")[:, 0, 0]
_, Xn, _ = coll_noisy.to_arrays()
xn = Xn[:, 0, 0]
tc, Xc, _ = coll_coarse.to_arrays()
xc = Xc[:, 0, 0]
v_coarse = coll_coarse.velocity_array(scheme="central")[:, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axes[0]
win = (t >= 22.0) & (t <= 30.0)
winc = (tc >= 22.0) & (tc <= 30.0)
ax.plot(t[win], x[win], lw=1.2, color=SFI_COLORS["exact"], label="true position")
ax.plot(t[win], xn[win], ".", ms=1.8, color=SFI_COLORS["data"], alpha=0.7,
        label=f"noisy frames, Δt = {dt}")
ax.plot(tc[winc], xc[winc], "o", ms=5, mfc="none", mew=1.3, color=SFI_COLORS["highlight"],
        label=f"one frame in {stride}, Δt = {dt_coarse:.1f}")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title(f"Localisation noise σ = {sigma}")
ax.legend(fontsize=7.5, loc="lower right")

for ax, xx, vv, ttl in ((axes[1], xn, v_noisy, f"Naive velocity, Δt = {dt}"),
                        (axes[2], xc, v_coarse, f"Naive velocity, Δt = {dt_coarse:.1f}")):
    ax.plot(x, v_fd, lw=0.5, color=SFI_COLORS["exact"], alpha=0.6, label="true cycle")
    ax.plot(xx, vv, ".", ms=1.5 if xx is xn else 3, color=SFI_COLORS["data"] if xx is xn
            else SFI_COLORS["highlight"], alpha=0.35 if xx is xn else 0.8, rasterized=True)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("x")
    ax.set_title(ttl)
axes[1].set_ylabel("finite-difference v")
axes[1].legend(fontsize=7.5, loc="upper left")
fig.suptitle("What the noise does to naive kinematics", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Left: the noise is a small blur on the positions.  Middle: at the full
# frame rate the finite-difference velocity is swamped.  Right: on the
# slow frames it is usable on the slow branches, but the fast jumps are
# cut into a few chords — a naive analysis would get the dynamics wrong
# there.
#
# Parametric inference on noisy, coarse frames
# ---------------------------------------------
#
# The parametric estimator
# (:meth:`~SFI.inference.UnderdampedLangevinInference.infer_force`)
# models the localisation noise explicitly and estimates it, and it
# integrates the dynamics across each frame interval instead of
# differencing.  Over :math:`\Delta t = 0.2` the force changes a lot on
# the fast branches, so ``n_substeps`` splits each interval into RK4
# sub-steps no longer than half the fastest relaxation time of the cycle,
# :math:`1/\max|\partial_v F| \approx 1/10` (the friction at the turning
# points).  PASTIS then selects the terms as before.

lam_max = mu * (np.abs(x).max() ** 2 - 1)          # max |dF/dv| on the cycle


def n_substeps_for(dt_obs):
    """RK4 sub-steps no longer than 1 / (2 lam_max)."""
    return int(np.ceil(2 * lam_max * dt_obs))


inf_coarse = UnderdampedLangevinInference(coll_coarse)
inf_coarse.infer_force(B, n_substeps=n_substeps_for(dt_coarse))
inf_coarse.compute_force_error()
nmse_pred_coarse = inf_coarse.force_predicted_MSE      # full 10-term fit (kept for the sweep)
inf_coarse.sparsify_force(criterion="PASTIS")
inf_coarse.compute_force_error()
inf_coarse.compare_to_exact(force_exact=proc.force_sf, data_exact=coll, maxpoints=2000)
inf_coarse.print_report()

sigma_hat = float(np.sqrt(np.asarray(inf_coarse.Lambda).ravel()[0]))
D_hat = float(np.asarray(inf_coarse.diffusion_average).ravel()[0])
print(f"max |dF/dv| = {lam_max:.1f}  ->  n_substeps = {n_substeps_for(dt_coarse)}")
print(f"estimated localisation noise sigma = {sigma_hat:.4f}  (true {sigma})")
print(f"estimated diffusion constant D     = {D_hat:.3f}   (true {D0})")
c_par, cov_par = inf_coarse.coeff_block(slice(None))
for lbl, tv in truth.items():
    i = labels.index(lbl)
    err = np.sqrt(cov_par[i, i])
    print(f"{lbl:>13} = {c_par[i]:+.3f} ± {err:.3f}   true {tv:+.0f}   "
          f"off by {100 * (c_par[i] / tv - 1):+.1f}% = {(c_par[i] - tv) / err:+.1f}σ")

# %%
# From 800 noisy frames the fit selects exactly the three true terms, and
# the noise level comes out close to the truth.  Its error bars are wider
# than on the clean data — the price of the noise and of the
# tenfold-fewer frames — and they are honest: the coefficients sit within
# about one standard error of the truth, free of the
# :math:`O(\Delta t)` bias of the linear fit.  The diffusion constant, on
# the other hand, is not measurable at this frame rate: the estimate
# printed above is far too large — across a fast jump the thermal kicks
# are a small part of the frame-to-frame scatter the fit has to explain.
# The force is what the slow frames still resolve.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ratio_panel(ax, [("linear, clean, Δt = 0.02", inf, SFI_COLORS["inferred"]),
                 (f"parametric, σ = {sigma}, Δt = {dt_coarse:.1f}", inf_coarse,
                  SFI_COLORS["highlight"])], labels, truth)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Recovered coefficients (±2σ)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Where does it break?
# --------------------
#
# The same noisy recording, fitted at six frame rates — every
# :math:`s`-th frame, for strides from 1 to 20 — with the sub-steps
# scaled to the interval as above.  We keep both the full 10-term fit's
# predicted error (``force_predicted_MSE``, available without knowing the
# truth) and the PASTIS-selected model.

strides = [1, 3, 5, 10, 15, 20]
sweep = {}
for s in strides:
    if s == stride:                                    # already fitted above
        sweep[s] = (inf_coarse, nmse_pred_coarse)
        continue
    inf_s = UnderdampedLangevinInference(coll_noisy.degrade(downsample=s))
    with warnings.catch_warnings():
        # at the finest stride the fit finds no force and warns; the table says so
        warnings.filterwarnings("ignore", message="infer_force: the parametric fit returned an all-zero")
        inf_s.infer_force(B, n_substeps=n_substeps_for(s * dt))
    inf_s.compute_force_error()
    nmse_pred = inf_s.force_predicted_MSE
    inf_s.sparsify_force(criterion="PASTIS")
    inf_s.compute_force_error()
    sweep[s] = (inf_s, nmse_pred)

true_idx = sorted(labels.index(lbl) for lbl in truth)
print(" stride    dt  frames  F dt²/σ  predicted NMSE   PASTIS selection")
for s in strides:
    inf_s, nmse_pred = sweep[s]
    sup = sorted(int(i) for i in np.asarray(inf_s.force_support))
    if sup == true_idx:
        found = "the 3 true terms"
    else:
        found = "wrong: " + (", ".join(labels[i] for i in sup) if sup else "no term")
    # A fit that finds no force at all has zero information: its NMSE is undefined.
    nmse_txt = f"{nmse_pred:.3g}" if np.isfinite(nmse_pred) else "no force found"
    print(f"  {s:>4}  {s * dt:.2f}  {Nsteps // s:>6}  {F_rms * (s * dt)**2 / sigma:7.2f}"
          f"  {nmse_txt:>14}   {found}")

# sphinx_gallery_start_ignore
dts = np.array(strides) * dt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
ax.axhline(1.0, color="#808080", lw=0.8, ls="--")
cols = [SFI_COLORS["inferred"], SFI_COLORS["highlight"], SFI_COLORS["tertiary"]]
for j, (lbl, tv) in enumerate(truth.items()):
    i = labels.index(lbl)
    for k, s in enumerate(strides):
        inf_s = sweep[s][0]
        sup = [int(q) for q in np.asarray(inf_s.force_support)]
        if sorted(sup) != true_idx:
            continue
        c = np.asarray(inf_s.force_coefficients)[sup.index(i)]
        e = np.asarray(inf_s.force_coefficients_stderr)[sup.index(i)]
        ax.errorbar(dts[k] * (1 + 0.06 * (j - 1)), c / tv, yerr=2 * e / abs(tv), fmt="o",
                    ms=5, capsize=3, color=cols[j], label=lbl if k == strides.index(stride) else None)
for k, s in enumerate(strides):
    sup = sorted(int(q) for q in np.asarray(sweep[s][0].force_support))
    if sup != true_idx:
        ax.plot(dts[k], 0.06, "x", ms=11, mew=2, color=SFI_COLORS["error"],
                transform=ax.get_xaxis_transform())
ax.plot([], [], "x", ms=9, mew=2, color=SFI_COLORS["error"], label="wrong model selected")
ax.set_xscale("log")
ax.set_xticks(dts)
ax.set_xticklabels([f"{d:g}" for d in dts])
ax.minorticks_off()
ax.set_xlim(dts[0] / 1.25, dts[-1] * 1.25)
ax.set_xlabel("frame interval Δt")
ax.set_ylabel("inferred / true  (±2σ)")
ax.set_title(f"PASTIS-selected coefficients, σ = {sigma}")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
nm = np.array([sweep[s][1] for s in strides], dtype=float)
ok = np.isfinite(nm)
ax.plot(dts[ok], nm[ok], "o-", color=SFI_COLORS["data"])
for k in np.flatnonzero(~ok):
    ax.plot(dts[k], 0.94, "x", ms=10, mew=2, color=SFI_COLORS["error"],
            transform=ax.get_xaxis_transform())
    ax.annotate("no force found", (dts[k], 0.86), xycoords=("data", "axes fraction"),
                ha="left", va="top", fontsize=8, color=SFI_COLORS["error"])
ax.set_xscale("log")
ax.set_yscale("log")
yt = [v for v in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
      if 0.5 * nm[ok].min() <= v <= 2 * nm[ok].max()]
ax.set_yticks(yt)
ax.set_yticklabels([f"{v:g}" for v in yt])
ax.set_xticks(dts)
ax.set_xticklabels([f"{d:g}" for d in dts])
ax.minorticks_off()
ax.set_xlim(dts[0] / 1.25, dts[-1] * 1.25)
ax.set_xlabel("frame interval Δt")
ax.set_ylabel("predicted NMSE")
ax.set_title("Full-basis predicted error")
for a in axes:   # shade the frame rates where PASTIS recovers the true model
    for k, s in enumerate(strides):
        if sorted(int(q) for q in np.asarray(sweep[s][0].force_support)) == true_idx:
            a.axvspan(dts[k] / 1.12, dts[k] * 1.12, color=SFI_COLORS["highlight"], alpha=0.12, lw=0)
fig.suptitle("Frame-rate sweep on the same noisy recording", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Two walls bound the usable window (shaded).  On the left, fast frames:
# the force signal per step, :math:`F\Delta t^2`, is below the
# localisation noise (column ``F dt²/σ``).  At the full frame rate the fit
# finds no force at all and PASTIS selects nothing; at strides 3 and 5
# the predicted error is still large and the selection is wrong.  The
# wall sits where :math:`F\Delta t^2` climbs past :math:`\sigma`.  On the right,
# slow frames: with a frame or two per fast jump the trajectory between
# frames is no longer pinned down, the predicted error climbs again and
# the selection fails at :math:`\Delta t = 0.4`.  In between, the
# predicted error — computed without the truth — is smallest where the
# selected coefficients are closest to the truth.  Hence a practical rule
# for a noisy recording: fit a few strides, keep the one with the
# smallest predicted error, and do not trust a selection made from a fit
# whose predicted error is large.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
ax_t.plot(xc, v_coarse, "o", ms=1.6, color=SFI_COLORS["highlight"], alpha=0.45, zorder=2)
ax_t.scatter(x, v_fd, c=t, s=0.6, cmap="viridis", alpha=0.9, rasterized=True, zorder=3)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
