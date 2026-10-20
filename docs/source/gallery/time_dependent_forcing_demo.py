r"""
Time-dependent forcing — protocols as extras
============================================

An experiment often comes with a **recorded protocol** — a stiffness, a
field, a temperature switched or ramped during the run.  Here a trap is
switched between two stiffnesses, and we infer from a single 1D trajectory
how the force depends on the protocol.

The protocol :math:`k(t)` enters SFI as a *time-dependent extra*: the
simulator consumes it frame by frame and attaches it to the output
collection, and the inference reads it back through the basis symbol
:func:`~SFI.bases.extra_scalar`.  A final section adds localisation noise
and switches to the parametric estimator, which reads the protocol the
same way.

.. rubric:: Tags

synthetic · overdamped · linear · 1D · time-dependent
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D", "time-dependent"]
# sphinx_gallery_thumbnail_number = 4

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
from SFI.utils.plotting import timeseries

apply_style()
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Model: a trap with switched stiffness
# -------------------------------------
#
# An overdamped particle in a harmonic trap whose stiffness alternates
# between two plateaus:
#
# .. math::
#
#    dx = -\bigl[k_0 + k(t)\bigr]\,x\,dt + \sqrt{2D}\,dW,
#
# with :math:`k_0 = 1`, a square-wave drive :math:`k(t) \in \{0, 1\}`,
# and :math:`D = 0.25`.  The force model is built compositionally: the
# drive is a basis symbol read from the dataset's extras.

import SFI
from SFI.bases import X, extra_scalar
from SFI.langevin import OverdampedProcess
from SFI.trajectory import time_series_extra

dt = 0.01
Nsteps = 60_000     # 600 time units
period = 1_000      # frames per plateau: 30 on/off cycles

k_protocol = (np.arange(Nsteps) // period % 2).astype(float)

# Two-feature basis: static trap x, and protocol-modulated trap k(t)·x.
B = X(dim=1) & (extra_scalar("k_drive", dim=1) * X(dim=1))

proc = OverdampedProcess(F=B, D=0.25, theta_F=jnp.array([-1.0, -1.0]))
proc.set_extras(extras_global={"k_drive": time_series_extra(k_protocol)})
proc.initialize(jnp.zeros(1))

coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(42), oversampling=4)

# %%
# Trajectory and protocol
# -----------------------
#
# When the drive is on, the trap is twice as stiff: the variance
# :math:`D/k` of the fluctuations halves and they relax twice as fast — a
# subtle change by eye (first 40 of 600 time units shown), but a clear
# signal for the inference.

# sphinx_gallery_start_ignore
t, X_traj, _ = coll.to_arrays()  # time axis for the protocol step panel
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3]})
axes[0].step(t, 1.0 + k_protocol, where="post", color=SFI_COLORS["exact"], lw=1.2)
axes[0].set_ylabel(r"$k_0 + k(t)$")
axes[0].set_yticks([1, 2])
timeseries(coll, dims=[0], ax=axes[1], lw=0.5, color=SFI_COLORS["data"])
axes[1].set_xlim(0, 40)
x_shown = X_traj[t <= 40].ravel()
axes[1].set_ylim(1.1 * x_shown.min(), 1.1 * x_shown.max())
axes[1].set_xlabel("time")
axes[1].set_ylabel("$x(t)$")
fig.suptitle("Switched-stiffness trap: protocol and trajectory")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inference with the protocol as a basis symbol
# ---------------------------------------------
#
# The *same* two-feature dictionary serves for inference: the linear
# estimator reads ``k_drive`` per frame from the collection's extras
# (the simulator attached it), so the static and driven contributions
# are separated by the protocol's variation.

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="MSD")      # clean data, noise-dominated steps
inf.infer_force_linear(B, preset="clean")
inf.compute_force_error()
inf.print_report()

# %%
# Recovered stiffness law
# -----------------------
#
# The two fitted coefficients reconstruct the time-dependent stiffness
# :math:`\hat k(t) = -[\hat\theta_1 + \hat\theta_2\,k(t)]`, compared with
# the ground truth.  The 1σ band propagates the full coefficient covariance
# returned by :meth:`~SFI.inference.OverdampedLangevinInference.coeff_block`
# (the two coefficients are anti-correlated: both explain the trap when the
# drive is on).

# sphinx_gallery_start_ignore
coeffs, cov = inf.coeff_block(B, field="force")
k_hat = -(coeffs[0] + coeffs[1] * k_protocol)
design = np.stack([np.ones_like(k_protocol), k_protocol], axis=1)   # d k_hat / d theta
k_std = np.sqrt(np.einsum("ti,ij,tj->t", design, cov, design))
k_true = 1.0 + k_protocol

fig, ax = plt.subplots(figsize=(8, 3))
ax.step(t, k_true, where="post", color=SFI_COLORS["exact"], lw=1.5, label="true $k_0+k(t)$")
ax.step(t, k_hat, where="post", color=SFI_COLORS["inferred"], lw=1.5, ls="--",
        label="inferred")
ax.fill_between(t, k_hat - k_std, k_hat + k_std,
                color=SFI_COLORS["inferred"], alpha=0.2, step="post")
ax.set_xlim(0, 60)
ax.set_xlabel("time")
ax.set_ylabel("trap stiffness")
ax.legend(loc="center right")
ax.set_title("Time-dependent stiffness: truth vs inference (1σ band)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# With measurement noise
# ----------------------
#
# Real tracking adds localisation noise.  Here every recorded position
# gets Gaussian noise of standard deviation :math:`\sigma = \sqrt{2D\Delta t}`
# — as large as a typical diffusive step.  The ``"clean"`` preset used
# above assumes noise-free positions and is now strongly biased.  The
# parametric estimator :meth:`~SFI.OverdampedLangevinInference.infer_force`
# profiles the noise amplitude :math:`\Lambda = \sigma^2` jointly with the
# force and reads the protocol frame by frame, exactly as the linear
# estimator does — the basis is unchanged.  For comparison, we also run
# the linear estimator with its default, noise-robust settings.

sigma = float(np.sqrt(2 * 0.25 * dt))
coll_noisy = coll.degrade(noise=sigma, seed=1)

inf_clean = SFI.OverdampedLangevinInference(coll_noisy)
inf_clean.compute_diffusion_constant(method="MSD")
inf_clean.infer_force_linear(B, preset="clean")
inf_clean.compute_force_error()

inf_default = SFI.OverdampedLangevinInference(coll_noisy)
inf_default.compute_diffusion_constant()
inf_default.infer_force_linear(B)
inf_default.compute_force_error()

inf_par = SFI.OverdampedLangevinInference(coll_noisy)
inf_par.infer_force(B)
inf_par.compute_force_error()


def plateaus(fit):
    """Stiffness with the drive off and on, -θ1 and -(θ1 + θ2), with 1σ errors."""
    c, cov = fit.coeff_block(B)
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    return -(design @ c), np.sqrt(np.einsum("pi,ij,pj->p", design, cov, design))


fits = {"linear, clean preset": inf_clean, "linear, default": inf_default,
        "parametric": inf_par}
for name, fit in fits.items():
    k, k_err = plateaus(fit)
    print(f"{name:21s} k_off = {k[0]:.2f} ± {k_err[0]:.2f}   k_on = {k[1]:.2f} ± {k_err[1]:.2f}")
print(f"{'truth':21s} k_off = 1.00          k_on = 2.00")
print(f"localisation noise: true σ = {sigma:.4f}, "
      f"parametric estimate {np.sqrt(float(np.ravel(inf_par.Lambda)[0])):.4f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hlines([1.0, 2.0], [-0.35, 0.65], [0.35, 1.35], color=SFI_COLORS["exact"], lw=2,
          label="true stiffness")
colors = [SFI_COLORS["error"], SFI_COLORS["data"], SFI_COLORS["inferred"]]
for j, ((name, fit), color) in enumerate(zip(fits.items(), colors)):
    k, k_err = plateaus(fit)
    ax.errorbar(np.arange(2) + 0.15 * (j - 1), k, yerr=k_err, fmt="o", ms=6,
                color=color, ecolor=color, capsize=4, label=name)
ax.set_xticks([0, 1], ["drive off", "drive on"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("trap stiffness")
ax.legend(loc="upper left")
ax.set_title(r"Noisy positions, $\sigma = \sqrt{2D\Delta t}$ (1σ error bars)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The clean preset mistakes the noise-induced jitter for a much stiffer
# trap.  Both noise-aware estimators recover the two plateaus within their
# error bars.  The parametric fit also returns the noise amplitude, and it
# is the estimator of choice when the sampling is coarse.
#
# Notes
# -----
#
# - In **simulation**, ``set_extras`` accepts a
#   :class:`~SFI.trajectory.TimeSeriesExtra` (one value per recorded
#   frame, held across oversampling substeps) or a plain callable
#   ``f(t)`` of physical time; the schedule is attached to the returned
#   collection.
# - In **inference**, any time-dependent extra in the dataset is sliced
#   per frame automatically; :func:`~SFI.bases.extra_scalar` turns it
#   into a composable basis symbol.
# - The static and driven features are collinear up to the protocol's
#   variation — strongly varying protocols (switches, large ramps)
#   identify the split best, and ``force_coefficients_stderr`` reports
#   exactly how well.
# - The parametric estimator evaluates each observation interval with
#   the protocol value at its first frame — the convention the simulator
#   uses — so measurement noise and recorded protocols combine without any
#   extra setup.  For a drive that varies *within* a frame (a continuous
#   oscillation of angular frequency :math:`\omega`), this left-point hold
#   lags the drive by half a frame, a phase :math:`\approx\omega\Delta t/2`:
#   negligible when the drive is slow compared with the sampling.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
shown = t <= 40.0
for t_on in t[shown][np.flatnonzero(np.diff(k_protocol[shown]) > 0) + 1]:   # drive-on plateaus
    ax_t.axvspan(t_on, t_on + period * dt, color=SFI_COLORS["exact"], alpha=0.12, lw=0)
ax_t.plot(t[shown], X_traj[shown, 0, 0], lw=0.5, color=SFI_COLORS["data"])
lift = 1.1 * np.abs(X_traj[shown, 0, 0]).max()       # stiffness steps drawn above the trace
ax_t.step(t[shown], lift + 0.6 * k_true[shown], where="post", color=SFI_COLORS["exact"], lw=2.5)
ax_t.step(t[shown], lift + 0.6 * k_hat[shown], where="post", color=SFI_COLORS["inferred"],
          lw=1.5, ls="--")
ax_t.set_xlim(0, 40)
ax_t.set_ylim(-lift, lift + 1.45)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
