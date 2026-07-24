"""
Getting started: end-to-end inference (Ornstein–Uhlenbeck)
===========================================================

This is the recommended **starting point** for new users: a complete,
runnable Stochastic Force Inference (SFI) workflow on the simplest
non-trivial system.  Every step below is executed when the documentation
is built, so the numbers and figures on this page are real output.

We work with a one-dimensional **Ornstein–Uhlenbeck (OU) process** — a
particle in a harmonic trap.  Its motion obeys the simplest overdamped
Langevin equation,

.. math::

   \\mathrm{d}x = \\underbrace{-k\\,x}_{F(x)}\\,\\mathrm{d}t
              + \\sqrt{2D}\\;\\mathrm{d}W ,

a linear restoring force :math:`F(x) = -k\\,x` that pulls the particle
back toward the origin, plus white noise whose strength is set by the
diffusion constant :math:`D`.  Because the force is linear and the noise
constant, we know the exact answer in advance — which makes OU the ideal
system for *watching the method work*.

Across this page we will:

#. simulate a trajectory from a known model;
#. estimate the diffusion constant and infer the force;
#. let SFI **select the minimal model** out of a deliberately
   over-complete basis;
#. validate the fit with residual diagnostics and a bootstrapped
   trajectory;
#. save the result to disk and load it back.

The whole workflow is a handful of method calls.  Real experimental data
follows the *same* path — you just load your trajectories instead of
simulating them; see :doc:`/gallery/experimental_workflow_demo` for that
template.

.. rubric:: Tags

synthetic · overdamped · linear · 1D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D"]
# sphinx_gallery_thumbnail_number = 2

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
from SFI.utils.plotting import plot_profile_1d, plot_recovery_bar, timeseries

apply_style()
# sphinx_gallery_end_ignore

# %%
# Simulate a trajectory
# ---------------------
#
# Before we can *infer* anything we need data.  We build the exact OU
# model and integrate it forward with
# :class:`~SFI.langevin.OverdampedProcess`.
#
# The force is the linear restoring law :math:`F(x) = -k\,x`, which we
# write directly as ``-k_true * X(dim=1)`` — here :func:`~SFI.bases.X`
# is the coordinate :math:`x`.  The diffusion is a single constant,
# ``D = 0.5``.

import SFI
from SFI.langevin import OverdampedProcess
from SFI.bases import X

k_true = 1.0      # trap stiffness  →  F(x) = -k x
D_true = 0.5      # diffusion constant
dt = 0.1          # time between recorded frames
Nsteps = 500      # number of frames

proc = OverdampedProcess(F=-k_true * X(dim=1), D=D_true)  # F(x) = -k·x
proc.initialize(jnp.array([0.0]))                         # start at the origin

key = random.PRNGKey(137)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=key, oversampling=4)
print(f"Simulated {coll.T} frames at dt = {dt}")

# %%
# The object returned by :meth:`~SFI.langevin.OverdampedProcess.simulate`
# is a :class:`~SFI.trajectory.TrajectoryCollection` — the universal data
# container in SFI.  Everything downstream (inference, degradation,
# bootstrapping) consumes a collection, whether it came from a simulation
# or from a real recording loaded with :meth:`~SFI.trajectory.TrajectoryCollection.from_arrays`.
#
# The trajectory shows the hallmark of a harmonic trap: the particle
# fluctuates around the origin, repeatedly pulled back by the restoring
# force.

# sphinx_gallery_start_ignore
t, X_full, _ = coll.to_arrays(dataset=0)   # t:(T,), X_full:(T, N, d)
Xarr = X_full[:, 0, :]                      # single particle -> (T, 1)
fig, ax = plt.subplots(figsize=(8, 3))
timeseries(coll, ax=ax, lw=0.8, color=SFI_COLORS["data"])
ax.axhline(0, ls=":", lw=0.5, color="#808080")
ax.set_xlabel("Time")
ax.set_ylabel("x")
ax.set_title("Ornstein–Uhlenbeck trajectory")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Set up inference and estimate the diffusion
# -------------------------------------------
#
# Inference runs through an engine built from the data — here
# :class:`~SFI.OverdampedLangevinInference`.  The first quantity to pin
# down is the **diffusion constant**, which sets the noise scale and
# weights the force regression.
#
# Our data is clean and free of measurement noise, so we use the plain
# mean-squared-displacement estimator, ``method="MSD"``, which is exact
# in the fine-sampling limit.  On real recordings with localization
# noise, drop the argument: the default ``method="auto"`` then selects a
# noise-robust estimator instead (see
# :doc:`/inference/noise_and_sampling`).
#
# .. note::
#
#    More broadly, this tutorial uses SFI's **linear estimators**
#    throughout — the right choice for clean, finely-sampled data.  For
#    measurement noise, coarse sampling, or models nonlinear in their
#    parameters, switch to the **parametric estimators** (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`
#    / :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`); the regime table in
#    :ref:`choosing-an-estimator` tells you when.

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="MSD")

# %%
# Infer the force
# ---------------
#
# Now the central step.  In a real experiment we would *not* know that
# the force is linear, so we hand SFI a generic, deliberately
# over-complete library of candidate terms — the monomials
# :math:`\{1,\,x,\,x^2\}` — and let the data decide which it needs.
# :func:`~SFI.bases.monomials_up_to` builds that basis; ``rank='vector'``
# makes each term a force component.

from SFI.bases import monomials_up_to

B = monomials_up_to(order=2, dim=1, rank="vector")   # candidates: 1, x, x²

inf.infer_force_linear(B)        # closed-form regression onto the basis
inf.compute_force_error()        # predicted (sampling-noise) NMSE
inf.print_report()

# %%
# The report lists the fitted coefficients and a *predicted* normalized
# mean-squared error — SFI's own estimate of how well the force is pinned
# down by this much data, available without any ground truth.
#
# Because this is a synthetic benchmark we *can* check against the truth.
# The inferred force :math:`\hat F(x)`, with its 1-σ confidence band,
# overlays the exact :math:`F(x) = -k\,x` almost perfectly.

x_jnp = jnp.array(np.linspace(Xarr.min() - 0.3, Xarr.max() + 0.3, 200)[:, None])

# capture the full-basis coefficients before model selection prunes them
coeffs_full = np.asarray(inf.force_coefficients_full)
stderr_full = np.asarray(inf.force_coefficients_stderr)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(4, 3))
plot_profile_1d(
    coll, inf.force_inferred, exact_field=proc.force_sf, ax=ax,
    label_exact=r"Exact  $F = -kx$",
    label_inferred=f"Inferred (pred. NMSE = {inf.force_predicted_MSE:.3f})",
)
ax.set_xlabel("x")
ax.set_ylabel("F(x)")
ax.set_title("Force recovery")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Which terms does the data support? Model selection
# --------------------------------------------------
#
# Our basis had three candidate terms, but only one — the linear term —
# is really present.  Looking at the fitted coefficients with their 1-σ
# error bars already tells the story: the constant and quadratic terms
# are statistically indistinguishable from zero, while the linear
# coefficient sits squarely at :math:`-k`.

labels = ["1", "x", "x²"]
coeffs_exact = np.array([0.0, -k_true, 0.0])

# sphinx_gallery_start_ignore
full_support = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(4.5, 3))
plot_recovery_bar(
    coeffs_full, full_support,
    coeffs_true=coeffs_exact, support_true=full_support,
    labels=labels, stderr=stderr_full, ax=ax,
)
ax.set_xlabel("basis term")
ax.set_title("Force coefficients (full basis)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Rather than read the bars by eye, let SFI make the choice.
# :meth:`~SFI.OverdampedLangevinInference.sparsify_force` runs a model
# search scored by an information criterion — the default is **PASTIS**,
# SFI's parsimonious criterion — and keeps only the terms the data
# actually warrant.

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# %%
# PASTIS keeps the single linear term and discards the constant and
# quadratic — recovering the exact *structure* of the model, not just its
# values.  Selecting the minimal model both sharpens the estimate and
# makes it interpretable: the answer is "a harmonic trap", stated in one
# coefficient.
#
# Consistency diagnostics
# -----------------------
#
# With real data there is no exact model to compare against, so the
# canonical sanity check is statistical.  SFI recomputes the standardised
# residuals
# :math:`z_t = (\Delta x_t - \hat F(x_t)\,\Delta t)/\sqrt{2\hat D\,\Delta t}`
# and asks whether they look like independent :math:`\mathcal N(0,1)`
# draws.  :func:`SFI.diagnostics.assess` (also reachable as
# ``inf.diagnose()``) runs a panel of tests and returns a report.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig = plot_summary(report)
plt.show()
# sphinx_gallery_end_ignore

# %%
# A well-specified fit shows a pooled residual mean ≈ 0, standard
# deviation ≈ 1, no autocorrelation, normal tails, and a realised error
# consistent with the predicted one.  Residual autocorrelation would
# point to missing dynamics; a standard deviation far from 1 to a
# mis-estimated diffusion.  See the
# :ref:`Diagnostics user guide <diagnostics_user_guide>` for the full
# inventory.
#
# Bootstrapped trajectory
# -----------------------
#
# A second, dynamical check: simulate a *new* trajectory directly from
# the inferred model.  If the fit is faithful, this bootstrap should be
# statistically indistinguishable from the original data — same spread,
# same relaxation.  Inferred models are first-class simulators, so this
# is a single call.

key_boot = random.PRNGKey(0)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(key_boot)

_, Xb_full, _ = coll_boot.to_arrays(dataset=0)
Xb = Xb_full[:, 0, 0]                                 # bootstrapped x(t)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
timeseries(coll, ax=axes[0], lw=0.8, color=SFI_COLORS["data"], label="Data")
axes[0].axhline(0, ls=":", lw=0.5, color="#808080")
axes[0].set_xlabel("")
axes[0].set_ylabel("x")
axes[0].legend(loc="upper right")
axes[0].set_title("Data vs bootstrapped trajectory")

timeseries(coll_boot, ax=axes[1], lw=0.8, color=SFI_COLORS["bootstrap"], label="Bootstrap")
axes[1].axhline(0, ls=":", lw=0.5, color="#808080")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("x")
axes[1].legend(loc="upper right")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The trajectories look alike, but the sharper test is in *statistics the
# fit never used directly*.  SFI only sees **short-time increments**
# :math:`\Delta x` over one step ``dt``; everything at longer lags — how
# fast correlations decay, how far the particle wanders — is a genuine
# *prediction* of the inferred model.  Comparing those long-time
# statistics between the data and a bootstrap is one of the most
# informative consistency checks available without ground truth.
#
# We look at two: the autocorrelation
# :math:`C(\tau) = \langle x(t)\,x(t+\tau)\rangle` (how memory fades) and
# the mean-squared displacement
# :math:`\mathrm{MSD}(\tau) = \langle [x(t+\tau)-x(t)]^2\rangle` (how far
# it spreads).  For an OU process both are exact exponentials, drawn here
# as dotted references.

def acf(x, n_lags):
    x = x - x.mean()
    c = np.correlate(x, x, mode="full")[len(x) - 1:]
    return c[:n_lags] / c[0]

def msd(x, n_lags):
    return np.array([np.mean((x[k:] - x[:-k]) ** 2) if k else 0.0
                     for k in range(n_lags)])

n_lags = 50
lags = np.arange(n_lags) * dt
acf_data, acf_boot = acf(Xarr[:, 0], n_lags), acf(Xb, n_lags)
msd_data, msd_boot = msd(Xarr[:, 0], n_lags), msd(Xb, n_lags)
acf_exact = np.exp(-k_true * lags)                       # OU: C(τ)/C(0) = e^{-kτ}
msd_exact = 2 * (D_true / k_true) * (1 - np.exp(-k_true * lags))

# sphinx_gallery_start_ignore
fig, (axc, axm) = plt.subplots(1, 2, figsize=(8, 3))
axc.plot(lags, acf_data, color=SFI_COLORS["data"], label="Data")
axc.plot(lags, acf_boot, "--", color=SFI_COLORS["bootstrap"], label="Bootstrap")
axc.plot(lags, acf_exact, ":", color="#808080", label="Exact OU")
axc.axhline(0, ls=":", lw=0.5, color="#808080")
axc.set_xlabel(r"lag $\tau$"); axc.set_ylabel(r"$C(\tau)/C(0)$")
axc.set_title("Autocorrelation"); axc.legend()

axm.plot(lags, msd_data, color=SFI_COLORS["data"], label="Data")
axm.plot(lags, msd_boot, "--", color=SFI_COLORS["bootstrap"], label="Bootstrap")
axm.plot(lags, msd_exact, ":", color="#808080", label="Exact OU")
axm.set_xlabel(r"lag $\tau$"); axm.set_ylabel(r"MSD$(\tau)$")
axm.set_title("Mean-squared displacement"); axm.legend()
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore
#
# Data and bootstrap track each other and the exact OU curves across all
# lags: the inferred short-time force and diffusion reproduce the
# emergent long-time behaviour the fit never saw directly.

# %%
# Save and load
# -------------
#
# Finally, persist the result.  SFI offers two complementary serializers:
#
# - :func:`~SFI.inference.save_results` writes a lightweight **summary**
#   (coefficients, errors, metadata) as ``.npz`` + ``.json``;
# - :func:`~SFI.inference.save_model` writes the fitted **callable model**
#   so the inferred force can be re-evaluated later.  Reloading it needs a
#   *template* that supplies the basis structure (the dictionary of
#   functions is not pickled).
#
# We write to a temporary directory here to keep the example
# self-contained.

import os
import tempfile

from SFI.inference import load_model, load_results, save_model, save_results

tmp = tempfile.mkdtemp()

# 1. lightweight summary
save_results(inf, os.path.join(tmp, "ou_inference"))
summary = load_results(os.path.join(tmp, "ou_inference"))
print("Reloaded summary keys:", sorted(summary)[:5], "...")

# 2. the callable force model
save_model(inf.force_inferred, os.path.join(tmp, "ou_force"))
force_reloaded = load_model(os.path.join(tmp, "ou_force"),
                            template=inf.force_inferred)

# the reloaded model evaluates to the same force
max_diff = float(np.max(np.abs(
    np.asarray(force_reloaded(x_jnp)) - np.asarray(inf.force_inferred(x_jnp))
)))
print(f"Reloaded model matches original (max |Δ| = {max_diff:.2e})")

# %%
# Next steps
# ----------
#
# That is the full arc — simulate, infer, select, validate, persist — in
# a few dozen lines.  Where to go next:
#
# - **Real experimental data** follows the identical workflow; you skip
#   the "exact model" comparison and lean on diagnostics and bootstrapping
#   for validation.  Start from :doc:`/start_here` for the routing guide,
#   then use :doc:`/gallery/experimental_workflow_demo` as the template.
# - **Noisy or coarsely-sampled recordings** need the parametric
#   estimators — see :doc:`/inference/noise_and_sampling`.
# - **Building models** (custom bases, interactions, masking):
#   :doc:`/statefunc/user_guide` and :doc:`/bases/user_guide`.
# - **Trajectory I/O** (loading CSV/Parquet/HDF5, multi-experiment data):
#   :doc:`/trajectory/user_guide`.
# - **Inference in depth** (overdamped vs. underdamped, diffusion
#   methods, error analysis): :doc:`/inference/user_guide`.

stamp_output()
