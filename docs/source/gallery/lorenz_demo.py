"""
Lorenz attractor — overdamped inference
========================================

Infer the force field of a 3D Lorenz system from a single simulated
trajectory using polynomial basis functions.

This example demonstrates the full SFI workflow on a classic chaotic
system: simulate → infer → sparsify → validate → bootstrap.

.. rubric:: Tags

synthetic · overdamped · nonlinear · sparsity · 3D · benchmark
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "sparsity", "3D", "benchmark"]
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
from _gallery_utils.helpers import (
    apply_style,
    plot_pareto_front,
    plot_recovery_bar,
    stamp_output,
)

apply_style()
# sphinx_gallery_end_ignore
# %%
# System definition
# ------------------
#
# The classic chaotic Lorenz system:
#
# .. math::
#
#    \dot{x} = \sigma(y - x), \quad
#    \dot{y} = x(\rho - z) - y, \quad
#    \dot{z} = x\,y - \beta\,z
#
# with :math:`(\sigma, \rho, \beta) = (20, 8, 2)`, i.e. closer to
# the critical point than the classic parameters.
# We add isotropic noise :math:`D_0 = 5` and simulate a short trajectory.

from SFI.bases import named_scalars, unit_axes, x_components
from SFI.langevin import OverdampedProcess
from SFI.utils import plotting

# Ground-truth Lorenz force, written directly as a symbolic composition
# of the basis primitives (no ``make_sf``, no raw array math).  Named
# scalars carry the reference parameter values as defaults, so the
# simulator can bind them automatically.
x, y, z = x_components(3)
ex, ey, ez = unit_axes(3)
sigma_p, rho_p, beta_p = named_scalars(sigma=20.0, rho=8.0, beta=2.0)

F_true = (
    (sigma_p * (y - x)) * ex
    + (x * (rho_p - z) - y) * ey
    + (x * y - beta_p * z) * ez
)

D0 = 5.0
dt = 0.005
Nsteps = 2000
seed = 0

key = random.PRNGKey(seed)

proc = OverdampedProcess(F_true, D=jnp.eye(3) * D0)
proc.initialize(jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32))

key, sub = random.split(key)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=sub, prerun=2000, oversampling=10)


# %%
# Trajectory
# -----------
#
# The characteristic butterfly structure is visible even on a short,
# noisy trajectory.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
plotting.phase2d(coll, dims=(0, 2), ax=ax, cmap="viridis", linewidth=0.8)
ax.set_xlabel("x"); ax.set_ylabel("z")
ax.set_title("(x, z) projection")

ax = axes[1]
plotting.timeseries(coll, dims=(0, 1, 2), ax=ax, lw=0.6)
ax.set_title("Time series")

fig.suptitle("Lorenz attractor — simulated trajectory", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Inference and sparsification
# ------------------------------
#
# A degree-2 polynomial basis in 3D gives 30 features.  PASTIS
# sparsification recovers the 7-term Lorenz model.

from SFI.bases import monomials_up_to
from SFI import OverdampedLangevinInference

degree = 2
B = monomials_up_to(order=degree, dim=3, rank='vector')

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="auto")
inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, maxpoints=2000)

inf.print_report()

# %%
# Pareto front and sparse model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

plot_pareto_front(inf.force_sparsity_result, ax=axes[0])

plot_recovery_bar(np.array(inf.force_coefficients), inf.force_support, labels=B.labels, ax=axes[1])
axes[1].set_title(f"Sparse model: {len(inf.force_support)} features selected")

fig.suptitle("PASTIS model selection", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Bootstrap trajectory
# ---------------------
#
# Generate a bootstrapped trajectory from the inferred sparse model.

key_boot = random.PRNGKey(seed + 99)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(key_boot, oversampling=10)

# %%
# Data vs bootstrap comparison
# ------------------------------
#
# Phase-space projection and time series of the data (top) vs the
# bootstrapped trajectory (bottom), straight from the toolkit plotters
# on each collection.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plotting.phase2d(coll, dims=(0, 2), ax=axes[0, 0], cmap="viridis", linewidth=0.6)
axes[0, 0].set_title("Data — (x, z) projection")
plotting.phase2d(coll_boot, dims=(0, 2), ax=axes[0, 1], cmap="magma", linewidth=0.6)
axes[0, 1].set_title("Bootstrap — (x, z) projection")
plotting.timeseries(coll, dims=(0, 1, 2), ax=axes[1, 0], lw=0.5, alpha=0.8)
axes[1, 0].set_title("Data — time series")
plotting.timeseries(coll_boot, dims=(0, 1, 2), ax=axes[1, 1], lw=0.5, alpha=0.8)
axes[1, 1].set_title("Bootstrap — time series")
fig.suptitle("Lorenz — data vs bootstrap", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Diagnostics
# -----------
#
# Once a fit is in hand, :func:`~SFI.diagnostics.assess` recomputes
# standardised residuals
# :math:`z_t = (\Delta x_t - \hat F(x_t)\,\Delta t)/\sqrt{2\hat D\,\Delta t}`
# and runs a panel of statistical tests.  A well-specified model
# should have residuals that are i.i.d. :math:`\mathcal{N}(0,1)` —
# no autocorrelation, no excess kurtosis, and a realised NMSE
# consistent with the predicted (sampling-noise) value.
#
# Here we diagnose the **PASTIS-selected linear model** (Itô
# estimator).

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Lorenz — diagnostics (PASTIS linear model)", y=1.02)
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
