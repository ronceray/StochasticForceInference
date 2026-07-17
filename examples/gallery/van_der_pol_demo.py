"""
Van der Pol oscillator — underdamped inference
===============================================

Infer the force field of a 1D Van der Pol oscillator from position-only
data.  Velocities are reconstructed internally by the ULI estimators.

This example demonstrates underdamped Langevin inference: the system
has inertia, but only positions are observed — as is typical for
experimental tracking data.

.. rubric:: Tags

synthetic · underdamped · nonlinear · 1D
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "nonlinear", "1D"]
# sphinx_gallery_thumbnail_number = 5

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

apply_style()
# sphinx_gallery_end_ignore
# %%
# System definition
# ------------------
#
# The Van der Pol oscillator:
#
# .. math::
#
#    \ddot{x} = \mu (1 - x^2)\,\dot{x} - x + \sqrt{2D}\,\xi(t)
#
# with :math:`\mu = 3` (strongly nonlinear) and :math:`D_0 = 0.05`.
# We use a time step :math:`\Delta t = 5 \times 10^{-3}` and a
# long trajectory to capture ~10 complete oscillation cycles.

from SFI.bases import named_scalar, unit_axes, v_components, x_components
from SFI.langevin import UnderdampedProcess
from SFI.utils import plotting

mu = 3.0
D0 = 0.05
dt = 0.02
Nsteps = 2000
oversampling = 8  # fine substeps → accurate Verlet integration of the SDE
seed_data = 0

# Ground-truth force written compositionally:
#     F(x, v) = mu * (1 - x²) * v - x
# Named scalar carries the reference value as a default, so the
# simulator binds the parameter automatically.
(_x,) = x_components(1)
(_v,) = v_components(1)
(_ex,) = unit_axes(1)
mu_p = named_scalar("mu", default=mu)
F_exact = (mu_p * (1 - _x * _x) * _v - _x) * _ex

key = random.PRNGKey(seed_data)
proc = UnderdampedProcess(F_exact, D=jnp.array([[D0]]))
proc.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))

key, sub = random.split(key)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=sub, prerun=200, oversampling=oversampling)

t, X, _ = coll.to_arrays(dataset=0)   # t:(T,), X:(T, N, d)
x = X[:, 0, 0]

# %%
# Phase portrait
# ---------------
#
# The Van der Pol limit cycle in the :math:`(x, \dot{x})` plane.
# Velocities are estimated via finite differences for visualisation;
# the inference engine uses more sophisticated ULI reconstructions.

# sphinx_gallery_start_ignore
v_fd = coll.velocity_array(dataset=0, scheme="central")[:, 0, 0]  # secant velocity; v is not stored

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
sc = ax.scatter(x, v_fd, c=t, s=0.4, cmap="viridis", alpha=0.7, rasterized=True)
ax.set_xlabel("x")
ax.set_ylabel("v (finite diff)")
ax.set_title("Phase portrait (time-coloured)")
plt.colorbar(sc, ax=ax, label="t", shrink=0.8)

ax = axes[1]
plotting.timeseries(coll, dims=[0], ax=ax, lw=0.6, color=SFI_COLORS["data"])
ax.set_title("Position time series")

fig.suptitle("Van der Pol oscillator — trajectory data", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Force inference
# ----------------
#
# Monomials in :math:`(x, v)` up to degree 3 capture the cubic
# nonlinearity.  :class:`~SFI.inference.UnderdampedLangevinInference` handles velocity
# reconstruction, diffusion estimation, and force projection.

from SFI.bases import monomials_up_to
from SFI import UnderdampedLangevinInference

degree = 3
B = monomials_up_to(order=degree, dim=1, include_v=True, rank='vector')

inf = UnderdampedLangevinInference(coll)
inf.compute_diffusion_constant()
# Default preset="auto": this trajectory is clean (Lambda_trace < 0), so the
# auto switch selects the sharper "clean"/WeakNoise estimator on its own.
inf.infer_force_linear(B)
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, data_exact=coll, maxpoints=2000)

inf.print_report()

# %%
# Force validation
# -----------------
#
# Scatter plot of exact vs inferred force, evaluated along the
# ULI-reconstructed trajectory.

F_true, F_pred = inf.force_comparison_arrays(model_exact=proc)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
plt.sca(ax)
plotting.comparison_scatter(F_true, F_pred, maxpoints=4000, alpha=0.15,
                            color=SFI_COLORS["data"])
ax.set_xlabel("F_true")
ax.set_ylabel("F_inferred")

ax = axes[1]
T2 = min(4000, len(F_true))
s = np.arange(T2)
ax.plot(s, F_true[:T2], lw=0.7, label="Exact", color=SFI_COLORS["exact"])
ax.plot(s, F_pred[:T2], lw=0.7, alpha=0.8, label="Inferred",
        color=SFI_COLORS["inferred"])
ax.set_xlabel("sample")
ax.set_ylabel("F")
ax.set_title("Force along trajectory")
ax.legend()

fig.suptitle("Underdamped force inference — validation", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Bootstrapped trajectory
# -------------------------
#
# Simulating from the inferred model and comparing the phase portrait
# provides a generative validation.

key_boot = random.PRNGKey(1)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(
    key_boot, oversampling=oversampling, simulate=True
)
_, X_boot, _ = coll_boot.to_arrays(dataset=0)
x_boot = X_boot[:, 0, 0]
v_fd_boot = coll_boot.velocity_array(dataset=0, scheme="central")[:, 0, 0]

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
tail = min(8000, x.shape[0], x_boot.shape[0])
ax.plot(x[-tail:], v_fd[-tail:], lw=0.3, alpha=0.6, label="Data",
        color=SFI_COLORS["data"])
ax.plot(x_boot[-tail:], v_fd_boot[-tail:], lw=0.3, alpha=0.5,
        label="Bootstrap", color=SFI_COLORS["bootstrap"])
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title("Phase portrait: data vs bootstrap")
ax.legend()

ax = axes[1]
plotting.timeseries(coll, dims=[0], ax=ax, lw=0.6, label="Data",
                    color=SFI_COLORS["data"])
plotting.timeseries(coll_boot, dims=[0], ax=ax, lw=0.6, alpha=0.8,
                    label="Bootstrap", color=SFI_COLORS["bootstrap"])
ax.set_title("Time series comparison")
ax.legend()

fig.suptitle("Generative validation — bootstrapped trajectory", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Diagnostics
# -----------
#
# The **underdamped** residual compares the symmetric finite-difference
# acceleration :math:`\hat a = (x_{t+1} - 2x_t + x_{t-1})/\Delta t^2`
# (the same kinematic the force estimator fits) to the inferred force
# :math:`\hat F(\hat x, \hat v)`, standardised by the noise scale of a
# second difference of integrated white noise,
# :math:`\tfrac23\,(2\hat D)/\Delta t`.  Adjacent accelerations overlap,
# so every second sample is used to keep the residuals independent.  A
# well-specified cubic-in-:math:`(x,v)` model then passes all checks.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Van der Pol — diagnostics (underdamped linear model)", y=1.02)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
tail = min(10000, x.shape[0])
ax_t.scatter(x[-tail:], v_fd[-tail:], c=t[-tail:], s=0.3,
             cmap="viridis", alpha=0.7, rasterized=True)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
