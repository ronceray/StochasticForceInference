r"""
Time-dependent forcing — protocols as extras
============================================

Infer a force law that depends on a **known time-dependent protocol**
— here a trap whose stiffness is switched between two values — from a
single 1D trajectory.

The protocol :math:`k(t)` enters SFI as a *time-dependent extra*: the
simulator consumes it per frame, attaches it to the output collection,
and the inference reads it back through the compositional symbol
:func:`~SFI.bases.extra_scalar`.  The same mechanism covers temperature
ramps, oscillating fields, or any drive recorded alongside the
trajectory.

.. rubric:: Tags

synthetic · overdamped · linear · 1D · time-dependent
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D", "time-dependent"]
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
from SFI.utils.plotting import timeseries

apply_style()
# sphinx_gallery_end_ignore
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
Nsteps = 20_000
period = 1_000  # frames per plateau

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
# When the drive is on, the trap is twice as stiff and the fluctuations
# shrink accordingly.

# sphinx_gallery_start_ignore
t, _, _ = coll.to_arrays()  # time axis for the protocol step panel
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3]})
axes[0].step(t, 1.0 + k_protocol, where="post", color=SFI_COLORS["exact"], lw=1.2)
axes[0].set_ylabel(r"$k_0 + k(t)$")
axes[0].set_yticks([1, 2])
timeseries(coll, dims=[0], ax=axes[1], lw=0.5, color=SFI_COLORS["data"])
axes[1].set_xlim(0, 40)
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
inf.compute_diffusion_constant()
inf.infer_force_linear(B)
inf.compute_force_error()
inf.print_report()

# %%
# Recovered stiffness law
# -----------------------
#
# The two fitted coefficients reconstruct the time-dependent stiffness
# :math:`\hat k(t) = -[\hat\theta_1 + \hat\theta_2\,k(t)]`, compared to
# the ground truth.

# sphinx_gallery_start_ignore
coeffs, cov = inf.coeff_block(B, field="force")
stderr = np.sqrt(np.diag(cov))
k_hat = -(coeffs[0] + coeffs[1] * k_protocol)
k_true = 1.0 + k_protocol

fig, ax = plt.subplots(figsize=(8, 3))
ax.step(t, k_true, where="post", color=SFI_COLORS["exact"], lw=1.5, label="true $k_0+k(t)$")
ax.step(t, k_hat, where="post", color=SFI_COLORS["inferred"], lw=1.5, ls="--",
        label="inferred")
ax.fill_between(t, k_hat - (stderr[0] + stderr[1] * k_protocol),
                k_hat + (stderr[0] + stderr[1] * k_protocol),
                color=SFI_COLORS["inferred"], alpha=0.2, step="post")
ax.set_xlim(0, 60)
ax.set_xlabel("Time")
ax.set_ylabel("trap stiffness")
ax.legend(loc="center right")
ax.set_title("Time-dependent stiffness: truth vs inference")
plt.show()
# sphinx_gallery_end_ignore
# %%
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

stamp_output()
