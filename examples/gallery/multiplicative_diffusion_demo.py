"""
Multiplicative noise — the Landauer blowtorch
==============================================

Recover a **state-dependent diffusion field** :math:`D(x)` together with
the force from a single overdamped trajectory.  The system is the
classic *Landauer blowtorch*: a bistable particle whose bath is hotter
on one side.  The temperature gradient redistributes the well
occupancies away from the naive Boltzmann weights — an effect that only
a joint inference of :math:`F(x)` *and* :math:`D(x)` can capture.

.. rubric:: Tags

synthetic · overdamped · multiplicative-noise · 1D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "multiplicative-noise", "1D"]
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
from SFI.utils.plotting import plot_profile_1d, timeseries_colored

apply_style()
# sphinx_gallery_end_ignore
# %%
# Model and simulation
# ---------------------
#
# A double-well particle in a bath with a linear temperature profile:
#
# .. math::
#
#    dx = (x - x^3)\,dt + \sqrt{2 D(x)}\,dW,
#    \qquad D(x) = D_0\,(1 + g\,x)
#
# interpreted in the **Itô convention** — the drift entering the SDE
# (and the force SFI infers) is the Itô drift.  With :math:`g > 0` the
# right well sits in the *hot* region.  Both fields are built
# compositionally from :mod:`SFI.bases` primitives and passed straight
# to the simulator.

import SFI
from SFI.bases import identity_matrix_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

D0 = 0.5      # baseline diffusion
g = 0.3       # temperature gradient
dt = 0.01
Nsteps = 50_000

(x,) = x_components(1)
(ex,) = unit_axes(1)
I = identity_matrix_basis(1)

F_model = (x - x**3) * ex                # F(x) = x - x³  (Itô drift)
D_model = D0 * (1.0 + g * x) * I         # D(x) = D₀(1 + g·x)

proc = OverdampedProcess(F=F_model, D=D_model,
                         theta_F=jnp.array([1.0]), theta_D=jnp.array([1.0]))
proc.initialize(jnp.array([0.5]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(7),
                     prerun=200, oversampling=10)
print(f"Trajectory: {coll.T} frames, dt={dt}")

# %%
# Trajectory
# -----------
#
# The particle hops between the two wells; excursions in the hot
# (right) well are visibly more agitated than in the cold one.

# sphinx_gallery_start_ignore
t, X, _mask = coll.to_arrays(dataset=0)
Xarr = X[:, 0, 0]
fig, ax = plt.subplots(figsize=(8, 3))
timeseries_colored(coll, color_fn=lambda x: D0 * (1 + g * x[:, 0]),
                   cmap="plasma", colorbar_label="local $D(x)$", s=1.5, ax=ax)
ax.axhline(1, ls=":", lw=0.5, color="#808080")
ax.axhline(-1, ls=":", lw=0.5, color="#808080")
ax.set_xlabel("Time")
ax.set_ylabel("x")
ax.set_title("Blowtorch trajectory — colored by local diffusivity")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Inference: force, then diffusion field
# ----------------------------------------
#
# The linear-estimator sequence, with one extra step: after the usual
# constant-:math:`D` estimate and force regression, we fit the full
# diffusion *field* with :meth:`infer_diffusion_linear` on a rank-2
# polynomial basis (``rank='symmetric_matrix'``).  The quadratic basis
# is deliberately larger than the truth — the linear gradient must be
# *discovered*, not assumed.

from SFI.bases import monomials_up_to

B_force = monomials_up_to(order=3, dim=1, rank="vector")
B_diff = monomials_up_to(order=2, dim=1, rank="symmetric_matrix")

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()              # constant-D baseline
inf.infer_force_linear(B_force)               # Itô drift regression
inf.compute_force_error()
inf.infer_diffusion_linear(B_diff)            # state-dependent D(x)

inf.compare_to_exact(model_exact=proc)
inf.print_report()

# %%
# Recovered force and diffusion fields
# --------------------------------------
#
# Both fields are recovered from the same trajectory.  The constant-D
# baseline (what a standard analysis would report) misses the gradient
# entirely.

x_grid = np.linspace(-1.9, 1.9, 200)
x_jnp = jnp.array(x_grid[:, None])
F_exact = x_grid - x_grid**3
D_exact = D0 * (1 + g * x_grid)
F_inf = np.asarray(inf.force_inferred(x_jnp))[:, 0]
D_inf = np.asarray(inf.diffusion_inferred(x_jnp))[:, 0, 0]
D_const = float(np.asarray(inf.diffusion_average).ravel()[0])

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
plot_profile_1d(coll, inf.force_inferred, exact_field=proc.force_sf, ax=axes[0])
axes[0].set_ylabel("F(x)")
axes[0].set_title("Force (Itô drift)")

plot_profile_1d(coll, inf.diffusion_inferred, exact_field=proc.diffusion_sf,
                component=0, ax=axes[1],
                label_exact="Exact $D(x)$", label_inferred="Inferred $D(x)$")
axes[1].axhline(D_const, ls="-.", lw=1.2, color=SFI_COLORS["highlight"],
                label="Constant-$D$ estimate")
axes[1].set_ylabel("D(x)")
axes[1].set_title("Diffusion field")
axes[1].legend()
plt.show()
# sphinx_gallery_end_ignore
# %%
# The blowtorch effect: occupancy statistics
# --------------------------------------------
#
# For Itô dynamics the zero-current stationary density is
#
# .. math::
#
#    p(x) \;\propto\; \frac{1}{D(x)}
#    \exp\!\Big(\int^x \frac{F(y)}{D(y)}\,dy\Big),
#
# *not* the Boltzmann weight :math:`e^{-U(x)/\bar D}` built from the
# potential and a constant diffusion estimate.  The hot well is
# depopulated.  Using the **inferred** :math:`\hat F` and
# :math:`\hat D(x)` in the formula above reproduces the empirical
# histogram; the constant-D Boltzmann prediction does not.

from scipy.integrate import cumulative_trapezoid


def stationary_density(F_vals, D_vals, x_vals):
    """Zero-current Itô stationary density on a grid (unnormalized)."""
    phi = cumulative_trapezoid(F_vals / D_vals, x_vals, initial=0.0)
    p = np.exp(phi - phi.max()) / D_vals
    return p / np.trapezoid(p, x_vals)


p_exact = stationary_density(F_exact, D_exact, x_grid)
p_inferred = stationary_density(F_inf, D_inf, x_grid)
p_naive = stationary_density(F_inf, np.full_like(x_grid, D_const), x_grid)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(Xarr, bins=80, density=True, alpha=0.45, color=SFI_COLORS["data"],
        label="Empirical histogram")
ax.plot(x_grid, p_inferred, lw=2, color=SFI_COLORS["inferred"],
        label=r"Inferred $\hat F,\hat D(x)$")
ax.plot(x_grid, p_naive, "-.", lw=1.6, color=SFI_COLORS["highlight"],
        label=r"Naive constant-$D$ Boltzmann")
ax.plot(x_grid, p_exact, ":", lw=1.6, color=SFI_COLORS["exact"], label="Exact")
ax.set_xlabel("x")
ax.set_ylabel("p(x)")
ax.set_title("Blowtorch occupancy shift")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore
# %%
# A note on spurious drifts
# ---------------------------
#
# With multiplicative noise the words "force" and "drift" need care.
# SFI infers the **Itô drift** :math:`F`: the conditional mean
# displacement per unit time, :math:`F(x) = \lim_{dt\to 0}
# \langle \Delta x \rangle / dt`.  Other stochastic conventions absorb
# part of the noise gradient into the drift — in 1D,
#
# .. math::
#
#    F_{\rm Strato} = F_{\rm Itô} - \tfrac12 D'(x), \qquad
#    F_{\rm anti\text{-}Itô} = F_{\rm Itô} - D'(x),
#
# and these *spurious drift* terms have real dynamical consequences:
# even with zero force, Itô dynamics accumulates particles in low-noise
# regions (:math:`p \propto 1/D`), so a bare drift measurement can
# suggest a force toward the cold side where none exists.  Whether the
# physical system at hand obeys Itô, Stratonovich or isothermal
# (anti-Itô) statistics is an experimental question — but once
# :math:`F_{\rm Itô}` and :math:`D(x)` are both inferred, converting
# between conventions is just the dictionary above.
# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
timeseries_colored(coll, color_fn=lambda x: D0 * (1 + g * x[:, 0]),
                   cmap="plasma", plot_colorbar=False, s=2.0, ax=ax_t)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
