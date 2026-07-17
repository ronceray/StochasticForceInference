"""
2D limit cycle — nonlinear overdamped inference
=================================================

Infer the force field of a 2D nonlinear system with a stable limit
cycle.  The radial force :math:`\\dot{r} = r(1 - r^2)` drives the
system to a unit circle, while an angular drift
:math:`\\dot{\\theta} = \\omega` generates rotation — a non-equilibrium
steady state with non-zero probability currents.

.. rubric:: Tags

synthetic · overdamped · nonlinear · non-equilibrium · 2D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "non-equilibrium", "2D"]
# sphinx_gallery_thumbnail_number = 5

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
# %%
# System definition
# ------------------
#
# In Cartesian coordinates the force is:
#
# .. math::
#
#    F_x = x(1 - x^2 - y^2) - \omega\, y, \qquad
#    F_y = y(1 - x^2 - y^2) + \omega\, x
#
# with angular frequency :math:`\omega = 2` and isotropic noise
# :math:`D = 0.1`.

from SFI.bases import named_scalars, ones_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

# Ground-truth force written compositionally:
#     F = (1 - r^2) (x e_x + y e_y) + omega (x e_y - y e_x)
# with r^2 = x^2 + y^2.  ``named_scalars`` holds ``omega`` with a
# default value so the simulator can bind parameters automatically.
x, y = x_components(2)
ex, ey = unit_axes(2)
one = ones_basis(2)
(omega_p,) = named_scalars(omega=2.0)

r2 = x * x + y * y
F_true = (one - r2) * (x * ex + y * ey) + omega_p * (x * ey - y * ex)

D0 = 0.1
dt = 0.05
Nsteps = 4_000
seed = 0

proc = OverdampedProcess(F_true, D=jnp.eye(2) * D0)
proc.initialize(jnp.array([0.5, 0.0]))

key = random.PRNGKey(seed)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=key, prerun=5000, oversampling=20)

# %%
# Trajectory and inferred force field
# --------------------------------------
#
# Build a polynomial basis, infer, sparsify with PASTIS, then overlay
# the inferred force field on the trajectory.

from SFI.bases import monomials_up_to
from SFI import OverdampedLangevinInference
from SFI.utils import plotting

degree = 3
B = monomials_up_to(order=degree, dim=2, include_constant=True, rank='vector')

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()
inf.infer_force_linear(B, M_mode="Ito")
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS")
inf.compare_to_exact(model_exact=proc, maxpoints=2000)

inf.print_report()

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]
plt.sca(ax)
plotting.phase2d(coll, dims=(0, 1), ax=ax, cmap="viridis", linewidth=0.5, alpha=0.5)
plotting.plot_field(
    coll, inf.force_inferred, N=12,
    color=SFI_COLORS["inferred"], autoscale=True, scale=0.2,
)
plotting.plot_field(
    coll, proc.force_sf, N=12,
    color=SFI_COLORS["exact"], autoscale=True, scale=0.2, alpha=0.4,
)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Trajectory + force field (gold=inferred, red=exact)")

ax = axes[1]
plt.sca(ax)
inf.comparison_scatter(
    model_exact=proc, field='force', ax=ax,
    maxpoints=3000, alpha=0.1,
)
ax.set_xlabel("Exact F")
ax.set_ylabel("Inferred F")

fig.suptitle("Limit cycle — trajectory and force field", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Force error norm
# -----------------
#
# A 2D map of the force-reconstruction error
# :math:`\|F_{\mathrm{exact}} - F_{\mathrm{inferred}}\|` evaluated
# on a regular grid covering the data domain.

# sphinx_gallery_start_ignore
theta_ring = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(figsize=(6, 5))
plotting.plot_field_error(coll, inf.force_inferred, proc.force_sf, N=60, ax=ax)
# overlay unit circle (the limit cycle at r = 1)
ax.plot(np.cos(theta_ring), np.sin(theta_ring), "--", color="#B0B0B0", lw=1, alpha=0.6)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Force error norm (2D map)")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Bootstrapped trajectory
# -------------------------
#
# A trajectory simulated from the inferred model should reproduce the
# limit-cycle structure and rotational dynamics.

key_boot = random.PRNGKey(seed + 123)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(key_boot, oversampling=20)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
plotting.phase2d(coll, dims=(0, 1), ax=ax, cmap="viridis", linewidth=0.5)
ax.set_title("Original data")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

ax = axes[1]
plotting.phase2d(coll_boot, dims=(0, 1), ax=ax, cmap="magma", linewidth=0.5)
ax.set_title("Bootstrapped (inferred model)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

fig.suptitle("Generative validation — limit cycle", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Diagnostics
# -----------
#
# :func:`~SFI.diagnostics.assess` recomputes standardised residuals and
# runs a panel of consistency tests; a well-specified fit yields
# residuals consistent with i.i.d. :math:`\mathcal{N}(0, 1)` — no
# autocorrelation, no excess kurtosis, and a realised NMSE consistent
# with the predicted value.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Limit cycle — diagnostics", y=1.02)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Thumbnail
# ---------
#
# Compact force-error map for the gallery grid.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
plotting.plot_field_error(coll, inf.force_inferred, proc.force_sf, N=60, ax=ax_t)
ax_t.plot(np.cos(theta_ring), np.sin(theta_ring), "--", color="#B0B0B0", lw=1, alpha=0.5)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
