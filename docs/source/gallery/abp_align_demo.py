"""
Aligning active Brownian particles — generic pairs API
========================================================

Infer pairwise interaction forces in a system of aligning active
Brownian particles (ABPs) using **generic pair-interaction building
blocks** from :mod:`SFI.bases.pairs`.

First we simulate the ABP system using the pre-built ABP model, then
save and reload the trajectory to demonstrate the I/O round-trip, and
finally construct a generic pair-interaction basis for linear
inference.

This example demonstrates:

1. Simulating a multi-particle ABP system with an example-local
   ABP helper (thin wrapper around :mod:`SFI.bases.pairs`).
2. Saving and loading trajectory data (CSV round-trip).
3. Constructing a generic **Basis** from
   :func:`~SFI.bases.pairs.heading_vector`,
   :func:`~SFI.bases.pairs.radial_pair_basis`, and
   :func:`~SFI.bases.pairs.angular_pair_basis`.
4. **Linear** force inference using that basis.
5. Recovering the learned interaction kernels vs the true model.

.. rubric:: Tags

synthetic · overdamped · multi-particle · linear
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "linear"]
# sphinx_gallery_thumbnail_number = 7

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output, plot_kernel_ci
from SFI.utils.plotting import dark_fig, wrap_positions, phase2d, plot_particles

apply_style()

# Force black save background for both static figures and video frames.
# The gallery mplstyle sets savefig.transparent=True, which produces
# white padding around dark-themed animations (H264 has no alpha).
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False

# -- Plotting style constants (not shown in rendered gallery) --
# Dark-theme figure helpers and periodic-box wrapping come from
# SFI.utils.plotting (dark_fig / wrap_positions); static heading-quiver
# snapshots use plot_particles. Only the bespoke FuncAnimation movies below
# still build scatter/quiver inline, so we keep these two style constants.

_ARROW_LEN = 1.5   # heading-arrow length in data units (movie quivers)
_PT_SIZE = 160      # scatter marker area (points²)
# sphinx_gallery_end_ignore
# %%
# System: aligning active Brownian particles
# --------------------------------------------
#
# Each particle has position :math:`(x, y)` and heading
# :math:`\theta` — three degrees of freedom.  The force includes:
#
# - **Self-propulsion**: :math:`c_0\,\hat{e}_\theta`
# - **Pairwise repulsion**: short-range, radially repulsive:
#   :math:`\varepsilon\, e^{-r/R_0}\,\hat{r}_{ij}`
# - **Pairwise alignment**: angular torque:
#   :math:`A\,\sin(\Delta\theta)\,e^{-r/L_0}`
#
# Positions obey periodic boundary conditions.

N_particles = 60
Lx, Ly = 30.0, 30.0       # periodic box dimensions
Nsteps = 2000              # simulation length
dt_sim = 0.02              # integration time step
D_iso = 0.05               # isotropic diffusion coefficient
seed = 0
box = jnp.array([Lx, Ly])

# True model parameters
c0_true = 1.0       # self-propulsion speed
eps_true = 2.0       # repulsion strength
A_true = 0.5         # alignment strength
R0_true = 1.0        # repulsion length scale
L0_true = 2.0        # alignment length scale

theta_F_exact = dict(c0=c0_true, eps=eps_true, A=A_true,
                     R0=R0_true, L0=L0_true)

# %%
# Simulating the true model
# ----------------------------
#
# We build the ABP model from the helpers in
# ``_gallery_utils/abp.py`` (a thin composition of
# :mod:`SFI.bases.pairs` primitives).  The result is a parametric
# state function (PSF) with named parameters ``c0``, ``eps``, ``A``,
# ``R0``, ``L0``.

from _gallery_utils.abp import make_abp_align_psf
from SFI.langevin import OverdampedProcess

F_psf = make_abp_align_psf(dim=3)
proc = OverdampedProcess(F_psf, D=D_iso, extras_global={"box": box})
proc.set_params(theta_F=theta_F_exact)

# Random initial conditions: uniform in box, random headings
key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0 = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

proc.initialize(x0)
key, sub = random.split(key)
coll = proc.simulate(dt=dt_sim, Nsteps=Nsteps, key=sub)

print(f"Trajectory: {coll.T} frames, "
      f"{coll.N} particles, dim={coll.d}")

# %%
# Simulation movie
# ------------------
#
# Animation of the simulated ABP system.  Particles are coloured by
# heading angle :math:`\theta`; white arrows show the propulsion
# direction.

Xfull = np.asarray(coll.X)  # full trajectory array (T × N × 3)

# sphinx_gallery_start_ignore
T_total = coll.T
skip_mov = max(1, T_total // 200)
Xmov_data = Xfull[::skip_mov]
T_mov_data = Xmov_data.shape[0]

fig_dm, ax_dm = dark_fig(figsize=(4.5, 4.5))
Xf0 = wrap_positions(Xmov_data[0], (Lx, Ly))
sc_dm = ax_dm.scatter(Xf0[:, 0], Xf0[:, 1], s=_PT_SIZE,
                      c=Xf0[:, 2], cmap="hsv",
                      vmin=-np.pi, vmax=np.pi, zorder=3,
                      edgecolors="white", lw=0.5)
qv_dm = ax_dm.quiver(Xf0[:, 0], Xf0[:, 1],
                     _ARROW_LEN * np.cos(Xf0[:, 2]),
                     _ARROW_LEN * np.sin(Xf0[:, 2]),
                     scale=1.0, scale_units="xy", width=0.008,
                     color="white", alpha=0.85, zorder=4)
ax_dm.set_xlim(0, Lx); ax_dm.set_ylim(0, Ly)
ax_dm.set_aspect("equal")
ax_dm.set_xlabel("x"); ax_dm.set_ylabel("y")
ttl_dm = ax_dm.set_title("t = 0.0")


def _update_data(frame):
    Xf = wrap_positions(Xmov_data[frame], (Lx, Ly))
    sc_dm.set_offsets(Xf[:, :2])
    sc_dm.set_array(Xf[:, 2])
    qv_dm.set_offsets(Xf[:, :2])
    qv_dm.set_UVC(_ARROW_LEN * np.cos(Xf[:, 2]),
                   _ARROW_LEN * np.sin(Xf[:, 2]))
    ttl_dm.set_text(f"t = {frame * skip_mov * dt_sim:.1f}")
    return sc_dm, qv_dm, ttl_dm


anim_data = FuncAnimation(fig_dm, _update_data,
                          frames=T_mov_data, interval=40, blit=True)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Saving and loading trajectory data
# -------------------------------------
#
# SFI trajectory collections can be serialised to CSV, Parquet, or
# HDF5 via :meth:`~SFI.trajectory.TrajectoryCollection.save`.
# Global extras — here the periodic box — are embedded in a YAML
# header and survive the round-trip.

import os, tempfile
from SFI.trajectory import TrajectoryCollection

csv_path = os.path.join(tempfile.gettempdir(), "sfi_abp_demo.csv")
coll.save(csv_path, format="csv")

with open(csv_path) as f:
    for line in f.readlines()[:10]:
        print(line.rstrip())
print("  ...")

# Reload into a fresh collection and verify
coll_reloaded = TrajectoryCollection.load(csv_path)
print(f"\nOriginal:  T={coll.T}, N={coll.N}, d={coll.d}  →  Loaded: T={coll_reloaded.T}, N={coll_reloaded.N}, d={coll_reloaded.d}")
max_err = float(np.max(np.abs(coll_reloaded.X - coll.X)))
print(f"Max |ΔX| = {max_err:.1e}  (numerical round-trip error)")

# %%
# Particle snapshot
# -------------------
#
# Final-frame snapshot with heading arrows (left) and individual
# particle trajectories (right).  Positions are wrapped into the
# periodic box; particles are coloured by heading angle.

n_show = min(5, N_particles)  # trajectories to display

# sphinx_gallery_start_ignore
fig_snap, axes_snap = dark_fig(1, 2, figsize=(12, 5.5))

plot_particles(coll, t_index=-1, color_dim=2, quiver=True, heading_dim=2,
               box=(Lx, Ly), vmin=-np.pi, vmax=np.pi, s=_PT_SIZE,
               ax=axes_snap[0])
axes_snap[0].set_title(f"Snapshot (t = {Nsteps * dt_sim:.0f})")

phase2d(coll, dims=(0, 1), particles=range(n_show), box=(Lx, Ly),
        ax=axes_snap[1])
axes_snap[1].set_title(f"Trajectories ({n_show} particles)")

fig_snap.suptitle(f"Aligning ABPs  (N = {N_particles})",
                  fontsize=13, color="white")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Building the pair-interaction basis
# ----------------------------------------
#
# For **inference**, we construct a generic basis from building blocks
# in :mod:`SFI.bases.pairs`.  The basis is deliberately over-complete
# — it contains many more radial kernels than the true model uses —
# so the linear solver must pick out the right combination.
#
# 1. **Self-propulsion** —
#    :func:`~SFI.bases.pairs.heading_vector` gives the unit heading
#    :math:`(\cos\theta, \sin\theta, 0)`.
# 2. **Radial repulsion** —
#    :func:`~SFI.bases.pairs.radial_pair_basis` with exponential-
#    polynomial kernels.
# 3. **Angular alignment** —
#    :func:`~SFI.bases.pairs.angular_pair_basis` with a
#    :math:`\sin(\Delta\theta)` coupling.

from SFI.bases.pairs import (
    angular_pair_basis,
    exp_poly_kernels,
    heading_vector,
    radial_pair_basis,
)
from SFI.statefunc import Basis

dim = 3  # (x, y, θ) per particle

# Radial kernels: r^n * exp(-r/L) for n ∈ {0,1}, L ∈ {0.5, 1, 2, 4}
repel_kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
align_kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])

# Self-propulsion: (cos θ, sin θ, 0)
B_heading = heading_vector(dim=dim, angle_index=2)

# Radial pair repulsion → dispatched Basis (acts in xy-plane)
inter_repel = radial_pair_basis(
    repel_kernels, dim=dim,
    box="extras",
    spatial_dims=slice(0, 2),          # xy positions for distance
    embed_dim=dim, embed_axes=[0, 1],  # embed radial force into xy
)
B_repel = inter_repel.dispatch_pairs(
    symmetric=True, exclude_self=True,
    owners="focal", reducer="sum", return_as="basis",
)

# Angular alignment → dispatched Basis (acts on θ)
inter_align = angular_pair_basis(
    align_kernels, jnp.sin, dim=dim,
    angle_index=2, output_index=2,     # coupling: sin(Δθ) → force on θ
    box="extras",
    spatial_dims=slice(0, 2),
)
B_align = inter_align.dispatch_pairs(
    symmetric=True, exclude_self=True,
    owners="focal", reducer="sum", return_as="basis",
)

# Stack into one combined basis for inference
B_full = Basis.stack([B_heading, B_repel, B_align])

n_heading = B_heading.n_features
n_repel = B_repel.n_features
n_align = B_align.n_features


# %%
# Linear force inference
# -------------------------
#
# Standard :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear` recovers the interaction
# coefficients in a single linear solve — no gradient-based
# optimisation needed.

from SFI import OverdampedLangevinInference
from SFI.utils import plotting

# Create inference object from the trajectory collection
inf = OverdampedLangevinInference(coll)

# Estimate diffusion coefficient
inf.compute_diffusion_constant(method="WeakNoise")

# Solve for force coefficients (one-shot linear solve)
inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")

# Estimate coefficient uncertainty (populates force_coefficients_covariance)
inf.compute_force_error()

# Compare to exact model for quantitative error metrics
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()

nmse_abp = float(inf.NMSE_force)  # reused in the force-scatter title

# %%
# Force scatter: exact vs inferred
# -----------------------------------
#
# All force components (translational and angular) along the
# trajectory, plotted as a scatter.  Points near the diagonal
# indicate good inference.

# The force is a multi-particle interacting field (per-frame neighbour
# structure + the ``box`` extra), so evaluate it per frame on ``coll.X``
# rather than via force_comparison_arrays (which flattens to single points).
Xe = proc.force_sf(coll.X, extras={"box": box})
Xi = inf.force_inferred(coll.X, extras={"box": box})

# sphinx_gallery_start_ignore
fig_sc, ax_sc = plt.subplots(figsize=(4.5, 4.5))
plotting.comparison_scatter(Xe, Xi, maxpoints=5000, alpha=0.05)
ax_sc.set_xlabel("Exact F")
ax_sc.set_ylabel("Inferred F")
ax_sc.set_title(f"Force scatter  (NMSE = {nmse_abp:.3f})")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Recovered interaction kernels
# --------------------------------
#
# The inferred coefficients weight the radial kernels.  Summing
# :math:`\sum_k c_k \phi_k(r)` reconstructs the effective radial
# interaction shape.  Shaded bands show 95 % confidence intervals
# derived from the coefficient covariance
# (see :func:`~SFI.inference.kernel_predict_ci`).

from SFI.inference import kernel_predict_ci

# Extract inferred coefficients and covariance sub-blocks for each basis block
c_heading, _ = inf.coeff_block(B_heading, field='force')
c_repel, cov_repel = inf.coeff_block(B_repel, field='force')
c_align, cov_align = inf.coeff_block(B_align, field='force')

# True coefficient vectors in the generic-basis coordinate system
# Kernel ordering from exp_poly_kernels(degrees=[0,1], lengths=[0.5,1,2,4]):
#   k=0: r⁰·e⁻ʳ/0.5   k=1: r⁰·e⁻ʳ/1   k=2: r⁰·e⁻ʳ/2   k=3: r⁰·e⁻ʳ/4
#   k=4: r¹·e⁻ʳ/0.5   k=5: r¹·e⁻ʳ/1   k=6: r¹·e⁻ʳ/2   k=7: r¹·e⁻ʳ/4
idx_repel = 1  # r⁰·e⁻ʳ/R₀ with R₀=1.0
idx_align = 2  # sin(Δθ)·r⁰·e⁻ʳ/L₀ with L₀=2.0

true_c_repel = np.zeros(n_repel)
true_c_repel[idx_repel] = -eps_true
true_c_align = np.zeros(n_align)
true_c_align[idx_align] = A_true

# Compute kernel profiles with confidence intervals
r_eval = np.linspace(0.01, 8.0, 200)
ci_repel = kernel_predict_ci(r_eval, repel_kernels, c_repel, cov_repel)
ci_align = kernel_predict_ci(r_eval, align_kernels, c_align, cov_align)

# True kernel profiles (for comparison)
true_repel = true_c_repel @ ci_repel["phi"]
true_align = true_c_align @ ci_align["phi"]

# sphinx_gallery_start_ignore
fig_kern, axes_kern = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (ci, true_k, ylabel, title) in zip(
    axes_kern,
    [
        (ci_repel, true_repel,
         r"$\phi_\mathrm{repel}(r)\;/\;\max|\phi_\mathrm{true}|$",
         "Radial repulsion kernel"),
        (ci_align, true_align,
         r"$\phi_\mathrm{align}(r)\;/\;\max|\phi_\mathrm{true}|$",
         "Angular alignment kernel"),
    ],
):
    plot_kernel_ci(ax, r_eval, true_k, ci, ci_label="95% CI")
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

axes_kern[1].set_ylim(-1, 5)
fig_kern.suptitle("Interaction kernel recovery", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Inferred self-propulsion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The heading coefficient recovers the active thrust magnitude
# :math:`c_0`:

# %%
# Bootstrapped trajectory
# -------------------------
#
# Simulating from the inferred model and comparing emergent collective
# behaviour is a powerful qualitative validation for multi-particle
# systems.

# Run a bootstrap simulation from the inferred coefficients
key_boot = random.PRNGKey(seed + 77)
proc_boot = inf.simulate_bootstrapped_trajectory(key_boot, simulate=False)
proc_boot.set_extras(extras_global={"box": box})
proc_boot.initialize(x0)                    # same initial conditions
key_boot, sub_boot = random.split(key_boot)
coll_boot = proc_boot.simulate(dt=dt_sim, Nsteps=Nsteps, key=sub_boot)

# sphinx_gallery_start_ignore
fig_bs_snap, axes_bs = dark_fig(1, 2, figsize=(12, 5.5))

plot_particles(coll, t_index=-1, color_dim=2, quiver=True, heading_dim=2,
               box=(Lx, Ly), vmin=-np.pi, vmax=np.pi, s=_PT_SIZE,
               ax=axes_bs[0])
axes_bs[0].set_title("Original (final frame)")

plot_particles(coll_boot, t_index=-1, color_dim=2, quiver=True, heading_dim=2,
               box=(Lx, Ly), vmin=-np.pi, vmax=np.pi, s=_PT_SIZE,
               ax=axes_bs[1])
axes_bs[1].set_title("Bootstrapped (inferred model)")

fig_bs_snap.suptitle("Bootstrap validation", fontsize=13, color="white")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Bootstrap comparison movie
# ----------------------------
#
# Side-by-side animation of the original trajectory (left) and
# a trajectory simulated from the *inferred* model (right).
# Both start from the same initial conditions.

Xboot_full = np.asarray(coll_boot.X)

# sphinx_gallery_start_ignore
skip = max(1, T_total // 200)
Xmov = Xfull[::skip]
Xboot_mov = Xboot_full[::skip]
T_mov = min(Xmov.shape[0], Xboot_mov.shape[0])

fig_bs, (ax_orig, ax_boot) = dark_fig(1, 2, figsize=(12, 6))

for _ax, _title in [(ax_orig, "Original"), (ax_boot, "Inferred model")]:
    _ax.set_xlim(0, Lx); _ax.set_ylim(0, Ly)
    _ax.set_aspect("equal")
    _ax.set_xlabel("x"); _ax.set_ylabel("y")
    _ax.set_title(_title)

Xf0_o = wrap_positions(Xmov[0], (Lx, Ly))
sc_o = ax_orig.scatter(Xf0_o[:, 0], Xf0_o[:, 1], s=_PT_SIZE,
                       c=Xf0_o[:, 2], cmap="hsv",
                       vmin=-np.pi, vmax=np.pi, zorder=3,
                       edgecolors="white", lw=0.5)
qv_o = ax_orig.quiver(Xf0_o[:, 0], Xf0_o[:, 1],
                       _ARROW_LEN * np.cos(Xf0_o[:, 2]),
                       _ARROW_LEN * np.sin(Xf0_o[:, 2]),
                       scale=1.0, scale_units="xy", width=0.008,
                       color="white", alpha=0.85, zorder=4)

Xf0_b = wrap_positions(Xboot_mov[0], (Lx, Ly))
sc_b = ax_boot.scatter(Xf0_b[:, 0], Xf0_b[:, 1], s=_PT_SIZE,
                       c=Xf0_b[:, 2], cmap="hsv",
                       vmin=-np.pi, vmax=np.pi, zorder=3,
                       edgecolors="white", lw=0.5)
qv_b = ax_boot.quiver(Xf0_b[:, 0], Xf0_b[:, 1],
                       _ARROW_LEN * np.cos(Xf0_b[:, 2]),
                       _ARROW_LEN * np.sin(Xf0_b[:, 2]),
                       scale=1.0, scale_units="xy", width=0.008,
                       color="white", alpha=0.85, zorder=4)

suptitle_bs = fig_bs.suptitle("Bootstrap comparison  t = 0.0",
                              fontsize=13, color="white")


def _update_bootstrap(frame):
    Xfo = wrap_positions(Xmov[frame], (Lx, Ly))
    sc_o.set_offsets(Xfo[:, :2])
    sc_o.set_array(Xfo[:, 2])
    qv_o.set_offsets(Xfo[:, :2])
    qv_o.set_UVC(_ARROW_LEN * np.cos(Xfo[:, 2]),
                  _ARROW_LEN * np.sin(Xfo[:, 2]))
    Xfb = wrap_positions(Xboot_mov[frame], (Lx, Ly))
    sc_b.set_offsets(Xfb[:, :2])
    sc_b.set_array(Xfb[:, 2])
    qv_b.set_offsets(Xfb[:, :2])
    qv_b.set_UVC(_ARROW_LEN * np.cos(Xfb[:, 2]),
                  _ARROW_LEN * np.sin(Xfb[:, 2]))
    suptitle_bs.set_text(
        f"Bootstrap comparison  t = {frame * skip * dt_sim:.1f}"
    )
    return sc_o, qv_o, sc_b, qv_b, suptitle_bs


anim_bootstrap = FuncAnimation(
    fig_bs, _update_bootstrap, frames=T_mov, interval=40, blit=True,
)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Both simulations start from the same initial conditions.
# Visual similarity confirms the inferred model captures the
# collective dynamics.

# %%
# Diagnostics
# -----------
#
# :func:`~SFI.diagnostics.assess` computes standardised Euler
# residuals pooled across **all particles and spatial components**
# (:math:`N_\mathrm{obs} \approx N_\mathrm{particles} \times T \times d`),
# giving high statistical power even from a short trajectory.
# All three force channels — translational :math:`(x,y)` and angular
# :math:`(\theta)` — are pooled into a single residual vector.
#
# The linear basis is over-complete by design, so the NMSE should
# be small but the coefficient z-scores for unused kernels will
# be near zero.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Aligning ABPs — diagnostics (linear model)", y=1.02)
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
