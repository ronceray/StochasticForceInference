# TODO: review this file
"""
Aligning active Brownian particles — manual example
==============================================
Infer pairwise interaction forces in a system of aligning active
Brownian particles (ABPs) using **generic pair-interaction building
blocks** from ``SFI.bases.pairs``.

This demo shows how to:

1. Build a simulation model from modular pair-interaction primitives
2. Build an over-complete inference basis from the same primitives
3. Recover the interaction kernels via linear force inference
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation

plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False

_OUTDIR = Path(__file__).resolve().parent / "abp_align_output"
_OUTDIR.mkdir(exist_ok=True)

# -- Plotting helpers --

_ARROW_LEN = 1.5   # heading-arrow length in data units
_PT_SIZE = 160      # scatter marker area (points²)

COL_EXACT = "#FF7A1A"     # orange
COL_INFERRED = "#FFC20A"  # gold


def _wrap_xy(X, Lx, Ly):
    """Wrap xy into [0, Lx) × [0, Ly); leave θ untouched."""
    X = np.array(X, copy=True)
    X[:, 0] = X[:, 0] % Lx
    X[:, 1] = X[:, 1] % Ly
    return X


def _wrap_trajectory(xy, box):
    """Wrap a (T, 2) trajectory and insert NaN at boundary jumps."""
    xy_w = xy.copy()
    xy_w[:, 0] = xy_w[:, 0] % box[0]
    xy_w[:, 1] = xy_w[:, 1] % box[1]
    jumps = np.any(np.abs(np.diff(xy_w, axis=0)) > 0.5 * box, axis=1)
    idx = np.flatnonzero(jumps)
    if idx.size:
        xy_w = np.insert(xy_w.astype(float), idx + 1, np.nan, axis=0)
    return xy_w


def _dark_ax(ax):
    """Style an axes for dark background."""
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("0.3")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _dark_fig(nrows=1, ncols=1, **kw):
    """Create a figure + axes with black background."""
    fig, axes = plt.subplots(nrows, ncols, **kw)
    fig.patch.set_facecolor("black")
    for ax in (np.atleast_1d(axes).flat if hasattr(axes, "flat") else [axes]):
        _dark_ax(ax)
    return fig, axes


def _draw_particles(ax, X, Lx, Ly):
    """Scatter + quiver for a single frame of wrapped ABPs."""
    Xw = _wrap_xy(X, Lx, Ly)
    ax.scatter(Xw[:, 0], Xw[:, 1], s=_PT_SIZE,
               c=Xw[:, 2], cmap="hsv",
               vmin=-np.pi, vmax=np.pi, zorder=3,
               edgecolors="white", lw=0.5)
    ax.quiver(Xw[:, 0], Xw[:, 1],
              _ARROW_LEN * np.cos(Xw[:, 2]),
              _ARROW_LEN * np.sin(Xw[:, 2]),
              scale=1.0, scale_units="xy", width=0.008,
              color="white", alpha=0.85, zorder=4)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
# %%
# System: aligning active Brownian particles
# --------------------------------------------
#
# Each particle has position (x, y) and heading θ — three degrees
# of freedom.  The force includes:
#
# - **Self-propulsion**: c₀ ê_θ
# - **Pairwise repulsion**: ε exp(-r/R₀) r̂_ij  (short-range, radial)
# - **Pairwise alignment**: A sin(Δθ) exp(-r/L₀)  (angular torque)
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

# %%
# Building the true model from pair-interaction building blocks
# ---------------------------------------------------------------
#
# We construct the exact simulation model from modular building
# blocks.  Each pair interaction is a parametric interactor
# (PSF) with its own prefactor and characteristic length as
# named parameters.  The same ``heading_vector`` basis will be
# reused later for inference.
#
# 1. **Self-propulsion** — ``heading_vector`` gives (cos θ, sin θ, 0),
#    promoted to a linear PSF with coefficient ``c0``.
# 2. **Radial repulsion** — a ``make_interactor`` PSF with params
#    ``eps`` (prefactor) and ``R0`` (range).
# 3. **Angular alignment** — a ``make_interactor`` PSF with params
#    ``A`` (prefactor) and ``L0`` (length scale).

from SFI.bases.pairs import (
    angular_pair_basis,
    exp_poly_kernels,
    heading_vector,
    pbc_displacement,
    radial_pair_basis,
    wrap_angle,
)
from SFI.statefunc import Basis, make_interactor
from SFI.langevin import OverdampedProcess

dim = 3  # (x, y, θ) per particle

# Self-propulsion: (cos θ, sin θ, 0) — reused for inference later
B_heading = heading_vector(dim=dim, angle_index=2)
F_active = B_heading.to_psf(coeff_key="c0")

# -- Radial repulsion interactor (parametric) --
def _repel_local(Xk, *, params, extras):
    """ε exp(-r/R₀) r̂_ij  embedded into (x, y, 0)."""
    xi, xj = Xk[0], Xk[1]
    dxy = pbc_displacement(xj[:2], xi[:2], extras["box"][:2])
    r = jnp.sqrt(jnp.sum(dxy ** 2) + 1e-12)
    rhat = dxy / r
    phi = -params["eps"] * jnp.exp(-r / params["R0"])
    return jnp.array([phi * rhat[0], phi * rhat[1], 0.0])

repel_interactor = make_interactor(
    _repel_local, dim=dim, rank=1, K=2, n_features=1,
    params={"eps": (), "R0": ()},
    extras_keys=("box",),
    labels=("repel_xy",),
)

# -- Angular alignment interactor (parametric) --
def _align_local(Xk, *, params, extras):
    """A sin(Δθ) exp(-r/L₀)  acting on θ only → (0, 0, torque)."""
    xi, xj = Xk[0], Xk[1]
    dxy = pbc_displacement(xj[:2], xi[:2], extras["box"][:2])
    r = jnp.sqrt(jnp.sum(dxy ** 2) + 1e-12)
    dtheta = wrap_angle(xj[2] - xi[2])
    tau = params["A"] * jnp.sin(dtheta) * jnp.exp(-r / params["L0"])
    return jnp.array([0.0, 0.0, tau])

align_interactor = make_interactor(
    _align_local, dim=dim, rank=1, K=2, n_features=1,
    params={"A": (), "L0": ()},
    extras_keys=("box",),
    labels=("align_z",),
)

# Concatenate pair interactors and dispatch over all pairs → PSF
F_pairs = (repel_interactor + align_interactor).dispatch_pairs(
    symmetric=True, exclude_self=True,
    owners="focal", reducer="sum", return_as="psf",
)

# Combine single-particle + pair-interaction PSFs
F_sim = F_active + F_pairs

theta_F_exact = dict(c0=c0_true, eps=eps_true, A=A_true,
                     R0=R0_true, L0=L0_true)

proc = OverdampedProcess(F_sim, D=D_iso, extras_global={"box": box})
proc.set_params(theta_F=theta_F_exact)

print(f"Simulation PSF with parameters: {list(theta_F_exact.keys())}")
print(f"True values: {theta_F_exact}")

# %%
# Running the simulation
# -------------------------

# Random initial conditions: uniform in box, random headings
key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0 = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

proc.initialize(x0)
key, sub = random.split(key)
coll = proc.simulate(dt=dt_sim, Nsteps=Nsteps, key=sub)

ds = coll.datasets[0]
print(f"Trajectory: {ds.X.shape[0]} frames, "
      f"{ds.X.shape[1]} particles, dim={ds.X.shape[-1]}")

# %%
# Simulation movie
# ------------------
#
# Animation of the simulated ABP system.  Particles are coloured by
# heading angle θ; white arrows show the propulsion direction.

Xfull = np.asarray(ds.X)  # full trajectory array (T × N × 3)

T_total = Xfull.shape[0]
skip_mov = max(1, T_total // 200)
Xmov_data = Xfull[::skip_mov]
T_mov_data = Xmov_data.shape[0]

fig_dm, ax_dm = _dark_fig(figsize=(4.5, 4.5))
Xf0 = _wrap_xy(Xmov_data[0], Lx, Ly)
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
    Xf = _wrap_xy(Xmov_data[frame], Lx, Ly)
    sc_dm.set_offsets(Xf[:, :2])
    sc_dm.set_array(Xf[:, 2])
    qv_dm.set_offsets(Xf[:, :2])
    qv_dm.set_UVC(_ARROW_LEN * np.cos(Xf[:, 2]),
                   _ARROW_LEN * np.sin(Xf[:, 2]))
    ttl_dm.set_text(f"t = {frame * skip_mov * dt_sim:.1f}")
    return sc_dm, qv_dm, ttl_dm


anim_data = FuncAnimation(fig_dm, _update_data,
                          frames=T_mov_data, interval=40, blit=True)
anim_data.save(_OUTDIR / "simulation.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {_OUTDIR / 'simulation.mp4'}")
plt.show()
# %%
# Saving and loading trajectory data
# -------------------------------------
#
# SFI trajectory collections can be serialised to CSV, Parquet, or
# HDF5.  Global extras — here the periodic box — are embedded in a
# YAML header and survive the round-trip.

from SFI.trajectory import TrajectoryCollection

csv_path = os.path.join(tempfile.gettempdir(), "sfi_abp_demo.csv")
coll.save(csv_path, format="csv")

with open(csv_path) as f:
    for line in f.readlines()[:10]:
        print(line.rstrip())
print("  ...")

# Reload into a fresh collection and verify
coll_reloaded = TrajectoryCollection.load(csv_path)
ds_re = coll_reloaded.datasets[0]
print(f"\nOriginal:  {ds.X.shape}  →  Loaded: {ds_re.X.shape}")
max_err = float(np.max(np.abs(np.asarray(ds_re.X) - np.asarray(ds.X))))
print(f"Max |ΔX| = {max_err:.1e}  (numerical round-trip error)")

# %%
# Particle snapshot
# -------------------
#
# Final-frame snapshot with heading arrows (left) and individual
# particle trajectories (right).  Positions are wrapped into the
# periodic box; particles are coloured by heading angle.

n_show = min(5, N_particles)  # trajectories to display

fig_snap, axes_snap = _dark_fig(1, 2, figsize=(12, 5.5))

_draw_particles(axes_snap[0], np.asarray(ds.X[-1]), Lx, Ly)
axes_snap[0].set_title(f"Snapshot (t = {Nsteps * dt_sim:.0f})")

for p in range(n_show):
    Xp = np.asarray(ds.X[:, p, :2])
    Xp_w = _wrap_trajectory(Xp, np.array([Lx, Ly]))
    axes_snap[1].plot(Xp_w[:, 0], Xp_w[:, 1], lw=0.8, alpha=0.85)
axes_snap[1].set_xlim(0, Lx); axes_snap[1].set_ylim(0, Ly)
axes_snap[1].set_aspect("equal")
axes_snap[1].set_xlabel("x"); axes_snap[1].set_ylabel("y")
axes_snap[1].set_title(f"Trajectories ({n_show} particles)")

fig_snap.suptitle(f"Aligning ABPs  (N = {N_particles})",
                  fontsize=13, color="white")
fig_snap.savefig(_OUTDIR / "snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'snapshot.png'}")
plt.show()
# %%
# Building the over-complete inference basis
# -----------------------------------------------
#
# For **inference**, we reuse the same building blocks but with an
# over-complete dictionary of radial kernels.  The basis contains
# many more kernels than the true model uses, so the linear solver
# must pick out the right combination.
#
# The ``heading_vector`` basis B_heading is shared with the
# simulation model above.

# Radial kernels: r^n * exp(-r/L) for n ∈ {0,1}, L ∈ {0.5, 1, 2, 4}
repel_kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
align_kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])

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
print(f"Basis: {B_full.n_features} features "
      f"(heading={n_heading}, repel={n_repel}, align={n_align})")

# %%
# Linear force inference
# -------------------------
#
# Standard ``infer_force_linear`` recovers the interaction
# coefficients in a single linear solve — no gradient-based
# optimisation needed.

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.utils import plotting

# Create inference object from the trajectory collection
inf = OverdampedLangevinInference(coll)

# Estimate diffusion coefficient
inf.compute_diffusion_constant(method="WeakNoise")

# Solve for force coefficients (one-shot linear solve)
inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")

# Compare to exact model for quantitative error metrics
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()

nmse_abp = float(inf.NMSE_force)
print(f"\nNMSE(force) = {nmse_abp:.4f}")

# %%
# Force scatter: exact vs inferred
# -----------------------------------
#
# All force components (translational and angular) along the
# trajectory, plotted as a scatter.  Points near the diagonal
# indicate good inference.

F_exact_vals = proc.force_sf(coll.X, extras={"box": box})
F_inf_vals = inf.force_inferred(coll.X, extras={"box": box})

fig_sc, ax_sc = plt.subplots(figsize=(4.5, 4.5))
plotting.comparison_scatter(F_exact_vals, F_inf_vals, maxpoints=5000, alpha=0.05)
ax_sc.set_xlabel("Exact F")
ax_sc.set_ylabel("Inferred F")
ax_sc.set_title(f"Force scatter  (NMSE = {nmse_abp:.3f})")
fig_sc.savefig(_OUTDIR / "force_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'force_scatter.png'}")
plt.show()
# %%
# Recovered interaction kernels
# --------------------------------
#
# The inferred coefficients weight the radial kernels.  Summing
# c_k φ_k(r) reconstructs the effective radial interaction shape.
# Both curves are normalised by the peak magnitude of the true
# kernel so the y-axis reads as a fraction of the true interaction
# strength.

coeffs = np.asarray(inf.force_coefficients)

# Split inferred coefficients into the three basis blocks
c_heading = coeffs[:n_heading]
c_repel = coeffs[n_heading : n_heading + n_repel]
c_align = coeffs[n_heading + n_repel :]

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

# Evaluate kernel basis functions on a radial grid
r_eval = np.linspace(0.01, 8.0, 200)
r_jax = jnp.array(r_eval)
phi_repel = np.array([np.asarray(fn(r_jax)) for fn, _ in repel_kernels])
phi_align = np.array([np.asarray(fn(r_jax)) for fn, _ in align_kernels])

# Reconstruct interaction profiles: coefficients @ kernel_matrix
true_repel = true_c_repel @ phi_repel
learned_repel = c_repel @ phi_repel
true_align = true_c_align @ phi_align
learned_align = c_align @ phi_align

scale_repel = np.max(np.abs(true_repel)) or 1.0
scale_align = np.max(np.abs(true_align)) or 1.0

fig_kern, axes_kern = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (true_k, learned_k, scale, ylabel, title) in zip(
    axes_kern,
    [
        (true_repel, learned_repel, scale_repel,
         r"$\phi_\mathrm{repel}(r)\;/\;\max|\phi_\mathrm{true}|$",
         "Radial repulsion kernel"),
        (true_align, learned_align, scale_align,
         r"$\phi_\mathrm{align}(r)\;/\;\max|\phi_\mathrm{true}|$",
         "Angular alignment kernel"),
    ],
):
    ax.plot(r_eval, true_k / scale, "--", lw=2, color=COL_EXACT,
            label="True")
    ax.plot(r_eval, learned_k / scale, lw=2, color=COL_INFERRED,
            label="Learned")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    # Focus y-axis on the true-kernel range with comfortable margin
    all_vals = np.concatenate([true_k / scale, learned_k / scale])
    ylo, yhi = float(all_vals.min()), float(all_vals.max())
    margin = 0.3 * max(yhi - ylo, 0.1)
    ax.set_ylim(ylo - margin, yhi + margin)

fig_kern.suptitle("Interaction kernel recovery", fontsize=13)
fig_kern.savefig(_OUTDIR / "kernels.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'kernels.png'}")
plt.show()
# %%
# Inferred self-propulsion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The heading coefficient recovers the active thrust magnitude c₀:

print(f"Self-propulsion: true c\u2080 = {float(c0_true):.2f}, "
      f"inferred = {float(c_heading[0]):.2f}")

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

fig_bs_snap, axes_bs = _dark_fig(1, 2, figsize=(12, 5.5))

_draw_particles(axes_bs[0], np.asarray(ds.X[-1]), Lx, Ly)
axes_bs[0].set_title("Original (final frame)")

_draw_particles(axes_bs[1], np.asarray(coll_boot.datasets[0].X[-1]), Lx, Ly)
axes_bs[1].set_title("Bootstrapped (inferred model)")

fig_bs_snap.suptitle("Bootstrap validation", fontsize=13, color="white")
fig_bs_snap.savefig(_OUTDIR / "bootstrap_snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'bootstrap_snapshot.png'}")
plt.show()
# %%
# Bootstrap comparison movie
# ----------------------------
#
# Side-by-side animation of the original trajectory (left) and
# a trajectory simulated from the *inferred* model (right).
# Both start from the same initial conditions.

Xboot_full = np.asarray(coll_boot.datasets[0].X)

skip = max(1, T_total // 200)
Xmov = Xfull[::skip]
Xboot_mov = Xboot_full[::skip]
T_mov = min(Xmov.shape[0], Xboot_mov.shape[0])

fig_bs, (ax_orig, ax_boot) = _dark_fig(1, 2, figsize=(12, 6))

for _ax, _title in [(ax_orig, "Original"), (ax_boot, "Inferred model")]:
    _ax.set_xlim(0, Lx); _ax.set_ylim(0, Ly)
    _ax.set_aspect("equal")
    _ax.set_xlabel("x"); _ax.set_ylabel("y")
    _ax.set_title(_title)

Xf0_o = _wrap_xy(Xmov[0], Lx, Ly)
sc_o = ax_orig.scatter(Xf0_o[:, 0], Xf0_o[:, 1], s=_PT_SIZE,
                       c=Xf0_o[:, 2], cmap="hsv",
                       vmin=-np.pi, vmax=np.pi, zorder=3,
                       edgecolors="white", lw=0.5)
qv_o = ax_orig.quiver(Xf0_o[:, 0], Xf0_o[:, 1],
                       _ARROW_LEN * np.cos(Xf0_o[:, 2]),
                       _ARROW_LEN * np.sin(Xf0_o[:, 2]),
                       scale=1.0, scale_units="xy", width=0.008,
                       color="white", alpha=0.85, zorder=4)

Xf0_b = _wrap_xy(Xboot_mov[0], Lx, Ly)
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
    Xfo = _wrap_xy(Xmov[frame], Lx, Ly)
    sc_o.set_offsets(Xfo[:, :2])
    sc_o.set_array(Xfo[:, 2])
    qv_o.set_offsets(Xfo[:, :2])
    qv_o.set_UVC(_ARROW_LEN * np.cos(Xfo[:, 2]),
                  _ARROW_LEN * np.sin(Xfo[:, 2]))
    Xfb = _wrap_xy(Xboot_mov[frame], Lx, Ly)
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
anim_bootstrap.save(_OUTDIR / "bootstrap_comparison.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {_OUTDIR / 'bootstrap_comparison.mp4'}")
plt.show()
# %%
# Both simulations start from the same initial conditions.
# Visual similarity confirms the inferred model captures the
# collective dynamics.
