"""
Nonreciprocal ABPs at large scale — 3 000 particles
=====================================================

Infer three pairwise interaction kernels — **repulsion**, **alignment**,
and **pursuit** — in a system of 3 000 active Brownian particles whose
interactions are **nonreciprocal** and **vision-gated**.

This example scales up the ABP pair-interaction demo to large particle
numbers using *truncated-range neighbor lists* (CSR format) and the
:func:`~SFI.langevin.chunked.simulate_chunked` helper that periodically
rebuilds the neighbor list during simulation.

It demonstrates:

1. Building a parametric simulation force from composable geometric
   primitives in :mod:`SFI.bases.pairs` — including
   :func:`~SFI.bases.pairs.vision_gate` for nonreciprocal coupling and
   :func:`~SFI.bases.pairs.particle_heading` for pursuit.
2. Chunked simulation with periodic neighbor-list rebuilds via
   :func:`~SFI.langevin.chunked.simulate_chunked`.
3. Constructing an overcomplete inference basis with
   :func:`~SFI.bases.pairs.scalar_pair_basis` and the same geometric
   building blocks.
4. **Linear** force inference on a subset of trajectory chunks.
5. Recovering all three interaction kernels vs the true model.

.. rubric:: Tags

synthetic · overdamped · multi-particle · linear · interactions · large-scale · nonreciprocal
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "linear", "interactions", "large-scale", "nonreciprocal"]
# sphinx_gallery_thumbnail_number = 5

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output, plot_kernel_ci
from SFI.utils.plotting import animate_particles, dark_fig, plot_particles, wrap_positions

apply_style()

# Force black save background for animations (H264 has no alpha).
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False
# sphinx_gallery_end_ignore
# %%
# System: nonreciprocal, vision-gated ABPs
# ------------------------------------------
#
# Each particle has state :math:`(x, y, \theta)`.  The deterministic
# force contains three pairwise interactions, each modulated by a
# **vision cone** :math:`v(\delta) = (1 + \cos\delta)/2` that depends
# on the bearing angle :math:`\delta` from the focal particle's
# heading to the neighbor — making the interactions *nonreciprocal*:
#
# - **Self-propulsion**: :math:`c_0\,\hat{e}_\theta`
# - **Repulsion**: :math:`-\varepsilon\, e^{-r/R_0}\,\hat{r}_{ij}`
# - **Alignment** (vision-gated): :math:`A\,v(\delta)\,\sin(\Delta\theta)\,
#   e^{-r/L_a}`
# - **Pursuit** (vision-gated): :math:`P\,v(\delta)\,\hat{e}_{\theta_j}\,
#   e^{-r/L_p}` — steers the focal particle toward the neighbor's heading.
#
# The system lives in a periodic box with ~3 000 particles at the
# same density as the 60-particle ABP demo.

N_particles = 3_000
density = 60 / (30.0 * 30.0)       # same as abp_align_demo
area = N_particles / density
Lx = Ly = float(np.sqrt(area))     # ≈ 387
Nsteps = 2000
dt_sim = 0.02
D_iso = 0.05
seed = 0
box = jnp.array([Lx, Ly])
box_np = np.array([Lx, Ly])

# Interaction range and neighbor-list parameters
cutoff = 12.0       # truncation radius
rebuild_every = 1   # rebuild neighbor list every step (fast with cKDTree)

# True model parameters
c0_true = 1.0       # self-propulsion
eps_true = 2.0      # repulsion strength
A_true = 0.3        # alignment strength
P_true = 1.5        # pursuit strength
R0_true = 1.0       # repulsion length scale
La_true = 2.0       # alignment length scale
Lp_true = 4.0       # pursuit length scale

theta_F_exact = dict(c0=c0_true, eps=eps_true, R0=R0_true,
                     A=A_true, La=La_true, P=P_true, Lp=Lp_true)

print(f"N = {N_particles},  box = {Lx:.1f}×{Ly:.1f},  cutoff = {cutoff}")

# %%
# Building the parametric simulation force
# --------------------------------------------
#
# We compose the simulation force from **geometric primitives** in
# :mod:`SFI.bases.pairs`.  Each primitive is a building block — a
# radial kernel, a direction vector, a gating function — that can
# be multiplied together and dispatched over pairs.

from SFI.bases.pairs import (
    angle_coupling,
    exp_poly_kernels,
    heading_vector,
    pair_direction,
    parametric_radial_kernel,
    particle_heading,
    scalar_pair_basis,
    vision_gate,
)
from SFI.statefunc import Basis

dim = 3  # (x, y, θ) per particle

# Geometric primitives
B_heading = heading_vector(dim=dim, angle_index=2)
e_ij = pair_direction(
    dim=dim, box="extras", spatial_dims=slice(0, 2),
    embed_dim=dim, embed_axes=[0, 1],
)
g_align = angle_coupling(jnp.sin, dim=dim, angle_index=2)
e_j = particle_heading(1, dim=dim, angle_index=2)
v = vision_gate(
    lambda d: (1 + jnp.cos(d)) / 2,
    dim=dim, angle_index=2,
    box="extras", spatial_dims=slice(0, 2),
)

# Parametric radial kernels
k_repel = parametric_radial_kernel(
    lambda r, p: -p["eps"] * jnp.exp(-r / p["R0"]),
    params={"eps": (), "R0": ()},
    dim=dim, box="extras", spatial_dims=slice(0, 2),
)
k_align = parametric_radial_kernel(
    lambda r, p: p["A"] * jnp.exp(-r / p["La"]),
    params={"A": (), "La": ()},
    dim=dim, box="extras", spatial_dims=slice(0, 2),
)
k_pursuit = parametric_radial_kernel(
    lambda r, p: p["P"] * jnp.exp(-r / p["Lp"]),
    params={"P": (), "Lp": ()},
    dim=dim, box="extras", spatial_dims=slice(0, 2),
)

# CSR dispatch keys (neighbor list stored in extras)
csr_kw = dict(indptr_key="indptr", indices_key="indices")

# Compose the full force: self-propulsion + repulsion + alignment + pursuit
F_sim = (
    B_heading.to_psf(coeff_key="c0")
    + (k_repel * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    + (k_align * v * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    + (k_pursuit * v * e_j).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
)

print(f"Force primitives: heading, repulsion, "
      f"vision-gated alignment, vision-gated pursuit")

# %%
# Chunked simulation with neighbor rebuilds
# --------------------------------------------
#
# At this scale (N = 3 000) a full N² pair list is infeasible.
# :func:`~SFI.langevin.chunked.simulate_chunked` builds a CSR
# neighbor list using :func:`scipy.spatial.cKDTree` (≈ 0.05 s per
# rebuild) and rebuilds it periodically.  Here we rebuild *every
# step* because the collective flock motion can exceed any practical
# Verlet skin in a single time step.
#
# The ``save_every`` parameter decouples the output chunk size (50
# frames) from the rebuild frequency (1 step).

from SFI.langevin import OverdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.utils.neighbors import make_neighbor_extras

# Initial conditions: uniform in box, random headings
key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0 = random.uniform(kth, (N_particles,),
                      minval=-jnp.pi, maxval=jnp.pi)
x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

# Build initial neighbor list
nbr0 = make_neighbor_extras(np.asarray(x0[:, :2]), cutoff, box_np)
extras0 = {"box": box}
extras0.update(nbr0)

proc = OverdampedProcess(F_sim, D=D_iso, extras_global=extras0)
proc.set_params(theta_F=theta_F_exact)
proc.initialize(x0)

# Run chunked simulation
print(f"Simulating {Nsteps} steps with neighbor rebuild every step ...")
t0 = time.perf_counter()
key, sub = random.split(key)
coll = simulate_chunked(
    proc, dt=dt_sim, Nsteps=Nsteps, key=sub,
    cutoff=cutoff, box=box_np,
    skin=0.0, rebuild_every=rebuild_every,
    save_every=50,
    spatial_dims=slice(0, 2),
    nnz_safety=3.0, verbose=False,
)
sim_time = time.perf_counter() - t0
n_chunks = len(coll.datasets)
print(f"Simulation done in {sim_time:.0f}s  ({n_chunks} chunks)")

# %%
# Simulation snapshot
# ----------------------
#
# Final-frame snapshot with 3 000 particles coloured by heading
# angle :math:`\theta`.  Collective flocking structures emerge
# from the pursuit and alignment interactions.

Xfull = coll.to_array(axis="time")  # (T, N, 3)

# sphinx_gallery_start_ignore
fig_snap, ax_snap = dark_fig(figsize=(8, 8))
plot_particles(coll, dataset=n_chunks - 1, t_index=-1,
               color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
               box=box_np, s=2, edgecolors="none", ax=ax_snap)
ax_snap.set_xlim(0, Lx); ax_snap.set_ylim(0, Ly)
ax_snap.set_title(f"N = {N_particles}  (final frame)",
                  color="white", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Simulation movie
# -------------------
#
# Full-frame animation of the flock dynamics showing the emergence
# and persistence of collective structures.

# sphinx_gallery_start_ignore
T_total = Xfull.shape[0]
skip = max(1, T_total // 200)

# Animate the FULL concatenated run: `coll` is chunked (40 datasets of 50
# frames) and animate_particles reads a single dataset, so passing `coll`
# directly would yield a 5-frame movie.
from SFI import TrajectoryCollection

coll_movie = TrajectoryCollection.from_arrays(X=np.asarray(Xfull), dt=dt_sim)
fig_dm, ax_dm = dark_fig(figsize=(8, 8))
anim_flock = animate_particles(coll_movie, color_dim=2, box=box_np, cmap='hsv',
                  vmin=-np.pi, vmax=np.pi, s=2, skip=skip, ax=ax_dm)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Building the overcomplete inference basis
# --------------------------------------------
#
# For inference we build an overcomplete basis from the same geometric
# primitives.  Instead of parametric kernels we use a grid of
# exponential-polynomial basis functions via
# :func:`~SFI.bases.pairs.exp_poly_kernels`, combined with the same
# vision gate :math:`v`, alignment coupling :math:`g`, and heading
# vector :math:`\hat{e}_j`.

kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
phi_r = scalar_pair_basis(kernels, dim=dim, box="extras",
                          spatial_dims=slice(0, 2))

B_repel = (phi_r * e_ij).dispatch_pairs_from_extras(
    **csr_kw, return_as="basis")
B_align = (phi_r * v * g_align).dispatch_pairs_from_extras(
    **csr_kw, return_as="basis")
B_pursuit = (phi_r * v * e_j).dispatch_pairs_from_extras(
    **csr_kw, return_as="basis")
B_full = Basis.stack([B_heading, B_repel, B_align, B_pursuit])

n_heading = B_heading.n_features
n_repel = B_repel.n_features
n_align = B_align.n_features
n_pursuit = B_pursuit.n_features

# %%
# Linear force inference on mid-trajectory chunks
# ---------------------------------------------------
#
# We select 8 evenly-spaced chunks from the middle of the trajectory
# (after the system has relaxed).  For each chunk we build a fresh
# CSR neighbor list from the first frame.

from SFI import OverdampedLangevinInference
from SFI.trajectory import TrajectoryCollection

INFER_CHUNKS = list(range(12, 28, 2))  # [12, 14, 16, 18, 20, 22, 24, 26]
datasets = []
for ci in INFER_CHUNKS:
    Xc = np.asarray(coll.to_arrays(dataset=ci)[1])  # (t, X, mask) -> X
    X0 = Xc[0]
    nbr = make_neighbor_extras(X0[:, :2], cutoff, box_np)
    eg = {"box": box}
    eg.update(nbr)
    ds = TrajectoryCollection.from_arrays(
        X=Xc, dt=dt_sim, extras_global=eg,
    ).datasets[0]
    datasets.append(ds)
    print(f"  chunk {ci}: nnz = {int(nbr['indptr'][-1])}")

coll_infer = TrajectoryCollection(
    datasets=datasets,
    weights=jnp.ones(len(datasets), dtype=jnp.float32),
).with_weights("pool")

# Inference
inf = OverdampedLangevinInference(coll_infer)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")

# Estimate coefficient uncertainty (populates force_coefficients_covariance)
inf.compute_force_error()

# Compare to exact model for error metrics
X0_ref = np.asarray(coll_infer.to_arrays(dataset=0)[1][0])  # X, first frame
nbr_ref = make_neighbor_extras(X0_ref[:, :2], cutoff, box_np)
extras_ref = {"box": box}
extras_ref.update(nbr_ref)
proc_ref = OverdampedProcess(F_sim, D=D_iso, extras_global=extras_ref)
proc_ref.set_params(theta_F=theta_F_exact)
proc_ref.initialize(jnp.array(X0_ref))

inf.compare_to_exact(model_exact=proc_ref, maxpoints=2000)
inf.print_report()

# NMSE(force) is already printed by inf.print_report(); keep the scalar
# only for the kernel-recovery figure title below.
nmse = float(inf.NMSE_force)

# Report inferred self-propulsion (true-vs-inferred narration)
c_heading, _ = inf.coeff_block((0, n_heading))

# %%
# Recovered interaction kernels
# --------------------------------
#
# Each interaction kernel is reconstructed as
# :math:`\sum_k c_k \phi_k(r)` using the inferred coefficients
# and the basis functions.  Shaded bands show 95 % confidence
# intervals derived from the coefficient covariance
# (see :func:`~SFI.inference.kernel_predict_ci`).

from SFI.inference import kernel_predict_ci

# Per-group coefficients and covariance sub-blocks (no hand-sliced
# offsets into the flat covariance — coeff_block returns both).
i0_repel = n_heading
i1_repel = n_heading + n_repel
i0_align = i1_repel
i1_align = i0_align + n_align
i0_pursuit = i1_align
i1_pursuit = i0_pursuit + n_pursuit
c_repel, cov_repel = inf.coeff_block((i0_repel, i1_repel))
c_align, cov_align = inf.coeff_block((i0_align, i1_align))
c_pursuit, cov_pursuit = inf.coeff_block((i0_pursuit, i1_pursuit))

# True coefficient vectors
# Kernel order from exp_poly_kernels(degrees=[0,1], lengths=[0.5,1,2,4]):
#   0: r⁰·exp(-r/0.5)  1: r⁰·exp(-r/1)  2: r⁰·exp(-r/2)  3: r⁰·exp(-r/4)
#   4: r¹·exp(-r/0.5)  5: r¹·exp(-r/1)  6: r¹·exp(-r/2)  7: r¹·exp(-r/4)
idx_repel = 1     # r⁰·exp(-r/R₀) with R₀=1
idx_align = 2     # r⁰·exp(-r/Lₐ) with Lₐ=2
idx_pursuit = 3   # r⁰·exp(-r/Lₚ) with Lₚ=4

true_c_repel = np.zeros(n_repel)
true_c_repel[idx_repel] = -eps_true
true_c_align = np.zeros(n_align)
true_c_align[idx_align] = A_true
true_c_pursuit = np.zeros(n_pursuit)
true_c_pursuit[idx_pursuit] = P_true

# Compute kernel profiles with confidence intervals
r_eval = np.linspace(0.01, 8.0, 200)
ci_repel = kernel_predict_ci(r_eval, kernels, c_repel, cov_repel)
ci_align = kernel_predict_ci(r_eval, kernels, c_align, cov_align)
ci_pursuit = kernel_predict_ci(r_eval, kernels, c_pursuit, cov_pursuit)

# True kernel profiles
true_repel = true_c_repel @ ci_repel["phi"]
true_align = true_c_align @ ci_align["phi"]
true_pursuit = true_c_pursuit @ ci_pursuit["phi"]

# sphinx_gallery_start_ignore
profiles = [
    ("Repulsion", ci_repel, true_repel),
    ("Alignment (vision-gated)", ci_align, true_align),
    ("Pursuit (vision-gated)", ci_pursuit, true_pursuit),
]

fig_kern, axes_kern = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, ci, true_k) in zip(axes_kern, profiles):
    plot_kernel_ci(ax, r_eval, true_k, ci, ci_label="95% CI")
    ax.set_xlabel("r")
    ax.set_title(title)
    ax.legend()

fig_kern.suptitle(
    f"Kernel recovery  (N = {N_particles}, NMSE = {nmse:.4f})",
    fontsize=13,
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Bootstrap simulation
# ----------------------
#
# We simulate a trajectory from the inferred model starting from
# a relaxed mid-trajectory frame and using chunked integration
# with periodic neighbor rebuilds.

key_boot = random.PRNGKey(seed + 77)

proc_boot = OverdampedProcess(
    inf.force_inferred._psf, inf.diffusion_inferred._psf,
)
proc_boot.set_params(
    theta_F=inf.force_inferred.params,
    theta_D=inf.diffusion_inferred.params,
)

# Start from a relaxed mid-trajectory frame
mid_frame = Xfull.shape[0] // 2
x0_boot = jnp.array(Xfull[mid_frame])

nbr_boot = make_neighbor_extras(np.asarray(x0_boot[:, :2]),
                                cutoff, box_np)
extras_boot = {"box": box}
extras_boot.update(nbr_boot)
proc_boot.set_extras(extras_global=extras_boot)
proc_boot.initialize(x0_boot)

boot_steps = 200
print(f"Bootstrap: {boot_steps} steps from mid-trajectory frame {mid_frame} ...")
key_boot, sub_boot = random.split(key_boot)
coll_boot = simulate_chunked(
    proc_boot, dt=dt_sim, Nsteps=boot_steps, key=sub_boot,
    cutoff=cutoff, box=box_np,
    skin=0.0, rebuild_every=rebuild_every,
    save_every=50,
    spatial_dims=slice(0, 2),
    nnz_safety=3.0, verbose=False,
)
Xboot = coll_boot.to_array(axis="time")
print(f"Bootstrap done: {Xboot.shape[0]} frames")

# sphinx_gallery_start_ignore
fig_bs, (ax_o, ax_b) = dark_fig(1, 2, figsize=(16, 7.5))
for _ax in (ax_o, ax_b):
    _ax.set_xlim(0, Lx); _ax.set_ylim(0, Ly); _ax.set_aspect("equal")
    _ax.set_xlabel("x"); _ax.set_ylabel("y")

Xw_o = wrap_positions(Xfull[mid_frame], box_np)

ax_o.scatter(Xw_o[:, 0], Xw_o[:, 1], s=2,
             c=Xw_o[:, 2], cmap="hsv",
             vmin=-np.pi, vmax=np.pi, edgecolors="none")
ax_o.set_title("Original (mid-trajectory)", color="white")

plot_particles(coll_boot, dataset=len(coll_boot.datasets) - 1,
               t_index=-1, color_dim=2, cmap='hsv',
               vmin=-np.pi, vmax=np.pi, box=box_np, s=2, ax=ax_b)
ax_b.set_title("Inferred model (bootstrap)", color="white")

fig_bs.suptitle("Bootstrap validation", fontsize=13, color="white")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Thumbnail
# ----------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
ax_t.set_facecolor("black")
fig_thumb.patch.set_facecolor("black")

# Plot a zoomed-in region showing flock structure
cx, cy = Lx / 2, Ly / 2
zoom = 60.0
Xw_t = wrap_positions(Xfull[-1], box_np)
mask_t = (
    (np.abs(Xw_t[:, 0] - cx) < zoom / 2) &
    (np.abs(Xw_t[:, 1] - cy) < zoom / 2)
)
Xt = Xw_t[mask_t]
ax_t.scatter(Xt[:, 0], Xt[:, 1], s=6,
             c=Xt[:, 2], cmap="hsv",
             vmin=-np.pi, vmax=np.pi, edgecolors="none")
ax_t.set_xlim(cx - zoom / 2, cx + zoom / 2)
ax_t.set_ylim(cy - zoom / 2, cy + zoom / 2)
ax_t.set_aspect("equal")
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
