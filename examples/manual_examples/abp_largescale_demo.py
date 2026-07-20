# TODO: review this file
"""
Large-scale active Brownian particles with truncated-range interactions
======================================================================

Scales the vision-gated pursuit demo to ~10,000 particles using:
  - Cell-list neighbor builder  (``SFI.utils.neighbors``)
  - Chunked simulation          (``SFI.langevin.chunked``)
  - CSR pair dispatch            (``dispatch_pairs_from_extras``)

No O(N²) all-pairs — the neighbor list is rebuilt between chunks.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

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
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.langevin import OverdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.statefunc import Basis
from SFI.utils.neighbors import make_neighbor_extras

_OUTDIR = Path(__file__).resolve().parent / "abp_largescale_output"
_OUTDIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

N_particles = 10_000
density     = 60 / (30.0 * 30.0)  # same density as small demo
area        = N_particles / density
Lx = Ly     = float(np.sqrt(area))  # ≈ 387
Nsteps      = 2000
dt          = 0.02
D           = 0.05
seed        = 0
box_np      = np.array([Lx, Ly])
box         = jnp.array([Lx, Ly])

cutoff      = 12.0        # interaction cutoff radius
rebuild_every = 50        # rebuild neighbor list every N steps

# True force parameters
c0  = 1.0    # self-propulsion
eps = 2.0    # repulsion
A   = 0.3    # alignment (gated)
P   = 1.5    # pursuit   (gated, nonreciprocal)
R0  = 1.0    # repulsion length
La  = 2.0    # alignment length
Lp  = 4.0    # pursuit length

print(f"N = {N_particles},  box = {Lx:.1f} × {Ly:.1f},  "
      f"cutoff = {cutoff},  rebuild_every = {rebuild_every}")


# ═══════════════════════════════════════════════════════════════════════
#  FORCE MODEL (CSR dispatch — no AutoPairs)
# ═══════════════════════════════════════════════════════════════════════

dim = 3  # (x, y, θ)

B_heading = heading_vector(dim=dim, angle_index=2)

e_ij = pair_direction(
    dim=dim, box="extras", spatial_dims=slice(0, 2),
    embed_dim=dim, embed_axes=[0, 1],
)
g_align = angle_coupling(jnp.sin, dim=dim, angle_index=2)
e_j     = particle_heading(1, dim=dim, angle_index=2)
v       = vision_gate(
    lambda d: (1 + jnp.cos(d)) / 2,
    dim=dim, angle_index=2,
    box="extras", spatial_dims=slice(0, 2),
)

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

# CSR dispatch — reads "indptr" / "indices" from extras
csr_kw = dict(indptr_key="indptr", indices_key="indices")

F_sim = (
    B_heading.to_psf(coeff_key="c0")
    + (k_repel * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    + (k_align * v * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    + (k_pursuit * v * e_j).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
)

theta_exact = dict(c0=c0, eps=eps, R0=R0, A=A, La=La, P=P, Lp=Lp)


# ═══════════════════════════════════════════════════════════════════════
#  INITIAL CONDITIONS + NEIGHBOR LIST
# ═══════════════════════════════════════════════════════════════════════

key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0   = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0    = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

# Build initial neighbor list
print("Building initial neighbor list ...")
t0 = time.perf_counter()
nbr_extras = make_neighbor_extras(np.asarray(x0[:, :2]), cutoff, box_np)
nnz0 = len(nbr_extras["indices"])
mean_nbr = nnz0 / N_particles
print(f"  nnz = {nnz0},  mean neighbors/particle = {mean_nbr:.1f}  "
      f"({time.perf_counter() - t0:.2f}s)")

extras0 = {"box": box}
extras0.update(nbr_extras)

proc = OverdampedProcess(F_sim, D=D, extras_global=extras0)
proc.set_params(theta_F=theta_exact)
proc.initialize(x0)


# ═══════════════════════════════════════════════════════════════════════
#  CHUNKED SIMULATION
# ═══════════════════════════════════════════════════════════════════════

print(f"\nSimulating {Nsteps} steps (chunks of {rebuild_every}) ...")
t0 = time.perf_counter()
key, sub = random.split(key)
coll = simulate_chunked(
    proc, dt=dt, Nsteps=Nsteps, key=sub,
    cutoff=cutoff, box=box_np,
    rebuild_every=rebuild_every,
    spatial_dims=slice(0, 2),
    nnz_safety=3.0,
    verbose=True,
)
elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s")

ds = coll.datasets[-1]  # last chunk for final snapshot
Xfull = np.asarray(ds.X)
print(f"Trajectory shape: {Xfull.shape} (last of {len(coll.datasets)} chunks)")


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════
#
# Build an overcomplete basis using the same geometric primitives
# but with CSR dispatch.

kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
phi_r   = scalar_pair_basis(kernels, dim=dim, box="extras", spatial_dims=slice(0, 2))

B_repel   = (phi_r * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
B_align   = (phi_r * v * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
B_pursuit = (phi_r * v * e_j).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
B_full    = Basis.stack([B_heading, B_repel, B_align, B_pursuit])

n_heading = B_heading.n_features
n_repel   = B_repel.n_features
n_align   = B_align.n_features
n_pursuit = B_pursuit.n_features
print(f"\nBasis: {B_full.n_features} features "
      f"(heading={n_heading}, repel={n_repel}, align={n_align}, pursuit={n_pursuit})")

print("Running inference ...")
t0 = time.perf_counter()
inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()
print(f"Inference done in {time.perf_counter() - t0:.1f}s")

nmse = float(inf.NMSE_force)
print(f"NMSE(force) = {nmse:.4f}")

coeffs = np.asarray(inf.force_coefficients)
i0 = 0
c_heading = coeffs[i0 : i0 + n_heading]; i0 += n_heading
c_repel   = coeffs[i0 : i0 + n_repel];   i0 += n_repel
c_align   = coeffs[i0 : i0 + n_align];   i0 += n_align
c_pursuit = coeffs[i0 : i0 + n_pursuit]; i0 += n_pursuit

print(f"Self-propulsion: true c0 = {c0:.2f}, inferred = {float(c_heading[0]):.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════


def _wrap_xy(X, Lx, Ly):
    X = np.array(X, copy=True)
    X[:, 0] %= Lx
    X[:, 1] %= Ly
    return X


# --- snapshot ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
Xw = _wrap_xy(Xfull[-1], Lx, Ly)
ax.scatter(Xw[:, 0], Xw[:, 1], s=2,
           c=Xw[:, 2], cmap="hsv",
           vmin=-np.pi, vmax=np.pi, edgecolors="none")
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect("equal")
ax.set_title(f"N = {N_particles}  (final frame)", color="white", fontsize=13)
ax.tick_params(colors="white")
fig.savefig(_OUTDIR / "snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'snapshot.png'}")
plt.show()

# --- zoom on a local patch ---
cx, cy = Lx / 2, Ly / 2
zoom = 30.0
fig_z, ax_z = plt.subplots(figsize=(6, 6))
ax_z.set_facecolor("black"); fig_z.patch.set_facecolor("black")
mask = (
    (np.abs(Xw[:, 0] - cx) < zoom / 2) &
    (np.abs(Xw[:, 1] - cy) < zoom / 2)
)
Xzoom = Xw[mask]
ax_z.scatter(Xzoom[:, 0], Xzoom[:, 1], s=40,
             c=Xzoom[:, 2], cmap="hsv",
             vmin=-np.pi, vmax=np.pi,
             edgecolors="white", lw=0.3)
ax_z.quiver(Xzoom[:, 0], Xzoom[:, 1],
            1.5 * np.cos(Xzoom[:, 2]),
            1.5 * np.sin(Xzoom[:, 2]),
            scale=1.0, scale_units="xy", width=0.005,
            color="white", alpha=0.7)
ax_z.set_xlim(cx - zoom / 2, cx + zoom / 2)
ax_z.set_ylim(cy - zoom / 2, cy + zoom / 2)
ax_z.set_aspect("equal")
ax_z.set_title(f"Zoom (30×30 patch)", color="white")
ax_z.tick_params(colors="white")
fig_z.savefig(_OUTDIR / "zoom.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'zoom.png'}")
plt.show()

# --- kernel recovery ---

idx_repel_true   = 1
idx_align_true   = 2
idx_pursuit_true = 3

true_c_repel   = np.zeros(n_repel);   true_c_repel[idx_repel_true]     = -eps
true_c_align   = np.zeros(n_align);   true_c_align[idx_align_true]     = A
true_c_pursuit = np.zeros(n_pursuit); true_c_pursuit[idx_pursuit_true] = P

r_eval = np.linspace(0.01, 8.0, 200)
r_jax  = jnp.array(r_eval)
phi_mat = np.array([np.asarray(fn(r_jax)) for fn, _ in kernels])

COL_EXACT    = "#FF7A1A"
COL_INFERRED = "#FFC20A"

profiles = [
    ("Repulsion",   true_c_repel   @ phi_mat, c_repel   @ phi_mat),
    ("Alignment (gated)",  true_c_align   @ phi_mat, c_align   @ phi_mat),
    ("Pursuit (gated)",    true_c_pursuit @ phi_mat, c_pursuit @ phi_mat),
]

fig_kern, axes_kern = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, true_k, learned_k) in zip(axes_kern, profiles):
    scale = np.max(np.abs(true_k)) or 1.0
    ax.plot(r_eval, true_k / scale, "--", lw=2, color=COL_EXACT, label="True")
    ax.plot(r_eval, learned_k / scale, lw=2, color=COL_INFERRED, label="Learned")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("r"); ax.set_title(title); ax.legend()
    vals = np.concatenate([true_k, learned_k]) / scale
    margin = 0.3 * max(vals.max() - vals.min(), 0.1)
    ax.set_ylim(vals.min() - margin, vals.max() + margin)

fig_kern.suptitle(f"Kernel recovery  (N = {N_particles}, NMSE = {nmse:.4f})", fontsize=13)
fig_kern.savefig(_OUTDIR / "kernels.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'kernels.png'}")
plt.show()

print(f"\nAll outputs in {_OUTDIR}")
