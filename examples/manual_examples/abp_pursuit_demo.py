# TODO: review this file
"""
Nonreciprocal active Brownian particles — vision-gated pursuit
==============================================================

Infer pairwise interactions that break action–reaction symmetry
and are gated by a **vision cone**: particle *i* only "sees"
neighbours that lie roughly ahead of its heading.

Force model:
  F = c0·ê_i  −  ε·exp(−r/R0)·r̂_ij
      +  v(φ_ij−θ_i) · A·sin(Δθ)·exp(−r/La)·θ̂
      +  v(φ_ij−θ_i) · P·exp(−r/Lp)·ê_j

  ⊕  self-propulsion        c0 · ê_i
  ⊕  short-range repulsion  −ε · exp(−r/R0) · r̂_ij      (isotropic)
  ⊖  alignment torque        v · A · sin(Δθ) · exp(−r/La) · θ̂  (GATED)
  ⊖  pursuit force           v · P · exp(−r/Lp) · ê_j           (GATED)

where v(δ) = (1 + cos δ)/2 is a smooth vision-cone gate:
  v = 1  when j is directly ahead of i  (φ_ij = θ_i)
  v = 0  when j is directly behind i    (φ_ij = θ_i + π)

Composable primitives from ``SFI.bases.pairs``:
  pair_direction     →  r̂_ij            (reciprocal,    rank-1)
  angle_coupling     →  sin(Δθ) θ̂       (reciprocal,    rank-1)
  particle_heading   →  ê_j             (nonreciprocal, rank-1)
  vision_gate        →  v(φ_ij − θ_i)   (nonreciprocal, rank-0)

The inference basis includes *both* reciprocal and nonreciprocal
terms; the fit correctly identifies which coefficients are nonzero.
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
from SFI.statefunc import Basis
from SFI.trajectory import TrajectoryCollection
from SFI.utils import plotting

_OUTDIR = Path(__file__).resolve().parent / "abp_pursuit_output"
_OUTDIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

_ARROW_LEN = 1.5
_PT_SIZE   = 160
COL_EXACT    = "#FF7A1A"
COL_INFERRED = "#FFC20A"

plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False


def _wrap_xy(X, Lx, Ly):
    X = np.array(X, copy=True)
    X[:, 0] %= Lx
    X[:, 1] %= Ly
    return X


def _wrap_trajectory(xy, box):
    xy_w = xy.copy()
    xy_w[:, 0] %= box[0]
    xy_w[:, 1] %= box[1]
    jumps = np.any(np.abs(np.diff(xy_w, axis=0)) > 0.5 * box, axis=1)
    idx = np.flatnonzero(jumps)
    if idx.size:
        xy_w = np.insert(xy_w.astype(float), idx + 1, np.nan, axis=0)
    return xy_w


def _dark_ax(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("0.3")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _dark_fig(nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, **kw)
    fig.patch.set_facecolor("black")
    for ax in (np.atleast_1d(axes).flat if hasattr(axes, "flat") else [axes]):
        _dark_ax(ax)
    return fig, axes


def _draw_particles(ax, X, Lx, Ly):
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


def _init_anim(ax, X0, Lx, Ly):
    Xw = _wrap_xy(X0, Lx, Ly)
    sc = ax.scatter(Xw[:, 0], Xw[:, 1], s=_PT_SIZE,
                    c=Xw[:, 2], cmap="hsv",
                    vmin=-np.pi, vmax=np.pi, zorder=3,
                    edgecolors="white", lw=0.5)
    qv = ax.quiver(Xw[:, 0], Xw[:, 1],
                   _ARROW_LEN * np.cos(Xw[:, 2]),
                   _ARROW_LEN * np.sin(Xw[:, 2]),
                   scale=1.0, scale_units="xy", width=0.008,
                   color="white", alpha=0.85, zorder=4)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return sc, qv


def _update_anim(sc, qv, Xframe, Lx, Ly):
    Xw = _wrap_xy(Xframe, Lx, Ly)
    sc.set_offsets(Xw[:, :2])
    sc.set_array(Xw[:, 2])
    qv.set_offsets(Xw[:, :2])
    qv.set_UVC(_ARROW_LEN * np.cos(Xw[:, 2]),
               _ARROW_LEN * np.sin(Xw[:, 2]))


# ═══════════════════════════════════════════════════════════════════════
#  PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

N_particles = 60
Lx, Ly      = 30.0, 30.0
Nsteps      = 2000
dt          = 0.02
D           = 0.05
seed        = 0
box         = jnp.array([Lx, Ly])

# True force parameters
c0  = 1.0    # self-propulsion speed
eps = 2.0    # repulsion strength
A   = 0.3    # alignment strength      (reciprocal)
P   = 1.5    # pursuit strength         (NONRECIPROCAL)
R0  = 1.0    # repulsion length
La  = 2.0    # alignment length
Lp  = 4.0    # pursuit length


# ═══════════════════════════════════════════════════════════════════════
#  TRUE MODEL — vision-gated nonreciprocal pair primitives
# ═══════════════════════════════════════════════════════════════════════
#
# Each particle has state (x, y, θ).  The overdamped force is:
#   F = c0 ê_i  −  ε exp(−r/R0) r̂_ij
#       +  v(φ_ij−θ_i) · A sin(Δθ) exp(−r/La) θ̂
#       +  v(φ_ij−θ_i) · P exp(−r/Lp) ê_j
#
# Repulsion is isotropic.  Alignment and pursuit are gated by the
# vision cone v(δ) = (1 + cos δ)/2 — only neighbours in front of
# particle i contribute.

dim = 3  # (x, y, θ) per particle

# --- geometric primitives ---

B_heading = heading_vector(dim=dim, angle_index=2)

e_ij = pair_direction(                       # r̂_ij embedded as (·, ·, 0)
    dim=dim, box="extras", spatial_dims=slice(0, 2),
    embed_dim=dim, embed_axes=[0, 1],
)

g_align = angle_coupling(jnp.sin, dim=dim, angle_index=2)  # sin(Δθ) on θ

e_j = particle_heading(1, dim=dim, angle_index=2)  # neighbour's heading

v = vision_gate(                                    # vision cone  ← NONRECIPROCAL
    lambda d: (1 + jnp.cos(d)) / 2,
    dim=dim, angle_index=2,
    box="extras", spatial_dims=slice(0, 2),
)

# --- parametric kernels ---

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

# --- compose & dispatch ---
#
# Repulsion is isotropic (symmetric=True, the default).
# Alignment and pursuit are gated by the vision cone v, which makes
# them nonreciprocal → symmetric=False.

F_sim = (
    B_heading.to_psf(coeff_key="c0")                                     # self-propulsion
    + (k_repel * e_ij).dispatch_pairs(return_as="psf")                    # repulsion (isotropic)
    + (k_align * v * g_align).dispatch_pairs(symmetric=False,             # alignment (gated) ⊖
                                             return_as="psf")
    + (k_pursuit * v * e_j).dispatch_pairs(symmetric=False,               # pursuit   (gated) ⊖
                                           return_as="psf")
)

theta_exact = dict(c0=c0, eps=eps, R0=R0, A=A, La=La, P=P, Lp=Lp)

proc = OverdampedProcess(F_sim, D=D, extras_global={"box": box})
proc.set_params(theta_F=theta_exact)

print(f"Model parameters: {theta_exact}")


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION
# ═══════════════════════════════════════════════════════════════════════

key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0   = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0    = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

proc.initialize(x0)
key, sub = random.split(key)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=sub)

ds = coll.datasets[0]
Xfull = np.asarray(ds.X)
print(f"Trajectory: {Xfull.shape}")


# ═══════════════════════════════════════════════════════════════════════
#  SAVE / RELOAD TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════

csv_path = os.path.join(tempfile.gettempdir(), "sfi_abp_pursuit_demo.csv")
coll.save(csv_path, format="csv")
coll_reloaded = TrajectoryCollection.load(csv_path)
print(f"CSV round-trip error: "
      f"{float(np.max(np.abs(np.asarray(coll_reloaded.datasets[0].X) - Xfull))):.1e}")


# ═══════════════════════════════════════════════════════════════════════
#  OVER-COMPLETE INFERENCE BASIS
# ═══════════════════════════════════════════════════════════════════════
#
# Same geometric primitives — now multiplied by an over-complete
# dictionary of scalar radial kernels.  Vision-gated terms use
# symmetric=False because the gate breaks pair symmetry.

kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])

phi_r = scalar_pair_basis(kernels, dim=dim, box="extras", spatial_dims=slice(0, 2))

B_repel   = (phi_r * e_ij).dispatch_pairs(return_as="basis")                              # isotropic
B_align   = (phi_r * v * g_align).dispatch_pairs(symmetric=False, return_as="basis")      # gated ⊖
B_pursuit = (phi_r * v * e_j).dispatch_pairs(symmetric=False, return_as="basis")           # gated ⊖
B_full    = Basis.stack([B_heading, B_repel, B_align, B_pursuit])

n_heading = B_heading.n_features
n_repel   = B_repel.n_features
n_align   = B_align.n_features
n_pursuit = B_pursuit.n_features
print(f"Basis: {B_full.n_features} features "
      f"(heading={n_heading}, repel={n_repel}, align={n_align}, pursuit={n_pursuit})")


# ═══════════════════════════════════════════════════════════════════════
#  LINEAR FORCE INFERENCE
# ═══════════════════════════════════════════════════════════════════════

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()

nmse = float(inf.NMSE_force)
print(f"\nNMSE(force) = {nmse:.4f}")

coeffs = np.asarray(inf.force_coefficients)
i0 = 0
c_heading = coeffs[i0 : i0 + n_heading];  i0 += n_heading
c_repel   = coeffs[i0 : i0 + n_repel];    i0 += n_repel
c_align   = coeffs[i0 : i0 + n_align];    i0 += n_align
c_pursuit = coeffs[i0 : i0 + n_pursuit];  i0 += n_pursuit

print(f"Self-propulsion: true c0 = {c0:.2f}, inferred = {float(c_heading[0]):.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  BOOTSTRAP SIMULATION
# ═══════════════════════════════════════════════════════════════════════

key_boot = random.PRNGKey(seed + 77)
proc_boot = inf.simulate_bootstrapped_trajectory(key_boot, simulate=False)
proc_boot.set_extras(extras_global={"box": box})
proc_boot.initialize(x0)
key_boot, sub_boot = random.split(key_boot)
coll_boot = proc_boot.simulate(dt=dt, Nsteps=Nsteps, key=sub_boot)
Xboot_full = np.asarray(coll_boot.datasets[0].X)


# ═══════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════

# --- simulation movie ---

T_total = Xfull.shape[0]
skip = max(1, T_total // 200)
Xmov = Xfull[::skip]

fig_dm, ax_dm = _dark_fig(figsize=(4.5, 4.5))
sc_dm, qv_dm = _init_anim(ax_dm, Xmov[0], Lx, Ly)
ttl_dm = ax_dm.set_title("t = 0.0")

def _update_sim(frame):
    _update_anim(sc_dm, qv_dm, Xmov[frame], Lx, Ly)
    ttl_dm.set_text(f"t = {frame * skip * dt:.1f}")
    return sc_dm, qv_dm, ttl_dm

anim_sim = FuncAnimation(fig_dm, _update_sim,
                         frames=Xmov.shape[0], interval=40, blit=True)
anim_sim.save(_OUTDIR / "simulation.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {_OUTDIR / 'simulation.mp4'}")
plt.show()

# --- snapshot + trajectories ---

n_show = min(5, N_particles)
fig_snap, axes_snap = _dark_fig(1, 2, figsize=(12, 5.5))

_draw_particles(axes_snap[0], Xfull[-1], Lx, Ly)
axes_snap[0].set_title(f"Snapshot (t = {Nsteps * dt:.0f})")

for p in range(n_show):
    xy_w = _wrap_trajectory(Xfull[:, p, :2], np.array([Lx, Ly]))
    axes_snap[1].plot(xy_w[:, 0], xy_w[:, 1], lw=0.8, alpha=0.85)
axes_snap[1].set_xlim(0, Lx); axes_snap[1].set_ylim(0, Ly)
axes_snap[1].set_aspect("equal")
axes_snap[1].set_xlabel("x"); axes_snap[1].set_ylabel("y")
axes_snap[1].set_title(f"Trajectories ({n_show} particles)")

fig_snap.suptitle(f"Vision-gated ABPs  (N = {N_particles})", fontsize=13, color="white")
fig_snap.savefig(_OUTDIR / "snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'snapshot.png'}")
plt.show()

# --- force scatter ---

F_exact = proc.force_sf(coll.X, extras={"box": box})
F_inf   = inf.force_inferred(coll.X, extras={"box": box})

fig_sc, ax_sc = plt.subplots(figsize=(4.5, 4.5))
plotting.comparison_scatter(F_exact, F_inf, maxpoints=5000, alpha=0.05)
ax_sc.set_xlabel("Exact F"); ax_sc.set_ylabel("Inferred F")
ax_sc.set_title(f"Force scatter  (NMSE = {nmse:.3f})")
fig_sc.savefig(_OUTDIR / "force_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'force_scatter.png'}")
plt.show()

# --- kernel recovery (3 panels: repulsion, alignment, pursuit) ---

idx_repel_true   = 1   # r^0·exp(−r/1)   matches R0 = 1
idx_align_true   = 2   # r^0·exp(−r/2)   matches La = 2
idx_pursuit_true = 3   # r^0·exp(−r/4)   matches Lp = 4

true_c_repel   = np.zeros(n_repel);   true_c_repel[idx_repel_true]     = -eps
true_c_align   = np.zeros(n_align);   true_c_align[idx_align_true]     = A
true_c_pursuit = np.zeros(n_pursuit); true_c_pursuit[idx_pursuit_true] = P

r_eval = np.linspace(0.01, 8.0, 200)
r_jax  = jnp.array(r_eval)
phi_mat = np.array([np.asarray(fn(r_jax)) for fn, _ in kernels])

profiles = [
    ("Repulsion (isotropic)",   true_c_repel   @ phi_mat, c_repel   @ phi_mat),
    ("Alignment (gated)",      true_c_align   @ phi_mat, c_align   @ phi_mat),
    ("Pursuit (gated)",        true_c_pursuit @ phi_mat, c_pursuit @ phi_mat),
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

fig_kern.suptitle("Kernel recovery  (isotropic / vision-gated)", fontsize=13)
fig_kern.savefig(_OUTDIR / "kernels.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'kernels.png'}")
plt.show()

# --- bootstrap snapshot ---

fig_bs, axes_bs = _dark_fig(1, 2, figsize=(12, 5.5))
_draw_particles(axes_bs[0], Xfull[-1], Lx, Ly)
axes_bs[0].set_title("Original (final frame)")
_draw_particles(axes_bs[1], Xboot_full[-1], Lx, Ly)
axes_bs[1].set_title("Bootstrapped (inferred model)")
fig_bs.suptitle("Bootstrap validation", fontsize=13, color="white")
fig_bs.savefig(_OUTDIR / "bootstrap_snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTDIR / 'bootstrap_snapshot.png'}")
plt.show()

# --- bootstrap comparison movie ---

Xmov_orig = Xfull[::skip]
Xmov_boot = Xboot_full[::skip]
T_mov = min(Xmov_orig.shape[0], Xmov_boot.shape[0])

fig_bm, (ax_o, ax_b) = _dark_fig(1, 2, figsize=(12, 6))
ax_o.set_title("Original"); ax_b.set_title("Inferred model")
sc_o, qv_o = _init_anim(ax_o, Xmov_orig[0], Lx, Ly)
sc_b, qv_b = _init_anim(ax_b, Xmov_boot[0], Lx, Ly)
suptitle_bm = fig_bm.suptitle("t = 0.0", fontsize=13, color="white")

def _update_boot(frame):
    _update_anim(sc_o, qv_o, Xmov_orig[frame], Lx, Ly)
    _update_anim(sc_b, qv_b, Xmov_boot[frame], Lx, Ly)
    suptitle_bm.set_text(f"Bootstrap comparison  t = {frame * skip * dt:.1f}")
    return sc_o, qv_o, sc_b, qv_b, suptitle_bm

anim_boot = FuncAnimation(fig_bm, _update_boot, frames=T_mov, interval=40, blit=True)
anim_boot.save(_OUTDIR / "bootstrap_comparison.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {_OUTDIR / 'bootstrap_comparison.mp4'}")
plt.show()
