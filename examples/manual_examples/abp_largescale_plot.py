# TODO: review this file
"""
ABP large-scale demo — Step 3: Figures
=======================================

Load saved trajectory + inference results and generate all plots.

Usage::

    MPLBACKEND=Agg python abp_largescale_plot.py
"""

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Dropbox compat

import json
from pathlib import Path
from glob import glob

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from abp_largescale_config import (
    N_particles, Lx, Ly,
    c0, eps, A, P, dt,
    build_inference_basis,
    OUTDIR, TRAJ_DIR, BOOT_TRAJ_DIR, RESULTS_PATH,
)


def _load_X_fast(traj_dir, n_chunks=None):
    """Load X arrays directly from H5 files, bypassing slow YAML parsing.
    
    If n_chunks is given, only loads ds_000 through ds_{n_chunks-1}
    to ignore Dropbox ghost files.
    """
    if n_chunks is not None:
        files = [str(Path(traj_dir) / f"ds_{i:03d}.h5") for i in range(n_chunks)]
    else:
        files = sorted(glob(str(Path(traj_dir) / "ds_*.h5")))
    chunks = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            t = f["table"]
            nrows = t["x0"].shape[0]
            tid = t["time_step"][:]
            ntimes = int(tid.max() - tid.min()) + 1
            npart = nrows // ntimes
            x0 = t["x0"][:].reshape(ntimes, npart)
            x1 = t["x1"][:].reshape(ntimes, npart)
            x2 = t["x2"][:].reshape(ntimes, npart)
            X = np.stack([x0, x1, x2], axis=-1)   # (T, N, 3)
        chunks.append(X)
    return np.concatenate(chunks, axis=0)


# ── load data ──
print("Loading trajectory (fast H5) ...")
Xfull = _load_X_fast(TRAJ_DIR, n_chunks=40)
print(f"  {Xfull.shape}")

print("Loading inference results ...")
res = np.load(str(RESULTS_PATH) + ".npz", allow_pickle=True)
with open(str(RESULTS_PATH) + ".json") as f:
    meta = json.load(f)
coeffs = res["force_coefficients"]
nmse   = meta.get("NMSE_force", float("nan"))
print(f"  NMSE = {nmse:.4f},  {len(coeffs)} coefficients")

_, sizes, kernels = build_inference_basis()
i0 = 0
c_heading = coeffs[i0 : i0 + sizes["heading"]]; i0 += sizes["heading"]
c_repel   = coeffs[i0 : i0 + sizes["repel"]];   i0 += sizes["repel"]
c_align   = coeffs[i0 : i0 + sizes["align"]];   i0 += sizes["align"]
c_pursuit = coeffs[i0 : i0 + sizes["pursuit"]];  i0 += sizes["pursuit"]


# ── helpers ──

def _wrap_xy(X, Lx, Ly):
    X = np.array(X, copy=True)
    X[:, 0] %= Lx
    X[:, 1] %= Ly
    return X


# ═══════════════════════════════════════════════════════════════════════
#  SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("black"); fig.patch.set_facecolor("black")
Xw = _wrap_xy(Xfull[-1], Lx, Ly)
ax.scatter(Xw[:, 0], Xw[:, 1], s=2,
           c=Xw[:, 2], cmap="hsv",
           vmin=-np.pi, vmax=np.pi, edgecolors="none")
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_aspect("equal")
ax.set_title(f"N = {N_particles}  (final frame)", color="white", fontsize=13)
ax.tick_params(colors="white")
fig.savefig(OUTDIR / "snapshot.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / 'snapshot.png'}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  ZOOM
# ═══════════════════════════════════════════════════════════════════════

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
ax_z.set_title("Zoom (30×30 patch)", color="white")
ax_z.tick_params(colors="white")
fig_z.savefig(OUTDIR / "zoom.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / 'zoom.png'}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  KERNEL RECOVERY
# ═══════════════════════════════════════════════════════════════════════

n_repel   = sizes["repel"]
n_align   = sizes["align"]
n_pursuit = sizes["pursuit"]

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
    ("Repulsion",          true_c_repel   @ phi_mat, c_repel   @ phi_mat),
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
fig_kern.savefig(OUTDIR / "kernels.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / 'kernels.png'}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION MOVIE (full frame, scatter only)
# ═══════════════════════════════════════════════════════════════════════

T_total = Xfull.shape[0]
skip = max(1, T_total // 200)
Xmov = Xfull[::skip]

fig_dm, ax_dm = plt.subplots(figsize=(8, 8))
fig_dm.patch.set_facecolor("black"); ax_dm.set_facecolor("black")
ax_dm.tick_params(colors="white")

Xw0 = _wrap_xy(Xmov[0], Lx, Ly)
sc_dm = ax_dm.scatter(Xw0[:, 0], Xw0[:, 1], s=2,
                      c=Xw0[:, 2], cmap="hsv",
                      vmin=-np.pi, vmax=np.pi, edgecolors="none")
ax_dm.set_xlim(0, Lx); ax_dm.set_ylim(0, Ly); ax_dm.set_aspect("equal")
ttl_dm = ax_dm.set_title("t = 0.0", color="white", fontsize=13)


def _update_sim(frame):
    Xw = _wrap_xy(Xmov[frame], Lx, Ly)
    sc_dm.set_offsets(Xw[:, :2])
    sc_dm.set_array(Xw[:, 2])
    ttl_dm.set_text(f"t = {frame * skip * dt:.1f}")
    return sc_dm, ttl_dm


anim_sim = FuncAnimation(fig_dm, _update_sim,
                         frames=Xmov.shape[0], interval=40, blit=True)
anim_sim.save(OUTDIR / "simulation.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {OUTDIR / 'simulation.mp4'}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  CLOSE-UP MOVIE (slowed down, arrows, ~100 particles)
# ═══════════════════════════════════════════════════════════════════════

# Pick a 30×30 region centered on a dense spot in the final frame
cx_zoom, cy_zoom = Lx / 2, Ly / 2
zoom_size = 30.0

# Use every 5th frame for close-up (still smooth, but manageable render time)
zoom_skip = 5
Xmov_zoom = Xfull[::zoom_skip]

fig_zm, ax_zm = plt.subplots(figsize=(8, 8))
fig_zm.patch.set_facecolor("black"); ax_zm.set_facecolor("black")
ax_zm.tick_params(colors="white")
ax_zm.set_xlim(cx_zoom - zoom_size / 2, cx_zoom + zoom_size / 2)
ax_zm.set_ylim(cy_zoom - zoom_size / 2, cy_zoom + zoom_size / 2)
ax_zm.set_aspect("equal")

_ARROW_LEN = 1.5

# Init with first frame
Xw_z0 = _wrap_xy(Xmov_zoom[0], Lx, Ly)
mask_z0 = (
    (np.abs(Xw_z0[:, 0] - cx_zoom) < zoom_size / 2) &
    (np.abs(Xw_z0[:, 1] - cy_zoom) < zoom_size / 2)
)
Xz0 = Xw_z0[mask_z0]
sc_zm = ax_zm.scatter(Xz0[:, 0], Xz0[:, 1], s=40,
                      c=Xz0[:, 2], cmap="hsv",
                      vmin=-np.pi, vmax=np.pi,
                      edgecolors="white", lw=0.3, zorder=3)
qv_zm = ax_zm.quiver(Xz0[:, 0], Xz0[:, 1],
                     _ARROW_LEN * np.cos(Xz0[:, 2]),
                     _ARROW_LEN * np.sin(Xz0[:, 2]),
                     scale=1.0, scale_units="xy", width=0.005,
                     color="white", alpha=0.7, zorder=4)
ttl_zm = ax_zm.set_title("Close-up  t = 0.0", color="white", fontsize=13)


def _update_zoom(frame):
    global sc_zm, qv_zm
    Xw = _wrap_xy(Xmov_zoom[frame], Lx, Ly)
    mask = (
        (np.abs(Xw[:, 0] - cx_zoom) < zoom_size / 2) &
        (np.abs(Xw[:, 1] - cy_zoom) < zoom_size / 2)
    )
    Xz = Xw[mask]
    # Remove old artists and redraw (particle count changes per frame)
    sc_zm.remove()
    qv_zm.remove()
    sc_zm = ax_zm.scatter(Xz[:, 0], Xz[:, 1], s=40,
                          c=Xz[:, 2], cmap="hsv",
                          vmin=-np.pi, vmax=np.pi,
                          edgecolors="white", lw=0.3, zorder=3)
    qv_zm = ax_zm.quiver(Xz[:, 0], Xz[:, 1],
                         _ARROW_LEN * np.cos(Xz[:, 2]),
                         _ARROW_LEN * np.sin(Xz[:, 2]),
                         scale=1.0, scale_units="xy", width=0.005,
                         color="white", alpha=0.7, zorder=4)
    ttl_zm.set_text(f"Close-up  t = {frame * zoom_skip * dt:.2f}")
    return sc_zm, qv_zm, ttl_zm


anim_zoom = FuncAnimation(fig_zm, _update_zoom,
                          frames=Xmov_zoom.shape[0], interval=80, blit=False)
anim_zoom.save(OUTDIR / "closeup.mp4", writer="ffmpeg", dpi=150)
print(f"Saved {OUTDIR / 'closeup.mp4'}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  BOOTSTRAP COMPARISON MOVIE
# ═══════════════════════════════════════════════════════════════════════

if BOOT_TRAJ_DIR.exists():
    print("Loading bootstrap trajectory (fast H5) ...")
    Xboot = _load_X_fast(BOOT_TRAJ_DIR)
    print(f"  {Xboot.shape}")

    Xmov_boot = Xboot[::skip]
    T_mov = min(Xmov.shape[0], Xmov_boot.shape[0])

    fig_bm, (ax_o, ax_b) = plt.subplots(1, 2, figsize=(16, 8))
    fig_bm.patch.set_facecolor("black")
    for ax in (ax_o, ax_b):
        ax.set_facecolor("black"); ax.tick_params(colors="white")
    ax_o.set_title("Original", color="white", fontsize=13)
    ax_b.set_title("Inferred model", color="white", fontsize=13)

    Xw_o0 = _wrap_xy(Xmov[0], Lx, Ly)
    Xw_b0 = _wrap_xy(Xmov_boot[0], Lx, Ly)
    sc_o = ax_o.scatter(Xw_o0[:, 0], Xw_o0[:, 1], s=2,
                        c=Xw_o0[:, 2], cmap="hsv",
                        vmin=-np.pi, vmax=np.pi, edgecolors="none")
    sc_b = ax_b.scatter(Xw_b0[:, 0], Xw_b0[:, 1], s=2,
                        c=Xw_b0[:, 2], cmap="hsv",
                        vmin=-np.pi, vmax=np.pi, edgecolors="none")
    for ax in (ax_o, ax_b):
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_aspect("equal")
    suptitle_bm = fig_bm.suptitle("t = 0.0", fontsize=13, color="white")

    def _update_boot(frame):
        Xw_o = _wrap_xy(Xmov[frame], Lx, Ly)
        sc_o.set_offsets(Xw_o[:, :2]); sc_o.set_array(Xw_o[:, 2])
        Xw_b = _wrap_xy(Xmov_boot[frame], Lx, Ly)
        sc_b.set_offsets(Xw_b[:, :2]); sc_b.set_array(Xw_b[:, 2])
        suptitle_bm.set_text(
            f"Bootstrap comparison  t = {frame * skip * dt:.1f}")
        return sc_o, sc_b, suptitle_bm

    anim_boot = FuncAnimation(fig_bm, _update_boot,
                              frames=T_mov, interval=40, blit=True)
    anim_boot.save(OUTDIR / "bootstrap_comparison.mp4",
                   writer="ffmpeg", dpi=150)
    print(f"Saved {OUTDIR / 'bootstrap_comparison.mp4'}")
    plt.show()
else:
    print(f"No bootstrap trajectory found at {BOOT_TRAJ_DIR}; skipping movie.")

print(f"\nAll outputs in {OUTDIR}")
