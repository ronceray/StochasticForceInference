# TODO: review this file
"""
ABP large-scale demo — Step 2: Inference
=========================================

Load the saved trajectory, run force inference, and save results.

Usage::

    MPLBACKEND=Agg python abp_largescale_infer.py
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"        # CPU-only — avoid GPU OOM
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Dropbox compat

import time
from glob import glob
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.langevin import OverdampedProcess
from SFI.trajectory import TrajectoryCollection
from SFI.utils.neighbors import build_neighbor_csr, make_neighbor_extras

from abp_largescale_config import (
    N_particles, Lx, Ly, Nsteps, dt, D, seed,
    box, box_np, cutoff, skin, rebuild_every,
    theta_exact, build_sim_force, build_inference_basis,
    TRAJ_DIR, BOOT_TRAJ_DIR, RESULTS_PATH, OUTDIR,
)


# ═══════════════════════════════════════════════════════════════════════
#  FAST H5 LOADER (bypasses slow YAML parsing)
# ═══════════════════════════════════════════════════════════════════════

def _load_X_fast(traj_dir, n_chunks=40):
    """Load X arrays directly from H5 files, bypassing slow YAML parsing.
    
    Only loads ds_000 through ds_{n_chunks-1} to ignore Dropbox ghost files.
    """
    chunks = []
    for i in range(n_chunks):
        fp = Path(traj_dir) / f"ds_{i:03d}.h5"
        with h5py.File(str(fp), "r") as f:
            t = f["table"]
            nrows = t["x0"].shape[0]
            tid = t["time_step"][:]
            ntimes = int(tid.max() - tid.min()) + 1
            npart = nrows // ntimes
            x0 = t["x0"][:].reshape(ntimes, npart)
            x1 = t["x1"][:].reshape(ntimes, npart)
            x2 = t["x2"][:].reshape(ntimes, npart)
            X = np.stack([x0, x1, x2], axis=-1)
        chunks.append(X)
    return chunks


# ═══════════════════════════════════════════════════════════════════════
#  NEIGHBOR DRIFT DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

def _neighbor_set(indptr, indices, i):
    """Return the set of neighbors for particle i."""
    return set(indices[indptr[i]:indptr[i+1]].tolist())


def check_neighbor_drift(Xfull, cutoff, box_np, rebuild_every, max_checks=10):
    """Compare neighbor lists at chunk boundaries to detect stale neighbors.

    Samples up to `max_checks` evenly-spaced chunks to avoid building
    all 2×n_chunks neighbor lists.
    """
    T = Xfull.shape[0]
    N = Xfull.shape[1]
    n_chunks = (T - 1) // rebuild_every
    # Sample a subset of chunks
    check_indices = np.linspace(0, n_chunks - 1, min(max_checks, n_chunks), dtype=int)

    stats = []
    for c in check_indices:
        t_start = c * rebuild_every
        t_end = min(t_start + rebuild_every, T - 1)
        if t_end <= t_start:
            continue

        pos_start = Xfull[t_start, :, :2]
        pos_end   = Xfull[t_end,   :, :2]

        iptr_s, idx_s = build_neighbor_csr(pos_start, cutoff, box_np)
        iptr_e, idx_e = build_neighbor_csr(pos_end,   cutoff, box_np)

        gained_list = []
        lost_list   = []
        for i in range(N):
            ns = _neighbor_set(iptr_s, idx_s, i)
            ne = _neighbor_set(iptr_e, idx_e, i)
            gained_list.append(len(ne - ns))
            lost_list.append(len(ns - ne))

        gained = np.array(gained_list)
        lost   = np.array(lost_list)
        nnz_start = len(idx_s)
        mean_nbrs = nnz_start / N

        stats.append(dict(
            chunk=int(c),
            t_start=t_start * dt,
            t_end=t_end * dt,
            mean_nbrs=mean_nbrs,
            mean_gained=gained.mean(),
            max_gained=gained.max(),
            mean_lost=lost.mean(),
            max_lost=lost.max(),
            frac_changed=(gained > 0).mean(),
        ))

    return stats


# ═══════════════════════════════════════════════════════════════════════
#  LOAD 4 MID-CHUNKS (not the first ones — they haven't fully relaxed)
# ═══════════════════════════════════════════════════════════════════════

INFER_CHUNKS = [18, 19, 20, 21]       # middle of the 40-chunk trajectory

print("Loading trajectory (fast H5) ...")
t0 = time.perf_counter()
all_chunks = _load_X_fast(TRAJ_DIR)
n_total = len(all_chunks)
X_chunks = [all_chunks[i] for i in INFER_CHUNKS]
print(f"  Selected chunks {INFER_CHUNKS} out of {n_total}  "
      f"({time.perf_counter() - t0:.1f}s)")

# Build per-chunk datasets with proper CSR extras
# Use cutoff + skin to match the simulation Verlet list
cutoff_list = cutoff + skin
print(f"Building per-chunk datasets with CSR extras (cutoff_list={cutoff_list}) ...")
t0 = time.perf_counter()
datasets = []
for ci, Xc in zip(INFER_CHUNKS, X_chunks):
    X0 = Xc[0]
    nbr = make_neighbor_extras(X0[:, :2], cutoff_list, box_np)
    eg = {"box": box}
    eg.update(nbr)
    ds = TrajectoryCollection.from_arrays(
        X=Xc, dt=dt, extras_global=eg,
    ).datasets[0]
    datasets.append(ds)
    print(f"    chunk {ci:2d}: nnz = {int(nbr['indptr'][-1])}")
coll = TrajectoryCollection(
    datasets=datasets,
    weights=jnp.ones(len(datasets), dtype=jnp.float32),
).with_weights("pool")
print(f"  {len(datasets)} datasets ready  ({time.perf_counter() - t0:.1f}s)")

# Also load full trajectory for neighbor drift check
Xfull = np.concatenate(all_chunks, axis=0)
del all_chunks   # free memory


# ═══════════════════════════════════════════════════════════════════════
#  NEIGHBOR DRIFT CHECK (full trajectory, sampled)
# ═══════════════════════════════════════════════════════════════════════

print("\nChecking neighbor drift ...")
t0 = time.perf_counter()
drift_stats = check_neighbor_drift(Xfull, cutoff, box_np, rebuild_every)
print(f"  done ({time.perf_counter() - t0:.1f}s)")
print(f"  {'chunk':>5} {'t':>8} {'<nbrs>':>7} {'<gain>':>7} {'max_g':>6} "
      f"{'<lost>':>7} {'max_l':>6} {'%changed':>9}")
for s in drift_stats:
    print(f"  {s['chunk']:5d} {s['t_end']:8.1f} {s['mean_nbrs']:7.1f} "
          f"{s['mean_gained']:7.2f} {s['max_gained']:6d} "
          f"{s['mean_lost']:7.2f} {s['max_lost']:6d} "
          f"{s['frac_changed']:9.1%}")

# Save drift stats
drift_arr = np.array([(s['chunk'], s['t_end'], s['mean_nbrs'],
                        s['mean_gained'], s['max_gained'],
                        s['mean_lost'], s['max_lost'],
                        s['frac_changed']) for s in drift_stats])
np.savetxt(OUTDIR / "neighbor_drift.csv", drift_arr,
           header="chunk t_end mean_nbrs mean_gained max_gained mean_lost max_lost frac_changed",
           fmt="%.4f")
print(f"  Saved {OUTDIR / 'neighbor_drift.csv'}")

del Xfull   # free memory before inference


# ═══════════════════════════════════════════════════════════════════════
#  BUILD BASIS & RUN INFERENCE
# ═══════════════════════════════════════════════════════════════════════

B_full, sizes, kernels = build_inference_basis()
print(f"\nBasis: {B_full.n_features} features  {sizes}")

# Force CPU for inference — large multi-dataset collections can
# trigger GPU memory / cuSolver handle issues.
cpu = jax.devices("cpu")[0]

print("Running inference ...")
t0 = time.perf_counter()
with jax.default_device(cpu):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B_full, M_mode="Ito", G_mode="rectangle")

    # ── compare to exact (requires rebuilding the simulation model) ──
    X0 = np.asarray(coll.datasets[0].X[0])
    nbr_extras = make_neighbor_extras(X0[:, :2], cutoff_list, box_np)
    extras0 = {"box": box}
    extras0.update(nbr_extras)

    F_sim = build_sim_force()
    proc = OverdampedProcess(F_sim, D=D, extras_global=extras0)
    proc.set_params(theta_F=theta_exact)
    proc.initialize(jnp.array(X0))

    inf.compare_to_exact(model_exact=proc, maxpoints=2000)
    inf.print_report()
    elapsed = time.perf_counter() - t0
    print(f"Inference done in {elapsed:.1f}s")

    nmse = float(inf.NMSE_force)
    print(f"NMSE(force) = {nmse:.4f}")

    coeffs = np.asarray(inf.force_coefficients)
    i0 = 0
    for name in ("heading", "repel", "align", "pursuit"):
        n = sizes[name]
        c = coeffs[i0 : i0 + n]
        i0 += n
        if name == "heading":
            print(f"Self-propulsion: true c0 = 1.00, inferred = {float(c[0]):.2f}")

    # ── save results ──
    inf.save_results(str(RESULTS_PATH))
    print(f"Inference results saved to {RESULTS_PATH}.*")


# ═══════════════════════════════════════════════════════════════════════
#  BOOTSTRAP SIMULATION (inferred model, single chunk)
# ═══════════════════════════════════════════════════════════════════════

key_boot = random.PRNGKey(seed + 77)

# Build the bootstrapped process manually (simulate_bootstrapped_trajectory
# doesn't support multi-dataset collections due to extras_global access).
proc_boot = OverdampedProcess(
    inf.force_inferred._psf, inf.diffusion_inferred._psf,
)
proc_boot.set_params(
    theta_F=inf.force_inferred.params,
    theta_D=inf.diffusion_inferred.params,
)

# Use same initial conditions as the original simulation
key_ic = random.PRNGKey(seed)
_, kx, kth = random.split(key_ic, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0   = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0    = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

# Build initial neighbor list for the bootstrap process
nbr_extras_boot = make_neighbor_extras(np.asarray(x0[:, :2]), cutoff_list, box_np)
extras_boot = {"box": box}
extras_boot.update(nbr_extras_boot)
proc_boot.set_extras(extras_global=extras_boot)
proc_boot.initialize(x0)

# Single chunk: 50 steps, neighbor list stays valid for short sim
boot_steps = 50
print(f"\nBootstrap simulation ({boot_steps} steps, single chunk) ...")
t0 = time.perf_counter()
key_boot, sub_boot = random.split(key_boot)
coll_boot = proc_boot.simulate(dt=dt, Nsteps=boot_steps, key=sub_boot)
elapsed = time.perf_counter() - t0
print(f"Bootstrap done in {elapsed:.1f}s")

coll_boot.save(str(BOOT_TRAJ_DIR), format="h5")
print(f"Bootstrap trajectory saved to {BOOT_TRAJ_DIR}/")
