# TODO: review this file
"""
ABP large-scale demo — Step 1: Simulation
==========================================

Run the chunked simulation with periodic neighbor-list rebuilds.
Saves the trajectory to disk via ``coll.save(..., format="h5")``.

Usage::

    MPLBACKEND=Agg python abp_largescale_simulate.py
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI.langevin import OverdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.utils.neighbors import make_neighbor_extras

from abp_largescale_config import (
    N_particles, Lx, Ly, Nsteps, dt, D, seed,
    box, box_np, cutoff, skin, rebuild_every,
    theta_exact, build_sim_force, TRAJ_DIR,
)

print(f"N = {N_particles},  box = {Lx:.1f} × {Ly:.1f},  "
      f"cutoff = {cutoff},  skin = {skin},  rebuild_every = {rebuild_every}")

# ── force model ──
F_sim = build_sim_force()

# ── initial conditions ──
key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N_particles, 2)) * jnp.array([Lx, Ly])
TH0   = random.uniform(kth, (N_particles,), minval=-jnp.pi, maxval=jnp.pi)
x0    = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

# ── initial neighbor list (cutoff + skin for Verlet buffer) ──
cutoff_list = cutoff + skin
print("Building initial neighbor list ...")
t0 = time.perf_counter()
nbr_extras = make_neighbor_extras(np.asarray(x0[:, :2]), cutoff_list, box_np)
nnz0 = len(nbr_extras["indices"])
print(f"  nnz = {nnz0},  mean neighbors/particle = {nnz0 / N_particles:.1f}  "
      f"({time.perf_counter() - t0:.2f}s)")

extras0 = {"box": box}
extras0.update(nbr_extras)

proc = OverdampedProcess(F_sim, D=D, extras_global=extras0)
proc.set_params(theta_F=theta_exact)
proc.initialize(x0)

# ── chunked simulation ──
print(f"\nSimulating {Nsteps} steps (chunks of {rebuild_every}) ...")
t0 = time.perf_counter()
key, sub = random.split(key)
coll = simulate_chunked(
    proc, dt=dt, Nsteps=Nsteps, key=sub,
    cutoff=cutoff, box=box_np,
    skin=skin,
    rebuild_every=rebuild_every,
    save_every=50,
    spatial_dims=slice(0, 2),
    nnz_safety=3.0,
    verbose=True,
)
elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s  ({len(coll.datasets)} chunks)")

# ── save trajectory ──
coll.save(str(TRAJ_DIR), format="h5")
print(f"Trajectory saved to {TRAJ_DIR}/")
