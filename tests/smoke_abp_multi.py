#!/usr/bin/env python
# TODO: review this file
"""Quick smoke test: one cell of the multi-particle ABP benchmark."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

print("=== Smoke test: multi-particle ABP Parametric SFI ===")

# -- build model --
from SFI.bases.pairs import (
    angle_coupling, heading_vector, pair_direction, parametric_radial_kernel,
)
from SFI.langevin import OverdampedProcess

DIM = 3
N = 10  # 10 interacting particles
LX, LY = 10.0, 10.0
D_ISO = 0.05
box = jnp.array([LX, LY])
theta_F = dict(c0=1.0, eps=2.0, A=0.5, R0=1.0, L0=2.0)

B_heading = heading_vector(dim=DIM, angle_index=2)
e_ij = pair_direction(dim=DIM, box="extras", spatial_dims=slice(0,2),
                      embed_dim=DIM, embed_axes=[0,1])
g_align = angle_coupling(jnp.sin, dim=DIM, angle_index=2)
k_repel = parametric_radial_kernel(
    lambda r, p: -p["eps"]*jnp.exp(-r/p["R0"]),
    params={"eps":(),"R0":()}, dim=DIM, box="extras", spatial_dims=slice(0,2))
k_align = parametric_radial_kernel(
    lambda r, p: p["A"]*jnp.exp(-r/p["L0"]),
    params={"A":(),"L0":()}, dim=DIM, box="extras", spatial_dims=slice(0,2))
F_psf = (B_heading.to_psf(coeff_key="c0")
         + (k_repel * e_ij).dispatch_pairs(return_as="psf")
         + (k_align * g_align).dispatch_pairs(return_as="psf"))

proc = OverdampedProcess(F_psf, D=D_ISO, extras_global={"box": box})
proc.set_params(theta_F=theta_F)

# -- simulate --
key = random.PRNGKey(42)
key, kx, kth = random.split(key, 3)
X0_xy = random.uniform(kx, (N, 2)) * jnp.array([LX, LY])
TH0 = random.uniform(kth, (N,), minval=-jnp.pi, maxval=jnp.pi)
x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)
proc.initialize(x0)
key, sub = random.split(key)

print("Simulating N=%d particles..." % N)
coll = proc.simulate(dt=0.01, Nsteps=500, key=sub, prerun=50, oversampling=5)
ds = coll.datasets[0]
print(f"  Trajectory: {ds.X.shape}")  # expect (T, N, DIM)

# -- test PSF call with (N,d) input --
X_test = ds.X[0]  # (N, DIM)
forces = F_psf(X_test, params=theta_F, extras={"box": box})
print(f"  PSF call: input {X_test.shape} → output {forces.shape}")
assert forces.shape == (N, DIM), f"Expected ({N},{DIM}), got {forces.shape}"

# -- test Parametric SFI inference (the multi-particle path) --
print("Testing Parametric SFI inference (multi-particle, full N-body)...")
from SFI.inference.overdamped import OverdampedLangevinInference

theta0_F = {k: 0.5 * jnp.ones_like(jnp.asarray(v)) for k, v in theta_F.items()}
inf2 = OverdampedLangevinInference(coll)
inf2.infer_force(
    F_psf, theta0_F,
    n_substeps=4,
    max_outer=3,
)
print(f"  Parametric SFI done: params_F = {inf2.force_inferred.params}")
print("=== SMOKE TEST PASSED ===")
