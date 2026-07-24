#!/usr/bin/env python
# TODO: review this file
"""
Correctness test for the efficient _same_particle_jacobian protocol.

Compares:
  (A) result from the new protocol (per-edge jacfwd + scatter for dispatchers,
      chain rule via JVP for composites)
  (B) reference via manual per-particle jacfwd (guaranteed correct)

Tests:
  1. Pure interaction dispatcher (pair interaction)
  2. Interaction + single-particle composite (MapNNode addition)
  3. Batched input (leading batch axis)
"""
from __future__ import annotations
import time
import pytest
import jax
_PREV_X64 = jax.config.jax_enable_x64
jax.config.update("jax_enable_x64", True)  # script-style file: runs at import; restored at EOF
import jax.numpy as jnp
from jax import random


# ── Build test system ─────────────────────────────────────────────
N = 10
dim = 3
seed = 42

from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _abp_helpers import etheta_vec3, make_alignment_local, make_repulsion_local  # noqa: E402

# Active self-propulsion (single-particle term)
B_heading_psf = etheta_vec3(dim=dim)
F_active = B_heading_psf.to_psf(coeff_key="c0")

# Pair interactions
repel_loc = make_repulsion_local(dim=dim)
align_loc = make_alignment_local(dim=dim)
inter_local = repel_loc + align_loc
F_pairs = inter_local.dispatch_pairs(
    symmetric=True, exclude_self=True,
    owners="focal", reducer="sum",
    return_as="psf",
)

# Full force = interaction + single-particle
F_full = F_active + F_pairs

# ── Set up test data ──────────────────────────────────────────────
key = random.PRNGKey(seed)
key, kx = random.split(key)
Lx, Ly = 10.0, 10.0
box = jnp.array([Lx, Ly])

X0_xy = random.uniform(kx, (N, 2)) * box
key, kth = random.split(key)
TH0 = random.uniform(kth, (N,), minval=-jnp.pi, maxval=jnp.pi)
x = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)  # (N, 3)

theta_exact = dict(c0=1.0, eps=2.0, A=0.5, R0=1.0, L0=2.0)
# Build params dict with jnp arrays
params = {k: jnp.asarray(v) for k, v in theta_exact.items()}
extras = {"box": box}


# ── Reference: per-particle jacfwd (guaranteed correct) ───────────



def reference_spj(expr, x, *, params, extras):
    """Manual same-particle Jacobian via per-particle jacfwd.

    expr should be the full PSF/StateExpr (not the raw root node),
    so that params are coerced correctly.
    """
    P = x.shape[0]

    def f_p(i):
        xp = x[i]
        def f_one(xp_):
            x_mod = x.at[i].set(xp_)
            return expr(x_mod, params=params, extras=extras)[i]
        J = jax.jacfwd(f_one)(xp)
        return J

    J_all = jax.vmap(f_p)(jnp.arange(P, dtype=jnp.int32))
    return jnp.moveaxis(J_all, -1, 1)


# ==================================================================
# Test 1: Interaction dispatcher alone
# ==================================================================
print("Test 1: Interaction dispatcher (F_pairs) ...")
dFdx_pairs = F_pairs.d_x(same_particle=True)

# New protocol path
t0 = time.perf_counter()
J_new = jax.jit(lambda x: dFdx_pairs(x, params=params, extras=extras))(x)
J_new.block_until_ready()
t_jit = time.perf_counter() - t0

t0 = time.perf_counter()
J_new2 = jax.jit(lambda x: dFdx_pairs(x, params=params, extras=extras))(x)
J_new2.block_until_ready()
t_exec = time.perf_counter() - t0

# Reference
J_ref = jax.jit(lambda x: reference_spj(F_pairs, x, params=params, extras=extras))(x)
J_ref.block_until_ready()

# Normalize: derivative expr may drop trailing F=1 that the reference keeps
if J_ref.ndim > J_new.ndim and J_ref.shape[-1] == 1:
    J_ref = J_ref[..., 0]
err = float(jnp.max(jnp.abs(J_new - J_ref)))
rel = float(jnp.linalg.norm(J_new - J_ref) / jnp.maximum(1e-30, jnp.linalg.norm(J_ref)))
print(f"  Shape: {J_new.shape}")
print(f"  Max abs error: {err:.2e}")
print(f"  Relative error: {rel:.2e}")
print(f"  JIT: {t_jit:.3f}s, exec: {t_exec:.4f}s")
assert err < 1e-10, f"Test 1 FAILED: max abs error {err:.2e}"
print("  PASSED ✓")
print()


# ==================================================================
# Test 2: Full composite (interaction + single-particle)
# ==================================================================
print("Test 2: Full composite (F_full = F_pairs + F_active) ...")
dFdx_full = F_full.d_x(same_particle=True)

t0 = time.perf_counter()
J_new = jax.jit(lambda x: dFdx_full(x, params=params, extras=extras))(x)
J_new.block_until_ready()
t_jit = time.perf_counter() - t0

t0 = time.perf_counter()
J_new2 = jax.jit(lambda x: dFdx_full(x, params=params, extras=extras))(x)
J_new2.block_until_ready()
t_exec = time.perf_counter() - t0

J_ref = jax.jit(lambda x: reference_spj(F_full, x, params=params, extras=extras))(x)
J_ref.block_until_ready()

if J_ref.ndim > J_new.ndim and J_ref.shape[-1] == 1:
    J_ref = J_ref[..., 0]
err = float(jnp.max(jnp.abs(J_new - J_ref)))
rel = float(jnp.linalg.norm(J_new - J_ref) / jnp.maximum(1e-30, jnp.linalg.norm(J_ref)))
print(f"  Shape: {J_new.shape}")
print(f"  Max abs error: {err:.2e}")
print(f"  Relative error: {rel:.2e}")
print(f"  JIT: {t_jit:.3f}s, exec: {t_exec:.4f}s")
assert err < 1e-10, f"Test 2 FAILED: max abs error {err:.2e}"
print("  PASSED ✓")
print()


# ==================================================================
# Test 3: Batched input
# ==================================================================
print("Test 3: Batched input (3 samples) ...")
key, k2 = random.split(key)
x_batch = x[None, :, :] + 0.01 * random.normal(k2, (3, N, dim))

J_batch = jax.jit(lambda xb: dFdx_full(xb, params=params, extras=extras))(x_batch)
J_batch.block_until_ready()

# Compare each sample in the batch against single-sample reference
max_err = 0.0
for b in range(3):
    J_b = jax.jit(lambda xb: dFdx_full(xb, params=params, extras=extras))(x_batch[b])
    J_b.block_until_ready()
    err_b = float(jnp.max(jnp.abs(J_batch[b] - J_b)))
    max_err = max(max_err, err_b)
print(f"  Batch shape: {J_batch.shape}")
print(f"  Max abs error (batch vs single): {max_err:.2e}")
assert max_err < 1e-12, f"Test 3 FAILED: max error {max_err:.2e}"
print("  PASSED ✓")
print()


# ==================================================================
# Test 4: Scaling comparison
# ==================================================================
print("Test 4: Scaling — timing new vs old at N=10,20 ...")

from SFI.statefunc.nodes.ops.derivative import _same_particle_grad, _move_deriv_axis

for N_test in [10, 20]:
    key, kx2 = random.split(key)
    x_test = random.uniform(kx2, (N_test, dim)) * jnp.array([Lx, Ly, 2*jnp.pi])
    x_test = x_test.at[:, 2].add(-jnp.pi)

    # New protocol (via d_x)
    fn_new = jax.jit(lambda xt: dFdx_full(xt, params=params, extras=extras))
    J_n = fn_new(x_test)
    J_n.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(3):
        J_n = fn_new(x_test)
        J_n.block_until_ready()
    t_new = (time.perf_counter() - t0) / 3

    # Old generic path (for comparison)
    child = F_full.root
    coerced_params = F_full.template.coerce(params)
    fn_old = jax.jit(
        lambda xt: _move_deriv_axis(
            _same_particle_grad(
                child, jax.jacfwd, xt, None, coerced_params,
                var="x", mask=None, extras=extras
            ),
            rank=int(child.rank), pdepth_old=int(child.pdepth),
            particles_input=True, same_particle=True,
        )
    )
    J_o = fn_old(x_test)
    J_o.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(3):
        J_o = fn_old(x_test)
        J_o.block_until_ready()
    t_old = (time.perf_counter() - t0) / 3

    # Drop trailing F=1 from old path to compare
    if J_o.ndim > J_n.ndim and J_o.shape[-1] == 1:
        J_o = J_o[..., 0]
    err = float(jnp.max(jnp.abs(J_n - J_o)))
    print(f"  N={N_test:3d}: new={t_new:.4f}s  old={t_old:.4f}s  "
          f"speedup={t_old/max(t_new,1e-9):.1f}x  err={err:.2e}")

print()
print("All tests passed.")

# Restore the global precision flag for the rest of the pytest session.
jax.config.update("jax_enable_x64", _PREV_X64)
