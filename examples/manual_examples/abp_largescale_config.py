# TODO: review this file
"""
ABP large-scale demo — shared configuration
=============================================

Physical parameters, force model definition, and helper functions
shared by the simulation, inference, and plotting scripts.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

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

# ── paths ──
OUTDIR = Path(__file__).resolve().parent / "abp_largescale_output"
OUTDIR.mkdir(exist_ok=True)
TRAJ_DIR      = OUTDIR / "trajectory"            # per-chunk HDF5 files
BOOT_TRAJ_DIR = OUTDIR / "bootstrap_trajectory"  # bootstrap per-chunk HDF5
RESULTS_PATH  = OUTDIR / "inference"              # → inference.npz + inference.json

# ── physical parameters ──
N_particles   = 10_000
density       = 60 / (30.0 * 30.0)
area          = N_particles / density
Lx = Ly       = float(np.sqrt(area))  # ≈ 387
Nsteps        = 2000
dt            = 0.02
D             = 0.05
seed          = 0
box_np        = np.array([Lx, Ly])
box           = jnp.array([Lx, Ly])

cutoff        = 12.0
skin          = 0.0
rebuild_every = 1

# true force parameters
c0  = 1.0;  eps = 2.0;  A = 0.3;  P = 1.5
R0  = 1.0;  La  = 2.0;  Lp = 4.0
theta_exact = dict(c0=c0, eps=eps, R0=R0, A=A, La=La, P=P, Lp=Lp)

# ── geometric primitives ──
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

# ── parametric kernels (simulation model) ──
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

# CSR dispatch keys
csr_kw = dict(indptr_key="indptr", indices_key="indices")


def build_sim_force():
    """Return the parametric simulation force (PSF sum)."""
    return (
        B_heading.to_psf(coeff_key="c0")
        + (k_repel * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
        + (k_align * v * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
        + (k_pursuit * v * e_j).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    )


def build_inference_basis():
    """Return (B_full, section_sizes) for the overcomplete inference basis."""
    kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
    phi_r   = scalar_pair_basis(kernels, dim=dim, box="extras", spatial_dims=slice(0, 2))

    B_repel   = (phi_r * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
    B_align   = (phi_r * v * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
    B_pursuit = (phi_r * v * e_j).dispatch_pairs_from_extras(**csr_kw, return_as="basis")
    B_full    = Basis.stack([B_heading, B_repel, B_align, B_pursuit])

    sizes = dict(
        heading=B_heading.n_features,
        repel=B_repel.n_features,
        align=B_align.n_features,
        pursuit=B_pursuit.n_features,
    )
    return B_full, sizes, kernels
