# TODO: review this file
"""
ABP (Active Brownian Particle) model building blocks — example-local.

Composed entirely from the geometric primitives in :mod:`SFI.bases.pairs`
(``heading_vector``, ``pair_direction``, ``angle_coupling``,
``parametric_radial_kernel``) using the ``*`` / ``+`` Interactor algebra.
No call to :func:`~SFI.statefunc.make_interactor`: the full ABP force is
a sum of products of off-the-shelf bricks.

These helpers are example-specific and not part of the public ``SFI``
API.  They previously lived in ``SFI.bases.abp_align`` /
``SFI.bases.abp_blocks``.
"""
from __future__ import annotations

import jax.numpy as jnp

from SFI.bases.pairs import (
    angle_coupling,
    heading_vector,
    pair_direction,
    parametric_radial_kernel,
)
from SFI.statefunc import PSF, Basis, Interactor


# ---- single-particle brick --------------------------------------------------

def etheta_vec3(*, dim: int = 3, theta_index: int = 2) -> Basis:
    """Unit heading vector ``(cos θ, sin θ, 0)`` embedded in *dim*-d space."""
    return heading_vector(dim=dim, angle_index=theta_index, spatial_axes=(0, 1))


# ---- pair interactor bricks -------------------------------------------------

def make_repulsion_local(*, dim: int = 3) -> Interactor:
    r"""Local 2-body repulsion: :math:`-\varepsilon/(1+(r/R_0)^2)\,\hat r_{xy}`.

    Composed as ``k_repel * e_ij`` where ``k_repel`` is a rank-0
    parametric radial kernel and ``e_ij`` is the xy unit displacement
    embedded in the 3-D (x, y, θ) state.  PBC box read from ``extras``.
    """
    k_repel = parametric_radial_kernel(
        lambda r, p: -p["eps"] / (1.0 + (r / p["R0"]) ** 2),
        params={"eps": (), "R0": ()},
        dim=dim,
        box="extras",
        spatial_dims=slice(0, 2),
    )
    e_ij = pair_direction(
        dim=dim,
        box="extras",
        spatial_dims=slice(0, 2),
        embed_dim=dim,
        embed_axes=[0, 1],
    )
    return k_repel * e_ij


def make_alignment_local(*, dim: int = 3) -> Interactor:
    r"""Local 2-body alignment torque: :math:`A\,e^{-r/L_0}\,\sin(\theta_j-\theta_i)\,\hat z`.

    Composed as ``k_align * g_align`` where ``k_align`` is a rank-0
    parametric radial kernel and ``g_align`` is ``sin(Δθ)`` placed on
    the θ-axis (index 2).
    """
    k_align = parametric_radial_kernel(
        lambda r, p: p["A"] * jnp.exp(-r / p["L0"]),
        params={"A": (), "L0": ()},
        dim=dim,
        box="extras",
        spatial_dims=slice(0, 2),
    )
    g_align = angle_coupling(jnp.sin, dim=dim, angle_index=2, output_index=2)
    return k_align * g_align


def make_abp_align_psf(*, dim: int = 3) -> PSF:
    """Full ABP force (active thrust + repulsion + alignment) as a PSF."""
    F_active = etheta_vec3(dim=dim).to_psf(coeff_key="c0")

    inter_local = make_repulsion_local(dim=dim) + make_alignment_local(dim=dim)
    F_pairs = inter_local.dispatch_pairs(
        symmetric=True,
        exclude_self=True,
        owners="focal",
        reducer="sum",
        normalize_by_degree=False,
        return_as="psf",
    )

    return F_active + F_pairs


__all__ = [
    "etheta_vec3",
    "make_repulsion_local",
    "make_alignment_local",
    "make_abp_align_psf",
]
