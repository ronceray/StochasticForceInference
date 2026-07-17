# TODO: review this file
"""
Particle → grid coarse-graining (example-local helper).

Bilinear deposition of particle quantities onto a periodic 2-D Cartesian
grid, followed by a wrapped Gaussian convolution. Used by the
``abp_to_spde_demo`` to obtain smooth (ρ, m) hydrodynamic fields from
agent-based simulations.

This helper is example-specific and not part of the public ``SFI`` API.
"""
from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("grid_shape",))
def _bilinear_deposit(
    positions: jnp.ndarray,        # (N, 2), wrapped to [0, L) per axis
    weights: jnp.ndarray,          # (N, K)  — quantities to deposit per particle
    box: jnp.ndarray,              # (2,)
    grid_shape: tuple[int, int],
) -> jnp.ndarray:
    """Bilinear deposition with periodic boundary conditions.

    Returns array of shape ``(*grid_shape, K)``.
    """
    Nx, Ny = grid_shape
    Lx, Ly = box[0], box[1]
    dx = Lx / Nx
    dy = Ly / Ny

    # Cell-centred fractional indices
    fx = (positions[:, 0] / dx) - 0.5
    fy = (positions[:, 1] / dy) - 0.5

    i0 = jnp.floor(fx).astype(jnp.int32)
    j0 = jnp.floor(fy).astype(jnp.int32)
    tx = fx - i0
    ty = fy - j0

    i1 = i0 + 1
    j1 = j0 + 1

    # Periodic wrap of indices
    i0 = jnp.mod(i0, Nx); i1 = jnp.mod(i1, Nx)
    j0 = jnp.mod(j0, Ny); j1 = jnp.mod(j1, Ny)

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    K = weights.shape[1]
    field = jnp.zeros((Nx, Ny, K), dtype=weights.dtype)

    field = field.at[i0, j0, :].add(w00[:, None] * weights)
    field = field.at[i1, j0, :].add(w10[:, None] * weights)
    field = field.at[i0, j1, :].add(w01[:, None] * weights)
    field = field.at[i1, j1, :].add(w11[:, None] * weights)
    return field


def _gaussian_kernel_1d(sigma_cells: float, half_width: int) -> jnp.ndarray:
    """Static-shape symmetric 1-D Gaussian kernel, normalised to sum=1."""
    import numpy as np
    if half_width <= 0 or sigma_cells <= 0.0:
        return jnp.array([1.0])
    k = np.arange(-half_width, half_width + 1, dtype=np.float64)
    g = np.exp(-0.5 * (k / float(sigma_cells)) ** 2)
    g = g / g.sum()
    return jnp.asarray(g)


@partial(jax.jit, static_argnames=("half_width",))
def _smooth_2d_pbc(
    field: jnp.ndarray,        # (Nx, Ny, K)
    g: jnp.ndarray,            # (2*half_width+1,) — pre-computed kernel
    half_width: int,
) -> jnp.ndarray:
    """Periodic separable Gaussian convolution along the two grid axes."""
    out = jnp.zeros_like(field)
    for s in range(2 * half_width + 1):
        out = out + g[s] * jnp.roll(field, s - half_width, axis=0)
    field_x = out
    out = jnp.zeros_like(field)
    for s in range(2 * half_width + 1):
        out = out + g[s] * jnp.roll(field_x, s - half_width, axis=1)
    return out


def coarse_grain_polar(
    X: jnp.ndarray,            # (T, N, 3) with (x, y, θ)
    *,
    box: jnp.ndarray,          # (2,)
    grid_shape: tuple[int, int],
    sigma_cells: float = 1.5,
    angle_index: int = 2,
) -> jnp.ndarray:
    """Coarse-grain ABP positions+orientations to (ρ, m_x, m_y) on a grid.

    Per-frame bilinear deposition of particle mass and polar moment onto
    a periodic ``grid_shape`` grid covering ``box``, followed by a
    wrapped Gaussian convolution with width ``sigma_cells`` (in grid
    cells). The deposited fields are *densities* (per unit area):
    integrating over the box returns the total particle count and the
    total polar moment.

    Returns
    -------
    fields : (T, Nx*Ny, 3) array
        Channel layout: ``[ρ, m_x, m_y]`` flattened in C order over
        the grid, ready for ``TrajectoryCollection.from_arrays``.
    """
    Nx, Ny = grid_shape
    Lx, Ly = float(box[0]), float(box[1])
    dx = Lx / Nx
    dy = Ly / Ny
    cell_area = dx * dy

    half_width = max(1, int(math.ceil(3.0 * sigma_cells)))
    kernel = _gaussian_kernel_1d(sigma_cells, half_width)

    @jax.jit
    def _one_frame(Xt: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.stack([jnp.mod(Xt[:, 0], Lx), jnp.mod(Xt[:, 1], Ly)], axis=1)
        theta = Xt[:, angle_index]
        ones = jnp.ones_like(theta)
        weights = jnp.stack([ones, jnp.cos(theta), jnp.sin(theta)], axis=1)
        deposited = _bilinear_deposit(xy, weights, box, grid_shape) / cell_area
        smoothed = _smooth_2d_pbc(deposited, kernel, half_width)
        return smoothed.reshape(Nx * Ny, 3)

    # Sequential per-frame loop with JIT cache reuse — keeps memory low
    # and avoids a huge vmap trace on long trajectories.
    frames_out = []
    for t in range(X.shape[0]):
        frames_out.append(_one_frame(X[t]))
    fields = jnp.stack(frames_out, axis=0)
    return fields


__all__ = ["coarse_grain_polar", "coarse_grain_nematic"]


def coarse_grain_nematic(
    X: jnp.ndarray,            # (T, N, ≥3) with (x, y, θ, ...)
    *,
    box: jnp.ndarray,
    grid_shape: tuple[int, int],
    sigma_cells: float = 1.5,
    angle_index: int = 2,
    include_polar: bool = True,
) -> jnp.ndarray:
    r"""Coarse-grain rod-shaped agents to (ρ, m_x, m_y, Q_xx, Q_xy) on a grid.

    Per-frame bilinear deposition of particle mass, polar moment
    :math:`(\cos\theta_i, \sin\theta_i)`, and nematic order tensor
    :math:`(\cos 2\theta_i, \sin 2\theta_i)` — the two independent
    components of the traceless 2×2 director outer product
    :math:`Q_{ij} = 2\langle n_i n_j\rangle - \delta_{ij}` with
    :math:`\mathbf n = (\cos\theta,\sin\theta)`. Followed by a periodic
    Gaussian convolution of width ``sigma_cells``.

    The deposited fields are *densities* per unit area: integrating over
    the box returns the corresponding total moments. ``Q_{xx}`` and
    ``Q_{xy}`` are *Q-densities* (i.e. ``ρ * local_mean_Q``), which is
    the natural quantity for hydrodynamic SPDE inference.

    Parameters
    ----------
    X : (T, N, ≥3) array
        Trajectory. The angle coordinate is at ``angle_index``.
    box : (2,) array
        Periodic-box dimensions (Lx, Ly).
    grid_shape : (Nx, Ny)
        Cartesian grid shape.
    sigma_cells : float, default 1.5
        Gaussian smoothing width in grid-cell units.
    angle_index : int, default 2
        Column index of the angle in ``X``.
    include_polar : bool, default True
        If True, return both polar momentum ``m`` and nematic ``Q``
        (5 channels: ρ, m_x, m_y, Q_xx, Q_xy). If False, return only
        density and nematic order (3 channels: ρ, Q_xx, Q_xy).

    Returns
    -------
    fields : (T, Nx*Ny, K) array
        Channel layout: ``[ρ, m_x, m_y, Q_xx, Q_xy]`` (K=5) or
        ``[ρ, Q_xx, Q_xy]`` (K=3), flattened in C order over the grid.
    """
    Nx, Ny = grid_shape
    Lx, Ly = float(box[0]), float(box[1])
    dx = Lx / Nx
    dy = Ly / Ny
    cell_area = dx * dy

    half_width = max(1, int(math.ceil(3.0 * sigma_cells)))
    kernel = _gaussian_kernel_1d(sigma_cells, half_width)

    K = 5 if include_polar else 3

    @jax.jit
    def _one_frame(Xt: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.stack([jnp.mod(Xt[:, 0], Lx), jnp.mod(Xt[:, 1], Ly)], axis=1)
        theta = Xt[:, angle_index]
        ones = jnp.ones_like(theta)
        c1, s1 = jnp.cos(theta), jnp.sin(theta)
        # Q components: cos(2θ) = 2cos²θ - 1, sin(2θ) = 2 sinθ cosθ
        c2, s2 = jnp.cos(2.0 * theta), jnp.sin(2.0 * theta)
        if include_polar:
            weights = jnp.stack([ones, c1, s1, c2, s2], axis=1)
        else:
            weights = jnp.stack([ones, c2, s2], axis=1)
        deposited = _bilinear_deposit(xy, weights, box, grid_shape) / cell_area
        smoothed = _smooth_2d_pbc(deposited, kernel, half_width)
        return smoothed.reshape(Nx * Ny, K)

    frames_out = []
    for t in range(X.shape[0]):
        frames_out.append(_one_frame(X[t]))
    return jnp.stack(frames_out, axis=0)
