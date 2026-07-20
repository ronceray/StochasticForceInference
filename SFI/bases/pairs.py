"""
SFI.bases.pairs
===============

Generic building blocks for multi-particle (pair-interaction) systems.

This module provides:

- **PBC utilities**: minimum-image displacement in arbitrary dimension.
- **Kernel families**: pre-built 1-D radial kernel functions.
- **Radial / scalar pair bases**: build Interactor objects from kernel
  families, ready for dispatch over neighbor lists.
- **Angular coupling bases**: weighted orientation-coupling interactors.
- **Heading vector**: single-particle heading vector from an angle coordinate.
- **Tensor pair features**: rank-2 dyadic basis for diffusion tensors, nematic order, etc.

"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax.numpy as jnp

from ..statefunc import Basis, Interactor, make_basis, make_interactor

# ═══════════════════════════════════════════════════════════════════════
#  PBC UTILITIES
# ═══════════════════════════════════════════════════════════════════════


def pbc_displacement(xj, xi, box):
    """Minimum-image displacement ``xj - xi`` under periodic boundaries.

    Works in any dimension.  All inputs are plain JAX arrays of shape
    ``(d,)`` (or broadcastable).

    Parameters
    ----------
    xj, xi : array, shape ``(d,)``
        Positions (or sub-positions) of two particles.
    box : array, shape ``(d,)``
        Box lengths along each axis.

    Returns
    -------
    dx : array, shape ``(d,)``
        ``xj - xi`` folded via minimum-image convention.
    """
    dx = xj - xi
    return dx - box * jnp.round(dx / box)


def wrap_angle(a):
    """Wrap angle(s) to ``(-π, π]``."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _pairwise_dr(XK, *, box=None, spatial_dims=None, eps=1e-12):
    """Compute displacement, distance and unit vector for a K=2 pair.

    Parameters
    ----------
    XK : array, shape ``(2, dim)``
        Stacked pair ``[xi, xj]``.
    box : array or None
        PBC box lengths (applied to ``spatial_dims`` only).
    spatial_dims : slice or index array, optional
        Which dimensions of state vector are spatial coordinates.
        Default: all.
    eps : float
        Regularisation to avoid division by zero.

    Returns
    -------
    dx : array, shape ``(d_spatial,)``
    r : scalar
    rhat : array, shape ``(d_spatial,)``
    """
    xi, xj = XK[0], XK[1]
    if spatial_dims is not None:
        xi_s, xj_s = xi[spatial_dims], xj[spatial_dims]
    else:
        xi_s, xj_s = xi, xj
    if box is not None:
        dx = pbc_displacement(xj_s, xi_s, box)
    else:
        dx = xj_s - xi_s
    r = jnp.sqrt(jnp.sum(dx**2) + eps)
    rhat = dx / r
    return dx, r, rhat


# ═══════════════════════════════════════════════════════════════════════
#  KERNEL FAMILIES
# ═══════════════════════════════════════════════════════════════════════


def exp_poly_kernels(degrees, lengths):
    r"""Radial kernels :math:`\phi_{k,L}(r) = r^k \exp(-r/L)`.

    Parameters
    ----------
    degrees : sequence of int
        Polynomial degrees *k*.
    lengths : sequence of float
        Exponential decay lengths *L*.

    Returns
    -------
    list of (callable, str)
        Each entry is ``(phi, label)`` where ``phi(r) -> scalar``.
    """
    out = []
    for k in degrees:
        for L in lengths:
            k_, L_ = int(k), float(L)

            def _phi(r, _k=k_, _L=L_):
                return (r**_k) * jnp.exp(-r / _L)

            out.append((_phi, f"r^{k_}·exp(-r/{L_:g})"))
    return out


def gaussian_kernels(sigmas):
    r"""Radial kernels :math:`\phi_\sigma(r) = \exp(-r^2 / 2\sigma^2)`.

    Parameters
    ----------
    sigmas : sequence of float
        Gaussian widths.

    Returns
    -------
    list of (callable, str)
    """
    out = []
    for s in sigmas:
        s_ = float(s)

        def _phi(r, _s=s_):
            return jnp.exp(-(r**2) / (2.0 * _s**2))

        out.append((_phi, f"r^0·gauss(σ={s_:g})"))
    return out


def power_kernels(degrees):
    r"""Radial kernels :math:`\phi_k(r) = r^k`.

    Parameters
    ----------
    degrees : sequence of int

    Returns
    -------
    list of (callable, str)
    """
    out = []
    for k in degrees:
        k_ = int(k)

        def _phi(r, _k=k_):
            return r**_k

        out.append((_phi, f"r^{k_}"))
    return out


def compact_kernels(degrees, cutoff):
    r"""Compactly-supported kernels :math:`r^k (1 - r/r_c)^2` for :math:`r < r_c`.

    Parameters
    ----------
    degrees : sequence of int
    cutoff : float
        Support radius *r_c*.

    Returns
    -------
    list of (callable, str)
    """
    rc = float(cutoff)
    out = []
    for k in degrees:
        k_ = int(k)

        def _phi(r, _k=k_, _rc=rc):
            w = jnp.where(r < _rc, (1.0 - r / _rc) ** 2, 0.0)
            return (r**_k) * w

        out.append((_phi, f"r^{k_}·comp(rc={rc:g})"))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  RADIAL / SCALAR PAIR BASES
# ═══════════════════════════════════════════════════════════════════════


def radial_pair_basis(
    kernels: Sequence[tuple[Callable, str]],
    *,
    dim: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
    embed_dim: int | None = None,
    embed_axes: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
) -> Interactor:
    r"""Build a rank-1 pair Interactor with radial-kernel features.

    Each feature is :math:`\phi_\alpha(r_{ij})\,\hat{\mathbf{r}}_{ij}`
    where :math:`r_{ij}` is the pairwise distance (optionally with PBC).

    .. physics:: Radial pair interaction basis
       :label: radial-pair-basis
       :category: Basis functions

       .. math::

          f_\alpha(\mathbf{r}_{ij})
          = \phi_\alpha(r_{ij})\;\hat{\mathbf{r}}_{ij}

       Scalar radial kernel :math:`\phi_\alpha` times the unit displacement
       vector.  Available kernel families: exponential-polynomial,
       Gaussian, power-law, and compactly supported.

    Parameters
    ----------
    kernels : list of (callable, str)
        1-D kernel functions and their labels, as returned by
        :func:`exp_poly_kernels`, :func:`gaussian_kernels`, etc.
    dim : int
        Full state-vector dimension per particle.
    box : array, ``"extras"``, or None
        PBC box lengths.  ``None`` (default) = free-space, no periodic
        boundaries.  Pass an array for a static box captured in the
        closure, or ``"extras"`` to read the box from
        ``extras["box"]`` at evaluation time.  The box is applied over
        ``spatial_dims`` only.
    spatial_dims : slice or index array, optional
        Which axes of the state vector are spatial coordinates (the ones
        over which distances are computed and the output vector is
        defined).  Default: ``slice(None)`` (all axes are spatial).
    embed_dim : int, optional
        If the output should be embedded into a larger vector (e.g., the
        displacement lives in 2D but the state vector is 3D with an
        angle), set ``embed_dim`` to the output vector size.  Spatial
        components are placed at ``embed_axes`` indices; remaining indices
        are zero.
    embed_axes : sequence of int, optional
        Indices into the ``embed_dim``-length output where spatial
        components are placed.  Required when ``embed_dim is not None``.
    labels : sequence of str, optional
        Override labels (one per kernel).

    Returns
    -------
    Interactor
        Rank-1 (vector) interactor with ``K=2``, ``n_features=len(kernels)``.
        Call ``.dispatch_pairs(...)`` to stream over neighbour lists.
    """
    n_feat = len(kernels)
    if labels is None:
        labels = [lab for _, lab in kernels]
    fns = [fn for fn, _ in kernels]

    # Determine output vector size
    if embed_dim is not None:
        out_d = embed_dim
        if embed_axes is None:
            raise ValueError("embed_axes required when embed_dim is set")
        _embed_axes = jnp.array(embed_axes, dtype=jnp.int32)
        if len(embed_axes) != (
            dim if spatial_dims is None
            else len(range(*spatial_dims.indices(dim))) if isinstance(spatial_dims, slice)
            else len(spatial_dims)
        ):
            raise ValueError(
                f"len(embed_axes)={len(embed_axes)} must equal the number of spatial "
                f"dimensions selected by spatial_dims."
            )
    else:
        out_d = (
            dim
            if spatial_dims is None
            else len(range(*spatial_dims.indices(dim)))
            if isinstance(spatial_dims, slice)
            else len(spatial_dims)
        )
        _embed_axes = None

    # Resolve box mode
    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        # Resolve box
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        dx, r, rhat = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)

        # Evaluate kernels → (n_feat,)
        vals = jnp.stack([fn(r) for fn in fns])  # (F,)

        # Build output: (out_d, F) = rhat[:, None] * vals[None, :]
        if _embed_axes is not None:
            # Embed into larger vector
            if len(embed_axes) != rhat.shape[0]:
                raise ValueError(
                    f"embed_axes length ({len(embed_axes)}) must equal the number of "
                    f"selected spatial dims ({rhat.shape[0]}) from spatial_dims."
                )
            full = jnp.zeros((out_d, n_feat), dtype=XK.dtype)
            for k, ax in enumerate(embed_axes):
                full = full.at[ax, :].set(rhat[k] * vals)
            return full
        else:
            return rhat[:, None] * vals[None, :]  # (d_spatial, F)

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=n_feat,
        extras_keys=extras_keys,
        labels=list(labels),
        descriptor="radial-pair-basis",
    )


def scalar_pair_basis(
    kernels: Sequence[tuple[Callable, str]],
    *,
    dim: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
) -> Interactor:
    r"""Build a rank-0 pair Interactor with scalar radial-kernel features.

    Each feature is :math:`\phi_\alpha(r_{ij})` — the raw kernel value
    without the directional :math:`\hat{r}` factor.

    Use this for energy-like quantities, as radial weights for angular
    coupling, or as building blocks for tensor pair features composed
    via the ``*`` operator.

    Parameters
    ----------
    kernels, dim, box, spatial_dims, labels
        Same as :func:`radial_pair_basis`.

    Returns
    -------
    Interactor
        Rank-0 (scalar) interactor with ``K=2``.
    """
    n_feat = len(kernels)
    if labels is None:
        labels = [lab for _, lab in kernels]
    fns = [fn for fn, _ in kernels]

    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        _, r, _ = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        return jnp.stack([fn(r) for fn in fns])  # (F,)

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=0,
        K=2,
        n_features=n_feat,
        extras_keys=extras_keys,
        labels=list(labels),
        descriptor="scalar-pair-basis",
    )


# ═══════════════════════════════════════════════════════════════════════
#  ANGULAR / ORIENTATION COUPLING
# ═══════════════════════════════════════════════════════════════════════


def angular_pair_basis(
    kernels: Sequence[tuple[Callable, str]],
    coupling_fn: Callable,
    *,
    dim: int,
    angle_index: int,
    output_index: int | None = None,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
    coupling_label: str = "g",
    labels: Sequence[str] | None = None,
) -> Interactor:
    r"""Build a rank-1 pair Interactor for orientation coupling.

    Each feature computes
    :math:`\phi_\alpha(r_{ij})\,g(\theta_j - \theta_i)` and embeds the
    result along ``output_index`` in a ``dim``-d output vector.

    Parameters
    ----------
    kernels : list of (callable, str)
        Radial weight functions (same format as other kernel factories).
    coupling_fn : callable
        Scalar function of the angle difference, e.g. ``jnp.sin`` for
        alignment, ``lambda a: jnp.cos(2*a)`` for nematic coupling.
    dim : int
        Full state-vector dimension.
    angle_index : int
        Index of the angle coordinate in the state vector.
    output_index : int, optional
        Index along which the coupled output is placed.  Defaults to
        ``angle_index``.
    box, spatial_dims
        PBC and spatial-dimension controls (same as :func:`radial_pair_basis`).
    coupling_label : str
        Short label for the coupling (appears in feature labels).
    labels : list of str, optional
        Override labels.

    Returns
    -------
    Interactor
        Rank-1 (vector) interactor with ``K=2``.
    """
    n_feat = len(kernels)
    if output_index is None:
        output_index = angle_index
    if labels is None:
        labels = [f"{coupling_label}·{lab}" for _, lab in kernels]
    fns = [fn for fn, _ in kernels]

    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        xi, xj = XK[0], XK[1]
        # distance
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        _, r, _ = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        # angle coupling
        dth = wrap_angle(xj[angle_index] - xi[angle_index])
        g = coupling_fn(dth)
        vals = jnp.stack([fn(r) * g for fn in fns])  # (F,)
        # embed along output_index
        out = jnp.zeros((dim, n_feat), dtype=XK.dtype)
        out = out.at[output_index, :].set(vals)
        return out  # (dim, F)

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=n_feat,
        extras_keys=extras_keys,
        labels=list(labels),
        descriptor="angular-pair-basis",
    )


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-PARTICLE: HEADING VECTOR
# ═══════════════════════════════════════════════════════════════════════


def heading_vector(dim: int, angle_index: int, *, spatial_axes: tuple[int, ...] | None = None) -> Basis:
    r"""Single-particle heading vector from an angle coordinate.

    Returns a rank-1 Basis whose single feature is the unit vector
    :math:`(\cos\theta, \sin\theta)` embedded in a ``dim``-d vector,
    with the cosine and sine placed at ``spatial_axes[0]`` and
    ``spatial_axes[1]`` respectively.

    Parameters
    ----------
    dim : int
        State-vector dimension.
    angle_index : int
        Index of the angle coordinate :math:`\theta`.
    spatial_axes : (int, int), optional
        Indices for (cos θ, sin θ) in the output.
        Default: ``(0, 1)`` — i.e. the first two axes.

    Returns
    -------
    Basis
        Rank-1, 1-feature heading-vector basis.
    """
    if spatial_axes is None:
        spatial_axes = (0, 1)
    ax_cos, ax_sin = spatial_axes[0], spatial_axes[1]

    def _f(x, *, mask=None):
        th = x[angle_index]
        out = jnp.zeros(dim, dtype=x.dtype)
        out = out.at[ax_cos].set(jnp.cos(th))
        out = out.at[ax_sin].set(jnp.sin(th))
        return out[:, None]  # (dim, 1) — rank-1, 1 feature

    return make_basis(
        _f,
        dim=dim,
        rank=1,
        n_features=1,
        labels=("e_heading",),
        descriptor="heading-vector",
    )


# ═══════════════════════════════════════════════════════════════════════
#  TENSOR PAIR FEATURES
# ═══════════════════════════════════════════════════════════════════════


def dyadic_pair_basis(
    kernels: Sequence[tuple[Callable, str]],
    *,
    dim: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
) -> Interactor:
    r"""Build a rank-2 (tensor) pair Interactor: :math:`\phi(r)\,\hat{r}\otimes\hat{r}`.

    Each feature is the outer product of the unit displacement vector
    with itself, weighted by a radial kernel.  Useful for directional
    diffusion tensors, nematic order parameters, etc.

    Parameters
    ----------
    kernels, dim, box, spatial_dims, labels
        Same as :func:`radial_pair_basis`.

    Returns
    -------
    Interactor
        Rank-2 (matrix) interactor with ``K=2``.
    """
    n_feat = len(kernels)
    if labels is None:
        labels = [f"rr·{lab}" for _, lab in kernels]
    fns = [fn for fn, _ in kernels]

    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        _, r, rhat = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        vals = jnp.stack([fn(r) for fn in fns])  # (F,)
        # rhat ⊗ rhat: (d, d) then weight by each kernel → (d, d, F)
        rr = jnp.outer(rhat, rhat)  # (d, d)
        return rr[:, :, None] * vals[None, None, :]  # (d, d, F)

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=2,
        K=2,
        n_features=n_feat,
        extras_keys=extras_keys,
        labels=list(labels),
        descriptor="dyadic-pair-basis",
    )


# ═══════════════════════════════════════════════════════════════════════
#  COMPOSABLE GEOMETRIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════
#
# Single-feature Interactors designed for composition via ``*``
# (element-wise spatial multiplication with feature Cartesian product).
# Combine a direction (rank-1), a scalar gate (rank-0), and a
# parametric radial kernel (rank-0) to build rich pair forces.
# ═══════════════════════════════════════════════════════════════════════


def pair_direction(
    *,
    dim: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
    embed_dim: int | None = None,
    embed_axes: Sequence[int] | None = None,
) -> Interactor:
    r"""Unit displacement vector :math:`\hat{r}_{ij}` as a rank-1 Interactor.

    Returns a single-feature, rank-1 pair Interactor whose output is the
    unit vector pointing from particle *i* to particle *j*.

    Parameters
    ----------
    dim : int
        Full state-vector dimension per particle.
    box : array, ``"extras"``, or None
        PBC box lengths (same semantics as :func:`radial_pair_basis`).
    spatial_dims : slice or index array, optional
        Which axes are spatial coordinates.
    embed_dim : int, optional
        Embed into a larger output vector (e.g. 2-D displacement in 3-D state).
    embed_axes : sequence of int, optional
        Indices for embedding (required when ``embed_dim`` is set).

    Returns
    -------
    Interactor
        Rank-1, 1-feature interactor with ``K=2``.
    """
    if embed_dim is not None:
        out_d = embed_dim
        if embed_axes is None:
            raise ValueError("embed_axes required when embed_dim is set")
    else:
        out_d = (
            dim
            if spatial_dims is None
            else (len(range(*spatial_dims.indices(dim))) if isinstance(spatial_dims, slice) else len(spatial_dims))
        )

    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        _, _, rhat = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        if embed_dim is not None:
            full = jnp.zeros(out_d, dtype=XK.dtype)
            for k, ax in enumerate(embed_axes):
                full = full.at[ax].set(rhat[k])
            return full[:, None]  # (out_d, 1)
        return rhat[:, None]  # (d_spatial, 1)

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=1,
        extras_keys=extras_keys,
        labels=("r̂_ij",),
        descriptor="pair-direction",
    )


def angle_coupling(
    coupling_fn: Callable,
    *,
    dim: int,
    angle_index: int,
    output_index: int | None = None,
    label: str = "g",
) -> Interactor:
    r"""Scalar orientation coupling embedded as a rank-1 Interactor.

    Computes ``coupling_fn(θ_j − θ_i)`` and places the result along
    ``output_index`` in a ``dim``-d output vector.

    Parameters
    ----------
    coupling_fn : callable
        Scalar function of the wrapped angle difference, e.g. ``jnp.sin``.
    dim : int
        Full state-vector dimension.
    angle_index : int
        Index of the angle coordinate in the state vector.
    output_index : int, optional
        Output axis for the coupling value.  Defaults to ``angle_index``.
    label : str
        Short label used in feature names.

    Returns
    -------
    Interactor
        Rank-1, 1-feature interactor with ``K=2``.
    """
    if output_index is None:
        output_index = angle_index

    def _pair_local(XK, *, extras=None):
        xi, xj = XK[0], XK[1]
        dth = wrap_angle(xj[angle_index] - xi[angle_index])
        g = coupling_fn(dth)
        out = jnp.zeros(dim, dtype=XK.dtype)
        out = out.at[output_index].set(g)
        return out[:, None]  # (dim, 1)

    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=1,
        labels=(label,),
        descriptor="angle-coupling",
    )


def particle_heading(
    which: int,
    *,
    dim: int,
    angle_index: int,
    spatial_axes: tuple[int, ...] | None = None,
) -> Interactor:
    r"""Heading vector of one particle in a pair, as a rank-1 Interactor.

    Returns :math:`(\cos\theta, \sin\theta)` of the selected particle
    (``which=0`` for the focal particle, ``which=1`` for the neighbor),
    embedded in a ``dim``-d output vector.

    Parameters
    ----------
    which : int
        ``0`` for the focal particle, ``1`` for the neighbor.
    dim : int
        Full state-vector dimension.
    angle_index : int
        Index of the angle coordinate.
    spatial_axes : (int, int), optional
        Indices for (cos θ, sin θ) in the output.  Default: ``(0, 1)``.

    Returns
    -------
    Interactor
        Rank-1, 1-feature interactor with ``K=2``.
    """
    if spatial_axes is None:
        spatial_axes = (0, 1)
    ax_cos, ax_sin = spatial_axes[0], spatial_axes[1]

    def _pair_local(XK, *, extras=None):
        th = XK[which][angle_index]
        out = jnp.zeros(dim, dtype=XK.dtype)
        out = out.at[ax_cos].set(jnp.cos(th))
        out = out.at[ax_sin].set(jnp.sin(th))
        return out[:, None]  # (dim, 1)

    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=1,
        labels=(f"ê_θ[{which}]",),
        descriptor="particle-heading",
    )


def vision_gate(
    gate_fn: Callable,
    *,
    dim: int,
    angle_index: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
) -> Interactor:
    r"""Scalar vision-cone gate as a rank-0 Interactor.

    Computes the bearing angle :math:`\delta` from the focal particle's
    heading to the displacement toward the neighbor, then returns
    ``gate_fn(δ)``.  This makes interactions *nonreciprocal*: the gate
    value depends on whether the neighbor is "in front" or "behind" the
    focal particle.

    Parameters
    ----------
    gate_fn : callable
        Scalar function of the bearing angle, e.g.
        ``lambda d: (1 + jnp.cos(d)) / 2`` for a cosine vision cone.
    dim : int
        Full state-vector dimension.
    angle_index : int
        Index of the angle coordinate.
    box : array, ``"extras"``, or None
        PBC box (same semantics as :func:`radial_pair_basis`).
    spatial_dims : slice or index array, optional
        Which axes are spatial coordinates (for the displacement).
        Must select exactly 2 dimensions — bearing angle is 2-D only.

    Returns
    -------
    Interactor
        Rank-0, 1-feature interactor with ``K=2``.
    """
    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, extras=None):
        xi, _ = XK[0], XK[1]
        # Spatial displacement
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        dx, _, _ = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        if dx.shape[0] != 2:
            raise ValueError(
                f"vision_gate requires exactly 2 spatial dimensions, got {dx.shape[0]}. "
                "Use spatial_dims to select 2 axes from a higher-dimensional state vector."
            )
        # Bearing angle: direction of neighbour relative to heading
        phi_ij = jnp.arctan2(dx[1], dx[0])
        theta_i = xi[angle_index]
        delta = wrap_angle(phi_ij - theta_i)
        return gate_fn(delta)[None]  # (1,) — rank-0, 1 feature

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=0,
        K=2,
        n_features=1,
        extras_keys=extras_keys,
        labels=("vision",),
        descriptor="vision-gate",
    )


def parametric_radial_kernel(
    kernel_fn: Callable,
    *,
    params: dict,
    dim: int,
    box: Any = None,
    spatial_dims: slice | Sequence[int] | None = None,
) -> Interactor:
    r"""Parametric scalar radial kernel as a rank-0 Interactor.

    Wraps a user-supplied function ``kernel_fn(r, params)`` into a
    rank-0 pair Interactor with learnable parameters.

    Parameters
    ----------
    kernel_fn : callable
        ``kernel_fn(r, params) -> scalar`` where *r* is the inter-particle
        distance and *params* is a dict of JAX arrays.
    params : dict
        Parameter specification passed to :func:`make_interactor`, e.g.
        ``{"eps": (), "R0": ()}`` for two scalar parameters.
    dim : int
        Full state-vector dimension.
    box : array, ``"extras"``, or None
        PBC box (same semantics as :func:`radial_pair_basis`).
    spatial_dims : slice or index array, optional
        Which axes are spatial coordinates.

    Returns
    -------
    Interactor
        Rank-0, 1-feature parametric interactor with ``K=2``.
    """
    _use_extras_box = box == "extras"
    _box = None if (box is None or _use_extras_box) else jnp.asarray(box)

    def _pair_local(XK, *, params=None, extras=None):
        b = _box
        if b is None and _use_extras_box and extras is not None:
            b = extras["box"]
            if spatial_dims is not None:
                b = b[spatial_dims] if b.ndim > 0 else b
        _, r, _ = _pairwise_dr(XK, box=b, spatial_dims=spatial_dims)
        return kernel_fn(r, params)[None]  # (1,) — rank-0, 1 feature

    extras_keys = ("box",) if _use_extras_box else ()
    return make_interactor(
        _pair_local,
        dim=dim,
        rank=0,
        K=2,
        n_features=1,
        params=params,
        extras_keys=extras_keys,
        labels=("k(r)",),
        descriptor="parametric-radial-kernel",
    )


# ═══════════════════════════════════════════════════════════════════════
#  VELOCITY-DEPENDENT PAIR BASES
# ═══════════════════════════════════════════════════════════════════════


def pair_velocity_difference(
    *,
    dim: int,
) -> Interactor:
    r"""Velocity difference :math:`\mathbf{v}_j - \mathbf{v}_i` as a rank-1 Interactor.

    Returns a single-feature, rank-1 pair Interactor whose output is the
    velocity difference between neighbor and focal particle.  Designed
    for composition with scalar pair Interactors via the ``*`` operator,
    e.g.::

        scalar_pair_basis(kernels, dim=d) * pair_velocity_difference(dim=d)

    Parameters
    ----------
    dim : int
        State-vector dimension per particle.

    Returns
    -------
    Interactor
        Rank-1, 1-feature interactor with ``K=2``, ``needs_v=True``.
    """

    def _pair_local(XK, *, v, extras=None):
        dv = v[1] - v[0]  # (dim,)
        return dv[:, None]  # (dim, 1)

    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=1,
        needs_v=True,
        labels=("Δv_ij",),
        descriptor="pair-velocity-difference",
    )


def particle_velocity(
    which: int,
    *,
    dim: int,
) -> Interactor:
    r"""Velocity of one particle in a pair, as a rank-1 Interactor.

    Returns the velocity of either the focal particle (``which=0``) or
    the neighbor (``which=1``) as a rank-1 Interactor.  Designed for
    composition with scalar pair Interactors via the ``*`` operator, e.g.::

        scalar_pair_basis(kernels, dim=d) * particle_velocity(which=1, dim=d)

    Parameters
    ----------
    which : int
        ``0`` for the focal particle's velocity, ``1`` for the neighbor's.
    dim : int
        State-vector dimension per particle.

    Returns
    -------
    Interactor
        Rank-1, 1-feature interactor with ``K=2``, ``needs_v=True``.
    """

    def _pair_local(XK, *, v, extras=None):
        return v[which][:, None]  # (dim, 1)

    return make_interactor(
        _pair_local,
        dim=dim,
        rank=1,
        K=2,
        n_features=1,
        needs_v=True,
        labels=(f"v[{which}]",),
        descriptor="particle-velocity",
    )
