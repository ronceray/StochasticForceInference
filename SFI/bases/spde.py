"""SPDE-oriented basis helpers.

Grid operators here are designed to be *box-polymorphic*: geometry is provided
via per-dataset extras (typically ``collection.extras_global``), so the same
basis/PSF object can run on different grid sizes without re-creation.

Important conventions
---------------------
We represent a regular grid as a "particle system":

* **particle axis** corresponds to flattened grid index ``p = 0...P-1``;
* **n_fields** (historically called ``dim``) is the number of field components
  per grid site (e.g. 2 for Gray-Scott U/V, 1 for a scalar concentration).

Grid-site positions are **purely implicit**: they are reconstructed from
``box/grid_shape`` and ``box/dx`` stored in extras.  The state vector
``X[p, :]`` contains only field values, never spatial coordinates.

The Laplacian helper returns a **scalar-rank** basis with
``n_features = n_fields``: feature ``a`` is the Laplacian of field
component ``X[..., a]``.

This is deliberate: it composes cleanly with generic StateExpr machinery
(slice a component and re-embed it into a vector drift with unit vectors).

Box extras
----------
Every operator reads two global extras (namespaced ``box`` by default):

* ``{box}/grid_shape`` : int array, shape ``(ndim,)``
* ``{box}/dx``         : float array, shape ``(ndim,)``

Use :func:`square_grid_extras` to create them.

.. tip::

   For GPU acceleration with JAX, ensure a CUDA-enabled ``jaxlib`` is
   installed.  All operators are JIT-compiled and run transparently on
   any JAX backend (CPU / GPU / TPU).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..statefunc.factory import make_interactor
from ..statefunc.nodes.interactions.stencils import (
    PreparedSquareStencilFromBox,
    square_biharmonic_offsets,
    square_cross_offsets,
)
from ..statefunc.nodes.interactions.stencils import (
    box_extras as _stencils_box_extras,
)

if TYPE_CHECKING:
    from ..langevin.noise import ConservedNoise

# ============================================================================
# Box extras  (grid geometry -- shared by ALL stencil operators)
# ============================================================================


def square_grid_extras(
    grid_shape: Sequence[int],
    dx: float | Sequence[float] = 1.0,
    *,
    prefix: str = "box",
) -> dict[str, Array]:
    """Return minimal box extras needed by **all** square-grid SPDE operators.

    Creates two arrays in the returned dict:

    * ``{prefix}/grid_shape`` -- integer grid dimensions ``(Nx, Ny, ...)``
    * ``{prefix}/dx``         -- float grid spacing per axis

    Parameters
    ----------
    grid_shape : sequence of int
        Number of grid points along each spatial axis.
    dx : float or sequence of float
        Grid spacing (uniform or per-axis).
    prefix : str
        Namespace prefix for the extras keys (default ``"box"``).
    """
    return _stencils_box_extras(grid_shape=grid_shape, dx=dx, prefix=prefix)


# ============================================================================
# Stencil algebra: Minkowski sum for operator composition
# ============================================================================


def minkowski_sum_offsets(offsets_a, offsets_b):
    """Compute the Minkowski sum of two stencil offset arrays.

    Given outer offsets *O_a* and inner offsets *O_b*, computes the
    deduplicated set ``{a + b : a in O_a, b in O_b}`` and an index
    mapping suitable for composing the two FD formulas.

    Parameters
    ----------
    offsets_a : array, shape ``(Ka, ndim)``
        Outer operator stencil offsets.
    offsets_b : array, shape ``(Kb, ndim)``
        Inner operator stencil offsets.

    Returns
    -------
    fused : jnp.ndarray, shape ``(K_fused, ndim)``
        Deduplicated sorted offset vectors.
    index_maps : jnp.ndarray, shape ``(Ka, Kb)``
        ``index_maps[i, j]`` is the position of ``offsets_a[i] + offsets_b[j]``
        in *fused*.
    """
    a = np.asarray(offsets_a, dtype=np.int32)
    b = np.asarray(offsets_b, dtype=np.int32)
    Ka = a.shape[0]
    ndim = a.shape[1]
    sums = a[:, None, :] + b[None, :, :]  # (Ka, Kb, ndim)
    flat = sums.reshape(-1, ndim)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Convention: center offset (0,…,0) must sit at slot 0 so that
    # focal_index=0 remains valid after fusion.
    center = np.zeros((1, ndim), dtype=np.int32)
    center_idx = int(np.where(np.all(unique == center, axis=1))[0][0])
    if center_idx != 0:
        unique[[0, center_idx]] = unique[[center_idx, 0]]
        swap_0 = inverse == 0
        swap_c = inverse == center_idx
        inverse[swap_c] = 0
        inverse[swap_0] = center_idx

    index_maps = inverse.reshape(Ka, b.shape[0])
    return jnp.asarray(unique, dtype=jnp.int32), jnp.asarray(index_maps, dtype=jnp.int32)


_StencilMeta = namedtuple(
    "_StencilMeta",
    [
        "offsets",  # jnp.ndarray (K, ndim)
        "inner_root",  # callable — the deepest pointwise root
        "n_fields",  # int (dim of deepest expression)
        "n_features_out",  # int (output features of this expression)
        "fd_only_fn",  # callable (K, n_feat_in) -> (n_feat_out,) — FD chain only
        "ndim",
        "bc",
        "order",
        "key_grid_shape",
        "key_dx",
        "labels",
    ],
)


# ============================================================================
# Helper: shared dispatch boilerplate  (cross *or* arbitrary stencil)
# ============================================================================


def _dispatch_stencil_operator(
    local_fn,
    *,
    offsets,
    n_fields: int,
    ndim: int,
    rank: int,
    n_features: int,
    bc: str,
    order: str,
    key_grid_shape: str,
    stencil_prefix: str,
    include_slot_extras_offset: bool,
    return_as: str,
    labels: list[str],
):
    """Wire local_fn + stencil offsets -> interactor -> prepared spec -> dispatch."""
    K = int(offsets.shape[0])

    inter = make_interactor(
        local_fn,
        dim=n_fields,
        rank=rank,
        K=K,
        n_features=n_features,
        labels=labels,
    )

    spec = PreparedSquareStencilFromBox(
        offsets=offsets,
        bc=bc,
        order=order,
        key_grid_shape=key_grid_shape,
        key_prefix=stencil_prefix,
        include_slot_extras_offset=include_slot_extras_offset,
    )

    return inter.dispatch(
        spec,
        owners="focal",
        focal_index=0,
        reducer="sum",
        return_as=return_as,
    )


def _default_cross_prefix(ndim: int, bc: str) -> str:
    """Shared stencil cache key for the cross stencil."""
    return f"_cache/stencil/cross_{ndim}d_{bc}"


def _default_biharmonic_prefix(ndim: int, bc: str) -> str:
    """Shared stencil cache key for the biharmonic stencil."""
    return f"_cache/stencil/biharmonic_{ndim}d_{bc}"


# ============================================================================
# StencilOp: composable differential operators that act on StateExpr
# ============================================================================


class StencilOp(ABC):
    r"""Composable finite-difference operator on a regular Cartesian grid.

    A ``StencilOp`` wraps a stencil-based finite-difference formula and can
    be **applied to any pointwise** :class:`~SFI.statefunc.StateExpr` to
    produce a new :class:`~SFI.statefunc.Basis`:

    .. code-block:: python

        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        lap_phi  = Lap(phi)       # standard  ∇²φ
        lap_phi3 = Lap(phi**3)    # ∇²(φ³) — exactly conservative on PBC

    The operator evaluates the inner expression on each stencil neighbor,
    then applies the finite-difference formula to the *transformed* values.
    This guarantees exact discrete conservation properties (e.g., zero
    spatial sum for the Laplacian on periodic grids) regardless of the
    inner nonlinearity.

    **Operator composition** is fully supported via automatic stencil
    fusion.  Nesting operators like ``Lap(Lap(phi))`` computes the
    Minkowski sum of the two stencils at construction time so that only
    a single dispatcher pass is needed at evaluation time:

    .. code-block:: python

        bih = Lap(Lap(phi))      # ≡ ∇⁴φ, fused 13-point stencil
        div_grad = Div(Grad(phi))  # ≡ ∇²φ via composition

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    bc : str
        Boundary condition: ``"pbc"`` (periodic) or ``"noflux"`` (reflecting).
    order : str
        Flattening order for the spatial grid (``"C"`` row-major or ``"F"``
        column-major).  Must match the layout used to build the
        ``extras["box/grid_shape"]`` array passed at evaluation time.
    key_grid_shape : str
        Extras key for the grid shape array.
    key_dx : str
        Extras key for the grid spacing array.
    """

    def __init__(
        self,
        *,
        ndim: int,
        bc: str = "noflux",
        order: str = "C",
        key_grid_shape: str = "box/grid_shape",
        key_dx: str = "box/dx",
    ):
        self._ndim = int(ndim)
        self._bc = bc
        self._order = order
        self._key_grid_shape = key_grid_shape
        self._key_dx = key_dx

    # Whether results may carry fusion metadata (overridden by AdvectionBy).
    _composable = True

    # --- subclass hooks (override) ------------------------------------

    @abstractmethod
    def _offsets(self):
        """Return stencil offsets array, shape ``(K, ndim)``."""

    @abstractmethod
    def _default_prefix(self) -> str:
        """Return default shared stencil cache prefix."""

    @abstractmethod
    def _build_local_fn(self, inner_root, *, n_features_out):
        """Return a ``local_fn(Xk, *, mask, extras)`` for the composed op."""

    def _n_features_out(self, n_features_inner: int) -> int:
        """Number of output features given inner expression's features."""
        return n_features_inner

    def _output_rank(self) -> int:
        return 0

    def _make_labels(self, label, n_features_out, inner_labels):
        """Build human-readable output labels."""
        name = label or type(self).__name__
        if len(inner_labels) >= n_features_out:
            return [f"{name}({il})" for il in inner_labels[:n_features_out]]
        return [f"{name}[{j}]" for j in range(n_features_out)]

    # --- public API ---------------------------------------------------

    def __call__(
        self,
        expr,
        *,
        return_as: str = "basis",
        label: str | None = None,
    ):
        """Apply the operator to a pointwise expression.

        Parameters
        ----------
        expr : StateExpr
            Pointwise expression (``rank=0``, ``particles_input=False``).
            Typical examples: ``field_component(0, n_fields=1)``, ``phi**3``,
            ``jnp.sin(phi)``, or any algebraic combination of such.

            May also be the result of another ``StencilOp`` — in that case,
            the two stencils are automatically fused via Minkowski sum.
        return_as : str
            ``"basis"`` (default) or ``"psf"``.
        label : str, optional
            Override for the output label prefix.

        Returns
        -------
        Basis or PSF
            The composed stencil operator evaluated on the expression.

        Raises
        ------
        TypeError
            If *expr* is not a :class:`~SFI.statefunc.StateExpr`.
        ValueError
            If *expr* has ``particles_input=True`` (and is not a stencil
            composition) or ``rank != 0``.
        """
        from ..statefunc.stateexpr import StateExpr

        if not isinstance(expr, StateExpr):
            raise TypeError(f"StencilOp expects a StateExpr, got {type(expr).__name__}")

        # --- Composition path: fuse stencils if inner is a stencil result ---
        inner_meta = getattr(expr, "_stencil_meta", None)
        if inner_meta is not None:
            if not self._composable:
                raise ValueError(
                    f"{type(self).__name__} does not support stencil composition.  Apply it to a pointwise expression."
                )
            return self._compose_stencils(
                inner_meta,
                return_as=return_as,
                label=label,
            )

        # --- Normal path: pointwise inner expression -----------------------
        if expr.particles_input:
            raise ValueError(
                "StencilOp requires a pointwise inner expression "
                "(particles_input=False).  For compositions like "
                "∇⁴ = ∇²∘∇², simply nest: Lap(Lap(phi))."
            )
        if expr.rank != 0:
            raise ValueError(f"StencilOp requires rank-0 inner expression, got rank={expr.rank}.")

        n_fields = expr.dim
        n_features_inner = expr.n_features
        n_features_out = self._n_features_out(n_features_inner)

        inner_root = expr.root
        local_fn = self._build_local_fn(
            inner_root,
            n_features_out=n_features_out,
        )

        offsets = self._offsets()
        stencil_prefix = self._default_prefix()

        # Labels
        inner_labels = getattr(expr, "labels", None)
        if inner_labels is None:
            inner_labels = [f"f{j}" for j in range(n_features_inner)]
        labels = self._make_labels(label, n_features_out, inner_labels)

        basis = _dispatch_stencil_operator(
            local_fn,
            offsets=offsets,
            n_fields=n_fields,
            ndim=self._ndim,
            rank=self._output_rank(),
            n_features=n_features_out,
            bc=self._bc,
            order=self._order,
            key_grid_shape=self._key_grid_shape,
            stencil_prefix=stencil_prefix,
            include_slot_extras_offset=True,
            return_as=return_as,
            labels=labels,
        )

        # Tag result with stencil metadata for potential composition
        if self._composable:
            fd_fn = self._build_local_fn(
                lambda x: x,
                n_features_out=n_features_out,
            )
            meta = _StencilMeta(
                offsets=offsets,
                inner_root=inner_root,
                n_fields=n_fields,
                n_features_out=n_features_out,
                fd_only_fn=fd_fn,
                ndim=self._ndim,
                bc=self._bc,
                order=self._order,
                key_grid_shape=self._key_grid_shape,
                key_dx=self._key_dx,
                labels=labels,
            )
            object.__setattr__(basis, "_stencil_meta", meta)

        return basis

    # --- composition (stencil fusion) ------------------------------------

    def _compose_stencils(self, inner_meta, *, return_as, label):
        """Fuse this operator's stencil with a prior stencil expression."""
        # --- compatibility checks ---
        if inner_meta.ndim != self._ndim:
            raise ValueError(f"Cannot compose: ndim mismatch ({self._ndim} vs {inner_meta.ndim})")
        if inner_meta.bc != self._bc:
            raise ValueError(f"Cannot compose: bc mismatch ({self._bc!r} vs {inner_meta.bc!r})")

        inner_offsets = inner_meta.offsets
        inner_root = inner_meta.inner_root
        inner_fd = inner_meta.fd_only_fn
        inner_n_fields = inner_meta.n_fields
        inner_n_features_out = inner_meta.n_features_out

        outer_offsets = self._offsets()
        K_outer = int(outer_offsets.shape[0])

        # --- Minkowski sum ---
        fused_offsets, index_maps = minkowski_sum_offsets(
            outer_offsets,
            inner_offsets,
        )

        # --- output info ---
        n_features_out = self._n_features_out(inner_n_features_out)
        outer_fd = self._build_local_fn(
            lambda x: x,
            n_features_out=n_features_out,
        )
        assert outer_fd is not None

        # --- build fused FD-only function ---
        _idx = index_maps  # capture for closure
        _Ko = K_outer

        def fused_fd(fXk_fused, *, mask=None, extras=None):
            intermediate = []
            for a in range(_Ko):
                idx_a = _idx[a]
                subset = fXk_fused[idx_a]
                sub_mask = mask[idx_a] if mask is not None else None
                intermediate.append(inner_fd(subset, mask=sub_mask, extras=extras))
            intermediate = jnp.stack(intermediate)
            outer_mask = None
            if mask is not None:
                outer_mask = mask[_idx[:, 0]]
            return outer_fd(intermediate, mask=outer_mask, extras=extras)

        # --- build full local_fn (root eval + fused FD) ---
        def fused_local_fn(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)
            return fused_fd(fXk, mask=mask, extras=extras)

        # --- unique stencil prefix ---
        _h = hashlib.md5(np.asarray(fused_offsets).tobytes()).hexdigest()[:8]
        stencil_prefix = f"_cache/stencil/fused_{self._ndim}d_{self._bc}_{_h}"

        # --- labels ---
        inner_labels = inner_meta.labels or [f"f{j}" for j in range(inner_n_features_out)]
        labels = self._make_labels(label, n_features_out, inner_labels)

        basis = _dispatch_stencil_operator(
            fused_local_fn,
            offsets=fused_offsets,
            n_fields=inner_n_fields,
            ndim=self._ndim,
            rank=self._output_rank(),
            n_features=n_features_out,
            bc=self._bc,
            order=self._order,
            key_grid_shape=self._key_grid_shape,
            stencil_prefix=stencil_prefix,
            include_slot_extras_offset=True,
            return_as=return_as,
            labels=labels,
        )

        # Tag for further composition
        meta = _StencilMeta(
            offsets=fused_offsets,
            inner_root=inner_root,
            n_fields=inner_n_fields,
            n_features_out=n_features_out,
            fd_only_fn=fused_fd,
            ndim=self._ndim,
            bc=self._bc,
            order=self._order,
            key_grid_shape=self._key_grid_shape,
            key_dx=self._key_dx,
            labels=labels,
        )
        object.__setattr__(basis, "_stencil_meta", meta)
        return basis

    # --- ASCII stencil visualization -------------------------------------

    def visualize_stencil(self):
        """Return an ASCII picture of the stencil pattern (2D only).

        For ``ndim != 2`` a simple offset listing is returned.

        Returns
        -------
        str
            Multi-line string.
        """
        offsets = self._offsets()
        assert offsets is not None
        ndim = self._ndim
        name = type(self).__name__
        K = int(offsets.shape[0])
        header = f"{name}(ndim={ndim}, bc={self._bc!r}) \u2014 {K}-point stencil"

        if ndim != 2:
            lines = [header, "Offsets:"]
            for i in range(K):
                off = tuple(int(v) for v in offsets[i])
                tag = " (center)" if all(v == 0 for v in off) else ""
                lines.append(f"  {off}{tag}")
            return "\n".join(lines)

        off_np = np.asarray(offsets, dtype=np.int32)
        xs, ys = off_np[:, 0], off_np[:, 1]
        offset_set = {(int(r[0]), int(r[1])) for r in off_np}
        lines = [header]
        for y in range(int(ys.max()), int(ys.min()) - 1, -1):
            row = []
            for x in range(int(xs.min()), int(xs.max()) + 1):
                if (x, y) == (0, 0):
                    row.append(" \u25cf ")
                elif (x, y) in offset_set:
                    row.append(" \u25cb ")
                else:
                    row.append(" \u00b7 ")
            lines.append("".join(row))
        return "\n".join(lines)

    def __repr__(self):
        return f"{type(self).__name__}(ndim={self._ndim}, bc={self._bc!r})"


class Laplacian(StencilOp):
    r"""Composable Laplacian operator :math:`\nabla^2` on a Cartesian grid.

    .. math::

       [\nabla^2 f(\phi)]_i = \sum_{\alpha=1}^{n_\mathrm{dim}}
       \frac{f(\phi_{i+e_\alpha}) + f(\phi_{i-e_\alpha}) - 2\,f(\phi_i)}
       {\Delta x_\alpha^2}

    where :math:`f` is the inner expression evaluated at each grid site.

    On periodic grids, the Laplacian matrix has zero column sums, so
    :math:`\sum_i [\nabla^2 g]_i = 0` for **any** function :math:`g`.
    This guarantees exact discrete conservation for expressions like
    :math:`\nabla^2(\phi^3)`.

    Examples
    --------
    >>> from SFI.bases import field_component
    >>> from SFI.bases.spde import Laplacian
    >>> Lap = Laplacian(ndim=2, bc="pbc")
    >>> phi = field_component(0, n_fields=1)
    >>> lap_phi  = Lap(phi)       # ∇²φ
    >>> lap_phi3 = Lap(phi**3)    # ∇²(φ³) — conservative
    """

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Stencil layout from square_cross_offsets: idx 0 = center,
        # then ndim pairs (plus, minus) → plus_idx = 1,3,5,…; minus_idx = 2,4,6,…
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        def local_lap_composed(Xk, *, mask=None, extras=None):
            # Evaluate inner expression on each gathered neighbor.
            # Xk: (K, n_fields) → inner_root treats K as batch → (K, n_feat)
            fXk = inner_root(Xk)

            f0 = fXk[0, :]  # center: (n_feat,)
            fp = fXk[plus_idx, :]  # (ndim, n_feat)
            fm = fXk[minus_idx, :]  # (ndim, n_feat)

            if mask is not None:
                mp = mask[plus_idx]  # (ndim,) bool
                mm = mask[minus_idx]
                fp = jnp.where(mp[:, None], fp, f0[None, :])
                fm = jnp.where(mm[:, None], fm, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"Laplacian requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_dx2 = 1.0 / (dx * dx)

            diffs = fp + fm - 2.0 * f0[None, :]  # (ndim, n_feat)
            lap = jnp.sum(diffs * inv_dx2[:, None], axis=0)  # (n_feat,)

            if mask is not None:
                lap = lap * mask[0]

            return lap

        return local_lap_composed

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "∇²"
        return [f"{name}({il})" for il in inner_labels[:n_features_out]]


class Biharmonic(StencilOp):
    r"""Composable biharmonic operator :math:`\nabla^4` on a Cartesian grid.

    Uses the standard 13-point stencil (2D) or 25-point stencil (3D).
    Requires ``ndim >= 2``.

    On periodic grids, :math:`\sum_i [\nabla^4 g]_i = 0` for any :math:`g`,
    so compositions like :math:`\nabla^4(\phi^2)` are exactly conservative.

    Examples
    --------
    >>> Bih = Biharmonic(ndim=2, bc="pbc")
    >>> phi = field_component(0, n_fields=1)
    >>> bih_phi = Bih(phi)         # ∇⁴φ
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._ndim < 2:
            raise ValueError(f"Biharmonic requires ndim >= 2 (got {self._ndim})")

    def _offsets(self):
        return square_biharmonic_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_biharmonic_prefix(self._ndim, self._bc)

    def _build_local_fn(self, inner_root, *, n_features_out):
        import itertools

        ndim = self._ndim
        key_dx = self._key_dx
        n_r2 = 2 * ndim
        n_r1 = 2 * ndim
        diag_pairs = list(itertools.combinations(range(ndim), 2))

        center = 0
        r2_plus = jnp.array([1 + 2 * a for a in range(ndim)])
        r2_minus = jnp.array([2 + 2 * a for a in range(ndim)])
        r1_base = 1 + n_r2
        r1_plus = jnp.array([r1_base + 2 * a for a in range(ndim)])
        r1_minus = jnp.array([r1_base + 2 * a + 1 for a in range(ndim)])
        diag_base = 1 + n_r2 + n_r1

        def local_bih_composed(Xk, *, mask=None, extras=None):
            # Evaluate inner expression on all stencil neighbors.
            fXk = inner_root(Xk)  # (K, n_feat)
            f0 = fXk[center, :]

            if extras is None or key_dx not in extras:
                raise KeyError(f"Biharmonic requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            inv_dx2 = 1.0 / (dx * dx)
            result = jnp.zeros_like(f0)

            for a in range(ndim):
                h4_inv = inv_dx2[a] * inv_dx2[a]
                f_2p = fXk[r2_plus[a], :]
                f_2m = fXk[r2_minus[a], :]
                f_p = fXk[r1_plus[a], :]
                f_m = fXk[r1_minus[a], :]
                result = result + h4_inv * (f_2p - 4.0 * f_p + 6.0 * f0 - 4.0 * f_m + f_2m)

            for pair_i, (a, b) in enumerate(diag_pairs):
                hab_inv = inv_dx2[a] * inv_dx2[b]
                base_s = diag_base + pair_i * 4
                f_pp = fXk[base_s, :]
                f_pm = fXk[base_s + 1, :]
                f_mp = fXk[base_s + 2, :]
                f_mm = fXk[base_s + 3, :]
                f_pa = fXk[r1_plus[a], :]
                f_ma = fXk[r1_minus[a], :]
                f_pb = fXk[r1_plus[b], :]
                f_mb = fXk[r1_minus[b], :]

                mixed = f_pp + f_pm + f_mp + f_mm - 2.0 * (f_pa + f_ma + f_pb + f_mb) + 4.0 * f0
                result = result + 2.0 * hab_inv * mixed

            if mask is not None:
                result = result * mask[center]

            return result

        return local_bih_composed

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "∇⁴"
        return [f"{name}({il})" for il in inner_labels[:n_features_out]]


class Gradient(StencilOp):
    r"""Composable gradient operator :math:`\nabla` on a Cartesian grid.

    .. math::

       [\nabla f(\phi)]_{i,\alpha} =
       \frac{f(\phi_{i+e_\alpha}) - f(\phi_{i-e_\alpha})}{2\,\Delta x_\alpha}

    Output features = ``n_features_inner × ndim``, ordered as
    ``(feature_0/dx_0, feature_0/dx_1, ..., feature_1/dx_0, ...)``.

    Examples
    --------
    >>> Grad = Gradient(ndim=2, bc="pbc")
    >>> phi  = field_component(0, n_fields=1)
    >>> grad_phi  = Grad(phi)       # ∇φ  (2 features)
    >>> grad_phi2 = Grad(phi**2)    # ∇(φ²) ≈ 2φ∇φ
    """

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _n_features_out(self, n_features_inner):
        return n_features_inner * self._ndim

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Same stencil layout as Laplacian (see above)
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        def local_grad_composed(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)  # (K, n_feat_inner)
            f0 = fXk[0, :]
            fp = fXk[plus_idx, :]  # (ndim, n_feat_inner)
            fm = fXk[minus_idx, :]

            if mask is not None:
                mp = mask[plus_idx]
                mm = mask[minus_idx]
                fp = jnp.where(mp[:, None], fp, f0[None, :])
                fm = jnp.where(mm[:, None], fm, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"Gradient requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_2dx = 1.0 / (2.0 * dx)

            diffs = (fp - fm) * inv_2dx[:, None]  # (ndim, n_feat_inner)
            # Reorder to (n_feat_inner, ndim) then flatten
            diffs = jnp.moveaxis(diffs, 0, -1)  # (n_feat_inner, ndim)
            result = diffs.reshape((-1,))  # (n_features_out,)

            if mask is not None:
                result = result * mask[0]

            return result

        return local_grad_composed

    def _make_labels(self, label, n_features_out, inner_labels):
        ndim = self._ndim
        n_inner = n_features_out // ndim
        labels = []
        for il in inner_labels[:n_inner]:
            for a in range(ndim):
                labels.append(f"d({il})/dx{a}")
        return labels


class LaplacianOfGradientSquared(StencilOp):
    r"""Composable operator :math:`\nabla^2(|\nabla f(\phi)|^2)`.

    This is the **genuine Active Model B+** term that breaks time-reversal
    symmetry.  It cannot be written as :math:`\nabla^2(g(\phi))` for any
    pointwise function :math:`g`, because :math:`|\nabla\phi|^2` depends
    on spatial derivatives, not just the local field value.

    The computation proceeds in two stages on the biharmonic (radius-2)
    stencil:

    1. Evaluate the inner expression :math:`f(\phi)` at all stencil
       neighbors (13 points in 2D, 25 in 3D).
    2. Compute :math:`|\nabla f|^2` at each of the Laplacian's 5
       stencil points (center + nearest neighbours) using central
       differences among the gathered values.
    3. Apply the standard Laplacian formula to the resulting
       :math:`|\nabla f|^2` field.

    On periodic grids, the outer Laplacian guarantees
    :math:`\sum_i [\nabla^2 G]_i = 0` exactly, so the operator
    is **conservative** even though :math:`G = |\nabla f|^2` is
    nonlinear in the spatial derivatives.

    .. math::

       [\nabla^2(|\nabla f|^2)]_i
       = \sum_\alpha \frac{G_{i+e_\alpha} + G_{i-e_\alpha} - 2\,G_i}
                          {\Delta x_\alpha^2}

    where

    .. math::

       G_j = |\nabla f(\phi)|^2_j
           = \sum_\beta \left(\frac{f(\phi_{j+e_\beta}) - f(\phi_{j-e_\beta})}
                               {2\,\Delta x_\beta}\right)^{\!2}

    Requires ``ndim >= 2``.

    Examples
    --------
    >>> LGS = LaplacianOfGradientSquared(ndim=2, bc="pbc")
    >>> phi = field_component(0, n_fields=1)
    >>> lgs_phi  = LGS(phi)       # ∇²(|∇φ|²)  (AMB+ active term)
    >>> lgs_phi2 = LGS(phi**2)    # ∇²(|∇(φ²)|²)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._ndim < 2:
            raise ValueError(f"LaplacianOfGradientSquared requires ndim >= 2 (got {self._ndim})")

    def _offsets(self):
        return square_biharmonic_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_biharmonic_prefix(self._ndim, self._bc)

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx

        # ----- Build offset → stencil-index lookup -----
        offsets = square_biharmonic_offsets(ndim, include_center=True)
        offset_to_idx: dict[tuple[int, ...], int] = {}
        for i in range(len(offsets)):
            offset_to_idx[tuple(int(x) for x in offsets[i])] = i

        center_idx = 0

        def _shifted(base_off, axis, sign):
            """Return tuple of base_off shifted by ±e_axis."""
            off = list(base_off)
            off[axis] += sign
            return tuple(off)

        # ----- Precompute gradient index pairs for center + cross nbrs -----
        # For each evaluation point q, grad_pairs[q][b] = (idx_plus, idx_minus)
        # so that gradient along axis b = (f[idx_plus] - f[idx_minus]) / (2*dx_b)

        def _grad_pairs(base_off):
            pairs = {}
            for b in range(ndim):
                ip = offset_to_idx[_shifted(base_off, b, +1)]
                im = offset_to_idx[_shifted(base_off, b, -1)]
                pairs[b] = (ip, im)
            return pairs

        origin = (0,) * ndim
        gp_center = _grad_pairs(origin)

        # Cross neighbours: +e_a and -e_a for each axis a
        gp_cross = {}  # gp_cross[(a, s)] where s ∈ {+1, -1}
        cross_idx = {}  # cross_idx[(a, s)] = stencil index of s*e_a
        for a in range(ndim):
            for s in (+1, -1):
                off_a = [0] * ndim
                off_a[a] = s
                off_a = tuple(off_a)
                cross_idx[(a, s)] = offset_to_idx[off_a]
                gp_cross[(a, s)] = _grad_pairs(off_a)

        # ----- The local function (called inside vmap over grid sites) -----

        def local_lap_grad_sq(Xk, *, mask=None, extras=None):
            # Evaluate inner expression at all stencil neighbours
            fXk = inner_root(Xk)  # (K, n_feat)
            f0 = fXk[center_idx, :]

            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"LaplacianOfGradientSquared requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_2dx = 1.0 / (2.0 * dx)
            inv_dx2 = 1.0 / (dx * dx)

            def _grad_sq(pairs):
                """Compute |∇f|² at a point from its gradient pairs."""
                gsq = jnp.zeros_like(f0)
                for b in range(ndim):
                    ip, im = pairs[b]
                    diff_b = (fXk[ip, :] - fXk[im, :]) * inv_2dx[b]
                    gsq = gsq + diff_b**2
                return gsq

            # |∇f|² at center
            G0 = _grad_sq(gp_center)

            # Laplacian of |∇f|²
            lap_G = jnp.zeros_like(f0)
            for a in range(ndim):
                Gp = _grad_sq(gp_cross[(a, +1)])
                Gm = _grad_sq(gp_cross[(a, -1)])
                lap_G = lap_G + inv_dx2[a] * (Gp + Gm - 2.0 * G0)

            if mask is not None:
                lap_G = lap_G * mask[center_idx]

            return lap_G

        return local_lap_grad_sq

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "∇²|∇|²"
        return [f"{name}({il})" for il in inner_labels[:n_features_out]]


# ============================================================================
# Same-space vector operators (require n_features == ndim)
# ============================================================================


class Divergence(StencilOp):
    r"""Composable divergence :math:`\nabla\!\cdot` on a Cartesian grid.

    .. math::

       [\nabla\!\cdot\mathbf{f}]_i = \sum_{\alpha}
       \frac{f_\alpha(\phi_{i+e_\alpha}) - f_\alpha(\phi_{i-e_\alpha})}
       {2\,\Delta x_\alpha}

    The inner expression must produce exactly ``ndim`` features.

    Examples
    --------
    >>> Div = Divergence(ndim=2, bc="pbc")
    >>> v   = vector_field(n_fields=2)
    >>> div_v = Div(v)           # scalar divergence (1 feature)

    Composition with :class:`Gradient` recovers the Laplacian:

    >>> Grad = Gradient(ndim=2, bc="pbc")
    >>> phi  = field_component(0, n_fields=1)
    >>> lap  = Div(Grad(phi))    # ∇·(∇φ) = ∇²φ
    """

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _n_features_out(self, n_features_inner):
        if n_features_inner != self._ndim:
            raise ValueError(f"Divergence requires n_features == ndim={self._ndim}, got {n_features_inner}")
        return 1

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Same stencil layout as Laplacian (see above)
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        def local_div(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)  # (K, ndim)
            f0 = fXk[0, :]
            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"Divergence requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_2dx = 1.0 / (2.0 * dx)

            result = jnp.zeros(())
            for a in range(ndim):
                result = result + (fXk[plus_idx[a], a] - fXk[minus_idx[a], a]) * inv_2dx[a]

            if mask is not None:
                result = result * mask[0]
            return result

        return local_div

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "∇·"
        return [f"{name}({','.join(inner_labels[: self._ndim])})"]


class Curl(StencilOp):
    r"""Composable curl operator on a Cartesian grid.

    **2D** (scalar curl / vorticity, 1 output feature):

    .. math::

       [\nabla\!\times\!\mathbf{f}]_i =
       \frac{f_1(i\!+\!e_0) - f_1(i\!-\!e_0)}{2\Delta x_0}
       - \frac{f_0(i\!+\!e_1) - f_0(i\!-\!e_1)}{2\Delta x_1}

    **3D** (vector curl, 3 output features):

    .. math::

       (\nabla\!\times\!\mathbf{f})_\alpha
       = \varepsilon_{\alpha\beta\gamma}\,
         \frac{f_\gamma(i\!+\!e_\beta) - f_\gamma(i\!-\!e_\beta)}
              {2\,\Delta x_\beta}

    The inner expression must produce exactly ``ndim`` features.
    Requires ``ndim >= 2``.

    Examples
    --------
    >>> Curl2D = Curl(ndim=2, bc="pbc")
    >>> v = vector_field(n_fields=2)
    >>> omega = Curl2D(v)   # scalar vorticity

    Identity check — curl of a gradient vanishes:

    >>> Grad = Gradient(ndim=2, bc="pbc")
    >>> phi  = field_component(0, n_fields=1)
    >>> should_vanish = Curl2D(Grad(phi))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._ndim < 2:
            raise ValueError(f"Curl requires ndim >= 2 (got {self._ndim})")
        if self._ndim > 3:
            raise ValueError(f"Curl is only defined for ndim 2 or 3 (got {self._ndim})")

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _n_features_out(self, n_features_inner):
        if n_features_inner != self._ndim:
            raise ValueError(f"Curl requires n_features == ndim={self._ndim}, got {n_features_inner}")
        return 1 if self._ndim == 2 else 3

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Same stencil layout as Laplacian (see above)
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        if ndim == 2:

            def local_curl(Xk, *, mask=None, extras=None):
                fXk = inner_root(Xk)
                f0 = fXk[0, :]
                if mask is not None:
                    fXk = jnp.where(mask[:, None], fXk, f0[None, :])

                if extras is None or key_dx not in extras:
                    raise KeyError(f"Curl requires extras[{key_dx!r}]")
                dx = jnp.asarray(extras[key_dx])
                dx = dx if dx.ndim == 1 else jnp.full((2,), dx)
                inv_2dx = 1.0 / (2.0 * dx)

                # df_1/dx_0 - df_0/dx_1
                dfy_dx = (fXk[plus_idx[0], 1] - fXk[minus_idx[0], 1]) * inv_2dx[0]
                dfx_dy = (fXk[plus_idx[1], 0] - fXk[minus_idx[1], 0]) * inv_2dx[1]
                result = jnp.array([dfy_dx - dfx_dy])

                if mask is not None:
                    result = result * mask[0]
                return result

            return local_curl

        # ndim == 3
        def local_curl_3d(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)
            f0 = fXk[0, :]
            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"Curl requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((3,), dx)
            inv_2dx = 1.0 / (2.0 * dx)

            def df(i, j):
                return (fXk[plus_idx[j], i] - fXk[minus_idx[j], i]) * inv_2dx[j]

            curl = jnp.array(
                [
                    df(2, 1) - df(1, 2),
                    df(0, 2) - df(2, 0),
                    df(1, 0) - df(0, 1),
                ]
            )

            if mask is not None:
                curl = curl * mask[0]
            return curl

        return local_curl_3d

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "∇×"
        if self._ndim == 2:
            return [f"{name}({','.join(inner_labels[:2])})"]
        return [f"({name})_{a}" for a in ("x", "y", "z")]


class SymGrad(StencilOp):
    r"""Symmetric gradient (strain-like) on a Cartesian grid.

    Computes the symmetric part of the gradient tensor of a vector field:

    .. math::

       S_{ij} = \tfrac{1}{2}\!\left(
       \frac{\partial f_i}{\partial x_j}
       + \frac{\partial f_j}{\partial x_i}
       \right)

    For velocity fields this is the **strain-rate** tensor; for
    displacement fields the **strain** tensor.

    Output features = ``ndim*(ndim+1)//2`` (upper triangle), ordered:
    ``(0,0), (0,1), ..., (0,d-1), (1,1), (1,2), ..., (d-1,d-1)``.

    The inner expression must produce exactly ``ndim`` features.
    Requires ``ndim >= 2``.

    Examples
    --------
    >>> SG = SymGrad(ndim=2, bc="pbc")
    >>> v  = vector_field(n_fields=2)
    >>> strain = SG(v)   # 3 features: S_00, S_01, S_11
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._ndim < 2:
            raise ValueError(f"SymGrad requires ndim >= 2 (got {self._ndim})")

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _n_features_out(self, n_features_inner):
        ndim = self._ndim
        if n_features_inner != ndim:
            raise ValueError(f"SymGrad requires n_features == ndim={ndim}, got {n_features_inner}")
        return ndim * (ndim + 1) // 2

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Same stencil layout as Laplacian (see above)
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        pairs = [(i, j) for i in range(ndim) for j in range(i, ndim)]

        def local_symgrad(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)
            f0 = fXk[0, :]
            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"SymGrad requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_2dx = 1.0 / (2.0 * dx)

            comps = []
            for i, j in pairs:
                dfi_dxj = (fXk[plus_idx[j], i] - fXk[minus_idx[j], i]) * inv_2dx[j]
                if i == j:
                    comps.append(dfi_dxj)
                else:
                    dfj_dxi = (fXk[plus_idx[i], j] - fXk[minus_idx[i], j]) * inv_2dx[i]
                    comps.append(0.5 * (dfi_dxj + dfj_dxi))

            result = jnp.array(comps)
            if mask is not None:
                result = result * mask[0]
            return result

        return local_symgrad

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "S"
        ndim = self._ndim
        return [f"{name}_{i}{j}" for i in range(ndim) for j in range(i, ndim)]


class SkewGrad(StencilOp):
    r"""Antisymmetric gradient (rotation-like) on a Cartesian grid.

    Computes the antisymmetric part of the gradient tensor:

    .. math::

       W_{ij} = \tfrac{1}{2}\!\left(
       \frac{\partial f_i}{\partial x_j}
       - \frac{\partial f_j}{\partial x_i}
       \right)

    For velocity fields this is the **rotation-rate** (vorticity) tensor.

    Output features = ``ndim*(ndim-1)//2`` (strict upper triangle),
    ordered: ``(0,1), (0,2), ..., (1,2), ...``.

    In 2D the single component equals half the scalar curl.

    The inner expression must produce exactly ``ndim`` features.
    Requires ``ndim >= 2``.

    Examples
    --------
    >>> W  = SkewGrad(ndim=2, bc="pbc")
    >>> v  = vector_field(n_fields=2)
    >>> omega = W(v)   # 1 feature: W_01
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._ndim < 2:
            raise ValueError(f"SkewGrad requires ndim >= 2 (got {self._ndim})")

    def _offsets(self):
        return square_cross_offsets(self._ndim, include_center=True)

    def _default_prefix(self):
        return _default_cross_prefix(self._ndim, self._bc)

    def _n_features_out(self, n_features_inner):
        ndim = self._ndim
        if n_features_inner != ndim:
            raise ValueError(f"SkewGrad requires n_features == ndim={ndim}, got {n_features_inner}")
        return ndim * (ndim - 1) // 2

    def _build_local_fn(self, inner_root, *, n_features_out):
        ndim = self._ndim
        key_dx = self._key_dx
        # Same stencil layout as Laplacian (see above)
        plus_idx  = 1 + 2 * jnp.arange(ndim)
        minus_idx = 2 + 2 * jnp.arange(ndim)

        pairs = [(i, j) for i in range(ndim) for j in range(i + 1, ndim)]

        def local_skewgrad(Xk, *, mask=None, extras=None):
            fXk = inner_root(Xk)
            f0 = fXk[0, :]
            if mask is not None:
                fXk = jnp.where(mask[:, None], fXk, f0[None, :])

            if extras is None or key_dx not in extras:
                raise KeyError(f"SkewGrad requires extras[{key_dx!r}]")
            dx = jnp.asarray(extras[key_dx])
            dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
            inv_2dx = 1.0 / (2.0 * dx)

            comps = []
            for i, j in pairs:
                dfi_dxj = (fXk[plus_idx[j], i] - fXk[minus_idx[j], i]) * inv_2dx[j]
                dfj_dxi = (fXk[plus_idx[i], j] - fXk[minus_idx[i], j]) * inv_2dx[i]
                comps.append(0.5 * (dfi_dxj - dfj_dxi))

            result = jnp.array(comps)
            if mask is not None:
                result = result * mask[0]
            return result

        return local_skewgrad

    def _make_labels(self, label, n_features_out, inner_labels):
        name = label or "W"
        ndim = self._ndim
        return [f"{name}_{i}{j}" for i in range(ndim) for j in range(i + 1, ndim)]


# ============================================================================
# Advection factory
# ============================================================================


def AdvectionBy(v_expr, *, ndim, bc="pbc", **stencil_kwargs):
    r"""Return a stencil operator for advection :math:`(\mathbf{v}\!\cdot\!\nabla)f`.

    Parameters
    ----------
    v_expr : StateExpr
        Pointwise expression with ``n_features == ndim`` giving the
        advecting vector field.
    ndim : int
        Number of spatial dimensions.
    bc : str
        Boundary condition (``"pbc"`` or ``"noflux"``).
    **stencil_kwargs
        Forwarded to ``StencilOp.__init__``.

    Returns
    -------
    StencilOp
        Operator computing :math:`\sum_\alpha v_\alpha\,\partial f / \partial x_\alpha`.

    Notes
    -----
    Results of this operator are **not** further composable because the
    advection FD formula references a second expression (the velocity).

    Examples
    --------
    >>> v   = vector_field(n_fields=2)
    >>> phi = field_component(0, n_fields=2)
    >>> Adv = AdvectionBy(v, ndim=2, bc="pbc")
    >>> adv_phi = Adv(phi)   # (v·∇)φ
    """
    from ..statefunc.stateexpr import StateExpr

    if not isinstance(v_expr, StateExpr):
        raise TypeError(f"v_expr must be a StateExpr, got {type(v_expr).__name__}")
    if v_expr.n_features != ndim:
        raise ValueError(f"v_expr must have n_features == ndim={ndim}, got {v_expr.n_features}")

    v_root = v_expr.root

    class _Advection(StencilOp):
        _composable = False

        def _offsets(self):
            return square_cross_offsets(self._ndim, include_center=True)

        def _default_prefix(self):
            return _default_cross_prefix(self._ndim, self._bc)

        def _build_local_fn(self, inner_root, *, n_features_out):
            _ndim = self._ndim
            _key_dx = self._key_dx
            _v_root = v_root
            _plus_idx = 1 + 2 * jnp.arange(_ndim)
            _minus_idx = 2 + 2 * jnp.arange(_ndim)

            def local_advection(Xk, *, mask=None, extras=None):
                fXk = inner_root(Xk)  # (K, n_feat_f)
                f0 = fXk[0, :]
                vXk = _v_root(Xk)  # (K, ndim)
                v_center = vXk[0, :]  # (ndim,)

                if mask is not None:
                    fXk = jnp.where(mask[:, None], fXk, f0[None, :])

                if extras is None or _key_dx not in extras:
                    raise KeyError(f"AdvectionBy requires extras[{_key_dx!r}]")
                dx = jnp.asarray(extras[_key_dx])
                dx = dx if dx.ndim == 1 else jnp.full((_ndim,), dx)
                inv_2dx = 1.0 / (2.0 * dx)

                fp = fXk[_plus_idx, :]  # (ndim, n_feat_f)
                fm = fXk[_minus_idx, :]
                grad_f = (fp - fm) * inv_2dx[:, None]  # (ndim, n_feat_f)
                result = jnp.sum(
                    v_center[:, None] * grad_f,
                    axis=0,
                )  # (n_feat_f,)

                if mask is not None:
                    result = result * mask[0]
                return result

            return local_advection

        def _make_labels(self, label, n_features_out, inner_labels):
            name = label or "(v·∇)"
            return [f"{name}({il})" for il in inner_labels[:n_features_out]]

        def __repr__(self):
            return f"AdvectionBy(ndim={self._ndim}, bc={self._bc!r})"

    return _Advection(ndim=ndim, bc=bc, **stencil_kwargs)


# ============================================================================
# Convenience helpers
# ============================================================================


def vector_field(n_fields: int, *, labels: list[str] | None = None):
    """All field components as a rank-0 expression with ``n_features == n_fields``.

    Shorthand for concatenating :func:`~SFI.bases.linear.field_component`
    over all indices.  Useful as input to same-space operators
    (:class:`Divergence`, :class:`Curl`, :class:`SymGrad`, …).

    Parameters
    ----------
    n_fields : int
        Number of field components per grid site.
    labels : list of str, optional
        Human-readable labels; defaults to ``field[0], field[1], …``.

    Returns
    -------
    Basis
        ``n_features == n_fields``, ``rank == 0``.
    """
    from .linear import x_coordinates

    if labels is None:
        labels = [f"field[{i}]" for i in range(n_fields)]
    return x_coordinates(list(range(n_fields)), dim=n_fields, labels=labels)


# ============================================================================
# Noise model factories  (grid-aware wrappers around langevin.noise)
# ============================================================================


def conserved_noise_pbc(
    sigma: float,
    *,
    grid_shape: "Sequence[int]",
    dx: "float | Sequence[float]" = 1.0,
    n_fields: int = 1,
) -> "ConservedNoise":  # noqa: F821
    r"""Build a conserved-noise model for a periodic square grid.

    Returns a :class:`~SFI.langevin.noise.ConservedNoise` instance configured
    for the given grid geometry.  The noise implements

    .. math::

       \eta(\mathbf{x}, t) = \nabla\!\cdot\!(\sigma\,\vec{\Lambda})

    where :math:`\vec{\Lambda}` is spatiotemporal white vector noise with
    periodic boundary conditions.  This conserves the spatial average:
    :math:`\sum_i \eta_i = 0` exactly at every time step.

    Parameters
    ----------
    sigma : float
        Continuum noise amplitude.
    grid_shape : sequence of int
        Grid dimensions ``(Nx, Ny, ...)``.
    dx : float or sequence of float
        Grid spacing (uniform or per-axis).
    n_fields : int
        Number of field components per site.

    Returns
    -------
    ConservedNoise
        Ready to pass as ``D=`` to :class:`~SFI.langevin.OverdampedProcess`.

    Example
    -------
    >>> from SFI.bases.spde import conserved_noise_pbc, square_grid_extras
    >>> noise = conserved_noise_pbc(sigma=0.3, grid_shape=(64, 64), dx=1.0)
    >>> proc  = OverdampedProcess(basis, D=noise)
    """
    from ..langevin.noise import ConservedNoise

    return ConservedNoise(sigma, grid_shape=grid_shape, dx=dx, n_fields=n_fields)
