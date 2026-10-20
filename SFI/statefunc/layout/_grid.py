"""GridLayout — structured-dimensions layout for SPDE systems.

Provides :class:`GridLayout`, a layout for regular-grid SPDE systems that
carries ``ndim`` (spatial dimension) and stencil parameters (``bc``, ``order``).
Differential operators are exposed as methods (``grad``, ``lap``, ``div``,
``biharmonic``, etc.).  The :meth:`~GridLayout.embed` method compiles
inner-world :class:`~SFI.statefunc.structexpr.StructuredExpr` trees into
outer-world :class:`~SFI.statefunc.stateexpr.StateExpr` objects.

Generic stencil composition
---------------------------
Any expression tree — including nested differential operators like
``layout.div(Q * layout.grad(phi))`` — compiles to a *single* fat-stencil
gather + locally-unrolled evaluation function.  The stencil footprint is
computed via Minkowski sums of per-operator template offsets (see
:mod:`._fd_atoms`), and the evaluation function is recursively compiled
with trace-time offset resolution (see :mod:`._eval_compiler`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp

from ..nodes.contract import Rank
from ..structexpr import (
    StructuredExpr,
    _BinaryOp,
    _ConcatOp,
    _ConstNode,
    _DiffOpNode,
    _EinsumOp,
    _SectorLeaf,
    _SliceOp,
    _StackOp,
    _StructNode,
    _UnaryOp,
)
from ._base import _BaseLayout
from ._sectors import (
    ScalarSector,
    Sector,
    SymTensorSector,
    TensorSector,
    VectorSector,
)

# =====================================================================
# StencilMeta — opaque metadata attached to _DiffOpNode
# =====================================================================


@dataclass(frozen=True, slots=True)
class StencilMeta:
    """Metadata for a finite-difference differential operator node.

    Stored on ``_DiffOpNode.engine_meta`` and consumed by the embed
    compiler to wire stencil dispatch.
    """

    op_name: str
    ndim: int
    bc: str
    order: str
    key_grid_shape: str
    key_dx: str


# =====================================================================
# Voigt helpers
# =====================================================================


def _voigt_expand(raw: jax.Array, sector: SymTensorSector) -> jax.Array:
    """Expand Voigt-packed ``(n_data,)`` to full ``(sdim, sdim)`` tensor.

    For traceless sectors, the last diagonal is reconstructed as
    ``-sum(other diagonals)``.
    """
    sdim = sector.sdim
    pairs = sector.voigt_pairs

    # Build full symmetric matrix
    mat = jnp.zeros((sdim, sdim), dtype=raw.dtype)
    for k, (i, j) in enumerate(pairs):
        mat = mat.at[i, j].set(raw[k])
        if i != j:
            mat = mat.at[j, i].set(raw[k])

    if sector.traceless:
        # Reconstruct last diagonal: -(sum of other diags)
        diag_sum = jnp.zeros((), dtype=raw.dtype)
        for d in range(sdim - 1):
            diag_sum = diag_sum + mat[d, d]
        mat = mat.at[sdim - 1, sdim - 1].set(-diag_sum)

    return mat


def _voigt_pack(mat: jax.Array, sector: SymTensorSector) -> jax.Array:
    """Extract independent Voigt components from ``(sdim, sdim)`` tensor."""
    return jnp.array([mat[i, j] for i, j in sector.voigt_pairs])


def _sector_pack(val: jax.Array, sector: Sector) -> jax.Array:
    """Pack a structured value to flat ``(n_data,)`` for a sector.

    Handles Voigt packing for SymTensorSector, flat reshape for others.
    """
    if isinstance(sector, SymTensorSector):
        return _voigt_pack(val, sector)
    elif isinstance(sector, ScalarSector):
        return val.reshape(1)
    else:
        # VectorSector, TensorSector: flat reshape
        return val.reshape(-1)


def _sector_expand(raw: jax.Array, sector: Sector) -> jax.Array:
    """Expand flat ``(n_data,)`` to structured ``(*sdims)``."""
    if isinstance(sector, SymTensorSector):
        return _voigt_expand(raw, sector)
    elif isinstance(sector, ScalarSector):
        return raw[0]
    elif isinstance(sector, VectorSector):
        return raw
    elif isinstance(sector, TensorSector):
        return raw.reshape(sector.sdims)
    else:
        raise TypeError(f"Unknown sector type: {type(sector)}")


# =====================================================================
# Pointwise tree compiler
# =====================================================================


def _compile_pointwise_fn(
    node: _StructNode,
    sectors: dict[str, Sector],
) -> Callable:
    """Compile a pointwise _StructNode tree to ``fn(x) -> (*sdims) array``.

    The resulting function takes a single state vector ``x: (dim,)`` and
    returns the structured value with shape ``(*sdims)``.

    Only used for *pure-pointwise* paths (no ``_DiffOpNode`` in tree).
    Trees with stencil nodes go through :func:`._eval_compiler.compile_eval`.
    """
    if isinstance(node, _SectorLeaf):
        indices = jnp.array(node.indices)
        sector = sectors[node.sector_name]
        if isinstance(sector, SymTensorSector):

            def _fn(x):
                return _voigt_expand(x[indices], sector)

            return _fn
        elif isinstance(sector, ScalarSector):
            idx0 = node.indices[0]

            def _fn(x):
                return x[idx0]

            return _fn
        else:
            # VectorSector or TensorSector
            def _fn(x):
                raw = x[indices]
                if node.sdims:
                    return raw.reshape(node.sdims)
                return raw

            return _fn

    if isinstance(node, _ConstNode):
        val = node.value
        if isinstance(val, (int, float)):
            arr = jnp.array(val, dtype=jnp.float32)

            def _fn(x):
                return arr.astype(x.dtype)

            return _fn
        else:
            arr = jnp.asarray(val)

            def _fn(x):
                return arr

            return _fn

    if isinstance(node, _BinaryOp):
        left_fn = _compile_pointwise_fn(node.left, sectors)
        right_fn = _compile_pointwise_fn(node.right, sectors)
        op_map = {
            "+": jnp.add,
            "-": jnp.subtract,
            "*": jnp.multiply,
            "/": jnp.true_divide,
            "**": jnp.power,
        }
        op_fn = op_map[node.op]

        def _fn(x):
            return op_fn(left_fn(x), right_fn(x))

        return _fn

    if isinstance(node, _EinsumOp):
        child_fns = [_compile_pointwise_fn(c, sectors) for c in node.children]
        spec = node.spec

        def _fn(x):
            operands = [cfn(x) for cfn in child_fns]
            return jnp.einsum(spec, *operands)

        return _fn

    if isinstance(node, _UnaryOp):
        child_fn = _compile_pointwise_fn(node.child, sectors)
        if node.op == "neg":

            def _fn(x):
                return -child_fn(x)

            return _fn
        elif node.op == "T":

            def _fn(x):
                v = child_fn(x)
                if v.ndim >= 2:
                    return jnp.swapaxes(v, -2, -1)
                return v

            return _fn
        elif node.op == "ew":
            ew = node.fn
            assert ew is not None

            def _fn(x):
                return ew(child_fn(x))

            return _fn
        else:
            raise ValueError(f"Unknown unary op: {node.op!r}")

    if isinstance(node, _StackOp):
        child_fns = [_compile_pointwise_fn(c, sectors) for c in node.children]

        def _fn(x):
            return jnp.stack([cfn(x) for cfn in child_fns], axis=0)

        return _fn

    if isinstance(node, _SliceOp):
        child_fn = _compile_pointwise_fn(node.child, sectors)
        idx = node.idx

        def _fn(x):
            v = child_fn(x)
            return v[..., idx]

        return _fn

    raise NotImplementedError(f"Cannot compile {type(node).__name__} as pointwise.")


# =====================================================================
# GridLayout
# =====================================================================


class GridLayout(_BaseLayout):
    """Layout for SPDE systems on regular Cartesian grids.

    Provides differential operator methods and an ``embed()`` compiler
    that turns inner-world :class:`~SFI.statefunc.structexpr.StructuredExpr`
    trees into outer-world ``Basis`` / ``PSF`` objects.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    bc : {``"pbc"``, ``"noflux"``}
        Boundary condition for stencil operators.
    order : str
        Grid flattening order (``"C"`` default).
    key_grid_shape : str
        Extras key for the grid shape array.
    key_dx : str
        Extras key for the grid spacing.
    **sectors : Sector
        Named sectors (keyword arguments).

    Examples
    --------
    >>> layout = GridLayout(
    ...     velocity=VectorSector([0, 1], sdim=2, spatial=True),
    ...     Q=SymTensorSector([2, 3], sdim=2, traceless=True),
    ...     dim=4, ndim=2, bc="pbc",
    ... )
    >>> v = layout.velocity
    >>> Q = layout.Q
    >>> force = layout.embed(rank=1,
    ...     velocity=layout.lap(v),
    ...     Q=layout.lap(Q),
    ... )
    """

    def __init__(
        self,
        *,
        dim: int,
        ndim: int,
        bc: Literal["pbc", "noflux"] = "pbc",
        order: Literal["C", "F"] = "C",
        key_grid_shape: str = "box/grid_shape",
        key_dx: str = "box/dx",
        **sectors: Sector,
    ) -> None:
        self._ndim = int(ndim)
        self._bc: Literal["pbc", "noflux"] = bc
        self._order: Literal["C", "F"] = order
        self._key_grid_shape = key_grid_shape
        self._key_dx = key_dx
        super().__init__(dim=dim, **sectors)
        self._validate_spatial_sectors()

    # --- validation ---------------------------------------------------

    def _validate_spatial_sectors(self) -> None:
        for name, sector in self._sectors.items():
            if isinstance(sector, VectorSector) and sector.spatial:
                if sector.sdim != self._ndim:
                    raise ValueError(f"Sector '{name}' is spatial but sdim={sector.sdim} != ndim={self._ndim}")

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self._ndim

    @property
    def bc(self) -> Literal["pbc", "noflux"]:
        """Boundary condition."""
        return self._bc

    # --- private: create StencilMeta ----------------------------------

    def _make_meta(self, op_name: str) -> StencilMeta:
        return StencilMeta(
            op_name=op_name,
            ndim=self._ndim,
            bc=self._bc,
            order=self._order,
            key_grid_shape=self._key_grid_shape,
            key_dx=self._key_dx,
        )

    # --- private: validate expression belongs to this layout ----------

    def _check_layout(self, expr: StructuredExpr, method: str) -> None:
        if expr._layout_id != self._layout_id:
            from ..structexpr import _FREE_LAYOUT

            if expr._layout_id != _FREE_LAYOUT:
                raise TypeError(
                    f"{method}: expression belongs to a different layout (id {expr._layout_id} vs {self._layout_id})"
                )

    # ==================================================================
    # Differential operator methods
    # ==================================================================

    # ------------------------------------------------------------------
    # Internal label helper
    # ------------------------------------------------------------------

    @staticmethod
    def _op_label(prefix: str, expr: StructuredExpr, suffix: str = "") -> tuple:
        """Return a single-element label tuple for a unary differential op.

        Uses the operand's existing single label if available; otherwise
        returns an empty tuple so the fallback in ``embed`` is used.
        """
        if expr.labels and len(expr.labels) == 1:
            return (f"{prefix}{expr.labels[0]}{suffix}",)
        return ()

    def grad(self, expr: StructuredExpr) -> StructuredExpr:
        """Spatial gradient.  Adds an ``(ndim,)`` trailing axis.

        Input sdims ``S`` → output sdims ``S + (ndim,)``.
        """
        self._check_layout(expr, "grad")
        out_sdims = expr.sdims + (self._ndim,)
        return expr._new(
            sdims=out_sdims,
            n_features=expr.n_features,
            labels=self._op_label("∇", expr),
            node=_DiffOpNode(
                op_name="grad",
                child=expr._node,
                engine_meta=self._make_meta("grad"),
            ),
        )

    def lap(self, expr: StructuredExpr) -> StructuredExpr:
        """Laplacian ∇².  Rank-preserving (same sdims as input)."""
        self._check_layout(expr, "lap")
        return expr._new(
            sdims=expr.sdims,
            n_features=expr.n_features,
            labels=self._op_label("∇²", expr),
            node=_DiffOpNode(
                op_name="lap",
                child=expr._node,
                engine_meta=self._make_meta("lap"),
            ),
        )

    def div(self, expr: StructuredExpr) -> StructuredExpr:
        """Divergence.  Contracts the last axis (must be ``ndim``).

        Input sdims ``(*S, ndim)`` → output sdims ``S``.
        """
        self._check_layout(expr, "div")
        if not expr.sdims or expr.sdims[-1] != self._ndim:
            raise ValueError(f"div requires last sdim == ndim={self._ndim}, got sdims={expr.sdims}")
        out_sdims = expr.sdims[:-1]
        return expr._new(
            sdims=out_sdims,
            n_features=expr.n_features,
            labels=self._op_label("∇·", expr),
            node=_DiffOpNode(
                op_name="div",
                child=expr._node,
                engine_meta=self._make_meta("div"),
            ),
        )

    def biharmonic(self, expr: StructuredExpr) -> StructuredExpr:
        """Biharmonic ∇⁴.  Rank-preserving.  Requires ``ndim >= 2``."""
        self._check_layout(expr, "biharmonic")
        if self._ndim < 2:
            raise ValueError("biharmonic requires ndim >= 2")
        return expr._new(
            sdims=expr.sdims,
            n_features=expr.n_features,
            labels=self._op_label("∇⁴", expr),
            node=_DiffOpNode(
                op_name="biharmonic",
                child=expr._node,
                engine_meta=self._make_meta("biharmonic"),
            ),
        )

    def lap_of_grad_sq(self, expr: StructuredExpr) -> StructuredExpr:
        r"""∇²(\|∇f\|²) — the Active Model B+ term.

        Convenience sugar for ``self.lap(self.grad(expr).dot(self.grad(expr)))``.
        Compiles to the biharmonic-radius stencil automatically via footprint
        computation.  Requires ``ndim >= 2``.
        """
        self._check_layout(expr, "lap_of_grad_sq")
        if self._ndim < 2:
            raise ValueError("lap_of_grad_sq requires ndim >= 2")
        g = self.grad(expr)
        return self.lap(g.dot(g))

    # ------------------------------------------------------------------
    # Convenience differential operator methods
    # ------------------------------------------------------------------

    def advection_by(
        self,
        velocity: StructuredExpr,
        transported: StructuredExpr,
    ) -> StructuredExpr:
        r"""Advection :math:`(\mathbf{v}\cdot\nabla)\phi`.

        *velocity* must have sdims ``(…, ndim)`` and *transported* is
        arbitrary.  Contracts the spatial gradient with the velocity
        vector.
        """
        self._check_layout(velocity, "advection_by")
        self._check_layout(transported, "advection_by")
        v_lbl = velocity.labels[0] if len(velocity.labels) == 1 else "v"
        f_lbl = transported.labels[0] if len(transported.labels) == 1 else "f"
        g = self.grad(transported)  # sdims: (*T, ndim)
        # Contract the trailing ndim axis of grad with trailing ndim of v.
        result = g.dot(velocity)
        return result._new(
            labels=(f"({v_lbl}·∇){f_lbl}",),
            node=result._node,
        )

    def strain_rate(self, v_expr: StructuredExpr) -> StructuredExpr:
        r"""Symmetric strain rate :math:`E = (\nabla v + (\nabla v)^T)/2`.

        Input: vector ``(ndim,)``. Output: symmetric tensor ``(ndim, ndim)``.
        """
        self._check_layout(v_expr, "strain_rate")
        v_lbl = v_expr.labels[0] if len(v_expr.labels) == 1 else "v"
        gv = self.grad(v_expr)  # (ndim, ndim)
        result = (gv + gv.T) * 0.5
        return result._new(labels=(f"E[{v_lbl}]",), node=result._node)

    def vorticity(self, v_expr: StructuredExpr) -> StructuredExpr:
        r"""Antisymmetric vorticity :math:`\Omega = (\nabla v - (\nabla v)^T)/2`.

        Input: vector ``(ndim,)``. Output: antisymmetric tensor ``(ndim, ndim)``.
        """
        self._check_layout(v_expr, "vorticity")
        gv = self.grad(v_expr)  # (ndim, ndim)
        return (gv - gv.T) * 0.5

    # ==================================================================
    # embed() — the compilation boundary
    # ==================================================================

    def embed(
        self,
        rank: int = 1,
        **named_fields: StructuredExpr,
    ) -> Any:
        """Compile inner-world expressions into an outer-world Basis/PSF.

        Parameters
        ----------
        rank : int
            Output rank (1 for forces, 2 for diffusion matrices).
        **named_fields
            ``sector_name=expression`` pairs.  Each expression's sdims
            must match the corresponding sector's sdims.

        Returns
        -------
        Basis
            Outer-world expression ready for inference.
        """
        if rank not in (1, 2):
            raise ValueError(f"Only rank=1 and rank=2 are supported, got {rank}")

        # Validate sector names and sdims
        for name, expr in named_fields.items():
            if name not in self._sectors:
                raise ValueError(f"Unknown sector '{name}'. Available: {list(self._sectors.keys())}")
            sector = self._sectors[name]
            if expr.sdims != sector.sdims:
                raise ValueError(f"Sector '{name}' has sdims={sector.sdims}, but expression has sdims={expr.sdims}")

        # Compile each sector
        sector_bases = []
        for name, expr in named_fields.items():
            sector = self._sectors[name]
            compiled = self._compile_sector(expr._node, name, sector, rank, labels=list(expr.labels))
            sector_bases.append(compiled)

        # Combine all sectors with feature concatenation
        if not sector_bases:
            raise ValueError("At least one sector expression is required")

        result = sector_bases[0]
        for sb in sector_bases[1:]:
            result = result & sb
        return result

    # ------------------------------------------------------------------
    # Sector compilation — footprint-based dispatch
    # ------------------------------------------------------------------

    def _compile_sector(
        self,
        node: _StructNode,
        sector_name: str,
        sector: Sector,
        rank: int,
        labels: list | None = None,
    ) -> Any:
        """Compile a single sector's _StructNode tree to outer StateExpr.

        Uses footprint analysis: if the tree reads only the local site
        (footprint = {origin}), compile as pure-pointwise.  Otherwise
        compile the *entire tree* — including nested stencil ops,
        nonlinear algebra, etc. — as a single fat-stencil dispatch.

        Parameters
        ----------
        labels : list[str] or None
            Human-readable term labels propagated from the outer
            ``StructuredExpr.labels``.  One label per feature (i.e. per
            term after ``&``-concatenation).  ``None`` / empty list falls
            back to ``"stencil(<sector_name>)"``.
        """
        from ._fd_atoms import _origin, compute_footprint

        labels = labels or []

        # Handle _ConcatOp at top level (feature concatenation)
        if isinstance(node, _ConcatOp):
            parts = []
            for i, c in enumerate(node.children):
                child_lbl = [labels[i]] if i < len(labels) else []
                parts.append(self._compile_sector(c, sector_name, sector, rank, labels=child_lbl))
            result = parts[0]
            for p in parts[1:]:
                result = result & p
            return result

        footprint = compute_footprint(node, self._ndim)
        origin = _origin(self._ndim)

        if footprint == origin:
            # Pure pointwise — no stencil needed
            return self._compile_pointwise_term(node, sector_name, sector, rank, labels=labels)
        else:
            # Stencil tree (may include nested diff ops + algebra)
            return self._compile_stencil_tree(node, footprint, sector_name, sector, rank, labels=labels)

    # ------------------------------------------------------------------
    # Stencil tree compilation (generic, supports nesting)
    # ------------------------------------------------------------------

    def _compile_stencil_tree(
        self,
        node: _StructNode,
        footprint,
        sector_name: str,
        sector: Sector,
        rank: int,
        labels: list | None = None,
    ) -> Any:
        """Compile an arbitrary tree with stencil nodes to outer Basis.

        1. Convert footprint → sorted offset array + o2i dict.
        2. Recursively compile the tree via ``_eval_compiler.compile_eval``.
        3. Wrap the eval function with sector pack + scatter.
        4. Dispatch via ``make_interactor`` + ``PreparedSquareStencilFromBox``.
        """
        from ..factory import make_interactor
        from ..nodes.interactions.stencils import PreparedSquareStencilFromBox
        from ._eval_compiler import compile_eval, wrap_as_local_fn
        from ._fd_atoms import build_offset_to_idx, offsets_to_sorted_array

        ndim = self._ndim
        offsets_array = offsets_to_sorted_array(footprint)
        o2i = build_offset_to_idx(offsets_array)

        # Build the recursive eval function (trace-time offset resolution)
        eval_fn = compile_eval(node, o2i, self._sectors, ndim, self._key_dx)

        # Wrap: mask pre-processing + evaluation at origin
        raw_local_fn = wrap_as_local_fn(eval_fn, ndim, o2i)

        # Determine output sdims
        out_sdims = _node_sdims(node, self._sectors)

        # Wrap with sector pack + scatter to dim-wide output
        local_fn = self._wrap_scatter(raw_local_fn, out_sdims, sector, rank)

        K = int(offsets_array.shape[0])

        # Determine stencil cache prefix from the footprint size
        # (The prefix is only needed for the hyper-table cache key)
        prefix = f"_cache/stencil/composed_{ndim}d_{self._bc}_K{K}"

        labels = labels or []
        label_str = labels[0] if labels else f"stencil({sector_name})"
        inter = make_interactor(
            local_fn,
            dim=self._dim,
            rank=Rank(rank),
            K=K,
            n_features=1,
            labels=[label_str],
        )

        spec = PreparedSquareStencilFromBox(
            offsets=jnp.asarray(offsets_array),
            bc=self._bc,
            order=self._order,
            key_grid_shape=self._key_grid_shape,
            key_prefix=prefix,
            include_slot_extras_offset=True,
        )

        return inter.dispatch(
            spec,
            owners="focal",
            focal_index=0,
            reducer="sum",
            return_as="basis",
        )

    # ------------------------------------------------------------------
    # Pointwise term compilation
    # ------------------------------------------------------------------

    def _compile_pointwise_term(
        self,
        node: _StructNode,
        sector_name: str,
        sector: Sector,
        rank: int,
        labels: list | None = None,
    ) -> Any:
        """Compile a pure pointwise _StructNode to outer Basis."""
        from ..factory import make_basis

        labels = labels or []
        label_str = labels[0] if labels else sector_name

        pw_fn = _compile_pointwise_fn(node, self._sectors)
        dim = self._dim
        indices = jnp.array(sector.indices)

        if rank == 1:

            def fn(x):
                val = pw_fn(x)  # (*sdims)
                packed = _sector_pack(val, sector)  # (n_data,)
                out = jnp.zeros(dim, dtype=x.dtype)
                out = out.at[indices].set(packed)
                return out  # (dim,) → rank=1, n_features=1

            return make_basis(
                fn,
                dim=self._dim,
                rank=1,
                n_features=1,
                labels=[label_str],
            )
        elif rank == 2:
            # Block-diagonal embedding
            def fn(x):
                val = pw_fn(x)  # (*sdims)
                packed = _sector_pack(val, sector)  # (n_data,)
                out = jnp.zeros((dim, dim), dtype=x.dtype)
                # Place as diagonal block
                for i_local, i_global in enumerate(sector.indices):
                    out = out.at[i_global, i_global].set(packed[i_local] if i_local < len(packed) else 0.0)
                return out  # (dim, dim) → rank=2

            return make_basis(
                fn,
                dim=self._dim,
                rank=2,
                n_features=1,
                labels=[label_str],
            )
        else:
            raise ValueError(f"Unsupported rank={rank}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_scatter(
        self,
        fd_fn: Callable,
        out_sdims: tuple[int, ...],
        sector: Sector,
        rank: int,
    ) -> Callable:
        """Wrap an FD local function with sector scatter to dim-wide output."""
        dim = self._dim
        indices = jnp.array(sector.indices)

        if rank == 1:

            def local_fn(Xk, *, mask=None, extras=None):
                fd_result = fd_fn(Xk, mask=mask, extras=extras)  # (n_flat_out,)
                # Reshape to structured
                structured = fd_result.reshape(out_sdims) if out_sdims else fd_result
                # Pack for sector
                packed = _sector_pack(structured, sector)  # (n_data,)
                # Scatter to dim-wide
                out = jnp.zeros(dim, dtype=fd_result.dtype)
                out = out.at[indices].set(packed)
                return out  # (dim,)  → rank=1, n_features=1

            return local_fn
        elif rank == 2:

            def local_fn(Xk, *, mask=None, extras=None):
                fd_result = fd_fn(Xk, mask=mask, extras=extras)
                structured = fd_result.reshape(out_sdims) if out_sdims else fd_result
                packed = _sector_pack(structured, sector)
                out = jnp.zeros((dim, dim), dtype=fd_result.dtype)
                for i_local, i_global in enumerate(sector.indices):
                    out = out.at[i_global, i_global].set(packed[i_local] if i_local < len(packed) else 0.0)
                return out  # (dim, dim)

            return local_fn
        else:
            raise ValueError(f"Unsupported rank={rank}")

    def __repr__(self) -> str:
        parts = [f"dim={self._dim}", f"ndim={self._ndim}", f"bc={self._bc!r}"]
        for name, sector in self._sectors.items():
            parts.append(f"{name}={sector!r}")
        return f"GridLayout({', '.join(parts)})"


# =====================================================================
# Tree utilities
# =====================================================================


def _node_sdims(
    node: _StructNode,
    sectors: dict[str, Sector],
) -> tuple[int, ...]:
    """Infer sdims of a node from the tree structure."""
    if isinstance(node, _SectorLeaf):
        return node.sdims
    if isinstance(node, _ConstNode):
        return node.sdims
    if isinstance(node, _BinaryOp):
        left_s = _node_sdims(node.left, sectors)
        right_s = _node_sdims(node.right, sectors)
        # Broadcasting: scalar with anything -> anything
        if left_s == ():
            return right_s
        if right_s == ():
            return left_s
        return left_s  # should match
    if isinstance(node, _UnaryOp):
        child_s = _node_sdims(node.child, sectors)
        if node.op == "T" and len(child_s) >= 2:
            return child_s[:-2] + (child_s[-1], child_s[-2])
        return child_s
    if isinstance(node, _EinsumOp):
        # Parse the spec to get output sdims
        from ..structexpr import _parse_einsum

        operand_specs, rhs = _parse_einsum(node.spec)
        letter_size: dict[str, int] = {}
        for spec_str, child in zip(operand_specs, node.children):
            child_sdims = _node_sdims(child, sectors)
            for ch, sz in zip(spec_str, child_sdims):
                letter_size[ch] = sz
        return tuple(letter_size[ch] for ch in rhs)
    if isinstance(node, _StackOp):
        return (node.sdim,) + _node_sdims(node.children[0], sectors)
    if isinstance(node, _SliceOp):
        return _node_sdims(node.child, sectors)[:-1]
    if isinstance(node, _ConcatOp):
        return _node_sdims(node.children[0], sectors)
    if isinstance(node, _DiffOpNode):
        child_s = _node_sdims(node.child, sectors)
        meta = node.engine_meta
        return _op_output_sdims(node.op_name, child_s, meta.ndim)
    return ()


def _op_output_sdims(
    op_name: str,
    child_sdims: tuple[int, ...],
    ndim: int,
) -> tuple[int, ...]:
    """Compute output sdims for a differential operator."""
    if op_name == "lap":
        return child_sdims
    elif op_name == "grad":
        return child_sdims + (ndim,)
    elif op_name == "div":
        return child_sdims[:-1]
    elif op_name == "biharmonic":
        return child_sdims
    else:
        raise ValueError(f"Unknown op: {op_name!r}")
