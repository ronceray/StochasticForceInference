"""Recursive stencil-tree compiler for GridLayout.

Given an arbitrary ``_StructNode`` tree (possibly containing nested
``_DiffOpNode`` s mixed with pointwise algebra) and a pre-computed
**offset-to-index** dict, this module compiles the tree into a single
JAX local function ``local_fn(Xk, *, mask, extras)``  that reads the
gathered data ``Xk: (K, dim)`` and produces the output for one grid
site.

Architecture
------------

The core abstraction is **base_offset**:  a *Python tuple* of ints
that represents "the grid offset at which we are evaluating this
sub-expression".  It starts at ``(0, …, 0)`` for the outermost call
and gets shifted when we descend into a ``_DiffOpNode``:  the
gradient needs values at ``base + e_a`` and ``base − e_a``, so the
child is called with those shifted base offsets.

Because ``base_offset`` is a Python tuple and ``o2i`` lookup is done
at **trace time**, all stencil index resolution becomes static integer
constants in the XLA graph — no runtime dict lookups.

Layout-independence
-------------------

This compiler is specific to **finite-difference** stencils on a grid.
A future PINN layout would implement the same ``_DiffOpNode`` IR nodes
but compile them via ``jax.grad`` instead of FD formulas.  The
``StructuredExpr`` algebra and IR tree are layout-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

# =====================================================================
# Offset arithmetic (trace-time only — pure Python)
# =====================================================================


def _shift(base: tuple[int, ...], axis: int, sign: int) -> tuple[int, ...]:
    """Shift *base* by ±e_axis."""
    lst = list(base)
    lst[axis] += sign
    return tuple(lst)


# =====================================================================
# Recursive evaluator builder
# =====================================================================


def compile_eval(
    node: Any,
    o2i: dict[tuple[int, ...], int],
    sectors: dict[str, Any],
    ndim: int,
    key_dx: str,
) -> Callable:
    """Compile a ``_StructNode`` tree into a trace-time-unrolled evaluator.

    Returns ``eval_fn(Xk, extras, mask, base_offset) -> jnp.array``
    where ``Xk: (K, dim)`` is the gathered neighbour data and
    ``base_offset`` is a Python int-tuple resolved at trace time.

    The returned function is meant to be called at the top level with
    ``base_offset = (0,) * ndim``.
    """
    from ..structexpr import (
        _BinaryOp,
        _ConcatOp,
        _ConstNode,
        _DiffOpNode,
        _EinsumOp,
        _SectorLeaf,
        _SliceOp,
        _StackOp,
        _UnaryOp,
    )
    from ._grid import _voigt_expand  # reuse existing helper

    # ------------------------------------------------------------------
    # Leaf: read from Xk at o2i[base_offset]
    # ------------------------------------------------------------------

    if isinstance(node, _SectorLeaf):
        indices = tuple(node.indices)
        sector = sectors[node.sector_name]
        sdims = node.sdims

        # Check if it's a SymTensor (needs Voigt expansion)
        from ._sectors import ScalarSector, SymTensorSector

        if isinstance(sector, SymTensorSector):

            def _eval(Xk, extras, mask, base_offset):
                idx = o2i[base_offset]
                raw = Xk[idx]  # (dim,)
                vals = jnp.array([raw[i] for i in indices])
                return _voigt_expand(vals, sector)

            return _eval

        elif isinstance(sector, ScalarSector):
            i0 = indices[0]

            def _eval(Xk, extras, mask, base_offset):
                idx = o2i[base_offset]
                return Xk[idx, i0]

            return _eval

        else:
            # VectorSector / TensorSector
            def _eval(Xk, extras, mask, base_offset):
                idx = o2i[base_offset]
                raw = jnp.array([Xk[idx, i] for i in indices])
                if sdims:
                    return raw.reshape(sdims)
                return raw

            return _eval

    # ------------------------------------------------------------------
    # Constant: independent of Xk and base_offset
    # ------------------------------------------------------------------

    if isinstance(node, _ConstNode):
        val = node.value
        if isinstance(val, (int, float)):
            arr = jnp.array(val, dtype=jnp.float32)

            def _eval(Xk, extras, mask, base_offset):
                return arr.astype(Xk.dtype)

            return _eval
        else:
            arr = jnp.asarray(val)

            def _eval(Xk, extras, mask, base_offset):
                return arr

            return _eval

    # ------------------------------------------------------------------
    # Binary op: + - * / **
    # ------------------------------------------------------------------

    if isinstance(node, _BinaryOp):
        left_fn = compile_eval(node.left, o2i, sectors, ndim, key_dx)
        right_fn = compile_eval(node.right, o2i, sectors, ndim, key_dx)
        _ops = {
            "+": jnp.add,
            "-": jnp.subtract,
            "*": jnp.multiply,
            "/": jnp.true_divide,
            "**": jnp.power,
        }
        op_fn = _ops[node.op]

        def _eval(Xk, extras, mask, base_offset):
            return op_fn(
                left_fn(Xk, extras, mask, base_offset),
                right_fn(Xk, extras, mask, base_offset),
            )

        return _eval

    # ------------------------------------------------------------------
    # Unary op: neg, T, ew (element-wise function)
    # ------------------------------------------------------------------

    if isinstance(node, _UnaryOp):
        child_fn = compile_eval(node.child, o2i, sectors, ndim, key_dx)

        if node.op == "neg":

            def _eval(Xk, extras, mask, base_offset):
                return -child_fn(Xk, extras, mask, base_offset)

            return _eval

        elif node.op == "T":

            def _eval(Xk, extras, mask, base_offset):
                v = child_fn(Xk, extras, mask, base_offset)
                if v.ndim >= 2:
                    return jnp.swapaxes(v, -2, -1)
                return v

            return _eval

        elif node.op == "ew":
            ew = node.fn
            assert ew is not None

            def _eval(Xk, extras, mask, base_offset):
                return ew(child_fn(Xk, extras, mask, base_offset))

            return _eval

        else:
            raise ValueError(f"Unknown unary op: {node.op!r}")

    # ------------------------------------------------------------------
    # Einsum
    # ------------------------------------------------------------------

    if isinstance(node, _EinsumOp):
        child_fns = [compile_eval(c, o2i, sectors, ndim, key_dx) for c in node.children]
        spec = node.spec

        def _eval(Xk, extras, mask, base_offset):
            operands = [cfn(Xk, extras, mask, base_offset) for cfn in child_fns]
            return jnp.einsum(spec, *operands)

        return _eval

    # ------------------------------------------------------------------
    # Stack
    # ------------------------------------------------------------------

    if isinstance(node, _StackOp):
        child_fns = [compile_eval(c, o2i, sectors, ndim, key_dx) for c in node.children]

        def _eval(Xk, extras, mask, base_offset):
            return jnp.stack(
                [cfn(Xk, extras, mask, base_offset) for cfn in child_fns],
                axis=0,
            )

        return _eval

    # ------------------------------------------------------------------
    # Concat
    # ------------------------------------------------------------------

    if isinstance(node, _ConcatOp):
        child_fns = [compile_eval(c, o2i, sectors, ndim, key_dx) for c in node.children]

        def _eval(Xk, extras, mask, base_offset):
            return jnp.concatenate(
                [cfn(Xk, extras, mask, base_offset) for cfn in child_fns],
                axis=0,
            )

        return _eval

    # ------------------------------------------------------------------
    # Slice
    # ------------------------------------------------------------------

    if isinstance(node, _SliceOp):
        child_fn = compile_eval(node.child, o2i, sectors, ndim, key_dx)
        idx = node.idx

        def _eval(Xk, extras, mask, base_offset):
            v = child_fn(Xk, extras, mask, base_offset)
            return v[..., idx]

        return _eval

    # ------------------------------------------------------------------
    # DiffOpNode — THE KEY CASE: FD formula with shifted base_offset
    # ------------------------------------------------------------------

    if isinstance(node, _DiffOpNode):
        op_name = node.op_name
        child_fn = compile_eval(node.child, o2i, sectors, ndim, key_dx)

        if op_name == "lap":
            return _compile_lap(child_fn, ndim, key_dx)
        elif op_name == "grad":
            return _compile_grad(child_fn, ndim, key_dx)
        elif op_name == "div":
            return _compile_div(child_fn, ndim, key_dx)
        elif op_name == "biharmonic":
            return _compile_biharmonic(child_fn, ndim, key_dx)
        else:
            raise ValueError(f"Unknown diff op in eval compiler: {op_name!r}")

    raise NotImplementedError(f"compile_eval: unsupported node type {type(node).__name__}")


# =====================================================================
# FD formula implementations (all operate on base_offset, trace-time)
# =====================================================================


def _compile_lap(child_fn: Callable, ndim: int, key_dx: str) -> Callable:
    """Laplacian: Σ_a (f(+e_a) + f(−e_a) − 2·f(0)) / dx_a²."""

    def _eval(Xk, extras, mask, base_offset):
        dx = jnp.asarray(extras[key_dx])
        dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
        inv_dx2 = 1.0 / (dx * dx)

        f0 = child_fn(Xk, extras, mask, base_offset)
        result = jnp.zeros_like(f0)

        for a in range(ndim):
            fp = child_fn(Xk, extras, mask, _shift(base_offset, a, +1))
            fm = child_fn(Xk, extras, mask, _shift(base_offset, a, -1))
            result = result + inv_dx2[a] * (fp + fm - 2.0 * f0)

        return result

    return _eval


def _compile_grad(child_fn: Callable, ndim: int, key_dx: str) -> Callable:
    """Gradient: (f(+e_a) − f(−e_a)) / (2·dx_a) for each axis a.

    Appends an ``(ndim,)`` trailing axis.
    """

    def _eval(Xk, extras, mask, base_offset):
        dx = jnp.asarray(extras[key_dx])
        dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
        inv_2dx = 1.0 / (2.0 * dx)

        components = []
        for a in range(ndim):
            fp = child_fn(Xk, extras, mask, _shift(base_offset, a, +1))
            fm = child_fn(Xk, extras, mask, _shift(base_offset, a, -1))
            components.append((fp - fm) * inv_2dx[a])

        # Stack along a new trailing axis
        # Each component has shape (*child_sdims), result: (*child_sdims, ndim)
        return jnp.stack(components, axis=-1)

    return _eval


def _compile_div(child_fn: Callable, ndim: int, key_dx: str) -> Callable:
    """Divergence: Σ_a ∂_a f[…, a].  Contracts last axis (must be ndim)."""

    def _eval(Xk, extras, mask, base_offset):
        dx = jnp.asarray(extras[key_dx])
        dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
        inv_2dx = 1.0 / (2.0 * dx)

        # Evaluate at center to get shape
        f0 = child_fn(Xk, extras, mask, base_offset)
        # f0 has shape (*prefix, ndim) — we contract the last axis
        result = jnp.zeros(f0.shape[:-1], dtype=f0.dtype)

        for a in range(ndim):
            fp = child_fn(Xk, extras, mask, _shift(base_offset, a, +1))
            fm = child_fn(Xk, extras, mask, _shift(base_offset, a, -1))
            # Select the a-th component of the last axis
            diff_a = fp[..., a] - fm[..., a]
            result = result + diff_a * inv_2dx[a]

        return result

    return _eval


def _compile_biharmonic(child_fn: Callable, ndim: int, key_dx: str) -> Callable:
    """Biharmonic ∇⁴ using the dedicated stencil.

    Uses the direct 4th-order formula rather than composing lap∘lap
    (which would give the same mathematical result but on a wider
    stencil if other operators are also present in the tree).
    """
    import itertools as _it

    diag_pairs = list(_it.combinations(range(ndim), 2))

    def _eval(Xk, extras, mask, base_offset):
        dx = jnp.asarray(extras[key_dx])
        dx = dx if dx.ndim == 1 else jnp.full((ndim,), dx)
        inv_dx2 = 1.0 / (dx * dx)

        f0 = child_fn(Xk, extras, mask, base_offset)
        result = jnp.zeros_like(f0)

        # Pure 4th derivatives along each axis:
        # (f(+2e) - 4f(+e) + 6f(0) - 4f(-e) + f(-2e)) / dx^4
        for a in range(ndim):
            h4_inv = inv_dx2[a] * inv_dx2[a]
            f_2p = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, +1), a, +1))
            f_p = child_fn(Xk, extras, mask, _shift(base_offset, a, +1))
            f_m = child_fn(Xk, extras, mask, _shift(base_offset, a, -1))
            f_2m = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, -1), a, -1))
            result = result + h4_inv * (f_2p - 4.0 * f_p + 6.0 * f0 - 4.0 * f_m + f_2m)

        # Mixed 4th derivatives: 2 * d^4f/(dx_a^2 dx_b^2) for each pair
        for a, b in diag_pairs:
            hab_inv = inv_dx2[a] * inv_dx2[b]

            # f(+e_a+e_b), f(+e_a-e_b), f(-e_a+e_b), f(-e_a-e_b)
            f_pp = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, +1), b, +1))
            f_pm = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, +1), b, -1))
            f_mp = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, -1), b, +1))
            f_mm = child_fn(Xk, extras, mask, _shift(_shift(base_offset, a, -1), b, -1))

            # f(+e_a), f(-e_a), f(+e_b), f(-e_b)
            f_pa = child_fn(Xk, extras, mask, _shift(base_offset, a, +1))
            f_ma = child_fn(Xk, extras, mask, _shift(base_offset, a, -1))
            f_pb = child_fn(Xk, extras, mask, _shift(base_offset, b, +1))
            f_mb = child_fn(Xk, extras, mask, _shift(base_offset, b, -1))

            mixed = f_pp + f_pm + f_mp + f_mm - 2.0 * (f_pa + f_ma + f_pb + f_mb) + 4.0 * f0
            result = result + 2.0 * hab_inv * mixed

        return result

    return _eval


# =====================================================================
# Top-level wrapper: eval_fn → stencil local_fn
# =====================================================================


def wrap_as_local_fn(
    eval_fn: Callable,
    ndim: int,
    o2i: dict[tuple[int, ...], int],
) -> Callable:
    """Wrap a compiled eval function as a stencil ``local_fn``.

    The returned function has signature
    ``local_fn(Xk, *, mask, extras) -> (n_flat_out,)``
    and evaluates the full tree at ``base_offset = (0,) * ndim``.

    Mask handling: for noflux BCs, out-of-bounds neighbours are
    replaced by the center value *before* the eval function sees them.
    This is done by pre-processing ``Xk`` using the mask.
    """
    origin = (0,) * ndim
    center_idx = o2i[origin]

    def local_fn(Xk, *, mask=None, extras=None):
        # Pre-apply mask: replace masked-out rows with center row
        if mask is not None:
            x0 = Xk[center_idx]
            Xk = jnp.where(mask[:, None], Xk, x0[None, :])

        result = eval_fn(Xk, extras, mask, origin)

        # Final mask: zero out if center itself is masked
        if mask is not None:
            result = result * mask[center_idx]

        return result

    return local_fn
