"""StructuredExpr — declarative structured tensor expressions (inner world).

Defines :class:`StructuredExpr`, the user-facing expression type for the
inner physics world, and the lightweight ``_StructNode`` IR that encodes
the computation graph.  Expressions are symbolic and cannot be evaluated
directly; use ``Layout.embed()`` to compile them into outer-world
``StateExpr`` objects.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp

from .params import ParamSpec, ParamSuite

# =====================================================================
# Constants
# =====================================================================

_FREE_LAYOUT: int = -1  # layout-agnostic sentinel (compatible with any layout)
_EINSUM_LETTERS: str = "ijklmnopqrstuvwxyz"

# Sentinel for "not provided" (distinct from None)
_MISSING: object = object()


# =====================================================================
# _StructNode: lightweight IR  (frozen dataclasses, no eval logic)
# =====================================================================
# Users never see these directly.  The embed compiler (Phase 3) walks
# them to produce outer-world StateExpr nodes.


@dataclass(frozen=True, slots=True)
class _StructNode:
    """Base marker for all IR nodes."""


# --- Leaves -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SectorLeaf(_StructNode):
    """Field extracted from a Layout sector."""

    sector_name: str
    indices: tuple[int, ...]
    sdims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ConstNode(_StructNode):
    """Compile-time constant (scalar, eye, etc.)."""

    value: Any  # int | float | jax.Array
    sdims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ParamLeaf(_StructNode):
    """Learnable parameter block."""

    param_spec: ParamSpec
    sdims: tuple[int, ...]


# --- Layout-engine operators ------------------------------------------


@dataclass(frozen=True, slots=True)
class _DiffOpNode(_StructNode):
    """Differential operator (grad, lap, div, …).  Engine-specific."""

    op_name: str
    child: _StructNode
    engine_meta: Any  # opaque to algebra


@dataclass(frozen=True, slots=True)
class _InteractionNode(_StructNode):
    """K-body interaction.  Engine-specific."""

    fn: Callable
    reads: tuple
    writes: Any
    arity: int
    spec_factory: Any
    engine_meta: Any


# --- Pure algebra (engine-agnostic) -----------------------------------


@dataclass(frozen=True, slots=True)
class _BinaryOp(_StructNode):
    """Binary arithmetic: ``+  -  *  /  **``."""

    op: str
    left: _StructNode
    right: _StructNode


@dataclass(frozen=True, slots=True)
class _EinsumOp(_StructNode):
    """Einstein summation over rank axes."""

    spec: str  # e.g. 'ij,jk->ik'
    children: tuple[_StructNode, ...]


@dataclass(frozen=True, slots=True)
class _ConcatOp(_StructNode):
    """Feature concatenation (``&``)."""

    children: tuple[_StructNode, ...]


@dataclass(frozen=True, slots=True)
class _StackOp(_StructNode):
    """Stack expressions along a new leading rank axis."""

    children: tuple[_StructNode, ...]
    sdim: int


@dataclass(frozen=True, slots=True)
class _SliceOp(_StructNode):
    """Feature selection (``expr[idx]``)."""

    child: _StructNode
    idx: Any  # int | slice | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _UnaryOp(_StructNode):
    """Unary: neg, transpose, elementwisemap."""

    op: str  # "neg" | "T" | "ew"
    child: _StructNode
    fn: Callable | None = None  # only for "ew"


@dataclass(frozen=True, slots=True)
class _ReshapeOp(_StructNode):
    """Reshape between rank axes and features."""

    child: _StructNode
    target_sdims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _DenseOp(_StructNode):
    """Learnable affine layer on features."""

    child: _StructNode
    n_out: int
    param_spec: ParamSpec


# =====================================================================
# Helpers
# =====================================================================


def _merge_params(
    a: ParamSuite | None,
    b: ParamSuite | None,
) -> ParamSuite | None:
    """Merge two optional ParamSuites (sharing-by-name)."""
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _resolve_layout(a: int, b: int) -> int:
    """Resolve layout IDs.  ``_FREE_LAYOUT`` is compatible with anything."""
    if a == _FREE_LAYOUT:
        return b
    if b == _FREE_LAYOUT:
        return a
    if a != b:
        raise TypeError(f"Cannot combine expressions from different Layouts (layout ids {a} vs {b})")
    return a


def _parse_einsum(spec: str) -> tuple[list[str], str]:
    """Parse ``'ij,jk->ik'`` → ``(['ij', 'jk'], 'ik')``."""
    if "->" not in spec:
        raise ValueError(f"Einsum spec must contain '->': {spec!r}")
    lhs, rhs = spec.split("->", 1)
    return lhs.split(","), rhs


def _validate_einsum(
    operand_specs: list[str],
    rhs: str,
    operands: Sequence[StructuredExpr],
) -> tuple[int, ...]:
    """Validate einsum letter→size mapping; return output sdims."""
    if len(operand_specs) != len(operands):
        raise ValueError(f"Einsum spec has {len(operand_specs)} operands, got {len(operands)} expressions")
    letter_size: dict[str, int] = {}
    for spec_str, expr in zip(operand_specs, operands):
        if len(spec_str) != expr.srank:
            raise ValueError(
                f"Einsum operand spec '{spec_str}' has {len(spec_str)} axes, "
                f"but expression has srank={expr.srank} (sdims={expr.sdims})"
            )
        for ch, sz in zip(spec_str, expr.sdims):
            if ch in letter_size:
                if letter_size[ch] != sz:
                    raise ValueError(f"Einsum letter '{ch}' has conflicting sizes: {letter_size[ch]} vs {sz}")
            else:
                letter_size[ch] = sz
    for ch in rhs:
        if ch not in letter_size:
            raise ValueError(f"Einsum output letter '{ch}' not found in any input operand")
    return tuple(letter_size[ch] for ch in rhs)


def _coerce_scalar(value: Any) -> StructuredExpr | None:
    """Wrap a Python/JAX scalar as a constant ``StructuredExpr``, or None."""
    if isinstance(value, (int, float)):
        return StructuredExpr(
            sdims=(),
            n_features=1,
            param_suite=None,
            labels=(),
            _layout_id=_FREE_LAYOUT,
            _node=_ConstNode(value=value, sdims=()),
        )
    if hasattr(value, "shape") and getattr(value, "shape") == ():
        return StructuredExpr(
            sdims=(),
            n_features=1,
            param_suite=None,
            labels=(),
            _layout_id=_FREE_LAYOUT,
            _node=_ConstNode(value=value, sdims=()),
        )
    return None


# =====================================================================
# Auto-label helpers
# =====================================================================

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")

# Characters in a label that signal it is "compound" and needs
# parenthesisation when used as a factor in a product.
_MUL_PAREN_CHARS = frozenset("+-/\u00b7")


def _int_superscript(n: int) -> str:
    """Convert an integer to a unicode superscript string."""
    return str(n).translate(_SUPERSCRIPT_DIGITS)


def _complete_labels(labels: tuple[str, ...], n_features: int) -> bool:
    """True when *labels* has exactly one non-empty entry per feature."""
    return len(labels) == n_features > 0 and all(labels)


def _is_const_value(node: _StructNode, value: float | int) -> bool:
    """True when *node* is a ``_ConstNode`` with the given Python scalar."""
    if not isinstance(node, _ConstNode):
        return False
    v = node.value
    return isinstance(v, (int, float)) and v == value


def _const_int(node: _StructNode) -> int | None:
    """Return the integer value of a ``_ConstNode``, or ``None``."""
    if not isinstance(node, _ConstNode):
        return None
    v = node.value
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v) and abs(v) < 1000:
        return int(v)
    return None


def _paren_for_pow(label: str) -> str:
    """Wrap in parentheses when raising to a power (multi-char labels)."""
    return f"({label})" if len(label) > 1 else label


def _paren_for_mul(label: str) -> str:
    """Wrap in parentheses when used as a factor in a product."""
    check = label[1:] if label.startswith("-") else label
    return f"({label})" if any(c in _MUL_PAREN_CHARS for c in check) else label


def _auto_mul_labels(
    a_labels: tuple[str, ...],
    a_nf: int,
    a_node: _StructNode,
    b_labels: tuple[str, ...],
    b_nf: int,
    b_node: _StructNode,
) -> tuple[str, ...]:
    """Auto-generate labels for element-wise multiplication ``a * b``."""
    # multiply by 1 -> identity
    if _is_const_value(b_node, 1) and _complete_labels(a_labels, a_nf):
        return a_labels
    if _is_const_value(a_node, 1) and _complete_labels(b_labels, b_nf):
        return b_labels
    # both fully labelled -> Cartesian product (juxtaposition)
    if _complete_labels(a_labels, a_nf) and _complete_labels(b_labels, b_nf):
        return tuple(_paren_for_mul(la) + _paren_for_mul(lb) for la in a_labels for lb in b_labels)
    return ()


def _auto_pow_labels(
    base_labels: tuple[str, ...],
    base_nf: int,
    exp_node: _StructNode,
) -> tuple[str, ...]:
    """Auto-generate labels for ``base ** exp``."""
    n = _const_int(exp_node)
    if n is None or not _complete_labels(base_labels, base_nf):
        return ()
    if n == 0:
        return tuple("1" for _ in base_labels)
    if n == 1:
        return base_labels
    sup = _int_superscript(n)
    return tuple(f"{_paren_for_pow(lab)}{sup}" for lab in base_labels)


def _auto_sub_labels(
    a_labels: tuple[str, ...],
    a_nf: int,
    b_labels: tuple[str, ...],
    b_nf: int,
) -> tuple[str, ...]:
    """Auto-generate labels for ``a - b``."""
    if not _complete_labels(a_labels, a_nf) or not _complete_labels(b_labels, b_nf):
        return ()
    if a_nf != b_nf:
        return ()
    return tuple(f"{la}-{_paren_for_mul(lb)}" for la, lb in zip(a_labels, b_labels))


def _auto_div_labels(
    a_labels: tuple[str, ...],
    a_nf: int,
    b_labels: tuple[str, ...],
    b_nf: int,
) -> tuple[str, ...]:
    """Auto-generate labels for ``a / b``."""
    if not _complete_labels(a_labels, a_nf) or not _complete_labels(b_labels, b_nf):
        return ()
    if b_nf == 1:
        d = _paren_for_mul(b_labels[0])
        return tuple(f"{_paren_for_mul(la)}/{d}" for la in a_labels)
    if a_nf == b_nf:
        return tuple(f"{_paren_for_mul(la)}/{_paren_for_mul(lb)}" for la, lb in zip(a_labels, b_labels))
    return ()


def _auto_add_labels(
    a_labels: tuple[str, ...],
    a_nf: int,
    b_labels: tuple[str, ...],
    b_nf: int,
) -> tuple[str, ...]:
    """Auto-generate labels for ``a + b``.

    When both sides are fully labelled, produces ``"a+b"`` per feature.
    Otherwise falls back to first-non-empty-wins (the previous default).
    """
    if _complete_labels(a_labels, a_nf) and _complete_labels(b_labels, b_nf) and a_nf == b_nf:
        return tuple(f"{la}+{lb}" for la, lb in zip(a_labels, b_labels))
    return a_labels or b_labels


def _auto_einsum_labels(
    operand_info: Sequence[tuple[tuple[str, ...], int]],
) -> tuple[str, ...]:
    """Auto-generate labels for ``einsum`` (Cartesian product with ``\u00b7``)."""
    for labels, nf in operand_info:
        if not _complete_labels(labels, nf):
            return ()
    label_lists = [labels for labels, _nf in operand_info]
    return tuple("\u00b7".join(combo) for combo in itertools.product(*label_lists))


def _auto_ew_labels(
    fn: Callable,
    labels: tuple[str, ...],
    n_features: int,
    name: str | None = None,
) -> tuple[str, ...]:
    """Auto-generate labels for ``elementwisemap(fn, \u2026)``."""
    if not _complete_labels(labels, n_features):
        return ()
    nm = name or getattr(fn, "__name__", "")
    if not nm or nm == "<lambda>":
        return ()
    return tuple(f"{nm}({lab})" for lab in labels)


def _feature_count(idx: Any, n_features: int) -> int:
    """Compute output feature count for a ``__getitem__`` index."""
    if isinstance(idx, int):
        if idx < -n_features or idx >= n_features:
            raise IndexError(f"Feature index {idx} out of range for n_features={n_features}")
        return 1
    if isinstance(idx, slice):
        return len(range(*idx.indices(n_features)))
    if isinstance(idx, (list, tuple)):
        for i in idx:
            if not isinstance(i, int):
                raise TypeError(f"Feature indices must be ints, got {type(i).__name__}")
            if i < -n_features or i >= n_features:
                raise IndexError(f"Feature index {i} out of range for n_features={n_features}")
        return len(idx)
    raise TypeError(f"Unsupported index type: {type(idx).__name__}")


# =====================================================================
# StructuredExpr
# =====================================================================


@dataclass(frozen=True, slots=True, eq=False)
class StructuredExpr:
    """Declarative structured tensor expression (inner world).

    Users build these via Layout methods and algebra.  They are symbolic
    and cannot be evaluated directly — use ``Layout.embed()`` to compile
    to an outer-world ``StateExpr``.

    Attributes
    ----------
    sdims : tuple[int, ...]
        Per-axis sizes.  ``()`` = scalar, ``(2,)`` = 2-vector, etc.
    n_features : int
        Number of independent regression channels (last axis).
    param_suite : ParamSuite | None
        Learnable parameters (``None`` = pure basis).
    labels : tuple[str, ...]
        Human-readable feature labels.
    """

    sdims: tuple[int, ...]
    n_features: int
    param_suite: ParamSuite | None
    labels: tuple[str, ...]
    _layout_id: int
    _node: _StructNode

    # --- properties ---------------------------------------------------

    @property
    def srank(self) -> int:
        """Number of structured dimensions (rank axes)."""
        return len(self.sdims)

    # --- private helpers ----------------------------------------------

    def _new(
        self,
        *,
        sdims: tuple[int, ...] | None = None,
        n_features: int | None = None,
        param_suite: Any = _MISSING,
        labels: tuple[str, ...] = (),
        layout_id: int | None = None,
        node: _StructNode,
    ) -> StructuredExpr:
        """Shortcut: create a new expr, inheriting metadata from *self*."""
        return StructuredExpr(
            sdims=sdims if sdims is not None else self.sdims,
            n_features=(n_features if n_features is not None else self.n_features),
            param_suite=(param_suite if param_suite is not _MISSING else self.param_suite),
            labels=labels,
            _layout_id=(layout_id if layout_id is not None else self._layout_id),
            _node=node,
        )

    def _check_layout(self, other: StructuredExpr) -> None:
        _resolve_layout(self._layout_id, other._layout_id)

    # =================================================================
    # Arithmetic:  +  -  *  /  **  (unary -)
    # =================================================================

    def __add__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            other = _coerce_scalar(other)
            if other is None:
                return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        if self.sdims != other.sdims:
            raise ValueError(f"Cannot add: sdims mismatch {self.sdims} vs {other.sdims}")
        if self.n_features != other.n_features:
            raise ValueError(f"Cannot add: n_features mismatch {self.n_features} vs {other.n_features}")
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features,
            param_suite=_merge_params(self.param_suite, other.param_suite),
            labels=_auto_add_labels(
                self.labels,
                self.n_features,
                other.labels,
                other.n_features,
            ),
            _layout_id=_resolve_layout(self._layout_id, other._layout_id),
            _node=_BinaryOp("+", self._node, other._node),
        )

    def __radd__(self, other: Any) -> StructuredExpr:
        return self.__add__(other)  # addition is commutative

    def __sub__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            other = _coerce_scalar(other)
            if other is None:
                return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        if self.sdims != other.sdims:
            raise ValueError(f"Cannot subtract: sdims mismatch {self.sdims} vs {other.sdims}")
        if self.n_features != other.n_features:
            raise ValueError(f"Cannot subtract: n_features mismatch {self.n_features} vs {other.n_features}")
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features,
            param_suite=_merge_params(self.param_suite, other.param_suite),
            labels=_auto_sub_labels(
                self.labels,
                self.n_features,
                other.labels,
                other.n_features,
            ),
            _layout_id=_resolve_layout(self._layout_id, other._layout_id),
            _node=_BinaryOp("-", self._node, other._node),
        )

    def __rsub__(self, other: Any) -> StructuredExpr:
        other_expr = _coerce_scalar(other)
        if other_expr is None:
            return NotImplemented  # type: ignore[return-value]
        return other_expr.__sub__(self)

    def __mul__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            other = _coerce_scalar(other)
            if other is None:
                return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        a, b = self, other
        if a.srank == 0:
            sdims = b.sdims
        elif b.srank == 0:
            sdims = a.sdims
        elif a.sdims == b.sdims:
            sdims = a.sdims
        else:
            raise TypeError(
                f"Cannot multiply expressions with incompatible sdims "
                f"{a.sdims} and {b.sdims}. Use einsum() for mixed-rank "
                f"contractions."
            )
        return StructuredExpr(
            sdims=sdims,
            n_features=a.n_features * b.n_features,
            param_suite=_merge_params(a.param_suite, b.param_suite),
            labels=_auto_mul_labels(
                a.labels,
                a.n_features,
                a._node,
                b.labels,
                b.n_features,
                b._node,
            ),
            _layout_id=_resolve_layout(a._layout_id, b._layout_id),
            _node=_BinaryOp("*", a._node, b._node),
        )

    def __rmul__(self, other: Any) -> StructuredExpr:
        return self.__mul__(other)  # multiplication is commutative

    def __truediv__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            other = _coerce_scalar(other)
            if other is None:
                return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        if other.n_features not in (1, self.n_features):
            raise ValueError(
                f"Division requires divisor n_features=1 or matching "
                f"n_features={self.n_features}, got {other.n_features}"
            )
        if other.srank > 0 and other.sdims != self.sdims:
            raise ValueError(f"Cannot divide: sdims mismatch {self.sdims} vs {other.sdims}")
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features,
            param_suite=_merge_params(self.param_suite, other.param_suite),
            labels=_auto_div_labels(
                self.labels,
                self.n_features,
                other.labels,
                other.n_features,
            ),
            _layout_id=_resolve_layout(self._layout_id, other._layout_id),
            _node=_BinaryOp("/", self._node, other._node),
        )

    def __rtruediv__(self, other: Any) -> StructuredExpr:
        other_expr = _coerce_scalar(other)
        if other_expr is None:
            return NotImplemented  # type: ignore[return-value]
        return other_expr.__truediv__(self)

    def __pow__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            other = _coerce_scalar(other)
            if other is None:
                return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        if other.srank != 0 or other.n_features != 1:
            raise TypeError(
                f"Exponent must be a scalar with n_features=1, got sdims={other.sdims}, n_features={other.n_features}"
            )
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features,
            param_suite=_merge_params(self.param_suite, other.param_suite),
            labels=_auto_pow_labels(
                self.labels,
                self.n_features,
                other._node,
            ),
            _layout_id=_resolve_layout(self._layout_id, other._layout_id),
            _node=_BinaryOp("**", self._node, other._node),
        )

    def __neg__(self) -> StructuredExpr:
        return self._new(labels=self.labels, node=_UnaryOp("neg", self._node))

    def __pos__(self) -> StructuredExpr:
        return self

    # =================================================================
    # Human-readable label
    # =================================================================

    def with_label(self, label: str) -> StructuredExpr:
        """Return a copy of this expression with a single human-readable label.

        Useful for annotating derived quantities (arithmetic, einsum, …)
        so that ``print_report`` shows a meaningful term name instead of
        a generic fallback.

        Parameters
        ----------
        label : str
            Human-readable name for this term, e.g. ``"|Q|²Q"``.

        Returns
        -------
        StructuredExpr
            Identical expression with ``labels=(label,)``.
        """
        if self.n_features != 1:
            raise ValueError(
                f"with_label() requires n_features=1, "
                f"got n_features={self.n_features}. "
                "Use the & operator to concatenate labelled single-feature "
                "terms instead of labelling a multi-feature block."
            )
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features,
            param_suite=self.param_suite,
            labels=(label,),
            _layout_id=self._layout_id,
            _node=self._node,
        )

    # =================================================================
    # Feature concatenation  (&)
    # =================================================================

    def __and__(self, other: Any) -> StructuredExpr:
        if not isinstance(other, StructuredExpr):
            return NotImplemented  # type: ignore[return-value]
        self._check_layout(other)
        if self.sdims != other.sdims:
            raise ValueError(f"Cannot concatenate (&): sdims mismatch {self.sdims} vs {other.sdims}")
        # Flatten left-side concats so labels[i] aligns with children[i]
        # (avoids misalignment when _compile_sector iterates node.children)
        if isinstance(self._node, _ConcatOp):
            new_children = self._node.children + (other._node,)
        else:
            new_children = (self._node, other._node)
        return StructuredExpr(
            sdims=self.sdims,
            n_features=self.n_features + other.n_features,
            param_suite=_merge_params(self.param_suite, other.param_suite),
            labels=self.labels + other.labels,
            _layout_id=_resolve_layout(self._layout_id, other._layout_id),
            _node=_ConcatOp(new_children),
        )

    # =================================================================
    # Feature selection  (expr[idx])
    # =================================================================

    def __getitem__(self, idx: Any) -> StructuredExpr:
        n = _feature_count(idx, self.n_features)
        if isinstance(idx, int):
            lbls = (self.labels[idx],) if idx < len(self.labels) else ()
        elif isinstance(idx, slice):
            lbls = tuple(self.labels[idx]) if self.labels else ()
        else:
            lbls = ()
        return self._new(
            n_features=n,
            labels=lbls,
            node=_SliceOp(self._node, idx),
        )

    # =================================================================
    # Transpose
    # =================================================================

    @property
    def T(self) -> StructuredExpr:
        """Swap last two rank axes.  Requires ``srank >= 2``."""
        if self.srank < 2:
            raise TypeError(f"Transpose requires srank >= 2, got srank={self.srank} (sdims={self.sdims})")
        new_sdims = self.sdims[:-2] + (self.sdims[-1], self.sdims[-2])
        return self._new(
            sdims=new_sdims,
            node=_UnaryOp("T", self._node),
        )

    # =================================================================
    # Matmul  (@)
    # =================================================================

    def __matmul__(self, other: Any) -> StructuredExpr:
        """Contract last axis of *self* with first axis of *other*."""
        if not isinstance(other, StructuredExpr):
            return NotImplemented  # type: ignore[return-value]
        if self.srank < 1 or other.srank < 1:
            raise TypeError(f"Matmul requires srank >= 1 on both sides, got {self.srank} and {other.srank}")
        if self.sdims[-1] != other.sdims[0]:
            raise ValueError(
                f"Matmul contraction mismatch: self.sdims[-1]={self.sdims[-1]} vs other.sdims[0]={other.sdims[0]}"
            )
        pool = iter(_EINSUM_LETTERS)
        a_letters = [next(pool) for _ in range(self.srank)]
        shared = a_letters[-1]
        b_letters = [shared] + [next(pool) for _ in range(other.srank - 1)]
        spec = "".join(a_letters) + "," + "".join(b_letters) + "->" + "".join(a_letters[:-1]) + "".join(b_letters[1:])
        return type(self).einsum(spec, self, other)

    # =================================================================
    # Dot  (contract last axes of both)
    # =================================================================

    def dot(self, other: StructuredExpr) -> StructuredExpr:
        """Contract last axis of *self* with last axis of *other*."""
        if not isinstance(other, StructuredExpr):
            raise TypeError(f"Expected StructuredExpr, got {type(other).__name__}")
        if self.srank < 1 or other.srank < 1:
            raise TypeError(f"dot requires srank >= 1 on both sides, got {self.srank} and {other.srank}")
        if self.sdims[-1] != other.sdims[-1]:
            raise ValueError(
                f"dot contraction mismatch: self.sdims[-1]={self.sdims[-1]} vs other.sdims[-1]={other.sdims[-1]}"
            )
        pool = iter(_EINSUM_LETTERS)
        a_rest = [next(pool) for _ in range(self.srank - 1)]
        b_rest = [next(pool) for _ in range(other.srank - 1)]
        shared = next(pool)
        spec = "".join(a_rest + [shared]) + "," + "".join(b_rest + [shared]) + "->" + "".join(a_rest + b_rest)
        return type(self).einsum(spec, self, other)

    # =================================================================
    # Einsum  (static method)
    # =================================================================

    @staticmethod
    def einsum(spec: str, *operands: StructuredExpr) -> StructuredExpr:
        """Einstein summation over rank axes.

        Example::

            Q = StructuredExpr.einsum('i,j->ij', n, n)
        """
        operand_specs, rhs = _parse_einsum(spec)
        output_sdims = _validate_einsum(operand_specs, rhs, operands)

        layout_id = _FREE_LAYOUT
        params: ParamSuite | None = None
        for op in operands:
            layout_id = _resolve_layout(layout_id, op._layout_id)
            params = _merge_params(params, op.param_suite)

        n_features = math.prod(op.n_features for op in operands)
        return StructuredExpr(
            sdims=output_sdims,
            n_features=n_features,
            param_suite=params,
            labels=_auto_einsum_labels([(op.labels, op.n_features) for op in operands]),
            _layout_id=layout_id,
            _node=_EinsumOp(spec, tuple(op._node for op in operands)),
        )

    # =================================================================
    # Stack  (classmethod — build vector from scalars)
    # =================================================================

    @classmethod
    def stack(
        cls,
        exprs: Sequence[StructuredExpr],
        *,
        sdim: int | None = None,
    ) -> StructuredExpr:
        """Stack expressions along a new leading rank axis.

        All inputs must share the same ``sdims`` and ``n_features``.
        ``sdim`` defaults to ``len(exprs)``.
        """
        exprs = list(exprs)
        if not exprs:
            raise ValueError("stack requires at least one expression")
        if sdim is None:
            sdim = len(exprs)
        if sdim != len(exprs):
            raise ValueError(f"sdim={sdim} does not match number of expressions ({len(exprs)})")
        ref = exprs[0]
        layout_id = ref._layout_id
        params = ref.param_suite
        for e in exprs[1:]:
            layout_id = _resolve_layout(layout_id, e._layout_id)
            if e.sdims != ref.sdims:
                raise ValueError(f"stack: all expressions must have same sdims, got {ref.sdims} and {e.sdims}")
            if e.n_features != ref.n_features:
                raise ValueError(
                    f"stack: all expressions must have same n_features, got {ref.n_features} and {e.n_features}"
                )
            params = _merge_params(params, e.param_suite)
        return StructuredExpr(
            sdims=(sdim,) + ref.sdims,
            n_features=ref.n_features,
            param_suite=params,
            labels=(),
            _layout_id=layout_id,
            _node=_StackOp(tuple(e._node for e in exprs), sdim),
        )

    # =================================================================
    # Eye  (classmethod — identity matrix)
    # =================================================================

    @classmethod
    def eye(
        cls,
        sdim: int,
        *,
        layout_id: int = _FREE_LAYOUT,
    ) -> StructuredExpr:
        """Identity matrix with ``sdims=(sdim, sdim)``, ``n_features=1``."""
        return StructuredExpr(
            sdims=(sdim, sdim),
            n_features=1,
            param_suite=None,
            labels=("I",),
            _layout_id=layout_id,
            _node=_ConstNode(value=jnp.eye(sdim), sdims=(sdim, sdim)),
        )

    # =================================================================
    # Math convenience methods
    # =================================================================

    def sin(self) -> StructuredExpr:
        return self.elementwisemap(jnp.sin, name="sin")

    def cos(self) -> StructuredExpr:
        return self.elementwisemap(jnp.cos, name="cos")

    def exp(self) -> StructuredExpr:
        return self.elementwisemap(jnp.exp, name="exp")

    def log(self) -> StructuredExpr:
        return self.elementwisemap(jnp.log, name="log")

    def tanh(self) -> StructuredExpr:
        return self.elementwisemap(jnp.tanh, name="tanh")

    def abs(self) -> StructuredExpr:
        return self.elementwisemap(jnp.abs, name="abs")

    def sqrt(self) -> StructuredExpr:
        return self.elementwisemap(jnp.sqrt, name="sqrt")

    def elementwisemap(
        self,
        fn: Callable,
        *,
        name: str | None = None,
    ) -> StructuredExpr:
        """Apply a JAX-traceable function elementwise.

        Records a ``_UnaryOp("ew", \u2026)`` node in the IR tree.  At embed
        time this compiles to ``StateExpr.elementwisemap(fn)``.

        Parameters
        ----------
        name : str, optional
            Override the function name used in auto-generated labels.
            Defaults to ``fn.__name__``.
        """
        return self._new(
            labels=_auto_ew_labels(fn, self.labels, self.n_features, name),
            node=_UnaryOp("ew", self._node, fn),
        )

    # =================================================================
    # Reshape operations
    # =================================================================

    def rank_to_features(self) -> StructuredExpr:
        """Flatten all rank axes into the features axis.

        ``sdims=(2,3), n_features=5``  →  ``sdims=(), n_features=30``.
        """
        new_nf = self.n_features * math.prod(self.sdims) if self.sdims else self.n_features
        return self._new(
            sdims=(),
            n_features=new_nf,
            node=_ReshapeOp(self._node, target_sdims=()),
        )

    def features_to_rank(self, target_sdims: tuple[int, ...]) -> StructuredExpr:
        """Promote features into rank axes.  Requires ``srank == 0``.

        ``sdims=(), n_features=12``  →  ``features_to_rank((3,4))``
        →  ``sdims=(3,4), n_features=1``.
        """
        if self.srank != 0:
            raise TypeError(
                f"features_to_rank requires srank == 0 (scalar input), "
                f"got srank={self.srank}. Use rank_to_features() first."
            )
        p = math.prod(target_sdims)
        if p == 0:
            raise ValueError("target_sdims must have all positive sizes")
        if self.n_features % p != 0:
            raise ValueError(f"n_features={self.n_features} is not divisible by prod(target_sdims)={p}")
        return self._new(
            sdims=target_sdims,
            n_features=self.n_features // p,
            node=_ReshapeOp(self._node, target_sdims=target_sdims),
        )

    # =================================================================
    # Dense  (learnable affine layer on features)
    # =================================================================

    def dense(self, n_out: int, *, name: str = "dense") -> StructuredExpr:
        """Affine projection of features.  Adds a learnable weight matrix."""
        ps = ParamSpec(name=name, shape=(n_out, self.n_features))
        return StructuredExpr(
            sdims=self.sdims,
            n_features=n_out,
            param_suite=_merge_params(self.param_suite, ParamSuite([ps])),
            labels=(),
            _layout_id=self._layout_id,
            _node=_DenseOp(self._node, n_out, ps),
        )

    # =================================================================
    # repr
    # =================================================================

    def __repr__(self) -> str:
        parts = [
            f"sdims={self.sdims}",
            f"n_features={self.n_features}",
        ]
        if self.labels:
            parts.append(f"labels={self.labels}")
        if self.param_suite is not None:
            parts.append("has_params=True")
        return f"StructuredExpr({', '.join(parts)})"
