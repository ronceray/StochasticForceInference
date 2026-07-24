from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .timeops import TimeOp

import jax
import jax.numpy as jnp
import opt_einsum as oe  # required

__all__ = ["ExprOperand", "TimeOperand", "ConstOperand", "Term", "Integrand"]


@dataclass(frozen=True)
class ExprOperand:
    """
    Wrap a state expression evaluated at x (and optionally v).

    Contract
    --------
    expr(x, v=..., mask=..., extras=..., params=...) -> array
      - features sit on the last axis
      - 'mask' is forwarded from the 'mask_out' stream when present
    """

    expr: object
    x: "TimeOp"
    v: Optional["TimeOp"] = None
    # Template / default parameters. Call-time params may override this.
    params_template: Optional[Mapping] = None
    alias: str = "E"

    @property
    def required_extras(self) -> Tuple[str, ...]:
        """Keys this expression demands in the extras mapping."""
        re = getattr(self.expr, "required_extras", ()) or ()
        return tuple(re)

    @property
    def requires(self) -> Set[str]:
        req = set(self.x.requires)
        if self.v is not None:
            req |= set(self.v.requires)
        if self.required_extras:
            req.add("extras")  # ensure collection streams extras
        return req


@dataclass(frozen=True)
class TimeOperand:
    """Time operand (scalar/vector/tensor) built from streams via a TimeOp."""

    op: "TimeOp"
    alias: str = "T"

    @property
    def requires(self) -> Set[str]:
        return set(self.op.requires)


@dataclass(frozen=True)
class ConstOperand:
    """Constant array captured by the program (e.g. A_inv)."""

    value: jnp.ndarray
    alias: str = "C"


@dataclass(frozen=True)
class Term:
    """
    One einsum contraction over the named operands.

    Axis conventions
    ----------------
    i : particle axis (kept until the integrator reduces over i)
    m,n,... : spatial indices
    a,b,... : feature axes (always last on any ExprOperand output)
    """

    eq: str
    ops: Tuple[str, ...]
    scale: float = 1.0


class Integrand:
    """
    Compose state expressions and time operands via einsum per time slice.

    ``require()`` -> ``Set[str]``
        Stream keys needed to evaluate this program on one time slice.

    ``__call__(**streams)`` -> ``jnp.ndarray``
        Evaluate on one time slice. The collection handles mask and reduction.

    ``estimate_bytes_per_sample(sample_streams)`` -> ``Optional[int]``
        Upper bound on per-particle bytes using a real sample (via opt_einsum).
    """

    def __init__(
        self,
        exprs: Sequence[ExprOperand] = (),
        times: Sequence[TimeOperand] = (),
        consts: Sequence[ConstOperand] = (),
        terms: Sequence[Term] = (),
    ):
        self._check_aliases(exprs, times, consts)
        self._exprs: Dict[str, ExprOperand] = {e.alias: e for e in exprs}
        self._times: Dict[str, TimeOperand] = {t.alias: t for t in times}
        self._consts: Dict[str, ConstOperand] = {c.alias: c for c in consts}
        self._terms: Tuple[Term, ...] = tuple(terms)

    @staticmethod
    def _check_aliases(
        exprs: Sequence[ExprOperand],
        times: Sequence[TimeOperand],
        consts: Sequence[ConstOperand],
    ) -> None:
        seen: Dict[str, str] = {}
        groups = [("expr", exprs), ("time", times), ("const", consts)]
        for kind, ops in groups:
            for op in ops:
                alias = op.alias  # type: ignore[union-attr]
                if alias in seen:
                    raise ValueError(
                        f"Duplicate alias {alias!r}: already registered as "
                        f"{seen[alias]}, now also as {kind}."
                    )
                seen[alias] = kind

    @staticmethod
    def _merge_operand_dicts(
        d1: Dict[str, Any],
        d2: Dict[str, Any],
        kind: str,
    ) -> Dict[str, Any]:
        """Merge two alias→operand dicts; allow same-object sharing, raise on conflict."""
        merged = dict(d1)
        for alias, op in d2.items():
            if alias in merged:
                if merged[alias] is not op:
                    raise ValueError(
                        f"Cannot add Integrands: alias {alias!r} maps to "
                        f"different {kind} operands in the two summands."
                    )
            else:
                merged[alias] = op
        return merged

    # ---------- integration API ----------
    def require(self) -> Set[str]:
        req = set()
        for e in self._exprs.values():
            req |= e.requires
        for t in self._times.values():
            req |= t.requires
        return req

    def required_extras(self) -> Set[str]:
        keys: Set[str] = set()
        for E in self._exprs.values():
            keys.update(E.required_extras)
        return keys

    def __call__(self, *, params: Optional[Any] = None, **streams):
        bufs: Dict[str, jnp.ndarray] = {}

        # --- early extras validation (fail fast, clear errors)
        extras = streams.get("extras", None)
        for a, E in self._exprs.items():
            if E.required_extras:
                try:
                    # StateExpr provides the validator
                    E.expr._validate_extras_presence(extras)  # type: ignore[attr-defined]
                except KeyError:
                    avail = sorted(list(extras.keys())) if isinstance(extras, Mapping) else None
                    msg = f"[{a}] missing extras {list(E.required_extras)}"
                    if avail is not None:
                        msg += f"; available: {avail}"
                    raise KeyError(msg) from None

            x = E.x(**streams)
            v = None if E.v is None else E.v(**streams)
            kwargs: Dict[str, Any] = {}
            if v is not None:
                kwargs["v"] = v
            if "mask_out" in streams and streams["mask_out"] is not None:
                kwargs["mask"] = streams["mask_out"]
            if "extras" in streams:
                kwargs["extras"] = streams["extras"]

            # Parameter routing:
            # - if params is not None: use it as the actual parameter object
            # - else: fall back to the operand's params_template
            par = E.params_template if params is None else params
            if par is not None:
                kwargs["params"] = par

            bufs[a] = E.expr(x, **kwargs)

        # time operands
        for a, T in self._times.items():
            bufs[a] = T.op(**streams)

        # constants
        for a, C in self._consts.items():
            bufs[a] = C.value

        # contractions
        out = None
        for term in self._terms:
            args = [bufs[a] for a in term.ops]
            val = jnp.einsum(term.eq, *args)
            out = term.scale * val if out is None else out + term.scale * val
        return out

    def batch_call(self, *, params: Optional[Any] = None, **streams):
        """Evaluate with streams that carry a leading batch (K) axis.

        This is the batched counterpart of :meth:`__call__`.  Streams such as
        ``X`` have shape ``(K, N, d)`` instead of ``(N, d)``, and ``dt`` has
        shape ``(K,)`` instead of being a scalar.

        State-expression operands receive the full ``(K, N, d)`` tensor and
        handle arbitrary leading batch dimensions internally (the leaf's
        ``_apply_user_func`` flattens the batch prefix and uses a single
        ``jax.vmap``).

        Time operands (e.g. velocity) are evaluated on the batch streams
        directly; the batch-safe ``velocity`` TimeOp handles dt broadcasting.

        Einsum contractions are vmapped over the leading K axis so that
        existing subscript strings (which reference particle/spatial/feature
        axes only) work unchanged.

        Returns
        -------
        jax.Array
            Result with a leading ``K`` axis.  Shape is
            ``(K, <per-row output shape>)``.
        """
        bufs: Dict[str, jnp.ndarray] = {}

        # --- extras validation (same as __call__) ---
        extras = streams.get("extras", None)
        for a, E in self._exprs.items():
            if E.required_extras:
                try:
                    E.expr._validate_extras_presence(extras)
                except KeyError:
                    avail = sorted(list(extras.keys())) if isinstance(extras, Mapping) else None
                    msg = f"[{a}] missing extras {list(E.required_extras)}"
                    if avail is not None:
                        msg += f"; available: {avail}"
                    raise KeyError(msg) from None

            x = E.x(**streams)  # (K, N, d)
            v = None if E.v is None else E.v(**streams)  # (K, N, d) or None
            kwargs: Dict[str, Any] = {}
            if v is not None:
                kwargs["v"] = v
            if "mask_out" in streams and streams["mask_out"] is not None:
                kwargs["mask"] = streams["mask_out"]  # (K, N)
            if "extras" in streams:
                kwargs["extras"] = streams["extras"]

            par = E.params_template if params is None else params
            if par is not None:
                kwargs["params"] = par

            bufs[a] = E.expr(x, **kwargs)
            # Output: (K, N, ..., F) or (K, ..., F) depending on pdepth/rank

        # --- time operands: call directly if batch_safe, else vmap over K ---
        for a, T in self._times.items():
            if getattr(T.op, "batch_safe", False):
                # TimeOp already handles leading batch dims — call directly
                bufs[a] = T.op(**streams)
            else:
                req_keys = T.op.requires
                req_streams = {k: streams[k] for k in req_keys if k in streams}
                # Separate batched (ndim≥1) vs scalar entries for in_axes
                in_axes_t = {k: 0 if v.ndim >= 1 else None for k, v in req_streams.items()}
                bufs[a] = jax.vmap(lambda s: T.op(**s), in_axes=(in_axes_t,))(req_streams)

        # --- constants ---
        for a, C in self._consts.items():
            bufs[a] = C.value

        # --- contractions: vmap each einsum over the leading K axis ---
        const_aliases = frozenset(self._consts.keys())
        out = None
        for term in self._terms:
            operands = [bufs[a] for a in term.ops]
            # Constants have no K axis → in_axes=None; others → 0
            in_axes = tuple(None if a in const_aliases else 0 for a in term.ops)
            val = jax.vmap(
                lambda *args, _eq=term.eq: jnp.einsum(_eq, *args),
                in_axes=in_axes,
            )(*operands)
            out = term.scale * val if out is None else out + term.scale * val

        return out

    # ---------- sugar for linear combos ----------
    def __add__(self, other: "Integrand") -> "Integrand":
        me = self._merge_operand_dicts(self._exprs, other._exprs, "expr")
        mt = self._merge_operand_dicts(self._times, other._times, "time")
        mc = self._merge_operand_dicts(self._consts, other._consts, "const")
        return Integrand(
            exprs=list(me.values()),
            times=list(mt.values()),
            consts=list(mc.values()),
            terms=[*self._terms, *other._terms],
        )

    def __radd__(self, other):  # allow sum([...], start=0)
        return self if other == 0 else NotImplemented

    def __mul__(self, alpha: float) -> "Integrand":
        return Integrand(
            exprs=self._exprs.values(),
            times=self._times.values(),
            consts=self._consts.values(),
            terms=[Term(eq=t.eq, ops=t.ops, scale=t.scale * alpha) for t in self._terms],
        )

    __rmul__ = __mul__

    # ---------- memory hint on a real sample ----------
    def estimate_bytes_per_sample(
        self,
        sample_streams: Mapping[str, jnp.ndarray],
        *,
        dtypesize: int = 4,
    ) -> Optional[int]:
        """
        Conservative upper bound in bytes per **time-sample** (one k-row).
        Uses StateExpr static hints + shapes-only einsum path. No evaluation.
        """
        _X = sample_streams.get("X", None)
        _dtype = None if _X is None else _X.dtype
        P = int(sample_streams["N_total"])  # use provided, conservative

        # ---- shapes without evaluation
        def expr_shape(E) -> tuple[int, ...]:
            p = int(getattr(E.expr, "pdepth", 0))
            r = int(getattr(E.expr, "rank", 0))
            d = int(getattr(E.expr, "dim", 0))
            F = int(getattr(E.expr, "n_features", 1))
            paxes = () if p == 0 else (P,) * p  # full particle axes (per-sample)
            return (*paxes, *(d,) * r, F)

        def time_shape(T) -> tuple[int, ...]:
            arr = T.op(**sample_streams)  # shape-only probe
            return tuple(arr.shape)

        def const_shape(C) -> tuple[int, ...]:
            return tuple(C.value.shape)

        shapes = {}
        for a, E in self._exprs.items():
            shapes[a] = expr_shape(E)
        for a, T in self._times.items():
            shapes[a] = time_shape(T)
        for a, C in self._consts.items():
            shapes[a] = const_shape(C)

        def match_len(shape: tuple[int, ...], subscript: str) -> tuple[int, ...]:
            L = len(subscript)
            return (*shape, *([1] * (L - len(shape)))) if len(shape) < L else shape

        def size_for_term(t: Term) -> int:
            lhs = t.eq.split("->", 1)[0]
            subs = [s.strip() for s in lhs.split(",")]
            ops_shapes = [match_len(shapes[a], subs[i]) for i, a in enumerate(t.ops)]
            try:
                _, info = oe.contract_path(t.eq, *ops_shapes, shapes=True, optimize="greedy")
                if hasattr(info, "largest_intermediate"):
                    return int(info.largest_intermediate) * int(dtypesize)
                if hasattr(info, "intermediate_shapes") and info.intermediate_shapes:
                    mx = max(int(jnp.prod(jnp.array(sh))) for sh in info.intermediate_shapes)  # type: ignore
                    return mx * int(dtypesize)
            except Exception:
                pass
            total_el = sum(int(jnp.prod(jnp.array(s))) for s in ops_shapes)
            return total_el * int(dtypesize)

        peak_terms = max((size_for_term(t) for t in self._terms), default=0)

        state_bytes = sum(int(jnp.prod(jnp.array(shapes[a]))) * int(dtypesize) for a in self._exprs.keys())

        expr_transient = 0
        for E in self._exprs.values():
            estimator = getattr(E.expr, "estimate_bytes_per_sample", None)
            if callable(estimator):
                expr_transient += int(estimator(dtype=_dtype, particle_size=P))

        total = expr_transient + state_bytes + peak_terms
        return int(total) if total > 0 else None
