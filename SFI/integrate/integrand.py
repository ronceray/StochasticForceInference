from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .timeops import TimeOp

import jax
import jax.numpy as jnp
import opt_einsum as oe  # required

__all__ = ["ExprOperand", "TimeOperand", "ConstOperand", "Term", "Integrand"]


def _state_mask(streams):
    """Mask handed to state expressions: positions known (``mask_state``).

    Not the fit mask ``mask_out``: a particle whose increment is excluded from
    the fit (dynamic mask) must still be visible to its neighbours'
    interactions.  Falls back to ``mask_out`` for producers that do not emit
    ``mask_state``.
    """
    m = streams.get("mask_state", None)
    return streams.get("mask_out", None) if m is None else m


#: Reserved per-particle extra fixing the periodic image of every pair
#: (:func:`SFI.bases.pairs.pbc_displacement`).
PBC_REF_KEY = "pbc_ref"

# Frames read by the position / increment streams (offsets from the row's anchor).
_STREAM_FRAMES = {
    "X_m4": (-4,), "X_m3": (-3,), "X_minusminus": (-2,), "X_minus": (-1,), "X": (0,),
    "X_plus": (1,), "X_plusplus": (2,), "X_p3": (3,), "X_p4": (4,),
    "dX_minus": (-1, 0), "dX": (0, 1), "dX_plus": (1, 2),
}
_POSITION_STREAMS = {f[0]: k for k, f in _STREAM_FRAMES.items() if not k.startswith("d")}


def _frames(required) -> Set[int]:
    """Frame offsets the given stream keys read (the anchor included)."""
    from SFI.trajectory.dataset import touched_offsets

    out: Set[int] = set()
    for k in required:
        if k in _STREAM_FRAMES:
            out.update(_STREAM_FRAMES[k])
        else:
            out.update(touched_offsets({k}))
    return out


def _pbc_reference_stream(exprs, times, scope=()) -> Optional[str]:
    """Frame fixing the periodic image of every pair in one row of a program.

    Re-picking each pair's minimum image at every frame makes
    direction-carrying pair features (``φ(r) r̂_ij``) jump when a pair
    crosses ``|Δx| = L/2``; a program combining pair features at several
    frames (Stratonovich trapezoid, shifted Gram, ULI) integrates those jumps
    into an O(1) bias.  All its evaluations are therefore put on the image of
    one reference frame, making the row integrand a smooth function of the
    frames it reads.  Only periodic pair features (``box`` set) take part;
    free-space bases and programs evaluating them at a single state (one
    frame, or one combination of frames) are untouched.

    The reference must precede every increment (or velocity) the row reads.
    A frame inside an increment carries its localisation noise, which the
    increment carries divided by Δt: the image would switch with it and the
    jump term return at O(σ²/Δt) — the correlation the noise-decorrelating
    stencils are built to avoid.  A frame after the start of an increment
    depends on the increment itself (``X_{t+2} = X_t + ΔX_t + ΔX_{t+1}``), so
    the image would correlate with the dynamical noise even on clean data.
    The reference is

    1. the earliest frame the pair features are evaluated at, when it
       precedes every increment (the features keep its minimum image), else
    2. the latest admissible frame the row already reads — this program's
       stencil or the estimator's shared row scope ``scope``
       (:func:`SFI.integrate.shared_rows`) — so no row is lost to it, else
    3. the frame just before everything the row reads (one more frame must
       be observed).
    """
    pbc = [E for E in exprs if E.has_pbc_pairs]
    if not pbc:
        return None
    # One evaluation state (a frame, or one combination of frames such as a
    # reconstructed position) has no frame-to-frame difference to protect.
    if len({frozenset(E.x.requires) for E in pbc}) <= 1:
        return None
    pair_frames = set()  # frames the pair displacements are evaluated at
    for E in pbc:
        pair_frames |= _frames(E.x.requires)
    noisy = set()  # frames entering increments / velocities
    for T in times:
        noisy |= _frames(T.requires)
    for E in exprs:
        for op in (E.v, E.tx, E.tv):
            if op is not None:
                noisy |= _frames(op.requires)
    read = set(pair_frames) | noisy | {0}
    for E in exprs:
        read |= _frames(E.requires)
    first_noisy = min(noisy) if noisy else None
    if first_noisy is None or min(pair_frames) < first_noisy:
        ref = min(pair_frames)
    else:
        admissible = {f for f in (read | set(scope)) if f < first_noisy and f in _POSITION_STREAMS}
        ref = max(admissible) if admissible else min(read) - 1
    if ref not in _POSITION_STREAMS:
        raise NotImplementedError(
            f"periodic pair features: no position stream for the image reference frame {ref}."
        )
    return _POSITION_STREAMS[ref]


def _with_pbc_ref(E: "ExprOperand", ref: Optional[str], kwargs: Dict[str, Any], streams) -> None:
    """Hand a periodic pair expression the row's image reference positions."""
    if ref is None or not E.has_pbc_pairs:
        return
    extras = dict(kwargs.get("extras") or {})
    extras[PBC_REF_KEY] = streams[ref]
    kwargs["extras"] = extras


@dataclass(frozen=True)
class ExprOperand:
    """
    Wrap a state expression evaluated at x (and optionally v).

    Contract
    --------
    expr(x, v=..., mask=..., extras=..., params=...) -> array
      - features sit on the last axis
      - 'mask' is forwarded from the 'mask_state' stream when present (positions
        known at every frame the row reads; a dynamically masked particle stays
        visible, e.g. as an interaction neighbour), else from 'mask_out'.  Rows are
        excluded from the reduction by 'mask_out', downstream in the integrator.
    """

    expr: object
    x: "TimeOp"
    v: Optional["TimeOp"] = None
    # Template / default parameters. Call-time params may override this.
    params_template: Optional[Mapping] = None
    alias: str = "E"
    # Optional tangents: the operand is then the directional derivative
    # d/de expr(x + e tx, v + e tv) at e = 0 (one forward-mode jvp), same shape
    # as expr.  Used to contract a Jacobian with a vector without forming it.
    tx: Optional["TimeOp"] = None
    tv: Optional["TimeOp"] = None

    @property
    def required_extras(self) -> Tuple[str, ...]:
        """Keys this expression demands in the extras mapping."""
        re = getattr(self.expr, "required_extras", ()) or ()
        return tuple(re)

    @property
    def has_pbc_pairs(self) -> bool:
        """Periodic pair features (accept the ``pbc_ref`` image reference)."""
        return PBC_REF_KEY in (getattr(self.expr, "particle_extras", ()) or ())

    @property
    def requires(self) -> Set[str]:
        req = set(self.x.requires)
        if self.v is not None:
            req |= set(self.v.requires)
        for t in (self.tx, self.tv):
            if t is not None:
                req |= set(t.requires)
        if self.required_extras or getattr(self.expr, "structural_extras", ()):
            req.add("extras")  # ensure collection streams extras (incl. dispatcher-owned CSR tables)
        return req


def _evaluate(E: "ExprOperand", x, v, kwargs: Dict[str, Any], streams) -> jnp.ndarray:
    """``E.expr`` at (x, v), or its jvp along the operand's tangents."""
    if E.tx is None and E.tv is None:
        return E.expr(x, v=v, **kwargs) if v is not None else E.expr(x, **kwargs)
    if E.tv is not None and v is None:
        raise ValueError(f"[{E.alias}] a velocity tangent needs a velocity operand")
    tx = None if E.tx is None else E.tx(**streams)
    tv = None if E.tv is None else E.tv(**streams)
    if v is None:
        return jax.jvp(lambda x_: E.expr(x_, **kwargs), (x,), (tx,))[1]
    if tx is None:
        return jax.jvp(lambda v_: E.expr(x, v=v_, **kwargs), (v,), (tv,))[1]
    if tv is None:
        return jax.jvp(lambda x_: E.expr(x_, v=v, **kwargs), (x,), (tx,))[1]
    return jax.jvp(lambda x_, v_: E.expr(x_, v=v_, **kwargs), (x, v), (tx, tv))[1]


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
    def _stencil(self) -> Set[str]:
        req = set()
        for e in self._exprs.values():
            req |= e.requires
        for t in self._times.values():
            req |= t.requires
        return req

    def pbc_reference_in(self, scope=()) -> Optional[str]:
        """Stream fixing the pair images of a row given the row scope (see :func:`_pbc_reference_stream`)."""
        return _pbc_reference_stream(self._exprs.values(), self._times.values(), scope)

    @property
    def pbc_reference(self) -> Optional[str]:
        """Stream fixing the pair images of a row, within the active ``shared_rows`` scope."""
        from .api import _SHARED_ROW_FRAMES

        return self.pbc_reference_in(_SHARED_ROW_FRAMES.get())

    def require(self) -> Set[str]:
        req = self._stencil()
        ref = self.pbc_reference
        if ref is not None:
            req.add(ref)
        return req

    def required_extras(self) -> Set[str]:
        keys: Set[str] = set()
        for E in self._exprs.values():
            keys.update(E.required_extras)
        return keys

    def structural_extras(self) -> Set[str]:
        """Dispatcher-owned extras (CSR / hyperedge tables) read by any operand expression."""
        keys: Set[str] = set()
        for E in self._exprs.values():
            keys.update(getattr(E.expr, "structural_extras", ()) or ())
        return keys

    def __call__(self, *, params: Optional[Any] = None, **streams):
        bufs: Dict[str, jnp.ndarray] = {}
        ref = self.pbc_reference

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
            state_mask = _state_mask(streams)
            if state_mask is not None:
                kwargs["mask"] = state_mask
            if "extras" in streams:
                kwargs["extras"] = streams["extras"]
            _with_pbc_ref(E, ref, kwargs, streams)

            # Parameter routing:
            # - if params is not None: use it as the actual parameter object
            # - else: fall back to the operand's params_template
            par = E.params_template if params is None else params
            if par is not None:
                kwargs["params"] = par

            bufs[a] = _evaluate(E, x, v, kwargs, streams)

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
        ref = self.pbc_reference

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
            state_mask = _state_mask(streams)
            if state_mask is not None:
                kwargs["mask"] = state_mask  # (K, N)
            if "extras" in streams:
                kwargs["extras"] = streams["extras"]
            _with_pbc_ref(E, ref, kwargs, streams)

            par = E.params_template if params is None else params
            if par is not None:
                kwargs["params"] = par

            bufs[a] = _evaluate(E, x, v, kwargs, streams)
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
        dtypesize: Optional[int] = None,
    ) -> Optional[int]:
        """
        Conservative upper bound in bytes per **time-sample** (one k-row).
        Uses StateExpr static hints + shapes-only einsum path. No evaluation.

        The runtime hands every expression the full ``(N, d)`` particle block
        of a row (``(K, N, d)`` in the batched kernel).  A ``pdepth=0``
        expression is therefore *broadcast* over the particle axis and its
        output is ``(N, ..., F)`` -- one copy per particle, exactly like a
        ``pdepth=1`` output -- and so is every einsum intermediate built from
        it (e.g. the ``(N, F, F)`` Gram block before the particle sum).  The
        estimate counts ``max(1, pdepth)`` particle axes accordingly; dropping
        the broadcast axis under-estimates by a factor ``N`` and silently
        disables chunking for stacked ``(T, N, d)`` data.

        ``dtypesize`` defaults to the item size of the ``X`` stream (4 bytes
        when it is unavailable).
        """
        _X = sample_streams.get("X", None)
        _dtype = None if _X is None else _X.dtype
        if dtypesize is None:
            dtypesize = 4 if _dtype is None else int(jnp.dtype(_dtype).itemsize)
        P = int(sample_streams["N_total"])  # use provided, conservative

        # ---- shapes without evaluation
        def expr_pdepth(E) -> int:
            return int(getattr(E.expr, "pdepth", 0))

        def expr_shape(E) -> tuple[int, ...]:
            p = expr_pdepth(E)
            r = int(getattr(E.expr, "rank", 0))
            d = int(getattr(E.expr, "dim", 0))
            F = int(getattr(E.expr, "n_features", 1))
            # Full particle axes per sample; a pdepth=0 output still carries
            # the (broadcast) particle axis of the data block.
            paxes = (P,) * max(1, p)
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
                b = int(estimator(dtype=_dtype, particle_size=P))
                if expr_pdepth(E) == 0:
                    # Node hints are per evaluation row; a pdepth=0 graph is
                    # evaluated once per particle of the block.
                    b *= P
                expr_transient += b

        total = expr_transient + state_bytes + peak_terms
        return int(total) if total > 0 else None
