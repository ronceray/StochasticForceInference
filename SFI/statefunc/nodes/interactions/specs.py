from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

# ──────────────────────────────────────────────────────────────────────────────
# Specs = “what to dispatch”
# These are plain holders (or builders via rules) that describe the set of
# interactions. They do not compute anything by themselves.
# ──────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# Convention: cache-only extras keys
# -----------------------------------------------------------------------------
CACHE_PREFIX = "_cache/"
"""
Keys starting with ``_cache/`` are reserved for auto-generated structural extras.

These objects should:
- be produced host-side (outside JIT),
- be purgeable whenever dataset/collection context changes,
- typically be marked as structural (dispatcher-owned) and not forwarded to
  children operators.
"""


@dataclass(frozen=True)
class PairsCSR:
    """
    K=2 pairs encoded in CSR form.

    Optional metadata (forwarded to interactor):
      - edge_extras: per-edge arrays aligned with `indices` order (shape (..., nnz, ...))
      - slot_extras: per-edge-per-slot arrays (shape (..., nnz, 2, ...))
    """

    indptr: Array
    indices: Array
    edge_extras: Optional[Mapping[str, Array]] = None
    slot_extras: Optional[Mapping[str, Array]] = None

    def __repr__(self) -> str:
        try:
            p = int(self.indptr.shape[-1]) - 1
            m = int(self.indices.shape[-1])
            return f"PairsCSR(P={p}, nnz={m})"
        except Exception:
            return "PairsCSR(?)"


@dataclass(frozen=True)
class HyperFixed:
    """
    Fixed-K hyperedges: each row lists K participants in one interaction.

    This is the natural container for regular-grid stencils:
      - M = number of hyperedges (often M == P, one per focal site),
      - K = number of participants in each hyperedge (center + neighbors).

    Attributes
    ----------
    hyper
        Integer array of shape (..., M, K) with indices in [0, P).
        For grid stencils built by :func:`~SFI.statefunc.nodes.interactions.stencils.hyperfixed_square_stencil`,
        the shape is typically (P, K).
    slot_mask
        Optional boolean array of shape (..., M, K). If present, marks which
        slots are *in-bounds* (or otherwise active). This is useful for "drop"
        boundary conditions and for operators that want to treat boundary slots
        differently. For "noflux", you may still keep it for diagnostics.
    edge_extras
        Optional mapping of per-hyperedge arrays, each shaped (..., M, ...).
        Forwarded to the interactor as-is.
    slot_extras
        Optional mapping of per-slot arrays, each shaped (..., M, K, ...).
        Typical entry: ``{"offset": (M,K,ndim)}`` storing integer offset vectors
        for each slot (directionality).
    """

    hyper: Array
    slot_mask: Optional[Array] = None
    edge_extras: Optional[Mapping[str, Any]] = None
    slot_extras: Optional[Mapping[str, Any]] = None

    def __repr__(self) -> str:
        try:
            m, k = int(self.hyper.shape[-2]), int(self.hyper.shape[-1])
            return f"HyperFixed(M={m}, K={k})"
        except Exception:
            return "HyperFixed(?)"


@dataclass(frozen=True)
class HyperCSR:
    """
    Variable-K hyperedges (CSR-like).

      - he_indptr: shape (..., M+1)
      - he_indices: shape (..., total_K)

    Optional metadata:
      - slot_ids: integer codes aligned with he_indices (shape (..., total_K,))
      - edge_extras: per-hyperedge arrays (shape (..., M, ...))
      - slot_extras: per-slot arrays aligned with he_indices (shape (..., total_K, ...))
    """

    he_indptr: Array
    he_indices: Array
    slot_ids: Optional[Array] = None
    edge_extras: Optional[Mapping[str, Array]] = None
    slot_extras: Optional[Mapping[str, Array]] = None

    def __repr__(self) -> str:
        try:
            m = int(self.he_indptr.shape[-1]) - 1
            nz = int(self.he_indices.shape[-1])
            return f"HyperCSR(M={m}, total_K={nz})"
        except Exception:
            return "HyperCSR(?)"


# ──────────────────────────────────────────────────────────────────────────────
# Rules = “how to build a spec from (x, …) at call-time”
# A rule advertises arity for constructor-time K-compat checks and builds the
# concrete spec in __call__/build.
# ──────────────────────────────────────────────────────────────────────────────


class SpecRule(eqx.Module):
    """Base class for spec-building rules.

    A rule advertises its arity for constructor-time K-compat checks and can
    declare two kinds of extras:

        - structural_extras(): keys owned by the rule/dispatcher (CSR arrays,
          neighbor lists, etc.). They are required to build the schedule and are
          **never forwarded** to children.

        - required_extras(): presence-only keys that must be present at call time
          for this rule to build, and that will be forwarded downstream (globals).
          No broadcasting is enforced here; the runtime handles presence.

    Cache-only extras convention
    ----------------------------
    Any auto-generated structural arrays should live under the ``_cache/`` prefix.
    Degradation and other context-changing transforms are expected to drop them.

    """

    def arity(self) -> Tuple[Literal["fixed", "variable"], Optional[int]]:
        """
        Return ("fixed", K) or ("variable", None).
        Used by the dispatcher constructor for K-compat checks.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement arity")

    def required_extras(self) -> tuple[str, ...]:
        """
        Presence-only extras the rule expects at build time and that will be
        forwarded downstream (globals). Use this sparingly—most rules either
        use no globals or only structural extras.
        """
        return ()

    def structural_extras(self) -> tuple[str, ...]:
        """
        Structural arrays used by the rule/dispatcher (e.g., CSR 'indptr/indices').
        They are *not* forwarded to children and must not be broadcast-checked.
        """
        return ()

    def build(
        self,
        x: Array,
        *,
        v: Array | None = None,
        mask: Array | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> PairsCSR | HyperFixed | HyperCSR:
        raise NotImplementedError(f"{type(self).__name__} must implement build")


class AutoPairs(SpecRule):
    """
    Build an all-pairs CSR on the fly.

    symmetric=True → directed i→j for all i≠j (each unordered pair counted twice).
    symmetric=False → keep only i<j rows (each unordered pair counted once).
    """

    symmetric: bool = eqx.field(static=True, default=True)
    exclude_self: bool = eqx.field(static=True, default=True)

    def arity(self) -> Tuple[Literal["fixed", "variable"], Optional[int]]:
        return ("fixed", 2)

    def build(self, x, *, v=None, mask=None, extras=None) -> PairsCSR:
        P = x.shape[-2]
        i = jnp.arange(P, dtype=jnp.int32)

        if self.symmetric:
            if self.exclude_self:
                # Each row has degree P-1; per-row columns are [0..P-1] \ {i}.
                deg = jnp.full((P,), P - 1, dtype=jnp.int32)
                indptr = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(deg)])
                base = jnp.arange(P - 1, dtype=jnp.int32)  # (P-1,)
                ii = i[:, None]  # (P,1)
                # For each row i, map base -> [0..i-1, i+1..P-1] via +1 shift after i
                cols = base + (base >= ii).astype(jnp.int32)  # (P, P-1)
                indices = cols.reshape(-1)  # grouped by row
                return PairsCSR(indptr=indptr, indices=indices)
            else:
                # Degree P per row; columns are 0..P-1 for every row
                deg = jnp.full((P,), P, dtype=jnp.int32)
                indptr = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(deg)])
                cols = jnp.tile(i, (P, 1))  # (P, P)
                indices = cols.reshape(-1)
                return PairsCSR(indptr=indptr, indices=indices)

        # Asymmetric: keep only j > i (strict upper triangle).
        # Per-row degree is P-1-i; indptr follows directly.
        deg = (P - 1 - i).astype(jnp.int32)
        indptr = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(deg)])
        # jnp.triu_indices avoids boolean gathers and returns rows grouped, then cols
        _, cols = jnp.triu_indices(P, k=1)
        indices = cols.astype(jnp.int32)
        return PairsCSR(indptr=indptr, indices=indices)


class FromExtrasPairsCSR(SpecRule):
    """
    Build a pairs spec by reading CSR arrays from `extras`.
    These are *structural* arrays (dispatcher-owned, never forwarded).
    """

    key_indptr: str = eqx.field(static=True)
    key_indices: str = eqx.field(static=True)

    def arity(self) -> Tuple[Literal["fixed", "variable"], Optional[int]]:
        return ("fixed", 2)

    def required_extras(self) -> tuple[str, ...]:
        # Presence is checked in build(); we don't list CSR here to avoid forwarding.
        return ()

    def structural_extras(self) -> tuple[str, ...]:
        # Dispatcher will consume these; children never see them.
        return (self.key_indptr, self.key_indices)

    def build(
        self,
        x: Array,
        *,
        v: Array | None = None,
        mask: Array | None = None,
        extras=None,
    ) -> PairsCSR:
        if extras is None:
            raise KeyError("FromExtrasPairsCSR: extras is required")
        return PairsCSR(indptr=extras[self.key_indptr], indices=extras[self.key_indices])


class CachedRule(SpecRule):
    """Cache the concrete spec built by an underlying rule.

    This is a *Python-side* cache keyed by:
      (P, rule.cache_key(extras))

    It is designed to avoid rebuilding large structural objects (CSR, hyper tables)
    every time the dispatcher is evaluated.

    Notes / caveats
    ---------------
    - The cache lives inside the rule instance (static field), so it is shared
      across all uses of that exact node object.
    - `cache_key(extras)` must return a **hashable** object (typically tuples of
      Python ints/floats/strings).
    - The rule is expected to be called in contexts where `extras` contains
      **concrete** values (not abstract tracers). If you call a function that
      triggers `build()` under JIT tracing with tracer-valued extras, any attempt
      to convert to Python scalars inside `cache_key` will fail. In that case,
      either:
        (i) keep `cache_key` shape-only, or
        (ii) ensure spec building happens outside the traced region.
    """

    rule: SpecRule = eqx.field(static=True)
    _cache: dict = eqx.field(static=True, default_factory=dict)

    def arity(self):
        return self.rule.arity()

    def required_extras(self):
        return self.rule.required_extras()

    def structural_extras(self):
        return self.rule.structural_extras()

    def _key(self, x, extras):
        # P disambiguates different particle counts. For grids, P = prod(grid_shape).
        P = int(x.shape[-2])

        # Optional secondary key advertised by the wrapped rule.
        # Must be hashable, typically a tuple of Python scalars.
        k2 = getattr(self.rule, "cache_key", lambda e: ())(extras)
        return (P, k2)

    def build(self, x, *, v=None, mask=None, extras=None):
        key = self._key(x, extras)
        if key in self._cache:
            return self._cache[key]
        spec = self.rule.build(x, v=v, mask=mask, extras=extras)
        self._cache[key] = spec
        return spec


class FromExtrasHyperFixed(SpecRule):
    """
    Build a HyperFixed spec by *reading* arrays from `extras`.

    This rule is deliberately JIT-safe:
      - no numpy conversions,
      - no Python hashing on tracer values,
      - no caching logic.

    It is meant to be paired with a host-side preparation step that inserts
    the structural arrays (hyper tables, slot masks, slot offsets, ...) into
    `extras_global` before any jitted StateExpr evaluation.
    """

    key_hyper: str = eqx.field(static=True)
    key_slot_mask: Optional[str] = eqx.field(static=True, default=None)
    key_slot_offset: Optional[str] = eqx.field(static=True, default=None)

    def arity(self) -> Tuple[Literal["fixed", "variable"], Optional[int]]:
        # fixed-K, but K is not known at constructor time here
        return ("fixed", None)

    def structural_extras(self) -> tuple[str, ...]:
        keys = [self.key_hyper]
        if self.key_slot_mask is not None:
            keys.append(self.key_slot_mask)
        if self.key_slot_offset is not None:
            keys.append(self.key_slot_offset)
        return tuple(keys)

    def build(self, x, *, v=None, mask=None, extras: Mapping[str, Any] | None = None) -> HyperFixed:
        if extras is None:
            raise KeyError("FromExtrasHyperFixed: extras is required")

        hyper = extras[self.key_hyper]
        slot_mask = extras[self.key_slot_mask] if (self.key_slot_mask is not None) else None

        slot_extras = None
        if self.key_slot_offset is not None:
            slot_extras = {"offset": extras[self.key_slot_offset]}

        return HyperFixed(hyper=hyper, slot_mask=slot_mask, slot_extras=slot_extras)
