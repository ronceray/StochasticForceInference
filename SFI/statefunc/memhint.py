# SFI/statefunc/memhint.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import jax.numpy as jnp

# ──────────────────────────────────────────────────────────────────────────────
# Core types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemHint:
    """
    Conservative memory footprint per SINGLE sample.
    - per_sample_bytes scales with the number of samples.
    - persistent_bytes is model state / CSR / weights, does NOT scale with samples.
    """

    per_sample_bytes: int = 0
    persistent_bytes: int = 0

    def __add__(self, other: "MemHint") -> "MemHint":
        return MemHint(
            self.per_sample_bytes + other.per_sample_bytes,
            self.persistent_bytes + other.persistent_bytes,
        )

    def scaled(self, k: int) -> "MemHint":
        """Scale only the transient part (useful if you ever convert to total bytes)."""
        return MemHint(self.per_sample_bytes * int(k), self.persistent_bytes)


@dataclass(frozen=True)
class SampleMeta:
    """
    Optional single-sample context to refine estimates.

    P: number of particles in ONE sample (for nodes with particle axes, pdepth>0).
    K: arity for interaction gathers when fixed/known (pairs=2, etc.).
    has_v: whether a velocity block will be present.
    has_mask: whether a boolean mask participates in the call.
    """

    P: Optional[int] = None
    K: Optional[int] = None
    has_v: Optional[bool] = None
    has_mask: Optional[bool] = None

    @staticmethod
    def from_arrays(x=None, v=None, mask=None) -> "SampleMeta":
        """
        Try to recover P from provided single-sample arrays without allocating anything.
        Heuristics:

          - If x has at least 2 dims and looks like (..., dim) with a leading particle axis,
            we guess P from the axis before `dim`. If ambiguous, we leave P=None.
          - v and mask only toggle flags; they don't change P.
        """
        P = None
        if x is not None and hasattr(x, "shape"):
            # super conservative: if it looks like (..., P, dim) with small dim
            if x.ndim >= 2:
                dim_guess = int(x.shape[-1])
                if 1 <= dim_guess <= 16:
                    # take the previous axis as P
                    P = int(x.shape[-2])
        return SampleMeta(
            P=P,
            K=None,
            has_v=(v is not None),
            has_mask=(mask is not None),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def itemsize_of(dtype) -> int:
    """Return dtype itemsize as an int; defaults to float32 when dtype is None."""
    return int(jnp.dtype(jnp.float32 if dtype is None else dtype).itemsize)


class _HasContract(Protocol):
    # Minimal surface we read from nodes (already present on your BaseNode/ops).
    rank: int
    dim: Optional[int]
    pdepth: int
    n_features: int


def resolve_P(particle_size: Optional[int], sample) -> Optional[int]:
    """Pick P from explicit particle_size, else from SampleMeta or array, else None."""
    if particle_size is not None:
        return int(particle_size)
    if sample is None:
        return None

    # Accept either a SampleMeta or an array-like (single-sample x)
    if isinstance(sample, SampleMeta):
        sm = sample
    else:
        # best-effort duck-typing: treat it as x and try to read P
        try:
            sm = SampleMeta.from_arrays(x=sample)
        except Exception:
            sm = None

    return None if (sm is None or sm.P is None) else int(sm.P)


def output_elems_per_sample(node: _HasContract, *, particle_size: Optional[int]) -> int:
    """
    Count output elements of a node for ONE sample from its static contract.

    Output shape suffix is ``(*particle_axes, *rank_axes, n_features)``
    with ``particle_axes = (P,) * pdepth`` and ``rank_axes = (dim,) * rank``.

    If ``particle_size`` is None, treat ``P = 1`` (safe lower bound for
    batch picking).
    """
    n_features = int(getattr(node, "n_features", 0) or 0)
    if n_features <= 0:
        return 0

    dim = getattr(node, "dim", None)
    rank = int(getattr(node, "rank", 0) or 0)
    if dim is None and rank > 0:
        return 0  # cannot infer rank block without dim

    pdepth = int(getattr(node, "pdepth", 0) or 0)
    P = 1 if particle_size is None else int(particle_size)

    elems = (dim or 1) ** rank
    if pdepth:
        elems *= P**pdepth
    elems *= n_features
    return int(elems)


def output_bytes_per_sample(node: _HasContract, *, dtype, particle_size: Optional[int]) -> int:
    """Translate element count to bytes using dtype itemsize."""
    return output_elems_per_sample(node, particle_size=particle_size) * itemsize_of(dtype)


def default_leaf_hint(node: _HasContract, *, dtype, particle_size: Optional[int], mode: str) -> MemHint:
    """
    Default for leaf-like nodes: count only the output buffer per sample.
    Composite nodes will also include their children.
    """
    return MemHint(per_sample_bytes=output_bytes_per_sample(node, dtype=dtype, particle_size=particle_size))


def broadcast_extra_bytes_for_children(*, children: Iterable, dtype, particle_size: Optional[int]) -> int:
    """
    When children have different particle depths, lower-depth outputs are broadcast
    to match the max. Materializing that broadcast costs memory. We conservatively
    account for an additional slab equal to (P^Δ - 1) times the child's OUTPUT bytes.
    """
    # target = max pdepth among children
    target = 0
    pd = []
    for ch in children:
        p = int(getattr(ch, "pdepth", 0) or 0)
        pd.append(p)
        if p > target:
            target = p
    if target == 0:
        return 0

    extra = 0
    P = 1 if particle_size is None else int(particle_size)
    for ch in children:
        p = int(getattr(ch, "pdepth", 0) or 0)
        if p < target:
            delta = target - p
            # only the OUTPUT slab needs broadcasting; not the child's internal working set
            base_out = output_bytes_per_sample(ch, dtype=dtype, particle_size=particle_size)
            factor = P**delta - 1
            if factor > 0 and base_out > 0:
                extra += base_out * factor
    return int(extra)


def default_op_hint(
    node: _HasContract,
    *,
    children: Iterable,
    dtype,
    particle_size: Optional[int],
    mode: str,
) -> MemHint:
    """
    Default for composite ops: sum(children.hint) + my output buffer + broadcast overhead.
    We assume child outputs coexist while the op constructs its result.
    """
    hint = MemHint()
    for ch in children:
        hint = hint + ch.memory_hint(dtype=dtype, particle_size=particle_size, mode=mode)
    # add broadcast materialization if children pdepth differ
    hint = MemHint(
        hint.per_sample_bytes
        + broadcast_extra_bytes_for_children(children=children, dtype=dtype, particle_size=particle_size),
        hint.persistent_bytes,
    )
    # add this op's own output
    return hint + default_leaf_hint(node, dtype=dtype, particle_size=particle_size, mode=mode)


def inflate_for_grad(hint: MemHint, *, factor: float = 2.0) -> MemHint:
    """
    Blanket inflation for gradient-mode nodes to account for tangents/tapes.
    Keep it simple and conservative; adjust locally if you get better bounds.
    """
    return MemHint(
        per_sample_bytes=int(hint.per_sample_bytes * factor),
        persistent_bytes=hint.persistent_bytes,
    )
