from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jaxtyping import Array

from ...memhint import MemHint, SampleMeta, itemsize_of, resolve_P
from ..base import BaseNode, BaseOpNode
from ..contract import Rank, _ContractMixin
from ..leaf import InteractionLeaf
from .specs import AutoPairs, FromExtrasPairsCSR, HyperCSR, HyperFixed, PairsCSR, SpecRule


# Small helper for rank axis sizes (vector/matrix → dim per axis).
class _BatchedParticleExtra:
    """Trace-time tag: a particle extra carrying the state's batch axes, ``(*batch, P, ...)``."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _rank_shape(rank: Rank, d: int) -> Tuple[int, ...]:
    r = int(rank)
    return () if r == 0 else (d,) * r


# Walk the interactor subtree and infer K mode; “fixed wins”.
def _infer_interactor_K(interactor: BaseNode):
    """Return ``("fixed", K)`` or ``("variable", Kmax)`` for an interactor.

    Every leaf of one interactor receives the same gathered neighbourhood, so
    the arities must agree: one fixed ``K``, one ``Kmax``, and a variable
    leaf's ``Kmax`` equal to a fixed leaf's ``K`` when both are present.
    """
    fixed_Ks, var_Kmaxs = set(), set()

    def _walk(n):
        if isinstance(n, InteractionLeaf):
            if n.mode == "fixed":
                fixed_Ks.add(int(n.K))
            elif n.Kmax is not None:
                var_Kmaxs.add(int(n.Kmax))
        for ch in getattr(n, "children", ()):
            _walk(ch)

    _walk(interactor)

    if len(fixed_Ks) > 1:
        raise ValueError(f"Interactor mixes fixed arities {sorted(fixed_Ks)}; split across dispatchers.")
    if len(var_Kmaxs) > 1:
        raise ValueError(f"Interactor mixes variable arities Kmax={sorted(var_Kmaxs)}; use a single Kmax.")
    if fixed_Ks:
        K = int(next(iter(fixed_Ks)))
        if var_Kmaxs and var_Kmaxs != {K}:
            raise ValueError(
                f"Interactor mixes fixed K={K} with variable Kmax={sorted(var_Kmaxs)}; "
                "they receive the same neighbourhood, so Kmax must equal K."
            )
        return "fixed", K
    if var_Kmaxs:
        return "variable", int(next(iter(var_Kmaxs)))

    # No InteractionLeaves found → likely a SimpleLeaf graph. Not allowed as interactor.
    raise ValueError("Interactor must be built from InteractionLeaves (particles_input=True, pdepth=0).")


def _interactor_per_slot(interactor: BaseNode) -> bool:
    """True if the interactor is a per-slot leaf (see ``make_interactor(per_slot=True)``).

    Composite nodes check per-tuple output shapes, so a per-slot leaf must be
    the interactor itself rather than part of a composition.
    """
    found = []

    def _walk(n):
        if isinstance(n, InteractionLeaf) and getattr(n, "per_slot", False):
            found.append(n)
        for ch in getattr(n, "children", ()):
            _walk(ch)

    _walk(interactor)
    if not found:
        return False
    if not (len(found) == 1 and found[0] is interactor):
        raise NotImplementedError(
            "A per-slot interactor (per_slot=True) must be dispatched directly: "
            "compose the dispatched bases instead of the interactors."
        )
    return True


class InteractionDispatcher(BaseOpNode):
    """
    Stream K-body interactions defined by `spec` (or a rule), evaluate a local
    interactor that consumes (K, d), and scatter-reduce to per-particle or global
    outputs.

    Returns:
      owners='global' → (..., ``*rank``, m)      # pdepth = 0
      else             → (..., P, ``*rank``, m)  # pdepth = 1

    Contract invariants:
      - This node always *consumes* a particle axis (…, P, d) → particles_input=True.
      - Output pdepth depends only on `owners`.
      - Interactor must be local: particles_input=True, pdepth=0.

    structural_extras : tuple[str, ...]
        Keys used by the rule/dispatcher (e.g., CSR arrays). They are never forwarded
        to the interactor subtree.
    """

    CONTRACT_MODE = None  # we use unary inherit

    # Child
    interactor: BaseNode = eqx.field(static=True)
    # Spec or rule
    spec: PairsCSR | HyperFixed | HyperCSR | SpecRule = eqx.field(static=True)

    # Reduction / ownership
    owners: Literal["focal", "all", "custom", "global"] = eqx.field(static=True, default="focal")
    focal_index: int = eqx.field(static=True, default=0)
    owner_weights: Optional[Array] = eqx.field(static=False, default=None)
    reducer: Literal["sum", "mean", "max"] = eqx.field(static=True, default="sum")
    normalize_by_degree: bool = eqx.field(static=True, default=False)
    exclude_self: bool = eqx.field(static=True, default=True)
    chunk_size: Optional[int] = eqx.field(static=True, default=None)

    # K-mode info (constructor-time check; “fixed wins”)
    _kmode: Literal["fixed", "variable"] = eqx.field(static=True, repr=False)
    _K: Optional[int] = eqx.field(static=True, repr=False, default=None)
    _enforce_fixedK_on_hypercsr: bool = eqx.field(static=True, repr=False, default=False)
    # Kernel returns one value per slot, (M, K, *rank, F); see make_interactor(per_slot=True).
    _per_slot: bool = eqx.field(static=True, repr=False, default=False)

    # ───────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        interactor: BaseNode,
        *,
        spec: PairsCSR | HyperFixed | HyperCSR | SpecRule,
        owners: Literal["focal", "all", "custom", "global"] = "focal",
        focal_index: int = 0,
        owner_weights: Optional[Array] = None,
        reducer: Literal["sum", "mean", "max"] = "sum",
        normalize_by_degree: bool = False,
        exclude_self: bool = True,
        chunk_size: Optional[int] = None,
    ):
        # Interactor must be local (no global P leaking through)
        if not interactor.particles_input or interactor.pdepth != 0:
            raise ValueError("Interactor must have particles_input=True and pdepth=0.")

        object.__setattr__(self, "interactor", interactor)
        object.__setattr__(self, "spec", spec)

        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "focal_index", int(focal_index))
        object.__setattr__(self, "owner_weights", owner_weights)
        object.__setattr__(self, "reducer", reducer)
        object.__setattr__(self, "normalize_by_degree", bool(normalize_by_degree))
        object.__setattr__(self, "exclude_self", bool(exclude_self))
        object.__setattr__(self, "chunk_size", chunk_size)

        # Interactor K-mode (K is the fixed arity, or Kmax for a variable interactor)
        kmode, K = _infer_interactor_K(interactor)
        spec_mode, spec_K = self._spec_arity(spec)
        if K is not None and spec_K is not None and K != spec_K:
            if kmode == "fixed":
                raise ValueError(f"Fixed-K mismatch between interactor (K={K}) and spec (K={spec_K}).")
            raise ValueError(f"Arity mismatch between variable interactor (Kmax={K}) and fixed-K spec (K={spec_K}).")

        # “Fixed wins” across interactor+spec
        if kmode == "fixed" or spec_mode == "fixed":
            kmode, K = "fixed", (K if K is not None else spec_K)
            if isinstance(spec, HyperCSR):
                object.__setattr__(self, "_enforce_fixedK_on_hypercsr", True)

        object.__setattr__(self, "_kmode", kmode)
        object.__setattr__(self, "_K", K)

        per_slot = _interactor_per_slot(interactor)
        if per_slot and isinstance(spec, (PairsCSR, AutoPairs, FromExtrasPairsCSR)):
            raise NotImplementedError("per-slot interactors are supported for hyperedge specs, not pair specs.")
        object.__setattr__(self, "_per_slot", per_slot)

        # Build BaseOpNode with one child (unifies params/extras contracts)
        super().__init__(interactor)

        # If rule advertises broadcast-checked extras, union them (currently none).
        # Structural extras are owned by the rule/dispatcher and are never forwarded downstream.
        if isinstance(spec, SpecRule):
            req = set(self.extras_required).union(spec.required_extras())
            object.__setattr__(self, "extras_required", tuple(sorted(req)))
            object.__setattr__(self, "structural_extras", tuple(spec.structural_extras()))
        else:
            object.__setattr__(self, "structural_extras", ())

        # Dispatcher *always* consumes a particle axis
        object.__setattr__(self, "particles_input", True)

    # ───────────────────────── arity helpers ──────────────────────────
    @staticmethod
    def _spec_arity(spec) -> tuple[Literal["fixed", "variable"], Optional[int]]:
        if isinstance(spec, PairsCSR):
            return "fixed", 2
        if isinstance(spec, HyperFixed):
            return "fixed", int(spec.hyper.shape[-1])
        if isinstance(spec, HyperCSR):
            return "variable", None
        if isinstance(spec, SpecRule):
            return spec.arity()
        raise TypeError(f"Unknown spec type {type(spec)}")

    # ───────────────────────── contract merge ─────────────────────────
    # Unary inherit: take child's contract and override pdepth/particles_input.
    def _merge_static(self, children):
        ch = children[0]
        new_pdepth = 0 if (self.owners == "global") else 1
        return _ContractMixin.inherit_contract(
            ch,
            pdepth=new_pdepth,
            particles_input=True,
        )

    # ───────────────────────── runtime ────────────────────────────────
    def __call__(self, x, *, params=None, v=None, mask=None, extras=None):
        # ------------------------------------------------------------------
        # 1) Determine which extras are "structural" for THIS spec/rule.
        #    Structural extras are consumed by the dispatcher/spec builder and
        #    must not be forwarded to child local-ops.
        # ------------------------------------------------------------------
        if isinstance(self.spec, SpecRule):
            structural = set(self.spec.structural_extras() or ())
        else:
            structural = set()

        # Extras forwarded to child local-ops (and used for broadcast validation).
        extras_child = extras
        if extras is not None and structural:
            extras_child = {k: v for k, v in extras.items() if k not in structural}

        # ------------------------------------------------------------------
        # 2) Centralized input checks should be done on *forwarded* extras only:
        #    - structural arrays like (P,K) must not be required/broadcast-checked
        #      as they are not part of the child API.
        # ------------------------------------------------------------------
        self._assert_inputs(x, v, mask, extras_child)

        # ------------------------------------------------------------------
        # 3) Build concrete spec. This may require structural extras, so build
        #    from the original `extras`, not the filtered version.
        # ------------------------------------------------------------------
        spec = self.spec.build(x, v=v, mask=mask, extras=extras) if isinstance(self.spec, SpecRule) else self.spec

        # ------------------------------------------------------------------
        # 4) Run dispatch. Child should only see `extras_child`.
        # ------------------------------------------------------------------
        if isinstance(spec, PairsCSR):
            y = self._pairs_csr_vectorized(x, v=v, mask=mask, extras=extras_child, params=params, spec=spec)
        elif isinstance(spec, HyperFixed):
            y = self._hyper_fixed_vectorized(x, v=v, mask=mask, extras=extras_child, params=params, spec=spec)
        elif isinstance(spec, HyperCSR):
            y = self._hyper_csr_vectorized(x, v=v, mask=mask, extras=extras_child, params=params, spec=spec)
        else:
            raise TypeError(f"Unknown spec type {type(spec)}")

        self._assert_outputs(x, y)
        return y

    # ───────────────────────── PAIRS / CSR path ───────────────────────
    def _pair_keep(self, k, row, col, iptr, mb):
        """Edge validity ``(M,)`` for a pairs CSR.

        Entries past ``indptr[-1]`` are padding (e.g. ``pad_neighbor_csr``);
        self-edges are dropped when ``exclude_self``; a pair with a masked member
        (position unknown: the integrator passes the static ``mask_state``) never
        contributes, as for hyperedge slots.  Excluded pairs are removed with
        ``jnp.where``, so NaN positions at masked entries cannot leak.
        """
        keep = k < iptr[-1]
        if self.exclude_self:
            keep = jnp.logical_and(keep, row != col)
        if mb is not None:
            keep = jnp.logical_and(keep, jnp.logical_and(mb[row], mb[col]))
        return keep

    def _pairs_csr_vectorized(self, x, *, v, mask, extras, params, spec: PairsCSR):
        if self._per_slot:
            raise NotImplementedError("per-slot interactors are supported for hyperedge specs, not PairsCSR.")
        *batch, P, d = x.shape
        globals_extras, particle_arrays = self._split_extras(extras or {}, P=P, batch=tuple(batch))
        particle_static, particle_batched = self._split_batched(particle_arrays, P=P)

        def _per_sample(xb, vb, mb, iptr, idx, pb):
            Pb = xb.shape[0]
            particle_arrays = {**particle_static, **pb} if pb else particle_static

            # Flat (row, col) without jnp.repeat
            M = idx.shape[0]
            k = jnp.arange(M, dtype=jnp.int32)
            row = jnp.searchsorted(iptr[1:], k, side="right")  # (M,)
            col = idx  # (M,)

            keep = self._pair_keep(k, row, col, iptr, mb)
            # degrees per row (for degree-normalized 'mean'): contributing neighbours only
            deg = jnp.zeros((Pb,), jnp.int32).at[row].add(keep.astype(jnp.int32))

            # ---------------------------- UNCHUNKED ----------------------------
            if self.chunk_size is None:
                Xi, Xj = xb[row], xb[col]  # (M,d)
                Xk = jnp.stack([Xi, Xj], axis=1)  # (M,2,d)

                Vk = None
                if vb is not None:
                    Vi, Vj = vb[row], vb[col]
                    Vk = jnp.stack([Vi, Vj], axis=1)

                Mk = None
                if mb is not None:
                    mi, mj = mb[row], mb[col]
                    Mk = jnp.stack([mi, mj], axis=1)

                ex_edge = self._gather_particle_extras(particle_arrays, row, col)  # (M, 2, ...)
                ex_local = globals_extras if not ex_edge else ({**globals_extras, **ex_edge})
                yraw = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local)

                km = keep.reshape((-1,) + (1,) * (yraw.ndim - 1))
                yloc = jnp.where(km, yraw, 0)

                if self.owners == "global":
                    if self.reducer == "sum":
                        return yloc.sum(axis=0)
                    elif self.reducer == "mean":
                        return yloc.sum(axis=0) / jnp.maximum(1, keep.sum())
                    elif self.reducer == "max":
                        return jnp.where(km, yraw, -jnp.inf).max(axis=0)
                    else:
                        raise ValueError(f"Unknown reducer {self.reducer!r}")

                # per-particle
                if self.reducer == "mean" and self.normalize_by_degree:
                    y_r = yloc / jnp.maximum(1, deg[row]).reshape((-1,) + (1,) * (yloc.ndim - 1))
                    y_c = yloc / jnp.maximum(1, deg[col]).reshape((-1,) + (1,) * (yloc.ndim - 1))
                else:
                    y_r = yloc
                    y_c = yloc

                outP = jnp.zeros((Pb,) + yloc.shape[1:], dtype=xb.dtype)
                if self.owners == "focal":
                    outP = outP.at[row].add(y_r)
                elif self.owners == "all":
                    outP = outP.at[row].add(y_r).at[col].add(y_c)
                elif self.owners == "custom":
                    w = self.owner_weights
                    if w is None or w.shape[0] != 2:
                        raise ValueError("owners='custom' requires owner_weights of shape (2,)")
                    outP = outP.at[row].add(w[0] * y_r).at[col].add(w[1] * y_c)
                else:
                    raise ValueError(f"Unknown owners {self.owners!r}")
                return outP

            # ----------------------------- CHUNKED -----------------------------
            CS = int(self.chunk_size)
            pad = (CS - (M % CS)) % CS
            Mpad = M + pad

            # Pad arrays and build validity masks
            row_pad = jnp.pad(row, (0, pad))
            col_pad = jnp.pad(col, (0, pad))
            keep_pad = jnp.pad(keep, (0, pad))
            # marks true data vs. padded tail
            data_mask = jnp.concatenate([jnp.ones((M,), dtype=bool), jnp.zeros((pad,), dtype=bool)], axis=0)

            # Accumulators
            rb = _rank_shape(self.interactor.rank, d)
            if self.owners == "global":
                if self.reducer == "max":
                    acc = jnp.full(rb + (int(self.n_features),), -jnp.inf, dtype=xb.dtype)
                else:
                    acc = jnp.zeros(rb + (int(self.n_features),), dtype=xb.dtype)
                cnt = jnp.zeros((), dtype=jnp.int32)
            else:
                acc = jnp.zeros((Pb,) + rb + (int(self.n_features),), dtype=xb.dtype)
                cnt = jnp.zeros((), dtype=jnp.int32)  # only used for global-mean

            steps = Mpad // CS

            def body(i, carry):
                acc, cnt = carry
                start = i * CS
                r_blk = jax.lax.dynamic_slice_in_dim(row_pad, start, CS, axis=0)
                c_blk = jax.lax.dynamic_slice_in_dim(col_pad, start, CS, axis=0)
                k_blk = jax.lax.dynamic_slice_in_dim(keep_pad, start, CS, axis=0)
                d_blk = jax.lax.dynamic_slice_in_dim(data_mask, start, CS, axis=0)
                valid = jnp.logical_and(k_blk, d_blk)

                Xi, Xj = xb[r_blk], xb[c_blk]
                Xk = jnp.stack([Xi, Xj], axis=1)

                Vk = None
                if vb is not None:
                    Vi, Vj = vb[r_blk], vb[c_blk]
                    Vk = jnp.stack([Vi, Vj], axis=1)

                Mk = None
                if mb is not None:
                    mi, mj = mb[r_blk], mb[c_blk]
                    Mk = jnp.stack([mi, mj], axis=1)

                ex_edge_blk = self._gather_particle_extras(particle_arrays, r_blk, c_blk)  # (CS, 2, ...)
                ex_local_blk = globals_extras if not ex_edge_blk else ({**globals_extras, **ex_edge_blk})
                yraw = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local_blk)
                vm = valid.reshape((-1,) + (1,) * (yraw.ndim - 1))
                yloc = jnp.where(vm, yraw, 0)

                if self.owners == "global":
                    if self.reducer in ("sum", "mean"):
                        acc = acc + yloc.sum(axis=0)
                        cnt = cnt + valid.sum().astype(cnt.dtype)
                    elif self.reducer == "max":
                        acc = jnp.maximum(acc, jnp.where(vm, yraw, -jnp.inf).max(axis=0))
                    else:
                        raise ValueError(f"Unknown reducer {self.reducer!r}")
                    return acc, cnt

                # per-particle owners
                if self.reducer == "mean" and self.normalize_by_degree:
                    y_r = yloc / jnp.maximum(1, deg[r_blk]).reshape((-1,) + (1,) * (yloc.ndim - 1))
                    y_c = yloc / jnp.maximum(1, deg[c_blk]).reshape((-1,) + (1,) * (yloc.ndim - 1))
                else:
                    y_r = yloc
                    y_c = yloc

                if self.owners == "focal":
                    acc = acc.at[r_blk].add(y_r)
                elif self.owners == "all":
                    acc = acc.at[r_blk].add(y_r).at[c_blk].add(y_c)
                elif self.owners == "custom":
                    w = self.owner_weights
                    if w is None or w.shape[0] != 2:
                        raise ValueError("owners='custom' requires owner_weights of shape (2,)")
                    acc = acc.at[r_blk].add(w[0] * y_r).at[c_blk].add(w[1] * y_c)
                else:
                    raise ValueError(f"Unknown owners {self.owners!r}")
                return acc, cnt

            accN, cntN = jax.lax.fori_loop(0, steps, body, (acc, cnt))
            if self.owners == "global" and self.reducer == "mean":
                accN = accN / jnp.maximum(1, cntN)
            return accN

        # vmap over batch prefix (or call once)
        if batch:
            # CSR arrays may or may not carry a leading batch axis.
            # Static graphs: indptr (N+1,), indices (nnz,)  → in_axes=None
            # Per-step graphs: indptr (K,N+1), indices (K,nnz) → in_axes=0
            iptr_ax = 0 if spec.indptr.ndim >= 2 else None
            idx_ax = 0 if spec.indices.ndim >= 2 else None
            vm = jax.vmap(
                _per_sample,
                in_axes=(
                    0,
                    0 if v is not None else None,
                    0 if mask is not None else None,
                    iptr_ax,
                    idx_ax,
                    0 if particle_batched else None,
                ),
            )
            return vm(x, v, mask, spec.indptr, spec.indices, particle_batched or None)
        else:
            return _per_sample(x, v, mask, spec.indptr, spec.indices, None)

    ### Hyper branch:

    def _hyper_valid(self, hyper: Array, slot_mask: Array | None, K: int, fi: int) -> Array:
        """Structural slot validity ``(M, K)``: the spec's slot mask AND ``exclude_self``.

        With ``exclude_self``, a non-focal slot holding the focal participant is
        invalid (grid "noflux" boundaries and ragged-row padding encode absent
        neighbours that way).  Invalid slots receive no scattered contribution,
        and a hyperedge whose focal slot is invalid contributes nothing at all
        (an empty or padding row).
        """
        valid = jnp.ones(hyper.shape, dtype=bool) if slot_mask is None else jnp.asarray(slot_mask, dtype=bool)
        if self.exclude_self:
            focal = hyper[:, fi]
            slots = jnp.arange(K, dtype=jnp.int32)[None, :]
            valid = jnp.logical_and(valid, jnp.logical_or(slots == fi, hyper != focal[:, None]))
        return valid

    def _scatter_hyper(self, yloc: Array, hyper: Array, valid: Array, P: int, K: int, fi: int) -> Array:
        """
        Scatter hyperedge-local outputs back onto the particle axis.

        The interactor is called on gathered neighborhoods::

            Xk = x[hyper]                    # (M, K, d)
            yloc = interactor(Xk, ...)       # (M, ...) or (M, K, ...) if per-slot

        Two output conventions are supported:

        1. **Per-hyperedge outputs** (typical for PDE stencils):
           ``yloc.shape == (M, *rank, F)``.  ``owners="focal"`` sends it to
           the focal participant; ``"all"``/``"custom"`` broadcast it to every
           valid slot.
        2. **Per-slot outputs** (``make_interactor(per_slot=True)``):
           ``yloc.shape == (M, K, *rank, F)``.  Slot ``s``'s value goes to the
           participant in slot ``s`` (``"all"``/``"custom"``) or, for
           ``"focal"``, only the focal slot's value is kept.

        Only slots marked valid (see :meth:`_hyper_valid`) receive a
        contribution.  Repeated indices are summed (``.at[idx].add``).  No
        degree normalization is applied here.

        Parameters
        ----------
        yloc
            Local outputs, ``(M, ...)`` or ``(M, K, ...)``.
        hyper
            Integer participant table ``(M, K)``.
        valid
            Boolean slot validity ``(M, K)``.
        P, K, fi
            Number of particles, slots per hyperedge, focal slot index.

        Returns
        -------
        out
            ``(P, *rank, F)``.
        """
        M = yloc.shape[0]
        if self._per_slot:
            contrib = yloc
        else:
            contrib = jnp.broadcast_to(yloc[:, None, ...], (M, K) + yloc.shape[1:])
        out = jnp.zeros((P,) + contrib.shape[2:], dtype=yloc.dtype)

        if self.owners == "focal":
            c = contrib[:, fi]
            vf = valid[:, fi].reshape((-1,) + (1,) * (c.ndim - 1))
            return out.at[hyper[:, fi]].add(jnp.where(vf, c, 0))

        if self.owners in ("all", "custom"):
            vm = valid.reshape(valid.shape + (1,) * (contrib.ndim - 2))
            contrib = jnp.where(vm, contrib, 0)
            if self.owners == "custom":
                w = self.owner_weights
                if w is None or w.shape[0] != K:
                    raise ValueError(f"owners='custom' requires owner_weights of shape (K,) with K={K}")
                contrib = contrib * w.reshape((1, K) + (1,) * (contrib.ndim - 2))
            return out.at[hyper.reshape((-1,))].add(contrib.reshape((M * K,) + contrib.shape[2:]))

        raise ValueError(f"Unknown owners {self.owners!r}")

    def _reduce_global_hyper(self, yloc: Array, valid: Array, fi: int):
        """Masked ``owners="global"`` reduction over hyperedges (and slots, if per-slot).

        Returns ``(value, count)``: the masked sum (``"sum"``/``"mean"``) or
        maximum (``"max"``), and the number of contributing entries.
        """
        w = valid if self._per_slot else valid[:, fi]
        axes = tuple(range(w.ndim))
        wm = w.reshape(w.shape + (1,) * (yloc.ndim - w.ndim))
        cnt = w.sum().astype(jnp.int32)
        if self.reducer in ("sum", "mean"):
            return jnp.where(wm, yloc, 0).sum(axis=axes), cnt
        if self.reducer == "max":
            return jnp.where(wm, yloc, -jnp.inf).max(axis=axes), cnt
        raise ValueError(f"Unknown reducer {self.reducer!r}")

    def _hyper_fixed_vectorized(self, x, *, v, mask, extras, params, spec: HyperFixed):
        """
        Dispatch a fixed-K hyperedge program (HyperFixed) on a state array.

        This is the HyperFixed analogue of the PairsCSR path. It gathers the K
        participants for each hyperedge, calls the local interactor, and scatters
        the result back onto the particle axis according to ``owners``.

        Shape conventions
        -----------------
        - State: ``x`` has shape ``(..., P, d)`` (leading batch axes optional).
        - Spec: ``hyper`` has shape ``(M, K)`` or ``(..., M, K)`` (optionally batched).
        - Gathered neighborhood: ``Xk = x[hyper]`` has shape ``(..., M, K, d)``.
        - Mask: ``mask`` (if provided) has shape ``(..., P)`` and is gathered to
          ``Mk = mask[hyper]`` of shape ``(..., M, K)``.
        - Slot validity: ``spec.slot_mask`` (optional, shape matching ``hyper``)
          AND the ``exclude_self`` rule (see :meth:`_hyper_valid`).  It is ANDed
          into ``Mk`` and decides which slots receive scattered contributions.

        Chunking
        --------
        If ``self.chunk_size`` is not None, hyperedges are processed in blocks of
        size ``chunk_size`` to control peak memory use.

        Notes on masking / no-flux semantics
        -----------------------------------
        This dispatcher *only* combines and forwards masks; it does not encode a
        specific boundary condition for holes/masked pixels. For “no flux through
        masked pixels”, implement it in the local operator by replacing a masked
        neighbor by the focal value (or any other consistent scheme). The mask
        information is provided as ``mask[..., slot]`` to the kernel.

        Parameters
        ----------
        x
            State array of shape ``(..., P, d)``.
        v
            Optional velocity-like array (same shape as x); forwarded to interactor.
        mask
            Optional per-particle boolean mask of shape ``(..., P)``; gathered to slots.
        extras
            Extra arrays; split into global vs particle-aligned by ``_split_extras``.
            Particle-aligned extras are gathered to hyperedge slots and passed to the kernel.
        params
            Parameters forwarded to the interactor.
        spec
            A :class:`~SFI.statefunc.nodes.interactions.specs.HyperFixed` schedule.

        Returns
        -------
        y
            Dispatched result.
            - For owners != "global": shape ``(..., P, <rank-shape>, n_features)``.
            - For owners == "global": shape ``(..., <rank-shape>, n_features)``.
        """
        *batch, P, d = x.shape

        # Separate extras into:
        # - globals_extras: arrays not aligned with particle axis (passed through unchanged)
        # - particle_arrays: arrays aligned with particle axis P (to be gathered on hyperedges)
        globals_extras, particle_arrays = self._split_extras(extras or {}, P=P, batch=tuple(batch))
        particle_static, particle_batched = self._split_batched(particle_arrays, P=P)

        hyper = spec.hyper
        slot_mask = getattr(spec, "slot_mask", None)

        # Optional, spec-aligned metadata.
        #
        # These are already aligned with hyperedges/slots, so unlike particle_extras
        # they do not require indexing by `hyper`.
        #
        # Note: we treat dicts of arrays as pytrees, which lets us vmap over them
        # when they carry a batch prefix.
        edge_extras = getattr(spec, "edge_extras", None) or {}
        slot_extras = getattr(spec, "slot_extras", None) or {}

        K = int(hyper.shape[-1])
        fi = int(self.focal_index)
        if fi < 0 or fi >= K:
            raise ValueError(f"focal_index={fi} out of bounds for K={K}")

        def _per_sample(xb, vb, mb, hyper_b, slot_mask_b, edge_extras_b, slot_extras_b, pb):
            """
            Per-sample worker used both for non-batched and vmapped calls.

            xb: (P, d)
            hyper_b: (M, K)
            """
            Pb = xb.shape[0]
            particle_arrays = {**particle_static, **pb} if pb else particle_static
            M = int(hyper_b.shape[0])

            # Structural validity (spec slot mask AND exclude_self), (M, K).
            valid = self._hyper_valid(hyper_b, slot_mask_b, K, fi)

            # -------------------------- non-chunked path -------------------------- #
            if self.chunk_size is None:
                # Gather neighborhood state: (M, K, d)
                Xk = xb[hyper_b]

                # Gather optional v and mask to neighborhood slots; the kernel's
                # mask combines the particle mask with the structural validity.
                Vk = vb[hyper_b] if vb is not None else None  # (M, K, d) or None
                Mk = valid if mb is None else jnp.logical_and(mb[hyper_b], valid)

                # Gather particle-aligned extras to (M, K, ...).
                ex_slots = self._gather_particle_extras_hyper(particle_arrays, hyper_b)

                # Build extras dict passed to the interactor:
                # - global extras (unchanged)
                # - gathered particle extras (slot-shaped)
                # - optional spec-provided extras
                ex_local = globals_extras if not ex_slots else ({**globals_extras, **ex_slots})
                if edge_extras_b:
                    ex_local = {**ex_local, **edge_extras_b}  # per-hyperedge arrays
                if slot_extras_b:
                    ex_local = {**ex_local, **slot_extras_b}  # per-slot arrays

                # Evaluate kernel on gathered neighborhoods.
                yloc = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local)

                if self.owners == "global":
                    val, cnt = self._reduce_global_hyper(yloc, valid, fi)
                    return val / jnp.maximum(1, cnt) if self.reducer == "mean" else val

                # Scatter to particle axis.
                return self._scatter_hyper(yloc, hyper_b, valid, Pb, K, fi)

            # ---------------------------- chunked path ---------------------------- #
            CS = int(self.chunk_size)

            # Pad M to a multiple of chunk size so that we can use a simple fori_loop.
            # Padded rows are invalid, so they contribute nothing.
            pad = (CS - (M % CS)) % CS
            Mpad = M + pad
            steps = Mpad // CS
            hyper_pad = jnp.pad(hyper_b, ((0, pad), (0, 0)))
            valid_pad = jnp.pad(valid, ((0, pad), (0, 0)))

            # Output shape template: rank_shape + (n_features,)
            rb = _rank_shape(self.interactor.rank, d)

            # Accumulator differs for global vs per-particle output.
            if self.owners == "global":
                # For global reductions, we keep one tensor (no particle axis).
                if self.reducer == "max":
                    acc = jnp.full(rb + (int(self.n_features),), -jnp.inf, dtype=xb.dtype)
                else:
                    acc = jnp.zeros(rb + (int(self.n_features),), dtype=xb.dtype)
            else:
                # For scattered outputs, keep a full particle-axis accumulator.
                acc = jnp.zeros((Pb,) + rb + (int(self.n_features),), dtype=xb.dtype)
            cnt = jnp.zeros((), dtype=jnp.int32)  # used only for global "mean"

            def body(i, carry):
                """Process one chunk of hyperedges."""
                acc, cnt = carry
                start = i * CS

                hyper_blk = jax.lax.dynamic_slice_in_dim(hyper_pad, start, CS, axis=0)  # (CS, K)
                valid_blk = jax.lax.dynamic_slice_in_dim(valid_pad, start, CS, axis=0)  # (CS, K)

                # Gather state / v / mask on this chunk.
                Xk = xb[hyper_blk]  # (CS, K, d)
                Vk = vb[hyper_blk] if vb is not None else None  # (CS, K, d) or None
                Mk = valid_blk if mb is None else jnp.logical_and(mb[hyper_blk], valid_blk)

                # Gather particle extras for this chunk.
                ex_slots_blk = self._gather_particle_extras_hyper(particle_arrays, hyper_blk)
                ex_local_blk = globals_extras if not ex_slots_blk else ({**globals_extras, **ex_slots_blk})

                # Slice spec-provided extras if present (padded like the hyper table).
                # Edge extras are (M, ...); slot extras are (M, K, ...) typically.
                if edge_extras_b:
                    ex_edge_blk = self._slice_extras_1d(self._pad_extras_1d(edge_extras_b, pad), start, CS)
                    ex_local_blk = {**ex_local_blk, **ex_edge_blk}
                if slot_extras_b:
                    ex_slot_blk = self._slice_extras_1d(self._pad_extras_1d(slot_extras_b, pad), start, CS)
                    ex_local_blk = {**ex_local_blk, **ex_slot_blk}

                # Kernel evaluation on this chunk.
                yloc = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local_blk)

                if self.owners == "global":
                    val, c = self._reduce_global_hyper(yloc, valid_blk, fi)
                    acc = jnp.maximum(acc, val) if self.reducer == "max" else acc + val
                    return acc, cnt + c

                # Scatter chunk contributions onto particles and accumulate.
                acc = acc + self._scatter_hyper(yloc, hyper_blk, valid_blk, Pb, K, fi)
                return acc, cnt

            accN, cntN = jax.lax.fori_loop(0, steps, body, (acc, cnt))

            # Finalize mean for global reduction (avoid division by 0 when M=0).
            if self.owners == "global" and self.reducer == "mean":
                accN = accN / jnp.maximum(1, cntN)
            return accN

        # ----------------------------- batch handling ----------------------------- #
        # If x has leading batch axes, we vmap over them. The schedule (hyper) can be:
        # - shared across the batch: hyper has shape (M, K) -> in_axes=None
        # - provided per sample:     hyper has shape (..., M, K) -> in_axes=0
        if batch:
            hyper_in = 0 if hyper.ndim == (len(batch) + 2) else None
            sm_in = 0 if (slot_mask is not None and slot_mask.ndim == (len(batch) + 2)) else None
            edge_in = hyper_in if edge_extras else None
            slot_in = hyper_in if slot_extras else None
            vm = jax.vmap(
                _per_sample,
                in_axes=(
                    0,
                    0 if v is not None else None,
                    0 if mask is not None else None,
                    hyper_in,
                    sm_in,
                    edge_in,
                    slot_in,
                    0 if particle_batched else None,
                ),
            )
            return vm(x, v, mask, hyper, slot_mask, edge_extras or None, slot_extras or None, particle_batched or None)

        return _per_sample(x, v, mask, hyper, slot_mask, edge_extras or None, slot_extras or None, None)

    def _hyper_csr_to_fixed(self, spec: HyperCSR, *, batched: bool) -> HyperFixed:
        """Pad a ragged :class:`HyperCSR` schedule into a dense :class:`HyperFixed` (JIT-safe).

        Each hyperedge is padded to ``K = self._K`` slots (the interactor's
        ``K`` or ``Kmax``) by repeating its first participant; a ``slot_mask``
        marks the real slots.  Entries of ``he_indices`` past ``he_indptr[-1]``
        are treated as padding.  A hyperedge with more than ``K`` participants
        raises instead of being truncated (at trace time when the schedule is
        concrete, at run time otherwise).

        With ``batched=True`` every schedule array carries a leading sample
        axis (per-sample topologies).
        """
        if self._K is None:
            raise NotImplementedError(
                "HyperCSR dispatch requires a static K: use a fixed-K interactor (K=...) or a variable one (Kmax=...)."
            )
        K = int(self._K)
        msg = f"HyperCSR: a hyperedge has more than K={K} participants; raise the interactor's K / Kmax."

        def _one(indptr, indices, slot_ids_b, slot_extras_b):
            total = int(indices.shape[0])
            M = int(indptr.shape[0]) - 1
            if isinstance(indptr, Tracer):
                indptr = eqx.error_if(indptr, jnp.any(indptr[1:] - indptr[:-1] > K), msg)
            elif M > 0 and int(np.max(np.diff(np.asarray(indptr)))) > K:
                raise ValueError(msg)

            if total == 0 or M == 0:
                hyper = jnp.zeros((M, K), dtype=indices.dtype)
                slot_mask = jnp.zeros((M, K), dtype=bool)
                r = c = None
            else:
                # Hyperedge and in-edge position of every flattened slot; padding
                # entries and overflow positions are sent out of range (dropped).
                pos = jnp.arange(total, dtype=jnp.int32)
                edge_id = jnp.searchsorted(indptr[1:], pos, side="right")
                in_edge = pos - indptr[jnp.minimum(edge_id, M - 1)]
                ok = jnp.logical_and(pos < indptr[-1], in_edge < K)
                r = jnp.where(ok, edge_id, M)
                c = jnp.where(ok, in_edge, 0)

                # Pad each row with its first participant (a valid particle index).
                first = indices[jnp.clip(indptr[:-1], 0, total - 1)]
                hyper = jnp.broadcast_to(first[:, None], (M, K))
                hyper = hyper.at[r, c].set(indices, mode="drop")
                slot_mask = jnp.zeros((M, K), dtype=bool).at[r, c].set(True, mode="drop")

            # Dense per-slot extras: slot ids (-1 on padded slots) and explicit slot extras.
            dense: Dict[str, Array] = {}
            if slot_ids_b is not None:
                sid = jnp.full((M, K), -1, dtype=slot_ids_b.dtype)
                dense["slot_id"] = sid if r is None else sid.at[r, c].set(slot_ids_b, mode="drop")
            for kname, arr in (slot_extras_b or {}).items():
                z = jnp.zeros((M, K) + arr.shape[1:], dtype=arr.dtype)
                dense[kname] = z if r is None else z.at[r, c].set(arr, mode="drop")
            return hyper, slot_mask, dense

        slot_ids = getattr(spec, "slot_ids", None)
        slot_extras = getattr(spec, "slot_extras", None) or {}
        args = (spec.he_indptr, spec.he_indices, slot_ids, slot_extras or None)
        if batched:
            in_axes = (0, 0, 0 if slot_ids is not None else None, 0 if slot_extras else None)
            hyper, slot_mask, dense = jax.vmap(_one, in_axes=in_axes)(*args)
        else:
            hyper, slot_mask, dense = _one(*args)

        return HyperFixed(
            hyper=hyper,
            slot_mask=slot_mask,
            edge_extras=getattr(spec, "edge_extras", None) or None,
            slot_extras=dense or None,
        )

    def _hyper_csr_vectorized(self, x, *, v, mask, extras, params, spec: HyperCSR):
        """Dispatch a CSR hyperedge schedule.

        :class:`~SFI.statefunc.nodes.interactions.specs.HyperCSR` stores a ragged list
        of hyperedges:

        - ``he_indptr``: shape ``(..., M+1)``
        - ``he_indices``: shape ``(..., total_slots)``

        where hyperedge ``e`` uses participants ``he_indices[indptr[e]:indptr[e+1]]``.
        The schedule is padded to ``K`` slots (see :meth:`_hyper_csr_to_fixed`)
        and dispatched through the HyperFixed path.  A schedule with a leading
        axis is per-sample (one topology per batch row); one without is shared.
        """
        *batch, _P, _d = x.shape
        batched = spec.he_indptr.ndim >= 2
        if batched and not batch:
            raise ValueError("HyperCSR: a batched schedule (leading axis) needs a batched state x.")
        hf = self._hyper_csr_to_fixed(spec, batched=batched)
        return self._hyper_fixed_vectorized(x, v=v, mask=mask, extras=extras, params=params, spec=hf)

    @staticmethod
    def _pad_extras_1d(extras: Mapping[str, Array], pad: int) -> Dict[str, Array]:
        """Zero-pad a dict of arrays along axis 0 (matches the padded hyper table)."""
        if not pad:
            return dict(extras)
        return {k: jnp.pad(arr, ((0, pad),) + ((0, 0),) * (arr.ndim - 1)) for k, arr in extras.items()}

    def _split_extras(self, extras: dict | None, *, P: int, batch: tuple = ()):
        """
        Return (globals_extras, particle_arrays) where:
          - globals_extras : dict of non-structural, non-particle keys (pass-through)
          - particle_arrays: dict[key -> (P, ...)] gathered per edge by the dispatcher,
            or ``(*batch, P, ...)`` for a per-sample value (e.g. the ``"pbc_ref"``
            reference positions, which follow the state's batch axes)
        """
        if not extras:
            return {}, {}
        particle_keys = set(getattr(self.interactor, "particle_extras", ()) or ())
        structural = set(getattr(self, "structural_extras", ()) or ())
        globals_extras, particle_arrays = {}, {}
        nb = len(batch)
        for k, v in extras.items():
            if k in structural:
                continue  # dispatcher-owned; never forward
            if k in particle_keys:
                shape = tuple(getattr(v, "shape", ()) or ())
                if nb and shape[: nb + 1] == tuple(batch) + (P,):
                    v = _BatchedParticleExtra(v)
                elif not (shape and shape[0] == P):
                    raise ValueError(f"Particle extra '{k}' must have shape (P, ...); got {getattr(v, 'shape', None)}")
                particle_arrays[k] = v
            else:
                globals_extras[k] = v
        return globals_extras, particle_arrays

    @staticmethod
    def _split_batched(particle_arrays: dict, *, P: int):
        """Split particle extras into shared ``(P, ...)`` and per-sample ``(*batch, P, ...)`` dicts."""
        static = {k: v for k, v in particle_arrays.items() if not isinstance(v, _BatchedParticleExtra)}
        batched = {k: v.value for k, v in particle_arrays.items() if isinstance(v, _BatchedParticleExtra)}
        return static, batched

    def _gather_particle_extras(self, particle_arrays: dict, row: Array, col: Array) -> dict:
        """For pairs (K=2), gather per-edge slices into shape (E, 2, ...)."""
        if not particle_arrays:
            return {}
        gathered = {}
        for k, arr in particle_arrays.items():
            ei = arr[row]  # (E, ...)
            ej = arr[col]  # (E, ...)
            gathered[k] = jnp.stack([ei, ej], axis=1)  # (E, 2, ...)
        return gathered

    def _gather_particle_extras_hyper(self, particle_arrays: dict, hyper: Array) -> dict:
        """For hyperedges (fixed K), gather per-slot slices into shape (M, K, ...)."""
        if not particle_arrays:
            return {}
        return {k: arr[hyper] for k, arr in particle_arrays.items()}

    @staticmethod
    def _slice_extras_1d(extras: Mapping[str, Array], start: int, length: int) -> Dict[str, Array]:
        """Dynamic-slice a dict of arrays along axis=0."""
        # This helper is used in chunked execution to take a consistent slice
        # of spec-aligned metadata (edge_extras or slot_extras) for the current
        # chunk. It assumes those arrays are aligned with the hyperedge axis
        # (axis 0 = hyperedges), which is the convention used by HyperFixed.
        if not extras:
            return {}
        return {k: jax.lax.dynamic_slice_in_dim(arr, start, length, axis=0) for k, arr in extras.items()}

    # ──────────── same-particle diagonal Jacobian (efficient) ─────────────────
    def _same_particle_jacobian(self, x, *, var="x", v=None, params=None, mask=None, extras=None):
        """Per-edge ``jacfwd`` + scatter: O(M·d²) instead of generic O(P·M·d).

        Returns ``(P, d_jac, rank..., F)`` in contract layout, matching the
        convention of :meth:`BaseNode._same_particle_jacobian`.
        """
        if self.owners == "global":
            raise ValueError("same-particle Jacobian is not defined for owners='global'")

        # ── resolve spec (same logic as __call__) ──
        if isinstance(self.spec, SpecRule):
            structural = set(self.spec.structural_extras() or ())
        else:
            structural = set()
        extras_child = extras
        if extras is not None and structural:
            extras_child = {k: val for k, val in extras.items() if k not in structural}
        spec = self.spec.build(x, v=v, mask=mask, extras=extras) if isinstance(self.spec, SpecRule) else self.spec

        # ── dispatch by spec type ──
        if isinstance(spec, PairsCSR):
            return self._pairs_csr_spj(
                x,
                v=v,
                mask=mask,
                extras=extras_child,
                params=params,
                spec=spec,
                var=var,
            )
        if isinstance(spec, HyperCSR):
            spec = self._hyper_csr_to_fixed(spec, batched=False)
        if isinstance(spec, HyperFixed):
            return self._hyper_fixed_spj(
                x,
                v=v,
                mask=mask,
                extras=extras_child,
                params=params,
                spec=spec,
                var=var,
            )
        # Unknown spec: fall back to generic AD path
        return super()._same_particle_jacobian(x, var=var, v=v, params=params, mask=mask, extras=extras)

    # ·············· PairsCSR per-edge Jacobian ··············
    def _pairs_csr_spj(self, x, *, v, mask, extras, params, spec, var):
        if self._per_slot:
            raise NotImplementedError("per-slot interactors are supported for hyperedge specs, not PairsCSR.")
        P, d = x.shape
        globals_ex, particle_arrays = self._split_extras(extras or {}, P=P)

        indptr, indices = spec.indptr, spec.indices
        M = indices.shape[0]
        k = jnp.arange(M, dtype=jnp.int32)
        row = jnp.searchsorted(indptr[1:], k, side="right")
        col = indices

        # Same edge validity as the forward path (padding, self-edges, masked particles).
        keep = self._pair_keep(k, row, col, indptr, mask)

        # Gather stencils (same as forward path)
        Xk = jnp.stack([x[row], x[col]], axis=1)  # (M, 2, d)
        Vk = jnp.stack([v[row], v[col]], axis=1) if v is not None else None
        Mk = jnp.stack([mask[row], mask[col]], axis=1) if mask is not None else None
        ex_edge = self._gather_particle_extras(particle_arrays, row, col)

        # — per-edge jacfwd —
        J_all = self._vmap_jacfwd_edges(
            Xk,
            Vk,
            Mk,
            ex_edge,
            globals_ex,
            params,
            var,
        )  # (M, rank..., F, K=2, d)

        # Apply keep mask
        km = keep.reshape((-1,) + (1,) * (J_all.ndim - 1))
        J_all = jnp.where(km, J_all, 0)

        # Degree normalization (if applicable): contributing neighbours, as in the forward path
        deg = None
        if self.reducer == "mean" and self.normalize_by_degree:
            deg = jnp.zeros((P,), jnp.int32).at[row].add(keep.astype(jnp.int32))

        # Scatter based on owners
        fi = self.focal_index
        out_shape = (P,) + J_all.shape[1:-2] + (J_all.shape[-1],)
        out = jnp.zeros(out_shape, dtype=x.dtype)

        if self.owners == "focal":
            J_focal = J_all[..., fi, :]  # (M, rank..., F, d)
            if deg is not None:
                J_focal = J_focal / jnp.maximum(1, deg[row]).reshape((-1,) + (1,) * (J_focal.ndim - 1))
            out = out.at[row].add(J_focal)
        elif self.owners in ("all", "custom"):
            w = self.owner_weights
            for s in range(2):
                J_s = J_all[..., s, :]
                idx = row if s == fi else col
                if deg is not None:
                    J_s = J_s / jnp.maximum(1, deg[idx]).reshape((-1,) + (1,) * (J_s.ndim - 1))
                if w is not None:
                    J_s = w[s] * J_s
                out = out.at[idx].add(J_s)
        else:
            raise ValueError(f"Unsupported owners='{self.owners}' for same-particle Jacobian")

        # (P, rank..., F, d) → (P, d, rank..., F)
        return jnp.moveaxis(out, -1, 1)

    # ·············· HyperFixed per-edge Jacobian ··············
    def _hyper_fixed_spj(self, x, *, v, mask, extras, params, spec, var):
        """Same-particle Jacobian of the HyperFixed forward path, per hyperedge.

        Mirrors the forward scatter exactly: only valid slots receive a block,
        and a participant that fills several slots of one hyperedge (row padding,
        no-flux boundary slots) gets the sum of the derivatives through all of
        them, so the result equals the full-AD diagonal blocks.
        """
        P, d = x.shape
        globals_ex, particle_arrays = self._split_extras(extras or {}, P=P)

        hyper = spec.hyper  # (M, K)
        edge_extras_spec = getattr(spec, "edge_extras", None) or {}
        slot_extras_spec = getattr(spec, "slot_extras", None) or {}

        K = int(hyper.shape[-1])
        M = int(hyper.shape[0])
        fi = int(self.focal_index)
        valid = self._hyper_valid(hyper, getattr(spec, "slot_mask", None), K, fi)

        # Gather stencils; the kernel mask matches the forward path.
        Xk = x[hyper]  # (M, K, d)
        Vk = v[hyper] if v is not None else None
        Mk = valid if mask is None else jnp.logical_and(mask[hyper], valid)

        # Collect all per-edge varying extras
        edge_varying = {}
        ex_slots = self._gather_particle_extras_hyper(particle_arrays, hyper)
        if ex_slots:
            edge_varying.update(ex_slots)
        if edge_extras_spec:
            edge_varying.update(edge_extras_spec)
        if slot_extras_spec:
            edge_varying.update(slot_extras_spec)

        # — per-edge jacfwd —  (M, *rank, F, K, d), or (M, K, *rank, F, K, d) per-slot
        J_all = self._vmap_jacfwd_edges(
            Xk,
            Vk,
            Mk,
            edge_varying,
            globals_ex,
            params,
            var,
        )

        # Sum the derivative over every slot holding the same participant.
        same = (hyper[:, :, None] == hyper[:, None, :]).astype(J_all.dtype)  # (M, K_out, K_in)
        if self._per_slot:
            J_eff = jnp.einsum("mst,ms...td->ms...d", same, J_all)  # (M, K, *rank, F, d)
        else:
            J_eff = jnp.einsum("mst,m...td->ms...d", same, J_all)  # (M, K, *rank, F, d)
        J_eff = jnp.where(valid.reshape(valid.shape + (1,) * (J_eff.ndim - 2)), J_eff, 0)

        # Scatter based on owners
        out = jnp.zeros((P,) + J_eff.shape[2:], dtype=x.dtype)
        if self.owners == "focal":
            out = out.at[hyper[:, fi]].add(J_eff[:, fi])
        elif self.owners in ("all", "custom"):
            w = self.owner_weights
            if w is not None:
                J_eff = J_eff * w.reshape((1, K) + (1,) * (J_eff.ndim - 2))
            out = out.at[hyper.reshape((-1,))].add(J_eff.reshape((M * K,) + J_eff.shape[2:]))
        else:
            raise ValueError(f"Unsupported owners='{self.owners}' for same-particle Jacobian")

        # (P, rank..., F, d) → (P, d, rank..., F)
        return jnp.moveaxis(out, -1, 1)

    # ·············· shared: vmap(jacfwd) over edges ··············
    def _vmap_jacfwd_edges(
        self,
        Xk,
        Vk,
        Mk,
        edge_varying,
        globals_ex,
        params,
        var,
    ):
        """Compute per-edge Jacobian w.r.t. position or velocity.

        Parameters
        ----------
        Xk : (M, K, d) positions
        Vk : (M, K, d) or None
        Mk : (M, K) or None
        edge_varying : dict of (M, ...) arrays  (per-edge extras)
        globals_ex   : dict of non-batched extras (closed over)
        params       : parameter dict
        var          : 'x' or 'v'

        Returns
        -------
        J : (M, rank..., F, K, d)
        """
        interactor = self.interactor

        def _jac_one(xk_one, vk_one, mk_one, ev_one):
            # Merge global (closed over) + per-edge extras
            ex = dict(globals_ex) if globals_ex else {}
            if ev_one:
                ex.update(ev_one)
            extras_arg = ex if ex else None

            if var == "x":
                return jax.jacfwd(lambda xk: interactor(xk, v=vk_one, mask=mk_one, params=params, extras=extras_arg))(
                    xk_one
                )
            else:
                return jax.jacfwd(lambda vk: interactor(xk_one, v=vk, mask=mk_one, params=params, extras=extras_arg))(
                    vk_one
                )

        in_vk = 0 if Vk is not None else None
        in_mk = 0 if Mk is not None else None
        in_ev = jax.tree_util.tree_map(lambda _: 0, edge_varying) if edge_varying else {}

        return jax.vmap(_jac_one, in_axes=(0, in_vk, in_mk, in_ev))(
            Xk,
            Vk,
            Mk,
            edge_varying if edge_varying else {},
        )

    # ─────────────────────── memory hint (custom) ─────────────────────────────
    def memory_hint(
        self,
        *,
        dtype=None,  # default float32 when None
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> MemHint:
        """
        Conservative per-sample footprint for interaction streaming.

        Counts:
          - gathered edge chunk (Xk and optionally Vk/Mask) ~ (chunk_E, K, d)
          - per-edge temporary outputs: (chunk_E, ``*rank``, n_features)
          - accumulator:
               owners='global' → (``*rank``, n_features)
               else            → (P, ``*rank``, n_features)
          - small integer/boolean work arrays for row/col/valid/deg
        """
        P = resolve_P(particle_size, sample) or 1
        d = int(getattr(self, "dim", 0) or 0)
        mfeat = int(getattr(self, "n_features", 0) or 0)
        r = int(getattr(self.interactor, "rank", 0) or 0)
        K = self._effective_K() or (sample.K if sample and sample.K else 2)

        isz = itemsize_of(dtype)
        isz_i32 = jnp.dtype(jnp.int32).itemsize
        isz_bool = jnp.dtype(jnp.bool_).itemsize

        rb = (d or 1) ** r
        out_per_edge_bytes = rb * mfeat * isz * (K if self._per_slot else 1)

        # persistent (weights etc.) only from child; transient will be re-counted here
        child_hint = self.interactor.memory_hint(dtype=dtype, particle_size=None, sample=None, mode=mode)
        persistent = int(child_hint.persistent_bytes)

        M_est = self._estimate_edges(P)
        chunk_E = int(M_est if self.chunk_size is None else int(self.chunk_size))

        gather_bytes_per_edge = K * d * isz
        if bool(getattr(self.interactor, "needs_v", False)) or (sample and sample.has_v):
            gather_bytes_per_edge += K * d * isz
        # cheap mask slab
        if sample and sample.has_mask:
            gather_bytes_per_edge += K * isz_bool
        # row/col/valid bookkeeping
        gather_bytes_per_edge += 2 * isz_i32 + 2 * isz_bool

        gather_bytes = chunk_E * gather_bytes_per_edge
        working_outputs = chunk_E * out_per_edge_bytes

        # Final accumulator that persists while chunks stream
        if self.owners == "global":
            acc_bytes = rb * mfeat * isz
        else:
            acc_bytes = P * rb * mfeat * isz
            acc_bytes += P * isz_i32  # degrees for normalized means

        per_sample = int(gather_bytes + working_outputs + acc_bytes)
        return MemHint(per_sample_bytes=per_sample, persistent_bytes=persistent)

    # ───────────────────── self-tuning chunk size from budget ─────────────────
    def suggest_chunk_size(
        self,
        *,
        max_per_sample_bytes: int,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
        clamp_to_total: bool = True,
    ) -> int:
        """
        Choose the largest chunk size (edges per chunk) that keeps the
        *per-sample* memory under `max_per_sample_bytes`.

        Returns a positive integer. If `clamp_to_total`, the result is capped
        by the estimated total number of edges for one sample.
        """
        P = resolve_P(particle_size, sample) or 1
        d = int(getattr(self, "dim", 0) or 0)
        mfeat = int(getattr(self, "n_features", 0) or 0)
        r = int(getattr(self.interactor, "rank", 0) or 0)
        K = self._effective_K() or (sample.K if sample and sample.K else 2)

        isz = itemsize_of(dtype)
        isz_i32 = jnp.dtype(jnp.int32).itemsize
        isz_bool = jnp.dtype(jnp.bool_).itemsize

        rb = (d or 1) ** r
        out_per_edge_bytes = rb * mfeat * isz * (K if self._per_slot else 1)

        gather_bytes_per_edge = K * d * isz
        if bool(getattr(self.interactor, "needs_v", False)) or (sample and sample.has_v):
            gather_bytes_per_edge += K * d * isz
        if sample and sample.has_mask:
            gather_bytes_per_edge += K * isz_bool
        gather_bytes_per_edge += 2 * isz_i32 + 2 * isz_bool

        # accumulator is independent of chunk size
        if self.owners == "global":
            acc_bytes = rb * mfeat * isz
        else:
            acc_bytes = P * rb * mfeat * isz
            acc_bytes += P * isz_i32

        # budget inequality:
        #   chunk_E * (gather_bytes_per_edge + out_per_edge_bytes) + acc_bytes <= max_bytes
        per_edge = gather_bytes_per_edge + out_per_edge_bytes
        budget = int(max_per_sample_bytes) - int(acc_bytes)
        if budget <= 0:
            return 1
        chunk_max = max(1, budget // int(per_edge))

        if clamp_to_total:
            M_est = self._estimate_edges(P)
            chunk_max = min(chunk_max, int(max(1, M_est)))

        return int(chunk_max)

    # ───────────────────────── helpers used by memory_hint ─────────────────────
    def _effective_K(self) -> int | None:
        """Return the interaction arity (fixed K or Kmax) when known; otherwise None."""
        K = getattr(self, "_K", None)
        if K is not None:
            return int(K)
        try:
            mode, K = self._spec_arity(self.spec)  # existing helper
            if mode == "fixed" and K is not None:
                return int(K)
        except Exception:
            pass
        return None

    def _estimate_edges(self, P: int) -> int:
        """
        Best-effort conservative estimate of edge count for ONE sample.
        Uses concrete spec shapes when available, else worst-case.
        """
        spec = self.spec
        # exact rows/nnz when available
        if isinstance(spec, PairsCSR):
            try:
                return int(spec.indices.shape[-1])
            except Exception:
                return int(P * (P - 1)) if getattr(self, "exclude_self", False) else int(P * P)

        if isinstance(spec, HyperFixed):
            try:
                return int(spec.hyper.shape[-2])  # M rows
            except Exception:
                return int(P * max(0, P - 1))

        if isinstance(spec, HyperCSR):
            try:
                return int(spec.he_indptr.shape[-1]) - 1
            except Exception:
                return int(P * max(0, P - 1))

        if isinstance(spec, SpecRule):
            try:
                mode, K = spec.arity()
                if mode == "fixed" and K == 2:
                    return int(P * (P - 1)) if getattr(self, "exclude_self", False) else int(P * P)
            except Exception:
                pass
            return int(P * max(0, P - 1))

        # unknown spec → pessimistic bound
        return int(P * max(0, P - 1))
