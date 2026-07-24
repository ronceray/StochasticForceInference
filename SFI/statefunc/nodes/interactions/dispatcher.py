from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...memhint import MemHint, SampleMeta, itemsize_of, resolve_P
from ..base import BaseNode, BaseOpNode
from ..contract import Rank, _ContractMixin
from ..leaf import InteractionLeaf
from .specs import HyperCSR, HyperFixed, PairsCSR, SpecRule


# Small helper for rank axis sizes (vector/matrix → dim per axis).
def _rank_shape(rank: Rank, d: int) -> Tuple[int, ...]:
    r = int(rank)
    return () if r == 0 else (d,) * r


# Walk the interactor subtree and infer K mode; “fixed wins”.
def _infer_interactor_K(interactor: BaseNode):
    fixed_Ks, var_seen = set(), False

    def _walk(n):
        nonlocal var_seen
        if isinstance(n, InteractionLeaf):
            if n.mode == "fixed":
                fixed_Ks.add(int(n.K))
            else:
                var_seen = True
        for ch in getattr(n, "children", ()):
            _walk(ch)

    _walk(interactor)

    if fixed_Ks:
        if len(fixed_Ks) != 1:
            raise ValueError(f"Interactor mixes fixed arities {sorted(fixed_Ks)}; split across dispatchers.")
        return "fixed", int(next(iter(fixed_Ks)))
    if var_seen:
        return "variable", None

    # No InteractionLeaves found → likely a SimpleLeaf graph. Not allowed as interactor.
    raise ValueError("Interactor must be built from InteractionLeaves (particles_input=True, pdepth=0).")


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

        # Interactor K-mode
        kmode, K = _infer_interactor_K(interactor)
        spec_mode, spec_K = self._spec_arity(spec)

        # “Fixed wins” across interactor+spec
        if kmode == "fixed" or spec_mode == "fixed":
            K_eff = spec_K if spec_mode == "fixed" else K
            if kmode == "fixed" and K is not None and K_eff is not None and K != K_eff:
                raise ValueError(f"Fixed-K mismatch between interactor (K={K}) and spec (K={K_eff}).")
            kmode, K = "fixed", (K or K_eff)
            if isinstance(spec, HyperCSR):
                object.__setattr__(self, "_enforce_fixedK_on_hypercsr", True)

        object.__setattr__(self, "_kmode", kmode)
        object.__setattr__(self, "_K", K)

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
    def _pairs_csr_vectorized(self, x, *, v, mask, extras, params, spec: PairsCSR):
        *batch, P, d = x.shape
        globals_extras, particle_arrays = self._split_extras(extras or {}, P=P)

        def _per_sample(xb, vb, mb, iptr, idx):
            Pb = xb.shape[0]

            # degrees per row (for degree-normalized 'mean')
            deg = (iptr[1:] - iptr[:-1]).astype(jnp.int32)  # (P,)

            # Flat (row, col) without jnp.repeat
            M = idx.shape[0]
            k = jnp.arange(M, dtype=jnp.int32)
            row = jnp.searchsorted(iptr[1:], k, side="right")  # (M,)
            col = idx  # (M,)

            # Keep mask for self-edges; used only arithmetically
            keep = jnp.logical_not(row == col) if self.exclude_self else jnp.ones((M,), dtype=bool)

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
                yloc = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local)

                km = keep.reshape((-1,) + (1,) * (yloc.ndim - 1))
                yloc = yloc * km

                if self.owners == "global":
                    if self.reducer == "sum":
                        return yloc.sum(axis=0)
                    elif self.reducer == "mean":
                        return yloc.sum(axis=0) / jnp.maximum(1, keep.sum())
                    elif self.reducer == "max":
                        return yloc.max(axis=0)
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
                yloc = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local_blk)
                vm = valid.reshape((-1,) + (1,) * (yloc.ndim - 1))
                yloc = yloc * vm

                if self.owners == "global":
                    if self.reducer in ("sum", "mean"):
                        acc = acc + yloc.sum(axis=0)
                        cnt = cnt + valid.sum().astype(cnt.dtype)
                    elif self.reducer == "max":
                        acc = jnp.maximum(acc, yloc.max(axis=0))
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
                ),
            )
            return vm(x, v, mask, spec.indptr, spec.indices)
        else:
            return _per_sample(x, v, mask, spec.indptr, spec.indices)

    ### Hyper branch:

    def _scatter_hyper(self, yloc: Array, hyper: Array, P: int, K: int, fi: int) -> Array:
        """
        Scatter hyperedge-local outputs back onto the particle axis.

        This method turns *local* contributions produced by the interactor on each
        hyperedge into a *global* array indexed by particle id.

        The interactor is called on gathered neighborhoods::

            Xk = x[hyper]                    # (M, K, d)  (plus optional batch axes)
            yloc = interactor(Xk, ...)       # (M, ...)

        Two output conventions are supported:

        1. **Per-hyperedge outputs** (typical for PDE stencils):
           ``yloc.shape == (M, ...)``

           The same hyperedge contribution is scattered to one or more owners
           depending on ``self.owners``.

        2. **Per-slot outputs** (slot-resolved):
           ``yloc.shape == (M, K, ...)``

           Each slot has its own contribution; this is useful when the kernel
           produces distinct terms for focal vs neighbors and wants to scatter
           them differently.

        Parameters
        ----------
        yloc
            Local outputs for each hyperedge. Shape is either ``(M, ...)`` or
            ``(M, K, ...)``.
        hyper
            Integer participant list of shape ``(M, K)``. ``hyper[m, s]`` is the
            particle id at slot ``s`` of hyperedge ``m``.
        P
            Number of particles (grid sites) in the global state.
        K
            Number of participants per hyperedge (stencil size).
        fi
            Slot index of the "focal" participant (typically 0 in PDE stencils).

        Returns
        -------
        out
            Scattered global array.
            - If ``owners == "global"``: shape matches ``yloc`` reduced over hyperedges.
            - Else: shape is ``(P, ...)`` where ``...`` matches the non-hyperedge axes
              of ``yloc``.

        Notes
        -----
        - Scattering uses ``.at[idx].add(...)``, so repeated indices are summed
          (as expected for additive contributions).
        - No degree normalization is applied here; if you want mean-per-particle
          behavior, implement it at the reduction stage (or add a separate
          normalization pass) to keep this primitive simple and predictable.
        """
        # Detect whether the kernel returned per-slot contributions.
        # Per-hyperedge yloc has shape (M, *rank_axes, n_features) → ndim = 2 + rank
        # Per-slot yloc has shape       (M, K, *rank_axes, n_features) → ndim = 3 + rank
        # The shape-only heuristic ``yloc.shape[1] == K`` is ambiguous when
        # ``dim == K`` (e.g. dim=5 state with a 5-point 2D Laplacian stencil); use
        # rank+ndim as the unambiguous discriminator.
        r_axes = int(self.interactor.rank)
        expected_per_hyper_ndim = 2 + r_axes
        per_slot = yloc.ndim > expected_per_hyper_ndim and yloc.shape[1] == K

        # "global" ownership means: do *not* scatter onto particle ids at all.
        # Instead reduce across hyperedges and return a single tensor.
        if self.owners == "global":
            if self.reducer == "sum":
                return yloc.sum(axis=0)
            if self.reducer == "mean":
                return yloc.mean(axis=0)
            if self.reducer == "max":
                return yloc.max(axis=0)
            raise ValueError(f"Unknown reducer {self.reducer!r}")

        # Allocate the global destination array.
        # For per-slot yloc: yloc is (M, K, ...), so drop axes (M, K) -> keep "..."
        # For per-hyperedge yloc: yloc is (M, ...), so drop axis (M) -> keep "..."
        out_shape = (P,) + (yloc.shape[2:] if per_slot else yloc.shape[1:])
        out = jnp.zeros(out_shape, dtype=yloc.dtype)

        # Owners="focal": each hyperedge contributes only to its focal participant.
        if self.owners == "focal":
            idx = hyper[:, fi]  # (M,)
            contrib = yloc[:, fi] if per_slot else yloc  # (M, ...) in both cases
            return out.at[idx].add(contrib)

        # Owners="all": scatter the same contribution to all K participants (or per-slot values if provided).
        # Owners="custom": same as "all" but multiply each slot by a slot weight vector.
        if self.owners in ("all", "custom"):
            if self.owners == "custom":
                w = self.owner_weights
                if w is None or w.shape[0] != K:
                    raise ValueError(f"owners='custom' requires owner_weights of shape (K,) with K={K}")
            else:
                w = None

            # Flatten all target indices for scattering: (M*K,)
            idx = hyper.reshape((-1,))

            # Build a per-slot contribution array of shape (M, K, ...).
            if per_slot:
                # Kernel already returned per-slot values.
                contrib = yloc
            else:
                # Kernel returned a single value per hyperedge; broadcast across slots.
                contrib = jnp.broadcast_to(yloc[:, None, ...], (yloc.shape[0], K) + yloc.shape[1:])

            # Optional per-slot weighting (useful e.g. to send +/- fluxes to different slots).
            if w is not None:
                # Reshape weights for broadcasting over the trailing axes of contrib.
                wm = w.reshape((1, K) + (1,) * (contrib.ndim - 2))
                contrib = contrib * wm

            # Flatten to align with flattened idx: (M*K, ...)
            contrib = contrib.reshape((yloc.shape[0] * K,) + contrib.shape[2:])
            return out.at[idx].add(contrib)

        raise ValueError(f"Unknown owners {self.owners!r}")

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
        - Slot mask: ``spec.slot_mask`` (optional) has shape matching ``hyper`` and
          is ANDed with ``Mk``. This is how boundary/invalid slots can be disabled
          while preserving fixed K.

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
        globals_extras, particle_arrays = self._split_extras(extras or {}, P=P)

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

        def _per_sample(xb, vb, mb, hyper_b, slot_mask_b, edge_extras_b, slot_extras_b):
            """
            Per-sample worker used both for non-batched and vmapped calls.

            xb: (P, d)
            hyper_b: (M, K)
            """
            Pb = xb.shape[0]
            M = int(hyper_b.shape[0])

            # ---------------------------------------------------------------------
            # Optional exclusion of self-interactions.
            #
            # In grid stencils with "noflux" boundaries, out-of-bounds neighbors are
            # often encoded as "neighbor index = focal index". Treating those slots
            # as invalid (mask=False) is a convenient way to express boundary handling
            # without special cases in the kernel.
            #
            # We therefore compute `slot_keep` that is True for:
            #   - the focal slot itself
            #   - any slot whose participant index differs from the focal index
            # and AND it into Mk.
            # ---------------------------------------------------------------------
            if self.exclude_self:
                focal = hyper_b[:, fi]  # (M,)
                slots = jnp.arange(K, dtype=jnp.int32)[None, :]  # (1, K)
                slot_keep = jnp.logical_or(slots == fi, hyper_b != focal[:, None])  # (M, K)
            else:
                slot_keep = None

            # -------------------------- non-chunked path -------------------------- #
            if self.chunk_size is None:
                # Gather neighborhood state: (M, K, d)
                Xk = xb[hyper_b]

                # Gather optional v and mask to neighborhood slots.
                Vk = vb[hyper_b] if vb is not None else None  # (M, K, d) or None
                Mk = mb[hyper_b] if mb is not None else None  # (M, K) or None

                # Combine all slot-wise masks:
                # - `Mk` from particle mask (dynamic, per sample)
                # - `slot_mask_b` from the spec (static boundary/invalid slots)
                # - `slot_keep` from exclude_self (static, computed from hyper table)
                if slot_mask_b is not None:
                    Mk = slot_mask_b if Mk is None else jnp.logical_and(Mk, slot_mask_b)
                if slot_keep is not None:
                    Mk = slot_keep if Mk is None else jnp.logical_and(Mk, slot_keep)

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

                # Scatter to particle axis.
                return self._scatter_hyper(yloc, hyper_b, Pb, K, fi)

            # ---------------------------- chunked path ---------------------------- #
            CS = int(self.chunk_size)

            # Pad M to a multiple of chunk size so that we can use a simple fori_loop.
            pad = (CS - (M % CS)) % CS
            Mpad = M + pad
            steps = Mpad // CS

            # Pad the hyper list; padded rows are arbitrary but will be masked out via data_mask.
            hyper_pad = jnp.pad(hyper_b, ((0, pad), (0, 0)))

            # Mask marking which hyperedges are real vs padded.
            data_mask = jnp.concatenate([jnp.ones((M,), dtype=bool), jnp.zeros((pad,), dtype=bool)], axis=0)  # (Mpad,)

            # Pad slot_mask and slot_keep to match padded hyper list.
            slot_mask_pad = jnp.pad(slot_mask_b, ((0, pad), (0, 0))) if slot_mask_b is not None else None
            slot_keep_pad = jnp.pad(slot_keep, ((0, pad), (0, 0))) if slot_keep is not None else None

            # Output shape template: rank_shape + (n_features,)
            rb = _rank_shape(self.interactor.rank, d)

            # Accumulator differs for global vs per-particle output.
            if self.owners == "global":
                # For global reductions, we keep one tensor (no particle axis).
                if self.reducer == "max":
                    acc = jnp.full(rb + (int(self.n_features),), -jnp.inf, dtype=xb.dtype)
                else:
                    acc = jnp.zeros(rb + (int(self.n_features),), dtype=xb.dtype)
                cnt = jnp.zeros((), dtype=jnp.int32)  # used only for "mean"
            else:
                # For scattered outputs, keep a full particle-axis accumulator.
                acc = jnp.zeros((Pb,) + rb + (int(self.n_features),), dtype=xb.dtype)
                cnt = jnp.zeros((), dtype=jnp.int32)

            def body(i, carry):
                """
                Process one chunk of hyperedges.

                The chunk is masked by `valid` so that padded hyperedges contribute 0.
                """
                acc, cnt = carry
                start = i * CS

                # Slice the current chunk of hyperedges and validity mask.
                hyper_blk = jax.lax.dynamic_slice_in_dim(hyper_pad, start, CS, axis=0)  # (CS, K)
                valid = jax.lax.dynamic_slice_in_dim(data_mask, start, CS, axis=0)  # (CS,)

                # Gather state / v / mask on this chunk.
                Xk = xb[hyper_blk]  # (CS, K, d)
                Vk = vb[hyper_blk] if vb is not None else None  # (CS, K, d) or None
                Mk = mb[hyper_blk] if mb is not None else None  # (CS, K) or None

                # Combine slot masks (schedule boundary mask, exclude_self, …).
                if slot_mask_pad is not None:
                    sm = jax.lax.dynamic_slice_in_dim(slot_mask_pad, start, CS, axis=0)  # (CS, K)
                    Mk = sm if Mk is None else jnp.logical_and(Mk, sm)
                if slot_keep_pad is not None:
                    sk = jax.lax.dynamic_slice_in_dim(slot_keep_pad, start, CS, axis=0)  # (CS, K)
                    Mk = sk if Mk is None else jnp.logical_and(Mk, sk)

                # Gather particle extras for this chunk.
                ex_slots_blk = self._gather_particle_extras_hyper(particle_arrays, hyper_blk)
                ex_local_blk = globals_extras if not ex_slots_blk else ({**globals_extras, **ex_slots_blk})

                # Slice spec-provided extras if present.
                # Edge extras are (M, ...); slot extras are (M, K, ...) typically.
                if edge_extras_b:
                    ex_edge_blk = self._slice_extras_1d(edge_extras_b, start, CS)
                    ex_local_blk = {**ex_local_blk, **ex_edge_blk}
                if slot_extras_b:
                    ex_slot_blk = self._slice_extras_1d(slot_extras_b, start, CS)
                    ex_local_blk = {**ex_local_blk, **ex_slot_blk}

                # Kernel evaluation on this chunk.
                yloc = self.interactor(Xk, v=Vk, mask=Mk, params=params, extras=ex_local_blk)

                # Mask out padded hyperedges: multiply by valid[..., None, ...] broadcast.
                yloc = yloc * valid.reshape((-1,) + (1,) * (yloc.ndim - 1))

                if self.owners == "global":
                    # Reduce on-the-fly for global ownership.
                    if self.reducer in ("sum", "mean"):
                        acc = acc + yloc.sum(axis=0)
                        cnt = cnt + valid.sum().astype(cnt.dtype)
                    elif self.reducer == "max":
                        acc = jnp.maximum(acc, yloc.max(axis=0))
                    else:
                        raise ValueError(f"Unknown reducer {self.reducer!r}")
                    return acc, cnt

                # Scatter chunk contributions onto particles and accumulate.
                acc = acc + self._scatter_hyper(yloc, hyper_blk, Pb, K, fi)
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
                ),
            )
            return vm(x, v, mask, hyper, slot_mask, edge_extras or None, slot_extras or None)

        return _per_sample(x, v, mask, hyper, slot_mask, edge_extras or None, slot_extras or None)

    def _hyper_csr_vectorized(self, x, *, v, mask, extras, params, spec: HyperCSR):
        """Dispatch a CSR hyperedge schedule.

        :class:`~SFI.statefunc.nodes.interactions.specs.HyperCSR` stores a ragged list
        of hyperedges:

        - ``he_indptr``: shape ``(..., M+1)``
        - ``he_indices``: shape ``(..., total_slots)``

        where hyperedge ``e`` uses participants ``he_indices[indptr[e]:indptr[e+1]]``.

        For JAX-friendly execution, the dispatcher converts this ragged representation
        into a *dense* fixed-K representation by padding each hyperedge to
        ``K = self._K`` and providing a ``slot_mask`` that marks which slots are real.

        This conversion is fully traceable as long as ``K`` is static (fixed-K mode).
        """
        if self._K is None:
            raise NotImplementedError(
                "HyperCSR dispatch requires a static K. Use a fixed-K interactor (or pass a fixed K override)."
            )
        K = int(self._K)

        he_indptr = spec.he_indptr
        he_indices = spec.he_indices
        slot_ids = getattr(spec, "slot_ids", None)
        edge_extras = getattr(spec, "edge_extras", None) or {}
        slot_extras = getattr(spec, "slot_extras", None) or {}

        def _csr_to_hyperfixed(indptr, indices, slot_ids_b, edge_extras_b, slot_extras_b):
            """Convert one CSR schedule (no batch axis) into :class:`HyperFixed`.

            The conversion keeps:
            - ``edge_extras``: aligned with hyperedges (axis 0 = M)
            - ``slot_extras``: aligned with flattened slots (axis 0 = total_slots)
              and scattered into a dense (M, K, ...) representation.
            """
            total = indices.shape[0]
            M = indptr.shape[0] - 1

            # Each flattened slot knows which hyperedge it belongs to.
            pos = jnp.arange(total, dtype=jnp.int32)
            edge_id = jnp.searchsorted(indptr[1:], pos, side="right")  # (total,)
            in_edge = pos - indptr[edge_id]  # (total,)

            in_range = in_edge < K

            # Initialize dense participant table by repeating the first participant
            # of each hyperedge (convenient padding value).
            first = indices[indptr[:-1]]  # (M,)
            hyper = jnp.repeat(first[:, None], K, axis=1)
            slot_mask = jnp.zeros((M, K), dtype=jnp.bool_)

            # Scatter the actual participants into their slot positions.
            hyper = hyper.at[edge_id[in_range], in_edge[in_range]].set(indices[in_range])
            slot_mask = slot_mask.at[edge_id[in_range], in_edge[in_range]].set(True)

            # Convert slot_ids into a dense per-slot extra (if present).
            slot_extras_fixed: Dict[str, Array] = {}
            if slot_ids_b is not None:
                sid = jnp.full((M, K), -1, dtype=slot_ids_b.dtype)
                sid = sid.at[edge_id[in_range], in_edge[in_range]].set(slot_ids_b[in_range])
                slot_extras_fixed["slot_id"] = sid

            # Scatter any explicit slot extras (aligned with `indices`) to (M, K, ...).
            for kname, arr in (slot_extras_b or {}).items():
                z = jnp.zeros((M, K) + arr.shape[1:], dtype=arr.dtype)
                z = z.at[edge_id[in_range], in_edge[in_range]].set(arr[in_range])
                slot_extras_fixed[kname] = z

            return HyperFixed(
                hyper=hyper,
                slot_mask=slot_mask,
                edge_extras=edge_extras_b or None,
                slot_extras=slot_extras_fixed or None,
            )

        # Support shared schedule (no batch) vs per-sample schedule (vmapped).
        *batch, _P, _d = x.shape
        if batch:
            # Assume schedule arrays (indptr, indices, extras) follow the same
            # batching pattern as the state x.
            vm = jax.vmap(
                _csr_to_hyperfixed,
                in_axes=(
                    0,
                    0,
                    0 if slot_ids is not None else None,
                    0 if edge_extras else None,
                    0 if slot_extras else None,
                ),
            )
            hf = vm(
                he_indptr,
                he_indices,
                slot_ids,
                edge_extras or None,
                slot_extras or None,
            )
        else:
            hf = _csr_to_hyperfixed(
                he_indptr,
                he_indices,
                slot_ids,
                edge_extras or None,
                slot_extras or None,
            )

        return self._hyper_fixed_vectorized(x, v=v, mask=mask, extras=extras, params=params, spec=hf)

    def _split_extras(self, extras: dict | None, *, P: int):
        """
        Return (globals_extras, particle_arrays) where:
          - globals_extras : dict of non-structural, non-particle keys (pass-through)
          - particle_arrays: dict[key -> (P, ...)] gathered per edge by the dispatcher
        """
        if not extras:
            return {}, {}
        particle_keys = set(getattr(self.interactor, "particle_extras", ()) or ())
        structural = set(getattr(self, "structural_extras", ()) or ())
        globals_extras, particle_arrays = {}, {}
        for k, v in extras.items():
            if k in structural:
                continue  # dispatcher-owned; never forward
            if k in particle_keys:
                if not (hasattr(v, "shape") and v.shape and v.shape[0] == P):
                    raise ValueError(f"Particle extra '{k}' must have shape (P, ...); got {getattr(v, 'shape', None)}")
                particle_arrays[k] = v
            else:
                globals_extras[k] = v
        return globals_extras, particle_arrays

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
        # HyperCSR / unknown: fall back to generic AD path
        return super()._same_particle_jacobian(x, var=var, v=v, params=params, mask=mask, extras=extras)

    # ·············· PairsCSR per-edge Jacobian ··············
    def _pairs_csr_spj(self, x, *, v, mask, extras, params, spec, var):
        P, d = x.shape
        globals_ex, particle_arrays = self._split_extras(extras or {}, P=P)

        indptr, indices = spec.indptr, spec.indices
        M = indices.shape[0]
        k = jnp.arange(M, dtype=jnp.int32)
        row = jnp.searchsorted(indptr[1:], k, side="right")
        col = indices

        keep = jnp.logical_not(row == col) if self.exclude_self else jnp.ones(M, dtype=bool)

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
        J_all = J_all * km

        # Degree normalization (if applicable)
        deg = None
        if self.reducer == "mean" and self.normalize_by_degree:
            deg = (indptr[1:] - indptr[:-1]).astype(jnp.int32)

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
        P, d = x.shape
        globals_ex, particle_arrays = self._split_extras(extras or {}, P=P)

        hyper = spec.hyper  # (M, K)
        slot_mask_spec = getattr(spec, "slot_mask", None)
        edge_extras_spec = getattr(spec, "edge_extras", None) or {}
        slot_extras_spec = getattr(spec, "slot_extras", None) or {}

        K = int(hyper.shape[-1])
        fi = self.focal_index

        # Gather stencils
        Xk = x[hyper]  # (M, K, d)
        Vk = v[hyper] if v is not None else None
        Mk = mask[hyper] if mask is not None else None

        # Combine slot masks
        if slot_mask_spec is not None:
            Mk = slot_mask_spec if Mk is None else jnp.logical_and(Mk, slot_mask_spec)
        if self.exclude_self:
            focal = hyper[:, fi]
            slots = jnp.arange(K, dtype=jnp.int32)[None, :]
            slot_keep = jnp.logical_or(slots == fi, hyper != focal[:, None])
            Mk = slot_keep if Mk is None else jnp.logical_and(Mk, slot_keep)

        # Collect all per-edge varying extras
        edge_varying = {}
        ex_slots = self._gather_particle_extras_hyper(particle_arrays, hyper)
        if ex_slots:
            edge_varying.update(ex_slots)
        if edge_extras_spec:
            edge_varying.update(edge_extras_spec)
        if slot_extras_spec:
            edge_varying.update(slot_extras_spec)

        # — per-edge jacfwd —
        J_all = self._vmap_jacfwd_edges(
            Xk,
            Vk,
            Mk,
            edge_varying,
            globals_ex,
            params,
            var,
        )  # (M, rank..., F, K, d)

        # Scatter based on owners
        out_shape = (P,) + J_all.shape[1:-2] + (J_all.shape[-1],)
        out = jnp.zeros(out_shape, dtype=x.dtype)

        if self.owners == "focal":
            J_focal = J_all[..., fi, :]
            out = out.at[hyper[:, fi]].add(J_focal)
        elif self.owners in ("all", "custom"):
            w = self.owner_weights
            for s in range(K):
                J_s = J_all[..., s, :]
                if w is not None:
                    J_s = w[s] * J_s
                out = out.at[hyper[:, s]].add(J_s)
        else:
            raise ValueError(f"Unsupported owners='{self.owners}' for same-particle Jacobian")

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
        out_per_edge_bytes = rb * mfeat * isz

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
        out_per_edge_bytes = rb * mfeat * isz

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
        """Return interaction arity if fixed; otherwise None."""
        if getattr(self, "_kmode", None) == "fixed" and getattr(self, "_K", None) is not None:
            return int(self._K)
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
