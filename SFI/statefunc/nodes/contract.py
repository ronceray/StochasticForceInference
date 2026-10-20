# SFI/statefunc/contract.py
# --------------------------

from enum import IntEnum
from typing import Any, Optional, Sequence

import equinox as eqx
from jaxtyping import Array

from ..params import ParamSuite


# ---------------------------------------------------------------------------
# Enums - ease of reading & cheap comparisons
# ---------------------------------------------------------------------------
class Rank(IntEnum):
    SCALAR = 0
    VECTOR = 1
    MATRIX = 2
    TENSOR3 = 3
    TENSOR4 = 4  # extend if/when needed


# ---------------------------------------------------------------------
# Contract mix-in  (rank | dim | pdepth | … + runtime guards)
# ---------------------------------------------------------------------
class _ContractMixin(eqx.Module):
    """
    Static contract shared by every StateExpr node.

    Parameters are stored as *eqx.field(static=True)* so they never flow into AD.

    Attributes
    ----------
    rank        : Rank enum
        Tensor rank of each basis element (0=scalar, 1=vector, ...).
    dim         : int | None
        Spatial dimension (None = agnostic).
    needs_v     : bool
        Dictionary requires velocity input?
    pdepth      : int
        Particle depth: how many particle axes the node's **output** carries.
        Inputs with `particles_input=True` *consume* one particle axis from `x`.
    n_features  : int
        Number of basis functions/features carried by this node (feature-last).
    param_suite : ParamSuite | None
        - None → deterministic node
        - ParamSuite → parametric node (no broadcasting of params).
    extras_required : tuple[str, ...]
        Presence-only requirements for the `extras` mapping. **No broadcasting is
        validated here.** Dispatcher/leaves decide how extras are consumed.
    particles_input : bool
        If True, `x` is expected to have a particle axis: layout `batch · P · dim`.
    sdims : tuple[int, ...] | None
        Spatial dimensions of the node's output. None = agnostic. Useful for structured
        data like grids, where each rank axis may have a different size. If provided, it
        is used for validating outputs and merging contracts of composite nodes (e.g.
        concatenation requires matching sdims on the concatenated axis). For leaf nodes,
        if sdims is None but dim is provided, we treat all rank axes as having size `dim`.
        For composite nodes, sdims must be explicitly provided if any child has sdims; we
        do not attempt to infer or broadcast sdims from dim in that case.
    """

    # ---------- static metadata --------------------------------------
    rank: "Rank" = eqx.field(static=True)
    dim: Optional[int] = eqx.field(static=True, default=None)
    needs_v: bool = eqx.field(static=True, default=False)
    pdepth: int = eqx.field(static=True, default=0)
    n_features: int = eqx.field(static=True, default=1)
    param_suite: ParamSuite | None = eqx.field(static=True, default=None)
    extras_required: tuple[str, ...] = eqx.field(static=True, default=())
    particles_input: bool = eqx.field(static=True, default=False)
    sdims: tuple[int, ...] | None = eqx.field(static=True, default=None)

    @staticmethod
    def inherit_contract(child: "_ContractMixin", **overrides) -> dict:
        """
        Start from child's static contract and override only what changes.
        Keys: rank, dim, pdepth, n_features, needs_v, particles_input, sdims
        """
        out = dict(
            rank=child.rank,
            dim=child.dim,
            pdepth=child.pdepth,
            n_features=child.n_features,
            needs_v=child.needs_v,
            particles_input=child.particles_input,
            sdims=child.sdims,
        )
        out.update(overrides)
        return out

    # -----------------------------------------------------------------
    # merge helper used by composite nodes
    # -----------------------------------------------------------------
    @classmethod
    def merge_contract(
        cls,
        children: Sequence["_ContractMixin"],
        *,
        mode: str,
        spec: str | None = None,
    ) -> dict:
        """
        Single source of truth for merging static contracts of composite nodes.

        Parameters
        ----------
        children : sequence of nodes
            The operands to merge.
        mode : {'concat','map','einsum'}
            Semantic of the operation.
            ``'concat'``: feature-axis concatenation.
            ``'map'``: elementwise Hadamard-like map (identical shapes/features).
            ``'einsum'``: spatial contraction; requires ``spec``.
        spec : str | None
            Einstein notation for 'einsum' mode, lowercase letters only
            on spatial axes. Example: "m,n->mn", "m,m->", ",n->n".

        Returns
        -------
        dict with keys: rank, dim, pdepth, n_features, needs_v, particles_input

        Notes
        -----
        - Particle-depth and particles_input are merged with the broadcast rule:
            Allowed mixes:
              - identical (pdepth, particles_input) for all children
              - OR (pdepth=0, particles_input=False) combined with (pdepth=1, particles_input=True)
                nodes. In that case the input particle axis is treated as broadcast by the False node,
                which aligns and conceptually matches the particle axis in the True node.
            The merged depth is the **maximum**; particles_input is True if any child has it.
        - This function is purposely strict on error messages to make debugging easy.
        - Extras policy: composite nodes **union** children’s `extras_required` at construction
          (presence-only). This function does not validate extras shapes or broadcasting.
        """
        if not children:
            raise ValueError("merge_contract(mode=%r): needs at least one child" % mode)

        # ---------------- helpers ----------------
        def _merge_pdepth(nodes: Sequence["_ContractMixin"]) -> tuple[int, bool]:
            pd, flag = nodes[0].pdepth, nodes[0].particles_input
            for ch in nodes[1:]:
                if (ch.pdepth, ch.particles_input) == (pd, flag):
                    continue
                # benign broadcast case
                if (pd, flag) == (0, False) and ch.particles_input and ch.pdepth == 1:
                    pd, flag = ch.pdepth, True
                    continue
                if (ch.pdepth, ch.particles_input) == (0, False) and flag and pd == 1:
                    # keep current pd/flag
                    continue
                raise ValueError(
                    "merge_contract(%s): incompatible particle-depth mix: "
                    "A(pdepth=%d, particles_input=%s) vs B(pdepth=%d, particles_input=%s)"
                    % (mode, pd, flag, ch.pdepth, ch.particles_input)
                )
            return int(pd), bool(flag)

        def _unify_dim(nodes: Sequence["_ContractMixin"]) -> int | None:
            dims = [n.dim for n in nodes]
            concrete = [d for d in dims if d is not None]
            if not concrete:
                return None
            d0 = concrete[0]
            if any(d != d0 for d in concrete[1:]):
                raise ValueError("merge_contract(%s): children disagree on `dim`: %s" % (mode, tuple(dims)))
            return d0

        def _all_equal(attrs: tuple[str, ...]) -> tuple[bool, dict]:
            """Check equality across children for the given attributes."""
            vals = {a: tuple(getattr(ch, a) for ch in children) for a in attrs}
            ok = all(len(set(vals[a])) == 1 for a in attrs)
            return ok, vals

        def _unify_sdims(nodes: Sequence["_ContractMixin"]) -> tuple[int, ...] | None:
            """Merge sdims: all must agree, or all be None."""
            all_sd = [n.sdims for n in nodes]
            if all(s is None for s in all_sd):
                return None
            if any(s is None for s in all_sd):
                raise ValueError(
                    "merge_contract(%s): some children have sdims=None while "
                    "others have explicit sdims: %s" % (mode, all_sd)
                )
            s0 = all_sd[0]
            if any(s != s0 for s in all_sd[1:]):
                raise ValueError("merge_contract(%s): children disagree on sdims: %s" % (mode, all_sd))
            return s0

        # -------------- fields independent of mode --------------
        needs_v = any(ch.needs_v for ch in children)
        pdepth, pflag = _merge_pdepth(children)
        dim = _unify_dim(children)

        # -------------- mode-specific logic ---------------------
        mode = mode.lower()
        if mode == "concat":
            ok, vals = _all_equal(("rank",))
            if not ok:
                raise ValueError(f"merge_contract(concat): children must share rank; got rank={vals['rank']}")
            n_features = sum(int(ch.n_features) for ch in children)
            rank = int(children[0].rank)
            sdims = _unify_sdims(children)

        elif mode in {"map"}:
            ok, vals = _all_equal(("rank", "n_features"))
            if not ok:
                raise ValueError(
                    f"merge_contract({mode}): children must share rank and n_features; "
                    f"got rank={vals['rank']}, n_features={vals['n_features']}"
                )
            n_features = int(children[0].n_features)
            rank = int(children[0].rank)
            sdims = _unify_sdims(children)

        elif mode == "einsum":
            if not spec or "->" not in spec:
                raise ValueError(
                    "merge_contract(einsum): a valid `spec` like 'i,j->ij' is required"
                    " (with RHS; no ellipses; batch is implicit)"
                )
            if "..." in spec:
                raise ValueError(
                    "merge_contract(einsum): ellipsis '...' is reserved for batch and must NOT appear in specs"
                )

            lhs, rhs = spec.replace(" ", "").split("->")
            lhs_ops = lhs.split(",")

            if len(lhs_ops) != len(children):
                raise ValueError(
                    "merge_contract(einsum): spec lists %d operands but %d children were provided"
                    % (len(lhs_ops), len(children))
                )

            def _letters_only(op: str) -> str:
                # Only lowercase letters; no dots allowed at all.
                if "." in op:
                    raise ValueError(f"merge_contract(einsum): malformed operand {op!r} (dots/ellipsis not allowed)")
                bad = [c for c in op if not c.islower()]
                if bad:
                    raise ValueError(f"merge_contract(einsum): operand {op!r} contains non-lowercase symbols: {bad}")
                return op

            # LHS validation and rank compatibility (spatial rank == len(token))
            lhs_clean = []
            for idx, (ch, op) in enumerate(zip(children, lhs_ops)):
                core = _letters_only(op)
                if len(core) != int(ch.rank):
                    raise ValueError(
                        "merge_contract(einsum): rank mismatch for child %d: child.rank=%d, "
                        "operand %r has %d spatial letters" % (idx, int(ch.rank), op, len(core))
                    )
                lhs_clean.append(core)

            # RHS validation: subset of union of LHS letters
            rhs_core = _letters_only(rhs)
            if set(rhs_core) - set("".join(lhs_clean)):
                raise ValueError("merge_contract(einsum): RHS contains letters not present on LHS")

            rank = len(rhs_core)

            # Feature axes: Cartesian product
            from functools import reduce
            from operator import mul

            n_features = reduce(mul, (int(ch.n_features) for ch in children), 1)

            # --- sdims for einsum: build per-letter size map ---
            any_has_sdims = any(ch.sdims is not None for ch in children)
            if any_has_sdims:
                letter_size: dict[str, int] = {}
                for ch, op in zip(children, lhs_clean):
                    ch_sdims = ch.sdims
                    if ch_sdims is None:
                        # Uniform dim: all rank axes are dim-sized
                        if ch.dim is None:
                            raise ValueError(
                                "merge_contract(einsum): child has sdims=None and "
                                "dim=None but other children have sdims; cannot "
                                "determine letter sizes"
                            )
                        ch_sdims = (ch.dim,) * int(ch.rank)
                    for letter, size in zip(op, ch_sdims):
                        if letter in letter_size:
                            if letter_size[letter] != size:
                                raise ValueError(
                                    f"merge_contract(einsum): letter {letter!r} has "
                                    f"inconsistent sizes: {letter_size[letter]} vs {size}"
                                )
                        else:
                            letter_size[letter] = size
                sdims = tuple(letter_size[c] for c in rhs_core) if rhs_core else ()
            else:
                sdims = None

        else:
            raise ValueError("merge_contract: unknown mode %r" % mode)

        return dict(
            rank=Rank(rank),
            dim=dim,
            sdims=sdims,
            pdepth=int(pdepth),
            n_features=int(n_features),
            needs_v=bool(needs_v),
            particles_input=bool(pflag),
        )

    def _assert_inputs(
        self,
        x: Array,
        v: Optional[Array],
        mask: Optional[Array],
        extras: Any | None = None,
    ):
        # ---------- spatial dimension ---------------------------------
        if (self.dim is not None) and (x.shape[-1] != self.dim):
            raise ValueError(f"[Node] expected x.shape[-1]=={self.dim}, got {x.shape[-1]}")

        # ---------- minimal rank of x (batch · [P?] · dim) ------------
        min_ndim = 2 if self.particles_input else 1
        if x.ndim < min_ndim:
            need = "batch·P·dim" if self.particles_input else "batch·dim"
            raise ValueError(f"[Node] x.ndim={x.ndim} < required for {need}")

        # Prefixes used for later checks
        prefix_shape = x.shape[:-1]  # everything except spatial dim (includes P if present)

        # ---------- velocity ------------------------------------------
        if self.needs_v:
            if v is None:
                raise ValueError("[Node] velocity `v` required but not provided")
            if v.shape != x.shape:
                raise ValueError("[Node] `v` must have same shape as `x`")

        # ---------- helpers -------------------------------------------
        def _strip_trailing_ones(shp: tuple[int, ...]) -> tuple[int, ...]:
            k = len(shp)
            while k > 0 and shp[k - 1] == 1:
                k -= 1
            return shp[:k]

        def _broadcastable(dst: tuple[int, ...], src: tuple[int, ...]) -> bool:
            # NumPy broadcasting: align from the right
            for a, b in zip(dst[::-1], src[::-1]):
                if b not in (1, a):
                    return False
            if len(src) > len(dst):
                lead = src[: len(src) - len(dst)]
                return all(d == 1 for d in lead)
            return True

        # ---------- mask ----------------------------------------------
        if mask is not None:
            # Allow boolean OR numeric masks. Require broadcast to prefix_shape (batch·[P]),
            # without stripping trailing ones (so (B,1) can broadcast to (B,P)).
            if hasattr(mask, "shape"):
                mshape = tuple(mask.shape)

                if not _broadcastable(prefix_shape, mshape):
                    raise ValueError(
                        "[Node] mask must broadcast to x's batch/particle prefix: "
                        f"expected {prefix_shape}, got {mask.shape}"
                    )
            else:
                # Non-array (e.g. Python float) is treated as scalar broadcast.
                pass

        # ----- extras: presence-only contract (no broadcasting checks here) -----
        req = tuple(getattr(self, "extras_required", ()) or ())

        if req:
            if extras is None:
                # Named keys were declared but no mapping was provided at all.
                raise KeyError(f"[Node] missing extras keys: {req}")
            missing = [k for k in req if k not in extras]
            if missing:
                raise KeyError(f"[Node] missing extras keys: {tuple(missing)}")
        # (If extras is provided, we do not validate shapes here—dispatcher/leaf runtime handles that.)

    # -----------------------------------------------------------------
    def _assert_outputs(self, x: Array, y: Array):
        # -------- feature axis size -----------------------------------------
        if y.shape[-1] != self.n_features:
            raise ValueError(f"[Node] expected last axis length {self.n_features}, got {y.shape[-1]}")

        # -------- rank axes size ---------------------------------------------
        r_axes = int(self.rank)  # 0 = scalar, 1 = vector, etc.
        rank_shape = () if r_axes == 0 else y.shape[-(r_axes + 1) : -1]
        if r_axes > 0:
            if self.sdims is not None:
                # Structured dimensions: per-axis sizes from sdims
                expected_rank = self.sdims
            elif self.dim is not None:
                # Uniform dimensions: all rank axes are dim-sized
                expected_rank = (self.dim,) * r_axes
            else:
                expected_rank = None
            if expected_rank is not None and rank_shape != expected_rank:
                raise ValueError(f"[Node] rank axes shape {rank_shape} ≠ expected {expected_rank}")

        # -------- batch / particle prefix --------------------
        # Layouts:
        #   x : batch · [P_in?] · dim
        #   y : batch · [P_out^pdepth] · (rank) · feature
        y_prefix = y.shape[: -(r_axes + 1)]
        x_batch = x.shape[:-2] if self.particles_input else x.shape[:-1]

        if self.particles_input:
            P_in = x.shape[-2]
            expected = x_batch + ((P_in,) * int(self.pdepth))
        else:
            expected = x_batch if int(self.pdepth) == 0 else None

        if expected is None or y_prefix != expected:
            raise ValueError(
                "[Node] output prefix incompatible: "
                f"y_prefix={y_prefix}, expected={expected}, "
                f"particles_input={self.particles_input}, pdepth={self.pdepth}"
            )
