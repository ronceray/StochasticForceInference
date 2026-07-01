# SFI/statefunc/nodes/ops/derivative.py
"""Derivative operator node."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from ...memhint import SampleMeta, inflate_for_grad
from ..base import BaseNode, BaseOpNode
from ..contract import _ContractMixin


# ──────────────────────────────────────────────────────────────────────────────
#  DerivativeNode – first-order Jacobian with optional cross-particle block
# ──────────────────────────────────────────────────────────────────────────────
class DerivativeNode(BaseOpNode):
    """First-order derivative wrapper around *one* child node.

    Parameters
    ----------
    child  : BaseNode
    var    : {'x', 'v', 'theta'}
        Choose which variable to differentiate with respect to.
    same_particle  : bool, default False
        Only meaningful for ``var in {'x','v'}``.
        Requires ``child.particles_input is True`` and ``child.pdepth == 1``.

        - ``True``  -> same-particle Jacobian  df[i]/dx[i]  (work/memory ``O(P)``).
        - ``False`` -> full cross Jacobian    df[i]/dx[j]  (work/memory ``O(P**2)``).

    mode   : {'auto','fwd','rev'}
        JAX autodiff flavour.  ``'auto'`` maps to:
        - x, v  → ``jacfwd`` (forward mode)
        - theta → ``jacrev`` (reverse mode)

    Shapes produced
    ---------------
    *Suffix; batch prefix unchanged.*

    Let ``P_out`` denote the child's **output** particle axis count (pdepth),
    and ``P_in`` the **input** particle axis (present **only when** the model
    has particle inputs). Let ``dim`` be spatial dimension, ``rank`` the existing
    spatial rank axes, and ``F`` the feature axis.

    **Noninteracting case** (pdepth=0, particles_input=False):

    For noninteracting nodes, inputs have shape ``batch · dim`` and outputs
    ``batch · rank … · F``. First-order derivatives w.r.t. ``x``/``v`` are computed
    **per sample** by temporarily un-batching the input to shape ``(dim,)`` and
    then vmapping the per-sample Jacobian back over the batch prefix. The result
    shape is::

        d_x(), d_v():   batch · dim · rank … · F

    No cross-batch Jacobian axes are formed.

    **Interacting case** (particles_input=True):

    ================  ===========================================================
    call               output suffix (after derivative)
    ================  ===========================================================
    d_x(same=True)     ``P_out^1 · dim · rank … · F``               (pdepth unchanged)
    d_x(same=False)    ``P_in · P_out^1 · dim · rank … · F``        (**pdepth → pdepth+1**; only if particles_input)
    d_v(same=True)     same as d_x(same=True)
    d_v(same=False)    same as d_x(same=False)
    d_theta()          ``rank … · (F x n_param)``
    ================  ===========================================================

    Notes
    -----
    - Derivatives w.r.t. x/v *increase spatial rank by 1* (insert new derivative-dim
      immediately **before** the existing rank block).
    - For not ``same_particle`` with particle inputs, the derivative introduces a new
      **input particle axis** positioned **before** the child's pdepth block, and
      the node's **pdepth is incremented by 1**. If there is **no** particle input,
      no extra axis is added and pdepth is unchanged.
    - Theta-Jacobian is formed leafwise and concatenated along the final axis, fusing
      (features x total_param_elems).
    """

    var: str = eqx.field(static=True)
    same_particle: bool = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(
        self,
        child: "BaseNode",
        *,
        var: str,  # 'x' | 'v' | 'theta'
        same_particle: bool = False,
        mode: str = "auto",
    ):
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "var", var)
        object.__setattr__(self, "same_particle", bool(same_particle))
        object.__setattr__(self, "mode", mode)

        if var not in {"x", "v", "theta"}:
            raise ValueError("var must be 'x', 'v', or 'theta'")
        if mode not in {"auto", "fwd", "rev"}:
            raise ValueError("mode must be 'auto', 'fwd', or 'rev'")
        if var == "theta" and (child.param_suite is None or child.param_suite.size == 0):
            raise ValueError("θ-gradient requested but child has no parameters")

        if var in {"x", "v"} and self.same_particle:
            if not child.particles_input or child.pdepth != 1:
                raise ValueError(
                    "same-particle derivative requires particles_input=True and pdepth==1 "
                    "(differentiate after slicing/selecting a single particle output)."
                )
        if var == "v" and not child.needs_v:
            raise ValueError("v-gradient requested but child does not depend on v")

        super().__init__(child)
        object.__setattr__(self, "param_suite", child.param_suite)

    def _merge_static(self, children):
        ch = children[0]
        new_rank = ch.rank + (1 if self.var in {"x", "v"} else 0)

        add_p = (self.var in {"x", "v"}) and ch.particles_input and (not self.same_particle)
        new_pdepth = ch.pdepth + (1 if add_p else 0)

        new_nfeat = ch.n_features
        if self.var == "theta":
            assert ch.param_suite is not None
            new_nfeat *= max(1, ch.param_suite.size)

        return _ContractMixin.inherit_contract(ch, rank=new_rank, pdepth=new_pdepth, n_features=new_nfeat)

    # ------------------------------------------------------------------
    #  Feature-aware flatten  (auto-labelling for derivatives)
    # ------------------------------------------------------------------
    def flatten(self):
        child = self.children[0]
        funcs_ch, labels_ch, descs_ch = child.flatten()

        prefix = {"x": "∂ₓ", "v": "∂ᵥ", "theta": "∂θ"}[self.var]

        def _wrap(lab):
            return lab if len(lab) == 1 else f"({lab})"

        if self.var in {"x", "v"}:
            # Feature count unchanged; just decorate labels.
            labels = [f"{prefix}{_wrap(lab)}" for lab in labels_ch]
        else:
            # theta: fused axis = n_features × n_params_total
            assert child.param_suite is not None
            n_params = max(1, child.param_suite.size)
            labels = [f"{prefix}({lab},{k})" for lab in labels_ch for k in range(n_params)]

        n_feat = len(labels)

        def _slice_feature(j):
            return lambda x, *, v=None, mask=None, extras=None, params=None, _j=j: (
                self(x, v=v, mask=mask, extras=extras, params=params)[..., _j]
            )

        funcs = [_slice_feature(j) for j in range(n_feat)]
        descs = [
            {"derivative": self.var, "child": d} for d in (descs_ch * (n_feat // len(descs_ch) if descs_ch else 1))
        ]

        return funcs, labels, descs

    def __call__(self, x, *, v=None, params=None, mask=None, extras=None):
        """Compute the requested first-order Jacobian with proper un-batching."""
        self._assert_inputs(x, v, mask, extras)
        child = self.children[0]

        # pick AD flavour
        mode = (
            "fwd"
            if (self.mode == "auto" and self.var in {"x", "v"})
            else "rev"
            if (self.mode == "auto" and self.var == "theta")
            else self.mode
        )
        diff = jax.jacfwd if mode == "fwd" else jax.jacrev

        # ----- θ derivative ------------------------------------------
        if self.var == "theta":

            def gθ(_x, _v, _θ):
                return child(_x, v=_v, params=_θ, mask=mask, extras=extras)

            Jtree = diff(gθ, argnums=2)(x, v, params)
            J_leaves, treedefJ = jax.tree_util.tree_flatten(Jtree)
            P_leaves, treedefP = jax.tree_util.tree_flatten(params)
            if treedefJ != treedefP:
                raise ValueError("[DerivativeNode] Jacobian tree does not match params tree structure")
            folded = []
            for J_leaf, p_leaf in zip(J_leaves, P_leaves):
                pr = int(getattr(p_leaf, "ndim", 0))
                A = (
                    jnp.reshape(J_leaf, J_leaf.shape + (1,))
                    if pr == 0
                    else jnp.reshape(J_leaf, J_leaf.shape[:-pr] + (-1,))
                )
                # fuse the last two axes (ranked output axis × param axis)
                A = jnp.reshape(A, A.shape[:-2] + (A.shape[-2] * A.shape[-1],))
                folded.append(A)
            if not folded:
                return child(x, v=v, params=params, mask=mask, extras=extras)
            return jnp.concatenate(folded, axis=-1)

        # ----- x/v derivative paths: un-batch then vmap back ----------------------
        # core dims for the diff'ed argument
        expect_particles = bool(child.particles_input)
        core_nd = 2 if expect_particles else 1  # (P,dim) vs (dim,)
        b_nd = max(0, x.ndim - core_nd)  # how many leading batch axes in x

        # Helpers to broadcast/bucket mask/extras to the batch prefix and flatten it
        def _broadcast_prefix_to(a, bshape):
            if a is None:
                return None
            if not (hasattr(a, "shape") and hasattr(a, "dtype")):
                return a  # non-array leaf stays scalar/constant
            # prepend ones so we can broadcast to the full prefix
            pad = len(bshape) - a.ndim if a.ndim < len(bshape) else 0
            if pad > 0:
                a = jnp.reshape(a, (1,) * pad + a.shape)
            # target shape = bshape + trailing
            trailing = a.shape[len(bshape) :]
            target = tuple(bshape) + tuple(trailing)
            a = jnp.broadcast_to(a, target)
            return a

        def _prep_tree_for_flat_map(tree, bshape):
            return jax.tree_util.tree_map(lambda leaf: _broadcast_prefix_to(leaf, bshape), tree)

        def _flatten_tree_prefix(tree, bshape):
            """Flatten the batch prefix of every leaf so it matches x_flat."""

            def _flat(a):
                if a is None:
                    return None
                if not (hasattr(a, "shape") and hasattr(a, "dtype")):
                    return a
                trailing = a.shape[len(bshape) :]
                return jnp.reshape(a, (-1,) + trailing)

            return jax.tree_util.tree_map(_flat, tree)

        # same-particle case: only valid for interacting (already validated in __init__)
        if self.same_particle:
            if b_nd == 0:
                # Direct call — _same_particle_jacobian returns contract layout
                return child._same_particle_jacobian(x, var=self.var, v=v, params=params, mask=mask, extras=extras)

            # vmap per-sample across all leading batch axes
            bshape = x.shape[:b_nd]

            # 1) Prepare/broadcast and flatten mask to batch prefix
            mask_flat = _prep_tree_for_flat_map(mask, bshape)
            mask_flat = _flatten_tree_prefix(mask_flat, bshape)

            # 2) Flatten x (and v); extras are **closed over** unchanged.
            x_flat = jnp.reshape(x, (-1,) + x.shape[-core_nd:])
            v_flat = None if v is None else jnp.reshape(v, (-1,) + v.shape[-core_nd:])

            in_axes_mask_sp = jax.tree_util.tree_map(lambda a: 0 if (hasattr(a, "shape")) else None, mask_flat)

            def map_one_sp(xi, vi, mi):
                return child._same_particle_jacobian(xi, var=self.var, v=vi, params=params, mask=mi, extras=extras)

            vm = jax.vmap(
                map_one_sp,
                in_axes=(0, 0 if v_flat is not None else None, in_axes_mask_sp),
            )
            J_flat = vm(x_flat, v_flat, mask_flat)
            return jnp.reshape(J_flat, bshape + J_flat.shape[1:])

        # cross-particle (or noninteracting) derivative → jacobian wrt x or v
        argnum = 0 if self.var == "x" else 1

        if b_nd == 0:
            # simple per-sample diff on (dim,) or (P,dim)
            J = diff(
                lambda _x, _v: child(_x, v=_v, params=params, mask=mask, extras=extras),
                argnums=argnum,
            )(x, v)
            return _move_deriv_axis(
                J,
                rank=child.rank,
                pdepth_old=child.pdepth,
                particles_input=child.particles_input,
                same_particle=False,
            )

        # batched case: vmap over the entire batch prefix (flattened), then restore shape
        bshape = x.shape[:b_nd]
        mask_b = _prep_tree_for_flat_map(mask, bshape)
        mask_b = _flatten_tree_prefix(mask_b, bshape)  # (K,N,...) → (K*N,...)
        if mask_b is not None and hasattr(mask_b, "shape"):
            mask_b = jnp.reshape(mask_b, (-1,) + mask_b.shape[len(bshape) :])
        x_flat = jnp.reshape(x, (-1,) + x.shape[-core_nd:])
        v_flat = None if v is None else jnp.reshape(v, (-1,) + v.shape[-core_nd:])

        def map_one(xi, vi, mi):
            return diff(
                lambda _x, _v: child(_x, v=_v, params=params, mask=mi, extras=extras),
                argnums=argnum,
            )(xi, vi)

        in_axes_mask = jax.tree_util.tree_map(lambda a: 0 if (hasattr(a, "shape")) else None, mask_b)
        vm = jax.vmap(map_one, in_axes=(0, 0 if v_flat is not None else None, in_axes_mask))
        J_flat = vm(x_flat, v_flat, mask_b)
        J = jnp.reshape(J_flat, bshape + J_flat.shape[1:])
        return _move_deriv_axis(
            J,
            rank=child.rank,
            pdepth_old=child.pdepth,
            particles_input=child.particles_input,
            same_particle=False,
        )

    # Derivatives tend to keep tangents/tapes alive. Inflate conservatively in grad mode.
    def memory_hint(
        self,
        *,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ):
        base = super().memory_hint(dtype=dtype, particle_size=particle_size, sample=sample, mode=mode)
        return inflate_for_grad(base, factor=2.0) if mode == "grad" else base


def _same_particle_grad(g, diff, x, v, params, *, var: str = "x", **g_kwargs):
    """
    Compute same-particle blocks without forming the full PxP.

    The callable `g` must have signature:
        g(x, v, params, **g_kwargs) -> y
    and should already close over any global/edge-local context (e.g. `extras`, `mask`)
    if not provided via `g_kwargs`.

    For var == 'x': returns ∂f_i / ∂x_i
    For var == 'v': returns ∂f_i / ∂v_i

    Let x.shape == B₁ · B₂ · … · Bk · P · dim (k ≥ 0 batch axes).
    Result shape:
        B₁ · B₂ · … · Bk · P · dim · (rank …) · (feature?)

    Implementation detail:
        - vmap across ALL batch axes (k nested vmaps), then over the particle index.
        - “Added particle axis” elsewhere refers to input particles and only matters
          when `particles_input=True`.
    """
    *batch_shape, P, dim = x.shape
    particles_axis = -2  # P is last-but-one

    def per_batch(_x, _v):
        def inner(i):
            if var == "x":
                yi = lax.dynamic_index_in_dim(_x, i, axis=particles_axis, keepdims=False)  # (dim,)

                def repl(y_local):
                    x_mod = lax.dynamic_update_index_in_dim(_x, y_local, i, axis=particles_axis)
                    # NOTE: extras/mask are expected to be captured by `g` or provided via **g_kwargs
                    y_full = g(x_mod, v=_v, params=params, **g_kwargs)  # shape: P_out(=1) · rank … · F
                    y_i = lax.dynamic_index_in_dim(y_full, i, axis=0, keepdims=False)
                    return y_i

                return diff(repl)(yi)  # dim × rank … × F
            elif var == "v":
                if _v is None:
                    raise ValueError("same-particle v-derivative requires v")
                yi = lax.dynamic_index_in_dim(_v, i, axis=particles_axis, keepdims=False)  # (dim,)

                def repl(y_local):
                    v_mod = lax.dynamic_update_index_in_dim(_v, y_local, i, axis=particles_axis)
                    y_full = g(_x, v=v_mod, params=params, **g_kwargs)
                    y_i = lax.dynamic_index_in_dim(y_full, i, axis=0, keepdims=False)
                    return y_i

                return diff(repl)(yi)  # dim × rank … × F
            else:
                raise ValueError("var must be 'x' or 'v' for same-particle")

        # use int32 indices for JAX gather niceness
        return jax.vmap(inner)(jnp.arange(P, dtype=jnp.int32))  # P × dim × rank … × F

    # vmap over ALL batch axes (k nested vmaps)
    def vmap_n(f, n, has_v):
        gfun = f
        for _ in range(n):
            gfun = jax.vmap(gfun, in_axes=(0, 0 if has_v else None))
        return gfun

    vmapped = vmap_n(per_batch, len(batch_shape), v is not None)
    return vmapped(x, v)


def _move_deriv_axis(
    arr,
    *,
    rank: int,
    pdepth_old: int,
    particles_input: bool,
    same_particle: bool,
):
    """
    Re-arrange the raw Jacobian from JAX to the contract:

        output (full cross, only if particles_input=True):
            batch · [P_in]^1 · [P_out]^{pdepth_old} · dim · rank · F
            and the node's new pdepth is (pdepth_old + 1)

        output (same-particle):
            batch · [P_out]^{pdepth_old} · dim · rank · F
            (no new particle axis; pdepth unchanged)

    Parameters
    ----------
    arr : jnp.ndarray           # output of jacfwd / jacrev
    rank : int                  # number of spatial rank axes in the child
    pdepth_old : int            # child's output particle-depth
    particles_input : bool      # whether inputs have a P axis
    same_particle : bool        # if True, produce the same-particle layout
    """
    nd = arr.ndim
    has_pin = particles_input and (not same_particle)

    # Raw jacfwd layout assumption:
    #   B · P_out^{pdepth_old} · rank · F · [P_in?] · dim
    n_tail = rank + 1 + (1 if has_pin else 0) + 1  # rank + F + P_in? + dim
    B = nd - (pdepth_old + n_tail)

    idx_batch = list(range(0, B))
    idx_pout = list(range(B, B + pdepth_old))
    idx_rank = list(range(B + pdepth_old, B + pdepth_old + rank))
    idx_feat = [B + pdepth_old + rank]
    idx_pin = [B + pdepth_old + rank + 1] if has_pin else []
    idx_dim = [nd - 1]

    if has_pin:
        # Target:  B · P_in · P_out^{pdepth_old} · dim · rank · F
        order = idx_batch + idx_pin + idx_pout + idx_dim + idx_rank + idx_feat
    else:
        # Target:  B · P_out^{pdepth_old} · dim · rank · F
        order = idx_batch + idx_pout + idx_dim + idx_rank + idx_feat

    return jnp.transpose(arr, order)
