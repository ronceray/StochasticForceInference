import math
from typing import Any, Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..memhint import MemHint, SampleMeta, default_leaf_hint, itemsize_of, resolve_P
from ..params import ParamSuite
from .base import BaseNode
from .contract import Rank


# -----------------------------------------------------------------------
# helper: leaf‑local fill policy (root already did a "zero" premask) -----
# -----------------------------------------------------------------------
def _apply_fill(arr: Array, mask: Optional[Array], policy: str) -> Array:
    """Optionally replace masked entries before dangerous maths.

    Gradient behaviour
    ------------------
    ``jnp.where(mask, arr, fill)`` naturally provides zero tangents for the
    masked branch in both forward and reverse mode (the fill value is a
    constant with zero tangent).  We therefore do **not** wrap the filled
    array in ``stop_gradient`` — doing so would kill the Jacobian for
    *all* entries (including active, non-masked ones), breaking
    ``jacfwd``-based derivative nodes such as ``basis.d_x()``.

    If the user-function has a singularity at the fill value (e.g. ``1/x``
    filled with 0), use ``fill_policy='eps'`` instead.
    """
    if mask is None or policy == "zero":
        return arr

    # Guard: squeeze trailing singleton dims from mask so that it matches
    # arr's non-spatial prefix.  This prevents `mask[..., None]` from
    # broadcasting *into* arr and inflating its shape (e.g. mask (1,) with
    # arr (d,) would produce (1, d) without this squeeze).
    if hasattr(mask, "ndim"):
        prefix_ndim = arr.ndim - 1  # everything except the spatial dim
        while mask.ndim > prefix_ndim and mask.shape[-1] == 1:
            mask = mask[..., 0]

    if policy == "eps":
        eps = jnp.finfo(arr.dtype).eps
        return jnp.where(mask[..., None], arr, eps)
    if policy == "zerostop":
        return jnp.where(mask[..., None], arr, 0)
    if policy == "nanstop":
        return jnp.where(mask[..., None], arr, jnp.nan)
    if callable(policy):
        return policy(arr, mask)
    raise ValueError(f"Unknown fill policy {policy!r}")


# ---------------------------------------------------------------------------
# Generic primitive leaf  –  deterministic *or* parametric
# ---------------------------------------------------------------------------
class BaseLeaf(BaseNode):
    """
    One callable that returns **n_features** stacked on the last axis.

    Parameters
    ----------
    func        : Callable
        Signature may include any subset of

            (x, v=None, mask=None, extras=None, params=None)

        - in any capitalisation.  ``params`` is omitted if
        ``param_suite is None`` (deterministic leaf).
    param_suite  : ParamSuite | None
        * ``None``                   → deterministic dictionary element
        * ParamSuite                 → parameters are required at call time
    rank, dim, pdepth, n_features, needs_v
        Static contract.
    labels      : tuple[str, ...]    - human-readable titles, length == n_features
    descriptor  : Any                - optional JSON-serialisable metadata
    fill_policy : "zero" | "eps" | "zerostop" | "nanstop" | Callable
    extras_keys : tuple[str, ...]    - required keys in the `extras` mapping

    Mask handling
    -------------
    ``fill_policy`` controls how masked entries are preprocessed **before**
    calling your function:

    * ``"zero"``     : no-op (pass inputs through unchanged).
    * ``"eps"``      : replace masked entries by machine epsilon of the dtype (safe for logs/div).
    * ``"zerostop"`` : replace masked entries by 0 (fill value is a constant, so
      ``jnp.where`` gives zero tangent for masked entries).
    * ``"nanstop"``  : replace masked entries by NaN (fail-fast in JITed code; same
      zero-tangent property as ``"zerostop"``).
    * ``callable``   : custom policy ``fn(arr, mask) -> arr`` applied elementwise.

    ``mask`` must broadcast to the prefix of ``x`` **including** the particle axis, i.e.
    ``batch·P`` if ``particles_input=True`` or ``batch`` otherwise.

    Extras policy
    -------------
    If a leaf declares ``extras``, values are read from the ``extras`` mapping at call time.
    All extras are treated as **global constants**: they are closed over inside the vmap and
    passed verbatim to every per-sample call. No shape-broadcasting is attempted.

    Named requirements can be declared via ``extras_keys=("alpha","beta",...)``. If the function
    declares an ``extras`` argument but no keys are specified, we only enforce **presence** of the
    mapping (no specific keys).

    """

    # ---------------- static fields ----------------
    func: Callable = eqx.field(static=True)
    param_suite: ParamSuite | None = eqx.field(static=True, default=None)
    labels: tuple[str, ...] = eqx.field(static=True, default=())
    descriptor: Any = eqx.field(static=True, default=None)
    fill_policy: str | Callable = eqx.field(static=True, default="zerostop")
    extras_keys: tuple[str, ...] = eqx.field(static=True, default=())
    particle_extras: tuple[str, ...] = eqx.field(static=True, default=())
    particles_input: bool = eqx.field(static=True, default=False)
    # Optional ``k -> StateExpr`` recipe folding this leaf at condition ``k``
    # (used by ``StateExpr.specialize``; see per_dataset_scalar/dataset_indicator).
    specialize_at: Any = eqx.field(static=True, repr=False, default=None)
    _signature: Any = eqx.field(static=True, repr=False, default=None)
    _sig_params: set[str] = eqx.field(static=True, repr=False, default_factory=set)
    _sig_param_map: dict[str, str] = eqx.field(static=True, repr=False, default_factory=dict)

    # -------------- constructor --------------------
    def __init__(
        self,
        *,
        func: Callable,
        rank: Rank,
        dim: int | None,
        n_features: int,
        pdepth: int = 0,
        particles_input: bool = False,
        needs_v: bool = False,
        param_suite=None,
        labels: tuple[str, ...] = (),
        descriptor: Any = "custom",
        fill_policy: str | Callable = "zerostop",
        extras_keys: tuple[str, ...] = (),
        particle_extras: tuple[str, ...] = (),
        specialize_at: Callable | None = None,
    ):
        # static contract
        object.__setattr__(self, "specialize_at", specialize_at)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "pdepth", pdepth)
        object.__setattr__(self, "needs_v", needs_v)
        object.__setattr__(self, "particles_input", particles_input)
        object.__setattr__(self, "n_features", int(n_features))

        # static metadata
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "param_suite", param_suite)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "fill_policy", fill_policy)
        # normalize and store extras_keys
        if extras_keys is None:
            extras_keys = ()
        elif isinstance(extras_keys, str):
            extras_keys = (extras_keys,)
        extras_keys = tuple(k for k in extras_keys if k != "*")  # no wildcard sentinel
        object.__setattr__(self, "extras_keys", extras_keys)
        object.__setattr__(self, "particle_extras", tuple(particle_extras or ()))

        if labels and len(labels) != n_features:
            raise ValueError("labels length must equal n_features")
        object.__setattr__(self, "labels", tuple(labels or [f"f{j}" for j in range(n_features)]))

        # --- signature analysis (single pass) --------------------------------
        import inspect as _inspect

        sig = _inspect.signature(func)
        object.__setattr__(self, "_signature", sig)

        # canonical-name → original-name (case-insensitive lookup)
        lower2orig = {p.name.lower(): p.name for p in sig.parameters.values()}
        object.__setattr__(self, "_sig_params", set(lower2orig))
        object.__setattr__(self, "_sig_param_map", lower2orig)

        params_by = {p.name: p for p in sig.parameters.values()}

        # needs_v → function must actually accept 'v'
        if self.needs_v and ("v" not in params_by):
            raise ValueError(f"{type(self).__name__}: needs_v=True but the function does not declare 'v'.")

        # extras_keys → function must actually accept 'extras'
        if self.extras_keys and ("extras" not in params_by):
            raise ValueError(f"{type(self).__name__}: extras_keys provided but the function has no 'extras' parameter.")

        # ParamLeaf must accept 'params' if it actually has parameters
        has_params_kw = "params" in params_by
        has_nonempty_suite = (self.param_suite is not None) and (getattr(self.param_suite, "size", 0) > 0)
        if has_nonempty_suite and (not has_params_kw):
            raise ValueError(
                f"{type(self).__name__}: function must declare a 'params' keyword to build a non-empty ParamLeaf."
            )

        # extras presence policy
        has_extras_param = "extras" in params_by
        if has_extras_param:
            required = tuple(self.extras_keys) if self.extras_keys else ()
        else:
            required = ()
        object.__setattr__(self, "extras_required", required)

        # ---- static contract sanity ----
        if not isinstance(self.pdepth, int) or self.pdepth < 0:
            raise ValueError(f"{type(self).__name__}: pdepth must be a non-negative int, got {self.pdepth!r}")
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError(f"{type(self).__name__}: rank must be a non-negative int, got {self.rank!r}")
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError(f"{type(self).__name__}: n_features must be a positive int, got {self.n_features!r}")
        if (self.dim is not None) and (not isinstance(self.dim, int) or self.dim < 1):
            raise ValueError(f"{type(self).__name__}: dim must be None or a positive int, got {self.dim!r}")
        if (not self.particles_input) and (self.pdepth > 0):
            raise ValueError(
                f"{type(self).__name__}: pdepth>0 requires particles_input=True "
                "(cannot create particle axes without a particle input axis)."
            )

    # -------------- call ---------------------------
    def __call__(self, x, *, params=None, v=None, mask=None, extras=None):
        self._assert_inputs(x, v, mask, extras)
        x_safe = _apply_fill(x, mask, self.fill_policy)

        y = self._apply_user_func(x_safe, v=v, mask=mask, extras=extras, params=params)

        self._assert_outputs(x, y)
        return y

    # -----------------------------------------------------------------
    #  Canonicalise shapes, batch-vmap once, call user func
    # -----------------------------------------------------------------
    def _apply_user_func(
        self,
        x,
        *,
        v=None,
        mask=None,
        extras=None,
        params=None,
    ):
        """
        Ensure the user-provided function sees a single sample:
          - x: (dim,)            or (P, dim) if particles_input=True
          - v: same shape as x   (only if the function declares 'v')
          - mask: scalar or (P,) (only if the function declares 'mask')
          - extras: per-sample dict (only if the function declares 'extras')
          - params: dict (if the function declares 'params', even for BasisLeaf)

        Batching over any leading axes is handled via a single vmap.
        """
        import jax.numpy as jnp

        # ---------- 1. Expected inner shapes ----------
        inner_ndim = 1 + int(self.particles_input)  # dim (+P)
        if x.ndim < inner_ndim:
            if self.dim == 1 and x.ndim == inner_ndim - 1:
                x = jnp.expand_dims(x, axis=-1)
                if v is not None:
                    v = jnp.expand_dims(v, axis=-1)
            else:
                raise ValueError(
                    f"{type(self).__name__}: input x.ndim={x.ndim} too small for particles_input={self.particles_input}"
                )

        batch_shape = x.shape[:-inner_ndim]  # tuple of batch dims
        inner_shape = x.shape[-inner_ndim:]
        flat_B = int(math.prod(batch_shape)) if batch_shape else 1

        # ---------- 2. Flatten batch ----------
        def _flatten(arr):
            return None if arr is None else arr.reshape((flat_B,) + inner_shape)

        x_f = _flatten(x)
        v_f = _flatten(v)
        m_f = None
        if mask is not None:
            # Let mask carry only its own dims; library already checked broadcast.
            # We broadcast mask over the full batch prefix and then flatten.
            mask_arr = jnp.asarray(mask)
            # Pad to at least len(batch_shape) dims
            pad = max(0, len(batch_shape) - mask_arr.ndim)
            mask_arr = jnp.reshape(mask_arr, (1,) * pad + mask_arr.shape)
            # Broadcast the first len(batch_shape) dims to batch_shape (remaining dims unchanged)
            target = tuple(batch_shape) + mask_arr.shape[len(batch_shape) :]
            mask_arr = jnp.broadcast_to(mask_arr, target)
            m_f = mask_arr.reshape((flat_B,) + mask_arr.shape[len(batch_shape) :])

        # ---------- 3. Prepare extras ------------
        # Policy:
        #   - extras are GLOBAL constants (closed over), not vmapped — except
        #     keys declared in ``particle_extras`` whose leading axes match
        #     the batch (the dispatcher's per-edge gathered arrays): those
        #     are vmapped alongside x so the user function sees its own
        #     ``(K, ...)`` slice per sample.
        e_f = None
        e_mapped = None
        if "extras" in self._sig_params:
            if extras is None:
                e_f = None
            else:
                # If keys were declared, keep those; otherwise pass the whole mapping.
                keys = self.extras_keys if getattr(self, "extras_keys", ()) else tuple(extras.keys())
                e_all = {k: extras[k] for k in keys}
                pkeys = set(getattr(self, "particle_extras", ()) or ())
                nb = len(batch_shape)

                def _as_mapped(val):
                    """Reshape a declared per-sample extra to (flat_B, ...).

                    Accepts a full batch-prefix match, or a trailing-axes
                    match (e.g. a per-particle ``(N, ...)`` array under a
                    batched ``(K, N)`` evaluation), broadcast over the
                    leading batch dims.  Returns None when the value does
                    not align (treated as a constant).
                    """
                    if not hasattr(val, "shape") or nb == 0:
                        return None
                    shape = tuple(val.shape)
                    if shape[:nb] == tuple(batch_shape):
                        return val.reshape((flat_B,) + shape[nb:])
                    for k_off in range(1, nb):
                        lead = nb - k_off
                        if shape[:lead] == tuple(batch_shape[k_off:]):
                            rest = shape[lead:]
                            tgt = tuple(batch_shape) + rest
                            return jnp.broadcast_to(val, tgt).reshape((flat_B,) + rest)
                    return None

                e_mapped = {}
                for k in pkeys & set(e_all):
                    mapped = _as_mapped(e_all[k])
                    if mapped is not None:
                        e_mapped[k] = mapped
                e_f = {k: v for k, v in e_all.items() if k not in e_mapped}
                if not e_mapped:
                    e_mapped = None

        # ---------- 4. Batched caller ----------

        def _caller(x_, v_, mask_, e_m):
            kwargs = {}
            if "v" in self._sig_params:
                kwargs[self._sig_param_map["v"]] = v_
            if "mask" in self._sig_params:
                kwargs[self._sig_param_map["mask"]] = mask_
            if "extras" in self._sig_params:
                ex = e_f if e_m is None else {**(e_f or {}), **e_m}
                kwargs[self._sig_param_map["extras"]] = ex
            if "params" in self._sig_params:
                kwargs[self._sig_param_map["params"]] = params
            return self.func(x_, **kwargs)

        in_axes = (
            0,
            0 if v_f is not None else None,
            0 if m_f is not None else None,
            0 if e_mapped is not None else None,
        )
        y_f = jax.vmap(_caller, in_axes=in_axes)(x_f, v_f, m_f, e_mapped)

        # ---------- 5. Restore batch shape ----------
        y = y_f.reshape(batch_shape + y_f.shape[1:])

        # ---------- 6. Contract check (strict pdepth) + auto feature axis ----------
        trail = y_f.shape[1:]  # excluding the vmapped axis
        P = inner_shape[0] if (self.particles_input and x.ndim >= 2) else None

        particle_block = (P,) * int(self.pdepth) if self.pdepth else ()
        rank_block = (self.dim,) * int(self.rank) if int(self.rank) else ()
        base_suffix = particle_block + rank_block

        if trail == base_suffix + (self.n_features,):
            return y
        if self.n_features == 1 and trail == base_suffix:
            return y[..., None]  # auto-insert singleton feature axis

        raise ValueError(
            f"[{type(self).__name__}] output shape mismatch.\n"
            f"  expected suffix : {base_suffix + (self.n_features,)}\n"
            f"  got             : {trail}"
        )

    # -------------- flatten ------------------------
    def flatten(self):
        funcs = [
            lambda _x, *, v=None, mask=None, extras=None, params=None, _j=j:  # noqa: E731
            self(_x, v=v, mask=mask, extras=extras, params=params)[..., _j]
            for j in range(self.n_features)
        ]
        descs = [self.descriptor] * self.n_features
        return funcs, list(self.labels), descs


# ---------------------------------------------------------------------------
# Leaf aliases for intelligible construction
# ---------------------------------------------------------------------------
class SimpleLeaf(BaseLeaf):
    """
    Standard leaf: never reads or produces a particle axis.
    Treats any particle dimension in `x` as batch and vmaps over it.
    Output suffix is (``*rank``, n_features) with feature last.
    """

    def __init__(self, **kwargs):
        if kwargs.get("particles_input", False):
            raise ValueError("SimpleLeaf forbids particles_input=True (it never sees P).")
        if kwargs.get("pdepth", 0) != 0:
            raise ValueError("SimpleLeaf requires pdepth=0 (it never produces a particle axis).")
        kwargs["particles_input"] = False
        kwargs["pdepth"] = 0
        super().__init__(**kwargs)


class InteractionLeaf(BaseLeaf):
    """
    Local interaction leaf: **consumes** a `(K, dim)` particle slice and returns
    feature-last output with **no** particle axis (pdepth=0).

    Modes
    -----
    - mode="fixed", K=int           → assert `x.shape[-2] == K`
    - mode="variable", Kmax=int     → assert `x.shape[-2] == Kmax`, pass (optional)
                                      1D mask of length Kmax to the user func.

    Extras semantics
    ----------------
    This leaf supports explicit separation of extras kinds (declared at construction time):
      - ``global_extras``   : keys whose values are **global constants** for the call
                              (e.g., PBC box). They are passed through verbatim.
      - ``particle_extras`` : keys whose values are **per-particle** arrays (shape ``(P, ...)``).
                              The dispatcher is responsible for **gathering** these into the
                              local K-order before calling the leaf. The leaf sees per-sample
                              payload only (no batch axes).
    """

    mode: Literal["fixed", "variable"] = eqx.field(static=True)
    K: int | None = eqx.field(static=True, default=None)
    Kmax: int | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        *,
        mode: Literal["fixed", "variable"],
        K: int | None = None,
        Kmax: int | None = None,
        func,
        dim: int,
        rank: "Rank",
        n_features: int = 1,
        needs_v: bool = False,
        param_suite=None,
        labels: tuple[str, ...] = (),
        descriptor=None,
        fill_policy: str | Callable = "zero",
        extras_keys: tuple[str, ...] = (),
        particle_extras: tuple[str, ...] = (),
    ):
        if mode == "fixed":
            if K is None:
                raise ValueError("InteractionLeaf(mode='fixed') requires K.")
        elif mode == "variable":
            if Kmax is None:
                raise ValueError("InteractionLeaf(mode='variable') requires Kmax.")
            # A safer default for ragged padding
            if fill_policy == "zero":
                fill_policy = "zerostop"
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        super().__init__(
            func=func,
            rank=rank,
            dim=dim,
            n_features=n_features,
            pdepth=0,  # never *produces* a particle axis
            particles_input=True,  # but it *consumes* (K?, dim)
            needs_v=needs_v,
            param_suite=param_suite,
            labels=labels,
            descriptor=descriptor,
            fill_policy=fill_policy,
            extras_keys=extras_keys,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "K", None if mode != "fixed" else int(K))
        object.__setattr__(self, "Kmax", None if mode != "variable" else int(Kmax))
        # NOTE: must be set *after* super().__init__, which resets the
        # base-class default to ().
        if particle_extras is None:
            particle_extras = ()
        object.__setattr__(self, "particle_extras", tuple(particle_extras))

    def __call__(self, x: Array, *, params=None, v=None, mask=None, extras=None) -> Array:
        if x.ndim < 2:
            raise ValueError(f"{type(self).__name__} expects (..., K, dim); got {x.shape}")
        Kin = x.shape[-2]
        if self.mode == "fixed":
            if Kin != self.K:
                raise ValueError(f"{type(self).__name__} expected K={self.K}, got {Kin}")
        else:  # variable
            if Kin != self.Kmax:
                raise ValueError(f"{type(self).__name__} expected Kmax={self.Kmax}, got {Kin}")
            # If the user func declared `mask`, BaseLeaf will route it; fill_policy applies.
        return super().__call__(x, params=params, v=v, mask=mask, extras=extras)

    # Interaction leaves often need a small live working set of the gathered neighborhood.
    # Count it conservatively as K*dim elements in addition to the output buffer.
    def memory_hint(
        self,
        *,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> MemHint:
        P = resolve_P(particle_size, sample)
        base = default_leaf_hint(self, dtype=dtype, particle_size=P, mode=mode)
        dim = int(getattr(self, "dim", 0) or 0)
        Kwork = self.K if self.mode == "fixed" else self.Kmax
        wset = (int(Kwork) * dim * itemsize_of(dtype)) if (Kwork is not None and dim) else 0
        return MemHint(
            per_sample_bytes=base.per_sample_bytes + wset,
            persistent_bytes=base.persistent_bytes,
        )
