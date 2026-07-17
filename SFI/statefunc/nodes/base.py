from typing import Any, Callable

import equinox as eqx
from jaxtyping import Array

from ..memhint import MemHint, SampleMeta, default_leaf_hint, default_op_hint, resolve_P
from ..params import ParamSuite
from .contract import _ContractMixin


# ---------------------------------------------------------------------------
# Abstract node – every dictionary component subclasses this
# ---------------------------------------------------------------------------
class BaseNode(_ContractMixin):
    """
    Single node in the immutable dictionary tree.

    Required methods
    ----------------
    ``__call__(x, *, params=None, v=None, mask=None, extras=None) -> Array``
        Returns ``(..., *rank_axes, n_features)``.

    ``flatten() -> (funcs, labels, descriptors)``
        Each func follows the same keyword-only signature as above.

    ``_tree_id() -> int``
        Optional hint for caching/equality.

    Notes
    -----
    - ``params`` is a ``dict[str, Array]`` for PSF/SF; Basis nodes ignore it.
    - ``extras`` is a plain mapping passed per call.  Composite nodes *union*
      children's ``extras_required`` (presence-only).
    - ``particle_extras``: ``tuple[str, ...]`` -- pure metadata.  Names of
      extras expected to be per-particle at dispatch time.

    """

    # ---------------- static fields ----------------
    extras_required: tuple[str, ...] = eqx.field(static=True, default=())

    def __call__(self, x: Array, *, params=None, v=None, mask=None, extras=None) -> Array:
        raise NotImplementedError(f"{type(self).__name__} must implement __call__")

    def flatten(
        self,
    ) -> tuple[list[Callable], list[str], list[Any]]:  # pragma: no cover
        """
        Return three parallel lists:
            - callables     (each f(x, *, v=None, mask=None, extras=None, params=None) -> Array)
            - labels        (human-readable strings)
            - descriptors   (arbitrary metadata, JSON-serialisable)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement flatten")

    # Optional fast-path identity used by caching layers
    def _tree_id(self) -> int:  # pragma: no cover
        return id(self)

    # ---------------- memory hint API ----------------
    def memory_hint(
        self,
        *,
        dtype=None,  # default float32 when None
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> MemHint:
        """
        Default conservative rule for LEAF-style nodes: output buffer only.

        Composite ops override this and add their children.
        ``particle_size`` or ``sample.P`` lets callers specify P when this node
        PRODUCES particle axes. If both are None, P=1 is assumed.
        """
        P = resolve_P(particle_size, sample)
        return default_leaf_hint(self, dtype=dtype, particle_size=P, mode=mode)

    # Friendly shortcut for callers that need just the number
    def estimate_bytes_per_sample(
        self,
        *,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> int:
        """Return ONLY the transient bytes per single sample."""
        return self.memory_hint(dtype=dtype, particle_size=particle_size, sample=sample, mode=mode).per_sample_bytes

    # ─────────── same-particle diagonal Jacobian protocol ─────────────
    def _same_particle_jacobian(
        self,
        x: Array,
        *,
        var: str = "x",
        v=None,
        params=None,
        mask=None,
        extras=None,
    ) -> Array:
        """Same-particle diagonal Jacobian ∂f_p/∂x_p (or ∂f_p/∂v_p).

        Returns array in **contract layout**: ``(P, d_jac, rank..., F)``.

        This is the generic fallback using per-particle AD; cost is O(P²·d).
        Subclasses override for efficiency:

        - :class:`BaseOpNode` propagates via chain rule through ``_op``.
        - :class:`InteractionDispatcher` uses per-edge ``jacfwd`` + scatter: O(M·d²).
        """
        import jax

        from .ops.derivative import _move_deriv_axis, _same_particle_grad

        J_raw = _same_particle_grad(self, jax.jacfwd, x, v, params, var=var, mask=mask, extras=extras)
        return _move_deriv_axis(
            J_raw,
            rank=int(self.rank),
            pdepth_old=int(self.pdepth),
            particles_input=bool(self.particles_input),
            same_particle=True,
        )


# ---------------------------------------------------------------------------
# Generic composite node – all operators with ≥1 child inherit from here
# ---------------------------------------------------------------------------
class BaseOpNode(BaseNode):
    """
    Generic **composite** backend node.

    Unifies boilerplate that every multi-child operator needs:

    - store/validate ``children``
    - union ``extras_required``
    - union ``particle_extras``
    - merge *static contract* via ``_merge_static()``
    - OR-reduce ``needs_v``
    - single runtime ``__call__`` that evaluates children once and
      delegates combination logic to ``_op(outputs, *, params)``
    - default ``flatten()`` that splices children sequentially
    - deterministic ``._tree_id()`` for JIT-cache keys

    Sub-classes **must** implement two hooks:

    ``_merge_static(children) -> dict``
        Return at least the keys ``rank, dim, pdepth, n_features, needs_v``.

    ``_op(outputs, *, params) -> jnp.ndarray``
        Combine child outputs into the final tensor.

    Notes
    -----
    Extras policy: composite nodes **union** children's ``extras_required``
    at construction (presence-only).  Shapes/broadcast are not validated here.

    """

    children: tuple[BaseNode, ...] = eqx.field(static=True)
    # Which merge mode to use for this composite node.
    # Subclasses must set one of: 'concat' | 'map' | 'elementwise' | 'einsum'
    CONTRACT_MODE: str | None = None

    # ------------------------------------------------------------------
    def __init__(self, *children: BaseNode):
        if not children:
            raise ValueError(f"{type(self).__name__}() needs at least one child")
        object.__setattr__(self, "children", tuple(children))

        # ---------------- merge static contract -----------------------
        contract = self._merge_static(self.children)
        required = {"rank", "dim", "pdepth", "n_features", "needs_v", "particles_input"}
        if missing := required - contract.keys():
            raise RuntimeError(f"{type(self).__name__}._merge_static() missing {missing}")
        for k, v in contract.items():
            object.__setattr__(self, k, v)

        # --------------- union extras + param template ----------------
        # Presence-only: union of children requirements; ignore wildcards.
        parts: list[set[str]] = []
        for c in self.children:
            ks = getattr(c, "extras_required", ())
            # Normalize: tuples/strings/iterables → iterate, drop wildcard '*'
            if ks == "*":
                continue
            if isinstance(ks, str):
                parts.append({ks})
                continue
            try:
                s = {k for k in ks if k != "*"}
            except TypeError:
                # Non-iterable fallback (shouldn't happen): ignore
                s = set()
            if s:
                parts.append(s)
        req = set().union(*parts) if parts else set()
        object.__setattr__(self, "extras_required", tuple(sorted(req)))

        # Union of particles_extra:
        decl: set[str] = set()
        for c in self.children:
            pe = getattr(c, "particle_extras", ())
            if pe:
                decl |= set(pe)
        object.__setattr__(self, "particle_extras", tuple(sorted(decl)))

        suite = ParamSuite.merge_many(*(ch.param_suite for ch in self.children))
        object.__setattr__(self, "param_suite", suite)

    # ------------------------------------------------------------------
    # default runtime call
    # ------------------------------------------------------------------
    def __call__(self, x, *, params=None, v=None, mask=None, extras=None):
        self._assert_inputs(x, v, mask, extras)
        outs = [c(x, params=params, v=v, mask=mask, extras=extras) for c in self.children]
        y = self._op(outs, params=params)
        self._assert_outputs(x, y)
        return y

    def _merge_static(self, children):
        mode = getattr(self, "CONTRACT_MODE", None)
        if not mode:
            raise NotImplementedError(f"{type(self).__name__} must define CONTRACT_MODE")
        spec = getattr(self, "spec", None) if mode == "einsum" else None
        return _ContractMixin.merge_contract(children, mode=mode, spec=spec)

    # default flatten: linear splice of children
    def flatten(self):
        funcs, labels, desc = [], [], []
        for ch in self.children:
            f, lab, d = ch.flatten()
            funcs.extend(f)
            labels.extend(lab)
            desc.extend(d)
        return funcs, labels, desc

    def _tree_id(self):
        return hash((type(self).__name__, tuple(ch._tree_id() for ch in self.children)))

    # interface stubs ---------------------------------------------------
    def _op(self, outs, *, params):  # noqa: D401
        raise NotImplementedError(f"{type(self).__name__} must implement _op")

    # ------------------------------------------------------------------
    # graph rewriting (see StateExpr.specialize)
    # ------------------------------------------------------------------
    def with_children(self, new_children):
        """Rebuild this composite with ``new_children``, recomputing all derived
        static state (contract, extras union, parameter suite) via the
        constructor. Subclasses with extra config must override to preserve it.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.with_children is not implemented; this node "
            "type is not yet supported by StateExpr.specialize()."
        )

    # ─────────── same-particle Jacobian: chain rule through _op ───────
    def _same_particle_jacobian(
        self,
        x: Array,
        *,
        var: str = "x",
        v=None,
        params=None,
        mask=None,
        extras=None,
    ) -> Array:
        """Chain rule: propagate children's Jacobians through ``_op`` via JVP.

        For each spatial direction *j*, a single ``jax.jvp`` call through
        ``_op`` gives the *j*-th column of the composite Jacobian.
        Cost: ``d_jac`` (=2 or 3) lightweight JVP evaluations of ``_op``.
        """
        import jax
        import jax.numpy as jnp

        # Fall back if _op is not implemented (DerivativeNode, InteractionDispatcher
        # override __call__ directly and leave _op as the stub).
        if type(self)._op is BaseOpNode._op:
            return super()._same_particle_jacobian(x, var=var, v=v, params=params, mask=mask, extras=extras)

        kw = dict(v=v, params=params, mask=mask, extras=extras)
        child_vals = [c(x, **kw) for c in self.children]
        child_jacs = [c._same_particle_jacobian(x, var=var, **kw) for c in self.children]

        d_jac = child_jacs[0].shape[1]
        p = params

        def op_fn(*outs_tuple):
            return self._op(list(outs_tuple), params=p)

        primals = tuple(child_vals)
        cols = []
        for j in range(d_jac):
            tangents = tuple(cj[:, j, ...] for cj in child_jacs)
            _, y_dot = jax.jvp(op_fn, primals, tangents)
            cols.append(y_dot)

        return jnp.stack(cols, axis=1)

    # ---------------- memory hint for composite ops ----------------
    def memory_hint(
        self,
        *,
        dtype=None,  # default float32 when None
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> MemHint:
        """
        Conservative default for composite nodes.

        ``sum(children.hint) + my output buffer + broadcast overhead``.
        We assume child outputs coexist while this op constructs its result.
        """
        P = resolve_P(particle_size, sample)
        return default_op_hint(self, children=self.children, dtype=dtype, particle_size=P, mode=mode)
