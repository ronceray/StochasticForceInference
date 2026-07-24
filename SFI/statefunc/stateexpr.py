"""High-level immutable façade over backend node trees.

This module exposes three public classes:

- ``Basis``  -- deterministic dictionary of features (no parameters),
- ``PSF``    -- parametric state-function family ``F(x; theta)``,
- ``SF``     -- bound state function with ``theta`` fixed.

They all share the **StateExpr** algebra (broadcasting, linear ops, feature
product/concatenation rules, differentiation builders).

---------------------------------------------------------------------------
Shape conventions (runtime evaluation)
---------------------------------------------------------------------------
Users call `Basis/PSF/SF` on single inputs or batched arrays; the library
handles batching and vectorization internally. Let `x` be the runtime input:

- If `particles_input=False`:      `x.shape == batch · dim`
- If `particles_input=True`:       `x.shape == batch · P · dim`

Single inputs have batch=(,). Outputs always end with the **feature axis**
(length `n_features`):

    y.shape == batch · [P]^pdepth · (dim)^rank · n_features

`pdepth` is strict: outputs have *exactly* `pdepth` particle axes.
If `particles_input=False`, then `pdepth` must be 0 (no particle axes can be
created without a particle input axis). Any particle axis is simply treated as
batch in that case.

---------------------------------------------------------------------------
User function contract (single–sample)
---------------------------------------------------------------------------
Factories (`make_basis`, `make_psf`) accept *single-sample* callables. Your
function never sees batch axes; it receives:

- `x`: `(dim,)` or `(P, dim)` if `particles_input=True`
- Optional keywords: any subset of `{v, mask, extras, params}` that you declare
  in the signature. We only pass those you declare.

Return shape (no batch axis):

    (P,)*pdepth + (dim,)*rank + (n_features?,)

If `n_features==1` you may omit the last axis; we insert a singleton feature
axis automatically.

``mask`` semantics: must broadcast to the prefix of ``x`` **including** the particle
axis (i.e. ``batch * P`` when present). Numeric or boolean masks are accepted.

``extras`` semantics: Extras are pass-through data for user functions. The expression
enforces **presence only**:

- If a leaf declares ``extras_keys=("a","b",...)``, those keys must be present
  in the ``extras`` mapping at call time.
- If a leaf declares ``extras`` but no keys, only the *presence* of a mapping
  is required; no keys are enforced.

No shape/broadcasting of extras is performed by the expression. Three kinds
exist by *declaration* (never by shape):

- global -- default; forwarded unchanged to user functions;
- particle -- declared only by interaction leaves; gathered by the
  **dispatcher** per edge and then forwarded downstream as globals;
- structural -- rule/dispatcher-owned (e.g., CSR arrays); never forwarded.

---------------------------------------------------------------------------
JAX use & autodiff
---------------------------------------------------------------------------
All computations are JAX-friendly. Write user functions with ``jax.numpy``.
Expressions compose under ``jit``/``vmap``, and support automatic
differentiation:

- ``.d_x()`` adds a spatial derivative axis to the rank block, and adds a
  particle axis when ``particles_input=True`` (coming from JAX; we only
  permute/reshape), **unless pdepth=1 and same_particle is True**.
- ``.d_v()`` similarly (if ``needs_v=True``).
- ``.d_theta()`` (PSF only) returns a Jacobian with the feature axis fused
  with parameters.

Internally, derivative axis ordering is canonicalized by permutation only;
we **never create particle axes** ourselves—if `particles_input=True`, extra
particle axes in a Jacobian come from JAX itself.

"""

import string
from typing import Any, Callable, Sequence

import equinox as eqx
import jax.numpy as jnp

from .core.runtime import _JIT_ENABLED, _eager_eval, _jitted_eval
from .memhint import SampleMeta
from .nodes import (
    BaseNode,
    ConcatNode,
    DenseNode,
    DerivativeNode,
    EinsumNode,
    MapNNode,
    Rank,
    ReshapeRankNode,
    SimpleLeaf,
    SliceFeaturesNode,
)

# 26 unique letters for einsum spatial indices.
_EINSUM_LETTERS: str = string.ascii_lowercase


# ---------------------------------------------------------------------
#  StateExpr  –  public façade class
# ---------------------------------------------------------------------
class StateExpr(eqx.Module):
    """Immutable *state expression* backed by a static node tree.

    Think: a read-only NumPy array whose **last axis is features**. Every algebraic
    operation returns a **new** ``StateExpr`` (functional style), and static contract
    metadata (``rank``, ``dim``, ``pdepth``, ``n_features``, ``needs_v``, ``particles_input``)
    is validated at graph-construction time.

    Runtime shapes
    --------------
    Inputs are **batched** at call time; the library handles batching.

    * If ``particles_input=False``:   ``x.shape == batch · dim``
    * If ``particles_input=True``:    ``x.shape == batch · P · dim``

    Outputs always end with the **feature axis** (length ``n_features``)::

        y.shape == batch · [P]^pdepth · (dim)^rank · n_features

    If ``particles_input=False``, ``pdepth`` must be ``0``.

    User function contract (single-sample)
    ---------------------------------------
    Factories accept *single-sample* callables; user code never sees batch axes.
    Your function gets ``x`` of shape ``(dim,)`` (or ``(P, dim)`` if ``particles_input``) and
    any subset of keyword-only args it declares: ``{v, mask, extras, params}``.
    Return shape (no batch axis): ``(P,)*pdepth + (dim,)*rank + (n_features?,)``.
    If ``n_features==1``, you may omit the last axis; a singleton is inserted.

    * ``mask`` must broadcast to the prefix of ``x`` **including** the particle axis.
    * ``extras`` presence: if a leaf declares ``extras`` with no explicit keys, presence
      is required (any dict). If ``extras_keys`` is given, those keys are required.
      Values may be scalars or arrays that broadcast over **batch only**.

    Operators
    ---------
    **Element-wise arithmetic**

    * ``+  -  *  /``  -- element-wise on spatial axes; **features must match**.
      Scalars and 1-D vectors (length ``n_features``) broadcast along features.
    * Unary: ``+expr``, ``-expr``.
    * NumPy/JAX ufuncs: ``sin``, ``exp``, etc. forward to element-wise maps with the
      same broadcasting rules; binary ufuncs accept ``StateExpr ∘ const`` and
      ``StateExpr ∘ StateExpr`` (features must match for the latter).

    **Linear-algebra-like**

    * ``@`` (matmul): true matrix multiplication on spatial axes,
      ``(..., m, k) @ (..., k, n) -> (..., m, n)``; **features form a Cartesian
      product** between operands (result features = ``F_left × F_right``).
    * ``.einsum(*others, spec=...)``: generic spatial contraction; **features take a
      Cartesian product across all operands** (no implicit feature reduction).
    * ``.dot(other)``: Spatial inner product between last rank axis of self and first
      rank axis of other. Cartesian product over features.
    * ``.sqrtm()``: matrix square root per-feature; requires ``rank==2``.

    **Feature-axis manipulation**

    * ``expr1 & expr2``  /  ``StateExpr.stack([...])``: **concatenate features**. Static
      spatial contracts must match; labels (if present) are concatenated.
    * ``expr[idx]``: feature selection (slice/list/bool/int). Spatial contract is
      unchanged; labels are subset when available.
    * ``.elementwisemap(func, label_fn=None)``: apply a scalar-to-scalar map to each
      feature independently (spatial axes untouched). Optional ``label_fn`` updates
      labels for ``Basis``.

    Differentiation builders
    ------------------------
    All builders return **new expressions** (no evaluation).

    * ``.d_x(same_particle=False, mode='auto')``  -- spatial Jacobian dF/dx.

      * Adds **one derivative-dim axis** immediately **before** the rank block.
      * If ``particles_input=True``:

        - when ``same_particle=False`` (default), builds the full cross-particle
          Jacobian df_i/dx_j and a second particle axis appears (from JAX);
        - when ``same_particle=True`` and ``pdepth=1``, computes the same-particle
          Jacobian df_i/dx_i without adding a new particle axis; otherwise an
          error is raised.

    * ``.d_v(same_particle=False, mode='auto')``  -- velocity Jacobian dF/dv
      (requires ``needs_v=True``). Same axis rules as ``.d_x()``.
    * ``.d_theta(mode='auto')`` -- Jacobian w.r.t. parameters (PSF only); the final axis becomes
      ``features × n_params_total``. Batch/particle/rank prefixes are preserved.

    Type mixing and broadcasting
    ----------------------------
    * Scalars and ndarrays are treated as **purely spatial constants**: they must
      be broadcastable to the spatial rank block ``(dim,)*rank`` and are then
      broadcast uniformly across the feature axis. Bare arrays cannot target the
      feature axis directly.
    * Combining two ``StateExpr`` requires matching static contracts for ``rank``,
      ``dim``, and ``pdepth``.
    * For element-wise ops such as ``+``, ``-`` and most binary ufuncs, ``n_features``
      must match (per-feature operations).
    * For multiplicative ops (``*``, ``/`` and their ufuncs), as well as ``@`` and
      ``.einsum``, feature axes take a **Cartesian product** between operands:
      ``F_out = F_left × F_right``. When both operands have more than one feature a
      one-off warning is emitted, as this can grow ``n_features`` quickly.
    * ``needs_v`` is **OR-combined**: if any operand needs ``v``, the result does.
    * ``particles_input`` is **OR-combined**: if any operand uses particle
      input, the result does too. An operand without particle input is
      broadcast uniformly across the particle axis.

    Array interop
    -------------
    Plain JAX/NumPy arrays are accepted in binary ops with StateExpr.
    They are treated as **spatial constants** with a single feature.
    Arrays broadcast over spatial axes and batch/particles only.
    Features never arise from arrays and are never contracted unless
    requested by explicit feature-aware APIs.

    Supported operations with arrays:

    - Elementwise: ``+``, ``-``, ``*``, ``/``, ``**``, and their reflected forms.
    - Linear algebra: ``A @ B``, ``B @ A``.
    - Tensor algebra: ``einsum(eq, ...)``, ``dot(...)``, ``tensordot(...)``.

    JAX compatibility and autodiff
    ------------------------------
    Write user functions with ``jax.numpy as jnp``. Expressions compose under ``jit``
    / ``vmap``, and support automatic differentiation:

    * ``.d_x()``, ``.d_v()`` add a derivative-dim axis (and a particle axis when
      ``particles_input=True``).
    * ``.d_theta()`` fuses ``features × n_params`` on the last axis.
      Derivative axis ordering is canonicalized by permutation only.

    """

    # tell NumPy we take precedence when mixing with ndarrays
    __array_priority__ = 1_000_000

    root: "BaseNode"

    def __init__(self, root):
        object.__setattr__(self, "root", root)
        # Tree-wide static sanity: catches invalid nodes even if built manually
        for _n in _walk_nodes(root):
            _static_contract_sanity_node(_n)

    def _validate_extras_presence(self, extras):
        _validate_extras_presence(self.required_extras, extras)

    # -----------------------------------------------------------------
    #  Static-contract passthrough
    # -----------------------------------------------------------------
    @property
    def rank(self):
        return self.root.rank

    @property
    def dim(self):
        return self.root.dim

    @property
    def pdepth(self):
        return self.root.pdepth

    @property
    def n_features(self):
        return self.root.n_features

    @property
    def needs_v(self):
        return self.root.needs_v

    @property
    def particles_input(self):
        return self.root.particles_input

    @property
    def sdims(self):
        return self.root.sdims

    # -------- Memory hints surfaced at the expression level --------
    def memory_hint(
        self,
        *,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ):
        """
        Conservative per-sample memory footprint for the WHOLE expression tree.
        Delegates to the root node, which sums children + own output along the way.
        """
        return self.root.memory_hint(dtype=dtype, particle_size=particle_size, sample=sample, mode=mode)

    def estimate_bytes_per_sample(
        self,
        *,
        dtype=None,
        particle_size: int | None = None,
        sample: SampleMeta | None = None,
        mode: str = "forward",
    ) -> int:
        """Small convenience wrapper returning only the transient bytes/sample."""
        return self.root.estimate_bytes_per_sample(dtype=dtype, particle_size=particle_size, sample=sample, mode=mode)

    # =================================================================
    #  INTERNAL HELPERS
    # =================================================================
    def _with_node(self, new_root: BaseNode):  # pragma: no cover
        """Dispatch to the concrete subclass constructor."""
        return type(self)(new_root)  # Basis/PSF/SF override if needed

    def specialize(self, *, dataset: int) -> "StateExpr":
        """Collapse a pooled model to its single-condition specialization.

        Returns a new expression in which every ``dataset_index``-reading
        primitive (e.g. :func:`~SFI.bases.per_dataset_scalar`,
        :func:`~SFI.bases.dataset_indicator`) is folded at condition
        ``dataset``: per-condition parameter arrays collapse to that condition's
        slice and the reserved ``dataset_index`` extra drops out of
        :attr:`required_extras`. The pooled-ness is an inference-time concern;
        once a condition is chosen the model stands alone (no dataset concept).

        On a bound :class:`~SFI.statefunc.SF` the stored parameter values are
        projected to match the shrunken template; on an unbound ``PSF`` the
        template's per-condition specs become scalars.
        """
        new_root = _specialize_node(self.root, int(dataset))
        return self._with_node(new_root)

    def _caller(self, x, v, mask, extras, params):
        if _JIT_ENABLED:
            y = _jitted_eval(self.root, x, v, mask, extras, params)
        else:
            y = _eager_eval(self.root, x, v, mask, extras, params)
        return y

    def _binary(self, other, fn: Callable[[Any, Any], Any], *, swap=False, label_fn=None):
        """Generic binary op; *other* may be StateExpr or scalar/ndarray.

        Broadcasting policy:
          - If `other` is StateExpr → element-wise MapN over both nodes (features must match).
          - If `other` is array-like (incl. scalars):
              - Treat it as **purely spatial**: it must be broadcastable to the rank block
                `(dim,)*rank`. No feature axis is allowed/assumed.
              - We then broadcast it uniformly across the **feature axis**.
        """
        if isinstance(other, StateExpr):
            left, right = (other.root, self.root) if swap else (self.root, other.root)
            node = MapNNode(lambda a, b, _fn=fn: _fn(a, b), left, right, label_fn=label_fn)
            return self._with_node(node)

        # constant path – strictly spatial
        const = jnp.asarray(other)
        if const.ndim == 0:
            const_spatial = const[..., None]  # broadcast across features uniformly
        else:
            rank = int(self.rank)
            dim = int(self.dim)
            if not _is_broadcastable_to_rank(tuple(const.shape), rank=rank, dim=dim):
                raise TypeError(
                    "Constant has incompatible shape. Allowed: scalar or a tensor "
                    f"broadcastable to the spatial rank block (dim={dim}, rank={rank}). "
                    f"Got shape={tuple(const.shape)}."
                )
            const_spatial = const[..., None]  # add feature singleton

        if swap:
            node = MapNNode(lambda a, c=const_spatial, _fn=fn: _fn(c, a), self.root)
        else:
            node = MapNNode(lambda a, c=const_spatial, _fn=fn: _fn(a, c), self.root)
        return self._with_node(node)

    def _scalar(self, fn: Callable[[Any], Any]):
        node = MapNNode(lambda a, _fn=fn: _fn(a), self.root)
        return self._with_node(node)

    # -----------------------------------------------------------------
    def __repr__(self):
        # Compact summary showing the static contract; useful in notebooks/tracebacks.
        return (
            f"{self.__class__.__name__}(rank={self.rank}, dim={self.dim}, "
            f"pdepth={self.pdepth}, n_features={self.n_features}, "
            f"needs_v={self.needs_v}, particles_input={self.particles_input})"
        )

    # -----------------------------------------------------------------
    # Gradient builders
    # -----------------------------------------------------------------
    def d_x(self, *, same_particle: bool = False, mode: str = "auto"):
        """Build an expression for the spatial Jacobian dF/dx.

        Axis effects
        ------------
        - Adds **one derivative-dim** immediately **before** the rank block.
        - If ``particles_input=True``:

          * when ``same_particle=True``:
            if pdepth=1, compute df_i/dx_i (no extra P axis); the particle
            dimension behaves like a broadcasted index.
            Otherwise, raises an error.
          * when ``same_particle=False`` (default): compute the full
            cross-particle Jacobian df_i/dx_j; **an extra particle axis
            appears** (from JAX). We never create P axes ourselves; we
            only permute to canonical order.

        Parameters
        ----------
        same_particle : bool
            See axis effects above.
        mode : {'auto', ...}
            Backend differentiation mode; 'auto' selects a sane default.

        Returns
        -------
        StateExpr
            A new expression representing the Jacobian.

        Notes
        -----
        This method triggers no evaluation; it returns a *new graph*.
        """
        node = DerivativeNode(self.root, var="x", same_particle=same_particle, mode=mode)
        return self._with_node(node)

    def d_v(self, *, same_particle: bool = False, mode: str = "auto"):
        """Build an expression for the velocity Jacobian ∂F/∂v.

        Same rules as `.d_x()`. Requires `needs_v=True` on the underlying expression.
        """
        if not self.needs_v:
            raise AttributeError("Underlying expression does not depend on v")
        node = DerivativeNode(self.root, var="v", same_particle=same_particle, mode=mode)
        return self._with_node(node)

    # =================================================================
    #  BASIC ARITHMETIC & ELEMENT-WISE OPS
    # =================================================================
    def __add__(self, o):
        return self._binary(o, lambda a, b: a + b, label_fn=lambda la, lb: f"{la}+{lb}")

    __radd__ = __add__

    def __sub__(self, o):
        return self._binary(
            o,
            lambda a, b: a - b,
            label_fn=lambda la, lb: f"{la}-({lb})" if any(c in lb for c in "+-") else f"{la}-{lb}",
        )

    def __rsub__(self, o):
        return self._binary(
            o,
            lambda a, b: a - b,
            swap=True,
            label_fn=lambda la, lb: f"{la}-({lb})" if any(c in lb for c in "+-") else f"{la}-{lb}",
        )

    def __mul__(self, o):
        """
        Multiplication.

        - If `o` is a StateExpr:
            * same-rank: spatially elementwise, features combine as a Cartesian
              product F_out = F_self × F_other (via `einsum`);
            * scalar × any-rank (either side): spatially elementwise scaling,
              still with feature Cartesian product.
        - If `o` is array-like: fall back to `_binary`, i.e. aligned features
          and spatial broadcasting.
        """
        if isinstance(o, StateExpr):
            r_self = int(self.rank)
            r_other = int(o.rank)

            def _check_rank(r: int) -> None:
                if r > len(_EINSUM_LETTERS):
                    raise ValueError(
                        f"rank={r} too large for implicit '*' spec; use .einsum() "
                        "with an explicit spatial contraction string."
                    )

            # Same-rank case
            if r_self == r_other:
                _check_rank(r_self)
                if self.n_features > 1 and o.n_features > 1:
                    _warn_cartesian_multi_feature("*")
                tok = _EINSUM_LETTERS[:r_self]
                # spatially elementwise, feature Cartesian product
                return type(self).einsum(f"{tok},{tok}->{tok}", self, o)

            # Allow scalar * any-rank when either side is scalar (rank 0)
            if r_self == 0 and r_other > 0:
                _check_rank(r_other)
                if self.n_features > 1 and o.n_features > 1:
                    _warn_cartesian_multi_feature("*")
                tok = _EINSUM_LETTERS[:r_other]
                # scalar (no spatial letters) times higher-rank
                return type(self).einsum(f",{tok}->{tok}", self, o)

            if r_other == 0 and r_self > 0:
                _check_rank(r_self)
                if self.n_features > 1 and o.n_features > 1:
                    _warn_cartesian_multi_feature("*")
                tok = _EINSUM_LETTERS[:r_self]
                # higher-rank times scalar (no spatial letters on rhs)
                return type(self).einsum(f"{tok},->{tok}", self, o)

            # Remaining mismatched-rank cases are ambiguous
            raise TypeError(
                "Multiplication between StateExprs requires matching spatial rank "
                "or one scalar rank; use .einsum() for general contractions."
            )

        # scalar / array path: old behaviour, aligned features + spatial broadcasting
        return self._binary(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        """
        Division.

        - If `o` is a StateExpr: divide via multiplication by its inverse,
          so feature axes combine as in `*`.
        - If `o` is array-like: aligned features via `_binary`.
        """
        if isinstance(o, StateExpr):
            inv = o._scalar(lambda x: 1 / x)
            return self * inv
        return self._binary(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        if isinstance(o, StateExpr):
            inv_self = self._scalar(lambda x: 1 / x)
            return o * inv_self
        # scalar / array on the left: swap arguments in `_binary`
        return self._binary(o, lambda a, b: a / b, swap=True)

    def __neg__(self):
        return self._scalar(lambda a: -a)

    def __pos__(self):
        return self

    def __pow__(self, exponent):
        """Element-wise power: ``expr ** n``.

        *exponent* must be a scalar or array-like constant (not a StateExpr).
        For StateExpr exponents, use ``jnp.power(base, exp)`` via
        ``__array_ufunc__``.
        """
        if isinstance(exponent, StateExpr):
            return self._binary(exponent, lambda a, b: a**b)
        e = jnp.asarray(exponent)
        return self._scalar(lambda a, _e=e: a**_e)

    def __rpow__(self, base):
        """Element-wise power with constant base: ``n ** expr``."""
        b = jnp.asarray(base)
        return self._scalar(lambda a, _b=b: _b**a)

    # =================================================================
    #  LINEAR-ALGEBRA-LIKE
    # =================================================================
    @classmethod
    def einsum(cls, spec: str, *operands):
        """
        General contraction on **spatial** axes (like `jnp.einsum`).

        Important
        ---------
        * Use only lowercase letters.
        * `spec` refers **only to spatial axes** (not the feature axis).
        * **Features take a Cartesian product** across operands (no implicit
          feature reduction or alignment). If you need feature concatenation,
          use `&`/`stack`. For per-feature ops, use element-wise maps or binary
          ops where features must match.

        Arrays in `operands` are accepted and coerced to spatial-constant
        expressions with a **single feature**. Only spatial letters in
        `spec` are interpreted. If no StateExpr is present, a TypeError
        is raised because `dim` cannot be inferred.

        Examples
        --------
        Vector inner product (per-feature), two rank-1 inputs:
        >>> # a, b: i × F
        >>> c = StateExpr.einsum("i,i->", a, b)    # result:  × F

        Matrix–vector product (per-feature), rank-2 with rank-1:
        >>> # M: ij × F1,  v: j × F2  →  i × (F1×F2)
        >>> y = StateExpr.einsum("ij,j->i", M, v)

        Outer product (per-feature Cartesian product):
        >>> # u: i × F1,  v: j × F2  →  ij × (F1×F2)
        >>> O = StateExpr.einsum("i,j->ij", u, v)


        Parameters
        ----------
        spec : str
            An einsum string over spatial indices, e.g. "ij,j->i".
        operands : mix[StateExpr, array-like]
            Any mix of StateExpr and arrays.
        """
        if "..." in spec:
            raise ValueError("Ellipsis '...' is not supported in einsum specs.")

        ref = next((op for op in operands if isinstance(op, StateExpr)), None)
        if ref is None:
            raise TypeError("einsum needs at least one StateExpr to infer spatial dim.")

        lhs, sep, rhs = spec.partition("->")
        terms = [t.strip() for t in lhs.split(",")]
        if len(terms) != len(operands):
            raise ValueError("einsum: number of terms and operands mismatch.")

        # Coerce arrays → spatial-constant exprs
        coerced = []
        for term, op in zip(terms, operands):
            if "..." in term:
                raise ValueError("Ellipsis is not supported in operand terms.")
            if isinstance(op, StateExpr):
                coerced.append(op)
            else:
                rank = len(term) if term else 0
                coerced.append(ref._const_expr_from_array(op, rank_override=rank))

        # Canonicalize RHS: put letters owned by a StateExpr **first** (stable), then others.
        # This matches tests that expect B.einsum("j,i->ji", A, B) == np.einsum("i,j->ij", x, A).
        if sep:
            rhs_letters = [c for c in rhs if c.isalpha()]
            se_idxs = {i for i, op in enumerate(operands) if isinstance(op, StateExpr)}

            def owner(letter: str) -> int | None:
                for k, term in enumerate(terms):
                    if letter in term:
                        return k
                return None

            rhs_se = [c for c in rhs_letters if owner(c) in se_idxs]
            rhs_cons = [c for c in rhs_letters if owner(c) not in se_idxs]
            rhs_new = "".join(rhs_se + rhs_cons)
            if rhs_new != rhs:
                spec = f"{lhs}->{rhs_new}"

        return cls._einsum_impl(spec, *coerced)

    @classmethod
    def _einsum_impl(cls, spec: str, *exprs: "StateExpr") -> "StateExpr":
        """
        Internal: build an EinsumNode from operands and the spatial einsum string.
        Features are handled in the node (Cartesian-product rule).
        """
        if not exprs:
            raise ValueError("einsum needs at least one operand")
        node = EinsumNode(*(e.root for e in exprs), spec=spec)
        return exprs[0]._with_node(node)

    def __matmul__(self, other):
        # True matrix multiplication: (..., m, k) @ (..., k, n) -> (..., m, n)
        if not isinstance(other, StateExpr):
            other = self._const_expr_from_array(other)
        ra, rb = self.rank, other.rank
        if ra < 1 or rb < 1:
            raise TypeError("matmul requires rank >= 1 for both operands.")

        aL = list(_EINSUM_LETTERS[:ra])
        bL = list(_EINSUM_LETTERS[ra : ra + rb])

        # contract A's last with B's first
        bL[0] = aL[-1]
        out = "".join(aL[:-1] + bL[1:])
        eq = f"{''.join(aL)},{''.join(bL)}->{out}"
        return type(self).einsum(eq, self, other)

    def __rmatmul__(self, other):
        # True matrix multiplication for array/expr on the left
        if not isinstance(other, StateExpr):
            other = self._const_expr_from_array(other)
        ra, rb = other.rank, self.rank
        if ra < 1 or rb < 1:
            raise TypeError("matmul requires rank >= 1 for both operands.")

        aL = list(_EINSUM_LETTERS[:ra])
        bL = list(_EINSUM_LETTERS[ra : ra + rb])

        # contract A's last with B's first
        bL[0] = aL[-1]
        out = "".join(aL[:-1] + bL[1:])
        eq = f"{''.join(aL)},{''.join(bL)}->{out}"
        return type(self).einsum(eq, other, self)

    def dot(self, other, axes=None):
        """
        Spatial tensordot via einsum.

        Semantics:
          - axes=None: contract last axis of self with first axis of other.
          - axes=int:
              * if self.rank == other.rank: contract **all** axes (Frobenius/trace for rank-2).
              * else: contract `axes` trailing axes of self with `axes` leading axes of other.
          - axes=(a_axes, b_axes): NumPy-style explicit lists.

        Arrays are accepted and coerced to spatial constants.
        """
        if not isinstance(other, StateExpr):
            other = self._const_expr_from_array(other)

        ra, rb = self.rank, other.rank

        # Normalize axes
        if axes is None:
            a_axes, b_axes = (ra - 1,), (0,)
        elif isinstance(axes, int):
            if ra == rb:
                # full Frobenius contraction when ranks match (matches tests)
                a_axes = tuple(range(ra))
                b_axes = tuple(range(rb))
            else:
                if axes < 0 or axes > min(ra, rb):
                    raise ValueError("axes out of range")
                a_axes = tuple(range(ra - axes, ra))
                b_axes = tuple(range(axes))
        else:
            a_axes = tuple(axes[0])
            b_axes = tuple(axes[1])
            if len(a_axes) != len(b_axes):
                raise ValueError("axes lengths must match")

        aL = list(_EINSUM_LETTERS[:ra])
        bL = list(_EINSUM_LETTERS[ra : ra + rb])

        # Share letters on contracted axes
        for ai, bi in zip(a_axes, b_axes):
            bL[bi] = aL[ai]

        # Output keeps non-contracted axes in order: self then other
        outA = [c for i, c in enumerate(aL) if i not in a_axes]
        outB = [c for j, c in enumerate(bL) if j not in b_axes]

        eq = f"{''.join(aL)},{''.join(bL)}->{''.join(outA + outB)}"
        return type(self).einsum(eq, self, other)

    def tensordot(self, other, axes=1):
        """Alias of .dot with NumPy-compatible `axes`."""
        return self.dot(other, axes=axes)

    def _const_expr_from_array(self, arr, *, rank_override: int | None = None):
        """
        Wrap a JAX/NumPy array as a spatial-constant StateExpr with one feature.
        Broadcasts over spatial axes and batch/particles. Feature count = 1.
        If `rank_override` is given, ignore `arr.ndim` and use that rank
        for spatial semantics (useful in einsum parsing).
        """
        const = jnp.asarray(arr)
        rank = int(rank_override) if rank_override is not None else int(const.ndim)
        dim = int(self.dim)
        if not _is_broadcastable_to_rank(tuple(const.shape), rank=rank, dim=dim):
            raise TypeError(
                "Array has incompatible shape for matmul with this expression. "
                f"Expected a tensor broadcastable to (dim={dim},)*rank; got {tuple(const.shape)}."
            )
        node = SimpleLeaf(
            func=lambda x, **kw: const,
            rank=rank,
            dim=dim,
            n_features=1,
            needs_v=False,
        )
        return self._with_node(node)

    def sqrtm(self):
        from ..utils.maths import sqrtm_psd

        if self.rank != Rank.MATRIX:
            raise ValueError("sqrtm only valid for rank-2 tensors")
        node = MapNNode(sqrtm_psd, self.root, label_fn=lambda s: f"sqrtm({s})")
        return self._with_node(node)

    # =================================================================
    #  FEATURE AXIS MANIPULATION
    # =================================================================
    def __and__(self, other: "StateExpr"):
        """Concatenate along the **feature axis**.

        Static contracts must match (rank/dim, compatible pdepth).
        """
        node = ConcatNode(self.root, other.root)
        return self._with_node(node)

    # stack helper
    @classmethod
    def stack(cls, exprs: Sequence["StateExpr"]):
        """Concatenate along the **feature axis**.

        Static contracts must match (rank/dim, compatible pdepth).
        """
        if not exprs:
            raise ValueError("stack() received empty sequence")
        node = ConcatNode(*(e.root for e in exprs))
        return exprs[0]._with_node(node)

    # feature slicing / fancy indexing
    def __getitem__(self, idx):
        """Feature selection via slices, lists, boolean masks, or integers.

        Returns a new expression with the same spatial contract; labels are
        subsetted when applicable.
        """
        node = SliceFeaturesNode(self.root, idx)
        return self._with_node(node)

    # -----------------------------------------------------------------
    #  RANK LIFTING: scalar → vector / matrix
    # -----------------------------------------------------------------
    def vectorize(self, dim: int | None = None, axes=None):
        """Lift a **scalar** expression to **rank-1 (vector)**.

        Equivalent to ``self * unit_vector_basis(dim, axes=axes)``, i.e. a
        Cartesian product of the feature axis with unit vectors.

        Parameters
        ----------
        dim : int, optional
            Spatial dimension.  Inferred from the expression's contract when
            possible.
        axes : sequence of int, optional
            Subset of spatial axes to include (default: all ``dim`` axes).

        Returns
        -------
        StateExpr
            Vector expression with ``n_features = self.n_features × len(axes)``.
        """
        from ..bases.constants import unit_vector_basis

        if self.rank != 0:
            raise TypeError(f"vectorize() requires a scalar (rank-0) expression; got rank={self.rank}")
        if getattr(self.root, "particles_input", False):
            raise TypeError(
                "vectorize() cannot be used on an undispatched Interactor. "
                "Either dispatch it first (e.g. inter.dispatch_pairs().vectorize()), "
                "or use a vector pair basis directly (e.g. radial_pair_basis)."
            )
        if dim is None:
            dim = getattr(self.root, "dim", None)
            if dim is None:
                raise ValueError("Cannot infer dim; pass it explicitly.")
        return self * unit_vector_basis(dim, axes=axes)

    def tensorize(self, dim: int | None = None, mode: str = "symmetric"):
        """Lift a **scalar** expression to **rank-2 (matrix)**.

        Parameters
        ----------
        dim : int, optional
            Spatial dimension.  Inferred when possible.
        mode : str
            ``'symmetric'`` (default) uses :func:`symmetric_matrix_basis`
            (d(d+1)/2 features per scalar feature, spans all symmetric
            matrices).  ``'identity'`` uses :func:`identity_matrix_basis`
            (1 feature per scalar feature, isotropic).

        Returns
        -------
        StateExpr
            Matrix expression.
        """
        from ..bases.constants import identity_matrix_basis, symmetric_matrix_basis

        if self.rank != 0:
            raise TypeError(f"tensorize() requires a scalar (rank-0) expression; got rank={self.rank}")
        if getattr(self.root, "particles_input", False):
            raise TypeError(
                "tensorize() cannot be used on an undispatched Interactor. "
                "Either dispatch it first (e.g. inter.dispatch_pairs().tensorize()), "
                "or use a tensor pair basis directly (e.g. dyadic_pair_basis)."
            )
        if dim is None:
            dim = getattr(self.root, "dim", None)
            if dim is None:
                raise ValueError("Cannot infer dim; pass it explicitly.")
        if mode == "symmetric":
            return self * symmetric_matrix_basis(dim)
        if mode == "identity":
            return self * identity_matrix_basis(dim)
        raise ValueError(f"Unknown mode={mode!r}; choose 'symmetric' or 'identity'.")

    # -----------------------------------------------------------------
    #  RANK ↔ FEATURE RESHAPING (lossless, invertible)
    # -----------------------------------------------------------------
    def rank_to_features(self):
        """Fold **all** spatial (rank) axes into the feature axis → rank-0.

        The output layout changes from::

            batch · (dim,)^rank · n_features

        to::

            batch · (n_features × dim^rank,)

        with rank = 0.  This is a pure reshape (no copy, no learnable
        parameters) and is the exact **inverse** of
        ``features_to_rank(original_rank)``.

        Returns
        -------
        StateExpr (same subclass)
            Scalar expression whose feature count is
            ``self.n_features × self.dim ** self.rank``.

        Raises
        ------
        TypeError
            If the expression is already rank‑0 (no-op would be confusing).

        Examples
        --------
        Prepare a rank-1 position vector for dense layers::

            >>> X(dim=2).rank_to_features()   # rank-0, 2 features

        The round-trip is the identity::

            >>> expr.rank_to_features().features_to_rank(expr.rank)  # same as expr
        """
        if self.rank == 0:
            raise TypeError("rank_to_features() on a rank-0 expression is a no-op; the features are already scalar.")
        node = ReshapeRankNode(self.root, target_rank=0)
        return self._with_node(node)

    def features_to_rank(self, rank: int):
        """Unfold features into spatial axes → given *rank*.

        The output layout changes from the current::

            batch · (dim,)^self.rank · n_features

        to::

            batch · (dim,)^rank · (n_features / dim^(rank − self.rank),)

        where the new innermost spatial axes are carved out of the
        feature axis.  This is a pure reshape and is the exact
        **inverse** of ``rank_to_features()`` when restoring the
        original rank.

        Parameters
        ----------
        rank : int
            Target tensor rank (must be greater than the current rank).

        Returns
        -------
        StateExpr (same subclass)
            Expression at the requested rank with fewer features.

        Raises
        ------
        ValueError
            If ``n_features`` is not divisible by ``dim^Δrank``.
        TypeError
            If ``rank ≤ self.rank`` (use ``rank_to_features`` to go down).

        Examples
        --------
        Turn a dense layer's output back into a vector field::

            >>> scalar_expr.features_to_rank(1)  # rank-1, F/dim features

        Build a 2→H→H→2 MLP force field::

            >>> mlp = (
            ...     X(dim=2)
            ...     .rank_to_features()                     # rank-0, 2 features
            ...     .dense(32, weight="W1", bias="b1")
            ...     .elementwisemap(jnp.tanh)
            ...     .dense(2, weight="W2", bias="b2")       # rank-0, 2 features
            ...     .features_to_rank(1)                     # rank-1, 1 feature
            ... )
        """
        if rank <= self.rank:
            raise TypeError(
                f"features_to_rank({rank}) requires rank > current rank "
                f"({self.rank}); use rank_to_features() to decrease rank."
            )
        node = ReshapeRankNode(self.root, target_rank=rank)
        return self._with_node(node)

    # -----------------------------------------------------------------
    #  ELEMENT-WISE TRANSFORM OF FEATURE AXIS
    # -----------------------------------------------------------------
    def elementwisemap(
        self,
        func: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        label_fn: Callable[[str], str] | None = None,
    ):
        """
        Apply *func* element-wise to every **feature** (spatial axes untouched).

        `func` must be a pure JAX function from scalar→scalar (rank-0 arrays OK).
        If the expression carries feature labels (e.g., a `Basis` or an `SF` bound
        from a `Basis`), `label_fn` (if provided) is applied to each feature label.

        Example
        -------
        >>> B = ...   # Basis with 4 features
        >>> C = B.elementwisemap(jnp.tanh, label_fn=lambda s: f"tanh({s})")
        """
        node = MapNNode(lambda a, _f=func: _f(a), self.root, label_fn=label_fn)
        return self._with_node(node)

    # -----------------------------------------------------------------
    #  DENSE (AFFINE) LAYER ON FEATURE AXIS
    # -----------------------------------------------------------------
    def dense(
        self,
        n_out: int,
        *,
        weight: str = "W",
        bias: str | None = "b",
    ):
        """Apply a learnable affine map on the **feature axis**.

        ``y[..., j] = sum_i x[..., i] * W[i, j] + b[j]``

        Spatial (rank) axes are untouched: the same ``W, b`` are shared
        across every spatial component. The result is always a `PSF`
        (since the dense layer introduces learnable parameters).

        Parameters
        ----------
        n_out : int
            Number of output features.
        weight : str
            Name for the weight parameter (default ``"W"``).  Use distinct
            names (``"W1"``, ``"W2"``, …) when stacking multiple layers.
        bias : str | None
            Name for the bias parameter (default ``"b"``; ``None`` to omit).
            Use distinct names (``"b1"``, ``"b2"``, …) when stacking layers.

        Returns
        -------
        PSF
            A parametric state function wrapping the dense layer.

        Examples
        --------
        Build the hidden layers of an MLP force field::

            >>> from SFI.bases import X
            >>> import jax.numpy as jnp
            >>> mlp = (
            ...     X(dim=2).vectorize(2)
            ...     .dense(32, weight="W1", bias="b1")
            ...     .elementwisemap(jnp.tanh)
            ...     .dense(1, weight="W2", bias="b2")
            ... )
        """
        from .psf import PSF  # deferred to avoid circular import

        node = DenseNode(self.root, n_out=n_out, weight=weight, bias=bias)
        return PSF(node)

    # =================================================================
    #  NUMPY / JAX UNARY UFUNC FORWARDING
    # =================================================================
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # We only implement the standard call "ufunc(expr [, const])"
        if method != "__call__":
            return NotImplemented

        # Unary: sin(expr)
        if ufunc.nin == 1 and inputs == (self,):
            node = MapNNode(lambda a, _uf=ufunc, _kw=kwargs: _uf(a, **_kw), self.root)
            return self._with_node(node)

        # Binary: expr + const  OR  const + expr
        if ufunc.nin == 2 and len(inputs) == 2:
            a, b = inputs
            if a is self and not isinstance(b, StateExpr):
                return self._binary(b, lambda x, y: ufunc(x, y, **kwargs))
            if b is self and not isinstance(a, StateExpr):
                return self._binary(a, lambda x, y: ufunc(x, y, **kwargs), swap=True)
            if isinstance(a, StateExpr) and isinstance(b, StateExpr):
                return a._binary(b, lambda x, y: ufunc(x, y, **kwargs))

        return NotImplemented

    @property
    def required_extras(self) -> tuple[str, ...]:
        """
        Presence-only extras required by the expression, forwarded from the root node.
        No shape/broadcast semantics here.
        """
        return tuple(getattr(self.root, "extras_required", ()) or ())

    @property
    def particle_extras(self) -> tuple[str, ...]:
        """
        Pure metadata, forwarded from the root node.

        Names of extras declared as per-particle somewhere in the underlying node tree
        (typically by interaction leaves). The dispatcher reads this to know which keys
        to gather from `(P, ...)` into `(E, K, ...)` per edge before calling locals.
        """
        return tuple(getattr(self.root, "particle_extras", ()) or ())


### Helpers ###


def _validate_extras_presence(required, extras):
    """
    Presence rules (new policy):

    - If ``required`` is empty, no check is performed.
    - Otherwise: all keys in ``required`` must be present in ``extras``.

    Legacy ``'*'`` (presence-only) is ignored here; leaves should not emit it.
    """
    required = tuple(k for k in (required or ()) if k != "*")
    if not required:
        return
    if extras is None:
        raise KeyError(f"Missing extras: {required}")
    missing = [k for k in required if k not in extras]
    if missing:
        raise KeyError(f"Missing extras keys: {missing}")


def _walk_nodes(node):
    yield node
    if hasattr(node, "children") and isinstance(node.children, (tuple, list)):
        for ch in node.children:
            yield from _walk_nodes(ch)
    for attr in ("child", "inner"):
        if hasattr(node, attr):
            yield from _walk_nodes(getattr(node, attr))


def _specialize_node(node: BaseNode, dataset: int) -> BaseNode:
    """Recursively fold ``dataset_index``-reading leaves at condition ``dataset``.

    A leaf that carries a ``specialize_at`` recipe is replaced by its
    condition-``dataset`` form; composite nodes are rebuilt from specialized
    children via ``with_children`` (which recomputes the static contract, extras
    union, and parameter suite); all other leaves pass through unchanged.
    """
    hook = getattr(node, "specialize_at", None)
    if hook is not None:
        return hook(dataset).root
    children = getattr(node, "children", None)
    if children:
        new_children = [_specialize_node(c, dataset) for c in children]
        # Only rebuild when a leaf was actually specialized.  If every child is
        # unchanged (the common case: no ``dataset_index``-reading leaves in this
        # subtree) the node is returned as-is, so node types that legitimately
        # do not implement ``with_children`` (e.g. ``InteractionDispatcher``)
        # pass through specialize() untouched instead of raising.
        if all(nc is oc for nc, oc in zip(new_children, children)):
            return node
        return node.with_children(new_children)
    return node


def _static_contract_sanity_node(n):
    # Only validate fields that exist on this node
    pdepth = getattr(n, "pdepth", None)
    rank = getattr(n, "rank", None)
    n_features = getattr(n, "n_features", None)
    dim = getattr(n, "dim", None)
    particles_input = getattr(n, "particles_input", None)

    if pdepth is not None:
        if not isinstance(pdepth, int) or pdepth < 0:
            raise ValueError(f"[Contract] {type(n).__name__}: pdepth must be a non-negative int, got {pdepth!r}")
    if rank is not None:
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"[Contract] {type(n).__name__}: rank must be a non-negative int, got {rank!r}")
    if n_features is not None:
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError(f"[Contract] {type(n).__name__}: n_features must be a positive int, got {n_features!r}")
    if dim is not None:
        if not isinstance(dim, int) or dim < 1:
            raise ValueError(f"[Contract] {type(n).__name__}: dim must be None or a positive int, got {dim!r}")
    if (particles_input is not None) and (pdepth is not None):
        if (not particles_input) and (pdepth > 0):
            raise ValueError(
                f"[Contract] {type(n).__name__}: pdepth>0 requires particles_input=True "
                "(cannot create particle axes without a particle input axis)."
            )


def _is_broadcastable_to_rank(shape: tuple[int, ...], *, rank: int, dim: int) -> bool:
    """True iff `shape` numpy-broadcasts to the spatial rank block `(dim,)*rank`.
    Right-align against the rank block; each aligned axis must be 1 or dim.
    """
    if len(shape) > rank:
        return False
    for s in shape[::-1]:
        if s != 1 and s != dim:
            return False
    return True


# One-off warning for rare multi-feature feature-Cartesian binary ops
_CARTESIAN_FEATURE_WARNED = False


def _warn_cartesian_multi_feature(op: str) -> None:
    """
    Emit a single warning the first time we apply a Cartesian product over
    feature axes in a binary op between two multi-feature expressions.
    """
    import warnings

    global _CARTESIAN_FEATURE_WARNED
    if _CARTESIAN_FEATURE_WARNED:
        return
    warnings.warn(
        f"[StateExpr] Binary op {op!r} between multi-feature expressions uses a "
        "Cartesian product over feature axes (F_out = F_left × F_right). "
        "This can grow n_features quickly; use '&' (stack) or aligned maps if "
        "you intended per-feature operations.",
        RuntimeWarning,
        stacklevel=3,
    )
    _CARTESIAN_FEATURE_WARNED = True
