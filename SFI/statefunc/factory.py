import logging
from typing import Any, Callable, Iterable, Optional, Sequence

import jax.numpy as jnp

from .basis import Basis
from .interactor import Interactor
from .nodes import SimpleLeaf
from .nodes.contract import Rank
from .nodes.leaf import InteractionLeaf
from .params import ParamSpec, ParamSuite
from .psf import PSF
from .sf import SF

logger = logging.getLogger(__name__)


def _probe_call(obj, *, dim: int, rank: int, n_features: int, label: str):
    """Run a zero-valued test call to catch shape/signature errors early.

    Called when ``dim`` is known at factory construction time.
    Logs a warning on failure instead of raising, so construction still
    succeeds (the user may intend a shape that only works with real data).
    """
    x_probe = jnp.zeros((1, dim))  # (N=1, dim)
    try:
        out = obj(x_probe)
    except Exception as exc:
        logger.warning(
            "%s: probe call with x of shape %s failed — the function may error at runtime.  Original error: %s",
            label,
            x_probe.shape,
            exc,
        )
        return

    out = jnp.asarray(out)
    if out.shape[0] != 1:
        logger.warning(
            "%s: probe call returned shape %s but expected leading axis 1 (matching N=1 input).  Check output shape.",
            label,
            out.shape,
        )
        return
    logger.debug("%s: probe call OK, output shape %s", label, out.shape)


def make_basis(
    func: Callable,
    *,
    dim: int | None = None,
    rank: int,
    n_features: int = 1,
    needs_v: bool = False,
    labels: Optional[Sequence[str]] = None,
    descriptor: Any = "custom",
    extras_keys: Optional[Sequence[str]] = None,
    particle_extras: Optional[Sequence[str]] = None,
    specialize_at: Optional[Callable] = None,
) -> Basis:
    """
    Construct a **deterministic Basis** from a *single-sample* user function,
    with **no particle semantics**. Particle axes (if present in ``x`` at call
    time) are treated purely as batch axes and vmapped over.

    ``particle_extras`` names extras keys whose values are **per-sample**
    arrays aligned with the batch/particle axes (e.g. an ``extras_local``
    entry of shape ``(N, ...)``): they are vmapped alongside ``x``, so the
    single-sample function sees *its own* particle's value instead of the
    whole array — the route to per-particle terms in single-particle
    bases (home-range centres, individual labels, ...).

    User function signature — declare **only** the kwargs you need:

    - Simplest:  ``f(x) -> array``
    - With velocity:  ``f(x, *, v) -> array``
    - With extras:  ``f(x, *, extras) -> array``

    The full signature is ``f(x, *, v=None, mask=None, extras=None)``;
    we introspect and pass only the kwargs you declare.

    Shapes (single sample)::

        x: (dim,)
        return: (*rank_axes, m)  # feature last; m == n_features

    If ``n_features == 1`` you may omit the last axis; a singleton feature
    axis is auto-inserted.

    Extras
    ~~~~~~
    If ``extras`` is declared, you may provide ``extras_keys=(...)`` to
    enforce keys. Extras arrays must broadcast over the **batch prefix**
    (never over rank/feature).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from SFI.statefunc import make_basis
    >>> B = make_basis(lambda x: x, dim=2, rank=1, n_features=1) # (equivalent to the built-in X(dim=2))

    JAX
    ~~~
    Write ``f`` with ``jax.numpy`` and keep it pure; works with
    jit/vmap/autodiff.
    """
    leaf = SimpleLeaf(
        func=func,
        n_features=int(n_features),
        labels=tuple(labels) if labels is not None else tuple(f"f{j}" for j in range(n_features)),
        descriptor=descriptor,
        dim=dim,
        rank=Rank(rank),
        needs_v=bool(needs_v),
        # SimpleLeaf forbids particles & pdepth by construction (pdepth=0, particles_input=False).
        extras_keys=tuple(extras_keys) if extras_keys is not None else (),
        particle_extras=tuple(particle_extras) if particle_extras is not None else (),
        specialize_at=specialize_at,
    )
    result = Basis(leaf)
    if dim is not None and not extras_keys and not needs_v:
        _probe_call(result, dim=dim, rank=rank, n_features=n_features, label="make_basis")
    return result


def make_psf(
    func: Callable,
    *,
    dim: int | None = None,
    rank: int,
    n_features: int = 1,
    drop_features: bool = True,
    needs_v: bool = False,
    labels: Optional[Sequence[str]] = None,
    descriptor: Any = "parametric",
    params: ParamSuite | Iterable[ParamSpec] | dict[str, Any],
    extras_keys: Optional[Sequence[str]] = None,
    specialize_at: Optional[Callable] = None,
) -> PSF:
    """
    Construct a **parametric state-function family (PSF)** from a *single-sample*
    user function, **without particle semantics**.

    User function signature — declare **only** the kwargs you need:

    - Simplest:  ``f(x, *, params) -> array``
    - With velocity:  ``f(x, *, v, params) -> array``
    - With extras:  ``f(x, *, params, extras) -> array``

    The full signature is ``f(x, *, params, v=None, mask=None, extras=None)``;
    we introspect and pass only the kwargs you declare.

    Shapes (single sample)::

        x: (dim,)
        return: (*rank_axes, m)  # feature last; m == n_features

    If ``n_features == 1`` you may omit the last axis; we auto-insert a
    singleton feature axis.

    Parameters (``params``) may be described as:

    - a ``ParamSuite``,
    - an iterable of ``ParamSpec``,
    - a dict of shapes, e.g. ``{'W': (d,d), 'b': ()}``,
    - or a dict of **sample arrays** from which (shape, dtype) are inferred.

    Extras
    ~~~~~~
    Same rules as ``make_basis`` (``extras_keys`` optional; broadcast over
    batch prefix).

    JAX
    ~~~
    Works with jit/vmap/autodiff w.r.t. inputs and parameters.
    """
    suite = ParamSuite.parse(params)

    leaf = SimpleLeaf(
        func=func,
        n_features=int(n_features),
        labels=tuple(labels) if labels is not None else tuple(f"f{j}" for j in range(n_features)),
        descriptor=descriptor,
        dim=dim,
        rank=Rank(rank),
        needs_v=bool(needs_v),
        param_suite=suite,  # Parametric path is enabled by providing a ParamSuite.
        extras_keys=tuple(extras_keys) if extras_keys is not None else (),
        specialize_at=specialize_at,
    )
    return PSF(leaf, drop_features=bool(drop_features))


def make_sf(
    func: Callable,
    *,
    dim: int | None = None,
    rank: int,
    n_features: int = 1,
    drop_features: bool = True,
    needs_v: bool = False,
    labels: Optional[Sequence[str]] = None,
    descriptor: Any = "custom_sf",
    extras_keys: Optional[Sequence[str]] = None,
) -> SF:
    """
    Construct an **SF** (bound state function) directly from a parameter-free
    user function — no ``Basis`` or ``PSF`` intermediate needed.

    This is the simplest entry point when you have a known, fixed function
    (e.g. an exact model for comparison, or a hand-coded feature) and just
    want a callable that participates in the SFI expression-tree ecosystem.

    The resulting ``SF`` supports ``.d_x()``, ``.d_v()``, and can be passed
    to ``compare_to_exact``, ``integrate``, or any other API that accepts
    an ``SF`` / ``StateExpr``.

    User function signature — declare **only** the kwargs you need:

    - Simplest:  ``f(x) -> array``
    - With velocity:  ``f(x, *, v) -> array``
    - With extras:  ``f(x, *, extras) -> array``

    Shapes (single sample)::

        x:      (dim,)
        return: (*rank_axes, n_features)

    If ``n_features == 1`` you may omit the trailing feature axis; it is
    auto-inserted.  The resulting SF squeezes it back when
    ``drop_features=True`` (default).

    Parameters
    ----------
    func : callable
        Pure JAX function, compatible with jit/vmap/autodiff.
    dim : int or None
        Spatial dimensionality (None = infer at first call).
    rank : int
        Tensor rank of the output (0=scalar, 1=vector, 2=matrix).
    n_features : int
        Number of output features (default 1).
    drop_features : bool
        Remove trailing size-1 feature axis (default True).
    needs_v : bool
        Whether ``func`` requires velocity ``v``.
    labels : sequence of str or None
        Human-readable feature labels (auto-generated if None).
    descriptor : any
        Metadata tag stored on the leaf node.
    extras_keys : sequence of str or None
        Required keys in the ``extras`` mapping.

    Returns
    -------
    SF
        A bound, callable state function with no free parameters.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from SFI.statefunc import make_sf
    >>> harmonic = make_sf(lambda x: -x, rank=1, dim=2)
    >>> harmonic(jnp.array([1.0, 2.0]))
    Array([-1., -2.], dtype=float32)
    """
    # Build a SimpleLeaf with an *empty* (but non-None) ParamSuite so the
    # node is accepted by PSF.  The empty suite carries zero parameters,
    # so PSF.__call__ auto-supplies params={} and SF.bind freezes nothing.
    empty_suite = ParamSuite([])

    leaf = SimpleLeaf(
        func=func,
        n_features=int(n_features),
        labels=tuple(labels) if labels is not None else tuple(f"f{j}" for j in range(n_features)),
        descriptor=descriptor,
        dim=dim,
        rank=Rank(rank),
        needs_v=bool(needs_v),
        param_suite=empty_suite,
        extras_keys=tuple(extras_keys) if extras_keys is not None else (),
    )
    psf = PSF(leaf, drop_features=bool(drop_features))
    result = psf.bind({})
    if dim is not None and not extras_keys and not needs_v:
        _probe_call(result, dim=dim, rank=rank, n_features=n_features, label="make_sf")
    return result


def make_interactor(
    func: Callable,
    *,
    dim: int,
    rank: Rank,
    # arity:
    K: int | None = None,
    Kmax: int | None = None,
    # features & plumbing:
    n_features: int = 1,
    needs_v: bool = False,
    labels: Iterable[str] = (),
    descriptor=None,
    params: ParamSuite | None = None,
    extras_keys: Iterable[str] = (),
    particle_extras: Iterable[str] = (),
):
    """
    Build a local interaction dictionary (Interactor) from a single-sample user
    function that consumes (K, dim) and returns feature-last.

    Pass exactly one of:
      - K=int           → fixed arity
      - Kmax=int        → variable arity (ragged via mask)

    ``particle_extras`` names the extras keys whose values are
    **per-particle** arrays (shape ``(P, ...)``): the dispatcher gathers
    them per edge member, so inside ``func`` they arrive with shape
    ``(K, ...)`` — one entry per member of the local tuple.  The
    reserved ``"particle_index"`` extra (injected by
    :class:`~SFI.trajectory.TrajectoryCollection`) combined with a
    ``(P,)``-shaped parameter gives per-particle inferred parameters::

        def local(Xk, *, params, extras):
            mob = params["mob"][extras["particle_index"]]   # (K,)
            ...
    """
    suite = ParamSuite.parse(params)
    if (K is None) == (Kmax is None):
        raise ValueError("Provide exactly one of K or Kmax.")

    leaf = InteractionLeaf(
        mode="fixed" if K is not None else "variable",
        K=K,
        Kmax=Kmax,
        func=func,
        dim=dim,
        rank=rank,
        n_features=n_features,
        needs_v=needs_v,
        param_suite=suite,
        labels=tuple(labels),
        descriptor=descriptor,
        extras_keys=tuple(extras_keys),
        particle_extras=tuple(particle_extras),
    )
    return Interactor(leaf)
