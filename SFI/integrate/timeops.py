from __future__ import annotations

import inspect
from typing import Callable, FrozenSet, Iterable, Optional

import jax.numpy as jnp

__all__ = [
    "TimeOp",
    "timeop",
    "stream",
    "velocity",
    "scale",
    "add",
]


class TimeOp:
    """
    A time-local operator evaluated on one time slice.

    A TimeOp declares which streamed fields it needs either via its function
    signature or the explicit *requires* parameter.

    Example
    -------
    >>> @timeop
    ... def vel(dX_minus, dt_minus):
    ...     return dX_minus / dt_minus[..., None]
    >>> vel.requires
    frozenset({'dX_minus', 'dt_minus'})
    """

    def __init__(
        self,
        fn: Callable[..., jnp.ndarray],
        name: str | None = None,
        *,
        batch_safe: bool = False,
        requires: Optional[FrozenSet[str]] = None,
    ):
        self.fn = fn
        self._name = name or fn.__name__
        self.batch_safe = batch_safe
        if requires is not None:
            self._requires: FrozenSet[str] = frozenset(requires)
        else:
            sig = inspect.signature(fn)
            self._requires = frozenset(
                p.name
                for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            )

    def __call__(self, **streams) -> jnp.ndarray:
        return self.fn(**{k: streams[k] for k in self._requires})

    @property
    def requires(self) -> FrozenSet[str]:
        return self._requires

    @property
    def name(self) -> str:
        return self._name


def timeop(
    fn: Callable[..., jnp.ndarray] | None = None,
    *,
    name: str | None = None,
    batch_safe: bool = False,
    requires: Optional[FrozenSet[str]] = None,
):
    """
    Decorator: convert a function into a TimeOp.

    Parameters
    ----------
    batch_safe : bool
        If True, the function already handles a leading batch (K) axis
        in its inputs without requiring an additional ``jax.vmap``.
    requires : frozenset, optional
        Explicit set of required stream keys.  When given, the function
        signature is not inspected for parameter names.  Use this when
        the function accepts ``**streams`` and the required keys cannot
        be inferred from the signature.

    Notes
    -----
    Without *requires*, the function's parameter names define the required
    stream keys (``**kwargs``/``*args`` parameters are ignored).
    """

    def _wrap(f: Callable[..., jnp.ndarray]) -> TimeOp:
        return TimeOp(f, name=name, batch_safe=batch_safe, requires=requires)

    return _wrap if fn is None else _wrap(fn)


# --------- Stock, composable timeops ---------


def stream(key: str, *, name: str | None = None) -> TimeOp:
    """Return a TimeOp that passes stream *key* through unchanged."""

    def _op(**streams: jnp.ndarray) -> jnp.ndarray:
        return streams[key]

    return TimeOp(_op, name=name or f"stream[{key}]", batch_safe=True, requires=frozenset({key}))


def velocity(dx_key: str, dt_key: str, *, name: str | None = None) -> TimeOp:
    """Return a TimeOp that computes dx/dt, broadcasting dt over arbitrary leading dims."""

    def _op(**streams: jnp.ndarray) -> jnp.ndarray:
        dx = streams[dx_key]
        dt = streams[dt_key]
        # Expand dt to broadcast with dx for arbitrary leading batch dims.
        # Single-row: dt scalar, dx (N, d) -> dt becomes (1, 1) -> OK.
        # Batch:      dt (K,),   dx (K, N, d) -> dt becomes (K, 1, 1) -> OK.
        while dt.ndim < dx.ndim:
            dt = dt[..., jnp.newaxis]
        return dx / dt

    return TimeOp(
        _op,
        name=name or f"V[{dx_key}/{dt_key}]",
        batch_safe=True,
        requires=frozenset({dx_key, dt_key}),
    )


def scale(op: TimeOp, alpha: float | int, *, name: str | None = None) -> TimeOp:
    """Return a TimeOp that multiplies *op* by scalar *alpha*."""

    def _op(**streams: jnp.ndarray) -> jnp.ndarray:
        return alpha * op(**streams)

    return TimeOp(
        _op,
        name=name or f"{alpha}*{op.name}",
        batch_safe=op.batch_safe,
        requires=op.requires,
    )


def add(ops: Iterable[TimeOp], *, name: str | None = None) -> TimeOp:
    """Return a TimeOp that sums the outputs of all *ops* element-wise.

    Parameters
    ----------
    ops : iterable of TimeOp
        At least one operand is required.
    """
    ops = tuple(ops)
    if not ops:
        raise ValueError(
            "add() requires at least one TimeOp; got an empty sequence."
        )
    nm = name or " + ".join(o.name for o in ops)
    req = frozenset().union(*(o.requires for o in ops))
    all_safe = all(o.batch_safe for o in ops)

    def _op(**streams: jnp.ndarray) -> jnp.ndarray:
        out = None
        for o in ops:
            cur = o(**streams)
            out = cur if out is None else out + cur
        return out  # type: ignore[return-value]  # out is set because ops is non-empty

    return TimeOp(_op, name=nm, batch_safe=all_safe, requires=req)
