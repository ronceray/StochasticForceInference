import functools
import itertools

import jax.numpy as jnp

from ..statefunc import Basis, Rank, make_basis

_RANK_ALIASES = {
    "scalar": None,
    "vector": "vector",
    "matrix": "symmetric",
    "symmetric_matrix": "symmetric",
    "identity_matrix": "identity",
}


def _lift(B: Basis, rank: str, dim: int):
    """Lift a scalar Basis to higher rank via Cartesian product with a structural basis."""
    import warnings

    from .constants import identity_matrix_basis, symmetric_matrix_basis, unit_vector_basis

    if rank not in _RANK_ALIASES:
        raise ValueError(f"Unknown rank={rank!r}. Choose from: {list(_RANK_ALIASES.keys())}")
    mode = _RANK_ALIASES[rank]
    if mode is None:
        return B
    # Suppress the Cartesian-product warning — this is intentional lifting.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Cartesian product.*")
        if mode == "vector":
            return B * unit_vector_basis(dim)
        if mode == "identity":
            return B * identity_matrix_basis(dim)
        if mode == "symmetric":
            return B * symmetric_matrix_basis(dim)
    raise AssertionError("unreachable")  # pragma: no cover


@functools.lru_cache(maxsize=64)
def _exponent_mats(dim: int, degree: int, include_x: bool, include_v: bool):
    """Return (ex_x, ex_v, labels) for all monomials of exact total degree."""
    if not include_x and not include_v:
        raise ValueError("At least one of include_x / include_v must be True.")
    L = (dim if include_x else 0) + (dim if include_v else 0)

    # stars-and-bars enumeration for total degree == degree
    if degree == 0:
        gammas = [tuple(0 for _ in range(L))]
    else:
        gammas = []
        for stars in itertools.combinations_with_replacement(range(L), degree):
            arr = [0] * L
            for s in stars:
                arr[s] += 1
            gammas.append(tuple(arr))

    ex_x, ex_v, labels = [], [], []
    for g in gammas:
        alpha = g[:dim] if include_x else (0,) * dim
        beta = g[-dim:] if include_v else (0,) * dim

        parts = []
        for k, p in enumerate(alpha):
            if p:
                parts.append(f"x{k}^{p}" if p > 1 else f"x{k}")
        for k, p in enumerate(beta):
            if p:
                parts.append(f"v{k}^{p}" if p > 1 else f"v{k}")
        lab = "1" if not parts else "·".join(parts)

        ex_x.append(alpha)
        ex_v.append(beta)
        labels.append(lab)

    ex_x = jnp.array(ex_x, dtype=jnp.int32)  # (F, dim)
    ex_v = jnp.array(ex_v, dtype=jnp.int32)  # (F, dim)
    return ex_x, ex_v, tuple(labels)


def monomials_degree(
    degree: int,
    *,
    dim: int,
    include_x: bool = True,
    include_v: bool = False,
    rank: str = "scalar",
):
    """
    All monomials of **exact** total degree in x and/or v.

    Parameters
    ----------
    degree : int
        Exact total polynomial degree.
    dim : int
        Spatial dimension.
    include_x, include_v : bool
        Which variables to include.
    rank : str
        Output rank. ``'scalar'`` (default) returns a scalar Basis with
        ``F`` features.  ``'vector'`` lifts to rank-1 via Cartesian product
        with ``unit_vector_basis(dim)`` (F × dim features).
        ``'matrix'`` / ``'symmetric_matrix'`` lifts to rank-2 via
        ``symmetric_matrix_basis(dim)`` (F × dim(dim+1)/2 features).
        ``'identity_matrix'`` lifts via ``identity_matrix_basis(dim)``
        (F × 1 features, isotropic).
    """
    ex_x, ex_v, labels = _exponent_mats(dim, degree, include_x, include_v)
    F = int(ex_x.shape[0])
    use_x, use_v = bool(include_x), bool(include_v)

    # Precompute static stuff outside eval (captured in the closure)
    Dx = int(ex_x.max()) if use_x else 0  # max degree for x
    Dv = int(ex_v.max()) if use_v else 0  # max degree for v
    Ix = ex_x.T if use_x else None  # (dim, F) gather indices
    Iv = ex_v.T if use_v else None  # (dim, F)

    def _power_table_fixed(z, D):
        """
        z: (dim,) -> (dim, D+1) = [1, z, z^2, ..., z^D]

        Important: avoid jnp.power with float exponents because its JVP/VJP uses log(z)
        and yields NaNs at z=0 even for integer powers (e.g. 0^0 pathways).
        This iterative construction is polynomial and differentiable everywhere.
        """
        dim = z.shape[0]
        one = jnp.ones((dim, 1), dtype=z.dtype)
        if D == 0:
            return one
        # powers[:, k] = z^k
        powers = [one, z[:, None]]
        for _ in range(2, D + 1):
            powers.append(powers[-1] * z[:, None])
        return jnp.concatenate(powers, axis=1)

    if use_x and use_v:

        def _eval(x, *, v=None, mask=None):
            # x,v: (dim,) ; returns (F,)
            if v is None:
                v = jnp.zeros_like(x)
            Tx = _power_table_fixed(x, Dx)  # (dim, Dx+1)
            Tv = _power_table_fixed(v, Dv)  # (dim, Dv+1)
            vx = jnp.take_along_axis(Tx, Ix, axis=1)  # (dim, F)
            vv = jnp.take_along_axis(Tv, Iv, axis=1)  # (dim, F)
            return jnp.prod(vx * vv, axis=0)  # (F,)
    elif use_x:

        def _eval(x, *, v=None, mask=None):
            Tx = _power_table_fixed(x, Dx)  # (dim, Dx+1)
            vx = jnp.take_along_axis(Tx, Ix, axis=1)  # (dim, F)
            return jnp.prod(vx, axis=0)  # (F,)
    else:  # use_v only

        def _eval(x, *, v=None, mask=None):
            Tv = _power_table_fixed(v if v is not None else jnp.zeros_like(x), Dv)
            vv = jnp.take_along_axis(Tv, Iv, axis=1)  # (dim, F)
            return jnp.prod(vv, axis=0)  # (F,)

    B = make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=F,
        needs_v=use_v,
        labels=list(labels),
    )
    return _lift(B, rank, dim)


def monomials_up_to(
    order: int,
    *,
    dim: int,
    include_constant: bool = True,
    include_x: bool = True,
    include_v: bool = False,
    rank: str = "scalar",
):
    r"""
    Concatenate degree-wise monomial bases for degrees 0..order (ascending).

    .. physics:: Multivariate polynomial basis
       :label: polynomial-basis
       :category: Basis functions

       .. math::

          f_\\alpha(x) = \\prod_{k=1}^{d} x_k^{\\alpha_k},
          \\qquad |\\alpha| \\le \\texttt{order}

       Full polynomial dictionary up to a given total degree, optionally
       including velocity monomials and lifted to vector or matrix rank.

    Parameters
    ----------
    order : int
        Maximum total polynomial degree.
    dim : int
        Spatial dimension.
    include_constant : bool
        If False, skip degree-0 (constant) term.
    include_x, include_v : bool
        Which variables to include.
    rank : str
        Output tensor rank.  See :func:`monomials_degree` for allowed
        values: ``'scalar'``, ``'vector'``, ``'matrix'`` /
        ``'symmetric_matrix'``, ``'identity_matrix'``.

        For force inference, use ``rank='vector'`` (the most common choice).
    """
    start = 0 if include_constant else 1
    if start > order:
        raise ValueError("No monomials to include: set include_constant=True or order>=1.")
    Bs = [monomials_degree(d, dim=dim, include_x=include_x, include_v=include_v) for d in range(start, order + 1)]
    B = Basis.stack(Bs)
    return _lift(B, rank, dim)
