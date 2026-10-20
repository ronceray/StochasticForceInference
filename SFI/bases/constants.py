from typing import Sequence

import jax.numpy as jnp

from SFI.statefunc import Basis, Rank
from SFI.statefunc.nodes import SimpleLeaf


def ones_basis(dim: int, pdepth: int = 0) -> Basis:
    def _ones(x):
        return jnp.ones((*x.shape[:-1], 1))

    return Basis(
        SimpleLeaf(
            func=_ones,
            n_features=1,
            labels=("1",),
            descriptor="scalar-one",
            dim=dim,
            rank=Rank.SCALAR,
            pdepth=pdepth,
        )
    )


def constant_array(A, *, label: str = "const", descriptor: str = "constant-array", as_sf=True):
    """
    Constant basis/sf with a single feature whose value is a fixed tensor A of shape
    ``(dim,)*rank`` (rank inferred from ``A.ndim``, dim from ``A.shape[0]``).

    - Errors if A is not a hypercube tensor (all axes same length).
    - Broadcasts over batch/particles: output shape is
      ``(*x.shape[:-1], (dim,)*rank, 1)``.
    """
    A = jnp.asarray(A)
    if A.ndim == 0:
        raise ValueError("Use ones_basis()/scalar constants for rank=SCALAR.")
    if len(set(A.shape)) != 1:
        raise ValueError(f"A must have shape (dim,)*rank (hypercube). Got {A.shape}.")

    dim = int(A.shape[0])
    rank = int(A.ndim)

    def _const(x):
        # x: (..., dim). We broadcast A over all leading axes of x (particles/time/...)
        lead = x.shape[:-1]
        return jnp.broadcast_to(A[..., None], (*lead, *A.shape, 1))

    leaf = SimpleLeaf(
        func=_const,
        n_features=1,
        labels=(label,),
        descriptor=descriptor,
        dim=dim,
        rank=rank,
        pdepth=0,
    )

    B = Basis(leaf)
    if as_sf:
        return B.to_psf().bind(params={"coeff": 1.0})
    return B


def unit_vector_basis(dim: int, axes: Sequence[int] | None = None) -> Basis:
    if axes is None:
        axes = list(range(dim))

    labels = [f"e{i}" for i in axes]

    def _f(x):
        out = jnp.zeros((dim, len(axes)))
        for i, a in enumerate(axes):
            out = out.at[a, i].set(1)
        return out

    return Basis(
        SimpleLeaf(
            func=_f,
            n_features=len(axes),
            labels=tuple(labels),
            descriptor="unit-vector-set",
            dim=dim,
            rank=Rank.VECTOR,
            pdepth=0,
            particles_input=False,
            needs_v=False,
        )
    )


def identity_matrix_basis(dim: int, pdepth: int = 0) -> Basis:
    def _f(x):
        eye = jnp.eye(dim)
        return jnp.broadcast_to(eye, (*x.shape[:-1], dim, dim))

    return Basis(
        SimpleLeaf(
            func=_f,
            n_features=1,
            labels=("I",),
            descriptor="identity",
            dim=dim,
            rank=Rank.MATRIX,
            pdepth=pdepth,
        )
    )


def symmetric_matrix_basis(dim: int, pdepth: int = 0) -> Basis:
    """Constant symmetric-matrix templates spanning the space of real symmetric
    ``dim × dim`` matrices.

    For ``dim=d`` there are ``d(d+1)/2`` features: one per upper-triangle
    entry ``(i,j)`` with ``i <= j``.  Each feature is a ``(dim, dim)``
    matrix: ``S_{(i,j)} = δ_{ia}δ_{jb} + δ_{ib}δ_{ja}`` (so the
    off-diagonal templates equal 1 in both symmetric slots, diagonal
    templates equal 1 on the diagonal).

    Rank is ``MATRIX`` (2), and the output shape is ``(dim, dim, F)``.
    """
    pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
    labels = [f"S{i}{j}" for (i, j) in pairs]

    # Pre-build the (dim, dim, F) template at module level
    import numpy as _np  # deferred: only needed at build time, not at JAX eval time

    tpl = _np.zeros((dim, dim, len(pairs)), dtype="float32")
    for k, (i, j) in enumerate(pairs):
        tpl[i, j, k] = 1.0
        tpl[j, i, k] = 1.0  # symmetric: both slots
    tpl_jnp = jnp.array(tpl)

    def _f(x):
        return tpl_jnp

    return Basis(
        SimpleLeaf(
            func=_f,
            n_features=len(pairs),
            labels=tuple(labels),
            descriptor="symmetric-matrix-templates",
            dim=dim,
            rank=Rank.MATRIX,
            pdepth=pdepth,
            particles_input=False,
            needs_v=False,
        )
    )


# ---------------------------------------------------------------------------
# Named scalar parameters (rank-0 PSFs whose value is a single named param)
# ---------------------------------------------------------------------------
def named_scalar(name: str, default=None, *, dim: int | None = None, label: str | None = None):
    """Rank-0, 1-feature PSF whose value is a single named scalar parameter.

    The returned PSF carries a single :class:`ParamSpec` with shape ``()``,
    optional ``default``, and label ``label or name``.

    Parameters
    ----------
    name : str
        Parameter name; also the default feature label.
    default : scalar or None
        Optional default value.  When set, the PSF can be evaluated, bound,
        or passed to a simulator without explicit ``params``.
    dim : int or None
        Spatial dimensionality; ``None`` (default) lets it be inferred at
        first call (the value is independent of ``x``).
    label : str or None
        Optional human-readable feature label (defaults to ``name``).

    Examples
    --------
    >>> sigma = named_scalar("sigma", default=20.0)
    >>> sigma()                       # uses default
    Array(20., dtype=float32)
    >>> sigma(params={"sigma": 30.})  # explicit override
    Array(30., dtype=float32)
    """
    from ..statefunc import make_psf
    from ..statefunc.params import ParamSpec

    spec = ParamSpec(name, (), default=default)

    def _f(x, *, params):
        return params[name]

    return make_psf(
        _f,
        dim=dim,
        rank=0,
        n_features=1,
        labels=[label or name],
        descriptor=f"named-scalar({name})",
        params=[spec],
    )


def extra_scalar(name: str, *, dim: int | None = None, label: str | None = None):
    """Rank-0, 1-feature Basis whose value is read from ``extras[name]``.

    The compositional symbol for data-carried quantities: an extra
    (per-experiment constant, a time-dependent drive delivered per frame,
    a per-particle property) becomes an expression that composes with
    ``x_components`` / ``unit_axes`` / ``named_scalar`` through the usual
    algebra — and, being a parameter-free :class:`Basis`, slots directly
    into linear-estimator dictionaries.

    Parameters
    ----------
    name : str
        Extras key to read; also the default feature label.
    dim : int or None
        Spatial dimensionality; ``None`` (default) lets it be inferred at
        first call (the value is independent of ``x``).
    label : str or None
        Optional human-readable feature label (defaults to ``name``).

    Examples
    --------
    >>> from SFI.bases import X, extra_scalar
    >>> k_t = extra_scalar("k_drive")          # delivered by the dataset
    >>> B = X(dim=2) & (k_t * X(dim=2))        # static + driven trap terms
    >>> # simulate: OverdampedProcess(F=B, theta_F=jnp.array([-1.0, -1.0]))
    >>> # infer:    inf.infer_force_linear(B)

    Notes
    -----
    At inference time the trajectory layer materializes ``extras[name]``
    per frame (slicing :class:`~SFI.trajectory.TimeSeriesExtra` values);
    in simulation, ``set_extras`` accepts the same time-dependent forms.
    """
    from ..statefunc import make_basis

    def _f(x, *, extras):
        return jnp.asarray(extras[name])[None]

    return make_basis(
        _f,
        dim=dim,
        rank=0,
        n_features=1,
        labels=[label or name],
        descriptor=f"extra-scalar({name})",
        extras_keys=(name,),
    )


def time_fourier(
    n_modes: int,
    period: float | None = None,
    *,
    dim: int | None = None,
    label: str | None = None,
):
    r"""Rank-0 time-Fourier dictionary read from the reserved ``time`` extra.

    Emits ``1 + 2 * n_modes`` parameter-free features

    .. math::

        \bigl\{\,1,\; \cos(k\omega t),\; \sin(k\omega t)\,\bigr\}_{k=1}^{n_\text{modes}},
        \qquad \omega = 2\pi / P,

    evaluated at each frame's absolute time ``t`` — the auto-injected
    ``time`` extra (see :meth:`TrajectoryDataset.build_extras`), so no
    bookkeeping is required.  Tensor it with a spatial basis to learn an
    *unknown* time-dependent force field by expansion: ``time_fourier(4) *
    X(dim=1)`` recovers a time-varying stiffness :math:`k(t)`, and
    ``time_fourier(4) * unit_vector_basis(1)`` a moving trap centre.
    Sparse selection (:term:`PASTIS`) keeps only the harmonics the data
    support.

    Parameters
    ----------
    n_modes : int
        Number of harmonics; produces ``1 + 2 * n_modes`` features.
    period : float or None
        Fundamental period :math:`P`.  If ``None`` (default), it defaults
        to the **full trajectory duration** (read from the auto-injected
        ``duration`` extra), i.e. the fundamental frequency is the inverse
        of the total observation time.
    dim : int or None
        Spatial dimensionality; ``None`` lets it be inferred (the value is
        independent of ``x``).
    label : str or None
        Optional label prefix for the features.

    Examples
    --------
    >>> from SFI.bases import X, time_fourier
    >>> B = time_fourier(4) * X(dim=1)        # learn k(t) over the trajectory
    >>> inf.infer_force_linear(B)
    """
    from ..statefunc import make_basis

    if n_modes < 1:
        raise ValueError("time_fourier requires n_modes >= 1")

    keys = ("time",) if period is not None else ("time", "duration")
    n_feat = 1 + 2 * n_modes
    pre = (label + " ") if label else ""
    labels = [f"{pre}1"]
    for k in range(1, n_modes + 1):
        labels += [f"{pre}cos({k}wt)", f"{pre}sin({k}wt)"]

    ks = jnp.arange(1, n_modes + 1, dtype=float)

    def _f(x, *, extras):
        t = jnp.asarray(extras["time"], dtype=float)
        P = period if period is not None else jnp.asarray(extras["duration"], dtype=float)
        ang = ks * (2.0 * jnp.pi / P) * t                                   # (n_modes,)
        cs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1).reshape(-1)   # (2 n_modes,)
        return jnp.concatenate([jnp.ones((1,), dtype=cs.dtype), cs])        # (1 + 2 n_modes,)

    return make_basis(
        _f,
        dim=dim,
        rank=0,
        n_features=n_feat,
        labels=labels,
        descriptor=f"time-fourier(n={n_modes})",
        extras_keys=keys,
    )


def per_dataset_scalar(name: str, n_datasets: int, default=None, *, dim: int | None = None):
    """Rank-0, 1-feature PSF whose value is dataset-specific.

    Carries a parameter array of shape ``(n_datasets,)`` and reads the
    entry of the current dataset through the reserved
    ``extras["dataset_index"]`` (injected automatically by
    :class:`~SFI.trajectory.TrajectoryCollection`).  Compose it with
    shared :func:`named_scalar` terms to fit pooled multi-experiment
    models where part of the parameters is experiment-specific and the
    rest is shared.

    Parameters
    ----------
    name : str
        Parameter name (one entry per dataset).
    n_datasets : int
        Number of datasets in the collection the model will be fit on.
    default : scalar or array of shape (n_datasets,), optional
        Optional default value(s); a scalar is broadcast.
    dim : int or None
        Spatial dimensionality (``None`` → inferred).

    Notes
    -----
    Indexed parameter access is nonlinear in the bookkeeping sense, so
    the parametric estimators fit models containing this primitive on
    the L-BFGS path.  For the **linear estimators**, use the one-hot
    route instead: :func:`dataset_indicator`.
    """
    from ..statefunc import make_psf
    from ..statefunc.params import ParamSpec

    if default is not None:
        default = jnp.broadcast_to(jnp.asarray(default, dtype=float), (n_datasets,))
    spec = ParamSpec(name, (int(n_datasets),), default=default)

    def _f(x, *, params, extras):
        return params[name][extras["dataset_index"]]

    def _specialize_at(k, _name=name, _dim=dim, _default=default):
        # Condition-k slice: the (n_datasets,) param collapses to a scalar and
        # the dataset_index read disappears (see StateExpr.specialize).
        d_k = None if _default is None else jnp.asarray(_default)[int(k)]
        spec_k = ParamSpec(_name, (), default=d_k)

        def _fk(x, *, params):
            return params[_name]

        return make_psf(
            _fk,
            dim=_dim,
            rank=0,
            n_features=1,
            labels=[_name],
            descriptor=f"per-dataset-scalar({_name})@{int(k)}",
            params=[spec_k],
        )

    return make_psf(
        _f,
        dim=dim,
        rank=0,
        n_features=1,
        labels=[name],
        descriptor=f"per-dataset-scalar({name})",
        params=[spec],
        extras_keys=("dataset_index",),
        specialize_at=_specialize_at,
    )


def dataset_indicator(n_datasets: int, *, dim: int | None = None):
    """Rank-0 Basis of ``n_datasets`` one-hot features ``1{dataset == d}``.

    The **linear-estimator** route to per-dataset coefficients: multiply
    a feature by the indicator and concatenate, and each dataset gets an
    independent linear coefficient for that feature (the Gram is
    block-diagonal across datasets), PASTIS-prunable like any feature.

    .. code-block:: python

       B = B_shared & (dataset_indicator(n) * X(dim))

    Reads the reserved ``extras["dataset_index"]`` injected by
    :class:`~SFI.trajectory.TrajectoryCollection`.
    """

    from ..statefunc import make_basis

    n = int(n_datasets)

    def _f(x, *, extras):
        return (jnp.arange(n) == extras["dataset_index"]).astype(float)

    def _specialize_at(k, _n=n, _dim=dim):
        # Condition-k one-hot becomes a constant vector; no dataset_index read.
        onehot = (jnp.arange(_n) == int(k)).astype(float)

        def _fk(x):
            return onehot

        return make_basis(
            _fk,
            dim=_dim,
            rank=0,
            n_features=_n,
            labels=[f"ds{i}" for i in range(_n)],
            descriptor=f"dataset-indicator({_n})@{int(k)}",
        )

    return make_basis(
        _f,
        dim=dim,
        rank=0,
        n_features=n,
        labels=[f"ds{i}" for i in range(n)],
        descriptor=f"dataset-indicator({n})",
        extras_keys=("dataset_index",),
        specialize_at=_specialize_at,
    )


# Per-particle inferred parameters ("params_local paralleling extras_local")
# are expressed inside interactor kernels: declare the reserved
# ``particle_index`` extra as particle-aligned and index a (P,)-shaped
# parameter with it —
#
#     def local(Xk, *, params, extras):
#         mob = params["mob"][extras["particle_index"]]   # (K,) per edge
#         ...
#     inter = make_interactor(local, ..., params={"mob": (P,)},
#                             extras_keys=("particle_index",),
#                             particle_extras=("particle_index",))
#
# See :func:`SFI.statefunc.make_interactor`.


def named_scalars(*args, **kwargs):
    """Unpack named scalar parameters, one PSF per name.

    Two equivalent call styles, both with deterministic ordering:

    Positional names (no defaults)::

        sigma, rho, beta = named_scalars("sigma", "rho", "beta")

    Keyword names with defaults (Python preserves call-site order)::

        sigma, rho, beta = named_scalars(sigma=20.0, rho=8.0, beta=2.0)

    Returns
    -------
    tuple[PSF, ...]
        One :class:`PSF` per name, in the order given.
    """
    if args and kwargs:
        raise TypeError("named_scalars: pass either positional names OR keyword name=default, not both.")
    if kwargs:
        return tuple(named_scalar(name, default=val) for name, val in kwargs.items())
    if not args:
        raise TypeError("named_scalars: at least one name is required.")
    if not all(isinstance(a, str) for a in args):
        raise TypeError("named_scalars: positional arguments must be parameter names (str).")
    return tuple(named_scalar(name) for name in args)
