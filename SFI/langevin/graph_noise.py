"""Graph noise: noise shared between particles through sources on a hypergraph (EXPERIMENTAL).

.. warning:: Experimental -- not exported from :mod:`SFI.langevin`; the API may change.

Model.  Each *source* ``s`` (a particle, a pair, a cell, ...) carries its own
independent Gaussian kick ``xi_s ~ N(0, I_m)``, and each member ``k`` of the source
receives ``B_{s,k}(x) xi_s``::

    dx_i = F_i dt + sqrt(2 dt) sum_{s ∋ i} B_{s,i}(x) xi_s ,
    D_ij = sum_{s ∋ i, j} B_{s,i} B_{s,j}^T               (positive semi-definite by construction).

Sources come in *families* ``q`` with known amplitude shapes and an intensity
``theta_q``: ``B_{s,k} = sqrt(theta_q) G_{s,k}(x)``, so ``D = sum_q theta_q D^q`` is
linear in the intensities.  A family is a per-slot interaction kernel
``G(Xk, mask, extras) -> (K, d, m)`` evaluated on each source's members, exactly
like a per-slot interactor (:func:`SFI.statefunc.make_interactor`), plus where its
sources live: a hyperedge CSR table in ``extras`` (``indptr_key``, ``indices_key``)
or ``None`` for one source per particle (independent white noise).

* :class:`GraphNoise` is a :class:`~SFI.langevin.noise.NoiseModel`: it samples by
  running each family's kernel, multiplied by the source's kick, through the
  standard hyperedge dispatcher (the kick is a per-hyperedge spec extra), so masks,
  padding, chunking and per-step topology work as for forces.
* :func:`noise_graph` and :func:`assemble_blocks` build ``D`` on the *noise graph*
  (pairs of particles that share a source): per-particle diagonal blocks and
  per-edge off-diagonal blocks, used by the estimator
  (:mod:`SFI.inference.graph_diffusion`) and by consistency checks.
* :func:`conserved_bond_family` / :func:`conserved_bond_noise`: conserved (divergence-
  form) noise on a lattice field, one source per nearest-neighbour bond
  (``D = M/dV * (-lattice Laplacian)``; periodic or no-flux boundaries, field
  selection, field-dependent mobility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .noise import NoiseModel

Array = jnp.ndarray

_XI = "graph_noise/xi"  # reserved per-source spec extra carrying the kicks


@dataclass(frozen=True)
class NoiseFamily:
    """One family of noise sources with a known amplitude shape.

    Parameters
    ----------
    amplitude : callable or None
        ``G(Xk, mask, extras) -> (K, d, m)``: the amplitude on each of the ``K``
        member slots of one source (``Xk`` the members' states, ``mask`` the slot
        validity).  ``None`` with ``indptr_key=None`` means one source per particle
        with ``G = I_d`` (independent white noise).
    indptr_key, indices_key : str or None
        Hyperedge CSR of the sources in ``extras`` (source ``s`` has members
        ``indices[indptr[s]:indptr[s+1]]``).  ``None``: one source per particle.
    n_noise : int
        Dimension ``m`` of each source's kick.
    Kmax : int
        Maximum number of members of a source.
    extras_keys : tuple of str
        Global extras the amplitude reads (e.g. ``"box"``).
    label : str
        Name used in reports.
    """

    amplitude: Optional[Callable] = None
    indptr_key: Optional[str] = None
    indices_key: Optional[str] = None
    n_noise: int = 1
    Kmax: int = 1
    extras_keys: Tuple[str, ...] = ()
    label: str = "family"

    @property
    def per_particle(self) -> bool:
        return self.indptr_key is None

    @staticmethod
    def white(dim: int, label: str = "self") -> "NoiseFamily":
        """Independent per-particle noise ``G = I_d`` (one source per particle)."""
        return NoiseFamily(amplitude=None, n_noise=dim, Kmax=1, label=label)

    def amplitudes(self, Xk: Array, mask: Array, extras) -> Array:
        """``(K, d, m)`` amplitudes of one source, zero on invalid slots."""
        if self.amplitude is None:
            d = Xk.shape[-1]
            G = jnp.broadcast_to(jnp.eye(d, dtype=Xk.dtype), (Xk.shape[0], d, d))
        else:
            G = self.amplitude(Xk, mask, extras)
        return jnp.where(mask[:, None, None], G, 0.0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class GraphNoise(NoiseModel):
    """Noise shared through sources on a hypergraph (see module docstring).

    Parameters
    ----------
    families : sequence of :class:`NoiseFamily`
    theta : sequence of float
        Intensity of each family (``B = sqrt(theta) G``).
    dim : int
        State dimension per particle.
    chunk_size : int, optional
        Hyperedge chunk size of the dispatchers.

    Notes
    -----
    ``sample`` follows the :class:`~SFI.langevin.noise.NoiseModel` convention: it
    returns the increment already scaled by ``sqrt(2)``, i.e. ``sqrt(2) sum_s B_s xi_s``.
    The topology tables are read from ``extras`` at every step, so neighbour lists
    refreshed between simulation chunks (or per-frame schedules) are followed.
    """

    def __init__(self, families: Sequence[NoiseFamily], theta: Sequence[float], *, dim: int,
                 chunk_size: Optional[int] = None) -> None:
        super().__init__(n_fields=dim)
        if len(families) != len(theta):
            raise ValueError("one intensity per family")
        if any(t < 0 for t in theta):
            raise ValueError("intensities must be non-negative")
        self.families = tuple(families)
        self.theta = tuple(float(t) for t in theta)
        self._dispatchers = tuple(None if f.per_particle else self._dispatcher(f, dim, chunk_size) for f in self.families)

    @staticmethod
    def _dispatcher(fam: NoiseFamily, dim: int, chunk_size):
        """Per-slot interactor ``G_{s,k} xi_s`` dispatched over the family's hyperedges."""
        from SFI.statefunc import make_interactor
        from SFI.statefunc.nodes.interactions import FromExtrasHyperCSR

        def kernel(Xk, mask, extras):
            G = fam.amplitudes(Xk, mask, extras)  # (K, d, m)
            return jnp.einsum("kdm,m->kd", G, extras[_XI])[..., None]  # (K, d, 1)

        leaf = make_interactor(kernel, dim=dim, rank=1, Kmax=fam.Kmax, n_features=1, per_slot=True,
                               extras_keys=tuple(fam.extras_keys), spec_extras=(_XI,))
        return leaf.dispatch(FromExtrasHyperCSR(fam.indptr_key, fam.indices_key, edge_extras_keys=(_XI,)),
                             owners="all", chunk_size=chunk_size)

    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        if x.ndim != 2:
            raise ValueError(f"GraphNoise samples one (N, d) configuration, got x of shape {tuple(x.shape)}")
        inc = jnp.zeros_like(x)
        keys = jax.random.split(key, len(self.families))
        for fam, th, disp, k in zip(self.families, self.theta, self._dispatchers, keys):
            if th == 0.0:
                continue
            if fam.per_particle:
                inc = inc + jnp.sqrt(th) * jax.random.normal(k, x.shape, x.dtype)
                continue
            M = extras[fam.indptr_key].shape[-1] - 1
            xi = jax.random.normal(k, (M, fam.n_noise), x.dtype)
            y = disp(x, extras={**extras, _XI: xi})[..., 0]  # (N, d)
            inc = inc + jnp.sqrt(th) * y
        return jnp.sqrt(2.0) * inc

    def effective_D_per_site(self, extras: dict) -> Array:
        """Per-particle diagonal D of the per-particle (white) families only.

        The graph families' diagonal blocks depend on the state and the topology;
        simulator observables that use a constant ``D^{-1}`` are approximate.
        """
        D = jnp.zeros((self.dim, self.dim))
        for fam, th in zip(self.families, self.theta):
            if fam.per_particle:
                D = D + th * jnp.eye(self.dim)
        return D

    def __repr__(self) -> str:
        fams = ", ".join(f"{f.label}={t:g}" for f, t in zip(self.families, self.theta))
        return f"GraphNoise({fams}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Noise graph and D blocks
# ---------------------------------------------------------------------------
def csr_to_padded(indptr, indices, Kmax: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Host-side ``(M, K)`` member table and slot mask from a hyperedge CSR."""
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    sizes = np.diff(indptr)
    K = int(sizes.max()) if Kmax is None else int(Kmax)
    if sizes.max() > K:
        raise ValueError(f"a source has {sizes.max()} members > Kmax = {K}")
    M = len(sizes)
    tab = np.zeros((M, K), np.int32)
    msk = np.zeros((M, K), bool)
    for s in range(M):
        members = indices[indptr[s]: indptr[s + 1]]
        tab[s, : len(members)] = members
        msk[s, : len(members)] = True
    return tab, msk


@dataclass(frozen=True)
class NoiseGraph:
    """Source tables of several families and the union noise graph (host arrays).

    ``tabs[q], msks[q]``: ``(M_q, K_q)`` members and slot masks of family ``q``;
    ``edges``: ``(E, 2)`` particle pairs ``i < j`` sharing a source of some family;
    ``slot_edge[q]``: ``(M_q, K_q, K_q)`` edge index of slot pair ``(a, b)`` when
    ``tab[s, a] < tab[s, b]``, else ``E`` (dropped).
    """

    N: int
    tabs: Tuple[np.ndarray, ...]
    msks: Tuple[np.ndarray, ...]
    edges: np.ndarray
    slot_edge: Tuple[np.ndarray, ...] = field(repr=False)

    @property
    def E(self) -> int:
        return int(self.edges.shape[0])


def noise_graph(families: Sequence[NoiseFamily], extras, N: int) -> NoiseGraph:
    """Build the union noise graph of ``families`` from the (static) topology in ``extras``."""
    tabs, msks = [], []
    for fam in families:
        if fam.per_particle:
            tabs.append(np.arange(N, dtype=np.int32)[:, None])
            msks.append(np.ones((N, 1), bool))
        else:
            t, m = csr_to_padded(extras[fam.indptr_key], extras[fam.indices_key], fam.Kmax)
            tabs.append(t)
            msks.append(m)
    pairs = set()
    for t, m in zip(tabs, msks):
        for row, mrow in zip(t, m):
            mem = row[mrow]
            for a in range(len(mem)):
                for b in range(len(mem)):
                    if mem[a] < mem[b]:
                        pairs.add((int(mem[a]), int(mem[b])))
    edges = np.array(sorted(pairs), np.int32).reshape(-1, 2)
    eid = {p: e for e, p in enumerate(map(tuple, edges))}
    E = len(edges)
    slot_edge = []
    for t, m in zip(tabs, msks):
        M, K = t.shape
        se = np.full((M, K, K), E, np.int32)
        for s in range(M):
            for a in range(K):
                for b in range(K):
                    if m[s, a] and m[s, b] and t[s, a] < t[s, b]:
                        se[s, a, b] = eid[(int(t[s, a]), int(t[s, b]))]
        slot_edge.append(se)
    return NoiseGraph(N=N, tabs=tuple(tabs), msks=tuple(msks), edges=edges, slot_edge=tuple(slot_edge))


def family_amplitudes(fam: NoiseFamily, x: Array, tab: Array, msk: Array, extras) -> Array:
    """``(M, K, d, m)`` amplitudes of every source of ``fam`` at state ``x`` (N, d)."""
    ex = {k: extras[k] for k in fam.extras_keys}
    return jax.vmap(lambda Xk, m: fam.amplitudes(Xk, m, ex))(x[tab], msk)


def assemble_blocks(G: Array, tab: Array, msk: Array, slot_edge: Array, N: int, E: int) -> Tuple[Array, Array]:
    """``D`` of one family (unit intensity) on the noise graph.

    Returns ``(D_diag (N, d, d), D_edge (E, d, d))`` with ``D_edge[e] = D_{ij}`` for
    ``edges[e] = (i, j)``, ``i < j``; ``D_ji = D_ij^T``.
    """
    d = G.shape[-2]
    O = jnp.einsum("skdm,slem->sklde", G, G)  # (M, K, K, d, d): G_k G_l^T
    K = G.shape[1]
    diag = O[:, jnp.arange(K), jnp.arange(K)]  # (M, K, d, d)
    D_diag = jnp.zeros((N, d, d), G.dtype).at[tab.reshape(-1)].add(
        jnp.where(msk.reshape(-1)[:, None, None], diag.reshape(-1, d, d), 0.0))
    D_edge = jnp.zeros((E, d, d), G.dtype).at[slot_edge.reshape(-1)].add(O.reshape(-1, d, d), mode="drop")
    return D_diag, D_edge


def dense_D(D_diag: Array, D_edge: Array, edges) -> Array:
    """``(N d, N d)`` dense diffusion matrix from noise-graph blocks (small systems / tests)."""
    N, d, _ = D_diag.shape
    D = jnp.zeros((N, N, d, d), D_diag.dtype)
    D = D.at[jnp.arange(N), jnp.arange(N)].set(D_diag)
    e = jnp.asarray(edges)
    if e.shape[0]:
        D = D.at[e[:, 0], e[:, 1]].set(D_edge).at[e[:, 1], e[:, 0]].set(jnp.swapaxes(D_edge, -1, -2))
    return jnp.transpose(D, (0, 2, 1, 3)).reshape(N * d, N * d)


# ---------------------------------------------------------------------------
# Lattice conserved noise: one source per nearest-neighbour bond
# ---------------------------------------------------------------------------
def lattice_bonds(grid_shape: Sequence[int], *, bc: str = "pbc", axis: Optional[int] = None,
                  order: str = "C") -> np.ndarray:
    """Nearest-neighbour bonds ``(n_bonds, 2)`` of a regular grid, flat site indices.

    Bond ``(p, q)`` joins site ``p`` to its ``+1`` neighbour ``q`` along one axis
    (``axis``, or every axis in turn).  ``bc="pbc"``: bonds wrap around (an axis of
    length 2 then carries two bonds between the same pair, as the periodic Laplacian
    stencil counts that neighbour twice); ``bc="noflux"``: no bond crosses the
    boundary.  Axes of length 1 carry no bond.  ``order`` is the grid flattening order
    (``"C"``, as :func:`SFI.bases.spde.square_grid_extras` and the stencil operators).
    """
    if bc not in ("pbc", "noflux"):
        raise ValueError(f"bc must be 'pbc' or 'noflux', got {bc!r}")
    shape = tuple(int(n) for n in grid_shape)
    sites = np.arange(int(np.prod(shape))).reshape(shape, order=order)
    axes = range(len(shape)) if axis is None else [int(axis)]
    out = []
    for ax in axes:
        n = shape[ax]
        if n < 2:
            continue
        nb = np.roll(sites, -1, axis=ax)
        keep = np.ones(shape, bool)
        if bc == "noflux":  # drop the wrap-around bond from the last layer
            idx = [slice(None)] * len(shape)
            idx[ax] = n - 1
            keep[tuple(idx)] = False
        out.append(np.stack([sites[keep], nb[keep]], axis=1))
    return np.concatenate(out, 0).astype(np.int32) if out else np.zeros((0, 2), np.int32)


def conserved_bond_family(grid_shape: Sequence[int], dx=1.0, *, bc: str = "pbc", n_fields: int = 1,
                          fields: Optional[Sequence[int]] = None, mobility: Optional[Callable] = None,
                          axis: Optional[int] = None, prefix: str = "bond", order: str = "C",
                          label: Optional[str] = None) -> Tuple[NoiseFamily, dict]:
    r"""Conserved noise on a lattice as a :class:`NoiseFamily` of bond sources (EXPERIMENTAL).

    Each nearest-neighbour bond ``b = (p, q)`` carries one independent kick per conserved
    field, given ``+a`` to site ``p`` and ``-a`` to site ``q``, with
    ``a = sqrt(m(xbar_b) / (dV dx^2))`` (``dV`` the cell volume, ``dx`` the bond length).
    The current through each bond is what one site loses and the other gains, so the
    noise conserves ``sum_p phi_p`` exactly, for periodic and no-flux boundaries alike.
    At intensity ``theta = M`` and constant mobility the covariance is

    .. math::

       D = \frac{M}{\Delta V\,\Delta x^2}\, L_{\rm lattice} = -\frac{M}{\Delta V}\,\nabla^2_{\rm lattice},

    with ``L_lattice`` the graph Laplacian of the bonds (degree minus adjacency), i.e. the
    lattice discretisation of :math:`\nabla\cdot(\sqrt{2M}\,\vec\Lambda)` -- the same
    operator as the :class:`~SFI.bases.spde.Laplacian` stencil for the same ``bc``.  It
    agrees with the spectral :class:`~SFI.langevin.noise.ConservedNoise` (``sigma^2 = M``,
    multiplier ``|k|^2``) at small ``k`` and differs at the lattice scale, where the bond
    form has the lattice dispersion ``sum_a (4/dx_a^2) sin^2(k_a dx_a/2)``.

    Parameters
    ----------
    grid_shape : sequence of int
        Grid dimensions; sites are flattened in ``order``.
    dx : float or sequence of float
        Grid spacing.  Per-axis spacings differ between bond directions, so an
        anisotropic ``dx`` requires one family per axis (``axis=``).
    bc : {"pbc", "noflux"}
        Boundary condition (see :func:`lattice_bonds`).
    n_fields : int
        Number of fields per site (state dimension ``d``).
    fields : sequence of int, optional
        Fields that receive conserved noise (default: all); each gets its own
        independent bond kicks.  Combine with other families (e.g.
        :meth:`NoiseFamily.white`) for the remaining fields.
    mobility : callable, optional
        Field-dependent mobility shape ``m(xbar) -> scalar or (len(fields),)``, evaluated
        at the bond value ``xbar = (x_p + x_q) / 2`` (arithmetic mean of the two sites'
        ``(d,)`` states).  Negative values are clipped to zero.  The family intensity
        multiplies it: ``D = theta * sum_b m(xbar_b) ...``.  Default: ``m = 1``.  The
        simulator integrates the noise in the Ito sense.
    axis : int, optional
        Only the bonds along this axis.
    prefix : str
        Extras namespace of the bond table (``{prefix}/indptr``, ``{prefix}/indices``).
    order : str
        Grid flattening order.
    label : str, optional
        Family label (default ``prefix``).

    Returns
    -------
    family : NoiseFamily
    extras : dict
        The bond table (hyperedge CSR) the family reads; merge it into the process and
        collection extras next to :func:`~SFI.bases.spde.square_grid_extras`.
    """
    shape = tuple(int(n) for n in grid_shape)
    ndim = len(shape)
    dxv = np.broadcast_to(np.asarray(dx, float), (ndim,))
    if axis is None and not np.allclose(dxv, dxv[0]):
        raise ValueError("anisotropic dx: build one family per axis (axis=0, 1, ...), or use conserved_bond_noise")
    h = float(dxv[0] if axis is None else dxv[int(axis)])
    dV = float(np.prod(dxv))
    fld = tuple(range(n_fields)) if fields is None else tuple(int(f) for f in fields)
    if not fld or any(f < 0 or f >= n_fields for f in fld):
        raise ValueError(f"fields must be a non-empty subset of range({n_fields}), got {fields}")
    m = len(fld)
    a = 1.0 / np.sqrt(dV * h * h)
    sel = np.zeros((n_fields, m))
    sel[list(fld), np.arange(m)] = 1.0
    sgn_sel = jnp.asarray(np.array([1.0, -1.0])[:, None, None] * sel[None] * a)  # (2, d, m)

    def amplitude(Xk, mask, extras):
        if mobility is None:
            return sgn_sel
        xbar = 0.5 * (Xk[0] + Xk[1])
        s = jnp.sqrt(jnp.maximum(jnp.broadcast_to(jnp.asarray(mobility(xbar), Xk.dtype), (m,)), 0.0))
        return sgn_sel * s[None, None, :]

    bonds = lattice_bonds(shape, bc=bc, axis=axis, order=order)
    extras = {f"{prefix}/indptr": jnp.asarray(np.arange(0, 2 * len(bonds) + 1, 2, dtype=np.int32)),
              f"{prefix}/indices": jnp.asarray(bonds.reshape(-1))}
    fam = NoiseFamily(amplitude=amplitude, indptr_key=f"{prefix}/indptr", indices_key=f"{prefix}/indices",
                      n_noise=m, Kmax=2, label=label or prefix)
    return fam, extras


class LatticeConservedNoise(GraphNoise):
    r"""Conserved noise on a lattice, one independent kick per nearest-neighbour bond (EXPERIMENTAL).

    The lattice counterpart of the spectral :class:`~SFI.langevin.noise.ConservedNoise`,
    with the same ``sigma`` convention (mobility ``M = sigma**2``):

    .. math::

       D = \frac{\sigma^2}{\Delta V}\,\bigl(-\nabla^2_{\rm lattice}\bigr),

    the operator of the :class:`~SFI.bases.spde.Laplacian` stencil for the same ``bc``.
    Each bond gives ``+a`` to one site and ``-a`` to the other, so ``sum_p phi_p`` is
    conserved exactly for periodic **and** no-flux boundaries.  At large scales it agrees
    with ``ConservedNoise``; at the grid scale it has the lattice dispersion
    ``sum_a (4/dx_a^2) sin^2(k_a dx_a/2)`` instead of ``|k|^2``.  Unlike the spectral
    sampler it supports no-flux boundaries, noise on a subset of the fields and a
    field-dependent mobility, and it is a :class:`GraphNoise`, so
    :func:`~SFI.inference.graph_diffusion.estimate_graph_noise` recovers ``sigma**2``.

    The bond tables are carried by the noise (:attr:`extras`) and added to the extras it
    samples with: the process only needs :func:`~SFI.bases.spde.square_grid_extras`.

    Parameters
    ----------
    sigma : float
        Noise amplitude; the mobility is ``sigma**2``.
    grid_shape : sequence of int
        Grid dimensions (sites flattened in ``order``, as the stencil operators).
    dx : float or sequence of float
        Grid spacing.  An anisotropic spacing uses one bond family per axis.
    bc : {"pbc", "noflux"}
        Boundary condition, as :class:`~SFI.bases.spde.Laplacian`.
    n_fields : int
        Number of fields per site.
    fields : sequence of int, optional
        Fields that receive conserved noise (default: all), each with independent kicks.
    mobility : callable, optional
        Field-dependent mobility shape ``m(xbar) -> scalar or (len(fields),)`` at the bond
        value ``xbar = (x_p + x_q) / 2``; negative values are clipped to zero.  The
        mobility is ``sigma**2 * m``.  Integrated in the Ito sense by the simulator.
    prefix : str
        Extras namespace of the bond tables.
    order : str
        Grid flattening order.
    chunk_size : int, optional
        Bond chunk size of the dispatchers.

    Examples
    --------
    >>> noise = LatticeConservedNoise(0.3, grid_shape=(64, 64), dx=1.0, bc="noflux")
    >>> proc = OverdampedProcess(drift, D=noise, theta_F=theta)
    >>> proc.set_extras(extras_global=square_grid_extras(grid_shape=(64, 64), dx=1.0))
    """

    def __init__(self, sigma: float, *, grid_shape: Sequence[int], dx=1.0, bc: str = "pbc", n_fields: int = 1,
                 fields: Optional[Sequence[int]] = None, mobility: Optional[Callable] = None,
                 prefix: str = "bond", order: str = "C", chunk_size: Optional[int] = None) -> None:
        shape = tuple(int(n) for n in grid_shape)
        dxv = np.broadcast_to(np.asarray(dx, float), (len(shape),))
        kw = dict(bc=bc, n_fields=n_fields, fields=fields, mobility=mobility, order=order)
        if np.allclose(dxv, dxv[0]):
            built = [conserved_bond_family(shape, dxv, prefix=prefix, **kw)]
        else:
            built = [conserved_bond_family(shape, dxv, axis=a, prefix=f"{prefix}/axis{a}", **kw)
                     for a in range(len(shape)) if shape[a] > 1]
        self.extras = {}
        for _, ex in built:
            self.extras.update(ex)
        self.sigma = float(sigma)
        self.grid_shape, self.bc = shape, bc
        fams = [f for f, _ in built]
        super().__init__(fams, [self.sigma**2] * len(fams), dim=n_fields, chunk_size=chunk_size)

    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        return super().sample(key, x, {**self.extras, **(extras or {})})

    def __repr__(self) -> str:
        return f"LatticeConservedNoise(sigma={self.sigma:g}, grid_shape={self.grid_shape}, bc={self.bc!r}, dim={self.dim})"
