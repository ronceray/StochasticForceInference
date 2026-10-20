"""
Janus-colloid tracks — data access and observables (example-local).

Helpers for ``janus_colloids_demo.py``: download the experimental tracks
distributed with Hem et al., *Soft Matter* (2025), turn them into dense
arrays with a validity mask, build per-frame neighbour lists, and measure
the collective observables the paper compares with simulations.

These helpers are example-specific and not part of the public ``SFI`` API.
"""
from __future__ import annotations

import hashlib
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # numpy < 2
    from numpy.core.multiarray import _reconstruct

# First 100 frames of the flocking dataset ("set A", Iwasawa, Nishiguchi & Sano,
# Phys. Rev. Research 3, 043104, 2021), as distributed with the code of the
# paper (github.com/jhem272/sfiabp_py, pinned commit).
DATA_URL = ("https://raw.githubusercontent.com/jhem272/sfiabp_py/"
            "7bf78a5298c87cbd40a211d61e26e65525a1b6d2/tutorial/data/"
            "20180927ordered_df_pv_m_100f.pkl")
DATA_SHA256 = "7d7ef2460ceee5f83a4dc42df4f4ba60849fe6d32c6ed18089afb13c319ec542"


def _create_array(f, args, state, npdict=None):
    """Stand-in for ``dill._dill._create_array`` (rebuilds one numpy array)."""
    a = f(*args)
    a.__setstate__(state)
    if npdict:
        a.__dict__.update(npdict)
    return a


class _ArraysOnlyUnpickler(pickle.Unpickler):
    """Unpickler that only rebuilds numpy arrays: no other code can run."""

    def find_class(self, module, name):
        if (module, name) == ("dill._dill", "_create_array"):
            return _create_array
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return _reconstruct
        if (module, name) in (("numpy", "ndarray"), ("numpy", "dtype")):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_set_a(cache_dir: Path):
    """Download (once) and read the set-A tracks.

    Returns
    -------
    frames : list of arrays ``(N_t, 4)``
        Per frame: ``x, y`` (µm), heading ``φ`` (rad, in ``[0, 2π)``) and
        the integer track label of each detected particle.
    dt : float
        Frame interval (s).
    fov : ndarray ``(2,)``
        Width and height of the field of view (µm).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "janus_set_a_100f.pkl"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        path.unlink()
        raise RuntimeError("checksum mismatch for the downloaded tracks; removed the file")
    with open(path, "rb") as fh:
        d = _ArraysOnlyUnpickler(fh).load()
    fov = np.array([d["xframelim"][1], d["yframelim"][1]], dtype=float)
    return [np.asarray(f, dtype=float) for f in d["X_Raw"]], float(d["dtframe"]), fov


def tracks_to_arrays(frames):
    """Dense ``(T, N, 3)`` states, one column per track label, plus the mask.

    Absent particles are NaN with ``mask = False``.  Headings are unwrapped
    along each track so that heading increments are continuous.
    """
    labels = np.unique(np.concatenate([f[:, 3] for f in frames]).astype(int))
    slot = {lab: i for i, lab in enumerate(labels)}
    T, N = len(frames), len(labels)
    X = np.full((T, N, 3), np.nan)
    mask = np.zeros((T, N), bool)
    for t, f in enumerate(frames):
        s = np.array([slot[int(lab)] for lab in f[:, 3]])
        X[t, s] = f[:, :3]
        mask[t, s] = True
    for n in range(N):
        X[mask[:, n], n, 2] = np.unwrap(X[mask[:, n], n, 2])
    return X, mask


def per_frame_neighbours(X, mask, radius):
    """Per-frame CSR neighbour lists over the present particles, as extras."""
    from SFI.trajectory import TimeSeriesExtra
    from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr

    T, N = mask.shape
    indptrs, indices = [], []
    for t in range(T):
        present = np.flatnonzero(mask[t])
        ip, ix = build_neighbor_csr(X[t, present, :2], radius)
        counts = np.zeros(N, int)
        counts[present] = np.diff(ip)
        indptrs.append(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
        indices.append(present[ix].astype(np.int32))
    nnz = max(len(ix) for ix in indices)
    indices = [pad_neighbor_csr(ip, ix, nnz)[1] for ip, ix in zip(indptrs, indices)]
    return {"indptr": TimeSeriesExtra(np.stack(indptrs)), "indices": TimeSeriesExtra(np.stack(indices))}


def interior_rows(X, mask, fov, margin, lag=2):
    """Fit mask: particles at least ``margin`` from the edge ``lag`` frames earlier."""
    x, y = X[..., 0], X[..., 1]
    with np.errstate(invalid="ignore"):
        inner = mask & (x > margin) & (x < fov[0] - margin) & (y > margin) & (y < fov[1] - margin)
    rows = np.zeros_like(mask)
    rows[lag:] = inner[:-lag]
    return rows & mask


# ---- collective observables ------------------------------------------------

def _tiles(frame, fov, side):
    """Tile index of each particle (square tiles of the given side) and tile count."""
    nx, ny = int(fov[0] // side), int(fov[1] // side)
    xy = np.mod(frame[:, :2], fov)
    ix, iy = (xy[:, 0] // side).astype(int), (xy[:, 1] // side).astype(int)
    ok = (ix < nx) & (iy < ny)
    return ix * ny + iy, ok, nx * ny


def polar_order_vs_area(frames, fov, sides):
    r"""Polar order :math:`C(S) = \langle|\langle e^{i\varphi}\rangle_{\rm tile}|\rangle` for tiles of area side²."""
    out = []
    for side in sides:
        vals = []
        for f in frames:
            tile, ok, n = _tiles(f, fov, side)
            z = np.exp(1j * f[ok, 2])
            s = np.bincount(tile[ok], z.real, n) + 1j * np.bincount(tile[ok], z.imag, n)
            cnt = np.bincount(tile[ok], None, n)
            vals.append(np.abs(s[cnt > 0]) / cnt[cnt > 0])
        out.append(np.mean(np.concatenate(vals)))
    return np.array(out)


def polarity_autocorrelation(phi, lags):
    r""":math:`f_p(\tau) = \langle \cos[\varphi(t+\tau) - \varphi(t)]\rangle` from headings ``(T, N)`` (NaN = absent)."""
    return np.array([np.nanmean(np.cos(phi[k:] - phi[:len(phi) - k])) for k in lags])


def number_fluctuations(frames, fov, sides):
    """Mean and standard deviation of the particle number in square tiles."""
    mean, std = [], []
    for side in sides:
        counts = []
        for f in frames:
            tile, ok, n = _tiles(f, fov, side)
            counts.append(np.bincount(tile[ok], None, n))
        counts = np.array(counts)
        mean.append(counts.mean())
        std.append(counts.std(axis=0).mean())
    return np.array(mean), np.array(std)


def body_frame_density(frames, fov, *, half_width=25.0, bins=50, periodic=False):
    r"""Neighbour density around a particle, in its own frame (:math:`x_\parallel` along its heading).

    Normalised by the mean density; with open boundaries only particles
    farther than the window from the edges are used as centres.
    """
    reach = half_width * np.sqrt(2)
    H = np.zeros((bins, bins))
    n_centres, density = 0, 0.0
    for f in frames:
        xy = np.mod(f[:, :2], fov) if periodic else f[:, :2]
        pairs = cKDTree(xy, boxsize=fov if periodic else None).query_pairs(reach, output_type="ndarray")
        pairs = np.concatenate([pairs, pairs[:, ::-1]])
        if periodic:
            n_centres += len(xy)
        else:
            inner = np.all((xy > reach) & (xy < fov - reach), axis=1)
            pairs = pairs[inner[pairs[:, 0]]]
            n_centres += inner.sum()
        d = xy[pairs[:, 1]] - xy[pairs[:, 0]]
        if periodic:
            d -= fov * np.round(d / fov)
        c, s = np.cos(f[pairs[:, 0], 2]), np.sin(f[pairs[:, 0], 2])
        H += np.histogram2d(c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], bins=bins,
                            range=[[-half_width, half_width]] * 2)[0]
        density += len(xy) / np.prod(fov)
    return H / (n_centres * density / len(frames) * (2 * half_width / bins) ** 2)


def relative_orientations(frames, r_lo, r_hi):
    r"""Angles :math:`(\theta_1, \theta_2)` of all ordered pairs with :math:`r_{lo} \le r < r_{hi}`.

    :math:`\theta_k = \varphi_k - \arg(\mathbf r_1 - \mathbf r_2)`, the
    convention of the paper (particle 2 faces particle 1 at :math:`\theta_2 = 0`).
    """
    t1, t2 = [], []
    for f in frames:
        pairs = cKDTree(f[:, :2]).query_pairs(r_hi, output_type="ndarray")
        pairs = np.concatenate([pairs, pairs[:, ::-1]])
        d = f[pairs[:, 0], :2] - f[pairs[:, 1], :2]
        r = np.hypot(d[:, 0], d[:, 1])
        keep = r >= r_lo
        phi_r = np.arctan2(d[keep, 1], d[keep, 0])
        t1.append(f[pairs[keep, 0], 2] - phi_r)
        t2.append(f[pairs[keep, 1], 2] - phi_r)
    return np.mod(np.concatenate(t1), 2 * np.pi), np.mod(np.concatenate(t2), 2 * np.pi)



def tracks_collection(frames, dt, fov, *, cutoff, margin, lag=2):
    """Experimental tracks as a :class:`~SFI.TrajectoryCollection`, ready for pair inference.

    One column per track label (absent particles masked), headings unwrapped,
    per-frame neighbour lists of radius ``cutoff`` + a Verlet skin measured on
    the tracks (twice the largest displacement over ``lag`` frames), and a
    fit mask (``dynamic_mask``) keeping particles farther than ``margin`` from
    the edges ``lag`` frames before the row; the others stay visible as
    neighbours.
    """
    from SFI import TrajectoryCollection, TrajectoryDataset

    X, mask = tracks_to_arrays(frames)
    skin = 2.0 * np.nanmax(np.hypot(*np.moveaxis(X[lag:, :, :2] - X[:-lag, :, :2], -1, 0)))
    ds = TrajectoryDataset.from_arrays(X=X, dt=dt, mask=mask, dynamic_mask=interior_rows(X, mask, fov, margin, lag),
                                       extras_global=per_frame_neighbours(X, mask, cutoff + skin))
    return TrajectoryCollection.from_dataset(ds)


# ---- the generic pair-interaction basis of Hem et al. ----------------------

def tapered_polyexp_kernels(*, cutoff, onset, degrees=range(0, 16, 2), r0=1.0, taper=4.0):
    r"""Radial functions :math:`g_l(r) = (r/r_0)^l e^{-r/r_0}/l!` of the paper's code, tapered.

    The window rises smoothly over ``onset`` (a pair of distances) and falls
    to zero, with its slope, over the last ``taper`` before ``cutoff``.
    Returns ``(callable, label)`` pairs for :func:`~SFI.bases.pairs.scalar_pair_basis`.
    """
    import jax.numpy as jnp
    from math import lgamma

    def step(u):
        u = jnp.clip(u, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    lo, hi = onset

    def g(l):
        c = lgamma(l + 1)
        return lambda r: (jnp.exp(l * jnp.log(r / r0) - r / r0 - c)
                          * step((r - lo) / (hi - lo)) * step((cutoff - r) / taper))
    return [(g(l), f"g{l}") for l in degrees]


def angular_modes(Na):
    """``(k1, k2)`` with ``|k1| + |k2| <= Na``, one per pair ``±(k1, k2)``."""
    return [(n - abs(k), k) for n in range(Na + 1) for k in (range(-n + 1, n + 1) if n else [0])]


def _relative_frame(XK, extras, periodic):
    import jax.numpy as jnp
    from SFI.bases.pairs import pbc_displacement

    x1, x2 = XK[0], XK[1]
    d = -pbc_displacement(x2[:2], x1[:2], extras["box"][:2]) if periodic else x1[:2] - x2[:2]
    phi_r = jnp.arctan2(d[1], d[0])
    return phi_r, x1[2] - phi_r, x2[2] - phi_r, jnp.sqrt(jnp.sum(d ** 2))


def orientation_features(Na, *, box=None, sector=None):
    r"""Angular part of the paper's basis, as a rank-1 pair interactor on states ``(x, y, φ)``.

    Headings :math:`\theta_{1,2}` are measured from :math:`\mathbf r_1 - \mathbf r_2`;
    :math:`\cos(k_1\theta_1 + k_2\theta_2)` along :math:`\hat e_r`, the sines along
    :math:`\hat e_\theta` and on the heading (torque) — the parity-symmetric
    Fourier series of order ``Na``.  ``sector="translation"`` keeps the
    :math:`\hat e_r, \hat e_\theta` features, ``sector="rotation"`` the torque ones.
    """
    import jax.numpy as jnp
    from SFI.statefunc import make_interactor

    cos_m = angular_modes(Na)
    sin_m = [m for m in cos_m if m != (0, 0)]
    kc, ks = jnp.array(cos_m, float).reshape(-1, 2), jnp.array(sin_m, float).reshape(-1, 2)

    def local(XK, *, extras=None):
        phi_r, t1, t2, _ = _relative_frame(XK, extras, box == "extras")
        e_r = jnp.array([jnp.cos(phi_r), jnp.sin(phi_r), 0.0])
        e_t = jnp.array([-jnp.sin(phi_r), jnp.cos(phi_r), 0.0])
        e_w = jnp.array([0.0, 0.0, 1.0])
        c = jnp.cos(kc[:, 0] * t1 + kc[:, 1] * t2)
        s = jnp.sin(ks[:, 0] * t1 + ks[:, 1] * t2)
        parts = {"translation": [e_r[:, None] * c, e_t[:, None] * s], "rotation": [e_w[:, None] * s]}
        return jnp.concatenate(sum((parts[k] for k in keep), []), axis=1)

    keep = ("translation", "rotation") if sector is None else (sector,)
    names = {"translation": [f"vr cos({a}θ1{b:+d}θ2)" for a, b in cos_m] + [f"vθ sin({a}θ1{b:+d}θ2)" for a, b in sin_m],
             "rotation": [f"ω sin({a}θ1{b:+d}θ2)" for a, b in sin_m]}
    labels = sum((names[k] for k in keep), [])
    return make_interactor(local, dim=3, rank=1, K=2, n_features=len(labels), labels=labels,
                           extras_keys=("box",) if box == "extras" else ())


def interaction_reader(interactor):
    r"""Evaluate a pair interactor's features at given ``(r, θ1, θ2)``.

    Returns ``read(coef, cov, r, t1, t2) -> (value, stderr)``, each of shape
    ``(..., 3)`` for :math:`(v_r, v_\theta, \omega)`: the features are evaluated on a
    two-particle configuration (particle 2 at :math:`(-r, 0)`, so
    :math:`\hat e_r = \hat x`) and contracted with the coefficients.
    """
    import jax.numpy as jnp

    probe = interactor.dispatch_pairs(return_as="basis", exclude_self=True, owners="focal", reducer="sum")

    def read(coef, cov, r, t1, t2):
        r, t1, t2 = np.broadcast_arrays(np.asarray(r, float), t1, t2)
        X2 = np.zeros(r.shape + (2, 3))
        X2[..., 0, 2], X2[..., 1, 0], X2[..., 1, 2] = t1, -r, t2
        Phi = np.asarray(probe(jnp.asarray(X2.reshape(-1, 2, 3))))[:, 0]
        value = Phi @ np.ravel(coef)
        err = (np.zeros_like(value) if cov is None
               else np.sqrt(np.einsum("nca,ab,ncb->nc", Phi, np.asarray(cov), Phi)))
        return value.reshape(r.shape + (3,)), err.reshape(r.shape + (3,))

    return read


def fitted_pair_force(kernels, Na, coef):
    """The fitted series as a single periodic pair feature — fast to simulate."""
    import jax.numpy as jnp
    from SFI.statefunc import make_interactor

    fns = [fn for fn, _ in kernels]
    cos_m = angular_modes(Na)
    sin_m = [m for m in cos_m if m != (0, 0)]
    kc, ks = jnp.array(cos_m, float).reshape(-1, 2), jnp.array(sin_m, float).reshape(-1, 2)
    A = jnp.asarray(np.ravel(coef).reshape(len(fns), -1))
    nc, ns = len(cos_m), len(sin_m)

    def local(XK, *, extras=None):
        phi_r, t1, t2, r = _relative_frame(XK, extras, True)
        w = jnp.stack([g(r) for g in fns]) @ A
        s = jnp.sin(ks[:, 0] * t1 + ks[:, 1] * t2)
        vr = w[:nc] @ jnp.cos(kc[:, 0] * t1 + kc[:, 1] * t2)
        vt, om = w[nc:nc + ns] @ s, w[nc + ns:] @ s
        return jnp.array([vr * jnp.cos(phi_r) - vt * jnp.sin(phi_r),
                          vr * jnp.sin(phi_r) + vt * jnp.cos(phi_r), om])[:, None]

    return make_interactor(local, dim=3, rank=1, K=2, n_features=1, extras_keys=("box",),
                           labels=("fitted pair force",)).dispatch_pairs_from_extras(
        indptr_key="indptr", indices_key="indices", return_as="basis")


def simulate_fitted(inf, one_body, pair_block, kernels, Na, x0, duration, *, D, box, dt, key, cutoff,
                    sigma=3.17, eps_steric=800.0, n_sub=10, skin=6.0):
    """Run a fitted model from ``x0`` in a periodic box; one saved frame per ``dt``.

    The fitted pair series is summed into one feature (:func:`fitted_pair_force`)
    and a soft steric repulsion ``V = (ε/2)(1 - r/σ)²`` (as in the paper's
    simulations) keeps particles from overlapping.
    """
    import jax.numpy as jnp
    from SFI.bases.pairs import radial_pair_basis
    from SFI.langevin import OverdampedProcess
    from SFI.langevin.chunked import simulate_chunked
    from SFI.statefunc import Basis
    from SFI.utils.neighbors import make_neighbor_extras

    steric = radial_pair_basis([(lambda r: jnp.where(r < sigma, 1.0 - r / sigma, 0.0), "steric")], dim=3,
                               box="extras", spatial_dims=slice(0, 2), embed_dim=3, embed_axes=[0, 1]
                               ).dispatch_pairs_from_extras(indptr_key="indptr", indices_key="indices",
                                                            return_as="basis")
    c_one = [float(np.ravel(inf.coeff_block(b)[0])[0]) for b in one_body]
    F = Basis.stack([*one_body, fitted_pair_force(kernels, Na, inf.coeff_block(pair_block)[0]), steric])
    proc = OverdampedProcess(F=F, D=jnp.asarray(D), theta_F=jnp.array(c_one + [1.0, -eps_steric / sigma]),
                             extras_global={"box": jnp.asarray(box),
                                            **make_neighbor_extras(np.mod(x0[:, :2], box), cutoff + skin, box)})
    proc.initialize(jnp.asarray(x0))
    coll = simulate_chunked(proc, dt=dt / n_sub, Nsteps=int(round(duration / dt)) * n_sub, key=key,
                            cutoff=cutoff, skin=skin, box=np.asarray(box), rebuild_every=n_sub,
                            spatial_dims=slice(0, 2))
    return np.concatenate([np.asarray(x0)[None], np.asarray(coll.to_array(axis="time"))[n_sub - 1::n_sub]])


def residual_noise(inf, coll):
    r"""Measurement noise and diffusion from the residuals of a force fit.

    SFI's :math:`\hat\Lambda = -\langle \Delta X_t \Delta X_{t-1}\rangle` is
    biased by :math:`-\langle F_t F_{t-1}\rangle\Delta t^2` when the drift is
    persistent (self-propulsion).  On the residuals
    :math:`r_t = \Delta X_t - \hat F(X_t)\Delta t` the drift is gone:
    :math:`\Lambda = -\langle r_t r_{t-1}\rangle` and
    :math:`D = \langle r_t r_t\rangle / 2\Delta t - \Lambda/\Delta t`
    (symmetrised; fitted rows only).  Returns ``(Lambda, D)``, ``(d, d)`` each.
    """
    import jax.numpy as jnp
    from SFI.trajectory import TimeSeriesExtra

    lag1, sq, n1, n0, dts = 0.0, 0.0, 0, 0, []
    for ds in coll.datasets:
        X = np.asarray(ds.X)
        dt = float(np.mean(np.asarray(ds.dt)))
        rows = np.asarray(ds.dynamic_mask if ds.dynamic_mask is not None else ds.mask, bool)
        valid = np.asarray(ds.mask, bool)
        F = []
        for t in range(X.shape[0] - 1):
            extras = {k: jnp.asarray(v.data[t] if isinstance(v, TimeSeriesExtra) else v)
                      for k, v in (ds.extras_global or {}).items()}
            F.append(np.asarray(inf.force_inferred(jnp.asarray(np.nan_to_num(X[t])), extras=extras)))
        r = X[1:] - X[:-1] - np.asarray(F) * dt
        ok = rows[:-1] & valid[1:] & valid[:-1]
        r = np.where(ok[..., None], r, 0.0)
        both = ok[1:] & ok[:-1]
        lag1 = lag1 + np.einsum("tnm,tnk->mk", np.where(both[..., None], r[1:], 0.0), r[:-1])
        sq = sq + np.einsum("tnm,tnk->mk", r, r)
        n1, n0 = n1 + both.sum(), n0 + ok.sum()
        dts.append(dt)
    dt = float(np.mean(dts))
    L = -lag1 / n1
    L = 0.5 * (L + L.T)
    return L, sq / n0 / (2 * dt) - L / dt


def periodic_tracks_collection(X, dt, box, *, cutoff, lag=2):
    """Simulated tracks in a periodic box, with per-frame Verlet lists."""
    from SFI import TrajectoryCollection
    from SFI.trajectory import TimeSeriesExtra
    from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr

    d = X[lag:, :, :2] - X[:-lag, :, :2]
    skin = 2.0 * np.hypot(*np.moveaxis(d - box * np.round(d / box), -1, 0)).max()
    lists = [build_neighbor_csr(np.mod(f[:, :2], box), cutoff + skin, box) for f in X]
    nnz = max(len(ix) for _, ix in lists)
    extras = {"box": np.asarray(box), "indptr": TimeSeriesExtra(np.stack([ip for ip, _ in lists])),
              "indices": TimeSeriesExtra(np.stack([pad_neighbor_csr(ip, ix, nnz)[1] for ip, ix in lists]))}
    return TrajectoryCollection.from_arrays(X=X, dt=dt, extras_global=extras)


def sample_pair_configurations(X, box, cutoff, n, rng):
    """``n`` random ``(r, θ1, θ2)`` of pairs closer than ``cutoff`` in periodic tracks."""
    out = []
    for f in X:
        p = cKDTree(np.mod(f[:, :2], box), boxsize=box).query_pairs(cutoff, output_type="ndarray")
        p = np.concatenate([p, p[:, ::-1]])
        d = f[p[:, 0], :2] - f[p[:, 1], :2]
        d -= box * np.round(d / box)
        phi_r = np.arctan2(d[:, 1], d[:, 0])
        out.append(np.stack([np.hypot(d[:, 0], d[:, 1]), f[p[:, 0], 2] - phi_r, f[p[:, 1], 2] - phi_r], 1))
    out = np.concatenate(out)
    return out[rng.choice(len(out), n, replace=False)]

__all__ = [
    "DATA_URL", "load_set_a", "tracks_to_arrays", "per_frame_neighbours", "interior_rows",
    "tracks_collection", "tapered_polyexp_kernels", "angular_modes", "orientation_features",
    "interaction_reader", "fitted_pair_force", "simulate_fitted", "residual_noise", "periodic_tracks_collection", "sample_pair_configurations",
    "polar_order_vs_area", "polarity_autocorrelation", "number_fluctuations",
    "body_frame_density", "relative_orientations",
]
