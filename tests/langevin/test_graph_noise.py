"""GraphNoise (experimental): noise shared between particles through sources on a hypergraph."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.langevin import UnderdampedProcess
from SFI.langevin.graph_noise import (
    GraphNoise,
    NoiseFamily,
    assemble_blocks,
    dense_D,
    family_amplitudes,
    noise_graph,
)
from SFI.statefunc import make_basis

N, d = 8, 2
RING = [[i, (i + 1) % N] for i in range(N)]
TRI = [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [2, 3, 4]]  # overlapping cells: shared edges


def _csr(rows):
    return (jnp.asarray(np.concatenate([[0], np.cumsum([len(r) for r in rows])]), jnp.int32),
            jnp.asarray(np.concatenate(rows), jnp.int32))


def _extras():
    ex = {}
    ex["pair/indptr"], ex["pair/indices"] = _csr(RING)
    ex["tri/indptr"], ex["tri/indices"] = _csr(TRI)
    return ex


def _families():
    sgn = jnp.array([1.0, -1.0])
    pair = NoiseFamily(lambda Xk, m, e: sgn[:, None, None] * jnp.eye(d)[None] / jnp.sqrt(2.0),
                       "pair/indptr", "pair/indices", n_noise=d, Kmax=2, label="pair")

    def cell_amp(Xk, m, e):  # state-dependent: members pushed along their offset from the centroid
        c = jnp.sum(jnp.where(m[:, None], Xk, 0.0), 0) / jnp.maximum(m.sum(), 1)
        r = Xk - c
        return jnp.eye(d)[None] / 3.0 + 0.2 * jnp.einsum("kd,e->kde", r, jnp.ones(d))

    cell = NoiseFamily(cell_amp, "tri/indptr", "tri/indices", n_noise=d, Kmax=3, label="cell")
    return [pair, cell, NoiseFamily.white(d)]


THETA = [0.3, 0.6, 0.05]


def _dense(fams, theta, x, ex):
    g = noise_graph(fams, ex, N)
    D = 0.0
    for f, th, tb, mk, se in zip(fams, theta, g.tabs, g.msks, g.slot_edge):
        G = family_amplitudes(f, x, jnp.asarray(tb), jnp.asarray(mk), ex)
        Dg, De = assemble_blocks(G, jnp.asarray(tb), jnp.asarray(mk), jnp.asarray(se), N, g.E)
        D = D + th * np.asarray(dense_D(Dg, De, g.edges))
    return D


def _brute_dense(fams, theta, x, ex):
    """D = sum_s B_s B_s^T assembled source by source, without the noise graph."""
    D = np.zeros((N * d, N * d))
    x = np.asarray(x)
    for f, th in zip(fams, theta):
        if f.per_particle:
            D += th * np.eye(N * d)
            continue
        ptr, idx = np.asarray(ex[f.indptr_key]), np.asarray(ex[f.indices_key])
        for s in range(len(ptr) - 1):
            mem = idx[ptr[s]:ptr[s + 1]]
            K = len(mem)
            Xk = np.zeros((f.Kmax, d))
            Xk[:K] = x[mem]
            msk = np.arange(f.Kmax) < K
            G = np.asarray(f.amplitudes(jnp.asarray(Xk), jnp.asarray(msk), {}))
            Bs = np.zeros((N * d, f.n_noise))
            for a, i in enumerate(mem):
                Bs[i * d:(i + 1) * d] += G[a]
            D += th * Bs @ Bs.T
    return D


def test_noise_graph_blocks_match_source_sum():
    ex, fams = _extras(), _families()
    x = jnp.asarray(np.random.default_rng(0).standard_normal((N, d)), jnp.float32)
    np.testing.assert_allclose(_dense(fams, THETA, x, ex), _brute_dense(fams, THETA, x, ex), atol=1e-5)
    g = noise_graph(fams, ex, N)
    assert {tuple(e) for e in g.edges} == {tuple(sorted(p)) for r in RING + TRI for p in
                                           [(a, b) for a in r for b in r if a != b]}


def test_sample_covariance_is_2D():
    ex, fams = _extras(), _families()
    x = jnp.asarray(np.random.default_rng(1).standard_normal((N, d)), jnp.float32)
    gn = GraphNoise(fams, THETA, dim=d)
    n = 40000
    S = jax.vmap(lambda k: gn.sample(k, x, ex))(jax.random.split(jax.random.PRNGKey(0), n))
    C = np.cov(np.asarray(S).reshape(n, N * d).T)
    D = _dense(fams, THETA, x, ex)
    assert np.abs(C - 2 * D).max() < 0.05 * np.abs(2 * D).max()


def test_rejects_negative_intensities_and_mismatched_lengths():
    fams = _families()
    with pytest.raises(ValueError):
        GraphNoise(fams, [0.1, -0.1, 0.1], dim=d)
    with pytest.raises(ValueError):
        GraphNoise(fams, [0.1, 0.1], dim=d)


def test_underdamped_process_accepts_a_noise_model():
    """Velocity increments of the UD integrator carry the shared-noise covariance 2 D dt."""
    ex = _extras()
    fams = _families()[:1] + _families()[2:]  # pair + white
    theta = [0.4, 0.05]
    gn = GraphNoise(fams, theta, dim=d)
    F = make_basis(lambda x, v: jnp.stack([x, v], -1), dim=d, rank=1, n_features=2, needs_v=True)
    proc = UnderdampedProcess(F, D=gn, theta_F=jnp.array([-1.0, -0.5]))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.zeros((N, d), jnp.float32))
    dt = 0.02
    X = np.asarray(proc.simulate(dt=dt, Nsteps=6000, key=jax.random.PRNGKey(2), oversampling=4).datasets[0].X)
    u = (X[2:] - 2 * X[1:-1] + X[:-2]).reshape(-1, N * d)  # Cov = (2/3) 2 D dt^3 for velocity noise
    C = np.cov(u.T) / ((2.0 / 3.0) * 2 * dt**3)
    D = _dense(fams, theta, jnp.zeros((N, d)), ex)
    assert np.abs(C - D).max() < 0.1 * np.abs(D).max()
