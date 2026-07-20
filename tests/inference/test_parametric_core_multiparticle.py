"""Multiparticle recovery via the SAME infer_force (no separate route)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_collection(X, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _per_particle_linear_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x
    return make_psf(f, dim=d, rank=1, n_features=1,
                    params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)])


# ── Phase 1: non-interacting (per-particle force) ──────────────────────


def test_infer_force_non_interacting_multiparticle():
    """N independent OU particles, one shared per-particle force → recover −k."""
    from SFI.inference.overdamped import OverdampedLangevinInference

    d, N, T, dt = 2, 8, 4000, 0.01
    k = np.array([[1.0, 0.1], [0.1, 1.5]])
    D = np.diag([0.5, 0.3])
    s2D = np.sqrt(2 * D)
    rng = np.random.default_rng(3)
    X = np.zeros((T, N, d))
    X[0] = rng.normal(0, 0.5, size=(N, d))
    for t in range(T - 1):
        X[t + 1] = (X[t] + np.einsum("ij,nj->ni", -k, X[t]) * dt
                    + (rng.normal(size=(N, d)) @ s2D.T) * np.sqrt(dt))

    coll = _make_collection(X, dt)
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_per_particle_linear_psf(d), n_substeps=4, max_outer=5)
    A = np.asarray(inf.force_coefficients_full).reshape(d, d)

    rel = np.linalg.norm(A - (-k)) / np.linalg.norm(k)
    # diagonal recovered to <1%; the small off-diagonal coupling (0.1) carries
    # ~10% single-realization sampling error and dominates the relative norm.
    assert rel < 0.10, f"A not recovered (rel={rel:.3e}):\n{A}\nvs {-k}"


# ── Phase 2: interacting (harmonic trap + pairwise spring) ─────────────


def _ud_linear_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v
    return make_psf(f, dim=d, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def test_infer_force_ud_non_interacting_multiparticle():
    """N independent UD harmonic oscillators → recover (k, γ) via the same engine."""
    from SFI.inference.underdamped import UnderdampedLangevinInference

    N, T, dt = 6, 8000, 0.05
    k, gamma, D = 1.0, 0.5, 0.1
    s = np.sqrt(2 * D)
    rng = np.random.default_rng(5)
    X = np.zeros((T, N, 1)); V = np.zeros((T, N, 1))
    X[0] = rng.normal(0, 0.3, size=(N, 1)); V[0] = rng.normal(0, 0.3, size=(N, 1))
    for t in range(T - 1):
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k * X[t] - gamma * V[t]) * dt + s * rng.normal(size=(N, 1)) * np.sqrt(dt)

    coll = _make_collection(X, dt)
    inf = UnderdampedLangevinInference(coll)
    inf.infer_force(_ud_linear_psf(1), n_substeps=4, max_outer=6)
    p = _ud_linear_psf(1).unflatten_params(inf.force_coefficients_full)
    k_hat = float(np.asarray(p["k"])); g_hat = float(np.asarray(p["gamma"]))

    assert abs(k_hat - k) / k < 0.15, f"k={k_hat}"
    assert abs(g_hat - gamma) / gamma < 0.30, f"gamma={g_hat}"


def _build_interacting_model(dim=2):
    from SFI.bases.pairs import parametric_radial_kernel, pair_direction
    from SFI.statefunc import make_psf

    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x, params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r, params={"k_spring": ()}, dim=dim)
    return F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")


def test_infer_force_interacting_multiparticle():
    """Trap + pairwise spring: recover (k_trap, k_spring) via the interacting path."""
    from SFI.inference.overdamped import OverdampedLangevinInference
    from SFI.langevin import OverdampedProcess

    dim, N = 2, 4
    F_psf = _build_interacting_model(dim)
    proc = OverdampedProcess(F_psf, D=0.3)
    proc.set_params(theta_F={"k_trap": 1.0, "k_spring": 0.5})
    proc.initialize(random.normal(random.PRNGKey(0), (N, dim)) * 0.5)
    coll = proc.simulate(dt=0.01, Nsteps=4000, key=random.PRNGKey(1), prerun=100, oversampling=5)

    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_build_interacting_model(dim), n_substeps=4, max_outer=5)
    p = _build_interacting_model(dim).unflatten_params(inf.force_coefficients_full)
    k_trap = float(np.asarray(p["k_trap"])); k_spring = float(np.asarray(p["k_spring"]))

    assert abs(k_trap - 1.0) < 0.15, f"k_trap={k_trap}"
    assert abs(k_spring - 0.5) < 0.15, f"k_spring={k_spring}"


def _build_ud_interacting(dim=2):
    from SFI.bases.pairs import parametric_radial_kernel, pair_direction
    from SFI.statefunc import make_psf

    F_trap = make_psf(
        lambda x, *, v, params: -params["k_trap"] * x - params["gamma"] * v,
        params={"k_trap": (), "gamma": ()}, dim=dim, rank=1, needs_v=True)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r, params={"k_spring": ()}, dim=dim)
    return F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")


def test_infer_force_ud_interacting_multiparticle():
    """Underdamped trap+damping+spring: recover (k_trap, γ, k_spring) — interacting UD."""
    from SFI.inference.underdamped import UnderdampedLangevinInference
    from SFI.langevin import UnderdampedProcess

    dim, N = 2, 3
    F_psf = _build_ud_interacting(dim)
    proc = UnderdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"k_trap": 1.0, "gamma": 0.5, "k_spring": 0.3})
    proc.initialize(random.normal(random.PRNGKey(0), (N, dim)) * 0.4,
                    v0=random.normal(random.PRNGKey(1), (N, dim)) * 0.2)
    coll = proc.simulate(dt=0.02, Nsteps=4000, key=random.PRNGKey(2), prerun=100)

    inf = UnderdampedLangevinInference(coll)
    inf.infer_force(_build_ud_interacting(dim), n_substeps=3, max_outer=3)
    p = _build_ud_interacting(dim).unflatten_params(inf.force_coefficients_full)
    k_trap = float(np.asarray(p["k_trap"])); gamma = float(np.asarray(p["gamma"]))
    k_spring = float(np.asarray(p["k_spring"]))

    # interacting UD uses the frozen-background phase Jacobian + shorter trajectory
    # → looser tolerance (this path is the runtime/optimization target)
    assert abs(k_trap - 1.0) / 1.0 < 0.25, f"k_trap={k_trap}"
    assert abs(gamma - 0.5) / 0.5 < 0.35, f"gamma={gamma}"
    assert abs(k_spring - 0.3) / 0.3 < 0.40, f"k_spring={k_spring}"
