"""Shared scenario definitions for the parametric golden regression suite.

Each scenario is a (make_data, run) pair:

* ``make_data()`` builds the input arrays ONCE (numpy RNG / seeded JAX
  simulation).  The generator stores these inputs in the golden ``.npz``
  so the pinned outputs never depend on future changes to simulators.
* ``run(data)`` executes the estimator on the stored inputs and returns a
  flat ``{key: np.ndarray}`` of outputs to pin.

Regenerate the goldens with ``python scripts/gen_parametric_goldens.py``
(run from the repo root, float64 is forced by the script).  The test
consumer is ``tests/inference/test_parametric_golden.py``.

Tolerance tiers (consumed by the test):

* ``"exact"``   — rtol 1e-9: the scenario must be bit-stable under
  exact-rewrite refactors (fp reassociation only).
* ``"repinned"`` — a looser, documented tier for scenarios whose pinned
  values were deliberately regenerated after a tolerance-gated change
  (e.g. the UD ψ θ-recursion swap); the regeneration commit must record
  the old-vs-new delta.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SCENARIOS", "GOLDEN_PATH"]

GOLDEN_PATH = "tests/inference/_golden/parametric_golden.npz"


# ── data builders (numpy loops: RNG streams are stable across versions) ──


def _ou_data(T, dt, d=2, seed=42, noise=0.0):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    D = np.diag(np.array([0.5, 0.3], dtype=np.float64)[:d])
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ (rng.normal(size=d) * np.sqrt(dt))
    if noise:
        X = X + noise * rng.standard_normal(X.shape)
    return X


def _ud_harmonic_data(T, dt, seed=42, k=1.0, gamma=0.5, D=0.1, noise=0.0):
    rng = np.random.default_rng(seed)
    sqrt2D = np.sqrt(2 * D)
    X = np.zeros((T, 1))
    V = np.zeros((T, 1))
    X[0] = rng.normal(0, 0.3, size=1)
    V[0] = rng.normal(0, 0.3, size=1)
    for t in range(T - 1):
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k * X[t] - gamma * V[t]) * dt + sqrt2D * rng.normal(size=1) * np.sqrt(dt)
    if noise:
        X = X + noise * rng.standard_normal(X.shape)
    return X


def _interacting_data():
    """Trap + pairwise spring, N=4, d=2 — simulated once and stored."""
    from jax import random

    from SFI.langevin import OverdampedProcess

    F_psf = _interacting_model(2)
    proc = OverdampedProcess(F_psf, D=0.3)
    proc.set_params(theta_F={"k_trap": 1.0, "k_spring": 0.5})
    proc.initialize(random.normal(random.PRNGKey(0), (4, 2)) * 0.5)
    coll = proc.simulate(dt=0.01, Nsteps=3000, key=random.PRNGKey(1),
                         prerun=100, oversampling=5)
    return np.asarray(coll.datasets[0].X)


# ── model builders ──


def _coll(X, dt, mask=None):
    import jax.numpy as jnp

    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    if X.ndim == 2:
        X = X[:, None, :]
    kw = {} if mask is None else {"mask": jnp.array(mask)}
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X), dt=float(dt), **kw)
    return TrajectoryCollection.from_dataset(ds)


def _od_vector_basis():
    from SFI.bases import monomials_up_to

    return monomials_up_to(1, dim=2, rank="vector")


def _linear_drift_psf(d=2):
    import jax.numpy as jnp

    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x

    return make_psf(f, dim=d, rank=1, n_features=1,
                    params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)])


def _ud_linear_psf(d=1):
    import jax.numpy as jnp

    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    return make_psf(f, dim=d, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def _ud_velocity_D_psf():
    import jax.numpy as jnp

    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def fD(x, *, v, params, mask=None, extras=None):
        return jnp.array([[params["c0"] + params["c1"] * v[0] ** 2]])

    return make_psf(fD, dim=1, rank=2, n_features=1, needs_v=True,
                    params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                            ParamSpec("c1", shape=(), dtype=jnp.float64)])


def _interacting_model(dim=2):
    from SFI.bases.pairs import pair_direction, parametric_radial_kernel
    from SFI.statefunc import make_psf

    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x,
                      params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r,
                                        params={"k_spring": ()}, dim=dim)
    return F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")


# ── output packing ──


def _force_outputs(res):
    return {
        "theta": np.asarray(res.theta),
        "D": np.asarray(res.D),
        "Lambda": np.asarray(res.Lambda),
        "G": np.asarray(res.G),
        "f": np.asarray(res.f),
        "theta_cov": np.asarray(res.theta_cov),
        "eiv_w": np.asarray(res.info["eiv_w"]),
    }


# ── scenario runners ──


def _run_od_force(data, *, basis=None, **kw):
    from SFI.inference.parametric_core.solve import solve_force_od

    coll = _coll(data["X"], data["dt"],
                 mask=data.get("mask"))
    F = basis if basis is not None else _od_vector_basis()
    return _force_outputs(solve_force_od(coll, F, max_outer=6, **kw))


def _run_ud_force(data, **kw):
    from SFI.inference.parametric_core.solve import solve_force_ud

    coll = _coll(data["X"], data["dt"])
    return _force_outputs(solve_force_ud(coll, _ud_linear_psf(1), max_outer=6, **kw))


def _run_od_diffusion(data):
    from SFI.bases import symmetric_matrix_basis
    from SFI.inference.parametric_core.solve import solve_diffusion_od, solve_force_od

    coll = _coll(data["X"], data["dt"])
    basis = _od_vector_basis()
    fres = solve_force_od(coll, basis, max_outer=6)
    dres = solve_diffusion_od(coll, basis, fres.theta, symmetric_matrix_basis(2),
                              Lambda=fres.Lambda)
    return {"theta_D": np.asarray(dres.theta_D),
            "nll": np.asarray(dres.info["nll"]),
            "theta_F": np.asarray(fres.theta)}


def _run_ud_diffusion(data):
    from SFI.inference.parametric_core.solve import solve_diffusion_ud, solve_force_ud

    coll = _coll(data["X"], data["dt"])
    fres = solve_force_ud(coll, _ud_linear_psf(1), max_outer=6)
    dres = solve_diffusion_ud(coll, _ud_linear_psf(1), fres.theta, _ud_velocity_D_psf(),
                              Lambda=fres.Lambda, theta_D0={"c0": 0.08, "c1": 0.0})
    return {"theta_D": np.asarray(dres.theta_D),
            "nll": np.asarray(dres.info["nll"]),
            "theta_F": np.asarray(fres.theta)}


def _run_od_interacting(data):
    """Engine-level scenario (exercises the wrapper attribute wiring too)."""
    from SFI.inference.overdamped import OverdampedLangevinInference

    coll = _coll(data["X"], data["dt"])
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_interacting_model(2), max_outer=5)
    return {
        "theta": np.asarray(inf.force_coefficients_full),
        "G": np.asarray(inf.force_G),
        "theta_cov": np.asarray(inf.force_G_pinv),
        "Lambda": np.asarray(inf.Lambda),
        "D": np.asarray(inf.diffusion_average),
    }


class _Scenario:
    def __init__(self, make_data, run, tier="exact"):
        self.make_data = make_data
        self.run = run
        self.tier = tier


def _masked_ou_data():
    X = _ou_data(T=5000, dt=0.02, seed=11, noise=0.02)
    mask = np.ones((5000, 1), dtype=bool)
    mask[1000:1100] = False
    mask[3000:3050] = False
    return {"X": X, "dt": 0.02, "mask": mask}


SCENARIOS = {
    "od_gn_eiv": _Scenario(
        lambda: {"X": _ou_data(T=6000, dt=0.02, seed=7, noise=0.03), "dt": 0.02},
        _run_od_force),
    "od_gn_mle": _Scenario(
        lambda: {"X": _ou_data(T=6000, dt=0.02, seed=7, noise=0.03), "dt": 0.02},
        lambda data: _run_od_force(data, eiv=False)),
    "od_psf_lbfgs": _Scenario(
        lambda: {"X": _ou_data(T=4000, dt=0.02, seed=9), "dt": 0.02},
        lambda data: _run_od_force(data, basis=_linear_drift_psf(2))),
    "od_euler": _Scenario(
        lambda: {"X": _ou_data(T=4000, dt=0.02, seed=13), "dt": 0.02},
        lambda data: _run_od_force(data, integrator="euler")),
    "od_masked": _Scenario(_masked_ou_data, _run_od_force),
    "od_interacting": _Scenario(
        lambda: {"X": _interacting_data(), "dt": 0.01},
        _run_od_interacting),
    "ud_gn_eiv": _Scenario(
        lambda: {"X": _ud_harmonic_data(T=8000, dt=0.05, seed=7), "dt": 0.05},
        _run_ud_force),
    "ud_gn_mle": _Scenario(
        lambda: {"X": _ud_harmonic_data(T=8000, dt=0.05, seed=7), "dt": 0.05},
        lambda data: _run_ud_force(data, eiv=False)),
    "od_diffusion": _Scenario(
        lambda: {"X": _ou_data(T=6000, dt=0.02, seed=3), "dt": 0.02},
        _run_od_diffusion),
    "ud_diffusion": _Scenario(
        lambda: {"X": _ud_harmonic_data(T=8000, dt=0.05, seed=4), "dt": 0.05},
        _run_ud_diffusion),
}
