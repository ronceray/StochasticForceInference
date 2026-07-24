"""Per-dataset / per-particle parameters in pooled multi-experiment inference."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import OverdampedLangevinInference
from SFI.bases import (
    X,
    dataset_indicator,
    extra_scalar,
    named_scalar,
    per_dataset_scalar,
    unit_vector_basis,
)
from SFI.langevin import OverdampedProcess
from SFI.trajectory import TrajectoryCollection


def _simulate_trap(k, drift, seed, Nsteps=6000, dt=0.01, D=0.2):
    """1D OU with stiffness k and constant drift."""
    B = X(dim=1) & unit_vector_basis(1)
    proc = OverdampedProcess(F=B, D=D, theta_F=jnp.array([-k, drift]))
    proc.initialize(jnp.zeros(1))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), oversampling=4, prerun=50)


def _pooled(*colls):
    return colls[0].concat(list(colls[1:]), weights="pool")


def test_dataset_index_injected_and_reserved():
    coll = _pooled(_simulate_trap(1.0, 0.0, 0), _simulate_trap(2.0, 0.0, 1))
    row0 = coll.datasets[0].make_producer({"X", "extras"}, dataset_index=0)(jnp.asarray(0))
    row1 = coll.datasets[1].make_producer({"X", "extras"}, dataset_index=1)(jnp.asarray(0))
    assert int(row0["extras"]["dataset_index"]) == 0
    assert int(row1["extras"]["dataset_index"]) == 1
    np.testing.assert_array_equal(np.asarray(row0["extras"]["particle_index"]), [0])

    # reserved-key collision
    bad = TrajectoryCollection.from_arrays(
        X=np.zeros((10, 1, 1)), dt=0.1, extras_global={"dataset_index": 7}
    )
    with pytest.raises(ValueError, match="reserved"):
        bad.datasets[0].make_producer({"X", "extras"}, dataset_index=0)(jnp.asarray(0))


def test_linear_route_dataset_indicator():
    """Per-dataset stiffness via one-hot indicator features (linear estimators)."""
    k1, k2 = 1.0, 2.0
    coll = _pooled(_simulate_trap(k1, 0.0, 2), _simulate_trap(k2, 0.0, 3))

    B = dataset_indicator(2, dim=1) * X(dim=1)  # 2 features: 1{ds=d}·x
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)
    inf.compute_force_error()

    coeffs = np.asarray(inf.force_coefficients).ravel()
    stderr = np.asarray(inf.force_coefficients_stderr).ravel()
    assert np.all(np.abs(coeffs - [-k1, -k2]) < 4.0 * stderr + 0.05)


def test_bootstrap_specializes_to_single_condition_and_reinfers():
    """Bootstrapping a pooled per-dataset force collapses it to the chosen
    experiment's standalone model: no ``dataset_index`` in the exported model or
    trajectory, so re-inference runs on a plain single-condition basis."""
    coll = _pooled(_simulate_trap(1.0, 0.0, 2), _simulate_trap(2.0, 0.0, 3))
    B = dataset_indicator(2, dim=1) * X(dim=1)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)

    coll_boot, proc = inf.simulate_bootstrapped_trajectory(random.PRNGKey(0), dataset=1)
    assert coll_boot.datasets[0].T > 1

    # Exported model + trajectory carry no dataset concept.
    assert "dataset_index" not in proc.F.required_extras
    assert "dataset_index" not in (coll_boot.datasets[0].extras_global or {})

    # Re-inference on the clean trajectory runs with a plain basis (regression:
    # the pooled basis needed dataset_index and crashed on the single export).
    re = OverdampedLangevinInference(coll_boot)
    re.compute_diffusion_constant(method="WeakNoise")
    re.infer_force_linear(X(dim=1))
    assert np.asarray(re.force_coefficients).size >= 1


def test_bootstrap_simulate_false_settles_at_own_attractor():
    """`simulate=False` + manual `initialize()` + simulate must reproduce EACH
    dataset's own attractor, not dataset 0's.

    Regression for the pooled-force default-index bug: the pre-fix path
    returned the pooled force without a ``dataset_index`` override, so every
    bootstrap fell back to dataset 0's attractor.
    """
    # F = -k x + drift  ⇒  equilibrium center at drift/k. ds0 → +5, ds1 → −5.
    coll = _pooled(_simulate_trap(1.0, +5.0, 7), _simulate_trap(1.0, -5.0, 8))
    # shared stiffness + per-dataset constant drift (distinct centers)
    B = (dataset_indicator(2, dim=1) * X(dim=1)) & (dataset_indicator(2, dim=1) * unit_vector_basis(1))
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)

    for k, center in [(0, +5.0), (1, -5.0)]:
        proc = inf.simulate_bootstrapped_trajectory(random.PRNGKey(0), dataset=k, simulate=False)
        assert "dataset_index" not in proc.F.required_extras
        x0 = np.asarray(coll.datasets[k].X).reshape(-1, 1)[0]
        proc.initialize(x0=jnp.asarray(x0))
        data = proc.simulate(dt=0.01, Nsteps=8000, key=random.PRNGKey(1), oversampling=4, prerun=200)
        traj = np.asarray(data.datasets[0].X).ravel()
        settled = float(np.mean(traj[len(traj) // 2 :]))
        assert abs(settled - center) < 1.5, f"dataset {k}: settled {settled:+.2f}, expected {center:+.0f}"


def test_parametric_route_shared_plus_per_dataset():
    """Shared stiffness + per-dataset drift offset with the parametric estimator."""
    k_true, a_true = 1.0, (0.5, -0.5)
    coll = _pooled(
        _simulate_trap(k_true, a_true[0], 4),
        _simulate_trap(k_true, a_true[1], 5),
    )

    k = named_scalar("k", default=0.5)
    a = per_dataset_scalar("a", 2, default=0.0)
    F = a * unit_vector_basis(1) - k * X(dim=1)

    inf = OverdampedLangevinInference(coll)
    inf.infer_force(F)

    theta = inf.force_inferred.params  # bound parameter tree
    k_hat = float(np.asarray(theta["k"]))
    a_hat = np.asarray(theta["a"]).ravel()
    assert abs(k_hat - k_true) < 0.15
    np.testing.assert_allclose(a_hat, a_true, atol=0.15)


def test_per_particle_params_in_interactor():
    """params_local paralleling extras_local: a (P,)-shaped parameter
    indexed by the gathered reserved particle_index inside a pair kernel."""
    from SFI.statefunc import make_interactor

    def local(Xk, *, params, extras):
        mob = params["mob"][extras["particle_index"]]  # (2,) per edge
        dx = Xk[1] - Xk[0]
        return (mob[0] * dx)[..., None]  # focal particle's own mobility

    inter = make_interactor(
        local,
        dim=1,
        rank=1,
        K=2,
        n_features=1,
        params={"mob": (3,)},
        extras_keys=("particle_index",),
        particle_extras=("particle_index",),
        labels=("mob_pair",),
    )
    P = inter.dispatch_pairs()

    coll = TrajectoryCollection.from_arrays(
        X=np.arange(3, dtype=float).reshape(1, 3, 1), dt=0.1
    )
    row = coll.datasets[0].make_producer({"X", "extras"}, dataset_index=0)(jnp.asarray(0))
    y = P(row["X"], extras=row["extras"], params={"mob": jnp.asarray([1.0, 10.0, 100.0])})
    # focal i sums mob[i]·(xj − xi): i=0 → 1·3, i=1 → 0, i=2 → 100·(−3)
    np.testing.assert_allclose(np.asarray(y).ravel(), [3.0, 0.0, -300.0], atol=1e-5)
