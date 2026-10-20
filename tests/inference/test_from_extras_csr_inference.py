"""Bases dispatched on a user-supplied CSR (``FromExtras*`` rules) stream extras into ``integrate``.

The CSR keys are dispatcher-owned (structural): they are never forwarded to child nodes and so do
not appear in ``required_extras``. Evaluators used to decide extras streaming from
``required_extras`` alone, so a basis whose only extras were structural was evaluated without
extras and the rule raised ``KeyError: 'FromExtrasPairsCSR: extras is required'``.
"""
import jax
import jax.numpy as jnp
import numpy as np

from SFI import OverdampedLangevinInference, TrajectoryCollection, UnderdampedLangevinInference
from SFI.integrate.integrand import ExprOperand
from SFI.integrate import stream
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import make_basis, make_interactor

N, d = 12, 2


def _ring():
    nbrs = np.array([[(i - 1) % N, (i + 1) % N] for i in range(N)], np.int32).reshape(-1)
    return {"ring/indptr": jnp.asarray(np.arange(0, 2 * N + 1, 2, dtype=np.int32)), "ring/indices": jnp.asarray(nbrs)}


def _coupling(needs_v=False):
    if needs_v:
        f = lambda Xk, v: (v[1] - v[0])[:, None]  # noqa: E731
    else:
        f = lambda Xk: (Xk[1] - Xk[0])[:, None]  # noqa: E731
    inter = make_interactor(f, dim=d, rank=1, K=2, n_features=1, needs_v=needs_v)
    return inter.dispatch_pairs_from_extras(indptr_key="ring/indptr", indices_key="ring/indices")


def test_structural_keys_are_reported_and_request_extras():
    coup = _coupling()
    assert set(coup.structural_extras) == {"ring/indptr", "ring/indices"}
    assert coup.required_extras == ()
    assert "extras" in ExprOperand(coup, x=stream("X")).requires
    single = make_basis(lambda x: x[:, None], dim=d, rank=1, n_features=1)
    assert "extras" in ExprOperand(single & coup, x=stream("X")).requires
    assert "extras" not in ExprOperand(single, x=stream("X")).requires


def test_overdamped_fit_on_user_csr():
    ex = _ring()
    single = make_basis(lambda x: x[:, None], dim=d, rank=1, n_features=1)
    B = single & _coupling()
    proc = OverdampedProcess(B, D=0.1, theta_F=jnp.array([-1.0, 0.5]))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.asarray(np.random.default_rng(0).standard_normal((N, d)), jnp.float32))
    X = np.asarray(proc.simulate(dt=0.01, Nsteps=4000, key=jax.random.PRNGKey(0)).datasets[0].X)
    inf = OverdampedLangevinInference(TrajectoryCollection.from_arrays(X=X, dt=0.01, extras_global=ex))
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(B)
    np.testing.assert_allclose(np.asarray(inf.force_coefficients), [-1.0, 0.5], atol=0.15)


def test_underdamped_fit_on_user_csr():
    ex = _ring()
    single = make_basis(lambda x, v: jnp.stack([x, v], -1), dim=d, rank=1, n_features=2, needs_v=True)
    B = single & _coupling(needs_v=True)
    proc = UnderdampedProcess(B, D=0.1, theta_F=jnp.array([-1.0, -0.5, 0.8]))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.asarray(np.random.default_rng(0).standard_normal((N, d)), jnp.float32))
    X = np.asarray(proc.simulate(dt=0.02, Nsteps=4000, key=jax.random.PRNGKey(1), oversampling=5).datasets[0].X)
    inf = UnderdampedLangevinInference(TrajectoryCollection.from_arrays(X=X, dt=0.02, extras_global=ex))
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)
    np.testing.assert_allclose(np.asarray(inf.force_coefficients), [-1.0, -0.5, 0.8], atol=0.25)
