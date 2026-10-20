"""Derivative nodes slice per-particle time-series extras per sample, as the leaf does."""
import jax
import jax.numpy as jnp
import numpy as np

from SFI import TrajectoryCollection, UnderdampedLangevinInference
from SFI.statefunc import make_basis
from SFI.trajectory.dataset import time_series_extra


def _basis():
    return make_basis(lambda x, v, extras: jnp.stack([x[0], v[0], extras["s"] * v[0]])[None, :], dim=1, rank=1,
                      n_features=3, needs_v=True, extras_keys=("s",), particle_extras=("s",))


def test_d_v_matches_per_sample_jacobian():
    K, N = 4, 3
    rng = np.random.default_rng(0)
    x, v = (jnp.asarray(rng.standard_normal((K, N, 1)), jnp.float32) for _ in range(2))
    s = jnp.asarray(rng.standard_normal((K, N)), jnp.float32)
    B = _basis()
    J = B.d_v()(x, v=v, extras={"s": s})
    ref = jax.vmap(jax.vmap(lambda xi, vi, si: jax.jacfwd(lambda w: B(xi, v=w, extras={"s": si}))(vi)))(x, v, s)
    np.testing.assert_allclose(np.asarray(J).reshape(K, N, -1), np.asarray(ref).reshape(K, N, -1), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(J)[..., 0, 0, 2], np.asarray(s), rtol=1e-6)  # d(s v)/dv = s


def test_uli_with_per_particle_time_series_extra():
    T, N, dt = 400, 3, 0.01
    rng = np.random.default_rng(1)
    X = np.cumsum(np.cumsum(rng.standard_normal((T, N, 1)) * dt, 0) * dt, 0).astype(np.float32)
    s = rng.standard_normal((T, N)).astype(np.float32)
    coll = TrajectoryCollection.from_arrays(X=X, dt=dt, extras_local={"s": time_series_extra(s)})
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(_basis(), preset="clean")
    assert np.all(np.isfinite(np.asarray(inf.force_coefficients)))
