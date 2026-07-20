"""make_basis(particle_extras=...): per-sample extras are vmapped per particle."""

import jax.numpy as jnp
import numpy as np

from SFI.statefunc import make_basis


def test_per_particle_extra_seen_per_sample():
    """A (N,)-aligned extra reaches each particle's own single-sample call."""
    N = 4

    def f(x, *, extras):
        # returns the particle's own scalar (rank 0, 1 feature)
        return jnp.asarray(extras["w"])[None]

    B = make_basis(f, dim=2, rank=0, n_features=1, extras_keys=("w",), particle_extras=("w",))

    x = jnp.zeros((N, 2))                 # (N, d) — particle axis as batch
    w = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    y = np.asarray(B(x, extras={"w": w})).ravel()
    np.testing.assert_allclose(y, [10.0, 20.0, 30.0, 40.0])


def test_per_particle_onehot_home_basis():
    """The home-range idiom: feature i is non-zero only for particle i."""
    N, d = 3, 2

    def home(x, *, extras):
        pull = -(x - extras["x0"])                          # (d,) own anchor
        onehot = (jnp.arange(N) == extras["home_id"]).astype(x.dtype)
        return pull[:, None] * onehot[None, :]              # (d, N)

    B = make_basis(
        home, dim=d, rank=1, n_features=N,
        extras_keys=("x0", "home_id"), particle_extras=("x0", "home_id"),
    )
    x = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])     # (N, d)
    x0 = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    y = np.asarray(B(x, extras={"x0": x0, "home_id": jnp.arange(N)}))  # (N, d, N)

    # particle i's feature i = -(x_i - x0_i); all other features zero
    for i in range(N):
        np.testing.assert_allclose(y[i, :, i], -(np.asarray(x)[i] - np.asarray(x0)[i]))
        off = [j for j in range(N) if j != i]
        np.testing.assert_allclose(y[i, :, off], 0.0)


def test_non_particle_extra_still_global():
    """An undeclared extra remains a shared constant (no per-sample split)."""
    N = 3

    def f(x, *, extras):
        return jnp.asarray(extras["g"])[None]

    B = make_basis(f, dim=1, rank=0, n_features=1, extras_keys=("g",))  # not in particle_extras
    x = jnp.zeros((N, 1))
    y = np.asarray(B(x, extras={"g": jnp.asarray(7.0)})).ravel()
    np.testing.assert_allclose(y, [7.0, 7.0, 7.0])
