"""Same-particle Jacobian of a one-body & interacting concatenation.

The one-body child is block-diagonal in the particles, so its same-particle Jacobian is
obtained from one JVP per coordinate (no whole-frame evaluation per particle); the
concatenated basis must match finite differences, masked particles included.
"""

import jax
import jax.numpy as jnp
import numpy as np

from SFI.statefunc import make_basis, make_interactor


def test_one_body_concat_pairs_same_particle_jacobian_matches_finite_differences():
    P = 6
    x = jnp.asarray(np.random.default_rng(1).normal(size=(P, 2)))
    one = make_basis(lambda y: jnp.exp(-(y**2).sum()) * jnp.eye(2), dim=2, rank=1, n_features=2, labels=["a", "b"])

    def pair(XK):
        d = XK[0] - XK[1]
        return (d * jnp.exp(-(d**2).sum()))[:, None] * jnp.ones((1, 2))

    nb = np.array([[(i + 1) % P, (i - 1) % P] for i in range(P)]).ravel().astype(np.int32)
    ex = {"ip": jnp.asarray(np.arange(0, 2 * P + 1, 2, dtype=np.int32)), "ix": jnp.asarray(nb)}
    inter = make_interactor(pair, dim=2, rank=1, K=2, n_features=2, labels=["p", "q"])
    B = one & inter.dispatch_pairs_from_extras(indptr_key="ip", indices_key="ix")
    mask = jnp.ones(P, bool).at[2].set(False)

    J = B.d_x(same_particle=True)(x, mask=mask, extras=ex)  # (P, d_jac, rank, F)
    f = lambda y: B(y, mask=mask, extras=ex)
    e = 1e-3
    for i in range(P):
        for j in range(2):
            fd = (f(x.at[i, j].add(e))[i] - f(x.at[i, j].add(-e))[i]) / (2 * e)
            np.testing.assert_allclose(np.asarray(J[i, j]), np.asarray(fd), rtol=1e-3, atol=1e-3)
