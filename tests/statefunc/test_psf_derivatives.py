# TODO: review this file
import jax.numpy as jnp

from SFI.statefunc import *


def _quad_basis(dim):
    return make_basis(
        lambda x, **_: x[0] ** 2,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=1,
        needs_v=False,
        labels=["x2"],
    )


def test_d_theta_gradient_via_psf_facade():
    basis = _quad_basis(1)
    psf = basis.to_psf()
    theta = {"coeff": jnp.array([2.0])}
    x = jnp.array([[3.0]])
    J = psf.d_theta()(x, params=theta)
    assert J.shape == (1, 1)
    assert jnp.allclose(J, x**2)


def test_d_x_gradient_of_psf():
    basis = _quad_basis(1)
    psf = basis.to_psf(drop_features=False)
    theta = {"coeff": jnp.array([5.0])}
    x = jnp.array([[4.0]])
    J = psf.bind(theta).d_x()(x)  # (..., dim, features)
    assert J.shape == (1, 1, 1)
    assert jnp.allclose(J[..., 0], 2 * theta["coeff"] * x)


def test_make_psf_basic_and_dtheta():
    # F(x;θ) = a*x0 + b*x1^2
    def f(x, *, params):
        a, b = params["a"], params["b"]
        return jnp.stack([a * x[..., 0] + b * x[..., 1] ** 2], axis=-1)  # (...,1)

    suite = ParamSuite.from_specs(ParamSpec("a", ()), ParamSpec("b", ()))
    P = make_psf(
        f, dim=2, rank=Rank.SCALAR, n_features=1, params=suite, drop_features=False
    )
    theta = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    x = jnp.array([[1.0, 2.0]])
    y = P(x, params=theta)
    assert y.shape == (1, 1)
    assert jnp.allclose(y, 2 * 1 + 3 * 4)
    J = P.d_theta()(x, params=theta)
    assert J.shape == (1, 2)
    assert jnp.allclose(J, jnp.array([[x[0, 0], x[0, 1] ** 2]]))
