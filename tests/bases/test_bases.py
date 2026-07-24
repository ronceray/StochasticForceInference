# TODO: review this file
import jax.numpy as jnp

from SFI.bases import monomials_degree, monomials_up_to


def test_monomials_degree_counts():
    # dim=2, x-only: F(d)=C(2+d-1,d)=d+1
    for d in range(5):
        B = monomials_degree(d, dim=2, include_x=True, include_v=False)
        x = jnp.ones((3, 2))  # P=3
        y = B(x)
        assert y.shape == (3, d + 1)


def test_monomials_labels_and_values():
    B = monomials_degree(2, dim=2, include_x=True, include_v=False)
    # degree-2 labels should include: x0^2, x0·x1, x1^2 (order may vary but count is 3)
    assert sum(lbl.startswith("x0^2") for lbl in B.labels) == 1
    x = jnp.array([[2.0, 3.0]])  # P=1
    y = B(x)  # (1,3)
    # check numeric content against naive eval
    vals = sorted([4.0, 6.0, 9.0])
    assert sorted(list(jnp.array(y[0]).tolist())) == vals


def test_monomials_up_to_concat():
    B = monomials_up_to(3, dim=2, include_x=True, include_v=False)
    x = jnp.ones((5, 2))
    y = B(x)
    # total features sum_{d=0..3} (d+1) = 10
    assert y.shape == (5, 10)
