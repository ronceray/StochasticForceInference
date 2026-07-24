"""`StateExpr.specialize(dataset=k)` — pooled → single-condition collapse.

Specialization folds every `dataset_index`-reading primitive at index `k`, so
the result no longer depends on the reserved `dataset_index` extra and carries
only condition-`k`'s parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from SFI.bases import (
    X,
    dataset_indicator,
    named_scalar,
    per_dataset_scalar,
    unit_vector_basis,
)
from SFI.statefunc import StateExpr


def test_specialize_per_dataset_scalar_drops_index_and_matches():
    K, k = 3, 1
    ps = per_dataset_scalar("s", K)
    assert "dataset_index" in ps.required_extras

    sp = ps.specialize(dataset=k)

    # No longer reads the reserved coordinate; param shrank to a scalar.
    assert "dataset_index" not in sp.required_extras
    assert sp.template["s"].shape == ()

    # Unbound equivalence: specialized fed the k-th param == pooled fed the
    # full (K,) param with dataset_index=k.
    x = jnp.zeros((1, 1))
    vK = jnp.array([10.0, 20.0, 30.0])
    out_pool = ps(x, params={"s": vK}, extras={"dataset_index": k})
    out_spec = sp(x, params={"s": vK[k]})
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_dataset_indicator_becomes_constant_onehot():
    K, k = 3, 2
    di = dataset_indicator(K, dim=1)
    assert "dataset_index" in di.required_extras

    sp = di.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    x = jnp.zeros((4, 1))  # batch of 4 samples
    out_pool = di(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))
    np.testing.assert_allclose(np.asarray(out_spec)[0], np.eye(K)[k])


def test_specialize_composite_indicator_times_X_matches():
    K, k = 3, 1
    B = dataset_indicator(K, dim=1) * X(dim=1)  # MapNNode over two leaves
    assert "dataset_index" in B.required_extras

    sp = B.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    x = jnp.array([[2.0], [5.0]])  # (2 samples, dim=1)
    out_pool = B(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_bound_sf_projects_params_and_matches():
    K, k = 2, 1
    a = per_dataset_scalar("a", K)
    stiff = named_scalar("stiff")
    F = a * unit_vector_basis(1) - stiff * X(dim=1)  # parametric PSF
    theta = {"a": jnp.array([0.5, -0.5]), "stiff": jnp.array(2.0)}
    sf = F.bind(theta)
    assert "dataset_index" in sf.required_extras

    sp = sf.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    # per-dataset 'a' folded to its k-th scalar; shared 'stiff' unchanged.
    assert sp.params["a"].shape == ()
    np.testing.assert_allclose(np.asarray(sp.params["a"]), -0.5)
    np.testing.assert_allclose(np.asarray(sp.params["stiff"]), 2.0)

    x = jnp.array([[1.0], [3.0]])
    out_pool = sf(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_through_derivative_of_potential():
    """specialize() must rebuild a ``DerivativeNode`` when a ``dataset_index``
    leaf below it is folded.

    Regression for the ``DerivativeNode.with_children is not implemented`` crash
    hit when the force is the gradient of a per-dataset potential
    (``F = -∇U``): the pooled attractor could not be collapsed to one condition.
    """
    K, k = 3, 1
    # Per-dataset scalar potential U_d(x) = |x|^2, then F = ∂ₓU (a DerivativeNode).
    Usq = StateExpr.einsum("i,i->", X(dim=1), X(dim=1))
    F = (dataset_indicator(K, dim=1) * Usq).d_x()
    assert "dataset_index" in F.required_extras

    sp = F.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    x = jnp.array([[2.0], [5.0]])
    out_pool = F(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_through_slice_features():
    """specialize() must rebuild a ``SliceFeaturesNode`` around a folded child."""
    K, k = 3, 2
    B = (dataset_indicator(K, dim=1) * X(dim=1))[::-1]  # feature-reordered
    assert "dataset_index" in B.required_extras

    sp = B.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    x = jnp.array([[2.0], [5.0]])
    out_pool = B(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_through_reshape_rank():
    """specialize() must rebuild a ``ReshapeRankNode`` around a folded child."""
    K, k = 3, 0
    # rank-1, K features (dim=2) → fold the rank axis into the feature axis.
    B = (dataset_indicator(K, dim=2) * X(dim=2)).rank_to_features()
    assert "dataset_index" in B.required_extras

    sp = B.specialize(dataset=k)
    assert "dataset_index" not in sp.required_extras

    x = jnp.array([[2.0, -1.0], [0.5, 3.0]])
    out_pool = B(x, extras={"dataset_index": k})
    out_spec = sp(x)
    np.testing.assert_allclose(np.asarray(out_spec), np.asarray(out_pool))


def test_specialize_noop_through_interaction_dispatcher():
    """specialize() must pass through an ``InteractionDispatcher`` unchanged
    when nothing needs specializing (no ``dataset_index``-reading leaves).

    Regression for the
    ``InteractionDispatcher.with_children is not implemented`` crash that
    broke ``simulate_bootstrapped_trajectory`` on interacting models (the
    ABP / SPDE gallery demos): ``_specialize_node`` used to rebuild every
    composite via ``with_children`` even when no child changed.
    """
    from SFI.statefunc import Rank, make_interactor
    from SFI.statefunc.nodes.interactions import AutoPairs

    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]

    inter = make_interactor(f, dim=2, rank=Rank.VECTOR, K=2)
    basis = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        return_as="basis",
    )
    # No dataset_index leaves -> specialize is a no-op and must not raise.
    out = basis.specialize(dataset=0)
    assert out is not None
