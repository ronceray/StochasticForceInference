"""`StateExpr.specialize(dataset=k)` — pooled → single-condition collapse.

Specialization folds every `dataset_index`-reading primitive at index `k`, so
the result no longer depends on the reserved `dataset_index` extra and carries
only condition-`k`'s parameters (see
`docs/superpowers/specs/2026-06-24-pooled-model-specialization-design.md`).
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
