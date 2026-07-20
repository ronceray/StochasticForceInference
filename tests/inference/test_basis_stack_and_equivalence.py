"""Regression test for GitHub issue #16.

`Basis.stack([B1, B2, B3])` and `B1 & B2 & B3` must produce identical feature
tensors *and* identical inferred force coefficients.  The two construction
paths produce different ConcatNode tree shapes (flat vs. nested), but that
structural difference must be invisible to inference.
"""

import jax.numpy as jnp
import numpy as np

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.statefunc import Basis
from SFI.statefunc.factory import make_basis
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ou_collection(T=400, dt=0.02, dim=2, seed=42):
    rng = np.random.default_rng(seed)
    k = np.ones(dim, dtype=np.float32) * 0.8
    D = np.ones(dim, dtype=np.float32) * 0.3
    x = np.zeros((T, 1, dim), dtype=np.float32)
    for t in range(T - 1):
        eta = rng.normal(size=(1, dim)).astype(np.float32)
        x[t + 1, 0] = x[t, 0] + dt * (-k * x[t, 0]) + np.sqrt(2.0 * D * dt) * eta[0]
    ds = TrajectoryDataset.from_arrays(X=jnp.array(x), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _three_vector_bases(dim=2):
    """Return three distinct rank-1 bases to combine."""
    B1 = make_basis(
        lambda x, **_: x[..., None],  # identity: shape (dim, 1)
        dim=dim, rank=1, n_features=1,
    )
    B2 = make_basis(
        lambda x, **_: (x**2)[..., None],  # squares
        dim=dim, rank=1, n_features=1,
    )
    B3 = make_basis(
        lambda x, **_: jnp.ones((*x.shape[:-1], dim, 1), dtype=x.dtype),  # constant
        dim=dim, rank=1, n_features=1,
    )
    return B1, B2, B3


# ---------------------------------------------------------------------------
# Test 1 – feature tensor equivalence (statefunc level)
# ---------------------------------------------------------------------------

def test_stack_and_and_produce_identical_feature_tensors():
    """stack([B1,B2,B3]) and B1&B2&B3 must evaluate identically on every input."""
    B1, B2, B3 = _three_vector_bases(dim=2)

    B_stack = Basis.stack([B1, B2, B3])
    B_and = B1 & B2 & B3

    assert B_stack.n_features == B_and.n_features

    x = jnp.array([[1.0, 2.0], [-0.5, 3.0], [0.0, -1.0]])  # (3, 2)
    out_stack = B_stack(x)
    out_and = B_and(x)

    assert jnp.allclose(out_stack, out_and, atol=0.0), (
        "Basis.stack and & produced different feature tensors — "
        "regression for issue #16"
    )


def test_stack_and_and_produce_identical_labels():
    """Labels must be identical regardless of construction path."""
    B1, B2, B3 = _three_vector_bases(dim=2)
    assert Basis.stack([B1, B2, B3]).labels == (B1 & B2 & B3).labels


# ---------------------------------------------------------------------------
# Test 2 – inference coefficient equivalence (end-to-end, issue #16 core)
# ---------------------------------------------------------------------------

def test_infer_force_linear_identical_for_stack_and_and():
    """infer_force_linear must give bit-identical coefficients for stack vs &.

    This is the core regression for issue #16: feature tensors were confirmed
    identical but inference results differed due to a metadata inconsistency in
    the old statefunc.py.  This test guards against that class of regression.
    """
    coll = _make_ou_collection()
    B1, B2, B3 = _three_vector_bases(dim=2)

    B_stack = Basis.stack([B1, B2, B3])
    B_and = B1 & B2 & B3

    inf1 = OverdampedLangevinInference(coll)
    inf1.compute_diffusion_constant(method="WeakNoise")
    inf1.infer_force_linear(B_stack, M_mode="Ito", G_mode="rectangle")

    inf2 = OverdampedLangevinInference(coll)
    inf2.compute_diffusion_constant(method="WeakNoise")
    inf2.infer_force_linear(B_and, M_mode="Ito", G_mode="rectangle")

    assert jnp.allclose(inf1.force_coefficients, inf2.force_coefficients, atol=1e-6), (
        "infer_force_linear produced different coefficients for Basis.stack vs & — "
        "regression for issue #16"
    )
    assert jnp.allclose(inf1.force_G_full, inf2.force_G_full, atol=1e-6), (
        "Gram matrices differ between stack and & — regression for issue #16"
    )
