"""Rows used by periodic fits: the image reference costs a frame only when unavoidable.

Programs combining periodic pair features at several frames fix every pair's
image at a frame preceding every increment of the row
(``Integrand.pbc_reference_in``).  When the estimator's shared row scope
already reads such a frame it is reused, so most presets sum over exactly the
rows of the same fit on a free-space basis.  Only stencils whose increments
start at the earliest frame they read need one more observed frame; free-space
bases never do.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from SFI import OverdampedLangevinInference, TrajectoryCollection, UnderdampedLangevinInference
from SFI.bases.pairs import exp_poly_kernels, radial_pair_basis

L = 10.0


def _data():
    rng = np.random.default_rng(0)
    X = np.cumsum(0.1 * rng.standard_normal((300, 8, 2)), axis=0) + rng.uniform(0, L, (8, 2))
    return X


def _basis(box):
    inter = radial_pair_basis(exp_poly_kernels([0, 1], [1.0, 3.0]), dim=2, box=box)
    return inter.dispatch_pairs(exclude_self=True, owners="focal", reducer="sum", return_as="basis")


def _coll(X, box):
    return TrajectoryCollection.from_arrays(X=X, dt=0.01, extras_global={"box": jnp.array([L, L])} if box else None)


@pytest.mark.parametrize("preset, extra", [("robust", set()), ("clean", set()), ("KM", set()),
                                           ("legacy-sfi-v1.0", {-1})])
def test_overdamped_rows_match_free_space(preset, extra):
    X = _data()
    frames = {}
    for box in ("extras", None):
        inf = OverdampedLangevinInference(_coll(X, box))
        inf.compute_diffusion_constant(method="MSD")
        inf.infer_force_linear(_basis(box), preset=preset)
        frames[box] = inf._force_row_frames
    assert frames["extras"] == frames[None] | extra


@pytest.mark.parametrize("preset, extra", [("robust", {-2}), ("clean", {-2}), ("legacy-clean-v1.0", set()),
                                           ("legacy-noisy-v1.0", set())])
def test_underdamped_rows_match_free_space(preset, extra):
    X = np.cumsum(_data() - _data()[:1], axis=0) * 0.05 + _data()[0]
    frames = {}
    for box in ("extras", None):
        inf = UnderdampedLangevinInference(_coll(X, box))
        inf.compute_diffusion_constant()
        inf.infer_force_linear(_basis(box), preset=preset)
        frames[box] = inf._force_linear_frames()
    assert frames["extras"] == frames[None] | extra


def test_entropy_production_widens_rows_when_fit_reads_no_reference():
    """A KM fit reads frames {t, t+1}: no admissible image reference, so the
    Stratonovich v-moments and their Gram run on rows with t-1 observed too
    (row set and particle-time of the estimate widened consistently).  The
    rate agrees with the one after a Stratonovich fit, whose rows already
    read t+2, within its error bar."""
    X = _data()
    out = {}
    for preset in ("KM", "legacy-sfi-v1.0"):
        inf = OverdampedLangevinInference(_coll(X, "extras"))
        inf.compute_diffusion_constant(method="MSD")
        inf.infer_force_linear(_basis("extras"), preset=preset)
        out[preset] = (inf, inf.compute_entropy_production(support="full"))
    inf_km, ep_km = out["KM"]
    assert inf_km._force_row_frames == frozenset({1})
    assert ep_km["tauN"] == pytest.approx(inf_km._rows_teff(frozenset({-1, 1})))
    assert ep_km["tauN"] < inf_km._rows_teff(frozenset({1}))
    ep_st = out["legacy-sfi-v1.0"][1]
    assert abs(ep_km["DeltaS"] - ep_st["DeltaS"]) < ep_st["error_DeltaS"]
