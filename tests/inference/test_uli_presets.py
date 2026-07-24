"""ULI linear force-inference presets and the Lambda-based ``auto`` switch.

The combinatorial ``(M_mode, G_mode, diffusion_method)`` surface of
``UnderdampedLangevinInference.infer_force_linear`` is consolidated into two
named presets plus an ``auto`` switch (see ``_resolve_force_linear_preset``):

* ``robust`` — ``symmetric`` / ``trapeze`` / ``noisy``: the broadly applicable
  default; flat across measurement noise within its feasible band.
* ``clean``  — ``symmetric`` / ``trapeze`` / ``WeakNoise``: a sharper refinement
  for verified noise-free data (helps in the coarse-dt, low-noise corner).

``auto`` routes between them on the sign of the measurement-noise estimate
``Lambda_trace`` (validated against the ``vdp_uli_10x`` sweep: the zero-crossing
tracks the noisy<->weak estimator transition with no harmful misroutes).
"""
from __future__ import annotations

import pytest

from SFI.inference.underdamped import _resolve_force_linear_preset


def test_robust_preset_is_symmetric_trapeze_noisy():
    (M, G, D), name = _resolve_force_linear_preset("robust")
    assert (M, G, D) == ("symmetric", "trapeze", "noisy")
    assert name == "robust"


def test_clean_preset_is_symmetric_trapeze_weaknoise():
    (M, G, D), name = _resolve_force_linear_preset("clean")
    assert (M, G, D) == ("symmetric", "trapeze", "WeakNoise")
    assert name == "clean"


def test_auto_with_measurement_noise_picks_noisy():
    (M, G, D), name = _resolve_force_linear_preset("auto", noise_detected=True)
    assert (M, G, D) == ("symmetric", "trapeze", "noisy")
    assert name == "robust"


def test_auto_without_measurement_noise_picks_weaknoise():
    (M, G, D), name = _resolve_force_linear_preset("auto", noise_detected=False)
    assert (M, G, D) == ("symmetric", "trapeze", "WeakNoise")
    assert name == "clean"


def test_auto_falls_back_to_robust_when_indicator_unknown():
    # No measurement-noise indicator available -> safe (noisy) default.
    (M, G, D), name = _resolve_force_linear_preset("auto", noise_detected=None)
    assert (M, G, D) == ("symmetric", "trapeze", "noisy")
    assert name == "robust"


def test_manual_override_wins_over_preset():
    (M, G, D), _ = _resolve_force_linear_preset("robust", diffusion_method="MSD")
    assert (M, G, D) == ("symmetric", "trapeze", "MSD")


def test_manual_override_wins_over_auto():
    (M, G, D), _ = _resolve_force_linear_preset(
        "auto", G_mode="shift", noise_detected=False
    )
    assert (M, G, D) == ("symmetric", "shift", "WeakNoise")


def test_partial_overrides_compose_with_preset():
    (M, G, D), _ = _resolve_force_linear_preset(
        "clean", M_mode="early", diffusion_method="MSD"
    )
    assert (M, G, D) == ("early", "trapeze", "MSD")


def test_legacy_clean_v1_preset():
    (M, G, D), name = _resolve_force_linear_preset("legacy-clean-v1.0")
    assert (M, G, D) == ("early", "rectangle", "MSD")
    assert name == "legacy-clean-v1.0"


def test_legacy_noisy_v1_preset():
    (M, G, D), name = _resolve_force_linear_preset("legacy-noisy-v1.0")
    assert (M, G, D) == ("symmetric", "rectangle", "noisy")
    assert name == "legacy-noisy-v1.0"


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        _resolve_force_linear_preset("turbo")


# --------------------------------------------------------------------------- #
# Integration: end-to-end routing of preset='auto' on the Lambda_trace sign
# --------------------------------------------------------------------------- #

import jax.numpy as jnp
from jax import random

from SFI.bases.constants import identity_matrix_basis
from SFI.bases.monomials import monomials_up_to
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import UnderdampedProcess
from SFI.statefunc.factory import make_basis


def _vdp_process():
    def vdp_force(x, *, v, mask=None):
        return jnp.array([2.0 * (1.0 - x[0] ** 2) * v[0] - x[0]])

    F = make_basis(vdp_force, dim=1, rank=1, n_features=1, needs_v=True).to_psf()
    D = identity_matrix_basis(1).to_psf()
    proc = UnderdampedProcess(F, D=D)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))
    return proc


def _force_basis():
    return monomials_up_to(
        order=3, dim=1, include_constant=True, include_x=True, include_v=True, rank="vector"
    )


@pytest.fixture(scope="module")
def vdp_clean():
    """Clean VdP trajectory in the dt~0.02 band where Lambda_trace<0 (force persistence)."""
    proc = _vdp_process()
    coll = proc.simulate(dt=0.02, Nsteps=20000, key=random.PRNGKey(0), prerun=100, oversampling=10)
    return proc, coll


def test_auto_picks_weaknoise_on_clean_data(vdp_clean):
    _, coll = vdp_clean
    inf = UnderdampedLangevinInference(coll, verbosity=0)
    inf.compute_diffusion_constant(method="auto")
    assert inf.Lambda_trace < 0  # clean data: force persistence dominates
    inf.infer_force_linear(_force_basis(), preset="auto")
    assert inf.metadata["force_diffusion_method"] == "WeakNoise"
    assert inf.metadata["force_preset"] == "auto"


def test_auto_picks_noisy_on_noisy_data(vdp_clean):
    _, coll = vdp_clean
    noisy = coll.degrade(noise=0.05, seed=3)
    inf = UnderdampedLangevinInference(noisy, verbosity=0)
    inf.compute_diffusion_constant(method="auto")
    assert inf.Lambda_trace > 0  # measurement noise dominates the increment structure
    inf.infer_force_linear(_force_basis(), preset="auto")
    assert inf.metadata["force_diffusion_method"] == "noisy"


def test_robust_preset_forces_noisy_on_clean_data(vdp_clean):
    _, coll = vdp_clean
    inf = UnderdampedLangevinInference(coll, verbosity=0)
    inf.compute_diffusion_constant(method="auto")
    inf.infer_force_linear(_force_basis(), preset="robust")
    assert inf.metadata["force_diffusion_method"] == "noisy"


def test_auto_requires_diffusion_constant_like_all_presets(vdp_clean):
    # The force inference needs A_inv from compute_diffusion_constant(); the
    # preset machinery does not change that contract.  Without it, inference
    # raises the same diffusion-missing error regardless of preset.
    _, coll = vdp_clean
    inf = UnderdampedLangevinInference(coll, verbosity=0)
    with pytest.raises(RuntimeError, match="diffusion"):
        inf.infer_force_linear(_force_basis(), preset="auto")


def test_explicit_diffusion_method_overrides_auto(vdp_clean):
    _, coll = vdp_clean
    inf = UnderdampedLangevinInference(coll, verbosity=0)
    inf.compute_diffusion_constant(method="auto")
    inf.infer_force_linear(_force_basis(), preset="auto", diffusion_method="MSD")
    assert inf.metadata["force_diffusion_method"] == "MSD"


def test_legacy_explicit_modes_still_supported(vdp_clean):
    # Power-user path: explicit (M_mode, G_mode, diffusion_method) unaffected by presets.
    _, coll = vdp_clean
    inf = UnderdampedLangevinInference(coll, verbosity=0)
    inf.compute_diffusion_constant(method="auto")
    inf.infer_force_linear(
        _force_basis(), M_mode="symmetric", G_mode="shift", diffusion_method="MSD"
    )
    assert inf.metadata["force_M_mode"] == "symmetric"
    assert inf.metadata["force_G_mode"] == "shift"
    assert inf.metadata["force_diffusion_method"] == "MSD"
