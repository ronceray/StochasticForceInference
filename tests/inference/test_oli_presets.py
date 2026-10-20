"""OLI linear force-inference presets and the Lambda-based ``auto`` switch.

``OverdampedLangevinInference.infer_force_linear`` consolidates the
``(M_mode, G_mode)`` convention surface into a single ``preset`` keyword,
mirroring the underdamped engine (see ``_resolve_force_linear_preset``):

* ``robust`` — ``Strato`` / ``shift``: noise-robust default.
* ``clean``  — ``Ito`` / ``trapeze``: sharper on verified clean / fine data.
* ``KM``     — ``Ito`` / ``rectangle``: plain Kramers–Moyal moments.
* ``legacy-sfi-v1.0`` — ``Strato`` / ``rectangle``: the published SFI v1.0 convention.

``auto`` routes between ``robust`` and ``clean`` on the sign of the
measurement-noise estimate ``Lambda_trace``.
"""
from __future__ import annotations

import pytest

from SFI.inference.overdamped import _resolve_force_linear_preset


def test_robust_preset_is_strato_shift():
    M, G, name = _resolve_force_linear_preset("robust")
    assert (M, G) == ("Strato", "shift")
    assert name == "robust"


def test_clean_preset_is_ito_trapeze():
    M, G, name = _resolve_force_linear_preset("clean")
    assert (M, G) == ("Ito", "trapeze")
    assert name == "clean"


def test_km_preset_is_ito_rectangle():
    M, G, name = _resolve_force_linear_preset("KM")
    assert (M, G) == ("Ito", "rectangle")
    assert name == "KM"


def test_legacy_sfi_v1_is_strato_rectangle():
    M, G, name = _resolve_force_linear_preset("legacy-sfi-v1.0")
    assert (M, G) == ("Strato", "rectangle")
    assert name == "legacy-sfi-v1.0"


def test_auto_with_noise_picks_robust():
    M, G, name = _resolve_force_linear_preset("auto", noise_detected=True)
    assert (M, G, name) == ("Strato", "shift", "robust")


def test_auto_without_noise_picks_clean():
    M, G, name = _resolve_force_linear_preset("auto", noise_detected=False)
    assert (M, G, name) == ("Ito", "trapeze", "clean")


def test_auto_requires_noise_indicator():
    # preset='auto' needs the measurement-noise estimate to route.
    with pytest.raises(RuntimeError, match="auto"):
        _resolve_force_linear_preset("auto", noise_detected=None)


def test_explicit_modes_override_preset():
    M, G, _ = _resolve_force_linear_preset("clean", M_mode="Strato")
    assert (M, G) == ("Strato", "trapeze")
    M, G, _ = _resolve_force_linear_preset("robust", G_mode="rectangle")
    assert (M, G) == ("Strato", "rectangle")


def test_legacy_m_mode_auto_synonym_uses_preset():
    # Back-compat: M_mode="auto" is treated as "use the preset".
    M, G, name = _resolve_force_linear_preset(
        "auto", M_mode="auto", noise_detected=False
    )
    assert (M, G, name) == ("Ito", "trapeze", "clean")


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        _resolve_force_linear_preset("turbo")
