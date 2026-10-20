"""Golden regression gate for the parametric estimators (OD + UD).

Re-runs every scenario in ``_parametric_golden_scenarios`` on the inputs
stored in ``_golden/parametric_golden.npz`` and compares the solver
outputs against the pinned values.

Purpose: any refactor of ``SFI/inference/parametric_core`` claimed to be
an *exact rewrite* must keep these numbers to fp-reassociation tolerance
(tier ``"exact"``: rtol 1e-9).  Deliberate numerical changes must
regenerate the goldens (``scripts/gen_parametric_goldens.py``) in their
own commit, recording the old-vs-new deltas the generator prints.

The inputs (trajectories, masks) are stored in the archive, so pinned
outputs never drift because a simulator changed.
"""

import pathlib

import jax
import numpy as np
import pytest

from _parametric_golden_scenarios import GOLDEN_PATH, SCENARIOS

_GOLDEN = pathlib.Path(__file__).resolve().parents[2] / GOLDEN_PATH

RTOL = {"exact": 1e-9, "repinned": 1e-9}


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Goldens are generated and compared in float64."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def golden_store():
    if not _GOLDEN.exists():
        pytest.skip(f"golden archive missing ({_GOLDEN}); run "
                    "scripts/gen_parametric_goldens.py")
    with np.load(_GOLDEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _scale(arr):
    return max(float(np.max(np.abs(arr))), 1e-30)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden(name, golden_store):
    sc = SCENARIOS[name]
    inputs = {k.split("/", 2)[2]: golden_store[k]
              for k in golden_store if k.startswith(f"{name}/in/")}
    if not inputs:
        pytest.skip(f"no golden entry for {name}; regenerate the archive")
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in inputs.items()}
    outputs = sc.run(data)

    rtol = RTOL[sc.tier]
    for key, actual in outputs.items():
        expected = golden_store[f"{name}/out/{key}"]
        actual = np.asarray(actual)
        if key == "f":
            # The converged score is a near-cancellation: compare against
            # the magnitude of its contributions (G·θ), not against ~0.
            atol = rtol * _scale(golden_store[f"{name}/out/G"]) * max(
                1.0, _scale(golden_store[f"{name}/out/theta"]))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol,
                                       err_msg=f"{name}/{key}")
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=rtol * _scale(expected),
                err_msg=f"{name}/{key}")
