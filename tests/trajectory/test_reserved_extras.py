"""Reserved-extras registry and the single extras resolver."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.trajectory.dataset import time_series_extra
from SFI.trajectory.reserved_extras import (
    RESERVED_NAMES,
    ExtrasContext,
    resolve_extras,
    resolve_reserved,
    slice_frame_extras,
)


def _ctx():
    return ExtrasContext(
        n_particles=3,
        dataset_index=2,
        frame_times=jnp.asarray([0.0, 0.1, 0.2]),
        duration=jnp.asarray(0.2),
    )


def test_reserved_names():
    assert RESERVED_NAMES == frozenset({"time", "duration", "dataset_index", "particle_index"})


def test_resolve_reserved_values():
    out = resolve_reserved(_ctx())
    assert set(out) == RESERVED_NAMES
    assert int(out["dataset_index"]) == 2
    assert out["dataset_index"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out["particle_index"]), [0, 1, 2])
    assert out["particle_index"].dtype == jnp.int32
    np.testing.assert_allclose(np.asarray(out["time"]), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(float(out["duration"]), 0.2)


def test_resolve_extras_merges_user_and_reserved():
    out = resolve_extras({"c": jnp.asarray(1.5)}, _ctx())
    assert set(out) == RESERVED_NAMES | {"c"}
    assert float(out["c"]) == 1.5
    assert int(out["dataset_index"]) == 2


def test_reserved_name_collision_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        resolve_extras({"time": jnp.asarray(0.0)}, _ctx())


def test_slice_frame_extras():
    schedule = time_series_extra(jnp.asarray([10.0, 20.0, 30.0]))
    user = slice_frame_extras({"a": schedule, "box": jnp.asarray(5.0)}, None, frame_idx=jnp.asarray(1))
    assert float(user["a"]) == 20.0  # TimeSeriesExtra sliced at frame 1
    assert float(user["box"]) == 5.0  # static, forwarded unchanged


def test_injected_pbc_ref_is_reserved():
    """``pbc_ref`` is injected per row by the integrator: a user extra of that name
    would be read as the periodic-image reference, so it is rejected."""
    from SFI.bases import pairs
    from SFI.integrate.integrand import PBC_REF_KEY
    from SFI.trajectory.reserved_extras import INJECTED_NAMES, is_reserved

    assert PBC_REF_KEY in INJECTED_NAMES and pairs._PBC_REF_KEY == PBC_REF_KEY
    assert is_reserved("pbc_ref") and not is_reserved("box")
    with pytest.raises(ValueError, match="reserved"):
        resolve_extras({"pbc_ref": jnp.zeros((3, 2))}, _ctx())


def test_parametric_runner_rejects_user_pbc_ref():
    import SFI
    from SFI import TrajectoryCollection
    from SFI.bases import monomials_up_to

    X = np.cumsum(np.random.default_rng(0).standard_normal((50, 1, 1)), axis=0)
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global={"pbc_ref": jnp.zeros((1, 1))})
    inf = SFI.OverdampedLangevinInference(coll)
    with pytest.raises(ValueError, match="reserved"):
        inf.infer_force(monomials_up_to(1, dim=1, rank="vector"), D=jnp.eye(1), Lambda=jnp.zeros((1, 1)))
