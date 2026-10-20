"""Per-particle extras (``extras_local``) must flow through the exact
parametric core.

Regression: the exact runner's ``_dataset_extras`` assembled only
``extras_global`` + reserved keys, so any model declaring
``particle_extras`` (e.g. the home-range gallery demo's per-agent
``home_id``) raised ``KeyError: missing extras`` inside the flow
Jacobians.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import OverdampedLangevinInference
from SFI.langevin import OverdampedProcess
from SFI.statefunc import make_basis, make_sf
from SFI.trajectory import TrajectoryCollection

K_TRUE = 1.4


@pytest.fixture(scope="module")
def coll_with_local_extras():
    F = make_sf(lambda x: -K_TRUE * x, dim=1, rank=1)
    proc = OverdampedProcess(F, D=0.5 * jnp.eye(1))
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=0.02, Nsteps=6000, key=random.PRNGKey(0), prerun=100)
    t, X, _ = coll.to_arrays(dataset=0)
    # Rebuild with a per-particle extra attached (N = 1 particle).
    return TrajectoryCollection.from_arrays(
        X=jnp.asarray(X), dt=0.02,
        extras_local={"gain": jnp.array([2.0])},
    )


def _gain_basis():
    def feat(x, *, extras):
        # single feature: -gain * x  -> true coefficient K_TRUE / gain
        return (-extras["gain"] * x)[:, None]

    return make_basis(feat, dim=1, rank=1, n_features=1,
                      extras_keys=("gain",), particle_extras=("gain",))


def test_infer_force_threads_extras_local(coll_with_local_extras):
    inf = OverdampedLangevinInference(coll_with_local_extras)
    inf.infer_force(_gain_basis())          # KeyError('gain') pre-fix
    c = float(np.asarray(inf.force_coefficients_full).ravel()[0])
    assert c == pytest.approx(K_TRUE / 2.0, rel=0.15), c


def test_infer_force_linear_still_threads_extras_local(coll_with_local_extras):
    # The linear path already worked; pin it so the two engines agree.
    inf = OverdampedLangevinInference(coll_with_local_extras)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(_gain_basis())
    c = float(np.asarray(inf.force_coefficients_full).ravel()[0])
    assert c == pytest.approx(K_TRUE / 2.0, rel=0.15), c
