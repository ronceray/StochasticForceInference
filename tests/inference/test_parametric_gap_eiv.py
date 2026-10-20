"""The EIV instrument must not leak measurement noise at mask-gap
boundaries.

Regression: where the instrument's lagged base positions were masked (or
at the dataset front), the left test-function fell back to the plain
regressor ψ — which is η-DIRTY under measurement noise.  Every gap then
contributed a few noise-correlated score terms, biasing the noise-
sensitive odd (friction) sector explosively: on this fixture γ came out
+172% (and up to +7000% at stronger noise×gap settings); the home-range
gallery demo showed γ 1.6 → 3.1 at 0.4% noise × 3% gaps.  Neither noise
alone nor gaps alone bias the fit — only their combination.

The fix EXCLUDES instrument-invalid residuals from the IV estimating
equation (zero left row; the whitened left columns are scale-only, so a
zero row contributes exactly nothing).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import UnderdampedLangevinInference
from SFI.langevin import UnderdampedProcess
from SFI.statefunc import make_basis
from SFI.trajectory import TrajectoryCollection

K_TRUE, GAMMA_TRUE, D0, DT, T = 2.0, 1.6, 0.5, 0.02, 6000


def _basis():
    def force(x, v, *, extras=None):
        return jnp.stack([-x, -v], axis=-1)   # features: stiffness, friction

    return make_basis(force, dim=2, rank=1, n_features=2, needs_v=True)


@pytest.fixture(scope="module")
def noisy_gapped_coll():
    B = _basis()
    proc = UnderdampedProcess(B, D=D0 * jnp.eye(2),
                              theta_F=jnp.array([K_TRUE, GAMMA_TRUE]))
    proc.initialize(jnp.zeros(2), v0=jnp.zeros(2))
    cc = proc.simulate(dt=DT, Nsteps=T, key=random.PRNGKey(0),
                       oversampling=8, prerun=100)
    _, X, _ = cc.to_arrays(dataset=0)
    rng = np.random.default_rng(1)
    Xn = np.asarray(X) + rng.normal(scale=0.01, size=np.asarray(X).shape)
    m = np.ones((T, 1), bool)
    m[rng.choice(np.arange(2, T - 2), size=int(0.08 * T), replace=False)] = False
    return TrajectoryCollection.from_arrays(X=jnp.asarray(Xn), dt=DT,
                                            mask=jnp.asarray(m))


def test_gap_boundaries_do_not_leak_noise_into_eiv(noisy_gapped_coll):
    inf = UnderdampedLangevinInference(noisy_gapped_coll)
    inf.infer_force(_basis())                       # eiv auto (instrument on)
    k_hat, gamma_hat = np.asarray(inf.force_coefficients_full).ravel()
    # pre-fix: gamma ~ 4.4 (+172%) on this fixture
    assert abs(gamma_hat - GAMMA_TRUE) / GAMMA_TRUE < 0.35, gamma_hat
    assert abs(k_hat - K_TRUE) / K_TRUE < 0.35, k_hat
