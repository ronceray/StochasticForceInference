"""Missing frames must not bias the entropy-production estimators.

Both estimators combine several integrals — Gram, velocity / parity moments,
the particle-time ``tau_N`` and (underdamped) the block sums of the error
bar — and every one of them must run over the same rows
(``SFI.integrate.shared_rows``).  Before, the overdamped ``tau_N`` counted
the rows with frames ``t, t+1`` observed while ``G`` and ``M_v`` summed over
the fit's wider stencil, so the rate came out low by the ratio of row
counts (``0.95^3`` for the robust preset at 5 % loss: ``Sdot`` 1.44 instead
of 1.97, 1.68 once ``tau_N`` counts the fit's rows); the underdamped block error bar was built from the raw positions,
ignoring the mask.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases import monomials_up_to, unit_axes, v_components, x_components
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.trajectory import TrajectoryCollection
from SFI.trajectory.dataset import frame_key


def _curl_trap(eps, *, v=False):
    """Harmonic trap with a rotational drive (and friction when ``v``): an exact NESS."""
    (x0, x1), (e0, e1) = x_components(2), unit_axes(2)
    F0, F1 = -x0 - eps * x1, -x1 + eps * x0
    if v:
        v0, v1 = v_components(2)
        F0, F1 = F0 - v0, F1 - v1
    return F0 * e0 + F1 * e1


def _simulate(proc, Nsteps, oversampling):
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    return proc.simulate(dt=0.01, Nsteps=Nsteps, key=random.PRNGKey(0), prerun=500, oversampling=oversampling)


def _nan_masked(coll):
    """Same collection with the masked positions replaced by NaN (as tracking data)."""
    _, X, M = coll.to_arrays(dataset=0)
    X = np.where(M[..., None], X, np.nan)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(X), dt=float(coll.datasets[0].dt), mask=jnp.asarray(M))


@pytest.fixture(scope="module")
def od_data():
    full = _simulate(OverdampedProcess(_curl_trap(1.0), D=0.5), 20_000, 4)
    return full, [full.degrade(data_loss_fraction=0.1, seed=s) for s in (1, 2, 3)]


def _od_ep(coll, **kw):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=2, include_constant=False, rank="vector"), **kw)
    return inf, inf.compute_entropy_production()


@pytest.mark.parametrize("kw", [dict(preset="robust"), dict(M_mode="Strato", G_mode="rectangle")])
def test_od_entropy_rate_unbiased_by_frame_loss(od_data, kw):
    full, lossy = od_data
    _, o0 = _od_ep(full, **kw)
    rates = []
    for c in lossy:
        inf, o1 = _od_ep(c, **kw)
        # tau_N is the particle-time of the fit's rows (the Gram's measure) ...
        rows = {"X", "__dt__"} | {frame_key(f) for f in inf._force_row_frames}
        assert o1["tauN"] == pytest.approx(c.Teff(rows), rel=1e-5)
        rates.append(o1["Sdot"])
    # ... so the rate matches the full-data one.  Mean over three loss patterns:
    # dropping rows alone moves a single estimate by ~sigma.  Pre-fix the robust
    # preset was low by the row-count ratio 0.9^3 (~3.6 sigma).
    assert abs(np.mean(rates) - o0["Sdot"]) < 2.0 * o0["Sdot_error"], (o0["Sdot"], rates)


@pytest.fixture(scope="module")
def ud_pair():
    full = _simulate(UnderdampedProcess(_curl_trap(0.5, v=True), D=0.5), 40_000, 8)
    return full, full.degrade(data_loss_fraction=0.1, seed=1)


def _ud_ep(coll):
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(monomials_up_to(1, dim=2, include_constant=False, include_v=True, rank="vector"))
    return inf.compute_entropy_production(), inf.metadata["entropy_production"]


def test_ud_entropy_unbiased_by_frame_loss_and_block_error_sees_the_mask(ud_pair):
    full, lossy = ud_pair
    o0, m0 = _ud_ep(full)
    o1, m1 = _ud_ep(_nan_masked(lossy))
    # Masked frames hold NaN: every integral, the block error bar included,
    # must skip them (pre-fix the block bar was NaN and silently dropped).
    assert np.isfinite(o1["DeltaS"]) and np.isfinite(m1["error_block"])
    assert 0.5 < m1["error_block"] / m0["error_block"] < 1.5
    assert abs(o1["Sdot"] - o0["Sdot"]) < 2.0 * o0["Sdot_error"], (o0, o1)
