"""Time-dependent extras in simulation, and the simulate→infer round trip."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import OverdampedLangevinInference, make_sf
from SFI.bases import X, extra_scalar, identity_matrix_basis, unit_vector_basis
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.trajectory.dataset import TimeSeriesExtra, time_series_extra


@pytest.mark.parametrize("method", ["euler", "heun"])
@pytest.mark.parametrize("oversampling,prerun", [(1, 0), (4, 5)])
def test_shift_convention_pins_frame_alignment(method, oversampling, prerun):
    """With F = a(t) (state-independent) and D≈0:
    X[k+1] - X[k] must equal dt * a[k] frame by frame."""
    Nsteps, dt = 12, 0.1
    a = np.zeros(Nsteps)
    a[Nsteps // 2 :] = 1.0  # step protocol

    F = extra_scalar("a", dim=1) * unit_vector_basis(1)
    proc = OverdampedProcess(F=F.to_psf() if hasattr(F, "to_psf") else F, D=1e-14)
    proc.set_extras(extras_global={"a": time_series_extra(a)})
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(
        dt=dt,
        Nsteps=Nsteps,
        key=random.PRNGKey(0),
        oversampling=oversampling,
        prerun=prerun,
        method=method,
    )
    Xr = np.asarray(coll.datasets[0]._X3d())[:, 0, 0]
    dX = np.diff(Xr)
    np.testing.assert_allclose(dX, dt * a[:-1], atol=1e-6)
    # the schedule is attached to the output, frame-aligned
    out = coll.datasets[0].extras_global["a"]
    assert isinstance(out, TimeSeriesExtra)
    np.testing.assert_allclose(np.asarray(out.data), a)


def test_round_trip_driven_trap():
    """Simulate F = -(k0 + k(t)) x with a square-wave protocol, then
    recover both coefficients with the linear estimator via extra_scalar.

    The x / k(t)·x features are collinear up to the protocol variance, so
    the split carries real statistical error: assertions are 3σ bounds
    using the estimator's own stderr, plus a ground-truth force NMSE.
    """
    Nsteps, dt = 20000, 0.01
    # Square wave (max protocol variance → best identifiability)
    k_t = (np.arange(Nsteps) // 500 % 2).astype(float)

    # F = -x - k(t)·x, built from the same two-feature basis the
    # inference will fit (coefficients -1, -1).
    B = (X(dim=1)) & (extra_scalar("k_drive", dim=1) * X(dim=1))

    proc = OverdampedProcess(F=B, D=0.25, theta_F=jnp.array([-1.0, -1.0]))
    proc.set_extras(extras_global={"k_drive": time_series_extra(k_t)})
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(1), oversampling=4)

    assert isinstance(coll.datasets[0].extras_global["k_drive"], TimeSeriesExtra)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)
    inf.compute_force_error()

    coeffs = np.asarray(inf.force_coefficients).ravel()
    stderr = np.asarray(inf.force_coefficients_stderr).ravel()
    assert np.all(np.abs(coeffs - [-1.0, -1.0]) < 3.0 * stderr + 1e-3)
    assert np.all(stderr < 0.4)  # the protocol must actually identify the split

    inf.compare_to_exact(model_exact=proc)
    assert float(inf.NMSE_force) < 0.05


def test_callable_of_time_materialized():
    Nsteps, dt = 8, 0.5
    proc = OverdampedProcess(F=X(dim=1), D=1e-14, theta_F=jnp.array([0.0]))
    proc.set_extras(extras_global={"a": lambda t: 2.0 * t})
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(2), oversampling=1)
    out = coll.datasets[0].extras_global["a"]
    assert isinstance(out, TimeSeriesExtra)
    np.testing.assert_allclose(np.asarray(out.data), 2.0 * dt * np.arange(Nsteps))


def test_wrong_length_schedule_raises():
    proc = OverdampedProcess(F=X(dim=1), D=0.1, theta_F=jnp.array([-1.0]))
    proc.set_extras(extras_global={"a": time_series_extra(np.zeros(5))})
    proc.initialize(jnp.zeros(1))
    with pytest.raises(ValueError, match="Nsteps"):
        proc.simulate(dt=0.1, Nsteps=10, key=random.PRNGKey(0))


def test_underdamped_schedule_smoke():
    """UD harmonic oscillator with a ramped drive amplitude runs and
    attaches the schedule."""
    Nsteps, dt = 200, 0.05
    ramp = np.linspace(0.5, 1.5, Nsteps)

    drive = extra_scalar("amp", dim=1) * unit_vector_basis(1)
    Fxv = make_sf(
        lambda x, v, *, extras: (-x - 0.5 * v + extras["amp"])[..., None],
        dim=1,
        rank=1,
        n_features=1,
        needs_v=True,
        extras_keys=("amp",),
    )
    proc = UnderdampedProcess(Fxv.to_psf() if hasattr(Fxv, "to_psf") else Fxv, D=0.1)
    proc.set_extras(extras_global={"amp": time_series_extra(ramp)})
    proc.initialize(jnp.zeros(1), v0=jnp.zeros(1))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(4), oversampling=2)
    assert np.asarray(coll.datasets[0]._X3d()).shape[0] == Nsteps
    out = coll.datasets[0].extras_global["amp"]
    assert isinstance(out, TimeSeriesExtra)
    np.testing.assert_allclose(np.asarray(out.data), ramp)
    assert "time_dependent_extras" in coll.datasets[0].meta
    _ = drive  # compositional form exercised in the OD tests


def test_reserved_time_clock_is_unified_across_sim_and_inference():
    """A force reading the reserved per-frame ``time`` can be simulated, and the
    inference layer materialises the identical clock for the produced data."""
    dt, Nsteps = 0.1, 30
    # F(x, t) = c * t * e_x — drift driven purely by the reserved clock.
    F = extra_scalar("time", dim=1) * unit_vector_basis(1)
    proc = OverdampedProcess(F=F, D=0.01, theta_F=jnp.array([1.0]))
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(0))

    x = np.asarray(coll.datasets[0]._X3d())[:, 0, 0]
    assert x[-1] > 2.0  # the growing time-driven drift carries x forward

    # Inference/diagnostics resolve the same per-frame clock.
    t_extra = np.asarray(coll.datasets[0].build_extras(jnp.arange(5))["time"])
    np.testing.assert_allclose(t_extra, np.arange(5) * dt, atol=1e-6)


def test_state_dependent_diffusion_with_schedule_smoke():
    """Covers the _B_fn extras-argument change."""
    Nsteps, dt = 100, 0.02
    ramp = np.linspace(1.0, 2.0, Nsteps)

    # D(x) = (0.1 + 0.05 x^2) I — state-dependent, schedule alongside.
    D_sf = make_sf(
        lambda x, *, extras: (0.1 + 0.05 * jnp.sum(x**2)) * jnp.eye(x.shape[-1]),
        dim=1,
        rank=2,
        n_features=1,
    )
    F = -(extra_scalar("k", dim=1) * X(dim=1))
    proc = OverdampedProcess(F=F.to_psf() if hasattr(F, "to_psf") else F, D=D_sf)
    proc.set_extras(extras_global={"k": time_series_extra(ramp)})
    proc.initialize(jnp.ones(1) * 0.3)
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(5), oversampling=2)
    assert np.asarray(coll.datasets[0]._X3d()).shape[0] == Nsteps
