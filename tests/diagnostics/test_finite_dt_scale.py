"""``finite_dt_scale``: the per-step drift-gradient scale of a fitted force.

On a linear OU force F = -K x the Jacobian is constant, so the RMS scale is
‖K‖_F Δt exactly (up to the statistical error of the fitted K): it doubles
with Δt and reads ``"fine"`` at Δt‖K‖ ≈ 0.014.  For an underdamped fit the
gradient is taken with respect to the velocity (the friction per step,
Δt γ).  A stiff OU sampled coarsely is reported as coarse / unreliable and
flagged by ``assess``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI.bases import monomials_up_to, unit_axes, v_components, x_components
from SFI.diagnostics import assess, finite_dt_scale
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import OverdampedProcess, UnderdampedProcess


def _ou_inferer(dt, K=(0.8, 1.2), D0=0.2, Nsteps=3000, seed=0):
    x0c, x1c = x_components(2)
    e0, e1 = unit_axes(2)
    F = (-K[0] * x0c) * e0 + (-K[1] * x1c) * e1
    proc = OverdampedProcess(F, D=D0)
    proc.initialize(jnp.zeros(2, dtype=jnp.float32))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=50, oversampling=5)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=2, rank="vector"), preset="clean")
    return inf


def test_scale_is_the_drift_gradient_per_step_and_linear_in_dt():
    r1 = finite_dt_scale(_ou_inferer(0.01))
    r2 = finite_dt_scale(_ou_inferer(0.02))
    k_norm = float(np.sqrt(0.8 ** 2 + 1.2 ** 2))
    assert r1["available"] and r1["variable"] == "x"
    assert abs(r1["grad_phi_dt"] / (k_norm * 0.01) - 1.0) < 0.15, r1
    assert abs(r2["grad_phi_dt"] / r1["grad_phi_dt"] - 2.0) < 0.3, (r1, r2)
    assert r1["verdict"] == "fine" and r2["verdict"] == "fine"
    assert r1["grad_p99_dt"] >= 0.9 * r1["grad_phi_dt"]      # constant Jacobian: p99 ≈ rms


def test_underdamped_scale_is_the_friction_per_step():
    x0c, x1c = x_components(2)
    v0c, v1c = v_components(2)
    e0, e1 = unit_axes(2)
    gamma, dt = 1.0, 0.01
    F = (-x0c - gamma * v0c) * e0 + (-x1c - gamma * v1c) * e1
    proc = UnderdampedProcess(F, D=0.5)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    coll = proc.simulate(dt=dt, Nsteps=6000, key=random.PRNGKey(1), prerun=200, oversampling=8)
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(monomials_up_to(1, dim=2, include_v=True, rank="vector"), preset="clean")
    r = finite_dt_scale(inf)
    assert r["available"] and r["variable"] == "v"
    assert abs(r["grad_phi_dt"] / (gamma * np.sqrt(2.0) * dt) - 1.0) < 0.25, r
    assert r["verdict"] == "fine"


def test_assess_reports_and_flags_coarse_sampling():
    inf = _ou_inferer(0.05, K=(8.0, 12.0), Nsteps=4000)
    rep = assess(inf, level="standard")
    fd = rep.finite_dt
    assert fd["available"] and fd["verdict"] in ("coarse", "unreliable"), fd
    assert fd["grad_phi_dt"] > 0.2
    assert any("finite_dt" in m for m in rep.flag_issues())
    rep.print_summary()
    assert "finite_dt" in rep.to_dict()

    fine = assess(_ou_inferer(0.01), level="standard")
    assert fine.finite_dt["verdict"] == "fine"
    assert not any("finite_dt" in m for m in fine.flag_issues())
    assert assess(_ou_inferer(0.01), level="minimal").finite_dt == {}
