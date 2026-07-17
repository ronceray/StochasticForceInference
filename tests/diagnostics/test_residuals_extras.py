"""Regression: diagnostics must pass global extras to the force.

The underdamped residual builder used to evaluate the inferred force without
the dataset's extras on the single-particle path
(``SFI/diagnostics/residuals.py``), so any force that references a global
extra raised on a single-particle, multi-dataset run — exactly the overdamped
builder's behaviour, but inconsistently omitted for underdamped.
"""

from __future__ import annotations

import numpy as np
from jax import random
import jax.numpy as jnp

from SFI import (
    OverdampedLangevinInference,
    TrajectoryCollection,
    UnderdampedLangevinInference,
)
from SFI.bases import (
    X,
    dataset_indicator,
    extra_scalar,
    identity_matrix_basis,
    monomials_up_to,
    unit_vector_basis,
)
from SFI.diagnostics import assess
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc.factory import make_basis


def _ud_inferer_with_per_dataset_extra():
    """Two single-particle UD datasets, each carrying its own global extra 'c'.

    The inference basis includes ``extra_scalar('c')`` so the fitted force
    references the extra; diagnostics must supply it when re-evaluating F.
    """
    def force(x, *, v, mask=None):
        return jnp.array([-x[0] - 0.5 * v[0]])

    F = make_basis(force, dim=1, rank=1, n_features=1, needs_v=True)

    sims = []
    for c_val, seed in [(0.3, 1), (-0.4, 2)]:
        proc = UnderdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
        proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([0.3])})
        proc.initialize(jnp.zeros(1, dtype=jnp.float32), v0=jnp.zeros(1, dtype=jnp.float32))
        coll = proc.simulate(dt=0.02, Nsteps=4000, key=random.PRNGKey(seed), prerun=100, oversampling=5)
        ds = coll.datasets[0]
        sims.append(
            TrajectoryCollection.from_arrays(
                X=np.asarray(ds.X), dt=float(ds.dt), extras_global={"c": jnp.array(float(c_val))}
            )
        )
    data = sims[0].concat(sims[1:], weights="pool")

    inf = UnderdampedLangevinInference(data)
    inf.compute_diffusion_constant(method="auto")
    U = unit_vector_basis(1)
    monomials = monomials_up_to(order=1, dim=1, include_constant=True, include_x=True, include_v=True)
    basis = (monomials * U) & (extra_scalar("c", dim=1) * U)  # [1, x, v, c] (x) e_x
    inf.infer_force_linear(basis)
    inf.compute_force_error()
    return inf


def test_assess_underdamped_multidataset_global_extras():
    """assess() must not raise when the UD force references a global extra."""
    inf = _ud_inferer_with_per_dataset_extra()
    report = assess(inf, level="standard")
    assert report.meta["regime"] == "UD"
    assert report.residuals["moments"]["n"] > 0


def _simulate_trap(k, seed, Nsteps=6000, dt=0.01, D=0.2):
    """1-D OU trap with stiffness k (single particle)."""
    B = X(dim=1) & unit_vector_basis(1)
    proc = OverdampedProcess(F=B, D=D, theta_F=jnp.array([-k, 0.0]))
    proc.initialize(jnp.zeros(1))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), oversampling=4, prerun=50)


def test_assess_overdamped_multidataset_dataset_indicator():
    """assess() must inject the reserved ``dataset_index`` for pooled forces."""
    coll = _simulate_trap(1.0, 2).concat([_simulate_trap(2.0, 3)], weights="pool")
    basis = dataset_indicator(2, dim=1) * X(dim=1)  # per-dataset stiffness: 1{ds=d}.x

    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis)
    inf.compute_force_error()

    report = assess(inf, level="standard")
    assert report.meta["regime"] == "OD"
    assert report.residuals["moments"]["n"] > 0
