"""Diagnostics work for multiparticle bases that read per-particle extras.

Regression for the residual builders threading per-particle extras (e.g.
a per-agent ``home_id``) through to the re-evaluated force.  A pair
interaction is included so the composite force has ``pdepth=1`` (a valid
multiparticle force) — which is also the ``N > 1`` branch the fix
touches.
"""

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI import OverdampedLangevinInference, UnderdampedLangevinInference
from SFI.bases import V
from SFI.bases.pairs import gaussian_kernels, radial_pair_basis
from SFI.diagnostics import DiagnosticsReport, assess
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import make_basis


def _per_agent_well(N, dim):
    """One feature per agent: -(x - x0_i) with anchors at the origin."""
    def f(x, *, extras):
        onehot = (jnp.arange(N) == extras["home_id"]).astype(x.dtype)
        return (-x)[:, None] * onehot[None, :]          # (dim, N)
    return make_basis(f, dim=dim, rank=1, n_features=N,
                      extras_keys=("home_id",), particle_extras=("home_id",))


def _pair(dim):
    return radial_pair_basis(gaussian_kernels([0.8]), dim=dim).dispatch_pairs()


def test_overdamped_assess_with_per_particle_extra():
    N, dim = 3, 2
    B = _per_agent_well(N, dim) & _pair(dim)            # pdepth=1, reads home_id
    theta = jnp.asarray(list(np.linspace(0.8, 1.4, N)) + [-0.3])
    proc = OverdampedProcess(B, D=0.2, theta_F=theta)
    proc.set_extras(extras_local={"home_id": jnp.arange(N)})
    proc.initialize(jnp.array([[0.0, 0.0], [1.2, 0.0], [0.6, 1.0]]))
    coll = proc.simulate(dt=0.01, Nsteps=3000, key=random.PRNGKey(0), oversampling=5)

    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(B)
    inf.compute_force_error()

    rep = assess(inf, level="standard")                 # must not raise on home_id
    assert isinstance(rep, DiagnosticsReport)
    assert rep.residuals["moments"]["n"] > 100
    assert 0.5 < rep.residuals["moments"]["std"] < 2.0  # clean, well-specified


def test_underdamped_assess_with_per_particle_extra():
    N, dim = 3, 2
    B = _per_agent_well(N, dim) & V(dim=dim) & _pair(dim)
    theta = jnp.asarray(list(np.linspace(0.8, 1.4, N)) + [-0.6, -0.3])
    proc = UnderdampedProcess(B, D=0.2, theta_F=theta)
    proc.set_extras(extras_local={"home_id": jnp.arange(N)})
    proc.initialize(jnp.array([[0.0, 0.0], [1.2, 0.0], [0.6, 1.0]]),
                    v0=jnp.zeros((N, dim)))
    coll = proc.simulate(dt=0.02, Nsteps=3000, key=random.PRNGKey(1), oversampling=5)

    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(B)
    inf.compute_force_error()

    rep = assess(inf, level="standard")                 # must not raise
    assert isinstance(rep, DiagnosticsReport)
    assert rep.residuals["moments"]["n"] > 100
