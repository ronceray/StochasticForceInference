"""Tests for inf.force/diffusion_comparison_arrays and inf.comparison_scatter."""

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np

from SFI import OverdampedLangevinInference, UnderdampedLangevinInference
from SFI.bases import monomials_up_to
from SFI.langevin import OverdampedProcess, UnderdampedProcess


def _od_fit():
    from SFI.bases import X

    proc = OverdampedProcess(F=-1.0 * X(dim=2), D=0.4)
    proc.initialize(jnp.array([0.5, -0.3]))
    coll = proc.simulate(dt=0.02, Nsteps=3000, key=jax.random.PRNGKey(3))
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(monomials_up_to(order=1, dim=2, rank="vector"))
    inf.compute_force_error()
    return inf, proc


def test_force_comparison_arrays_shapes_and_agreement():
    inf, proc = _od_fit()
    Fe, Fi = inf.force_comparison_arrays(model_exact=proc, maxpoints=1500)
    assert Fe.shape == Fi.shape
    assert Fe.ndim == 2 and Fe.shape[1] == 2
    # linear OU recovered well: normalized MSE small
    nmse = np.mean((Fe - Fi) ** 2) / np.mean(Fe**2)
    assert nmse < 0.2


def test_comparison_scatter_returns_axes():
    inf, proc = _od_fit()
    ax = inf.comparison_scatter(model_exact=proc, field="force", maxpoints=800)
    assert ax is not None
    assert len(ax.collections) >= 1  # the scatter


def test_diffusion_comparison_arrays_constant():
    inf, proc = _od_fit()
    De, Di = inf.diffusion_comparison_arrays(model_exact=proc, maxpoints=500)
    assert De.shape == Di.shape
    assert De.shape[1:] == (2, 2)


def test_force_comparison_arrays_underdamped_velocity_path():
    # underdamped: force depends on v, exercises the fd_velocity branch
    from SFI.bases import V, X

    proc = UnderdampedProcess(F=-1.0 * X(dim=1) - 0.5 * V(dim=1), D=0.3)
    proc.initialize(jnp.array([1.0]), jnp.array([0.0]))
    coll = proc.simulate(dt=0.01, Nsteps=4000, key=jax.random.PRNGKey(5))
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(monomials_up_to(order=1, dim=1, include_v=True, rank="vector"))
    inf.compute_force_error()
    Fe, Fi = inf.force_comparison_arrays(model_exact=proc, maxpoints=1500)
    assert Fe.shape == Fi.shape and Fe.shape[1] == 1
