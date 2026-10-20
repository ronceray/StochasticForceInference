"""compute_diffusion_error: sandwich covariance of the linear diffusion fit.

For a constant-D overdamped process with a single constant feature, the
local estimators have closed-form long-run variances per frame (clean data,
small dt):

* MSD       ``D̂_t = dX_t²/(2dt)``                 →  2 D²  (no serial correlation)
* WeakNoise ``D̂_t = (dX_{t+1} - dX_t)²/(4dt)``    →  2 D² (1 + 2·¼) = 3 D²

so the standard error of the fitted constant is ``D·sqrt(v/N)``.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import SFI
from SFI.bases import identity_matrix_basis, monomials_up_to, unit_axes, x_components
from SFI.langevin import OverdampedProcess

D_TRUE = 0.5
T = 20_000


@pytest.fixture(scope="module")
def ou_coll():
    (x,) = x_components(1)
    (ex,) = unit_axes(1)
    proc = OverdampedProcess(-1.0 * x * ex, D=D_TRUE)
    proc.initialize(jnp.array([0.0]))
    return proc.simulate(dt=0.01, Nsteps=T, key=random.PRNGKey(3), prerun=200, oversampling=4)


@pytest.mark.parametrize("method, long_run_var", [("MSD", 2.0), ("WeakNoise", 3.0)])
def test_constant_D_stderr_matches_closed_form(ou_coll, method, long_run_var):
    inf = SFI.OverdampedLangevinInference(ou_coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(identity_matrix_basis(1), M_mode=method)
    inf.compute_diffusion_error()
    se = float(np.asarray(inf.diffusion_coefficients_stderr)[0])
    expected = D_TRUE * np.sqrt(long_run_var / T)
    assert se == pytest.approx(expected, rel=0.1)
    assert inf.diffusion_inferred.param_cov is not None


def test_covariance_is_symmetric_psd_on_tensor_basis():
    x, y = x_components(2)
    ex, ey = unit_axes(2)
    proc = OverdampedProcess((-1.0 * x) * ex + (-1.5 * y) * ey, D=jnp.array([[0.5, 0.2], [0.2, 0.3]]))
    proc.initialize(jnp.zeros(2))
    coll = proc.simulate(dt=0.02, Nsteps=4000, key=random.PRNGKey(5), prerun=200, oversampling=4)
    inf = SFI.OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=2, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(monomials_up_to(1, dim=2, rank="symmetric_matrix"))
    inf.compute_diffusion_error()
    cov = np.asarray(inf.diffusion_coefficients_covariance)
    p = len(np.asarray(inf.diffusion_support))
    assert cov.shape == (p, p)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_integrate_rows_sums_to_integrate(ou_coll):
    from SFI.integrate.api import integrate, integrate_rows

    inf = SFI.OverdampedLangevinInference(ou_coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(identity_matrix_basis(1), M_mode="WeakNoise")
    from SFI.integrate.integrand import Integrand, Term

    D_op = inf.get_diffusion_timeop("WeakNoise")
    prog = Integrand(exprs=[inf._diffusion_basis_operand()], times=[D_op],
                     terms=[Term(eq="imna,imn->ia", ops=("B", D_op.alias))])
    total = np.asarray(integrate(ou_coll, prog, reduce="sum", weight_by_dt=False))
    rows = integrate_rows(ou_coll, prog, weight_by_dt=False)
    summed = sum(np.asarray(Y).sum(axis=0) for _, Y in rows.values())
    np.testing.assert_allclose(summed, total, rtol=1e-4)
    (t_idx, _), = rows.values()
    assert np.all(np.diff(t_idx) == 1)
