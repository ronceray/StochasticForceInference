"""Missing frames must not bias estimators built from several integrals.

The linear force fit combines a Gram matrix, moments and (Stratonovich /
ULI) a gradient correction, each reading different frames; the linear
diffusion fit combines a Gram matrix and moments.  With randomly missing
frames every integral used to keep the rows where *its own* stencil was
observed, biasing ``G^-1 M`` by the ratio of row counts (e.g. the robust
preset gave k = -0.85 for a true -1 at 5 % loss).  All integrals of one
estimator now share one row set (``SFI.integrate.shared_rows``).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import SFI
from SFI.bases import X, identity_matrix_basis, monomials_up_to, unit_axes, v_components, x_components
from SFI.integrate import Integrand, ExprOperand, Term, integrate, required_frames, shared_rows, stream
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.trajectory import TrajectoryCollection, TrajectoryDataset

LOSS = 0.2


@pytest.fixture(scope="module")
def ou_pair():
    proc = OverdampedProcess(F=-1.0 * X(dim=1), D=0.5)
    proc.initialize(jnp.zeros((40, 1)))
    coll = proc.simulate(dt=0.05, Nsteps=1500, key=random.PRNGKey(0), oversampling=4)
    return coll, coll.degrade(data_loss_fraction=LOSS, seed=1)


def _force_k(coll, preset):
    inf = SFI.OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(order=1, dim=1, rank="vector"), preset=preset)
    inf.compute_force_error()
    return float(inf.force_coefficients_full[1]), float(inf.force_coefficients_stderr[1])


@pytest.mark.parametrize("preset", ["clean", "robust"])
def test_od_force_unbiased_by_frame_loss(ou_pair, preset):
    full, lossy = ou_pair
    k0, _ = _force_k(full, preset)
    k1, err = _force_k(lossy, preset)
    # Same trajectory minus 20 % of the frames: the estimates differ only by
    # the dropped rows' sampling noise (pre-fix: robust shifted by ~10 sigma).
    assert abs(k1 - k0) < 2.0 * err


@pytest.mark.parametrize("method", ["WeakNoise", "noisy"])
def test_od_diffusion_unbiased_by_frame_loss(ou_pair, method):
    full, lossy = ou_pair
    vals = []
    for c in (full, lossy):
        inf = SFI.OverdampedLangevinInference(c)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_diffusion_linear(identity_matrix_basis(1), M_mode=method)
        vals.append(float(np.ravel(inf.diffusion_coefficients_full)[0]))
    # Pre-fix the lossy value was ~20 % low (the Gram counted rows the
    # moments' stencil could not use).
    assert abs(vals[1] / vals[0] - 1.0) < 0.02


def test_ud_force_unbiased_by_frame_loss():
    (x,), (v,), (ex,) = x_components(1), v_components(1), unit_axes(1)
    proc = UnderdampedProcess((-1.0 * x - 1.0 * v) * ex, D=0.5)
    proc.initialize(jnp.zeros((40, 1)), v0=jnp.zeros((40, 1)))
    full = proc.simulate(dt=0.05, Nsteps=1500, key=random.PRNGKey(0), oversampling=8)
    lossy = full.degrade(data_loss_fraction=LOSS, seed=1)
    B = monomials_up_to(order=1, dim=1, include_v=True, rank="vector")
    out = []
    for c in (full, lossy):
        inf = SFI.UnderdampedLangevinInference(c)
        inf.compute_diffusion_constant()
        inf.infer_force_linear(B, preset="robust")
        inf.compute_force_error()
        out.append((np.asarray(inf.force_coefficients_full), np.asarray(inf.force_coefficients_stderr)))
    (c0, _), (c1, e1) = out
    assert np.all(np.abs(c1[1:] - c0[1:]) < 2.0 * e1[1:])


def test_shared_rows_aligns_integrals_and_checks_every_read_frame():
    rng = np.random.default_rng(0)
    T, N = 60, 3
    Xa = rng.normal(size=(T, N, 1))
    M = rng.random((T, N)) > 0.2
    coll = TrajectoryCollection([TrajectoryDataset.from_arrays(X=Xa, dt=0.1, mask=M)], jnp.ones(1))
    ds = coll.datasets[0]

    one = ExprOperand(expr=identity_matrix_basis(1), x=stream("X"), alias="I")
    p_short = Integrand(exprs=[one], terms=[Term(eq="imna->ia", ops=("I",))])
    one_p = ExprOperand(expr=identity_matrix_basis(1), x=stream("X_plusplus"), alias="Ip")
    p_long = Integrand(exprs=[one_p], terms=[Term(eq="imna->ia", ops=("Ip",))])

    kw = dict(reduce="sum", weight_by_dt=False)
    n_short, n_long = float(integrate(coll, p_short, **kw)[0]), float(integrate(coll, p_long, **kw)[0])
    assert n_short > n_long  # each integral keeps the rows its own stencil sees
    frames = required_frames(p_short, p_long)
    assert frames == frozenset({2})  # X_plusplus reads t+2 only (t+1 comes from dt)
    with shared_rows(frames):
        a, b = float(integrate(coll, p_short, **kw)[0]), float(integrate(coll, p_long, **kw)[0])
    assert a == b == n_long

    # dX_plus reads frames t+1 and t+2: both must be observed.
    t = jnp.arange(T - 2)
    m = np.asarray(ds._output_mask_batch({"dX_plus"}, t))
    np.testing.assert_array_equal(m, M[:-2] & M[1:-1] & M[2:])


def test_integrate_returns_the_measure_of_its_own_rows():
    rng = np.random.default_rng(1)
    T, N = 50, 2
    M = rng.random((T, N)) > 0.25
    ds = [TrajectoryDataset.from_arrays(X=rng.normal(size=(T, N, 1)), dt=0.1, mask=M) for _ in range(2)]
    coll = TrajectoryCollection(ds, jnp.array([1.0, 3.0]))
    one = ExprOperand(expr=identity_matrix_basis(1), x=stream("X_plusplus"), alias="I")
    prog = Integrand(exprs=[one], terms=[Term(eq="imna->ia", ops=("I",))])
    # The measure is sum_d w_d sum_t N_active dt over the rows the value sums,
    # so value / measure is the average of the integrand (here 1).
    val, meas = integrate(coll, prog, reduce="sum", return_measure=True)
    assert float(val[0]) == pytest.approx(meas, rel=1e-6)
    want = sum(w * d.Teff(prog.require() | {"__dt__"}) for w, d in zip((1.0, 3.0), ds))
    assert meas == pytest.approx(want, rel=1e-6)
    with shared_rows({-1}):
        val2, meas2 = integrate(coll, prog, reduce="sum", return_measure=True)
    assert meas2 < meas and float(val2[0]) == pytest.approx(meas2, rel=1e-6)
    mean, m3 = integrate(coll, prog, reduce="mean", return_measure=True)
    assert float(mean[0]) == pytest.approx(1.0) and m3 == pytest.approx(meas)
