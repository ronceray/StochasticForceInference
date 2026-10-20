"""Noise shared between particles: stencils on particle pairs, model-free corrections, intensities (experimental)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI import OverdampedLangevinInference, TrajectoryCollection, UnderdampedLangevinInference
from SFI.inference import overdamped as od
from SFI.inference import underdamped as ud
from SFI.inference.graph_diffusion import estimate_graph_noise, stencil_correction_program, stencil_directions
from SFI.integrate import integrate, stream
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.langevin.graph_noise import GraphNoise, NoiseFamily
from SFI.statefunc import make_basis, make_interactor


def _csr(rows):
    return (jnp.asarray(np.concatenate([[0], np.cumsum([len(r) for r in rows])]), jnp.int32),
            jnp.asarray(np.concatenate(rows), jnp.int32))


def _ring(N, d):
    nb = np.array([[(i - 1) % N, (i + 1) % N] for i in range(N)], np.int32).reshape(-1)
    ex = {"ring/indptr": jnp.asarray(np.arange(0, 2 * N + 1, 2, dtype=np.int32)), "ring/indices": jnp.asarray(nb)}
    ex["edge/indptr"], ex["edge/indices"] = _csr([[i, (i + 1) % N] for i in range(N)])
    sgn = jnp.array([1.0, -1.0])  # antisymmetric bond kick: D_ij = -theta/2, D_ii += theta/2 per bond
    pair = NoiseFamily(lambda Xk, m, e: sgn[:, None, None] * jnp.eye(d)[None] / jnp.sqrt(2.0),
                       "edge/indptr", "edge/indices", n_noise=d, Kmax=2, label="pair")
    return ex, [pair, NoiseFamily.white(d)]


def _od_basis(d):
    coup = make_interactor(lambda Xk: (Xk[1] - Xk[0])[:, None], dim=d, rank=1, K=2, n_features=1)
    return make_basis(lambda y: y[:, None], dim=d, rank=1, n_features=1) & coup.dispatch_pairs_from_extras(
        indptr_key="ring/indptr", indices_key="ring/indices")


def _ud_basis(d):
    coup = make_interactor(lambda Xk, v: (v[1] - v[0])[:, None], dim=d, rank=1, K=2, n_features=1, needs_v=True)
    return make_basis(lambda x, v: jnp.stack([x, v], -1), dim=d, rank=1, n_features=2,
                      needs_v=True) & coup.dispatch_pairs_from_extras(indptr_key="ring/indptr", indices_key="ring/indices")


@pytest.mark.parametrize("dynamics,method", [(dy, m) for dy in ("od", "ud") for m in ("MSD", "WeakNoise", "noisy")])
def test_stencil_diagonal_is_the_sfi_estimator(dynamics, method):
    rng = np.random.default_rng(0)
    s = {k: jnp.asarray(rng.standard_normal((5, 2)), jnp.float32) for k in ("dX", "dX_minus", "dX_plus")}
    s.update(dt=jnp.asarray(0.1), dt_minus=jnp.asarray(0.13), dt_plus=jnp.asarray(0.08), mask_out=jnp.ones(5, bool))
    full = sum(jnp.einsum("im,jn->ijmn", L(**s), y(**s)) for L, y in stencil_directions(method, dynamics))
    ref = {("od", "MSD"): od._D_msd, ("od", "WeakNoise"): od._D_weaknoise, ("od", "noisy"): od._D_noisy,
           ("ud", "MSD"): ud._D_msd_uli, ("ud", "WeakNoise"): ud._D_weaknoise_uli,
           ("ud", "noisy"): ud._D_noisy_uli}[(dynamics, method)]
    expected = np.asarray(ref(**{k: v for k, v in s.items() if k != "mask_out"}))
    np.testing.assert_allclose(np.asarray(jnp.einsum("iimn->imn", full)), expected, rtol=2e-5,
                               atol=1e-6 * np.abs(expected).max())
    np.testing.assert_allclose(np.asarray(full), np.asarray(jnp.transpose(full, (1, 0, 3, 2))),
                               atol=1e-6 * np.abs(expected).max())  # symmetric N d x N d matrix


def test_stencil_correction_equals_explicit_all_pairs_contraction():
    N, d, T = 6, 2, 12
    ex, _ = _ring(N, d)
    B = _od_basis(d)
    X = np.random.default_rng(1).standard_normal((T, N, d)).astype(np.float32)
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=ex)
    A_inv = np.array([[1.5, 0.3], [0.3, 0.8]], np.float32)
    dirs = stencil_directions("noisy", "od")
    val = np.asarray(integrate(coll, stencil_correction_program(B, A_inv, dirs, x=stream("X")), reduce="sum",
                               weight_by_dt=False))
    ref = 0
    for t in range(1, T - 1):
        dX, dXm = X[t + 1] - X[t], X[t] - X[t - 1]
        D = 0.25 * (np.einsum("im,jn->ijmn", dX, dX) + 2 * np.einsum("im,jn->ijmn", dX, dXm)
                    + 2 * np.einsum("im,jn->ijmn", dXm, dX) + np.einsum("im,jn->ijmn", dXm, dXm)) / 0.1
        J = np.asarray(jax.jacfwd(lambda xx: B(xx, extras=ex))(jnp.asarray(X[t])))  # (i, o, a, j, m)
        ref = ref + np.einsum("ijnm,no,ioajm->a", D, A_inv, J)
    np.testing.assert_allclose(val, ref, rtol=1e-4)


def _od_ring_data(dt, nsteps, loc=0.0, theta=(0.4, 0.05), N=12, d=2, seed=0):
    ex, fams = _ring(N, d)
    B = _od_basis(d)
    proc = OverdampedProcess(B, D=GraphNoise(fams, list(theta), dim=d), theta_F=jnp.array([-1.0, 0.8]))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.asarray(np.random.default_rng(seed).standard_normal((N, d)), jnp.float32))
    X = np.asarray(proc.simulate(dt=dt, Nsteps=nsteps, key=jax.random.PRNGKey(seed + 3), oversampling=4).datasets[0].X)
    if loc:
        X = X + loc * np.random.default_rng(seed + 7).standard_normal(X.shape).astype(X.dtype)
    return TrajectoryCollection.from_arrays(X=X, dt=dt, extras_global=ex), B, fams


def test_intensities_from_stencils_with_localisation_noise():
    coll, _, fams = _od_ring_data(0.01, 12000, loc=0.03)
    noisy = estimate_graph_noise(coll, fams, stencil="noisy")
    msd = estimate_graph_noise(coll, fams, stencil="MSD")
    np.testing.assert_allclose(noisy.theta, [0.4, 0.05], atol=0.05)
    assert msd.theta[1] > 0.05 + 0.05  # the MSD stencil puts Lambda/dt into the per-particle family


def test_overdamped_stratonovich_with_shared_noise():
    from SFI.inference.graph_diffusion import SharedNoiseOLI

    coll, B, _ = _od_ring_data(0.01, 16000)

    def coeffs(inf, **kw):
        inf.compute_diffusion_constant(method="MSD")
        inf.infer_force_linear(B, preset="robust", **kw)
        return np.asarray(inf.force_coefficients)

    c_same = coeffs(OverdampedLangevinInference(coll), cross_particle="same-particle")
    c_default = coeffs(OverdampedLangevinInference(coll), correction_stencil="noisy")
    c_new = coeffs(SharedNoiseOLI(coll))
    assert abs(c_same[0] + 1.0) > 2.0 and c_same[1] < 0  # stiffness x4, coupling sign flipped
    np.testing.assert_allclose(c_new, [-1.0, 0.8], atol=0.1)
    np.testing.assert_allclose(c_default, c_new, rtol=1e-5, atol=1e-6)  # all pairs is the engine default
    np.testing.assert_allclose(coeffs(OverdampedLangevinInference(coll)), [-1.0, 0.8], atol=0.1)  # lag-2 stencil


def test_underdamped_with_shared_noise():
    from SFI.inference.graph_diffusion import SharedNoiseULI

    N, d, dt = 16, 2, 0.02
    ex, fams = _ring(N, d)
    B = _ud_basis(d)
    theta_F = np.array([-1.0, -0.5, 0.8])
    proc = UnderdampedProcess(B, D=GraphNoise(fams, [0.4, 0.05], dim=d), theta_F=jnp.asarray(theta_F))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.asarray(np.random.default_rng(0).standard_normal((N, d)), jnp.float32))
    X = np.asarray(proc.simulate(dt=dt, Nsteps=12000, key=jax.random.PRNGKey(1), oversampling=10).datasets[0].X)
    coll = TrajectoryCollection.from_arrays(X=X, dt=dt, extras_global=ex)
    np.testing.assert_allclose(estimate_graph_noise(coll, fams, dynamics="ud", stencil="WeakNoise").theta,
                               [0.4, 0.05], atol=0.04)

    def coeffs(inf, **kw):
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, preset="robust", **kw)
        return np.asarray(inf.force_coefficients)

    c_same = coeffs(UnderdampedLangevinInference(coll), cross_particle="same-particle")
    c_default = coeffs(UnderdampedLangevinInference(coll))
    c_new = coeffs(SharedNoiseULI(coll))
    assert abs(c_same[1] - theta_F[1]) > 1.0 and c_same[2] < 0  # friction x6, coupling sign flipped
    np.testing.assert_allclose(c_new, theta_F, atol=0.15)
    np.testing.assert_allclose(c_default, c_new, rtol=1e-5, atol=1e-6)  # all pairs is the engine default


def test_lag2_stencil_removes_the_dt_bias_of_the_stratonovich_estimator():
    """OD OU data from the exact discretisation: the Vestergaard-Stratonovich + shift-Gram estimate has an
    O(dt) bias; adding -1/2 dX_{t-2} (x) dX_t / dt to the stencil cancels it (exact analysis for linear
    dynamics), and leaves the cancellation of localisation noise intact."""
    from SFI.inference.graph_diffusion import SharedNoiseOLI

    rng = np.random.default_rng(0)
    N, T, dt, k, D = 3000, 1200, 0.15, 1.0, 1.0
    a = np.exp(-k * dt)
    X = np.zeros((T, N, 1))
    X[0] = rng.standard_normal((N, 1)) * np.sqrt(D / k)
    for t in range(1, T):
        X[t] = a * X[t - 1] + np.sqrt(D / k * (1 - a * a)) * rng.standard_normal((N, 1))
    B = make_basis(lambda x: x[:, None], dim=1, rank=1, n_features=1)

    def fit(stencil, loc=0.0, cls=OverdampedLangevinInference):
        Y = X + loc * rng.standard_normal(X.shape) if loc else X
        inf = cls(TrajectoryCollection.from_arrays(X=Y.astype(np.float32), dt=dt))
        kw = {"correction_stencil": stencil}
        if cls is SharedNoiseOLI:
            inf.set_cross_particle("stencil", stencil=stencil)
            kw = {}
        inf.compute_diffusion_constant(method="noisy")
        inf.infer_force_linear(B, preset="robust", **kw)
        return float(np.asarray(inf.force_coefficients)[0])

    e1, e2 = abs(fit("noisy") + k), abs(fit("noisy_lag2") + k)
    assert fit(None) == fit("noisy_lag2")  # the robust preset's default
    assert e1 > 0.03 and e2 < 0.3 * e1
    assert abs(fit("noisy_lag2", loc=0.1) + k) < 0.3 * e1
    assert fit("noisy_lag2", cls=SharedNoiseOLI) == pytest.approx(fit("noisy_lag2"), rel=1e-5)
