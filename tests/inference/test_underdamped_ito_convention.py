"""Itô-convention control for the underdamped linear estimators.

Under multiplicative noise the velocity-space drift that any estimator can
return is the Itô drift ``a = ∇_v·D + D F``; the symmetric ULI stencil
converts its mid-point projection to it through the gradient term.  On the
exact-in-basis model

    dv = (−x − v) dt + sqrt(2 (D0 + D2 v²)) dW        (Itô)

the v-coefficient of the fitted acceleration discriminates the conventions:
−1 (Itô drift), −1 − 2 D2 (Itô − ∂_v D), −1 − D2 (Itô − ½ ∂_v D).  A raw
regression of the symmetric second difference ⟨Δv̂⟩/Δt on (x̂, v̂) returns
≈ 0 for the same coefficient: a mid-point projection is not a drift and
must never be exposed as one.

Reference numbers (SpuriousFun ``run_uli_control2.py``, 400 walkers ×
40 000 frames): Δt = 0.005 → v-coefficient −0.9986, D2 = 0.0987, raw
regression v-coefficient −0.005; Δt = 0.02 → −1.058 (the O(Δt γ) bias).
"""

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI.bases import monomials_up_to, unit_axes, v_components, x_components
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import UnderdampedProcess
from SFI.statefunc import make_sf

D0, D2 = 0.5, 0.10
DT = 0.005
N_WALKERS, N_STEPS = 400, 5_000


def _simulate(seed=0):
    x0c, = x_components(1)
    v0c, = v_components(1)
    e0, = unit_axes(1)
    F = (-x0c - v0c) * e0
    D = make_sf(lambda x, v: (D0 + D2 * v[0] ** 2) * jnp.eye(1), dim=1, rank=2, needs_v=True)
    proc = UnderdampedProcess(F, D=D)
    rng = np.random.default_rng(seed)
    proc.initialize(jnp.asarray(rng.normal(size=(N_WALKERS, 1))), v0=jnp.asarray(rng.normal(size=(N_WALKERS, 1))))
    return proc.simulate(dt=DT, Nsteps=N_STEPS, key=random.PRNGKey(seed), prerun=500, oversampling=10)


def test_uli_returns_the_ito_drift_under_multiplicative_noise():
    coll = _simulate()
    B3 = monomials_up_to(3, dim=1, include_x=True, include_v=True, rank="vector")
    B2 = monomials_up_to(2, dim=1, include_x=True, include_v=True, rank="identity_matrix")

    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(B3, preset="clean")
    c = np.asarray(inf.force_coefficients).ravel()
    iv = list(B3.labels).index("v0·e0")
    ix = list(B3.labels).index("x0·e0")

    # Itô drift: -1.00; the alternatives -1.2 (Itô - d_vD) and -1.1 (Itô - d_vD/2) are excluded.
    assert abs(c[iv] + 1.0) < 0.06, c[iv]
    assert abs(c[ix] + 1.0) < 0.06, c[ix]
    assert abs(c[iv] - (-1.0 - 2 * D2)) > 0.1
    assert abs(c[iv] - (-1.0 - D2)) > 0.05

    inf.infer_diffusion_linear(B2)
    cd = np.asarray(inf.diffusion_coefficients).ravel()
    lab = list(B2.labels)
    assert abs(cd[lab.index("1·I")] - D0) < 0.03, cd
    assert abs(cd[lab.index("v0^2·I")] - D2) < 0.02, cd
    for name in ("x0·I", "v0·I", "x0^2·I", "(x0·v0)·I"):
        assert abs(cd[lab.index(name)]) < 0.03, (name, cd)

    # The mid-point (symmetric second-difference) regression is NOT a drift:
    # its v-coefficient vanishes on this model.
    X = np.asarray(coll.datasets[0].X, dtype=np.float64)[:, :, 0]      # (T, B)
    xh = (X[:-2] + X[1:-1] + X[2:]) / 3.0
    vh = (X[2:] - X[:-2]) / (2.0 * DT)
    dvh = (X[2:] - 2.0 * X[1:-1] + X[:-2]) / DT
    A = np.c_[xh.ravel(), vh.ravel(), np.ones(xh.size)]
    coef = np.linalg.lstsq(A, (dvh / DT).ravel(), rcond=None)[0]
    assert abs(coef[1]) < 0.1, coef            # v-coefficient ≈ 0 (reference: -0.005)
    assert abs(coef[0] + 1.0) < 0.15, coef     # x-coefficient ≈ -1 (reference: -0.92)
