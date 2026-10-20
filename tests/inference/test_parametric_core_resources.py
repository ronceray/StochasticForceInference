"""Resource bounds of the parametric core — scheduling only, same numbers.

* **Measured chunk planning**: each per-chunk program (Gram, NLL, the
  fixed-θ cache, the differentiated L-BFGS loss) is sized from XLA's buffer
  assignment of the compiled program, so ``max_memory_gb`` bounds the
  working set — the static per-point estimate missed the per-stage
  θ-recursion intermediates by an order of magnitude, and the old
  1000-residual floor overrode small budgets.  Chunking leaves results
  unchanged up to float reassociation.
* **∂F/∂θ AD mode**: reverse mode when the model has more parameters than
  the frame force has entries (a neural network on one particle), forward
  otherwise — the same derivative.
* **Early Gauss–Newton runaway fallback**: an accepted step whose trial Gram
  has collapsed is the sensitivity runaway; it is flagged at that step,
  before the (D, Λ) reprofile and the next Gram build, and the ``auto``
  fallback (the frozen-precision L-BFGS, restarted from the initial θ) is
  bit-identical to an explicit ``inner="lbfgs"`` solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases import X
from SFI.inference.parametric_core import flow_multi
from SFI.inference.parametric_core.driver import gn_minimize
from SFI.inference.parametric_core.gram import unpack_gram
from SFI.inference.parametric_core.runner import make_exact_runs_od
from SFI.integrate.api import _xla_bytes
from SFI.statefunc import make_psf
from SFI.statefunc.params import ParamSpec
from SFI.trajectory.collection import TrajectoryCollection


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# ── fixtures ─────────────────────────────────────────────────────────────


def _double_well(T=1500, dt=0.02, seed=3, noise=0.0):
    rng = np.random.default_rng(seed)
    X_ = np.zeros((T, 2))
    for t in range(T - 1):
        x = X_[t]
        F = np.array([-x[0] * (x[0] ** 2 - 1.0), -x[1]])
        X_[t + 1] = x + F * dt + np.sqrt(2 * 0.5 * dt) * rng.normal(size=2)
    X_ = X_ + noise * rng.standard_normal(X_.shape)
    return TrajectoryCollection.from_arrays(X=X_, dt=dt)


def _mlp(H=4):
    """Potential MLP force (float32 parameter template, the library default)."""
    U = (X(dim=2).rank_to_features()
         .dense(H, weight="W1", bias="b1").elementwisemap(jnp.tanh)
         .dense(H, weight="W2", bias="b2").elementwisemap(jnp.tanh)
         .dense(1, weight="W3", bias=None))
    return -(U.d_x())


def _mlp_theta(F, seed=0, scale=0.5):
    n = int(F.template.size)
    return scale * random.normal(random.PRNGKey(seed), (n,), dtype=jnp.float64)


def _tanh_net_f64(H=6):
    """Nonlinear-in-θ vector field with a float64 parameter template:
    ``F(x) = Aᵀ tanh(Wᵀx + b)`` — 5H parameters, 2 outputs."""
    def f(x, *, params):
        z = jnp.tanh(x @ params["W"] + params["b"])
        return z @ params["A"]

    return make_psf(f, dim=2, rank=1, params=[
        ParamSpec("W", (2, H), dtype=jnp.float64),
        ParamSpec("b", (H,), dtype=jnp.float64),
        ParamSpec("A", (H, 2), dtype=jnp.float64)])


def _linear_psf_f64():
    """Linear-in-θ field outside the Basis fast path: ``F(x) = Mx + c``."""
    def f(x, *, params):
        return params["M"] @ x + params["c"]

    return make_psf(f, dim=2, rank=1, params=[
        ParamSpec("M", (2, 2), dtype=jnp.float64),
        ParamSpec("c", (2,), dtype=jnp.float64)])


def _gram(runs, theta, D, Lam, n):
    G, f, H, nll = unpack_gram(runs.gram(theta, D, Lam), n)
    return np.asarray(G), np.asarray(f), np.asarray(H), float(nll)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _noise():
    """(D, Λ) in float64 — built after the module's x64 switch."""
    return 0.5 * jnp.eye(2, dtype=jnp.float64), 1e-3 * jnp.eye(2, dtype=jnp.float64)


# ── measured chunk planning ──────────────────────────────────────────────


def test_measured_chunks_respect_the_budget_per_program():
    coll = _double_well()
    ds = coll.datasets[0]
    F = _mlp(H=4)
    n = int(F.template.size)
    budget = 3 * 2**20
    runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4",
                              w=1.0, chunk_target_bytes=budget)
    C_gram = runs._resolve_chunk_pts(ds, "gram", dtype=jnp.float64)
    C_nll = runs._resolve_chunk_pts(ds, "nll", dtype=jnp.float64)
    # sized below the old 1000-residual floor, separately per program (the
    # Gram carries n-wide ψ / instrument columns, the NLL does not)
    assert 1 <= C_gram < 1000
    assert C_gram < C_nll
    for kind, C in (("gram", C_gram), ("nll", C_nll)):
        fn, args = runs._probe_program(kind, (), ds, 0, C, jnp.float64)
        assert _xla_bytes(fn, *args) <= budget, kind

    # chunking leaves the Gram unchanged (float reassociation only)
    theta = _mlp_theta(F)
    ref = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4",
                             w=1.0)                      # single chunk
    for a, b in zip(_gram(runs, theta, *_noise(), n), _gram(ref, theta, *_noise(), n)):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9 * np.max(np.abs(b)))


def test_chunked_loss_value_and_gradient_match_single_chunk():
    # Several loss chunks run under per-chunk rematerialisation
    # (jax.checkpoint); value and gradient match the single-chunk program.
    coll = _double_well(T=1200)
    ds = coll.datasets[0]
    F = _mlp(H=4)
    theta_f = _mlp_theta(F, seed=1)
    theta_l = theta_f + 0.05 * _mlp_theta(F, seed=2)
    small = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4",
                               w=0.0, chunk_target_bytes=2**20)
    assert small._resolve_chunk_pts(ds, "loss", dtype=jnp.float64) < 1199
    ref = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4",
                             w=0.0)

    def vg(runs):
        return jax.jit(jax.value_and_grad(
            lambda tl: runs.loss(tl, theta_f, *_noise())))(theta_l)

    (v1, g1), (v0, g0) = vg(small), vg(ref)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-11)
    # The gradient reaches θ through the float32 parameter template, whose
    # reverse-mode cast rounds each chunk's cotangent to float32 (both runs
    # sit ~3e-8 from the forward-mode gradient): float32-level agreement.
    assert _rel(np.asarray(g1), np.asarray(g0)) < 1e-6


def test_planner_falls_back_to_static_guess_without_memory_analysis(monkeypatch):
    import SFI.integrate.api as api

    monkeypatch.setattr(api, "_xla_bytes", lambda fn, *a: None)
    coll = _double_well(T=800)
    F = _mlp(H=4)
    runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4",
                              w=1.0, chunk_target_bytes=2**20)
    ds = coll.datasets[0]
    C = runs._resolve_chunk_pts(ds, "gram", dtype=jnp.float64)
    assert C == min(799, runs._static_chunk_pts(ds, "gram"))


# ── ∂F/∂θ: reverse vs forward mode ───────────────────────────────────────


def test_ad_mode_rule_compares_parameter_and_output_sizes(monkeypatch):
    used = []
    real_rev, real_fwd = jax.jacrev, jax.jacfwd

    def spy(name, real):
        def wrapped(*a, **k):
            used.append(name)
            return real(*a, **k)
        return wrapped

    monkeypatch.setattr(jax, "jacrev", spy("rev", real_rev))
    monkeypatch.setattr(jax, "jacfwd", spy("fwd", real_fwd))
    th = jnp.arange(5.0)
    flow_multi._param_jacobian(lambda t: jnp.stack([t.sum(), t @ t]), th)  # 5 > 2
    flow_multi._param_jacobian(lambda t: jnp.tile(t, 2), th)               # 5 < 10
    flow_multi._param_jacobian(lambda t: t * t, th)                        # tie
    assert used == ["rev", "fwd", "fwd"]


@pytest.mark.parametrize("model", ["tanh_net", "linear"])
def test_reverse_mode_dFdth_matches_forward_exactly(model, monkeypatch):
    # float64 parameter templates: forward and reverse mode agree to float64
    # round-off, on the per-frame ∂F/∂θ and on the assembled Gram.
    F = _tanh_net_f64() if model == "tanh_net" else _linear_psf_f64()
    n = int(F.template.size)
    assert n > 2                                     # → reverse mode
    theta = 0.7 * random.normal(random.PRNGKey(4), (n,), dtype=jnp.float64)
    Xf = jnp.asarray(np.random.default_rng(0).normal(size=(5, 1, 2)))
    rev = jax.vmap(flow_multi._dFdth_builder(F, theta, None))(Xf)
    fwd = jax.vmap(lambda x: jax.jacfwd(
        lambda th: F(x, params=F.unflatten_params(th)))(theta))(Xf)
    assert _rel(np.asarray(rev), np.asarray(fwd)) < 1e-13

    coll = _double_well(T=600)
    out = {}
    for mode in ("auto", "fwd"):
        if mode == "fwd":
            monkeypatch.setattr(flow_multi, "_param_jacobian",
                                lambda f, th, f_rev=None: jax.jacfwd(f)(th))
        runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1,
                                  integrator="rk4", w=1.0)
        out[mode] = _gram(runs, theta, *_noise(), n)
    for a, b in zip(out["auto"][:3], out["fwd"][:3]):
        assert _rel(a, b) < 1e-11


def test_reverse_mode_is_exact_under_the_float32_template(monkeypatch):
    # The parameter template defaults to float32.  Before float64 values
    # were kept by float specs (``params._cast_dtype``), plain reverse mode
    # rounded each cotangent to float32 at the parameter leaves (~1e-7 noise
    # in ψ, amplified by an ill-conditioned Gauss–Newton Gram); the builder
    # differentiates in float64 either way.  Both must now match forward
    # mode to float64 round-off.
    F = _mlp(H=6)
    n = int(F.template.size)
    theta = _mlp_theta(F, seed=5)
    Xf = jnp.asarray(np.random.default_rng(1).normal(size=(7, 1, 2)))
    rev = jax.vmap(flow_multi._dFdth_builder(F, theta, None))(Xf)
    fwd = jax.vmap(lambda x: jax.jacfwd(
        lambda th: F(x, params=F.unflatten_params(th)))(theta))(Xf)
    plain_rev = jax.vmap(lambda x: jax.jacrev(
        lambda th: F(x, params=F.unflatten_params(th)))(theta))(Xf)
    assert _rel(np.asarray(rev), np.asarray(fwd)) < 1e-13
    assert _rel(np.asarray(plain_rev), np.asarray(fwd)) < 1e-13   # no float32 rounding left

    coll = _double_well(T=600)
    out = {}
    for mode in ("auto", "fwd"):
        if mode == "fwd":
            monkeypatch.setattr(flow_multi, "_param_jacobian",
                                lambda f, th, f_rev=None: jax.jacfwd(f)(th))
        runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1,
                                  integrator="rk4", w=1.0)
        out[mode] = _gram(runs, theta, *_noise(), n)
    for a, b in zip(out["auto"][:3], out["fwd"][:3]):
        assert _rel(a, b) < 1e-11


# ── early Gauss–Newton runaway fallback ──────────────────────────────────


def test_gn_flags_a_collapsed_first_step_before_reprofiling():
    # Synthetic runaway: the GN step leads where the Gram has collapsed by
    # 1e-9 while the merit ‖f‖ still drops — the accepted step is the
    # runaway.  It is flagged at iteration 0: no (D, Λ) reprofile, no
    # further Gram build, and the healthy start is returned.
    n = 3
    theta0 = jnp.ones(n)
    grams, profiles = [], []

    def gram_fn(th, D, L):
        grams.append(np.asarray(th))
        if float(jnp.linalg.norm(th - theta0)) < 1e-12:
            return jnp.eye(n), -1e2 * jnp.ones(n), 0.0      # healthy start
        return 1e-9 * jnp.eye(n), 1e-9 * jnp.ones(n), 0.0  # collapsed

    def profile_fn(th, D, L):
        profiles.append(1)
        return D, L

    th, D, L, info = gn_minimize(theta0, gram_fn, jnp.eye(2), jnp.eye(2),
                                 profile_fn=profile_fn, max_iter=5,
                                 reprofile_iters=1, merit="phi")
    assert info["diverged"] is True
    assert info["outer_iterations"] == 1
    assert not profiles
    assert len(grams) == 2                          # start + the trial step
    np.testing.assert_array_equal(np.asarray(th), np.asarray(theta0))


def test_auto_fallback_is_bit_identical_to_explicit_lbfgs(monkeypatch):
    from SFI.inference.parametric_core import runner as R
    from SFI.inference.parametric_core.solve import solve_force_od

    prepares = []
    real_prepare = R.ExactRuns.prepare

    def spy_prepare(self, *a, **k):
        prepares.append(self._w)
        return real_prepare(self, *a, **k)

    monkeypatch.setattr(R.ExactRuns, "prepare", spy_prepare)
    coll = _double_well(T=2500, noise=0.0)
    F = _mlp(H=8)
    theta0 = _mlp_theta(F, seed=123, scale=0.5)
    kw = dict(theta0=theta0, max_outer=1, inner_maxiter=10,
              chunk_target_bytes=2**28)
    with pytest.warns(RuntimeWarning, match="Gauss-Newton diverged"):
        auto = solve_force_od(coll, F, inner="auto", **kw)
    assert auto.info["inner"] == "lbfgs"
    # the runaway was caught at the first GN step: the only (D, Λ) profile
    # is the L-BFGS pass's (w = 0), none on the instrument pass (w = 1)
    assert prepares == [0.0], prepares
    lb = solve_force_od(coll, F, inner="lbfgs", **kw)
    for name in ("theta", "D", "Lambda", "theta_cov", "G_raw", "H", "f"):
        np.testing.assert_array_equal(np.asarray(getattr(auto, name)),
                                      np.asarray(getattr(lb, name)), err_msg=name)
