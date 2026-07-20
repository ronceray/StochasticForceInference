"""Tests for SFI.inference.parametric_core.flow_multi — multiparticle residuals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _per_particle_linear_psf(d):
    """particles_input=False linear drift F(x)=A x (non-interacting)."""
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x
    return make_psf(f, dim=d, rank=1, n_features=1,
                    params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)])


def test_non_interacting_multi_matches_per_particle_single():
    """For a per-particle force, multi residuals == single-particle kernel per particle."""
    from SFI.inference.parametric_core.flow_multi import multi_step_residuals
    from SFI.inference.parametric_core.flow import od_window_residuals

    d, W, N = 2, 6, 3
    F = _per_particle_linear_psf(d)
    A = jnp.array([[-1.0, 0.2], [0.1, -1.5]])
    theta = jnp.asarray(F.flatten_params({"A": A}), dtype=jnp.float64)
    struct = F.unflatten_params(theta)
    rng = np.random.default_rng(0)
    X_w = jnp.array(rng.standard_normal((W, N, d)))

    r, J = multi_step_residuals(F, struct, X_w, None, dt=0.05, n_substeps=4, integrator="rk4")
    assert r.shape == (W - 1, N, d) and J.shape == (W - 1, N, d, d)

    def drift(x):
        return F(x[None], params=struct)[0]
    for p in range(N):
        rp, Jp = od_window_residuals(drift, X_w[:, p, :], 0.05, 4, "rk4")
        np.testing.assert_allclose(np.asarray(r[:, p, :]), np.asarray(rp), atol=1e-10)
        np.testing.assert_allclose(np.asarray(J[:, p, :, :]), np.asarray(Jp), atol=1e-10)


def test_interacting_residuals_vanish_on_deterministic_trajectory():
    """particles_input=True force: residuals ~0 on a noiseless interacting trajectory."""
    from SFI.inference.parametric_core.flow_multi import multi_step_residuals
    from SFI.statefunc import make_psf
    from SFI.bases.pairs import parametric_radial_kernel, pair_direction

    dim, N = 2, 3
    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x, params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r, params={"k_spring": ()}, dim=dim)
    F_psf = F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")
    struct = {"k_trap": 1.0, "k_spring": 0.5}

    from SFI.integrate.rk4 import ode_flow

    dt, n_sub = 0.02, 4
    def full_drift(Xflat):
        X = Xflat.reshape(N, dim)
        return F_psf(X, params=struct).reshape(-1)
    rng = np.random.default_rng(1)
    X = jnp.array(rng.standard_normal((N, dim)) * 0.4)
    Xs = [X]
    for _ in range(7):
        Xn = ode_flow(full_drift, X.reshape(-1), dt, n_sub).reshape(N, dim)
        Xs.append(Xn); X = Xn
    X_w = jnp.stack(Xs, axis=0)  # (W, N, dim)

    r, J = multi_step_residuals(F_psf, struct, X_w, None, dt=dt, n_substeps=n_sub, integrator="rk4")
    assert r.shape == (7, N, dim)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-7)


# ── η-clean instrument dispatchers (multi_od_instrument / multi_ud_instrument) ──


def _trap_spring_psf(dim=2):
    from SFI.bases.pairs import parametric_radial_kernel, pair_direction
    from SFI.statefunc import make_psf

    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x,
                      params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r,
                                        params={"k_spring": ()}, dim=dim)
    return F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")


def test_multi_od_instrument_non_interacting_matches_vmapped_single():
    """particles_input=False: dispatcher == per-particle od_instrument vmap."""
    from SFI.inference.parametric_core.flow import od_instrument
    from SFI.inference.parametric_core.flow_multi import multi_od_instrument

    d, N = 2, 4
    F = _per_particle_linear_psf(d)
    theta = jnp.asarray(F.flatten_params({"A": jnp.array([[-1.0, 0.2], [0.1, -1.5]])}))
    rng = np.random.default_rng(2)
    X_base = jnp.array(rng.standard_normal((N, d)))

    got = multi_od_instrument(F, theta, X_base, None, 0.05, 2, "rk4")

    def drift_of_theta(th):
        struct = F.unflatten_params(th)
        return lambda y: F(y[None], params=struct)[0]

    ref = jax.vmap(lambda xb: od_instrument(drift_of_theta, theta, xb, 0.05, 2, "rk4"))(X_base)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_multi_od_instrument_interacting_pair_column_tracks_regressor():
    """particles_input=True: the frame-level instrument keeps every parameter
    column at the regressor's scale — in particular the pair-interaction column,
    which the old isolated-frame (N=1) construction zeroed out (or crashed on),
    making the IV Gram structurally singular on interaction parameters."""
    from SFI.inference.parametric_core.flow_multi import (
        multi_od_instrument, multi_step_residuals)

    dim, N, dt = 2, 5, 0.02
    F = _trap_spring_psf(dim)
    theta = jnp.asarray(F.flatten_params({"k_trap": 1.0, "k_spring": 0.5}))
    rng = np.random.default_rng(3)
    X_base = jnp.array(rng.standard_normal((N, dim)) * 0.5)

    psi_inst = multi_od_instrument(F, theta, X_base, None, dt, 1, "rk4")
    assert psi_inst.shape == (N, dim, 2)

    X_w = jnp.stack([X_base, X_base], axis=0)

    def resid(th):
        r, _ = multi_step_residuals(F, F.unflatten_params(th), X_w, None, dt, 1, "rk4")
        return r

    psi_right = jax.jacfwd(resid)(theta)[0]            # (N, d, 2)
    names = list(F.unflatten_params(theta).keys())
    for a, name in enumerate(names):
        n_inst = float(jnp.linalg.norm(psi_inst[..., a]))
        n_right = float(jnp.linalg.norm(psi_right[..., a]))
        assert n_inst > 0.3 * n_right, (
            f"instrument column '{name}' lost the regressor scale: "
            f"{n_inst:.3e} vs {n_right:.3e}")


def _flock_pair_psf(dim=2):
    from SFI.statefunc import make_interactor
    from SFI.statefunc.params import ParamSpec

    def _pair(x, *, v, params):
        dr = x[1] - x[0]
        dv = v[1] - v[0]
        return (params["k_coh"] * dr + params["k_alg"] * dv)[..., None]

    return make_interactor(
        _pair, dim=dim, rank=1, K=2, n_features=1, needs_v=True,
        params=[ParamSpec("k_coh", shape=(), default=0.5),
                ParamSpec("k_alg", shape=(), default=0.8)],
    ).dispatch_pairs(drop_features=True)


def test_multi_ud_instrument_non_interacting_matches_vmapped_single():
    """particles_input=False: dispatcher == per-particle ud_instrument vmap."""
    from SFI.inference.parametric_core.flow_ud import ud_instrument
    from SFI.inference.parametric_core.flow_multi import multi_ud_instrument
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    F = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                 params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                         ParamSpec("gamma", shape=(), dtype=jnp.float64)])
    theta = jnp.asarray(F.flatten_params({"k": 1.0, "gamma": 0.5}))
    rng = np.random.default_rng(4)
    Y_a = jnp.array(rng.standard_normal((3, 1)) * 0.3)
    Y_b = Y_a + 0.01 * jnp.array(rng.standard_normal((3, 1)))

    got = multi_ud_instrument(F, theta, Y_a, Y_b, None, 0.01, 2, "rk4")

    def force_of_theta(th):
        struct = F.unflatten_params(th)
        return lambda x, v: F(x[None], v=v[None], params=struct)[0]

    ref = jax.vmap(
        lambda a, b: ud_instrument(force_of_theta, theta, a, b, 0.01, 2, "rk4")
    )(Y_a, Y_b)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_multi_ud_instrument_interacting_columns_nonzero():
    """particles_input=True (pure pair force): every instrument column nonzero —
    the old isolated-frame construction returned all-zero columns here."""
    from SFI.inference.parametric_core.flow_multi import multi_ud_instrument

    dim, N, dt = 2, 4, 0.01
    F = _flock_pair_psf(dim)
    theta = jnp.asarray(F.flatten_params({"k_coh": 0.5, "k_alg": 0.8}))
    rng = np.random.default_rng(5)
    Y_a = jnp.array(rng.standard_normal((N, dim)) * 0.5)
    Y_b = Y_a + 0.02 * jnp.array(rng.standard_normal((N, dim)))

    psi_inst = multi_ud_instrument(F, theta, Y_a, Y_b, None, dt, 1, "rk4")
    assert psi_inst.shape == (N, dim, 2)
    for a, name in enumerate(F.unflatten_params(theta).keys()):
        n_col = float(jnp.linalg.norm(psi_inst[..., a]))
        assert n_col > 1e-8, f"instrument column '{name}' is zero"


# ── frozen-background ψ_right (same-particle θ-recursion) ───────────────────


def test_od_frozen_background_psi_matches_exact_jacfwd():
    """multi_step_residuals_with_psi: r and J bit-identical to the plain
    interacting path; ψ_right matches the exact full-flow jacfwd to the
    O(h²) cross-particle substep-feedback terms (the same ones the
    frozen-background J drops)."""
    from SFI.inference.parametric_core.flow_multi import (
        multi_step_residuals, multi_step_residuals_with_psi)

    dim, N, W, dt = 2, 5, 4, 0.02
    F = _trap_spring_psf(dim)
    theta = jnp.asarray(F.flatten_params({"k_trap": 1.0, "k_spring": 0.5}))
    rng = np.random.default_rng(6)
    X_w = jnp.array(rng.standard_normal((W, N, dim)) * 0.6)

    r1, J1, psi_fb = multi_step_residuals_with_psi(F, theta, X_w, None, dt, 2, "rk4")
    r0, J0 = multi_step_residuals(F, F.unflatten_params(theta), X_w, None, dt, 2, "rk4")
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r0))
    np.testing.assert_array_equal(np.asarray(J1), np.asarray(J0))

    def resid(th):
        r, _ = multi_step_residuals(F, F.unflatten_params(th), X_w, None, dt, 2, "rk4")
        return r

    psi_exact = jax.jacfwd(resid)(theta)
    rel = float(jnp.linalg.norm(psi_fb - psi_exact) / jnp.linalg.norm(psi_exact))
    assert rel < 0.02, f"frozen-background psi too far from exact: rel={rel:.3e}"


def test_ud_frozen_background_psi_matches_exact_jacfwd():
    """ud_multi_step_residuals_with_psi: r and α-propagators bit-identical;
    ψ_right matches the exact jacfwd to O(h²)."""
    from SFI.inference.parametric_core.flow_multi import (
        ud_multi_step_residuals, ud_multi_step_residuals_with_psi)

    dim, N, W, dt = 2, 4, 6, 0.01
    F = _flock_pair_psf(dim)
    theta = jnp.asarray(F.flatten_params({"k_coh": 0.5, "k_alg": 0.8}))
    rng = np.random.default_rng(7)
    Y = [jnp.array(rng.standard_normal((N, dim)) * 0.5)]
    V = jnp.array(rng.standard_normal((N, dim)) * 0.3)
    for _ in range(W - 1):
        Y.append(Y[-1] + V * dt)
    Y_w = jnp.stack(Y)

    r1, ap1, a01, am1, vh1, psi_fb = ud_multi_step_residuals_with_psi(
        F, theta, Y_w, None, dt, 2, "rk4")
    r0, ap0, a00, am0, vh0 = ud_multi_step_residuals(
        F, F.unflatten_params(theta), Y_w, None, dt, 2, "rk4")
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r0))
    np.testing.assert_array_equal(np.asarray(a01), np.asarray(a00))
    np.testing.assert_array_equal(np.asarray(vh1), np.asarray(vh0))

    def resid(th):
        r, *_ = ud_multi_step_residuals(F, F.unflatten_params(th), Y_w, None, dt, 2, "rk4")
        return r

    psi_exact = jax.jacfwd(resid)(theta)
    rel = float(jnp.linalg.norm(psi_fb - psi_exact) / jnp.linalg.norm(psi_exact))
    assert rel < 0.02, f"frozen-background psi too far from exact: rel={rel:.3e}"
