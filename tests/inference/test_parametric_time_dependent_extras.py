"""Time-dependent extras on the parametric (exact-core) path.

Regression: the exact runner forwarded ``extras_global`` verbatim, so a
:class:`TimeSeriesExtra` reached the jitted chunk pipeline as a
non-array (``TypeError ... extras['k_drive']``) and the reserved
per-frame ``time`` was dropped.  Per-frame values are now materialised
per dataset, sliced per block, and read at each interval's left frame —
the simulator's convention (``dX[k]`` generated under ``a[k]``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import SFI
from SFI.bases import V, X, extra_scalar, identity_matrix_basis, unit_vector_basis
from SFI.inference.parametric_core.precompute import (
    od_point_tensors,
    ud_point_tensors,
)
from SFI.inference.parametric_core.runner import (
    make_exact_runs_od,
    make_exact_runs_ud,
)
from SFI.langevin import OverdampedProcess
from SFI.trajectory import TrajectoryCollection, time_series_extra



@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Float64 for the exact-residual checks, restored afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)

DT, T = 0.1, 40
A = np.where(np.arange(T) % 10 < 5, 1.0, -2.0)          # switched drive


def _drive_psf(key="a"):
    return (extra_scalar(key, dim=1) * unit_vector_basis(1)).to_psf()


def _block_inputs(runs, coll):
    ds = coll.datasets[0]
    ex = runs._dataset_extras(ds, 0)
    blocks = list(runs._iter_blocks(ds))
    assert len(blocks) == 1
    _, _, lo, hi, X_block, dt_block, _, _ = blocks[0]
    return X_block, dt_block, runs._block_extras(ex, lo, hi)


def test_od_residuals_read_left_frame():
    """Drift F = a(t): exact data ``X[k+1] − X[k] = a[k]·Δt`` gives zero
    residuals at θ = 1 (a right-frame read misses by ``(a[k+1]−a[k])Δt``
    at every switch), for the residual, instrument and convexity paths."""
    Xs = np.concatenate([[0.0], np.cumsum(A[:-1] * DT)])[:, None]
    coll = TrajectoryCollection.from_arrays(
        X=jnp.asarray(Xs), dt=DT, extras_global={"a": time_series_extra(A)})
    F = _drive_psf()
    runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4")
    X_block, dt_block, extras = _block_inputs(runs, coll)
    pt = od_point_tensors(F, jnp.ones(1), X_block, extras, dt_block, 1,
                          with_instrument=True,
                          conv_lambda=1e-4 * jnp.eye(1))
    np.testing.assert_allclose(np.asarray(pt["r"]), 0.0, atol=1e-12)
    # ψ = −∂Φ/∂θ = −a[k]Δt; the instrument (base k−1, residual frame k)
    # reproduces it exactly for a state-independent drift
    psi = np.asarray(pt["psi"])[:, 0, 0, 0]
    np.testing.assert_allclose(psi, -A[:-1] * DT, atol=1e-12)
    inst = np.asarray(pt["psi_inst"])[1:, 0, 0, 0]
    np.testing.assert_allclose(inst, psi[1:], atol=1e-12)


def test_ud_residuals_read_left_frame():
    """Acceleration a(t), piecewise constant per interval: exact data
    (RK4 is exact for constant force) gives zero UD shooting residuals
    when the shooting interval reads frame n and the prediction n+1."""
    x, v, xs = 0.0, 0.3, []
    for k in range(T):
        xs.append(x)
        x, v = x + v * DT + 0.5 * A[k] * DT**2, v + A[k] * DT
    coll = TrajectoryCollection.from_arrays(
        X=jnp.asarray(np.array(xs)[:, None]), dt=DT,
        extras_global={"a": time_series_extra(A)})
    # a zero-coefficient friction term gives the model its v-dependence
    F = ((extra_scalar("a", dim=1) * unit_vector_basis(1)) & V(dim=1)).to_psf()
    runs = make_exact_runs_ud(coll, F, dt=None, n_substeps=1, integrator="rk4")
    X_block, dt_block, extras = _block_inputs(runs, coll)
    pt = ud_point_tensors(F, jnp.array([1.0, 0.0]), X_block, extras,
                          dt_block, 1, with_instrument=True)
    np.testing.assert_allclose(np.asarray(pt["r"]), 0.0, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(pt["psi_inst"])))


def test_reserved_time_reaches_parametric_core():
    """The reserved per-frame ``time`` is no longer dropped: drift
    F = θ·t on data ``X[k+1] − X[k] = t_k·Δt``."""
    t = np.arange(T) * DT
    Xs = np.concatenate([[0.0], np.cumsum(t[:-1] * DT)])[:, None]
    coll = TrajectoryCollection.from_arrays(X=jnp.asarray(Xs), dt=DT)
    F = _drive_psf("time")
    runs = make_exact_runs_od(coll, F, dt=None, n_substeps=1, integrator="rk4")
    X_block, dt_block, extras = _block_inputs(runs, coll)
    pt = od_point_tensors(F, jnp.ones(1), X_block, extras, dt_block, 1)
    np.testing.assert_allclose(np.asarray(pt["r"]), 0.0, atol=1e-12)


def test_constant_series_matches_static_extra():
    """A constant TimeSeriesExtra and the same value as a static extra
    give the same parametric fit (the per-frame path changes nothing but
    where the value is read)."""
    F_true = X(dim=1) & (extra_scalar("k", dim=1) * X(dim=1))
    proc = OverdampedProcess(F=F_true, D=0.25, theta_F=jnp.array([-1.0, -0.5]))
    proc.set_extras(extras_global={"k": 1.0})
    proc.initialize(jnp.zeros(1))
    Xs = proc.simulate(dt=0.01, Nsteps=3000, key=random.PRNGKey(3)).to_arrays()[1]
    B = X(dim=1) * extra_scalar("k", dim=1)
    fits = []
    for k in (jnp.asarray(1.0), time_series_extra(np.ones(Xs.shape[0]))):
        coll = TrajectoryCollection.from_arrays(
            X=jnp.asarray(Xs), dt=0.01, extras_global={"k": k})
        inf = SFI.OverdampedLangevinInference(coll)
        inf.infer_force(B)
        fits.append(np.asarray(inf.force_coefficients))
    np.testing.assert_allclose(fits[1], fits[0], rtol=1e-6)


def test_switched_trap_with_localisation_noise():
    """The gallery's switched-stiffness trap with localisation noise
    σ ≈ the diffusive step √(2DΔt): parametric ``infer_force`` (crashed
    pre-fix) recovers both coefficients within 2σ of its own stderr."""
    dt, Nsteps, period, D = 0.01, 20_000, 1_000, 0.25
    k_protocol = (np.arange(Nsteps) // period % 2).astype(float)
    B = X(dim=1) & (extra_scalar("k_drive", dim=1) * X(dim=1))
    proc = OverdampedProcess(F=B, D=D, theta_F=jnp.array([-1.0, -1.0]))
    proc.set_extras(extras_global={"k_drive": time_series_extra(k_protocol)})
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(42),
                         oversampling=4)
    sigma = np.sqrt(2 * D * dt)
    noisy = coll.degrade(noise=sigma, seed=1)

    inf = SFI.OverdampedLangevinInference(noisy)
    inf.infer_force(B)
    inf.compute_force_error()
    theta = np.asarray(inf.force_coefficients)
    se = np.asarray(inf.force_coefficients_stderr)
    assert np.all(np.abs(theta + 1.0) < 2.0 * se), (theta, se)
    # the parametric fit records its basis labels, so coeff_block locates
    # a Basis block (the gallery's plateau plot)
    c, cov = inf.coeff_block(B, field="force")
    np.testing.assert_allclose(c, theta)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), se, rtol=1e-6)
    lam = float(np.ravel(inf.metadata["force_parametric_info"]["Lambda"])[0])
    assert np.sqrt(lam) == pytest.approx(sigma, rel=0.1)


def test_diffusion_reads_protocol_the_force_does_not():
    """Switched diffusion D(t) = d0 + d1·k(t) under a static force: the
    force fit drops the unread series, and the parametric diffusion fit
    still receives it per frame (the runner materialises the extras of
    the diffusion model as well)."""
    Nsteps = 10_000
    k_protocol = (np.arange(Nsteps) // 1_000 % 2).astype(float)
    DB = identity_matrix_basis(1) & (extra_scalar("k", dim=1)
                                     * identity_matrix_basis(1))
    proc = OverdampedProcess(F=X(dim=1), D=DB, theta_F=jnp.array([-1.0]),
                             theta_D=jnp.array([0.1, 0.3]))
    proc.set_extras(extras_global={"k": time_series_extra(k_protocol)})
    proc.initialize(jnp.zeros(1))
    coll = proc.simulate(dt=0.01, Nsteps=Nsteps, key=random.PRNGKey(0),
                         oversampling=4)
    inf = SFI.OverdampedLangevinInference(coll)
    inf.infer_force(X(dim=1))
    inf.infer_diffusion(DB)
    # relative standard error of a variance from n increments: √(2/n)
    d = np.asarray(inf.diffusion_coefficients)
    d_off, d_on = d[0], d[0] + d[1]
    n_half = Nsteps / 2
    assert abs(d_off / 0.1 - 1) < 3 * np.sqrt(2 / n_half), d
    assert abs(d_on / 0.4 - 1) < 3 * np.sqrt(2 / n_half), d


def test_per_frame_neighbour_lists_reach_the_parametric_core():
    """Per-frame CSR neighbour tables are dispatcher-owned (structural) extras,
    absent from ``required_extras``: the runner dropped them (``KeyError:
    'nb/indptr'``).  Sliced per frame, lists with a skin give the dense-pair fit."""
    from SFI.bases.pairs import compact_kernels, radial_pair_basis
    from SFI.inference.parametric_core.frame_extras import time_varying_keys
    from SFI.trajectory import TimeSeriesExtra
    from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr

    L, RC, SKIN = 10.0, 3.0, 1.5
    rng = np.random.default_rng(0)
    Xp = np.cumsum(0.2 * rng.standard_normal((30, 8, 2)), axis=0) + rng.uniform(0, L, (8, 2))
    ips, ids = [], []
    for t in range(len(Xp)):
        ip, ix = build_neighbor_csr(Xp[t] % L, RC + SKIN, np.array([L, L]))
        ips.append(ip)
        ids.append(ix)
    nnz = max(len(i) for i in ids)
    ids = [pad_neighbor_csr(ip, ix, nnz)[1] for ip, ix in zip(ips, ids)]
    box = {"box": jnp.array([L, L])}
    coll_csr = TrajectoryCollection.from_arrays(X=Xp, dt=0.1, extras_global={
        **box, "nb/indptr": TimeSeriesExtra(jnp.asarray(np.stack(ips), jnp.int32)),
        "nb/indices": TimeSeriesExtra(jnp.asarray(np.stack(ids), jnp.int32))})
    coll_dense = TrajectoryCollection.from_arrays(X=Xp, dt=0.1, extras_global=box)
    # Compact kernels vanish beyond RC: within the skin the lists hold every interacting pair.
    inter = radial_pair_basis(compact_kernels([0, 1], RC), dim=2, box="extras")
    kw = dict(exclude_self=True, owners="focal", reducer="sum")
    B_csr = inter.dispatch_pairs_from_extras(indptr_key="nb/indptr", indices_key="nb/indices", **kw)
    B_dense = inter.dispatch_pairs(return_as="basis", **kw)
    assert set(time_varying_keys(coll_csr.datasets[0], set(B_csr.structural_extras))) == {"nb/indptr", "nb/indices"}

    theta = []
    for B, coll in ((B_csr, coll_csr), (B_dense, coll_dense)):
        inf = SFI.OverdampedLangevinInference(coll)
        inf.infer_force(B, D=0.02 * jnp.eye(2), Lambda=jnp.zeros((2, 2)), eiv=False)
        theta.append(np.asarray(inf.force_coefficients))
    np.testing.assert_allclose(theta[0], theta[1], rtol=1e-6, atol=1e-9)
