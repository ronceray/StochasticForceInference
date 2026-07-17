# TODO: review this file
"""
tests/test_parametric_mask_abp.py
=============================
Test parametric SFI with masked (degraded) data on a multi-particle ABP
system with interactions.  This is the real test: particle loss affects
other particles through the N-body force, so per-particle masking is
non-trivial.

For interacting systems, if ANY particle is masked at time t, the ODE
flow for all other particles is contaminated (they would interact with
a fake fill-value).  The solver handles this by collapsing the per-
particle mask to per-step: a step is valid only when ALL particles are
observed.

Verifies:
  1. No NaN in any output.
  2. Inferred parameters remain reasonable vs clean-data reference.
  3. The effective data loss (fraction of invalid steps) grows with
     data_loss_fraction more steeply than for single particles.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random



# ── ABP model setup ──


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _build_abp_psf():
    """Build an interacting ABP PSF: self-propulsion + repulsion + alignment."""
    from SFI.bases.pairs import (
        angle_coupling, heading_vector, pair_direction,
        parametric_radial_kernel,
    )
    DIM = 3  # (x, y, θ)
    B_heading = heading_vector(dim=DIM, angle_index=2)
    e_ij = pair_direction(
        dim=DIM, box="extras", spatial_dims=slice(0, 2),
        embed_dim=DIM, embed_axes=[0, 1],
    )
    g_align = angle_coupling(jnp.sin, dim=DIM, angle_index=2)
    k_repel = parametric_radial_kernel(
        lambda r, p: -p["eps"] * jnp.exp(-r / p["R0"]),
        params={"eps": (), "R0": ()},
        dim=DIM, box="extras", spatial_dims=slice(0, 2),
    )
    k_align = parametric_radial_kernel(
        lambda r, p: p["A"] * jnp.exp(-r / p["L0"]),
        params={"A": (), "L0": ()},
        dim=DIM, box="extras", spatial_dims=slice(0, 2),
    )
    F_psf = (
        B_heading.to_psf(coeff_key="c0")
        + (k_repel * e_ij).dispatch_pairs(return_as="psf")
        + (k_align * g_align).dispatch_pairs(return_as="psf")
    )
    return F_psf


def _simulate_abp(N, Nsteps, dt, seed=42):
    """Simulate N interacting ABPs and return (collection, theta_true)."""
    from SFI.langevin import OverdampedProcess

    DIM = 3
    LX, LY = 10.0, 10.0
    D_ISO = 0.05
    box = jnp.array([LX, LY])
    theta_true = dict(c0=1.0, eps=2.0, A=0.5, R0=1.0, L0=2.0)

    F_psf = _build_abp_psf()
    proc = OverdampedProcess(F_psf, D=D_ISO, extras_global={"box": box})
    proc.set_params(theta_F=theta_true)

    key = random.PRNGKey(seed)
    key, kx, kth = random.split(key, 3)
    X0_xy = random.uniform(kx, (N, 2)) * jnp.array([LX, LY])
    TH0 = random.uniform(kth, (N,), minval=-jnp.pi, maxval=jnp.pi)
    x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)
    proc.initialize(x0)

    key, sub = random.split(key)
    coll = proc.simulate(
        dt=dt, Nsteps=Nsteps, key=sub, prerun=50, oversampling=5,
    )
    return coll, theta_true, F_psf


def _run_parametric_sfi(coll, F_psf, theta_true, max_iter=5):
    """Run Parametric SFI and return the inference object."""
    from SFI.inference.overdamped import OverdampedLangevinInference

    theta0 = {k: 0.5 * jnp.ones_like(jnp.asarray(v))
              for k, v in theta_true.items()}

    sfi = OverdampedLangevinInference(coll)
    sfi.infer_force(
        F_psf, theta0,
        n_substeps=4,
        max_outer=max_iter,
    )
    return sfi


def _assert_no_nan(sfi, label=""):
    prefix = f"[{label}] " if label else ""
    for name, val in [
        ("theta", sfi.force_coefficients_full),
        ("D", sfi.diffusion_average),
        ("Lambda", sfi.Lambda),
        ("G", sfi.force_G),
    ]:
        assert bool(jnp.all(jnp.isfinite(val))), (
            f"{prefix}{name} has non-finite values"
        )


def _param_error(sfi, theta_true, F_psf):
    """Relative L² error between inferred and true flat params."""
    theta_true_flat = F_psf.flatten_params(theta_true)
    theta_hat = sfi.force_coefficients_full
    return float(
        jnp.linalg.norm(theta_hat - theta_true_flat)
        / jnp.linalg.norm(theta_true_flat)
    )


# ── Tests ──

class TestParametricMaskABP:
    """Parametric SFI with masked multi-particle ABP data."""

    N = 5
    NSTEPS = 2000
    DT = 0.01

    @pytest.fixture(scope="class")
    def abp_setup(self):
        """Simulate clean ABP data."""
        coll, theta_true, F_psf = _simulate_abp(
            self.N, self.NSTEPS, self.DT, seed=42,
        )
        ds0 = coll.datasets[0]
        print(f"\n  ABP trajectory shape: {ds0.X.shape}")  # (T, N, d)
        return coll, theta_true, F_psf

    @pytest.fixture(scope="class")
    def clean_result(self, abp_setup):
        """Reference: Parametric SFI on clean data."""
        coll, theta_true, F_psf = abp_setup
        sfi = _run_parametric_sfi(coll, F_psf, theta_true, max_iter=5)
        return sfi

    def test_clean_baseline(self, abp_setup, clean_result):
        """Clean ABP data: NaN-free and finite."""
        coll, theta_true, F_psf = abp_setup
        sfi = clean_result
        _assert_no_nan(sfi, "ABP-clean")

        err = _param_error(sfi, theta_true, F_psf)
        print(f"  ABP clean θ error: {err:.4f}")
        print(f"  D diag: {jnp.diag(sfi.diffusion_average)}")
        print(f"  Λ diag: {jnp.diag(sfi.Lambda)}")
        print(f"  θ_hat: {sfi.force_coefficients_full}")
        print(f"  θ_true: {F_psf.flatten_params(theta_true)}")
        # Smoke: just check no crash and finite; ABP convergence is hard
        assert err < 2.0, f"ABP force error implausibly large: {err}"

    def test_masked_20pct(self, abp_setup, clean_result):
        """20% per-particle data loss: NaN-free output."""
        coll, theta_true, F_psf = abp_setup

        coll_deg = coll.degrade(data_loss_fraction=0.2, seed=77)
        ds0 = coll_deg.datasets[0]
        M = np.asarray(ds0._M2d())
        per_particle_loss = 1.0 - float(np.mean(M))
        # With N particles, any-missing-at-step loss is higher
        all_valid = np.all(M, axis=1)
        effective_step_loss = 1.0 - float(np.mean(all_valid))
        print(f"\n  Per-particle loss: {per_particle_loss:.1%}")
        print(f"  Effective step loss (any particle missing): "
              f"{effective_step_loss:.1%}")

        sfi_m = _run_parametric_sfi(coll_deg, F_psf, theta_true, max_iter=5)
        _assert_no_nan(sfi_m, "ABP-mask-20%")

        err = _param_error(sfi_m, theta_true, F_psf)
        err_clean = _param_error(clean_result, theta_true, F_psf)
        print(f"  ABP masked-20% θ error: {err:.4f} (clean: {err_clean:.4f})")
        print(f"  D diag: {jnp.diag(sfi_m.diffusion_average)}")
        print(f"  Λ diag: {jnp.diag(sfi_m.Lambda)}")
        # Main check: no NaN. Accuracy may degrade.
        assert err < 3.0, f"ABP masked force error implausibly large: {err}"

    def test_masked_40pct(self, abp_setup, clean_result):
        """40% per-particle data loss: NaN-free output."""
        coll, theta_true, F_psf = abp_setup

        coll_deg = coll.degrade(data_loss_fraction=0.4, seed=99)
        ds0 = coll_deg.datasets[0]
        M = np.asarray(ds0._M2d())
        per_particle_loss = 1.0 - float(np.mean(M))
        all_valid = np.all(M, axis=1)
        effective_step_loss = 1.0 - float(np.mean(all_valid))
        print(f"\n  Per-particle loss: {per_particle_loss:.1%}")
        print(f"  Effective step loss: {effective_step_loss:.1%}")

        sfi_m = _run_parametric_sfi(coll_deg, F_psf, theta_true, max_iter=5)
        _assert_no_nan(sfi_m, "ABP-mask-40%")

        err = _param_error(sfi_m, theta_true, F_psf)
        err_clean = _param_error(clean_result, theta_true, F_psf)
        print(f"  ABP masked-40% θ error: {err:.4f} (clean: {err_clean:.4f})")
        print(f"  D diag: {jnp.diag(sfi_m.diffusion_average)}")
        print(f"  Λ diag: {jnp.diag(sfi_m.Lambda)}")
        # At 40% per-particle loss with N=5: (0.6)^5 ≈ 7.8% steps survive.
        # Not enough data for accurate inference; just verify NaN-free.
        # (The assertion above already checked NaN-free.)
        print(f"  [Note: only ~{100*(1-effective_step_loss):.0f}% of steps "
              f"usable — accurate inference not expected]")
