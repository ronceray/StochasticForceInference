"""Stratonovich force moments on a periodic box: no minimum-image jump bias.

Aligning ABPs (the ``abp_align`` gallery setup: 60 particles, box 30,
exp-poly pair kernels up to length 4).  The kernels do not vanish at
``r = L/2``, so re-picking each pair's minimum image at ``X_plus`` flips
``φ(r) r̂_ij`` whenever a pair crosses the cut during a step; the trapezoid
integrates those jumps into an O(1) bias that does not shrink with Δt
(Strato NMSE 2.0e-2 against 3.6e-4 for Itô before the fix).  With both
trapezoid frames on the image of the preceding frame (frame t enters the
increment, so it cannot fix the image), Strato gives 4.1e-4.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _abp_helpers import make_abp_align_psf  # noqa: E402

from SFI import OverdampedLangevinInference  # noqa: E402
from SFI.bases.pairs import angular_pair_basis, exp_poly_kernels, heading_vector, radial_pair_basis  # noqa: E402
from SFI.langevin import OverdampedProcess  # noqa: E402
from SFI.statefunc import Basis  # noqa: E402


N, L, DT, NSTEPS = 60, 30.0, 0.02, 5000


@pytest.fixture(scope="module")
def abp_run():
    proc = OverdampedProcess(make_abp_align_psf(dim=3), D=jnp.diag(jnp.array([0.05, 0.05, 0.3])),
                             extras_global={"box": jnp.array([L, L])})
    proc.set_params(theta_F=dict(c0=1.0, eps=2.0, R0=1.0, A=0.5, L0=2.0))
    _, kx, kth, ksim = random.split(random.PRNGKey(0), 4)
    x0 = jnp.concatenate([random.uniform(kx, (N, 2)) * L,
                          random.uniform(kth, (N, 1), minval=-jnp.pi, maxval=jnp.pi)], axis=1)
    proc.initialize(x0)
    coll = proc.simulate(dt=DT, Nsteps=NSTEPS, key=ksim)

    # Pairs do cross the minimum-image cut (the scenario under test).
    X = np.asarray(coll.to_array())[..., :2]
    img = np.round((X[:, None] - X[:, :, None]) / L)
    assert (img[1:] != img[:-1]).any()

    kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
    kw = dict(symmetric=True, exclude_self=True, owners="focal", reducer="sum", return_as="basis")
    B = Basis.stack([
        heading_vector(dim=3, angle_index=2),
        radial_pair_basis(kernels, dim=3, box="extras", spatial_dims=slice(0, 2),
                          embed_dim=3, embed_axes=[0, 1]).dispatch_pairs(**kw),
        angular_pair_basis(kernels, jnp.sin, dim=3, angle_index=2, output_index=2,
                           box="extras", spatial_dims=slice(0, 2)).dispatch_pairs(**kw),
    ])
    return proc, coll, B


def _nmse(proc, coll, coll_true, B, diffusion, **fit):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method=diffusion)
    inf.infer_force_linear(B, **fit)
    inf.compute_force_error()
    inf.compare_to_exact(model_exact=proc, data_exact=coll_true, maxpoints=2000)
    return float(inf.NMSE_force)


@pytest.mark.slow
def test_strato_matches_ito_on_periodic_abps(abp_run):
    proc, coll, B = abp_run
    nmse = {m: _nmse(proc, coll, coll, B, "MSD", M_mode=m, G_mode="rectangle") for m in ("Ito", "Strato")}
    # Same level as Itô (the jump bias made it 57x larger).
    assert nmse["Strato"] < 2 * nmse["Ito"], nmse


@pytest.mark.slow
def test_noise_robust_stencils_on_periodic_abps(abp_run):
    """Localisation noise + missing detections (the gallery degradation).

    The image reference must not share noise with the row's increment:
    fixing it at the anchor frame X made the Itô-shift moments pick up the
    jump through the noise of X (Itô-shift/shift NMSE 9e-4 -> 6e-2).  With the
    reference rule all three stencils agree (1.0e-3, 1.6e-3, 1.5e-3); before
    the fix both Stratonovich fits sat at 2e-2.
    """
    proc, coll_true, B = abp_run
    coll = coll_true.degrade(noise=0.05, data_loss_fraction=0.02, seed=1)
    nmse = {
        "Ito-shift/shift": _nmse(proc, coll, coll_true, B, "noisy", M_mode="Ito-shift", G_mode="shift"),
        "Strato/shift": _nmse(proc, coll, coll_true, B, "noisy", M_mode="Strato", G_mode="shift"),
        "Strato/rectangle": _nmse(proc, coll, coll_true, B, "noisy", M_mode="Strato", G_mode="rectangle"),
    }
    assert max(nmse.values()) < 2 * min(nmse.values()), nmse
