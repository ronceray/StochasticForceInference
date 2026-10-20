"""Dead pixels on lattice fields with the noise-robust (all-pairs) force estimator.

``degrade_spatial_data(..., stencil_radius=r)`` removes from the fit every cell
whose stencil touches a missing pixel; such a cell stays *observed* (its value
and increments are known).  Two bugs biased the robust preset on these data:

* the all-pairs Stratonovich correction zeroed the increments of every
  neighbour excluded from the *fit* (``mask_out``) instead of only the
  neighbours whose position is unknown (``mask_state``), dropping the
  cross-site terms ``D_ij d_j b_i`` next to every gap;
* the erosion covered frames ``t-1 .. t+2`` while the default ``noisy_lag2``
  correction also reads ``t+3``: a neighbour missing there is hidden from the
  whole row's stencils.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI import OverdampedLangevinInference, TrajectoryCollection
from SFI.bases.spde import conserved_noise_pbc, square_grid_extras
from SFI.inference.graph_diffusion import stencil_directions
from SFI.langevin import OverdampedProcess
from SFI.statefunc.layout import GridLayout, ScalarSector
from SFI.trajectory.degrade import degrade_spatial_data

L, DT = 16, 0.02
TRUTH = np.array([-0.2, 0.5])  # phi, lap phi


@pytest.mark.parametrize("method", ["noisy", "noisy_lag2"])
def test_probe_increments_of_fit_excluded_neighbours_are_kept(method):
    rng = np.random.default_rng(0)
    N, d = 6, 1
    s = {k: jnp.asarray(rng.standard_normal((N, d)), jnp.float32)
         for k in ("dX", "dX_plus", "dX_minus", "X_plusplus", "X_p3")}
    s.update(dt=jnp.asarray(0.1), dt_plus=jnp.asarray(0.1))
    observed = jnp.array([True, True, True, True, True, False])
    excluded = jnp.array([True, False, True, False, True, False])  # fit mask ⊆ observed
    for L_op, y_op in stencil_directions(method, "od"):
        y_fit = np.asarray(y_op(**s, mask_out=excluded, mask_state=observed))
        y_obs = np.asarray(y_op(**s, mask_out=observed))
        np.testing.assert_array_equal(y_fit, y_obs)        # only unobserved sites are zeroed
        assert np.all(y_fit[5] == 0) and np.any(y_fit[1] != 0)


@pytest.fixture(scope="module")
def heat():
    layout = GridLayout(U=ScalarSector([0]), dim=1, ndim=2, bc="pbc")
    B = layout.embed(rank=1, U=layout.U & layout.lap(layout.U))
    ex = square_grid_extras(grid_shape=(L, L), dx=1.0)
    proc = OverdampedProcess(B, D=conserved_noise_pbc(sigma=0.3, grid_shape=(L, L), dx=1.0),
                             theta_F=jnp.asarray(TRUTH, jnp.float32))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.zeros((L * L, 1), jnp.float32))
    coll = proc.simulate(dt=DT, Nsteps=4000, key=jax.random.PRNGKey(0), oversampling=4, prerun=100,
                         compute_observables=False)
    return coll, B


def _robust(coll, B):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="noisy")
    inf.infer_force_linear(B, preset="robust")
    inf.compute_force_error()
    return np.asarray(inf.force_coefficients), np.asarray(inf.force_coefficients_stderr)


def test_robust_fit_unbiased_by_dead_pixels(heat):
    coll, B = heat
    c_full, _ = _robust(coll, B)
    lossy = degrade_spatial_data(coll, downscale=1, data_loss_fraction=0.02, stencil_radius=1,
                                 bc="pbc", seed=1)
    c_loss, e_loss = _robust(lossy, B)
    # Same movie minus the cells next to gaps: the fits differ by the dropped
    # rows' sampling noise only (pre-fix: the Laplacian coefficient was off by
    # many standard errors).
    assert np.all(np.abs(c_loss - c_full) < 3.0 * e_loss), (c_full, c_loss, e_loss)
