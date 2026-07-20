"""Smoke tests for the new/extended SFI.utils.plotting helpers.

These assert the helpers run on a tiny simulated collection and return an
artist/axes without error (visual correctness is checked in the gallery
build).
"""

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from SFI.bases import X
from SFI.langevin import OverdampedProcess
from SFI.utils import plotting as P


@pytest.fixture
def coll2d():
    proc = OverdampedProcess(F=-1.0 * X(dim=2), D=0.3)
    proc.initialize(jnp.array([0.4, -0.4]))
    return proc.simulate(dt=0.02, Nsteps=600, key=jax.random.PRNGKey(7))


@pytest.fixture
def coll3d():
    proc = OverdampedProcess(F=-1.0 * X(dim=3), D=0.3)
    proc.initialize(jnp.array([0.4, -0.4, 0.2]))
    return proc.simulate(dt=0.02, Nsteps=400, key=jax.random.PRNGKey(8))


def _fresh():
    plt.close("all")
    return plt.figure()


# ---- A2: phase-space ----

def test_phase2d_extensions(coll2d):
    _fresh()
    P.phase2d(coll2d, dims=(0, 1), color="#40B0A6")  # solid color branch
    _fresh()
    P.phase2d(coll2d, dims=(0, 1), transform=np.tanh)  # transform branch
    _fresh()
    P.phase2d(coll2d, dims=(0, 1), box=(2.0, 2.0))  # periodic wrap branch


def test_phase2d_color_colorbar_conflict(coll2d):
    _fresh()
    with pytest.raises(ValueError):
        P.phase2d(coll2d, color="r", plot_colorbar=True)


def test_phase2d_scalar(coll2d):
    _fresh()
    lc = P.phase2d_scalar(coll2d, color_fn=lambda x: np.linalg.norm(x, axis=-1))
    assert lc is not None


def test_timeseries_colored(coll2d):
    _fresh()
    ax = P.timeseries_colored(coll2d, color_fn=lambda x: x[:, 0] ** 2, dims=[0])
    assert ax is not None


def test_phase3d(coll3d):
    _fresh()
    ax = P.phase3d(coll3d)
    assert ax is not None


def test_trajectory_scatter(coll2d):
    _fresh()
    ax = P.trajectory_scatter(coll2d)
    assert ax is not None
    ax2 = P.trajectory_scatter(coll2d, cmap="viridis")
    assert ax2 is not None


# ---- A3: field plots ----

@pytest.fixture
def fit2d(coll2d):
    from SFI import OverdampedLangevinInference
    from SFI.bases import monomials_up_to

    proc = OverdampedProcess(F=-1.0 * X(dim=2), D=0.3)
    proc.initialize(jnp.array([0.4, -0.4]))
    inf = OverdampedLangevinInference(coll2d)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(monomials_up_to(order=1, dim=2, rank="vector"))
    inf.infer_diffusion_linear()
    return inf, proc, coll2d


def test_plot_field_extensions(fit2d):
    inf, proc, coll = fit2d
    _fresh()
    P.plot_field(coll, inf.force_inferred, N=8, mask_unvisited=True, clip_magnitude=5.0)


def test_plot_tensor_field_ellipse(fit2d):
    inf, proc, coll = fit2d
    _fresh()
    P.plot_tensor_field(coll, inf.diffusion_inferred, N=6, mode="ellipse", scale=0.2)


def test_plot_profile_1d(fit2d):
    inf, proc, coll = fit2d
    _fresh()
    ax = P.plot_profile_1d(coll, inf.force_inferred, exact_field=proc.force_sf, dim=0, N=40, samples=True)
    assert ax is not None


def test_plot_field_error(fit2d):
    inf, proc, coll = fit2d
    _fresh()
    ax = P.plot_field_error(coll, inf.force_inferred, proc.force_sf, N=12)
    assert ax is not None


def test_stream_field(fit2d):
    inf, proc, coll = fit2d
    _fresh()
    ax = P.stream_field(coll, inf.force_inferred, N=12)
    assert ax is not None


# ---- A4: particles / active matter ----

@pytest.fixture
def active_coll():
    from SFI.trajectory import TrajectoryCollection

    rng = np.random.RandomState(0)
    T, N = 20, 12
    xy = np.cumsum(0.05 * rng.randn(T, N, 2), axis=0) % 5.0
    th = (rng.rand(T, N) * 2 * np.pi)[..., None]
    Xarr = np.concatenate([xy, th], axis=-1).astype(np.float32)  # (T, N, 3)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(Xarr), dt=0.1)


def test_plot_particles_heading(active_coll):
    _fresh()
    ax = P.plot_particles(active_coll, color_dim=2, quiver=True, heading_dim=2, box=(5.0, 5.0), cmap="hsv")
    assert ax is not None


def test_plot_particles_legacy_default(active_coll):
    _fresh()
    ax = P.plot_particles(active_coll)  # back-compat path
    assert ax is not None


def test_plot_rods(active_coll):
    _fresh()
    ax = plt.gca()
    Xframe = np.asarray(active_coll.to_arrays()[1][-1])  # (N, 3)
    lc = P.plot_rods(ax, Xframe, angle_index=2, length=0.4)
    assert lc is not None


def test_plot_nematic_director():
    _fresh()
    ax = plt.gca()
    g = np.linspace(0, 1, 16)
    GX, GY = np.meshgrid(g, g)
    Qxx = np.cos(2 * np.pi * GX)
    Qxy = np.sin(2 * np.pi * GY)
    rho = np.ones_like(GX)
    ax.imshow(rho, origin="lower")
    out = P.plot_nematic_director(ax, Qxx, Qxy, rho, skip=2)
    assert out is not None


# ---- A5: SPDE snapshots & animation ----

@pytest.fixture
def grid_coll():
    from SFI.trajectory import TrajectoryCollection

    rng = np.random.RandomState(1)
    T, n = 10, 8
    N = n * n
    scal = rng.rand(T, N, 1)
    vec = 0.1 * rng.randn(T, N, 2)
    Xarr = np.concatenate([scal, vec], axis=-1).astype(np.float32)  # (T, N, 3)
    coll = TrajectoryCollection.from_arrays(X=jnp.asarray(Xarr), dt=0.1)
    return coll, (n, n)


def test_plot_spde_snapshot_single_and_multi(grid_coll):
    coll, gs = grid_coll
    _fresh()
    P.plot_spde_snapshot(coll, 0, grid_shape=gs, vector_channels=(1, 2), render="streamplot")
    _fresh()
    axes = P.plot_spde_snapshot(coll, [0, 5], grid_shape=gs, vector_channels=(1, 2), render="quiver")
    assert len(np.atleast_1d(axes)) == 2


def test_spatial_acorr2d():
    rng = np.random.RandomState(2)
    f = rng.randn(16, 16)
    r, C = P.spatial_acorr2d(f, dx=1.0)
    assert r.shape == C.shape
    assert np.isfinite(C[0])


def test_animate_particles(active_coll):
    _fresh()
    anim = P.animate_particles(active_coll, trail=3, skip=2)
    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)
    anim._func(0)  # exercise one frame


def test_animate_spde_comparison(grid_coll):
    coll, gs = grid_coll
    _fresh()
    anim = P.animate_spde_comparison(coll, coll, grid_shape=gs, skip=2)
    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)
    anim._func(0)


# ---- A6: recovery / reporting ----

def test_plot_recovery_bar_extensions():
    _fresh()
    ax = P.plot_recovery_bar(
        [1.0, -0.5, 0.2], [0, 2, 3],
        coeffs_true=[1.0, -0.4], support_true=[0, 2],
        stderr=[0.1, 0.1, 0.05], labels=["1", "x", "x2", "x3", "x4"],
        sort=True, show_pruned=True,
    )
    assert ax is not None


def test_plot_recovery_bar_multi():
    _fresh()
    ax = P.plot_recovery_bar_multi(
        [[1.0, 0.2, 0.0], [0.9, 0.25, 0.05]], ["a", "b", "c"],
        coeffs_true=[1.0, 0.2, 0.0], group_names=["GN", "L-BFGS"],
    )
    assert ax is not None


def test_plot_recovery_matrix():
    _fresh()
    true = np.eye(3)
    inferred = np.eye(3) + 0.05 * np.random.RandomState(0).randn(3, 3)
    axes = P.plot_recovery_matrix(true, inferred, row_labels=list("xyz"), col_labels=list("xyz"))
    assert len(axes) == 2


def test_plot_field_shared_axes_savefig(fit2d):
    """plot_field's equal-aspect must survive savefig on SHARED axes.

    Regression: the legacy ``plt.axis("equal")`` (adjustable='datalim')
    raises at draw time when axes are shared — which only surfaces during
    sphinx-gallery's savefig, not a no-op ``plt.show()``. ``_equal_aspect``
    falls back to adjustable='box'. (This broke nn_force in the gallery build.)
    """
    import io

    inf, proc, coll = fit2d
    _fresh()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.sca(axes[0])
    P.plot_field(coll, inf.force_inferred, N=6)
    fig.savefig(io.BytesIO(), format="png")  # triggers draw -> apply_aspect
