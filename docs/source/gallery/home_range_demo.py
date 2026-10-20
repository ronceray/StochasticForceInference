r"""
Home ranges in a shared landscape, from noisy gappy data
========================================================

Four tracked agents roam overlapping territories crossed by winding
river valleys.  From their noisy, gappy, positions-only tracks we ask:
**where does each agent live** — the centre, size and orientation of
its home range — **how strongly do they avoid each other**, and **how
much does the terrain steer them**?

Each agent :math:`i` is tethered to its own anchor
:math:`\mathbf{x}^0_i` by its own **tensor** stiffness
:math:`\mathsf{A}_i` (an ellipse, not a circle); agents **repel** one
another; and all feel the slope of a known topographic map :math:`h`:

.. math::

   \dot{\mathbf{v}}_i = -\mathsf{A}_i(\mathbf{x}_i - \mathbf{x}^0_i)
                        - \gamma\,\mathbf{v}_i
                        + \sum_{j\neq i}\phi(r_{ij})\,\hat{\mathbf{r}}_{ij}
                        - k\,\nabla h(\mathbf{x}_i)
                        + \sqrt{2D}\,\boldsymbol{\xi}_i .

Everything stays **linear in the parameters**: expanding
:math:`-\mathsf{A}_i(\mathbf{x}-\mathbf{x}^0_i) = -\mathsf{A}_i\mathbf{x}
+ \mathbf{c}_i`, we fit :math:`(\mathsf{A}_i, \mathbf{c}_i)` and recover
the anchor :math:`\mathbf{x}^0_i = \mathsf{A}_i^{-1}\mathbf{c}_i`; the
known map enters as one fixed feature :math:`-\nabla h` with a shared
coupling :math:`k`.  The fit is the noise-aware parametric estimator,
checked against ground truth, by the residual diagnostics, and by a
bootstrap simulation.

.. rubric:: Tags

synthetic · underdamped · multi-particle · 2D · per-particle
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "multi-particle", "2D", "per-particle"]
# sphinx_gallery_thumbnail_number = 3

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Ellipse

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import animate_particles_comparison, dark_fig

apply_style()
# Dark figures with an opaque background: the movie is encoded without alpha.
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False

AGENT_COLORS = ["#FFC20A", "#FF2D6F", "#9BE564", "#D35FB7"]
RIVER_CMAP = LinearSegmentedColormap.from_list("river", ["#3d8fc4", "#1d4560", "#15181d"])


def _ellipse_axes(A, kappa=0.9):
    """Iso-potential contour {(x-x0)·A·(x-x0)=kappa^2} as (width, height, angle°)."""
    w, Vv = np.linalg.eigh(A)
    semi = kappa / np.sqrt(np.maximum(w, 1e-6))
    return 2 * semi[0], 2 * semi[1], np.degrees(np.arctan2(Vv[1, 0], Vv[0, 0]))
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A colony in a river landscape
# -----------------------------
#
# ``N = 4`` agents.  Each stiffness tensor :math:`\mathsf{A}_i` has its
# own eigenvalues and tilt, and the anchors are close enough for the
# ranges to overlap.  The map :math:`h(\mathbf{x})` carves two
# **meandering river valleys** — Gaussian channels whose centrelines wind
# sinusoidally, with structure finer than a home range.

import SFI
from SFI.bases import V, identity_matrix_basis
from SFI.bases.pairs import gaussian_kernels, radial_pair_basis
from SFI.langevin import UnderdampedProcess
from SFI.statefunc import make_basis

N, d = 4, 2
eigs = np.array([[1.4, 0.5], [0.45, 1.3], [1.1, 0.4], [0.6, 1.0]])
tilts = np.array([0.3, 1.1, -0.6, 2.0])
A_true = np.zeros((N, d, d))
for i in range(N):
    c, s = np.cos(tilts[i]), np.sin(tilts[i])
    R = np.array([[c, -s], [s, c]])
    A_true[i] = R @ np.diag(eigs[i]) @ R.T

anchors = np.array([[0.0, 0.0], [1.6, 0.5], [0.7, 1.7], [-1.0, 1.2]])
gamma_true, k_rep, ell, D_true, k_topo = 1.6, -0.4, 0.7, 0.18, 0.6  # k_rep<0 ⇒ repulsive

# Known topography h: two meandering Gaussian valleys along a tilted axis.
_ang = 0.6
_t_hat = jnp.array([np.cos(_ang), np.sin(_ang)])     # along-valley axis
_n_hat = jnp.array([-np.sin(_ang), np.cos(_ang)])    # across-valley axis
_chans = ((0.55, 0.9, 1.1, 0.10, -0.55),             # (depth, wavelength, phase,
          (0.40, 1.3, -0.7, 0.09, 0.75))             #  width, centre offset)
_MEANDER = 0.35                                      # centreline amplitude


def grad_h(x):
    u = jnp.dot(x, _t_hat)                            # along-valley
    vperp = jnp.dot(x, _n_hat)                        # across-valley
    G = jnp.zeros(2)
    for depth, L, ph, wd, offc in _chans:
        phase = 2.0 * jnp.pi * u / L + ph
        cu = offc + _MEANDER * jnp.sin(phase)        # winding centreline
        dcu = _MEANDER * (2.0 * jnp.pi / L) * jnp.cos(phase)
        s = vperp - cu
        g = jnp.exp(-0.5 * (s / wd) ** 2)
        G = G + depth * (s / wd**2) * g * (_n_hat - dcu * _t_hat)
    return G


def topo_height(xy):                              # for plotting the landscape
    th, nh = np.asarray(_t_hat), np.asarray(_n_hat)
    u, vperp = xy @ th, xy @ nh
    H = np.zeros(xy.shape[0])
    for depth, L, ph, wd, offc in _chans:
        cu = offc + _MEANDER * np.sin(2.0 * np.pi * u / L + ph)
        H -= depth * np.exp(-0.5 * ((vperp - cu) / wd) ** 2)
    return H

# %%
# The force basis: 5N + 3 linear features
# ---------------------------------------
#
# SFI's built-in bases are shared by all particles; *individual*
# coefficients need one custom brick — the only custom code on this
# page.  :func:`~SFI.statefunc.make_basis` reads each agent's id from its
# per-particle extras and emits five features (three for the symmetric
# tensor, two for :math:`\mathbf{c}_i`) that are non-zero only for that
# agent.  Friction, a Gaussian pair repulsion and the landscape slope
# :math:`-\nabla h` are shared.

def home(x, *, extras):
    i = extras["home_id"]
    x0c, x1c = x[0], x[1]
    blocks = jnp.stack([
        jnp.array([-x0c, 0.0]),     # A_xx
        jnp.array([0.0, -x1c]),     # A_yy
        jnp.array([-x1c, -x0c]),    # A_xy
        jnp.array([1.0, 0.0]),      # c_x
        jnp.array([0.0, 1.0]),      # c_y
    ], axis=1)
    onehot = (jnp.arange(N) == i).astype(x.dtype)
    return (onehot[None, :, None] * blocks[:, None, :]).reshape(d, 5 * N)


B_home = make_basis(home, dim=d, rank=1, n_features=5 * N,
                    extras_keys=("home_id",), particle_extras=("home_id",),
                    labels=[f"{p}[{i}]" for i in range(N) for p in ("Axx", "Ayy", "Axy", "cx", "cy")])
B_friction = V(dim=d)
B_repulsion = radial_pair_basis(gaussian_kernels([ell]), dim=d).dispatch_pairs()
B_landscape = make_basis(lambda x: (-grad_h(x))[:, None], dim=d, rank=1, n_features=1,
                         labels=["-grad h"])
B = B_home & B_friction & B_repulsion & B_landscape

# %%
# Simulate, then corrupt the recording
# ------------------------------------
#
# The recording gets Gaussian localisation noise and 3% of the positions
# are dropped.  The relevant scale for the noise is the displacement per
# frame, not the range size: velocities and accelerations are
# differences of positions, so the noise enters the acceleration as
# :math:`\sigma/\Delta t^2`.

theta_well = []
for i in range(N):
    A, x0 = A_true[i], anchors[i]
    c = A @ x0
    theta_well += [A[0, 0], A[1, 1], A[0, 1], c[0], c[1]]
theta_true = jnp.asarray(theta_well + [-gamma_true, k_rep, k_topo])

proc = UnderdampedProcess(B, D=D_true, theta_F=theta_true)
proc.set_extras(extras_local={"home_id": jnp.arange(N)})
proc.initialize(jnp.asarray(anchors), v0=jnp.zeros((N, d)))
coll_clean = proc.simulate(
    dt=0.02, Nsteps=12000, key=random.PRNGKey(0), oversampling=8, prerun=200,
)

sigma_loc = 0.008                       # localisation noise (std per coordinate)
coll = coll_clean.degrade(noise=sigma_loc, data_loss_fraction=0.03, seed=3)
dt = coll.dt

X_clean = coll_clean.to_array()         # (T, N, d), ground truth (for the comparison only)
spread = float(X_clean.std(axis=0).mean())
step = float(np.sqrt(np.mean(np.diff(X_clean, axis=0) ** 2)))
print(f"{coll.T} frames;  range size ~{spread:.2f};  displacement per frame ~{step:.4f}")
print(f"localisation noise σ = {sigma_loc}  (~{sigma_loc / step:.1f}× the per-frame step)")

# %%
# Left: the whole recording on the map (river beds bright).  Right: a
# close-up of one agent over 200 frames — the observed positions (dots)
# scatter around the true path by as much as it advances per frame.

# sphinx_gallery_start_ignore
Xd = coll.to_array()                    # observed, NaN where dropped
pad = 0.25
x_lo, x_hi = np.nanmin(Xd[..., 0]) - pad, np.nanmax(Xd[..., 0]) + pad
y_lo, y_hi = np.nanmin(Xd[..., 1]) - pad, np.nanmax(Xd[..., 1]) + pad
gx, gy = np.linspace(x_lo, x_hi, 260), np.linspace(y_lo, y_hi, 220)
GX, GY = np.meshgrid(gx, gy)
Hgrid = topo_height(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GX.shape)


def _landscape(ax):
    ax.contourf(GX, GY, Hgrid, levels=24, cmap=RIVER_CMAP, zorder=0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")


fig, (ax, ax_z) = dark_fig(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
_landscape(ax)
for i in range(N):
    ax.scatter(Xd[:, i, 0], Xd[:, i, 1], s=0.6, alpha=0.08, color=AGENT_COLORS[i],
               rasterized=True, zorder=2)
    ax.plot([], [], "o", color=AGENT_COLORS[i], label=f"agent {i}")
ax.legend(loc="upper left", fontsize=8, framealpha=0.3, markerscale=1.2)
ax.set_title("Observed positions, on the topographic map")
ax.set_xlabel("x"); ax.set_ylabel("y")

i0, t0, nz = 0, 3000, 200
ax_z.plot(X_clean[t0:t0 + nz, i0, 0], X_clean[t0:t0 + nz, i0, 1], "-", lw=1.2,
          color="#B0B0B0", label="true path")
ax_z.plot(Xd[t0:t0 + nz, i0, 0], Xd[t0:t0 + nz, i0, 1], "o", ms=2.2,
          color=AGENT_COLORS[i0], label="observed")
ax_z.set_aspect("equal", adjustable="datalim")
ax_z.set_title(f"Agent {i0}, {nz} frames: noise ≈ step")
ax_z.set_xlabel("x"); ax_z.set_ylabel("y")
ax_z.legend(loc="best", fontsize=8, framealpha=0.3)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Noise-aware parametric fit
# --------------------------
#
# At this noise level the linear estimator is overwhelmed; the
# parametric :meth:`~SFI.inference.UnderdampedLangevinInference.infer_force`
# models the noise and estimates the diffusion and the noise level
# itself.  For the bootstrap below we also fit an isotropic diffusion
# :math:`D\,\mathsf{I}` (the motion is isotropic, and a full matrix
# would add a poorly constrained direction).

inf = SFI.UnderdampedLangevinInference(coll)
inf.infer_force(B)
inf.compute_force_error()
inf.infer_diffusion(identity_matrix_basis(d))

# Each agent's tensor and anchor, from its five coefficients.
c_hat = np.asarray(inf.force_coefficients).ravel()
s_hat = np.asarray(inf.force_coefficients_stderr).ravel()
A_hat = np.zeros((N, d, d))
x0_hat = np.zeros((N, d))
for i in range(N):
    a, b, cc, cx, cy = c_hat[5 * i:5 * i + 5]
    A_hat[i] = np.array([[a, cc], [cc, b]])
    x0_hat[i] = np.linalg.solve(A_hat[i], [cx, cy])

for name, j, true in [("friction γ", 5 * N, gamma_true), ("repulsion", 5 * N + 1, k_rep),
                      ("landscape k", 5 * N + 2, k_topo)]:
    sign = -1 if j == 5 * N else 1                   # the friction feature is +v
    print(f"{name:>12}: {sign * c_hat[j]:+.2f} ± {s_hat[j]:.2f}   (true {true:+.2f})")
print(f"localisation noise: σ̂ = {np.sqrt(np.mean(np.diag(np.asarray(inf.Lambda)))):.4f}"
      f"   (true {sigma_loc})")
print(f"anchor error, mean over agents: {np.mean(np.linalg.norm(x0_hat - anchors, axis=1)):.3f}"
      f"   (range size ~{spread:.2f})")

# %%
# Validate: ground truth and residual diagnostics
# -----------------------------------------------
#
# On a synthetic benchmark the inferred force can be compared with the
# truth directly: the realised error (``Normalized MSE``) should be of
# the order of the predicted one.

inf.compare_to_exact(model_exact=proc)
inf.print_report()

# %%
# Without ground truth, the residual tests are the check.  They account
# for the estimated localisation noise — including the correlation it
# induces between successive accelerations — so a well-specified fit
# passes them despite the noise.

report = inf.diagnose()
report.print_summary()

# %%
# Bootstrap: simulate the inferred model
# --------------------------------------
#
# A fresh trajectory drawn from the fitted force and diffusion (right)
# next to the data (left), both over the *inferred* home ranges (dashed):
# the same territories, overlaps and drift along the river valleys.

coll_boot, _ = inf.simulate_bootstrapped_trajectory(key=random.PRNGKey(5), oversampling=8)

# sphinx_gallery_start_ignore
fig_m, axes_m = dark_fig(1, 2, figsize=(6.6, 3.4))
for ax_, ttl in zip(axes_m, ["data", "bootstrap (inferred model)"]):
    ax_.contourf(GX, GY, Hgrid, levels=24, cmap=RIVER_CMAP, zorder=0)
    for i in range(N):
        wH, hH, aH = _ellipse_axes(A_hat[i])
        ax_.add_patch(Ellipse(x0_hat[i], wH, hH, angle=aH, fill=False,
                              edgecolor=AGENT_COLORS[i], lw=1.2, ls="--", zorder=4))
    ax_.set_title(ttl, fontsize=9)
anim = animate_particles_comparison(
    [coll, coll_boot], axes=axes_m, trail=300, skip=60, interval=70, s=30,
    cmap=ListedColormap(AGENT_COLORS),
)
for ax_ in axes_m:
    ax_.set_xlim(x_lo, x_hi); ax_.set_ylim(y_lo, y_hi)
    ax_.set_xticks([]); ax_.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# True vs inferred home ranges
# ----------------------------
#
# Recovered ellipses (dashed) over the ground truth (solid): each
# agent's range — centre, elongation, tilt — is recovered from noisy,
# gappy, positions-only data, together with the shared friction,
# repulsion and river coupling.

# sphinx_gallery_start_ignore
fig, ax = dark_fig(figsize=(6.4, 5.2))
_landscape(ax)
for i in range(N):
    col = AGENT_COLORS[i]
    ax.scatter(Xd[:, i, 0], Xd[:, i, 1], s=0.5, alpha=0.05, color=col, rasterized=True, zorder=2)
    wT, hT, aT = _ellipse_axes(A_true[i])
    ax.add_patch(Ellipse(anchors[i], wT, hT, angle=aT, fill=False,
                         edgecolor=col, lw=2.6, ls="-", zorder=4))
    wH, hH, aH = _ellipse_axes(A_hat[i])
    ax.add_patch(Ellipse(x0_hat[i], wH, hH, angle=aH, fill=False,
                         edgecolor="white", lw=1.5, ls="--", zorder=5))
    ax.plot(*anchors[i], "+", color=col, ms=12, mew=2.2, zorder=6)
    ax.plot(*x0_hat[i], "x", color="white", ms=7, mew=1.5, zorder=6)
ax.plot([], [], "-", color="#D0D0D0", lw=2.6, label="true range  (+ anchor)")
ax.plot([], [], "--", color="white", lw=1.5, label="inferred range  (× anchor)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Home ranges: truth (solid) vs inferred (dashed)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.4)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Notes
# -----
#
# - One :func:`~SFI.statefunc.make_basis` brick with ``particle_extras=``
#   gives **per-agent tensor coefficients**; the centre stays a *linear*
#   parameter via :math:`\mathbf{c}_i = \mathsf{A}_i\mathbf{x}^0_i`.
# - A **known landscape** enters as a fixed feature :math:`-\nabla h` with
#   one shared coupling — the same pattern fits any external field whose
#   shape is known (a river, a thermal gradient, an illumination map).
# - Underdamped dynamics with localisation noise is the hardest regime SFI
#   targets: velocities are never observed and the noise is amplified by
#   :math:`1/\Delta t^2`.  See :doc:`/inference/underdamped` and
#   :doc:`/inference/noise_and_sampling`.
