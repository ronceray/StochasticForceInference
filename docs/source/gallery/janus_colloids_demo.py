r"""
Janus colloids — learning general pair interactions from experimental tracks
=============================================================================

Self-propelled Janus colloids interact through electrostatic and
hydrodynamic forces that cannot be computed from first principles.  Hem,
Poncet, Ronceray, Nishiguchi and Démery (*Soft Matter* 2025,
`doi:10.1039/D5SM00655D <https://doi.org/10.1039/D5SM00655D>`_) learned
them from tracking data instead, with a **generic** basis: a Fourier
series in the two relative orientations times smooth radial functions,
for the radial velocity, the transverse velocity and the torque that one
colloid induces on another.

This page redoes that analysis with SFI on the experimental tracks the
authors distribute with their code — the first 100 frames of their
flocking dataset (set A, Iwasawa, Nishiguchi & Sano, *Phys. Rev. Research*
2021), about 1 700 colloids at 15 frames per second — then simulates the
learned model to see whether it flocks.

.. rubric:: Tags

real-data · overdamped · multi-particle · linear · custom-basis · 2D · non-equilibrium
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["real-data", "overdamped", "multi-particle", "linear", "custom-basis", "2D", "non-equilibrium"]
# sphinx_gallery_thumbnail_number = 7

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import warnings
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from _gallery_utils.janus_colloids import (
    body_frame_density,
    interaction_reader,
    load_set_a,
    orientation_features,
    periodic_tracks_collection,
    polar_order_vs_area,
    polarity_autocorrelation,
    relative_orientations,
    residual_noise,
    sample_pair_configurations,
    simulate_fitted,
    tracks_collection,
    tracks_to_arrays,
    tapered_polyexp_kernels,
)
from SFI import TrajectoryCollection
from SFI.utils.plotting import animate_particles, animate_particles_comparison, dark_fig

apply_style()
plt.rcParams["savefig.facecolor"] = "auto"   # movies on their own dark background
warnings.formatwarning = lambda msg, cat, *args, **kw: f"{cat.__name__}: {msg}\n"
warnings.filterwarnings("ignore", message=r".*Cartesian product over feature axes.*")

cache_dir = Path(__file__).resolve().parent / "_cache" if "__file__" in dir() else Path("_cache")
SIGMA = 3.17                                        # particle diameter (µm)
TH = np.linspace(0, 2 * np.pi, 97)
T1, T2 = np.meshgrid(TH, TH, indexing="ij")
TITLES = [r"$v_r/U$", r"$v_\theta/U$", r"$\sigma\omega/U$"]


def _angle_axes(ax):
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")


def _particle_axes(axes, fov):
    for ax in np.atleast_1d(axes):
        ax.set_xlim(0, fov[0]); ax.set_ylim(0, fov[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for col in ax.collections:
            col.set_edgecolor("none")
# sphinx_gallery_end_ignore
stamp_output()

# %%
# The experiment
# --------------
#
# Silica spheres of diameter :math:`\sigma = 3.17\,\mu`\ m, half-coated
# with titanium, swim metal side first under a vertical AC field and
# organise into a polar flock.  The movie of this experiment published
# with Iwasawa *et al.* (arXiv:2011.14548, ancillary file, 30 fps):
#
# .. raw:: html
#
#    <video controls loop muted playsinline width="100%"
#           src="https://arxiv.org/src/2011.14548v1/anc/3_Ordered.mp4"></video>
#
# The tracks — position, heading :math:`\varphi` and a track label per
# detected particle and frame — come from the authors' repository (5 MB,
# read without executing code from the file).  Particles enter and leave
# the field of view, so the tracks become a masked collection, with
# per-frame neighbour lists and a fit mask that keeps particles near the
# edges (whose neighbours were partly not recorded) as neighbours only:

frames, dt, fov = load_set_a(cache_dir)
R_C = 21.0                                          # interaction range (µm), as in the paper
coll = tracks_collection(frames, dt, fov, cutoff=R_C, margin=40.0)
print(f"{len(frames)} frames at dt = {dt:.4f} s, {coll.datasets[0].X.shape[1]} tracks, "
      f"{np.mean([len(f) for f in frames]):.0f} particles per frame")

# sphinx_gallery_start_ignore
X_obs, present = tracks_to_arrays(frames)
fig_tr, ax_tr = dark_fig(figsize=(6.5, 5.2))
anim_tracks = animate_particles(TrajectoryCollection.from_arrays(X=X_obs, mask=present, dt=dt),
                                color_dim=2, cmap="hsv", vmin=0, vmax=2 * np.pi, s=4.0,
                                interval=1000 * dt, ax=ax_tr)
_particle_axes(ax_tr, fov)
ax_tr.set_title("tracks, coloured by heading", color="#D0D0D0")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The generic interaction basis
# -----------------------------
#
# The effect of particle 2 on particle 1 is a radial velocity
# :math:`v_r`, a transverse velocity :math:`v_\theta` and a torque
# :math:`\omega`, functions of the distance :math:`r` and of the headings
# :math:`\theta_1, \theta_2` measured from :math:`\mathbf r_1 - \mathbf
# r_2`.  The paper expands them as
# :math:`\sum_l g_l(r) \sum_{|k_1|+|k_2| \le N_a} a_{l,k_1k_2}
# \cos\text{ or }\sin(k_1\theta_1 + k_2\theta_2)` — cosines for
# :math:`v_r`, sines for :math:`v_\theta` and :math:`\omega`, as parity
# requires — with :math:`g_l(r) = r^l e^{-r}/l!` (:math:`r` in µm,
# :math:`l = 0, 2, \ldots, 14`, as in the paper's code).  In SFI this is
# the product of a radial and an angular pair interactor; we multiply the
# radial functions by a window that vanishes smoothly at :math:`r_c` and
# below :math:`1.5\,\sigma`, where the tracks hold almost no pairs.
# Self-propulsion and three more one-body terms complete the basis.

from SFI.bases import unit_axes, x_components
from SFI.bases.pairs import heading_vector, scalar_pair_basis
from SFI.statefunc import Basis

kernels = tapered_polyexp_kernels(cutoff=R_C, onset=(1.0 * SIGMA, 1.5 * SIGMA))


def pair_interactions(Na, box=None):
    return scalar_pair_basis(kernels, dim=3, box=box, spatial_dims=slice(0, 2)) * orientation_features(Na, box=box)


def pair_basis(Na, box=None):
    return pair_interactions(Na, box).dispatch_pairs_from_extras(
        indptr_key="indptr", indices_key="indices", return_as="basis")


x, y, phi = x_components(3)
e_x, e_y, _ = unit_axes(3)
e_heading = heading_vector(dim=3, angle_index=2)                                   # U ê_φ
e_side = phi.elementwisemap(jnp.sin) * e_x - phi.elementwisemap(jnp.cos) * e_y     # ê_φ⊥
one_body = [e_x, e_y, e_heading, e_side]

pairs_2 = pair_basis(Na=2)
basis = Basis.stack([*one_body, pairs_2])
print(f"{basis.n_features} features, {pairs_2.n_features} of them pair interactions")

# %%
# Inference
# ---------
#
# Linear SFI, on every frame.  The tracks carry localisation noise (we
# measure it below), so we use **Itô-shift** moments — the basis is
# evaluated one frame before each displacement, whose noise it then does
# not see — with the trapeze Gram matrix.  The ill-conditioned radial
# functions (SFI warns about the Gram matrix) are why this script runs JAX
# in double precision.

from SFI import OverdampedLangevinInference

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(basis, M_mode="Ito-shift", G_mode="trapeze")
inf.compute_force_error()

U, U_cov = inf.coeff_block(e_heading)
U = float(np.ravel(U)[0])
print(f"self-propulsion U = {U:.2f} ± {float(np.sqrt(np.ravel(U_cov)[0])):.2f} µm/s")

# %%
# The learned interactions
# ------------------------
#
# The analogue of the paper's Fig. 2, at :math:`r = 2.5\,\sigma`, the most
# frequent neighbour distance.  Left: how often each pair of relative
# orientations occurs — neighbours mostly move in the same direction
# (:math:`\theta_1 \approx \theta_2`).  Hatching marks where the 95 %
# confidence interval is wider than the value.

read = interaction_reader(pair_interactions(2))
value, stderr = read(*inf.coeff_block(pairs_2), 2.5 * SIGMA, T1, T2)
t1_obs, t2_obs = relative_orientations(frames, 2.25 * SIGMA, 2.75 * SIGMA)

# sphinx_gallery_start_ignore
scales = [1 / U, 1 / U, SIGMA / U]
fig_maps, axes_maps = plt.subplots(1, 4, figsize=(16, 3.9))
h = axes_maps[0].hist2d(t1_obs, t2_obs, bins=36, range=[[0, 2 * np.pi]] * 2, cmap="magma", rasterized=True)
plt.colorbar(h[3], ax=axes_maps[0], label="pairs")
axes_maps[0].set_title(r"pairs at $r \approx 2.5\,\sigma$", fontsize=11)
_angle_axes(axes_maps[0])
for j, ax in enumerate(axes_maps[1:]):
    im = ax.pcolormesh(TH, TH, (value[..., j] * scales[j]).T, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                       shading="auto", rasterized=True)
    with plt.rc_context({"hatch.color": "#B0B0B0"}):
        ax.contourf(TH, TH, (1.96 * stderr[..., j] > np.abs(value[..., j])).T.astype(float),
                    levels=[0.5, 1.5], hatches=["xxx"], colors="none")
    _angle_axes(ax)
    ax.set_title(TITLES[j], fontsize=11)
fig_maps.colorbar(im, ax=axes_maps[1:], shrink=0.85)
plt.show()
# sphinx_gallery_end_ignore

# %%
# A repulsion strongest when particle 1 is in front of particle 2 and
# turned sideways, a weak transverse velocity, and a torque that turns
# particle 1 away from particle 2, most strongly when particle 2 faces it
# (:math:`\theta_2 \approx 0`) — the "modulated turning away" of the
# paper.  Along :math:`r`, with the distribution of pair distances:

r = np.linspace(1.3 * SIGMA, 6.5 * SIGMA, 120)
profiles = {t1: read(*inf.coeff_block(pairs_2), r, t1, 0.0) for t1 in (np.pi / 2, 3 * np.pi / 2)}

# sphinx_gallery_start_ignore
from scipy.spatial import cKDTree

r_pairs = np.concatenate([np.hypot(*(f[p[:, 0], :2] - f[p[:, 1], :2]).T) for f in frames[::10]
                          for p in [cKDTree(f[:, :2]).query_pairs(6.5 * SIGMA, output_type="ndarray")]])
fig_rad, axes_rad = plt.subplots(1, 2, figsize=(12, 4))
for ax, j in [(axes_rad[0], 0), (axes_rad[1], 2)]:
    ax_h = ax.twinx()
    ax_h.hist(r_pairs / SIGMA, bins=80, range=(1.3, 6.5), color="#808080", alpha=0.15)
    ax_h.set_yticks([]); ax_h.spines["right"].set_visible(False)
    ax.set_zorder(ax_h.get_zorder() + 1); ax.patch.set_visible(False)
    for (t1, (v, e)), col, lab in zip(profiles.items(), [SFI_COLORS["inferred"], SFI_COLORS["highlight"]],
                                      [r"$(\theta_1, \theta_2) = (\pi/2, 0)$", r"$(3\pi/2, 0)$"]):
        ax.plot(r / SIGMA, v[:, j] * scales[j], color=col, lw=2, label=lab)
        ax.fill_between(r / SIGMA, (v[:, j] - 1.96 * e[:, j]) * scales[j], (v[:, j] + 1.96 * e[:, j]) * scales[j],
                        color=col, alpha=0.25)
    ax.axhline(0, color="#808080", lw=0.8)
    ax.set_xlabel(r"$r/\sigma$"); ax.set_ylabel(TITLES[j]); ax.legend(fontsize=9)
plt.show()
# sphinx_gallery_end_ignore

# %%
# How much noise, and does it matter?
# -----------------------------------
#
# SFI's measurement-noise estimate :math:`\hat\Lambda = -\langle \Delta
# X_t\,\Delta X_{t-1}\rangle` comes out *negative* on the positions:
# successive displacements of a swimmer are correlated, and that
# persistence, :math:`\langle F_t F_{t-1}\rangle\Delta t^2`, hides the
# noise.  On the residuals of the fit, :math:`r_t = \Delta X_t - \hat
# F(X_t)\Delta t`, the drift is gone and the same estimator works:

Lam, D_res = residual_noise(inf, coll)
noise_sd = np.sqrt(np.clip(np.diag(Lam), 0, None))
D_r = D_res[2, 2]
print("Λ̂ from displacements (x, y, φ):", np.diag(np.asarray(inf.Lambda)).round(4))
print("Λ̂ from residuals     (x, y, φ):", np.diag(Lam).round(4))
print(f"localisation noise {noise_sd[0]:.2f}, {noise_sd[1]:.2f} µm and {np.degrees(noise_sd[2]):.1f}°;  "
      f"rotational diffusion D_r = {D_r:.3f} 1/s")

# %%
# About two pixels of positional noise — as large as the distance a
# colloid swims in a frame — and a few degrees on the headings.  To see
# what that does to the inference, we let the learned model generate data
# with a known answer: simulate it from the first frame (periodic box of
# the size of the field of view, the rotational diffusion just measured,
# translational diffusion neglected as in the paper), add noise of the
# measured size, and
# refit — with Itô-shift, and with the paper's convention (Stratonovich
# moments, rectangle Gram matrix, ``preset="legacy-sfi-v1.0"``).

D_sim = np.diag([0.0, 0.0, D_r])
x0 = frames[0][:, :3]
sim = dict(box=fov, dt=dt, cutoff=R_C)
X_true = simulate_fitted(inf, one_body, pairs_2, kernels, 2, x0, 10 + len(frames) * dt, D=D_sim,
                         key=jax.random.PRNGKey(1), **sim)[-len(frames):]
rng = np.random.default_rng(0)
X_noisy = X_true + noise_sd * rng.standard_normal(X_true.shape)

configs = sample_pair_configurations(X_true[::10], fov, R_C, 20000, rng)
v_true, _ = read(inf.coeff_block(pairs_2)[0], None, *configs.T)
print(f"{'refit on simulated noisy tracks':34s}    U    error on  v_r   v_θ    ω   (true U = {U:.2f})")
for label, opts in [("Itô-shift, trapeze", dict(M_mode="Ito-shift", G_mode="trapeze")),
                    ("paper: Stratonovich, rectangle", dict(preset="legacy-sfi-v1.0"))]:
    pairs_b = pair_basis(2, box="extras")
    refit = OverdampedLangevinInference(periodic_tracks_collection(X_noisy, dt, fov, cutoff=R_C))
    refit.compute_diffusion_constant(method="WeakNoise")
    refit.infer_force_linear(Basis.stack([*one_body, pairs_b]), **opts)
    v_fit, _ = read(refit.coeff_block(pairs_b)[0], None, *configs.T)
    err = np.sqrt(np.mean((v_fit - v_true) ** 2, 0) / np.mean(v_true ** 2, 0))
    print(f"{label:34s} {float(np.ravel(refit.coeff_block(e_heading)[0])[0]):6.2f}"
          f"   {'':9s}{err[0]:5.2f} {err[1]:5.2f} {err[2]:5.2f}")

# %%
# (Relative RMS errors over the simulated pairs.)  Itô-shift recovers the
# self-propulsion and, to within roughly 15–30 %, the interactions; the
# transverse velocity, the weakest of the three, is the least precise.  The
# Stratonovich moments of the paper are thrown off by the noise on the
# positions: their Itô-to-Stratonovich correction rests on a per-step
# diffusion estimate that, for particles swimming at :math:`U`, is
# dominated by :math:`U^2\Delta t`.
#
# Does the learned model flock?
# -----------------------------
#
# As in the paper we fit models of angular order :math:`N_a = 0`
# (isotropic radial interactions only), 1 and 2, and simulate each for
# 60 s from the first experimental frame.  The flock is a balance between
# turning-away torques and rotational noise, so we also run the
# first-order model with the paper's rotational diffusion,
# :math:`D_r = 0.092` s⁻¹.

models = {2: (inf, pairs_2)}
for Na in (0, 1):
    pairs_Na = pair_basis(Na)
    models[Na] = (OverdampedLangevinInference(coll), pairs_Na)
    models[Na][0].compute_diffusion_constant(method="WeakNoise")
    models[Na][0].infer_force_linear(Basis.stack([*one_body, pairs_Na]), M_mode="Ito-shift", G_mode="trapeze")

def run(Na, D, seed):
    fitted, pairs_Na = models[Na]
    return simulate_fitted(fitted, one_body, pairs_Na, kernels, Na, x0, 60.0, D=D, key=jax.random.PRNGKey(seed), **sim)


runs = {Na: run(Na, D_sim, 10 + Na) for Na in (0, 1, 2)}
runs["1, paper D_r"] = run(1, np.diag([0.0, 0.0, 0.092]), 11)
for name, X in runs.items():
    print(f"N_a = {name}: polar order after 60 s = {np.abs(np.exp(1j * X[-1, :, 2]).mean()):.2f}")

# %%
# The experiment (left) and the second-order model (right), from the same
# frame, over the 6.7 s of the recording:

# sphinx_gallery_start_ignore
wrapped = runs[2][:len(frames)].copy()
wrapped[..., :2] = np.mod(wrapped[..., :2], fov)
fig_mov, axes_mov = dark_fig(1, 2, figsize=(9, 4))
anim_cmp = animate_particles_comparison(
    [TrajectoryCollection.from_arrays(X=X_obs, mask=present, dt=dt),
     TrajectoryCollection.from_arrays(X=wrapped, dt=dt)],
    titles=["experiment", r"learned model ($N_a = 2$)"], color_dim=2, cmap="hsv",
    vmin=0, vmax=2 * np.pi, s=3.0, interval=1000 * dt, axes=axes_mov)
_particle_axes(axes_mov, fov)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Statistics the fit never used: the polar order :math:`C(S)` in square
# regions of area :math:`S` (second half of each run), the polarity
# autocorrelation :math:`f_p(\tau) = \langle\cos[\varphi(t+\tau) -
# \varphi(t)]\rangle`, and the density of neighbours around a particle in
# its own frame.

sides = np.array([10.0, 20.0, 40.0, 84.0, 168.0])
lags = np.arange(0, 451, 5)
C_data = polar_order_vs_area(frames, fov, sides)
fp_data = polarity_autocorrelation(np.where(present, X_obs[..., 2], np.nan), lags[lags < len(frames)])
C_sim = {Na: polar_order_vs_area(list(X[len(X) // 2:]), fov, sides) for Na, X in runs.items()}
fp_sim = {Na: polarity_autocorrelation(X[..., 2], lags[lags < len(X)]) for Na, X in runs.items()}
rho_data = body_frame_density(frames, fov)
rho_sim = body_frame_density(list(runs[2][len(runs[2]) // 2::5]), fov, periodic=True)

# sphinx_gallery_start_ignore
colors = {0: SFI_COLORS["tertiary"], 1: SFI_COLORS["highlight"], 2: SFI_COLORS["inferred"],
          "1, paper D_r": SFI_COLORS["error"]}
fig_obs, (ax_C, ax_f) = plt.subplots(1, 2, figsize=(12, 4.2))
ax_C.loglog(sides ** 2, C_data, "o-", color=SFI_COLORS["data"], lw=2, label="experiment")
ax_f.plot(lags[lags < len(frames)] * dt, fp_data, color=SFI_COLORS["data"], lw=3, label="experiment")
for Na in runs:
    lab = rf"$N_a = 1$, $D_r = 0.092$" if Na == "1, paper D_r" else f"model $N_a = {Na}$"
    ax_C.loglog(sides ** 2, C_sim[Na], "s--", color=colors[Na], label=lab)
    ax_f.plot(lags[:len(fp_sim[Na])] * dt, fp_sim[Na], "--", color=colors[Na], label=lab)
ax_C.loglog(sides ** 2, C_sim[0][0] * sides[0] / sides, ":", color="#808080", lw=1, label=r"$\propto S^{-1/2}$")
ax_C.set_xlabel(r"$S$ [$\mu$m$^2$]"); ax_C.set_ylabel(r"$C(S)$"); ax_C.set_title("polar order")
ax_f.set_xlabel(r"$\tau$ [s]"); ax_f.set_ylabel(r"$f_p(\tau)$"); ax_f.set_title("polarity autocorrelation")
ax_f.axhline(0, color="#808080", lw=0.8)
for ax in (ax_C, ax_f):
    ax.legend(fontsize=9)
plt.show()

fig_rho, axes_rho = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, rho, lab in [(axes_rho[0], rho_data, "experiment"), (axes_rho[1], rho_sim, r"model $N_a = 2$")]:
    im = ax.imshow(rho.T, origin="lower", extent=[-25 / SIGMA, 25 / SIGMA] * 2, cmap="magma", vmin=0, vmax=1.6)
    ax.add_patch(plt.Circle((0, 0), 0.5, color=SFI_COLORS["inferred"]))
    ax.set_xlabel(r"$x_\parallel/\sigma$ (along the heading)"); ax.set_ylabel(r"$x_\perp/\sigma$"); ax.set_title(lab)
fig_rho.colorbar(im, ax=axes_rho, shrink=0.85, label="density / mean density")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Isotropic interactions alone (:math:`N_a = 0`) do not flock: the order
# of the first frame is lost and :math:`C(S)` falls as :math:`S^{-1/2}`.
# The second-order model reproduces the experiment — the polar order at
# all region sizes, the polarity autocorrelation over the recording, the
# ring of neighbours at :math:`r \approx 2.5\,\sigma`.  The first-order
# model keeps a weaker order with the rotational diffusion measured here
# and none with the paper's larger :math:`D_r`: as in the paper, the
# second angular order is needed.
#
# Comparison with the paper
# -------------------------
#
# With 3 % of the data the qualitative picture of the paper is recovered —
# the self-propulsion speed, a mostly isotropic repulsion, a weak transverse
# velocity, a modulated turning-away torque, and a flocking simulation.
# Where we differ:
#
# * **Localisation noise.**  About two pixels on positions, not "below one
#   pixel".  At this level the paper's Stratonovich estimator overestimates
#   :math:`U` and misses the radial velocity substantially in the
#   bootstrap above; the amplitudes of its Fig. 2 are uncertain to that
#   extent.
# * **Rotational diffusion.**  Measured on the residuals of the fit,
#   :math:`D_r` comes out about a third below the paper's 0.092 s⁻¹.  The
#   conclusion that the second angular order is needed holds at both values,
#   but how much order the first-order model keeps depends on it.
# * **Radial basis.**  The paper's text describes :math:`g_l` with
#   :math:`l = 0 \ldots 7` and :math:`r_0 = 2\,\mu`\ m; its code — and this
#   page — use :math:`l = 0, 2, \ldots, 14` with :math:`r_0 = 1\,\mu`\ m.
#   The code also truncates them at :math:`r_c` while still nonzero, which
#   biases Stratonovich moments; the window here vanishes smoothly.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb, ax_t = plt.subplots(figsize=(4, 4))
ax_t.pcolormesh(TH, TH, (value[..., 2] * scales[2]).T, cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                shading="auto", rasterized=True)
ax_t.set_xticks([]); ax_t.set_yticks([])
for sp in ax_t.spines.values():
    sp.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore
