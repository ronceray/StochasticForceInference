r"""
Aligning active Brownian particles — interaction kernels from noisy tracks
==========================================================================

Infer the **pair interactions** of a population of aligning active
Brownian particles (ABPs) from tracking data — positions *and* headings
corrupted by localisation noise, with a few detections missing — using a
generic, over-complete dictionary of pair kernels from
:mod:`SFI.bases.pairs`.

Nothing about the kernel shapes is assumed: the dictionary contains
exponential–polynomial kernels of several ranges, :term:`PASTIS` keeps
the ones the data support, and the recovered kernels come with
confidence bands.  The inferred model is then simulated from the first
recorded frame and played side by side with the recording, and its
collective order — an emergent statistic the fit never used — is
compared with the data's.

.. rubric:: Tags

synthetic · overdamped · multi-particle · linear · sparsity
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "linear", "sparsity"]
# sphinx_gallery_thumbnail_number = 6

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.abp import make_abp_align_psf
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output, plot_kernel_ci
from SFI.utils.plotting import (
    animate_particles_comparison,
    dark_fig,
    plot_pareto_front,
    plot_particles,
    plot_recovery_bar,
)

apply_style()

# Movies are encoded on the figure's own (black) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"

# Print library warnings without the local source path of this script.
_formatwarning = warnings.formatwarning
warnings.formatwarning = lambda msg, cat, *args, **kw: f"{cat.__name__}: {msg}\n"


def _pin_arrows(axes, length=1.25, width=0.12):
    """Heading arrows of fixed length (box units) drawn under the particles."""
    from matplotlib.quiver import Quiver

    for ax in np.atleast_1d(axes):
        for art in ax.collections:
            if isinstance(art, Quiver):
                art.scale, art.width = 1.0 / length, width
                art.set_facecolor("white")
                art.set_zorder(1)
            else:
                art.set_zorder(2)
# sphinx_gallery_end_ignore
stamp_output()

# %%
# System: aligning active Brownian particles
# ------------------------------------------
#
# Each particle carries a position :math:`(x, y)` and a heading
# :math:`\theta`.  It swims at speed :math:`c_0` along
# :math:`\hat e_\theta`, is pushed away from close neighbours by a soft
# repulsion :math:`-\varepsilon/(1 + r^2/R_0^2)\,\hat r_{ij}`, and turns
# toward their headings with a torque
# :math:`A\,e^{-r/L_0}\sin(\theta_j - \theta_i)`.  Positions diffuse
# weakly, headings more strongly (rotational noise :math:`D_\theta`), in
# a periodic box.
#
# The rotational noise is chosen close to the onset of collective order:
# the population is neither disordered nor a rigid flock — its global
# alignment waxes and wanes as local groups align and lose their
# common heading.  This is also the regime where the alignment
# interaction is informative: in a fully polarised flock
# :math:`\sin(\theta_j - \theta_i) \approx 0` for every pair and the
# torque cannot be measured.

from SFI.langevin import OverdampedProcess

N_particles = 60
L = 30.0                     # periodic box side
box = jnp.array([L, L])
dt = 0.02
Nsteps = 5000                # 100 time units ≈ 30 rotational decorrelation times
D_xy, D_theta = 0.05, 0.3    # translational / rotational diffusion
theta_true = dict(c0=1.0, eps=2.0, R0=1.0, A=0.5, L0=2.0)

proc = OverdampedProcess(
    make_abp_align_psf(dim=3),
    D=jnp.diag(jnp.array([D_xy, D_xy, D_theta])),
    extras_global={"box": box},
)
proc.set_params(theta_F=theta_true)

key, kx, kth, ksim = random.split(random.PRNGKey(0), 4)
x0 = jnp.concatenate(
    [random.uniform(kx, (N_particles, 2)) * L,
     random.uniform(kth, (N_particles, 1), minval=-jnp.pi, maxval=jnp.pi)],
    axis=1,
)
proc.initialize(x0)
coll_true = proc.simulate(dt=dt, Nsteps=Nsteps, key=ksim)


def polar_order(coll):
    """Polar order parameter |<exp(i theta)>| per frame."""
    theta = np.asarray(coll.to_array())[..., 2]
    return np.abs(np.exp(1j * theta).mean(axis=1))


phi_true = polar_order(coll_true)
t_frames = np.arange(len(phi_true)) * dt
print(f"{coll_true.T} frames, {coll_true.N} particles; "
      f"polar order ranges over [{phi_true.min():.2f}, {phi_true.max():.2f}]")

# %%
# The polar order parameter :math:`\varphi = |\langle e^{i\theta}\rangle|`
# keeps fluctuating instead of settling.  The snapshot shows the most
# ordered frame (dot on the trace), with particles coloured by heading: a
# dominant direction is visible, but far from unanimous.

# sphinx_gallery_start_ignore
fig = plt.figure(figsize=(11, 4.2))
fig.patch.set_facecolor("black")
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
ax_phi = fig.add_subplot(gs[0])
ax_phi.plot(t_frames, phi_true, lw=0.8, color=SFI_COLORS["data"])
ax_phi.set_xlabel("time")
ax_phi.set_ylabel(r"polar order $\varphi$")
ax_phi.set_ylim(0, 1)
ax_phi.set_title("Collective order fluctuates")
i_snap = int(np.argmax(phi_true))
ax_phi.plot(t_frames[i_snap], phi_true[i_snap], "o", ms=7, color=SFI_COLORS["inferred"])
ax_snap = fig.add_subplot(gs[1])
plot_particles(coll_true, t_index=i_snap, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
               quiver=True, heading_dim=2, box=(L, L), s=60, ax=ax_snap, zorder=2,
               quiver_kw=dict(color="white", scale=0.8, width=0.12, zorder=1))
ax_snap.set_xlim(0, L); ax_snap.set_ylim(0, L)
ax_snap.set_title(rf"t = {t_frames[i_snap]:.0f}, $\varphi$ = {phi_true[i_snap]:.2f} (N = {N_particles})",
                  color="white")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Corrupt the recording
# ---------------------
#
# Real tracking adds **localisation noise** to every coordinate — here
# :math:`\sigma = 0.05` on positions and on headings, comparable to the
# displacement in one frame — and **misses detections**: 2 % of the
# particle positions are dropped at random.  The masked detections are
# simply absent from the data; SFI skips every increment and feature that
# would need them.

sigma = 0.05
coll = coll_true.degrade(noise=sigma, data_loss_fraction=0.02, seed=1)

# %%
# An over-complete dictionary of pair kernels
# -------------------------------------------
#
# Three families of features, all built from :mod:`SFI.bases.pairs`:
#
# 1. **self-propulsion** — the heading vector :math:`\hat e_\theta`
#    (:func:`~SFI.bases.pairs.heading_vector`);
# 2. **radial forces** — :math:`\phi_k(r)\,\hat r_{ij}` summed over
#    neighbours (:func:`~SFI.bases.pairs.radial_pair_basis`);
# 3. **alignment torques** — :math:`\phi_k(r)\sin(\theta_j-\theta_i)`
#    on the heading (:func:`~SFI.bases.pairs.angular_pair_basis`).
#
# The radial kernels :math:`\phi_k(r) = r^n e^{-r/\ell}` span
# :math:`n \in \{0, 1\}` and :math:`\ell \in \{0.5, 1, 2, 4\}` — 17
# candidate features in total.  The true repulsion (a Lorentzian) is
# *not* in the dictionary: the kernels can only approximate it, as with
# real data.

from SFI.bases.pairs import (
    angular_pair_basis,
    exp_poly_kernels,
    heading_vector,
    radial_pair_basis,
)
from SFI.statefunc import Basis

dim = 3  # (x, y, θ)
kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
pair_kw = dict(symmetric=True, exclude_self=True, owners="focal",
               reducer="sum", return_as="basis")

B_heading = heading_vector(dim=dim, angle_index=2)
B_repel = radial_pair_basis(
    kernels, dim=dim, box="extras", spatial_dims=slice(0, 2),
    embed_dim=dim, embed_axes=[0, 1],
).dispatch_pairs(**pair_kw)
B_align = angular_pair_basis(
    kernels, jnp.sin, dim=dim, angle_index=2, output_index=2,
    box="extras", spatial_dims=slice(0, 2),
).dispatch_pairs(**pair_kw)
B = Basis.stack([B_heading, B_repel, B_align])

# %%
# Inference and model selection
# -----------------------------
#
# A single linear solve, then PASTIS.  With noisy tracks the
# localisation error of frame :math:`t` enters both the increment
# :math:`\Delta x_t` and the features evaluated at :math:`x_t`, and that
# correlation biases a plain regression.  The default settings measure
# the localisation noise and, finding it, switch to the noise-robust
# estimators (Stratonovich moments and a shifted Gram), in which the
# noise realisations of features and increments cancel or are
# independent.
#
# SFI also warns that the full dictionary is ill-conditioned.  That is
# expected of an over-complete dictionary — kernels of neighbouring
# ranges overlap so much that the data cannot tell all 17 coefficients
# apart — and it is exactly what model selection resolves: the few
# kernels PASTIS keeps are well separated, and their sampling error bars
# are well defined.

from SFI import OverdampedLangevinInference

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()                 # noise-robust estimator (auto)
inf.infer_force_linear(B)                        # preset="auto"
print(f"localisation noise detected -> preset {inf.metadata['force_preset']!r}")
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, data_exact=coll_true, maxpoints=2000)
inf.print_report()

print(f"\nNMSE(force): predicted {inf.force_predicted_MSE:.1e}, "
      f"realised (vs ground truth) {float(inf.NMSE_force):.1e}")
print(f"D_xx = {float(inf.diffusion_average[0, 0]):.3f} (true {D_xy}),  "
      f"D_θθ = {float(inf.diffusion_average[2, 2]):.3f} (true {D_theta})")

# %%
# The Pareto front shows the information gained as terms are added: it
# rises steeply for the first few features and then plateaus, and PASTIS
# stops at the plateau.  The selected model keeps the self-propulsion,
# a few radial kernels that together reproduce the repulsion, and a
# single alignment kernel.  The realised error (known here because the
# data are synthetic) is about an order of magnitude above the predicted
# sampling error: the remainder is the bias left by the corrupted
# recording and by the approximate repulsion shape.
#
# The positional diffusion :math:`D_{xx}` also comes out high.  The
# noise-robust estimator of :math:`D` multiplies consecutive increments,
# and a persistent drift adds :math:`\tfrac32 \langle F_x^2\rangle\Delta t
# = \tfrac34 c_0^2\Delta t \approx 0.015` to it — an
# :math:`O(\Delta t)` bias that only matters because the swimming is
# strong compared with the weak positional noise.  It is harmless here:
# the particles' spreading is dominated by swimming (effective diffusivity
# :math:`c_0^2/2D_\theta \approx 1.7`), and the rotational noise that
# controls the order is recovered accurately.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.4]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
plot_recovery_bar(np.asarray(inf.force_coefficients), np.asarray(inf.force_support),
                  labels=list(B.labels), stderr=np.asarray(inf.force_coefficients_stderr),
                  ax=axes[1])
axes[1].tick_params(axis="x", labelsize=7)
axes[1].set_title(f"Selected: {len(inf.force_support)} / {B.n_features} features")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Recovered interaction kernels
# -----------------------------
#
# Summing the selected kernels, :math:`\sum_k c_k \phi_k(r)`, gives the
# radial force and the alignment strength as functions of distance, with
# 95 % confidence bands from the coefficient covariance
# (:func:`~SFI.inference.kernel_predict_ci`).  The grey zone marks
# distances where fewer than 1 % of the observed pairs lie: repulsion
# keeps particles apart, so the data say little there and the
# reconstruction is an extrapolation.

from SFI.inference import kernel_predict_ci

# Per-block coefficients and covariance (pruned terms are exact zeros).
c_heading, _ = inf.coeff_block(B_heading)
c_repel, cov_repel = inf.coeff_block(B_repel)
c_align, cov_align = inf.coeff_block(B_align)
print(f"self-propulsion c0 = {float(np.ravel(c_heading)[0]):.3f}  (true {theta_true['c0']})")

r_eval = np.linspace(0.05, 8.0, 200)
ci_repel = kernel_predict_ci(r_eval, kernels, c_repel, cov_repel)
ci_align = kernel_predict_ci(r_eval, kernels, c_align, cov_align)
true_repel = -theta_true["eps"] / (1 + (r_eval / theta_true["R0"]) ** 2)
true_align = theta_true["A"] * np.exp(-r_eval / theta_true["L0"])

# Pair distances actually sampled (minimum image), from a subset of frames.
Xs = np.asarray(coll.to_array())[::50, :, :2]
dxy = Xs[:, :, None, :] - Xs[:, None, :, :]
dxy -= L * np.round(dxy / L)
r_pairs = np.linalg.norm(dxy, axis=-1)[:, ~np.eye(N_particles, dtype=bool)]
r_low = float(np.percentile(r_pairs[r_pairs < 8.0], 1))

# Alignment strength at a typical neighbour distance, with its 95 % error.
a2 = kernel_predict_ci(np.array([2.0]), kernels, c_align, cov_align)
a2_true = theta_true["A"] * np.exp(-2.0 / theta_true["L0"])
print(f"alignment strength at r = 2: {a2['mean'][0]:.3f} ± {1.96 * a2['std'][0]:.3f} "
      f"(true {a2_true:.3f})")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, ci, true_k, title, ylabel in [
    (axes[0], ci_repel, true_repel, "Radial force", r"$f(r)$ (negative = repulsive)"),
    (axes[1], ci_align, true_align, "Alignment strength", r"$a(r)$"),
]:
    plot_kernel_ci(ax, r_eval, true_k, ci, scale=1.0, ci_label="95% CI")
    ax.axvspan(0, r_low, color="#808080", alpha=0.25, lw=0, label="unsampled (<1% of pairs)")
    ax.set_xlim(0, r_eval[-1])
    ax.set_xlabel("pair distance r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Interaction kernels: truth vs inferred", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Where pairs are actually sampled, the radial force is reproduced by
# kernels that are individually quite unlike the Lorentzian.  The
# alignment kernel :math:`e^{-r/2}` is in the dictionary and PASTIS picks
# exactly it, but with an amplitude above the truth by more than its
# error bar: like the realised force error above, this is a bias left by
# the corrupted recording, which the sampling error bars do not include.
#
# Bootstrap: does the inferred model reproduce the collective behaviour?
# ----------------------------------------------------------------------
#
# The fit only used single-step increments.  The polar order, its
# fluctuations and its time scale are emergent, many-body properties —
# a genuine prediction.  We simulate the inferred model (force *and*
# diffusion) from the **first recorded frame**, for as long as the
# recording, in a few independent replicas.

proc_boot = inf.simulate_bootstrapped_trajectory(random.PRNGKey(7), simulate=False)
x0_rec = jnp.asarray(coll.to_array()[0])         # first recorded frame
boots = []
for seed in range(4):
    proc_boot.initialize(x0_rec)
    boots.append(proc_boot.simulate(dt=dt, Nsteps=coll.T, key=random.PRNGKey(100 + seed),
                                    compute_observables=False))  # positions only: saves memory

# %%
# Side by side, the recording (left; missed detections blink out) and the
# first replica (right) start from the same configuration and soon part
# ways — individual trajectories are random, so a bootstrap can only be
# compared *statistically*.  What should match is the collective
# behaviour: small groups that align, merge, turn together and break up.

# sphinx_gallery_start_ignore
fig_m, axes_m = dark_fig(1, 2, figsize=(6.6, 3.7))
anim = animate_particles_comparison(
    [coll, boots[0]], titles=["recording", "inferred model"],
    color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi, quiver=True, heading_dim=2,
    box=(L, L), s=34, trail=40, skip=25, interval=80, axes=axes_m,
)
_pin_arrows(axes_m, length=1.25, width=0.14)
for ax in axes_m:
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# To make "alike" quantitative, compare the polar order of the data with
# that of the replicas.  A single run of 100 time units contains only a
# few order–disorder excursions, so replicas differ visibly from one
# another.  Because the data are synthetic, we can also run replicas of
# the *true* model from the same start: they show how much the order of
# the recording itself could have fluctuated.

true_reps = []
for seed in range(4):
    proc.initialize(x0)
    true_reps.append(proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(200 + seed),
                                   compute_observables=False))

half = len(phi_true) // 2  # second halves: after the initial ordering transient
phi_data = polar_order(coll)
phi_boot = np.array([polar_order(c) for c in boots])
phi_ref = np.array([polar_order(c) for c in true_reps])
m_ref, m_boot = phi_ref[:, half:].mean(axis=1), phi_boot[:, half:].mean(axis=1)
print(f"mean polar order (second halves): data {phi_data[half:].mean():.2f}\n"
      f"  true-model replicas      {np.round(m_ref, 2)}  (average {m_ref.mean():.2f})\n"
      f"  inferred-model replicas  {np.round(m_boot, 2)}  (average {m_boot.mean():.2f})")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
for i, ph in enumerate(phi_boot):
    axes[0].plot(t_frames, ph, lw=1.1 if i == 0 else 0.7, alpha=1.0 if i == 0 else 0.6,
                 color=SFI_COLORS["tertiary"],
                 label="replica in the movie" if i == 0 else ("other replicas" if i == 1 else None))
axes[0].plot(t_frames, phi_data, lw=1.1, color=SFI_COLORS["data"], label="data")
axes[0].set_xlabel("time"); axes[0].set_ylabel(r"polar order $\varphi$")
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8, ncol=3, loc="upper left")
bins = np.linspace(0, 1, 26)
axes[1].hist(phi_data[half:], bins=bins, density=True, alpha=0.5,
             color=SFI_COLORS["data"], label="data")
axes[1].hist(phi_ref[:, half:].ravel(), bins=bins, density=True, histtype="step", lw=1.4,
             ls="--", color="#B0B0B0", label="true model (pooled)")
axes[1].hist(phi_boot[:, half:].ravel(), bins=bins, density=True, histtype="step", lw=1.6,
             color=SFI_COLORS["tertiary"], label="inferred model (pooled)")
axes[1].set_xlabel(r"$\varphi$ (second halves)"); axes[1].legend(fontsize=8)
fig.suptitle("Emergent order: data vs inferred model", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The verdict is instructive.  The inferred model reproduces the
# phenomenology — partial, fluctuating order of the right size and time
# scale — but its replicas are on average more ordered than the data,
# whereas the replicas of the true model bracket it.  Close to the onset
# of collective motion, the order is a steep function of the ratio of
# alignment to rotational noise, so it amplifies the small excess of
# inferred alignment found above.  An emergent statistic near a
# transition is a far more stringent test than the single-step fit —
# which is exactly why a bootstrap is worth running on real data, where
# no ground truth is available to reveal such a bias.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb, ax_t = dark_fig(figsize=(4, 4))
plot_particles(coll_true, t_index=i_snap, color_dim=2, cmap="hsv",
               vmin=-np.pi, vmax=np.pi, quiver=True, heading_dim=2, box=(L, L), s=90, ax=ax_t,
               zorder=2, quiver_kw=dict(color="white", scale=0.7, width=0.16, zorder=1))
ax_t.set_xlim(0, L); ax_t.set_ylim(0, L)
plt.show()
warnings.formatwarning = _formatwarning
# sphinx_gallery_end_ignore
