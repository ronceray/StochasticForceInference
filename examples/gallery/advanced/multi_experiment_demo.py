r"""
Multi-experiment ABP inference — shared interactions, per-condition motility
=============================================================================

Infer **one interaction model** from several recordings made under
different conditions — here three active-particle experiments at
different densities and box sizes, whose particles also swim at
**different speeds** (think of different temperatures, fuel or
illumination levels).

Each recording is a separate :class:`~SFI.trajectory.TrajectoryDataset`
with its own periodic box in ``extras_global``; the datasets are merged
into one :class:`~SFI.trajectory.TrajectoryCollection` and fitted
jointly.  The interaction kernels are **shared** — they must explain all
three experiments at once — while the self-propulsion speed gets **one
coefficient per experiment** through
:func:`~SFI.bases.dataset_indicator`.  Everything stays linear in the
coefficients, so the fit is a single linear solve followed by
:term:`PASTIS`.  The pooled model is then validated experiment by
experiment, and each recording is replayed side by side with a
simulation of the model collapsed to that condition.

.. note::

   For per-experiment parameters that enter *nonlinearly* (a
   per-condition interaction range, say), the same idea is expressed with
   :func:`~SFI.bases.per_dataset_scalar` inside a parametric model fitted
   by :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`.
   Dataset pooling and weights are covered in :doc:`/trajectory/user_guide`;
   the single-experiment version of this system, with noisy tracks, is
   :doc:`/gallery/abp_align_demo`.

.. rubric:: Tags

synthetic · overdamped · multi-particle · multi-experiment · linear · sparsity
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "multi-experiment", "linear", "sparsity"]
# sphinx_gallery_thumbnail_number = 6

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gallery_utils.abp import make_abp_align_psf
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import (
    animate_particles_comparison,
    comparison_scatter,
    dark_fig,
    plot_particles,
    plot_pareto_front,
)

apply_style()

# Movies are encoded on the figure's own (black) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"

# Print library warnings without the local source path of this script.
_formatwarning = warnings.formatwarning
warnings.formatwarning = lambda msg, cat, *args, **kw: f"{cat.__name__}: {msg}\n"


def _style_particles(ax, L, *, diameter=0.8, arrow=1.25, width=0.14):
    """Same length scale in every box: marker diameter and arrow size in box units."""
    from matplotlib.quiver import Quiver

    pt_per_unit = ax.get_window_extent().width * 72 / ax.figure.dpi / L
    for art in ax.collections:
        if isinstance(art, Quiver):
            art.scale, art.width = 1.0 / arrow, width
            art.set_facecolor("white")
            art.set_zorder(1)
        else:
            art.set_sizes([(diameter * pt_per_unit) ** 2])
            art.set_zorder(2)
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Three experiments, one interaction law
# --------------------------------------
#
# All experiments share the same aligning-ABP interactions — soft
# repulsion and short-range alignment (see :doc:`/gallery/abp_align_demo`)
# — but differ in particle number, box size and swimming speed
# :math:`c_0`.  The same interactions then produce three different
# collective states — from a disordered gas to a polar flock — which is
# exactly why pooling them constrains the interactions better than any
# single recording.

from SFI.langevin import OverdampedProcess

dt = 0.02
Nsteps = 2000
D = jnp.diag(jnp.array([0.05, 0.05, 0.3]))        # translational / rotational noise
shared = dict(eps=2.0, R0=1.0, A=0.5, L0=2.0)      # interaction parameters (shared)

# (label, N particles, box side, swimming speed c0)
experiments = [
    ("dilute",   10, 30.0, 0.8),
    ("moderate", 30, 20.0, 1.0),
    ("crowded",  60, 15.0, 1.3),
]

collections, processes = [], []
key = random.PRNGKey(42)
for label, N, L, c0 in experiments:
    key, kx, kth, ksim = random.split(key, 4)
    x0 = jnp.concatenate(
        [random.uniform(kx, (N, 2)) * L,
         random.uniform(kth, (N, 1), minval=-jnp.pi, maxval=jnp.pi)],
        axis=1,
    )
    proc = OverdampedProcess(make_abp_align_psf(dim=3), D=D,
                             extras_global={"box": jnp.array([L, L])})
    proc.set_params(theta_F=dict(c0=c0, **shared))
    proc.initialize(x0)
    collections.append(proc.simulate(dt=dt, Nsteps=Nsteps, key=ksim))
    processes.append(proc)
    print(f"{label:>8}: N={N:2d}, box {L:.0f}×{L:.0f}, density {N / L**2:.3f}, c0={c0}")

# sphinx_gallery_start_ignore
fig, axes = dark_fig(1, 3, figsize=(12, 4.4))
for ax, (label, N, L, c0), coll_k in zip(axes, experiments, collections):
    plot_particles(coll_k, t_index=-1, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                   quiver=True, heading_dim=2, box=(L, L), ax=ax)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_title(f"{label}: N = {N}, box {L:.0f}, $c_0$ = {c0}", color="white", fontsize=10)
fig.canvas.draw()
for ax, (_, _, L, _) in zip(axes, experiments):
    _style_particles(ax, L)
fig.suptitle("Final frames (colour and arrows = heading; same length scale in every box)",
             fontsize=12, color="white")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Shared kernels, per-experiment speed
# ------------------------------------
#
# The dictionary is that of the single-experiment demo, with one change:
# the heading feature is multiplied by
# :func:`~SFI.bases.dataset_indicator`, a one-hot vector that reads the
# dataset index the collection injects automatically.  The product gives
# three self-propulsion features, each active in one experiment only, so
# each experiment gets its own fitted speed; the 16 pair-kernel features
# are shared.

from SFI.bases import dataset_indicator
from SFI.bases.pairs import angular_pair_basis, exp_poly_kernels, heading_vector, radial_pair_basis
from SFI.statefunc import Basis

dim = 3
kernels = exp_poly_kernels(degrees=[0, 1], lengths=[0.5, 1.0, 2.0, 4.0])
pair_kw = dict(symmetric=True, exclude_self=True, owners="focal",
               reducer="sum", return_as="basis")

B_speed = dataset_indicator(len(experiments), dim=dim) * heading_vector(dim=dim, angle_index=2)
B_repel = radial_pair_basis(
    kernels, dim=dim, box="extras", spatial_dims=slice(0, 2),
    embed_dim=dim, embed_axes=[0, 1],
).dispatch_pairs(**pair_kw)
B_align = angular_pair_basis(
    kernels, jnp.sin, dim=dim, angle_index=2, output_index=2,
    box="extras", spatial_dims=slice(0, 2),
).dispatch_pairs(**pair_kw)
B = Basis.stack([B_speed, B_repel, B_align])
print(f"{B.n_features} candidate features: {B_speed.n_features} per-experiment, "
      f"{B_repel.n_features + B_align.n_features} shared")

# %%
# Joint inference
# ---------------
#
# Merge the recordings — each keeps its own box — and fit.  With
# ``weights="pool"`` every frame of every experiment counts equally.  The
# simulated tracks carry no localisation noise, so we ask for the
# ``"clean"`` estimators directly rather than letting the default settings
# test for noise, and estimate :math:`D` with the ``"WeakNoise"`` stencil,
# which is insensitive to the strong swimming drift.
#
# As in the single-experiment demo, the full, deliberately over-complete
# dictionary is ill-conditioned — see the force Gram condition number in
# the report below (neighbouring kernels overlap); PASTIS resolves it by
# keeping a few well-separated kernels.

from SFI import OverdampedLangevinInference

coll = collections[0].merge(collections[1:], weights="pool")

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")   # clean data, strong drift
inf.infer_force_linear(B, preset="clean")            # noise-free simulated tracks
inf.compute_force_error()
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# sphinx_gallery_start_ignore
coeffs = np.asarray(inf.force_coefficients_full)
support = np.asarray(inf.force_support)
stderr_full = np.zeros(B.n_features)
stderr_full[support] = np.asarray(inf.force_coefficients_stderr)
c0_true = np.array([c0 for *_, c0 in experiments])

fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
xk = np.arange(3)
axes[1].axhline(0.0, color="#808080", lw=1)
axes[1].errorbar(xk, coeffs[:3] - c0_true, yerr=1.96 * stderr_full[:3], fmt="o", ms=8,
                 capsize=6, color=SFI_COLORS["inferred"], ecolor="#B0B0B0", label="inferred ± 95% CI")
for x_, c_hat, se, c_true in zip(xk, coeffs[:3], stderr_full[:3], c0_true):
    axes[1].annotate(f"{c_hat:.3f} ± {se:.3f}\n(true {c_true})", (x_, c_hat - c_true),
                     xytext=(12, 0), textcoords="offset points", va="center", fontsize=8,
                     color="#D0D0D0")
axes[1].set_xticks(xk, [label for label, *_ in experiments])
axes[1].set_xlim(-0.5, 2.9)
lim = 1.2 * float(np.max(np.abs(coeffs[:3] - c0_true) + 1.96 * stderr_full[:3]))
axes[1].set_ylim(-lim, lim)
axes[1].set_ylabel(r"$\hat c_0 - c_0$")
axes[1].set_title("Per-experiment swimming speed: error")
axes[1].legend(fontsize=8, loc="lower right")
plt.show()
# sphinx_gallery_end_ignore

# %%
# PASTIS keeps the three speeds, a couple of radial kernels and a single
# alignment kernel — the true :math:`e^{-r/2}`, with an amplitude within
# its error bar of the truth (table above).  Each experiment's speed is
# recovered within its error bar too: the shared interactions are pinned
# down by all three recordings at once, which is what lets the speeds be
# separated cleanly.
#
# Per-experiment validation
# -------------------------
#
# A pooled fit could explain the average while failing one condition.
# Evaluating the inferred force against the truth separately in each
# experiment rules this out.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
# sphinx_gallery_end_ignore
for k, ((label, N, L, c0), coll_k, proc_k) in enumerate(zip(experiments, collections, processes)):
    X_k = jnp.asarray(coll_k.to_array())[::10]
    extras_k = {"box": jnp.array([L, L]), "dataset_index": k}
    F_exact = np.asarray(proc_k.force_sf(X_k, extras=extras_k))
    F_inf = np.asarray(inf.force_inferred(X_k, extras=extras_k))
    nmse = np.mean((F_exact - F_inf) ** 2) / np.mean(F_exact ** 2)
    print(f"{label:>8}: NMSE(force) = {nmse:.4f}")
    # sphinx_gallery_start_ignore
    plt.sca(axes[k])
    comparison_scatter(F_exact, F_inf, maxpoints=3000, alpha=0.25,
                       color=SFI_COLORS["inferred"], mode="none")
    for ln in axes[k].lines:  # thin diagonal under the points
        ln.set_zorder(0); ln.set_lw(0.8)
    axes[k].set_xlabel("exact force component"); axes[k].set_ylabel("inferred")
    axes[k].set_title(f"{label}: NMSE = {nmse:.4f}", fontsize=10)
    # sphinx_gallery_end_ignore
# sphinx_gallery_start_ignore
fig.suptitle("Force recovery in each experiment", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Reproducing each experiment from the pooled fit
# -----------------------------------------------
#
# ``simulate_bootstrapped_trajectory(..., dataset=k)`` collapses the
# pooled model to condition :math:`k` — its own swimming speed and its
# box — and simulates it from that experiment's first frame, for as long
# as the recording.

n_replicas = 3
boots = [
    [inf.simulate_bootstrapped_trajectory(random.PRNGKey(10 * k + r), dataset=k)[0]
     for r in range(n_replicas)]
    for k in range(len(experiments))
]

# %%
# Top row: the three recordings; bottom row: the inferred model (the
# first replica of each condition), started from the same frames.
# Individual trajectories part ways quickly — the dynamics is
# stochastic — but the collective states match: independent swimmers in
# the dilute box, transient aligned groups at moderate density, and in
# the crowded box a polar flock that forms within a few time units.  The
# two crowded flocks need not travel the same way: the direction of a
# flock is a spontaneously broken symmetry, chosen at random as it forms.
# All boxes are drawn on the same length scale.

# sphinx_gallery_start_ignore
fig_m, axes_m = dark_fig(2, 3, figsize=(6.6, 4.7))
boxes = [(L, L) for _, _, L, _ in experiments]
anim = animate_particles_comparison(
    collections + [b[0] for b in boots], axes=axes_m.ravel(), box=boxes * 2,
    titles=[f"{label}, $c_0$ = {c0}" for label, _, _, c0 in experiments] + [""] * 3,
    color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi, quiver=True, heading_dim=2,
    trail=30, skip=10, interval=80,
)
for ax in axes_m.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    ax.title.set_fontsize(9)
axes_m[0, 0].set_ylabel("recording", fontsize=9)
axes_m[1, 0].set_ylabel("inferred model", fontsize=9)
fig_m.canvas.draw()
for row in axes_m:
    for ax, (L, _) in zip(row, boxes):
        _style_particles(ax, L)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The polar order :math:`\varphi = |\langle e^{i\theta}\rangle|`, an
# emergent quantity the fit never used, makes the comparison
# quantitative.  For reference, :math:`N` independent random headings
# give :math:`\varphi \approx \sqrt{\pi/4N}` by chance alone.
#
# * **Dilute**: data and replicas sit at that chance level — no collective
#   order.
# * **Crowded**: a stable flock, with the same order in the data and in
#   every replica.
# * **Moderate**: order well above chance, but it comes and goes on long
#   time scales, so one run of 40 time units is a small sample: the
#   replicas' mean orders spread widely, and the data lie just above the
#   top of that spread.  Only the statistics of many runs could tell a
#   genuine discrepancy here.


def polar_order(c):
    theta = np.asarray(c.to_array())[..., 2]
    return np.abs(np.exp(1j * theta).mean(axis=1))


t_axis = np.arange(Nsteps) * dt
# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
# sphinx_gallery_end_ignore
for k, (label, N, _, _) in enumerate(experiments):
    phi_d = polar_order(collections[k])
    phi_b = [polar_order(c) for c in boots[k]]
    print(f"{label:>8}: mean polar order, data {phi_d.mean():.2f}, "
          f"bootstrap replicas {np.round([p.mean() for p in phi_b], 2)}  "
          f"(random headings: {np.sqrt(np.pi / (4 * N)):.2f})")
    # sphinx_gallery_start_ignore
    for r, p in enumerate(phi_b):
        axes[k].plot(t_axis[:len(p)], p, lw=1.0 if r == 0 else 0.6, alpha=1.0 if r == 0 else 0.6,
                     color=SFI_COLORS["tertiary"],
                     label="replica in the movie" if r == 0 else ("other replicas" if r == 1 else None))
    axes[k].plot(t_axis[:len(phi_d)], phi_d, lw=1.0, color=SFI_COLORS["data"], label="data")
    axes[k].axhline(np.sqrt(np.pi / (4 * N)), color="#B0B0B0", ls=":", lw=1.2,
                    label=r"random headings, $\sqrt{\pi/4N}$")
    axes[k].set_title(f"{label} (N = {N})", fontsize=10); axes[k].set_xlabel("time")
    # sphinx_gallery_end_ignore
# sphinx_gallery_start_ignore
axes[0].set_ylabel(r"polar order $\varphi$"); axes[0].set_ylim(0, 1)
axes[2].legend(fontsize=8, loc="lower right")
fig.suptitle("Emergent order: each experiment vs bootstraps of the pooled model", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb, ax_t = dark_fig(figsize=(4, 4))
L_c = experiments[2][2]  # crowded: a polar flock
plot_particles(collections[2], t_index=-1, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
               quiver=True, heading_dim=2, box=(L_c, L_c), ax=ax_t)
ax_t.set_xlim(0, L_c); ax_t.set_ylim(0, L_c)
fig_thumb.canvas.draw()
_style_particles(ax_t, L_c, diameter=0.9, arrow=1.4, width=0.18)
plt.show()
warnings.formatwarning = _formatwarning
# sphinx_gallery_end_ignore
