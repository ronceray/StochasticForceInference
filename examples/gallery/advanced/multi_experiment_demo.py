"""
Multi-experiment ABP inference
===============================

Infer a **shared interaction model** from multiple independent ABP
experiments that differ in both particle number and box size.

Real experimental data often consists of several recordings made
under different conditions — different densities, confinements, or
observation windows.  SFI can handle them natively: each experiment
is a separate :class:`~SFI.trajectory.TrajectoryDataset` carrying
its own periodic box via ``extras_global``, and all datasets are
concatenated into a single :class:`~SFI.trajectory.TrajectoryCollection`
for joint inference.  The inferred force law is global — it must
explain *all* experiments with the same parameters.

.. note::

   This is an **advanced** example: the force is fit with the parametric
   estimator (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`) on a PSF parameterisation.  For
   multi-experiment inference with the linear estimators, see
   ``custom_basis_demo`` in the main gallery; dataset pooling and
   weights are covered in :doc:`/trajectory/user_guide`.

.. rubric:: Tags

synthetic · overdamped · multi-particle · multi-experiment · nonlinear
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "multi-experiment", "nonlinear"]
# sphinx_gallery_thumbnail_number = 4

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.formatting import model_summary
from SFI.utils.plotting import (
    comparison_scatter,
    dark_ax,
    dark_fig,
    plot_particles,
    wrap_positions,
)

apply_style()

# Force black save background (same as single-experiment ABP demo).
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["savefig.transparent"] = False

# -- Marker-scaling constants (not shown in rendered gallery) --

_ARROW_LEN = 1.0   # heading-arrow length in data units
_Lref = 15.0        # reference box side for marker scaling
_s_base = 160       # scatter size (pts²) at reference scale
# sphinx_gallery_end_ignore
# %%
# System: aligning ABPs with varying conditions
# -----------------------------------------------
#
# All experiments share the **same** underlying ABP force law
# (self-propulsion + repulsion + alignment), but differ in:
#
# - **Particle number** *N*
# - **Box size** :math:`L_x \times L_y`
#
# This creates a spectrum from dilute to crowded conditions, which
# stress-tests whether SFI can recover a unique model.

from _gallery_utils.abp import make_abp_align_psf
from SFI.langevin import OverdampedProcess

dt_sim = 0.02
Nsteps = 2000
D_iso = 0.05
seed = 42

# Shared exact force parameters
theta_F_exact = dict(c0=1.0, eps=2.0, A=0.5, R0=1.0, L0=2.0)

# Three experiments: (label, N_particles, Lx, Ly)
experiments = [
    ("Dilute / large box",    10, 30.0, 30.0),
    ("Moderate / medium box", 30, 20.0, 20.0),
    ("Crowded / small box",   60, 15.0, 15.0),
]

F_psf = make_abp_align_psf(dim=3)

# %%
# Simulate each experiment
# -------------------------
#
# Each experiment produces its own trajectory collection with a
# per-dataset ``box`` in ``extras_global``.

collections = []
key = random.PRNGKey(seed)

for label, N, Lx, Ly in experiments:
    box = jnp.array([Lx, Ly])
    key, kx, kth, ksim = random.split(key, 4)
    X0_xy = random.uniform(kx, (N, 2)) * jnp.array([Lx, Ly])
    TH0 = random.uniform(kth, (N,), minval=-jnp.pi, maxval=jnp.pi)
    x0 = jnp.concatenate([X0_xy, TH0[:, None]], axis=1)

    proc = OverdampedProcess(F_psf, D=D_iso, extras_global={"box": box})
    proc.set_params(theta_F=theta_F_exact)
    proc.initialize(x0)
    coll = proc.simulate(dt=dt_sim, Nsteps=Nsteps, key=ksim)
    collections.append(coll)

    density = N / (Lx * Ly)
    print(f"  {label}: N={N}, L={Lx:.0f}×{Ly:.0f}, "
          f"ρ={density:.3f}, frames={coll.T}")

# %%
# Snapshots from each experiment
# --------------------------------
#
# Final-frame snapshots illustrate the different densities and box
# sizes.  Positions are wrapped into the periodic box.

n_experiments = len(experiments)  # panel count for the snapshot grid

# sphinx_gallery_start_ignore
fig, axes = dark_fig(1, 3, figsize=(14, 4.5))

for ax, (label, N, Lx, Ly), coll in zip(axes, experiments, collections):
    s_scaled = _s_base * (_Lref / Lx) ** 2
    plot_particles(coll, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                   quiver=True, heading_dim=2, box=(Lx, Ly), s=s_scaled,
                   edgecolors="white", lw=0.5, zorder=3,
                   quiver_kw=dict(color="white", alpha=0.85, width=0.008,
                                  zorder=4),
                   ax=ax)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    density = N / (Lx * Ly)
    ax.set_title(f"{label}\nN={N}, ρ={density:.3f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle("Three ABP experiments at different conditions", fontsize=13,
             color="white")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Concatenate and infer
# -----------------------
#
# We merge all three collections into one and run a single nonlinear
# (PSF) inference.  SFI handles the per-dataset box automatically via
# ``extras_global``.

from SFI import OverdampedLangevinInference

# Merge trajectory collections with effective-temperature weighting
coll_all = collections[0].merge(collections[1:], weights="pool")

print(f"Combined: {len(coll_all.datasets)} datasets, "
      f"weights = {np.asarray(coll_all.weights).round(3)}")

# Create inference object from the combined collection
inf = OverdampedLangevinInference(coll_all)

# Parametric force inference — the force law is shared across experiments
theta0 = jnp.zeros(F_psf.template.size) + 0.5
inf.infer_force(F_psf, theta0)

inf.compute_force_error()

# %%
# Inference report
# -----------------
#
# The coefficient table now includes **SNR** and a **significance**
# marker.  Significant terms (``|SNR| ≥ 2``) are highlighted.

inf.print_report()

# Compare inferred parameters to the known ground truth
param_cmp = inf.compare_params_to_exact(theta_F_exact, psf=F_psf)
print(model_summary(
    list(param_cmp),
    [float(np.ravel(r["inferred"])[0]) for r in param_cmp.values()],
    coeffs_true=[float(np.ravel(r["true"])[0]) for r in param_cmp.values()],
    title="Parameter comparison",
))

# %%
# Per-experiment validation
# --------------------------
#
# For each experiment, evaluate the inferred force error separately.
# A good global model should explain all conditions, not just the
# average.

# Rebuild the exact force SF to evaluate ground-truth forces
exact_sf = F_psf.bind(theta_F_exact)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (label, N, Lx, Ly), coll in zip(axes, experiments, collections):
    box = jnp.array([Lx, Ly])
    F_exact_vals = np.asarray(exact_sf(coll.X, extras={"box": box}))
    F_inf_vals = np.asarray(inf.force_inferred(coll.X, extras={"box": box}))

    plt.sca(ax)
    comparison_scatter(F_exact_vals, F_inf_vals, maxpoints=50, alpha=0.15,
                       color=SFI_COLORS["inferred"], mode="none")

    # Per-experiment NMSE (bespoke metric kept as the panel title)
    mse = np.mean((F_exact_vals - F_inf_vals) ** 2)
    var = np.mean(F_exact_vals ** 2)
    nmse = mse / var if var > 0 else float("inf")
    ax.set_title(f"{label}\nNMSE = {nmse:.3f}", fontsize=10)

fig.suptitle("Per-experiment force scatter", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Animated multi-experiment snapshots
# -------------------------------------
#
# Side-by-side animation of all three experiments using the **same**
# inferred model — shown by bootstrapping trajectories.

n_frames = Nsteps // max(1, Nsteps // 150)  # frame count for animation

# sphinx_gallery_start_ignore
fig_mov, axes_mov = dark_fig(1, 3, figsize=(14, 4.5))
artists = []

for ax, (label, N, Lx, Ly), coll in zip(axes_mov, experiments, collections):
    s_scaled = _s_base * (_Lref / Lx) ** 2
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    density = N / (Lx * Ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _, Xc, _ = coll.to_arrays(dataset=0)  # (T, N, d) frames via collection API
    Xf0 = wrap_positions(Xc[0], (Lx, Ly))
    sc = ax.scatter(Xf0[:, 0], Xf0[:, 1], s=s_scaled, c=Xf0[:, 2],
                    cmap="hsv", vmin=-np.pi, vmax=np.pi, zorder=3,
                    edgecolors="white", lw=0.5)
    qv = ax.quiver(Xf0[:, 0], Xf0[:, 1],
                   _ARROW_LEN * np.cos(Xf0[:, 2]),
                   _ARROW_LEN * np.sin(Xf0[:, 2]),
                   scale=1.0, scale_units="xy", width=0.008,
                   color="white", alpha=0.85, zorder=4)
    ttl = ax.set_title(f"{label}\nρ={density:.3f}  t=0.0", fontsize=9)
    artists.append((sc, qv, ttl, Xc, label, density, Lx, Ly))

skip = max(1, Nsteps // 150)


def _update_multi(frame):
    idx = frame * skip
    outs = []
    for sc, qv, ttl, Xc, label, density, Lx, Ly in artists:
        Xraw = Xc[min(idx, Xc.shape[0] - 1)]
        Xf = wrap_positions(Xraw, (Lx, Ly))
        sc.set_offsets(Xf[:, :2])
        sc.set_array(Xf[:, 2])
        qv.set_offsets(Xf[:, :2])
        qv.set_UVC(_ARROW_LEN * np.cos(Xf[:, 2]),
                    _ARROW_LEN * np.sin(Xf[:, 2]))
        ttl.set_text(f"{label}\nρ={density:.3f}  t={idx * dt_sim:.1f}")
        outs.extend([sc, qv, ttl])
    return outs


anim_multi = FuncAnimation(
    fig_mov, _update_multi, frames=n_frames, interval=50, blit=True,
)
fig_mov.suptitle("Multi-experiment ABP dynamics", fontsize=13, color="white")
plt.show()
# sphinx_gallery_end_ignore
# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
fig_thumb.patch.set_facecolor("black")
ax_t = fig_thumb.add_subplot(111)
dark_ax(ax_t)
coll_dense = collections[-1]
_, _, Lx_d, Ly_d = experiments[-1]
s_thumb = _s_base * (_Lref / Lx_d) ** 2 * 0.4
plot_particles(coll_dense, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
               quiver=True, heading_dim=2, box=(Lx_d, Ly_d), s=s_thumb,
               edgecolors="white", lw=0.5, zorder=3,
               quiver_kw=dict(color="white", alpha=0.85, width=0.008,
                              zorder=4),
               ax=ax_t)
ax_t.set_xlim(0, Lx_d); ax_t.set_ylim(0, Ly_d)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Going further: experiment-specific parameters
# ---------------------------------------------
#
# Here the force law is fully **shared** — every experiment must be
# explained by the same parameters.  When part of the physics is
# experiment-specific (a per-batch propulsion speed, a per-sample
# temperature), keep the shared terms and add per-dataset parameters
# through the reserved ``dataset_index`` extra (injected automatically
# for every collection):
#
# .. code-block:: python
#
#    from SFI.bases import named_scalar, per_dataset_scalar
#
#    v0 = per_dataset_scalar("v0", n_datasets=len(collections))  # per experiment
#    k  = named_scalar("k_align", default=1.0)                   # shared
#    # ... compose v0 and k into the force model, then inf.infer_force(F)
#
# The parametric estimator fits shared and per-dataset parameters
# jointly (L-BFGS path).  On the linear estimators, the same idea is
# expressed with one-hot features — ``dataset_indicator(n) * feature``
# gives an independent coefficient per experiment.  See the
# multi-experiment section of :doc:`/trajectory/user_guide`.
#
# To **reproduce one experiment** from a pooled fit, bootstrap it with
# ``dataset=k``.  The pooled model is collapsed to that condition (its
# per-dataset parameters folded at ``k`` via ``inf.force_inferred.specialize``)
# and the returned process and trajectory are standalone — they carry no
# ``dataset_index``, so re-inference uses a plain single-condition basis:
#
# .. code-block:: python
#
#    coll_k, proc_k = inf.simulate_bootstrapped_trajectory(key, dataset=k)
#    # proc_k is experiment k's own model; coll_k is a clean single trajectory

stamp_output()
