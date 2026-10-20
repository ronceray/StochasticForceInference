r"""
Velocity-dependent noise — underdamped multiplicative diffusion
=================================================================

Recover a **velocity-dependent diffusion field** :math:`D(v)` for an
inertial particle, from positions only.  Many active and driven systems
fluctuate harder the faster they move — motility noise, flight-force
fluctuations, turbulent drag.  Underdamped SFI reconstructs the
unobserved velocity and infers :math:`F(x,v)` and :math:`D(x,v)`
jointly; here the noise amplitude grows several-fold across the
explored speeds.  We fit :math:`D` on a generic polynomial basis, let PASTIS pick
the noise model, cross-check with the likelihood-based estimator, and
close on the regime where velocity-dependent noise becomes genuinely
hard.

.. rubric:: Tags

synthetic · underdamped · multiplicative-noise · sparsity · 1D
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "multiplicative-noise", "sparsity", "1D"]
# sphinx_gallery_thumbnail_number = 6

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import phase2d_scalar, plot_pareto_front, plot_profile_1d

apply_style()


def ratio_panel(ax, fits, labels, truth):
    """Selected coefficients / truth with ±2σ bars, one marker series per fit."""
    true_idx = [labels.index(k) for k in truth]
    shown = sorted(set(true_idx).union(*[set(int(i) for i in f[1]) for f in fits]))
    ax.axhline(1.0, color="#808080", lw=0.8, ls="--")
    offsets = np.linspace(-0.12, 0.12, len(fits)) if len(fits) > 1 else [0.0]
    for (name, support, coeffs, stderr, color), off in zip(fits, offsets):
        val = {int(i): c for i, c in zip(support, coeffs)}
        err = {int(i): e for i, e in zip(support, stderr)}
        first = True
        for j, i in enumerate(shown):
            if i in true_idx:
                t = truth[labels[i]]
                ax.errorbar(j + off, val.get(i, 0.0) / t, yerr=2 * err.get(i, 0.0) / abs(t),
                            fmt="o", ms=6, capsize=4, color=color, ecolor=color,
                            label=name if first else None)
                first = False
            elif i in val:  # selected but absent from the truth
                ax.plot(j + off, 0.0, "x", ms=8, color=SFI_COLORS["error"])
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([labels[i] for i in shown])
    ax.set_ylabel("inferred / true")
    ax.set_title("Selected diffusion coefficients (±2σ)")
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Model and simulation
# ---------------------
#
# A damped harmonic oscillator whose bath kicks grow with speed:
#
# .. math::
#
#    \dot x = v, \qquad
#    dv = (-k x - \gamma v)\,dt + \sqrt{2 D(v)}\,dW,
#    \qquad D(v) = D_0\left(1 + (v/v_c)^2\right),
#
# in the **Itô convention**: the force SFI infers is the Itô drift of
# the velocity equation.  Both fields are composed from coordinate
# primitives — note the velocity primitive ``v`` entering the diffusion.

import SFI
from SFI.bases import identity_matrix_basis, unit_axes, v_components, x_components
from SFI.langevin import UnderdampedProcess

k = 1.0       # stiffness
gamma = 1.0   # friction
D0 = 0.3      # diffusion at rest
vc = 1.0      # velocity scale of the noise growth
dt = 0.01
Nsteps = 80_000

(x,) = x_components(1)
(v,) = v_components(1)
(ex,) = unit_axes(1)
I = identity_matrix_basis(1)

F_model = (-k * x - gamma * v) * ex            # Itô drift of dv
D_model = D0 * (1.0 + (v / vc) ** 2) * I       # D(v) — multiplicative in v

proc = UnderdampedProcess(F=F_model, D=D_model,
                          theta_F=jnp.ones(F_model.n_features),
                          theta_D=jnp.ones(D_model.n_features))
proc.initialize(jnp.array([0.0]), v0=jnp.array([0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(5),
                     prerun=500, oversampling=20)
print(f"Trajectory: {coll.T} frames, dt={dt} (positions only)")

# %%
# Phase portrait
# ---------------
#
# Velocity is *not* observed.  For this plot only we estimate it by
# central finite differences; the inference engine does its own
# reconstruction and never needs it.  Fast excursions (top and bottom)
# are noticeably noisier than slow ones.

# sphinx_gallery_start_ignore
# Central-difference velocity for plotting, packed with x into a phase-space
# (x, v) collection so the canonical plotters can sweep the v axis.
_t, _Xpos, _ = coll.to_arrays()
v_rec = coll.velocity_array(scheme="central")
coll_xv = SFI.TrajectoryCollection.from_arrays(
    X=np.concatenate([np.asarray(_Xpos), np.asarray(v_rec)], axis=-1), dt=dt,
)

fig, ax = plt.subplots(figsize=(5, 4))
phase2d_scalar(
    coll_xv,
    color_fn=lambda M: D0 * (1 + (M[:, 1] / vc) ** 2),  # local D(v) at each midpoint
    dims=(0, 1), cmap="plasma", colorbar_label=r"local $D(v)$",
    linewidth=0.6, alpha=0.7, ax=ax,
)
ax.set_xlabel("x")
ax.set_ylabel(r"$\hat v$ (reconstructed)")
ax.set_title("Phase portrait — coloured by local noise amplitude")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inference
# ----------
#
# The data are clean positions, so the linear underdamped sequence
# applies, with the clean-data estimators chosen explicitly (the
# ``WeakNoise`` diffusion constant and the ``clean`` force preset; the
# automatic choice would also guard against measurement noise, which
# these synthetic data do not have).
# :meth:`~SFI.inference.UnderdampedLangevinInference.infer_force_linear`
# fits the Itô drift :math:`F(x,v)`, then
# :meth:`~SFI.inference.UnderdampedLangevinInference.infer_diffusion_linear`
# fits :math:`D(x,v)` on a polynomial basis in *both* variables
# (``include_v=True``) spanning :math:`\{1, x, v, x^2, xv, v^2\}`.  Only
# :math:`1` and :math:`v^2` should carry weight.  Both fields come with
# predicted standard errors.

from SFI.bases import monomials_up_to

B_force = monomials_up_to(order=1, dim=1, include_v=True, rank="vector")
B_diff = monomials_up_to(order=2, dim=1, include_v=True, rank="symmetric_matrix")
D_labels = list(B_diff.labels)

inf = SFI.UnderdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_force, preset="clean")
inf.compute_force_error()
inf.infer_diffusion_linear(B_diff)
inf.compute_diffusion_error()
inf.compare_to_exact(model_exact=proc)

F_true = {"x0·e0": -k, "v0·e0": -gamma}
D_true = {"1·S00": D0, "v0^2·S00": D0 / vc**2}
print(f"NMSE vs truth: force {float(inf.NMSE_force):.4f}, diffusion {float(inf.NMSE_diffusion):.4f}")
print("Force F(x, v):")
for lbl, c, e in zip(B_force.labels, np.asarray(inf.force_coefficients_full),
                     np.asarray(inf.force_coefficients_stderr)):
    print(f"  {lbl:>8} = {c:+.4f} ± {e:.4f}   (true {F_true.get(lbl, 0.0):+.2f})")
print("Diffusion D(x, v):")
for lbl, c, e in zip(D_labels, np.asarray(inf.diffusion_coefficients),
                     np.asarray(inf.diffusion_coefficients_stderr)):
    print(f"  {lbl:>8} = {c:+.4f} ± {e:.4f}   (true {D_true.get(lbl, 0.0):+.2f})")

# %%
# The two true terms dominate and the four others are small.
#
# Selecting the noise model
# --------------------------
#
# :meth:`~SFI.inference.UnderdampedLangevinInference.sparsify_diffusion`
# turns "small" into a decision: a PASTIS search over subsets of the six
# candidate terms, scored by the likelihood of the diffusion fit in the
# metric of its own sandwich covariance;
# :meth:`~SFI.inference.UnderdampedLangevinInference.compute_diffusion_error`
# then gives the covariance of the selected model.

inf.sparsify_diffusion(criterion="PASTIS")
inf.compute_diffusion_error()          # covariance of the selected model
print("selected diffusion terms:")
for i, c, e in zip(np.asarray(inf.diffusion_support), np.asarray(inf.diffusion_coefficients),
                   np.asarray(inf.diffusion_coefficients_stderr)):
    lbl = D_labels[i]
    tv = D_true.get(lbl, 0.0)
    # format option "z": a deviation rounding to zero prints as +0.0, not -0.0
    print(f"  {lbl:>8} = {c:+.4f} ± {e:.4f}   (true {tv:+.2f}, off by {(c - tv) / e:+z.1f}σ)")

# %%
# Left: the Pareto front of the diffusion fit — the information gain
# jumps when :math:`1` and :math:`v^2` enter and is flat afterwards, so
# PASTIS stops at two terms.  Right: the selected coefficients divided by
# their true values, with ±2σ error bars.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
plot_pareto_front(inf.diffusion_sparsity_result, ax=axes[0])
axes[0].set_title(f"Pareto front — PASTIS keeps {len(np.asarray(inf.diffusion_support))} / {len(D_labels)}")
ratio_panel(axes[1], [("linear", np.asarray(inf.diffusion_support),
                       np.asarray(inf.diffusion_coefficients),
                       np.asarray(inf.diffusion_coefficients_stderr), SFI_COLORS["inferred"])],
            D_labels, D_true)
axes[1].set_xlim(-0.6, 1.6)
fig.suptitle("Noise-model selection", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Recovered diffusion profile
# -----------------------------
#
# Evaluating the inferred tensor along the velocity axis recovers the
# parabolic noise profile (band: ±2σ predicted, from the covariance of
# the selected coefficients); a constant-:math:`D` analysis would report
# only the mean.

# The inferred tensor as a callable on phase-space points [x, v]; sweeping the
# v axis (x held at 0) recovers the parabolic noise profile.
def D_inferred(pts):
    pts = jnp.asarray(pts)
    return inf.diffusion_inferred(pts[:, :1], v=pts[:, 1:2])  # (N, 1, 1)


def D_exact(pts):
    v = np.asarray(pts)[:, 1]
    return D0 * (1 + (v / vc) ** 2)


# Band on the v axis at x = 0, from the covariance of the selected
# coefficients (coeff_block embeds it in the full basis order).
v_grid = np.linspace(float(v_rec.min()), float(v_rec.max()), 200)
feats = np.asarray(B_diff(jnp.zeros((200, 1)), v=jnp.asarray(v_grid[:, None])))[:, 0, 0, :]
C_D, Cov_D = inf.coeff_block(slice(None), field="diffusion")
D_mid = feats @ C_D
D_band = np.sqrt(np.einsum("ti,ij,tj->t", feats, Cov_D, feats))
# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.5))
plot_profile_1d(
    coll_xv, D_inferred, exact_field=D_exact, dim=1, component=0, samples=True,
    ax=ax, label_exact="Exact $D(v)$", label_inferred="Inferred $D(v)$",
)
ax.fill_between(v_grid, D_mid - 2 * D_band, D_mid + 2 * D_band,
                color=SFI_COLORS["inferred"], alpha=0.25, label="±2σ")
ax.set_xlabel("v")
ax.set_ylabel("D(v)")
ax.set_title("Velocity-dependent diffusion recovery")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The parametric route
# ---------------------
#
# The parametric estimators fit the same fields by maximising the
# likelihood of the observed positions (and would also absorb
# localisation noise or coarse sampling).
# :meth:`~SFI.inference.UnderdampedLangevinInference.infer_diffusion`
# returns its coefficients with error bars from the inverse Fisher
# information of that likelihood, and the same
# :meth:`~SFI.inference.UnderdampedLangevinInference.sparsify_diffusion`
# call selects the terms, then re-fits the selected model exactly.

inf_par = SFI.UnderdampedLangevinInference(coll)
inf_par.infer_force(B_force)
inf_par.infer_diffusion(B_diff)
inf_par.sparsify_diffusion(criterion="PASTIS")
inf_par.compute_diffusion_error()
print("parametric, selected diffusion terms:")
for i, c, e in zip(np.asarray(inf_par.diffusion_support), np.asarray(inf_par.diffusion_coefficients),
                   np.asarray(inf_par.diffusion_coefficients_stderr)):
    lbl = D_labels[i]
    tv = D_true.get(lbl, 0.0)
    print(f"  {lbl:>8} = {c:+.4f} ± {e:.4f}   (true {tv:+.2f}, off by {(c - tv) / e:+z.1f}σ)")

# %%
# Both routes select the same two terms, and their coefficients agree
# with each other within the error bars; the parametric :math:`v^2`
# coefficient lands about two of its own standard errors above the truth
# (printed above), at the edge of its ±2σ bar.  Agreement between two
# estimators built on different principles is itself a useful check on
# real data, where the truth is unknown.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ratio_panel(ax, [("linear", np.asarray(inf.diffusion_support), np.asarray(inf.diffusion_coefficients),
                  np.asarray(inf.diffusion_coefficients_stderr), SFI_COLORS["inferred"]),
                 ("parametric", np.asarray(inf_par.diffusion_support),
                  np.asarray(inf_par.diffusion_coefficients),
                  np.asarray(inf_par.diffusion_coefficients_stderr), SFI_COLORS["highlight"])],
            D_labels, D_true)
ax.set_xlim(-0.6, 1.6)
ax.legend(fontsize=8)
ax.set_title("Linear vs parametric (±2σ)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# When is this hard?
# --------------------
#
# Velocity-dependent noise has a genuinely hard regime.  With
# :math:`D(v) \propto v^2` at large speed, the velocity distribution
# develops **power-law tails** — for a free particle the stationary
# density decays as :math:`|v|^{-(2 + \gamma/D_0)}`.  If friction is weak
# relative to the noise gradient, rare high-speed bursts dominate the
# statistics, and no finite sampling rate resolves them — estimates of
# the noise floor :math:`D_0` then degrade no matter how small
# :math:`\Delta t` is.  Here :math:`\gamma/D_0 \approx 3.3` keeps the
# tails integrable and the inference quantitative, yet the reconstructed
# speeds are already visibly heavier-tailed than a Gaussian of the same
# variance (below).  For strongly driven systems, prefer
# *saturating* noise models (e.g.
# :math:`D_0 + \Delta D\, v^2/(v^2 + v_s^2)`).

# sphinx_gallery_start_ignore
v_all = np.asarray(v_rec).ravel()
v_all = v_all[np.isfinite(v_all)]
s_v = v_all.std()
fig, ax = plt.subplots(figsize=(5.5, 3.5))
bins = np.linspace(-1.05 * np.abs(v_all).max(), 1.05 * np.abs(v_all).max(), 81)
ax.hist(v_all, bins=bins, density=True, histtype="stepfilled", alpha=0.6,
        color=SFI_COLORS["data"], label=r"reconstructed $\hat v$")
vv = np.linspace(bins[0], bins[-1], 400)
ax.plot(vv, np.exp(-vv**2 / (2 * s_v**2)) / np.sqrt(2 * np.pi * s_v**2), lw=2,
        color=SFI_COLORS["exact"], label="Gaussian, same variance")
ax.set_yscale("log")
ax.set_ylim(0.5 / (len(v_all) * (bins[1] - bins[0])), None)
ax.set_xlabel("v")
ax.set_ylabel("probability density")
ax.set_title("Heavy velocity tails")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
phase2d_scalar(
    coll_xv,
    color_fn=lambda M: D0 * (1 + (M[:, 1] / vc) ** 2),
    dims=(0, 1), cmap="plasma", plot_colorbar=False, ax=ax_t,
)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
