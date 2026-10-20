r"""
2D limit cycle — nonlinear overdamped inference
===============================================

A noisy oscillator: a radial force pulls the system onto the unit
circle while a constant angular drift turns it around — a
non-equilibrium steady state with a circulating probability current.
From one trajectory, we recover the full nonlinear force field with a
generic cubic dictionary,
check it with residual diagnostics and simulations of the inferred model,
and then repeat the inference on positions blurred by localisation noise.

.. rubric:: Tags

synthetic · overdamped · nonlinear · non-equilibrium · 2D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "non-equilibrium", "2D"]
# sphinx_gallery_thumbnail_number = 7

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
from matplotlib.colors import ListedColormap
from SFI import TrajectoryCollection
from SFI.diagnostics import plot_summary
from SFI.utils.plotting import (
    animate_particles_comparison,
    dark_fig,
    phase2d,
    plot_field,
    plot_field_error,
    plot_pareto_front,
    plot_recovery_bar,
    stream_field,
    timeseries,
)

apply_style()
# Movies are encoded on the figure's own (dark) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"
# sphinx_gallery_end_ignore
stamp_output()

# %%
# The system
# ----------
#
# In polar coordinates the drift is :math:`\dot r = r(1 - r^2)`,
# :math:`\dot\theta = \omega`; in Cartesian coordinates the force reads
#
# .. math::
#
#    F_x = x(1 - x^2 - y^2) - \omega\, y, \qquad
#    F_y = y(1 - x^2 - y^2) + \omega\, x,
#
# with :math:`\omega = 2` (period :math:`\pi`) and isotropic noise
# :math:`D = 0.1`.  We record 200 time units — about 64 turns — at
# :math:`\Delta t = 0.05`.

from SFI.bases import named_scalars, ones_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

# The true force, written symbolically:
#     F = (1 - r^2) (x e_x + y e_y) + omega (x e_y - y e_x),  r^2 = x^2 + y^2
x, y = x_components(2)
ex, ey = unit_axes(2)
one = ones_basis(2)
(omega_p,) = named_scalars(omega=2.0)
F_true = (one - (x * x + y * y)) * (x * ex + y * ey) + omega_p * (x * ey - y * ex)

D0, dt, Nsteps = 0.1, 0.05, 4_000
proc = OverdampedProcess(F_true, D=jnp.eye(2) * D0)
proc.initialize(jnp.array([0.5, 0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(0), prerun=5000, oversampling=20)

# sphinx_gallery_start_ignore
theta_ring = np.linspace(0, 2 * np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
phase2d(coll, dims=(0, 1), ax=axes[0], cmap="viridis", linewidth=0.4, alpha=0.8)
axes[0].plot(np.cos(theta_ring), np.sin(theta_ring), "--", color="#D0D0D0", lw=1)
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
axes[0].set_title("Trajectory (colour = time)")
timeseries(coll, dims=(0, 1), ax=axes[1], lw=0.8)
axes[1].set_xlim(0, 20)
axes[1].set_xlabel("time")
axes[1].set_ylabel("$x(t)$, $y(t)$")
axes[1].legend(["x", "y"], loc="upper right", ncol=2)
axes[1].set_title("Noisy oscillations (first 20 time units)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inference and sparse selection
# ------------------------------
#
# A cubic polynomial dictionary in :math:`(x, y)` offers 20 candidate
# terms; the true force uses 8.  The data are noise-free and the drift per
# step is large compared with the noise, so we estimate the diffusion with
# ``"WeakNoise"`` and use the ``"clean"`` preset of the linear force
# estimator; :term:`PASTIS` then selects the terms.

from SFI import OverdampedLangevinInference
from SFI.bases import monomials_up_to

B = monomials_up_to(order=3, dim=2, include_constant=True, rank="vector")

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B, preset="clean")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()

# sphinx_gallery_start_ignore
# Truth coefficients: the exact force projected onto the same dictionary.
labels = list(B.labels)
_pts = jnp.asarray(np.random.default_rng(0).uniform(-1.5, 1.5, (400, 2)))
_Phi = np.asarray(B(_pts)).reshape(-1, B.n_features)
theta_true = np.linalg.lstsq(_Phi, np.asarray(proc.force_sf(_pts)).ravel(), rcond=None)[0]
theta_true[np.abs(theta_true) < 1e-6] = 0.0
idx = np.arange(B.n_features)


def coefficients_with_errors(fit):
    """Full-length coefficient and 1σ vectors (zeros off the support)."""
    c, cov = fit.coeff_block(slice(None))
    return np.asarray(c), np.sqrt(np.diag(np.asarray(cov)))


coeffs_sel, stderr_sel = coefficients_with_errors(inf)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
plot_recovery_bar(coeffs_sel, idx, coeffs_true=theta_true, support_true=idx,
                  stderr=stderr_sel, labels=labels, ax=axes[1])
axes[1].set_title(f"PASTIS keeps {len(inf.force_support)} / {B.n_features} terms (1σ error bars)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The Pareto front (left) jumps with the two rotation terms, climbs more
# gently as the six radial terms enter, and is flat beyond: PASTIS keeps
# exactly the eight true terms, each within about one error bar of its
# true value (right).  In space, the inferred field (gold arrows) matches
# the exact one where the data are; the error map (right) shows the
# reconstruction error growing only towards the corners, outside the
# explored region.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
phase2d(coll, dims=(0, 1), ax=ax, color=SFI_COLORS["data"], linewidth=0.3, alpha=0.35)
plt.sca(ax)
plot_field(coll, proc.force_sf, N=12, color=SFI_COLORS["exact"], autoscale=True,
           scale=0.25, width=0.03)
plot_field(coll, inf.force_inferred, N=12, color=SFI_COLORS["inferred"], autoscale=True,
           scale=0.25, width=0.012)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Force field: inferred (gold) over exact (orange)")
ax = axes[1]
plot_field_error(coll, inf.force_inferred, proc.force_sf, N=60, ax=ax)
ax.plot(np.cos(theta_ring), np.sin(theta_ring), "--", color="#D0D0D0", lw=1, alpha=0.7)
fig.axes[-1].set_ylabel(r"$\|\hat F - F_{\rm exact}\|$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Reconstruction error (dashed: limit cycle)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Diagnostics
# -----------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.diagnose` tests the
# standardised residuals of the fit: they should be i.i.d.
# :math:`\mathcal{N}(0, 1)`, with a realised error consistent with the
# predicted one.

report = inf.diagnose()
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Residuals of the limit-cycle fit")
plt.show()
# sphinx_gallery_end_ignore

# %%
# All residual tests pass.  The ``finite_dt`` flag notes that
# :math:`\Delta t\,|\nabla F|` is not small, so the linear estimates carry
# an :math:`O(\Delta t)` bias; here it stays below the sampling error,
# since the realised error is consistent with its prediction.
#
# Simulating the inferred model
# -----------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.simulate_bootstrapped_trajectory`
# runs the inferred force and diffusion from the first recorded point.
# The movie plays the data and the simulation side by side over the
# inferred flow lines (first 25 time units): same radius, same rotation
# speed, same jitter.

coll_boot, _ = inf.simulate_bootstrapped_trajectory(random.PRNGKey(123), oversampling=20)

# sphinx_gallery_start_ignore
fig_m, axes_m = dark_fig(1, 2, figsize=(6.6, 3.5))
for ax in axes_m:
    stream_field(coll, inf.force_inferred, N=30, color="#4A5468", density=0.9,
                 linewidth=0.6, arrowsize=0.7, ax=ax)
n_movie = 500                                   # first 25 time units
anim = animate_particles_comparison(
    [TrajectoryCollection.from_arrays(X=c.to_array()[:n_movie], dt=dt) for c in (coll, coll_boot)],
    titles=["data", "inferred model"], s=70,
    cmap=ListedColormap([SFI_COLORS["data"]]), trail=30, skip=2, interval=70, axes=axes_m,
)
for ax in axes_m:
    for trail_line in ax.lines:
        trail_line.set(color="#B0C4DE", alpha=0.8, linewidth=1.0)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Robustness: localisation noise
# ------------------------------
#
# Tracked positions carry a localisation error.  We add Gaussian noise of
# standard deviation :math:`\sigma = 0.08` to every position — almost the
# size of the diffusive step :math:`\sqrt{2D\Delta t} = 0.1`.  The linear
# estimator with its default settings is compared with the parametric
# estimator :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
# which models the noise explicitly (profiling its amplitude
# :math:`\Lambda = \sigma^2` jointly with :math:`D`) and integrates the
# flow over each interval.  PASTIS runs on both.

coll_noisy = coll.degrade(noise=0.08, seed=1)

inf_lin = OverdampedLangevinInference(coll_noisy)
inf_lin.compute_diffusion_constant()
inf_lin.infer_force_linear(B)
inf_lin.sparsify_force(criterion="PASTIS")
inf_lin.compute_force_error()
inf_lin.compare_to_exact(force_exact=proc.force_sf, data_exact=coll, maxpoints=2000)

inf_par = OverdampedLangevinInference(coll_noisy)
inf_par.infer_force(B)
inf_par.sparsify_force(criterion="PASTIS")
inf_par.compute_force_error()
inf_par.compare_to_exact(force_exact=proc.force_sf, data_exact=coll, maxpoints=2000)

for name, fit in [("linear (default)", inf_lin), ("parametric", inf_par)]:
    D_hat = np.diag(np.asarray(fit.diffusion_average))
    sigma_hat = np.sqrt(np.maximum(np.diag(np.asarray(fit.Lambda)), 0.0))
    print(f"{name:17s} terms kept = {len(fit.force_support):2d}   "
          f"D = {D_hat.round(3)}   σ = {sigma_hat.round(3)}   "
          f"force NMSE vs truth = {fit.NMSE_force:.4f} (predicted {fit.force_predicted_MSE:.4f})")
print("truth             terms     =  8   D = [0.1 0.1]   σ = [0.08 0.08]")

# sphinx_gallery_start_ignore
vmax = 0.6
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, fit, title in zip(axes, (inf_lin, inf_par), ("linear estimator (default)", "parametric estimator")):
    plot_field_error(coll, fit.force_inferred, proc.force_sf, N=60, vmax=vmax, ax=ax)
    ax.plot(np.cos(theta_ring), np.sin(theta_ring), "--", color="#D0D0D0", lw=1, alpha=0.7)
    ax.set_title(f"{title}: {len(fit.force_support)} terms")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
for cax in fig.axes[2:]:
    cax.set_ylabel(r"$\|\hat F - F_{\rm exact}\|$")
fig.suptitle(r"Noisy positions ($\sigma = 0.08$): force reconstruction error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The default linear estimator is overwhelmed: localisation noise inflates
# its diffusion estimate and hides the weak radial terms, so PASTIS keeps
# only the strongest ones and the realised error far exceeds the predicted
# one — the signature of a bias.  The parametric estimator recovers the
# eight true terms, both :math:`D` and :math:`\sigma`, and a force error at
# its predicted, sampling-limited level, as on clean data.
#
# Takeaways
# ---------
#
# * A generic cubic dictionary plus :term:`PASTIS` finds the eight terms of
#   the limit cycle, with error bars and a predicted error that the
#   comparison with the truth confirms.
# * Residual diagnostics and simulations of the inferred model validate the
#   fit without reference to the truth.
# * When positions are noisy, switch to the parametric estimator: it
#   separates localisation noise from diffusion and keeps the force
#   unbiased.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
phase2d(coll, dims=(0, 1), ax=ax_t, color=SFI_COLORS["data"], linewidth=0.4, alpha=0.6)
stream_field(coll, inf.force_inferred, N=30, color=SFI_COLORS["inferred"], density=0.6,
             linewidth=0.8, arrowsize=0.8, ax=ax_t)
ax_t.set_aspect("equal")
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
