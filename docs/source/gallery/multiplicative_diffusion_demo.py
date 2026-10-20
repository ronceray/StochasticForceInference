r"""
Multiplicative noise — the Landauer blowtorch
==============================================

A particle hops in a **symmetric** double well, yet it spends most of
its time on the left.  The reason is invisible in the force: the bath is
**hotter on the right** — Landauer's *blowtorch*.  This page recovers
the state-dependent diffusion field :math:`D(x)` together with the force
from a single trajectory, lets PASTIS decide which terms :math:`D(x)`
needs, and shows that the inferred force and diffusion field together —
and only together — predict the lopsided occupancy.  The last section
repeats the analysis on a recording with localisation noise.

.. rubric:: Tags

synthetic · overdamped · multiplicative-noise · 1D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "multiplicative-noise", "1D"]
# sphinx_gallery_thumbnail_number = 5

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
from SFI.utils.plotting import plot_pareto_front, timeseries_colored

apply_style()
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A double well in a temperature gradient
# ---------------------------------------
#
# .. math::
#
#    dx = (x - x^3)\,dt + \sqrt{2 D(x)}\,dW,
#    \qquad D(x) = D_0\,(1 + g\,x),
#
# in the **Itô convention** — the drift in the SDE, and the force SFI
# infers, is the Itô drift.  The drift is symmetric under
# :math:`x\to -x`; with :math:`g > 0` the right well sits in the hot
# region.  Both fields are built from :mod:`SFI.bases` primitives.

import SFI
from SFI.bases import identity_matrix_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

D0 = 0.5      # baseline diffusion
g = 0.3       # temperature gradient
dt = 0.01

(x,) = x_components(1)
(ex,) = unit_axes(1)
I = identity_matrix_basis(1)

F_model = (x - x**3) * ex                # F(x) = x - x³  (Itô drift)
D_model = D0 * (1.0 + g * x) * I         # D(x) = D₀(1 + g·x)

proc = OverdampedProcess(F=F_model, D=D_model,
                         theta_F=jnp.array([1.0]), theta_D=jnp.array([1.0]))
proc.initialize(jnp.array([0.5]))
coll = proc.simulate(dt=dt, Nsteps=50_000, key=random.PRNGKey(7),
                     prerun=200, oversampling=10)

t, X, _mask = coll.to_arrays(dataset=0)
Xarr = np.asarray(X[:, 0, 0])
print(f"{coll.T} frames at dt = {dt};  time spent in the hot half (x > 0): "
      f"{np.mean(Xarr > 0):.0%}")

# %%
# Excursions in the hot well are visibly more agitated — and shorter:
# the occupancy (right) is lopsided although the drift is symmetric.

# sphinx_gallery_start_ignore
fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(9, 3), sharey=True,
                               gridspec_kw={"width_ratios": [5, 1]})
timeseries_colored(coll, color_fn=lambda x: D0 * (1 + g * x[:, 0]),
                   cmap="plasma", colorbar_label="local $D(x)$", s=1.5, ax=ax)
for ax_ in (ax, ax_h):
    ax_.axhline(1, ls=":", lw=0.6, color="#808080")
    ax_.axhline(-1, ls=":", lw=0.6, color="#808080")
ax.set_xlabel("time")
ax.set_ylabel("$x$")
ax.set_title("Blowtorch trajectory, coloured by the local diffusivity")
ax_h.hist(Xarr, bins=70, density=True, orientation="horizontal",
          color=SFI_COLORS["data"], alpha=0.7)
ax_h.set_xlabel("$p(x)$")
ax_h.set_title("occupancy")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infer the force, then the diffusion field
# -----------------------------------------
#
# The usual linear sequence plus one step:
# :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`
# fits the diffusion *field* on a matrix-valued polynomial basis
# (``rank="symmetric_matrix"``) spanning :math:`\{1, x, x^2\}` — wider
# than the truth, so the shape of :math:`D(x)` is read off the data.  The
# data are clean, so the ``"clean"`` force preset and the ``WeakNoise``
# local diffusion estimator apply (unlike the plain mean-squared
# displacement, it carries no :math:`F^2\Delta t` bias).

from SFI.bases import monomials_up_to

B_force = monomials_up_to(order=3, dim=1, rank="vector")
B_diff = monomials_up_to(order=2, dim=1, rank="symmetric_matrix")

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")   # constant-D baseline
inf.infer_force_linear(B_force, preset="clean")       # Itô drift
inf.compute_force_error()
inf.infer_diffusion_linear(B_diff, M_mode="WeakNoise")  # D(x)
inf.compute_diffusion_error()
inf.compare_to_exact(model_exact=proc)
inf.print_report()

D_true = {"1·S00": D0, "x0·S00": D0 * g, "x0^2·S00": 0.0}
for lbl, c, e in zip(B_diff.labels, np.asarray(inf.diffusion_coefficients),
                     np.asarray(inf.diffusion_coefficients_stderr)):
    print(f"  D[{lbl:>8}] = {c:+.4f} ± {e:.4f}   (true {D_true[lbl]:+.3f})")

# %%
# The diffusion error bars are sandwich estimates from the data alone;
# they are calibrated here because the basis contains the true
# :math:`D(x)`.
#
# Which terms does :math:`D(x)` need?
# -----------------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_diffusion`
# runs the PASTIS search over subsets of the diffusion basis, exactly as
# :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force` does
# for the force.

inf.sparsify_diffusion(criterion="PASTIS")
inf.compute_diffusion_error()           # covariance of the selected model
for i, c, e in zip(np.asarray(inf.diffusion_support), np.asarray(inf.diffusion_coefficients),
                   np.asarray(inf.diffusion_coefficients_stderr)):
    lbl = B_diff.labels[i]
    print(f"  selected D[{lbl:>8}] = {c:+.4f} ± {e:.4f}   (true {D_true[lbl]:+.3f})")

# %%
# PASTIS keeps the constant and the gradient and drops the quadratic
# term (right: the information gain saturates at two terms).  The
# constant-:math:`D` estimate a standard analysis would report (dash-dot)
# misses the gradient entirely.  (The force band is computed with the
# average diffusion, so it is somewhat too narrow in the hot well.)

x_grid = np.linspace(-1.9, 1.9, 200)
x_jnp = jnp.array(x_grid[:, None])
F_ci = inf.force_inferred.predict_ci(x_jnp)       # mean and pointwise std
D_ci = inf.diffusion_inferred.predict_ci(x_jnp)
F_inf, F_std = np.asarray(F_ci["mean"])[:, 0], np.asarray(F_ci["std"])[:, 0]
D_inf, D_std = np.asarray(D_ci["mean"])[:, 0, 0], np.asarray(D_ci["std"])[:, 0, 0]
D_const = float(np.asarray(inf.diffusion_average).ravel()[0])
F_exact = x_grid - x_grid**3
D_exact = D0 * (1 + g * x_grid)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
axes[0].plot(x_grid, F_exact, "--", lw=2, color=SFI_COLORS["exact"], label="exact")
axes[0].plot(x_grid, F_inf, lw=2, color=SFI_COLORS["inferred"], label="inferred")
axes[0].fill_between(x_grid, F_inf - 2 * F_std, F_inf + 2 * F_std,
                     color=SFI_COLORS["inferred"], alpha=0.25, label=r"$\pm 2\sigma$")
axes[0].axhline(0, color="#808080", lw=0.5)
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$F(x)$")
axes[0].set_title("Force (Itô drift)")
axes[0].legend(fontsize=8)

axes[1].plot(x_grid, D_exact, "--", lw=2, color=SFI_COLORS["exact"], label="exact")
axes[1].plot(x_grid, D_inf, lw=2, color=SFI_COLORS["inferred"], label="inferred (selected)")
axes[1].fill_between(x_grid, D_inf - 2 * D_std, D_inf + 2 * D_std,
                     color=SFI_COLORS["inferred"], alpha=0.25, label=r"$\pm 2\sigma$")
axes[1].axhline(D_const, ls="-.", lw=1.2, color=SFI_COLORS["highlight"],
                label="constant-$D$ estimate")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$D(x)$")
axes[1].set_title("Diffusion field")
axes[1].legend(fontsize=8)

plot_pareto_front(inf.diffusion_sparsity_result, criteria=("PASTIS",), ax=axes[2])
axes[2].set_xticks(range(len(B_diff.labels) + 1))
axes[2].set_title(r"Selecting $D(x)$ terms")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The blowtorch effect, explained
# -------------------------------
#
# For Itô dynamics the zero-current stationary density is
#
# .. math::
#
#    p(x) \;\propto\; \frac{1}{D(x)}
#    \exp\!\Big(\int^x \frac{F(y)}{D(y)}\,dy\Big),
#
# *not* the Boltzmann weight of the drift with a constant temperature.
# Evaluated with the **inferred** :math:`\hat F` and :math:`\hat D(x)`,
# it reproduces the empirical histogram; with the constant-:math:`D`
# estimate it predicts two nearly equal wells.

from scipy.integrate import cumulative_trapezoid


def stationary_density(F_vals, D_vals, x_vals):
    """Zero-current Itô stationary density on a grid (normalised)."""
    phi = cumulative_trapezoid(F_vals / D_vals, x_vals, initial=0.0)
    p = np.exp(phi - phi.max()) / D_vals
    return p / np.trapezoid(p, x_vals)


p_exact = stationary_density(F_exact, D_exact, x_grid)
p_inferred = stationary_density(F_inf, D_inf, x_grid)
p_naive = stationary_density(F_inf, np.full_like(x_grid, D_const), x_grid)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(Xarr, bins=80, density=True, alpha=0.45, color=SFI_COLORS["data"],
        label="data histogram")
ax.plot(x_grid, p_inferred, lw=2.2, color=SFI_COLORS["inferred"],
        label=r"inferred $\hat F$ and $\hat D(x)$")
ax.plot(x_grid, p_naive, "-.", lw=1.6, color=SFI_COLORS["highlight"],
        label=r"$\hat F$ with constant $\hat D$")
ax.plot(x_grid, p_exact, ":", lw=1.6, color=SFI_COLORS["exact"], label="exact")
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title("The hot well is depopulated")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# With localisation noise
# -----------------------
#
# Tracked particles come with localisation error.  We add Gaussian noise
# of standard deviation :math:`\sigma = 0.1`, as large as a single-frame
# displacement :math:`\sqrt{2D_0\Delta t}`.  The linear estimators assume
# clean positions; the parametric ones model the noise:
# :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` profiles
# the noise level along with the force, and
# :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` then
# fits :math:`D(x)` on the same basis with the force held fixed.

sigma_loc = 0.1
coll_noisy = coll.degrade(noise=sigma_loc, seed=1)

inf_naive = SFI.OverdampedLangevinInference(coll_noisy)      # noise-blind
inf_naive.compute_diffusion_constant(method="WeakNoise")
inf_naive.infer_force_linear(B_force, preset="clean")
inf_naive.infer_diffusion_linear(B_diff, M_mode="WeakNoise")

inf_par = SFI.OverdampedLangevinInference(coll_noisy)        # noise-aware
inf_par.infer_force(B_force)
inf_par.infer_diffusion(B_diff)
inf_par.sparsify_diffusion(criterion="PASTIS")
inf_par.compute_diffusion_error()

sigma_hat = float(np.sqrt(np.asarray(inf_par.Lambda).ravel()[0]))
print(f"localisation noise: true σ = {sigma_loc},  estimated σ̂ = {sigma_hat:.3f}")
for name, inf_k in [("linear (noise-blind)", inf_naive), ("parametric", inf_par)]:
    print(f"{name:>21}:  F coeffs {np.round(np.asarray(inf_k.force_coefficients_full), 3)}"
          f"   D coeffs {np.round(np.asarray(inf_k.diffusion_coefficients), 3)}")
print(f"{'truth':>21}:  F coeffs [ 0.  1.  0. -1.]   D coeffs [{D0} {D0 * g} 0.]")
for i, c, e in zip(np.asarray(inf_par.diffusion_support), np.asarray(inf_par.diffusion_coefficients),
                   np.asarray(inf_par.diffusion_coefficients_stderr)):
    lbl = B_diff.labels[i]
    print(f"  parametric, selected D[{lbl:>8}] = {c:+.4f} ± {e:.4f}   (true {D_true[lbl]:+.3f})")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for inf_k, lbl, col, ls in [(inf_naive, "linear (noise-blind)", SFI_COLORS["error"], "-."),
                            (inf_par, "parametric (noise-aware)", SFI_COLORS["inferred"], "-")]:
    axes[0].plot(x_grid, np.asarray(inf_k.force_inferred(x_jnp))[:, 0], ls, lw=2,
                 color=col, label=lbl)
    axes[1].plot(x_grid, np.asarray(inf_k.diffusion_inferred(x_jnp))[:, 0, 0], ls, lw=2,
                 color=col, label=lbl)
Dk = inf_par.diffusion_inferred.predict_ci(x_jnp)
m_, s_ = np.asarray(Dk["mean"])[:, 0, 0], np.asarray(Dk["std"])[:, 0, 0]
axes[1].fill_between(x_grid, m_ - 2 * s_, m_ + 2 * s_, color=SFI_COLORS["inferred"], alpha=0.25)
axes[0].plot(x_grid, F_exact, "--", lw=2, color=SFI_COLORS["exact"], label="exact")
axes[1].plot(x_grid, D_exact, "--", lw=2, color=SFI_COLORS["exact"], label="exact")
axes[0].set_ylim(-4, 4)
axes[1].set_ylim(0, None)
for ax_, yl, ttl in [(axes[0], "$F(x)$", "Force"), (axes[1], "$D(x)$", "Diffusion field")]:
    ax_.axhline(0, color="#808080", lw=0.5)
    ax_.set_xlabel("$x$")
    ax_.set_ylabel(yl)
    ax_.set_title(rf"{ttl} — noisy recording ($\sigma = {sigma_loc}$)")
axes[0].legend(fontsize=8, loc="upper right")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The noise-blind fit distorts the force several-fold and reads the
# localisation noise as extra diffusion (its :math:`\hat D` is about four
# times too large).  The parametric fit recovers both fields and the
# noise level itself, and PASTIS selects the same two diffusion terms as
# on clean data (shaded: :math:`\pm 2\sigma` on :math:`D(x)`).  Two
# caveats on its error bars: the diffusion ones treat the estimated noise
# level as known, and since the noise level and :math:`D_0` trade off
# they are somewhat optimistic on noisy data; the force ones assume a
# constant diffusion and are too narrow in the hot well, which is why no
# force band is drawn here.
#
# Which "force"? Spurious drifts
# ------------------------------
#
# SFI infers the **Itô drift** :math:`F(x) = \lim_{\Delta t\to 0}\langle
# \Delta x \rangle / \Delta t`.  Other conventions move part of the
# noise gradient into the drift — in 1D,
# :math:`F_{\rm Strato} = F_{\rm Itô} - \tfrac12 D'(x)` and
# :math:`F_{\rm anti\text{-}Itô} = F_{\rm Itô} - D'(x)`.  Which one is
# the physical force is an experimental question; once :math:`F_{\rm
# Itô}` and :math:`D(x)` are both inferred, the conversion is immediate:

dD = np.gradient(D_inf, x_grid)            # D'(x) of the inferred field
F_strato = F_inf - 0.5 * dD
F_anti_ito = F_inf - dD
print(f"inferred D'(x) = {dD.mean():.3f}  (true {D0 * g})")

# %%
# Here :math:`D'` is constant, so the conventions differ by a uniform
# shift of the drift — enough to move its zeros, the apparent well
# bottoms.  Note that the density formula above peaks where
# :math:`F_{\rm Itô} = D'`: at the zeros of the *anti-Itô* drift, which
# is why the occupancy peaks sit slightly left of :math:`\pm 1`.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
timeseries_colored(coll, color_fn=lambda x: D0 * (1 + g * x[:, 0]),
                   cmap="plasma", plot_colorbar=False, s=2.0, ax=ax_t)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
