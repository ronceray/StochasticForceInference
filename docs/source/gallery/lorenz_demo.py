r"""
Noise-driven Lorenz system — hopping between two foci
=====================================================

A 3D Lorenz-type system *below* its chaotic threshold: without noise it
would settle on one of two stable foci.  Strong noise kicks it around
each focus and, every so often, across to the other.  From a single
recording of 10 time units, can we recover the seven Lorenz terms among
30 polynomial candidates — and a model that hops like the data?

The workflow: linear inference with :term:`PASTIS` model selection →
residual diagnostics → a cross-check with the parametric estimator →
simulations of the inferred model, compared with the data.

.. rubric:: Tags

synthetic · overdamped · nonlinear · sparsity · 3D · benchmark
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "sparsity", "3D", "benchmark"]
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
from _gallery_utils.helpers import SFI_COLORS, apply_style, plot_pareto_front, stamp_output
from SFI.diagnostics import plot_summary
from SFI.utils.plotting import (
    animate_particles_comparison,
    dark_fig,
    phase2d_scalar,
    timeseries,
)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

apply_style()
# Movies are encoded on the figure's own (dark) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A Lorenz system below chaos
# ---------------------------
#
# .. math::
#
#    \dot{x} = \sigma(y - x), \quad
#    \dot{y} = x(\rho - z) - y, \quad
#    \dot{z} = x\,y - \beta\,z
#
# with :math:`(\sigma, \rho, \beta) = (20, 8, 2)`.  Since :math:`\rho` is
# well below the chaotic threshold
# :math:`\rho_H = \sigma(\sigma+\beta+3)/(\sigma-\beta-1) \approx 29`, the
# two foci :math:`C_\pm = (\pm\sqrt{14}, \pm\sqrt{14}, 7)` are stable.
# Isotropic noise :math:`D_0 = 5` drives all the large-scale motion.  We
# record 10 time units at :math:`\Delta t = 0.005`.

from SFI.bases import named_scalars, unit_axes, x_components
from SFI.langevin import OverdampedProcess

# The true force, written symbolically with the basis primitives; the
# named scalars carry the parameter values as defaults.
x, y, z = x_components(3)
ex, ey, ez = unit_axes(3)
sigma_p, rho_p, beta_p = named_scalars(sigma=20.0, rho=8.0, beta=2.0)
F_true = (
    (sigma_p * (y - x)) * ex
    + (x * (rho_p - z) - y) * ey
    + (x * y - beta_p * z) * ez
)

D0, dt, Nsteps = 5.0, 0.005, 2000
proc = OverdampedProcess(F_true, D=jnp.eye(3) * D0)
proc.initialize(jnp.array([1.0, 1.0, 1.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(9), prerun=2000, oversampling=10)

# sphinx_gallery_start_ignore
x_focus = np.sqrt(14.0)                     # x-coordinate of the foci C+-
foci = np.array([[x_focus, x_focus, 7.0], [-x_focus, -x_focus, 7.0]])


def _clear_panes(ax3d):
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color("#808080")


fig = plt.figure(figsize=(11, 4.2), layout="none")
ax3 = fig.add_axes([0.0, 0.04, 0.4, 0.86], projection="3d")
X_data = np.asarray(coll.to_array())[:, 0, :]
segments = Line3DCollection(np.stack([X_data[:-1], X_data[1:]], axis=1), cmap="coolwarm",
                            norm=plt.Normalize(-2 * x_focus, 2 * x_focus), linewidth=0.8)
segments.set_array(0.5 * (X_data[:-1, 0] + X_data[1:, 0]))
ax3.add_collection3d(segments)
ax3.set(xlim=(-11, 11), ylim=(-11, 11), zlim=(-1, 16), xlabel="x", ylabel="y", zlabel="z",
        xticks=[-10, -5, 0, 5, 10], yticks=[-10, -5, 0, 5, 10], zticks=[0, 5, 10, 15])
ax3.scatter(*foci.T, s=60, marker="X", color=SFI_COLORS["exact"], depthshade=False,
            label=r"stable foci $C_\pm$")
ax3.view_init(elev=22, azim=-60)
_clear_panes(ax3)
ax3.legend(loc="upper left")
ax3.set_title("Trajectory (colour = x)")

ax = fig.add_axes([0.5, 0.13, 0.48, 0.77])
timeseries(coll, dims=(0, 1), ax=ax, lw=0.7)
for s in (+1, -1):
    ax.axhline(s * x_focus, color=SFI_COLORS["exact"], ls="--", lw=1)
ax.set_xlabel("time")
ax.set_ylabel(r"$x(t)$, $y(t)$")
ax.legend(["x", "y", r"foci $\pm\sqrt{14}$"], loc="lower right", ncol=3)
ax.set_title("Hopping between the two foci")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The noise carries the system around one focus, then across to the other.
# :math:`y` follows :math:`x` closely: the strong coupling
# :math:`\sigma = 20` slaves :math:`y` to :math:`x` within
# :math:`1/\sigma = 0.05` time units — a near-collinearity that will matter
# for model selection.
#
# Linear inference and sparse selection
# -------------------------------------
#
# A degree-2 polynomial basis in 3D offers 30 candidate terms; the true
# model uses 7.  The data are noise-free, so we use the clean-data
# estimators: ``"WeakNoise"`` for the diffusion (the plain MSD would be
# biased by this strong drift) and ``preset="clean"`` for the force.
# :term:`PASTIS` then selects the terms.

from SFI import OverdampedLangevinInference
from SFI.bases import monomials_up_to

B = monomials_up_to(order=2, dim=3, rank="vector")

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B, preset="clean")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, maxpoints=2000)
inf.print_report()

# sphinx_gallery_start_ignore
labels = list(B.labels)
# The seven Lorenz terms, by basis label.
TRUE = {
    "x0·e0": -20.0, "x1·e0": 20.0,                       # sigma (y - x)
    "x0·e1": 8.0, "x1·e1": -1.0, "(x0·x2)·e1": -1.0,     # x (rho - z) - y
    "(x0·x1)·e2": 1.0, "x2·e2": -2.0,                    # x y - beta z
}
EQ_NAMES = [r"$\dot x$", r"$\dot y$", r"$\dot z$"]


def _tex(label):
    """'(x0·x1)·e2' -> '$xy$' (the monomial of a basis term, in x, y, z)."""
    mono = label.rsplit("·e", 1)[0].strip("()")
    for k, v in (("x0", "x"), ("x1", "y"), ("x2", "z"), ("·", "")):
        mono = mono.replace(k, v)
    return f"${mono}$"


def coefficient_panels(fit, axes):
    """One panel per equation: selected coefficients (1σ bars) next to the truth."""
    sup = np.asarray(fit.force_support, dtype=int).tolist()
    c = np.asarray(fit.force_coefficients)
    e = np.asarray(fit.force_coefficients_stderr)
    sel = {labels[i]: (c[k], e[k]) for k, i in enumerate(sup)}
    for eq, ax in enumerate(axes):
        terms = [lab for lab in labels if lab.endswith(f"e{eq}") and (lab in TRUE or lab in sel)]
        xs = np.arange(len(terms))
        ax.bar(xs - 0.2, [sel.get(t, (0, 0))[0] for t in terms], width=0.4,
               yerr=[sel.get(t, (0, 0))[1] for t in terms], color=SFI_COLORS["inferred"],
               capsize=3, ecolor="#B0B0B0", label="inferred")
        ax.bar(xs + 0.2, [TRUE.get(t, 0.0) for t in terms], width=0.4, fill=False,
               edgecolor=SFI_COLORS["exact"], lw=1.8, label="exact")
        ax.axhline(0, color="#808080", lw=0.6)
        ax.set_xticks(xs, [_tex(t) for t in terms])
        ax.set_title(EQ_NAMES[eq])
    axes[0].set_ylabel("coefficient")


fig, axes = plt.subplots(1, 4, figsize=(12, 3.8),
                         gridspec_kw={"width_ratios": [1.5, 0.8, 1, 1]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
coefficient_panels(inf, axes[1:])
axes[1].legend(loc="lower right")
fig.suptitle(f"PASTIS keeps {len(inf.force_support)} of {B.n_features} terms "
             "(error bars: 1σ)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# PASTIS (left) stops where the information gain flattens, keeping six
# terms with 1σ error bars.  The one departure from the true model comes
# from the slaving of :math:`y` to :math:`x`: the weak :math:`-y` in
# :math:`\dot y` is absorbed into the :math:`x` term, fitted near
# :math:`\rho - 1 = 7`.  Such near-collinear terms are interchangeable at
# this data length — on another 10-unit recording, PASTIS may pick
# :math:`y^2` for :math:`xy` — but the force field along the data stays
# accurate either way.
#
# Diagnostics: can we trust the fit?
# ----------------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.diagnose` recomputes
# the standardised residuals
# :math:`z_t = (\Delta x_t - \hat F(x_t)\,\Delta t)/\sqrt{2\hat D\,\Delta t}`
# and tests them: a well-specified model leaves i.i.d.
# :math:`\mathcal{N}(0,1)` residuals and a realised error consistent with
# the predicted one.  It also measures how coarse the sampling is for this
# force.

report = inf.diagnose()
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Residuals of the linear fit")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The residuals pass every test, but the ``finite_dt`` flag warns that
# :math:`\Delta t\,|\nabla F|` is not small (it is set by
# :math:`\sigma\Delta t = 0.1`): the linear estimators treat the force as
# constant over a sampling step, which biases them at order
# :math:`\Delta t`.  The flag suggests a cross-check with the parametric
# estimator :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
# which integrates the force over each step.  The data are noise-free, so
# we switch off its measurement-noise instrument (``eiv=False``).

inf_par = OverdampedLangevinInference(coll)
inf_par.infer_force(B, eiv=False)          # clean data: no noise instrument
inf_par.sparsify_force(criterion="PASTIS")
inf_par.compute_force_error()

# Change between the two fits, in units of their combined error bars.
c_lin, cov_lin = inf.coeff_block(slice(None))
c_par, cov_par = inf_par.coeff_block(slice(None))
err = np.sqrt(np.diag(cov_lin) + np.diag(cov_par))
kept = err > 0                             # terms kept by either fit
same = set(np.asarray(inf.force_support).tolist()) == set(np.asarray(inf_par.force_support).tolist())
print(f"same terms selected: {same}")
print(f"largest coefficient change: {np.max(np.abs(c_par - c_lin)[kept] / err[kept]):.2f} σ")

# %%
# The two estimators select the same terms and agree to a small fraction
# of an error bar: the finite-:math:`\Delta t` bias is negligible here, and
# the fast linear fit stands.
#
# Bootstrap: does the inferred model hop like the data?
# -----------------------------------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.simulate_bootstrapped_trajectory`
# runs the inferred model from the first recorded point, with the data's
# length and sampling.  The movie shows both in the :math:`(x, z)` plane,
# coloured by :math:`x` (blue near :math:`C_-`, red near :math:`C_+`).

coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(random.PRNGKey(99), oversampling=10)

# sphinx_gallery_start_ignore
fig_m, axes_m = dark_fig(1, 2, figsize=(6.6, 3.3))
anim = animate_particles_comparison(
    [coll, coll_boot], titles=["data", "inferred model"], dims=(0, 2),
    color_dim=0, cmap="coolwarm", vmin=-2 * x_focus, vmax=2 * x_focus,
    s=90, trail=120, skip=8, interval=70, axes=axes_m,
)
for ax in axes_m:
    for trail_line in ax.lines:
        trail_line.set(color="#9AA4B8", alpha=0.75, linewidth=1.0)
    ax.scatter(foci[:, 0], foci[:, 2], marker="X", s=45, color=SFI_COLORS["exact"], zorder=0)
    ax.set_xlim(-11, 11)
    ax.set_ylim(-1, 16)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
plt.show()
# sphinx_gallery_end_ignore

# %%
# One bootstrap is a single random realisation, so the comparison must be
# statistical.  We run the inferred model — and, since the data are
# synthetic, the true model — fifty times longer than the recording, and
# compare the distributions of :math:`x` and :math:`z` and the rate of
# focus-to-focus hops (:math:`x` travelling from one focus value
# :math:`\pm\sqrt{14}` to the other).


def hop_rate(c):
    """Focus-to-focus hops per unit time, with a Poisson error bar."""
    xs = np.asarray(c.to_array())[:, 0, 0]
    side = np.sign(xs) * (np.abs(xs) >= np.sqrt(14.0))      # +1 / -1 at a focus
    n_hops = np.count_nonzero(np.diff(side[side != 0]))
    T = len(xs) * dt
    return n_hops / T, np.sqrt(max(n_hops, 1)) / T


long_runs = {}
for name, p, seed in [("inferred model", proc_boot, 1), ("true model", proc, 2)]:
    p.initialize(jnp.array([1.0, 1.0, 1.0]))
    long_runs[name] = p.simulate(dt=dt, Nsteps=50 * Nsteps, key=random.PRNGKey(seed),
                                 prerun=2000, oversampling=10)

for name, c in [("data", coll), *long_runs.items()]:
    r, e = hop_rate(c)
    print(f"{name:15s} hops per unit time = {r:.2f} ± {e:.2f}")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, dim, lab in zip(axes, (0, 2), ("x", "z")):
    bins = np.linspace(-11, 11, 45) if dim == 0 else np.linspace(-1, 17, 37)
    ax.hist(np.asarray(coll.to_array())[:, 0, dim], bins=bins, density=True,
            color=SFI_COLORS["data"], alpha=0.45, label="data (10 time units)")
    for (name, c), col, ls in zip(long_runs.items(),
                                  (SFI_COLORS["inferred"], SFI_COLORS["exact"]), ("-", "--")):
        ax.hist(np.asarray(c.to_array())[:, 0, dim], bins=bins, density=True,
                histtype="step", color=col, ls=ls, lw=2, label=f"{name} (500 time units)")
    ax.set_xlabel(lab)
    ax.set_ylabel(f"$p({lab})$")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Stationary statistics: data vs long simulations")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The distributions and the hopping rate of the inferred model match
# those of the true dynamics within statistical errors.  The recording
# itself, with only two hops, agrees within its large counting error: 10
# time units are a small sample of this process, which is why the
# coefficients carry the error bars shown above.
#
# Takeaways
# ---------
#
# * A generic quadratic dictionary plus :term:`PASTIS` recovers the Lorenz
#   structure from 2000 frames of a strongly noise-driven trajectory, with
#   an error bar on every term; near-collinear terms are where model
#   selection is least certain.
# * The diagnostics do more than pass or fail: the ``finite_dt`` flag
#   called for a cross-check, and the parametric estimator confirmed the
#   fit.
# * Simulations of the inferred model validate it on what the fit never
#   used directly: the hopping statistics between the two foci.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
phase2d_scalar(coll, dims=(0, 2), color_fn=lambda X: X[:, 0], cmap="coolwarm",
               linewidth=1.4, plot_colorbar=False, ax=ax_t)
ax_t.scatter(foci[:, 0], foci[:, 2], marker="X", s=60, color=SFI_COLORS["exact"], zorder=3)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
