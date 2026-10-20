r"""
Inertial dissipation: entropy production from positions only
=============================================================

The driven optical trap of the overdamped entropy-production page, but
the bead now keeps its **inertia** — and the camera still records only
its positions.  How much does it dissipate, and can we tell from
positions alone?

As in the :doc:`overdamped demo </gallery/entropy_production_demo>`, a
curl force stirs the trapped bead; with inertia the state is
:math:`(\mathbf{x}, \mathbf{v})`, and the bath exerts friction and noise
on the velocity,

.. math::

   \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v},
   \qquad
   \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t}
   = \underbrace{-k\,\mathbf{x}
     + \varepsilon\,(\hat z \times \mathbf{x})}_{\text{trap + curl drive}}
   \;\underbrace{-\;\gamma\,\mathbf{v}}_{\text{friction}}
   \;+\; \sqrt{2D}\,\boldsymbol{\xi}(t),

and — crucially — **the velocities are never observed**.  Underdamped
Langevin Inference (ULI) reconstructs them, fits the full
acceleration field :math:`F(\mathbf{x},\mathbf{v})`, and splits it by
time-reversal parity into a *reversible* part :math:`F^{+}` (trap and
drive) and an *irreversible* part :math:`F^{-}` (friction).  The entropy
production follows from the fitted model, evaluated on the data.

Inertia is not a detail here.  For this linear system the exact rate is

.. math::

   \sigma(\gamma) \;=\; \frac{2\,\varepsilon^2\,\gamma}{\gamma^2 k - \varepsilon^2},

which *grows* as friction decreases — diverging at the stability
boundary :math:`\gamma \to \varepsilon/\sqrt{k}` where the curl
overcomes the trap — and only settles onto the overdamped result
:math:`2\varepsilon^2/(\gamma k)` at strong friction.  An overdamped
analysis of inertial data misses this amplification.

On this page we fit :math:`F(\mathbf{x},\mathbf{v})` from positions and
let PASTIS find the trap, curl and friction terms; split the fit by
time-reversal parity; **measure the entropy production** with
:meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`;
sweep the friction to watch inertia amplify dissipation; and coarsen the
sampling to see where the measurement stops being reliable.

.. rubric:: Tags

synthetic · underdamped · linear · 2D · non-equilibrium · thermodynamics
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "linear", "2D", "non-equilibrium", "thermodynamics"]
# sphinx_gallery_thumbnail_number = 7

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import phase2d, plot_pareto_front, plot_recovery_bar, stream_field

apply_style()
# sphinx_gallery_end_ignore
stamp_output()

# %%
# An inertial bead in the driven trap
# -----------------------------------
#
# The same trap and rotational drive as the overdamped page (a little
# stronger here, :math:`\varepsilon = 0.5`) — but the bead now carries
# momentum (think of a levitated microsphere in low-pressure gas rather
# than a colloid in water).  At :math:`\gamma = \sqrt{k}` (quality factor
# near one) the exact rate is already a third above the overdamped
# formula.
#
# The simulator integrates the phase-space dynamics, but the returned
# collection contains **positions only** — exactly what a camera sees.
# ``compute_observables=True`` makes it accumulate the ground-truth
# entropy along the way, using the true (otherwise discarded) velocities.

import SFI
from SFI.bases import unit_axes, v_components, x_components
from SFI.langevin import UnderdampedProcess

k_true = 1.0       # trap stiffness
eps_true = 0.5     # rotational drive:  F_curl = eps * (z_hat x x)
gamma_true = 1.0   # friction
D_true = 0.5       # velocity-space diffusion
dt = 0.01          # time between recorded frames
Nsteps = 320_000   # frames (3200 velocity relaxation times)

x0c, x1c = x_components(2)
v0c, v1c = v_components(2)
e0, e1 = unit_axes(2)


def make_trap(eps, gamma):
    """Trap + curl drive + friction, as a basis expression F(x, v)."""
    return (
        (-k_true * x0c - eps * x1c - gamma * v0c) * e0
        + (-k_true * x1c + eps * x0c - gamma * v1c) * e1
    )


def simulate_trap(eps, gamma, seed, *, Nsteps=Nsteps):
    proc = UnderdampedProcess(make_trap(eps, gamma), D=D_true)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    return proc.simulate(
        dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed),
        prerun=500, oversampling=8, compute_observables=True,
    )


def sigma_exact(eps, gamma):
    return 2.0 * eps**2 * gamma / (gamma**2 * k_true - eps**2)


coll = simulate_trap(eps_true, gamma_true, seed=3)
sigma = sigma_exact(eps_true, gamma_true)

print(f"{coll.T} frames at dt = {dt} (positions only)")
print(f"exact entropy production rate : {sigma:.4f} kB / time")
print(f"overdamped formula would give : {2 * eps_true**2 / (gamma_true * k_true):.4f} kB / time")

# sphinx_gallery_start_ignore
t_arr, X_plot, _ = coll.to_arrays(dataset=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
n_view = 4000
axes[0].plot(np.asarray(t_arr)[:n_view], np.asarray(X_plot)[:n_view, 0, 0],
             lw=0.7, color=SFI_COLORS["data"], alpha=0.9)
axes[0].axhline(0, ls=":", lw=0.5, color="#808080")
axes[0].set_xlabel("time")
axes[0].set_ylabel(r"$x_0$")
axes[0].set_title("Position vs time: smooth, inertial")

phase2d(coll, tmin=0, tmax=n_view, ax=axes[1], cmap="viridis",
        linewidth=0.5, alpha=0.85)
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlabel(r"$x_0$")
axes[1].set_ylabel(r"$x_1$")
axes[1].set_title("The bead's path")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Fit the acceleration field from positions
# -----------------------------------------
#
# The workflow mirrors the overdamped one; the engine handles the
# velocity reconstruction.  We hand ULI an over-complete polynomial basis
# in :math:`(\mathbf{x}, \mathbf{v})` — 30 candidate terms — and let
# PASTIS decide.  The data are clean and finely sampled, hence the
# ``"clean"`` preset and the ``WeakNoise`` diffusion estimator.

from SFI.bases import monomials_up_to

inf = SFI.UnderdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")

B = monomials_up_to(2, dim=2, include_constant=True, include_v=True, rank="vector")
inf.infer_force_linear(B, preset="clean")
inf.compute_force_error()

coeffs_full = np.asarray(inf.force_coefficients_full)
stderr_full = np.asarray(inf.force_coefficients_stderr)

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# %%
# PASTIS keeps exactly six terms: the isotropic trap, the two curl
# couplings and the two friction terms.  From here on we fit these six
# terms directly.

# sphinx_gallery_start_ignore
labels_full = list(B.labels)
coeffs_true_full = np.zeros(len(labels_full))
for name, val in {
    "x0·e0": -k_true, "x1·e1": -k_true,
    "x1·e0": -eps_true, "x0·e1": eps_true,
    "v0·e0": -gamma_true, "v1·e1": -gamma_true,
}.items():
    coeffs_true_full[labels_full.index(name)] = val

fig, (ax_p, ax) = plt.subplots(1, 2, figsize=(12, 3.4),
                               gridspec_kw={"width_ratios": [1, 2.6]})
plot_pareto_front(inf.force_sparsity_result, ax=ax_p)
plot_recovery_bar(
    coeffs_full, np.arange(len(labels_full)),
    coeffs_true=coeffs_true_full, support_true=np.arange(len(labels_full)),
    labels=labels_full, stderr=stderr_full, ax=ax,
)
ax.tick_params(axis="x", labelsize=6)
ax.set_title("30 candidate terms (full fit, ±1σ): trap, curl and friction stand out")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Split the dynamics by the arrow of time
# ---------------------------------------
#
# Under time reversal positions are even and velocities odd, so the
# fitted field splits into :math:`\hat F^{\pm}(\mathbf{x},\mathbf{v}) =
# \tfrac12[\hat F(\mathbf{x},\mathbf{v}) \pm \hat
# F(\mathbf{x},-\mathbf{v})]`.  Here :math:`\hat F^{+}` is the trap plus
# the curl drive (all positional) and :math:`\hat F^{-}` is the friction —
# the part that knows which way time flows.

F_even, F_odd = inf.time_reversal_split()

x_probe = jnp.array([[1.0, 0.0]])
v_probe = jnp.array([[0.0, 1.0]])
print("at x = (1, 0), v = (0, 1):")
print("  F       =", np.round(np.asarray(inf.force_inferred(x_probe, v=v_probe))[0], 3))
print("  F_even  =", np.round(np.asarray(F_even(x_probe, v_probe))[0], 3), "  (reversible)")
print("  F_odd   =", np.round(np.asarray(F_odd(x_probe, v_probe))[0], 3), "  (irreversible)")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

stream_field(coll, lambda X: np.asarray(F_even(jnp.asarray(X), jnp.zeros_like(jnp.asarray(X)))),
             ax=axes[0], N=24, density=0.8,
             color=SFI_COLORS["inferred"], linewidth=0.8, arrowsize=0.7)
axes[0].set_title(r"$\hat F^{+}(\mathbf{x}, 0)$: trap + drive (even)")
axes[0].set_xlabel(r"$x_0$")
axes[0].set_ylabel(r"$x_1$")
axes[0].set_aspect("equal")

v_line = np.linspace(-2.5, 2.5, 100)
Fodd_line = np.asarray(F_odd(
    jnp.zeros((100, 2)), jnp.stack([jnp.asarray(v_line), jnp.zeros(100)], axis=-1)
))[:, 0]
axes[1].plot(v_line, Fodd_line, color=SFI_COLORS["inferred"], lw=2.4,
             label=r"inferred $\hat F^{-}_0(0, v)$")
axes[1].plot(v_line, -gamma_true * v_line, ":", lw=1.6, color=SFI_COLORS["exact"],
             label=rf"exact $-\gamma v$ ($\gamma={gamma_true}$)")
axes[1].axhline(0, ls=":", lw=0.5, color="#808080")
axes[1].axvline(0, ls=":", lw=0.5, color="#808080")
axes[1].set_xlabel(r"$v_0$")
axes[1].set_ylabel(r"$\hat F^{-}_0$")
axes[1].set_title(r"$\hat F^{-}$: friction (odd)")
axes[1].legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Measure the dissipation — from positions only
# ---------------------------------------------
#
# :meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`
# evaluates
#
# .. math::
#
#    \Delta\hat S = \sum_t \bigl(\mathrm{d}\hat v_t - \hat F^{+}\,\mathrm{d}t\bigr)^{\!\top}
#    \circ\, \bar D^{-1} \hat F^{-}
#
# on the reconstructed kinematics, with a counterterm for their
# finite-:math:`\Delta t` biases; :math:`\circ` denotes the Stratonovich
# (midpoint) product.  Its error bar is set by the fluctuations of the odd
# (friction) sector, which are large even though its *mean*
# irreversibility is small.
#
# One more subtlety: evaluating the functional with coefficients fitted
# **on the same trajectory** adds a small positive :math:`O(1/\tau_N)`
# plug-in bias.  **Cross-fitting** — fit one half, evaluate on the other,
# via the ``coefficients`` argument — removes it.

B_sel = B[[int(i) for i in inf.force_support]]    # the six PASTIS-selected terms


def entropy_of(coll_run):
    """Fit the selected terms on a recording and measure its dissipation."""
    inf_run = SFI.UnderdampedLangevinInference(coll_run)
    inf_run.compute_diffusion_constant(method="WeakNoise")
    inf_run.infer_force_linear(B_sel, preset="clean")
    return inf_run.compute_entropy_production()


out = entropy_of(coll)

# Cross-fitting: fit each half, evaluate on the other one.
from SFI import TrajectoryCollection

X_ud = coll.to_array()                            # positions, (T, 1, 2)
T_half = coll.T // 2
halves = [TrajectoryCollection.from_arrays(X=X_ud[:T_half], dt=dt),
          TrajectoryCollection.from_arrays(X=X_ud[T_half:], dt=dt)]
fits = []
for coll_h in halves:
    inf_h = SFI.UnderdampedLangevinInference(coll_h)
    inf_h.compute_diffusion_constant(method="WeakNoise")
    inf_h.infer_force_linear(B_sel, preset="clean")
    fits.append(inf_h)
out_xf = [fits[1].compute_entropy_production(coefficients=fits[0].force_coefficients_full),
          fits[0].compute_entropy_production(coefficients=fits[1].force_coefficients_full)]
Sdot_xf = 0.5 * (out_xf[0]["Sdot"] + out_xf[1]["Sdot"])
err_xf = 0.5 * (out_xf[0]["Sdot_error"] ** 2 + out_xf[1]["Sdot_error"] ** 2) ** 0.5

S_sim = coll.datasets[0].meta["observables"]["entropy"]   # simulator ground truth
print(f"exact rate                    : {sigma:.4f} kB / time")
print(f"inferred (same-sample)        : {out['Sdot']:.4f} ± {out['Sdot_error']:.4f}")
print(f"inferred (cross-fitted)       : {Sdot_xf:.4f} ± {err_xf:.4f}")
print(f"true model on this trajectory : {S_sim / out['tauN']:.4f}")

# %%
# Both estimates agree with the exact rate within their error bars, and
# with the entropy the *true* model produced along this very trajectory.
# At this recording length (thousands of relaxation times) the plug-in
# bias is far below the error bar; cross-fitting is the safeguard for
# short recordings.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.8, 3.2))
vals = [out["Sdot"], Sdot_xf, S_sim / out["tauN"]]
errs = [out["Sdot_error"], err_xf, 0.0]
names = ["same-sample", "cross-fitted", "true model\n(same run)"]
colors = [SFI_COLORS["inferred"], SFI_COLORS["highlight"], SFI_COLORS["bootstrap"]]
ax.axhline(sigma, ls=":", lw=1.2, color=SFI_COLORS["exact"],
           label=rf"exact $\sigma = {sigma:.3f}$")
ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="none", capsize=4,
            ecolor="#B0B0B0")
for i, (v, c) in enumerate(zip(vals, colors)):
    ax.plot([i], [v], "o", ms=8, color=c)
ax.set_xticks(range(len(vals)), names)
ax.set_xlim(-0.5, len(vals) - 0.5)
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_title("Dissipation from positions only")
ax.legend(loc="upper right")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inertia amplifies dissipation
# -----------------------------
#
# Now sweep the friction at fixed drive, on shorter runs (800 time units
# each).  The overdamped formula
# :math:`2\varepsilon^2/(\gamma k)` (dashed) already grows as
# :math:`\gamma` decreases; inertia adds to it: the exact rate pulls away
# and **diverges** at the stability boundary :math:`\gamma =
# \varepsilon/\sqrt{k}`, where the curl overcomes the trap.  The
# estimator, still fed positions only, tracks the exact curve.

gamma_sweep = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
sweep = {gam: entropy_of(simulate_trap(eps_true, gam, seed=20 + i, Nsteps=80_000))
         for i, gam in enumerate(gamma_sweep)}

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6.2, 3.6))
g_fine = np.geomspace(0.62, 6.0, 300)
ax.plot(g_fine, sigma_exact(eps_true, g_fine), ":", lw=1.6,
        color=SFI_COLORS["exact"], label=r"exact $2\varepsilon^2\gamma/(\gamma^2 k - \varepsilon^2)$")
ax.plot(g_fine, 2 * eps_true**2 / (g_fine * k_true), "--", lw=1.2,
        color="#808080", label=r"overdamped $2\varepsilon^2/(\gamma k)$")
ax.axvline(eps_true / k_true**0.5, color=SFI_COLORS["error"], lw=1.0, alpha=0.5)
ax.annotate("stability\nboundary", xy=(eps_true / k_true**0.5, 1.5),
            xytext=(0.85, 1.6), fontsize=8, color=SFI_COLORS["error"],
            arrowprops=dict(arrowstyle="->", color=SFI_COLORS["error"], lw=0.8))
g_arr = np.array(gamma_sweep)
sd = np.array([sweep[g]["Sdot"] for g in gamma_sweep])
se = np.array([sweep[g]["Sdot_error"] for g in gamma_sweep])
ax.errorbar(g_arr, sd, yerr=se, fmt="o", ms=5.5, capsize=3,
            color=SFI_COLORS["inferred"], label="inferred (positions only)")
ax.set_xlabel(r"friction $\gamma$")
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3, 5])
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_title("Inertia amplifies dissipation")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# When sampling is too coarse
# ---------------------------
#
# Everything above resolved the velocity correlation time
# :math:`\tau_v = 1/\gamma`.  Subsampling the recording stretches the
# effective :math:`\Delta t`.  The entropy functional itself carries a
# finite-:math:`\Delta t` counterterm, but the fitted *coefficients* keep
# an :math:`O(\gamma\Delta t)` error — the curl comes out too strong —
# which the dissipation amplifies, well outside the error bars (these
# describe sampling noise only).  The method reports a first-order
# estimate of this coefficient bias as ``Sdot_dt_bias``.

resolution = [(gamma_true * dt, out)]               # stride 1: the measurement above
for s in [10, 20, 40]:
    coll_s = TrajectoryCollection.from_arrays(X=X_ud[::s], dt=dt * s)
    resolution.append((gamma_true * dt * s, entropy_of(coll_s)))

for gdt, o in resolution:
    print(f"gamma*dt = {gdt:4.2f}:  Sdot = {o['Sdot']:.3f} ± {o['Sdot_error']:.3f}"
          f"  = {o['Sdot'] / sigma:.2f} x exact;   reported dt bias {o['Sdot_dt_bias']:+.3f}")

# %%
# The estimate climbs by about 20% at :math:`\gamma\Delta t = 0.1` and
# more than doubles by :math:`0.4`.  Subtracting the reported bias (open
# circles) brings it back to the exact rate up to
# :math:`\gamma\Delta t \approx 0.1`; beyond, the first-order estimate
# over-corrects.  Read it as a flag, not a correction: a reported bias
# comparable to the rate itself means the sampling is too coarse.
# **Inertial dissipation needs inertia resolved**, :math:`\gamma\Delta t
# \ll 1`.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.8, 3.4))
gdt_arr = np.array([r[0] for r in resolution])
sd_r = np.array([r[1]["Sdot"] for r in resolution])
se_r = np.array([r[1]["Sdot_error"] for r in resolution])
ax.axhline(sigma, ls=":", lw=1.4, color=SFI_COLORS["exact"], label=rf"exact $\sigma = {sigma:.3f}$")
ax.errorbar(gdt_arr, sd_r, yerr=se_r, fmt="o", ms=5.5, capsize=3,
            color=SFI_COLORS["inferred"], label="inferred (subsampled)")
ax.plot(gdt_arr, [r[1]["Sdot"] - r[1]["Sdot_dt_bias"] for r in resolution], "o", ms=5.5,
        mfc="none", color=SFI_COLORS["inferred"], label=r"minus reported $\Delta t$ bias")
ax.set_xscale("log")
ax.set_xticks(gdt_arr)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_ylim(0, None)
ax.set_xlabel(r"sampling coarseness $\gamma\,\Delta t$")
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_title(r"Velocities must be resolved: $\gamma\Delta t \ll 1$")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Next steps
# ----------
#
# - **The overdamped companion page**: the same trap without inertia,
#   the near-equilibrium detection limit, and the parametric energy
#   fit — :doc:`/gallery/entropy_production_demo`.
# - **Theory**: the odd-parity split, the estimator's error bars and the
#   cross-fitting recipe — :doc:`/inference/stochastic_thermodynamics`.
# - **Driven or inertial?**  The dynamics-order classifier tells a
#   rotational current from momentum — :doc:`/gallery/dynamics_order_demo`.
# - **Underdamped inference in depth**: presets, kinematics and
#   diagnostics — :doc:`/inference/underdamped`.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
ax_t.plot(g_fine, sigma_exact(eps_true, g_fine), ":", lw=2.0, color=SFI_COLORS["exact"])
ax_t.plot(g_fine, 2 * eps_true**2 / (g_fine * k_true), "--", lw=1.4, color="#808080")
ax_t.axvline(eps_true / k_true**0.5, color=SFI_COLORS["error"], lw=1.2, alpha=0.6)
ax_t.errorbar(g_arr, sd, yerr=se, fmt="o", ms=7, capsize=3, color=SFI_COLORS["inferred"])
ax_t.set_xscale("log")
ax_t.xaxis.set_major_locator(mticker.NullLocator())
ax_t.xaxis.set_minor_locator(mticker.NullLocator())
ax_t.set_yticks([])
ax_t.set_xlabel("")
ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
