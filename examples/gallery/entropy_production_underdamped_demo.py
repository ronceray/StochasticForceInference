"""
Inertial dissipation: entropy production from positions only
=============================================================

The :doc:`overdamped entropy-production demo </gallery/entropy_production_demo>`
measured the dissipation of a driven optical trap.  Here the same bead
keeps its **inertia**: the state is :math:`(\\mathbf{x}, \\mathbf{v})`,
the bath exerts friction and noise on the velocity,

.. math::

   \\frac{\\mathrm{d}\\mathbf{x}}{\\mathrm{d}t} = \\mathbf{v},
   \\qquad
   \\frac{\\mathrm{d}\\mathbf{v}}{\\mathrm{d}t}
   = \\underbrace{-k\\,\\mathbf{x}
     + \\varepsilon\\,(\\hat z \\times \\mathbf{x})}_{\\text{trap + curl drive}}
   \\;\\underbrace{-\\;\\gamma\\,\\mathbf{v}}_{\\text{friction}}
   \\;+\\; \\sqrt{2D}\\,\\boldsymbol{\\xi}(t),

and — crucially — **only positions are observed**.  Underdamped
Langevin Inference reconstructs the velocities, fits the full
acceleration field :math:`F(\\mathbf{x},\\mathbf{v})`, and splits it by
time-reversal parity (:math:`\\mathbf{x}` even, :math:`\\mathbf{v}`
odd): a *reversible* part :math:`F^{+}` (trap and drive) and an
*irreversible* part :math:`F^{-}` (friction).  The entropy production
is then the Stratonovich log path-probability ratio of the fitted
model, evaluated on the data.

Inertia is not a detail here.  For this linear system the exact rate is

.. math::

   \\sigma(\\gamma) \\;=\\; \\frac{2\\,\\varepsilon^2\\,\\gamma}{\\gamma^2 k - \\varepsilon^2},

which *grows* as friction decreases — diverging at the stability
boundary :math:`\\gamma \\to \\varepsilon/\\sqrt{k}` where the curl
overcomes the trap — and only settles onto the overdamped result
:math:`2\\varepsilon^2/(\\gamma k)` at strong friction.  Analysing
inertial data with an overdamped model misses this amplification.

Across this page we will:

#. simulate the inertial driven trap, recording positions only;
#. fit :math:`F(\\mathbf{x},\\mathbf{v})` with ULI and let PASTIS find
   the trap, curl and friction terms;
#. split the fit by time-reversal parity with
   :meth:`~SFI.inference.UnderdampedLangevinInference.time_reversal_split`;
#. **measure the entropy production** with
   :meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`,
   cross-check it against the simulator's ground truth, and remove the
   same-sample plug-in bias by cross-fitting;
#. sweep the friction: inertia amplifies dissipation, and the estimator
   tracks the exact :math:`\\sigma(\\gamma)`;
#. push the sampling interval past the velocity correlation time and
   watch the reconstruction fail — the :math:`\\Delta t \\lesssim \\tau_v`
   prerequisite.

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
from SFI.utils.plotting import phase2d, plot_recovery_bar, stream_field

apply_style()
# sphinx_gallery_end_ignore

# %%
# An inertial bead in the driven trap
# -----------------------------------
#
# Same trap, same faint rotational drive as the overdamped page — but the
# bead now carries momentum (think a levitated microsphere in low-pressure
# gas rather than a colloid in water).  We pick a moderately underdamped
# regime, :math:`\gamma = \sqrt{k}` (quality factor near one), where the
# exact rate :math:`\sigma = 2\varepsilon^2\gamma/(\gamma^2 k -
# \varepsilon^2)` is already **1.33×** the overdamped formula.
#
# The simulator integrates the phase-space dynamics but the returned
# collection contains **positions only** — exactly what a camera sees.
# ``compute_observables=True`` makes it accumulate the ground-truth
# information and entropy along the way, using the true internal
# velocities (which are otherwise discarded).

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


def simulate_trap(eps, gamma, seed, *, dt=dt, Nsteps=Nsteps, oversampling=8):
    proc = UnderdampedProcess(make_trap(eps, gamma), D=D_true)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    coll_run = proc.simulate(
        dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed),
        prerun=500, oversampling=oversampling, compute_observables=True,
    )
    return coll_run, proc


def sigma_exact(eps, gamma):
    return 2.0 * eps**2 * gamma / (gamma**2 * k_true - eps**2)


coll, proc_true = simulate_trap(eps_true, gamma_true, seed=3)

print(f"Simulated {coll.T} frames at dt = {dt} (positions only)")
print(f"Exact entropy production rate: sigma = {sigma_exact(eps_true, gamma_true):.4f} kB / time")
print(f"Overdamped formula would give:  {2 * eps_true**2 / (gamma_true * k_true):.4f} kB / time")

# sphinx_gallery_start_ignore
t_arr, X_ud, _ = coll.to_arrays(dataset=0)
X_ud = np.asarray(X_ud)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
n_view = 4000
axes[0].plot(np.asarray(t_arr)[:n_view], X_ud[:n_view, 0, 0],
             lw=0.7, color=SFI_COLORS["data"], alpha=0.9)
axes[0].axhline(0, ls=":", lw=0.5, color="#808080")
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"$x_0$")
axes[0].set_title("Position vs time: smoother than overdamped")

phase2d(coll, tmin=0, tmax=n_view, ax=axes[1], cmap="viridis",
        linewidth=0.5, alpha=0.85)
axes[1].set_xlabel(r"$x_0$")
axes[1].set_ylabel(r"$x_1$")
axes[1].set_aspect("equal")
axes[1].set_title("The bead's path: inertial loops")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Fit the acceleration field from positions
# -----------------------------------------
#
# The workflow mirrors the overdamped one — the engine transparently
# handles the velocity reconstruction.  We hand ULI an over-complete
# polynomial basis in :math:`(\mathbf{x}, \mathbf{v})` and let PASTIS
# decide: it should keep the linear trap terms, the faint curl, and the
# friction — and nothing else.

from SFI.bases import monomials_up_to

inf = SFI.UnderdampedLangevinInference(coll)
inf.compute_diffusion_constant()

B = monomials_up_to(2, dim=2, include_constant=True, include_v=True, rank="vector")
inf.infer_force_linear(B)
inf.compute_force_error()

coeffs_full = np.asarray(inf.force_coefficients_full)
stderr_full = np.asarray(inf.force_coefficients_stderr)

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# sphinx_gallery_start_ignore
labels_full = list(B.labels)
coeffs_true_full = np.zeros(len(labels_full))
for name, val in {
    "x0·e0": -k_true, "x1·e1": -k_true,
    "x1·e0": -eps_true, "x0·e1": eps_true,
    "v0·e0": -gamma_true, "v1·e1": -gamma_true,
}.items():
    coeffs_true_full[labels_full.index(name)] = val

fig, ax = plt.subplots(figsize=(9.5, 3.2))
plot_recovery_bar(
    coeffs_full, np.arange(len(labels_full)),
    coeffs_true=coeffs_true_full, support_true=np.arange(len(labels_full)),
    labels=labels_full, stderr=stderr_full, ax=ax,
)
ax.tick_params(axis="x", labelsize=6)
ax.set_title("30 candidate terms: trap, curl and friction recovered")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Split the dynamics by the arrow of time
# ---------------------------------------
#
# Under time reversal positions are even and velocities odd, so the
# fitted field splits into :math:`\hat F^{\pm}(\mathbf{x},\mathbf{v}) =
# \tfrac12[\hat F(\mathbf{x},\mathbf{v}) \pm \hat
# F(\mathbf{x},-\mathbf{v})]`.  For this system :math:`\hat F^{+}` is
# the trap plus the curl drive (all positional) and :math:`\hat F^{-}`
# is the friction — the part that knows which way time flows.

F_even, F_odd = inf.time_reversal_split()

x_probe = jnp.array([[1.0, 0.0]])
v_probe = jnp.array([[0.0, 1.0]])
print("F      (x=(1,0), v=(0,1)) =", np.asarray(inf.force_inferred(x_probe, v=v_probe))[0])
print("F_even (reversible part)  =", np.asarray(F_even(x_probe, v_probe))[0])
print("F_odd  (irreversible part)=", np.asarray(F_odd(x_probe, v_probe))[0])

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
axes[1].plot(v_line, Fodd_line, color=SFI_COLORS["inferred"], lw=1.8,
             label=r"$\hat F^{-}_0(0, v)$")
axes[1].plot(v_line, -gamma_true * v_line, ":", lw=1.4, color=SFI_COLORS["exact"],
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
# evaluates :math:`\Delta\hat S = \sum_t (\mathrm{d}\hat v_t \circ - \hat
# F^{+}\mathrm{d}t)^\top \bar D^{-1} \hat F^{-}` with the reconstructed
# kinematics.  Two subtleties, both reported by the method:
#
# - friction makes the odd sector *informative but barely irreversible*:
#   the log ratio's contributions largely cancel, so the error bar is set
#   by the odd-sector fluctuation scale (``Q_odd``), estimated
#   analytically and by block variance;
# - evaluating the functional with coefficients fitted **on the same
#   trajectory** adds a small positive :math:`O(1/\tau_N)` plug-in bias.
#   **Cross-fitting** — fit on one half, evaluate on the other — removes
#   it, via the ``coefficients`` argument.

out = inf.compute_entropy_production()

S_sim = coll.datasets[0].meta["observables"]["entropy"]
tauN = out["tauN"]
sigma = sigma_exact(eps_true, gamma_true)

# Cross-fitting: two halves, two fits, evaluate each on the other half.
from SFI import TrajectoryCollection

X_all = jnp.asarray(X_ud)
T_half = X_all.shape[0] // 2
halves = [
    TrajectoryCollection.from_arrays(X=X_all[:T_half], dt=dt),
    TrajectoryCollection.from_arrays(X=X_all[T_half:], dt=dt),
]
fits = []
for coll_h in halves:
    inf_h = SFI.UnderdampedLangevinInference(coll_h)
    inf_h.compute_diffusion_constant()
    inf_h.infer_force_linear(B)
    fits.append(inf_h)
out_xf = [
    fits[1].compute_entropy_production(coefficients=fits[0].force_coefficients_full),
    fits[0].compute_entropy_production(coefficients=fits[1].force_coefficients_full),
]
Sdot_crossfit = 0.5 * (out_xf[0]["Sdot"] + out_xf[1]["Sdot"])
err_crossfit = 0.5 * (out_xf[0]["Sdot_error"] ** 2 + out_xf[1]["Sdot_error"] ** 2) ** 0.5

print(f"exact rate                    : {sigma:.4f} kB / time")
print(f"inferred (same-sample)        : {out['Sdot']:.4f} +- {out['Sdot_error']:.4f}"
      f"   [odd dof bias {out['Sdot_bias']:.4f}]")
print(f"inferred (cross-fitted)       : {Sdot_crossfit:.4f} +- {err_crossfit:.4f}")
print(f"true model on this trajectory : {S_sim / tauN:.4f}")
print(f"irreversible fraction of the odd sector: {out['entropy_odd_ratio']:.3f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.8, 3.2))
vals = [out["Sdot"], Sdot_crossfit, S_sim / tauN]
errs = [out["Sdot_error"], err_crossfit, 0.0]
names = ["same-sample\nplug-in", "cross-fitted", "true model\n(same run)"]
colors = [SFI_COLORS["inferred"], SFI_COLORS["highlight"], SFI_COLORS["bootstrap"]]
ax.axhline(sigma, ls=":", lw=1.2, color=SFI_COLORS["exact"],
           label=rf"exact $\sigma = {sigma:.3f}$")
ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="none", capsize=4,
            ecolor="#B0B0B0")
for i, (v, c) in enumerate(zip(vals, colors)):
    ax.plot([i], [v], "o", ms=8, color=c)
ax.set_xticks(range(len(vals)), names)
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_title("Dissipation from positions only")
ax.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inertia amplifies dissipation
# -----------------------------
#
# Now sweep the friction at fixed drive.  Watch the exact rate: far from
# the overdamped intuition that less friction means less dissipation,
# :math:`\sigma(\gamma)` **grows** as :math:`\gamma` decreases — the
# freely-circulating bead extracts more work from the drive — and
# diverges at the stability boundary :math:`\gamma = \varepsilon/\sqrt{k}`.
# The estimator, still fed positions only, tracks the exact curve; the
# overdamped formula (dashed) is off by 30% already at :math:`\gamma = 1`.

B_lin = monomials_up_to(1, dim=2, include_constant=False, include_v=True, rank="vector")


def entropy_of(coll_run):
    inf_run = SFI.UnderdampedLangevinInference(coll_run)
    inf_run.compute_diffusion_constant()
    inf_run.infer_force_linear(B_lin)
    return inf_run.compute_entropy_production()


gamma_sweep = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
sweep = {}
for i, gam in enumerate(gamma_sweep):
    coll_g, _ = simulate_trap(eps_true, gam, seed=20 + i, Nsteps=80_000)
    sweep[gam] = entropy_of(coll_g)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6.2, 3.6))
g_fine = np.linspace(0.62, 6.0, 300)
ax.plot(g_fine, sigma_exact(eps_true, g_fine), ":", lw=1.6,
        color=SFI_COLORS["exact"], label=r"exact $2\varepsilon^2\gamma/(\gamma^2 k - \varepsilon^2)$")
ax.plot(g_fine, 2 * eps_true**2 / (g_fine * k_true), "--", lw=1.2,
        color="#808080", label=r"overdamped limit $2\varepsilon^2/(\gamma k)$")
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
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_title("Inertia amplifies dissipation")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# When sampling is too coarse
# ---------------------------
#
# Everything above relied on resolving the velocity correlation time
# :math:`\tau_v = 1/\gamma`.  Subsampling the same recording stretches
# the effective :math:`\Delta t`; beyond :math:`\gamma\Delta t \approx
# 0.5` the secant velocities stop resembling velocities, the estimator
# (which warns at exactly that threshold) degrades, and the measurement
# slides toward a coarse-grained lower bound.  There is no free lunch:
# **inertial dissipation is only measurable when inertia is resolved.**

strides = [1, 5, 10, 20, 40, 80]
resolution = []
for s in strides:
    coll_s = TrajectoryCollection.from_arrays(X=X_all[::s], dt=dt * s)
    resolution.append((gamma_true * dt * s, entropy_of(coll_s)))

for gdt, o in resolution:
    print(f"gamma*dt = {gdt:5.2f}:  Sdot = {o['Sdot']:7.4f} +- {o['Sdot_error']:.4f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.8, 3.4))
gdt_arr = np.array([r[0] for r in resolution])
sd_r = np.array([r[1]["Sdot"] for r in resolution])
se_r = np.array([r[1]["Sdot_error"] for r in resolution])
ax.axhline(sigma, ls=":", lw=1.4, color=SFI_COLORS["exact"], label=rf"exact $\sigma = {sigma:.3f}$")
ax.axvspan(0.5, gdt_arr.max() * 1.4, color=SFI_COLORS["error"], alpha=0.08)
ax.axvline(0.5, color=SFI_COLORS["error"], lw=1.0, alpha=0.5)
ax.errorbar(gdt_arr, sd_r, yerr=se_r, fmt="o", ms=5.5, capsize=3,
            color=SFI_COLORS["inferred"], label="inferred (subsampled)")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel(r"sampling coarseness $\gamma\,\Delta t$")
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_title(r"Velocities must be resolved: $\Delta t \lesssim \tau_v$")
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
# - **Theory**: the odd-parity split, the estimator's provisional error
#   bars and the cross-fitting recipe —
#   :doc:`/inference/stochastic_thermodynamics`.
# - **Driven or inertial?**  The dynamics-order classifier tells a
#   rotational current from momentum —
#   :doc:`/gallery/dynamics_order_demo`.
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

stamp_output()
