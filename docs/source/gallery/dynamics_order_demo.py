r"""
Overdamped or underdamped? Classifying dynamics from data
=========================================================

Before fitting a force, you must choose an *engine*:
:class:`~SFI.OverdampedLangevinInference` assumes first-order (overdamped)
dynamics, :class:`~SFI.UnderdampedLangevinInference` second-order (inertial)
dynamics.  :func:`SFI.classify_dynamics` decides from the positions alone —
no velocities needed, and robust to localisation noise.

It reads the lag-resolved displacement covariance
:math:`C_k = \langle \Delta x_t \cdot \Delta x_{t+k}\rangle` at several
sampling steps :math:`\Delta t`.  Two facts make this robust:

1. **White localisation noise touches only** :math:`C_0` **and** :math:`C_1`
   (it adds :math:`+2\sigma^2` and :math:`-\sigma^2`, and **exactly 0** at
   lag :math:`\ge 2`), so lag-2 statistics are immune to it.
2. **A force cannot fake inertia**: scanning :math:`\Delta t` separates the
   overdamped force confound (which vanishes as :math:`\Delta t\to0`) from
   genuine momentum persistence (which saturates).

We classify an overdamped double well and an underdamped oscillator under
the same heavy localisation noise, then two cases the method must handle
honestly: a coarsely sampled inertial particle (verdict ``inconclusive``)
and a driven, non-equilibrium 2D system that is still overdamped.

.. rubric:: Tags

diagnostics · overdamped · underdamped · 1D · 2D · non-equilibrium · synthetic
"""

# sphinx_gallery_tags = ["diagnostics", "overdamped", "underdamped", "1D", "2D", "non-equilibrium", "synthetic"]
# sphinx_gallery_thumbnail_number = 6

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.ticker import NullFormatter

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import phase2d, stream_field

apply_style()


def _tidy_log_axes(fig):
    """Hide minor tick labels on the log Δt axes (they overlap)."""
    for ax in fig.axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter())
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Two noisy recordings
# --------------------
#
# The overdamped process is a double well :math:`F(x) = x - x^3`; the
# underdamped one a damped harmonic oscillator :math:`\dot v = -k x -
# \gamma v + \sqrt{2D}\,\xi` with momentum relaxation time
# :math:`\tau_v = 1/\gamma = 1`.  Both are sampled at :math:`\Delta t = 0.02`,
# fine enough to resolve momentum if there is any, and both are degraded with
# the *same* localisation noise :math:`\sigma = 0.06`.

from SFI import classify_dynamics
from SFI.bases import unit_axes, v_components, x_components
from SFI.diagnostics import plot_dynamics_order
from SFI.langevin import OverdampedProcess, UnderdampedProcess

(x,) = x_components(1)
(v,) = v_components(1)
(ex,) = unit_axes(1)

# Overdamped double well, F(x) = x - x^3.
proc_od = OverdampedProcess((x - x * x * x) * ex, D=0.15)
proc_od.initialize(jnp.array([0.5]))
coll_od = proc_od.simulate(dt=0.02, Nsteps=12000, key=random.PRNGKey(0), prerun=200,
                           oversampling=10)

# Underdamped damped harmonic oscillator, F(x, v) = -k x - gamma v.
k, gamma = 1.0, 1.0
proc_ud = UnderdampedProcess((-k * x - gamma * v) * ex, D=1.0)
proc_ud.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))
coll_ud = proc_ud.simulate(dt=0.02, Nsteps=12000, key=random.PRNGKey(1), prerun=200,
                           oversampling=10)

sigma = 0.06
coll_od_noisy = coll_od.degrade(noise=sigma, seed=10)
coll_ud_noisy = coll_ud.degrade(noise=sigma, seed=11)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, clean, noisy, name in [(axes[0], coll_od, coll_od_noisy, "overdamped double well"),
                               (axes[1], coll_ud, coll_ud_noisy, "underdamped oscillator")]:
    t, Xc, _ = clean.to_arrays()
    _, Xn, _ = noisy.to_arrays()
    win = slice(2000, 2075)                     # 1.5 time units
    ax.plot(t[win], np.diff(Xn[:, 0, 0])[win], ".-", ms=5, lw=0.5,
            color=SFI_COLORS["data"], label="recorded increments (noisy)")
    ax.plot(t[win], np.diff(Xc[:, 0, 0])[win], color=SFI_COLORS["exact"], lw=1.8,
            label="increments of the true path")
    ax.axhline(0, color="#808080", lw=0.6)
    ax.set_title(name)
    ax.set_xlabel("time")
axes[0].set_ylabel(r"$\Delta x_t = x_{t+\Delta t} - x_t$")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Inertia makes increments persistent; localisation noise swamps them")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inertia shows up as *persistent* increments: those of the true
# oscillator path (right, orange) vary smoothly from one frame to the next,
# while the overdamped ones (left) are white.  But the localisation noise
# adds increments of typical size :math:`\sqrt2\,\sigma \approx 0.085`,
# several times the oscillator's own, so the recorded increments (blue)
# look equally erratic in both cases.  Any test built on velocities or
# single-step increments is blind here.
#
# Classify
# --------
#
# :func:`~SFI.diagnostics.classify_dynamics` runs the noise-immune scaling
# test, a parametric fit of the covariances (estimating :math:`\tau_v`,
# :math:`\sigma` and :math:`D`), and a cross-check on the residual
# autocorrelation of an overdamped fit.

report_od = classify_dynamics(coll_od_noisy)
report_ud = classify_dynamics(coll_ud_noisy)

print("### Overdamped double well (with localisation noise) ###")
report_od.print_summary()
print("\n### Underdamped damped oscillator (with localisation noise) ###")
report_ud.print_summary()

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
for row, rep, name in [(axes[0], report_od, "Double well"), (axes[1], report_ud, "Oscillator")]:
    plot_dynamics_order(rep, axes=row)
    row[0].set_title(f"{name}: {row[0].get_title()}")
_tidy_log_axes(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Top: the double well.  The lag-2 persistence :math:`\rho_2` stays near
# zero as :math:`\Delta t` shrinks, and the apparent kinetic energy grows as
# :math:`1/\Delta t` (slope :math:`-1`): a rough, non-differentiable path.
# Bottom: the oscillator.  :math:`\rho_2` rises to a positive plateau
# (momentum persistence) and the apparent kinetic energy flattens (slope
# :math:`\to 0`): a smooth path.  The localisation noise, confined to lags 0
# and 1, never enters these verdicts.
#
# Coarse sampling: an honest ``inconclusive``
# -------------------------------------------
#
# The underdamped verdict needs momentum to be *resolved*: sampled finely
# compared with :math:`\tau_v`.  Here is a **freely diffusing inertial
# particle** (velocity relaxing at rate :math:`\gamma = 1`, no confining
# force — the classifier's covariance model exactly) sampled at
# :math:`\Delta t = 0.7`, so :math:`\gamma\,\Delta t \approx 0.7` and
# momentum is barely half-resolved.  At such coarse sampling the velocity
# signal is weak, so many samples (500 000) are needed before the fit can
# detect inertia at all.

proc_free = UnderdampedProcess((-gamma * v) * ex, D=1.0)   # F = -γv, no confinement
proc_free.initialize(jnp.zeros(1), v0=jnp.zeros(1))
coll_coarse = proc_free.simulate(dt=0.7, Nsteps=500_000, key=random.PRNGKey(1), prerun=500,
                                 oversampling=10)

# The cross-check refits a full overdamped model; skip it on this large
# dataset — the verdict rests on the scaling test and the parametric fit.
report_inc = classify_dynamics(coll_coarse, cross_check=False)
print("### Freely diffusing inertial particle, coarsely sampled ###")
report_inc.print_summary()

# sphinx_gallery_start_ignore
fig_inc = plot_dynamics_order(report_inc)
fig_inc.set_size_inches(11, 3.8)
_tidy_log_axes(fig_inc)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The classifier **abstains** rather than guess.  The parametric fit does
# detect inertia (a large AICc gain), but at the finest available step the
# noise-immune lag-2 persistence has already decayed into the gray zone
# between the overdamped and underdamped signatures.  The summary reports
# :math:`\gamma\,\Delta t` and advises sampling finer than :math:`\tau_v`.
#
# Beyond 1D: a driven, non-equilibrium 2D system
# ----------------------------------------------
#
# The test pools spatial components, so it works in any dimension.  Here
# is a 2D **overdamped** particle in a harmonic trap with a rotational
# force, :math:`\mathbf{F}(\mathbf{x}) = -k\,\mathbf{x} + \omega\,(\hat z
# \times \mathbf{x})`: a curl that breaks detailed balance and drives a
# steady probability current, the kind of active or driven system SFI
# targets.  Being out of equilibrium is *not* being inertial — a
# first-order force cannot manufacture momentum — so the verdict should
# stay ``OD``.

x0, x1 = x_components(2)
e0, e1 = unit_axes(2)
k2, omega = 1.0, 1.0
F_rot = (-k2 * x0 - omega * x1) * e0 + (-k2 * x1 + omega * x0) * e1
proc_2d = OverdampedProcess(F_rot, D=0.15)
proc_2d.initialize(jnp.array([0.5, 0.0]))
coll_2d = proc_2d.simulate(dt=0.02, Nsteps=12000, key=random.PRNGKey(0), prerun=200,
                           oversampling=10)
coll_2d_noisy = coll_2d.degrade(noise=sigma, seed=12)

report_2d = classify_dynamics(coll_2d_noisy)
print("### 2D overdamped with a rotational (curl) force ###")
report_2d.print_summary()

# sphinx_gallery_start_ignore
fig_traj, ax_traj = plt.subplots(figsize=(4.2, 4.0))
stream_field(coll_2d_noisy, proc_2d.force_sf, N=25, color="#808080", linewidth=0.5,
             density=0.8, arrowsize=0.7, ax=ax_traj)
phase2d(coll_2d_noisy, dims=(0, 1), color=SFI_COLORS["data"], linewidth=0.3, alpha=0.5,
        ax=ax_traj)
ax_traj.set_xlabel("$x$")
ax_traj.set_ylabel("$y$")
ax_traj.set_title("2D driven overdamped trajectory")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The rotational force (grey streamlines) drives a circulating current; the
# noisy overdamped path (blue) fills the trap around it.  The circulation
# is real — yet every increment is still non-differentiable, which is what
# the verdict keys on:

# sphinx_gallery_start_ignore
fig_2d = plot_dynamics_order(report_2d)
fig_2d.set_size_inches(11, 3.8)
_tidy_log_axes(fig_2d)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Still ``OD``: the rotational drive leaves the lag-2 persistence near zero
# at fine sampling, and the apparent kinetic energy keeps its :math:`-1`
# slope.
#
# Takeaways
# ---------
#
# * The verdict is built from **lag-2** statistics, which white
#   localisation noise cannot reach — noise that dominates :math:`C_0` does
#   not flip the classification.
# * The :math:`\Delta t` scan defeats the force confound: an overdamped
#   force only adds an :math:`\mathcal{O}(\Delta t^2)` correction to a
#   variance that grows like :math:`\Delta t`, so it vanishes under
#   refinement, whereas inertia saturates.
# * When momentum is only marginally resolved (:math:`\gamma\,\Delta t \sim
#   1`), the classifier returns ``"inconclusive"`` rather than guessing.
# * Driving is not inertia: a non-equilibrium, first-order force is
#   correctly read as **overdamped**.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
dts = np.asarray(report_od.scan["dt"], dtype=float)
ax_t.loglog(dts, dts[0] / dts, ":", color="#808080", lw=1.2)       # overdamped: slope -1
ax_t.loglog(dts, np.ones_like(dts), "--", color="#808080", lw=1.2)  # inertial: slope 0
for rep, c in ((report_od, SFI_COLORS["data"]), (report_ud, SFI_COLORS["exact"])):
    K = np.asarray(rep.scaling["K"], dtype=float)                   # apparent kinetic energy
    ax_t.loglog(np.asarray(rep.scan["dt"]), K / K[0], "o-", ms=7, lw=2.5, color=c)
ax_t.minorticks_off()
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
