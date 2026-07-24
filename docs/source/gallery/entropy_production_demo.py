"""
Measuring dissipation: entropy production in a driven optical trap
===================================================================

A colloidal bead in an optical trap looks like the textbook equilibrium
system — yet real tweezers exert a small **non-conservative** force: the
scattering force has a curl component that silently stirs the bead
(Roichman et al., PRL 2008; Wu et al., PRL 2009).  This demo treats that
situation from the point of view of **stochastic thermodynamics**: the
bead obeys the overdamped Langevin equation

.. math::

   \\frac{\\mathrm{d}\\mathbf{x}}{\\mathrm{d}t}
   = \\underbrace{-k\\,\\mathbf{x}}_{\\text{trap}}
   + \\underbrace{\\varepsilon\\,(\\hat z \\times \\mathbf{x})}_{\\text{curl drive}}
   + \\sqrt{2D}\\,\\boldsymbol{\\xi}(t),

with a faint rotational drive :math:`\\varepsilon \\ll k` breaking detailed
balance.  The twist: the steady-state *density* of this driven trap is
**exactly** the Boltzmann distribution of the trap alone — no static
observable reveals the drive.  Its only signatures are dynamical: a
circulating probability current, and a positive **entropy production
rate**, exactly :math:`\\sigma = 2\\varepsilon^2/k` (in units of
:math:`k_B` per unit time) for this system.

Across this page we will:

#. simulate the driven trap and check that its density is Boltzmann;
#. infer force and diffusion, and let PASTIS find the faint curl term;
#. decompose the dynamics into an equilibrium part (a potential) and a
   drive (a current);
#. **measure the entropy production** with
   :meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
   and compare it to the exact rate;
#. ask how faint a drive is detectable — the thermodynamic limit of
   detection;
#. re-fit the model *parametrized by its energy*, using automatic
   differentiation.

The estimator is the one introduced with SFI itself (Frishman &
Ronceray, PRX 2020): project the phase-space velocity onto a basis,
contract with the inverse diffusion — dissipation becomes a measurable
quantity of the trajectory alone.

.. rubric:: Tags

synthetic · overdamped · linear · 2D · non-equilibrium · thermodynamics
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "2D", "non-equilibrium", "thermodynamics"]
# sphinx_gallery_thumbnail_number = 8

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
# A trap that hides its drive
# ---------------------------
#
# We simulate a bead in a two-dimensional isotropic harmonic trap of
# stiffness :math:`k`, plus a weak rotational force
# :math:`\varepsilon\,(\hat z\times\mathbf{x})` — the minimal model of a
# tweezer's non-conservative scattering component.  Units are natural
# (:math:`k_BT = 1`, mobility 1): for a micron-sized bead in water, one
# time unit is the trap relaxation time (:math:`\sim` tens of ms) and
# lengths are in units of the thermal trap width.
#
# The drive is *faint*: :math:`\varepsilon/k = 0.3`, and we also run an
# equilibrium control (:math:`\varepsilon = 0`) for comparison.  One long
# recording — a few thousand relaxation times — will turn out to be
# exactly what measuring this faint dissipation requires.

import SFI
from SFI.bases import unit_axes, x_components
from SFI.langevin import OverdampedProcess

k_true = 1.0       # trap stiffness
eps_true = 0.3     # rotational drive:  F_curl = eps * (z_hat x x)
D_true = 0.5       # diffusion constant
dt = 0.01          # time between recorded frames
Nsteps = 320_000   # frames per run (3200 trap relaxation times)

x0c, x1c = x_components(2)
e0, e1 = unit_axes(2)


def make_trap(eps):
    """Isotropic trap + curl drive of amplitude eps, as a basis expression."""
    return (-k_true * x0c - eps * x1c) * e0 + (-k_true * x1c + eps * x0c) * e1


def simulate_trap(eps, seed):
    proc = OverdampedProcess(make_trap(eps), D=D_true)
    key_init, key_run = random.split(random.PRNGKey(seed))
    x_init = jnp.sqrt(D_true / k_true) * random.normal(key_init, (2,))
    proc.initialize(x_init)
    coll_run = proc.simulate(
        dt=dt, Nsteps=Nsteps, key=key_run,
        prerun=500, oversampling=4, compute_observables=True,
    )
    return coll_run, proc


coll, proc_true = simulate_trap(eps_true, seed=4)
coll_eq, _ = simulate_trap(0.0, seed=2)

sigma_exact = 2.0 * eps_true**2 / k_true
print(f"Simulated {coll.T} frames at dt = {dt}"
      f"  (total observation time {Nsteps * dt:.0f})")
print(f"Exact entropy production rate: sigma = 2 eps^2 / k = {sigma_exact:.3f} kB / time")

# sphinx_gallery_start_ignore
t_arr, X_neq, _ = coll.to_arrays(dataset=0)   # X_neq: (T, 1, 2)
X_neq = np.asarray(X_neq)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
n_view = 4000  # 40 relaxation times
axes[0].plot(np.asarray(t_arr)[:n_view], X_neq[:n_view, 0, 0],
             lw=0.6, color=SFI_COLORS["data"], alpha=0.9)
axes[0].axhline(0, ls=":", lw=0.5, color="#808080")
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"$x_0$")
axes[0].set_title("Position vs time (looks like a trap...)")

phase2d(coll, tmin=0, tmax=n_view, ax=axes[1], cmap="viridis",
        linewidth=0.4, alpha=0.8)
axes[1].set_xlabel(r"$x_0$")
axes[1].set_ylabel(r"$x_1$")
axes[1].set_aspect("equal")
axes[1].set_title("The bead's path (driven run)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Statics look like equilibrium
# -----------------------------
#
# The curl force is divergence-free and everywhere tangent to the circles
# of constant :math:`|\mathbf{x}|` — so it stirs probability *along* the
# level sets of the trap's Boltzmann density without reshaping it.  The
# driven steady state is exactly
# :math:`\rho(\mathbf{x}) \propto e^{-k|\mathbf{x}|^2/2D}`, independent
# of :math:`\varepsilon`.
#
# The radial histograms of the driven and equilibrium runs confirm it:
# they are statistically indistinguishable, and both match the Boltzmann
# prediction.  **Snapshots cannot reveal this non-equilibrium system** —
# only the arrow of time in the trajectories can.

# sphinx_gallery_start_ignore
_, X_eq, _ = coll_eq.to_arrays(dataset=0)
r_neq = np.linalg.norm(X_neq.reshape(-1, 2), axis=1)
r_eq = np.linalg.norm(np.asarray(X_eq).reshape(-1, 2), axis=1)

r_grid = np.linspace(0, 3.0, 200)
p_boltzmann = (k_true / D_true) * r_grid * np.exp(-k_true * r_grid**2 / (2 * D_true))

fig, ax = plt.subplots(figsize=(5.5, 3.4))
bins = np.linspace(0, 3.0, 60)
ax.hist(r_neq, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
        color=SFI_COLORS["data"], label=rf"driven ($\varepsilon = {eps_true}$)")
ax.hist(r_eq, bins=bins, density=True, histtype="step", lw=1.4,
        color=SFI_COLORS["highlight"], label=r"equilibrium ($\varepsilon = 0$)")
ax.plot(r_grid, p_boltzmann, ":", lw=1.6, color="#808080",
        label=r"Boltzmann $\propto r\,e^{-kr^2/2D}$")
ax.set_xlabel(r"radius $r = |\mathbf{x}|$")
ax.set_ylabel(r"$p(r)$")
ax.set_title("Steady-state density: the drive is invisible")
ax.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infer the dynamics
# ------------------
#
# Standard SFI workflow: estimate the diffusion constant, then regress
# the force onto a deliberately over-complete polynomial basis and let
# **PASTIS** decide which terms the data support.  The interesting
# question: does model selection *keep the faint curl term*?

from SFI.bases import monomials_up_to

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="MSD")   # clean synthetic data

B = monomials_up_to(2, dim=2, rank="vector")   # 12 candidate terms
inf.infer_force_linear(B)
inf.compute_force_error()

coeffs_full = np.asarray(inf.force_coefficients_full)
stderr_full = np.asarray(inf.force_coefficients_stderr)

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# %%
# PASTIS retains exactly the four linear terms — the isotropic trap
# *and* the antisymmetric curl couplings, which are an order of magnitude
# smaller.  The faint drive is not just detected: its structure is
# identified.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(7.5, 3.2))
labels_full = list(B.labels)
coeffs_true_full = np.zeros(len(labels_full))
coeffs_true_full[labels_full.index("x0·e0")] = -k_true
coeffs_true_full[labels_full.index("x1·e1")] = -k_true
coeffs_true_full[labels_full.index("x1·e0")] = -eps_true
coeffs_true_full[labels_full.index("x0·e1")] = eps_true
plot_recovery_bar(
    coeffs_full, np.arange(len(labels_full)),
    coeffs_true=coeffs_true_full, support_true=np.arange(len(labels_full)),
    labels=labels_full, stderr=stderr_full, ax=ax,
)
ax.set_title("Force coefficients: the curl terms are small but significant")
plt.show()
# sphinx_gallery_end_ignore

# %%
# An equilibrium part and a drive
# -------------------------------
#
# For a linear force :math:`\mathbf{F} = M\mathbf{x}`, the symmetric part
# of :math:`M` is the (conservative) trap and the antisymmetric part is
# the non-conservative drive.  We read :math:`\hat M` off the inferred
# force field by automatic differentiation and split it:

import jax

F_hat = lambda x: inf.force_inferred(x[None, :])[0]
M_hat = np.asarray(jax.jacobian(F_hat)(jnp.zeros(2)))

k_hat = -0.5 * (M_hat[0, 0] + M_hat[1, 1])       # trap stiffness
eps_hat = 0.5 * (M_hat[1, 0] - M_hat[0, 1])      # curl amplitude

print(f"trap stiffness : k_hat   = {k_hat:.4f}   (true {k_true})")
print(f"curl amplitude : eps_hat = {eps_hat:.4f}   (true {eps_true})")

# %%
# The two parts play very different thermodynamic roles.  The symmetric
# part integrates to a potential :math:`\hat U = \tfrac12 \hat k
# |\mathbf{x}|^2` whose Boltzmann weight *is* the observed density.  The
# antisymmetric part is precisely the **mean local velocity**
# :math:`\hat{\mathbf{v}}(\mathbf{x}) = \hat\varepsilon\,(\hat z \times
# \mathbf{x})` — the circulating probability current that a movie of the
# bead would reveal but a snapshot cannot.

# sphinx_gallery_start_ignore
M_sym = 0.5 * (M_hat + M_hat.T)
M_anti = 0.5 * (M_hat - M_hat.T)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

# (a) recovered potential + conservative force
g = np.linspace(-2.2, 2.2, 120)
GX, GY = np.meshgrid(g, g)
U_hat = 0.5 * k_hat * (GX**2 + GY**2)
cs = axes[0].contour(GX, GY, U_hat, levels=6, colors=SFI_COLORS["inferred"],
                     linewidths=1.0, alpha=0.9)
axes[0].clabel(cs, inline=True, fontsize=6, fmt="%.1f")
stream_field(coll, lambda X: X @ M_sym.T, ax=axes[0], N=24, density=0.7,
             color="#808080", linewidth=0.6, arrowsize=0.6)
axes[0].set_title(r"Equilibrium part: potential $\hat U(\mathbf{x})$")
axes[0].set_aspect("equal")

# (b) current part over the data density
hb = axes[1].hexbin(np.asarray(X_neq).reshape(-1, 2)[:, 0],
                    np.asarray(X_neq).reshape(-1, 2)[:, 1],
                    gridsize=45, cmap="cividis", mincnt=1,
                    extent=(-2.2, 2.2, -2.2, 2.2), linewidths=0.1)
stream_field(coll, lambda X: X @ M_anti.T, ax=axes[1], N=24, density=0.8,
             color=SFI_COLORS["highlight"], linewidth=0.9, arrowsize=0.8)
axes[1].set_title(r"Drive: current $\hat{\mathbf{v}} = \hat\varepsilon\,(\hat z\times\mathbf{x})$")
axes[1].set_aspect("equal")

for ax_ in axes:
    ax_.set_xlabel(r"$x_0$")
    ax_.set_ylabel(r"$x_1$")
    ax_.set_xlim(-2.2, 2.2)
    ax_.set_ylim(-2.2, 2.2)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The price of the current: entropy production
# --------------------------------------------
#
# Maintaining that circulation costs free energy, dissipated into the
# bath at the **entropy production rate**
# :math:`\sigma = \langle \mathbf{v}\cdot D^{-1}\mathbf{v}\rangle`.
# :meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
# estimates it directly from the fit, by projecting the phase-space
# velocity onto the force basis (Frishman & Ronceray, PRX 2020) — with an
# error bar, and a *fluctuation bias* :math:`2N_b/\tau_N` that tells you
# how much apparent dissipation mere noise would produce with this basis
# and this much data.

out = inf.compute_entropy_production()

S_sim = coll.datasets[0].meta["observables"]["entropy"]  # simulator ground truth
tauN = out["tauN"]

print(f"exact rate                    : {sigma_exact:.4f} kB / time")
print(f"inferred (projection)         : {out['Sdot']:.4f} +- {out['Sdot_error']:.4f}"
      f"   [fluctuation bias {out['Sdot_bias']:.4f}]")
print(f"from the fitted model 2e^2/k  : {2 * eps_hat**2 / k_hat:.4f}")
print(f"true model on this trajectory : {S_sim / tauN:.4f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.2))
vals = [out["Sdot"], 2 * eps_hat**2 / k_hat, S_sim / tauN]
errs = [out["Sdot_error"], 0.0, 0.0]
names = ["projection\nestimator", r"$2\hat\varepsilon^2/\hat k$", "true model\n(same run)"]
colors = [SFI_COLORS["inferred"], SFI_COLORS["highlight"], SFI_COLORS["bootstrap"]]
ax.axhline(sigma_exact, ls=":", lw=1.2, color=SFI_COLORS["exact"],
           label=rf"exact $2\varepsilon^2/k = {sigma_exact:.3f}$")
ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="o", ms=7, capsize=4,
            color="#B0B0B0", markerfacecolor="none", markeredgewidth=0)
for i, (v, c) in enumerate(zip(vals, colors)):
    ax.plot([i], [v], "o", ms=8, color=c)
ax.set_xticks(range(len(vals)), names)
ax.set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
ax.set_title("Entropy production: three estimates agree")
ax.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# The report now carries the dissipation too — ``print_report()`` shows
# ``Entropy production (total Delta S, estimated error)`` once the
# estimator has run.  Note the estimator only uses the *inferred* force
# and diffusion: no ground truth enters.  A subtlety worth naming: what
# is measured is the entropy produced *along this trajectory*, which
# itself fluctuates from run to run — the reported error bar includes
# that fluctuation (the :math:`2\Delta\hat S` term), which is why all
# three estimates track each other more tightly than they track the
# ensemble value.
#
# How faint a drive can you detect?
# ---------------------------------
#
# Near equilibrium the current is linear in the drive, so the dissipation
# is *quadratic*: :math:`\sigma = 2\varepsilon^2/k`.  Detecting a
# twice-fainter drive takes four times the entropy — and the estimator's
# own fluctuation bias sets the floor: irreversibility becomes detectable
# only once the trajectory has dissipated a few :math:`k_B` more than the
# bias :math:`2N_b`.  We sweep :math:`\varepsilon` at fixed trajectory
# length, then sweep the trajectory length at fixed faint drive.

B_lin = monomials_up_to(1, dim=2, include_constant=False, rank="vector")


def entropy_of(coll_run):
    """MSD diffusion + linear force fit + entropy production, in one go."""
    inf_run = SFI.OverdampedLangevinInference(coll_run)
    inf_run.compute_diffusion_constant(method="MSD")
    inf_run.infer_force_linear(B_lin)
    return inf_run.compute_entropy_production()


eps_sweep = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
runs = {eps_true: coll}
for i, eps in enumerate(eps_sweep):
    if eps not in runs:
        runs[eps], _ = simulate_trap(eps, seed=10 + i)
sweep = {eps: entropy_of(runs[eps]) for eps in eps_sweep}

out_eq = entropy_of(coll_eq)
print(f"equilibrium control: Sdot = {out_eq['Sdot']:.4f} +- {out_eq['Sdot_error']:.4f}"
      f"  (bias {out_eq['Sdot_bias']:.4f})  -> consistent with zero")

# %%
# For the length sweep we truncate the :math:`\varepsilon = 0.1` run —
# dissipating only :math:`\sigma \approx 0.02\,k_B` per unit time — to
# increasingly short observation windows:

from SFI import TrajectoryCollection


def truncate(coll_run, T_keep):
    _, X_run, _ = coll_run.to_arrays(dataset=0)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(X_run[:T_keep]), dt=dt)


T_sweep = [20_000, 40_000, 80_000, 160_000, 320_000]
length_sweep = [entropy_of(truncate(runs[0.1], T)) for T in T_sweep]

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

# (a) rate vs drive amplitude
eps_arr = np.array(eps_sweep)
sd = np.array([sweep[e]["Sdot"] for e in eps_sweep])
se = np.array([sweep[e]["Sdot_error"] for e in eps_sweep])
bias0 = sweep[eps_sweep[0]]["Sdot_bias"]
eps_fine = np.geomspace(0.03, 1.0, 100)
axes[0].plot(eps_fine, 2 * eps_fine**2 / k_true, ":", lw=1.4,
             color=SFI_COLORS["exact"], label=r"exact $2\varepsilon^2/k$")
axes[0].axhline(bias0, ls="--", lw=1.0, color="#808080",
                label=r"fluctuation bias $2N_b/\tau_N$")
axes[0].errorbar(eps_arr, sd, yerr=se, fmt="o", ms=5, capsize=3,
                 color=SFI_COLORS["inferred"], label="inferred")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"drive amplitude $\varepsilon$")
axes[0].set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
axes[0].set_title(r"Quadratic near equilibrium: $\sigma \propto \varepsilon^2$")
axes[0].legend(fontsize=8)

# (b) rate vs observation time at faint drive
tau_arr = np.array([o["tauN"] for o in length_sweep])
sd_T = np.array([o["Sdot"] for o in length_sweep])
se_T = np.array([o["Sdot_error"] for o in length_sweep])
sigma_01 = 2 * 0.1**2 / k_true
axes[1].axhline(sigma_01, ls=":", lw=1.4, color=SFI_COLORS["exact"],
                label=rf"exact ($\varepsilon = 0.1$): {sigma_01:.3f}")
axes[1].plot(tau_arr, [o["Sdot_bias"] for o in length_sweep], "--", lw=1.0,
             color="#808080", label=r"fluctuation bias $2N_b/\tau_N$")
axes[1].errorbar(tau_arr, sd_T, yerr=se_T, fmt="o", ms=5, capsize=3,
                 color=SFI_COLORS["inferred"], label="inferred")
axes[1].set_xscale("log")
axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
axes[1].set_xlabel(r"observation time $\tau_N$")
axes[1].set_ylabel(r"$\hat{\dot S}$  [$k_B$ / time]")
axes[1].set_title("Detection emerges with time")
axes[1].legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The left panel shows the near-equilibrium scaling over more than two
# decades of dissipation, down to the detection floor where the inferred
# rate meets the fluctuation bias.  The right panel shows detection
# *emerging* as data accumulates: with :math:`\tau_N \lesssim 10^3` the
# faint drive is buried in the bias; a few thousand relaxation times
# resolve it cleanly.  The rule of thumb — the trajectory must dissipate
# a few :math:`k_B` beyond the :math:`2N_b` bias — is the thermodynamic
# cost of *measuring* irreversibility.
#
# Fit the energy, not the force
# -----------------------------
#
# Near equilibrium, the natural parametrization is thermodynamic: an
# energy landscape :math:`U_\theta` plus a drive amplitude.  With SFI's
# parametric estimator and JAX autodiff, we fit
# :math:`\mathbf{F}_\theta = -\nabla U_\theta +
# \varepsilon\,(\hat z\times\mathbf{x})` directly — here with
# :math:`U_\theta = \tfrac12 k |\mathbf{x}|^2 + \tfrac14 u |\mathbf{x}|^4`,
# where the quartic coefficient :math:`u` lets the data *bound the
# anharmonicity* of the trap.

from SFI.statefunc import ParamSpec, make_psf


def U_theta(x, params):
    r2 = jnp.sum(x**2)
    return 0.5 * params["k"] * r2 + 0.25 * params["u"] * r2**2


def F_theta(x, *, params):
    return -jax.grad(U_theta)(x, params) + params["eps"] * jnp.array([-x[1], x[0]])


F_psf = make_psf(
    F_theta, dim=2, rank=1,
    params=[ParamSpec("k", shape=()), ParamSpec("u", shape=()), ParamSpec("eps", shape=())],
)

inf_energy = SFI.OverdampedLangevinInference(coll)
inf_energy.infer_force(F_psf)
inf_energy.compute_force_error()

theta = F_psf.unflatten_params(inf_energy.force_coefficients_full)
theta_err = np.asarray(inf_energy.force_coefficients_stderr)
for name, true_val, err in zip(("k", "u", "eps"), (k_true, 0.0, eps_true), theta_err):
    print(f"{name:>4s} = {float(theta[name]):+.4f} +- {err:.4f}   (true {true_val:+.3f})")

# %%
# The energy parameters come back with error bars: the trap stiffness and
# drive amplitude are recovered, and the anharmonicity is *bounded* —
# consistent with zero.  Deriving forces from energies via autodiff
# scales to arbitrarily rich landscapes (see the Müller–Brown surface in
# :doc:`/gallery/advanced/nn_force_demo`).

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(4.5, 3))
names = [r"$k$", r"$u$", r"$\varepsilon$"]
vals = [float(theta["k"]), float(theta["u"]), float(theta["eps"])]
trues = [k_true, 0.0, eps_true]
plot_recovery_bar(
    np.array(vals), np.arange(3),
    coeffs_true=np.array(trues), support_true=np.arange(3),
    labels=names, stderr=theta_err, ax=ax,
)
ax.set_title("Energy parametrization recovered")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Next steps
# ----------
#
# - **Theory**: the estimator, its bias and error bars, the
#   Itô-information / Stratonovich-entropy duality, and what
#   coarse-graining does to these measurements —
#   :doc:`/inference/stochastic_thermodynamics`.
# - **Strongly driven systems**: a non-equilibrium steady state with
#   :math:`O(1)` currents — :doc:`/gallery/limitcycle_demo`.
# - **State-dependent noise** breaks the naive Boltzmann picture in a
#   different way (spurious drifts) —
#   :doc:`/gallery/multiplicative_diffusion_demo`.
# - **Driven vs inertial**: a rotational current is not inertia; the
#   dynamics-order classifier tells them apart —
#   :doc:`/gallery/dynamics_order_demo`.
# - **Real tweezer data** with localization noise:
#   :doc:`/gallery/experimental_workflow_demo`.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
ax_t.hexbin(np.asarray(X_neq).reshape(-1, 2)[:, 0],
            np.asarray(X_neq).reshape(-1, 2)[:, 1],
            gridsize=40, cmap="cividis", mincnt=1,
            extent=(-2.1, 2.1, -2.1, 2.1), linewidths=0.1)
stream_field(coll, lambda X: X @ M_anti.T, ax=ax_t, N=24, density=0.9,
             color=SFI_COLORS["inferred"], linewidth=1.1, arrowsize=0.9)
ax_t.set_xlim(-2.1, 2.1)
ax_t.set_ylim(-2.1, 2.1)
ax_t.set_aspect("equal")
ax_t.set_xticks([])
ax_t.set_yticks([])
ax_t.set_xlabel("")
ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
