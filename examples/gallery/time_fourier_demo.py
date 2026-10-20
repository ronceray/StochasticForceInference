r"""
Learning a time-dependent force field — time-Fourier basis
==========================================================

Recover an **unknown, time-varying** force law from trajectories alone,
by expanding time in a Fourier dictionary.

The companion :doc:`time_dependent_forcing_demo` infers a *known,
recorded* drive.  Here the drive is **not** recorded: a population of
overdamped particles sits in a common harmonic trap whose **centre**
:math:`a(t)` *and* **stiffness** :math:`k(t)` both wander in time,

.. math::

   \mathrm{d}x = -k(t)\,\bigl[x - a(t)\bigr]\,\mathrm{d}t
                 + \sqrt{2D}\,\mathrm{d}W .

SFI is handed only the trajectories and a
:func:`~SFI.bases.time_fourier` dictionary, which reads the clock
(``time``) that every dataset carries.  The linear estimator reconstructs
both time-dependent fields with error bands, :term:`PASTIS` keeps only
the harmonics the data support, and the parametric estimator repeats the
feat on positions blurred by localisation noise.

.. rubric:: Tags

synthetic · overdamped · linear · 1D · time-dependent
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D", "time-dependent"]
# sphinx_gallery_thumbnail_number = 6

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import (
    dark_fig,
    plot_pareto_front,
    plot_recovery_bar,
    plot_time_profile_comparison,
    timeseries,
)

apply_style()
# Movies are encoded on the figure's own (dark) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A trap that drifts and stiffens
# -------------------------------
#
# The stiffness :math:`k(t)` and the pull :math:`k(t)\,a(t)` are smooth
# but not single sinusoids: each is a short sum of harmonics of the
# fundamental :math:`\omega = 2\pi / T` (one period over the whole run).
# They are the ground truth; the estimator is never told them.

import SFI
from SFI.bases import X, extra_scalar, time_fourier, unit_vector_basis
from SFI.langevin import OverdampedProcess
from SFI.trajectory import time_series_extra

dt = 0.02
Nframes = 2500
Nparticles = 48
D_true = 0.08

t = np.arange(Nframes) * dt
T_tot = t[-1]
w = 2 * np.pi / T_tot

k_t = 2.0 + 0.8 * np.cos(w * t) + 0.4 * np.cos(2 * w * t)   # stiffness > 0
b_t = 0.9 * np.sin(w * t) + 0.3 * np.cos(2 * w * t)         # pull k·a
a_t = b_t / k_t                                            # trap centre

# %%
# Simulate an ensemble
# --------------------
#
# A single trapped particle cannot tell *"the trap is stiffer"* from *"the
# trap has moved"*: at any instant it just sits near the minimum.  A
# **population** breaks that degeneracy — at each time, the spread of
# particles around :math:`a(t)` fixes the slope :math:`-k(t)` and the
# offset :math:`k(t)\,a(t)`.  Initialising the process with one position
# per particle simulates ``Nparticles`` independent copies at once.

F_true = extra_scalar("neg_k") * X(dim=1) & extra_scalar("pull") * unit_vector_basis(1)
proc = OverdampedProcess(F=F_true, D=D_true, theta_F=jnp.array([1.0, 1.0]))
proc.set_extras(extras_global={
    "neg_k": time_series_extra(-k_t),
    "pull": time_series_extra(b_t),
})

x0 = np.random.default_rng(0).normal(0.0, 0.4, size=(Nparticles, 1))
proc.initialize(jnp.asarray(x0))
coll = proc.simulate(dt=dt, Nsteps=Nframes, key=random.PRNGKey(0), oversampling=3,
                     compute_observables=False)
print(f"ensemble: {coll.T} frames x {Nparticles} particles")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
timeseries(coll, particles=range(14), ax=ax[0], lw=0.4, alpha=0.5, color=SFI_COLORS["data"])
ax[0].plot(t, a_t, color=SFI_COLORS["exact"], lw=2.2, label=r"trap centre $a(t)$")
ax[0].set_xlabel("")
ax[0].set_ylabel("$x$")
ax[0].legend(loc="upper right")
ax[1].plot(t, k_t, color=SFI_COLORS["exact"], lw=2.2)
ax[1].set_ylabel(r"stiffness $k(t)$")
ax[1].set_xlabel("time")
fig.suptitle("Ensemble in a drifting, stiffening trap (14 of 48 particles shown)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The particles follow the moving centre, and their spread shrinks where
# the trap is stiff.  Neither curve is given to the estimator.
#
# Inference with a time-Fourier dictionary
# ----------------------------------------
#
# Tensor a Fourier-in-time dictionary with the spatial terms:
# ``time_fourier(n) * X`` carries the **stiffness**, and
# ``time_fourier(n) * unit_vector_basis(1)`` the **pull**.  With
# ``period=None`` the fundamental is the full duration of the recording.
# The data are noise-free, so we use the MSD diffusion estimate and the
# ``"clean"`` preset; :term:`PASTIS` then prunes the dictionary.

n_modes = 6
B_k = time_fourier(n_modes) * X(dim=1)                 # -> -k(t)
B_b = time_fourier(n_modes) * unit_vector_basis(1)     # ->  k(t) a(t)
B = B_k & B_b

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="MSD")
inf.infer_force_linear(B, preset="clean")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# sphinx_gallery_start_ignore
nf = 1 + 2 * n_modes
theta_true = np.zeros(2 * nf)
theta_true[[0, 1, 3]] = [-2.0, -0.8, -0.4]       # -k(t): 1, cos(wt), cos(2wt)
theta_true[[nf + 2, nf + 3]] = [0.9, 0.3]        # k a : sin(wt), cos(2wt)
coeffs, cov = inf.coeff_block(B)
idx = np.arange(2 * nf)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
plot_recovery_bar(coeffs, idx, coeffs_true=theta_true, support_true=idx,
                  stderr=np.sqrt(np.diag(cov)), labels=list(B.labels), ax=axes[1])
axes[1].set_title(f"PASTIS keeps {len(inf.force_support)} / {2 * nf} terms (1σ error bars)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The dictionary offers 26 terms (13 Fourier features per spatial term);
# the truth uses 5 — the constant and first two cosines of :math:`-k(t)`,
# the first sine and second cosine of :math:`k(t)\,a(t)`.  PASTIS keeps
# exactly those (right; left: the Pareto front).
#
# Reconstructed stiffness and centre
# ----------------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.predict_time_profile`
# contracts each coefficient block with its time dictionary: the ``x``
# block gives :math:`-k(t)` (evaluated at :math:`x = 1`), the constant
# block the pull :math:`k(t)\,a(t)`; both come with 1σ bands propagated
# from the coefficient covariance.  Their ratio is the trap centre.

neg_k_hat, k_std = inf.predict_time_profile(B_k, t, x=1.0, return_std=True)
b_hat, b_std = inf.predict_time_profile(B_b, t, return_std=True)
k_hat = -neg_k_hat
a_hat = b_hat / k_hat


def nmse(estimate, truth):
    return float(np.mean((estimate - truth) ** 2) / np.mean(truth ** 2))


print(f"k(t) reconstruction NMSE = {nmse(k_hat, k_t):.1e}")
print(f"a(t) reconstruction NMSE = {nmse(a_hat, a_t):.1e}")

# sphinx_gallery_start_ignore
axes = plot_time_profile_comparison(
    t, [k_t, b_t, a_t], [k_hat, b_hat, a_hat],
    labels=[r"stiffness $k(t)$", r"pull $k(t)\,a(t)$", r"trap centre $a(t)$"],
)
for ax, prof, std in zip(axes[:2], [k_hat, b_hat], [k_std, b_std]):
    ax.fill_between(t, prof - std, prof + std, color=SFI_COLORS["inferred"], alpha=0.3,
                    lw=0, label="1σ band")
for ax in axes:
    ax.legend(loc="upper center" if ax is axes[0] else "upper right", fontsize=8)
axes[0].figure.suptitle("Recovered time-dependent force field (never shown the protocol)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The movie plays the ensemble in its trap: each particle is drawn on the
# true potential :math:`U(x, t) = \tfrac12 k(t)\,[x - a(t)]^2` (orange), next
# to the inferred one (gold, dashed), with the stiffness shown on the
# right.

# sphinx_gallery_start_ignore
X_all = np.asarray(coll.to_array())[:, :, 0]               # (T, N)
xg = np.linspace(-1.7, 1.4, 200)
skip = 10

fig_m, (ax_m, ax_k) = dark_fig(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
(true_well,) = ax_m.plot([], [], color=SFI_COLORS["exact"], lw=2.4)
(inf_well,) = ax_m.plot([], [], color=SFI_COLORS["inferred"], lw=1.8, ls="--")
beads = ax_m.scatter(X_all[0], 0 * X_all[0], s=45, color=SFI_COLORS["data"],
                     edgecolor="#D0D0D0", lw=0.4, zorder=5)
ax_m.set(xlim=(xg[0], xg[-1]), ylim=(-0.02, 0.6), xlabel="$x$", ylabel=r"$U(x, t)$")
ax_k.plot(t, k_t, color=SFI_COLORS["exact"], lw=2.4, label="true")
ax_k.plot(t, k_hat, color=SFI_COLORS["inferred"], lw=1.6, ls="--", label="inferred")
# One legend for both panels (same colours), in the free space between the
# stiff ends; an opaque box keeps the time cursor from crossing the labels.
ax_k.legend(loc="upper center", fontsize=8, frameon=True, framealpha=1, edgecolor="none",
            facecolor=fig_m.get_facecolor())
cursor = ax_k.axvline(0, color="#D0D0D0", lw=1)
ax_k.set(xlabel="time", ylabel="$k(t)$")
for a in (ax_m, ax_k):
    a.spines[["top", "right"]].set_visible(False)


def _trap_frame(n):
    i = n * skip
    true_well.set_data(xg, 0.5 * k_t[i] * (xg - a_t[i]) ** 2)
    inf_well.set_data(xg, 0.5 * k_hat[i] * (xg - a_hat[i]) ** 2)
    beads.set_offsets(np.column_stack([X_all[i], 0.5 * k_t[i] * (X_all[i] - a_t[i]) ** 2]))
    cursor.set_xdata([t[i], t[i]])
    ax_m.set_title(f"t = {t[i]:.1f}")
    return []


anim = FuncAnimation(fig_m, _trap_frame, frames=Nframes // skip, interval=70)
plt.show()
# sphinx_gallery_end_ignore

# %%
# With localisation noise
# -----------------------
#
# Now blur every recorded position with Gaussian noise of standard
# deviation :math:`\sigma = 0.06`, about the size of the diffusive step
# :math:`\sqrt{2D\Delta t} \approx 0.057`.  The parametric estimator
# :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` profiles
# the noise amplitude jointly with the force and reads the clock frame by
# frame, like the linear estimator; the dictionary is unchanged.

coll_noisy = coll.degrade(noise=0.06, seed=1)

inf_noisy = SFI.OverdampedLangevinInference(coll_noisy)
inf_noisy.infer_force(B)
inf_noisy.sparsify_force(criterion="PASTIS")
inf_noisy.compute_force_error()

neg_k_n, k_std_n = inf_noisy.predict_time_profile(B_k, t, x=1.0, return_std=True)
b_n = inf_noisy.predict_time_profile(B_b, t)
print(f"terms kept: {[B.labels[i] for i in np.asarray(inf_noisy.force_support)]}")
print(f"profiled noise σ = {np.sqrt(float(np.ravel(inf_noisy.Lambda)[0])):.3f} (true 0.06), "
      f"D = {float(np.ravel(inf_noisy.diffusion_average)[0]):.3f} (true {D_true})")
print(f"k(t) reconstruction NMSE = {nmse(-neg_k_n, k_t):.1e}")
print(f"a(t) reconstruction NMSE = {nmse(b_n / -neg_k_n, a_t):.1e}")

# sphinx_gallery_start_ignore
axes = plot_time_profile_comparison(
    t, [k_t, a_t], [-neg_k_n, b_n / -neg_k_n],
    labels=[r"stiffness $k(t)$", r"trap centre $a(t)$"],
)
axes[0].fill_between(t, -neg_k_n - k_std_n, -neg_k_n + k_std_n, color=SFI_COLORS["inferred"],
                     alpha=0.3, lw=0, label="1σ band")
axes[0].legend(loc="upper center", fontsize=8)
axes[0].figure.suptitle(r"Recovered from noisy positions ($\sigma = 0.06$), parametric estimator")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The same five harmonics are selected and both profiles are recovered,
# with the noise level and the diffusion coefficient as by-products.
#
# Notes
# -----
#
# - **Time as an extra.**  Every dataset carries its clock as the reserved
#   ``time`` extra (see
#   :meth:`~SFI.trajectory.TrajectoryDataset.build_extras`), so
#   :func:`~SFI.bases.time_fourier` needs no recorded protocol.  With
#   ``period=None`` the fundamental is the full duration of the recording;
#   pass ``period=`` to fit a known repeat time.
# - **Sampling convention.**  The parametric estimator reads
#   time-dependent terms at the first frame of each sampling interval, as
#   the simulator applies them.  For a drive that varies continuously
#   within a frame, this left-point hold lags it by half a frame — a phase
#   :math:`\approx\omega\Delta t/2`, about :math:`3\times10^{-3}` rad for
#   the highest harmonic kept here, and nil for these simulated data,
#   whose drive is held constant over each frame.
# - **Why an ensemble.**  Stiffness and centre are jointly identifiable
#   only because many particles sample a spread of positions at each
#   instant.
# - **Beyond traps.**  Any slowly varying force field — ramps, oscillatory
#   drives, ageing — can be learned the same way: tensor
#   :func:`~SFI.bases.time_fourier` with the spatial basis the problem
#   needs, and let PASTIS keep the active modes.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
width = np.sqrt(float(np.ravel(inf.diffusion_average)[0]) / k_hat)   # inferred trap's spread
for i in range(6):
    ax_t.plot(t, X_all[:, i], lw=0.4, alpha=0.45, color=SFI_COLORS["data"])
ax_t.fill_between(t, a_hat - 2 * width, a_hat + 2 * width, color=SFI_COLORS["inferred"],
                  alpha=0.22, lw=0)
ax_t.plot(t, a_hat, color=SFI_COLORS["inferred"], lw=2.2)
ax_t.set_xlim(t[0], t[-1])
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
