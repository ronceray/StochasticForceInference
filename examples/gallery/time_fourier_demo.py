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

SFI never sees :math:`k(t)` or :math:`a(t)`.  It is handed only the
trajectories and a :func:`~SFI.bases.time_fourier` dictionary, which
reads the **auto-injected** ``time`` clock (no protocol has to be
recorded).  The linear estimator reconstructs both time-dependent
fields, and :term:`PASTIS` keeps only the harmonics the data support.

.. rubric:: Tags

synthetic · overdamped · linear · 1D · time-dependent
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "linear", "1D", "time-dependent"]
# sphinx_gallery_thumbnail_number = 2

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

apply_style()
# sphinx_gallery_end_ignore
# %%
# A trap that drifts and stiffens
# -------------------------------
#
# The centre :math:`a(t)` and stiffness :math:`k(t)` are smooth but
# *not* single sinusoids — each is a short sum of harmonics of the
# fundamental :math:`\omega = 2\pi / T_\text{tot}` (one period over the
# whole run).  They are the ground truth we will try to recover; the
# estimator is never told them.

import SFI
from SFI.bases import X, time_fourier, unit_vector_basis, extra_scalar
from SFI.langevin import OverdampedProcess
from SFI.trajectory import time_series_extra
from SFI.utils.plotting import plot_time_profile_comparison, timeseries

dt = 0.02
Nframes = 2500
Nparticles = 48
D_true = 0.08

t = np.arange(Nframes) * dt
T_tot = t[-1]
w = 2 * np.pi / T_tot

k_t = 2.0 + 0.8 * np.cos(w * t) + 0.4 * np.cos(2 * w * t)   # stiffness > 0
b_t = 0.9 * np.sin(w * t) + 0.3 * np.cos(2 * w * t)         # additive drift k·a
a_t = b_t / k_t                                            # implied trap centre

# %%
# Simulate an ensemble
# --------------------
#
# A single trapped particle cannot separate *"the trap is stiffer"* from
# *"the trap has moved"* — at any instant it just sits near the minimum.
# A **population** breaks that degeneracy: at each time the spread of
# particles around :math:`a(t)` fixes the slope :math:`-k(t)` and the
# offset :math:`k(t)\,a(t)`.  The particles are independent, so we
# simulate ``Nparticles`` single-particle runs and merge them into one
# ensemble collection.

F_true = extra_scalar("neg_k") * X(dim=1) & extra_scalar("drift") * unit_vector_basis(1)
proc = OverdampedProcess(F=F_true, D=D_true, theta_F=jnp.array([1.0, 1.0]))
proc.set_extras(extras_global={
    "neg_k": time_series_extra(-k_t),
    "drift": time_series_extra(b_t),
})

rng = np.random.default_rng(0)
runs = []
for i in range(Nparticles):
    proc.initialize(jnp.array([float(rng.normal() * 0.4)]))
    ci = proc.simulate(dt=dt, Nsteps=Nframes, key=random.PRNGKey(i),
                       oversampling=3, compute_observables=False)
    runs.append(ci)

coll = runs[0].merge(runs[1:])   # ensemble = many single-particle datasets
print(f"ensemble: {coll.datasets[0].T} frames x {Nparticles} particles")

# %%
# Trajectories and the hidden protocol
# ------------------------------------
#
# The cloud of particles tracks the moving centre; the bottom panel
# shows the stiffness they actually feel.  Neither curve is given to the
# estimator.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
for j in range(min(14, Nparticles)):
    timeseries(coll, dataset=j, particles=[0], ax=ax[0],
               lw=0.4, alpha=0.5, color=SFI_COLORS["data"])
ax[0].plot(t, a_t, color=SFI_COLORS["exact"], lw=2.2, label=r"trap centre $a(t)$")
ax[0].set_ylabel("x")
ax[0].legend(loc="upper right")
ax[1].plot(t, k_t, color=SFI_COLORS["exact"], lw=2.2)
ax[1].set_ylabel(r"stiffness $k(t)$")
ax[1].set_xlabel("time")
fig.suptitle("Ensemble in a drifting, stiffening trap")
fig.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inference with a time-Fourier dictionary
# ----------------------------------------
#
# Tensor a Fourier-in-time dictionary with the spatial terms:
# ``time_fourier(n) * X`` carries the **stiffness**, and
# ``time_fourier(n) * unit_vector_basis(1)`` the **moving centre**.
# :func:`~SFI.bases.time_fourier` reads the auto-injected ``time`` extra;
# with ``period=None`` its fundamental is the full trajectory duration.
# Nothing about :math:`a(t)` or :math:`k(t)` is supplied.

n_modes = 6
B = time_fourier(n_modes) * X(dim=1) & time_fourier(n_modes) * unit_vector_basis(1)

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="MSD")
inf.infer_force_linear(B)
inf.compute_force_error()

# %%
# Reconstructed stiffness and centre
# ----------------------------------
#
# The two coefficient blocks give the time functions :math:`-k(t)` (the
# ``x`` block) and :math:`k(t)\,a(t)` (the constant block); dividing
# recovers the centre.  Both match the hidden ground truth.

# sphinx_gallery_start_ignore
# ``coeff_block`` splits the flat coefficient vector into the two harmonic
# blocks without hand-computed offsets.  (``predict_time_profile`` would
# contract these for us, but it does not batch this ``time_fourier * X``
# block over time, so we keep the small local design matrix here.)
nf = 1 + 2 * n_modes
th_k, _ = inf.coeff_block((0, nf))           # x-block      -> -k(t)
th_b, _ = inf.coeff_block((nf, 2 * nf))      # const-block  ->  k(t)*a(t)


def _fourier_design(tt):
    ang = np.outer(np.arange(1, n_modes + 1), w * tt)
    rows = [np.ones_like(tt)]
    for j in range(n_modes):
        rows += [np.cos(ang[j]), np.sin(ang[j])]
    return np.stack(rows)


Fm = _fourier_design(t)
k_hat = -(np.asarray(th_k) @ Fm)
b_hat = np.asarray(th_b) @ Fm
a_hat = b_hat / np.where(np.abs(k_hat) > 1e-2, k_hat, np.nan)


def _nmse(h, g):
    return float(np.nanmean((h - g) ** 2) / np.mean(g ** 2))


axes = plot_time_profile_comparison(
    t,
    [k_t, a_t],
    [k_hat, a_hat],
    labels=[r"stiffness $k(t)$", r"trap centre $a(t)$"],
)
fig = axes[0].figure
fig.suptitle("Recovered time-dependent force field (never shown the protocol)")
fig.tight_layout()
plt.show()

print(f"k(t) reconstruction NMSE = {_nmse(k_hat, k_t):.4f}")
print(f"a(t) reconstruction NMSE = {_nmse(a_hat, a_t):.4f}")
# sphinx_gallery_end_ignore

# %%
# Sparse selection of harmonics
# -----------------------------
#
# The dictionary offers ``n_modes`` harmonics per spatial term, but the
# data support only a few.  :term:`PASTIS` prunes the rest — the surviving
# terms are exactly the harmonics built into :math:`k(t)` and
# :math:`k(t)\,a(t)`.

inf.sparsify_force(criterion="PASTIS")
inf.print_report()

# %%
# Notes
# -----
#
# - **Time as an extra.**  The reserved ``time`` extra is injected
#   automatically, per frame, by the trajectory layer (see
#   :meth:`~SFI.trajectory.TrajectoryDataset.build_extras`), so
#   :func:`~SFI.bases.time_fourier` needs no recorded protocol.  With
#   ``period=None`` the fundamental is the **full trajectory duration**;
#   pass ``period=`` to fit a known repeat time.
# - **Why an ensemble.**  The stiffness and the centre are jointly
#   identifiable only because many particles sample a spread of positions
#   at each instant; a single tightly-trapped trajectory cannot tell a
#   stiffer trap from a displaced one.
# - **Beyond traps.**  Any time-dependent force field — ramps,
#   oscillatory drives, slow aging — can be learned the same way:
#   tensor :func:`~SFI.bases.time_fourier` with whatever spatial basis the
#   problem needs, and let PASTIS keep the active modes.

stamp_output()
