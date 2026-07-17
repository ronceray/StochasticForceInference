r"""
Diagnostics — assessing fit quality
====================================

Once a fit is in hand, the next question is *should we trust it?*
The :mod:`SFI.diagnostics` submodule answers that by recomputing
standardised innovations from the fitted state function and the
inferred constant diffusion, then running a battery of statistical
tests.

This demo fits two models to the same 1-D double-well trajectory:

1. a **well-specified** cubic basis :math:`\{1, x, x^2, x^3\}`, and
2. a **deliberately wrong** linear basis :math:`\{1, x\}` (missing the
   cubic term),

and contrasts their diagnostic reports side-by-side.

.. rubric:: Tags

diagnostics · overdamped · linear · 1D · synthetic
"""

# sphinx_gallery_tags = ["diagnostics", "overdamped", "linear", "1D", "synthetic"]
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
from _gallery_utils.helpers import apply_style, stamp_output

apply_style()
# sphinx_gallery_end_ignore

# %%
# Simulate a 1-D double-well process
# ----------------------------------
#
# A standard test bed for misspecification: :math:`F(x) = x - x^3` has
# a *cubic* restoring force with two stable fixed points at
# :math:`x = \pm 1`. We sample at a moderate :math:`\Delta t = 0.05` —
# coarse enough that a missing force term leaves a residual the
# diagnostics can detect (see the closing note on sampling interval).

import SFI
from SFI.bases import monomials_up_to, unit_axes, x_components
from SFI.diagnostics import assess, plot_summary
from SFI.langevin import OverdampedProcess

# Double-well force F(x) = x − x³, written compositionally.
(_x,) = x_components(1)
(_ex,) = unit_axes(1)
F_sf = (_x - _x * _x * _x) * _ex

proc = OverdampedProcess(F_sf, D=0.15)
proc.initialize(jnp.array([0.5], dtype=jnp.float32))
coll = proc.simulate(
    dt=0.05, Nsteps=8000, key=random.PRNGKey(2), prerun=200, oversampling=10,
)

# %%
# Fit (1): well-specified cubic basis
# -----------------------------------

inf_good = SFI.OverdampedLangevinInference(coll)
inf_good.compute_diffusion_constant(method="WeakNoise")
B_good = monomials_up_to(order=3, dim=1, rank='vector')  # {1, x, x², x³}
inf_good.infer_force_linear(B_good, M_mode="Ito")
inf_good.compute_force_error()

# %%
# Fit (2): deliberately misspecified — linear force only (no cubic)
# ------------------------------------------------------------------

inf_bad = SFI.OverdampedLangevinInference(coll)
inf_bad.compute_diffusion_constant(method="WeakNoise")
B_bad = monomials_up_to(order=1, dim=1, rank='vector')  # {1, x} — misses x³
inf_bad.infer_force_linear(B_bad, M_mode="Ito")
inf_bad.compute_force_error()

# %%
# Run diagnostics on both
# -----------------------
#
# Each flagged issue in the ``-- Flags --`` block carries a one-line
# action hint pointing at the likely cure.

rep_good = assess(inf_good, level="standard")
rep_bad = assess(inf_bad, level="standard")

print("\n### Well-specified fit ###")
rep_good.print_summary()

print("\n### Misspecified fit ###")
rep_bad.print_summary()

# %%
# Visual summary
# --------------
#
# :func:`~SFI.diagnostics.plot_summary` lays out the three canonical
# panels — Q--Q, residual histogram, and residual autocorrelation (with
# a squared-residual overlay for volatility clustering) — for a given
# report.

fig_good = plot_summary(rep_good)
fig_good.suptitle("Well-specified fit", y=1.01)

# %%

fig_bad = plot_summary(rep_bad)
fig_bad.suptitle("Misspecified fit (constant force)", y=1.01)

plt.show()

# %%
# Reading the figures
# -------------------
#
# The well-specified cubic fit shows residuals lining up on the Q--Q
# diagonal, a histogram that hugs the :math:`\mathcal N(0,1)` density,
# and an ACF inside the Bartlett band; its printed report lists no
# flags.
#
# The misspecified linear fit shows the diagnostic signature of a
# missing structural term: the leftover cubic force tracks the slow
# well-to-well motion, so the residuals are **autocorrelated** (the
# Ljung--Box test fails) and the realised NMSE sits well above the
# predicted (sampling-noise) value — the data does not support the
# linear model.
#
# .. note::
#    How visible a missing term is depends on the sampling interval.
#    At very fine :math:`\Delta t` the diffusion estimate can absorb a
#    weak force misspecification, leaving the marginal residual tests
#    looking clean; coarser sampling (as here) makes the leftover
#    structure show up in the autocorrelation and NMSE-consistency
#    checks.

stamp_output()
