"""
Velocity-dependent noise — underdamped multiplicative diffusion
=================================================================

Recover a **velocity-dependent diffusion field** :math:`D(v)` for an
inertial particle, from positions only.  Many active and driven systems
fluctuate harder the faster they move — motility noise, flight-force
fluctuations, turbulent drag.  Underdamped SFI reconstructs the
unobserved velocity and infers :math:`F(x,v)` and :math:`D(x,v)`
jointly; here the noise amplitude doubles within the explored speed
range and is recovered model-free.

.. rubric:: Tags

synthetic · underdamped · multiplicative-noise · diffusion-field · 1D
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "multiplicative-noise", "diffusion-field", "1D"]
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
from SFI.utils.plotting import phase2d_scalar, plot_profile_1d

apply_style()
# sphinx_gallery_end_ignore
# %%
# Model and simulation
# ---------------------
#
# A damped harmonic oscillator whose bath kicks grow with speed:
#
# .. math::
#
#    \dot x = v, \qquad
#    dv = (-k x - \gamma v)\,dt + \sqrt{2 D(v)}\,dW,
#    \qquad D(v) = D_0\left(1 + (v/v_c)^2\right),
#
# in the **Itô convention**: the force SFI infers is the Itô drift of
# the velocity equation.  Both fields are composed from coordinate
# primitives — note the velocity primitive ``v`` entering the diffusion.

import SFI
from SFI.bases import identity_matrix_basis, unit_axes, v_components, x_components
from SFI.langevin import UnderdampedProcess

k = 1.0       # stiffness
gamma = 1.0   # friction
D0 = 0.3      # diffusion at rest
vc = 1.0      # velocity scale of the noise growth
dt = 0.01
Nsteps = 80_000

(x,) = x_components(1)
(v,) = v_components(1)
(ex,) = unit_axes(1)
I = identity_matrix_basis(1)

F_model = (-k * x - gamma * v) * ex            # Itô drift of dv
D_model = D0 * (1.0 + (v / vc) ** 2) * I       # D(v) — multiplicative in v

proc = UnderdampedProcess(F=F_model, D=D_model,
                          theta_F=jnp.ones(F_model.n_features),
                          theta_D=jnp.ones(D_model.n_features))
proc.initialize(jnp.array([0.0]), v0=jnp.array([0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(5),
                     prerun=500, oversampling=20)
print(f"Trajectory: {coll.T} frames, dt={dt} (positions only)")

# %%
# Phase portrait
# ---------------
#
# Velocity is *not* observed — for plotting we reconstruct it by finite
# differences, exactly as the inference engine does internally.  Fast
# excursions (top and bottom) are noticeably noisier than slow ones.

# sphinx_gallery_start_ignore
# Reconstruct v by finite differences (exactly as the engine does) and build a
# phase-space (x, v) collection so the canonical plotters can sweep the v axis.
_t, _Xpos, _ = coll.to_arrays()
v_rec = coll.velocity_array(scheme="central")
coll_xv = SFI.TrajectoryCollection.from_arrays(
    X=np.concatenate([np.asarray(_Xpos), np.asarray(v_rec)], axis=-1), dt=dt,
)

fig, ax = plt.subplots(figsize=(5, 4))
phase2d_scalar(
    coll_xv,
    color_fn=lambda M: D0 * (1 + (M[:, 1] / vc) ** 2),  # local D(v) at each midpoint
    dims=(0, 1), cmap="plasma", colorbar_label=r"local $D(v)$",
    linewidth=0.6, alpha=0.7, ax=ax,
)
ax.set_xlabel("x")
ax.set_ylabel(r"$\hat v$ (reconstructed)")
ax.set_title("Phase portrait — colored by local noise amplitude")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Inference
# ----------
#
# The underdamped parametric sequence: :meth:`infer_force` fits the
# Itô drift :math:`F(x,v)`, then :meth:`infer_diffusion` fits
# :math:`D(x,v)` on a polynomial basis in *both* variables
# (``include_v=True``).  The basis spans :math:`\{1, x, v, x^2, xv,
# v^2\}` — the inference must discover that only :math:`1` and
# :math:`v^2` carry weight.

from SFI.bases import monomials_up_to

B_force = monomials_up_to(order=1, dim=1, include_v=True, rank="vector")
B_diff = monomials_up_to(order=2, dim=1, include_v=True, rank="symmetric_matrix")

inf = SFI.UnderdampedLangevinInference(coll)
inf.infer_force(B_force)
inf.infer_diffusion(B_diff)

inf.compare_to_exact(model_exact=proc)
inf.print_report()
print(inf.summary(field="diffusion"))
print(f"  (true: [1] = {D0:+.3f}, [v²] = {D0 / vc**2:+.3f}, rest 0)")

# %%
# Recovered diffusion profile
# -----------------------------
#
# Evaluating the inferred tensor along the velocity axis recovers the
# parabolic noise profile; a constant-:math:`D` analysis would report
# only the mean.

# The inferred tensor as a callable on phase-space points [x, v]; sweeping the
# v axis (x held at 0) recovers the parabolic noise profile.
def D_inferred(pts):
    pts = jnp.asarray(pts)
    return inf.diffusion_inferred(pts[:, :1], v=pts[:, 1:2])  # (N, 1, 1)


def D_exact(pts):
    v = np.asarray(pts)[:, 1]
    return D0 * (1 + (v / vc) ** 2)


# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.5))
plot_profile_1d(
    coll_xv, D_inferred, exact_field=D_exact, dim=1, component=0, samples=True,
    ax=ax, label_exact="Exact $D(v)$", label_inferred="Inferred $D(v)$",
)
ax.set_xlabel("v")
ax.set_ylabel("D(v)")
ax.set_title("Velocity-dependent diffusion recovery")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore
# %%
# When is this hard?
# --------------------
#
# Velocity-dependent noise has a genuinely hard regime.  With
# :math:`D(v) \propto v^2` the velocity distribution develops
# **power-law tails** (tail exponent :math:`2 + \gamma/D_0`): if
# friction is weak relative to the noise gradient, rare high-speed
# bursts dominate the statistics, and no finite sampling rate resolves
# them — estimates of the noise floor :math:`D_0` then degrade no
# matter how small :math:`dt` is.  Here :math:`\gamma/D_0 \approx 3.3`
# keeps the tails integrable and the inference quantitative.  For
# strongly driven systems, prefer *saturating* noise models
# (e.g. :math:`D_0 + \Delta D\, v^2/(v^2 + v_s^2)`) — see the
# state-dependent diffusion benchmark for that case.

stamp_output()
