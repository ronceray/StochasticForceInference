"""
Anisotropic diffusion tensor field
====================================

Recover a **position-dependent, anisotropic diffusion tensor**
:math:`D(\\mathbf{x})` from a single 2D trajectory — model-free.  A
tracer in a ring trap experiences fluctuations whose radial component
grows with distance from the center (think of a probe in a radially
stretched gel, or near a topological defect in an active film): the
local noise ellipse rotates with position.  A polynomial tensor basis
recovers the full field, including its principal axes.

.. rubric:: Tags

synthetic · overdamped · multiplicative-noise · anisotropic · 2D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "multiplicative-noise", "anisotropic", "2D"]
# sphinx_gallery_thumbnail_number = 3

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
from SFI.utils.plotting import phase2d, plot_tensor_field

apply_style()
# sphinx_gallery_end_ignore
# %%
# Model and simulation
# ---------------------
#
# A particle in a ring potential :math:`U = \tfrac{k}{4}(r^2 - R^2)^2`
# with a radially anisotropic diffusion tensor (Itô convention):
#
# .. math::
#
#    D(\mathbf{x}) = D_0\,\mathbb{I}
#      + a\, r^2\, \hat{\mathbf{r}}\hat{\mathbf{r}}^{\!\top}
#    = D_0\,\mathbb{I} + a \begin{pmatrix} x^2 & xy \\ xy & y^2 \end{pmatrix}.
#
# Radial kicks grow with :math:`r`; tangential noise stays at
# :math:`D_0`.  Both fields are polynomial, so we can build them
# compositionally from coordinate and matrix-template primitives.

import SFI
from SFI.bases import identity_matrix_basis, symmetric_matrix_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

k = 4.0       # ring trap stiffness
R = 1.5       # ring radius
D0 = 0.2      # isotropic (tangential) diffusion
a = 0.15      # radial anisotropy growth
dt = 0.005
Nsteps = 50_000

x, y = x_components(2)
ex, ey = unit_axes(2)
I = identity_matrix_basis(2)
S = symmetric_matrix_basis(2)            # templates Sxx, Sxy, Syy
Sxx, Sxy, Syy = S[0], S[1], S[2]

r2 = x**2 + y**2
F_model = (-k * (r2 - R**2)) * (x * ex + y * ey)
D_model = D0 * I + a * (x**2 * Sxx + x * y * Sxy + y**2 * Syy)

proc = OverdampedProcess(F=F_model, D=D_model,
                         theta_F=jnp.ones(F_model.n_features),
                         theta_D=jnp.ones(D_model.n_features))
proc.initialize(jnp.array([R, 0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(11),
                     prerun=500, oversampling=10)

# %%
# Trajectory
# -----------
#
# The tracer diffuses around the annulus, sampling all orientations of
# the local anisotropy.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(4.5, 4.5))
phase2d(coll, dims=(0, 1), ax=ax, color=SFI_COLORS["data"], linewidth=0.3, alpha=0.6)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(R * np.cos(th), R * np.sin(th), ls=":", lw=1, color="#808080")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_title("Ring-trap trajectory")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Inference
# ----------
#
# Same canonical sequence as in 1D — the only change is the rank-2
# tensor basis for the diffusion field: degree-2 matrix monomials,
# spanning all symmetric-tensor fields with polynomial entries
# (18 features; the truth uses 5 of them).

from SFI.bases import monomials_up_to

B_force = monomials_up_to(order=3, dim=2, rank="vector")
B_diff = monomials_up_to(order=2, dim=2, rank="symmetric_matrix")

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()
inf.infer_force_linear(B_force)
inf.compute_force_error()
inf.infer_diffusion_linear(B_diff)

inf.compare_to_exact(model_exact=proc)
inf.print_report()

# %%
# Tensor glyphs: exact vs inferred
# ----------------------------------
#
# Each glyph is the ellipse :math:`\mathbf{u}^{\!\top} D^{-1}\mathbf{u}
# = \text{const}` of the local tensor — long axis along the fast
# direction.  Exact field on the left, inferred on the right: the
# radial orientation and the :math:`r^2` growth are both recovered.

# Glyph positions on a polar grid covering the sampled annulus, and a
# callable for the exact tensor field D(x) = D0 I + a r^2 r-hat r-hat^T.
r_glyph = np.linspace(0.9, 2.0, 4)
th_glyph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
pts = np.array([[r * np.cos(t), r * np.sin(t)] for r in r_glyph for t in th_glyph])


def D_exact_field(p):
    p = np.asarray(p).reshape(-1, 2)
    return np.array([D0 * np.eye(2) + a * np.outer(q, q) for q in p])


def D_inferred_field(p):
    return inf.diffusion_inferred(jnp.asarray(p))


# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
plt.sca(axes[0])
plot_tensor_field(coll, D_exact_field, mode="ellipse", positions=pts,
                  center=(0.0, 0.0), radius=2.4, scale=0.55,
                  color=SFI_COLORS["exact"])
axes[0].set_title("Exact $D(\\mathbf{x})$")
plt.sca(axes[1])
plot_tensor_field(coll, D_inferred_field, mode="ellipse", positions=pts,
                  center=(0.0, 0.0), radius=2.4, scale=0.55,
                  color=SFI_COLORS["inferred"])
axes[1].set_title("Inferred $D(\\mathbf{x})$")
for ax in axes:
    ax.plot(R * np.cos(th), R * np.sin(th), ls=":", lw=0.8, color="#808080")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Radial and tangential eigenvalues
# -----------------------------------
#
# Projecting the inferred tensor onto the local radial and tangential
# directions separates the two physical components: the radial
# diffusivity grows as :math:`D_0 + a r^2` while the tangential one
# stays flat at :math:`D_0`.

r_prof = np.linspace(0.8, 2.1, 30)
th_s = np.linspace(0, 2 * np.pi, 24, endpoint=False)
D_rr = np.zeros((len(r_prof), len(th_s)))
D_tt = np.zeros_like(D_rr)
for j, t_ in enumerate(th_s):
    pts_ray = np.column_stack([r_prof * np.cos(t_), r_prof * np.sin(t_)])
    D_ray = np.asarray(inf.diffusion_inferred(jnp.array(pts_ray)))
    rhat = np.array([np.cos(t_), np.sin(t_)])
    that = np.array([-np.sin(t_), np.cos(t_)])
    D_rr[:, j] = np.einsum("i,nij,j->n", rhat, D_ray, rhat)
    D_tt[:, j] = np.einsum("i,nij,j->n", that, D_ray, that)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(r_prof, D0 + a * r_prof**2, lw=2, color=SFI_COLORS["exact"],
        label=r"Exact $D_{rr} = D_0 + a r^2$")
ax.plot(r_prof, np.full_like(r_prof, D0), lw=2, ls=":",
        color=SFI_COLORS["exact"], label=r"Exact $D_{\theta\theta} = D_0$")
ax.plot(r_prof, D_rr.mean(axis=1), "--", lw=2, color=SFI_COLORS["inferred"],
        label="Inferred $D_{rr}$")
ax.fill_between(r_prof, D_rr.min(axis=1), D_rr.max(axis=1),
                color=SFI_COLORS["inferred"], alpha=0.2)
ax.plot(r_prof, D_tt.mean(axis=1), "--", lw=2, color=SFI_COLORS["highlight"],
        label=r"Inferred $D_{\theta\theta}$")
ax.fill_between(r_prof, D_tt.min(axis=1), D_tt.max(axis=1),
                color=SFI_COLORS["highlight"], alpha=0.2)
ax.set_xlabel("r")
ax.set_ylabel("eigenvalue of $D$")
ax.set_title("Radial vs tangential diffusivity")
ax.legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore
#
# Shaded bands show the spread over sampling angles — a measure of how
# isotropically the basis error is distributed.  The same workflow
# applies unchanged to experimental 2D tracking data; see the
# :doc:`multiplicative-noise demo <multiplicative_diffusion_demo>` for
# the Itô-convention caveats that come with state-dependent noise.

stamp_output()
