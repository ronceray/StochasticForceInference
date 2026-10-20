# TODO: review this file
from SFI.statefunc.structexpr import StructuredExpr

# ── Re-build all building blocks with the full expanded basis ─────────────────
# 19-term Landau-Ginzburg (LG) basis:
#   - Systematically includes all symmetry-allowed terms up to 2nd order in
#     fields (u, Q) and 2nd order in derivatives for both sectors.
#   - Two degeneracies resolved:
#
#   DESIGN NOTES:
#   • (u·∇)u EXCLUDED from u-sector: exact 2D identity E·u + Ω·u = (u·∇)u
#     makes those three terms perfectly degenerate → huge canceling coefficients.
#
#   • "Q" and "|Q|²Q" REPLACED by single Landau term (|Q|²-S₀²)Q:
#     In the bulk nematic |Q|² ≈ S₀² = const, so Q and |Q|²Q are near-
#     degenerate (columns nearly parallel in the regression matrix), producing
#     huge cancelling coefficients of order 10^7.  The combination (|Q|²-S₀²)Q
#     is orthogonal in bulk and cleanly encodes the Landau restoring potential
#     V ∝ (|Q|²-S₀²)².  Fitted coefficient c ≈ -0.221, τ_relax ≈ 13.2 frames.

Omega_expr  = layout.vorticity(u)
E_strain    = layout.strain_rate(u)
lap_u       = layout.lap(u)
div_Q       = layout.div(Q)
div_u       = layout.div(u)
grad_div_u  = layout.grad(div_u)
adv_Q       = layout.advection_by(u, Q)
lapQ_expr   = layout.lap(Q)
lap_E       = layout.lap(E_strain)

# Landau-Ginzburg Q term: equilibrium |Q|² computed from data
# (requires X4d to be available in the calling scope)
S0sq    = float(np.mean(X4d[:, :, :, CH_Q11]**2 + X4d[:, :, :, CH_Q12]**2))
S2      = StructuredExpr.einsum("ij,ij->", Q, Q).with_label("|Q|²")
LandauQ = ((S2 - S0sq) * Q).with_label("(|Q|²-S₀²)Q")

# ── u-sector (10 terms): degree ≤2 in fields, ≤2 in derivatives ──────────────
u_force = (
    lap_u                                                                        # ∇²u         viscosity
    & u                                                                          # u            Stokes drag
    & div_Q                                                                      # ∇·Q          Q-gradient force (Ericksen-Leslie)
    & grad_div_u                                                                 # ∇∇·u         compressional / gradient of divergence
    & StructuredExpr.einsum("ij,j->i", Q,          u         ).with_label("Q·u")   # Q·u          active back-stress
    & StructuredExpr.einsum("ij,j->i", E_strain,   u         ).with_label("E·u")   # E·u          strain coupling
    & StructuredExpr.einsum("ij,j->i", Omega_expr, u         ).with_label("Ω·u")   # Ω·u          vorticity coupling
    & (div_u * u).with_label("(∇·u)u")                                              # (∇·u)u       compressional nonlinearity
    & StructuredExpr.einsum("ij,j->i", Q,          lap_u     ).with_label("Q·∇²u") # Q·∇²u        higher-order elastic coupling
    & StructuredExpr.einsum("ij,j->i", Q,          grad_div_u).with_label("Q·∇∇u") # Q·∇∇u        higher-order compressional coupling
)

# ── Q-sector (9 terms): degree ≤2 in fields, ≤2 in derivatives ───────────────
Q_force = (
    LandauQ                                                                      # (|Q|²-S₀²)Q  Landau potential: c≈-0.221, τ≈13.2 fr
    & lapQ_expr                                                                  # ∇²Q           Frank elasticity
    & E_strain                                                                   # E[u]          flow-alignment (stretching/Jeffery)
    & lap_E                                                                      # ∇²E[u]        higher-order flow coupling
    & (Omega_expr @ Q - Q @ Omega_expr).with_label("[Ω,Q]")                     # [Ω,Q]         co-rotation
    & adv_Q                                                                      # (u·∇)Q        advection
    & (E_strain @ Q + Q @ E_strain).with_label("{E,Q}")                          # {E,Q}         flow-alignment (symmetric)
    & StructuredExpr.einsum("i,j->ij", u, u).with_label("u⊗u")                  # u⊗u           flow-induced ordering
    & (div_u * Q).with_label("(∇·u)Q")                                          # (∇·u)Q        compressional Q damping
)

BASIS_FULL = layout.embed(rank=1, u=u_force, Q=Q_force)

print(f"Full basis: {len(list(BASIS_FULL.labels))} terms  (S₀² = {S0sq:.5f},  S₀ = {S0sq**0.5:.5f})")
