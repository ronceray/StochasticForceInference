# SFI/inference/underdamped.py
from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from SFI.bases.constants import constant_array
from SFI.inference.base import BaseLangevinInference
from SFI.integrate.api import integrate, integrate_rows, required_frames, shared_rows
from SFI.integrate.integrand import (
    ConstOperand,
    ExprOperand,
    Integrand,
    Term,
    TimeOperand,
)
from SFI.integrate.timeops import TimeOp, scale, stream, timeop, velocity
from SFI.statefunc import PSF, SF, Basis
from SFI.utils.maths import sqrtm_psd, stable_pinv

from .result import InferenceResultSF
from .sparse import SparseScorer

logger = logging.getLogger(__name__)

# Positional defect M_x of the symmetric ULI position x̂ = (x_{t-1}+x_t+x_{t+1})/3:
# the â–x̂ noise cross-covariance, in units of D h, entering the moments as
# -2 M_x h <D̄A⁻¹ : ∂ₓb>.  Derived (Itô–Taylor two-point expansion), not fitted.
_ULI_POSITIONAL_DEFECT = 7.0 / 18.0


# ---------------------------------------------------------------------------- #
# ULI linear force-inference presets
# ---------------------------------------------------------------------------- #
# The full (M_mode, G_mode, diffusion_method) surface is combinatorial and hard
# for users to navigate.  The choice that actually matters is the diffusion
# estimator (``noisy`` vs ``WeakNoise``), keyed on measurement noise; the broadly
# robust kinematics/Gram settings are ``M_mode='symmetric'`` and
# ``G_mode='trapeze'`` (the ULI Gram is returned transposed — grid assessments on
# the VdP limit cycle and a 2-D noise-driven OU both place it at/near the per-cell
# oracle, ~1.07-1.08x, with a better in-band worst case).  We expose two named
# presets and an ``auto`` switch.
_FORCE_LINEAR_PRESETS: dict[str, tuple[str, str, str]] = {
    # broadly applicable default — flat across measurement noise in its band
    "robust": ("symmetric", "trapeze", "noisy"),
    # sharper refinement for verified noise-free data (coarse-dt, low-noise)
    "clean":  ("symmetric", "trapeze", "WeakNoise"),
    # SFI v1.0 reproductions (explicit only; never selected by 'auto')
    "legacy-clean-v1.0": ("early", "rectangle", "MSD"),
    "legacy-noisy-v1.0": ("symmetric", "rectangle", "noisy"),
}


def _resolve_force_linear_preset(
    preset: str,
    *,
    M_mode: str | None = None,
    G_mode: str | None = None,
    diffusion_method: str | None = None,
    noise_detected: bool | None = None,
) -> tuple[tuple[str, str, str], str]:
    """Resolve a force-inference ``preset`` to a concrete mode triple.

    Parameters
    ----------
    preset : {'auto', 'robust', 'clean'}
        ``'robust'`` and ``'clean'`` map directly to fixed mode triples (see
        ``_FORCE_LINEAR_PRESETS``).  ``'auto'`` selects ``'robust'`` (the
        ``noisy`` diffusion estimator) when measurement noise is present and
        ``'clean'`` (``WeakNoise``) otherwise.
    M_mode, G_mode, diffusion_method : str, optional
        Power-user overrides.  Any non-``None`` value replaces the preset's
        choice on that axis (so explicit callers are unaffected by the preset).
    noise_detected : bool, optional
        Measurement-noise indicator for ``preset='auto'`` (``Lambda_trace > 0``).
        ``True`` -> ``robust``/``noisy``; ``False`` -> ``clean``/``WeakNoise``.
        ``None`` (indicator unavailable) falls back to the safe ``robust`` path.

    Returns
    -------
    (M_mode, G_mode, diffusion_method) : tuple of str
        The resolved mode triple.
    resolved_preset : str
        The concrete base preset used (``'robust'`` or ``'clean'``) — useful for
        metadata/logging when ``preset='auto'``.
    """
    if preset == "auto":
        base = "clean" if noise_detected is False else "robust"
    elif preset in _FORCE_LINEAR_PRESETS:
        base = preset
    else:
        choices = ", ".join(repr(p) for p in ("auto", *_FORCE_LINEAR_PRESETS))
        raise ValueError(f"unknown preset {preset!r}; choose from {choices}")

    M0, G0, D0 = _FORCE_LINEAR_PRESETS[base]
    resolved = (
        M0 if M_mode is None else M_mode,
        G0 if G_mode is None else G_mode,
        D0 if diffusion_method is None else diffusion_method,
    )
    return resolved, base


class UnderdampedLangevinInference(BaseLangevinInference):
    r"""
    Stochastic Force Inference concrete class for underdamped systems (velocities unobserved).

    Core equations (Ito convention):
        dx/dt = v
        dv/dt = F(x, v) + sqrt(2 D(x, v)) dW_t

    Only x(t) is observed; v(t) is reconstructed from finite differences ("secant velocities").

    .. physics:: Underdamped Langevin SDE
       :label: underdamped-langevin-sde
       :category: Dynamical equations

       .. math::

          \frac{\mathrm{d}x}{\mathrm{d}t} = v, \qquad
          \frac{\mathrm{d}v}{\mathrm{d}t}
          = F(x,v) + \sqrt{2\,D(x,v)}\;\mathrm{d}W_t

       Only positions :math:`x(t)` are observed; velocities are
       reconstructed from finite differences.
    """

    # ------------------------------ public API ------------------------------ #

    def infer_force_linear(
        self,
        basis: Basis,
        *,
        preset: str = "auto",
        M_mode: str | None = None,
        G_mode: str | None = None,
        diffusion_method: str | None = None,
        cross_particle: str = "all-pairs",
    ) -> None:
        r"""
        Infer F(x, v) as a linear combination of basis functions.

        The estimator is configured by a ``preset`` that consolidates the
        ``(M_mode, G_mode, diffusion_method)`` surface into two broadly-validated
        choices, with an ``auto`` switch keyed on measurement noise.  Power users
        can still pass any of the three modes explicitly to override the preset.

        Parameters
        ----------
        preset : {'auto', 'robust', 'clean', 'legacy-clean-v1.0', 'legacy-noisy-v1.0'}, default 'auto'
            * ``'robust'`` — ``symmetric`` / ``trapeze`` / ``noisy``.  The
              broadly applicable default: flat across measurement noise within
              its feasible band and never catastrophic.
            * ``'clean'`` — ``symmetric`` / ``trapeze`` / ``WeakNoise``.  A
              sharper refinement for verified noise-free data (helps mainly in
              the coarse-dt, low-noise corner); fragile under measurement noise.
            * ``'auto'`` — pick ``'robust'`` when measurement noise is detected and
              ``'clean'`` otherwise, on the sign of the localization-noise estimate
              ``Lambda_trace`` (``>0`` -> noise present).  ``Lambda_trace`` is
              reused from a prior :meth:`compute_diffusion_constant` call or
              estimated inline.  Validated on the ``vdp_uli_10x`` benchmark: the
              zero-crossing tracks the ``noisy``<->``WeakNoise`` transition with no
              harmful misroutes.
            * ``'legacy-clean-v1.0'`` — ``early`` / ``rectangle`` / ``MSD``.
              Reproduces the published SFI v1.0 clean-data convention
              (explicit only; never selected by ``'auto'``).
            * ``'legacy-noisy-v1.0'`` — ``symmetric`` / ``rectangle`` /
              ``noisy``.  Reproduces the SFI v1.0 noisy-data convention
              (explicit only).
        basis : Basis
            Rank-1 basis producing per-particle vectors.
            Must accept v=... keyword in evaluation.
        M_mode : str, optional
            Override the preset's kinematics / moments convention:
            'symmetric' | 'early' | 'anticipated'.  ``None`` uses the preset.
        G_mode : str, optional
            Override the preset's Gram construction mode: 'rectangle' | 'trapeze'
            | 'shift' | 'doubleshift' (each returned transposed).  ``None`` uses
            the preset.
        diffusion_method : str, optional
            Override the preset's instantaneous diffusion estimator used in the
            ULI correction term: 'noisy' | 'WeakNoise' | 'MSD'.  ``None`` uses the
            preset (and re-enables the ``auto`` measurement-noise switch).
        cross_particle : {'all-pairs', 'same-particle'}, default 'all-pairs'
            Interacting bases only.  The ULI correction
            :math:`\langle D_t : A^{-1} \partial_v b \rangle` uses the
            instantaneous diffusion estimator on every pair of particles,
            :math:`\sum_j D_{t,ij} : A^{-1} \partial_{v_j} b_i` (``'all-pairs'``):
            exact when particles share noise (e.g. neighbouring cell centroids
            driven through common vertices) and consistent at :math:`O(\Delta t)`
            without.  Model-free (forward-mode derivatives along the
            increments).  ``'same-particle'`` keeps each particle's own block
            only (the pre-2.0.4 behaviour), valid for noise independent between
            particles.
        """
        # Legacy alias: M_mode='auto' historically meant 'symmetric'.
        if M_mode == "auto":
            M_mode = "symmetric"

        # Resolve the mode triple from the preset.  preset='auto' selects the
        # diffusion estimator on the sign of Lambda_trace, which is set by the
        # required prior compute_diffusion_constant() call (the force inference
        # also needs its A_inv normalization).  If absent, fall back to the safe
        # 'robust' branch; the missing-diffusion error surfaces downstream.
        noise_detected: bool | None = None
        if preset == "auto" and diffusion_method is None:
            lam = getattr(self, "Lambda_trace", None)
            noise_detected = None if lam is None else (lam > 0)
        (M_mode, G_mode, diffusion_method), resolved_preset = _resolve_force_linear_preset(
            preset,
            M_mode=M_mode,
            G_mode=G_mode,
            diffusion_method=diffusion_method,
            noise_detected=noise_detected,
        )
        self.metadata["force_preset"] = preset
        if preset == "auto":
            logger.info(
                "Auto force preset -> %s (M_mode=%s, G_mode=%s, diffusion_method=%s; "
                "Lambda_trace sign: %s)",
                resolved_preset, M_mode, G_mode, diffusion_method,
                None if noise_detected is None else ("+" if noise_detected else "-"),
            )

        self._validate_basis(basis, expected_rank=1, label="force basis")
        if cross_particle not in ("all-pairs", "same-particle"):
            raise ValueError("cross_particle must be 'all-pairs' or 'same-particle'")

        # Refuse before touching any fit attribute (the previous fit stays intact).
        if hasattr(self, "force_G_full"):
            raise RuntimeError("Force has already been inferred on this object - create a new instance to re-infer.")
        self._reset_fit_state("force")  # e.g. after a parametric PSF fit

        self.force_basis = basis
        self.__force_cross_particle__ = cross_particle
        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.__force_diffusion_method__ = diffusion_method
        self.metadata["force_M_mode"] = M_mode
        self.metadata["force_G_mode"] = G_mode
        self.metadata["force_diffusion_method"] = diffusion_method

        # Structural arrays are exposed only for this evaluation and never
        # persisted on the dataset (see ``_structural_scope``).
        with self._structural_scope(self.force_basis):
            # Gram, moments and the velocity-gradient correction read different
            # frames; with missing frames they must still sum over the same
            # rows, or C = G^-1 M is biased by the ratio of row counts.
            with shared_rows(self._force_linear_frames()):
                # The Gram's row measure (particle-time) normalizes rates and model selection.
                self.force_G_full, self._force_row_measure = self._force_G_matrix(return_measure=True)
                self.force_moments = self._force_moments()

        self._check_gram_conditioning(self.force_G_full, label="force")
        # Solve and expose a scorer for further model selection.
        self.force_scorer = SparseScorer(M=self.force_moments, G=self.force_G_full)
        self._update_force_coefficients(self.force_scorer.total_C)
        self.metadata["force_method"] = "linear"

    def infer_diffusion_linear(
        self,
        basis: Basis,
        *,
        M_mode: str = "noisy",
        G_mode: str = "rectangle",
    ) -> None:
        """
        Fit D(x, v) as a linear combination of basis functions.
        """
        if M_mode == "auto":
            M_mode = "noisy"

        # Refuse before touching any fit attribute (the previous fit stays intact).
        if hasattr(self, "diffusion_moments"):
            raise RuntimeError(
                "Diffusion has already been inferred on this object - create a new instance to re-infer."
            )
        self._reset_fit_state("diffusion")  # e.g. after a parametric infer_diffusion

        self.diffusion_basis = basis
        self.__diffusion_M_mode__ = M_mode
        self.__diffusion_G_mode__ = G_mode
        self.metadata["diffusion_M_mode"] = M_mode
        self.metadata["diffusion_G_mode"] = G_mode

        with self._structural_scope(self.diffusion_basis):
            # Gram and moments over one common row set (see infer_force_linear).
            frames = required_frames(self._diffusion_G_matrix(program_only=True),
                                     self._diffusion_moments(program_only=True))
            self._diffusion_row_frames = frames  # reused by compute_diffusion_error
            with shared_rows(frames):
                self.diffusion_G_full, self._diffusion_row_measure = self._diffusion_G_matrix(return_measure=True)
                self.diffusion_moments = self._diffusion_moments()

        self.diffusion_scorer = SparseScorer(M=self.diffusion_moments, G=self.diffusion_G_full)
        self._update_diffusion_coefficients(self.diffusion_scorer.total_C)
        self.metadata["diffusion_method"] = "linear"

    def _estimate_lambda_trace(self) -> float:
        """Estimate the localization (measurement) noise Λ and cache its trace.

        Sets ``self.Lambda`` (the ULI 'Lambda' instantaneous estimator,
        time-averaged) and ``self.Lambda_trace = tr(Λ)`` and returns the trace.
        ``Lambda_trace > 0`` indicates measurement noise dominates the
        velocity-increment autocorrelation; ``< 0`` indicates force persistence
        dominates (clean dynamics).  Shared by :meth:`compute_diffusion_constant`
        and the ``preset='auto'`` switch in :meth:`infer_force_linear`.
        """
        L_op = self.get_diffusion_timeop("Lambda")
        L_prog = Integrand(times=[L_op], terms=[Term(eq="imn->imn", ops=(L_op.alias,))])
        # Diffusion/noise are per-point quantities: combine per-point, not dt-weighted.
        self.Lambda = integrate(
            self.data, L_prog, reduce="mean", weight_by_dt=False,
            chunk_target_bytes=self._chunk_target_bytes,
        )
        self.Lambda_trace = float(jnp.trace(self.Lambda))
        logger.info("Measurement noise trace: %s", self.Lambda_trace)
        return self.Lambda_trace

    def compute_diffusion_constant(self, method: str = "auto") -> None:
        """
        Underdamped constant diffusion inference (ULI estimators).

          1) Estimate measurement noise Λ with method "Lambda".
          2) If method == 'auto': choose 'noisy' if tr(Λ)>0 else 'WeakNoise'.
          3) Average chosen instantaneous estimator to obtain D̄.
          4) Set A = 2 D̄ and derive A_inv, sqrtA, sqrtA_inv.
        """
        if hasattr(self, "diffusion_average"):
            raise RuntimeError("Diffusion already computed; create a new inference object to recompute.")

        # 1) measurement noise Λ
        self._estimate_lambda_trace()

        # 2) select instantaneous estimator
        if method == "auto":
            method = "noisy" if self.Lambda_trace > 0 else "WeakNoise"
        self.metadata["diffusion_constant_method"] = method

        # 3) time-average instantaneous diffusion
        D_op = self.get_diffusion_timeop(method)
        D_prog = Integrand(times=[D_op], terms=[Term(eq="imn->imn", ops=(D_op.alias,))])
        self.diffusion_average = integrate(
            self.data, D_prog, reduce="mean", weight_by_dt=False,
            chunk_target_bytes=self._chunk_target_bytes,
        )
        self.diffusion_inferred = constant_array(self.diffusion_average)

        # 4) normalization matrices
        self.A = 2.0 * self.diffusion_average
        self.A_inv = jnp.linalg.inv(self.A)
        self.sqrtA = sqrtm_psd(self.A)
        self.sqrtA_inv = jnp.linalg.inv(self.sqrtA)

    def compute_entropy_production(
        self, *, support: str = "current", coefficients=None
    ) -> dict:
        r"""Estimate the entropy production of the observed dynamics.

        Plug-in Stratonovich log path-probability ratio of the fitted
        acceleration field, split into its time-reversal even/odd parts in
        the (unobserved, reconstructed) velocity — :math:`x` even,
        :math:`v` odd, :math:`\hat F^{\pm}(x,v) = \tfrac12[\hat F(x,v) \pm
        \hat F(x,-v)]`:

        .. physics:: Entropy production estimator (underdamped, linear basis)
           :label: entropy-production-inference-underdamped
           :category: Observable

           .. math::

              \Delta\hat S
              \;=\; \sum_t \bigl(\mathrm{d}\hat v_t \circ
                    - \hat F^{+}\,\mathrm{d}t\bigr)^\top
                    \bar D^{-1}\, \hat F^{-}
              \;=\; 2\,C^\top \varphi^{-} \;-\; 2\,C^\top G^{+-} C ,

           .. math::

              \varphi^{-}_a = \sum_t \mathrm{d}t\;
                 \hat a_t \cdot A^{-1} b^{-}_a(\hat x_t, \hat v_t),
              \qquad
              G^{+-}_{ab} = \sum_t \mathrm{d}t\;
                 b^{+}_a \cdot A^{-1} b^{-}_b ,
              \qquad A = 2\bar D,

           with :math:`b^{\pm}(x,v) = \tfrac12[b(x,v) \pm b(x,-v)]`, the ULI
           symmetric kinematics :math:`(\hat x, \hat v, \hat a)`, and
           :math:`C` the coefficients.  In a steady state the
           expectation equals the entropy production rate whenever
           :math:`\nabla_v \cdot F^{+} = 0` (standard mechanical even
           forces).

           **Finite** :math:`\Delta t`.  The moment objects carry
           :math:`O(h)` kinematic biases (:math:`h = \Delta t`), removed by
           the counterterm :math:`\Delta S_{\Delta t} = C^\top K C + l^\top
           C` of :meth:`_entropy_dt_programs`, valid for any :math:`C`.  The
           coefficients are the force fit's, with its own finite-:math:`\Delta
           t` and noise properties; their first-order kinematic bias is
           *reported* as ``Sdot_dt_bias`` (see Notes).  The fluctuation scale
           is set by the odd
           quadratic
           :math:`Q^{-} = 2\,C^\top G^{--} C
           \approx \int \hat F^{-}\!\cdot \bar D^{-1} \hat F^{-}\,\mathrm{d}t`
           — the log ratio's two large contributions fluctuate even where
           their means cancel:

           .. math::

              \delta \Delta S = \max\Bigl[
              \sqrt{2\,Q^{-} + (2 N^{-})^2},\;
              \delta\Delta S_{\mathrm{block}}\Bigr],
              \qquad
              \dot S_{\mathrm{bias}} = \frac{2 N^{-}}{\tau_N},

           where :math:`N^{-} = \operatorname{rank} G^{--}` counts the
           odd-sector degrees of freedom (this reduces to the overdamped
           formula, where the odd sector is everything) and
           :math:`\delta\Delta S_{\mathrm{block}}` is an empirical
           block-variance estimate that additionally captures the
           slowly-correlated fluctuations of the
           :math:`F^{+}\!\cdot\bar D^{-1} F^{-}` current term.

        Parameters
        ----------
        support : {"current", "full"}
            ``"current"`` (default) uses the active (possibly sparsified)
            support.  Because off-support coefficients are zero, the
            plug-in value :math:`\Delta\hat S` is fixed by the fit —
            ``support`` only changes the reported :math:`N^{-}` / bias
            (unlike the overdamped projection estimator).
        coefficients : array, optional
            Full-basis coefficient vector to plug in instead of this
            object's fit — the **cross-fitting** hook.  Evaluating with
            coefficients fitted on *independent* data removes the
            same-sample plug-in correlation bias (see Notes): split the
            trajectory in two, fit each half, and average the two cross
            evaluations.

        Returns
        -------
        dict
            ``{"Sdot", "Sdot_error", "Sdot_bias", "Sdot_debiased", "DeltaS",
            "error_DeltaS", "DeltaS_debiased", "tauN", "Nminus",
            "DeltaS_phi", "DeltaS_G", "DeltaS_dt", "Q_odd",
            "entropy_odd_ratio", "Sdot_dt_bias"}`` — plain floats, units of
            :math:`k_B` (nats).  ``DeltaS = DeltaS_phi - DeltaS_G +
            DeltaS_dt``.  ``Sdot_dt_bias`` is the first-order estimate of
            the rate bias due to the coefficients' finite-:math:`\Delta t`
            error (not subtracted; ``None`` for interacting bases and for
            fits with the ``noisy`` measurement-noise correction).
            ``Sdot_debiased = Sdot - Sdot_bias`` subtracts the odd-sector
            fluctuation bias (the AIC-style :math:`+\tfrac12` per odd degree
            of freedom on the plug-in information; provisional, see Notes).
            ``entropy_odd_ratio = DeltaS / Q_odd`` is the dimensionless
            irreversible fraction of the odd sector (:math:`\approx 0` at
            equilibrium).

        Notes
        -----
        - **Finite-:math:`\Delta t` bias.**  With exact coefficients the
          counterterm leaves an :math:`O(h^2)` residual.  The dominant
          finite-:math:`\Delta t` bias, however, is in the coefficients:
          the default ULI fit (trapezoid Gram on secant velocities) has
          :math:`O(h)` errors — on the driven trap the curl comes out
          :math:`\approx(1+1.3\,\gamma\Delta t)` too strong — which the
          functional amplifies to :math:`\approx +2.2\,\gamma\Delta t`
          relative on :math:`\dot S` (1.24, 1.52, 2.22 :math:`\sigma` at
          :math:`\gamma\Delta t = 0.1, 0.2, 0.4`, exact infinite-data
          values).  ``Sdot_dt_bias`` estimates it to first order: the
          residual of a kinematically consistent normal system
          (:meth:`_entropy_force_system`, clean kinematics) at the fit's
          coefficients, propagated through the gradient of
          :math:`\Delta\hat S`.  It is an order-of-magnitude flag, not a
          correction: accurate to :math:`\gamma\Delta t \approx 0.2` on
          the linear trap, to :math:`\approx 0.05` with cubic friction
          (linearised at the biased coefficients).  A bias comparable to
          ``Sdot`` flags sampling too coarse for the fit.
        - **Error theory.**  The bias/error formulas above mirror the
          overdamped derivation with the odd-sector fluctuation scale
          (provisional).
        - **Same-sample plug-in bias.**  With same-sample coefficients, the
          correlation between the fit noise and the moment noise produces
          a positive :math:`O(1/\tau_N)` bias with a large prefactor
          (empirically :math:`\sim +0.5\,\sigma` at :math:`\tau_N \gamma
          \sim 300` with 8 basis features; not subtracted).  It decays with
          trajectory length and is removed by **cross-fitting** via the
          ``coefficients`` argument.
        - **Resolution prerequisite.**  Velocity reconstruction requires
          :math:`\Delta t \lesssim \tau_v` (velocity correlation time); a
          warning is emitted when :math:`\hat\gamma\,\Delta t > 0.5` with
          :math:`\hat\gamma = \mathrm{Tr}\,\bar D / \langle|\hat
          v|^2\rangle`.
        - The contraction uses the constant average :math:`\bar D`;
          measurement noise adds an :math:`O(\Lambda/\Delta t^2)` bias to
          the reconstructed kinematics (warning when
          :math:`\mathrm{Tr}\,\Lambda` is non-negligible relative to
          :math:`\mathrm{Tr}\,\bar D\,\Delta t^3`).
        - Parametric fits (:meth:`infer_force`) are not supported yet.
        """
        if self.metadata.get("force_method") != "linear":
            raise RuntimeError(
                "compute_entropy_production() requires a prior infer_force_linear() "
                "fit; parametric fits (infer_force) are not supported yet."
            )

        dt_val = float(self.data.peek_row(require={"dt"})["dt"])
        trD = float(jnp.trace(self.diffusion_average))
        lam = float(getattr(self, "Lambda_trace", 0.0))
        if lam > 0.01 * trD * dt_val**3:
            logger.warning(
                "Measurement noise detected (Tr Lambda = %.3g vs Tr(D) dt^3 = %.3g): "
                "reconstructed velocities carry an O(Lambda/dt^2) bias that is NOT "
                "included in the reported error bar.",
                lam, trD * dt_val**3,
            )
        v2 = self._mean_v2_trace()
        g_hat = trD / max(v2, 1e-300)
        if g_hat * dt_val > 0.5:
            logger.warning(
                "Sampling interval is coarse (estimated gamma*dt = %.2g > 0.5): "
                "velocities are underresolved and the reconstruction-based "
                "entropy estimate is biased.  Use finer sampling (dt << tau_v).",
                g_hat * dt_val,
            )

        if support == "current":
            S = self.force_support
        elif support == "full":
            S = jnp.arange(self.force_coefficients_full.shape[0])
        else:
            raise ValueError(f"support must be 'current' or 'full', got {support!r}")
        if coefficients is not None:
            coefficients = jnp.asarray(coefficients)
            if coefficients.shape != self.force_coefficients_full.shape:
                raise ValueError(
                    "coefficients must match the full basis size "
                    f"{self.force_coefficients_full.shape}, got {coefficients.shape}."
                )

        # Finite-dt (see Notes): the moment objects on the symmetric kinematics
        # with their O(h) counterterm.  The counterterm needs the single-site
        # v- and x-derivatives of the basis; interacting (pair) bases keep the
        # uncorrected objects on the fit's kinematics.
        fit_mode = getattr(self, "__force_M_mode__", "symmetric")
        single_site = getattr(self.force_basis, "pdepth", 0) == 0
        if not single_site:
            logger.warning(
                "compute_entropy_production: the finite-dt counterterm is not available "
                "for interacting bases."
            )
        mode = "symmetric" if single_site else fit_mode
        h = dt_val if single_site else None
        # The coefficient-bias diagnostic models clean kinematics: skipped when
        # the force fit declared measurement noise (``noisy`` correction).
        diagnose = single_site and getattr(self, "__force_diffusion_method__", None) != "noisy"

        # All moment objects, the block sums and tau_N over one row set (see
        # ``shared_rows``): the per-row plug-in program reads every operand.
        frames = required_frames(
            self._entropy_scalar_program(self.force_coefficients_full, mode=mode, dt_correction=h)
        )
        with self._structural_scope(self.force_basis), shared_rows(frames):
            phi_minus, G_pm, G_mm = self._entropy_parity_moments(mode=mode)
            if single_site:
                K_dt, l_dt = (
                    integrate(self.data, prog, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
                    for prog in self._entropy_dt_programs(h)
                )
            if diagnose:
                G_c, G_om, M_c = self._entropy_force_system(h)

        C = coefficients[S] if coefficients is not None else self.force_coefficients_full[S]
        DeltaS_phi = 2.0 * float(C @ phi_minus[S])
        DeltaS_G = 2.0 * float(C @ (G_pm[jnp.ix_(S, S)] @ C))
        DeltaS_dt = (
            float(C @ (K_dt[jnp.ix_(S, S)] @ C) + l_dt[S] @ C) if single_site else 0.0
        )
        DeltaS = DeltaS_phi - DeltaS_G + DeltaS_dt
        Q_odd = 2.0 * float(C @ (G_mm[jnp.ix_(S, S)] @ C))

        # First-order coefficient bias: the residual of the kinematically
        # consistent normal equations at C, dC = G_c^-1 (G_c C + G_om C - M_c),
        # propagated through the gradient of DeltaS.  Reported, not subtracted.
        DeltaS_coef_bias = None
        if diagnose:
            SS = jnp.ix_(S, S)
            dC = stable_pinv(G_c[SS]) @ (G_c[SS] @ C + G_om[SS] @ C - M_c[S])
            grad = (2.0 * phi_minus[S] - 2.0 * (G_pm[SS] + G_pm[SS].T) @ C
                    + (K_dt[SS] + K_dt[SS].T) @ C + l_dt[S])
            DeltaS_coef_bias = float(grad @ dC)

        # Odd-sector degrees of freedom: numerical rank of the PSD odd Gram.
        Nminus = self._psd_rank(G_mm[jnp.ix_(S, S)])
        tauN = self._entropy_row_measure

        # Error bar: the analytic (martingale) scale sqrt(2 Q_odd + (2 N-)^2)
        # misses the slowly-correlated current fluctuations of the F+·D̄⁻¹·F−
        # term; the empirical block variance includes them.  Report the max.
        err_analytic = float((2.0 * max(Q_odd, 0.0) + (2.0 * Nminus) ** 2) ** 0.5)
        C_full = jnp.zeros_like(self.force_coefficients_full).at[S].set(C)
        with self._structural_scope(self.force_basis), shared_rows(frames):
            err_block = self._entropy_block_error(C_full, mode=mode, dt_correction=h)

        self.DeltaS = DeltaS
        self.error_DeltaS = max(err_analytic, err_block or 0.0)
        self.DeltaS_debiased = DeltaS - 2.0 * Nminus
        self.Sdot = DeltaS / tauN
        self.Sdot_bias = 2.0 * Nminus / tauN
        self.Sdot_debiased = self.DeltaS_debiased / tauN
        self.Sdot_error = self.error_DeltaS / tauN
        self.entropy_odd_ratio = DeltaS / max(Q_odd, 1e-300)
        self.metadata["entropy_production"] = {
            "support": support,
            "Nminus": Nminus,
            "tauN": tauN,
            "DeltaS_phi": DeltaS_phi,
            "DeltaS_G": DeltaS_G,
            "DeltaS_dt": DeltaS_dt,
            "DeltaS_coef_bias": DeltaS_coef_bias,
            "Q_odd": Q_odd,
            "error_analytic": err_analytic,
            "error_block": err_block,
            "dt_corrected": single_site,
        }
        logger.info(
            "Entropy production (UD): DeltaS = %.4g +- %.4g "
            "(rate %.4g +- %.4g, bias %.4g, odd ratio %.3g)",
            DeltaS, self.error_DeltaS, self.Sdot, self.Sdot_error,
            self.Sdot_bias, self.entropy_odd_ratio,
        )
        return {
            "Sdot": self.Sdot,
            "Sdot_error": self.Sdot_error,
            "Sdot_bias": self.Sdot_bias,
            "Sdot_debiased": self.Sdot_debiased,
            "DeltaS": self.DeltaS,
            "error_DeltaS": self.error_DeltaS,
            "DeltaS_debiased": self.DeltaS_debiased,
            "tauN": tauN,
            "Nminus": Nminus,
            "DeltaS_phi": DeltaS_phi,
            "DeltaS_G": DeltaS_G,
            "DeltaS_dt": DeltaS_dt,
            "Q_odd": Q_odd,
            "entropy_odd_ratio": self.entropy_odd_ratio,
            "Sdot_dt_bias": None if DeltaS_coef_bias is None else DeltaS_coef_bias / tauN,
        }

    def time_reversal_split(self):
        r"""Split the fitted acceleration field by time-reversal parity.

        Returns
        -------
        (F_even, F_odd) : pair of callables ``f(x, v) -> array``
            :math:`\hat F^{\pm}(x,v) = \tfrac12[\hat F(x,v) \pm \hat
            F(x,-v)]` — the reversible (conservative + confining) and
            irreversible (friction + velocity-odd driving) parts of the
            dynamics.  Both close over :attr:`force_inferred` and are
            jit-safe.
        """
        if not hasattr(self, "force_inferred"):
            raise RuntimeError("time_reversal_split() requires a fitted force.")
        F = self.force_inferred

        def F_even(x, v):
            return 0.5 * (F(x, v=v) + F(x, v=-v))

        def F_odd(x, v):
            return 0.5 * (F(x, v=v) - F(x, v=-v))

        return F_even, F_odd

    def simulate_bootstrapped_trajectory(self, key, oversampling: int = 1, simulate: bool = True, dataset: int = 0):
        """
        Simulate an underdamped Langevin trajectory using the inferred force and diffusion.

        Args:
            key: JAX random key.
            oversampling: number of integration substeps between saved frames.
            simulate: if True, runs a simulation initialized from the first row of data;
                      if False, returns an uninitialized process object.
            dataset: which experiment of a pooled fit to reproduce. The inferred model is
                collapsed to this condition via :meth:`~SFI.statefunc.StateExpr.specialize`
                (per-dataset parameters folded at ``dataset``); the resulting process is a
                standalone single-condition model that does not read ``dataset_index``.
                Defaults to 0.

        Returns:
            If simulate=True: (TrajectoryCollection, UnderdampedProcess)
            Else: UnderdampedProcess
        """
        self._check_noise_consistency()
        from SFI.langevin import UnderdampedProcess

        if not hasattr(self, "force_inferred"):
            raise RuntimeError("Force not inferred. Call infer_force_linear() (or nonlinear later) first.")

        # Collapse a (possibly pooled) fit to the chosen experiment: per-dataset
        # parameters are folded at ``dataset`` and the reserved ``dataset_index``
        # disappears, so the simulated model is standalone (no dataset concept).
        force_k = self.force_inferred.specialize(dataset=int(dataset))
        diff_k = self.diffusion_inferred.specialize(dataset=int(dataset))
        self._check_diffusion_psd(diff_k, dataset=int(dataset))
        bootstrapped_process = UnderdampedProcess(force_k._psf, diff_k._psf)
        bootstrapped_process.set_params(theta_F=force_k.params, theta_D=diff_k.params)

        if not simulate:
            # Just return the process, leave extras and state uninitialized
            return bootstrapped_process

        ds = self.data.datasets[int(dataset)]
        bootstrapped_process.set_extras(extras_global=ds.extras_global, extras_local=ds.extras_local)
        # Only the particle count is framework-supplied; the specialized
        # force does not read ``dataset_index``.
        bootstrapped_process._reserved_overrides = {
            "particle_index": jnp.arange(int(ds.N)),
        }
        # Initialize from first available row; velocity from dX/dt as requested.
        from SFI.trajectory import TrajectoryCollection

        start_config = TrajectoryCollection.from_dataset(ds).peek_row(require={"X", "dX", "dt"})
        x0 = start_config["X"]
        v0 = start_config["dX"] / start_config["dt"]

        bootstrapped_process.initialize(x0, v0=v0)

        data_bootstrap = bootstrapped_process.simulate(
            dt=start_config["dt"],
            Nsteps=ds.T,
            key=key,
            prerun=0,
            oversampling=oversampling,
        )
        return data_bootstrap, bootstrapped_process

    # ------------------------------------------------------------------ #
    # Parametric — underdamped exact-core inference
    # ------------------------------------------------------------------ #

    def infer_force(
        self,
        F,
        theta0=None,
        *,
        D=None,
        Lambda=None,
        integrator: str = "rk4",
        n_substeps: int = 1,
        inner: str = "auto",
        eiv="auto",
        max_outer: int = 5,
        inner_maxiter: int = 80,
    ) -> None:
        r"""Infer force F(x,v) with the exact parametric estimator.

        Built on ``SFI.inference.parametric_core``: the unobserved velocity
        is resolved by shooting through an RK4 phase-space step, the
        pentadiagonal (bandwidth-2) 3-point residual covariance is whitened
        *exactly* by a reverse-time block-LDLᵀ innovations recursion, and
        the parameters are found by direct Gauss–Newton (linear-in-θ
        ``Basis``, with the skip-trick errors-in-variables instrument; the
        nonlinear-in-θ ``PSF`` path uses damped Gauss–Newton).  ``(D, Λ)``
        are profiled natively: moment-estimator init, then an exact
        banded-NLL refinement at the fitted θ on cached fixed-θ tensors.
        Non-uniform sampling (``t`` vector or per-step ``dt``) is fitted
        directly with per-interval shooting steps and Lyapunov-exact
        process covariances.

        .. physics:: Parametric force inference (underdamped)
           :label: parametric-force-underdamped
           :category: Inference

           The phase-space dynamics
           :math:`\dot{x}=v,\;\mathrm{d}v = F(x,v)\mathrm{d}t
           + \sqrt{2D}\,\mathrm{d}W` are factored into deterministic
           flow + noise.  Three-point shooting residuals follow a
           pentadiagonal Gaussian that is whitened exactly
           (reverse-time innovations) before entering the
           Gauss–Newton normal equations; the process noise enters at
           :math:`\Delta t^3`.  Under measurement noise the left factor
           is the η-clean *skip* instrument built from the lagged clean
           position pair (``eiv=True``) — consistent where the naive
           MLE is velocity-EIV-biased.

        Parameters
        ----------
        F : PSF (``needs_v=True``) or Basis
            Parametric force model.  A ``Basis`` is converted to a PSF
            internally (coefficients initialised to zero) and PASTIS
            sparsification is enabled; it runs the fast direct-GN path.
        theta0 : dict, array, or None
            Initial force parameters (default: zeros).
        D : array (d, d), optional
            Fixed diffusion matrix.  If both ``D`` and ``Lambda``
            are given, noise profiling is skipped entirely (fast path).
        Lambda : array ``(d, d)``, optional
            Fixed measurement-noise covariance.
        integrator : {"rk4", "euler"}
            Flow predictor (default ``"rk4"``, a single 4th-order step).
            A single ``"euler"`` step cannot carry the force into the
            position update, so the skip-trick is auto-disabled (with a
            warning) for ``integrator="euler", n_substeps=1``.
        n_substeps : int
            Integrator micro-steps per observation interval (default 1 —
            the single-step minimal estimator).
        inner : {"auto", "gn", "lbfgs"}
            Inner solver.  ``"auto"`` → Gauss–Newton (direct for a
            linear ``Basis``, damped for a nonlinear ``PSF``).
            ``"lbfgs"`` is unavailable on the underdamped exact core
            (the frozen-precision quadratic is unbounded under exact
            whitening) and redirects to Gauss–Newton with a warning.
        eiv : {"auto", True, False, float}
            Measurement-noise errors-in-variables instrument.  ``"auto"``
            (default) → ``True`` for all models (interacting models use
            the same N-body flow for the instrument as for the residual);
            ``True`` forces the η-clean skip instrument; ``False`` is the
            plain MLE; a float in ``[0, 1]`` blends.  ``"auto"`` does
            *not* inspect the noise level: for (near-)clean data at
            coarse effective sampling pass ``eiv=False`` (see
            :ref:`parametric-algorithm`).
        max_outer : int
            Outer Gauss–Newton / IRLS iterations (default 5).
        inner_maxiter : int
            Inner L-BFGS iterations per outer step on the PSF path
            (default 80; raise for large nonlinear families, e.g. NNs).

        Updates
        -------
        Sets standard ``force_*`` attributes: ``force_psf``,
        ``force_G``, ``force_G_pinv``, ``force_coefficients_full``,
        ``force_coefficients``, ``force_support``, ``force_moments``,
        ``force_inferred``, ``diffusion_average``, ``A``, ``A_inv``,
        ``Lambda``.

        When ``F`` is a ``Basis``, additionally sets
        ``force_basis``, ``force_G_full``, and ``force_scorer``
        so that ``sparsify_force()`` can be called afterwards.

        See Also
        --------
        :ref:`parametric-concept` : Mathematical foundations.
        :ref:`parametric-algorithm` : Detailed algorithm description.
        """
        import time as _time

        from SFI.inference.parametric_core.solve import _as_psf, solve_force_ud
        from SFI.statefunc import SF
        from SFI.statefunc.basis import Basis

        from .result import InferenceResultSF

        if hasattr(self, "force_inferred"):
            raise RuntimeError("Force inference has already been run on this object.")

        if integrator not in ("rk4", "euler"):
            raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")

        # ── Accept Basis → convert to PSF, enable PASTIS ──
        basis_mode = isinstance(F, Basis)
        if basis_mode:
            self.force_basis = F
            self.force_basis_labels = getattr(F, "labels", None)
        elif not (hasattr(F, "flatten_params") and hasattr(F, "unflatten_params")):
            raise TypeError("F must be a PSF or a Basis.  Got %s." % type(F).__name__)
        F_psf = _as_psf(F)

        # Structural arrays (interaction lists, etc.) are exposed only for this
        # parametric solve and never persisted (see ``_structural_scope``).
        with self._structural_scope(F_psf):
            import numpy as _np

            from SFI.inference.parametric_core.solve import _dt_steps

            # representative Δt for logging and the reported β — the median
            # step (identical to the single step at uniform sampling)
            dt_val = float(_np.median(_np.concatenate(
                [_dt_steps(ds) for ds in self.data.datasets])))

            logger.info(
                "[uli_parametric] UD minimal parametric solve (n_params=%d, dt=%.4g, %s·n%d, basis_mode=%s)...",
                int(F_psf.template.size), dt_val, integrator, n_substeps, basis_mode,
            )
            t0 = _time.perf_counter()

            res = solve_force_ud(
                self.data, F, theta0=theta0, D=D, Lambda=Lambda,
                integrator=integrator, n_substeps=n_substeps,
                inner=inner, eiv=eiv, max_outer=max_outer,
                inner_maxiter=inner_maxiter,
                chunk_target_bytes=self._chunk_target_bytes,
            )
            t_elapsed = _time.perf_counter() - t0

        # ── Standard force attributes ──
        theta_flat = res.theta
        G = res.G
        self._warn_if_null_parametric_force(theta_flat, res.info, label="infer_force")

        self.force_psf = F_psf
        self.diffusion_average = res.D
        # Callable constant-D field, so the full result surface (incl.
        # simulate_bootstrapped_trajectory) works without a separate
        # compute_diffusion_constant() call; a later infer_diffusion()
        # overwrites it with the state-dependent fit.
        self.diffusion_inferred = constant_array(res.D)
        self.A = 2 * res.D
        self.A_inv = jnp.linalg.inv(self.A)
        self.Lambda = res.Lambda
        self.Lambda_trace = float(jnp.trace(res.Lambda))

        self.force_G = G
        # Parameter covariance from the solver: the sandwich G⁻¹HG⁻ᵀ on the
        # IV (eiv) path, the inverse information on the symmetric path.
        G_inv = None
        if bool(jnp.all(jnp.isfinite(res.theta_cov))):
            G_inv = res.theta_cov
        self.force_G_pinv = G_inv

        self.force_coefficients_full = theta_flat
        self.force_coefficients = theta_flat
        self.force_support = jnp.arange(theta_flat.size)
        self.force_moments = G @ theta_flat

        # ── PASTIS plumbing (Basis mode only) ──
        if basis_mode:
            from SFI.inference.parametric_core.solve import force_selection_metric
            from SFI.inference.sparse import SparseScorer

            self.force_G_full = G
            # Select in the inverse-covariance metric (the GN Gram itself on
            # the symmetric path), so a kept support's coefficients and
            # covariance are those of the efficient restricted estimator.
            self.force_fisher_full = force_selection_metric(res)
            F_sel = G if self.force_fisher_full is None else self.force_fisher_full
            self.force_scorer = SparseScorer(M=F_sel @ theta_flat, G=F_sel)

        sf_F = SF(F_psf, F_psf.unflatten_params(theta_flat))
        beta = float(jnp.trace(res.Lambda) / (jnp.trace(res.D) * dt_val + 1e-30))

        meta_F = dict(
            kind="force",
            inference="parametric",
            n_params=int(theta_flat.size),
            beta=beta,
            A_inv=jnp.asarray(self.A_inv),
        )
        self.force_inferred = InferenceResultSF(
            sf_F,
            param_cov=G_inv,
            meta=meta_F,
        )

        self.metadata["force_method"] = "parametric"
        self.metadata["force_parametric_info"] = {
            **res.info,
            "integrator": integrator,
            "beta": beta,
            "D_matrix": res.D,
            "Lambda": res.Lambda,
        }

        logger.info(
            "[uli_parametric] Done in %.1f s.  Λ=%s, D=%s, β=%.3f",
            t_elapsed,
            jnp.diag(res.Lambda),
            jnp.diag(res.D),
            beta,
        )

    # ── State-dependent diffusion regression ──

    def infer_diffusion(
        self,
        basis=None,
        *,
        theta_D0=None,
        integrator: str = "rk4",
        n_substeps: int = 1,
        maxiter: int = 100,
        _support=None,
    ) -> None:
        r"""Infer state- and velocity-dependent diffusion D(x, v).

        Requires a prior parametric :meth:`infer_force` call.  Holds the
        fitted force fixed and minimises the exact banded NLL over the
        diffusion parameters on cached fixed-θ tensors (zero force or
        basis evaluations per L-BFGS iteration), with ``D(x, v; θ_D)``
        evaluated at the shooting velocity.

        .. physics:: State-dependent diffusion inference (underdamped)
           :label: parametric-diffusion-underdamped
           :category: Inference

           With the force :math:`\hat F(x,v)` held fixed, the
           state-dependent diffusion :math:`D(x,v;\theta_D)` is optimised
           by minimising the exact banded NLL of the pentadiagonal
           (bandwidth-2) residual covariance; :math:`\Lambda` from the
           force inference is held fixed.

        Parameters
        ----------
        basis : Basis (rank 2), PSF, or None
            Diffusion model.  A rank-2 ``Basis`` gives the linear
            parameterisation ``D = Σ_j θ_j d_j``; a ``PSF`` (optionally
            ``needs_v=True``) is used directly.  When ``None`` (default),
            ``symmetric_matrix_basis(d)`` — all constant symmetric
            ``d×d`` diffusion matrices.
        theta_D0 : dict, array, or None
            Initial diffusion parameters (default zeros).
        integrator : {"rk4", "euler"}
            Flow predictor (default ``"rk4"``, matching :meth:`infer_force`).
        n_substeps : int
            Integrator micro-steps per Δt (default 1).
        maxiter : int
            L-BFGS maximum iterations (default 100).

        Updates
        -------
        Sets ``diffusion_inferred``, ``diffusion_coefficients`` (and
        ``diffusion_basis`` when ``basis`` is a ``Basis``), plus metadata.
        ``diffusion_inferred.param_cov`` is the inverse observed Fisher
        information (inverse NLL Hessian at :math:`\hat\theta_D`, force and
        :math:`\Lambda` held fixed); :meth:`compute_diffusion_error` gives the
        standard errors, :meth:`sparsify_diffusion` selects terms.

        See Also
        --------
        :ref:`parametric-algorithm` : Full algorithm description.
        """
        from SFI.bases import symmetric_matrix_basis
        from SFI.inference.parametric_core.solve import _as_psf, solve_diffusion_ud

        if not hasattr(self, "force_inferred"):
            raise RuntimeError("infer_diffusion() requires a prior infer_force() call.")
        if self.metadata.get("force_method") not in ("parametric", "parametric_core"):
            raise RuntimeError(
                "infer_diffusion() requires parametric force inference (call infer_force(), not infer_force_linear)."
            )

        from SFI.statefunc.basis import Basis

        d = self.data.datasets[0].X.shape[-1]
        if basis is None:
            basis = symmetric_matrix_basis(d)
        if isinstance(basis, Basis):
            self._validate_basis(basis, expected_rank=2, label="diffusion basis")
            self.diffusion_basis = basis
        D_psf = _as_psf(basis)
        Lambda = getattr(self, "Lambda", None)
        if Lambda is None:
            Lambda = jnp.zeros((d, d))

        res = solve_diffusion_ud(
            self.data, self.force_psf, self.force_coefficients_full, D_psf,
            Lambda=Lambda, theta_D0=theta_D0,
            n_substeps=n_substeps, integrator=integrator, maxiter=maxiter,
            chunk_target_bytes=self._chunk_target_bytes, support=_support,
        )
        spec = dict(basis=basis, integrator=integrator, n_substeps=n_substeps, maxiter=maxiter)
        self._store_parametric_diffusion(res, D_psf, spec, support=_support)

    def _force_G_matrix(self, *, program_only: bool = False, return_measure: bool = False) -> jnp.ndarray:
        if not hasattr(self, "A_inv"):
            raise RuntimeError("A_inv not available. Compute diffusion first.")
        b_left = self.force_basis
        b_right = self.force_basis @ self.A_inv
        return self.__G_matrix__(b_left, b_right, self.__force_G_mode__, "ima,imb->iab",
                                 program_only=program_only, return_measure=return_measure)

    def _force_linear_frames(self) -> frozenset:
        """Frame offsets read by the linear force fit (Gram + moments + correction)."""
        progs = [self._force_G_matrix(program_only=True)]
        moments = self._force_moments(program_only=True)
        progs += list(moments) if isinstance(moments, (list, tuple)) else [moments]
        return required_frames(*progs)

    def _force_moments(self, *, mode=None, A_inv=None, D_op=None, program_only: bool = False) -> jnp.ndarray:
        r"""
        ULI force moments expressed via Integrand.

        With reconstructed kinematics (x̂, v̂, â):

            M_a = < â · A_inv · b_a(x̂, v̂) >  +  w * < - D_inst : (A_inv · ∂_v b_a) >

        where:
            - b_a is the vector basis (i, dim, a)
            - ∂_v b_a is the velocity-gradient basis (i, dim, dim, a)
            - w = (1 + 2 l)/3 with l=1 (symmetric), l=0 (early), l=-1/2 (anticipated)
              and the anticipated mode uses w=0 (no gradient correction).

        .. physics:: Underdamped force moments (ULI)
           :label: force-moments-underdamped
           :category: Inference

           .. math::

              M_a = \bigl\langle \hat a_t \cdot A^{-1} \cdot b_a(\hat x_t, \hat v_t)
              \bigr\rangle
              \;-\; w \,\bigl\langle (D_{\text{inst}}\,A^{-1})_{kl}\,
              \partial_{v_k} (b_a)_l \bigr\rangle

           where :math:`k` is the velocity-derivative index and :math:`l` the
           output (force) index of :math:`\partial_v b_a`.  The matrix acting
           on the Jacobian is :math:`D_{\text{inst}}\,A^{-1}` in this order:
           contracting :math:`D_{\text{inst}}` with the output index instead
           would give :math:`A^{-1} D_{\text{inst}}`, which differs whenever
           :math:`[A^{-1}, D_{\text{inst}}] \neq 0` (this is the overdamped
           gradient-correction convention, mirrored here).  The weight is
           :math:`w = (1+2\ell)/3`, with :math:`\ell=1` (symmetric),
           :math:`\ell=0` (early), :math:`\ell=-\tfrac{1}{2}` (anticipated).

        Contractions
        ------------
        - phi term:
            eq='im,mn,ina->ia' with ops (â, A_inv, b)

        - correction term:
            eq='imn,no,imoa->ia' with ops (D_inst, A_inv, ∂_v b)
            (∂_v b has axes (i, m, o, a): m is the velocity-DERIVATIVE index
            — contracted with D_inst — and o is the OUTPUT/force index —
            contracted with A_inv; a is the feature.  This yields
            (D_inst · A_inv) : ∂_v b.  Per stateexpr.py, d_v inserts the
            derivative axis immediately before the rank block, so it is the
            second axis, not the third.)
        """
        default_call = mode is None and A_inv is None and D_op is None and not program_only
        if A_inv is None:
            if not hasattr(self, "A_inv"):
                raise RuntimeError("A_inv not available. Compute diffusion first.")
            A_inv = self.A_inv
        A_inv = jnp.asarray(A_inv)
        if mode is None:
            mode = getattr(self, "__force_M_mode__", "symmetric")
        if mode not in ("symmetric", "early", "anticipated"):
            raise KeyError(f"Unknown __force_M_mode__: {mode}")

        if program_only:
            progs = [self._uli_v_moments(A_inv=A_inv, mode=mode, program_only=True)]
            if mode != "anticipated":
                progs.append(self._uli_grad_moments(A_inv=A_inv, D_op=D_op, mode=mode, program_only=True))
            return progs

        phi_moments = self._uli_v_moments(A_inv=A_inv, mode=mode)
        if default_call:
            self._phi_moments = phi_moments
        # Determine w prefactor
        if mode == "symmetric":
            lam = 1.0
        elif mode == "early":
            lam = 0.0
        else:  # anticipated
            lam = -0.5
        w_pref = (1.0 + 2.0 * lam) / 3.0

        # In anticipated mode, w_pref==0 and we skip the correction term.
        if w_pref == 0.0:
            return phi_moments

        logger.debug("Computing ULI correction term (D_inst × grad_v basis).")
        w_moments = self._uli_grad_moments(A_inv=A_inv, D_op=D_op, mode=mode, scale=-float(w_pref))
        if default_call:
            self._w_moments = w_moments

        return phi_moments + w_moments

    def _uli_v_moments(self, *, A_inv=None, mode=None, program_only: bool = False) -> jnp.ndarray:
        r"""Acceleration projection :math:`\sum_t \hat a_t \cdot A^{-1} b_a(\hat x_t, \hat v_t)`.

        Extensive, dt-weighted, mask-aware sum on the kinematics of ``mode``
        (default: the fit's ``__force_M_mode__``).  ``A_inv`` defaults to
        ``self.A_inv``; with the identity this is the plain projection
        :math:`\langle b_a \cdot \Delta\hat v_t\rangle/\Delta t` — on the
        symmetric stencil the mid-point (Stratonovich) projection in
        velocity space.
        """
        if A_inv is None:
            A_inv = self.A_inv
        if mode is None:
            mode = getattr(self, "__force_M_mode__", "symmetric")
        x_hat = self._get_x_hat(M_mode=mode)
        v_hat = self._get_v_hat(M_mode=mode)
        a_hat = self._get_a_hat(M_mode=mode)
        A = ConstOperand(jnp.asarray(A_inv), alias="A")
        a_op = TimeOperand(a_hat, alias="a")
        B = ExprOperand(expr=self.force_basis, x=x_hat, v=v_hat, alias="B")
        prog_phi = Integrand(
            exprs=[B],
            times=[a_op],
            consts=[A],
            terms=[Term(eq="im,mn,ina->ia", ops=("a", "A", "B"))],
        )
        if program_only:
            return prog_phi
        return integrate(self.data, prog_phi, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)

    def _uli_grad_moments(
        self, *, A_inv=None, D_op=None, mode=None, scale: float = 1.0, program_only: bool = False
    ) -> jnp.ndarray:
        r"""Velocity-gradient term :math:`\mathrm{scale}\sum_t (D_{\rm inst}(t) A^{-1})_{kl}\,\partial_{v_k}(b_a)_l`.

        The Itô-conversion term of the ULI moments without its
        :math:`-w` prefactor (passed as ``scale`` by :meth:`_force_moments`).
        ``D_op`` defaults to the fit's diffusion stencil; ``A_inv`` to
        ``self.A_inv``.  With the identity and ``scale=1`` this is
        :math:`\langle \partial_v b_a : D_t\rangle`, the quadratic-variation
        estimate of the Itô–Stratonovich difference in velocity space.
        """
        if A_inv is None:
            A_inv = self.A_inv
        if mode is None:
            mode = getattr(self, "__force_M_mode__", "symmetric")
        x_hat = self._get_x_hat(M_mode=mode)
        v_hat = self._get_v_hat(M_mode=mode)
        A = ConstOperand(jnp.asarray(A_inv), alias="A")
        method = getattr(self, "__force_diffusion_method__", "noisy")
        _interacting = getattr(self.force_basis, "pdepth", 0) >= 1
        all_pairs = _interacting and getattr(self, "__force_cross_particle__", "all-pairs") == "all-pairs"
        # The force fit's own correction (custom D_op: same-particle blocks below).
        # Its metadata and warning are recorded on evaluation only: program_only
        # (row-scope planning) has no side effects.
        own = D_op is None
        if own:
            if all_pairs:
                # Every particle pair: sum_j D_t,ij : A^-1 d b_i / d v_j, the same instantaneous
                # stencil on i != j, as one forward-mode derivative of the basis along each
                # increment direction of the stencil (no Jacobian block, no diffusion model).
                from SFI.inference.graph_diffusion import stencil_correction_program, stencil_directions

                prog_w = stencil_correction_program(self.force_basis, A_inv, stencil_directions(method, "ud"),
                                                    x=x_hat, v=v_hat, wrt="v", scale=float(scale))
                if program_only:
                    return prog_w
                self.metadata["force_noise_correction"] = "all-pairs"
                return integrate(self.data, prog_w, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
            D_op = self.get_diffusion_timeop(method)
        # For pair-dispatched bases (pdepth>=1), the Ito correction needs
        # the same-particle velocity Jacobian  ∂f_i/∂v_i  (not cross ∂f_i/∂v_j).
        _same = getattr(self.force_basis, "pdepth", 0) >= 1
        if _same and not program_only:
            self._check_same_site_correction(self.force_basis, label="ULI velocity correction")
        Gv = ExprOperand(
            expr=self.force_basis.d_v(same_particle=_same),
            x=x_hat,
            v=v_hat,
            alias="Gv",
        )

        # Contraction correctness relies on D_inst being SYMMETRIC: D_inst (m,n)
        # pairs with the velocity-derivative axis (m) of Gv and A_inv (n,o) with
        # the output axis (o), giving (D_inst . A_inv) : grad_v b.  Correct only
        # because D ("noisy") is symmetric; a non-symmetric instantaneous-D
        # estimator would need the D/A_inv index roles swapped (they differ
        # whenever [A_inv, D] != 0).  Mirrors overdamped._force_moments.
        prog_w = Integrand(
            exprs=[Gv],
            times=[D_op],
            consts=[A],
            terms=[
                Term(
                    eq="imn,no,imoa->ia",
                    ops=(D_op.alias, "A", "Gv"),
                    scale=float(scale),
                )
            ],
        )
        if program_only:
            return prog_w
        if own:
            self.metadata["force_noise_correction"] = "same-particle"
        return integrate(self.data, prog_w, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)

    def _entropy_parity_moments(self, mode=None):
        r"""Parity-resolved moment objects for :meth:`compute_entropy_production`.

        Builds, at the kinematic evaluation points :math:`(\hat x, \hat v)`
        of ``mode`` (default: the fit's) and with the v-parity flip realised
        by negating the velocity operand (``scale(v_hat, -1)``):

        - ``phi_minus[a]`` = :math:`\sum \mathrm{d}t\,\hat a\cdot A^{-1}
          b^-_a` (extensive; :math:`\equiv \sum \mathrm{d}\hat v \cdot
          A^{-1} b^-_a` since :math:`\hat a\,\mathrm{d}t = \mathrm{d}\hat
          v`),
        - ``G_pm[a,b]`` = :math:`\sum \mathrm{d}t\, b^+_a \cdot A^{-1}
          b^-_b`,
        - ``G_mm[a,b]`` = :math:`\sum \mathrm{d}t\, b^-_a \cdot A^{-1}
          b^-_b` (PSD; its rank counts odd-sector degrees of freedom).

        All three share the same evaluation measure by construction — do
        not mix with ``__G_matrix__`` outputs (different kinematics, and
        those are returned transposed).
        """
        if mode is None:
            mode = getattr(self, "__force_M_mode__", "symmetric")
        x_hat = self._get_x_hat(M_mode=mode)
        v_hat = self._get_v_hat(M_mode=mode)
        a_hat = self._get_a_hat(M_mode=mode)

        A = ConstOperand(self.A_inv, alias="A")
        a_op = TimeOperand(a_hat, alias="a")
        Bp = ExprOperand(expr=self.force_basis, x=x_hat, v=v_hat, alias="Bp")
        Bm = ExprOperand(expr=self.force_basis, x=x_hat, v=scale(v_hat, -1.0), alias="Bm")

        # phi^- : two-term (+1/2, -1/2) combination realising b^-.
        prog_phi = Integrand(
            exprs=[Bp, Bm],
            times=[a_op],
            consts=[A],
            terms=[
                Term(eq="im,mn,ina->ia", ops=("a", "A", "Bp"), scale=0.5),
                Term(eq="im,mn,ina->ia", ops=("a", "A", "Bm"), scale=-0.5),
            ],
        )
        # Its row measure (particle-time) is the estimator's tau_N.
        phi_minus, self._entropy_row_measure = integrate(
            self.data, prog_phi, reduce="sum", chunk_target_bytes=self._chunk_target_bytes,
            return_measure=True,
        )

        def _parity_gram(s_pp, s_pm, s_mp, s_mm):
            prog = Integrand(
                exprs=[Bp, Bm],
                consts=[A],
                terms=[
                    Term(eq="ima,mn,inb->iab", ops=("Bp", "A", "Bp"), scale=s_pp),
                    Term(eq="ima,mn,inb->iab", ops=("Bp", "A", "Bm"), scale=s_pm),
                    Term(eq="ima,mn,inb->iab", ops=("Bm", "A", "Bp"), scale=s_mp),
                    Term(eq="ima,mn,inb->iab", ops=("Bm", "A", "Bm"), scale=s_mm),
                ],
            )
            return integrate(
                self.data, prog, reduce="sum", chunk_target_bytes=self._chunk_target_bytes
            )

        # b^+ A^-1 b^- = 1/4 [BpABp - BpABm + BmABp - BmABm]
        G_pm = _parity_gram(0.25, -0.25, 0.25, -0.25)
        # b^- A^-1 b^- = 1/4 [BpABp - BpABm - BmABp + BmABm]
        G_mm = _parity_gram(0.25, -0.25, -0.25, 0.25)

        self._phi_minus, self._G_pm, self._G_mm = phi_minus, G_pm, G_mm
        return phi_minus, G_pm, G_mm

    def _entropy_scalar_program(self, C, *, mode=None, dt_correction=None) -> Integrand:
        r"""Per-row scalar integrand of the entropy plug-in functional.

        Row value (before the engine's dt-weighting):

            s_t = 2 [ â·A⁻¹ F̂⁻(x̂,v̂) ]  −  2 [ F̂⁺·A⁻¹·F̂⁻ ](x̂,v̂)

        with F̂(±v̂) = Σ_a C_a b_a(x̂, ±v̂), so that
        ``integrate(..., reduce="sum")`` over any time window returns that
        window's ΔŜ contribution.  Used for the block-variance error bar.
        With ``dt_correction=h`` the rows also carry the O(h) kinematic
        counterterms of :meth:`_entropy_dt_programs` (symmetric kinematics).
        """
        if mode is None:
            mode = getattr(self, "__force_M_mode__", "symmetric")
        if dt_correction is not None:
            return self._entropy_scalar_program(C, mode=mode) + self._entropy_dt_programs(
                dt_correction, C=C
            )
        x_hat = self._get_x_hat(M_mode=mode)
        v_hat = self._get_v_hat(M_mode=mode)
        a_hat = self._get_a_hat(M_mode=mode)

        A = ConstOperand(self.A_inv, alias="A")
        Cc = ConstOperand(jnp.asarray(C), alias="C")
        a_op = TimeOperand(a_hat, alias="a")
        Bp = ExprOperand(expr=self.force_basis, x=x_hat, v=v_hat, alias="Bp")
        Bm = ExprOperand(expr=self.force_basis, x=x_hat, v=scale(v_hat, -1.0), alias="Bm")

        # 2 Cᵀφ⁻ rows: +1·(a A Bp C) − 1·(a A Bm C)
        # −2 F̂⁺A⁻¹F̂⁻ rows: −0.5·[pp − pm + mp − mm]
        return Integrand(
            exprs=[Bp, Bm],
            times=[a_op],
            consts=[A, Cc],
            terms=[
                Term(eq="im,mn,ina,a->i", ops=("a", "A", "Bp", "C"), scale=1.0),
                Term(eq="im,mn,ina,a->i", ops=("a", "A", "Bm", "C"), scale=-1.0),
                Term(eq="ima,mn,inb,a,b->i", ops=("Bp", "A", "Bp", "C", "C"), scale=-0.5),
                Term(eq="ima,mn,inb,a,b->i", ops=("Bp", "A", "Bm", "C", "C"), scale=0.5),
                Term(eq="ima,mn,inb,a,b->i", ops=("Bm", "A", "Bp", "C", "C"), scale=-0.5),
                Term(eq="ima,mn,inb,a,b->i", ops=("Bm", "A", "Bm", "C", "C"), scale=0.5),
            ],
        )

    # Parity-resolved operand names of the O(h) entropy programs: name ->
    # (raw operand, s), the operand being 1/2 [raw(x̂, v̂) + s raw(x̂, -v̂)], or
    # raw(x̂, v̂) itself when s is None.  A v-derivative flips the parity of the
    # minus-v evaluation: (∂ᵥb)⁻ = 1/2 [∂ᵥb(v) + (∂ᵥb)(-v)].
    _ENTROPY_PARITY = {
        "b": ("B", None), "b+": ("B", 1.0), "b-": ("B", -1.0),
        "Dvb": ("V", None), "Dvb+": ("V", -1.0), "Dvb-": ("V", 1.0),
        "Dvvb": ("VV", None), "Dvvb+": ("VV", 1.0), "Dvvb-": ("VV", -1.0),
        "Dvvvb-": ("VVV", 1.0),
        "Dxb": ("X", None), "Dxb-": ("X", -1.0),
    }

    def _entropy_kinematic_integrand(self, terms, *, contract=None) -> Integrand:
        r"""Integrand of parity-resolved basis products on the symmetric ULI kinematics.

        ``terms`` is a list of ``(eq, ops, scale)`` with ``eq`` an einsum
        without output (row index ``i``, feature indices ``a``/``b``) and
        ``ops`` drawn from :attr:`_ENTROPY_PARITY` (basis ``b`` and its
        derivatives ``∂ᵥb``, ``∂ᵥ²b``, ``∂ᵥ³b``, ``∂ₓb`` with axes ``(i, k[, l, m], o,
        a)``), the kinematics ``"a"`` (:math:`\hat a`) and the constants
        ``"A"`` (:math:`A^{-1}`), ``"D"`` (:math:`\bar D`), ``"DA"``
        (:math:`\bar D A^{-1}`).  The output is ``"ia"`` / ``"iab"`` (by the
        feature indices present) or, with ``contract=C``, the per-row scalar
        with every feature index contracted with ``C``.
        """
        from itertools import product

        x_hat, v_hat = _X_sym_uli, _V_sym_uli
        b = self.force_basis
        raw = {"B": lambda: b, "V": b.d_v, "VV": lambda: b.d_v().d_v(),
               "VVV": lambda: b.d_v().d_v().d_v(), "X": b.d_x}
        rename = {"a": "ea", "A": "eA", "D": "eD", "DA": "eDA"}
        exprs, out_terms = {}, []
        for eq, ops, sc in terms:
            choices = []
            for op in ops:
                if op not in self._ENTROPY_PARITY:
                    choices.append([(rename[op], 1.0)])
                    continue
                base, s = self._ENTROPY_PARITY[op]
                for side in ("p",) if s is None else ("p", "m"):
                    alias = f"e{base}{side}"
                    if alias not in exprs:
                        v_op = v_hat if side == "p" else scale(v_hat, -1.0)
                        exprs[alias] = ExprOperand(expr=raw[base](), x=x_hat, v=v_op, alias=alias)
                choices.append([(f"e{base}p", 1.0)] if s is None
                               else [(f"e{base}p", 0.5), (f"e{base}m", 0.5 * s)])
            feats = "".join(c for c in "ab" if c in eq)
            if contract is not None:
                eq_full = eq + "".join("," + c for c in feats) + "->i"
                extra = ("eC",) * len(feats)
            else:
                eq_full, extra = eq + "->i" + feats, ()
            for combo in product(*choices):
                w = float(np.prod([c[1] for c in combo]))
                out_terms.append(Term(eq=eq_full, ops=tuple(c[0] for c in combo) + extra, scale=sc * w))
        D = jnp.asarray(self.diffusion_average)
        consts = [ConstOperand(self.A_inv, alias="eA"), ConstOperand(D, alias="eD"),
                  ConstOperand(D @ self.A_inv, alias="eDA")]
        if contract is not None:
            consts.append(ConstOperand(jnp.asarray(contract), alias="eC"))
        return Integrand(exprs=list(exprs.values()), times=[TimeOperand(_A_sym_uli, alias="ea")],
                         consts=consts, terms=out_terms)

    def _entropy_force_system(self, h):
        r"""Kinematically consistent linear force system, for the coefficient-bias diagnostic.

        The default ULI fit (trapezoid Gram on secant velocities, symmetric
        moments) carries an :math:`O(h)` coefficient bias that the entropy
        functional amplifies; the residual of this system at the fit's
        coefficients estimates it.  Gram and moments share the symmetric
        evaluation point :math:`(\hat x, \hat v)` and carry the derived
        :math:`O(h)` counterterms (row :math:`a` = test function, constant
        :math:`\bar D`, :math:`h = \Delta t`):

        .. math::

           G_{ab} = \bigl\langle b_a\!\cdot\! A^{-1} b_b
             + \tfrac{h}{3}\,\bar D_{kl}\bigl(2\,\partial_{v_k} b_a\!\cdot\!
               A^{-1}\partial_{v_l} b_b + b_a\!\cdot\! A^{-1}
               \partial_{v_k}\partial_{v_l} b_b\bigr)
             - \tfrac{7h}{12}\,\Omega^{\bar D}[b_b] : A^{-1}\partial_v b_a
             \bigr\rangle,

           M_a = \bigl\langle \hat a\cdot A^{-1} b_a
             - \bar D A^{-1} : \partial_v b_a
             - 2 M_x h\, \bar D A^{-1} : \partial_x b_a \bigr\rangle,

        with :math:`\Omega^{\bar D}[f] = J\bar D - \bar D J^\top`,
        :math:`J_{kl} = \partial_{v_l} f_k`, and :math:`M_x = 7/18` the
        positional defect of :math:`\hat x = (x_{t-1}+x_t+x_{t+1})/3`.  The
        :math:`h/3` terms are the cusp of the velocity autocorrelation
        (:math:`\langle f(\hat v)\rangle = \langle f\rangle - \tfrac h3
        \bar D:\langle\partial_v^2 f\rangle`); the :math:`\Omega` term is the
        antisymmetric (gyroscopic) velocity-Jacobian part of the
        :math:`\hat a`–:math:`\hat v` noise cross-covariance.  The residual
        coefficient bias is :math:`O(h^2)` (verified on exact Gaussian
        moments of linear systems, anisotropic :math:`\bar D` included).

        Returns ``(G, G_omega, M)`` with the :math:`\Omega` part of the Gram
        separate, to be applied explicitly (evaluated at given coefficients):
        solving with :math:`G + G_\Omega` has a spurious pole where
        :math:`\tfrac{7h}{6}\bar D` reaches :math:`\langle\hat v\hat
        v^\top\rangle` (:math:`\gamma\Delta t \approx 0.8` on the driven
        trap), an artefact of truncating the expansion.
        """
        G_terms = [
            ("ipa,pq,iqb", ("b", "A", "b"), 1.0),
            ("kl,ikpa,pq,ilqb", ("D", "Dvb", "A", "Dvb"), 2.0 * h / 3.0),
            ("kl,ipa,pq,iklqb", ("D", "b", "A", "Dvvb"), h / 3.0),
        ]
        Om_terms = [
            ("imkb,ml,kp,ilpa", ("Dvb", "D", "A", "Dvb"), -7.0 * h / 12.0),
            ("km,imlb,kp,ilpa", ("D", "Dvb", "A", "Dvb"), 7.0 * h / 12.0),
        ]
        M_terms = [
            ("im,mn,ina", ("a", "A", "b"), 1.0),
            ("mn,no,imoa", ("D", "A", "Dvb"), -1.0),
            ("mn,no,imoa", ("D", "A", "Dxb"), -2.0 * _ULI_POSITIONAL_DEFECT * h),
        ]
        G, G_om, M = (
            integrate(self.data, self._entropy_kinematic_integrand(terms), reduce="sum",
                      chunk_target_bytes=self._chunk_target_bytes)
            for terms in (G_terms, Om_terms, M_terms)
        )
        return G, G_om, M

    def _entropy_dt_programs(self, h, *, C=None):
        r"""O(h) kinematic counterterms of the entropy functional.

        On the symmetric kinematics the plug-in objects carry, to first order
        in :math:`h`, :math:`\mathbb E\,\varphi^-_a/\tau = \langle F\cdot
        A^{-1} b^-_a\rangle + \langle\bar D A^{-1}:\partial_v b^-_a\rangle +
        h\,T_a` and :math:`\mathbb E\,G^{+-}_{ab}/\tau = \langle b^+_a\cdot
        A^{-1} b^-_b\rangle - \tfrac h3 \bar D:\langle\partial_v^2(b^+_a\cdot
        A^{-1} b^-_b)\rangle`, with :math:`T_a` the moment defect of
        :meth:`_entropy_force_system`.  Subtracting both from
        :math:`2C^\top\varphi^- - 2C^\top G^{+-}C` leaves, per unit time,

        .. math::

           \tfrac{2h}{3}\bar D:\bigl[\hat F^-\!\cdot A^{-1}\partial_v^2\hat F^-
             - \partial_v^2\hat F^+\!\cdot A^{-1}\hat F^-
             - 2\,\partial_v\hat F^+\!\cdot A^{-1}\partial_v\hat F^-\bigr]
           - 4M_x h\,\bar D A^{-1}:\partial_x\hat F^-
           + \tfrac{7h}{6}\,\Omega^{\bar D}[\hat F]:A^{-1}\partial_v\hat F^-
           + \tfrac{2h}{3}\,\bar D_{kl}(\bar D A^{-1})_{mn}\,
             \partial_{v_k}\partial_{v_l}\partial_{v_m}\hat F^-_n .

        The last term is the cusp of the Stratonovich term
        :math:`\langle\bar D A^{-1}:\partial_v F^-\rangle`, which the raw
        :math:`\varphi^-` carries at :math:`\hat v` (the force moments
        subtract it at :math:`\hat v` too, so it is absent from the fit).

        Returns ``(K, l)`` integrands (``iab`` / ``ia``) such that the
        correction is :math:`C^\top K C + l^\top C`, or with ``C`` one per-row
        scalar integrand.
        """
        quad = [
            ("kl,ipa,pq,iklqb", ("D", "b-", "A", "Dvvb-"), 2.0 * h / 3.0),
            ("kl,iklpa,pq,iqb", ("D", "Dvvb+", "A", "b-"), -2.0 * h / 3.0),
            ("kl,ikpa,pq,ilqb", ("D", "Dvb+", "A", "Dvb-"), -4.0 * h / 3.0),
            ("imka,ml,kp,ilpb", ("Dvb", "D", "A", "Dvb-"), 7.0 * h / 6.0),
            ("km,imla,kp,ilpb", ("D", "Dvb", "A", "Dvb-"), -7.0 * h / 6.0),
        ]
        lin = [
            ("kl,ikla", ("DA", "Dxb-"), -4.0 * _ULI_POSITIONAL_DEFECT * h),
            ("kl,mn,iklmna", ("D", "DA", "Dvvvb-"), 2.0 * h / 3.0),
        ]
        if C is not None:
            return self._entropy_kinematic_integrand(quad + lin, contract=C)
        return (self._entropy_kinematic_integrand(quad), self._entropy_kinematic_integrand(lin))

    def _entropy_block_error(self, C, n_blocks: int = 16, *, mode=None, dt_correction=None):
        """Empirical block-variance error bar for the entropy plug-in.

        Evaluates the per-row plug-in ΔŜ contributions with the fitted
        coefficients held fixed (on the rows of the enclosing
        ``shared_rows`` block, masks and extras included), sums them over
        ``n_blocks`` contiguous windows of the frame axis (datasets laid end
        to end) and returns ``sqrt(K/(K-1) · K · Var_k)`` — an error bar that
        includes the slowly-correlated current fluctuations the analytic
        (martingale) formula misses.  The windows partition the rows of the
        estimate, so their sums add up to ΔŜ.  Returns ``None`` when a window
        would span fewer than 16 frames; callers fall back to the analytic
        bar.
        """
        T_total = sum(int(ds.T) for ds in self.data.datasets)
        block_len = T_total // n_blocks
        if block_len < 16:
            return None
        rows = integrate_rows(
            self.data, self._entropy_scalar_program(C, mode=mode, dt_correction=dt_correction),
            chunk_target_bytes=self._chunk_target_bytes,
        )
        offsets = np.cumsum([0] + [int(ds.T) for ds in self.data.datasets])
        vals = np.zeros(n_blocks)
        for ds_idx, (t_idx, Y) in rows.items():
            k = np.minimum((offsets[ds_idx] + t_idx) // block_len, n_blocks - 1)
            np.add.at(vals, k, np.asarray(Y, dtype=float))
        K = float(n_blocks)
        var_total = K * float(np.var(vals)) * K / (K - 1.0)
        return float(var_total ** 0.5)

    def _mean_v2_trace(self) -> float:
        """Time-average of |v̂|² (per row, summed over particles and components).

        Used for the γ̂ ≈ Tr(D̄)/⟨|v̂|²⟩ sampling-resolution heuristic; for
        multi-particle data the per-particle normalization cancels between
        numerator and denominator at the level of this order-of-magnitude
        check, so we normalize by the particle count.
        """
        mode = getattr(self, "__force_M_mode__", "symmetric")
        v_hat = self._get_v_hat(M_mode=mode)
        v_op = TimeOperand(v_hat, alias="v")
        prog = Integrand(times=[v_op], terms=[Term(eq="im,im->i", ops=("v", "v"))])
        total = float(
            integrate(
                self.data, prog, reduce="mean", weight_by_dt=False,
                chunk_target_bytes=self._chunk_target_bytes,
            )
        )
        n_particles = int(self.data.peek_row(require={"X"})["X"].shape[0])
        return total / max(n_particles, 1)

    @staticmethod
    def _psd_rank(G, rtol: float = 1e-5) -> int:
        """Numerical rank of a (nominally PSD) Gram matrix via eigvalsh.

        The tolerance is calibrated for float32 extensive sums: parity-even
        features cancel in the four-term ``G^{--}`` construction only to
        roundoff, leaving spurious eigenvalues ~1e-7 of the largest —
        comfortably below 1e-5 — while genuinely independent odd features
        sit orders of magnitude above it.
        """
        G = jnp.asarray(G)
        if G.size == 0:
            return 0
        w = jnp.linalg.eigvalsh(0.5 * (G + G.T))
        wmax = float(jnp.max(jnp.abs(w)))
        if wmax <= 0.0:
            return 0
        return int(jnp.sum(w > rtol * wmax))

    def _diffusion_G_matrix(self, *, program_only: bool = False, return_measure: bool = False) -> jnp.ndarray:
        # Diffusion is per-point: weight the projection per-point (weight_by_dt=False).
        return self.__G_matrix__(
            self.diffusion_basis,
            self.diffusion_basis,
            self.__diffusion_G_mode__,
            "imna,imnb->iab",
            weight_by_dt=False,
            program_only=program_only,
            return_measure=return_measure,
        )

    def _diffusion_basis_operand(self) -> ExprOperand:
        """Diffusion basis at the kinematics (X_t, V_t) of the chosen local estimator."""
        method = getattr(self, "__diffusion_M_mode__", None)
        if method is None:
            raise RuntimeError("Missing __diffusion_M_mode__. Call infer_diffusion_linear() first.")

        if method == "noisy":
            x_hat = _X_noisy_uli
            v_hat = _V_noisy_uli
        elif method == "MSD":
            # Reuse the symmetric ULI reconstruction
            x_hat = _X_sym_uli
            v_hat = _V_sym_uli
        elif method == "WeakNoise":
            x_hat = _X_weaknoise_uli
            v_hat = velocity("dX", "dt")
        else:
            raise KeyError(f"Unknown underdamped diffusion moment method: {method!r}")

        return ExprOperand(expr=self.diffusion_basis, x=x_hat, v=v_hat, alias="B")

    def _diffusion_moments(self, *, program_only: bool = False) -> jnp.ndarray:
        """
        ULI diffusion linear moments.

        For each method, define kinematics (X_t, V_t) and use the corresponding instantaneous
        diffusion estimator D_inst(t):

            noisy:
                X_t = 0.25*(X + X_minus + X_plus + X_plusplus)
                V_t = (4 dX + dX_plus + dX_minus) / (6 dt)
            MSD:
                X_t = (X + X_minus + X_plus)/3
                V_t = (dX + dX_minus)/(2 dt)
            WeakNoise:
                X_t = (X + X_plus)/2
                V_t = dX/dt

        Then:
            M_a = < B_a(X_t, V_t) : D_inst(t) >
                = sum_t sum_i  einsum('imna,imn->ia', B, D_inst)
        """
        method = getattr(self, "__diffusion_M_mode__", None)
        B = self._diffusion_basis_operand()
        logger.debug("Computing diffusion linear moments (method=%s).", method)

        # Instantaneous diffusion estimator matching the method string
        D_op = self.get_diffusion_timeop(method)

        prog = Integrand(
            exprs=[B],
            times=[D_op],
            terms=[Term(eq="imna,imn->ia", ops=("B", D_op.alias))],
        )
        # Per-point (weight_by_dt=False): each increment's diffusion estimate counts equally.
        if program_only:
            return prog
        return integrate(self.data, prog, reduce="sum", weight_by_dt=False,
                         chunk_target_bytes=self._chunk_target_bytes)

    def _update_force_inferred(self) -> None:
        """
        Materialize fitted force as SF and wrap in InferenceResultSF.
        """
        P: PSF = self.force_basis.to_psf()
        theta = {"coeff": jnp.asarray(self.force_coefficients_full)}
        sf = SF(P, theta)

        meta = dict(
            kind="force",
            modes=dict(
                M=getattr(self, "__force_M_mode__", None),
                G=getattr(self, "__force_G_mode__", None),
            ),
            diffusion_method=getattr(self, "__force_diffusion_method__", None),
            A_inv=jnp.asarray(getattr(self, "A_inv", None)) if hasattr(self, "A_inv") else None,
            basis_features=int(getattr(self.force_basis, "n_features", 0)),
            basis_labels=getattr(self.force_basis, "labels", None),
        )
        # New coefficients: their covariance is set by compute_force_error
        # (sparsify_force refreshes it), never inherited from previous ones.
        self.force_inferred = InferenceResultSF(sf, param_cov=None, meta=meta)

    def _update_diffusion_inferred(self) -> None:
        """
        Materialize fitted diffusion as SF and wrap in InferenceResultSF.
        """
        if self.diffusion_basis is None:
            raise RuntimeError("_update_diffusion_inferred called before diffusion was fitted.")
        P: PSF = self.diffusion_basis.to_psf()
        theta = {"coeff": jnp.asarray(self.diffusion_coefficients_full)}
        sf = SF(P, theta)

        meta = dict(
            kind="diffusion",
            modes=dict(
                M=getattr(self, "__diffusion_M_mode__", None),
                G=getattr(self, "__diffusion_G_mode__", None),
            ),
            A_inv=jnp.asarray(getattr(self, "A_inv", None)) if hasattr(self, "A_inv") else None,
            basis_features=int(getattr(self.diffusion_basis, "n_features", 0)),
            basis_labels=getattr(self.diffusion_basis, "labels", None),
        )
        # New coefficients: their covariance is set by compute_diffusion_error
        # (sparsify_diffusion refreshes it), never inherited from previous ones.
        self.diffusion_inferred = InferenceResultSF(sf, param_cov=None, meta=meta)

    # ------------------------------ shared helpers ------------------------------ #

    def __G_matrix__(
        self,
        b_left: callable,
        b_right: callable,
        G_mode: str,
        einsum_string: str,
        subsampling: int = 1,
        weight_by_dt: bool = True,
        program_only: bool = False,
        return_measure: bool = False,
    ) -> jnp.ndarray:
        """
        ULI Gram matrix.

        Uses secant velocities from raw displacements:
          V(t)      = dX/dt
          V_minus(t)= dX_minus/dt
          V_plus(t) = dX_plus/dt

        Modes — the returned Gram is the transpose Gᵀ of the tabulated outer
        product.  The averaging is asymmetric in the left/right factors, and
        grid assessments (VdP, OU) show Gᵀ is the better-conditioned regressor,
        so it is the canonical form:

        rectangle:   < bL(X, V) ⊗ bR(X, V) >
        shift:       < bL(X, V) ⊗ bR(X_minus, V_minus) >
        trapeze:     < bL(X, V) ⊗ 0.5*(bR(X, V) + bR(X_minus, V_minus)) >
        doubleshift: < bL(X_plus, V_plus) ⊗ bR(X_minus, V_minus) >
        """
        logger.debug(
            "Computing ULI G matrix (mode=%s) with einsum: %s",
            G_mode,
            einsum_string,
        )

        # Rectangle program
        BL = ExprOperand(expr=b_left, x=stream("X"), v=velocity("dX", "dt"), alias="BL")
        BR0 = ExprOperand(expr=b_right, x=stream("X"), v=velocity("dX", "dt"), alias="BR0")
        prog_rect = Integrand(exprs=[BL, BR0], terms=[Term(eq=einsum_string, ops=("BL", "BR0"))])

        if G_mode == "rectangle":
            prog = prog_rect

        elif G_mode in ("trapeze", "shift", "doubleshift"):
            BRm = ExprOperand(
                expr=b_right,
                x=stream("X_minus"),
                v=velocity("dX_minus", "dt"),
                alias="BRm",
            )
            prog_shift = Integrand(exprs=[BL, BRm], terms=[Term(eq=einsum_string, ops=("BL", "BRm"))])

            if G_mode == "shift":
                prog = prog_shift
            elif G_mode == "trapeze":
                prog = 0.5 * (prog_rect + prog_shift)
            else:  # doubleshift
                BLp = ExprOperand(
                    expr=b_left,
                    x=stream("X_plus"),
                    v=velocity("dX_plus", "dt"),
                    alias="BLp",
                )
                prog = Integrand(exprs=[BLp, BRm], terms=[Term(eq=einsum_string, ops=("BLp", "BRm"))])

        else:
            raise KeyError(f"Wrong G_mode argument for ULI: {G_mode}")
        if program_only:
            return prog

        # The ULI Gram is returned transposed (Gᵀ): the averaging is asymmetric
        # in the (left, right) factors and Gᵀ is the better-conditioned regressor.
        G = integrate(
            self.data,
            prog,
            reduce="sum",
            reduce_over_particles=True,
            subsampling=subsampling,
            weight_by_dt=weight_by_dt,
            chunk_target_bytes=self._chunk_target_bytes,
            return_measure=True,
        )
        G, measure = G
        # ``integrate`` returns the scalar ``0.0`` when the plan has no chunks,
        # i.e. the collection yielded no valid frames for this integrand. Feeding
        # that scalar to ``swapaxes`` below would raise a cryptic
        # "index -2 is out of bounds for axis 0 with size 0". Surface the real
        # cause instead.
        if jnp.ndim(G) < 2:
            raise ValueError(
                f"ULI Gram integration (G_mode={G_mode!r}) found no valid frames: "
                "the collection produced zero usable rows for this basis, so the "
                "Gram collapsed to a scalar instead of an (i, a, b) array. Common "
                "causes: the trajectory is too short for the requested stencil "
                "(secant velocities need a neighbouring frame), or — for an "
                "interaction/structural basis — the collection is missing the "
                "reserved structural extras the basis needs (frequent with "
                "bootstrapped collections built without those extras). Verify the "
                "collection has valid frames and carries the extras the basis requires."
            )
        G = jnp.swapaxes(G, -1, -2)
        return (G, measure) if return_measure else G

    # ------------------------------ diffusion timeops ------------------------------ #

    def _build_diffusion_timeoperands(self) -> None:
        if hasattr(self, "_diff_ops"):
            return
        self._diff_ops = {
            "MSD": TimeOperand(_D_msd_uli, alias="D_msd"),
            "WeakNoise": TimeOperand(_D_weaknoise_uli, alias="D_weaknoise"),
            "noisy": TimeOperand(_D_noisy_uli, alias="D_noisy"),
            "Lambda": TimeOperand(_Lambda_uli, alias="Lambda"),
        }

    def get_diffusion_timeop(self, method: str) -> TimeOperand:
        self._build_diffusion_timeoperands()
        try:
            return self._diff_ops[method]  # type: ignore[attr-defined]
        except KeyError as e:
            raise KeyError(f"Unknown underdamped diffusion estimator method: {method}") from e

    # ------------------------------ kinematics selectors ------------------------------ #

    def _build_kinematics_timeoperands(self) -> None:
        if hasattr(self, "_kin_ops"):
            return
        self._kin_ops = {
            "x": {
                "symmetric": _X_sym_uli,
                "early": _X_early_uli,
                "anticipated": _X_anticipated_uli,
            },
            "v": {
                "symmetric": _V_sym_uli,
                "early": _V_early_uli,
                "anticipated": _V_anticipated_uli,
            },
            "a": {
                "symmetric": _A_sym_uli,
                "early": _A_sym_uli,
                "anticipated": _A_anticipated_uli,
            },
        }

    def _get_x_hat(self, *, M_mode: str, at: str = "t") -> TimeOp:
        if at != "t":
            raise NotImplementedError("x_hat shifting not implemented yet (only at='t').")
        self._build_kinematics_timeoperands()
        return self._kin_ops["x"][M_mode]  # type: ignore[index]

    def _get_v_hat(self, *, M_mode: str, at: str = "t") -> TimeOp:
        if at != "t":
            raise NotImplementedError("v_hat shifting not implemented yet (only at='t').")
        self._build_kinematics_timeoperands()
        return self._kin_ops["v"][M_mode]  # type: ignore[index]

    def _get_a_hat(self, *, M_mode: str, at: str = "t") -> TimeOp:
        if at != "t":
            raise NotImplementedError("a_hat shifting not implemented yet (only at='t').")
        self._build_kinematics_timeoperands()
        return self._kin_ops["a"][M_mode]  # type: ignore[index]


# -------------------- Underdamped (ULI) kinematics as TimeOps -------------------- #
# .. physics:: ULI kinematic reconstructions
#    :label: uli-kinematics
#    :category: Kinematics
#
# The underdamped inference reconstructs (x̂, v̂, â) from observed positions
# via finite differences.  Three modes:
#
# Symmetric:   x̂ = (x_{t-1}+x_t+x_{t+1})/3,   v̂ = (dX+dX⁻)/(2dt),   â = (dX−dX⁻)/dt²
# Early:       x̂ = x_t,                          v̂ = dX⁻/dt
# Anticipated: x̂ = (x_t+x_{t+1}+x_{t+2})/3,    â = (dX⁺−dX)/dt²


@timeop(name="X_sym_uli")
def _X_sym_uli(**streams):
    r"""
    .. physics:: Symmetric ULI kinematic reconstruction
       :label: uli-kinematics-symmetric
       :category: Kinematics

       .. math::

          \\hat x(t) = \\tfrac{1}{3}\\bigl[x_{t-1}+x_t+x_{t+1}\\bigr],
          \\quad
          \\hat v(t) = \\frac{\\mathrm{d}X_t+\\mathrm{d}X_{t-1}}{2\\,\\mathrm{d}t},
          \\quad
          \\hat a(t) = \\frac{\\mathrm{d}X_t - \\mathrm{d}X_{t-1}}{\\mathrm{d}t^2}
    """
    # x̂(t) = (x(t-1) + x(t) + x(t+1)) / 3
    return (streams["X"] + streams["X_minus"] + streams["X_plus"]) / 3.0


_X_sym_uli._requires = frozenset({"X", "X_minus", "X_plus"})  # type: ignore[attr-defined]


@timeop(name="X_early_uli")
def _X_early_uli(**streams):
    # x̂(t) = x(t)
    return streams["X"]


_X_early_uli._requires = frozenset({"X"})  # type: ignore[attr-defined]


@timeop(name="X_anticipated_uli")
def _X_anticipated_uli(**streams):
    # x̂(t) = (x(t) + x(t+1) + x(t+2)) / 3
    return (streams["X"] + streams["X_plus"] + streams["X_plusplus"]) / 3.0


_X_anticipated_uli._requires = frozenset({"X", "X_plus", "X_plusplus"})  # type: ignore[attr-defined]


@timeop(name="V_sym_uli")
def _V_sym_uli(**streams):
    # v̂(t) = (dX(t) + dX_minus(t)) / (2 dt)
    dt = streams["dt"][..., None, None]
    return 0.5 * (streams["dX"] + streams["dX_minus"]) / dt


_V_sym_uli._requires = frozenset({"dX", "dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="V_early_uli")
def _V_early_uli(**streams):
    # v̂(t) = dX_minus(t) / dt
    dt = streams["dt"][..., None, None]
    return streams["dX_minus"] / dt


_V_early_uli._requires = frozenset({"dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="V_anticipated_uli")
def _V_anticipated_uli(**streams):
    # v̂(t) = dX_minus(t) / dt
    dt = streams["dt"][..., None, None]
    return streams["dX_minus"] / dt


_V_anticipated_uli._requires = frozenset({"dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="A_sym_uli")
def _A_sym_uli(**streams):
    # â(t) = (dX(t) - dX_minus(t)) / dt^2
    dt = streams["dt"][..., None, None]
    return (streams["dX"] - streams["dX_minus"]) / (dt**2)


_A_sym_uli._requires = frozenset({"dX", "dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="A_anticipated_uli")
def _A_anticipated_uli(**streams):
    # â(t) = (dX_plus(t) - dX(t)) / dt^2
    dt = streams["dt"][..., None, None]
    return (streams["dX_plus"] - streams["dX"]) / (dt**2)


_A_anticipated_uli._requires = frozenset({"dX_plus", "dX", "dt"})  # type: ignore[attr-defined]

# -------------------- Underdamped (ULI) diffusion kinematics as TimeOps -------------------- #


@timeop(name="X_noisy_uli")
def _X_noisy_uli(**streams):
    # X_t (noisy diffusion): 0.25*(X + X_minus + X_plus + X_plusplus)
    return 0.25 * (streams["X"] + streams["X_minus"] + streams["X_plus"] + streams["X_plusplus"])


_X_noisy_uli._requires = frozenset({"X", "X_minus", "X_plus", "X_plusplus"})  # type: ignore[attr-defined]


@timeop(name="V_noisy_uli")
def _V_noisy_uli(**streams):
    # V_t (noisy diffusion): (4 dX + dX_plus + dX_minus) / (6 dt)
    dt = streams["dt"][..., None, None]
    return (4.0 * streams["dX"] + streams["dX_plus"] + streams["dX_minus"]) / (6.0 * dt)


_V_noisy_uli._requires = frozenset({"dX", "dX_plus", "dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="X_weaknoise_uli")
def _X_weaknoise_uli(**streams):
    # X_t (WeakNoise diffusion): 0.5*(X + X_plus)
    return 0.5 * (streams["X"] + streams["X_plus"])


_X_weaknoise_uli._requires = frozenset({"X", "X_plus"})  # type: ignore[attr-defined]

# -------------------- Underdamped (ULI) diffusion estimators as TimeOps -------------------- #


def _symmetrize_imn(M: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


@timeop(name="D_msd_uli")
def _D_msd_uli(**streams):
    r"""
    ULI 'MSD' instantaneous diffusion estimator:
        D = (3/4) * ( (dX - dX_minus) ⊗ (dX - dX_minus) ) / dt^3
    Returns (N, d, d).

    .. physics:: MSD diffusion estimator (underdamped)
       :label: D-msd-underdamped
       :category: Estimator

       .. math::

          \hat D_{\text{MSD}}^{\text{ULI}}(t)
          = \frac{3}{4}\,
            \frac{(\mathrm{d}X_t - \mathrm{d}X_{t-1})
            \otimes (\mathrm{d}X_t - \mathrm{d}X_{t-1})}
            {\mathrm{d}t^3}
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    ddX = dX - dXm
    dt = streams["dt"][..., None, None]
    return 0.75 * jnp.einsum("im,in->imn", ddX, ddX) / (dt**3)


_D_msd_uli._requires = frozenset({"dX", "dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="D_weaknoise_uli")
def _D_weaknoise_uli(**streams):
    r"""
    ULI 'WeakNoise' estimator:
        D = (1/2) * ( (2 dX - dX_minus - dX_plus) ⊗ (2 dX - dX_minus - dX_plus) ) / dt^3
    Returns (N, d, d).

    .. physics:: Weak-noise diffusion estimator (underdamped)
       :label: D-weaknoise-underdamped
       :category: Estimator

       .. math::

          \hat D_{\text{WN}}^{\text{ULI}}(t)
          = \frac{1}{2}\,
            \frac{(2\,\mathrm{d}X_t - \mathrm{d}X_{t-1} - \mathrm{d}X_{t+1})
            \otimes (\cdots)}{\mathrm{d}t^3}
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    dXp = streams["dX_plus"]
    u = 2.0 * dX - dXm - dXp
    dt = streams["dt"][..., None, None]
    return 0.5 * jnp.einsum("im,in->imn", u, u) / (dt**3)


_D_weaknoise_uli._requires = frozenset({"dX", "dX_minus", "dX_plus", "dt"})  # type: ignore[attr-defined]


@timeop(name="D_noisy_uli")
def _D_noisy_uli(**streams):
    r"""
    ULI 'noisy' diffusion estimator:
        D = (3 / (11 dt^3)) * sym( -a + b + c - 3d + e + f )
    with:
        a = dX ⊗ dX
        b = dX_minus ⊗ dX_minus
        c = dX_plus ⊗ dX_plus
        d = dX_plus ⊗ dX_minus
        e = dX ⊗ dX_plus
        f = dX ⊗ dX_minus
    Returns (N, d, d).

    .. physics:: Noisy diffusion estimator (underdamped)
       :label: D-noisy-underdamped
       :category: Estimator

       .. math::

          \hat D_{\text{noisy}}^{\text{ULI}}(t)
          = \frac{3}{11\,\mathrm{d}t^3}\,\operatorname{sym}\!
            \bigl[-a + b + c - 3d + e + f\bigr]

       where :math:`a = \mathrm{d}X_t\otimes\mathrm{d}X_t`,
       :math:`b = \mathrm{d}X_{t-1}\otimes\mathrm{d}X_{t-1}`, etc.
       Optimally handles both signal and localization noise.

       Non-uniform sampling: two interval-pattern effects are handled.
       The stencil's *ballistic* cancellation — the :math:`\langle v^2
       \rangle` terms sum to :math:`Q(u,v,w) = -u^2+v^2+w^2-3wv+uw+uv`
       with :math:`(u,v,w) = (\mathrm{d}t, \mathrm{d}t^-, \mathrm{d}t^+)`,
       zero only at equal intervals — is restored by subtracting
       :math:`Q\,\widehat{v^2}` with the η-clean velocity-square estimate
       :math:`\widehat{v^2} = \mathrm{sym}(\mathrm{d}X^+\!\otimes
       \mathrm{d}X^-)/(\mathrm{d}t^+ \mathrm{d}t^-)` (no shared
       localization noise, so the Λ cancellation — which is
       interval-independent — is untouched).  The :math:`\mathrm{d}t^3`
       normalisation uses the local three-interval mean.  Both corrections
       vanish *bitwise* at uniform sampling; under jitter the estimator is
       initialisation-grade (the exact-core likelihood profile refines
       :math:`\hat D`).
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    dXp = streams["dX_plus"]

    a = jnp.einsum("im,in->imn", dX, dX)
    b = jnp.einsum("im,in->imn", dXm, dXm)
    c = jnp.einsum("im,in->imn", dXp, dXp)
    d = jnp.einsum("im,in->imn", dXp, dXm)
    e = jnp.einsum("im,in->imn", dX, dXp)
    f = jnp.einsum("im,in->imn", dX, dXm)

    M = (-1.0) * a + (1.0) * b + (1.0) * c + (-3.0) * d + (1.0) * e + (1.0) * f
    M = _symmetrize_imn(M)

    dt = streams["dt"][..., None, None]
    dtm = streams["dt_minus"][..., None, None]
    dtp = streams["dt_plus"][..., None, None]
    # ballistic-leakage correction: Q = −dt² + dt⁻² + dt⁺² − 3dt⁺dt⁻
    # + dt·dt⁺ + dt·dt⁻, written in the exact difference form
    # δ⁻² + δ⁺² − 3δ⁻δ⁺ (δ± = dt± − dt) so that Q ≡ 0.0 bitwise at equal
    # intervals; the η-clean dX⁺⊗dX⁻ estimate keeps the
    # localization-noise cancellation intact.
    dv, dw = dtm - dt, dtp - dt
    Q = dv * dv + dw * dw - 3.0 * dv * dw
    M = M - Q * _symmetrize_imn(d) / (dtp * dtm)
    # local 3-interval mean for the dt³ normalisation (bitwise dt at
    # uniform sampling: the correction term is exactly 0.0 there)
    dt_bar = dt + ((dtm - dt) + (dtp - dt)) / 3.0
    return (3.0 / 11.0) * M / (dt_bar**3)


_D_noisy_uli._requires = frozenset({"dX", "dX_minus", "dX_plus", "dt",
                                    "dt_minus", "dt_plus"})  # type: ignore[attr-defined]


@timeop(name="Lambda_uli")
def _Lambda_uli(**streams):
    r"""
    ULI 'Lambda' instantaneous measurement-noise estimator:
        Λ = sym( 10a + b + c + 8d - 10e - 10f ) / 44
    with the same (a..f) definitions as in _D_noisy_uli.
    Returns (N, d, d).

    .. physics:: Measurement noise estimator (underdamped)
       :label: Lambda-underdamped
       :category: Estimator

       .. math::

          \hat\Lambda^{\text{ULI}}
          = \frac{1}{44}\,\operatorname{sym}\!
            \bigl[10a + b + c + 8d - 10e - 10f\bigr]

       Same displacement products :math:`(a\ldots f)` as the noisy
       diffusion estimator.  Extracts localization noise for
       underdamped systems.
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    dXp = streams["dX_plus"]

    a = jnp.einsum("im,in->imn", dX, dX)
    b = jnp.einsum("im,in->imn", dXm, dXm)
    c = jnp.einsum("im,in->imn", dXp, dXp)
    d = jnp.einsum("im,in->imn", dXp, dXm)
    e = jnp.einsum("im,in->imn", dX, dXp)
    f = jnp.einsum("im,in->imn", dX, dXm)

    M = (10.0) * a + (1.0) * b + (1.0) * c + (8.0) * d + (-10.0) * e + (-10.0) * f
    return _symmetrize_imn(M) / 44.0


_Lambda_uli._requires = frozenset({"dX", "dX_minus", "dX_plus"})  # type: ignore[attr-defined]
