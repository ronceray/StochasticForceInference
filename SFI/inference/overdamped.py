from __future__ import annotations

import logging

import jax.numpy as jnp

from SFI.bases.constants import constant_array
from SFI.inference.base import BaseLangevinInference
from SFI.integrate.api import integrate, required_frames, shared_rows
from SFI.integrate.integrand import (
    ConstOperand,
    ExprOperand,
    Integrand,
    Term,
    TimeOperand,
)
from SFI.integrate.timeops import stream, timeop, velocity
from SFI.statefunc import PSF, SF, Basis
from SFI.utils.maths import sqrtm_psd, stable_pinv

from .result import InferenceResultSF  # fitted SF with param_cov/meta
from .sparse import SparseScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# OLI linear force-inference presets
# ---------------------------------------------------------------------------- #
# A single ``preset`` keyword bundles the (M_mode, G_mode) convention surface,
# mirroring the underdamped engine.  ``'auto'`` keys off measurement noise; the
# ``legacy-*`` presets reproduce published SFI v1.0 conventions.
_FORCE_LINEAR_PRESETS: dict[str, tuple[str, str]] = {
    # noise-robust default: Stratonovich moments + noise-decorrelating Gram
    "robust":          ("Strato", "shift"),
    # sharper for verified clean / fine-sampled data
    "clean":           ("Ito", "trapeze"),
    # Kramers–Moyal: plain Itô finite-difference moments, rectangle Gram
    "KM":              ("Ito", "rectangle"),
    # SFI v1.0 (2020 PRX): Stratonovich moments, rectangle Gram
    "legacy-sfi-v1.0": ("Strato", "rectangle"),
}


def _resolve_force_linear_preset(
    preset: str,
    *,
    M_mode: str | None = None,
    G_mode: str | None = None,
    noise_detected: bool | None = None,
) -> tuple[str, str, str]:
    """Resolve an overdamped force ``preset`` to a concrete ``(M_mode, G_mode)``.

    ``'auto'`` selects ``'robust'`` when measurement noise is detected and
    ``'clean'`` otherwise.  Any non-``None`` ``M_mode`` / ``G_mode`` (other than
    the legacy ``M_mode='auto'`` synonym) overrides the preset on that axis.
    Returns ``(M_mode, G_mode, resolved_preset)``.
    """
    if preset == "auto":
        if noise_detected is None:
            raise RuntimeError(
                "preset='auto' needs the measurement-noise estimate; call "
                "compute_diffusion_constant() first, or pass an explicit preset."
            )
        base = "robust" if noise_detected else "clean"
    elif preset in _FORCE_LINEAR_PRESETS:
        base = preset
    else:
        choices = ", ".join(repr(p) for p in ("auto", *_FORCE_LINEAR_PRESETS))
        raise ValueError(f"unknown preset {preset!r}; choose from {choices}")
    M0, G0 = _FORCE_LINEAR_PRESETS[base]
    M = M0 if M_mode in (None, "auto") else M_mode
    G = G0 if G_mode is None else G_mode
    return M, G, base


class OverdampedLangevinInference(BaseLangevinInference):
    r"""Stochastic Force Inference concrete class for overdamped systems

    This class provides tools for inferring force (drift) and
    diffusion tensors from stochastic trajectory data based on overdamped
    Langevin dynamics. It supports both linear and nonlinear basis
    function methods.

    Core Equation
    ~~~~~~~~~~~~~
    The dynamics are described by the 1st order autonomous stochastic
    differential equation (SDE)::

        dx/dt = F(x) + sqrt(2D(x)) dxi(t)

    where:

    - ``F(x)`` is the Ito drift (force) term.
    - ``D(x)`` is the diffusion tensor, evaluated in the Ito convention.
    - ``dxi(t)`` is Gaussian white noise.

    .. physics:: Overdamped Langevin SDE
       :label: overdamped-langevin-sde
       :category: Dynamical equations

       .. math::

          \frac{\mathrm{d}x}{\mathrm{d}t}
          = F(x) + \sqrt{2\,D(x)}\;\mathrm{d}\xi(t)

       :math:`F(x)` is the Itô drift, :math:`D(x)` the diffusion tensor
       (Itô convention), and :math:`\mathrm{d}\xi` is Gaussian white noise.

    Here x is a 2D array of shape Nparticles x dimension. All particles
    are assumed to have identical properties.

    This class provides tools to approximate F(x) and D(x) from a time series x(t) formatted as TrajectoryCollection.

    Note that the `Ito` and `Strato` variants of the force inference
    routines do NOT refer to the convention in which the SDE is
    expressed (which is always Ito), but to the way stochastic
    integrals are performed to compute parameters.

    Key Features
    ~~~~~~~~~~~~
    - Force Inference:
      - Linear combination of basis functions (`infer_force_linear`).
      - Parametric families (`infer_force` with a `Basis` or `PSF`).
    - Diffusion Inference:
      - Constant diffusion estimation (`compute_diffusion_constant`).
      - State-dependent diffusion with basis functions (`infer_diffusion_linear` and `infer_diffusion`).
    - Sparsification:
      - Force sparsification for linear inference `sparsify_force`, implementing PASTIS and other information criteria.
    - Error Estimation:
      - Normalized mean-squared error (MSE) prediction for both force and diffusion.
    - Comparison Tools for method benchmarking:
      - Evaluate inferred fields against known exact models (`compare_to_exact`).
    - Simulation:
      - Generate trajectories using inferred fields (`simulate_bootstrapped_trajectory`).

    Workflow
    ~~~~~~~~
    1. Initialize with `TrajectoryCollection` containing the trajectory.
    2. Use the `infer_*` methods to infer force and diffusion fields.
    3. Optionally compute error estimates and/or compare with exact data for validation.

    Indices Convention
    ~~~~~~~~~~~~~~~~~~
    The code uses jnp.einsum for array manipulation, with a consistent index naming scheme for clarity:
    - `t` : Time index, = 0..Ntimesteps-1
    - `a, b, c...` : Basis function indices, = 0..Nfunctions - 1.
    - `m, n, o...` : Spatial indices, = 0..dim-1.
    - `i, j...` : Particle indices.
    We also use these indices as shortcuts for array shapes. For
    instance `basis_linear : im -> iam` reads `basis_linear has input
    a jnp.array of shape (Nparticles,dim) and outputs a jnp.array of
    shape (Nparticles,Nfunctions,dim)`.

    Logging
    ~~~~~~~
    Progress messages use Python ``logging``.  Enable with
    ``SFI.enable_logging()`` or ``logging.getLogger('SFI').setLevel(logging.INFO)``.

    Example
    ~~~~~~~
    Fully documented examples in the "examples" folder: Lorenz model, ActiveBrownianParticles, Ornstein-Uhlenbeck...

    """

    def compute_diffusion_constant(self, method: str = "auto") -> None:
        """Estimate a constant (spatially uniform) diffusion matrix.

        Parameters
        ----------
        method : {"auto", "noisy", "WeakNoise", "MSD"}
            Estimator to use.  ``"noisy"`` is the noise-robust
            Vestergaard–Blainey–Flyvbjerg estimator.  ``"auto"`` selects
            ``"noisy"`` when the measurement-noise trace Tr(Λ) > 0
            (localization noise detected), and ``"WeakNoise"`` otherwise.

        Updates
        -------
        Sets ``diffusion_average``, ``diffusion_inferred``, ``A``,
        ``A_inv``, ``sqrtA``, ``sqrtA_inv``, ``Lambda``,
        ``Lambda_trace``, and ``metadata["diffusion_constant_method"]``.
        """

        if hasattr(self, "diffusion_average"):
            raise RuntimeError("Diffusion already computed; create a new inference object to recompute.")

        # 1) measurement noise Λ
        L_op = self.get_diffusion_timeop("Lambda")
        L_prog = Integrand(times=[L_op], terms=[Term(eq="imn->imn", ops=(L_op.alias,))])
        # Diffusion/noise are per-point quantities: each increment's estimate has
        # constant relative variance regardless of dt, so combine them per-point
        # (weight_by_dt=False), not dt-weighted (which is the force convention).
        self.Lambda = integrate(self.data, L_prog, reduce="mean", weight_by_dt=False,
                                chunk_target_bytes=self._chunk_target_bytes)
        self.Lambda_trace = float(jnp.trace(self.Lambda))
        logger.info("Measurement noise trace: %s", self.Lambda_trace)

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

    def infer_force_linear(
        self,
        basis: Basis,
        *,
        preset: str = "auto",
        M_mode: str | None = None,
        G_mode: str | None = None,
        cross_particle: str = "all-pairs",
        correction_stencil: str | None = None,
    ):
        r"""Infer the force field as a linear combination of basis functions (linear regression).

        .. physics:: Linear force regression (overdamped)
           :label: linear-force-regression-overdamped
           :category: Inference

           .. math::

              \hat F(x) = \sum_a C_a\, b_a(x)
              \qquad\text{where}\qquad
              G\,C = M

           :math:`G_{ab} = \langle b_a(x_t)\, b_b(x_t) \rangle` is the Gram
           matrix, :math:`M_a = \langle v_t \cdot A^{-1} \cdot b_a \rangle` are
           the force moments, and :math:`A = 2\bar D`.

        This method computes the force field coefficients (`self.force_coefficients`) using
        the provided Basis object. The force field is represented as:

            inferred_force(x) = sum_a basis_linear(x)[:,a] * force_coefficients[a]

        These coefficients are computed by solving a linear system::

            G . force_coefficients = force_moments

        and the different options account for the manner to compute G
        and force_moments. In its simplest form::

            G_ab = < b_a(xt) b_b(xt) >                [G_mode = 'rectangle']
            force_moments[a] = < dX[t]/dt b_a(xt) >   [mode = 'Ito' ]

        but this is rarely the best choice of parameters.

        Args:

            basis: Basis
                The fitting functions, encoded as a single callable Basis
                object (see SFI.statefunc submodule for doc and SFI.bases for examples).

            preset (str):
                Single-keyword convention bundle (mirrors the underdamped
                engine), default ``'auto'``.  ``'auto'`` picks ``'robust'``
                when measurement noise is detected (Tr(Λ) > 0) and ``'clean'``
                otherwise.  Presets: ``'robust'`` (Stratonovich moments +
                ``'shift'`` Gram, noise-robust); ``'clean'`` (Itô +
                ``'trapeze'``, clean / fine-sampled data); ``'KM'``
                (Kramers–Moyal: Itô + ``'rectangle'``); ``'legacy-sfi-v1.0'``
                (Stratonovich + ``'rectangle'``, the published SFI v1.0
                convention).

            M_mode (str, optional):
                Override the preset's moment convention: ``'Ito'``,
                ``'Ito-shift'``, or ``'Strato'``.  ``None`` (default) uses the
                preset.

            G_mode (str, optional):
                Override the preset's Gram normalization: ``'rectangle'`` (2020
                PRX), ``'trapeze'`` (2024 Amiri et al. PRR), or ``'shift'``
                (``<b_a(xt) b_b(x_t+dt)>``, decorrelates measurement noise).
                ``None`` (default) uses the preset.

            cross_particle (str):
                Stratonovich moments only, interacting bases only.  The
                correction :math:`\langle D_t : A^{-1} \nabla b \rangle` uses the
                instantaneous diffusion estimator on every pair of particles,
                :math:`\sum_j D_{t,ij} : A^{-1} \partial_{x_j} b_i`
                (``'all-pairs'``, default): exact with noise shared between
                particles (hydrodynamic coupling, lattice fields with conserved
                noise) and consistent at :math:`O(\Delta t)` without.  It needs
                no diffusion model (forward-mode derivatives along the
                increments).  ``'same-particle'`` keeps only each particle's own
                block (the pre-2.0.4 behaviour), valid only for noise
                independent between particles.

            correction_stencil (str, optional):
                Instantaneous diffusion estimator of the Stratonovich
                correction.  ``'noisy_lag2'`` adds
                :math:`\tfrac12[-a\otimes c - c\otimes a + b\otimes m]/\Delta t`
                to the forward Vestergaard stencil (:math:`a = \Delta X_t`,
                :math:`b = \Delta X_{t+1}`, :math:`c = \Delta X_{t+2}`,
                :math:`m = \Delta X_{t-1}`; left factor on the moment side):
                each product pairs increments that share no frame (localisation
                noise does not bias them) and together they cancel the
                :math:`O(\Delta t)` bias of Stratonovich moments with the
                ``'shift'`` Gram, for nonlinear dynamics as well.
                ``'noisy'`` is the Vestergaard stencil alone (lower variance
                when localisation noise dominates :math:`D\Delta t` at fine
                :math:`\Delta t`).  ``None`` (default): ``'noisy_lag2'`` with the
                ``'shift'`` Gram, ``'noisy'`` otherwise (the lag-2 term is
                derived for that pairing) and with
                ``cross_particle='same-particle'`` on interacting bases.

        Outputs:
            Updates the following attributes:
                - self.force_scorer: SparseScorer for model selection.
                - self.force_coefficients: The inferred coefficients for the basis functions.
                - self.force_inferred: Callable function representing the inferred force field.
                - self.force_G: The normalization matrix used in the inference process.

        """
        noise = self.Lambda_trace > 0.0 if hasattr(self, "Lambda_trace") else None
        M_mode, G_mode, _resolved_preset = _resolve_force_linear_preset(
            preset, M_mode=M_mode, G_mode=G_mode, noise_detected=noise,
        )
        logger.info(
            "Force inference: preset=%s -> M_mode=%s, G_mode=%s (Lambda trace: %s)",
            _resolved_preset, M_mode, G_mode, getattr(self, "Lambda_trace", None),
        )

        self._validate_basis(basis, expected_rank=1, label="force basis")

        if cross_particle not in ("all-pairs", "same-particle"):
            raise ValueError("cross_particle must be 'all-pairs' or 'same-particle'")
        _same_block = cross_particle == "same-particle" and getattr(basis, "particles_input", False)
        if correction_stencil is None:
            correction_stencil = "noisy_lag2" if G_mode == "shift" and not _same_block else "noisy"
        if correction_stencil not in ("noisy", "noisy_lag2"):
            raise ValueError("correction_stencil must be 'noisy' or 'noisy_lag2'")
        if correction_stencil != "noisy" and _same_block:
            raise ValueError("correction_stencil='noisy_lag2' on an interacting basis needs cross_particle='all-pairs'")

        # Refuse before touching any fit attribute (the previous fit stays intact).
        if hasattr(self, "force_G_full"):
            raise RuntimeError("Force has already been inferred on this object - create a new instance to re-infer.")
        self._reset_fit_state("force")  # e.g. after a parametric PSF fit

        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.__force_cross_particle__ = cross_particle
        self.__force_correction_stencil__ = correction_stencil
        self.force_basis = basis
        self.metadata["force_preset"] = _resolved_preset
        self.metadata["force_M_mode"] = M_mode
        self.metadata["force_G_mode"] = G_mode

        # Structural (CSR/stencil) arrays are exposed only for this evaluation and
        # never persisted on the dataset (see ``_structural_scope``).
        with self._structural_scope(self.force_basis):
            # Gram, moments and the Stratonovich correction read different
            # frames; with missing frames they must still sum over the same
            # rows, or C = G^-1 M is biased by the ratio of row counts.
            self._force_row_frames = self._force_linear_frames()  # reused by compute_entropy_production
            with shared_rows(self._force_row_frames):
                # The Gram's row measure (particle-time) normalizes rates and model selection.
                self.force_G_full, self._force_row_measure = self._force_G_matrix(return_measure=True)
                self.force_moments = self._force_moments()

        self._check_gram_conditioning(self.force_G_full, label="force")
        self.force_scorer = SparseScorer(M=self.force_moments, G=self.force_G_full)
        self._update_force_coefficients(self.force_scorer.total_C)
        self.metadata["force_method"] = "linear"

    def infer_diffusion_linear(
        self,
        basis: Basis = None,
        *,
        M_mode: str = "auto",
        G_mode: str = "rectangle",
    ) -> None:
        """
        Fit the diffusion field as a linear combination of basis functions.

        This method computes the coefficients of the diffusion tensor field (`self.diffusion_coefficients`) using
        the provided basis functions. The diffusion tensor is represented as:

            diffusion_inferred(x, mask) = sum_a basis_linear(x, mask)[:,a] * diffusion_coefficients[a]

        Args:
            basis (Basis with rank = 2 or None): the fitting functions.  When ``None``
                (default), ``symmetric_matrix_basis(d)`` is used, spanning all constant
                symmetric ``d×d`` diffusion matrices.  Requires a prior
                ``compute_diffusion_constant()`` call to determine ``d``.

            M_mode (str):
                The method used for local diffusion tensor estimation and moments computation.
                See _diffusion_estimator documentation for additional information.

            G_mode (str):
                The method used to compute the normalization matrix `G`.
                Not investigated extensively yet for diffusion inference.

        Updates:
            self.diffusion_coefficients: The inferred coefficients for the diffusion basis functions.
            self.diffusion_inferred: Callable representing the inferred diffusion tensor field.
            self.diffusion_G: The normalization matrix used in the inference process.

        Note:
            This inferred tensor field is not guaranteed to be nonnegative.
        """
        if basis is None:
            if not hasattr(self, "diffusion_average"):
                raise RuntimeError(
                    "infer_diffusion_linear() with no basis requires a prior "
                    "compute_diffusion_constant() call to determine spatial dimension."
                )
            from SFI.bases import symmetric_matrix_basis
            basis = symmetric_matrix_basis(self.diffusion_average.shape[0])

        if M_mode == "auto":
            if self.Lambda_trace > 0.0:
                M_mode = "noisy"
                G_mode = "rectangle"
            else:
                M_mode = "WeakNoise"
                G_mode = "rectangle"
            logger.info(
                "Auto-selecting diffusion inference: M_mode %s, G_mode %s (Lambda trace: %s)",
                M_mode,
                G_mode,
                self.Lambda_trace,
            )

        self._validate_basis(basis, expected_rank=2, label="diffusion basis")

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

    def compute_entropy_production(self, *, support: str = "current") -> dict:
        r"""Estimate the entropy production of the observed dynamics.

        Projects the phase-space velocity (mean local velocity) of the data
        onto the force basis and contracts it with the inverse average
        diffusion — the estimator of Frishman & Ronceray, Phys. Rev. X 10,
        021009 (2020), reconnected to the v2 moment engine.  Requires a
        prior :meth:`infer_force_linear` fit (any preset); the Stratonovich
        velocity moments are reused from the fit when available and computed
        on demand otherwise.

        .. physics:: Entropy production estimator (overdamped, linear basis)
           :label: entropy-production-inference-overdamped
           :category: Observable

           .. math::

              \Delta\hat S = 2\, M_v^\top G^{-1} M_v,
              \qquad
              M_{v,a} = \sum_t \mathrm{d}X_t \circ A^{-1}\, b_a,
              \qquad A = 2\bar D,

           .. math::

              \hat{\dot S} = \frac{\Delta\hat S}{\tau_N},
              \qquad
              \dot S_{\mathrm{bias}} = \frac{2 N_b}{\tau_N},
              \qquad
              \delta \Delta S = \sqrt{2\,\Delta\hat S + (2 N_b)^2}.

           :math:`\hat v = G^{-1} M_v` is the basis projection of the
           phase-space velocity :math:`v = F - D\nabla\ln\rho`, :math:`N_b`
           the number of basis degrees of freedom, and :math:`\tau_N` the
           total particle-time.  The projection makes :math:`\Delta\hat S` a
           *lower bound* on the entropy produced in the observed phase
           space, up to the :math:`+2N_b` fluctuation bias.  The bias is
           reported both ways: the raw plug-in value keeps it, and
           ``DeltaS_debiased`` :math:`= \Delta\hat S - 2N_b` subtracts it.
           This is exactly the **AIC complexity correction** applied to the
           trajectory information: :math:`\Delta\hat S` is four times the
           plug-in information carried by the projected velocity field, and
           each fitted degree of freedom inflates a maximized (plug-in)
           information by :math:`+\tfrac12` — the same counting that powers
           the ``AIC`` criterion of ``sparsify_force``.

        Parameters
        ----------
        support : {"current", "full"}
            ``"current"`` (default) restricts the projection to the active
            basis support — the full basis before sparsification, the
            selected support after :meth:`sparsify_force` (shrinking the
            :math:`2 N_b/\tau_N` bias).  ``"full"`` always uses the complete
            basis.

        Returns
        -------
        dict
            ``{"Sdot", "Sdot_error", "Sdot_bias", "Sdot_debiased", "DeltaS",
            "error_DeltaS", "DeltaS_debiased", "tauN", "Nb"}`` — the
            per-particle entropy production rate, its estimated error and
            fluctuation bias, the bias-subtracted (AIC-corrected) rate, the
            total entropy :math:`\Delta\hat S = \hat{\dot S}\,\tau_N` with
            error and its debiased value, the total particle-time and the
            basis size.  All plain floats, in units of :math:`k_B` (nats).

        Notes
        -----
        - The estimate assumes constant diffusion: the contraction uses the
          average :math:`\bar D` even when a state-dependent field was
          fitted (a warning is emitted in that case).  Extension point: a
          trajectory integral of :math:`\hat v \cdot D(x)^{-1} \hat v`.
        - ``Sdot_debiased`` can come out negative near equilibrium — that is
          the expected fluctuation of an unbiased estimate around zero, not
          an inconsistency.  Use the raw ``Sdot`` when a conservative
          plug-in value is wanted, the debiased one for point estimates and
          detection decisions.
        - **Independent measurement noise does not bias the estimator at
          leading order.**  The Stratonovich velocity moments are *odd*
          under time reversal while i.i.d. (signal-independent) localization
          noise is *even*, so the naive :math:`O(\Lambda/\Delta t)`
          noise–increment correlation cancels exactly — per step it
          telescopes to a boundary term.  The surviving effects are second
          order and *not* included in the error bar: an :math:`O(\Lambda)`
          basis-smoothing distortion (the moments see the
          :math:`\Lambda`-blurred basis functions) and a noise-inflated
          variance that raises the effective fluctuation bias.  A warning is
          emitted when the estimated noise dominates the per-step diffusive
          displacement (:math:`\mathrm{Tr}\,\Lambda \gtrsim 50\%` of
          :math:`\mathrm{Tr}\,\bar D\,\Delta t`).  Noise *correlated with
          the motion* — motion blur, exposure-time averaging — breaks the
          parity cancellation and does bias the estimate: correct or
          deconvolve it first (see :doc:`/inference/noise_and_sampling`).
        - Parametric fits (:meth:`infer_force`) are not supported yet.  The
          natural extension is the plug-in Stratonovich sum
          :math:`\sum_t \mathrm{d}X_t \circ \bar D^{-1}\,
          \tfrac12[\hat F(X_t)+\hat F(X_{t+1})]`.
        - :math:`\tau_N` is the particle-time measure of the force Gram's
          own rows, dataset weights included, so the rate is exact for
          masked and weighted multi-dataset collections alike.
        """
        if self.metadata.get("force_method") != "linear":
            raise RuntimeError(
                "compute_entropy_production() requires a prior infer_force_linear() "
                "fit; parametric fits (infer_force) are not supported yet."
            )
        lam = float(getattr(self, "Lambda_trace", 0.0))
        if lam > 0.0:
            # The leading O(Λ/Δt) noise bias cancels by time-reversal parity
            # for signal-independent noise (the Stratonovich moments are odd,
            # i.i.d. noise is even); what survives is second order.  Warn only
            # when noise dominates the per-step diffusive displacement.
            dt_val = float(self.data.peek_row(require={"dt"})["dt"])
            diffusive_scale = float(jnp.trace(self.diffusion_average)) * dt_val
            if lam > 0.5 * diffusive_scale:
                logger.warning(
                    "Strong measurement noise (Tr Lambda = %.3g, %.0f%% of "
                    "Tr(D)*dt).  Independent noise does not bias the entropy "
                    "estimate at leading order (parity cancellation), but "
                    "basis smoothing and noise-inflated variance (second "
                    "order) are NOT in the error bar — and noise correlated "
                    "with the motion (blur) does not cancel.",
                    lam, 100.0 * lam / max(diffusive_scale, 1e-300),
                )
        if self._diffusion_looks_state_dependent():
            logger.warning(
                "A state-dependent diffusion field was inferred; the entropy "
                "estimator contracts with the constant average diffusion and is "
                "only exact for constant D."
            )

        # Stratonovich v-moments: by-product of a Strato fit, else on demand —
        # on the fit's rows either way, as G and tau_N (the v-moment stencil,
        # frames t and t+1, is inside every linear force fit's: its moments
        # read the increment).  Exception: periodic pair features fix their
        # image at a frame outside the increment (``Integrand.pbc_reference_in``);
        # when the fit's rows read none, the v-moments and the Gram are
        # recomputed on the enlarged row set.
        frames = self._force_row_frames
        v_frames = required_frames(self._strato_v_moments(program_only=True))
        widen = not v_frames <= frames
        if widen:
            frames = frames | v_frames
        if widen or not hasattr(self, "force_v_moments"):
            with self._structural_scope(self.force_basis), shared_rows(frames):
                self.force_v_moments = self._strato_v_moments()
        # tau_N: the particle-time measure of the Gram's own rows.
        G_full, G_pinv_current, tauN = self.force_G_full, self.force_G_pinv, self._force_row_measure
        if widen:
            with self._structural_scope(self.force_basis), shared_rows(frames):
                G_full, tauN = self.__G_matrix__(self.force_basis, self.force_basis @ self.A_inv,
                                                 self.__force_G_mode__, "ima,imb->iab", return_measure=True)
            S = self.force_support
            G_pinv_current = stable_pinv(G_full[jnp.ix_(S, S)])

        if support == "current":
            S = self.force_support
            G_pinv = G_pinv_current
        elif support == "full":
            S = jnp.arange(self.force_v_moments.shape[0])
            G_pinv = stable_pinv(G_full)
        else:
            raise ValueError(f"support must be 'current' or 'full', got {support!r}")

        Mv = self.force_v_moments[S]
        Nb = int(S.shape[0])
        # A carries one factor of A_inv in both M_v and G, so the A-scaling
        # cancels in G^{-1} M_v; the leading 2 restores D̄^{-1} = 2 A^{-1}.
        DeltaS = float(2.0 * (Mv @ (G_pinv @ Mv))) if Nb > 0 else 0.0

        self.DeltaS = DeltaS
        self.error_DeltaS = float((2.0 * DeltaS + (2.0 * Nb) ** 2) ** 0.5)
        self.DeltaS_debiased = DeltaS - 2.0 * Nb
        self.Sdot = DeltaS / tauN
        self.Sdot_bias = 2.0 * Nb / tauN
        self.Sdot_debiased = self.DeltaS_debiased / tauN
        self.Sdot_error = self.error_DeltaS / tauN
        self.metadata["entropy_production"] = {
            "support": support,
            "Nb": Nb,
            "tauN": tauN,
        }
        logger.info(
            "Entropy production: DeltaS = %.4g +- %.4g "
            "(rate %.4g +- %.4g, fluctuation bias %.4g, debiased %.4g)",
            DeltaS, self.error_DeltaS, self.Sdot, self.Sdot_error,
            self.Sdot_bias, self.Sdot_debiased,
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
            "Nb": Nb,
        }

    def simulate_bootstrapped_trajectory(self, key, oversampling=1, simulate=True, dataset=0):
        """
        Simulate an overdamped Langevin trajectory with the inferred force and diffusion fields.

        This function generates a trajectory using the inferred force field and diffusion tensor inferred
        from the input data, matching the original time series and initial conditions.

        Args:
            key: JAX random key for generating noise in the simulation.
            oversampling (int, optional): Factor for oversampling (i.e. number of intermediate simulated
                points between two recorded points). Defaults to 1.
            simulate: if True, performs the simulation with the first data point as initial position;
                      if False, returns an uninitialized object which can be simulated with flexible
                      initial position and parameters.
            dataset (int, optional): Which experiment of a pooled fit to reproduce. The inferred
                model is collapsed to this condition via
                :meth:`~SFI.statefunc.StateExpr.specialize` (folding ``per_dataset_scalar`` /
                ``dataset_indicator`` at ``dataset``); the resulting process is a standalone
                single-condition model that does not read ``dataset_index``. Defaults to 0.

        Returns:
            OverdampedProcess: Simulated Langevin process object.
        """
        self._check_noise_consistency()
        from SFI.langevin import OverdampedProcess

        # Collapse a (possibly pooled) fit to the chosen experiment: per-dataset
        # parameters are folded at ``dataset`` and the reserved ``dataset_index``
        # disappears, so the simulated model is standalone (no dataset concept).
        force_k = self.force_inferred.specialize(dataset=int(dataset))
        diff_k = self.diffusion_inferred.specialize(dataset=int(dataset))
        self._check_diffusion_psd(diff_k, dataset=int(dataset))
        bootstrapped_process = OverdampedProcess(force_k._psf, diff_k._psf)
        bootstrapped_process.set_params(theta_F=force_k.params, theta_D=diff_k.params)
        ds = self.data.datasets[int(dataset)]
        bootstrapped_process.set_extras(extras_global=ds.extras_global, extras_local=ds.extras_local)
        # Only the particle count is framework-supplied; the specialized
        # force does not read ``dataset_index``.
        bootstrapped_process._reserved_overrides = {
            "particle_index": jnp.arange(int(ds.N)),
        }

        if simulate:
            from SFI.trajectory import TrajectoryCollection

            start_config = TrajectoryCollection.from_dataset(ds).peek_row(require={"X", "dt"})
            x0 = jnp.asarray(start_config["X"])
            if not jnp.all(jnp.isfinite(x0)):
                x0 = self._find_finite_x0()
            bootstrapped_process.initialize(x0)
            data_bootstrap = bootstrapped_process.simulate(
                dt=start_config["dt"],
                Nsteps=ds.T,
                key=key,
                prerun=0,
                oversampling=oversampling,
            )
            return data_bootstrap, bootstrapped_process
        return bootstrapped_process

    # ── Parametric (exact-core) force inference ──

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
        extra_radius: int = 1,
    ) -> None:
        r"""Infer the force field with the exact parametric estimator.

        Built on ``SFI.inference.parametric_core``: an RK4 flow step per
        observation interval defines the residual, the block-tridiagonal
        residual covariance is whitened *exactly* by a reverse-time
        block-LDLᵀ innovations recursion (no window truncation, no tuning
        knobs), and the parameters are found by direct Gauss–Newton
        (linear-in-θ ``Basis``, with the skip-trick errors-in-variables
        instrument) or L-BFGS (nonlinear-in-θ ``PSF``).  ``(D, Λ)`` are
        profiled natively: moment-estimator init, then an exact banded-NLL
        refinement at the fitted θ on cached fixed-θ tensors (zero basis
        evaluations per profile iteration).  Non-uniform sampling is
        supported end-to-end: datasets carrying an absolute time vector
        ``t`` or a per-step ``dt`` array fit directly, with per-interval
        flow steps and process covariances.

        .. physics:: Parametric force inference (overdamped)
           :label: parametric-force-overdamped
           :category: Inference

           The observed positions follow
           :math:`y_t = x_t + \eta_t` where
           :math:`\eta \sim \mathcal{N}(0, \Lambda)`.  The
           deterministic flow
           :math:`\Phi(x;\theta) = z(\Delta t) - x`
           (one RK4 step by default) defines the residual
           :math:`r_t = y_{t+1} - y_t - \Phi(y_t;\theta)`.
           Residuals follow a block-tridiagonal Gaussian that is
           whitened exactly (reverse-time innovations), and the
           whitened sensitivities enter the Gauss–Newton normal
           equations; under measurement noise the left factor is
           replaced by the η-clean *skip* instrument
           :math:`\psi_{\rm inst} = \partial\Phi/\partial\theta`
           evaluated at the lagged clean point (``eiv=True``),
           giving a consistent estimating equation.

        Parameters
        ----------
        F : PSF or Basis
            Parametric drift model.  A ``Basis`` is converted to a PSF
            internally (coefficients initialised to zero) and PASTIS
            sparsification is enabled; it runs the fast direct-GN path.
            A ``PSF`` (possibly nonlinear in θ) runs the L-BFGS path.
        theta0 : dict, array, or None
            Initial drift parameters (default: zeros).
        D : array (d, d), optional
            Fixed diffusion matrix.  If both ``D`` and ``Lambda``
            are given, noise profiling is skipped entirely (fast path).
        Lambda : array (d, d), optional
            Fixed measurement-noise covariance.
        integrator : {"rk4", "euler"}
            Flow predictor (default ``"rk4"``, a single 4th-order step).
        n_substeps : int
            Integrator micro-steps per observation interval (default 1 —
            the single-step minimal estimator).  At coarse effective
            sampling (fast dynamics per interval) raise to 2 — the RK4
            flow model saturates at ``n_substeps=2`` on typical chaotic
            benchmarks.  With per-step ``dt`` the substep is
            ``h_k = dt_k / n_substeps``.
        inner : {"auto", "gn", "lbfgs"}
            Inner solver.  ``"auto"`` → Gauss–Newton (direct for a
            linear ``Basis``, damped for a nonlinear ``PSF`` — the
            instrument stays active); ``"lbfgs"`` forces the
            frozen-precision quadratic route.
        eiv : {"auto", True, False, float}
            Measurement-noise errors-in-variables instrument.  ``"auto"``
            (default) → ``True`` for all models (interacting models use
            the same N-body flow for the instrument as for the residual);
            ``True`` forces the η-clean skip instrument (consistent under
            noise); ``False`` is the plain MLE; a float in ``[0, 1]``
            blends.  ``"auto"`` does *not* inspect the noise level: for
            (near-)clean data at coarse effective sampling pass
            ``eiv=False`` — the ``(D, Λ)`` profile cannot separate a
            Δt-independent Λ from the process increment structure there
            and can fit a spurious measurement noise (see
            :ref:`parametric-algorithm`).
        max_outer : int
            Outer Gauss–Newton / IRLS iterations (default 5).
        inner_maxiter : int
            Inner L-BFGS iterations per outer step on the PSF path
            (default 80; raise for large nonlinear families, e.g. NNs).
        extra_radius : int
            Deprecated, no effect (a ``DeprecationWarning`` fires when
            a non-default value is passed).

        Updates
        -------
        Sets standard ``force_*`` attributes:
        ``force_inferred``, ``force_psf``, ``force_G``,
        ``force_G_pinv``, ``force_coefficients_full``,
        ``force_coefficients``, ``force_support``,
        ``force_moments``.

        Also sets ``diffusion_average``, ``A``, ``A_inv``, ``Lambda``
        from the profiled ``(D, Λ)``.

        When ``F`` is a ``Basis``, additionally sets
        ``force_basis``, ``force_G_full``, and ``force_scorer``
        so that ``sparsify_force()`` can be called afterwards.

        See Also
        --------
        :ref:`parametric-concept` : Mathematical foundations.
        :ref:`parametric-algorithm` : Detailed algorithm description.
        """
        import time as _time

        from SFI.inference.parametric_core.solve import _as_psf, solve_force_od

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

        with self._structural_scope(F_psf):
            import numpy as _np

            from SFI.inference.parametric_core.solve import _dt_steps

            # representative Δt for logging and the reported β — the median
            # step (identical to the single step at uniform sampling)
            dt_val = float(_np.median(_np.concatenate(
                [_dt_steps(ds) for ds in self.data.datasets])))

            logger.info(
                "[infer_force] OD minimal parametric solve (n_params=%d, dt=%.4g, %s·n%d, basis_mode=%s)...",
                int(F_psf.template.size), dt_val, integrator, n_substeps, basis_mode,
            )
            t0 = _time.perf_counter()

            res = solve_force_od(
                self.data, F, theta0=theta0, D=D, Lambda=Lambda,
                integrator=integrator, n_substeps=n_substeps,
                inner=inner, eiv=eiv, max_outer=max_outer,
                inner_maxiter=inner_maxiter, extra_radius=extra_radius,
                chunk_target_bytes=self._chunk_target_bytes,
            )
            t_elapsed = _time.perf_counter() - t0

        # ── Store results (standard force_* attributes) ──
        theta_flat = res.theta
        G = res.G
        self._warn_if_null_parametric_force(theta_flat, res.info, label="infer_force")

        self.force_psf = F_psf
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

        self.diffusion_average = res.D
        # Callable constant-D field, so the full result surface (incl.
        # simulate_bootstrapped_trajectory) works without a separate
        # compute_diffusion_constant() call; a later infer_diffusion()
        # overwrites it with the state-dependent fit.
        self.diffusion_inferred = constant_array(res.D)
        self.A = 2.0 * res.D
        self.A_inv = jnp.linalg.inv(self.A)
        self.Lambda = res.Lambda
        self.Lambda_trace = float(jnp.trace(res.Lambda))

        beta = float(jnp.trace(res.Lambda) / (jnp.trace(res.D) * dt_val + 1e-30))

        sf_F = SF(F_psf, F_psf.unflatten_params(theta_flat))
        meta_F = dict(
            kind="force",
            inference="parametric",
            n_params=int(F_psf.template.size),
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
            "[infer_force] Done in %.1f s.  Λ=%s, D=%s, β=%.3f",
            t_elapsed,
            jnp.diag(res.Lambda),
            jnp.diag(res.D),
            beta,
        )

    # ── State-dependent diffusion from parametric residuals ──

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
        r"""Infer state-dependent diffusion D(x) from parametric residuals.

        Requires a prior parametric :meth:`infer_force` call.  Holds the
        fitted force fixed and minimises the exact banded NLL over the
        diffusion parameters (the log-det term makes the diffusion level
        identifiable) on cached fixed-θ tensors: the flow residuals and
        stage Jacobians are computed once at :math:`\hat\theta`, so each
        L-BFGS iteration costs linear algebra only — zero force or basis
        evaluations.  Non-uniform sampling (``t`` vector or per-step
        ``dt``) is supported.

        .. physics:: State-dependent diffusion inference (overdamped)
           :label: parametric-diffusion-overdamped
           :category: Inference

           With the force :math:`\hat F` held fixed, the state-dependent
           diffusion :math:`D(x;\theta_D)` is optimised by minimising the
           exact banded negative log-likelihood; :math:`\Lambda`
           from the force inference is held fixed.  A rank-2 basis gives
           :math:`D(x) = \sum_j (\theta_D)_j\, d_j(x)`; a PSF is evaluated
           directly.

        Parameters
        ----------
        basis : Basis (rank 2), PSF, or None
            Diffusion model.  A rank-2 ``Basis`` gives the linear
            parameterisation; a ``PSF`` is used directly
            (``D(x) = PSF(x; θ_D)``).  When ``None`` (default),
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
        from SFI.inference.parametric_core.solve import _as_psf, solve_diffusion_od

        if not hasattr(self, "force_inferred"):
            raise RuntimeError("infer_diffusion() requires a prior infer_force() call.")
        if self.metadata.get("force_method") not in ("parametric", "parametric_core"):
            raise RuntimeError(
                "infer_diffusion() requires parametric force inference (call infer_force(), not infer_force_linear)."
            )

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

        res = solve_diffusion_od(
            self.data, self.force_psf, self.force_coefficients_full, D_psf,
            Lambda=Lambda, theta_D0=theta_D0,
            n_substeps=n_substeps, integrator=integrator, maxiter=maxiter,
            chunk_target_bytes=self._chunk_target_bytes, support=_support,
        )
        spec = dict(basis=basis, integrator=integrator, n_substeps=n_substeps, maxiter=maxiter)
        self._store_parametric_diffusion(res, D_psf, spec, support=_support)


    #################################################################
    ################       BACKEND       ############################
    #################################################################

    # Hooks required by BaseLangevinInference:
    def _force_G_matrix(self, *, program_only: bool = False, return_measure: bool = False) -> jnp.ndarray:
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

    def _strato_v_moments(self, *, A_inv=None, program_only: bool = False) -> jnp.ndarray:
        r"""Stratonovich (trapezoid) phase-space velocity moments.

        Extensive, dt-weighted, mask-aware sums

        .. math::

           M_{v,a} = \sum_t \mathrm{d}X_t \cdot A^{-1} \cdot
                     \tfrac12\bigl[b_a(X_t) + b_a(X_{t+1})\bigr]

        — the by-product of Stratonovich force fitting and the raw material
        of :meth:`compute_entropy_production`.

        Parameters
        ----------
        A_inv : array (d, d), optional
            Contraction matrix; defaults to ``self.A_inv``.  Pass the
            identity to obtain the plain Stratonovich current projection
            :math:`\langle b_a \circ \mathrm{d}X_t\rangle`.
        """
        if A_inv is None:
            A_inv = self.A_inv
        A = ConstOperand(jnp.asarray(A_inv), alias="A")
        V = TimeOperand(velocity("dX", "dt"), alias="V")
        B0 = ExprOperand(expr=self.force_basis, x=stream("X"), alias="B0")
        Bp = ExprOperand(expr=self.force_basis, x=stream("X_plus"), alias="Bp")
        prog_B0 = Integrand(
            exprs=[B0],
            times=[V],
            consts=[A],
            terms=[Term(eq="im,mn,ina->ia", ops=("V", "A", "B0"))],
        )
        prog_Bp = Integrand(
            exprs=[Bp],
            times=[V],
            consts=[A],
            terms=[Term(eq="im,mn,ina->ia", ops=("V", "A", "Bp"))],
        )
        prog_strato_vb = 0.5 * (prog_B0 + prog_Bp)
        if program_only:
            return prog_strato_vb
        return integrate(
            self.data, prog_strato_vb, reduce="sum", chunk_target_bytes=self._chunk_target_bytes
        )

    def _diffusion_looks_state_dependent(self) -> bool:
        """Best-effort probe: does ``diffusion_inferred`` vary with x?

        Used by :meth:`compute_entropy_production` to warn when a
        state-dependent diffusion field was fitted (the estimator contracts
        with the constant average).  Conservative: unprobeable fields count
        as state-dependent.
        """
        D_sf = getattr(self, "diffusion_inferred", None)
        if D_sf is None:
            return False
        try:
            x0 = jnp.asarray(self.data.datasets[0].X[0][:1])  # (1, d)
            x1 = x0 + 1.0 + 0.5 * jnp.abs(x0)
            D_a = jnp.asarray(D_sf(x0))
            D_b = jnp.asarray(D_sf(x1))
            scale = 1.0 + float(jnp.max(jnp.abs(D_a)))
            return bool(float(jnp.max(jnp.abs(D_a - D_b))) > 1e-6 * scale)
        except Exception:  # pragma: no cover — conservative fallback
            return True

    def _force_moments(self, *, mode=None, A_inv=None, D_op=None, program_only: bool = False):
        r"""
        Compute force moments ⟨ v · A_inv · b ⟩ with Ito / Ito-shift / Strato flavors.

        .. physics:: Overdamped force moments (linear regression)
           :label: force-moments-overdamped
           :category: Inference

           **Itô moments:**

           .. math::

              M_a = \bigl\langle v_t \cdot A^{-1} \cdot b_a(X_t) \bigr\rangle

           **Stratonovich moments** (trapezoid + gradient correction):

           .. math::

              M_a^{\text{S}} = \tfrac{1}{2}\bigl\langle v_t \cdot A^{-1}
              \cdot \bigl[b_a(X_t) + b_a(X_{t+1})\bigr] \bigr\rangle
              \;-\; \bigl\langle (D_{\text{inst}}\,A^{-1})_{kl}\,
              \partial_k (b_a)_l \bigr\rangle

           where :math:`k` is the derivative index and :math:`l` the output
           (force) index of :math:`\nabla_x b_a`.  The matrix acting on the
           Jacobian is :math:`D_{\text{inst}}\,A^{-1}` in this order: contracting
           :math:`D_{\text{inst}}` with the output index instead would give
           :math:`A^{-1} D_{\text{inst}}`, which differs whenever
           :math:`[A^{-1}, D_{\text{inst}}] \neq 0` (anisotropic, non-aligned
           instantaneous diffusion with a non-trivial decorrelating Gram).

           The force coefficients solve :math:`G \cdot C = M`
           where :math:`G_{ab} = \langle b_a \cdot b_b \rangle` (with mode variants).

        Contractions
        ------------
        - RHS (Ito or Ito-shift):
            eq='im,mn,ina->ia' with ops (V, A, B)
            shapes:
              V: (N, m)       from velocity(dX, dt)
              A: (m, n)       constant A_inv
              B: (i, n, a)    basis at X or X_minus, features last

        - Stratonovich v∘b:
            trapezoid average over X and X_plus, same rhs contraction on each, then 0.5*(...+...)

        - Stratonovich gradient correction:
            eq='imn,inoa,om->ia' with ops (D, G, A)
            shapes:
              D: (i, m, n)    instantaneous diffusion (N, d, d) e.g. noisy
              G: (i, n, o, a) basis.d_x()(X) shape (N, d_deriv, d_force, F)
              A: (o, m)       A_inv (d, d)
            The derivative axis n of G is contracted with D and the output
            (force) axis o with A_inv, i.e. the correction is
            (D_inst . A_inv) : grad b.  Contracting D with the output axis
            instead computes A_inv . D_inst, which differs whenever D_inst and
            A_inv do not commute.

        Parameters
        ----------
        mode : {"Ito", "Ito-shift", "Strato"}, optional
            Moment convention; defaults to ``self.__force_M_mode__``.
        A_inv : array (d, d), optional
            Contraction matrix; defaults to ``self.A_inv`` (raises when the
            diffusion has not been computed).  Pass the identity for the
            un-normalized drift projections :math:`\langle b_a \cdot v_t\rangle`.
        D_op : TimeOperand, optional
            Instantaneous-diffusion operand of the Stratonovich gradient
            correction; defaults to the ``"noisy"`` stencil.

        The cached by-products ``force_v_moments`` and
        ``_force_D_grad_b_average`` are written only on the default call
        (no overrides), so that :meth:`compute_entropy_production` and
        :meth:`infer_force_linear` see exactly what they used to.

        Masking
        -------
        Mask is applied by the integrator on the leading particle axis i
        (`mask_out`); state-expression calls receive the static `mask_state`
        (positions known), so dynamically masked particles stay visible. The leaf
        fill policy (`zerostop` by default) replaces masked entries with 0
        via `jnp.where`, which naturally gives zero tangents for masked
        entries without blocking the Jacobian for active entries.
        """
        default_call = mode is None and A_inv is None and D_op is None and not program_only
        if A_inv is None:
            if not hasattr(self, "A_inv"):
                raise RuntimeError("A_inv not available. Compute diffusion first.")
            A_inv = self.A_inv
        A_inv = jnp.asarray(A_inv)

        # Common pieces
        A = ConstOperand(A_inv, alias="A")
        V = TimeOperand(velocity("dX", "dt"), alias="V")

        if mode is None:
            mode = getattr(self, "__force_M_mode__", "Ito")
        if mode not in ("Ito", "Ito-shift", "Strato"):
            raise KeyError(f"Unknown __force_M_mode__: {mode}")

        if mode in ("Ito", "Ito-shift"):
            x_key = "X_minus" if mode == "Ito-shift" else "X"
            B = ExprOperand(expr=self.force_basis, x=stream(x_key), alias="B")
            prog = Integrand(
                exprs=[B],
                times=[V],
                consts=[A],
                terms=[Term(eq="im,mn,ina->ia", ops=("V", "A", "B"))],
            )
            if program_only:
                return prog
            logger.debug(
                "Computing Ito-shift force coefficients."
                if mode == "Ito-shift"
                else "Computing Ito force coefficients."
            )
            return integrate(self.data, prog, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)

        # Stratonovich
        if program_only:
            return [self._strato_v_moments(A_inv=A_inv, program_only=True),
                    self._strato_grad_moments(A_inv=A_inv, D_op=D_op, program_only=True)]
        logger.debug("Computing Strato force coefficients.")

        # v ∘ b via trapezoid on basis
        v_moments = self._strato_v_moments(A_inv=A_inv)

        logger.debug("Computing Strato gradient term.")
        grad_term = self._strato_grad_moments(A_inv=A_inv, D_op=D_op)

        if default_call:
            self.force_v_moments = v_moments
            self._force_D_grad_b_average = grad_term
        return v_moments - grad_term

    def _strato_grad_moments(self, *, A_inv=None, D_op=None, program_only: bool = False) -> jnp.ndarray:
        r"""Gradient (Itô→Stratonovich conversion) term of the Strato moments.

        Extensive, dt-weighted, mask-aware sum

        .. math::

           \sum_t (D_{\text{inst}}(t)\,A^{-1})_{kl}\,\partial_k (b_a)_l(X_t)

        with :math:`D_{\text{inst}}` the ``"noisy"`` stencil by default (any
        symmetric instantaneous-diffusion ``TimeOperand`` via ``D_op``) and
        ``A_inv`` defaulting to ``self.A_inv``.  With ``A_inv`` the identity
        this is :math:`\langle \nabla b_a : D_t\rangle`, the
        quadratic-variation estimate of the Itô–Stratonovich difference.
        """
        if A_inv is None:
            A_inv = self.A_inv
        A = ConstOperand(jnp.asarray(A_inv), alias="A")

        _interacting = getattr(self.force_basis, "particles_input", False)
        cross = getattr(self, "__force_cross_particle__", "all-pairs")
        stencil = getattr(self, "__force_correction_stencil__", "noisy")
        # The force fit's own correction (custom D_op: same-particle blocks below).
        # Its metadata and warning are recorded on evaluation only: program_only
        # (row-scope planning) has no side effects.
        if D_op is None and not program_only:
            self.metadata["force_noise_correction"] = ("all-pairs" if _interacting and cross == "all-pairs"
                                                       else "same-particle")
            self.metadata["force_correction_stencil"] = stencil
        if D_op is None and ((_interacting and cross == "all-pairs") or stencil != "noisy"):
            # Every particle pair: sum_j D_t,ij : A^-1 d b_i / d x_j, with D_t,ij the same
            # stencil on i != j, evaluated as one forward-mode derivative of the basis along
            # each increment direction of the stencil (no Jacobian block, no diffusion model).
            # For a non-interacting basis only j = i contributes: the per-particle estimator.
            from SFI.inference.graph_diffusion import stencil_correction_program, stencil_directions

            prog_grad = stencil_correction_program(self.force_basis, A_inv, stencil_directions(stencil, "od"),
                                                   x=stream("X"), wrt="x")
        else:
            # Same-particle Jacobian block: d_x(same_particle=True) for interacting bases,
            # (N, d, d, F) = (i, o, m, a).  Exact only when the noise is independent between
            # particles (D_ij = 0 for i != j).
            if _interacting and not program_only:
                self._check_same_site_correction(self.force_basis, label="Stratonovich force correction")
            G = ExprOperand(
                expr=self.force_basis.d_x(same_particle=_interacting),
                x=stream("X"),
                alias="G",
            )
            D = self.get_diffusion_timeop("noisy") if D_op is None else D_op

            # Contraction correctness relies on D_inst being SYMMETRIC: the einsum
            # pairs D_inst with the derivative axis (n) of the Jacobian and A_inv
            # with the output axis (o), giving (D_inst . A_inv) : grad b.  This is
            # the intended order only because D ("noisy") is symmetric; a future
            # non-symmetric instantaneous-D estimator would need the index roles of
            # D and A_inv swapped (and would differ whenever [A_inv, D] != 0).
            prog_grad = Integrand(
                exprs=[G],
                times=[D],
                consts=[A],
                terms=[Term(eq="imn,inoa,om->ia", ops=(D.alias, "G", "A"))],
            )
        if program_only:
            return prog_grad
        return integrate(
            self.data, prog_grad, reduce="sum", chunk_target_bytes=self._chunk_target_bytes
        )

    def _diffusion_G_matrix(self, *, program_only: bool = False, return_measure: bool = False) -> jnp.ndarray:
        # Diffusion is a per-point quantity: weight the projection per-point
        # (weight_by_dt=False), consistently with _diffusion_moments.
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
        """Diffusion basis at the point its local estimator is attached to."""
        return ExprOperand(expr=self.diffusion_basis, x=stream("X"), alias="B")

    def _diffusion_moments(self, *, program_only: bool = False) -> jnp.ndarray:
        logger.debug("Computing diffusion linear moments.")
        D_op = self.get_diffusion_timeop(self.__diffusion_M_mode__)
        B = self._diffusion_basis_operand()
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
        Materialize the fitted force as an SF and wrap it with param covariance.
        Produces: self.force_inferred : InferenceResultSF
        """
        if hasattr(self, "force_basis"):
            # Basis -> PSF is the supported path; tests already use .to_psf()
            P = self.force_basis.to_psf()
            theta = {"coeff": jnp.asarray(self.force_coefficients_full)}
        elif hasattr(self, "force_psf"):
            P = self.force_psf
            theta = self.force_params_nonlinear

        sf = SF(P, theta)  # fixed-θ state function

        meta = dict(
            kind="force",
            modes=dict(
                M=getattr(self, "__force_M_mode__", None),
                G=getattr(self, "__force_G_mode__", None),
            ),
            A_inv=jnp.asarray(getattr(self, "A_inv", None)) if hasattr(self, "A_inv") else None,
            basis_features=int(getattr(self.force_basis, "n_features", 0)),
            basis_labels=getattr(self.force_basis, "labels", None),
        )
        # New coefficients: their covariance is set by compute_force_error
        # (sparsify_force refreshes it), never inherited from previous ones.
        self.force_inferred = InferenceResultSF(sf, param_cov=None, meta=meta)

    def _update_diffusion_inferred(self) -> None:
        """
        Materialize the fitted diffusion tensor as an SF and wrap it.
        Produces: self.diffusion_inferred : InferenceResultSF
        """
        if self.diffusion_basis is None:
            raise RuntimeError("_update_diffusion_inferred called before diffusion was fitted.")
        P: "PSF" = self.diffusion_basis.to_psf()  # rank-2 PSF
        theta = {"coeff": jnp.asarray(self.diffusion_coefficients_full)}
        sf = SF(P, theta)  # callable (x[, mask/extras]) -> (i, m, n)

        meta = dict(
            kind="diffusion",
            mode=getattr(self, "__diffusion_M_mode__", None),
            A_inv=jnp.asarray(getattr(self, "A_inv", None)) if hasattr(self, "A_inv") else None,
            basis_features=int(getattr(self.diffusion_basis, "n_features", 0)),
            basis_labels=getattr(self.diffusion_basis, "labels", None),
        )
        # New coefficients: their covariance is set by compute_diffusion_error
        # (sparsify_diffusion refreshes it), never inherited from previous ones.
        self.diffusion_inferred = InferenceResultSF(sf, param_cov=None, meta=meta)

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
        Compute Gram/normalization matrix G = < b_left ⊗ b_right > with chosen mode.

        Arguments
        ---------
        b_left, b_right : stateexpr callables
            Each must follow the contract expr(x, v=..., mask=..., extras=..., params=...),
            producing arrays with features on the last axis. Particle axis is leading i.
        G_mode : {'rectangle','trapeze','shift'}
            rectangle: b_left(X_t)  ⊗ b_right(X_t)
            trapeze:   b_left(X_t)  ⊗ 0.5 [ b_right(X_t) + b_right(X_{t+}) ]
            shift:     b_left(X_t)  ⊗ b_right(X_{t+})
        einsum_string : str
            Einstein string including the particle index in inputs (e.g. 'iam,ibn->iabmn').
        subsampling : int
            Use every `subsampling`-th time row.

        Returns
        -------
        jnp.ndarray
            Time-averaged Gram matrix with particle axis reduced by the integrator.
        """
        logger.debug("Computing G matrix with einsum: %s", einsum_string)

        BL = ExprOperand(expr=b_left, x=stream("X"), alias="BL")
        BR0 = ExprOperand(expr=b_right, x=stream("X"), alias="BR0")
        BRp = ExprOperand(expr=b_right, x=stream("X_plus"), alias="BRP")

        if G_mode == "rectangle":
            prog = Integrand(exprs=[BL, BR0], terms=[Term(eq=einsum_string, ops=("BL", "BR0"))])
        elif G_mode == "trapeze":
            rect = Integrand(exprs=[BL, BR0], terms=[Term(eq=einsum_string, ops=("BL", "BR0"))])
            shift = Integrand(exprs=[BL, BRp], terms=[Term(eq=einsum_string, ops=("BL", "BRP"))])
            prog = 0.5 * (rect + shift)
        elif G_mode == "shift":
            prog = Integrand(exprs=[BL, BRp], terms=[Term(eq=einsum_string, ops=("BL", "BRP"))])
        else:
            raise KeyError("Wrong G_mode argument")
        if program_only:
            return prog

        # Mask-aware reduction over particles; Teff mean over time
        return integrate(
            self.data,
            prog,
            reduce="sum",
            reduce_over_particles=True,
            subsampling=subsampling,
            weight_by_dt=weight_by_dt,
            chunk_target_bytes=self._chunk_target_bytes,
            return_measure=return_measure,
        )

    def _build_diffusion_timeoperands(self):
        """
        Construct and cache TimeOperands for diffusion estimators.
        Safe to call multiple times; idempotent.
        """
        if hasattr(self, "_diff_ops"):
            return
        self._diff_ops = {
            "MSD": TimeOperand(_D_msd, alias="D_msd"),
            "noisy": TimeOperand(_D_noisy, alias="D_noisy"),
            "WeakNoise": TimeOperand(_D_weaknoise, alias="D_weaknoise"),
            "Lambda": TimeOperand(_Lambda, alias="Lambda"),
        }

    def get_diffusion_timeop(self, method: str) -> TimeOperand:
        """
        Return the requested overdamped diffusion estimator as a TimeOperand.
        Output is (N, d, d). Required streams are declared on the wrapped TimeOp.
        """
        self._build_diffusion_timeoperands()
        try:
            return self._diff_ops[method]  # type: ignore[attr-defined]
        except KeyError as e:
            raise KeyError(f"Unknown diffusion estimator method: {method}") from e


# -------------------- Overdamped diffusion estimators as TimeOps --------------------


@timeop(name="D_msd", batch_safe=True)
def _D_msd(**streams):
    r"""
    MSD estimator (per particle): 0.5 * dX ⊗ (dX / dt)
    Returns (..., N, d, d) — batch-safe.

    .. physics:: MSD diffusion estimator (overdamped)
       :label: D-msd-overdamped
       :category: Estimator

       .. math::

          \hat D_{\text{MSD}}(t)
          = \tfrac{1}{2}\,\mathrm{d}X_t \otimes
            \frac{\mathrm{d}X_t}{\mathrm{d}t}

       Simplest estimator; biased by measurement noise.
    """
    dX = streams["dX"]
    dt = streams["dt"]
    while dt.ndim < dX.ndim:
        dt = dt[..., jnp.newaxis]
    v = dX / dt
    return 0.5 * jnp.einsum("...m,...n->...mn", dX, v)


_D_msd._requires = frozenset({"dX", "dt"})  # type: ignore[attr-defined]


@timeop(name="D_noisy", batch_safe=True)
def _D_noisy(**streams):
    r"""
    Noisy diffusion estimator — Vestergaard–Blainey–Flyvbjerg (per particle),
    attached to the first point of its stencil:
      1/4 [ dX⊗(dX/dt) + 2 dX⊗(dX^+/dt) + 2 dX^+⊗(dX/dt) + dX^+⊗(dX^+/dt) ].
    Returns (N, d, d).

    .. physics:: Noisy (Vestergaard–Blainey–Flyvbjerg) diffusion estimator
       :label: D-noisy
       :category: Estimator

       .. math::

          \hat D_{\text{noisy}}(t) = \tfrac{1}{4}\bigl[
              \mathrm{d}X_t \otimes v_t
            + 2\,\mathrm{d}X_t \otimes v_{t+1}
            + 2\,\mathrm{d}X_{t+1} \otimes v_t
            + \mathrm{d}X_{t+1} \otimes v_{t+1}
          \bigr]

       Two-point estimator robust to measurement noise
       (CL Vestergaard, PC Blainey, H Flyvbjerg - Physical Review E, 2014).

       The stencil is attached to its *first* point :math:`X_t` (increments
       :math:`\mathrm{d}X_t, \mathrm{d}X_{t+1}`).  Conditioned on
       :math:`X_t` both increments lie in the future, so the estimate is
       unbiased *state-resolved*; the shared-point attachment
       (:math:`\mathrm{d}X_{t-1}, \mathrm{d}X_t` at :math:`X_t`) is
       unbiased only in the mean, because the backward increment has a
       non-zero conditional mean given its end point.  The forward stencil
       costs the last frame of each trajectory instead of the first.

       Non-uniform sampling: the stencil expectation is
       :math:`2D(\mathrm{d}t + \mathrm{d}t^+)` with the localization terms
       cancelling at any spacing, so the denominator is the two-interval
       mean :math:`(\mathrm{d}t + \mathrm{d}t^+)/2` — identical to
       :math:`\mathrm{d}t` (bitwise) at uniform sampling, exactly unbiased
       under per-step ``dt``.
    """
    dX = streams["dX"]
    dXp = streams["dX_plus"]
    dt = streams["dt"]
    dtp = streams["dt_plus"]
    while dt.ndim < dX.ndim:
        dt = dt[..., jnp.newaxis]
        dtp = dtp[..., jnp.newaxis]
    invdt = 1.0 / (0.5 * (dt + dtp))
    a = jnp.einsum("...m,...n->...mn", dX, dX * invdt)
    b = jnp.einsum("...m,...n->...mn", dX, dXp * invdt)
    c = jnp.einsum("...m,...n->...mn", dXp, dX * invdt)
    d = jnp.einsum("...m,...n->...mn", dXp, dXp * invdt)
    return 0.25 * (a + 2 * b + 2 * c + d)


_D_noisy._requires = frozenset({"dX", "dX_plus", "dt", "dt_plus"})  # type: ignore[attr-defined]


@timeop(name="D_weaknoise", batch_safe=True)
def _D_weaknoise(**streams):
    r"""
    Weak-noise estimator (per particle), attached to the first point of its
    stencil:
      1/4 * ( (dX^+ - dX) ⊗ (dX^+/dt - dX/dt) ).
    Returns (N, d, d).

    .. physics:: Weak-noise diffusion estimator (overdamped)
       :label: D-weaknoise-overdamped
       :category: Estimator

       .. math::

          \hat D_{\text{WN}}(t)
          = \tfrac{1}{4}\bigl(\mathrm{d}X_{t+1} - \mathrm{d}X_t\bigr)
            \otimes \bigl(v_{t+1} - v_t\bigr)

       Uses successive-displacement differences, in which the drift cancels
       to :math:`O(\mathrm{d}t^2)` without a fit; suitable when localization
       noise is negligible.  The stencil is attached to its *first* point
       :math:`X_t`: conditioned on :math:`X_t` both increments lie in the
       future, so the estimate is unbiased state-resolved (the shared-point
       attachment :math:`\mathrm{d}X_t - \mathrm{d}X_{t-1}` is exact only in
       the mean and carries an :math:`O(\mathrm{d}t)` conditioning bias
       where the drift varies).  Costs the last frame of each trajectory
       instead of the first.

       Non-uniform sampling: ``E[(dX⁺ − dX)⊗(dX⁺ − dX)] = 2D(dt + dt⁺)``
       for clean data at any spacing, so the denominator is the
       two-interval mean ``(dt + dt⁺)/2`` — bitwise-identical to ``dt``
       at uniform sampling, exactly unbiased per-step (same
       generalization as the noisy estimator above).
    """
    dX = streams["dX"]
    dXp = streams["dX_plus"]
    dt = streams["dt"]
    dtp = streams["dt_plus"]
    while dt.ndim < dX.ndim:
        dt = dt[..., jnp.newaxis]
        dtp = dtp[..., jnp.newaxis]
    invdt = 1.0 / (0.5 * (dt + dtp))
    ddx = dXp - dX
    dv = dXp * invdt - dX * invdt
    return 0.25 * jnp.einsum("...m,...n->...mn", ddx, dv)


_D_weaknoise._requires = frozenset({"dX", "dX_plus", "dt", "dt_plus"})  # type: ignore[attr-defined]


@timeop(name="Lambda_meas_noise", batch_safe=True)
def _Lambda(**streams):
    r"""
    Measurement-noise cross term (per particle):
      Λ_i = -0.5 [ dX_i ⊗ dX^-_i + dX^-_i ⊗ dX_i ].
    Returns (N, d, d). No 1/dt factor inside.

    .. physics:: Measurement noise estimator (overdamped)
       :label: Lambda-overdamped
       :category: Estimator

       .. math::

          \hat\Lambda_i
          = -\,\tfrac{1}{2}\bigl[
              \mathrm{d}X_i \otimes \mathrm{d}X_{i-1}
            + \mathrm{d}X_{i-1} \otimes \mathrm{d}X_i
          \bigr]

       Estimates localization / measurement noise from anti-correlation
       of successive increments.
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    return -0.5 * (jnp.einsum("...m,...n->...mn", dX, dXm) + jnp.einsum("...m,...n->...mn", dXm, dX))


_Lambda._requires = frozenset({"dX", "dX_minus"})  # type: ignore[attr-defined]


# -------------------- Instantaneous-diffusion operand factories --------------------
# These close over data that is not a stream (a constant matrix, or a fitted
# drift field) and are therefore built per fit rather than registered once.


def _D_const(D):
    """Constant instantaneous diffusion ``D`` broadcast over the particle axis.

    Test/reduction hook: every ``D_inst``-consuming moment collapses to its
    constant-diffusion form when this operand replaces a data stencil.
    """
    D = jnp.asarray(D)

    @timeop(name="D_const", batch_safe=True, requires=frozenset({"dX"}))
    def _op(**streams):
        dX = streams["dX"]
        return jnp.broadcast_to(D, tuple(dX.shape[:-1]) + tuple(D.shape))

    return _op


def _needs_extras(field) -> bool:
    """Whether ``field`` reads extras: per-row ones or structural tables (e.g. CSR neighbour lists)."""
    return bool(tuple(getattr(field, "required_extras", ()) or ())
                or tuple(getattr(field, "structural_extras", ()) or ()))


def _field_requires(field, requires):
    """``requires`` plus the ``extras`` stream when ``field`` reads extras."""
    if _needs_extras(field):
        return frozenset(requires) | {"extras"}
    return frozenset(requires)


def _eval_field(field, x, streams, **kw):
    """``field(x, **kw)`` with the row's extras forwarded when ``field`` reads them."""
    if _needs_extras(field):
        kw["extras"] = streams["extras"]
    return jnp.asarray(field(x, **kw))


def _D_drift_corrected(drift):
    r"""Drift-corrected MSD stencil ``(dX − Φ̂(X⁻)·dt)^⊗2 / (2 dt)``.

    ``drift`` is a fitted Itô drift (an ``SF`` or any callable mapping
    ``(..., N, d) -> (..., N, d)``) evaluated at the *previous* point
    ``X_minus``: evaluating it at ``X_t`` would correlate the correction with
    the localization noise of the increment's own end points.  The raw MSD
    stencil carries the bias ``Φ Φᵀ dt / 2``; this removes it while keeping
    the ``Λ/dt`` offset of localization noise (a constant added diffusion).
    Single-particle (``pdepth = 0``) drifts only: the operand forwards the
    row's extras to the field, not masks.
    """

    @timeop(name="D_drift", batch_safe=True, requires=_field_requires(drift, {"dX", "dt", "X_minus"}))
    def _op(**streams):
        dX = streams["dX"]
        dt = streams["dt"]
        Xm = streams["X_minus"]
        while dt.ndim < dX.ndim:
            dt = dt[..., jnp.newaxis]
        r = dX - _eval_field(drift, Xm, streams) * dt
        return 0.5 * jnp.einsum("...m,...n->...mn", r, r / dt)

    return _op


def _D_noisy_dc(drift):
    r"""Forward Vestergaard stencil with both increments drift-corrected.

    Same stencil as :func:`_D_noisy` with ``dX → dX − Φ̂(X⁻)·dt`` and
    ``dX⁺ → dX⁺ − Φ̂(X⁻)·dt⁺``; the drift is evaluated once, at ``X_minus``
    (see :func:`_D_drift_corrected`).  The raw noisy stencil carries
    ``1.5 Φ Φᵀ dt`` of drift bias — three times the MSD stencil's.
    """

    @timeop(
        name="D_noisy_dc",
        batch_safe=True,
        requires=_field_requires(drift, {"dX", "dX_plus", "dt", "dt_plus", "X_minus"}),
    )
    def _op(**streams):
        dX = streams["dX"]
        dXp = streams["dX_plus"]
        dt = streams["dt"]
        dtp = streams["dt_plus"]
        Xm = streams["X_minus"]
        while dt.ndim < dX.ndim:
            dt = dt[..., jnp.newaxis]
            dtp = dtp[..., jnp.newaxis]
        phi = _eval_field(drift, Xm, streams)
        a_ = dX - phi * dt
        b_ = dXp - phi * dtp
        invdt = 1.0 / (0.5 * (dt + dtp))
        a = jnp.einsum("...m,...n->...mn", a_, a_ * invdt)
        b = jnp.einsum("...m,...n->...mn", a_, b_ * invdt)
        c = jnp.einsum("...m,...n->...mn", b_, a_ * invdt)
        d = jnp.einsum("...m,...n->...mn", b_, b_ * invdt)
        return 0.25 * (a + 2 * b + 2 * c + d)

    return _op
