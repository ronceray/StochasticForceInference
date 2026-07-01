from __future__ import annotations

import logging

import jax.numpy as jnp

from SFI.bases.constants import constant_array
from SFI.inference.base import BaseLangevinInference
from SFI.integrate.api import integrate
from SFI.integrate.integrand import (
    ConstOperand,
    ExprOperand,
    Integrand,
    Term,
    TimeOperand,
)
from SFI.integrate.timeops import stream, timeop, velocity
from SFI.statefunc import PSF, SF, Basis
from SFI.utils.maths import sqrtm_psd

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

        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.force_basis = basis
        self.metadata["force_preset"] = _resolved_preset
        self.metadata["force_M_mode"] = M_mode
        self.metadata["force_G_mode"] = G_mode

        if hasattr(self, "force_G_full"):
            raise RuntimeError("Force has already been inferred on this object - create a new instance to re-infer.")

        # Structural (CSR/stencil) arrays are exposed only for this evaluation and
        # never persisted on the dataset (see ``_structural_scope``).
        with self._structural_scope(self.force_basis):
            self.force_G_full = self._force_G_matrix()
            self.force_moments = self._force_moments()

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

        self.diffusion_basis = basis
        self.__diffusion_M_mode__ = M_mode
        self.__diffusion_G_mode__ = G_mode
        self.metadata["diffusion_M_mode"] = M_mode
        self.metadata["diffusion_G_mode"] = G_mode

        if hasattr(self, "diffusion_moments"):
            raise RuntimeError(
                "Diffusion has already been inferred on this object - create a new instance to re-infer."
            )

        with self._structural_scope(self.diffusion_basis):
            self.diffusion_G_full = self._diffusion_G_matrix()
            self.diffusion_moments = self._diffusion_moments()

        self.diffusion_scorer = SparseScorer(M=self.diffusion_moments, G=self.diffusion_G_full)
        self._update_diffusion_coefficients(self.diffusion_scorer.total_C)

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
        from SFI.langevin import OverdampedProcess

        # Collapse a (possibly pooled) fit to the chosen experiment: per-dataset
        # parameters are folded at ``dataset`` and the reserved ``dataset_index``
        # disappears, so the simulated model is standalone (no dataset concept).
        force_k = self.force_inferred.specialize(dataset=int(dataset))
        diff_k = self.diffusion_inferred.specialize(dataset=int(dataset))
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

    # ── Parametric (windowed) force inference ──

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
        r"""Infer the force field with the minimal parametric estimator.

        Built on ``SFI.inference.parametric_core``: a single RK4 flow step
        per observation interval defines the residual, the residual
        covariance gives a windowed-precision NLL, and the parameters are
        found by direct Gauss–Newton (linear-in-θ ``Basis``, with the
        skip-trick errors-in-variables instrument) or frozen-precision
        L-BFGS (nonlinear-in-θ ``PSF``).  ``(D, Λ)`` are profiled
        natively: moment-estimator init, then one windowed
        conditional-NLL refinement at the fitted θ.

        .. physics:: Parametric windowed force inference (overdamped)
           :label: parametric-force-overdamped
           :category: Inference

           The observed positions follow
           :math:`y_t = x_t + \eta_t` where
           :math:`\eta \sim \mathcal{N}(0, \Lambda)`.  The
           deterministic flow
           :math:`\Phi(x;\theta) = z(\Delta t) - x`
           (one RK4 step by default) defines the residual
           :math:`r_t = y_{t+1} - y_t - \Phi(y_t;\theta)`.
           Residuals follow a banded Gaussian whose local precision
           weights the Gauss–Newton normal equations; under
           measurement noise the left factor is replaced by the
           η-clean *skip* instrument
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
            the single-step minimal estimator).
        inner : {"auto", "gn", "lbfgs"}
            Inner solver.  ``"auto"`` → direct Gauss–Newton for a linear
            ``Basis``, L-BFGS for a ``PSF``.
        eiv : {"auto", True, False, float}
            Measurement-noise errors-in-variables instrument.  ``"auto"``
            (default) → ``True`` for all models (interacting models use
            the same N-body flow for the instrument as for the residual);
            ``True`` forces the η-clean skip instrument (consistent under
            noise); ``False`` is the plain MLE; a float in ``[0, 1]``
            blends.  Active on the GN path only.
        max_outer : int
            Outer Gauss–Newton / IRLS iterations (default 5).
        inner_maxiter : int
            Inner L-BFGS iterations per outer step on the PSF path
            (default 80; raise for large nonlinear families, e.g. NNs).
        extra_radius : int
            Precision-window padding beyond the covariance bandwidth
            (default 1).  Raise to 2–3 in the noise-dominated regime
            β = Tr(Λ)/Tr(2DΔt) ≫ 1, where the windowed precision
            decays slowly and the default window under-resolves it.

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
        elif not (hasattr(F, "flatten_params") and hasattr(F, "unflatten_params")):
            raise TypeError("F must be a PSF or a Basis.  Got %s." % type(F).__name__)
        F_psf = _as_psf(F)

        with self._structural_scope(F_psf):
            dt_val = float(self.data.peek_row(require={"dt"})["dt"])

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
            )
            t_elapsed = _time.perf_counter() - t0

        # ── Store results (standard force_* attributes) ──
        theta_flat = res.theta
        G = res.G

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
            from SFI.inference.sparse import SparseScorer

            self.force_G_full = G
            self.force_scorer = SparseScorer(
                M=self.force_moments,
                G=self.force_G_full,
            )

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
    ) -> None:
        r"""Infer state-dependent diffusion D(x) from parametric residuals.

        Requires a prior parametric :meth:`infer_force` call.  Holds the
        fitted force fixed and minimises the windowed conditional NLL over
        the diffusion parameters (the log-det term makes the diffusion
        level identifiable), reusing the same flow residuals and integrate
        engine as the force solve.

        .. physics:: State-dependent diffusion inference (overdamped)
           :label: parametric-diffusion-overdamped
           :category: Inference

           With the force :math:`\hat F` held fixed, the state-dependent
           diffusion :math:`D(x;\theta_D)` is optimised by minimising the
           windowed conditional negative log-likelihood; :math:`\Lambda`
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
        )
        theta_D = res.theta_D
        self.diffusion_coefficients = theta_D
        self.diffusion_coefficients_full = theta_D
        sf_D = SF(D_psf, D_psf.unflatten_params(theta_D))
        meta_D = dict(
            kind="diffusion",
            inference="parametric",
            n_params=int(D_psf.template.size),
            nll=res.info["nll"],
        )
        self.diffusion_inferred = InferenceResultSF(
            sf_D,
            param_cov=None,
            meta=meta_D,
        )
        self.metadata["diffusion_method"] = "parametric"
        self.metadata["diffusion_parametric_info"] = dict(res.info)


    #################################################################
    ################       BACKEND       ############################
    #################################################################

    # Hooks required by BaseLangevinInference:
    def _force_G_matrix(self) -> jnp.ndarray:
        b_left = self.force_basis
        b_right = self.force_basis @ self.A_inv
        return self.__G_matrix__(b_left, b_right, self.__force_G_mode__, "ima,imb->iab")

    def _force_moments(self):
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
              \;-\; \bigl\langle D_{\text{inst}} : (A^{-1} \cdot \nabla_x b_a) \bigr\rangle

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
            eq='imn,ioma,no->ia' with ops (D, G, A)
            shapes:
              D: (i, m, n)    instantaneous diffusion (N, d, d) e.g. noisy
              G: (i, o, m, a) basis.d_x()(X) shape (N, d_deriv, d_force, F)
              A: (m, n)       A_inv (d, d)

        Masking
        -------
        Mask is applied by the integrator on the leading particle axis i,
        and forwarded to state-expression calls via `mask_out`. The leaf
        fill policy (`zerostop` by default) replaces masked entries with 0
        via `jnp.where`, which naturally gives zero tangents for masked
        entries without blocking the Jacobian for active entries.
        """
        if not hasattr(self, "A_inv"):
            raise RuntimeError("A_inv not available. Compute diffusion first.")

        # Common pieces
        A = ConstOperand(self.A_inv, alias="A")
        V = TimeOperand(velocity("dX", "dt"), alias="V")

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
            logger.debug(
                "Computing Ito-shift force coefficients."
                if mode == "Ito-shift"
                else "Computing Ito force coefficients."
            )
            return integrate(self.data, prog, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)

        # Stratonovich
        logger.debug("Computing Strato force coefficients.")

        # v ∘ b via trapezoid on basis
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
        self.force_v_moments = integrate(
            self.data, prog_strato_vb, reduce="sum", chunk_target_bytes=self._chunk_target_bytes
        )

        logger.debug("Computing Strato gradient term.")

        # Gradient correction: G = basis.d_x() with mask forwarded normally.
        # The leaf's fill_policy='zerostop' uses jnp.where to zero masked
        # entries, which preserves the Jacobian for active particles.
        #
        # For interacting bases (pdepth >= 1), the full cross-particle
        # Jacobian d_x() has shape (N_out, N_in, d, d, F) which doesn't
        # match the Strato einsum 'ioma'.  We use same_particle=True to
        # get the diagonal block (N, d, d, F) = (i, o, m, a) instead.
        # This is physically correct: the Strato correction only needs
        # ∂b_a(x_i)/∂x_i (same-particle gradient) IF the diffusion is
        # diagonal on particle level (no noise correlations between particles).
        _interacting = getattr(self.force_basis, "particles_input", False)
        G = ExprOperand(
            expr=self.force_basis.d_x(same_particle=_interacting),
            x=stream("X"),
            alias="G",
        )
        D = self.get_diffusion_timeop("noisy")  # already a TimeOperand with alias

        prog_grad = Integrand(
            exprs=[G],
            times=[D],
            consts=[A],
            terms=[Term(eq="imn,ioma,no->ia", ops=(D.alias, "G", "A"))],
        )
        self._force_D_grad_b_average = integrate(
            self.data, prog_grad, reduce="sum", chunk_target_bytes=self._chunk_target_bytes
        )

        return self.force_v_moments - self._force_D_grad_b_average

    def _diffusion_G_matrix(self) -> jnp.ndarray:
        # Diffusion is a per-point quantity: weight the projection per-point
        # (weight_by_dt=False), consistently with _diffusion_moments.
        return self.__G_matrix__(
            self.diffusion_basis,
            self.diffusion_basis,
            self.__diffusion_G_mode__,
            "imna,imnb->iab",
            weight_by_dt=False,
        )

    def _diffusion_moments(self) -> jnp.ndarray:
        logger.debug("Computing diffusion linear moments.")
        D_op = self.get_diffusion_timeop(self.__diffusion_M_mode__)
        B = ExprOperand(expr=self.diffusion_basis, x=stream("X"), alias="B")
        prog = Integrand(
            exprs=[B],
            times=[D_op],
            terms=[Term(eq="imna,imn->ia", ops=("B", D_op.alias))],
        )
        # Per-point (weight_by_dt=False): each increment's diffusion estimate counts equally.
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
        cov = getattr(self, "force_coefficients_cov", None)  # may be absent
        self.force_inferred = InferenceResultSF(sf, param_cov=cov, meta=meta)

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
        cov = getattr(self, "diffusion_coefficients_cov", None)  # may be absent
        self.diffusion_inferred = InferenceResultSF(sf, param_cov=cov, meta=meta)

    def __G_matrix__(
        self,
        b_left: callable,
        b_right: callable,
        G_mode: str,
        einsum_string: str,
        subsampling: int = 1,
        weight_by_dt: bool = True,
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

        # Mask-aware reduction over particles; Teff mean over time
        return integrate(
            self.data,
            prog,
            reduce="sum",
            reduce_over_particles=True,
            subsampling=subsampling,
            weight_by_dt=weight_by_dt,
            chunk_target_bytes=self._chunk_target_bytes,
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
    Noisy diffusion estimator — Vestergaard–Blainey–Flyvbjerg (per particle):
      1/4 [ dX⊗(dX/dt) + 2 dX⊗(dX^-/dt) + 2 dX^-⊗(dX/dt) + dX^-⊗(dX^-/dt) ].
    Returns (N, d, d).

    .. physics:: Noisy (Vestergaard–Blainey–Flyvbjerg) diffusion estimator
       :label: D-noisy
       :category: Estimator

       .. math::

          \hat D_{\text{noisy}}(t) = \tfrac{1}{4}\bigl[
              \mathrm{d}X_t \otimes v_t
            + 2\,\mathrm{d}X_t \otimes v_{t-1}
            + 2\,\mathrm{d}X_{t-1} \otimes v_t
            + \mathrm{d}X_{t-1} \otimes v_{t-1}
          \bigr]

       Two-point estimator robust to measurement noise
       (CL Vestergaard, PC Blainey, H Flyvbjerg - Physical Review E, 2014).
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    dt = streams["dt"]
    while dt.ndim < dX.ndim:
        dt = dt[..., jnp.newaxis]
    invdt = 1.0 / dt
    a = jnp.einsum("...m,...n->...mn", dX, dX * invdt)
    b = jnp.einsum("...m,...n->...mn", dX, dXm * invdt)
    c = jnp.einsum("...m,...n->...mn", dXm, dX * invdt)
    d = jnp.einsum("...m,...n->...mn", dXm, dXm * invdt)
    return 0.25 * (a + 2 * b + 2 * c + d)


_D_noisy._requires = frozenset({"dX", "dX_minus", "dt"})  # type: ignore[attr-defined]


@timeop(name="D_weaknoise", batch_safe=True)
def _D_weaknoise(**streams):
    r"""
    Weak-noise estimator (per particle):
      1/4 * ( (dX - dX^-) ⊗ (dX/dt - dX^-/dt) ).
    Returns (N, d, d).

    .. physics:: Weak-noise diffusion estimator (overdamped)
       :label: D-weaknoise-overdamped
       :category: Estimator

       .. math::

          \hat D_{\text{WN}}(t)
          = \tfrac{1}{4}\bigl(\mathrm{d}X_t - \mathrm{d}X_{t-1}\bigr)
            \otimes \bigl(v_t - v_{t-1}\bigr)

       Uses successive-displacement differences; suitable when localization
       noise is negligible.
    """
    dX = streams["dX"]
    dXm = streams["dX_minus"]
    dt = streams["dt"]
    while dt.ndim < dX.ndim:
        dt = dt[..., jnp.newaxis]
    invdt = 1.0 / dt
    ddx = dX - dXm
    dv = dX * invdt - dXm * invdt
    return 0.25 * jnp.einsum("...m,...n->...mn", ddx, dv)


_D_weaknoise._requires = frozenset({"dX", "dX_minus", "dt"})  # type: ignore[attr-defined]


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
