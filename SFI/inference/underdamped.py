# SFI/inference/underdamped.py
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
from SFI.integrate.timeops import TimeOp, stream, timeop, velocity
from SFI.statefunc import PSF, SF, Basis
from SFI.utils.maths import sqrtm_psd

from .result import InferenceResultSF
from .sparse import SparseScorer

logger = logging.getLogger(__name__)


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
    ) -> None:
        """
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

        self.force_basis = basis
        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.__force_diffusion_method__ = diffusion_method
        self.metadata["force_M_mode"] = M_mode
        self.metadata["force_G_mode"] = G_mode
        self.metadata["force_diffusion_method"] = diffusion_method

        if hasattr(self, "force_G_full"):
            raise RuntimeError("Force has already been inferred on this object - create a new instance to re-infer.")

        # Structural arrays are exposed only for this evaluation and never
        # persisted on the dataset (see ``_structural_scope``).
        with self._structural_scope(self.force_basis):
            self.force_G_full = self._force_G_matrix()
            self.force_moments = self._force_moments()

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
        from SFI.langevin import UnderdampedProcess

        if not hasattr(self, "force_inferred"):
            raise RuntimeError("Force not inferred. Call infer_force_linear() (or nonlinear later) first.")

        # Collapse a (possibly pooled) fit to the chosen experiment: per-dataset
        # parameters are folded at ``dataset`` and the reserved ``dataset_index``
        # disappears, so the simulated model is standalone (no dataset concept).
        force_k = self.force_inferred.specialize(dataset=int(dataset))
        diff_k = self.diffusion_inferred.specialize(dataset=int(dataset))
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
    # Parametric — underdamped parametric windowed inference
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
        r"""Infer force F(x,v) with the minimal parametric estimator.

        Built on ``SFI.inference.parametric_core``: the unobserved velocity
        is resolved by shooting through a single RK4 phase-space step, the
        3-point residual covariance is pentadiagonal (bandwidth 2), and the
        parameters are found by direct Gauss–Newton (linear-in-θ ``Basis``,
        with the skip-trick errors-in-variables instrument) or
        frozen-precision L-BFGS (nonlinear-in-θ ``PSF``).  ``(D, Λ)`` are
        profiled natively: moment-estimator init, then one windowed
        conditional-NLL refinement at the fitted θ.

        .. physics:: Parametric windowed force inference (underdamped)
           :label: parametric-force-underdamped
           :category: Inference

           The phase-space dynamics
           :math:`\dot{x}=v,\;\mathrm{d}v = F(x,v)\mathrm{d}t
           + \sqrt{2D}\,\mathrm{d}W` are factored into deterministic
           flow + noise.  Three-point shooting residuals follow a
           pentadiagonal Gaussian whose local precision weights the
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
            Inner solver.  ``"auto"`` → direct Gauss–Newton for a linear
            ``Basis``, L-BFGS for a ``PSF``.
        eiv : {"auto", True, False, float}
            Measurement-noise errors-in-variables instrument.  ``"auto"``
            (default) → ``True`` for all models (interacting models use
            the same N-body flow for the instrument as for the residual);
            ``True`` forces the η-clean skip instrument; ``False`` is the
            plain MLE; a float in ``[0, 1]`` blends.  GN path only.
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
        elif not (hasattr(F, "flatten_params") and hasattr(F, "unflatten_params")):
            raise TypeError("F must be a PSF or a Basis.  Got %s." % type(F).__name__)
        F_psf = _as_psf(F)

        # Structural arrays (interaction lists, etc.) are exposed only for this
        # parametric solve and never persisted (see ``_structural_scope``).
        with self._structural_scope(F_psf):
            dt_val = float(self.data.peek_row(require={"dt"})["dt"])

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
            )
            t_elapsed = _time.perf_counter() - t0

        # ── Standard force attributes ──
        theta_flat = res.theta
        G = res.G

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
            from SFI.inference.sparse import SparseScorer

            self.force_G_full = G
            self.force_scorer = SparseScorer(
                M=self.force_moments,
                G=self.force_G_full,
            )

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
    ) -> None:
        r"""Infer state- and velocity-dependent diffusion D(x, v).

        Requires a prior parametric :meth:`infer_force` call.  Holds the
        fitted force fixed and minimises the pentadiagonal windowed
        conditional NLL over the diffusion parameters, with ``D(x, v;
        θ_D)`` evaluated at the shooting velocity.

        .. physics:: State-dependent diffusion inference (underdamped)
           :label: parametric-diffusion-underdamped
           :category: Inference

           With the force :math:`\hat F(x,v)` held fixed, the
           state-dependent diffusion :math:`D(x,v;\theta_D)` is optimised
           by minimising the windowed conditional NLL on pentadiagonal
           (bandwidth-2) covariance windows; :math:`\Lambda` from the
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

        from SFI.statefunc import SF
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

    def _force_G_matrix(self) -> jnp.ndarray:
        if not hasattr(self, "A_inv"):
            raise RuntimeError("A_inv not available. Compute diffusion first.")
        b_left = self.force_basis
        b_right = self.force_basis @ self.A_inv
        return self.__G_matrix__(b_left, b_right, self.__force_G_mode__, "ima,imb->iab")

    def _force_moments(self) -> jnp.ndarray:
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
              \;+\; w \,\bigl\langle -D_{\text{inst}} : (A^{-1}\cdot\partial_v b_a)
              \bigr\rangle

           where :math:`w = (1+2\ell)/3`, with :math:`\ell=1` (symmetric),
           :math:`\ell=0` (early), :math:`\ell=-\tfrac{1}{2}` (anticipated).

        Contractions
        ------------
        - phi term:
            eq='im,mn,ina->ia' with ops (â, A_inv, b)

        - correction term:
            eq='imn,no,imoa->ia' with ops (D_inst, A_inv, ∂_v b)
            (the third axis of ∂_v b is the derivative index; we label it 'o' above).
        """
        if not hasattr(self, "A_inv"):
            raise RuntimeError("A_inv not available. Compute diffusion first.")

        mode = getattr(self, "__force_M_mode__", "symmetric")
        if mode not in ("symmetric", "early", "anticipated"):
            raise KeyError(f"Unknown __force_M_mode__: {mode}")

        x_hat = self._get_x_hat(M_mode=mode)
        v_hat = self._get_v_hat(M_mode=mode)
        a_hat = self._get_a_hat(M_mode=mode)

        # Main moment term: < â · A_inv · b >
        A = ConstOperand(self.A_inv, alias="A")
        a_op = TimeOperand(a_hat, alias="a")
        B = ExprOperand(expr=self.force_basis, x=x_hat, v=v_hat, alias="B")

        prog_phi = Integrand(
            exprs=[B],
            times=[a_op],
            consts=[A],
            terms=[Term(eq="im,mn,ina->ia", ops=("a", "A", "B"))],
        )
        phi_moments = integrate(self.data, prog_phi, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
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

        D_op = self.get_diffusion_timeop(getattr(self, "__force_diffusion_method__", "noisy"))
        # For pair-dispatched bases (pdepth>=1), the Ito correction needs
        # the same-particle velocity Jacobian  ∂f_i/∂v_i  (not cross ∂f_i/∂v_j).
        _same = getattr(self.force_basis, "pdepth", 0) >= 1
        Gv = ExprOperand(
            expr=self.force_basis.d_v(same_particle=_same),
            x=x_hat,
            v=v_hat,
            alias="Gv",
        )

        prog_w = Integrand(
            exprs=[Gv],
            times=[D_op],
            consts=[A],
            terms=[
                Term(
                    eq="imn,no,imoa->ia",
                    ops=(D_op.alias, "A", "Gv"),
                    scale=-float(w_pref),
                )
            ],
        )
        w_moments = integrate(self.data, prog_w, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
        self._w_moments = w_moments

        return phi_moments + w_moments

    def _diffusion_G_matrix(self) -> jnp.ndarray:
        # Diffusion is per-point: weight the projection per-point (weight_by_dt=False).
        return self.__G_matrix__(
            self.diffusion_basis,
            self.diffusion_basis,
            self.__diffusion_G_mode__,
            "imna,imnb->iab",
            weight_by_dt=False,
        )

    def _diffusion_moments(self) -> jnp.ndarray:
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

        logger.debug("Computing diffusion linear moments (method=%s).", method)

        # Instantaneous diffusion estimator matching the method string
        D_op = self.get_diffusion_timeop(method)

        # Evaluate diffusion basis on (x_hat, v_hat)
        B = ExprOperand(expr=self.diffusion_basis, x=x_hat, v=v_hat, alias="B")

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
        cov = getattr(self, "force_coefficients_covariance", None)
        self.force_inferred = InferenceResultSF(sf, param_cov=cov, meta=meta)

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
        cov = getattr(self, "diffusion_coefficients_cov", None)
        self.diffusion_inferred = InferenceResultSF(sf, param_cov=cov, meta=meta)

    # ------------------------------ shared helpers ------------------------------ #

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
        )
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
        return jnp.swapaxes(G, -1, -2)

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
    return (3.0 / 11.0) * M / (dt**3)


_D_noisy_uli._requires = frozenset({"dX", "dX_minus", "dX_plus", "dt"})  # type: ignore[attr-defined]


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
