from __future__ import annotations

import contextlib
import dataclasses
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SFI.inference.sparse import SparsityResult

import jax
import jax.numpy as jnp
import numpy as np

from SFI.integrate.api import integrate
from SFI.integrate.integrand import (
    ConstOperand,
    ExprOperand,
    Integrand,
    Term,
    TimeOperand,
)
from SFI.integrate.timeops import stream, timeop, velocity
from SFI.statefunc.nodes.interactions.prepare import (
    prepare_collection_for_expr,
)
from SFI.utils.maths import stable_pinv

logger = logging.getLogger(__name__)


def _spd_inverse(F, label: str):
    """Inverse of a symmetric positive-definite information matrix.

    Raises if ``F`` is not positive definite (Cholesky fails): the fit is not
    at a non-degenerate likelihood optimum and no covariance follows from it.
    """
    F = 0.5 * (jnp.asarray(F) + jnp.asarray(F).T)
    L = jnp.linalg.cholesky(F)
    if not bool(jnp.all(jnp.isfinite(L))):
        raise RuntimeError(
            f"The {label} is not positive definite: the fit is not at a "
            "non-degenerate optimum (unconverged, on a positivity boundary, or "
            "with collinear parameters); no covariance is available."
        )
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(F.shape[0], dtype=F.dtype), lower=True)
    return Linv.T @ Linv


class BaseLangevinInference(ABC):
    """Stochastic Force Inference main class

    This class provides tools for inferring force (drift) and
    diffusion tensors from stochastic trajectory data based on
    Langevin dynamics. It contains the shared logic for Overdamped and
    Underdamped Langevin inference.

    These subclasses must implement a handful of hooks that depend on
    the physics (e.g. whether velocities are observed). The details of
    the physics assumptions and definitions, as well as extensive doc
    strings, are given in the headers of these classes.

    Key Features
    ------------

    - **Force inference** — linear combination of basis functions
      (``infer_force_linear`` with a ``Basis``, the canonical path) or
      parametric families (``infer_force`` with a ``Basis`` or ``PSF``, the
      single-step flow estimator with native (D, Λ) profiling).
    - **Diffusion inference** — constant (``compute_diffusion_constant``) or
      state-dependent via a linear basis (``infer_diffusion_linear``).
    - **Sparsification** — pluggable strategies (beam search, greedy, STLSQ,
      LASSO) with information-criterion selection (AIC, BIC, PASTIS).
    - **Error estimation** — normalized mean-squared-error (NMSE) prediction
      for force and diffusion.
    - **Comparison** — evaluate inferred fields against known exact models
      (``compare_to_exact``).
    - **Simulation** — generate trajectories from inferred fields
      (``simulate_bootstrapped_trajectory``).

    Workflow
    --------

    1. Initialize with a ``TrajectoryCollection`` holding the trajectory.
    2. Use the ``infer_*`` methods to infer force and diffusion fields.
    3. Optionally sparsify the results to mitigate overfitting.
    4. Optionally compute error estimates and/or compare with exact data.

    Indices Convention
    ------------------

    The code uses ``jnp.einsum`` with a consistent index naming scheme:

    - ``t`` — time index, 0..Ntimesteps-1.
    - ``a, b, c...`` — basis-function indices, 0..Nfunctions-1.
    - ``m, n, o...`` — state / spatial indices, 0..dim-1.
    - ``i, j...`` — particle indices (size Nparticles, or 1 if there is no
      particle structure).

    These also serve as array-shape shorthands: e.g.
    ``basis_linear : im -> iam`` means ``basis_linear`` takes an array of
    shape (Nparticles, dim) and returns one of shape
    (Nparticles, Nfunctions, dim).

    Logging levels control output (configure via ``logging`` or
    ``SFI.enable_logging()``):

    - INFO  -> inference steps, key results.
    - DEBUG -> detailed computation progress.

    """

    def __init__(self, data, *, max_memory_gb=1.0, **kwargs):
        """Initialize the inference object.

        Parameters
        ----------
        data : TrajectoryCollection
            The trajectory data to infer from.
        max_memory_gb : float
            Approximate memory budget for integration batches and for each
            chunk of the parametric estimator's programs (both sized from
            the compiled kernels' measured working sets).

        Raises
        ------
        TypeError
            If ``data`` is not a ``TrajectoryCollection``.
        """
        from SFI.trajectory.collection import TrajectoryCollection

        if not isinstance(data, TrajectoryCollection):
            raise TypeError(
                f"Expected a TrajectoryCollection, got {type(data).__name__}. "
                f"Use TrajectoryCollection.from_dataset(ds) or "
                f"TrajectoryCollection.from_arrays(X=..., dt=...) first."
            )
        self.data = data
        self.max_memory_gb = max_memory_gb
        self.metadata = {}

    @property
    def _chunk_target_bytes(self) -> int:
        """Convert ``max_memory_gb`` to bytes for the integration engine."""
        return max(1, int(self.max_memory_gb * 1024**3))

    def _rows_teff(self, frames) -> float:
        """Unweighted particle-time of the rows a ``shared_rows(frames)`` block sums over.

        A recount, for checks: estimators normalize by the measure
        :func:`~SFI.integrate.integrate` returns with their own sums
        (``return_measure``), which also carries the dataset weights.
        """
        from SFI.trajectory.dataset import frame_key

        return float(self.data.Teff({"X", "__dt__"} | {frame_key(f) for f in frames}))

    @contextlib.contextmanager
    def _structural_scope(self, *exprs):
        """Expose the structural (CSR/stencil) arrays required by ``exprs`` for the
        duration of the block, without ever persisting them on the dataset.

        On entry ``self.data`` is swapped for a transient collection carrying the
        freshly-built structural arrays; on exit the original collection object is
        restored. Nothing is left on the dataset, so a stale structural table can
        never survive a later transform, and teardown is exception-safe (there is
        no purge step to forget).
        """
        original = self.data
        try:
            exprs = tuple(e for e in exprs if e is not None)
            if exprs:
                self.data = prepare_collection_for_expr(original, *exprs)
            yield
        finally:
            self.data = original

    # ---- shared validation helpers ---------------------------------------- #

    # Condition-number thresholds of the force Gram, by floating-point width:
    # beyond them the pseudo-inverse solve mixes rounding noise into the
    # coefficients (float32 carries ~7 digits, float64 ~16).
    _GRAM_COND_MAX = {4: 1e6, 8: 1e8}

    def _check_gram_conditioning(self, G, *, label: str = "force") -> float:
        """Record the condition number of a Gram matrix and warn when it is extreme.

        Sets ``metadata[f"{label}_gram_cond"]``.  An ill-conditioned Gram
        (collinear or badly scaled features, power-law dictionaries on noisy
        positions, ...) makes the fitted coefficients and their error
        estimates unreliable even though the pseudo-inverse solve succeeds.
        """
        G = jnp.asarray(G)
        cond = float(jnp.linalg.cond(G)) if G.ndim == 2 and G.shape[0] > 1 else 1.0
        self.metadata[f"{label}_gram_cond"] = cond
        thr = self._GRAM_COND_MAX.get(int(jnp.dtype(G.dtype).itemsize), 1e8)
        if cond > thr:
            warnings.warn(
                f"{label} Gram condition number {cond:.2e} > {thr:.0e}: the basis is "
                "ill-conditioned on these data (collinear or badly scaled features); the "
                "coefficients and error estimates are unreliable - drop or rescale features, "
                "or sparsify.",
                stacklevel=3,
            )
        logger.info("%s Gram condition number: %.3g", label, cond)
        return cond

    def _noise_consistency_issue(self):
        """Message when the force and diffusion fits treat measurement noise differently.

        Under localization noise (``Lambda_trace > 0``) the drift and the
        diffusion must be estimated on the same footing - both raw, or both
        noise-corrected (``preset="robust"`` with the ``"noisy"`` diffusion
        stencil, which is what ``"auto"`` selects).  Correcting only one of
        them leaves an offset :math:`\\Lambda/\\Delta t` in the other, which
        the force :math:`\\bar D^{-1}\\Phi` turns into a spurious term
        proportional to :math:`\\nabla \\ln P`.  Returns ``None`` when the
        fits are consistent, or when no noise is detected, or when one of
        the two fits is missing.
        """
        md = self.metadata
        if md.get("force_method") != "linear" or "force_preset" not in md:
            return None
        lam = getattr(self, "Lambda_trace", None)
        if lam is None or float(lam) <= 0.0:
            return None
        force_corr = (
            md.get("force_preset") == "robust"
            or md.get("force_diffusion_method") == "noisy"
            or (md.get("force_M_mode") == "Strato" and md.get("force_G_mode") == "shift")
        )
        d_methods = {md.get("diffusion_constant_method"), md.get("diffusion_M_mode")} - {None}
        if not d_methods:
            return None
        if len(d_methods) > 1:
            return (
                f"the constant diffusion ({md.get('diffusion_constant_method')}) and the diffusion "
                f"field ({md.get('diffusion_M_mode')}) use different measurement-noise treatments"
            )
        diff_corr = d_methods == {"noisy"}
        if force_corr and not diff_corr:
            return (
                f"noise-robust force fit (preset {md.get('force_preset')!r}) combined with an "
                f"uncorrected diffusion estimate ({next(iter(d_methods))!r}) while measurement noise is "
                f"detected (Tr Lambda = {float(lam):.3g}); use compute_diffusion_constant(method='noisy') "
                "(the 'auto' choice) so that drift and diffusion see the same process"
            )
        if diff_corr and not force_corr:
            return (
                f"noise-corrected diffusion ('noisy') combined with an uncorrected force fit "
                f"(preset {md.get('force_preset')!r}) while measurement noise is detected "
                f"(Tr Lambda = {float(lam):.3g}): the offset Lambda/dt left in the drift becomes a "
                "spurious force along grad ln P; use preset='robust' (the 'auto' choice)"
            )
        return None

    def _check_diffusion_psd(self, D_sf, *, dataset: int = 0, n_points: int = 10_000) -> dict:
        """PSD check of a diffusion field on the data before it is simulated.

        Records ``metadata["diffusion_psd_fraction"]`` / ``["diffusion_min_eig"]``
        and warns when the field is not positive semi-definite on part of the
        data: the simulator clips negative eigenvalues to zero, so the
        generated noise silently differs from the fitted field there.
        """
        from SFI.diagnostics.psd import psd_fraction

        res = psd_fraction(D_sf, self.data, n_points=n_points, dataset=dataset)
        if res.get("available"):
            self.metadata["diffusion_psd_fraction"] = res["fraction"]
            self.metadata["diffusion_min_eig"] = res["min_eig"]
            if res["fraction"] > 0.0:
                warnings.warn(
                    f"the inferred diffusion is not positive semi-definite on {100 * res['fraction']:.1f}% "
                    f"of the data points (most negative eigenvalue {res['min_eig']:.3g}); the simulator "
                    "clips negative eigenvalues to zero, so the simulated noise differs from the fitted "
                    "field there - constrain the diffusion model (fewer features, a PSD parametrization) "
                    "before generative use",
                    stacklevel=3,
                )
        return res

    def _check_same_site_correction(self, basis, *, label: str) -> bool:
        """Warn when a same-site noise correction is applied to a lattice-field basis.

        The Stratonovich force correction (OLI) and the ULI velocity correction
        contract the instantaneous diffusion with each particle's own Jacobian
        block, which is exact when the noise is independent between particles.
        On a lattice field (a basis built from :mod:`SFI.bases.spde` stencil
        operators) the sites are the particles, and conserved or spatially
        correlated noise couples neighbouring sites: the cross-site terms
        :math:`\\sum_{j\\neq i} D_{ij}\\,\\partial_j b_i` are then missing and
        every coefficient is biased at O(1), signs included.  Returns whether
        the warning fired.
        """
        keys = getattr(basis, "structural_extras", ()) or ()
        if not any(str(k).startswith("_cache/stencil/") for k in keys):
            return False
        warnings.warn(
            f"{label} on a lattice-field (SPDE stencil) basis: the correction uses each "
            "site's own Jacobian block, which is exact only for noise independent between "
            "sites (white noise). With conserved or spatially correlated noise "
            "(e.g. ConservedNoise) the cross-site terms are missing and every coefficient "
            "is biased, signs included; use cross_particle='all-pairs' (the default), or "
            "M_mode='Ito', unless the noise is known to be white.",
            stacklevel=4,
        )
        return True

    @staticmethod
    def _warn_if_null_parametric_force(theta, info=None, *, label: str = "infer_force") -> bool:
        """Warn when a parametric force fit returns the all-zero solution.

        An all-zero coefficient vector means the Gauss–Newton solver never
        moved from its (zero) initial guess: no step lowered the objective.
        The fit then carries no information (``force_information = 0``,
        ``force_predicted_MSE`` NaN) and the profiled ``(D, Λ)`` are those
        of a zero force — previously silent.  The result is left untouched;
        returns whether the warning fired.
        """
        th = np.asarray(theta)
        if th.size == 0 or not np.all(np.isfinite(th)) or np.any(th != 0):
            return False
        it = (info or {}).get("outer_iterations")
        it_txt = f" after {it} outer iteration(s)" if it is not None else ""
        warnings.warn(
            f"{label}: the parametric fit returned an all-zero force (θ ≡ 0){it_txt} — the "
            "solver never moved from its zero initial guess, so the fit carries no "
            "information (force_information = 0, force_predicted_MSE undefined) and the "
            "profiled D and Λ belong to a zero force.  Typical cause: the per-frame force "
            "signal (|F|·Δt², underdamped; |F|·Δt, overdamped) is buried in localisation "
            "noise at this frame interval: fit coarser frames (coll.degrade(downsample=k)).",
            RuntimeWarning,
            stacklevel=3,
        )
        return True

    def _check_noise_consistency(self):
        """Warn (once per call) when :meth:`_noise_consistency_issue` reports a mismatch."""
        msg = self._noise_consistency_issue()
        if msg:
            warnings.warn("inconsistent measurement-noise treatment: " + msg, stacklevel=3)
        return msg

    @staticmethod
    def _validate_basis(basis, *, expected_rank: int, label: str = "basis"):
        """Check that ``basis`` is a Basis with the correct rank.

        Parameters
        ----------
        basis : object
            Should be a ``Basis`` instance.
        expected_rank : int
            Required spatial rank (1 for force, 2 for diffusion).
        label : str
            Human-readable name for error messages.

        Raises
        ------
        TypeError
            If ``basis`` is not a ``Basis``.
        ValueError
            If the rank does not match ``expected_rank``.
        """
        from SFI.statefunc import Basis as _Basis

        if not isinstance(basis, _Basis):
            from SFI.statefunc import PSF as _PSF
            from SFI.statefunc import SF as _SF

            if isinstance(basis, (_PSF, _SF)):
                raise TypeError(
                    f"`{label}` must be a Basis (deterministic dictionary), "
                    f"not a {type(basis).__name__}. "
                    f"For parametric inference with a PSF, use infer_force()."
                )
            raise TypeError(f"`{label}` must be a Basis object, got {type(basis).__name__}.")
        if basis.rank != expected_rank:
            raise ValueError(
                f"`{label}` must have rank={expected_rank} "
                f"(got rank={basis.rank}). "
                + (
                    "Force bases should produce vectors (rank 1)."
                    if expected_rank == 1
                    else "Diffusion bases should produce matrices (rank 2)."
                )
            )

    def sparsify_force(
        self,
        *,
        criterion: str = "PASTIS",
        p: float = 0.05,
        method: str = "beam",
        max_k: int | None = None,
        **strategy_kwargs,
    ) -> "SparsityResult":
        """Sparsify the inferred force by selecting a subset of basis functions.

        Builds a Pareto front of sparse models using the chosen
        ``method``, then selects the model that maximises the given
        information ``criterion``.

        Parameters
        ----------
        criterion : ``"PASTIS"`` | ``"AIC"`` | ``"BIC"`` | ``"EBIC"`` | ``"SIC"``, default ``"PASTIS"``
            Information criterion for model selection.
        p : float, default 0.05
            Prior-scale parameter :math:`p_0` for the PASTIS penalty.
        method : str, default ``"beam"``
            Search strategy.  One of:

            - ``"beam"`` — bidirectional beam search (PASTIS original).
              Extra kwargs: ``beam_width`` (int, default 3),
              ``aic_patience`` (int, default 2).
            - ``"greedy"`` — forward stepwise selection.
              Extra kwargs: ``direction`` (``"forward"`` | ``"backward"``
              | ``"both"``, default ``"forward"``).
            - ``"stlsq"`` — Sequential Thresholded Least Squares
              (SINDy-style). Extra kwargs: ``threshold`` (float or None),
              ``mode`` (``"relative"`` | ``"absolute"``),
              ``n_thresholds`` (int).
            - ``"lasso"`` — :math:`\\ell_1`-penalised coordinate descent.
              Extra kwargs: ``alpha`` (float or None),
              ``n_alphas`` (int).
            - ``"hillclimb"`` — stochastic hill-climbing
              (Gerardos & Ronceray, 2025).  Extra kwargs: ``ic``,
              ``patience`` (int), ``seed`` (int or None).
        max_k : int or None
            Maximum model size.  Defaults to the full basis size.
        **strategy_kwargs
            Passed to the strategy constructor.

        Returns
        -------
        SparsityResult
            The full Pareto-front result, also stored as
            ``self.force_sparsity_result``.

        Notes
        -----
        The error read-outs follow the selected coefficients.  If
        :meth:`compute_force_error` had been run, it is re-run on the
        selected support (no pass over the data: the covariance is the
        inverse selection metric on the support), so
        ``force_coefficients_covariance``, ``force_coefficients_stderr``,
        ``force_predicted_MSE`` and ``force_inferred.param_cov`` describe the
        new coefficients.  A parametric fit, whose ``param_cov`` is set at
        fit time, keeps one (the restricted estimator's).  Otherwise
        ``force_inferred.param_cov`` stays ``None`` until
        :meth:`compute_force_error`.
        """
        # BIC/EBIC penalise by the total trajectory time tau: the linear fit's
        # own row measure (the particle-time its Gram sums over), else Teff.
        if self.metadata.get("force_method") == "linear":
            tau = float(self._force_row_measure)
        else:
            tau = float(self.data.Teff({"dt"}))
        result, coeffs, support = self._select_support(
            self.force_scorer, criterion=criterion, p=p, method=method, max_k=max_k,
            tau=tau, strategy_kwargs=strategy_kwargs,
        )
        had_error, had_cov = self._error_state("force")
        self._update_force_coefficients(coeffs, support)
        self.force_sparsity_result = result
        self._refresh_error_after_selection("force", had_error=had_error, had_cov=had_cov)
        return result

    # Read-outs of compute_{field}_error: they describe one set of coefficients.
    _ERROR_ATTRS = ("coefficients_covariance", "coefficients_cov", "coefficients_stderr",
                    "information", "predicted_MSE")

    def _error_state(self, field: str):
        """``(had_error, had_cov)``: whether ``compute_{field}_error`` was run on the
        current coefficients, and whether ``{field}_inferred`` carries a ``param_cov``."""
        res = getattr(self, f"{field}_inferred", None)
        return (hasattr(self, f"{field}_coefficients_covariance"),
                getattr(res, "param_cov", None) is not None)

    def _drop_error_analysis(self, field: str) -> None:
        for attr in self._ERROR_ATTRS:
            self.__dict__.pop(f"{field}_{attr}", None)

    def _refresh_error_after_selection(self, field: str, *, had_error: bool, had_cov: bool) -> None:
        """Keep the error read-outs in step with the coefficients a selection just set.

        See the Notes of :meth:`sparsify_force`.  After a selection the
        covariance is the inverse selection metric on the support (no pass over
        the data), so re-running ``compute_{field}_error`` is cheap.
        """
        if had_error:
            try:
                getattr(self, f"compute_{field}_error")()
            except RuntimeError as err:
                self._drop_error_analysis(field)
                warnings.warn(f"sparsify_{field}: no covariance for the selected support ({err})",
                              RuntimeWarning, stacklevel=3)
            return
        if had_cov and field == "force":
            # Parametric fit (param_cov set at fit time): the restricted estimator's covariance.
            object.__setattr__(self.force_inferred, "param_cov",
                               self._embed_support_cov("force", self.force_G_pinv))

    def _embed_support_cov(self, field: str, cov):
        """Support-sized covariance in the fitted PSF's full parameter order
        (pruned coefficients are exact zeros, with zero variance)."""
        if cov is None:
            return None
        cov = jnp.asarray(cov)
        p_full = int(np.asarray(getattr(self, f"{field}_coefficients_full")).size)
        sup = np.asarray(getattr(self, f"{field}_support"), dtype=int).ravel()
        if sup.size == p_full and np.array_equal(sup, np.arange(p_full)):
            return cov
        return jnp.zeros((p_full, p_full), dtype=cov.dtype).at[jnp.ix_(sup, sup)].set(cov)

    def _reset_fit_state(self, field: str) -> None:
        """Forget what a previous fit of ``field`` left behind before a new fit.

        A linear fit may follow a parametric one on the same object (and a
        parametric diffusion fit a linear one): the method flag, the
        parametric state, the selection metric and the error analysis describe
        the previous coefficients and must not be read for the new ones.
        """
        stale = {
            "force": ("force_fisher_full", "force_psf"),
            "diffusion": ("diffusion_fisher_full", "_diffusion_parametric_current",
                          "_diffusion_parametric_full", "_diffusion_parametric_spec"),
        }[field]
        for attr in stale:
            self.__dict__.pop(attr, None)
        self._drop_error_analysis(field)
        self.metadata.pop(f"{field}_method", None)
        self.metadata.pop(f"{field}_parametric_info", None)

    def _select_support(self, scorer, *, criterion, p, method, max_k, tau, strategy_kwargs):
        """Pareto-front search on ``scorer`` + information-criterion selection.

        Shared by :meth:`sparsify_force` and :meth:`sparsify_diffusion`.
        Returns ``(result, coeffs, support)``.
        """
        from SFI.inference.sparse import (
            BeamSearchStrategy,
            GreedyStepwiseStrategy,
            HillClimbStrategy,
            LassoStrategy,
            STLSQStrategy,
        )

        if max_k is None:
            max_k = scorer.p

        # Build the strategy object
        _strategies = {
            "beam": BeamSearchStrategy,
            "greedy": GreedyStepwiseStrategy,
            "stlsq": STLSQStrategy,
            "lasso": LassoStrategy,
            "hillclimb": HillClimbStrategy,
        }
        key = method.lower()
        if key not in _strategies:
            raise ValueError(f"Unknown sparsity method {method!r}. Choose from {list(_strategies)}.")

        # Default beam_width for beam search
        if key == "beam":
            strategy_kwargs.setdefault("beam_width", 3)
            strategy_kwargs.setdefault("aic_patience", 2)
            strategy_kwargs.setdefault("report_time", True)
        elif key == "hillclimb":
            # Hill-climb uses the selection criterion as its acceptance
            # objective by default, so it stays consistent with the IC
            # used at the final `select_by_ic` step.
            strategy_kwargs.setdefault("ic", criterion)
            strategy_kwargs.setdefault("p_param", p)

        strategy = _strategies[key](**strategy_kwargs)
        result: SparsityResult = strategy.run(scorer, max_k=max_k)

        # Select the best model according to the information criterion.
        k, support, score, coeffs = result.select_by_ic(criterion, p_param=p, tau=tau)
        result = dataclasses.replace(
            result, selection={"criterion": criterion.upper(), "p_param": p, "tau": tau}
        )
        return result, coeffs, support

    def sparsify_diffusion(
        self,
        *,
        criterion: str = "PASTIS",
        p: float = 0.05,
        method: str = "beam",
        max_k: int | None = None,
        **strategy_kwargs,
    ) -> "SparsityResult":
        r"""Sparsify the inferred diffusion field by selecting basis functions.

        The diffusion counterpart of :meth:`sparsify_force`: same search
        strategies (``method``), same information criteria (``criterion``),
        applied to the fit of the last :meth:`infer_diffusion_linear` or
        :meth:`infer_diffusion` call.  The information it scores is the
        log-likelihood gain of the diffusion fit, so that a spurious basis
        function gains :math:`\chi^2_1/2` — the calibration every criterion
        assumes.

        .. physics:: Diffusion information for model selection
           :label: diffusion-information-selection
           :category: Inference

           **Linear fit.**  :math:`\hat C = G^{-1}M` solves the moment
           condition :math:`M - GC = \sum_t s_t(C) = 0`, whose long-run
           covariance is the sandwich meat :math:`S` of
           :meth:`compute_diffusion_error`.  Asymptotically
           :math:`M \sim \mathcal N(GC,\, S)`, so the log-likelihood of a
           support :math:`B` (coefficients zero off :math:`B`) is the
           efficient (optimal-weight) minimum-distance criterion

           .. math::

              \mathcal I(B) = \tfrac12\, m_B^\top \mathcal F_{BB}^{-1} m_B,
              \qquad
              \mathcal F = G^\top S^{-1} G, \quad m = G^\top S^{-1} M,

           maximised by :math:`\hat C_B = \mathcal F_{BB}^{-1} m_B`.  On the
           full basis :math:`\hat C_B` is the least-squares fit and
           :math:`\mathcal F^{-1} = G^{-1}SG^{-1}` its sandwich covariance;
           on a sub-support it is the efficient restricted estimator, with
           covariance :math:`\mathcal F_{BB}^{-1}`.  :math:`S` is evaluated
           once, at the full-basis fit, so every support is scored in the
           same metric and :math:`2[\mathcal I(\mathrm{full}) - \mathcal
           I(B)]` is the Hansen–Newey–West distance statistic,
           :math:`\chi^2_{n_0-|B|}` when the true field lies in :math:`B`.
           The plain Gram :math:`\tfrac12 M^\top G^{-1}M` is *not* an
           information: it carries the units of :math:`D^2` and ignores
           the estimator's variance.

           **Parametric fit.**  :math:`\mathcal F` is the Hessian of the
           banded NLL at :math:`\hat\theta_D` and :math:`m = \mathcal F
           \hat\theta_D` (the quadratic expansion of the NLL, as for the
           parametric force); the selected support is then re-fitted on
           the exact NLL.

        Parameters
        ----------
        criterion, p, method, max_k, **strategy_kwargs
            As in :meth:`sparsify_force`.  ``"BIC"``/``"EBIC"`` use the
            number of observed increments as sample size (the diffusion
            information grows with the number of steps, not with time).

        Returns
        -------
        SparsityResult
            The full Pareto-front result, also stored as
            ``self.diffusion_sparsity_result``.  The selected coefficients
            replace ``diffusion_coefficients`` (``diffusion_support`` gives
            their indices); run :meth:`compute_diffusion_error` for their
            covariance.  If it had already been run, it is re-run on the
            selected support, as in :meth:`sparsify_force`; a parametric fit's
            ``diffusion_inferred.param_cov`` is that of the re-fitted support.

        Notes
        -----
        Requires a basis-expanded diffusion fit: a rank-2 ``Basis`` (linear
        route) or a parametric fit whose parameters are the selection
        candidates (a ``Basis``, or a PSF whose zero parameters switch
        terms off).
        """
        from SFI.inference.sparse import SparseScorer

        if not hasattr(self, "diffusion_coefficients_full"):
            raise RuntimeError(
                "sparsify_diffusion() requires a diffusion fit: run infer_diffusion_linear() or infer_diffusion()."
            )
        if self.metadata.get("diffusion_method") == "parametric":
            dt_val = float(self.data.peek_row(require={"dt"})["dt"])
            n_obs = float(self.data.Teff({"dt"})) / dt_val
        else:
            # The linear fit's own row count (its Gram is per-point weighted).
            n_obs = float(self._diffusion_row_measure)

        if self.metadata.get("diffusion_method") == "parametric":
            fisher = getattr(self, "_diffusion_parametric_full", None)
            if fisher is None:
                raise RuntimeError("The parametric diffusion fit carries no NLL Hessian.")
            F_full, theta_full = fisher["hessian"], fisher["theta_D"]
            scorer = SparseScorer(M=F_full @ theta_full, G=F_full)
        else:
            F_full, m_full = self._diffusion_selection_metric()
            scorer = SparseScorer(M=m_full, G=F_full)

        result, coeffs, support = self._select_support(
            scorer, criterion=criterion, p=p, method=method, max_k=max_k,
            tau=n_obs, strategy_kwargs=strategy_kwargs,
        )
        support = np.asarray(support, dtype=int)
        had_error, had_cov = self._error_state("diffusion")
        if self.metadata.get("diffusion_method") == "parametric":
            self._refit_parametric_diffusion(support)
        else:
            self.diffusion_fisher_full = F_full
            self._update_diffusion_coefficients(jnp.asarray(coeffs), support)
        self.diffusion_sparsity_result = result
        self._refresh_error_after_selection("diffusion", had_error=had_error, had_cov=had_cov)
        return result

    def _diffusion_selection_metric(self):
        """Fisher metric ``(F, m)`` of the linear diffusion fit (full basis).

        ``F = Gᵀ S⁻¹ G``, ``m = Gᵀ S⁻¹ M`` with ``S`` the long-run score
        covariance at the full-basis least-squares fit; see
        :meth:`sparsify_diffusion`.
        """
        G = np.asarray(self.diffusion_G_full, dtype=float)
        M = np.asarray(self.diffusion_moments, dtype=float)
        C_full = np.asarray(self.diffusion_scorer.total_C, dtype=float)
        S = self._diffusion_score_covariance(coefficients=C_full)
        try:
            L = np.linalg.cholesky(0.5 * (S + S.T))
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                "The diffusion score covariance is not positive definite (too few "
                "samples for this basis, or collinear basis functions); the "
                "diffusion information is undefined."
            ) from err
        W = np.linalg.solve(L, G)  # L⁻¹G, so F = Wᵀ W
        F = W.T @ W
        m = W.T @ np.linalg.solve(L, M)
        return jnp.asarray(0.5 * (F + F.T)), jnp.asarray(m)

    def _store_parametric_diffusion(self, res, D_psf, spec: dict, support=None) -> None:
        """Write a parametric diffusion solve to the fit attributes.

        ``res`` is a ``DiffusionSolveResult``; ``spec`` the keyword arguments
        that reproduce the solve (kept for the re-fit of a selected support);
        ``support`` the free parameters (``None``: all).
        """
        from SFI.inference.result import InferenceResultSF
        from SFI.statefunc import SF, Basis

        if support is None:
            # A fresh fit (not the re-fit of a selected support): drop what a
            # previous diffusion fit left behind.
            self._reset_fit_state("diffusion")
            basis = spec.get("basis")
            self.diffusion_basis_labels = getattr(basis, "labels", None) if isinstance(basis, Basis) else None
        # The linear fit's selection metric never applies to a parametric fit.
        self.__dict__.pop("diffusion_fisher_full", None)

        theta_D = jnp.asarray(res.theta_D)
        n_D = int(theta_D.size)
        sup = np.arange(n_D) if support is None else np.asarray(support, dtype=int)
        self.diffusion_coefficients_full = theta_D
        self.diffusion_coefficients = theta_D if support is None else theta_D[sup]
        self.diffusion_support = jnp.asarray(sup)
        current = {"theta_D": theta_D, "hessian": jnp.asarray(res.hessian)}
        self._diffusion_parametric_current = current
        if support is None:
            self._diffusion_parametric_full = current
            self._diffusion_parametric_spec = spec

        # Inverse observed Fisher information on the support, embedded in the
        # PSF's full parameter order (fixed parameters have zero variance).
        cov_full = None
        try:
            cov = _spd_inverse(current["hessian"][jnp.ix_(sup, sup)], "diffusion NLL Hessian")
            cov_full = jnp.zeros((n_D, n_D), dtype=cov.dtype).at[jnp.ix_(sup, sup)].set(cov)
        except RuntimeError as err:
            warnings.warn(f"infer_diffusion: {err}", RuntimeWarning, stacklevel=3)

        meta_D = dict(
            kind="diffusion",
            inference="parametric",
            n_params=int(D_psf.template.size),
            nll=res.info["nll"],
        )
        self.diffusion_inferred = InferenceResultSF(
            SF(D_psf, D_psf.unflatten_params(theta_D)), param_cov=cov_full, meta=meta_D
        )
        self.metadata["diffusion_method"] = "parametric"
        self.metadata["diffusion_parametric_info"] = dict(res.info)

    def _refit_parametric_diffusion(self, support) -> None:
        """Re-fit the parametric diffusion on ``support`` (exact NLL), others at 0."""
        spec = dict(self._diffusion_parametric_spec)
        theta0 = jnp.asarray(self._diffusion_parametric_full["theta_D"])
        sup = np.asarray(support, dtype=int)
        theta0 = jnp.zeros_like(theta0).at[sup].set(theta0[sup])
        basis = spec.pop("basis")
        spec["theta_D0"] = theta0
        self.infer_diffusion(basis, _support=sup, **spec)

    ########################## ERROR ANALYSIS ###########################

    def compute_force_error(self):
        r"""
        Estimate sampling error for force inference.

        .. physics:: Force coefficient covariance & predicted error
           :label: force-error-covariance
           :category: Error analysis

           .. math::

              \operatorname{Cov}(C) = G^{-1},
                  \qquad
                  \mathbb{E}\!\left[\langle \delta F^\top A^{-1} \delta F \rangle\right]
                  = \operatorname{Tr}\!\left(G\,\operatorname{Cov}(C)\right),
              \qquad
              I_F = \tfrac{1}{2}\,C^\top M,
                  \qquad
                  \text{NMSE}_{F,\text{pred}}
                  = \frac{\operatorname{Tr}(G\cdot\operatorname{Cov}(C))}{C^\top M}
                  = \frac{\operatorname{Tr}(G\cdot\operatorname{Cov}(C))}{2 I_F}

           Assumes the sampling error dominates; measurement noise and
           discretization biases are not addressed.

        This method evaluates the covariance of the inferred force coefficients, the standard error,
        and computes the predicted normalized mean squared error (MSE) of the inferred force field. This analysis
        assumes that the sampling error dominates, and measurement noise or discretization biases are
        not explicitly addressed. It is common to OLI and ULI (by construction of the normal matrix G).

        Updates:
            self.force_coefficients_covariance (jnp.ndarray): Covariance matrix of the force coefficients.
            self.force_coefficients_stderr (jnp.ndarray): Standard error for each force coefficient.
            self.force_information (float): Estimated information content of the inferred force field.
            self.force_predicted_MSE (float): Predicted normalized mean squared error of the inferred force field.
            self.force_inferred.param_cov: The covariance in the full parameter order of the fitted
                PSF (pruned coefficients: zero rows and columns), for predict_var / predict_ci.

        """
        # Estimate the covariance of the force coefficients.
        # force_G is the NLL Hessian for all force methods:
        #   - linear:      G_ab = Σ_t dt b_a⊤ A⁻¹ b_b  (GLS Gram = Onsager–Machlup Hessian)
        #   - parametric:  G = Σ_t ψ_t⊤ Σ_t⁻¹ ψ_t       (Gauss–Newton NLL Hessian)
        #   - nonlinear:   G = H(NLL) from L-BFGS-B inverse Hessian
        # In all cases Cov(C) = G⁻¹ exactly (Fisher information bound).
        # Parametric EIV fit: the sandwich G_raw⁻¹HG_raw⁻ᵀ, and after
        # sparsify_force its support-restricted form, the inverse of
        # force_fisher_full[S, S] (see force_selection_metric).
        self.force_coefficients_covariance = self.force_G_pinv
        if self.force_coefficients_covariance is None:
            raise RuntimeError(
                "No force-coefficient covariance: the last force fit did not produce a "
                "curvature matrix (e.g. the parametric optimisation returned a non-finite "
                "loss — check theta0 and the fit report)."
            )

        # Calculate the standard error for each force coefficient
        self.force_coefficients_stderr = jnp.einsum("aa->a", self.force_coefficients_covariance) ** 0.5

        # Propagate into the existing InferenceResultSF (if present), in the order
        # of its (full-basis) parameter vector: off-support coefficients are exact zeros.
        if hasattr(self, "force_inferred") and self.force_inferred is not None:
            object.__setattr__(self.force_inferred, "param_cov",
                               self._embed_support_cov("force", self.force_coefficients_covariance))

        # Compute time-integrated squared error of the force field
        force_SSE = float(jnp.einsum("ab,ba", self.force_G, self.force_coefficients_covariance))

        # Compute normalized MSE
        if hasattr(self, "force_moments"):
            self.force_information = float(0.5 * self.force_coefficients_full @ self.force_moments)
            force_energy = 2.0 * self.force_information
        elif hasattr(self, "force_optimization_results_nonlinear"):
            self.force_information = -self.force_optimization_results_nonlinear["fun"]
            # For parametric/nonlinear, force_information = -NLL_min; energy = 2*I
            force_energy = 2.0 * self.force_information
        else:
            raise RuntimeError("Force information is unavailable. Run force inference before computing the error.")

        # Guard against empty support (zero inferred force energy → null model, MSE undefined)
        if force_energy == 0.0:
            self.force_predicted_MSE = float("nan")
        else:
            self.force_predicted_MSE = float(force_SSE / force_energy)

    def compute_diffusion_error(self):
        r"""
        Estimate sampling error for diffusion inference.

        Mirrors :meth:`compute_force_error` for the diffusion field.

        .. physics:: Diffusion coefficient covariance (sandwich estimator)
           :label: diffusion-error-sandwich
           :category: Error analysis

           The linear diffusion fit is an ordinary least-squares projection of
           a local estimator :math:`\hat D_t` onto the basis,
           :math:`\hat C = G^{-1} M` with :math:`G = \sum_t B_t^\top B_t` and
           :math:`M = \sum_t B_t^\top \hat D_t`.  Its covariance is the
           sandwich

           .. math::

              \operatorname{Cov}(\hat C) = G^{-1} S\, G^{-1}, \qquad
              S = \sum_{|k| \le K} \sum_t s_t\, s_{t+k}^\top, \qquad
              s_t = B_t^\top \bigl(\hat D_t - B_t \hat C\bigr),

           where the scores :math:`s_t` are summed over particles.  The local
           estimators use a few consecutive frames, so scores whose stencils
           share a frame are correlated (through shared increments, or a
           shared localization error); :math:`K` is the stencil's frame span,
           read from the streams the estimator requires — scores further apart
           share no frame and are uncorrelated.  No noise model is assumed:
           state-dependent diffusion, the estimator's own variance and its
           serial correlation all enter through the empirical scores.

        For a parametric diffusion fit (:meth:`infer_diffusion`) the
        covariance is the inverse observed Fisher information — the inverse
        Hessian of the banded NLL at :math:`\hat\theta_D`, with the force and
        :math:`\Lambda` held at their fitted values.  After
        :meth:`sparsify_diffusion` it is the covariance of the selected
        support's estimator, :math:`\mathcal F_{BB}^{-1}` (see there).

        Updates
        -------
        self.diffusion_coefficients_covariance : jnp.ndarray
            Covariance matrix of the diffusion coefficients (on the support).
        self.diffusion_coefficients_stderr : jnp.ndarray
            Standard error for each diffusion coefficient.
        self.diffusion_information : float
            Estimated information content of the inferred diffusion field.
        self.diffusion_predicted_MSE : float
            Predicted normalized mean squared error.
        """
        diffusion_method = self.metadata.get("diffusion_method", "linear")
        sup = np.asarray(self.diffusion_support, dtype=int)
        if diffusion_method == "parametric":
            # Inverse observed Fisher information of the banded NLL on the
            # support (force and Λ held at their fitted values).
            fisher = getattr(self, "_diffusion_parametric_current", None)
            if fisher is None:
                raise RuntimeError("The parametric diffusion fit carries no NLL Hessian.")
            F_sup = jnp.asarray(fisher["hessian"])[jnp.ix_(sup, sup)]
            cov = _spd_inverse(F_sup, "diffusion NLL Hessian")
        elif getattr(self, "diffusion_fisher_full", None) is not None:
            # After sparsify_diffusion: the efficient restricted estimator's
            # covariance in the selection metric (= the sandwich on the full basis).
            F_sup = jnp.asarray(self.diffusion_fisher_full)[jnp.ix_(sup, sup)]
            cov = _spd_inverse(F_sup, "diffusion Fisher metric")
        else:
            if not hasattr(self, "diffusion_G_pinv") or self.diffusion_G_pinv is None:
                raise RuntimeError(
                    "Diffusion Gram inverse is unavailable. Run infer_diffusion_linear() before computing the error."
                )
            S = self._diffusion_score_covariance()
            S = jnp.asarray(S)[jnp.ix_(sup, sup)]
            Gp = self.diffusion_G_pinv
            cov = Gp @ S @ Gp.T
            F_sup = None
        self.diffusion_coefficients_covariance = 0.5 * (cov + cov.T)
        self.diffusion_coefficients_cov = self.diffusion_coefficients_covariance

        self.diffusion_coefficients_stderr = jnp.einsum("aa->a", self.diffusion_coefficients_covariance) ** 0.5

        # Propagate into the existing InferenceResultSF, in the order of its
        # (full-basis) parameter vector: off-support coefficients are exact zeros.
        if hasattr(self, "diffusion_inferred") and self.diffusion_inferred is not None:
            object.__setattr__(self.diffusion_inferred, "param_cov",
                               self._embed_support_cov("diffusion", self.diffusion_coefficients_covariance))

        if diffusion_method == "parametric":
            # Fisher-metric analogue of the linear prediction (as for the
            # parametric force): Tr(F Cov) / (θᵀ F θ) on the support.
            theta = jnp.asarray(self.diffusion_coefficients_full)[sup]
            diffusion_SSE = float(jnp.einsum("ab,ba", F_sup, self.diffusion_coefficients_covariance))
            self.diffusion_information = float(0.5 * theta @ F_sup @ theta)
            diffusion_energy = 2.0 * self.diffusion_information
        else:
            diffusion_SSE = float(jnp.einsum("ab,ba", self.diffusion_G, self.diffusion_coefficients_covariance))
            self.diffusion_information = float(0.5 * self.diffusion_coefficients_full @ self.diffusion_moments)
            diffusion_energy = 2.0 * self.diffusion_information

        if diffusion_energy == 0.0:
            self.diffusion_predicted_MSE = float("nan")
        else:
            self.diffusion_predicted_MSE = float(diffusion_SSE / diffusion_energy)

    def _diffusion_score_covariance(self, coefficients=None) -> np.ndarray:
        """Long-run covariance ``S`` of the diffusion-fit scores (full basis).

        See :meth:`compute_diffusion_error`.  Rows are evaluated with the same
        operands, masks and dataset weights as the diffusion moments; lagged
        products are taken within each dataset, between frames ``K`` apart at
        most, ``K`` being the frame span of the estimator's stencil.  The
        residuals use ``coefficients`` (full basis; default: the current fit).
        """
        from SFI.integrate.api import integrate_rows, shared_rows
        from SFI.trajectory.dataset import TrajectoryDataset

        D_op = self.get_diffusion_timeop(self.__diffusion_M_mode__)
        B = self._diffusion_basis_operand()
        if coefficients is None:
            coefficients = self.diffusion_coefficients_full
        C = ConstOperand(jnp.asarray(coefficients), alias="Cd")
        prog = Integrand(
            exprs=[B],
            times=[D_op],
            consts=[C],
            terms=[
                Term(eq="imna,imn->ia", ops=("B", D_op.alias)),
                Term(eq="imna,imnc,c->ia", ops=("B", "B", "Cd"), scale=-1.0),
            ],
        )
        amin, amax = TrajectoryDataset._required_offsets(set(prog.require()))
        K = int(amax - amin)

        # Same row set as the fit's Gram and moments (see ``shared_rows``).
        with self._structural_scope(self.diffusion_basis), shared_rows(getattr(self, "_diffusion_row_frames", ())):
            rows = integrate_rows(
                self.data, prog, weight_by_dt=False, chunk_target_bytes=self._chunk_target_bytes
            )

        p = int(np.asarray(self.diffusion_coefficients_full).shape[0])
        S = np.zeros((p, p))
        for t_idx, Y in rows.values():
            Y = np.asarray(Y, dtype=float).reshape(len(t_idx), p)
            # Scatter onto the frame grid so lags are frame distances (gaps
            # and masked frames contribute zero rows).
            t0 = int(t_idx.min()) if len(t_idx) else 0
            grid = np.zeros((int(t_idx.max()) - t0 + 1 if len(t_idx) else 0, p))
            grid[np.asarray(t_idx) - t0] = Y
            S += grid.T @ grid
            for k in range(1, K + 1):
                if k >= grid.shape[0]:
                    break
                L = grid[:-k].T @ grid[k:]
                S += L + L.T
        return S

    def diagnose(self, *, level: str = "standard", **kwargs):
        """Run the consistency-check suite from :mod:`SFI.diagnostics`.

        Convenience wrapper for :func:`SFI.diagnostics.assess`. See its
        docstring for the available ``level`` presets.
        """
        from SFI.diagnostics import assess

        return assess(self, level=level, **kwargs)

    def holdout_score(self, data, *, require_error: bool = False) -> dict:
        """Held-out NMSE of the fitted force on an independent collection.

        A *side feature for data-abundant scenarios*: SFI estimates its
        own accuracy from the training data (``force_predicted_MSE``)
        and validates fits through the diagnostics suite, neither of
        which costs any data.  Reach for an explicit train/test split
        (:meth:`TrajectoryCollection.split_time
        <SFI.trajectory.TrajectoryCollection.split_time>`) only when
        data is plentiful, or to confirm a suspected bias floor: a
        ``ratio`` near 1 means the fit is sampling-limited, a ratio
        ``≫ 1`` means a bias floor (often measurement noise — see the
        noise-and-sampling guide).

        The score is the χ² excess of the residuals of ``force_inferred``
        on ``data`` over their expected noise level, in NMSE units (see
        :func:`~SFI.diagnostics.residual_tests.mse_consistency`): a scale,
        not a precision instrument — its resolution is set by the χ²
        fluctuations of the residuals, and since the finite-Δt expectation
        uses the fitted force, a force error moves it with either sign.

        Parameters
        ----------
        data : TrajectoryCollection
            Independent test data (e.g. the second half of
            ``coll.split_time(0.8)``).
        require_error : bool
            If True, run :meth:`compute_force_error` first when the
            predicted error is missing, so ``ratio`` is always defined.

        Returns
        -------
        dict
            ``{"holdout_NMSE", "predicted_NMSE", "ratio", "excess_z",
            "n_obs"}``.  Also stored as ``self.force_holdout_NMSE``.

        Notes
        -----
        Bases that read *time-dependent* extras are not supported on the
        held-out path (the residual builders pass extras unsliced).
        """
        from SFI.diagnostics.residual_tests import mse_consistency
        from SFI.diagnostics.residuals import build_residuals

        if require_error and getattr(self, "force_predicted_MSE", None) is None:
            self.compute_force_error()

        bundle = build_residuals(self, data=data)
        out = mse_consistency(self, bundle)
        result = {
            "holdout_NMSE": out.get("realised_NMSE"),
            "predicted_NMSE": out.get("predicted_NMSE"),
            "ratio": out.get("ratio"),
            "excess_z": out.get("excess_z"),
            "n_obs": bundle.n_obs,
        }
        self.force_holdout_NMSE = result["holdout_NMSE"]
        return result

    def print_report(self):
        """
        Print a summary report of the inference results.

        Provides insights into the inferred diffusion and force fields, along with error metrics
        such as sampling error, trajectory length, discretization bias, and measurement noise.
        """
        print("\n  --- StochasticForceInference Report --- ")

        # Average diffusion tensor
        print("Average diffusion tensor:\n", self.diffusion_average)

        # Measurement noise tensor
        print("Measurement noise tensor:\n", self.Lambda)

        # Entropy production
        if hasattr(self, "DeltaS"):
            print(
                "Entropy production (total Delta S, estimated error):",
                self.DeltaS,
                self.error_DeltaS,
            )
            if hasattr(self, "DeltaS_debiased"):
                print(
                    "Entropy production, fluctuation-bias subtracted (AIC):",
                    self.DeltaS_debiased,
                )

        # Force inference metrics
        if hasattr(self, "force_predicted_MSE"):
            print("Force estimated information:", self.force_information)
            print(
                "Force: estimated normalized mean squared error (sampling only):",
                self.force_predicted_MSE,
            )
            # To add: bias estimates
        if "force_gram_cond" in self.metadata:
            print(f"Force Gram condition number: {self.metadata['force_gram_cond']:.3g}")
        issue = self._noise_consistency_issue()
        if issue:
            print("WARNING - inconsistent measurement-noise treatment:", issue)

        # Diffusion error metrics
        if hasattr(self, "diffusion_predicted_MSE"):
            print("Diffusion estimated information:", self.diffusion_information)
            print(
                "Diffusion: estimated normalized mean squared error (sampling only):",
                self.diffusion_predicted_MSE,
            )
            # To add: bias estimates

        # Exact-comparison NMSE (set by compare_to_exact)
        if hasattr(self, "NMSE_force"):
            print(f"Normalized MSE (force):     {self.NMSE_force:.4f}")
        if hasattr(self, "NMSE_diffusion"):
            print(f"Normalized MSE (diffusion): {self.NMSE_diffusion:.4f}")

        # Force coefficient table
        if hasattr(self, "force_coefficients_full"):
            print()
            print(self.summary("force"))

    def summary(self, field: str = "force") -> str:
        """
        Return a formatted coefficient table for the inferred model.

        Parameters
        ----------
        field : ``"force"`` or ``"diffusion"``
            Which inferred field to summarize.

        Returns
        -------
        str
            Multi-line table ready for ``print()``.
        """
        import numpy as np

        from SFI.utils.formatting import model_summary

        if field == "force":
            labels = getattr(self, "force_basis_labels", None)
            coeffs = getattr(self, "force_coefficients_full", None)
            stderr = getattr(self, "force_coefficients_stderr", None)
            support = getattr(self, "force_support", None)
            title = "Force Coefficient Table"
        elif field == "diffusion":
            labels = getattr(self, "diffusion_basis_labels", None)
            coeffs = getattr(self, "diffusion_coefficients_full", None)
            stderr = getattr(self, "diffusion_coefficients_stderr", None)
            support = getattr(self, "diffusion_support", None)
            title = "Diffusion Coefficient Table"
        else:
            raise ValueError(f"Unknown field {field!r}; expected 'force' or 'diffusion'.")

        if coeffs is None:
            return f"  (No {field} coefficients available.)"

        coeffs = np.asarray(coeffs)
        n = coeffs.shape[0]

        auto_labels = labels is None
        if auto_labels:
            labels = [f"b{j}" for j in range(n)]

        if stderr is not None:
            stderr = np.asarray(stderr)

        # support may be full or sparse
        support_arr = None
        if support is not None:
            support_arr = np.asarray(support)
            if support_arr.shape[0] == n:
                support_arr = None  # full support, do not highlight

        return model_summary(
            labels,
            coeffs,
            stderr=stderr,
            support=support_arr,
            title=title,
            auto_labels=auto_labels,
        )

    def report_dict(self) -> dict:
        """Return a structured summary of inference results as a dictionary.

        This is the machine-readable counterpart of :meth:`print_report`.
        All values are plain Python scalars or numpy arrays (no JAX arrays).

        Returns
        -------
        dict
            Keys include ``"diffusion_average"``, ``"Lambda"``,
            ``"force_information"``, ``"force_predicted_MSE"``,
            ``"NMSE_force"``, ``"NMSE_diffusion"``, and others
            when available. Missing quantities are omitted.
        """
        import numpy as np

        d: dict = {}
        d["metadata"] = dict(self.metadata)

        if hasattr(self, "diffusion_average"):
            d["diffusion_average"] = np.asarray(self.diffusion_average)
        if hasattr(self, "Lambda"):
            d["Lambda"] = np.asarray(self.Lambda)

        if hasattr(self, "DeltaS"):
            d["DeltaS"] = float(self.DeltaS)
            d["error_DeltaS"] = float(self.error_DeltaS)
            if hasattr(self, "DeltaS_debiased"):
                d["DeltaS_debiased"] = float(self.DeltaS_debiased)

        if hasattr(self, "force_coefficients"):
            d["force_coefficients"] = np.asarray(self.force_coefficients)
        if hasattr(self, "force_coefficients_full"):
            d["force_coefficients_full"] = np.asarray(self.force_coefficients_full)
        if hasattr(self, "force_support"):
            d["force_support"] = np.asarray(self.force_support)
        if hasattr(self, "force_coefficients_stderr"):
            d["force_coefficients_stderr"] = np.asarray(self.force_coefficients_stderr)
        if hasattr(self, "force_information"):
            d["force_information"] = float(self.force_information)
        if hasattr(self, "force_predicted_MSE"):
            d["force_predicted_MSE"] = float(self.force_predicted_MSE)
        if hasattr(self, "NMSE_force"):
            d["NMSE_force"] = float(self.NMSE_force)
        if hasattr(self, "MSE_force"):
            d["MSE_force"] = float(self.MSE_force)

        if hasattr(self, "diffusion_coefficients"):
            d["diffusion_coefficients"] = np.asarray(self.diffusion_coefficients)
        if hasattr(self, "diffusion_information"):
            d["diffusion_information"] = float(self.diffusion_information)
        if hasattr(self, "diffusion_predicted_MSE"):
            d["diffusion_predicted_MSE"] = float(self.diffusion_predicted_MSE)
        if hasattr(self, "NMSE_diffusion"):
            d["NMSE_diffusion"] = float(self.NMSE_diffusion)
        if hasattr(self, "MSE_diffusion"):
            d["MSE_diffusion"] = float(self.MSE_diffusion)

        return d

    def compare_to_exact(
        self,
        *,
        model_exact=None,
        data_exact=None,
        force_exact=None,
        diffusion_exact=None,  # callable | float | (d,d) array
        maxpoints: int = 1000,
    ) -> None:
        r"""
        Compare inferred vs exact using dt-weighted time means via the integrate() API.

        This function evaluates the inferred force/diffusion against "exact" references
        on a (possibly exact/synthetic) dataset. It updates:

            self.MSE_force / self.NMSE_force
            self.MSE_diffusion / self.NMSE_diffusion

        Inputs: exact references
        ~~~~~~~~~~~~~~~~~~~~~~~~
        You can provide exact references in two ways:

        1) Preferred: `model_exact`
           A model object (from SFI.langevin submodule) exposing:

                             - model_exact.force_sf : exact force/drift (SF/StateExpr-like)
                             - model_exact.diffusion_sf : exact diffusion (SF/StateExpr-like)
                                 OR a constant (float or (d,d) matrix) via ``model_exact.D``

        2) Explicit: `force_exact`, `diffusion_exact`
           - `force_exact`: SF/StateExpr-like callable returning (N, d)
           - `diffusion_exact`:

                * callable returning (N, d, d), OR
                * float meaning σ·I, OR
                * (d,d) matrix constant diffusion.

           These are used if `model_exact` is not provided. If `model_exact` is provided,
           its members take precedence unless they are missing, in which case the explicit
           arguments can be used as fallback.

        Velocity provisioning (underdamped)
        -----------------------------------
        If an evaluated expression advertises `needs_v=True`, this routine supplies:

            v := dX/dt   (secant velocity from the data stream)

        i.e. it uses `velocity("dX", "dt")` as the `v=...` keyword argument. This works for both
        exact and inferred expressions and keeps underdamped comparisons possible even when
        the dataset only stores positions.

        Metrics
        -------
        Force:
            e = Fe - Fh
            MSE_force  = < e^T A_inv e >
            NMSE_force = MSE_force / < Fh^T A_inv Fh >

        Diffusion:
            E = De - Dh
            MSE_diffusion  = < tr(A_inv E A_inv E) >

        .. physics:: Normalized MSE metrics (force & diffusion)
           :label: nmse-metrics
           :category: Error analysis

           .. math::

              \text{NMSE}_F
              = \frac{\langle (F_{\text{exact}} - \hat F)^\top A^{-1}
                (F_{\text{exact}} - \hat F) \rangle}
              {\langle \hat F^\top A^{-1} \hat F \rangle}

           .. math::

              \text{NMSE}_D
              = \frac{\langle \operatorname{tr}(A^{-1} E\, A^{-1} E) \rangle}
              {\langle \operatorname{tr}(A^{-1} \hat D\, A^{-1} \hat D) \rangle}

           where :math:`E = D_{\text{exact}} - \hat D`.
            NMSE_diffusion = MSE_diffusion / < tr(A_inv Dh A_inv Dh) >

        where A_inv is `self.A_inv` (typically (2 D̄)^{-1} from the inferred constant diffusion normalization).

        Subsampling
        -----------
        Uses a simple subsampling heuristic so that the total number of evaluated points is ~<= `maxpoints`,
        accounting for the maximum number of particles.

        Requirements
        ------------
        - `self.A_inv` must exist (run compute_diffusion_constant() or otherwise set A_inv).
        - The dataset must provide streams `X`, `dt`, and if any evaluated expr needs v: `dX` as well.
        """
        data_exact = data_exact or self.data

        # ---------- resolve exact references ----------
        if model_exact is not None:
            F_exact = (
                getattr(model_exact, "force_sf", None)
                if force_exact is None
                else getattr(model_exact, "force_sf", force_exact)
            )
            D_exact = (
                getattr(model_exact, "diffusion_sf", None)
                if diffusion_exact is None
                else getattr(model_exact, "diffusion_sf", diffusion_exact)
            )
            # Fall back to model_exact.D when the bound SF is absent (e.g. constant
            # diffusion in OverdampedProcess sets _D_sf=None but stores the raw value
            # in the .D field).
            if D_exact is None:
                D_exact = diffusion_exact if diffusion_exact is not None else getattr(model_exact, "D", None)
        else:
            F_exact = force_exact
            D_exact = diffusion_exact

        if not hasattr(self, "A_inv"):
            raise RuntimeError("A_inv not available. Run compute_diffusion_constant() (or equivalent) first.")

        nsteps = int(getattr(data_exact, "Nsteps", 0) or 0)
        nmaxp = int(getattr(data_exact, "Nmaxparticles", 1) or 1)
        subsampling = max(1, nsteps // max(1, maxpoints // max(1, nmaxp))) if nsteps else 1

        logger.info("Comparing to exact data...")

        A = ConstOperand(jnp.asarray(self.A_inv), alias="A")
        d = int(jnp.asarray(self.A_inv).shape[0])

        # Helper: provide v only if the expression wants it.
        def _maybe_v(expr):
            return velocity("dX", "dt") if bool(getattr(expr, "needs_v", False)) else None

        # Helper: wrap constant diffusion into a TimeOperand returning (N,d,d)
        def _const_diffusion_operand(Dconst, *, alias: str):
            Dconst = jnp.asarray(Dconst)
            if Dconst.ndim == 0:
                # scalar σ -> σ I
                sigma = float(Dconst)
                Dmat = sigma * jnp.eye(d, dtype=jnp.asarray(self.A_inv).dtype)
            elif Dconst.ndim == 2:
                Dmat = Dconst
            else:
                raise TypeError("Constant diffusion must be a float (σ) or a (d,d) matrix.")

            @timeop(name=f"D_const_{alias}")
            def _Dconst(**streams):
                N = streams["X"].shape[0]
                return jnp.broadcast_to(Dmat[None, :, :], (N, Dmat.shape[0], Dmat.shape[1]))

            _Dconst._requires = frozenset({"X"})  # type: ignore[attr-defined]

            return TimeOperand(_Dconst, alias=alias)

        # Helper: build diffusion operand (ExprOperand or TimeOperand) with alias control.
        def _diffusion_operand(Dobj, *, alias: str):
            # NoiseModel instances (e.g. ConservedNoise) are not callable exprs
            # and cannot be converted to JAX arrays — return None so the caller
            # skips the (point-wise) diffusion comparison for them.
            from SFI.langevin.noise import NoiseModel

            if isinstance(Dobj, NoiseModel):
                return None
            if callable(Dobj):
                vD = _maybe_v(Dobj)
                return ExprOperand(expr=Dobj, x=stream("X"), v=vD, alias=alias)
            return _const_diffusion_operand(Dobj, alias=alias)

        # ------------------------------ FORCE ------------------------------ #
        if hasattr(self, "force_inferred") and (F_exact is not None):
            if not callable(F_exact):
                raise TypeError("Exact force must be callable (SF/StateExpr-like).")

            Fh = getattr(self.force_inferred, "sf", self.force_inferred)

            # Structural extras needed by expr graphs are built into a transient
            # collection (never persisted). Only StateExpr-like objects have them;
            # plain callables are skipped (the `root` attribute marks StateExprs).
            force_exprs = [e for e in (F_exact, Fh) if hasattr(e, "root")]
            if force_exprs:
                data_exact = prepare_collection_for_expr(data_exact, *force_exprs)

            Fe_op = ExprOperand(expr=F_exact, x=stream("X"), v=_maybe_v(F_exact), alias="Fe")
            Fh_op = ExprOperand(expr=Fh, x=stream("X"), v=_maybe_v(Fh), alias="Fh")

            # Numerator: ⟨(Fe−Fh)^T A (Fe−Fh)⟩
            num_prog = Integrand(
                exprs=[Fe_op, Fh_op],
                consts=[A],
                terms=[
                    Term(eq="im,mn,in->i", ops=("Fe", "A", "Fe"), scale=+1.0),
                    Term(eq="im,mn,in->i", ops=("Fe", "A", "Fh"), scale=-2.0),
                    Term(eq="im,mn,in->i", ops=("Fh", "A", "Fh"), scale=+1.0),
                ],
            )
            # Denominator: ⟨Fh^T A Fh⟩
            den_prog = Integrand(
                exprs=[Fh_op],
                consts=[A],
                terms=[Term(eq="im,mn,in->i", ops=("Fh", "A", "Fh"))],
            )

            num = integrate(
                data_exact,
                num_prog,
                reduce="mean",
                reduce_over_particles=True,
                subsampling=subsampling,
                chunk_target_bytes=self._chunk_target_bytes,
            )
            den = integrate(
                data_exact,
                den_prog,
                reduce="mean",
                reduce_over_particles=True,
                subsampling=subsampling,
                chunk_target_bytes=self._chunk_target_bytes,
            )

            self.MSE_force = num
            self.NMSE_force = num / (den + 1e-12)

            logger.info("Normalized MSE (force): %s", self.NMSE_force)

        # ------------------------------ DIFFUSION ------------------------------ #
        if D_exact is not None:
            # Inferred diffusion: prefer callable diffusion_inferred if available, else constant diffusion_average.
            Dh_obj = None
            if hasattr(self, "diffusion_inferred"):
                Dh_candidate = getattr(self.diffusion_inferred, "sf", self.diffusion_inferred)
                if callable(Dh_candidate):
                    Dh_obj = Dh_candidate

            if Dh_obj is None:
                if hasattr(self, "diffusion_average"):
                    Dh_obj = jnp.asarray(self.diffusion_average)
                else:
                    raise RuntimeError("No inferred diffusion callable and no diffusion_average available.")

            # Build operands (supports callable OR constant for both exact and inferred).
            De_op = _diffusion_operand(D_exact, alias="De")
            Dh_op = _diffusion_operand(Dh_obj, alias="Dh")

            if De_op is None or Dh_op is None:
                logger.info(
                    "Skipping diffusion NMSE: exact diffusion is a NoiseModel "
                    "(e.g. ConservedNoise) that cannot be compared point-wise."
                )
            else:
                # Prepare structural extras only for callable exprs with node trees.
                diff_exprs = []
                if isinstance(De_op, ExprOperand) and hasattr(D_exact, "root"):
                    diff_exprs.append(D_exact)
                if isinstance(Dh_op, ExprOperand) and hasattr(Dh_obj, "root"):
                    diff_exprs.append(Dh_obj)
                if diff_exprs:
                    data_exact = prepare_collection_for_expr(data_exact, *diff_exprs)

                exprs_num = [op for op in (De_op, Dh_op) if isinstance(op, ExprOperand)]
                times_num = [op for op in (De_op, Dh_op) if isinstance(op, TimeOperand)]
                exprs_den = [Dh_op] if isinstance(Dh_op, ExprOperand) else []
                times_den = [Dh_op] if isinstance(Dh_op, TimeOperand) else []

                # Numerator: ⟨ tr(A (De−Dh) A (De−Dh)) ⟩
                # Expanded form with contractions using eq="imn,iop,no,pm->i".
                num_prog = Integrand(
                    exprs=exprs_num,
                    times=times_num,
                    consts=[A],
                    terms=[
                        Term(eq="imn,iop,no,pm->i", ops=("De", "De", "A", "A"), scale=+1.0),
                        Term(eq="imn,iop,no,pm->i", ops=("De", "Dh", "A", "A"), scale=-2.0),
                        Term(eq="imn,iop,no,pm->i", ops=("Dh", "Dh", "A", "A"), scale=+1.0),
                    ],
                )
                den_prog = Integrand(
                    exprs=exprs_den,
                    times=times_den,
                    consts=[A],
                    terms=[Term(eq="imn,iop,no,pm->i", ops=("Dh", "Dh", "A", "A"))],
                )

                num = integrate(
                    data_exact,
                    num_prog,
                    reduce="mean",
                    reduce_over_particles=True,
                    subsampling=subsampling,
                    chunk_target_bytes=self._chunk_target_bytes,
                )
                den = integrate(
                    data_exact,
                    den_prog,
                    reduce="mean",
                    reduce_over_particles=True,
                    subsampling=subsampling,
                    chunk_target_bytes=self._chunk_target_bytes,
                )

                self.MSE_diffusion = num
                self.NMSE_diffusion = num / (den + 1e-12)

                logger.info("Normalized MSE (diffusion): %s", self.NMSE_diffusion)

    # ------------------------------------------------------------------
    # Exact-vs-inferred sample arrays + scatter (graphical comparison)
    # ------------------------------------------------------------------
    def _comparison_points(self, data=None, *, maxpoints: int = 2000, need_v: bool = False):
        """Subsample ``(X_flat, V_flat|None, mask_flat)`` to ~``maxpoints`` points."""
        import numpy as np

        data = data if data is not None else self.data
        t, X, M = data.to_arrays(dataset=0)
        X = np.asarray(X)
        M = np.asarray(M)
        T, _, d = X.shape
        stride = max(1, T // max(1, maxpoints // max(1, X.shape[1])))
        Xs, Ms = X[::stride], M[::stride]
        Vf = None
        if need_v:
            from SFI.utils.maths import fd_velocity

            dt = np.diff(np.asarray(t, dtype=float))
            Vf = np.asarray(fd_velocity(X, dt))[::stride].reshape(-1, d)
        return Xs.reshape(-1, d), Vf, Ms.reshape(-1).astype(bool)

    def _eval_on_points(self, field, Xf, Vf):
        """Evaluate a callable force/field on flat points, supplying ``v`` when needed."""
        import jax.numpy as jnp
        import numpy as np

        fn = getattr(field, "sf", field)
        needs_v = bool(getattr(field, "needs_v", False))
        if needs_v and Vf is not None:
            out = fn(jnp.asarray(Xf), v=jnp.asarray(Vf))
        else:
            out = fn(jnp.asarray(Xf))
        return np.asarray(out)

    def _eval_diffusion_on_points(self, Dobj, Xf, Vf):
        """Evaluate a diffusion field (callable) or broadcast a constant to ``(M, d, d)``."""
        import numpy as np

        if callable(getattr(Dobj, "sf", Dobj)):
            return self._eval_on_points(Dobj, Xf, Vf)
        Dc = np.asarray(Dobj)
        d = Xf.shape[1]
        if Dc.ndim == 0:
            Dmat = float(Dc) * np.eye(d)
        elif Dc.ndim == 2:
            Dmat = Dc
        else:
            raise TypeError("Constant diffusion must be a scalar or a (d, d) matrix.")
        return np.broadcast_to(Dmat[None, :, :], (Xf.shape[0], d, d))

    def force_comparison_arrays(self, *, model_exact=None, force_exact=None, data=None, maxpoints: int = 2000):
        """Return ``(F_exact, F_inferred)`` evaluated along the trajectory.

        Evaluates the exact and inferred force on the (subsampled, masked)
        trajectory points, supplying finite-difference velocities for
        underdamped fields.  Feeds :meth:`comparison_scatter`; also handy
        for custom diagnostics.

        Parameters
        ----------
        model_exact :
            Object exposing ``force_sf`` (e.g. an ``OverdampedProcess``).
        force_exact :
            Explicit callable exact force (overrides ``model_exact``).
        data :
            Collection to evaluate on (default: the training data).
        maxpoints :
            Approximate number of points to evaluate.

        Returns
        -------
        (F_exact, F_inferred) : tuple of ndarray, shape ``(n_points, d)``
        """
        F_exact = force_exact
        if F_exact is None and model_exact is not None:
            F_exact = getattr(model_exact, "force_sf", None)
        if F_exact is None or not callable(F_exact):
            raise ValueError(
                "force_comparison_arrays needs a callable exact force "
                "(model_exact.force_sf or force_exact=)."
            )
        if not hasattr(self, "force_inferred"):
            raise RuntimeError("No inferred force; run infer_force_linear / infer_force first.")
        need_v = bool(getattr(F_exact, "needs_v", False)) or bool(
            getattr(self.force_inferred, "needs_v", False)
        )
        Xf, Vf, mask = self._comparison_points(data, maxpoints=maxpoints, need_v=need_v)
        Fe = self._eval_on_points(F_exact, Xf, Vf)[mask]
        Fi = self._eval_on_points(self.force_inferred, Xf, Vf)[mask]
        return Fe, Fi

    def diffusion_comparison_arrays(self, *, model_exact=None, diffusion_exact=None, data=None, maxpoints: int = 2000):
        """Return ``(D_exact, D_inferred)`` evaluated along the trajectory.

        Like :meth:`force_comparison_arrays` but for the diffusion field;
        a constant exact/inferred diffusion is broadcast to ``(n_points,
        d, d)``.
        """
        import jax.numpy as jnp

        D_exact = diffusion_exact
        if D_exact is None and model_exact is not None:
            D_exact = getattr(model_exact, "diffusion_sf", None)
            if D_exact is None:
                D_exact = getattr(model_exact, "D", None)
        if D_exact is None:
            raise ValueError(
                "diffusion_comparison_arrays needs an exact diffusion "
                "(model_exact or diffusion_exact=)."
            )
        D_inf = getattr(self, "diffusion_inferred", None)
        if D_inf is None or not callable(getattr(D_inf, "sf", D_inf)):
            if not hasattr(self, "diffusion_average"):
                raise RuntimeError("No inferred diffusion callable and no diffusion_average available.")
            D_inf = jnp.asarray(self.diffusion_average)
        need_v = bool(getattr(D_exact, "needs_v", False)) or bool(getattr(D_inf, "needs_v", False))
        Xf, Vf, mask = self._comparison_points(data, maxpoints=maxpoints, need_v=need_v)
        De = self._eval_diffusion_on_points(D_exact, Xf, Vf)[mask]
        Di = self._eval_diffusion_on_points(D_inf, Xf, Vf)[mask]
        return De, Di

    def comparison_scatter(self, *, model_exact=None, field: str = "force", data=None, ax=None, maxpoints: int = 2000, **plot_kw):
        """Scatter inferred-vs-exact force (or diffusion) along the trajectory.

        Evaluates both fields on the data with
        :meth:`force_comparison_arrays` / :meth:`diffusion_comparison_arrays`
        and renders them with
        :func:`SFI.utils.plotting.comparison_scatter` (identity line +
        Pearson ``r`` + MSE).  Replaces hand-rolled exact-vs-inferred
        scatters in demos.

        Parameters
        ----------
        model_exact :
            Object exposing ``force_sf`` / ``diffusion_sf`` / ``D``.
        field : {"force", "diffusion"}
            Which field to compare.
        data :
            Collection to evaluate on (default: training data).
        ax :
            Target axes (default: current axes).
        maxpoints :
            Approximate number of points to evaluate.
        **plot_kw :
            Forwarded to :func:`SFI.utils.plotting.comparison_scatter`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        from SFI.utils import plotting

        if field == "force":
            exact, inferred = self.force_comparison_arrays(model_exact=model_exact, data=data, maxpoints=maxpoints)
        elif field == "diffusion":
            exact, inferred = self.diffusion_comparison_arrays(model_exact=model_exact, data=data, maxpoints=maxpoints)
        else:
            raise ValueError(f"Unknown field {field!r}; expected 'force' or 'diffusion'.")
        if ax is not None:
            plt.sca(ax)
        plotting.comparison_scatter(exact, inferred, **plot_kw)
        return plt.gca()

    def compare_params_to_exact(self, theta_true, *, psf=None) -> dict:
        """Compare inferred parametric coefficients to known ground truth.

        For a model fitted with a parametric family, returns a per-parameter
        dict of absolute and relative error.  ``theta_true`` may be a flat
        array (compared elementwise to ``force_coefficients_full``) or a
        ``{name: value}`` dict (unflattened from the fitted coefficients via
        ``psf.unflatten_params``, falling back to ``self.force_psf``).

        Returns
        -------
        dict
            ``{name: {"true", "inferred", "abs_error", "rel_error"}}``; also
            stored as ``self.parameter_comparison``.
        """
        import numpy as np

        theta_inf = np.asarray(self.force_coefficients_full).ravel()
        out: dict = {}
        if isinstance(theta_true, dict):
            psf = psf if psf is not None else getattr(self, "force_psf", None)
            inf_dict: dict = {}
            if psf is not None and hasattr(psf, "unflatten_params"):
                try:
                    inf_dict = {k: np.asarray(v) for k, v in dict(psf.unflatten_params(theta_inf)).items()}
                except Exception:
                    inf_dict = {}
            for name, tv in theta_true.items():
                tv = np.asarray(tv, dtype=float)
                iv = np.asarray(inf_dict.get(name, np.full(tv.shape, np.nan)), dtype=float)
                abs_err = float(np.sqrt(np.mean((iv - tv) ** 2))) if iv.shape == tv.shape else float("nan")
                rel = abs_err / (float(np.sqrt(np.mean(tv**2))) + 1e-12)
                out[name] = {"true": tv, "inferred": iv, "abs_error": abs_err, "rel_error": rel}
        else:
            tv = np.asarray(theta_true, dtype=float).ravel()
            iv = theta_inf[: tv.shape[0]]
            err = float(np.linalg.norm(iv - tv))
            out["theta"] = {
                "true": tv,
                "inferred": iv,
                "abs_error": err,
                "rel_error": err / (float(np.linalg.norm(tv)) + 1e-12),
            }
        self.parameter_comparison = out
        return out

    def coeff_block(self, block, *, field: str = "force"):
        """Return the coefficient (and covariance) slice for a basis sub-block.

        Compound bases (e.g. a multi-kernel or time-Fourier library) pack
        several conceptual blocks into one flat coefficient vector.  This
        returns the slice for one block without hand-computed offsets.

        Parameters
        ----------
        block :
            ``(start, stop)`` indices, a ``slice``, an ``int``, or a
            ``Basis`` (located by matching its labels as a contiguous run of
            the fitted basis labels).
        field : {"force", "diffusion"}

        Returns
        -------
        (coeffs, cov) : tuple
            ``coeffs`` is the 1-D slice; ``cov`` is the matching covariance
            block (or ``None`` if no covariance is available).
        """
        import numpy as np

        coeffs = np.asarray(getattr(self, f"{field}_coefficients_full"))
        cov = getattr(self, f"{field}_coefficients_covariance", None)
        cov = np.asarray(cov) if cov is not None else None
        if cov is not None and cov.shape[0] != coeffs.shape[0]:
            # After sparsification the covariance lives on the support only:
            # embed it in the full basis (pruned coefficients are exact zeros).
            support = np.asarray(getattr(self, f"{field}_support"), dtype=int)
            full = np.zeros((coeffs.shape[0], coeffs.shape[0]), dtype=cov.dtype)
            full[np.ix_(support, support)] = cov
            cov = full

        if isinstance(block, slice):
            i0 = block.start or 0
            i1 = block.stop if block.stop is not None else len(coeffs)
        elif isinstance(block, (tuple, list)) and len(block) == 2 and all(
            isinstance(v, (int, np.integer)) for v in block
        ):
            i0, i1 = int(block[0]), int(block[1])
        elif isinstance(block, (int, np.integer)):
            i0, i1 = int(block), int(block) + 1
        elif hasattr(block, "labels") and hasattr(block, "n_features"):
            full_labels = list(getattr(self, f"{field}_basis_labels", []) or [])
            block_labels = list(block.labels)
            nf = int(block.n_features)
            i0 = None
            for start in range(0, len(full_labels) - nf + 1):
                if full_labels[start : start + nf] == block_labels:
                    i0 = start
                    break
            if i0 is None:
                raise ValueError("Could not locate the basis block within the fitted basis labels.")
            i1 = i0 + nf
        else:
            raise TypeError("block must be (start, stop), a slice, an int, or a Basis.")

        cslice = coeffs[i0:i1]
        covslice = cov[i0:i1, i0:i1] if cov is not None else None
        return cslice, covslice

    def predict_time_profile(
        self, basis_block, t, *, field: str = "force", x=None, duration=None, return_std: bool = False
    ):
        """Evaluate a (time-dependent) basis block's coefficient profile at ``t``.

        Contracts the fitted coefficients of ``basis_block`` with the basis's
        own evaluation at times ``t`` (via the reserved ``time`` extra),
        returning the time profile — e.g. ``-k(t)`` for the ``x`` block of a
        time-Fourier trap (evaluate it at ``x=1``).  Avoids re-deriving the
        design matrix by hand.

        Parameters
        ----------
        basis_block : Basis
            A contiguous block of the fitted basis (located by its labels).
        t : array (T,)
            Times at which to evaluate the profile.
        x : array (dim,), optional
            State at which the block is evaluated (default: the origin).
        duration : float, optional
            Value of the reserved ``duration`` extra (sets the fundamental of
            a ``time_fourier`` dictionary with ``period=None``).  Defaults to
            ``t.max() - t.min()``; pass the recording's duration when ``t``
            does not span the whole trajectory.
        return_std : bool
            If True, also return the 1-σ standard error of the profile,
            propagated from the coefficient covariance.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        coeffs, cov = self.coeff_block(basis_block, field=field)
        t = np.asarray(t, dtype=float)
        dim = int(getattr(basis_block, "dim", 1))
        x0 = np.zeros(dim) if x is None else np.broadcast_to(np.asarray(x, dtype=float), (dim,))
        if duration is None:
            duration = float(t.max() - t.min()) if t.size > 1 else 1.0

        # Global extras are not batched over samples: evaluate one time at a
        # time (vmapped) so each row sees its own ``time``.
        def _row(ti):
            out = basis_block(jnp.asarray(x0)[None, :], extras={"time": ti, "duration": duration})
            return jnp.reshape(out, (-1,))

        D = np.asarray(jax.vmap(_row)(jnp.asarray(t)))
        prof = D @ np.asarray(coeffs)
        if not return_std:
            return prof
        if cov is None:
            raise RuntimeError(f"No coefficient covariance: call compute_{field}_error() first.")
        std = np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", D, np.asarray(cov), D), 0.0))
        return prof, std

    #################################################################
    ################       BACKEND       ############################
    #################################################################

    def _update_force_coefficients(self, coeffs, support=None, jit_inferred=True):
        """Write or update force coefficients and rebuild ``force_inferred``.

        Called after the initial linear solve and again after sparsification
        to set the active support and (re-)build the callable force field.

        Parameters
        ----------
        coeffs : Array
            Coefficient vector for the active basis functions.
        support : array-like or None
            Indices of the active basis functions.  ``None`` means all.
        jit_inferred : bool
            Whether to JIT-compile the resulting ``force_inferred``.
        """
        self.force_coefficients = coeffs
        if support is None:
            self.force_support = jnp.arange(self.force_scorer.p)
        else:
            self.force_support = jnp.array(support)

        # Persist basis labels for downstream reporting (best-effort)
        if hasattr(self, "force_basis"):
            self.force_basis_labels = getattr(self.force_basis, "labels", None)
        elif not hasattr(self, "force_basis_labels"):
            self.force_basis_labels = None
        ix = jnp.ix_(self.force_support, self.force_support)
        self.force_G = self.force_G_full[ix]
        # Covariance of the restricted estimator: the inverse of the selection
        # metric on the support (the EIV fit's G_rawᵀH⁻¹G_raw; else the Gram).
        fisher = getattr(self, "force_fisher_full", None)
        self.force_G_pinv = stable_pinv(self.force_G if fisher is None else fisher[ix])
        # Sparse coeffs on the complete basis:
        self.force_coefficients_full = jnp.zeros_like(self.force_moments)
        if len(self.force_support) > 0:
            self.force_coefficients_full = self.force_coefficients_full.at[self.force_support].set(coeffs)
        # Call the inferred-constructing subclass-specific hook:
        self._update_force_inferred()

    def _update_diffusion_coefficients(self, coeffs, support=None, jit_inferred=True):
        """Write or update diffusion coefficients and rebuild ``diffusion_inferred``."""
        self.diffusion_coefficients = coeffs
        if support is None:
            self.diffusion_support = jnp.arange(self.diffusion_scorer.p)
        else:
            self.diffusion_support = jnp.array(support)
        # Basis labels, so coeff_block can locate a Basis block (as for the force).
        self.diffusion_basis_labels = getattr(getattr(self, "diffusion_basis", None), "labels", None)
        self.diffusion_G = self.diffusion_G_full[jnp.ix_(self.diffusion_support, self.diffusion_support)]
        self.diffusion_G_pinv = stable_pinv(self.diffusion_G)
        # Sparse coeffs on the complete basis:
        self.diffusion_coefficients_full = jnp.zeros_like(self.diffusion_moments)
        self.diffusion_coefficients_full = self.diffusion_coefficients_full.at[self.diffusion_support].set(coeffs)
        # Call the inferred-constructing subclass-specific hook:
        self._update_diffusion_inferred()

    def _detach_from_jax(self):
        """Convert all JAX arrays inside this object to NumPy arrays
        to prevent memory leaks. Use this before deleting this object,
        as the Jax traces might persist otherwise. Important when
        performing a large number of inference runs in the same run
        (e.g. for benchmarking the method over many
        parameters/trajectories).

        """
        import gc

        for attr_name in vars(self):  # Loop through all attributes
            attr_value = getattr(self, attr_name)

            if isinstance(attr_value, jnp.ndarray):  # If it's a JAX array
                setattr(self, attr_name, jax.device_get(attr_value))  # Convert to NumPy

            elif isinstance(attr_value, dict):  # If it's a dictionary, check inside
                for key, value in attr_value.items():
                    if isinstance(value, jnp.ndarray):
                        attr_value[key] = jax.device_get(value)

            elif isinstance(attr_value, list):  # If it's a list, check each item
                for i in range(len(attr_value)):
                    if isinstance(attr_value[i], jnp.ndarray):
                        attr_value[i] = jax.device_get(attr_value[i])

        # Clear any lingering references in JAX's cache
        jax.clear_caches()
        jax.device_get(jax.numpy.zeros(1))  # Forces JAX to clear buffers
        gc.collect()

    # ---- simulation helpers ----------------------------------------------- #

    def _find_finite_x0(self, also_dx: bool = False):
        """Return the first X row (and optionally dX) with no NaN fill values.

        Masked datasets store NaN as a fill value at missing positions.
        Using such a row as a simulation initial condition propagates NaN
        through the whole trajectory.  This helper scans the datasets to
        find the first time step where every element of X is finite.

        Parameters
        ----------
        also_dx : bool
            If True, also return ``dX = X[t+1] - X[t]`` at the chosen step
            (both rows must be finite).

        Returns
        -------
        x0 : Array
            First fully-finite position row.
        dX : Array, only when ``also_dx=True``
            Increment at that time step.
        """
        import numpy as np

        for ds in self.data.datasets:
            X_np = np.asarray(ds.X)
            T = X_np.shape[0]
            flat = X_np.reshape(T, -1)
            rows_finite = np.all(np.isfinite(flat), axis=1)
            if also_dx:
                valid = np.where(rows_finite[:-1] & rows_finite[1:])[0]
            else:
                valid = np.where(rows_finite)[0]
            if len(valid) > 0:
                t0 = int(valid[0])
                x0 = jnp.asarray(X_np[t0])
                if also_dx:
                    return x0, jnp.asarray(X_np[t0 + 1] - X_np[t0])
                return x0

        # Fallback: peek_row + replace any remaining NaN with zeros.
        import warnings

        x0 = jnp.asarray(self.data.peek_row(require={"X"})["X"])
        x0_clean = jnp.where(jnp.isfinite(x0), x0, jnp.zeros_like(x0))
        warnings.warn(
            "simulate_bootstrapped_trajectory: no fully-finite time step found in "
            "the dataset. NaN fill values in the initial condition have been replaced "
            "with 0.0.",
            UserWarning,
            stacklevel=3,
        )
        if also_dx:
            return x0_clean, jnp.zeros_like(x0_clean)
        return x0_clean

    ### Subclass hooks ###

    @abstractmethod
    def _force_G_matrix(self) -> jnp.ndarray: ...

    @abstractmethod
    def _force_moments(self) -> jnp.ndarray: ...

    @abstractmethod
    def _diffusion_G_matrix(self) -> jnp.ndarray: ...

    @abstractmethod
    def _diffusion_moments(self) -> jnp.ndarray: ...

    @abstractmethod
    def get_diffusion_timeop(self, method: str) -> TimeOperand: ...

    @abstractmethod
    def _update_force_inferred(self) -> None: ...

    @abstractmethod
    def _update_diffusion_inferred(self) -> None: ...

    # ---- persistence ------------------------------------------------------ #

    def save_results(self, path) -> "Path":
        """Save ``report_dict()`` to ``<path>.npz`` + ``<path>.json``.

        See :func:`SFI.inference.serialization.save_results` for details.
        """
        from SFI.inference.serialization import save_results

        return save_results(self, path)

    @staticmethod
    def load_results(path) -> dict:
        """Reload a dict previously saved by :meth:`save_results`.

        See :func:`SFI.inference.serialization.load_results` for details.
        """
        from SFI.inference.serialization import load_results

        return load_results(path)
