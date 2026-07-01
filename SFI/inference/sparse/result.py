"""
SFI.inference.sparse.result — Sparsity result container
=======================================================

:class:`SparsityResult` is the return type of every search strategy.
It stores the Pareto front (best info / support / coefficients per
cardinality *k*) and provides information-criterion selection.

Supported information criteria
------------------------------
* **AIC** — Akaike (1974), penalty *k*.
* **BIC** — Schwarz (1978), penalty (k/2) ln τ.  Uses the
  continuous-time formulation of Gerardos & Ronceray (2025).
* **EBIC** — Chen & Chen (2008), BIC + 2 γ ln C(n₀, k).
* **PASTIS** — Gerardos & Ronceray (2025), penalty k ln(n₀/p₀).
* **SIC** — Secret Information Criterion (unpublished, Ronceray),
  penalty k ln(I_total).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

Array = jnp.ndarray


@dataclass(frozen=True)
class SparsityResult:
    """Frozen container for the output of a sparsity search.

    Attributes
    ----------
    p : int
        Total number of candidate basis functions.
    total_info : float
        Information gain of the full (dense) model.
    method : str
        Name of the strategy that produced this result (e.g.
        ``"beam"``, ``"greedy"``, ``"stlsq"``, ``"lasso"``).
    best_info_by_k : list[float]
        ``best_info_by_k[k]`` is the highest information gain found
        among all explored supports of cardinality *k*.  Unexplored
        cardinalities are ``-inf``.
    best_support_by_k : list[list[int]]
        The support achieving ``best_info_by_k[k]``.
    best_coeffs_by_k : list[Array | None]
        The corresponding coefficient vector.
    second_info_by_k : list[float]
        Second-best information gain per *k* (for robustness
        diagnostics).  May be all ``-inf`` if the strategy does not
        track runner-ups.
    second_support_by_k : list[list[int]]
        Support achieving the second-best info per *k*.
    """

    p: int
    total_info: float
    method: str

    best_info_by_k: list = field(default_factory=list)
    best_support_by_k: list = field(default_factory=list)
    best_coeffs_by_k: list = field(default_factory=list)

    second_info_by_k: list = field(default_factory=list)
    second_support_by_k: list = field(default_factory=list)

    # -----------------------------------------------------------------
    # Information-criterion selection
    # -----------------------------------------------------------------
    def select_by_ic(
        self,
        name: str,
        *,
        p_param: float = 1e-3,
        tau: Optional[float] = None,
        gamma: float = 0.5,
    ) -> Tuple[int, List[int], float, Optional[Array]]:
        r"""Return the support that maximises a given information criterion.

        .. physics:: Information criteria for sparse model selection
           :label: information-criteria
           :category: Model selection

           .. math::

              \text{AIC}(k)    &= \mathcal{I}(k) - k \\
              \text{BIC}(k)    &= \mathcal{I}(k) - \tfrac{1}{2}\,k\,\ln\tau \\
              \text{EBIC}(k)   &= \text{BIC}(k) - 2\gamma\,\ln\binom{n_0}{k} \\
              \text{PASTIS}(k) &= \mathcal{I}(k) - k\,\ln(n_0 / p_0) \\
              \text{SIC}(k)    &= \mathcal{I}(k) - k\,\ln(\mathcal{I}_{\text{total}})

           where :math:`\mathcal{I}(k)` is the log-likelihood gain with
           *k* basis terms out of :math:`n_0` candidates, :math:`\tau`
           is the total trajectory time, :math:`p_0` is the PASTIS
           significance level, and :math:`\gamma \in [0,1]` controls EBIC
           stringency.

        References
        ----------
        * **AIC** — Akaike, H. (1974). "A new look at the statistical
          model identification." *IEEE Trans. Automat. Control*, 19(6),
          716–723.
        * **BIC** — Schwarz, G. (1978). "Estimating the dimension of a
          model." *Ann. Statist.*, 6(2), 461–464.  The continuous-time
          formulation :math:`\tfrac{k}{2}\ln\tau` follows from the
          Laplace approximation of the SDE marginal likelihood
          (Gerardos & Ronceray, 2025).
        * **EBIC** — Chen, J. & Chen, Z. (2008). "Extended Bayesian
          information criteria for model selection with large model
          spaces." *Biometrika*, 95(3), 759–771.
        * **PASTIS** — Gerardos, A. & Ronceray, P. (2025).
          "Principled model selection for stochastic dynamics."
        * **SIC** — Unpublished (Ronceray).

        Parameters
        ----------
        name : ``"AIC"`` | ``"BIC"`` | ``"EBIC"`` | ``"PASTIS"`` | ``"SIC"``
            Information criterion to maximise.
        p_param : float, default 1e-3
            Significance level :math:`p_0` for the PASTIS penalty.
        tau : float or None
            Total trajectory time.  **Required** for BIC and EBIC.
        gamma : float, default 0.5
            EBIC tuning parameter (:math:`\gamma \in [0,1]`).  Only used
            when *name* is ``"EBIC"``.

        Returns
        -------
        k_star : int
            Selected model size.
        support : list[int]
            Basis-function indices of the chosen model.
        score : float
            Value of the information criterion at ``k_star``.
        coeffs : Array or None
            Coefficient vector for the selected support.
        """
        name = name.upper()
        n0 = self.p
        total_info = self.total_info

        # Validate tau for criteria that need it
        if name in ("BIC", "EBIC") and tau is None:
            raise ValueError(
                f"Criterion {name!r} requires the total trajectory time 'tau'.  Pass tau=<float> to select_by_ic()."
            )

        def _log_comb(n: int, k: int) -> float:
            """log C(n, k) via lgamma — exact for integer args."""
            if k < 0 or k > n:
                return 0.0
            return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

        def _score(k: int, info: float) -> float:
            if info == -np.inf:
                return -np.inf
            if name == "AIC":
                return info - k
            if name == "BIC":
                return info - 0.5 * k * math.log(tau)
            if name == "EBIC":
                return info - 0.5 * k * math.log(tau) - 2.0 * gamma * _log_comb(n0, k)
            if name == "PASTIS":
                return info - k * math.log(n0 / p_param)
            if name == "SIC":
                return info - k * math.log(total_info)
            raise ValueError(f"Unknown criterion {name!r}")

        scores = [_score(k, info) for k, info in enumerate(self.best_info_by_k)]
        k_star = int(np.argmax(scores))
        logger.info(
            "Criterion %s selected a model with %d terms out of %d.",
            name,
            k_star,
            self.p,
        )
        return (
            k_star,
            self.best_support_by_k[k_star],
            scores[k_star],
            self.best_coeffs_by_k[k_star],
        )

    # -----------------------------------------------------------------
    # Convenience: all ICs at once
    # -----------------------------------------------------------------
    def all_ic(
        self,
        *,
        p_param: float = 1e-3,
        tau: Optional[float] = None,
        gamma: float = 0.5,
        true_support: Optional[List[int]] = None,
        true_coeffs: Optional[List[float]] = None,
        Phi_test: Optional[Array] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Compute all information criteria and optionally compare to ground truth.

        Parameters
        ----------
        p_param : float
            PASTIS significance level.
        tau : float or None
            Total trajectory time.  If provided, BIC and EBIC are
            included; otherwise they are skipped.
        gamma : float, default 0.5
            EBIC tuning parameter.
        true_support, true_coeffs : optional
            Ground-truth support and coefficients for overlap metrics.
        Phi_test : optional Array
            Held-out design matrix for predictive NMSE.
        verbose : bool
            If *True*, log a summary table at INFO level.

        Returns
        -------
        dict
            Keyed by IC name, each value is a dict with ``k``,
            ``support``, ``score``, ``coeffs``, and optionally overlap
            and predictive-NMSE entries.
        """
        from .metrics import overlap_metrics, predictive_nmse

        # Build list of criteria — BIC/EBIC only when tau is available
        ic_names = ["AIC"]
        if tau is not None:
            ic_names += ["BIC", "EBIC"]
        ic_names += ["PASTIS", "SIC"]

        summary: Dict[str, Dict] = {}
        for ic_name in ic_names:
            k, support, score, coeffs = self.select_by_ic(
                ic_name,
                p_param=p_param,
                tau=tau,
                gamma=gamma,
            )
            entry: dict = dict(k=k, support=support, score=float(score), coeffs=coeffs)
            if true_support is not None:
                entry.update(overlap_metrics(true_support, support))
                if Phi_test is not None and true_coeffs is not None:
                    entry["predictive_NMSE"] = predictive_nmse(Phi_test, true_support, true_coeffs, support, coeffs)
            summary[ic_name] = entry

        if verbose:
            has_overlap = any("exact" in e for e in summary.values())
            has_nmse = any("predictive_NMSE" in e for e in summary.values())

            hdr_parts = [f"{'IC':<8}", f"{'k*':>3}", f"{'score':>10}"]
            if has_overlap:
                hdr_parts.append(f"{'TP/FP/FN':>15}")
                hdr_parts.append(f"{'exact':>10}")
            if has_nmse:
                hdr_parts.append(f"{'pred NMSE':>10}")
            hdr_parts.append("support")

            lines = ["=== Information-criterion summary ===", "  ".join(hdr_parts)]
            for ic_name, entry in summary.items():
                row_parts = [f"{ic_name:<8}", f"{entry['k']:>3}", f"{entry['score']:10.2f}"]
                if has_overlap:
                    row_parts.append(f"{entry['TP']}/{entry['FP']}/{entry['FN']:>9}")
                    row_parts.append(f"{str(entry['exact']):>10}")
                if has_nmse:
                    row_parts.append(f"{entry['predictive_NMSE']:10.4f}")
                row_parts.append(str(entry["support"]))
                lines.append("  ".join(row_parts))
            logger.info("\n".join(lines))

        return summary
