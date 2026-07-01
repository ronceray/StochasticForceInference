"""Top-level diagnostic dispatcher.

See :func:`assess` for the user-facing entry point.
"""

from __future__ import annotations

from typing import Optional

from .report import DiagnosticsReport
from .residual_tests import (
    autocorrelation_tests,
    mse_consistency,
    normality_test,
    residual_moments,
)
from .residuals import build_residuals

_LEVELS = ("minimal", "standard")


def assess(
    inferer,
    *,
    level: str = "standard",
    n_lags: Optional[int] = None,
    data=None,
) -> DiagnosticsReport:
    """Run residual-consistency checks on a fitted inference object.

    Recomputes the standardised residuals (Euler--Maruyama innovations,
    whitened by the inferred constant diffusion) and tests whether they
    look like an independent ``N(0, 1)`` sample.

    Parameters
    ----------
    inferer :
        A fitted ``OverdampedLangevinInference`` or
        ``UnderdampedLangevinInference``. Must have a callable
        ``force_inferred`` (run e.g. ``infer_force_linear`` first) and an
        inferred constant diffusion (``A_inv``).
    level :
        ``"minimal"`` — pooled residual moments only;
        ``"standard"`` (default) — adds autocorrelation, normality, and
        predicted-vs-realised MSE consistency.
    n_lags :
        Number of autocorrelation lags. Default ``min(20, n_eff // 5)``.
    data :
        Optional independent :class:`~SFI.trajectory.TrajectoryCollection`
        on which to evaluate the residuals (held-out diagnostics).
        Default: the training data attached to the inferer.

    Returns
    -------
    DiagnosticsReport
        Container with a ``residuals`` section and ``meta``.

    Notes
    -----
    The Euler-style residual is exact for the linear estimators; for the
    parametric estimators it is an approximation that
    is still asymptotically consistent under correct specification, so
    the autocorrelation and normality tests remain valid for flagging
    misspecification.
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS!r}; got {level!r}.")

    bundle = build_residuals(inferer, data=data)

    residuals: dict = {"moments": residual_moments(bundle)}
    if level == "standard":
        residuals["autocorr"] = autocorrelation_tests(bundle, n_lags=n_lags)
        residuals["normality"] = normality_test(bundle)
        residuals["mse_consistency"] = mse_consistency(inferer, bundle)

    meta = {
        "backend": bundle.backend,
        "regime": bundle.regime,
        "n_obs": bundle.n_obs,
        "n_particles": bundle.n_particles,
        "d": bundle.d,
        "level": level,
        "inferer_class": type(inferer).__name__,
    }

    return DiagnosticsReport(residuals=residuals, meta=meta)
