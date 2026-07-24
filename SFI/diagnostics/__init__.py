"""SFI diagnostics — residual-consistency checks for inference results.

Top-level user-facing entry point::

    from SFI.diagnostics import assess
    report = assess(inferer, level="standard")
    report.print_summary()

The module reuses the fitted force and the inferred constant diffusion to
recompute standardised residuals (innovations) and test whether they look
like an independent ``N(0, 1)`` sample:

* mean, variance, skewness and kurtosis (pooled and per spatial
  component), plus the per-component covariance;
* Ljung--Box autocorrelation of the residuals and their squares;
* Kolmogorov--Smirnov normality against ``N(0, 1)``;
* predicted-vs-realised normalised mean squared error of the force
  (a sampling-noise-aware chi-square check).

See :mod:`SFI.diagnostics.residual_tests`. Two preset levels are exposed:

* ``"minimal"`` — residual moments only;
* ``"standard"`` — adds autocorrelation, normality, and MSE consistency.

All backends (overdamped / underdamped, single / multi-particle, SPDE)
share a unified residual definition: the Euler--Maruyama innovation
:math:`r_t = \\Delta x_t - F(x_t)\\,\\Delta t` (overdamped) or the
secant-velocity innovation
:math:`r_t = \\Delta v_t - F(x_t, v_t)\\,\\Delta t` (underdamped), both
whitened by :math:`(2 \\bar D \\Delta t)^{-1/2}`.

References
----------
Ljung & Box (1978); Diebold, Gunther & Tay (1998).
"""

from .assess import assess
from .dynamics_order import DynamicsOrderReport, classify_dynamics
from .plotting import (
    plot_dynamics_order,
    plot_qq,
    plot_residual_acf,
    plot_residual_histogram,
    plot_summary,
)
from .report import DiagnosticsReport
from .residual_tests import parametric_four_point_diagnostic

__all__ = [
    "assess",
    "classify_dynamics",
    "DiagnosticsReport",
    "DynamicsOrderReport",
    "parametric_four_point_diagnostic",
    "plot_dynamics_order",
    "plot_qq",
    "plot_residual_acf",
    "plot_residual_histogram",
    "plot_summary",
]
