"""DiagnosticsReport dataclass — structured output of `assess()`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

#: One-line action hint per flag, appended to `flag_issues` messages.
#: Keys are ``(section, test)``; ``(section, "")`` is the section fallback.
#: Wording mirrors docs/source/diagnostics.rst ("Interpreting flags" and
#: "When a flag points beyond the linear estimators") — keep in lockstep.
_FLAG_HINTS: dict[tuple[str, str], str] = {
    ("autocorr", "ljung_box"): (
        "missing time-correlated feature — widen the basis; if it persists, "
        "suspect coarse sampling: the parametric estimator (infer_force) "
        "extends the usable Δt"
    ),
    ("autocorr", "ljung_box_squared"): (
        "diffusion mis-estimated or state-dependent — try the other "
        "compute_diffusion_constant method or a state-dependent diffusion "
        "basis; the parametric estimators profile (D, Λ)"
    ),
    ("normality", ""): (
        "non-Gaussian residuals — rare events not captured by the basis, "
        "or a non-Gaussian noise structure"
    ),
    ("moments", "mean"): (
        "non-zero residual mean — systematic drift bias; widen the basis"
    ),
    ("moments", "std"): (
        "whitened std far from 1 — D̄ likely wrong: try both "
        "compute_diffusion_constant methods, then the parametric estimator"
    ),
    ("mse_consistency", ""): (
        "realised error above predicted — model bias; on experimental data "
        "usually measurement noise: consider the parametric estimator "
        "(infer_force)"
    ),
}


def _hint(section: str, test: str = "") -> str:
    return _FLAG_HINTS.get((section, test)) or _FLAG_HINTS.get((section, ""), "")


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / jax arrays to plain Python objects."""
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


@dataclass
class DiagnosticsReport:
    """Container for residual-consistency test results.

    Attributes
    ----------
    residuals : dict
        Test results. Always holds ``"moments"``; at ``level="standard"``
        also ``"autocorr"``, ``"normality"`` and ``"mse_consistency"``.
    meta : dict
        Backend tag, regime, ``n_obs``, ``n_particles``, ``d``, level.
    """

    residuals: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the report."""
        return _to_jsonable(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------ #
    # Issue flagging
    # ------------------------------------------------------------------ #
    def flag_issues(self, alpha: float = 0.01, *, hints: bool = True) -> list[str]:
        """List human-readable warnings.

        Returns one line per test whose p-value is below ``alpha`` or
        whose statistic crosses a sane threshold (residual mean off zero,
        std far from one, MSE-consistency ``|z| > 5``).

        Parameters
        ----------
        alpha : float
            Significance level for the p-value tests.
        hints : bool
            When True (default), each message carries a one-line action
            hint (" — <what to do>"); set False for bare statistics
            (machine parsing).
        """
        msgs: list[str] = []

        def _emit(base: str, section: str, test: str = "") -> None:
            h = _hint(section, test) if hints else ""
            msgs.append(f"{base} — {h}" if h else base)

        # Normality / autocorrelation p-values
        for section_name, section in (
            ("normality", self.residuals.get("normality", {})),
            ("autocorr", self.residuals.get("autocorr", {})),
        ):
            for test_name, payload in section.items():
                if not isinstance(payload, Mapping):
                    continue
                p = payload.get("pvalue")
                if p is not None and p < alpha:
                    _emit(
                        f"[{section_name}/{test_name}] p={p:.2e} < {alpha}",
                        section_name,
                        test_name,
                    )

        # Residual moments
        moments = self.residuals.get("moments", {})
        m = moments.get("mean")
        s = moments.get("std")
        n = float(moments.get("n", 1))
        band = 4.0 / max(n**0.5, 1.0)
        if m is not None and s is not None and abs(m) > band:
            _emit(
                f"[moments/mean] |mean|={abs(m):.3g} (expected ~0, 4σ band={band:.3g})",
                "moments",
                "mean",
            )
        if s is not None and (s < 0.5 or s > 2.0):
            _emit(f"[moments/std] std={s:.3g}", "moments", "std")

        # Predicted vs realised MSE — flag on the chi^2 z-score of the
        # residual excess, which is sampling-noise-aware (the raw ratio is
        # too noisy at modest n_obs).  A decisive z (|z|>5) flags on its
        # own; a moderately significant z (|z|>2) flags when the realised/
        # predicted ratio is also an order of magnitude off — a large,
        # consistently-signed excess at modest n_obs (e.g. a structurally
        # misspecified force family) would otherwise stay silent.
        consistency = self.residuals.get("mse_consistency", {})
        excess_z = consistency.get("excess_z")
        ratio = consistency.get("ratio")
        if excess_z is not None and (
            abs(excess_z) > 5.0
            or (abs(excess_z) > 2.0 and ratio is not None and ratio > 10.0)
        ):
            _emit(
                f"[mse_consistency] residual chi^2 z-score = {excess_z:+.2f}, "
                f"realised/predicted NMSE = {ratio:.2g}",
                "mse_consistency",
            )

        return msgs

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #
    def print_summary(self, alpha: float = 0.01, *, hints: bool = True) -> None:
        """Print a human-readable summary of the diagnostic report.

        Each flagged issue carries a one-line action hint unless
        ``hints=False``.
        """
        print("\n=== SFI diagnostics report ===")
        meta = self.meta or {}
        if meta:
            print(f"backend  : {meta.get('backend', '?')}")
            print(f"regime   : {meta.get('regime', '?')}")
            print(
                f"n_obs    : {meta.get('n_obs', '?')}  "
                f"n_particles: {meta.get('n_particles', '?')}  "
                f"d: {meta.get('d', '?')}"
            )
            print(f"level    : {meta.get('level', '?')}")

        res = self.residuals or {}
        if res:
            print("\n-- Residuals --")
            mom = res.get("moments", {})
            if mom:
                print(
                    f"  mean = {mom.get('mean', float('nan')):+.4f}  "
                    f"std = {mom.get('std', float('nan')):.4f}  "
                    f"skew = {mom.get('skew', float('nan')):+.3f}  "
                    f"kurt-3 = {mom.get('excess_kurt', float('nan')):+.3f}  "
                    f"(n={mom.get('n', 0)})"
                )
            for sect_name, sect in (
                ("normality", res.get("normality", {})),
                ("autocorr", res.get("autocorr", {})),
            ):
                if not sect:
                    continue
                for test_name, payload in sect.items():
                    if not isinstance(payload, Mapping):
                        continue
                    stat = payload.get("statistic")
                    p = payload.get("pvalue")
                    if stat is None and p is None:
                        continue
                    flag = "✗" if (p is not None and p < alpha) else "✓"
                    s_str = f"stat={stat:.3g}" if stat is not None else ""
                    p_str = f"p={p:.3g}" if p is not None else ""
                    print(f"  {flag} {sect_name:9s} {test_name:18s} {s_str:14s} {p_str}")

            mc = res.get("mse_consistency", {})
            if mc:
                pred = mc.get("predicted_NMSE")
                real = mc.get("realised_NMSE")
                excess_z = mc.get("excess_z")
                pred_str = f"{pred:.3g}" if isinstance(pred, (int, float)) else str(pred)
                real_str = f"{real:.3g}" if isinstance(real, (int, float)) else str(real)
                z_str = f"{excess_z:+.2f}" if isinstance(excess_z, (int, float)) else str(excess_z)
                print(f"  predicted NMSE = {pred_str}  realised NMSE = {real_str}  χ² z = {z_str} (|z|>5 ⇒ bias)")

        issues = self.flag_issues(alpha=alpha, hints=hints)
        print("\n-- Flags --")
        if not issues:
            print("  (no issues at α = {:.2g})".format(alpha))
        else:
            for msg in issues:
                print(f"  ! {msg}")
        print()
