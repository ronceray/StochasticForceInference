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
        "residuals larger than the noise model predicts: model bias or "
        "finite-Δt effects; on experimental data usually measurement noise: "
        "consider the parametric estimator (infer_force)"
    ),
    ("mse_consistency", "low"): (
        "residuals smaller than the noise model predicts — D̄ over-estimated, "
        "the noise is not white (smoothed / coarse-grained data, correlated "
        "measurement error), or a mis-specified force (its error lowers the "
        "excess through ∇·F̂ for equilibrium-like forces): compare "
        "compute_diffusion_constant methods and check autocorr"
    ),
    ("finite_dt", "coarse"): (
        "O(Δt) biases of a few to tens of per cent — confirm with the "
        "parametric estimator (infer_force) or a stride-2 subsample"
    ),
    ("finite_dt", "unreliable"): (
        "sampling interval comparable to the local relaxation time — the "
        "linear estimators are biased at O(1); use the parametric estimator"
    ),
}


def _hint(section: str, test: str = "") -> str:
    return _FLAG_HINTS.get((section, test)) or _FLAG_HINTS.get((section, ""), "")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and bool(np.isfinite(x))


def _mse_consistency_message(mc: Mapping[str, Any]) -> str:
    """Flag text for the χ² consistency test, stating what is measured.

    The measured quantities are the residual χ² excess (mean whitened norm
    vs its finite-Δt expectation) and its z-score.  The NMSE is only the
    excess re-expressed in force units *if all of it were force error* —
    finite-Δt and noise-model effects also feed it, and a force error
    moves it with either sign — so it is labelled as an implied scale,
    never as a realised/predicted error ratio.
    """
    z = mc.get("excess_z")
    msg = f"[mse_consistency] residual chi^2 excess z = {z:+.2f}"
    mean_n, exp_n = mc.get("mean_mahalanobis_norm"), mc.get("expected_mahalanobis_norm")
    if _finite(mean_n) and _finite(exp_n):
        msg += f" (mean |z|^2 = {mean_n:.4g} vs expected {exp_n:.4g})"
    implied, pred, ratio = mc.get("realised_NMSE"), mc.get("predicted_NMSE"), mc.get("ratio")
    if _finite(implied) and implied > 0:
        msg += f"; if all of it were force error it would imply NMSE ~ {implied:.2g}"
        if _finite(ratio) and _finite(pred):
            msg += f" ({ratio:.2g}x the predicted {pred:.2g})"
        msg += ": a scale for the excess, not a measured force error"
    elif _finite(ratio) and ratio > 0:
        msg += (
            f"; if all of it were force error it would imply {ratio:.2g}x the predicted NMSE"
            ": a scale for the excess, not a measured error ratio"
        )
    return msg


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
    finite_dt : dict
        Per-step drift-gradient scale of the fitted force
        (:func:`~SFI.diagnostics.finite_dt_scale`): ``grad_phi_dt``,
        ``grad_p99_dt``, ``verdict``; filled at ``level="standard"``.
    """

    residuals: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    finite_dt: dict = field(default_factory=dict)

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
            _emit(_mse_consistency_message(consistency), "mse_consistency", "low" if excess_z < 0 else "")

        # Finite sampling interval: the drift gradient per step sets the O(dt)
        # bias of every linear stencil; flag the coarse and unreliable regimes.
        fd = self.finite_dt or {}
        if fd.get("available") and fd.get("verdict") in ("coarse", "unreliable"):
            _emit(
                f"[finite_dt] dt*|grad F| = {fd['grad_phi_dt']:.3g} (p99 {fd.get('grad_p99_dt', float('nan')):.3g}): "
                f"{fd['verdict']} sampling",
                "finite_dt",
                fd["verdict"],
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
                up = mc.get("realised_NMSE_upper95")
                if isinstance(up, (int, float)) and up > 0:
                    real_str += f" (≤ {up:.2g} at 95%)"
                z_str = f"{excess_z:+.2f}" if isinstance(excess_z, (int, float)) else str(excess_z)
                print(f"  predicted NMSE = {pred_str}  χ²-excess NMSE = {real_str}  χ² z = {z_str} (|z|>5 ⇒ bias)")
                if mc.get("note"):
                    print(f"  ({mc['note']})")

        fd = self.finite_dt or {}
        if fd.get("available"):
            print("\n-- Finite sampling interval --")
            print(
                f"  dt*|grad F| rms = {fd['grad_phi_dt']:.3g}  p99 = {fd['grad_p99_dt']:.3g}  "
                f"(w.r.t. {fd['variable']}, {fd['n_points']} points)  verdict: {fd['verdict']}"
            )
        elif fd:
            print(f"\n-- Finite sampling interval -- unavailable: {fd.get('reason', '?')}")

        issues = self.flag_issues(alpha=alpha, hints=hints)
        print("\n-- Flags --")
        if not issues:
            print("  (no issues at α = {:.2g})".format(alpha))
        else:
            for msg in issues:
                print(f"  ! {msg}")
        print()
