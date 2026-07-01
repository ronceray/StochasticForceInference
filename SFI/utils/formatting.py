# SFI/utils/formatting.py
"""Pretty-printing utilities for inferred models."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# ANSI escape codes for terminal highlighting
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def model_summary(
    labels: Sequence[str],
    coefficients: np.ndarray,
    *,
    stderr: Optional[np.ndarray] = None,
    support: Optional[np.ndarray] = None,
    coeffs_true: Optional[np.ndarray] = None,
    support_true: Optional[np.ndarray] = None,
    title: str = "Coefficient Table",
    max_rows: int = 60,
    significance_thresholds: tuple = (2.0, 10.0, 100.0),
    auto_labels: bool = False,
) -> str:
    """
    Build a human-readable coefficient table with SNR and significance.

    Only active (support) coefficients are shown in the table body.  Zeroed
    basis functions are listed separately below, unless labels are
    auto-generated.

    Parameters
    ----------
    labels : sequence of str
        One label per basis function.
    coefficients : 1-D array
        Coefficient vector (length must match *labels* or *support*).
    stderr : 1-D array or None
        Standard errors (same length as *coefficients*).  These reflect
        sampling error only; discretization (finite-time-step) bias is
        not included.
    support : 1-D int array or None
        Indices into *labels* that *coefficients* correspond to.
        If ``None``, full support is assumed (all basis functions
        have non-zero coefficients).
    title : str
        Section header printed above the table.
    max_rows : int
        If the table exceeds this many rows, truncate the middle.
    significance_thresholds : tuple of float
        Three SNR thresholds (in multiples of stderr) for the ``*``, ``**``,
        and ``***`` significance levels.  Defaults ``(2.0, 10.0, 100.0)``.
    auto_labels : bool
        If ``True``, labels were auto-generated (e.g. ``b0``, ``b1``, …)
        and the list of zeroed functions is suppressed.

    Returns
    -------
    str
        Ready-to-print multi-line table.  Terms are marked *, **, or ***
        according to *significance_thresholds*; *** terms are **bold**,
        * and ** terms are normal weight, non-significant active terms are dimmed.
    """
    coefficients = np.asarray(coefficients).ravel()
    n_labels = len(labels)

    # Expand coefficients to full length when support is given
    if support is not None:
        support = np.asarray(support).ravel().astype(int)
        if len(coefficients) == len(support):
            # coefficients is sparse; expand to full length
            full = np.zeros(n_labels)
            full[support] = coefficients
            coefficients = full
        elif len(coefficients) != n_labels:
            raise ValueError(
                f"len(coefficients)={len(coefficients)} matches neither "
                f"len(support)={len(support)} nor n_labels={n_labels}"
            )
        active = set(int(i) for i in support)
    else:
        active = None  # all active

    has_stderr = stderr is not None
    if has_stderr:
        stderr = np.asarray(stderr).ravel()
        if stderr.shape[0] != n_labels:
            # Might be sparse; try to expand
            if support is not None and stderr.shape[0] == len(support):
                se_full = np.zeros(n_labels)
                se_full[support] = stderr
                stderr = se_full
            else:
                raise ValueError(
                    f"len(stderr)={stderr.shape[0]} matches neither "
                    f"n_labels={n_labels} nor len(support)={len(support) if support is not None else 'N/A'}"
                )

    # Expand ground-truth coefficients to full length (paired "True" column)
    has_true = coeffs_true is not None
    true_full = None
    if has_true:
        ct = np.asarray(coeffs_true).ravel()
        true_full = np.zeros(n_labels)
        if support_true is not None:
            st = np.asarray(support_true).ravel().astype(int)
            true_full[st] = ct
        elif ct.shape[0] == n_labels:
            true_full = ct
        elif support is not None and ct.shape[0] == len(support):
            true_full[support] = ct
        else:
            true_full[: ct.shape[0]] = ct

    # ── Build rows ──
    # (index, label, coefficient, stderr_or_None, is_active, snr_or_None, true_or_None)
    rows: list = []
    for i, lab in enumerate(labels):
        c = float(coefficients[i])
        se = float(stderr[i]) if (stderr is not None and i < len(stderr)) else None
        is_active = (active is None) or (i in active)
        snr = abs(c / se) if (se is not None and se > 0) else None
        tv = float(true_full[i]) if has_true else None
        rows.append((i, lab, c, se, is_active, snr, tv))

    # ── Column widths ──
    idx_w = max(3, len(str(n_labels - 1)))
    lab_w = max(5, max(len(r[1]) for r in rows))
    coeff_fmt = "{:>12.5e}"
    se_fmt = "{:>12.5e}"
    snr_fmt = "{:>6.1f}"

    # Determine whether SNR column should be shown
    has_snr = has_stderr and any(r[5] is not None for r in rows)

    # ── Header ──
    sep_w = idx_w + lab_w + 30
    if has_stderr:
        sep_w += 15
    if has_snr:
        sep_w += 9  # "  SNR   "
    if has_true:
        sep_w += 15
    sep = "─" * sep_w

    lines: list[str] = []
    lines.append(f"  {title}")
    lines.append(f"  {sep}")

    hdr = f"  {'#':<{idx_w}}  {'Label':<{lab_w}}  {'Coefficient':>12}"
    if has_stderr:
        hdr += f"  {'Std.Err':>12}"
    if has_snr:
        hdr += f"  {'SNR':>6}"
    if has_true:
        hdr += f"  {'True':>12}"
    hdr += "  Sig"
    lines.append(hdr)
    lines.append(f"  {sep}")

    # ── Rows (with optional truncation) — only show active (support) entries ──
    thr1, thr2, thr3 = significance_thresholds

    def _sig_marker(snr):
        """Return (marker_str, is_bold) for a given SNR (or None)."""
        if snr is None:
            return "·", False
        if snr >= thr3:
            return "***", True
        if snr >= thr2:
            return "**", False
        if snr >= thr1:
            return "*", False
        return "·", False

    def _fmt_row(r):
        i, lab, c, se, act, snr, tv = r
        marker, bold = _sig_marker(snr if act else None)

        # Start with index and label
        s = f"  {i:<{idx_w}}  {lab:<{lab_w}}  {coeff_fmt.format(c)}"
        if has_stderr:
            s += f"  {se_fmt.format(se)}" if se is not None else f"  {'---':>12}"
        if has_snr:
            s += f"  {snr_fmt.format(snr)}" if snr is not None else f"  {'---':>6}"
        if has_true:
            s += f"  {coeff_fmt.format(tv)}" if tv is not None else f"  {'---':>12}"

        # Significance marker (padded to 3 chars for alignment)
        s += f"  {marker:<3}"

        # Apply ANSI formatting: bold for ***, dim for non-significant active
        if bold:
            s = _BOLD + s + _RESET
        elif marker == "·" and has_snr:
            s = _DIM + s + _RESET

        return s

    active_rows = [r for r in rows if r[4]]
    zero_rows = [r for r in rows if not r[4]]

    if len(active_rows) <= max_rows:
        for r in active_rows:
            lines.append(_fmt_row(r))
    else:
        half = (max_rows - 1) // 2
        for r in active_rows[:half]:
            lines.append(_fmt_row(r))
        lines.append(f"  {'...':^{sep_w}}")
        for r in active_rows[-half:]:
            lines.append(_fmt_row(r))

    lines.append(f"  {sep}")

    n_active = len(active_rows)
    summary_line = f"  {n_active}/{n_labels} basis functions in support"
    if has_snr:
        n1 = sum(1 for r in active_rows if r[5] is not None and r[5] >= thr1)
        n2 = sum(1 for r in active_rows if r[5] is not None and r[5] >= thr2)
        n3 = sum(1 for r in active_rows if r[5] is not None and r[5] >= thr3)
        summary_line += f", sig: {n1}* / {n2}** / {n3}*** (|SNR| ≥ {thr1:.4g} / {thr2:.4g} / {thr3:.4g})"
    lines.append(summary_line)
    if has_stderr:
        lines.append("  (Std.err. reflects sampling error only; discretization bias is not included.)")

    # ── Zeroed basis functions ──
    if zero_rows and not auto_labels:
        zero_labels = ", ".join(r[1] for r in zero_rows)
        lines.append(f"  Zeroed ({len(zero_rows)}): {zero_labels}")

    return "\n".join(lines)


def print_model_comparison(
    inferences,
    labels,
    *,
    extra_cols=None,
    metrics=None,
    title: str = "Model Comparison",
) -> str:
    """Build a multi-model comparison table from several inference objects.

    Parameters
    ----------
    inferences :
        Sequence of fitted inference objects.
    labels :
        One name per inference (row labels).
    metrics :
        Attribute names to read from each inference (default
        ``["n_params", "NMSE_force", "force_predicted_MSE"]``).  The special
        ``"n_params"`` counts ``force_coefficients_full``.
    extra_cols :
        Optional ``{column_name: {label: value}}`` of caller-supplied cells.
    title :
        Header line.

    Returns
    -------
    str
        Ready-to-print table.
    """
    if metrics is None:
        metrics = ["n_params", "NMSE_force", "force_predicted_MSE"]
    extra_cols = extra_cols or {}
    cols = ["Model"] + list(metrics) + list(extra_cols.keys())

    def _metric(inf, name):
        if name == "n_params":
            cf = getattr(inf, "force_coefficients_full", None)
            return int(np.asarray(cf).size) if cf is not None else None
        v = getattr(inf, name, None)
        return float(v) if v is not None else None

    rows = []
    for inf, lab in zip(inferences, labels):
        row = {"Model": lab}
        for name in metrics:
            row[name] = _metric(inf, name)
        for col, vals in extra_cols.items():
            row[col] = vals.get(lab) if isinstance(vals, dict) else None
        rows.append(row)

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    widths = {c: max(len(str(c)), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    lines = [f"  {title}"]
    header = "  " + "  ".join(f"{str(c):>{widths[c]}}" for c in cols)
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))
    for r in rows:
        lines.append("  " + "  ".join(f"{_fmt(r.get(c)):>{widths[c]}}" for c in cols))
    return "\n".join(lines)
