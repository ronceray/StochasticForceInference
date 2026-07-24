"""Flag messages from DiagnosticsReport carry actionable hints."""

from __future__ import annotations

from SFI.diagnostics import DiagnosticsReport

SEP = " — "


def _report(**residuals) -> DiagnosticsReport:
    return DiagnosticsReport(residuals=residuals, meta={})


def _flags(report, **kw):
    return report.flag_issues(alpha=0.01, **kw)


def test_autocorr_hint_mentions_parametric():
    rep = _report(autocorr={"ljung_box": {"statistic": 1.0, "pvalue": 1e-6}})
    (msg,) = _flags(rep)
    assert msg.startswith("[autocorr/ljung_box]")
    assert SEP in msg
    assert "parametric estimator" in msg


def test_autocorr_squared_hint_mentions_diffusion():
    rep = _report(autocorr={"ljung_box_squared": {"statistic": 1.0, "pvalue": 1e-6}})
    (msg,) = _flags(rep)
    assert "diffusion" in msg


def test_normality_hint():
    rep = _report(normality={"ks": {"statistic": 0.4, "pvalue": 1e-8}})
    (msg,) = _flags(rep)
    assert "non-Gaussian" in msg


def test_moments_mean_hint():
    rep = _report(moments={"mean": 5.0, "std": 1.0, "n": 100})
    msgs = _flags(rep)
    assert any("drift bias" in m for m in msgs if m.startswith("[moments/mean]"))


def test_moments_std_hint():
    rep = _report(moments={"mean": 0.0, "std": 3.0, "n": 100})
    (msg,) = _flags(rep)
    assert msg.startswith("[moments/std]")
    assert "compute_diffusion_constant" in msg


def test_mse_consistency_hint_mentions_infer_force():
    rep = _report(mse_consistency={"excess_z": 8.0, "ratio": 50.0})
    (msg,) = _flags(rep)
    assert msg.startswith("[mse_consistency]")
    assert "measurement noise" in msg
    assert "infer_force" in msg


def test_hints_false_strips_suffix():
    rep = _report(
        autocorr={"ljung_box": {"pvalue": 1e-6}},
        moments={"mean": 0.0, "std": 3.0, "n": 100},
        mse_consistency={"excess_z": 8.0, "ratio": 50.0},
    )
    for msg in _flags(rep, hints=False):
        assert SEP not in msg


def test_clean_report_has_no_flags(capsys):
    rep = _report(
        moments={"mean": 0.0, "std": 1.0, "n": 1000},
        autocorr={"ljung_box": {"pvalue": 0.5}},
        normality={"ks": {"statistic": 0.01, "pvalue": 0.7}},
        mse_consistency={"excess_z": 0.3, "ratio": 1.1},
    )
    assert _flags(rep) == []
    rep.print_summary()
    out = capsys.readouterr().out
    assert "no issues" in out


def test_print_summary_shows_hints(capsys):
    rep = _report(mse_consistency={"excess_z": 8.0, "ratio": 50.0})
    rep.print_summary()
    out = capsys.readouterr().out
    assert "infer_force" in out
    rep.print_summary(hints=False)
    out = capsys.readouterr().out
    assert "infer_force" not in out
