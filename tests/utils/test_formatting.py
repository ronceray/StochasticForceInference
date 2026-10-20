"""model_summary emits ANSI styling only on a terminal (or when asked)."""

import io
import sys

import numpy as np

from SFI.utils.formatting import model_summary

ESC = "\033["
LABELS = ["1", "x", "x^2"]
COEFFS = np.array([0.01, -1.0, 0.3])
STDERR = np.array([0.1, 0.001, 0.01])  # SNR 0.1 (dim), 1000 (bold), 30


class _Stream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def _table(monkeypatch, *, tty, color=None, no_color=False):
    monkeypatch.setattr(sys, "stdout", _Stream(tty))
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    return model_summary(LABELS, COEFFS, stderr=STDERR, color=color)


def test_plain_text_when_not_a_terminal(monkeypatch):
    """Regression: bold/dim ANSI codes leaked into files, notebooks and docs."""
    out = _table(monkeypatch, tty=False)
    assert ESC not in out
    assert "***" in out  # significance markers still carry the information


def test_styled_on_a_terminal(monkeypatch):
    out = _table(monkeypatch, tty=True)
    assert "\033[1m" in out and "\033[2m" in out


def test_explicit_color_overrides_detection(monkeypatch):
    assert ESC in _table(monkeypatch, tty=False, color=True)
    assert ESC not in _table(monkeypatch, tty=True, color=False)


def test_no_color_convention(monkeypatch):
    assert ESC not in _table(monkeypatch, tty=True, no_color=True)


def test_same_table_up_to_styling(monkeypatch):
    styled = _table(monkeypatch, tty=False, color=True)
    for code in ("\033[1m", "\033[2m", "\033[0m"):
        styled = styled.replace(code, "")
    assert styled == _table(monkeypatch, tty=False)
