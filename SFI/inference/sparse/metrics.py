"""
SFI.inference.sparse.metrics — Benchmark helpers
=================================================

Standalone functions for comparing inferred supports / coefficients
against ground truth.  Useful for benchmarking and papers but not
required for normal inference.
"""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp

Array = jnp.ndarray


def overlap_metrics(true_support: List[int], pred_support: List[int]) -> Dict:
    """Compare predicted support to the ground truth.

    Parameters
    ----------
    true_support, pred_support : list[int]
        Indices of the true and predicted active basis functions.

    Returns
    -------
    dict
        Keys: ``TP``, ``FP``, ``FN``, ``prec``, ``rec``, ``exact``.
    """
    true_set, pred_set = set(true_support), set(pred_support)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        prec=tp / (tp + fp) if tp + fp else 0.0,
        rec=tp / (tp + fn) if tp + fn else 0.0,
        exact=(fp == 0 and fn == 0),
    )


def predictive_nmse(
    Phi_test: Array,
    true_support: List[int],
    true_coeffs,
    inferred_support: List[int],
    inferred_coeffs,
) -> float:
    """Normalised mean-squared error on a held-out design matrix.

    Parameters
    ----------
    Phi_test : (n_test, p) Array
        Design matrix evaluated on test data.
    true_support : list[int]
        Ground-truth active indices.
    true_coeffs : array-like
        Ground-truth coefficient vector (length ``len(true_support)``).
    inferred_support : list[int]
        Inferred active indices.
    inferred_coeffs : array-like
        Inferred coefficient vector (length ``len(inferred_support)``).

    Returns
    -------
    float
        :math:`\\|\\hat y - y\\|^2 / \\|y\\|^2`.
    """
    if len(inferred_support) == 0:
        return 1.0
    true_signal = Phi_test[:, jnp.array(true_support)] @ jnp.array(true_coeffs)
    pred_signal = Phi_test[:, jnp.array(inferred_support)] @ jnp.array(inferred_coeffs)
    residual = true_signal - pred_signal
    return float(jnp.sum(residual**2) / jnp.sum(true_signal**2))
