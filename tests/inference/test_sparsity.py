# TODO: review this file
"""
Unit tests for the SFI.inference.sparse sub-package.

Uses a small synthetic problem where the ground truth is known:
    p = 10 basis functions, true support = {1, 4, 7}, random G and M
    constructed so that the true coefficients are [2, -3, 1].
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from SFI.inference.sparse import (
    SparseScorer,
    SparsityResult,
    BeamSearchStrategy,
    GreedyStepwiseStrategy,
    HillClimbStrategy,
    STLSQStrategy,
    LassoStrategy,
    overlap_metrics,
    predictive_nmse,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def synthetic():
    """Build a synthetic (M, G) pair with a known sparse ground truth.

    We construct G as Phi^T Phi for a random design matrix Phi (100 x 10)
    and M = Phi^T y where y = Phi[:, true_support] @ true_coeffs + noise.
    """
    rng = np.random.default_rng(42)
    p = 10
    n = 200
    true_support = [1, 4, 7]
    true_coeffs = np.array([2.0, -3.0, 1.0])

    Phi = rng.standard_normal((n, p))
    y = Phi[:, true_support] @ true_coeffs + 0.05 * rng.standard_normal(n)

    G = jnp.array(Phi.T @ Phi / n)
    M = jnp.array(Phi.T @ y / n)

    return dict(
        G=G,
        M=M,
        p=p,
        n=n,
        true_support=true_support,
        true_coeffs=true_coeffs,
        Phi=jnp.array(Phi),
    )


@pytest.fixture(scope="module")
def scorer(synthetic):
    return SparseScorer(M=synthetic["M"], G=synthetic["G"])


# ======================================================================
# SparseScorer
# ======================================================================


class TestSparseScorer:
    def test_total_solution_shape(self, scorer, synthetic):
        assert scorer.p == synthetic["p"]
        assert scorer.total_C.shape == (synthetic["p"],)
        assert float(scorer.total_info) > 0

    def test_single_support(self, scorer, synthetic):
        B = jnp.array(synthetic["true_support"], dtype=jnp.int32)
        info, C = scorer.info_and_coeffs(B)
        assert C.shape == (3,)
        assert float(info) > 0
        # Coefficients should be close to true
        np.testing.assert_allclose(
            np.asarray(C), synthetic["true_coeffs"], atol=0.3
        )

    def test_empty_support(self, scorer):
        B = jnp.array([], dtype=jnp.int32)
        info, C = scorer.info_and_coeffs(B)
        assert float(info) == 0.0
        assert C.shape == (0,)

    def test_full_support(self, scorer, synthetic):
        B = jnp.arange(synthetic["p"], dtype=jnp.int32)
        info, C = scorer.info_and_coeffs(B)
        np.testing.assert_allclose(float(info), float(scorer.total_info), rtol=1e-5)

    def test_vmap_info(self, scorer):
        batch = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        infos, coeffs = scorer.vmap_info(batch)
        assert infos.shape == (2,)
        assert coeffs.shape == (2, 3)


# ======================================================================
# Beam search
# ======================================================================


class TestBeamSearch:
    def test_basic(self, scorer, synthetic):
        strategy = BeamSearchStrategy(beam_width=5, report_time=True)
        result = strategy.run(scorer, max_k=6)
        assert isinstance(result, SparsityResult)
        assert result.method == "beam"
        assert result.p == synthetic["p"]
        # The Pareto front should have some positive entries
        assert max(result.best_info_by_k) > 0

    def test_recovers_true_support(self, scorer, synthetic):
        strategy = BeamSearchStrategy(beam_width=10)
        result = strategy.run(scorer, max_k=8)
        # AIC is the right criterion for this small problem (PASTIS needs large p)
        k, support, score, coeffs = result.select_by_ic("AIC")
        assert k <= 5  # should be close to 3
        om = overlap_metrics(synthetic["true_support"], support)
        assert om["rec"] >= 0.66  # at least 2 of 3 recovered

    def test_ic_selection(self, scorer):
        strategy = BeamSearchStrategy(beam_width=5)
        result = strategy.run(scorer, max_k=8)
        for ic_name in ("AIC", "PASTIS", "SIC"):
            k, support, score, coeffs = result.select_by_ic(ic_name)
            assert k >= 0
            assert isinstance(support, list)
        # BIC and EBIC require tau
        for ic_name in ("BIC", "EBIC"):
            k, support, score, coeffs = result.select_by_ic(ic_name, tau=100.0)
            assert k >= 0
            assert isinstance(support, list)


# ======================================================================
# Greedy stepwise
# ======================================================================


class TestGreedyStepwise:
    @pytest.mark.parametrize("direction", ["forward", "backward", "both"])
    def test_runs(self, scorer, synthetic, direction):
        strategy = GreedyStepwiseStrategy(direction=direction, report_time=True)
        result = strategy.run(scorer, max_k=6)
        assert isinstance(result, SparsityResult)
        assert result.method.startswith("greedy")

    def test_forward_recovers_support(self, scorer, synthetic):
        strategy = GreedyStepwiseStrategy(direction="forward")
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("AIC")
        om = overlap_metrics(synthetic["true_support"], support)
        assert om["rec"] >= 0.66

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            GreedyStepwiseStrategy(direction="invalid")


# ======================================================================
# STLSQ
# ======================================================================


class TestSTLSQ:
    def test_auto_sweep(self, scorer, synthetic):
        strategy = STLSQStrategy(n_thresholds=15, report_time=True)
        result = strategy.run(scorer, max_k=8)
        assert isinstance(result, SparsityResult)
        assert result.method == "stlsq"
        # At least one model should have positive info
        assert max(result.best_info_by_k) > 0

    def test_single_threshold(self, scorer, synthetic):
        strategy = STLSQStrategy(threshold=0.1, mode="relative")
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("AIC")
        assert k >= 0

    def test_absolute_mode(self, scorer, synthetic):
        strategy = STLSQStrategy(threshold=0.5, mode="absolute")
        result = strategy.run(scorer, max_k=8)
        assert isinstance(result, SparsityResult)


# ======================================================================
# LASSO
# ======================================================================


class TestLasso:
    def test_auto_path(self, scorer, synthetic):
        strategy = LassoStrategy(n_alphas=20, report_time=True)
        result = strategy.run(scorer, max_k=8)
        assert isinstance(result, SparsityResult)
        assert result.method == "lasso"
        assert max(result.best_info_by_k) > 0

    def test_single_alpha(self, scorer, synthetic):
        strategy = LassoStrategy(alpha=0.01)
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("AIC")
        assert k >= 0

    def test_recovers_support(self, scorer, synthetic):
        strategy = LassoStrategy(n_alphas=30)
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("AIC")
        om = overlap_metrics(synthetic["true_support"], support)
        # LASSO+AIC should get reasonable support
        assert om["rec"] >= 0.33


# ======================================================================
# SparsityResult
# ======================================================================


class TestSparsityResult:
    def test_all_ic(self, scorer):
        strategy = BeamSearchStrategy(beam_width=5)
        result = strategy.run(scorer, max_k=6)
        # Without tau: only AIC, PASTIS, SIC
        summary = result.all_ic(verbose=False)
        assert set(summary.keys()) == {"AIC", "PASTIS", "SIC"}
        for entry in summary.values():
            assert "k" in entry and "support" in entry
        # With tau: all five ICs
        summary_full = result.all_ic(tau=100.0, verbose=False)
        assert set(summary_full.keys()) == {"AIC", "BIC", "EBIC", "PASTIS", "SIC"}

    def test_all_ic_with_truth(self, scorer, synthetic):
        strategy = BeamSearchStrategy(beam_width=5)
        result = strategy.run(scorer, max_k=6)
        summary = result.all_ic(
            tau=100.0,
            true_support=synthetic["true_support"],
            true_coeffs=synthetic["true_coeffs"],
            Phi_test=synthetic["Phi"],
            verbose=False,
        )
        for entry in summary.values():
            assert "TP" in entry
            assert "predictive_NMSE" in entry

    def test_bic_requires_tau(self, scorer):
        strategy = BeamSearchStrategy(beam_width=5)
        result = strategy.run(scorer, max_k=6)
        with pytest.raises(ValueError, match="tau"):
            result.select_by_ic("BIC")
        with pytest.raises(ValueError, match="tau"):
            result.select_by_ic("EBIC")

    def test_bic_formula(self, scorer):
        """BIC penalty should be (k/2)*log(tau), not log(p)."""
        strategy = GreedyStepwiseStrategy(direction="forward")
        result = strategy.run(scorer, max_k=6)
        tau = 50.0
        k, support, score, coeffs = result.select_by_ic("BIC", tau=tau)
        # Verify the score manually for the selected k
        info_k = result.best_info_by_k[k]
        expected = info_k - 0.5 * k * np.log(tau)
        np.testing.assert_allclose(score, expected, rtol=1e-6)

    def test_ebic_stricter_than_bic(self, scorer):
        """EBIC (gamma>0) should select a model no larger than BIC."""
        strategy = BeamSearchStrategy(beam_width=5)
        result = strategy.run(scorer, max_k=8)
        tau = 50.0
        k_bic, *_ = result.select_by_ic("BIC", tau=tau)
        k_ebic, *_ = result.select_by_ic("EBIC", tau=tau, gamma=1.0)
        assert k_ebic <= k_bic


# ======================================================================
# Metrics helpers
# ======================================================================


class TestMetrics:
    def test_overlap_perfect(self):
        om = overlap_metrics([1, 4, 7], [1, 4, 7])
        assert om["exact"] is True
        assert om["prec"] == 1.0
        assert om["rec"] == 1.0

    def test_overlap_partial(self):
        om = overlap_metrics([1, 4, 7], [1, 2, 7])
        assert om["TP"] == 2
        assert om["FP"] == 1
        assert om["FN"] == 1

    def test_overlap_empty(self):
        om = overlap_metrics([1, 4, 7], [])
        assert om["TP"] == 0
        assert om["FN"] == 3

    def test_predictive_nmse_exact(self):
        rng = np.random.default_rng(99)
        Phi = jnp.array(rng.standard_normal((20, 5)))
        support = [0, 2]
        coeffs = [1.0, -1.0]
        nmse = predictive_nmse(Phi, support, coeffs, support, coeffs)
        assert nmse < 1e-10

    def test_predictive_nmse_empty_inferred(self):
        Phi = jnp.ones((10, 5))
        assert predictive_nmse(Phi, [0], [1.0], [], []) == 1.0


# ======================================================================
# Hill-climbing
# ======================================================================


class TestHillClimb:
    def test_basic(self, scorer, synthetic):
        strategy = HillClimbStrategy(
            ic="AIC", patience=100, seed=42, report_time=True,
        )
        result = strategy.run(scorer, max_k=6)
        assert isinstance(result, SparsityResult)
        assert result.method == "hillclimb"
        assert result.p == synthetic["p"]
        assert max(result.best_info_by_k) > 0

    def test_recovers_support(self, scorer, synthetic):
        strategy = HillClimbStrategy(ic="AIC", patience=200, seed=0)
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("AIC")
        om = overlap_metrics(synthetic["true_support"], support)
        assert om["rec"] >= 0.66

    def test_pastis_criterion(self, scorer, synthetic):
        strategy = HillClimbStrategy(
            ic="PASTIS", p_param=1e-3, patience=200, seed=7,
        )
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("PASTIS")
        assert k >= 0
        assert isinstance(support, list)

    def test_bic_criterion(self, scorer, synthetic):
        strategy = HillClimbStrategy(
            ic="BIC", tau=100.0, patience=200, seed=7,
        )
        result = strategy.run(scorer, max_k=8)
        k, support, score, coeffs = result.select_by_ic("BIC", tau=100.0)
        assert k >= 0

    def test_deterministic_with_seed(self, scorer):
        results = []
        for _ in range(2):
            s = HillClimbStrategy(ic="AIC", patience=100, seed=123)
            r = s.run(scorer, max_k=6)
            results.append(r.best_info_by_k)
        np.testing.assert_array_equal(results[0], results[1])
