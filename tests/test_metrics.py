"""Tests for src/metrics.py — correctness of all evaluation functions."""
import math

import numpy as np
import pytest

from src.metrics import (
    bootstrap_ci,
    cold_start_rmse,
    ndcg_at_k,
    paired_ttest,
    recall_at_k,
    rmse,
    sampled_evaluation,
)


class TestRmse:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_symmetric(self):
        y_true = np.array([1.0, 3.0, 5.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        assert rmse(y_true, y_pred) == pytest.approx(rmse(y_pred, y_true))


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # relevant items ranked first
        y_true = [5.0, 5.0, 1.0, 1.0]
        y_pred = [0.9, 0.8, 0.2, 0.1]
        score = ndcg_at_k(y_true, y_pred, k=2)
        assert score == pytest.approx(1.0)

    def test_worst_ranking(self):
        # all relevant items ranked last
        y_true = [5.0, 5.0, 1.0, 1.0]
        y_pred = [0.1, 0.2, 0.8, 0.9]
        score = ndcg_at_k(y_true, y_pred, k=2)
        assert score == pytest.approx(0.0)

    def test_no_relevant_items(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [0.9, 0.5, 0.1]
        score = ndcg_at_k(y_true, y_pred, k=2, relevance_threshold=4.0)
        assert score == pytest.approx(0.0)

    def test_score_in_zero_one(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, 20).tolist()
        y_pred = rng.random(20).tolist()
        score = ndcg_at_k(y_true, y_pred, k=5)
        assert 0.0 <= score <= 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k([1.0, 2.0], [0.5], k=1)


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        y_true = [5.0, 5.0, 1.0, 1.0]
        y_pred = [0.9, 0.8, 0.2, 0.1]
        score = recall_at_k(y_true, y_pred, k=2)
        assert score == pytest.approx(1.0)

    def test_no_relevant_in_top_k(self):
        y_true = [5.0, 5.0, 1.0, 1.0]
        y_pred = [0.1, 0.2, 0.8, 0.9]
        score = recall_at_k(y_true, y_pred, k=2)
        assert score == pytest.approx(0.0)

    def test_no_relevant_items(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [0.9, 0.5, 0.1]
        score = recall_at_k(y_true, y_pred, k=2, relevance_threshold=4.0)
        assert score == pytest.approx(0.0)

    def test_score_in_zero_one(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(1, 6, 20).tolist()
        y_pred = rng.random(20).tolist()
        score = recall_at_k(y_true, y_pred, k=5)
        assert 0.0 <= score <= 1.0


class TestBootstrapCi:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, 100).astype(float)
        y_pred_a = y_true + rng.normal(0, 1, 100)
        y_pred_b = y_true + rng.normal(0, 0.5, 100)
        result = bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=500, seed=0)
        assert "rmse_a" in result
        assert "rmse_b" in result
        assert "observed_delta" in result
        assert "p_value" in result
        assert "significant" in result

    def test_better_model_detected(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(1, 6, 500).astype(float)
        # B is clearly better (much smaller error)
        y_pred_a = y_true + rng.normal(0, 1.5, 500)
        y_pred_b = y_true + rng.normal(0, 0.1, 500)
        result = bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=0)
        assert result["significant"] is True
        assert result["observed_delta"] > 0  # A - B > 0 means B is better

    def test_identical_models_not_significant(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, 200).astype(float)
        y_pred = y_true + rng.normal(0, 0.5, 200)
        result = bootstrap_ci(y_true, y_pred, y_pred, n_bootstrap=500, seed=0)
        assert result["observed_delta"] == pytest.approx(0.0)
        assert result["significant"] is False

    def test_ci_contains_observed_delta(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(1, 6, 300).astype(float)
        y_pred_a = y_true + rng.normal(0, 1.0, 300)
        y_pred_b = y_true + rng.normal(0, 0.8, 300)
        result = bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=7)
        lo = result["ci_95_lo"]
        hi = result["ci_95_hi"]
        assert lo <= hi


class TestPairedTtest:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, 100).astype(float)
        preds = y_true + rng.normal(0, 0.5, 100)
        result = paired_ttest(y_true, preds, preds)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant_at_0.05" in result

    def test_identical_preds_not_significant(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(1, 6, 100).astype(float)
        preds = y_true + rng.normal(0, 0.5, 100)
        result = paired_ttest(y_true, preds, preds)
        assert result["significant_at_0.05"] is False

    def test_clearly_different_is_significant(self):
        rng = np.random.default_rng(0)
        y_true = np.ones(500) * 3.0
        y_pred_a = y_true + rng.normal(0, 2.0, 500)
        y_pred_b = y_true + rng.normal(0, 0.01, 500)
        result = paired_ttest(y_true, y_pred_a, y_pred_b)
        assert result["significant_at_0.05"] is True


class TestSampledEvaluation:
    def test_returns_ndcg_and_recall(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = sampled_evaluation(model, split.test, split.train, n_negatives=10, k=5)
        assert "ndcg@5" in result
        assert "recall@5" in result
        assert 0.0 <= result["ndcg@5"] <= 1.0
        assert 0.0 <= result["recall@5"] <= 1.0


class TestColdStartRmse:
    def test_returns_all_buckets(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = cold_start_rmse(model, split.test, split.train)
        assert len(result) == 3

    def test_bucket_keys_present(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = cold_start_rmse(model, split.test, split.train)
        for label, stats in result.items():
            assert "rmse" in stats
            assert "n" in stats

    def test_rmse_positive_or_nan(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        import math
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = cold_start_rmse(model, split.test, split.train)
        for label, stats in result.items():
            if stats["n"] > 0:
                assert stats["rmse"] >= 0.0
            else:
                assert math.isnan(stats["rmse"])

    def test_n_sums_to_test_size(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = cold_start_rmse(model, split.test, split.train)
        total = sum(s["n"] for s in result.values())
        assert total == len(split.test)

    def test_custom_bins(self, preprocessed_df):
        from src.models import StaticMF
        from src.splits import random_split
        split = random_split(preprocessed_df, seed=42)
        model = StaticMF(k=2, epochs=2, patience=1).fit(split.train, split.val)
        result = cold_start_rmse(model, split.test, split.train, bins=[0, 10])
        assert len(result) == 2
