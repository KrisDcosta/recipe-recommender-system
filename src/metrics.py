"""Evaluation metrics for rating prediction and top-N ranking."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating prediction
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Ranking metrics (sampled evaluation)
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain at k."""
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    y_true: list[float],
    y_pred: list[float],
    k: int,
    relevance_threshold: float = 4.0,
) -> float:
    """NDCG@k for a single user.

    Args:
        y_true: actual ratings for the candidate set (positives + negatives).
        y_pred: predicted scores for the same candidate set.
        k: cutoff rank.
        relevance_threshold: ratings >= this are considered relevant.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    binary_rel = [1.0 if r >= relevance_threshold else 0.0 for r in y_true]

    # rank by predicted score descending
    order = np.argsort(y_pred)[::-1]
    ranked_rel = [binary_rel[i] for i in order]

    # ideal: sort by actual relevance descending
    ideal_rel = sorted(binary_rel, reverse=True)

    dcg = _dcg(ranked_rel, k)
    idcg = _dcg(ideal_rel, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    y_true: list[float],
    y_pred: list[float],
    k: int,
    relevance_threshold: float = 4.0,
) -> float:
    """Recall@k for a single user."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    relevant = {i for i, r in enumerate(y_true) if r >= relevance_threshold}
    if not relevant:
        return 0.0

    order = np.argsort(y_pred)[::-1]
    top_k = set(order[:k])
    return len(relevant & top_k) / len(relevant)


def sampled_evaluation(
    model: Any,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_negatives: int = 100,
    k: int = 10,
    relevance_threshold: float = 4.0,
    seed: int = 42,
) -> dict[str, float]:
    """Compute NDCG@k and Recall@k using sampled negatives.

    For each user in test_df:
      - positives = items the user rated in test (not seen in training)
      - negatives = n_negatives random items the user has never rated
      - rank the union by model predictions
      - compute NDCG@k and Recall@k

    This follows standard sampled evaluation practice (He et al. 2017).
    """
    rng = np.random.default_rng(seed)
    all_items = np.array(list(
        set(train_df["recipe_id"].unique()) | set(test_df["recipe_id"].unique())
    ))

    # items each user has rated (train + test) — can't use as negatives
    user_rated: dict[Any, set] = defaultdict(set)
    for uid, iid in zip(train_df["user_id"], train_df["recipe_id"]):
        user_rated[uid].add(iid)
    for uid, iid in zip(test_df["user_id"], test_df["recipe_id"]):
        user_rated[uid].add(iid)

    ndcg_scores: list[float] = []
    recall_scores: list[float] = []

    for uid, group in test_df.groupby("user_id"):
        positives = list(zip(group["recipe_id"], group["rating"]))
        if not positives:
            continue

        rated = user_rated[uid]
        candidates = all_items[~np.isin(all_items, list(rated))]
        if len(candidates) == 0:
            continue

        neg_items = rng.choice(
            candidates,
            size=min(n_negatives, len(candidates)),
            replace=False,
        )

        items = [iid for iid, _ in positives] + list(neg_items)
        true_ratings = [r for _, r in positives] + [0.0] * len(neg_items)

        # get time_bin for TimeAwareMF compatibility
        time_bins = list(group["time_bin"]) if "time_bin" in group.columns else [None] * len(positives)
        time_bins_neg = [None] * len(neg_items)
        all_time_bins = time_bins + time_bins_neg

        preds = [
            model.predict(uid, iid, time_bin=tb)
            if hasattr(model, "_b_ut") else model.predict(uid, iid)
            for iid, tb in zip(items, all_time_bins)
        ]

        ndcg_scores.append(ndcg_at_k(true_ratings, preds, k, relevance_threshold))
        recall_scores.append(recall_at_k(true_ratings, preds, k, relevance_threshold))

    result = {
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "n_users_evaluated": len(ndcg_scores),
    }
    logger.info("Sampled evaluation (k=%d, n_neg=%d): %s", k, n_negatives, result)
    return result


# ---------------------------------------------------------------------------
# Statistical comparison (A/B framework)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval on the RMSE difference (A - B).

    A positive delta means B is better (lower RMSE).
    p_value is the fraction of bootstrap samples where A ≥ B (one-sided).

    Args:
        y_true: ground truth ratings.
        y_pred_a: predictions from model A.
        y_pred_b: predictions from model B.
        n_bootstrap: number of bootstrap resamples.
        alpha: significance level (default 0.05 → 95% CI).
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    n = len(y_true)
    observed_delta = rmse(y_true, y_pred_a) - rmse(y_true, y_pred_b)

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = rmse(y_true[idx], y_pred_a[idx]) - rmse(y_true[idx], y_pred_b[idx])

    ci_lo = float(np.percentile(deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    # p-value: fraction of bootstrap samples where delta ≤ 0 (B not better)
    p_value = float(np.mean(deltas <= 0))

    return {
        "rmse_a": rmse(y_true, y_pred_a),
        "rmse_b": rmse(y_true, y_pred_b),
        "observed_delta": observed_delta,  # positive → B is better
        f"ci_{int((1-alpha)*100)}_lo": ci_lo,
        f"ci_{int((1-alpha)*100)}_hi": ci_hi,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
    }


def paired_ttest(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """Paired t-test on per-sample squared errors.

    Tests H0: mean(SE_a) == mean(SE_b).
    A significant result means the difference in RMSE is unlikely due to chance.
    """
    y_true = np.asarray(y_true, dtype=float)
    se_a = (y_true - np.asarray(y_pred_a, dtype=float)) ** 2
    se_b = (y_true - np.asarray(y_pred_b, dtype=float)) ** 2
    t_stat, p_value = stats.ttest_rel(se_a, se_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "mean_se_a": float(se_a.mean()),
        "mean_se_b": float(se_b.mean()),
    }


# ---------------------------------------------------------------------------
# Convenience import
# ---------------------------------------------------------------------------

from collections import defaultdict  # noqa: E402 (needed for sampled_evaluation)
