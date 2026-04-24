"""Train / val / test splitting strategies for the recipe recommender."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def overlap_stats(self) -> dict[str, dict[str, int]]:
        """User and item overlap between splits — key cold-start diagnostic."""
        train_users = set(self.train["user_id"])
        train_items = set(self.train["recipe_id"])
        val_users = set(self.val["user_id"])
        val_items = set(self.val["recipe_id"])
        test_users = set(self.test["user_id"])
        test_items = set(self.test["recipe_id"])
        return {
            "val_users_in_train": len(val_users & train_users),
            "val_users_cold": len(val_users - train_users),
            "test_users_in_train": len(test_users & train_users),
            "test_users_cold": len(test_users - train_users),
            "val_items_in_train": len(val_items & train_items),
            "val_items_cold": len(val_items - train_items),
            "test_items_in_train": len(test_items & train_items),
            "test_items_cold": len(test_items - train_items),
        }


def random_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Randomly shuffle and split interactions.

    Warm-start evaluation: users and items appear across splits.
    Fractions must sum to 1.0.
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[idx[:n_train]].reset_index(drop=True)
    val = df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    test = df.iloc[idx[n_train + n_val :]].reset_index(drop=True)

    result = SplitResult(train=train, val=val, test=test)
    logger.info("Random split — %s", result.sizes())
    return result


def temporal_split(
    df: pd.DataFrame,
    test_cutoff_year: int = 2015,
    val_cutoff_year: int | None = None,
    date_col: str = "date",
) -> SplitResult:
    """Split by time: train on historical data, evaluate on future interactions.

    Simulates deployment: the model has never seen any test-set timestamps.

    Args:
        test_cutoff_year: interactions from this year onward go to test.
        val_cutoff_year: if given, interactions in [val_cutoff_year, test_cutoff_year)
            go to val; otherwise val is empty (use random val from training for
            hyperparameter tuning, temporal split only for final evaluation).
        date_col: column containing datetime or year values.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    years = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(years):
        years = years.dt.year
    elif years.dtype == object:
        years = pd.to_datetime(years, errors="coerce").dt.year

    test_mask = years >= test_cutoff_year

    if val_cutoff_year is not None:
        if val_cutoff_year >= test_cutoff_year:
            raise ValueError("val_cutoff_year must be < test_cutoff_year")
        val_mask = (years >= val_cutoff_year) & ~test_mask
        train_mask = years < val_cutoff_year
    else:
        val_mask = pd.Series(False, index=df.index)
        train_mask = ~test_mask

    train = df[train_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)

    result = SplitResult(train=train, val=val, test=test)
    stats = result.overlap_stats()
    logger.info(
        "Temporal split (test>=%d) — sizes: %s", test_cutoff_year, result.sizes()
    )
    logger.info(
        "Cold-start: %d cold users / %d cold items in test",
        stats["test_users_cold"],
        stats["test_items_cold"],
    )
    return result
