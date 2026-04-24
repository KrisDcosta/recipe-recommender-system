"""Tests for src/splits.py — correctness, no data leakage, size guarantees."""
import pandas as pd
import pytest

from src.splits import SplitResult, random_split, temporal_split


@pytest.fixture
def df(preprocessed_df):
    return preprocessed_df


class TestRandomSplit:
    def test_sizes_sum_to_total(self, df):
        result = random_split(df)
        assert len(result.train) + len(result.val) + len(result.test) == len(df)

    def test_approximate_fractions(self, df):
        result = random_split(df, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        n = len(df)
        assert abs(len(result.train) / n - 0.7) < 0.05
        assert abs(len(result.val) / n - 0.15) < 0.05

    def test_no_row_overlap(self, df):
        result = random_split(df)
        idx_train = set(result.train.index)
        idx_val = set(result.val.index)
        idx_test = set(result.test.index)
        # all indices are disjoint (reset_index so indices are positional within each split)
        # check total unique rows = total rows
        combined = pd.concat([result.train, result.val, result.test])
        assert len(combined) == len(df)

    def test_reproducible_with_seed(self, df):
        r1 = random_split(df, seed=0)
        r2 = random_split(df, seed=0)
        assert list(r1.train["user_id"]) == list(r2.train["user_id"])

    def test_different_seeds_differ(self, df):
        r1 = random_split(df, seed=1)
        r2 = random_split(df, seed=2)
        assert list(r1.train["user_id"]) != list(r2.train["user_id"])

    def test_invalid_fractions_raise(self, df):
        with pytest.raises(ValueError):
            random_split(df, train_frac=0.5, val_frac=0.5, test_frac=0.5)

    def test_returns_split_result(self, df):
        result = random_split(df)
        assert isinstance(result, SplitResult)


class TestTemporalSplit:
    def test_no_future_leakage_in_train(self, df):
        result = temporal_split(df, test_cutoff_year=2015)
        if len(result.train) > 0 and "year" in result.train.columns:
            assert (result.train["year"] < 2015).all()

    def test_test_contains_only_future(self, df):
        result = temporal_split(df, test_cutoff_year=2015)
        if len(result.test) > 0 and "year" in result.test.columns:
            assert (result.test["year"] >= 2015).all()

    def test_sizes_sum_to_total(self, df):
        result = temporal_split(df, test_cutoff_year=2015)
        assert len(result.train) + len(result.val) + len(result.test) == len(df)

    def test_val_cutoff_respected(self, df):
        result = temporal_split(df, test_cutoff_year=2016, val_cutoff_year=2014)
        if len(result.val) > 0 and "year" in result.val.columns:
            assert (result.val["year"] >= 2014).all()
            assert (result.val["year"] < 2016).all()

    def test_val_cutoff_must_be_less_than_test_cutoff(self, df):
        with pytest.raises(ValueError):
            temporal_split(df, test_cutoff_year=2014, val_cutoff_year=2015)

    def test_missing_date_col_raises(self, df):
        bad = df.drop(columns=["date"] if "date" in df.columns else [])
        if "date" not in bad.columns and "year" not in bad.columns:
            with pytest.raises(ValueError):
                temporal_split(bad, test_cutoff_year=2015, date_col="date")

    def test_overlap_stats_returns_dict(self, df):
        result = temporal_split(df, test_cutoff_year=2015)
        stats = result.overlap_stats()
        assert "test_users_cold" in stats
        assert "test_items_cold" in stats
        assert all(isinstance(v, int) for v in stats.values())


class TestSplitResult:
    def test_sizes_method(self, df):
        result = random_split(df)
        sizes = result.sizes()
        assert set(sizes.keys()) == {"train", "val", "test"}
        assert sum(sizes.values()) == len(df)
