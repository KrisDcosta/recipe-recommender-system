"""Tests for src/data.py."""
import pandas as pd
import pytest

from src.data import drop_zero_ratings, load_interactions, load_recipes, preprocess


class TestLoadInteractions:
    def test_returns_dataframe(self, tmp_path, raw_interactions):
        path = tmp_path / "interactions.csv"
        raw_interactions.to_csv(path, index=False)
        df = load_interactions(path)
        assert isinstance(df, pd.DataFrame)

    def test_date_parsed_to_datetime(self, tmp_path, raw_interactions):
        path = tmp_path / "interactions.csv"
        raw_interactions.to_csv(path, index=False)
        df = load_interactions(path)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_missing_required_column_raises(self, tmp_path, raw_interactions):
        path = tmp_path / "bad.csv"
        raw_interactions.drop(columns=["rating"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_interactions(path)


class TestLoadRecipes:
    def test_returns_dataframe(self, tmp_path, raw_recipes):
        path = tmp_path / "recipes.csv"
        raw_recipes.to_csv(path, index=False)
        df = load_recipes(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(raw_recipes)

    def test_missing_required_column_raises(self, tmp_path, raw_recipes):
        path = tmp_path / "bad.csv"
        raw_recipes.drop(columns=["id"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_recipes(path)


class TestPreprocess:
    def test_merge_produces_expected_columns(self, raw_interactions, raw_recipes):
        df = preprocess(raw_interactions, raw_recipes)
        assert "user_id" in df.columns
        assert "recipe_id" in df.columns
        assert "rating" in df.columns
        assert "year" in df.columns

    def test_row_count_preserved(self, raw_interactions, raw_recipes):
        df = preprocess(raw_interactions, raw_recipes)
        assert len(df) == len(raw_interactions)

    def test_nutrition_parsed(self, raw_interactions, raw_recipes):
        df = preprocess(raw_interactions, raw_recipes)
        assert "calories" in df.columns
        assert pd.api.types.is_float_dtype(df["calories"])


class TestDropZeroRatings:
    def test_removes_zeros(self, raw_interactions, raw_recipes):
        df = preprocess(raw_interactions, raw_recipes)
        cleaned = drop_zero_ratings(df)
        assert (cleaned["rating"] == 0).sum() == 0

    def test_nonzero_rows_preserved(self, raw_interactions, raw_recipes):
        df = preprocess(raw_interactions, raw_recipes)
        nonzero_before = (df["rating"] != 0).sum()
        cleaned = drop_zero_ratings(df)
        assert len(cleaned) == nonzero_before

    def test_index_reset(self, preprocessed_df):
        result = drop_zero_ratings(preprocessed_df)
        assert list(result.index) == list(range(len(result)))
