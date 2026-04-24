"""Data loading and preprocessing for the Food.com recipe recommender."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RECIPE_MERGE_COLS = ["id", "name", "ingredients", "minutes", "n_steps", "n_ingredients"]
_NUTRITION_COLS = [
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "saturated_fat_pdv", "carbs_pdv",
]


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load raw interactions CSV.

    Expected columns: user_id, recipe_id, date, rating, review
    """
    df = pd.read_csv(path)
    required = {"user_id", "recipe_id", "date", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interactions missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info("Loaded %d interactions from %s", len(df), path)
    return df


def load_recipes(path: str | Path) -> pd.DataFrame:
    """Load raw recipes CSV.

    Expected columns: id, name, minutes, n_steps, n_ingredients, nutrition, ingredients
    """
    df = pd.read_csv(path)
    required = {"id", "minutes", "n_steps", "n_ingredients"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Recipes missing columns: {missing}")
    logger.info("Loaded %d recipes from %s", len(df), path)
    return df


def _parse_nutrition(series: pd.Series) -> pd.DataFrame:
    """Parse the stringified nutrition list into float columns."""
    def _split(s: str) -> list[float | None]:
        parts = str(s).strip("[]").split(",")
        parts = [p.strip() for p in parts]
        if len(parts) < 7:
            parts += [None] * (7 - len(parts))
        return parts[:7]

    nut_df = series.apply(_split).apply(pd.Series)
    nut_df.columns = _NUTRITION_COLS
    for col in _NUTRITION_COLS:
        nut_df[col] = pd.to_numeric(nut_df[col], errors="coerce")
    return nut_df


def preprocess(
    interactions: pd.DataFrame,
    recipes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge interactions with recipe metadata and parse nutrition.

    Returns a flat DataFrame with all columns needed for modelling.
    """
    recipes = recipes.copy()

    if "nutrition" in recipes.columns:
        nut_df = _parse_nutrition(recipes["nutrition"])
        recipes = pd.concat([recipes, nut_df], axis=1)

    drop_cols = ["description", "submitted", "tags", "steps", "contributor_id", "nutrition"]
    recipes.drop(columns=[c for c in drop_cols if c in recipes.columns], inplace=True)

    df = interactions.merge(
        recipes,
        left_on="recipe_id",
        right_on="id",
        how="left",
    )

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    logger.info("Preprocessed dataset: %d rows, %d cols", *df.shape)
    return df


def drop_zero_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop interactions where rating == 0.

    Food.com allows reviews without an explicit rating; these appear as 0
    in the export. They are not 1-star ratings and pollute explicit-feedback
    models. See platform verification in FINDINGS.md.
    """
    before = len(df)
    df = df[df["rating"] != 0].reset_index(drop=True)
    dropped = before - len(df)
    logger.info("Dropped %d zero-rated interactions (%.1f%%)", dropped, 100 * dropped / before)
    return df
