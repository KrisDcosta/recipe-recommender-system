"""Shared pytest fixtures using synthetic data — no real dataset required."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def raw_interactions(rng):
    """500 synthetic interactions: 50 users × 30 recipes, years 2005–2020."""
    n = 500
    users = rng.integers(1, 51, n)
    recipes = rng.integers(1, 31, n)
    ratings = rng.integers(1, 6, n).astype(float)
    # inject some zeros to test zero-dropping
    zero_mask = rng.random(n) < 0.1
    ratings[zero_mask] = 0.0
    years = rng.integers(2005, 2021, n)
    months = rng.integers(1, 13, n)
    days = rng.integers(1, 29, n)
    dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)]
    return pd.DataFrame({
        "user_id": users,
        "recipe_id": recipes,
        "date": dates,
        "rating": ratings,
        "review": ["ok"] * n,
    })


@pytest.fixture(scope="session")
def raw_recipes(rng):
    """30 synthetic recipes."""
    n = 30
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "name": [f"recipe_{i}" for i in range(1, n + 1)],
        "minutes": rng.integers(10, 120, n),
        "n_steps": rng.integers(2, 15, n),
        "n_ingredients": rng.integers(3, 20, n),
        "nutrition": [str([float(rng.integers(100, 500))] + [float(rng.integers(0, 50))] * 6)] * n,
        "ingredients": ["['salt', 'pepper']"] * n,
        "description": ["A tasty recipe."] * n,
    })


@pytest.fixture(scope="session")
def preprocessed_df(raw_interactions, raw_recipes):
    from src.data import preprocess, drop_zero_ratings
    df = preprocess(raw_interactions, raw_recipes)
    return drop_zero_ratings(df)
