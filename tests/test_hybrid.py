"""Tests for src/hybrid.py — HybridMF."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# sentence_transformers stub (same pattern as test_embeddings.py)
# ---------------------------------------------------------------------------

def _install_st_stub(dim: int = 8) -> None:
    if "sentence_transformers" in sys.modules:
        return

    class _FakeModel:
        def encode(self, texts, **kwargs):
            rng = np.random.default_rng(0)
            vecs = rng.random((len(texts), dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.maximum(norms, 1e-8)

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self._m = _FakeModel()

        def encode(self, texts, **kwargs):
            return self._m.encode(texts, **kwargs)

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = stub


_install_st_stub()

from src.embeddings import RecipeEmbedder  # noqa: E402
from src.hybrid import HybridMF  # noqa: E402
from src.models import TimeAwareMF  # noqa: E402
from src.splits import random_split  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipes_df():
    return pd.DataFrame({
        "id": list(range(30)),
        "name": [f"Recipe {i}" for i in range(30)],
        "ingredients": [[f"ing{i}a", f"ing{i}b"] for i in range(30)],
    })


@pytest.fixture(scope="module")
def fitted_embedder(recipes_df):
    emb = RecipeEmbedder()
    emb.fit(recipes_df)
    return emb


@pytest.fixture(scope="module")
def split(preprocessed_df):
    return random_split(preprocessed_df, seed=42)


@pytest.fixture(scope="module")
def fitted_hybrid(split, fitted_embedder):
    model = HybridMF(
        mf=TimeAwareMF(k=2, epochs=2, patience=1),
        embedder=fitted_embedder,
    )
    model.fit(split.train, split.val)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHybridMFFit:
    def test_fit_returns_self(self, split, fitted_embedder):
        model = HybridMF(
            mf=TimeAwareMF(k=2, epochs=2, patience=1),
            embedder=fitted_embedder,
        )
        result = model.fit(split.train, split.val)
        assert result is model

    def test_fitted_flag(self, fitted_hybrid):
        assert fitted_hybrid._fitted is True

    def test_mf_also_fitted(self, fitted_hybrid):
        assert fitted_hybrid._mf_fitted is True

    def test_fit_no_embedder_no_recipes_raises(self, split):
        model = HybridMF(mf=TimeAwareMF(k=2, epochs=2, patience=1), embedder=None)
        with pytest.raises(ValueError, match="recipes_df"):
            model.fit(split.train, split.val)

    def test_fit_with_recipes_df(self, split, recipes_df):
        model = HybridMF(mf=TimeAwareMF(k=2, epochs=2, patience=1))
        model.fit(split.train, split.val, recipes_df=recipes_df)
        assert model._fitted


class TestHybridMFPredict:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            HybridMF().predict(1, 1)

    def test_predict_in_range(self, fitted_hybrid, split):
        for _, row in split.test.head(20).iterrows():
            p = fitted_hybrid.predict(row["user_id"], row["recipe_id"])
            assert 1.0 <= p <= 5.0, f"prediction {p} out of [1, 5]"

    def test_predict_unknown_user(self, fitted_hybrid, split):
        p = fitted_hybrid.predict(999_999_999, split.test.iloc[0]["recipe_id"])
        assert 1.0 <= p <= 5.0

    def test_predict_unknown_item(self, fitted_hybrid, split):
        p = fitted_hybrid.predict(split.test.iloc[0]["user_id"], 999_999_999)
        assert 1.0 <= p <= 5.0

    def test_predict_batch_length(self, fitted_hybrid, split):
        preds = fitted_hybrid.predict_batch(split.test.head(20))
        assert len(preds) == 20

    def test_predict_batch_in_range(self, fitted_hybrid, split):
        preds = fitted_hybrid.predict_batch(split.test.head(20))
        assert (preds >= 1.0).all() and (preds <= 5.0).all()

    def test_predictions_have_variance(self, fitted_hybrid, split):
        preds = fitted_hybrid.predict_batch(split.test)
        assert np.std(preds) > 0.0


class TestHybridMFSaveLoad:
    def test_save_load_roundtrip(self, fitted_hybrid, split, tmp_path):
        path = tmp_path / "hybrid.joblib"
        fitted_hybrid.save(path)
        loaded = HybridMF.load(path)
        row = split.test.iloc[0]
        p1 = fitted_hybrid.predict(row["user_id"], row["recipe_id"])
        p2 = loaded.predict(row["user_id"], row["recipe_id"])
        assert abs(p1 - p2) < 1e-5
