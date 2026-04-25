"""Tests for src/embeddings.py — RecipeEmbedder and feature helpers."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal stub so tests run without sentence-transformers installed.
# ---------------------------------------------------------------------------

def _install_st_stub(dim: int = 8) -> None:
    """Inject a lightweight sentence_transformers stub into sys.modules."""
    if "sentence_transformers" in sys.modules:
        return

    class _FakeModel:
        def encode(self, texts, **kwargs):
            rng = np.random.default_rng(0)
            vecs = rng.random((len(texts), dim)).astype(np.float32)
            # L2-normalise to mimic normalize_embeddings=True
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

from src.embeddings import RecipeEmbedder, _build_recipe_text, build_embedding_features, project_embeddings_2d  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def recipes_df():
    return pd.DataFrame({
        "id": [101, 102, 103],
        "name": ["Pasta Carbonara", "Caesar Salad", "Banana Bread"],
        "ingredients": [
            ["pasta", "egg", "bacon"],
            ["romaine", "croutons", "parmesan"],
            ["banana", "flour", "sugar"],
        ],
    })


@pytest.fixture
def fitted_embedder(recipes_df):
    emb = RecipeEmbedder()
    emb.fit(recipes_df)
    return emb


@pytest.fixture
def interactions_df():
    return pd.DataFrame({
        "recipe_id": [101, 102, 103, 101, 102],
        "user_id": [1, 1, 2, 2, 3],
        "rating": [5, 4, 3, 5, 4],
    })


# ---------------------------------------------------------------------------
# _build_recipe_text
# ---------------------------------------------------------------------------

class TestBuildRecipeText:
    def test_name_and_list_ingredients(self):
        row = pd.Series({"name": "Pasta", "ingredients": ["egg", "bacon"]})
        text = _build_recipe_text(row)
        assert "Pasta" in text
        assert "egg" in text

    def test_name_and_string_ingredients(self):
        row = pd.Series({"name": "Soup", "ingredients": "tomato, basil"})
        text = _build_recipe_text(row)
        assert "Soup" in text
        assert "tomato" in text

    def test_missing_name_uses_ingredients(self):
        row = pd.Series({"name": None, "ingredients": ["rice", "beans"]})
        text = _build_recipe_text(row)
        assert "rice" in text

    def test_both_missing_returns_empty(self):
        row = pd.Series({"name": None, "ingredients": None})
        assert _build_recipe_text(row) == ""


# ---------------------------------------------------------------------------
# RecipeEmbedder.fit / get / matrix
# ---------------------------------------------------------------------------

class TestRecipeEmbedderFit:
    def test_fit_returns_self(self, recipes_df):
        emb = RecipeEmbedder()
        result = emb.fit(recipes_df)
        assert result is emb

    def test_fitted_flag(self, fitted_embedder):
        assert fitted_embedder._fitted is True

    def test_all_ids_present(self, fitted_embedder, recipes_df):
        for rid in recipes_df["id"]:
            assert fitted_embedder.get(rid) is not None

    def test_embedding_dim_consistent(self, fitted_embedder, recipes_df):
        vecs = [fitted_embedder.get(rid) for rid in recipes_df["id"]]
        dims = {v.shape[0] for v in vecs}
        assert len(dims) == 1

    def test_embeddings_unit_norm(self, fitted_embedder, recipes_df):
        for rid in recipes_df["id"]:
            vec = fitted_embedder.get(rid)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_unknown_recipe_returns_none(self, fitted_embedder):
        assert fitted_embedder.get(999_999) is None

    def test_get_before_fit_returns_none(self):
        emb = RecipeEmbedder()
        assert emb.get(1) is None

    def test_matrix_before_fit_raises(self):
        emb = RecipeEmbedder()
        with pytest.raises(RuntimeError, match="fit"):
            emb.matrix()

    def test_matrix_shape(self, fitted_embedder, recipes_df):
        mat, ids = fitted_embedder.matrix()
        assert mat.shape[0] == len(recipes_df)
        assert len(ids) == len(recipes_df)

    def test_recipe_id_col_alias(self, recipes_df):
        df = recipes_df.rename(columns={"id": "recipe_id"})
        emb = RecipeEmbedder().fit(df)
        for rid in df["recipe_id"]:
            assert emb.get(rid) is not None


# ---------------------------------------------------------------------------
# RecipeEmbedder.similarity / most_similar
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_self_similarity_is_one(self, fitted_embedder, recipes_df):
        rid = recipes_df["id"].iloc[0]
        sim = fitted_embedder.similarity(rid, rid)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_in_range(self, fitted_embedder, recipes_df):
        a, b = recipes_df["id"].iloc[0], recipes_df["id"].iloc[1]
        sim = fitted_embedder.similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_unknown_id_similarity_zero(self, fitted_embedder, recipes_df):
        rid = recipes_df["id"].iloc[0]
        assert fitted_embedder.similarity(rid, 999_999) == pytest.approx(0.0)

    def test_most_similar_length(self, fitted_embedder, recipes_df):
        rid = recipes_df["id"].iloc[0]
        results = fitted_embedder.most_similar(rid, n=2)
        assert len(results) == 2

    def test_most_similar_excludes_self(self, fitted_embedder, recipes_df):
        rid = recipes_df["id"].iloc[0]
        results = fitted_embedder.most_similar(rid, n=10)
        assert all(r[0] != rid for r in results)

    def test_most_similar_unknown_returns_empty(self, fitted_embedder):
        assert fitted_embedder.most_similar(999_999) == []

    def test_most_similar_sorted_desc(self, fitted_embedder, recipes_df):
        rid = recipes_df["id"].iloc[0]
        results = fitted_embedder.most_similar(rid, n=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RecipeEmbedder.save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_roundtrip(self, fitted_embedder, recipes_df, tmp_path):
        path = tmp_path / "embedder.joblib"
        fitted_embedder.save(path)
        loaded = RecipeEmbedder.load(path)
        rid = recipes_df["id"].iloc[0]
        orig = fitted_embedder.get(rid)
        restored = loaded.get(rid)
        np.testing.assert_array_almost_equal(orig, restored)

    def test_load_preserves_fitted(self, fitted_embedder, tmp_path):
        path = tmp_path / "embedder.joblib"
        fitted_embedder.save(path)
        loaded = RecipeEmbedder.load(path)
        assert loaded._fitted is True


# ---------------------------------------------------------------------------
# build_embedding_features
# ---------------------------------------------------------------------------

class TestBuildEmbeddingFeatures:
    def test_shape(self, fitted_embedder, interactions_df, recipes_df):
        feats = build_embedding_features(interactions_df, fitted_embedder)
        _, ids = fitted_embedder.matrix()
        assert feats.shape == (len(interactions_df), fitted_embedder.get(ids[0]).shape[0])

    def test_known_recipe_nonzero(self, fitted_embedder, interactions_df):
        feats = build_embedding_features(interactions_df, fitted_embedder)
        assert np.any(feats[0] != 0.0)

    def test_unknown_recipe_zero_row(self, fitted_embedder):
        df = pd.DataFrame({"recipe_id": [999_999], "user_id": [1], "rating": [3]})
        feats = build_embedding_features(df, fitted_embedder)
        np.testing.assert_array_equal(feats[0], 0.0)

    def test_unfitted_embedder_raises(self, interactions_df):
        emb = RecipeEmbedder()
        with pytest.raises(RuntimeError, match="fitted"):
            build_embedding_features(interactions_df, emb)


# ---------------------------------------------------------------------------
# project_embeddings_2d
# ---------------------------------------------------------------------------

class TestProjectEmbeddings2d:
    def test_shape(self, fitted_embedder, recipes_df):
        coords, ids = project_embeddings_2d(fitted_embedder, method="tsne")
        assert coords.shape == (len(recipes_df), 2)
        assert len(ids) == len(recipes_df)

    def test_sample_n_caps_output(self, fitted_embedder, recipes_df):
        coords, ids = project_embeddings_2d(fitted_embedder, method="tsne", sample_n=2)
        assert len(ids) == 2
        assert coords.shape == (2, 2)

    def test_ids_are_known_recipe_ids(self, fitted_embedder, recipes_df):
        _, ids = project_embeddings_2d(fitted_embedder, method="tsne")
        known = set(recipes_df["id"].tolist())
        assert set(ids).issubset(known)

    def test_umap_missing_raises_importerror(self, fitted_embedder):
        import sys
        original = sys.modules.pop("umap", None)
        # also block the umap package if present under 'umap' key
        sys.modules["umap"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError, AttributeError)):
                project_embeddings_2d(fitted_embedder, method="umap")
        finally:
            if original is not None:
                sys.modules["umap"] = original
            else:
                sys.modules.pop("umap", None)
