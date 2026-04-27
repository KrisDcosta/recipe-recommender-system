"""Tests for optional FAISS vector search wrapper."""
from __future__ import annotations

import sys
import importlib.util
import types

import numpy as np
import pandas as pd
import pytest

from src.embeddings import RecipeEmbedder
from src.vector_store import FaissVectorStore, build_faiss_store_if_available


pytestmark = pytest.mark.skipif(
    "faiss" not in sys.modules and importlib.util.find_spec("faiss") is None,
    reason="faiss-cpu is optional on local test platforms",
)


def _install_st_stub(dim: int = 8) -> None:
    """Inject a lightweight sentence_transformers stub into sys.modules."""
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
    _install_st_stub()
    emb = RecipeEmbedder()
    emb.fit(recipes_df)
    return emb


class TestFaissVectorStore:
    def test_from_embedder_returns_same_neighbours_as_bruteforce(self, fitted_embedder, recipes_df):
        store = FaissVectorStore.from_embedder(fitted_embedder)
        rid = recipes_df.iloc[0]["id"]
        faiss_results = store.most_similar(rid, n=3)
        brute_results = fitted_embedder.most_similar(rid, n=3)
        assert [x[0] for x in faiss_results] == [x[0] for x in brute_results]

    def test_unknown_recipe_returns_empty(self, fitted_embedder):
        store = build_faiss_store_if_available(fitted_embedder)
        assert store is not None
        assert store.most_similar(999_999, n=3) == []
