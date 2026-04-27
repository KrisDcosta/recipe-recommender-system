"""Tests for optional FAISS vector search wrapper."""
from __future__ import annotations

import sys
import importlib.util

import pytest

from src.vector_store import FaissVectorStore, build_faiss_store_if_available


pytestmark = pytest.mark.skipif(
    "faiss" not in sys.modules and importlib.util.find_spec("faiss") is None,
    reason="faiss-cpu is optional on local test platforms",
)


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
