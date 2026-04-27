"""Vector search abstraction for recipe embeddings.

FAISS is used when installed; otherwise callers can continue to use the
RecipeEmbedder brute-force path. Keeping FAISS optional lets tests run on
platforms where faiss-cpu wheels are unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.embeddings import RecipeEmbedder

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """In-memory FAISS index over normalized recipe embeddings."""

    def __init__(self, index: Any, recipe_ids: list[Any]) -> None:
        self.index = index
        self.recipe_ids = recipe_ids
        self._id_to_pos = {rid: i for i, rid in enumerate(recipe_ids)}

    @classmethod
    def from_embedder(cls, embedder: RecipeEmbedder) -> "FaissVectorStore":
        """Build an IndexFlatIP from an already fitted RecipeEmbedder."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for FaissVectorStore") from exc

        mat, recipe_ids = embedder.matrix()
        mat = np.asarray(mat, dtype=np.float32)
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        logger.info("FAISS vector store built: %d vectors × %d dims", mat.shape[0], mat.shape[1])
        return cls(index=index, recipe_ids=list(recipe_ids))

    def most_similar(self, recipe_id: Any, n: int = 10) -> list[tuple[Any, float]]:
        """Return top-n similar recipe ids and scores, excluding the query id."""
        pos = self._id_to_pos.get(recipe_id)
        if pos is None:
            return []
        query = self._reconstruct(pos).reshape(1, -1)
        scores, idx = self.index.search(query, n + 1)

        out: list[tuple[Any, float]] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            rid = self.recipe_ids[int(i)]
            if rid == recipe_id:
                continue
            out.append((rid, float(score)))
            if len(out) >= n:
                break
        return out

    def save(self, path: str | Path) -> None:
        """Persist the index and id mapping with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FAISS vector store saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        obj = joblib.load(path)
        logger.info("FAISS vector store loaded from %s", path)
        return obj

    def _reconstruct(self, pos: int) -> np.ndarray:
        """Return vector at position. Handles FAISS wrappers with different APIs."""
        try:
            return self.index.reconstruct(pos)
        except TypeError:
            out = np.empty(self.index.d, dtype=np.float32)
            self.index.reconstruct(pos, out)
            return out


def build_faiss_store_if_available(embedder: RecipeEmbedder | None) -> FaissVectorStore | None:
    """Best-effort FAISS store creation used by API startup."""
    if embedder is None:
        return None
    try:
        return FaissVectorStore.from_embedder(embedder)
    except ImportError:
        logger.warning("faiss-cpu not installed; /similar will use brute-force embedding search")
        return None
