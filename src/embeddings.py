"""LLM-based recipe embeddings using sentence-transformers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _is_present(val) -> bool:
    """True if val is non-None, non-NaN, and (for scalars) not the string 'nan'."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    return True


def _build_recipe_text(row: pd.Series) -> str:
    """Concatenate recipe fields into a single embedding input string."""
    parts = []
    if _is_present(row.get("name")):
        parts.append(str(row["name"]))
    if _is_present(row.get("ingredients")):
        ing = row["ingredients"]
        if isinstance(ing, list):
            parts.append(", ".join(str(x) for x in ing))
        else:
            parts.append(str(ing))
    return ". ".join(parts) if parts else ""


class RecipeEmbedder:
    """Encodes recipes into dense vectors using a sentence-transformer model.

    Usage
    -----
    embedder = RecipeEmbedder()
    embedder.fit(recipes_df)               # encode all recipes
    vec = embedder.get(recipe_id)          # lookup single vector
    mat, ids = embedder.matrix()           # full embedding matrix
    embedder.save("models/embedder.joblib")
    embedder2 = RecipeEmbedder.load("models/embedder.joblib")
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, batch_size: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._embeddings: dict[Any, np.ndarray] = {}
        self._fitted = False

    def fit(self, recipes_df: pd.DataFrame) -> "RecipeEmbedder":
        """Encode all recipes. Expects columns: recipe_id (or id), name, ingredients."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required. Run: pip install 'recipe-recommender[embeddings]'"
            ) from exc

        id_col = "recipe_id" if "recipe_id" in recipes_df.columns else "id"
        texts = recipes_df.apply(_build_recipe_text, axis=1).tolist()
        ids = recipes_df[id_col].tolist()

        logger.info("Encoding %d recipes with %s …", len(texts), self.model_name)
        model = SentenceTransformer(self.model_name)
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self._embeddings = dict(zip(ids, vecs))
        self._fitted = True
        logger.info("RecipeEmbedder fitted — %d vectors, dim=%d", len(ids), vecs.shape[1])
        return self

    def get(self, recipe_id: Any) -> np.ndarray | None:
        """Return embedding for recipe_id, or None if unknown."""
        return self._embeddings.get(recipe_id)

    def matrix(self) -> tuple[np.ndarray, list]:
        """Return (N×D embedding matrix, list of recipe_ids) in insertion order."""
        if not self._fitted:
            raise RuntimeError("Call fit() before matrix()")
        ids = list(self._embeddings.keys())
        mat = np.stack([self._embeddings[i] for i in ids])
        return mat, ids

    def similarity(self, recipe_id_a: Any, recipe_id_b: Any) -> float:
        """Cosine similarity between two recipes (embeddings are L2-normalised)."""
        a = self.get(recipe_id_a)
        b = self.get(recipe_id_b)
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def most_similar(self, recipe_id: Any, n: int = 10) -> list[tuple[Any, float]]:
        """Return top-n similar recipe (id, score) pairs, excluding the query."""
        query = self.get(recipe_id)
        if query is None:
            return []
        mat, ids = self.matrix()
        scores = mat @ query
        order = np.argsort(scores)[::-1]
        return [(ids[i], float(scores[i])) for i in order if ids[i] != recipe_id][:n]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("RecipeEmbedder saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RecipeEmbedder":
        obj = joblib.load(path)
        logger.info("RecipeEmbedder loaded from %s", path)
        return obj


def project_embeddings_2d(
    embedder: RecipeEmbedder,
    method: str = "tsne",
    sample_n: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, list]:
    """Reduce recipe embeddings to 2D for scatter-plot visualisation.

    Returns (coords of shape (N, 2), list of recipe_ids).
    Uses t-SNE by default (sklearn, no extra deps).
    Set method='umap' for faster/cleaner layout (requires: pip install umap-learn).
    sample_n caps the number of recipes projected (full 231K is slow for t-SNE).
    """
    mat, ids = embedder.matrix()
    if len(ids) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ids), size=sample_n, replace=False)
        mat = mat[idx]
        ids = [ids[i] for i in idx]

    if method == "umap":
        try:
            import umap as umap_lib
            coords = umap_lib.UMAP(n_components=2, random_state=seed).fit_transform(mat)
        except ImportError as exc:
            raise ImportError("pip install umap-learn  (or use method='tsne')") from exc
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(
            n_components=2, perplexity=min(30, len(ids) - 1),
            random_state=seed, n_jobs=-1,
        ).fit_transform(mat)

    logger.info("project_embeddings_2d: %d points → 2D via %s", len(ids), method)
    return coords, ids


def build_embedding_features(
    df: pd.DataFrame,
    embedder: RecipeEmbedder,
) -> np.ndarray:
    """Look up the embedding vector for each recipe_id in df.

    Rows with unknown recipe_id get a zero vector. Returns (N, D) array.
    """
    if not embedder._fitted:
        raise RuntimeError("embedder must be fitted before building features")
    sample = next(iter(embedder._embeddings.values()))
    dim = sample.shape[0]

    out = np.zeros((len(df), dim), dtype=np.float32)
    for i, rid in enumerate(df["recipe_id"]):
        vec = embedder.get(rid)
        if vec is not None:
            out[i] = vec
    return out
