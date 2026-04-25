"""Hybrid recommender: time-aware MF + LLM content embeddings via Ridge regression."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.embeddings import RecipeEmbedder, build_embedding_features
from src.models import BaseMF, TimeAwareMF

logger = logging.getLogger(__name__)


class HybridMF(BaseMF):
    """Two-stage hybrid model.

    Stage 1  — TimeAwareMF produces a latent-factor score.
    Stage 2  — Ridge regressor blends the MF score with LLM content features.

    This captures both collaborative signal (who rated what) and content signal
    (what the recipe is about) in a single prediction.

    Prediction: r_hat = Ridge([mf_score | embedding_features])
    """

    def __init__(
        self,
        mf: TimeAwareMF | None = None,
        embedder: RecipeEmbedder | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.mf = mf or TimeAwareMF()
        self.embedder = embedder  # must be pre-fitted; fitted inside fit() if None
        self.alpha = alpha

        self._ridge: Ridge | None = None
        self._scaler: StandardScaler | None = None
        self._mf_fitted = False
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        recipes_df: pd.DataFrame | None = None,
    ) -> "HybridMF":
        """Fit MF stage, then Ridge blending stage.

        Parameters
        ----------
        train_df  : interactions train split
        val_df    : interactions val split (used for MF early stopping)
        recipes_df: recipe metadata for embedding; required if embedder not pre-fitted
        """
        # Stage 1 — fit MF
        logger.info("HybridMF: fitting TimeAwareMF …")
        self.mf.fit(train_df, val_df)
        self._mf_fitted = True

        # Stage 1b — fit/check embedder
        if self.embedder is None or not self.embedder._fitted:
            if recipes_df is None:
                raise ValueError(
                    "Provide recipes_df or a pre-fitted RecipeEmbedder to fit HybridMF"
                )
            logger.info("HybridMF: fitting RecipeEmbedder …")
            self.embedder = RecipeEmbedder()
            self.embedder.fit(recipes_df)

        # Stage 2 — build blending features on train
        X_train, y_train = self._build_features(train_df)
        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)

        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(X_train_s, y_train)

        self._fitted = True
        logger.info("HybridMF fitted — Ridge coefs shape %s", self._ridge.coef_.shape)
        return self

    def predict(self, user_id: Any, item_id: Any, time_bin: Any = None, **_: Any) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        mf_score = self.mf.predict(user_id, item_id, time_bin=time_bin)
        emb = self.embedder.get(item_id)
        emb_vec = emb if emb is not None else np.zeros(self._emb_dim, dtype=np.float32)
        feat = np.concatenate([[mf_score], emb_vec]).reshape(1, -1)
        feat_s = self._scaler.transform(feat)
        pred = float(self._ridge.predict(feat_s)[0])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, df: pd.DataFrame, **_: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X, _ = self._build_features(df)
        X_s = self._scaler.transform(X)
        preds = self._ridge.predict(X_s)
        return np.clip(preds, 1.0, 5.0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("HybridMF saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridMF":
        obj = joblib.load(path)
        logger.info("HybridMF loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) for Ridge blending stage."""
        mf_scores = self.mf.predict_batch(df).reshape(-1, 1)
        emb_feats = build_embedding_features(df, self.embedder)
        X = np.hstack([mf_scores, emb_feats])
        y = df["rating"].to_numpy(dtype=float) if "rating" in df.columns else np.zeros(len(df))
        return X, y

    @property
    def _emb_dim(self) -> int:
        if self.embedder is None or not self.embedder._fitted:
            return 0
        mat, _ = self.embedder.matrix()
        return mat.shape[1]
