"""Matrix factorization models for explicit rating prediction."""
from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseMF(ABC):
    """Abstract base for SGD matrix factorization models."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "BaseMF":
        ...

    @abstractmethod
    def predict(self, user_id: Any, item_id: Any, **kwargs: Any) -> float:
        ...

    def predict_batch(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Predict ratings for every row in df."""
        return np.array([
            self.predict(row["user_id"], row["recipe_id"], **kwargs)
            for _, row in df.iterrows()
        ])

    def save(self, path: str | Path) -> None:
        """Persist model to disk with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseMF":
        """Load a previously saved model."""
        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model


class StaticMF(BaseMF):
    """Matrix factorisation with user and item bias terms, trained with SGD.

    Prediction: r_ui = alpha + b_u + b_i + <p_u, q_i>

    Based on: Koren, Bell & Volinsky (2009) — "Matrix Factorization Techniques
    for Recommender Systems", IEEE Computer.
    """

    def __init__(
        self,
        k: int = 10,
        lr: float = 0.01,
        lamb: float = 0.02,
        epochs: int = 10,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.lr = lr
        self.lamb = lamb
        self.epochs = epochs
        self.patience = patience
        self.seed = seed

        self._alpha: float = 0.0
        self._b_u: dict[Any, float] = {}
        self._b_i: dict[Any, float] = {}
        self._P: dict[Any, np.ndarray] = {}
        self._Q: dict[Any, np.ndarray] = {}
        self._global_mean: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "StaticMF":
        random.seed(self.seed)
        np.random.seed(self.seed)

        self._global_mean = float(train_df["rating"].mean())
        self._alpha = self._global_mean

        triples = list(zip(train_df["user_id"], train_df["recipe_id"], train_df["rating"]))
        val_triples = list(zip(val_df["user_id"], val_df["recipe_id"], val_df["rating"]))

        users = {u for u, _, _ in triples}
        items = {i for _, i, _ in triples}

        self._b_u = {u: 0.0 for u in users}
        self._b_i = {i: 0.0 for i in items}
        self._P = {u: np.random.normal(0, 0.05, self.k).astype(np.float32) for u in users}
        self._Q = {i: np.random.normal(0, 0.05, self.k).astype(np.float32) for i in items}

        best_val = float("inf")
        best_state: tuple | None = None
        wait = 0

        for ep in range(1, self.epochs + 1):
            random.shuffle(triples)
            self._sgd_epoch(triples)
            val_rmse = self._eval_rmse(val_triples)
            train_rmse = self._eval_rmse(triples[:20_000])
            logger.debug("Epoch %2d | train_rmse=%.4f | val_rmse=%.4f", ep, train_rmse, val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = self._snapshot()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (best val RMSE %.4f)", ep, best_val)
                    break

        self._restore(best_state)
        self._fitted = True
        logger.info("StaticMF fitted — best val RMSE %.4f", best_val)
        return self

    def predict(self, user_id: Any, item_id: Any, **_: Any) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        return self._predict_raw(user_id, item_id)

    def _predict_raw(self, user_id: Any, item_id: Any) -> float:
        b_u = self._b_u.get(user_id, 0.0)
        b_i = self._b_i.get(item_id, 0.0)
        p_u = self._P.get(user_id)
        q_i = self._Q.get(item_id)
        mf = float(np.dot(p_u, q_i)) if (p_u is not None and q_i is not None) else 0.0
        return float(np.clip(self._alpha + b_u + b_i + mf, 1.0, 5.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sgd_epoch(self, triples: list[tuple]) -> None:
        lr, lamb = self.lr, self.lamb
        for u, i, r in triples:
            if u not in self._P or i not in self._Q:
                continue
            p_u, q_i = self._P[u], self._Q[i]
            err = r - (self._alpha + self._b_u[u] + self._b_i[i] + float(np.dot(p_u, q_i)))
            self._b_u[u] += lr * (err - lamb * self._b_u[u])
            self._b_i[i] += lr * (err - lamb * self._b_i[i])
            self._P[u] += lr * (err * q_i - lamb * p_u)
            self._Q[i] += lr * (err * p_u - lamb * q_i)

        resid = sum(
            r - (self._alpha + self._b_u.get(u, 0.0) + self._b_i.get(i, 0.0)
                 + float(np.dot(self._P[u], self._Q[i]))
                 if u in self._P and i in self._Q else
                 r - self._alpha)
            for u, i, r in triples[:20_000]
        )
        self._alpha += 0.01 * (resid / max(len(triples), 1))

    def _eval_rmse(self, triples: list[tuple]) -> float:
        if not triples:
            return float("inf")
        errors = [
            (r - self._predict_raw(u, i)) ** 2
            for u, i, r in triples
        ]
        return math.sqrt(sum(errors) / len(errors))

    def _snapshot(self) -> tuple:
        return (
            self._alpha,
            dict(self._b_u),
            dict(self._b_i),
            {u: v.copy() for u, v in self._P.items()},
            {i: v.copy() for i, v in self._Q.items()},
        )

    def _restore(self, state: tuple | None) -> None:
        if state is None:
            return
        self._alpha, self._b_u, self._b_i, self._P, self._Q = state


class TimeAwareMF(BaseMF):
    """Matrix factorisation with user×time and item×time bias terms.

    Prediction: r_ui = alpha + b_u + b_i + b_ut + b_it + <p_u, q_i>

    Captures temporal rating drift — empirically significant in the Food.com
    dataset which spans 18 years (2000–2018).

    Reference: Koren (2009) — "Collaborative Filtering with Temporal Dynamics",
    KDD 2009.
    """

    def __init__(
        self,
        k: int = 5,
        lr: float = 0.01,
        lamb: float = 0.02,
        epochs: int = 10,
        patience: int = 3,
        time_mode: str = "year",
        seed: int = 42,
    ) -> None:
        if time_mode not in ("year", "month"):
            raise ValueError("time_mode must be 'year' or 'month'")
        self.k = k
        self.lr = lr
        self.lamb = lamb
        self.epochs = epochs
        self.patience = patience
        self.time_mode = time_mode
        self.seed = seed

        self._alpha: float = 0.0
        self._b_u: dict[Any, float] = {}
        self._b_i: dict[Any, float] = {}
        self._b_ut: dict[tuple, float] = {}
        self._b_it: dict[tuple, float] = {}
        self._P: dict[Any, np.ndarray] = {}
        self._Q: dict[Any, np.ndarray] = {}
        self._global_mean: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "TimeAwareMF":
        random.seed(self.seed)
        np.random.seed(self.seed)

        train_df = self._ensure_time_bin(train_df)
        val_df = self._ensure_time_bin(val_df)

        train_df = train_df.dropna(subset=["time_bin"])
        val_df = val_df.dropna(subset=["time_bin"])

        self._global_mean = float(train_df["rating"].mean())
        self._alpha = self._global_mean

        quads = list(zip(
            train_df["user_id"], train_df["recipe_id"],
            train_df["time_bin"], train_df["rating"],
        ))
        val_quads = list(zip(
            val_df["user_id"], val_df["recipe_id"],
            val_df["time_bin"], val_df["rating"],
        ))

        users = {u for u, _, _, _ in quads}
        items = {i for _, i, _, _ in quads}

        self._b_u = {u: 0.0 for u in users}
        self._b_i = {i: 0.0 for i in items}
        self._b_ut = {}
        self._b_it = {}
        self._P = {u: np.random.normal(0, 0.05, self.k).astype(np.float32) for u in users}
        self._Q = {i: np.random.normal(0, 0.05, self.k).astype(np.float32) for i in items}

        best_val = float("inf")
        best_state: tuple | None = None
        wait = 0

        for ep in range(1, self.epochs + 1):
            random.shuffle(quads)
            self._sgd_epoch(quads)
            val_rmse = self._eval_rmse(val_quads)
            train_rmse = self._eval_rmse(quads[:20_000])
            logger.debug("Epoch %2d | train_rmse=%.4f | val_rmse=%.4f", ep, train_rmse, val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = self._snapshot()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (best val RMSE %.4f)", ep, best_val)
                    break

        self._restore(best_state)
        self._fitted = True
        logger.info("TimeAwareMF fitted — best val RMSE %.4f", best_val)
        return self

    def predict(self, user_id: Any, item_id: Any, time_bin: Any = None, **_: Any) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        return self._predict_raw(user_id, item_id, time_bin)

    def _predict_raw(self, user_id: Any, item_id: Any, time_bin: Any = None) -> float:
        b_u = self._b_u.get(user_id, 0.0)
        b_i = self._b_i.get(item_id, 0.0)
        b_ut = self._b_ut.get((user_id, time_bin), 0.0) if time_bin is not None else 0.0
        b_it = self._b_it.get((item_id, time_bin), 0.0) if time_bin is not None else 0.0
        p_u = self._P.get(user_id)
        q_i = self._Q.get(item_id)
        mf = float(np.dot(p_u, q_i)) if (p_u is not None and q_i is not None) else 0.0
        return float(np.clip(self._alpha + b_u + b_i + b_ut + b_it + mf, 1.0, 5.0))

    def predict_batch(self, df: pd.DataFrame, **_: Any) -> np.ndarray:
        df = self._ensure_time_bin(df)
        return np.array([
            self.predict(row["user_id"], row["recipe_id"], time_bin=row.get("time_bin"))
            for _, row in df.iterrows()
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_time_bin(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time_bin" in df.columns:
            return df
        df = df.copy()
        if "date" in df.columns:
            dt = df["date"] if pd.api.types.is_datetime64_any_dtype(df["date"]) \
                else pd.to_datetime(df["date"], errors="coerce")
            if self.time_mode == "year":
                df["time_bin"] = dt.dt.year
            else:
                df["time_bin"] = dt.dt.year * 100 + dt.dt.month
        else:
            df["time_bin"] = None
        return df

    def _sgd_epoch(self, quads: list[tuple]) -> None:
        lr, lamb = self.lr, self.lamb
        for u, i, t, r in quads:
            if u not in self._P or i not in self._Q:
                continue
            p_u, q_i = self._P[u], self._Q[i]
            b_ut = self._b_ut.get((u, t), 0.0)
            b_it = self._b_it.get((i, t), 0.0)
            pred = self._alpha + self._b_u[u] + self._b_i[i] + b_ut + b_it + float(np.dot(p_u, q_i))
            err = r - pred
            self._b_u[u] += lr * (err - lamb * self._b_u[u])
            self._b_i[i] += lr * (err - lamb * self._b_i[i])
            self._b_ut[(u, t)] = b_ut + lr * (err - lamb * b_ut)
            self._b_it[(i, t)] = b_it + lr * (err - lamb * b_it)
            self._P[u] += lr * (err * q_i - lamb * p_u)
            self._Q[i] += lr * (err * p_u - lamb * q_i)

        resid = sum(
            r - self._alpha
            for _, _, _, r in quads[:20_000]
        )
        self._alpha += 0.01 * (resid / max(len(quads), 1))

    def _eval_rmse(self, quads: list[tuple]) -> float:
        if not quads:
            return float("inf")
        errors = [
            (r - self._predict_raw(u, i, time_bin=t)) ** 2
            for u, i, t, r in quads
        ]
        return math.sqrt(sum(errors) / len(errors))

    def _snapshot(self) -> tuple:
        return (
            self._alpha,
            dict(self._b_u),
            dict(self._b_i),
            dict(self._b_ut),
            dict(self._b_it),
            {u: v.copy() for u, v in self._P.items()},
            {i: v.copy() for i, v in self._Q.items()},
        )

    def _restore(self, state: tuple | None) -> None:
        if state is None:
            return
        self._alpha, self._b_u, self._b_i, self._b_ut, self._b_it, self._P, self._Q = state
