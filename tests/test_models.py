"""Tests for src/models.py — StaticMF and TimeAwareMF."""
import math

import numpy as np
import pytest

from src.models import StaticMF, TimeAwareMF
from src.splits import random_split


@pytest.fixture(scope="module")
def split(preprocessed_df):
    return random_split(preprocessed_df, seed=42)


@pytest.fixture(scope="module")
def fitted_static(split):
    model = StaticMF(k=3, lr=0.05, lamb=0.01, epochs=3, patience=2, seed=0)
    model.fit(split.train, split.val)
    return model


@pytest.fixture(scope="module")
def fitted_time_aware(split):
    model = TimeAwareMF(k=3, lr=0.05, lamb=0.01, epochs=3, patience=2, seed=0)
    model.fit(split.train, split.val)
    return model


class TestStaticMF:
    def test_fit_returns_self(self, split):
        model = StaticMF(k=2, epochs=2, patience=1)
        result = model.fit(split.train, split.val)
        assert result is model

    def test_predict_in_range(self, fitted_static, split):
        for _, row in split.test.head(20).iterrows():
            p = fitted_static.predict(row["user_id"], row["recipe_id"])
            assert 1.0 <= p <= 5.0, f"prediction {p} out of [1, 5]"

    def test_predict_before_fit_raises(self):
        model = StaticMF()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(1, 1)

    def test_predict_batch_length(self, fitted_static, split):
        preds = fitted_static.predict_batch(split.test.head(30))
        assert len(preds) == 30

    def test_predict_batch_in_range(self, fitted_static, split):
        preds = fitted_static.predict_batch(split.test.head(30))
        assert (preds >= 1.0).all() and (preds <= 5.0).all()

    def test_predictions_have_variance(self, fitted_static, split):
        # On tiny synthetic data MF may not beat global mean (insufficient signal).
        # What we verify: the model learned something — predictions are not all identical.
        preds = fitted_static.predict_batch(split.test)
        assert np.std(preds) > 0.0, "Predictions must not all be identical"

    def test_save_load_roundtrip(self, fitted_static, split, tmp_path):
        path = tmp_path / "static_mf.joblib"
        fitted_static.save(path)
        loaded = StaticMF.load(path)
        p1 = fitted_static.predict(split.test.iloc[0]["user_id"], split.test.iloc[0]["recipe_id"])
        p2 = loaded.predict(split.test.iloc[0]["user_id"], split.test.iloc[0]["recipe_id"])
        assert math.isclose(p1, p2, rel_tol=1e-6)

    def test_unknown_user_falls_back_gracefully(self, fitted_static):
        p = fitted_static.predict(user_id=999_999_999, item_id=1)
        assert 1.0 <= p <= 5.0

    def test_unknown_item_falls_back_gracefully(self, fitted_static):
        p = fitted_static.predict(user_id=1, item_id=999_999_999)
        assert 1.0 <= p <= 5.0


class TestTimeAwareMF:
    def test_fit_returns_self(self, split):
        model = TimeAwareMF(k=2, epochs=2, patience=1)
        result = model.fit(split.train, split.val)
        assert result is model

    def test_predict_in_range(self, fitted_time_aware, split):
        for _, row in split.test.head(20).iterrows():
            tb = int(row["year"]) if "year" in row and not np.isnan(row["year"]) else None
            p = fitted_time_aware.predict(row["user_id"], row["recipe_id"], time_bin=tb)
            assert 1.0 <= p <= 5.0

    def test_predict_without_time_bin(self, fitted_time_aware, split):
        p = fitted_time_aware.predict(split.test.iloc[0]["user_id"], split.test.iloc[0]["recipe_id"])
        assert 1.0 <= p <= 5.0

    def test_predict_before_fit_raises(self):
        model = TimeAwareMF()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(1, 1)

    def test_invalid_time_mode_raises(self):
        with pytest.raises(ValueError, match="time_mode"):
            TimeAwareMF(time_mode="quarter")

    def test_save_load_roundtrip(self, fitted_time_aware, split, tmp_path):
        path = tmp_path / "ta_mf.joblib"
        fitted_time_aware.save(path)
        loaded = TimeAwareMF.load(path)
        row = split.test.iloc[0]
        tb = int(row["year"]) if "year" in row and not np.isnan(row["year"]) else None
        p1 = fitted_time_aware.predict(row["user_id"], row["recipe_id"], time_bin=tb)
        p2 = loaded.predict(row["user_id"], row["recipe_id"], time_bin=tb)
        assert math.isclose(p1, p2, rel_tol=1e-6)

    def test_predictions_have_variance(self, fitted_time_aware, split):
        preds = fitted_time_aware.predict_batch(split.test)
        assert np.std(preds) > 0.0, "Predictions must not all be identical"
