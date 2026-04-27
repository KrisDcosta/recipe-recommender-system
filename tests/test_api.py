"""API endpoint tests — no real models or dataset required.

Uses FastAPI TestClient with a fully mocked app.state.ctx so tests run
in CI without any .joblib files present.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# ── Stub sentence_transformers before any src import ─────────────────────────
def _install_st_stub(dim: int = 8) -> None:
    if "sentence_transformers" in sys.modules:
        return
    class _FakeST:
        def __init__(self, *a, **kw): pass
        def encode(self, texts, **kw):
            rng = np.random.default_rng(0)
            v = rng.random((len(texts), dim)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _FakeST
    sys.modules["sentence_transformers"] = stub

_install_st_stub()

# ── Imports ───────────────────────────────────────────────────────────────────
from fastapi.testclient import TestClient  # noqa: E402

from app.main import LatencyStats, _build_user_rated, _download_gcs_prefix, _parse_gcs_uri  # noqa: E402
from app.schemas import PredictRequest  # noqa: E402
from src.embeddings import RecipeEmbedder  # noqa: E402


# ── Minimal fake model ────────────────────────────────────────────────────────

class _FakeModel:
    """Predict always returns 4.0; supports predict_batch."""
    _fitted = True
    _b_u: dict = {}

    def predict(self, user_id, item_id, time_bin=None, **_):
        return 4.0

    def predict_batch(self, df, **_):
        return np.full(len(df), 4.0)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_embedder():
    import pandas as pd
    recipes = pd.DataFrame({
        "id": [101, 102, 103],
        "name": ["Pasta", "Salad", "Bread"],
        "ingredients": [["pasta"], ["lettuce"], ["flour"]],
    })
    emb = RecipeEmbedder()
    emb.fit(recipes)
    return emb


@pytest.fixture(scope="module")
def client(fake_embedder):
    """TestClient using a no-lifespan test app that shares the real routers.

    Bypasses model loading from disk entirely — no .joblib files needed.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from app.demo import router as demo_router
    from app.routers import explain, predict, recommend, similar
    from app.schemas import HealthResponse

    test_app = FastAPI()
    test_app.add_middleware(CORSMiddleware, allow_origins=["*"],
                            allow_methods=["*"], allow_headers=["*"])
    test_app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
    test_app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
    test_app.include_router(similar.router, prefix="/similar", tags=["Similarity"])
    test_app.include_router(explain.router, prefix="/explain", tags=["Explanation"])
    test_app.include_router(demo_router, tags=["Demo"])

    ctx = {
        "model": _FakeModel(),
        "model_name": "fake_model",
        "embedder": fake_embedder,
        "vector_store": None,
        "id_to_name": {101: "Pasta", 102: "Caesar Salad", 103: "Banana Bread"},
        "id_to_meta": {
            101: {
                "name": "Pasta",
                "minutes": 30,
                "ingredients": "pasta",
                "search_text": "pasta pasta",
            },
            102: {
                "name": "Caesar Salad",
                "minutes": 15,
                "ingredients": "lettuce",
                "search_text": "caesar salad lettuce",
            },
            103: {
                "name": "Banana Bread",
                "minutes": 60,
                "ingredients": "flour",
                "search_text": "banana bread flour",
            },
        },
        "all_recipe_ids": [101, 102, 103],
        "user_rated": {1: {101}},
    }
    test_app.state.ctx = ctx
    test_app.state.latency = LatencyStats()

    @test_app.get("/health", response_model=HealthResponse)
    def health():
        loaded = [ctx["model_name"]]
        if ctx.get("embedder"):
            loaded.append("embedder")
        return HealthResponse(status="ok", models_loaded=loaded,
                              n_recipes=len(ctx["all_recipe_ids"]))

    @test_app.get("/metrics")
    def metrics():
        return {
            "model": ctx["model_name"],
            "vector_store": "faiss" if ctx.get("vector_store") else "brute_force",
            "latency": test_app.state.latency.snapshot(),
        }

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_loaded_list(self, client):
        r = client.get("/health")
        loaded = r.json()["models_loaded"]
        assert "fake_model" in loaded
        assert "embedder" in loaded

    def test_n_recipes(self, client):
        r = client.get("/health")
        assert r.json()["n_recipes"] == 3


class TestMetrics:
    def test_metrics_returns_latency_snapshot(self, client):
        client.get("/health")
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == "fake_model"
        assert body["vector_store"] == "brute_force"
        assert "latency" in body


# ── /demo ─────────────────────────────────────────────────────────────────────

class TestDemo:
    def test_demo_page_serves_html(self, client):
        r = client.get("/demo")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Recipe Recommender" in r.text

    def test_demo_calls_recommend_endpoint(self, client):
        r = client.get("/demo")
        assert 'fetch("/recommend"' in r.text

    def test_demo_calls_new_user_and_explain_endpoints(self, client):
        r = client.get("/demo")
        assert 'fetch("/recommend/new-user"' in r.text
        assert 'fetch("/explain"' in r.text


# ── Startup artifact helpers ─────────────────────────────────────────────────

class TestStartupArtifactHelpers:
    def test_parse_gcs_uri_with_prefix(self):
        assert _parse_gcs_uri("gs://bucket/path/to/models") == ("bucket", "path/to/models")

    def test_parse_gcs_uri_with_bucket_only(self):
        assert _parse_gcs_uri("gs://bucket") == ("bucket", "")

    def test_parse_gcs_uri_rejects_non_gcs(self):
        with pytest.raises(ValueError):
            _parse_gcs_uri("s3://bucket/path")

    def test_download_gcs_prefix_uses_bucket_name(self, monkeypatch, tmp_path):
        calls = {}

        class _Blob:
            name = "models/time_aware_mf.joblib"

            def download_to_filename(self, target):
                calls["target"] = str(target)

        class _Client:
            def bucket(self, bucket_name):
                calls["bucket"] = bucket_name
                return object()

            def list_blobs(self, bucket_name, prefix):
                calls["list_blobs"] = (bucket_name, prefix)
                return [_Blob()]

        storage_stub = types.ModuleType("storage")
        storage_stub.Client = _Client
        google_cloud_stub = types.ModuleType("google.cloud")
        google_cloud_stub.storage = storage_stub
        monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_stub)
        monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_stub)

        assert _download_gcs_prefix("gs://bucket/models", tmp_path) == 1
        assert calls["list_blobs"] == ("bucket", "models")
        assert calls["target"].endswith("time_aware_mf.joblib")


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_200(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 101})
        assert r.status_code == 200

    def test_rating_in_range(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 101})
        rating = r.json()["predicted_rating"]
        assert 1.0 <= rating <= 5.0

    def test_response_fields(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 101})
        body = r.json()
        assert "predicted_rating" in body
        assert "model" in body
        assert body["user_id"] == 1
        assert body["recipe_id"] == 101

    def test_with_year(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 101, "year": 2015})
        assert r.status_code == 200

    def test_with_date(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 101, "date": "2015-06"})
        assert r.status_code == 200

    def test_year_date_mismatch_422(self, client):
        r = client.post(
            "/predict",
            json={"user_id": 1, "recipe_id": 101, "year": 2016, "date": "2015-06"},
        )
        assert r.status_code == 422

    def test_predict_request_derives_time_bin_from_date(self):
        body = PredictRequest(user_id=1, recipe_id=101, date="2015-06")
        assert body.time_bin == 2015

    def test_unknown_user_200(self, client):
        r = client.post("/predict", json={"user_id": 999999, "recipe_id": 101})
        assert r.status_code == 200

    def test_unknown_recipe_200(self, client):
        r = client.post("/predict", json={"user_id": 1, "recipe_id": 999999})
        assert r.status_code == 200

    def test_negative_user_id_422(self, client):
        r = client.post("/predict", json={"user_id": -1, "recipe_id": 101})
        assert r.status_code == 422

    def test_missing_user_id_422(self, client):
        r = client.post("/predict", json={"recipe_id": 101})
        assert r.status_code == 422

    def test_missing_recipe_id_422(self, client):
        r = client.post("/predict", json={"user_id": 1})
        assert r.status_code == 422


# ── /recommend ────────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_200(self, client):
        r = client.post("/recommend", json={"user_id": 1})
        assert r.status_code == 200

    def test_response_has_recommendations(self, client):
        r = client.post("/recommend", json={"user_id": 1})
        body = r.json()
        assert "recommendations" in body
        assert isinstance(body["recommendations"], list)

    def test_top_n_respected(self, client):
        r = client.post("/recommend", json={"user_id": 1, "top_n": 2})
        assert len(r.json()["recommendations"]) <= 2

    def test_exclude_rated_removes_seen(self, client):
        # user 1 has rated recipe 101 — should not appear with exclude_rated=true
        r = client.post("/recommend", json={"user_id": 1, "exclude_rated": True})
        ids = [rec["recipe_id"] for rec in r.json()["recommendations"]]
        assert 101 not in ids

    def test_include_rated_keeps_seen(self, client):
        r = client.post("/recommend", json={"user_id": 1, "exclude_rated": False})
        ids = [rec["recipe_id"] for rec in r.json()["recommendations"]]
        assert 101 in ids

    def test_recommendation_fields(self, client):
        r = client.post("/recommend", json={"user_id": 1})
        rec = r.json()["recommendations"][0]
        assert "recipe_id" in rec
        assert "name" in rec
        assert "predicted_rating" in rec

    def test_top_n_zero_422(self, client):
        r = client.post("/recommend", json={"user_id": 1, "top_n": 0})
        assert r.status_code == 422

    def test_top_n_over_limit_422(self, client):
        r = client.post("/recommend", json={"user_id": 1, "top_n": 101})
        assert r.status_code == 422

    def test_build_user_rated_groups_training_history(self):
        import pandas as pd

        interactions = pd.DataFrame({
            "user_id": [1, 1, 2],
            "recipe_id": [101, 102, 103],
        })

        assert _build_user_rated(interactions) == {1: {101, 102}, 2: {103}}


class TestNewUserRecommend:
    def test_liked_recipe_returns_cold_start_recs(self, client):
        r = client.post("/recommend/new-user", json={"liked_recipe_ids": [101], "top_n": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["search_backend"] == "brute_force"
        assert body["recommendations"]
        assert body["recommendations"][0]["source"] == "semantic_cold_start"

    def test_excludes_liked_and_disliked_recipes(self, client):
        r = client.post(
            "/recommend/new-user",
            json={"liked_recipe_ids": [101], "disliked_recipe_ids": [102], "top_n": 3},
        )
        ids = [rec["recipe_id"] for rec in r.json()["recommendations"]]
        assert 101 not in ids
        assert 102 not in ids

    def test_requires_signal(self, client):
        r = client.post("/recommend/new-user", json={"top_n": 2})
        assert r.status_code == 422

    def test_metadata_fallback_with_ingredient_preference(self, client):
        r = client.post(
            "/recommend/new-user",
            json={"preferences": {"ingredients": ["lettuce"]}, "top_n": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["search_backend"] == "metadata"
        assert [rec["recipe_id"] for rec in body["recommendations"]] == [102]

    def test_unknown_liked_recipe_404(self, client):
        r = client.post("/recommend/new-user", json={"liked_recipe_ids": [999_999]})
        assert r.status_code == 404


class TestExplain:
    def test_known_user_explain_falls_back_without_api_key(self, client, monkeypatch):
        monkeypatch.setenv("ENABLE_LLM_EXPLANATIONS", "false")
        r = client.post("/explain", json={"user_id": 1, "top_n": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["fallback"] is True
        assert body["provider"] == "rule_based"
        assert len(body["explanations"]) == 2

    def test_new_user_explain_payload(self, client, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        r = client.post(
            "/explain",
            json={
                "liked_recipe_ids": [101],
                "disliked_recipe_ids": [102],
                "recommendations": [{"recipe_id": 103, "name": "Banana Bread", "score": 0.8}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["explanations"][0]["recipe_id"] == 103
        assert "close to the recipe examples" in body["explanations"][0]["explanation"]

    def test_explain_requires_context(self, client):
        r = client.post("/explain", json={})
        assert r.status_code == 422

    def test_explain_uses_llm_when_configured(self, client, monkeypatch):
        from app.routers import explain as explain_router

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '[{"recipe_id":103,'
                                    '"explanation":"This matches your onboarding taste."}]'
                                )
                            }
                        }
                    ]
                }

        calls = {}

        def fake_post(url, headers, json, timeout):
            calls["url"] = url
            calls["model"] = json["model"]
            calls["auth"] = headers["Authorization"]
            return _FakeResponse()

        monkeypatch.setenv("ENABLE_LLM_EXPLANATIONS", "true")
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.x.ai/v1")
        monkeypatch.setattr(explain_router.httpx, "post", fake_post)

        r = client.post(
            "/explain",
            json={
                "recommendations": [{"recipe_id": 103, "name": "Banana Bread", "score": 0.8}],
            },
        )

        assert r.status_code == 200
        body = r.json()
        assert body["fallback"] is False
        assert body["model"] == "test-model"
        assert body["explanations"][0]["explanation"] == "This matches your onboarding taste."
        assert calls["url"] == "https://api.x.ai/v1/chat/completions"
        assert calls["auth"] == "Bearer test-key"


# ── /similar ──────────────────────────────────────────────────────────────────

class TestSimilar:
    def test_returns_200(self, client):
        r = client.post("/similar", json={"recipe_id": 101})
        assert r.status_code == 200

    def test_response_has_similar(self, client):
        r = client.post("/similar", json={"recipe_id": 101})
        assert "similar" in r.json()

    def test_excludes_self(self, client):
        r = client.post("/similar", json={"recipe_id": 101})
        ids = [s["recipe_id"] for s in r.json()["similar"]]
        assert 101 not in ids

    def test_top_n_respected(self, client):
        r = client.post("/similar", json={"recipe_id": 101, "top_n": 1})
        assert len(r.json()["similar"]) <= 1

    def test_similarity_field_present(self, client):
        r = client.post("/similar", json={"recipe_id": 101})
        item = r.json()["similar"][0]
        assert "similarity" in item
        assert "name" in item

    def test_search_backend_present(self, client):
        r = client.post("/similar", json={"recipe_id": 101})
        assert r.json()["search_backend"] == "brute_force"

    def test_unknown_recipe_404(self, client):
        r = client.post("/similar", json={"recipe_id": 999999})
        assert r.status_code == 404

    def test_no_embedder_503(self, client):
        # Temporarily remove embedder
        original = client.app.state.ctx["embedder"]
        client.app.state.ctx["embedder"] = None
        r = client.post("/similar", json={"recipe_id": 101})
        client.app.state.ctx["embedder"] = original
        assert r.status_code == 503
