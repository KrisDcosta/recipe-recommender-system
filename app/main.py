"""FastAPI application — recipe recommender inference service.

Startup loads all model artifacts once; all requests use the cached objects.
Models are stored in the `state` dict attached to `app.state`.

Environment variables
---------------------
MODEL_DIR      path to directory containing .joblib files  (default: models)
RECIPES_CSV    path to RAW_recipes.csv                     (default: data/dataset/RAW_recipes.csv)
MODEL_GCS_URI  optional gs://bucket/prefix for model artifacts
RECIPES_GCS_URI optional gs://bucket/path/RAW_recipes.csv
LOG_LEVEL      uvicorn/app log level                       (default: info)
"""
from __future__ import annotations

import logging
import math
import os
import time
from ast import literal_eval
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.routers import explain, predict, recommend, similar
from app.demo import router as demo_router
from app.schemas import HealthResponse
from src.vector_store import build_faiss_store_if_available

logger = logging.getLogger("app")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
RECIPES_CSV = Path(os.getenv("RECIPES_CSV", "data/dataset/RAW_recipes.csv"))
USER_RATED_PATH = Path(os.getenv("USER_RATED_PATH", MODEL_DIR / "user_rated.joblib"))
TRAIN_INTERACTIONS_CSV = Path(
    os.getenv("TRAIN_INTERACTIONS_CSV", "data/dataset/interactions_train.csv")
)
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI")
RECIPES_GCS_URI = os.getenv("RECIPES_GCS_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "time_aware_mf")


class LatencyStats:
    """Small in-process latency collector for demo/Cloud Run diagnostics."""

    def __init__(self) -> None:
        self._by_route: dict[str, list[float]] = {}

    def observe(self, route: str, elapsed_ms: float) -> None:
        values = self._by_route.setdefault(route, [])
        values.append(elapsed_ms)
        if len(values) > 500:
            del values[:-500]

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for route, values in self._by_route.items():
            arr = sorted(values)
            if not arr:
                continue
            p95_idx = min(len(arr) - 1, max(0, math.ceil(0.95 * len(arr)) - 1))
            out[route] = {
                "count": len(arr),
                "avg_ms": round(sum(arr) / len(arr), 2),
                "p95_ms": round(arr[p95_idx], 2),
                "max_ms": round(arr[-1], 2),
            }
        return out


class LatencyMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time-Ms headers and structured request latency logs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        route = request.scope.get("route")
        route_key = getattr(route, "path", request.url.path)
        request.app.state.latency.observe(route_key, elapsed_ms)
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, object_or_prefix) for a gs:// URI."""
    if not uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    without_scheme = uri[5:]
    bucket, _, path = without_scheme.partition("/")
    if not bucket:
        raise ValueError("GCS URI must include a bucket name")
    return bucket, path.strip("/")


def _download_gcs_prefix(uri: str, destination: Path) -> int:
    """Download all objects under a GCS prefix into destination."""
    from google.cloud import storage

    bucket_name, prefix = _parse_gcs_uri(uri)
    client = storage.Client()
    destination.mkdir(parents=True, exist_ok=True)

    n_downloaded = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = Path(blob.name[len(prefix):].lstrip("/")) if prefix else Path(blob.name)
        if str(relative) in ("", "."):
            relative = Path(blob.name).name
        if not str(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading gs://%s/%s -> %s", bucket_name, blob.name, target)
        blob.download_to_filename(target)
        n_downloaded += 1

    if n_downloaded == 0:
        raise RuntimeError(f"No GCS objects found under {uri}")
    return n_downloaded


def _download_gcs_file(uri: str, destination: Path) -> None:
    """Download one GCS object to destination."""
    from google.cloud import storage

    bucket_name, object_name = _parse_gcs_uri(uri)
    if not object_name:
        raise ValueError("GCS file URI must include an object path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading gs://%s/%s -> %s", bucket_name, object_name, destination)
    storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(destination)


def _ensure_cloud_artifacts() -> None:
    """Download Cloud Run artifacts when local mounts are absent."""
    has_local_model = (
        (MODEL_DIR / "hybrid_mf.joblib").exists()
        or (MODEL_DIR / "time_aware_mf.joblib").exists()
    )
    if MODEL_GCS_URI and not has_local_model:
        n_downloaded = _download_gcs_prefix(MODEL_GCS_URI, MODEL_DIR)
        logger.info("Downloaded %d model artifact(s) from %s", n_downloaded, MODEL_GCS_URI)

    if RECIPES_GCS_URI and not RECIPES_CSV.exists():
        _download_gcs_file(RECIPES_GCS_URI, RECIPES_CSV)


def _build_user_rated(interactions: pd.DataFrame) -> dict:
    """Build user_id -> set(recipe_id) for excluding seen recipes."""
    user_rated: dict = {}
    for user_id, group in interactions.groupby("user_id")["recipe_id"]:
        user_rated[user_id] = set(group)
    return user_rated


def _recipe_metadata(row: pd.Series) -> dict:
    """Return compact metadata used by demo filters and explanation prompts."""
    ingredients = row.get("ingredients", "")
    if isinstance(ingredients, str):
        try:
            parsed = literal_eval(ingredients)
            if isinstance(parsed, list):
                ingredients_text = ", ".join(str(x) for x in parsed)
            else:
                ingredients_text = ingredients
        except (SyntaxError, ValueError):
            ingredients_text = ingredients
    elif isinstance(ingredients, list):
        ingredients_text = ", ".join(str(x) for x in ingredients)
    else:
        ingredients_text = ""

    minutes = row.get("minutes")
    try:
        minutes_value = int(minutes) if not pd.isna(minutes) else None
    except (TypeError, ValueError):
        minutes_value = None

    name = str(row.get("name", ""))
    return {
        "name": name,
        "minutes": minutes_value,
        "ingredients": ingredients_text,
        "search_text": f"{name} {ingredients_text}".lower(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release on shutdown."""
    import joblib

    state: dict = {}
    _ensure_cloud_artifacts()

    # ── Rating model ─────────────────────────────────────────────────────
    # Time-aware MF is the verified production default. Hybrid remains opt-in
    # until its full-data metrics beat the default model.
    model_paths = {
        "time_aware_mf": MODEL_DIR / "time_aware_mf.joblib",
        "hybrid_mf": MODEL_DIR / "hybrid_mf.joblib",
    }
    if MODEL_NAME not in model_paths:
        raise RuntimeError(
            f"Unsupported MODEL_NAME={MODEL_NAME!r}; expected one of {sorted(model_paths)}"
        )

    model_path = model_paths[MODEL_NAME]
    if not model_path.exists():
        fallback_path = model_paths["time_aware_mf"]
        if MODEL_NAME != "time_aware_mf" and fallback_path.exists():
            logger.warning(
                "%s not found at %s; falling back to time_aware_mf at %s",
                MODEL_NAME,
                model_path,
                fallback_path,
            )
            model_path = fallback_path
            state["model_name"] = "time_aware_mf"
        else:
            raise RuntimeError(
                f"No model found at {model_path}. "
                "Run: python scripts/train.py --model time_aware --output-dir models"
            )
    else:
        state["model_name"] = MODEL_NAME

    logger.info("Loading %s from %s", state["model_name"], model_path)
    state["model"] = joblib.load(model_path)

    # ── Embedder ──────────────────────────────────────────────────────────
    embed_path = MODEL_DIR / "recipe_embedder.joblib"
    if embed_path.exists():
        logger.info("Loading RecipeEmbedder from %s", embed_path)
        state["embedder"] = joblib.load(embed_path)
    else:
        logger.warning("No embedder found at %s — /similar endpoint unavailable", embed_path)
        state["embedder"] = None

    # ── Vector store ──────────────────────────────────────────────────────
    # FAISS is best-effort: production uses it when faiss-cpu is installed, tests and
    # unsupported platforms fall back to RecipeEmbedder.most_similar.
    state["vector_store"] = build_faiss_store_if_available(state["embedder"])

    # ── Recipe lookup table ───────────────────────────────────────────────
    if RECIPES_CSV.exists():
        logger.info("Loading recipe names from %s", RECIPES_CSV)
        recipes = pd.read_csv(RECIPES_CSV, usecols=["id", "name", "minutes", "ingredients"])
        state["id_to_name"] = recipes.set_index("id")["name"].to_dict()
        state["id_to_meta"] = {
            int(row["id"]): _recipe_metadata(row)
            for _, row in recipes.iterrows()
        }
        state["all_recipe_ids"] = list(state["id_to_name"].keys())
    else:
        logger.warning("RAW_recipes.csv not found — recipe names unavailable")
        state["id_to_name"] = {}
        state["id_to_meta"] = {}
        state["all_recipe_ids"] = []

    # ── Rated items per user (from training set) ───────────────────────────
    # Used by /recommend to exclude already-rated recipes.
    if USER_RATED_PATH.exists():
        logger.info("Loading user-rated lookup from %s", USER_RATED_PATH)
        state["user_rated"] = joblib.load(USER_RATED_PATH)
    elif TRAIN_INTERACTIONS_CSV.exists():
        logger.info("Building user-rated lookup from %s", TRAIN_INTERACTIONS_CSV)
        train_interactions = pd.read_csv(
            TRAIN_INTERACTIONS_CSV,
            usecols=["user_id", "recipe_id"],
        )
        state["user_rated"] = _build_user_rated(train_interactions)
    else:
        logger.warning(
            "No user-rated lookup found; /recommend exclude_rated will be best-effort only"
        )
        state["user_rated"] = {}

    app.state.ctx = state
    logger.info(
        "Startup complete — model=%s  recipes=%d  users_with_history=%d  embedder=%s  vector_store=%s",
        state["model_name"],
        len(state["all_recipe_ids"]),
        len(state["user_rated"]),
        "loaded" if state["embedder"] else "missing",
        "faiss" if state["vector_store"] else "brute_force",
    )
    yield

    logger.info("Shutdown — releasing model state")
    app.state.ctx.clear()


app = FastAPI(
    title="Recipe Recommender API",
    description=(
        "Time-aware collaborative filtering + LLM embeddings for Food.com recipe recommendations. "
        "Trained on 1.1M interactions (231K recipes, 2000–2018)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.latency = LatencyStats()
app.add_middleware(LatencyMiddleware)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
app.include_router(similar.router, prefix="/similar", tags=["Similarity"])
app.include_router(explain.router, prefix="/explain", tags=["Explanation"])
app.include_router(demo_router, tags=["Demo"])


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    ctx = app.state.ctx
    loaded = [ctx["model_name"]]
    if ctx.get("embedder"):
        loaded.append("embedder")
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        n_recipes=len(ctx.get("all_recipe_ids", [])),
    )


@app.get("/", tags=["Meta"])
def root():
    return {"message": "Recipe Recommender API", "docs": "/docs"}


@app.get("/metrics", tags=["Meta"])
def metrics(request: Request):
    """Return lightweight in-process latency metrics for recent requests."""
    ctx = request.app.state.ctx
    return {
        "model": ctx["model_name"],
        "vector_store": "faiss" if ctx.get("vector_store") else "brute_force",
        "latency": request.app.state.latency.snapshot(),
    }
