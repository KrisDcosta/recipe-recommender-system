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
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, recommend, similar
from app.schemas import HealthResponse

logger = logging.getLogger("app")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
RECIPES_CSV = Path(os.getenv("RECIPES_CSV", "data/dataset/RAW_recipes.csv"))
USER_RATED_PATH = Path(os.getenv("USER_RATED_PATH", MODEL_DIR / "user_rated.joblib"))
TRAIN_INTERACTIONS_CSV = Path(
    os.getenv("TRAIN_INTERACTIONS_CSV", "data/dataset/interactions_train.csv")
)
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI")
RECIPES_GCS_URI = os.getenv("RECIPES_GCS_URI")


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
    for blob in client.list_blobs(bucket, prefix=prefix):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release on shutdown."""
    import joblib

    state: dict = {}
    _ensure_cloud_artifacts()

    # ── Hybrid MF (primary model) ─────────────────────────────────────────
    hybrid_path = MODEL_DIR / "hybrid_mf.joblib"
    ta_path = MODEL_DIR / "time_aware_mf.joblib"

    if hybrid_path.exists():
        logger.info("Loading HybridMF from %s", hybrid_path)
        state["model"] = joblib.load(hybrid_path)
        state["model_name"] = "hybrid_mf"
    elif ta_path.exists():
        logger.info("HybridMF not found; loading TimeAwareMF from %s", ta_path)
        state["model"] = joblib.load(ta_path)
        state["model_name"] = "time_aware_mf"
    else:
        raise RuntimeError(
            f"No model found in {MODEL_DIR}. "
            "Run: python scripts/train.py --output-dir models"
        )

    # ── Embedder ──────────────────────────────────────────────────────────
    embed_path = MODEL_DIR / "recipe_embedder.joblib"
    if embed_path.exists():
        logger.info("Loading RecipeEmbedder from %s", embed_path)
        state["embedder"] = joblib.load(embed_path)
    else:
        logger.warning("No embedder found at %s — /similar endpoint unavailable", embed_path)
        state["embedder"] = None

    # ── Recipe lookup table ───────────────────────────────────────────────
    if RECIPES_CSV.exists():
        logger.info("Loading recipe names from %s", RECIPES_CSV)
        recipes = pd.read_csv(RECIPES_CSV, usecols=["id", "name"])
        state["id_to_name"] = recipes.set_index("id")["name"].to_dict()
        state["all_recipe_ids"] = list(state["id_to_name"].keys())
    else:
        logger.warning("RAW_recipes.csv not found — recipe names unavailable")
        state["id_to_name"] = {}
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
        "Startup complete — model=%s  recipes=%d  users_with_history=%d  embedder=%s",
        state["model_name"],
        len(state["all_recipe_ids"]),
        len(state["user_rated"]),
        "loaded" if state["embedder"] else "missing",
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

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
app.include_router(similar.router, prefix="/similar", tags=["Similarity"])


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
