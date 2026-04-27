"""POST /recommend — top-N recipe recommendations for a user."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.schemas import (
    NewUserRecommendRequest,
    NewUserRecommendResponse,
    NewUserRecipeScore,
    RecipeScore,
    RecommendRequest,
    RecommendResponse,
)

router = APIRouter()

_MAX_CANDIDATES = 50_000  # cap scan for very large recipe catalogs


@router.post("", response_model=RecommendResponse)
def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
    """Return top-N recipes predicted to be most enjoyed by `user_id`.

    - Scores all candidate recipes (excluding already-rated if `exclude_rated=true`).
    - Caps candidate scan at 50K for latency; sufficient for Food.com catalog (231K).
    - Unknown user_id falls back to item-popularity ordering via model bias.
    """
    ctx = request.app.state.ctx
    model = ctx["model"]
    id_to_name: dict = ctx["id_to_name"]
    all_ids: list = ctx["all_recipe_ids"]

    rated: set = ctx["user_rated"].get(body.user_id, set())

    # Filter candidates
    candidates = [rid for rid in all_ids if not (body.exclude_rated and rid in rated)]
    if len(candidates) > _MAX_CANDIDATES:
        rng = np.random.default_rng(body.user_id % (2**32))
        idx = rng.choice(len(candidates), size=_MAX_CANDIDATES, replace=False)
        candidates = [candidates[i] for i in idx]

    # Score all candidates
    scores = np.array([
        model.predict(user_id=body.user_id, item_id=rid)
        for rid in candidates
    ])

    top_idx = np.argsort(scores)[::-1][: body.top_n]

    recommendations = [
        RecipeScore(
            recipe_id=candidates[i],
            name=id_to_name.get(candidates[i], f"Recipe {candidates[i]}"),
            predicted_rating=round(float(scores[i]), 4),
        )
        for i in top_idx
    ]

    return RecommendResponse(
        user_id=body.user_id,
        recommendations=recommendations,
        model=ctx["model_name"],
    )


def _metadata_matches(meta: dict, include: list[str], avoid: list[str], max_minutes: int | None) -> bool:
    search_text = str(meta.get("search_text", "")).lower()
    if include and not all(term.lower() in search_text for term in include):
        return False
    if avoid and any(term.lower() in search_text for term in avoid):
        return False
    minutes = meta.get("minutes")
    if max_minutes is not None and minutes is not None and minutes > max_minutes:
        return False
    return True


def _brute_force_profile_search(
    embedder,
    query: np.ndarray,
    top_k: int,
    exclude_ids: set[int],
) -> list[tuple[int, float]]:
    mat, ids = embedder.matrix()
    query = np.asarray(query, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm <= 0:
        return []
    query = query / norm
    scores = mat @ query
    order = np.argsort(scores)[::-1]
    out: list[tuple[int, float]] = []
    for i in order:
        rid = int(ids[i])
        if rid in exclude_ids:
            continue
        out.append((rid, float(scores[i])))
        if len(out) >= top_k:
            break
    return out


def _content_match_fallback(ctx: dict, body: NewUserRecommendRequest) -> list[tuple[int, float]]:
    include = body.preferences.ingredients
    avoid = body.preferences.avoid
    max_minutes = body.preferences.max_minutes
    id_to_meta = ctx.get("id_to_meta", {})
    out: list[tuple[int, float]] = []
    exclude_ids = set(body.liked_recipe_ids) | set(body.disliked_recipe_ids)
    for rid in ctx["all_recipe_ids"]:
        if rid in exclude_ids:
            continue
        meta = id_to_meta.get(rid, {})
        if not _metadata_matches(meta, include, avoid, max_minutes):
            continue
        score = 0.5 + min(len(include), 5) * 0.05
        out.append((rid, score))
        if len(out) >= body.top_n:
            break
    return out


@router.post("/new-user", response_model=NewUserRecommendResponse)
def recommend_new_user(body: NewUserRecommendRequest, request: Request) -> NewUserRecommendResponse:
    """Cold-start recommendations from liked/disliked recipes and simple preferences.

    This uses the recipe embedding space directly rather than the collaborative model,
    so it works before a new user has historical ratings.
    """
    ctx = request.app.state.ctx
    embedder = ctx.get("embedder")
    vector_store = ctx.get("vector_store")
    id_to_name: dict = ctx["id_to_name"]
    id_to_meta: dict = ctx.get("id_to_meta", {})

    exclude_ids = set(body.liked_recipe_ids) | set(body.disliked_recipe_ids)
    include = body.preferences.ingredients
    avoid = body.preferences.avoid
    max_minutes = body.preferences.max_minutes

    neighbours: list[tuple[int, float]] = []
    search_backend = "metadata"

    if embedder is not None and body.liked_recipe_ids:
        vectors = [
            embedder.get(rid)
            for rid in body.liked_recipe_ids
            if embedder.get(rid) is not None
        ]
        if not vectors:
            raise HTTPException(
                status_code=404,
                detail="None of the liked_recipe_ids were found in the embedding index.",
            )
        query = np.mean(np.stack(vectors), axis=0)
        search_k = min(max(body.top_n * 8, 50), len(ctx["all_recipe_ids"]))
        if vector_store is not None:
            neighbours = vector_store.search_vector(query, n=search_k, exclude_ids=exclude_ids)
            search_backend = "faiss"
        else:
            neighbours = _brute_force_profile_search(embedder, query, search_k, exclude_ids)
            search_backend = "brute_force"
    else:
        neighbours = _content_match_fallback(ctx, body)

    filtered: list[NewUserRecipeScore] = []
    for rid, score in neighbours:
        meta = id_to_meta.get(rid, {})
        if not _metadata_matches(meta, include, avoid, max_minutes):
            continue
        filtered.append(
            NewUserRecipeScore(
                recipe_id=int(rid),
                name=id_to_name.get(rid, f"Recipe {rid}"),
                score=round(float(score), 4),
                source="semantic_cold_start" if search_backend != "metadata" else "metadata_cold_start",
            )
        )
        if len(filtered) >= body.top_n:
            break

    if not filtered and search_backend != "metadata":
        for rid, score in _content_match_fallback(ctx, body):
            filtered.append(
                NewUserRecipeScore(
                    recipe_id=int(rid),
                    name=id_to_name.get(rid, f"Recipe {rid}"),
                    score=round(float(score), 4),
                    source="metadata_cold_start",
                )
            )
            if len(filtered) >= body.top_n:
                break

    return NewUserRecommendResponse(
        recommendations=filtered,
        search_backend=search_backend,
    )
