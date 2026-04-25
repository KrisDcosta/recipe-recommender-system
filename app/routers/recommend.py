"""POST /recommend — top-N recipe recommendations for a user."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

from app.schemas import RecipeScore, RecommendRequest, RecommendResponse

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
