"""POST /similar — find semantically similar recipes via embedding cosine similarity."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas import SimilarRecipe, SimilarRequest, SimilarResponse

router = APIRouter()


@router.post("", response_model=SimilarResponse)
def similar_recipes(body: SimilarRequest, request: Request) -> SimilarResponse:
    """Return top-N recipes most similar to `recipe_id` by embedding cosine similarity.

    Requires the embedder to be loaded at startup.
    Returns 503 if embedder is unavailable (run scripts/embed_recipes.py first).
    """
    ctx = request.app.state.ctx
    embedder = ctx.get("embedder")
    id_to_name: dict = ctx["id_to_name"]

    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Embedder not loaded. Run: python scripts/embed_recipes.py",
        )

    if embedder.get(body.recipe_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"recipe_id {body.recipe_id} not found in embedding index.",
        )

    neighbours = embedder.most_similar(body.recipe_id, n=body.top_n)

    return SimilarResponse(
        seed_recipe_id=body.recipe_id,
        similar=[
            SimilarRecipe(
                recipe_id=rid,
                name=id_to_name.get(rid, f"Recipe {rid}"),
                similarity=round(score, 4),
            )
            for rid, score in neighbours
        ],
    )
