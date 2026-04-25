"""POST /predict — predicted rating for a (user, recipe) pair."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("", response_model=PredictResponse)
def predict_rating(body: PredictRequest, request: Request) -> PredictResponse:
    """Predict the rating user `user_id` would give to recipe `recipe_id`.

    - Returns a float in [1.0, 5.0].
    - Unknown user or recipe falls back gracefully (global mean + biases).
    - Pass `year` or `date` for time-aware models to capture temporal drift.
    """
    ctx = request.app.state.ctx
    model = ctx["model"]

    predicted = model.predict(
        user_id=body.user_id,
        item_id=body.recipe_id,
        time_bin=body.time_bin,
    )

    return PredictResponse(
        user_id=body.user_id,
        recipe_id=body.recipe_id,
        predicted_rating=round(predicted, 4),
        model=ctx["model_name"],
    )
