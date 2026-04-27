"""POST /explain — concise explanations for recipe recommendations."""
from __future__ import annotations

import json
import os

import httpx
from fastapi import APIRouter, Request

from app.routers.recommend import recommend
from app.schemas import (
    ExplainRecommendation,
    ExplainRequest,
    ExplainResponse,
    RecipeExplanation,
    RecommendRequest,
)

router = APIRouter()


def _env_truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _llm_settings() -> tuple[bool, str, str, str | None, str]:
    provider = os.getenv("LLM_PROVIDER", "xai")
    model = os.getenv("LLM_MODEL", "grok-3-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1").rstrip("/")
    key_name = "XAI_API_KEY" if provider == "xai" else "OPENAI_API_KEY"
    api_key = os.getenv(key_name)
    enabled = _env_truthy("ENABLE_LLM_EXPLANATIONS")
    return enabled, provider, model, api_key, base_url


def _recipe_context(ctx: dict, recipe_id: int) -> str:
    meta = ctx.get("id_to_meta", {}).get(recipe_id, {})
    name = meta.get("name") or ctx["id_to_name"].get(recipe_id, f"Recipe {recipe_id}")
    ingredients = str(meta.get("ingredients", "")).strip()
    minutes = meta.get("minutes")
    parts = [str(name)]
    if minutes:
        parts.append(f"{minutes} minutes")
    if ingredients:
        parts.append(f"ingredients: {ingredients[:260]}")
    return " | ".join(parts)


def _default_explanations(body: ExplainRequest, recs: list[ExplainRecommendation]) -> list[RecipeExplanation]:
    out: list[RecipeExplanation] = []
    for rec in recs:
        if body.liked_recipe_ids:
            reason = "it is close to the recipe examples you liked during onboarding"
            if rec.score is not None:
                reason += f" with a semantic match score of {rec.score:.2f}"
        elif body.user_id is not None:
            reason = "the rating model ranked it highly for this user"
            if rec.predicted_rating is not None:
                reason += f" with an estimated rating of {rec.predicted_rating:.2f}"
        else:
            reason = "it fits the current recommendation context"

        if body.disliked_recipe_ids:
            reason += " while filtering out recipes you marked as dislikes"

        out.append(
            RecipeExplanation(
                recipe_id=rec.recipe_id,
                name=rec.name,
                explanation=f"{rec.name} is recommended because {reason}.",
            )
        )
    return out


def _build_prompt(ctx: dict, body: ExplainRequest, recs: list[ExplainRecommendation]) -> str:
    liked = [_recipe_context(ctx, rid) for rid in body.liked_recipe_ids[:5]]
    disliked = [_recipe_context(ctx, rid) for rid in body.disliked_recipe_ids[:5]]
    recommendations = [
        {
            "recipe_id": rec.recipe_id,
            "name": rec.name,
            "score": rec.score,
            "predicted_rating": rec.predicted_rating,
            "context": _recipe_context(ctx, rec.recipe_id),
        }
        for rec in recs
    ]
    payload = {
        "known_user_id": body.user_id,
        "liked_examples": liked,
        "disliked_examples": disliked,
        "recommendations": recommendations,
    }
    return (
        "Explain recipe recommendations in one short user-facing sentence each. "
        "Use only the provided context. Do not mention model internals unless useful. "
        "Return strict JSON as an array of objects with recipe_id and explanation. "
        f"Context: {json.dumps(payload, ensure_ascii=True)}"
    )


def _call_llm(ctx: dict, body: ExplainRequest, recs: list[ExplainRecommendation]) -> list[RecipeExplanation] | None:
    enabled, _provider, model, api_key, base_url = _llm_settings()
    if not enabled or not api_key:
        return None

    prompt = _build_prompt(ctx, body, recs)
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You write concise, honest recommendation explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception:
        return None

    by_id = {rec.recipe_id: rec for rec in recs}
    out: list[RecipeExplanation] = []
    if isinstance(parsed, list):
        for item in parsed:
            try:
                recipe_id = int(item["recipe_id"])
                explanation = str(item["explanation"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            rec = by_id.get(recipe_id)
            if rec and explanation:
                out.append(
                    RecipeExplanation(
                        recipe_id=recipe_id,
                        name=rec.name,
                        explanation=explanation,
                    )
                )
    return out if len(out) == len(recs) else None


@router.post("", response_model=ExplainResponse)
def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    """Explain known-user or new-user recommendations.

    If LLM credentials are absent or the provider call fails, the endpoint returns
    deterministic rule-based explanations instead of failing the product flow.
    """
    ctx = request.app.state.ctx
    recs = body.recommendations
    if body.user_id is not None and not recs:
        rec_response = recommend(
            RecommendRequest(user_id=body.user_id, top_n=body.top_n, exclude_rated=True),
            request,
        )
        recs = [
            ExplainRecommendation(
                recipe_id=rec.recipe_id,
                name=rec.name,
                predicted_rating=rec.predicted_rating,
            )
            for rec in rec_response.recommendations
        ]

    enabled, provider, model, api_key, _base_url = _llm_settings()
    llm_explanations = _call_llm(ctx, body, recs)
    fallback = llm_explanations is None

    return ExplainResponse(
        explanations=llm_explanations or _default_explanations(body, recs),
        provider=provider if enabled and api_key else "rule_based",
        model=model if enabled and api_key else "rule_based",
        fallback=fallback,
    )
