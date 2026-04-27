"""Pydantic v2 request/response schemas for the recommendation API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    recipe_id: int = Field(..., description="Recipe ID")
    year: Optional[int] = Field(None, ge=2000, le=2030, description="Year of interaction (optional)")
    date: Optional[str] = Field(
        None,
        description="Interaction date as YYYY, YYYY-MM, or YYYY-MM-DD (optional)",
    )

    @field_validator("user_id", "recipe_id")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("IDs must be non-negative")
        return v

    @field_validator("date")
    @classmethod
    def date_must_start_with_year(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            year = int(v[:4])
        except (TypeError, ValueError) as exc:
            raise ValueError("date must start with a four-digit year") from exc
        if year < 2000 or year > 2030:
            raise ValueError("date year must be between 2000 and 2030")
        return v

    @model_validator(mode="after")
    def year_and_date_must_agree(self) -> "PredictRequest":
        if self.year is not None and self.date is not None and self.year != int(self.date[:4]):
            raise ValueError("year must match the year encoded in date")
        return self

    @property
    def time_bin(self) -> int | None:
        if self.year is not None:
            return self.year
        if self.date is not None:
            return int(self.date[:4])
        return None


class PredictResponse(BaseModel):
    user_id: int
    recipe_id: int
    predicted_rating: float = Field(..., ge=1.0, le=5.0)
    model: str


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for")
    top_n: int = Field(10, ge=1, le=100, description="Number of recommendations")
    exclude_rated: bool = Field(True, description="Exclude recipes already rated in training")


class RecipeScore(BaseModel):
    recipe_id: int
    name: str
    predicted_rating: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[RecipeScore]
    model: str


class NewUserPreferences(BaseModel):
    ingredients: list[str] = Field(default_factory=list, description="Ingredients or terms to prefer")
    avoid: list[str] = Field(default_factory=list, description="Ingredients or terms to avoid")
    max_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Optional maximum cook time")


class NewUserRecommendRequest(BaseModel):
    liked_recipe_ids: list[int] = Field(default_factory=list, description="Recipes the new user liked")
    disliked_recipe_ids: list[int] = Field(default_factory=list, description="Recipes the new user disliked")
    preferences: NewUserPreferences = Field(default_factory=NewUserPreferences)
    top_n: int = Field(10, ge=1, le=100, description="Number of recommendations")

    @model_validator(mode="after")
    def has_signal(self) -> "NewUserRecommendRequest":
        if not self.liked_recipe_ids and not self.preferences.ingredients:
            raise ValueError("Provide at least one liked_recipe_id or preferred ingredient")
        return self


class NewUserRecipeScore(BaseModel):
    recipe_id: int
    name: str
    score: float
    source: str


class NewUserRecommendResponse(BaseModel):
    recommendations: list[NewUserRecipeScore]
    search_backend: str
    source: str = "semantic_cold_start"


# ---------------------------------------------------------------------------
# /similar
# ---------------------------------------------------------------------------

class SimilarRequest(BaseModel):
    recipe_id: int = Field(..., description="Seed recipe ID")
    top_n: int = Field(10, ge=1, le=100, description="Number of similar recipes")


class SimilarRecipe(BaseModel):
    recipe_id: int
    name: str
    similarity: float


class SimilarResponse(BaseModel):
    seed_recipe_id: int
    search_backend: str = Field(..., description="Vector search backend: faiss or brute_force")
    similar: list[SimilarRecipe]


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------

class ExplainRecommendation(BaseModel):
    recipe_id: int
    name: str
    score: Optional[float] = None
    predicted_rating: Optional[float] = None


class ExplainRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="Known user ID for collaborative recommendations")
    liked_recipe_ids: list[int] = Field(default_factory=list)
    disliked_recipe_ids: list[int] = Field(default_factory=list)
    recommendations: list[ExplainRecommendation] = Field(default_factory=list)
    top_n: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def has_context(self) -> "ExplainRequest":
        if self.user_id is None and not self.recommendations:
            raise ValueError("Provide user_id or recommendations to explain")
        return self


class RecipeExplanation(BaseModel):
    recipe_id: int
    name: str
    explanation: str


class ExplainResponse(BaseModel):
    explanations: list[RecipeExplanation]
    provider: str
    model: str
    fallback: bool


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    n_recipes: int
