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
    similar: list[SimilarRecipe]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    n_recipes: int
