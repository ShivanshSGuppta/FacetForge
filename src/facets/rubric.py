"""Typed facet and scoring configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


CategoryName = Literal["linguistic", "pragmatics", "safety", "emotion"]
EvidenceType = Literal["text_span", "context_window", "feature_vector", "mixed"]


class ScoreScale(BaseModel):
    """Ordered score labels used across the evaluator."""

    labels: list[int] = Field(min_length=5)
    description: str = "Ordered score labels used for all facets."

    @field_validator("labels")
    @classmethod
    def labels_must_be_ordered(cls, value: list[int]) -> list[int]:
        if value != sorted(value):
            raise ValueError("Score labels must be sorted in ascending order.")
        if len(set(value)) != len(value):
            raise ValueError("Score labels must be unique.")
        return value


class FacetDefinition(BaseModel):
    """A single facet definition loaded from YAML."""

    facet_id: str
    facet_name: str
    category: CategoryName
    description: str
    score_labels: list[int]
    rubric: str
    requires_context: bool
    applicable_speakers: list[str] = Field(default_factory=lambda: ["all"])
    evidence_type: EvidenceType = "mixed"
    version: str = "1.0.0"

    @field_validator("facet_id")
    @classmethod
    def facet_id_must_use_dot_notation(cls, value: str) -> str:
        if value.count(".") < 2:
            raise ValueError("facet_id must use category.subgroup.signal format")
        return value

    @field_validator("score_labels")
    @classmethod
    def facet_score_labels_must_be_ordered(cls, value: list[int]) -> list[int]:
        if value != sorted(value):
            raise ValueError("facet score labels must be ordered")
        return value


class RubricTemplate(BaseModel):
    """Template fragments used to build judge prompts."""

    system_instruction: str
    output_contract: str
    batch_instruction: str


class FacetBatch(BaseModel):
    """A batched group of facets sent together to the judge model."""

    batch_id: str
    category: CategoryName
    facets: list[FacetDefinition]

    @property
    def facet_ids(self) -> list[str]:
        return [facet.facet_id for facet in self.facets]

