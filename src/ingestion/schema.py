"""Typed schemas for ingestion and normalization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ColumnMapping(BaseModel):
    """Resolved source-to-canonical column mapping."""

    conversation_id: str | None = None
    turn_id: str | None = None
    speaker: str | None = None
    text: str
    turn_index: str | None = None
    timestamp: str | None = None
    parent_turn_id: str | None = None


class PreprocessOptions(BaseModel):
    """Runtime preprocessing options."""

    context_radius: int = Field(default=1, ge=0, le=5)
    export_processed: bool = True


class NormalizedTurn(BaseModel):
    """Structured normalized turn used by the rest of the pipeline."""

    conversation_id: str
    turn_id: str
    turn_index: int
    speaker_role: str
    text: str
    parent_turn_id: str | None = None
    is_first_turn: bool
    is_last_turn: bool
    prev_turn_text: str = ""
    next_turn_text: str = ""
    context_window: str = ""
    language_code: str = "unknown"
    char_count: int
    token_count: int
    sentence_count: int
    avg_word_length: float
    uppercase_ratio: float
    punctuation_ratio: float
    repetition_ratio: float
    spelling_error_estimate: float
    contains_question: bool
    contains_url: bool
    preprocessing_notes: str = ""
    timestamp_order_valid: bool = True
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_id", "turn_id", "speaker_role")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """Reject blank required string fields."""
        if not value:
            raise ValueError("value must not be blank")
        return value
