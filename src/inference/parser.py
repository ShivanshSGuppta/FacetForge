"""Robust JSON parsing and validation for judge outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class JudgeFacetResult(BaseModel):
    """Validated judge output for a single facet."""

    facet_id: str
    score: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    short_rationale: str = ""
    evidence_span: str = ""
    abstain: bool = False


class JudgeBatchResult(BaseModel):
    """Validated judge output for a batch."""

    facets: list[JudgeFacetResult]


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_fences(text)
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned
    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")
    indices = [index for index in (first_brace, first_bracket) if index >= 0]
    if not indices:
        return cleaned
    return cleaned[min(indices) :]


def parse_judge_output(raw_text: str) -> JudgeBatchResult:
    """Parse and validate judge model output."""
    candidate = _extract_json_candidate(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        compact = re.sub(r"[\x00-\x1f]", "", candidate)
        payload = json.loads(compact)
        if payload is None:
            raise ValueError(f"Unable to parse judge output: {error}") from error

    if isinstance(payload, list):
        payload = {"facets": payload}
    if isinstance(payload, dict) and "results" in payload and "facets" not in payload:
        payload = {"facets": payload["results"]}

    try:
        return JudgeBatchResult(**payload)
    except ValidationError as error:
        raise ValueError(f"Judge output failed schema validation: {error}") from error

