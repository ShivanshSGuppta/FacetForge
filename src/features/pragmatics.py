"""Deterministic pragmatic feature extraction."""

from __future__ import annotations

import re
from typing import Any

QUESTION_STARTERS = ("what", "why", "how", "when", "where", "who", "can", "could", "would", "should")
REQUEST_MARKERS = ("please", "can you", "could you", "would you", "help me", "need you to")
INSTRUCTION_MARKERS = ("step", "first", "then", "next", "must", "should", "do this", "follow")
REFUSAL_MARKERS = ("cannot", "can't", "won't", "unable", "not able", "i must refuse", "i can't help with")
APOLOGY_MARKERS = ("sorry", "apologize", "apologies")
THANKS_MARKERS = ("thanks", "thank you", "appreciate")
COMMITMENT_MARKERS = ("i will", "i'll", "going to", "let me", "i can")
HEDGING_MARKERS = ("maybe", "perhaps", "might", "could be", "possibly", "seems")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _jaccard_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\b[\w']+\b", left.lower()))
    right_tokens = set(re.findall(r"\b[\w']+\b", right.lower()))
    if not left_tokens and not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return round(1.0 - (overlap / union), 4)


def extract_pragmatic_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract pragmatic and discourse features from a normalized row."""
    text = str(row.get("text", ""))
    lowered = text.lower()
    prev_text = str(row.get("prev_turn_text", ""))
    prev_question = "?" in prev_text
    direct_answer_score = 0.25
    if prev_question and text:
        if lowered.startswith(("yes", "no", "because", "the", "it", "you", "we", "this")):
            direct_answer_score = 0.8
        elif len(text.split()) >= 4:
            direct_answer_score = 0.6
    elif text:
        direct_answer_score = 0.5

    request_count = sum(marker in lowered for marker in REQUEST_MARKERS)
    instruction_count = sum(marker in lowered for marker in INSTRUCTION_MARKERS)
    hedging_count = sum(marker in lowered for marker in HEDGING_MARKERS)

    return {
        "contains_question": bool(row.get("contains_question", False) or lowered.startswith(QUESTION_STARTERS)),
        "contains_request": _contains_any(text, REQUEST_MARKERS),
        "contains_instruction": _contains_any(text, INSTRUCTION_MARKERS),
        "contains_refusal": _contains_any(text, REFUSAL_MARKERS),
        "contains_apology": _contains_any(text, APOLOGY_MARKERS),
        "contains_thanks": _contains_any(text, THANKS_MARKERS),
        "contains_commitment": _contains_any(text, COMMITMENT_MARKERS),
        "direct_answer_score": round(direct_answer_score, 4),
        "topic_shift_score": _jaccard_distance(text, prev_text) if prev_text else 0.0,
        "request_marker_count": int(request_count),
        "instruction_marker_count": int(instruction_count),
        "hedging_phrase_count": int(hedging_count),
        "grounding_marker_count": int(sum(token in lowered for token in ("this", "that", "above", "earlier", "here"))),
    }

