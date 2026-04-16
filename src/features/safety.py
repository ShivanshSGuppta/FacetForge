"""Deterministic safety and risk feature extraction."""

from __future__ import annotations

import re
from typing import Any

PROFANITY = {"damn", "shit", "fuck", "bitch", "asshole"}
SENSITIVE_TOPICS = {
    "self-harm",
    "suicide",
    "kill",
    "weapon",
    "drugs",
    "bomb",
    "violence",
    "fraud",
    "steal",
    "explosive",
    "hack",
}
PRIVACY_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
)


def extract_safety_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract safety-relevant heuristics from a normalized row."""
    text = str(row.get("text", ""))
    lowered = text.lower()
    tokens = re.findall(r"\b[\w'-]+\b", lowered)
    profanity_count = sum(token in PROFANITY for token in tokens)
    sensitive_matches = sum(marker in lowered for marker in SENSITIVE_TOPICS)
    privacy_risk = any(pattern.search(text) for pattern in PRIVACY_PATTERNS)
    harm_intent_score = min(1.0, round((sensitive_matches * 0.2) + (0.3 if "how to" in lowered else 0.0), 4))

    if harm_intent_score >= 0.8 or privacy_risk:
        risk_bucket = "high"
    elif harm_intent_score >= 0.4 or profanity_count >= 2:
        risk_bucket = "medium"
    else:
        risk_bucket = "low"

    return {
        "contains_profanity": profanity_count > 0,
        "contains_sensitive_topic": sensitive_matches > 0,
        "contains_harm_indicator": harm_intent_score >= 0.4,
        "policy_risk_bucket": risk_bucket,
        "profanity_count": int(profanity_count),
        "sensitive_topic_count": int(sensitive_matches),
        "harm_intent_score": harm_intent_score,
        "privacy_risk_score": 1.0 if privacy_risk else 0.0,
    }

