"""Deterministic emotion and tone feature extraction."""

from __future__ import annotations

import re
from typing import Any

POSITIVE_WORDS = {"thanks", "thank", "glad", "happy", "great", "appreciate", "love", "helpful"}
NEGATIVE_WORDS = {"angry", "upset", "sad", "frustrated", "annoyed", "terrible", "awful"}
EMPATHY_MARKERS = ("i understand", "that sounds", "i'm sorry", "sorry", "i hear you", "that makes sense")
DEESCALATION_MARKERS = ("let's", "take a moment", "calm", "work through", "here to help")


def extract_emotion_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract tone, sentiment proxy, and empathy features."""
    text = str(row.get("text", ""))
    lowered = text.lower()
    tokens = re.findall(r"\b[\w'-]+\b", lowered)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    sentiment = 0.0
    if tokens:
        sentiment = round((positive - negative) / len(tokens), 4)

    intensity = min(
        1.0,
        round(
            (text.count("!") * 0.15)
            + (sum(character.isupper() for character in text) / max(len(text), 1))
            + (0.2 if negative > 0 else 0.0),
            4,
        ),
    )

    scores = {
        "joy": positive,
        "anger": sum(token in {"angry", "annoyed", "furious"} for token in tokens),
        "sadness": sum(token in {"sad", "upset", "hurt"} for token in tokens),
        "fear": sum(token in {"afraid", "scared", "worried"} for token in tokens),
    }
    emotion_primary = max(scores, key=scores.get) if any(scores.values()) else "neutral"

    empathy_count = sum(marker in lowered for marker in EMPATHY_MARKERS)
    deescalation_count = sum(marker in lowered for marker in DEESCALATION_MARKERS)

    return {
        "sentiment_polarity": sentiment,
        "emotion_primary": emotion_primary,
        "emotion_intensity": intensity,
        "empathy_marker_count": int(empathy_count),
        "deescalation_marker_count": int(deescalation_count),
        "warmth_score": round(min(1.0, max(0.0, 0.5 + sentiment + (empathy_count * 0.15))), 4),
    }

