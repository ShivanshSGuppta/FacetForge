"""Deterministic lexical and text-quality features."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(r"\b[\w']+\b")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def extract_lexical_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract explainable text-quality features from a normalized row."""
    text = str(row.get("text", ""))
    tokens = _tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    counts = Counter(tokens)
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    adjacent_repetitions = sum(
        1 for left, right in zip(tokens, tokens[1:]) if left == right
    )
    punctuation_irregularity = (
        0.5 * (1.0 if re.search(r"[!?.,]{4,}", text) else 0.0)
        + 0.5 * min(float(row.get("punctuation_ratio", 0.0)) * 2.0, 1.0)
    )
    rare_shape_ratio = (
        sum(1 for token in tokens if re.search(r"[a-zA-Z]\d|\d[a-zA-Z]|(.)\1{2,}", token)) / len(tokens)
        if tokens
        else 0.0
    )
    repetition_ratio = float(row.get("repetition_ratio", 0.0))

    return {
        "char_count": int(row.get("char_count", len(text))),
        "token_count": int(row.get("token_count", len(tokens))),
        "sentence_count": int(row.get("sentence_count", 0)),
        "avg_word_length": float(row.get("avg_word_length", 0.0)),
        "uppercase_ratio": float(row.get("uppercase_ratio", 0.0)),
        "punctuation_ratio": float(row.get("punctuation_ratio", 0.0)),
        "repetition_ratio": repetition_ratio,
        "spelling_error_estimate": float(row.get("spelling_error_estimate", 0.0)),
        "lexical_diversity": round(unique_ratio, 4),
        "adjacent_repetition_ratio": round(adjacent_repetitions / max(len(tokens), 1), 4),
        "punctuation_irregularity": round(min(punctuation_irregularity, 1.0), 4),
        "rare_token_shape_ratio": round(rare_shape_ratio, 4),
        "long_token_ratio": round(
            sum(1 for token in alpha_tokens if len(token) >= 10) / max(len(alpha_tokens), 1), 4
        ),
        "redundancy_pressure": round(
            sum(count - 1 for count in counts.values() if count > 1) / max(len(tokens), 1), 4
        ),
    }

