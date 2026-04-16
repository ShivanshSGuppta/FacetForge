"""Unified feature extraction entrypoints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from features.emotion import extract_emotion_features
from features.lexical import extract_lexical_features
from features.pragmatics import extract_pragmatic_features
from features.safety import extract_safety_features


def assemble_feature_frame(normalized_turns: pd.DataFrame) -> pd.DataFrame:
    """Build a dataframe of deterministic features for each normalized turn."""
    rows: list[dict[str, Any]] = []
    for record in normalized_turns.to_dict(orient="records"):
        merged = dict(record)
        merged.update(extract_lexical_features(record))
        merged.update(extract_pragmatic_features(record))
        merged.update(extract_safety_features(record))
        merged.update(extract_emotion_features(record))
        rows.append(merged)
    return pd.DataFrame(rows)

