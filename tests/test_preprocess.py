"""Tests for schema inference and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ingestion.loader import infer_column_mapping
from ingestion.preprocess import normalize_turns, preprocess_csv


def test_infer_column_mapping_with_flexible_headers() -> None:
    dataframe = pd.DataFrame(
        {
            "chat_id": ["a1"],
            "message_id": ["m1"],
            "author": ["user"],
            "content": ["Hello there"],
        }
    )
    mapping = infer_column_mapping(dataframe)
    assert mapping.conversation_id == "chat_id"
    assert mapping.turn_id == "message_id"
    assert mapping.speaker == "author"
    assert mapping.text == "content"


def test_normalize_turns_generates_context_and_ids_when_missing() -> None:
    dataframe = pd.DataFrame(
        {
            "speaker": ["user", "assistant"],
            "text": ["Can you help?", "Yes, I can help with that."],
        }
    )
    normalized = normalize_turns(dataframe)
    assert normalized.loc[0, "conversation_id"] == "conversation_0001"
    assert bool(normalized.loc[0, "is_first_turn"]) is True
    assert bool(normalized.loc[1, "is_last_turn"]) is True
    assert "CURRENT" in normalized.loc[0, "context_window"]
    assert bool(normalized.loc[0, "contains_question"]) is True


def test_preprocess_csv_exports_derived_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "conversation": ["conv1", "conv1"],
            "id": ["t1", "t2"],
            "role": ["user", "assistant"],
            "message": ["Visit https://example.com", "Sure. Here is the summary!"],
        }
    ).to_csv(csv_path, index=False)
    normalized, mapping = preprocess_csv(csv_path)
    assert mapping.text == "message"
    assert bool(normalized.loc[0, "contains_url"]) is True
    assert normalized.loc[1, "sentence_count"] >= 1
    assert normalized.loc[1, "char_count"] > 0
