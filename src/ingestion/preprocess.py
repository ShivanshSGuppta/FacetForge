"""Conversation turn normalization and derived column generation."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.loader import infer_column_mapping, load_input_csv
from ingestion.schema import ColumnMapping, NormalizedTurn, PreprocessOptions
from utils.constants import DEFAULT_CONTEXT_RADIUS, PROCESSED_DATA_DIR, SPEAKER_ALIASES
from utils.io import ensure_directory

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\b[\w']+\b")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
EMOTION_WORDS = {
    "sorry",
    "thanks",
    "thank",
    "understand",
    "frustrated",
    "happy",
    "sad",
    "angry",
    "glad",
}


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_speaker(value: Any) -> str:
    normalized = _safe_text(value).lower()
    return SPEAKER_ALIASES.get(normalized, normalized or "unknown")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _language_code(text: str) -> str:
    if not text.strip():
        return "unknown"
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    alpha_letters = sum(character.isalpha() for character in text)
    if alpha_letters == 0:
        return "unknown"
    ratio = ascii_letters / alpha_letters
    return "en" if ratio >= 0.8 else "mixed"


def _spelling_error_estimate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    anomalies = 0
    for token in tokens:
        if len(token) > 18:
            anomalies += 1
        elif re.search(r"(.)\1{3,}", token):
            anomalies += 1
        elif sum(character.isdigit() for character in token) >= max(2, len(token) // 2):
            anomalies += 1
    return round(anomalies / len(tokens), 4)


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return round(repeated / len(tokens), 4)


def _sort_turns(dataframe: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    ordered = dataframe.copy()
    if mapping.turn_index and mapping.turn_index in ordered.columns:
        ordered["_sort_turn_index"] = pd.to_numeric(ordered[mapping.turn_index], errors="coerce")
    else:
        ordered["_sort_turn_index"] = pd.NA

    if mapping.timestamp and mapping.timestamp in ordered.columns:
        ordered["_sort_timestamp"] = pd.to_datetime(ordered[mapping.timestamp], errors="coerce")
    else:
        ordered["_sort_timestamp"] = pd.NaT

    ordered["_sort_conversation"] = (
        ordered[mapping.conversation_id].astype(str)
        if mapping.conversation_id
        else "conversation_0001"
    )

    return ordered.sort_values(
        by=["_sort_conversation", "_sort_turn_index", "_sort_timestamp", "_source_row_number"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _build_context_window(group: pd.DataFrame, row_index: int, radius: int) -> str:
    start = max(0, row_index - radius)
    end = min(len(group), row_index + radius + 1)
    segments: list[str] = []
    for local_index in range(start, end):
        prefix = "CURRENT" if local_index == row_index else f"NEIGHBOR_{local_index - row_index:+d}"
        speaker = group.iloc[local_index]["speaker_role"]
        text = group.iloc[local_index]["text"]
        segments.append(f"{prefix} | {speaker}: {text}")
    return "\n".join(segments)


def normalize_turns(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping | None = None,
    options: PreprocessOptions | None = None,
) -> pd.DataFrame:
    """Normalize a raw dataframe into evaluation-ready turn records."""
    dataframe = dataframe.copy()
    if "_source_row_number" not in dataframe.columns:
        dataframe["_source_row_number"] = range(len(dataframe))
    mapping = mapping or infer_column_mapping(dataframe)
    options = options or PreprocessOptions(context_radius=DEFAULT_CONTEXT_RADIUS)
    ordered = _sort_turns(dataframe, mapping)

    if mapping.conversation_id is None:
        ordered["__conversation_id"] = "conversation_0001"
        conversation_column = "__conversation_id"
    else:
        conversation_column = mapping.conversation_id

    if mapping.turn_id is None:
        ordered["__turn_id"] = [f"turn_{index + 1:04d}" for index in range(len(ordered))]
        turn_id_column = "__turn_id"
    else:
        turn_id_column = mapping.turn_id

    if mapping.speaker is None:
        ordered["__speaker"] = "unknown"
        speaker_column = "__speaker"
    else:
        speaker_column = mapping.speaker

    normalized_rows: list[dict[str, Any]] = []

    for conversation_id, conversation_group in ordered.groupby(conversation_column, sort=False):
        conversation_group = conversation_group.reset_index(drop=True)
        timestamp_order_valid = True
        if mapping.timestamp and mapping.timestamp in conversation_group.columns:
            timestamp_series = pd.to_datetime(conversation_group[mapping.timestamp], errors="coerce")
            valid_timestamps = timestamp_series.dropna()
            timestamp_order_valid = valid_timestamps.is_monotonic_increasing if not valid_timestamps.empty else True

        for local_index, row in conversation_group.iterrows():
            text = _safe_text(row[mapping.text])
            tokens = _tokenize(text)
            sentence_count = max(1, len(SENTENCE_PATTERN.findall(text))) if text else 0
            words = [token for token in tokens if token.isalpha()]
            notes: list[str] = []
            if not text:
                notes.append("blank_text")
            if mapping.conversation_id is None:
                notes.append("generated_conversation_id")
            if mapping.turn_id is None:
                notes.append("generated_turn_id")
            if mapping.speaker is None:
                notes.append("generated_speaker")
            if URL_PATTERN.search(text):
                notes.append("url_present")
            if any(word in EMOTION_WORDS for word in words):
                notes.append("emotion_lexicon_detected")

            payload = {
                "conversation_id": _safe_text(conversation_id) or "conversation_0001",
                "turn_id": _safe_text(row[turn_id_column]) or f"turn_{local_index + 1:04d}",
                "turn_index": int(local_index),
                "speaker_role": _canonical_speaker(row[speaker_column]),
                "text": text,
                "parent_turn_id": _safe_text(row[mapping.parent_turn_id]) if mapping.parent_turn_id else None,
                "is_first_turn": local_index == 0,
                "is_last_turn": local_index == len(conversation_group) - 1,
                "prev_turn_text": "" if local_index == 0 else _safe_text(conversation_group.iloc[local_index - 1][mapping.text]),
                "next_turn_text": "" if local_index == len(conversation_group) - 1 else _safe_text(conversation_group.iloc[local_index + 1][mapping.text]),
                "context_window": _build_context_window(conversation_group.assign(speaker_role=conversation_group[speaker_column].map(_canonical_speaker), text=conversation_group[mapping.text].map(_safe_text)), local_index, options.context_radius),
                "language_code": _language_code(text),
                "char_count": len(text),
                "token_count": len(tokens),
                "sentence_count": sentence_count,
                "avg_word_length": round(sum(len(word) for word in words) / len(words), 4) if words else 0.0,
                "uppercase_ratio": round(sum(character.isupper() for character in text) / max(len(text), 1), 4),
                "punctuation_ratio": round(sum(not character.isalnum() and not character.isspace() for character in text) / max(len(text), 1), 4),
                "repetition_ratio": _repetition_ratio(tokens),
                "spelling_error_estimate": _spelling_error_estimate(tokens),
                "contains_question": "?" in text,
                "contains_url": bool(URL_PATTERN.search(text)),
                "preprocessing_notes": ",".join(notes),
                "timestamp_order_valid": timestamp_order_valid,
                "raw_metadata": {"source_row_number": int(row["_source_row_number"])},
            }
            normalized_rows.append(NormalizedTurn(**payload).model_dump())

    return pd.DataFrame(normalized_rows)


def preprocess_csv(
    csv_path: Path,
    mapping_overrides: dict[str, str] | None = None,
    options: PreprocessOptions | None = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load, infer schema, normalize, and return processed turns."""
    raw_dataframe = load_input_csv(csv_path)
    mapping = infer_column_mapping(raw_dataframe, overrides=mapping_overrides)
    normalized = normalize_turns(raw_dataframe, mapping=mapping, options=options)
    return normalized, mapping


def export_processed_dataframe(dataframe: pd.DataFrame, stem: str) -> dict[str, Path]:
    """Persist the normalized dataframe to CSV and JSONL artifacts."""
    ensure_directory(PROCESSED_DATA_DIR)
    csv_path = PROCESSED_DATA_DIR / f"{stem}.csv"
    jsonl_path = PROCESSED_DATA_DIR / f"{stem}.jsonl"
    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(jsonl_path, orient="records", lines=True, force_ascii=True)
    return {"csv": csv_path, "jsonl": jsonl_path}
