"""Raw data loading and source schema inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from ingestion.schema import ColumnMapping
from utils.constants import COLUMN_CANDIDATES, DEFAULT_ENCODING


def _normalize_header(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum() or character == "_")


def load_input_csv(path: Path, encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    """Load a raw CSV input file into a dataframe."""
    dataframe = pd.read_csv(path, encoding=encoding)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    dataframe["_source_row_number"] = range(len(dataframe))
    return dataframe


def infer_column_mapping(
    dataframe: pd.DataFrame,
    overrides: Mapping[str, str] | None = None,
) -> ColumnMapping:
    """Infer the canonical column mapping from a dataframe header set."""
    normalized_lookup = {_normalize_header(column): column for column in dataframe.columns}
    resolved: dict[str, str | None] = {}

    for canonical, candidates in COLUMN_CANDIDATES.items():
        resolved_value = None
        if overrides and canonical in overrides:
            override = overrides[canonical]
            if override in dataframe.columns:
                resolved_value = override
        if resolved_value is None:
            for candidate in candidates:
                original = normalized_lookup.get(_normalize_header(candidate))
                if original is not None:
                    resolved_value = original
                    break
        resolved[canonical] = resolved_value

    text_column = resolved.get("text")
    if text_column is None:
        raise ValueError("Unable to infer a text column from the input CSV.")

    return ColumnMapping(**resolved)

