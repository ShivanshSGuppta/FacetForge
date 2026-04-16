"""Evaluation summary metrics and report helpers."""

from __future__ import annotations

import pandas as pd


def build_category_summary(facet_results: pd.DataFrame) -> pd.DataFrame:
    """Create a category-level summary table."""
    if facet_results.empty:
        return pd.DataFrame(
            columns=["category", "mean_score", "mean_confidence", "abstain_rate", "facet_count"]
        )
    grouped = facet_results.groupby("category", as_index=False).agg(
        mean_score=("score", "mean"),
        mean_confidence=("confidence", "mean"),
        abstain_rate=("abstain", "mean"),
        facet_count=("facet_id", "count"),
    )
    return grouped.round(4)

