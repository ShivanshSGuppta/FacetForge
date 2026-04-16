"""Structured export helpers for predictions and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.metrics import build_category_summary
from utils.constants import PREDICTIONS_DIR, REPORTS_DIR
from utils.io import dump_json, dump_jsonl, ensure_directory


def export_run_outputs(
    run_id: str,
    normalized_turns: pd.DataFrame,
    feature_frame: pd.DataFrame,
    turn_results: list[dict[str, Any]],
    facet_results: pd.DataFrame,
) -> dict[str, Path]:
    """Persist predictions, reports, and auxiliary run artifacts."""
    ensure_directory(PREDICTIONS_DIR)
    ensure_directory(REPORTS_DIR)

    turn_jsonl_path = PREDICTIONS_DIR / f"{run_id}_turn_results.jsonl"
    flat_csv_path = PREDICTIONS_DIR / f"{run_id}_facet_results.csv"
    normalized_csv_path = REPORTS_DIR / f"{run_id}_normalized_turns.csv"
    features_csv_path = REPORTS_DIR / f"{run_id}_features.csv"
    category_summary_path = REPORTS_DIR / f"{run_id}_category_summary.csv"
    run_manifest_path = REPORTS_DIR / f"{run_id}_manifest.json"

    dump_jsonl(turn_jsonl_path, turn_results)
    facet_results.to_csv(flat_csv_path, index=False)
    normalized_turns.to_csv(normalized_csv_path, index=False)
    feature_frame.to_csv(features_csv_path, index=False)
    build_category_summary(facet_results).to_csv(category_summary_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifacts": {
            "turn_results_jsonl": str(turn_jsonl_path),
            "facet_results_csv": str(flat_csv_path),
            "normalized_turns_csv": str(normalized_csv_path),
            "features_csv": str(features_csv_path),
            "category_summary_csv": str(category_summary_path),
        },
    }
    dump_json(run_manifest_path, manifest)
    return {
        "turn_results_jsonl": turn_jsonl_path,
        "facet_results_csv": flat_csv_path,
        "normalized_turns_csv": normalized_csv_path,
        "features_csv": features_csv_path,
        "category_summary_csv": category_summary_path,
        "manifest_json": run_manifest_path,
    }

