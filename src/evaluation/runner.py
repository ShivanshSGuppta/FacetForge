"""End-to-end pipeline runner."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from evaluation.export import export_run_outputs
from features import assemble_feature_frame
from facets.batching import build_facet_batches, load_batching_config
from facets.registry import FacetRegistry, load_score_scale
from facets.rubric import RubricTemplate
from inference.llm_client import LLMClient, load_inference_settings
from scoring.aggregator import aggregate_facet_result
from scoring.judge import RubricJudge
from utils.constants import CONFIG_DIR
from utils.io import load_yaml
from utils.logging import get_logger
from ingestion.preprocess import export_processed_dataframe, preprocess_csv

LOGGER = get_logger(__name__)


@dataclass
class RunArtifacts:
    """Structured pipeline outputs returned to callers."""

    run_id: str
    normalized_turns: pd.DataFrame
    feature_frame: pd.DataFrame
    facet_results: pd.DataFrame
    turn_results: list[dict[str, Any]]
    export_paths: dict[str, Path]
    run_metadata: dict[str, Any]


def _load_rubric_template() -> RubricTemplate:
    payload = load_yaml(CONFIG_DIR / "scoring" / "rubric_templates.yaml") or {}
    return RubricTemplate(**payload)


def run_evaluation(
    input_csv: Path,
    run_id: str | None = None,
    resume: bool = False,
) -> RunArtifacts:
    """Execute the full evaluation pipeline from CSV to exported artifacts."""
    del resume
    run_id = run_id or f"run_{uuid4().hex[:10]}"
    LOGGER.info("Starting run %s for %s", run_id, input_csv)
    started_at = time.perf_counter()

    normalized_turns, mapping = preprocess_csv(input_csv)
    export_processed_dataframe(normalized_turns, stem=run_id)
    LOGGER.info("Normalized %s turns", len(normalized_turns))

    feature_frame = assemble_feature_frame(normalized_turns)
    LOGGER.info("Extracted deterministic features for %s turns", len(feature_frame))

    registry = FacetRegistry.from_directory()
    score_scale = load_score_scale()
    batching_config = load_batching_config()
    facet_batches = build_facet_batches(
        registry.all(),
        min_batch_size=batching_config["min_batch_size"],
        max_batch_size=batching_config["max_batch_size"],
    )
    LOGGER.info("Loaded %s facets across %s batches", len(registry.all()), len(facet_batches))

    client = LLMClient(load_inference_settings())
    judge = RubricJudge(client=client, template=_load_rubric_template())

    flat_facet_results: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []

    feature_lookup = {
        (row["conversation_id"], row["turn_id"]): row
        for row in feature_frame.to_dict(orient="records")
    }

    for turn_record in normalized_turns.to_dict(orient="records"):
        key = (turn_record["conversation_id"], turn_record["turn_id"])
        feature_record = feature_lookup[key]
        turn_facet_results: list[dict[str, Any]] = []

        for batch in facet_batches:
            judge_output = judge.score_batch(turn_record, feature_record, batch, score_scale)
            judge_lookup = {item.facet_id: item for item in judge_output.facets}
            for facet in batch.facets:
                final_result = aggregate_facet_result(
                    facet=facet,
                    score_scale=score_scale,
                    feature_record=feature_record,
                    judge_result=judge_lookup.get(facet.facet_id),
                )
                final_result.update(
                    {
                        "conversation_id": turn_record["conversation_id"],
                        "turn_id": turn_record["turn_id"],
                        "turn_index": turn_record["turn_index"],
                        "speaker_role": turn_record["speaker_role"],
                    }
                )
                flat_facet_results.append(final_result)
                turn_facet_results.append(final_result)

        scores_by_category: dict[str, list[int]] = defaultdict(list)
        for item in turn_facet_results:
            scores_by_category[item["category"]].append(item["score"])
        category_summary = {
            category: round(sum(values) / len(values), 4)
            for category, values in scores_by_category.items()
        }

        turn_results.append(
            {
                "conversation_id": turn_record["conversation_id"],
                "turn_id": turn_record["turn_id"],
                "turn_index": turn_record["turn_index"],
                "speaker_role": turn_record["speaker_role"],
                "text": turn_record["text"],
                "category_summary": category_summary,
                "facet_results": turn_facet_results,
                "processing_metadata": {
                    "model_provider": client.settings.provider,
                    "model_name": client.settings.model_name,
                    "mapping": mapping.model_dump(),
                },
                "human_review_needed": any(
                    item["confidence"] < 0.45 or item["abstain"] for item in turn_facet_results
                ),
            }
        )

    facet_results = pd.DataFrame(flat_facet_results)
    export_paths = export_run_outputs(
        run_id=run_id,
        normalized_turns=normalized_turns,
        feature_frame=feature_frame,
        turn_results=turn_results,
        facet_results=facet_results,
    )
    elapsed = round(time.perf_counter() - started_at, 4)
    run_metadata = {
        "run_id": run_id,
        "input_csv": str(input_csv),
        "turn_count": len(normalized_turns),
        "facet_count": len(registry.all()),
        "batch_count": len(facet_batches),
        "elapsed_seconds": elapsed,
        "model_provider": client.settings.provider,
        "model_name": client.settings.model_name,
    }
    LOGGER.info("Completed run %s in %s seconds", run_id, elapsed)

    return RunArtifacts(
        run_id=run_id,
        normalized_turns=normalized_turns,
        feature_frame=feature_frame,
        facet_results=facet_results,
        turn_results=turn_results,
        export_paths=export_paths,
        run_metadata=run_metadata,
    )

