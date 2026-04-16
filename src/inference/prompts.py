"""Prompt construction for grouped facet judging."""

from __future__ import annotations

import json
from typing import Any

from facets.rubric import FacetBatch, RubricTemplate, ScoreScale


def build_judge_messages(
    turn_record: dict[str, Any],
    feature_record: dict[str, Any],
    facet_batch: FacetBatch,
    score_scale: ScoreScale,
    rubric_template: RubricTemplate,
) -> list[dict[str, str]]:
    """Build a concise, reproducible structured prompt for a facet batch."""
    facet_payload = [
        {
            "facet_id": facet.facet_id,
            "facet_name": facet.facet_name,
            "description": facet.description,
            "rubric": facet.rubric,
            "requires_context": facet.requires_context,
            "applicable_speakers": facet.applicable_speakers,
            "evidence_type": facet.evidence_type,
            "score_labels": facet.score_labels,
        }
        for facet in facet_batch.facets
    ]

    input_payload = {
        "turn": {
            "conversation_id": turn_record["conversation_id"],
            "turn_id": turn_record["turn_id"],
            "speaker_role": turn_record["speaker_role"],
            "text": turn_record["text"],
            "prev_turn_text": turn_record.get("prev_turn_text", ""),
            "next_turn_text": turn_record.get("next_turn_text", ""),
            "context_window": turn_record.get("context_window", ""),
        },
        "features": {
            key: value
            for key, value in feature_record.items()
            if key not in {"text", "prev_turn_text", "next_turn_text", "context_window", "raw_metadata"}
        },
        "facet_batch": {
            "batch_id": facet_batch.batch_id,
            "category": facet_batch.category,
            "facets": facet_payload,
        },
        "score_scale": score_scale.labels,
    }

    return [
        {"role": "system", "content": rubric_template.system_instruction},
        {
            "role": "user",
            "content": "\n".join(
                [
                    rubric_template.batch_instruction,
                    rubric_template.output_contract,
                    "Input payload:",
                    json.dumps(input_payload, ensure_ascii=True, indent=2),
                ]
            ),
        },
    ]

