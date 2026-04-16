"""Grouped facet judging orchestration."""

from __future__ import annotations

from typing import Any

from facets.rubric import FacetBatch, RubricTemplate, ScoreScale
from inference.llm_client import LLMClient
from inference.parser import JudgeBatchResult, JudgeFacetResult, parse_judge_output
from inference.prompts import build_judge_messages


def _fallback_results(facet_batch: FacetBatch) -> JudgeBatchResult:
    return JudgeBatchResult(
        facets=[
            JudgeFacetResult(
                facet_id=facet.facet_id,
                score=None,
                confidence=0.0,
                short_rationale="Live judge skipped because no inference endpoint is configured.",
                evidence_span="",
                abstain=True,
            )
            for facet in facet_batch.facets
        ]
    )


class RubricJudge:
    """Judge orchestration wrapper around the inference client."""

    def __init__(self, client: LLMClient, template: RubricTemplate) -> None:
        self.client = client
        self.template = template

    def score_batch(
        self,
        turn_record: dict[str, Any],
        feature_record: dict[str, Any],
        facet_batch: FacetBatch,
        score_scale: ScoreScale,
    ) -> JudgeBatchResult:
        """Score a batch of facets for a single turn."""
        if not self.client.is_configured:
            return _fallback_results(facet_batch)

        messages = build_judge_messages(
            turn_record=turn_record,
            feature_record=feature_record,
            facet_batch=facet_batch,
            score_scale=score_scale,
            rubric_template=self.template,
        )
        response = self.client.generate(messages)
        return parse_judge_output(response.text)

