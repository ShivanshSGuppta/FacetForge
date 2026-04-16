"""Merge heuristic priors, judge outputs, and metadata into final facet results."""

from __future__ import annotations

from typing import Any

from facets.rubric import FacetDefinition, ScoreScale
from inference.parser import JudgeFacetResult
from scoring.calibrator import calibrate_final_score, derive_heuristic_prior
from scoring.confidence import derive_confidence


def aggregate_facet_result(
    facet: FacetDefinition,
    score_scale: ScoreScale,
    feature_record: dict[str, Any],
    judge_result: JudgeFacetResult | None,
) -> dict[str, Any]:
    """Aggregate a single facet into a final reviewable payload."""
    heuristic_score, rubric_coverage = derive_heuristic_prior(facet, feature_record, score_scale)
    judge_score = None if judge_result is None or judge_result.abstain else judge_result.score
    if judge_score is not None and judge_score not in score_scale.labels:
        judge_score = min(score_scale.labels, key=lambda label: abs(label - judge_score))

    final_score = calibrate_final_score(
        scale=score_scale,
        heuristic_score=heuristic_score,
        judge_score=judge_score,
        judge_confidence=judge_result.confidence if judge_result else 0.0,
    )
    agreement = 1.0 if judge_score is None else max(
        0.0,
        1.0 - (abs(judge_score - heuristic_score) / (score_scale.labels[-1] - score_scale.labels[0])),
    )
    confidence = derive_confidence(
        model_confidence=judge_result.confidence if judge_result else 0.0,
        parse_valid=bool(judge_result and not (judge_result.abstain and judge_result.score is None)),
        rubric_coverage=rubric_coverage,
        heuristic_agreement=agreement,
        repeated_run_consistency=None,
    )

    source = "heuristic_baseline" if judge_score is None else "hybrid"
    rationale = (
        judge_result.short_rationale
        if judge_result and judge_result.short_rationale
        else "Calibrated from deterministic features because live judge output was unavailable."
    )
    evidence = judge_result.evidence_span if judge_result and judge_result.evidence_span else ""
    final_abstain = judge_score is None and rubric_coverage < 0.15

    return {
        "facet_id": facet.facet_id,
        "facet_name": facet.facet_name,
        "category": facet.category,
        "score": final_score,
        "confidence": confidence,
        "short_rationale": rationale,
        "evidence_span": evidence,
        "abstain": final_abstain,
        "judge_abstain": bool(judge_result.abstain) if judge_result else False,
        "heuristic_score": heuristic_score,
        "judge_score": judge_score,
        "rubric_version": facet.version,
        "source": source,
        "requires_context": facet.requires_context,
    }
