"""Confidence computation from observable evaluator signals."""

from __future__ import annotations


def derive_confidence(
    model_confidence: float | None,
    parse_valid: bool,
    rubric_coverage: float,
    heuristic_agreement: float,
    repeated_run_consistency: float | None = None,
) -> float:
    """Compute a bounded confidence score from measurable inputs."""
    consistency = 0.5 if repeated_run_consistency is None else repeated_run_consistency
    confidence = (
        0.35 * max(0.0, min(1.0, model_confidence or 0.0))
        + 0.2 * (1.0 if parse_valid else 0.0)
        + 0.2 * max(0.0, min(1.0, rubric_coverage))
        + 0.15 * max(0.0, min(1.0, heuristic_agreement))
        + 0.1 * max(0.0, min(1.0, consistency))
    )
    return round(max(0.0, min(1.0, confidence)), 4)

