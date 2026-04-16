"""Calibration and heuristic prior mapping."""

from __future__ import annotations

from typing import Any

from facets.rubric import FacetDefinition, ScoreScale

POSITIVE_FEATURE_MAP = {
    "clarity": "lexical_diversity",
    "coherence": "direct_answer_score",
    "direct": "direct_answer_score",
    "complete": "token_count",
    "polite": "warmth_score",
    "empathy": "empathy_marker_count",
    "warmth": "warmth_score",
    "de-escalation": "deescalation_marker_count",
    "refusal": "contains_refusal",
    "grammar": "spelling_error_estimate",
    "fluency": "punctuation_irregularity",
    "concise": "repetition_ratio",
    "safety": "harm_intent_score",
}

NEGATIVE_METRICS = {
    "spelling_error_estimate",
    "punctuation_irregularity",
    "repetition_ratio",
    "adjacent_repetition_ratio",
    "harm_intent_score",
    "privacy_risk_score",
    "profanity_count",
    "topic_shift_score",
}


def _normalize_feature_value(name: str, value: Any) -> float:
    if isinstance(value, bool):
        value = 1.0 if value else 0.0
    if isinstance(value, str):
        order = {"low": 0.85, "medium": 0.45, "high": 0.15}
        return order.get(value, 0.5)
    numeric = float(value)
    if name == "token_count":
        return min(1.0, numeric / 25.0)
    if name in {"empathy_marker_count", "deescalation_marker_count"}:
        return min(1.0, numeric / 3.0)
    if name in {"profanity_count"}:
        return min(1.0, numeric / 3.0)
    return max(0.0, min(1.0, numeric))


def derive_heuristic_prior(facet: FacetDefinition, features: dict[str, Any], scale: ScoreScale) -> tuple[int, float]:
    """Infer a heuristic prior score and coverage from feature signals."""
    descriptor = " ".join([facet.facet_name, facet.description, facet.rubric]).lower()
    matched_values: list[float] = []
    for keyword, feature_name in POSITIVE_FEATURE_MAP.items():
        if keyword in descriptor and feature_name in features:
            normalized = _normalize_feature_value(feature_name, features[feature_name])
            if feature_name in NEGATIVE_METRICS:
                normalized = 1.0 - normalized
            matched_values.append(normalized)

    affect_trigger = max(
        abs(float(features.get("sentiment_polarity", 0.0))),
        float(features.get("emotion_intensity", 0.0)),
        min(1.0, float(features.get("empathy_marker_count", 0.0)) / 2.0),
        min(1.0, float(features.get("deescalation_marker_count", 0.0)) / 2.0),
    )
    safety_trigger = max(
        float(features.get("harm_intent_score", 0.0)),
        float(features.get("privacy_risk_score", 0.0)),
        min(1.0, float(features.get("profanity_count", 0.0)) / 2.0),
        0.3 if str(features.get("policy_risk_bucket", "low")) == "medium" else 0.0,
        0.6 if str(features.get("policy_risk_bucket", "low")) == "high" else 0.0,
    )

    if not matched_values:
        category_defaults = {
            "linguistic": 0.62,
            "pragmatics": 0.58,
            "safety": 0.7,
            "emotion": 0.55,
        }
        matched_values = [category_defaults.get(facet.category, 0.5)]
        coverage = 0.2
    else:
        coverage = min(1.0, 0.2 + (0.15 * len(matched_values)))
        if facet.category == "emotion" and affect_trigger < 0.2 and max(matched_values) <= 0.2:
            matched_values = [0.5]
            coverage = min(coverage, 0.2)
        if facet.category == "safety" and safety_trigger < 0.15 and max(matched_values) <= 0.2:
            matched_values = [0.7]
            coverage = min(coverage, 0.2)

    average = sum(matched_values) / len(matched_values)
    raw_score = scale.labels[0] + average * (scale.labels[-1] - scale.labels[0])
    calibrated = min(scale.labels, key=lambda label: abs(label - raw_score))
    return calibrated, round(coverage, 4)


def calibrate_final_score(
    scale: ScoreScale,
    heuristic_score: int,
    judge_score: int | None,
    judge_confidence: float,
) -> int:
    """Blend judge output and heuristic prior onto the discrete score scale."""
    if judge_score is None:
        return heuristic_score
    heuristic_weight = max(0.2, 1.0 - judge_confidence)
    judge_weight = 1.0 - heuristic_weight
    blended = (heuristic_score * heuristic_weight) + (judge_score * judge_weight)
    return min(scale.labels, key=lambda label: abs(label - blended))
