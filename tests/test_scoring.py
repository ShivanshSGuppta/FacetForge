"""Tests for feature extraction, parser handling, and aggregation."""

from __future__ import annotations

from facets.registry import FacetRegistry, load_score_scale
from features import assemble_feature_frame
from inference.parser import JudgeFacetResult, parse_judge_output
from scoring.aggregator import aggregate_facet_result
import pandas as pd


def test_feature_assembler_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn_id": "t1",
                "turn_index": 0,
                "speaker_role": "assistant",
                "text": "I'm sorry you're frustrated. Let's work through it.",
                "prev_turn_text": "I am upset about this.",
                "next_turn_text": "",
                "context_window": "CURRENT | assistant: I'm sorry you're frustrated. Let's work through it.",
                "language_code": "en",
                "char_count": 47,
                "token_count": 8,
                "sentence_count": 2,
                "avg_word_length": 4.0,
                "uppercase_ratio": 0.0,
                "punctuation_ratio": 0.05,
                "repetition_ratio": 0.0,
                "spelling_error_estimate": 0.0,
                "contains_question": False,
                "contains_url": False,
                "preprocessing_notes": "",
                "timestamp_order_valid": True,
                "raw_metadata": {},
            }
        ]
    )
    features = assemble_feature_frame(frame)
    assert "empathy_marker_count" in features.columns
    assert "direct_answer_score" in features.columns
    assert "policy_risk_bucket" in features.columns


def test_parse_judge_output_accepts_json_fences() -> None:
    parsed = parse_judge_output(
        """```json
        {"facets":[{"facet_id":"linguistic.grammar.subject_verb_agreement","score":75,"confidence":0.8,"short_rationale":"Clean grammar.","evidence_span":"Clean grammar.","abstain":false}]}
        ```"""
    )
    assert parsed.facets[0].score == 75


def test_aggregate_result_uses_judge_and_heuristics() -> None:
    registry = FacetRegistry.from_directory()
    scale = load_score_scale()
    facet = registry.get("emotion.empathy.emotion_recognition")
    feature_record = {
        "empathy_marker_count": 2,
        "warmth_score": 0.8,
        "direct_answer_score": 0.7,
        "repetition_ratio": 0.0,
    }
    judge_result = JudgeFacetResult(
        facet_id=facet.facet_id,
        score=75,
        confidence=0.9,
        short_rationale="Strong empathic acknowledgment.",
        evidence_span="I'm sorry you're dealing with that.",
        abstain=False,
    )
    aggregated = aggregate_facet_result(facet, scale, feature_record, judge_result)
    assert aggregated["score"] in scale.labels
    assert aggregated["source"] == "hybrid"
    assert aggregated["confidence"] > 0.5

