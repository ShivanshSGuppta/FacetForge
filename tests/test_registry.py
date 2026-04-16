"""Tests for facet registry loading and batching."""

from __future__ import annotations

from pathlib import Path

import pytest

from facets.batching import build_facet_batches, load_batching_config
from facets.registry import FacetRegistry, load_score_scale
from facets.rubric import FacetDefinition, ScoreScale


def test_registry_loads_expected_counts() -> None:
    registry = FacetRegistry.from_directory()
    summary = registry.summary()
    assert summary["linguistic"] == 90
    assert summary["pragmatics"] == 90
    assert summary["safety"] == 60
    assert summary["emotion"] == 60
    assert len(registry.all()) == 300


def test_batches_respect_size_limits() -> None:
    registry = FacetRegistry.from_directory()
    config = load_batching_config()
    batches = build_facet_batches(
        registry.all(),
        min_batch_size=config["min_batch_size"],
        max_batch_size=config["max_batch_size"],
    )
    assert batches
    assert all(len(batch.facets) <= config["max_batch_size"] for batch in batches)


def test_duplicate_facet_ids_are_rejected() -> None:
    facet = FacetDefinition(
        facet_id="linguistic.grammar.test_signal",
        facet_name="Test Signal",
        category="linguistic",
        description="A test facet.",
        score_labels=[10, 25, 50, 75, 90],
        rubric="A rubric.",
        requires_context=False,
        applicable_speakers=["assistant"],
        evidence_type="text_span",
        version="1.0.0",
    )
    with pytest.raises(ValueError):
        FacetRegistry([facet, facet])


def test_score_scale_rejects_unsorted_values() -> None:
    with pytest.raises(ValueError):
        ScoreScale(labels=[10, 50, 25, 75, 90])

