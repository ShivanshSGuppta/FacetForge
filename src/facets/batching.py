"""Facet batching logic for grouped rubric judging."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from facets.rubric import FacetBatch, FacetDefinition
from utils.constants import CONFIG_DIR, DEFAULT_BATCH_MAX_SIZE, DEFAULT_BATCH_MIN_SIZE
from utils.io import load_yaml


def _facet_subgroup_key(facet: FacetDefinition) -> str:
    parts = facet.facet_id.split(".")
    return ".".join(parts[:2])


def load_batching_config(path: Path | None = None) -> dict[str, int]:
    """Load batching thresholds from YAML."""
    path = path or CONFIG_DIR / "scoring" / "batching.yaml"
    document = load_yaml(path) or {}
    return {
        "min_batch_size": int(document.get("min_batch_size", DEFAULT_BATCH_MIN_SIZE)),
        "max_batch_size": int(document.get("max_batch_size", DEFAULT_BATCH_MAX_SIZE)),
    }


def build_facet_batches(
    facets: list[FacetDefinition],
    min_batch_size: int = DEFAULT_BATCH_MIN_SIZE,
    max_batch_size: int = DEFAULT_BATCH_MAX_SIZE,
) -> list[FacetBatch]:
    """Build grouped facet batches with stable ordering."""
    grouped: dict[str, list[FacetDefinition]] = defaultdict(list)
    for facet in sorted(facets, key=lambda item: item.facet_id):
        grouped[_facet_subgroup_key(facet)].append(facet)

    batches: list[FacetBatch] = []
    by_category: dict[str, list[tuple[str, list[FacetDefinition]]]] = defaultdict(list)
    for subgroup, subgroup_facets in grouped.items():
        by_category[subgroup_facets[0].category].append((subgroup, subgroup_facets))

    for category, subgroup_items in by_category.items():
        working: list[FacetDefinition] = []
        batch_number = 1
        for subgroup, subgroup_facets in subgroup_items:
            if working and len(working) + len(subgroup_facets) > max_batch_size:
                batch_id = f"{category}.batch_{batch_number:02d}"
                batches.append(FacetBatch(batch_id=batch_id, category=category, facets=working))
                batch_number += 1
                working = []
            working.extend(subgroup_facets)
            if len(working) >= min_batch_size:
                batch_id = f"{category}.batch_{batch_number:02d}"
                batches.append(FacetBatch(batch_id=batch_id, category=category, facets=working))
                batch_number += 1
                working = []
        if working:
            batch_id = f"{category}.batch_{batch_number:02d}"
            batches.append(FacetBatch(batch_id=batch_id, category=category, facets=working))
    return batches

