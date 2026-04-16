"""Facet registry loader, validator, and accessor APIs."""

from __future__ import annotations

from pathlib import Path

from facets.rubric import FacetDefinition, ScoreScale
from utils.constants import CONFIG_DIR
from utils.io import load_yaml


class FacetRegistry:
    """In-memory registry of facet definitions."""

    def __init__(self, facets: list[FacetDefinition]) -> None:
        facet_ids = [facet.facet_id for facet in facets]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("Duplicate facet_id values detected in registry.")
        self._facets = facets
        self._by_id = {facet.facet_id: facet for facet in facets}

    @classmethod
    def from_directory(cls, directory: Path | None = None) -> "FacetRegistry":
        """Load all facet YAML files from a directory."""
        directory = directory or CONFIG_DIR / "facets"
        facets: list[FacetDefinition] = []
        for path in sorted(directory.glob("*.yaml")):
            document = load_yaml(path) or {}
            for item in document.get("facets", []):
                facets.append(FacetDefinition(**item))
        return cls(facets)

    def all(self) -> list[FacetDefinition]:
        """Return all facets in registry order."""
        return list(self._facets)

    def by_category(self, category: str) -> list[FacetDefinition]:
        """Return facets belonging to a category."""
        return [facet for facet in self._facets if facet.category == category]

    def get(self, facet_id: str) -> FacetDefinition:
        """Return a facet by identifier."""
        return self._by_id[facet_id]

    def summary(self) -> dict[str, int]:
        """Return category counts for reporting."""
        summary: dict[str, int] = {}
        for facet in self._facets:
            summary[facet.category] = summary.get(facet.category, 0) + 1
        return summary


def load_score_scale(path: Path | None = None) -> ScoreScale:
    """Load the global score scale configuration."""
    path = path or CONFIG_DIR / "scoring" / "score_scale.yaml"
    document = load_yaml(path) or {}
    return ScoreScale(**document)

