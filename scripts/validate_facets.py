"""Validate the facet registry and print a summary."""

from __future__ import annotations

from facets.batching import build_facet_batches, load_batching_config
from facets.registry import FacetRegistry


def main() -> None:
    """Run registry validation and emit a short summary."""
    registry = FacetRegistry.from_directory()
    config = load_batching_config()
    batches = build_facet_batches(
        registry.all(),
        min_batch_size=config["min_batch_size"],
        max_batch_size=config["max_batch_size"],
    )
    print("Facet counts by category:")
    for category, count in registry.summary().items():
        print(f"  {category}: {count}")
    print(f"Total facets: {len(registry.all())}")
    print(f"Total batches: {len(batches)}")


if __name__ == "__main__":
    main()
