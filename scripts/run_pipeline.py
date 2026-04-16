"""CLI entrypoint for the FacetForge evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.runner import run_evaluation
from utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Run the FacetForge evaluation pipeline.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier.")
    return parser


def main() -> None:
    """Execute the CLI entrypoint."""
    configure_logging()
    args = build_parser().parse_args()
    artifacts = run_evaluation(Path(args.input), run_id=args.run_id)
    print(f"Run ID: {artifacts.run_id}")
    for name, path in artifacts.export_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

