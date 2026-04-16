"""FacetForge FastAPI service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from evaluation.runner import run_evaluation
from utils.constants import PREDICTIONS_DIR, REPORTS_DIR, RAW_DATA_DIR
from utils.io import ensure_directory, read_json
from utils.logging import configure_logging

configure_logging()

app = FastAPI(title="FacetForge API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health probe for container orchestration."""
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload a CSV and trigger an evaluation run."""
    ensure_directory(RAW_DATA_DIR / "uploads")
    run_id = f"api_{uuid4().hex[:8]}"
    csv_path = RAW_DATA_DIR / "uploads" / f"{run_id}_{file.filename}"
    contents = await file.read()
    csv_path.write_bytes(contents)
    artifacts = run_evaluation(csv_path, run_id=run_id)
    return {
        "run_id": artifacts.run_id,
        "run_metadata": artifacts.run_metadata,
        "artifacts": {key: str(value) for key, value in artifacts.export_paths.items()},
    }


@app.get("/results/{run_id}")
def get_results(run_id: str) -> dict[str, object]:
    """Return lightweight metadata and summaries for a prior run."""
    manifest_path = REPORTS_DIR / f"{run_id}_manifest.json"
    summary_path = REPORTS_DIR / f"{run_id}_category_summary.csv"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    summary_records = pd.read_csv(summary_path).to_dict(orient="records") if summary_path.exists() else []
    return {
        "run_id": run_id,
        "manifest": read_json(manifest_path),
        "category_summary": summary_records,
    }

