"""FacetForge Streamlit review UI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.runner import run_evaluation
from utils.constants import RAW_DATA_DIR, REPORTS_DIR, SAMPLE_DATA_DIR
from utils.io import ensure_directory

st.set_page_config(page_title="FacetForge", page_icon="FF", layout="wide")


def _latest_manifest() -> Path | None:
    manifests = sorted(REPORTS_DIR.glob("*_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def _load_run_from_manifest(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = pd.read_csv(manifest["artifacts"]["normalized_turns_csv"])
    facets = pd.read_csv(manifest["artifacts"]["facet_results_csv"])
    return normalized, facets


def main() -> None:
    """Run the Streamlit application."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 204, 153, 0.28), transparent 35%),
                radial-gradient(circle at top right, rgba(109, 168, 255, 0.18), transparent 30%),
                linear-gradient(180deg, #f7f5ef 0%, #f3efe5 100%);
        }
        div[data-testid="stMetricValue"] { font-size: 1.7rem; }
        div[data-testid="stMetricLabel"] { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("FacetForge")
    st.caption("Config-driven conversation turn evaluation with deterministic features and rubric-based scoring.")

    with st.sidebar:
        st.subheader("Run Controls")
        sample_path = SAMPLE_DATA_DIR / "sample_conversations.csv"
        uploaded_file = st.file_uploader("Upload conversation CSV", type=["csv"])
        data_source = st.radio(
            "Dataset",
            options=["Bundled sample dataset", "Uploaded CSV"],
            index=0 if sample_path.exists() else 1,
        )
        run_now = st.button("Run evaluation")

    input_path: Path | None = None
    if data_source == "Bundled sample dataset" and sample_path.exists():
        input_path = sample_path
    elif uploaded_file is not None:
        ensure_directory(RAW_DATA_DIR / "uploads")
        input_path = RAW_DATA_DIR / "uploads" / uploaded_file.name
        input_path.write_bytes(uploaded_file.getvalue())

    if run_now and input_path:
        with st.spinner("Running evaluation pipeline..."):
            artifacts = run_evaluation(input_path)
        st.success(f"Run completed: {artifacts.run_id}")
    elif run_now and input_path is None:
        st.error("Upload a CSV or keep the bundled sample dataset selected before running evaluation.")

    manifest_path = _latest_manifest()
    if manifest_path is None:
        st.info("No run artifacts found yet. Upload a CSV or use the bundled sample dataset.")
        return

    normalized_df, facets_df = _load_run_from_manifest(manifest_path)
    top_left, top_middle, top_right = st.columns(3)
    top_left.metric("Turns", int(normalized_df.shape[0]))
    top_middle.metric("Facet Scores", int(facets_df.shape[0]))
    top_right.metric("Latest Run", manifest_path.stem.replace("_manifest", ""))

    st.subheader("Normalized Turns")
    st.dataframe(normalized_df, width="stretch", height=260)

    st.subheader("Category Summary")
    summary_df = facets_df.groupby("category", as_index=False).agg(
        mean_score=("score", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    left, right = st.columns(2)
    left.bar_chart(summary_df.set_index("category")["mean_score"])
    right.bar_chart(summary_df.set_index("category")["mean_confidence"])

    st.subheader("Facet Browser")
    conversations = ["All"] + sorted(facets_df["conversation_id"].astype(str).unique().tolist())
    categories = ["All"] + sorted(facets_df["category"].astype(str).unique().tolist())
    selected_conversation = st.selectbox("Conversation", conversations)
    selected_category = st.selectbox("Category", categories)
    search_text = st.text_input("Facet or rationale contains")

    filtered = facets_df.copy()
    if selected_conversation != "All":
        filtered = filtered[filtered["conversation_id"].astype(str) == selected_conversation]
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if search_text:
        lowered = search_text.lower()
        filtered = filtered[
            filtered["facet_name"].str.lower().str.contains(lowered)
            | filtered["short_rationale"].str.lower().str.contains(lowered)
        ]

    st.dataframe(
        filtered[
            [
                "conversation_id",
                "turn_id",
                "category",
                "facet_name",
                "score",
                "confidence",
                "source",
                "short_rationale",
                "evidence_span",
            ]
        ],
        width="stretch",
        height=420,
    )

    st.download_button(
        label="Download facet results CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="facetforge_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
