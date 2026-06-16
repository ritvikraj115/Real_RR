#!/usr/bin/env python3
"""Streamlit sandbox for the Redrob ranker.

Robustly accepts .jsonl, .json, and .jsonl.gz candidate uploads, normalizes
them into strict JSONL, and then calls the core ranking pipeline in offline
mode using local model artifacts when available.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from rank_candidates import run
except Exception as exc:  # pragma: no cover
    run = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_JD_INDEX = APP_DIR / "jd_hybrid_index.json"
DEFAULT_MODELS_DIR = APP_DIR / "models"


st.set_page_config(
    page_title="Redrob Ranker Sandbox | Team: Real_RR",
    page_icon="🚀",
    layout="wide",
)


def _resolve_local_model(model_name: str) -> str:
    """Prefer local artifacts if present; fall back to the HF id only if needed."""
    safe = model_name.replace("/", "__")
    local_dir = DEFAULT_MODELS_DIR / safe
    if local_dir.exists():
        return str(local_dir)
    # Keep the original model id as fallback for environments that still have internet.
    return model_name


def _read_uploaded_bytes(uploaded_file) -> bytes:
    data = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".gz") or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def _normalize_candidates_to_jsonl(uploaded_file) -> str:
    """Return strict JSONL text regardless of whether the upload is jsonl/json/gz."""
    raw_bytes = _read_uploaded_bytes(uploaded_file)
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded file is not valid UTF-8 text.") from exc

    stripped = text.strip()
    if not stripped:
        raise ValueError("Uploaded file is empty.")

    # First try full JSON.
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    records: list[dict[str, Any]] = []

    if isinstance(payload, list):
        records = [obj for obj in payload if isinstance(obj, dict)]
        if len(records) != len(payload):
            raise ValueError("JSON array must contain only candidate objects.")
    elif isinstance(payload, dict):
        if isinstance(payload.get("candidates"), list):
            records = [obj for obj in payload["candidates"] if isinstance(obj, dict)]
            if len(records) != len(payload["candidates"]):
                raise ValueError("'candidates' must contain only candidate objects.")
        elif payload.get("candidate_id"):
            records = [payload]
        else:
            raise ValueError("JSON upload must be a candidate object, list of candidates, or {\"candidates\": [...]} structure.")
    else:
        # Fallback to strict JSONL.
        records = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at uploaded file line {line_no}: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL at line {line_no}: each line must be a JSON object.")
            records.append(obj)

    if not records:
        raise ValueError("No candidate records were found in the uploaded file.")

    # Normalize to one compact JSON object per line.
    return "\n".join(json.dumps(obj, ensure_ascii=False) for obj in records) + "\n"


def _write_temp_jsonl(uploaded_file) -> Path:
    normalized = _normalize_candidates_to_jsonl(uploaded_file)
    tmp_dir = Path(tempfile.mkdtemp(prefix="redrob_sandbox_"))
    temp_path = tmp_dir / "sample_candidates.jsonl"
    temp_path.write_text(normalized, encoding="utf-8")
    return temp_path


@st.cache_resource(show_spinner=False)
def load_models_into_memory() -> bool:
    """Load local models once; never require network in the sandbox."""
    if run is None:
        st.error(f"Could not import rank_candidates.py: {IMPORT_ERROR}")
        return False

    # If the core ranker already handles local model paths, this warm-up simply
    # confirms the artifacts exist.
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except Exception as exc:
        st.error(f"sentence-transformers is not available: {exc}")
        return False

    try:
        with st.spinner("Initializing offline models from local artifacts..."):
            SentenceTransformer(_resolve_local_model("BAAI/bge-small-en-v1.5"))
            CrossEncoder(_resolve_local_model("cross-encoder/ms-marco-MiniLM-L-6-v2"))
        return True
    except Exception as exc:
        st.error(
            "Could not load local models. "
            "Make sure the model artifacts exist under ./models "
            "or provide a network-enabled dev environment."
        )
        st.exception(exc)
        return False


def _run_ranker(input_candidates: Path, output_csv: Path) -> int:
    """Call the core ranking pipeline using the same argument names it expects."""
    args = argparse.Namespace(
        jd_index=str(DEFAULT_JD_INDEX),
        candidates=str(input_candidates),
        output=str(output_csv),
        metadata_output=str(output_csv.with_suffix(".metadata.json")),
        l0_report_output="",
        stage1_report_output="",
        stage2_report_output="",
        scores_output="",
        chunk_scores_output="",
        chunk_scores_limit=1000,
        pre_shortlist_size=800,
        shortlist_size=120,
        top_k=100,
        allow_fewer_than_top_k=True,
        max_candidates=None,
        as_of_date="2026-06-09",
        vector_backend="sentence-transformers",
        embedding_model="BAAI/bge-small-en-v1.5",
        model_cache_dir=str(DEFAULT_MODELS_DIR),
        jd_embeddings_cache=str(APP_DIR / "cache" / "jd_vector_embeddings.npz"),
        vector_scores_cache=str(output_csv.with_suffix(".vector_scores.npz")),
        cross_encoder_backend="sentence-transformers",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_scores_cache=str(output_csv.with_suffix(".cross_scores.npz")),
        require_cross_encoder=True,
        batch_size=16,
        prepare_cache_only=False,
        validate_only=False,
    )
    return int(run(args) or 0)


def main() -> None:
    st.title("🚀 Redrob Hackathon: Stage 1 Sandbox")
    st.markdown(
        f"""
**Team ID:** `Real_RR`  
This sandbox runs the ranking pipeline end-to-end in offline mode.

Supported uploads:
- `.jsonl`
- `.json`
- `.jsonl.gz`

The file is normalized to strict JSONL before ranking, so pretty-printed JSON or
gzipped input will not trigger a JSONL parse error.
"""
    )

    if not DEFAULT_JD_INDEX.exists():
        st.error(f"Missing JD index: {DEFAULT_JD_INDEX}")
        st.stop()

    if not load_models_into_memory():
        st.stop()

    st.markdown("### 1. Upload Candidates")
    uploaded_file = st.file_uploader(
        "Upload sample candidates",
        type=["jsonl", "json", "gz"],
        accept_multiple_files=False,
        help="You can upload JSONL, JSON, or gzipped JSONL.",
    )

    if uploaded_file is None:
        st.info("Upload a sample candidate file to run the pipeline.")
        return

    try:
        temp_input_path = _write_temp_jsonl(uploaded_file)
    except Exception as exc:
        st.error("The uploaded file could not be converted into strict JSONL.")
        st.exception(exc)
        return

    st.success(f"Uploaded and normalized to JSONL: {uploaded_file.name}")

    st.markdown("### 2. Run Pipeline")
    if st.button("▶️ Execute Ranking Engine", type="primary"):
        tmp_out_dir = Path(tempfile.mkdtemp(prefix="redrob_output_"))
        output_csv = tmp_out_dir / "Real_RR.csv"

        with st.spinner("Executing BM25 → Bi-Encoder → Cross-Encoder..."):
            try:
                exit_code = _run_ranker(temp_input_path, output_csv)
                if exit_code in (0, None) and output_csv.exists():
                    st.success("✅ Pipeline executed successfully.")
                    df = pd.read_csv(output_csv)

                    st.markdown("### 3. Review & Download Results")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    with open(output_csv, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Real_RR.csv",
                            data=f,
                            file_name="Real_RR.csv",
                            mime="text/csv",
                        )
                else:
                    st.error(f"Pipeline failed (exit code: {exit_code}).")
            except Exception as exc:
                st.error("Execution failed.")
                st.exception(exc)
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
