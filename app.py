#!/usr/bin/env python3
"""Streamlit sandbox for the Redrob ranker.

Robustly accepts .jsonl, .json, and .jsonl.gz candidate uploads, normalizes
them into strict JSONL, and then calls the core ranking pipeline in offline
mode using local model artifacts only.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Hard offline mode for sandbox reproducibility
# ---------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_JD_INDEX = APP_DIR / "jd_hybrid_index.json"
DEFAULT_MODELS_DIR = APP_DIR / "models"
DEFAULT_CACHE_DIR = APP_DIR / "cache"

try:
    from rank_candidates import run
except Exception as exc:  # pragma: no cover
    run = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

st.set_page_config(
    page_title="Redrob Ranker Sandbox | Team: Real_RR",
    page_icon="🚀",
    layout="wide",
)


def _safe_query_params() -> dict[str, str]:
    try:
        qp = dict(st.query_params)
        return {str(k): str(v) for k, v in qp.items()}
    except Exception:
        return {}


# Optional health check for hosted sandboxes
if _safe_query_params().get("ping") == "true":
    st.write("App is awake and healthy.")
    st.stop()


def _resolve_local_model(model_name: str) -> str:
    """Prefer local artifacts only. Never fall back to network downloads."""
    safe = model_name.replace("/", "__")
    local_dir = DEFAULT_MODELS_DIR / safe
    if local_dir.exists():
        return str(local_dir)
    raise FileNotFoundError(
        f"Local model artifact not found: {local_dir}. "
        f"Please include the offline model folder in ./models."
    )


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

    def _as_records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            records = [obj for obj in payload if isinstance(obj, dict)]
            if len(records) != len(payload):
                raise ValueError("JSON array must contain only candidate objects.")
            return records

        if isinstance(payload, dict):
            if isinstance(payload.get("candidates"), list):
                records = [obj for obj in payload["candidates"] if isinstance(obj, dict)]
                if len(records) != len(payload["candidates"]):
                    raise ValueError("'candidates' must contain only candidate objects.")
                return records
            if payload.get("candidate_id"):
                return [payload]

        raise ValueError(
            'JSON upload must be a candidate object, list of candidates, or {"candidates": [...]} structure.'
        )

    # First try full JSON (pretty-printed JSON is common in uploads)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    records: list[dict[str, Any]]
    if payload is not None:
        records = _as_records(payload)
    else:
        # Fallback to strict JSONL parsing.
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

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except Exception as exc:
        st.error(f"sentence-transformers is not available: {exc}")
        return False

    try:
        # Only verify local artifacts; do not allow network fallback.
        with st.spinner("Initializing offline models from local artifacts..."):
            SentenceTransformer(_resolve_local_model("BAAI/bge-small-en-v1.5"))
            CrossEncoder(_resolve_local_model("cross-encoder/ms-marco-MiniLM-L-6-v2"))
        return True
    except Exception as exc:
        st.error(
            "Could not load local models. Make sure the model artifacts exist under ./models."
        )
        st.exception(exc)
        return False


def _run_ranker(input_candidates: Path, output_csv: Path) -> int:
    """Call the core ranking pipeline using the same argument names it expects."""
    args = argparse.Namespace(
        jd_index=str(DEFAULT_JD_INDEX),
        candidates=str(input_candidates),
        output=str(output_csv),
        metadata_output="",
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
        jd_embeddings_cache=str(DEFAULT_CACHE_DIR / "jd_vector_embeddings.npz"),
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


def _read_csv_bytes_as_df(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(csv_bytes))


def main() -> None:
    st.title("🚀 Redrob Hackathon: Stage 1 Sandbox")
    st.markdown(
        """
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
        with tempfile.TemporaryDirectory(prefix="redrob_output_") as out_dir_str:
            out_dir = Path(out_dir_str)
            output_csv = out_dir / "Real_RR.csv"

            with st.spinner("Executing BM25 → Bi-Encoder → Cross-Encoder..."):
                try:
                    exit_code = _run_ranker(temp_input_path, output_csv)
                    if exit_code in (0, None) and output_csv.exists():
                        st.success("✅ Pipeline executed successfully.")

                        csv_bytes = output_csv.read_bytes()
                        df = _read_csv_bytes_as_df(csv_bytes)

                        st.markdown("### 3. Review & Download Results")
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            label="⬇️ Download Real_RR.csv",
                            data=csv_bytes,
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