# Repository Structure

```text
.
├── models/                         # Cached HuggingFace models
├── app.py                          # Streamlit demo (optional)
├── jd_hybrid_index.json            # Preprocessed job description index
├── rank_candidates.py              # Main ranking pipeline
├── validate_submission.py          # Submission validator
├── submission_metadata.yaml
├── requirements.txt
├── README.md
├── ranking_pipeline_README.md
└── final_all_scores_features.csv   # Analysis output (generated/optional)
```

> **Note:** The candidate dataset (`candidates.jsonl`) is **not included in this repository** because of its large size. Before running the pipeline, place the provided `candidates.jsonl` file in the project root (same directory as `rank_candidates.py`).

---

# Installation

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

---

# Required Files

Before running the ranker, your project directory should contain:

```text
.
├── candidates.jsonl          ← Add this file before running
├── jd_hybrid_index.json
├── rank_candidates.py
├── requirements.txt
└── ...
```

---

# Running the Ranking Pipeline

Run the main ranking pipeline:

```bash
python rank_candidates.py \
    --candidates candidates.jsonl \
    --jd-index jd_hybrid_index.json \
    --output Real_RR.csv
```

The pipeline will:

1. Load the candidate dataset.
2. Load the preprocessed JD index.
3. Apply structured L0 candidate triage.
4. Compute Behavior Validation Score (BVS).
5. Build the engineered candidate document.
6. Perform BM25 lexical retrieval.
7. Perform Bi-Encoder semantic retrieval.
8. Select family-aware evidence for reranking.
9. Run Cross-Encoder verification.
10. Fuse semantic, behavioral, structured, and evidence-based scores.
11. Generate the final ranked submission.

---

# Validate Submission

After ranking completes, validate the generated CSV.

```bash
python validate_submission.py Real_RR.csv
```

---

# Output

The pipeline generates:

```text
Real_RR.csv
```

which contains the final ranked candidate list in the required submission format.

---

# Notes

- `jd_hybrid_index.json` is already preprocessed and ready to use.
- `candidates.jsonl` must be provided separately before execution.
- The pipeline is designed for CPU-only execution and uses local caching to improve runtime.
- HuggingFace models will be downloaded automatically on the first run and reused from the `models/` directory on subsequent runs.