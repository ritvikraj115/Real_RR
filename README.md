# Redrob Hackathon Ranker

This repository contains our submission for the **Redrob Intelligent Candidate Discovery Challenge**.

The project implements a production-inspired, multi-stage candidate ranking pipeline that combines structured eligibility filtering, hybrid retrieval, neural reranking, and recruiter-oriented behavioral signals to produce an explainable ranking of candidates for a given job description.

The complete ranking pipeline is:

**L0 Triage → Behavior Validation Score (BVS) → BM25 Retrieval → Dense Vector Retrieval → Cross-Encoder Reranking → Coverage & Evidence Scoring → Final Score Fusion**

---

# Repository Structure

```text
.
├── models/                         # Cached SentenceTransformer and Cross-Encoder models
├── cache/                          # Local embedding and scoring caches
├── app.py                          # Optional Streamlit demo
├── jd_hybrid_index.json            # Preprocessed hybrid JD index
├── rank_candidates.py              # Main ranking pipeline
├── validate_submission.py          # Submission validator
├── submission_metadata.yaml        # Challenge metadata
├── requirements.txt                # Python dependencies
├── README.md
├── ranking_pipeline_README.md      # Detailed pipeline documentation
└── final_all_scores_features.csv   # Optional analysis output
```

---

# Environment

The pipeline was developed and tested using the following environment.

| Component        | Version         |
| ---------------- | --------------- |
| Python           | **3.12.x**      |
| Execution        | CPU Only        |
| Operating System | Windows / Linux |
| GPU              | Not Required    |

> **Note**
>
> This project was validated using **Python 3.12**. Some versions of `torch`, `sentence-transformers`, and related dependencies may behave differently on older Python versions. For reproducible results, we recommend running the project with **Python 3.12**.

---

# Installation

We recommend using a clean virtual environment.

```bash
# Create virtual environment (Python 3.12)
python3.12 -m venv .venv

# Activate

# Linux / macOS
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

# Running the Pipeline

Before running the pipeline, place the candidate dataset (for example `candidates.jsonl`) in the repository root. The dataset is **not included** in this repository because of its size.

Run the ranker:

```bash
python rank_candidates.py --candidates candidates.jsonl --jd-index jd_hybrid_index.json --output Real_RR.csv
```

Validate the submission:

```bash
python validate_submission.py Real_RR.csv
```

---

# Pipeline Overview

1. Load the preprocessed JD hybrid index.
2. Read candidate profiles.
3. Apply conservative L0 structured triage.
4. Compute Behavior Validation Score (BVS).
5. Build enriched candidate retrieval documents.
6. Perform BM25 lexical retrieval.
7. Perform dense semantic retrieval using Sentence Transformers.
8. Apply family-aware candidate recall.
9. Rerank the strongest evidence with a Cross-Encoder.
10. Combine semantic relevance, recruiter behavior, coverage, evidence density, and calibrated negative confidence into the final ranking score.
11. Generate the final submission CSV.

---

# Notes

* Candidate data is intentionally excluded from this repository due to file size limitations.
* The first execution may take longer because embedding models and local caches are created.
* Subsequent executions reuse cached models and embeddings for faster runtime.
* The pipeline is designed for CPU execution and produces deterministic, explainable rankings suitable for hackathon evaluation.
