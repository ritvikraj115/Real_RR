# Redrob Hackathon Ranker

This repository contains our submission for the **Redrob Intelligent Candidate Discovery Challenge**.

The project implements a **production-inspired, multi-stage hybrid candidate ranking pipeline** that progressively narrows the candidate search space while increasing the sophistication of evaluation at each stage. It combines structured recruiter signals, hybrid information retrieval, semantic reranking, and explainable score fusion to rank candidates for a given job description under the hackathon's CPU-only runtime constraints.

Rather than relying on a single ranking signal, the system combines **structured eligibility**, **lexical retrieval**, **semantic retrieval**, **pairwise neural verification**, and **behavior-aware scoring** into one interpretable ranking framework.

---

# Repository Structure

```text
.
├── models/                         # Cached SentenceTransformer and Cross-Encoder models
├── cache/                          # Local embedding and scoring caches
├── app.py                          # Optional Streamlit demonstration
├── jd_hybrid_index.json            # Preprocessed hybrid JD index
├── rank_candidates.py              # Main ranking pipeline
├── validate_submission.py          # Submission validator
├── submission_metadata.yaml        # Challenge metadata
├── requirements.txt                # Python dependencies
├── README.md
├── ranking_pipeline_README.md      # Detailed pipeline explanation
└── final_all_scores_features.csv   # Optional diagnostic output
```

---

# Environment

The pipeline was developed and validated using the following environment.

| Component        | Version         |
| ---------------- | --------------- |
| Python           | **3.12.x**      |
| Execution        | CPU Only        |
| Operating System | Windows / Linux |
| GPU              | Not Required    |

> **Important**
>
> The project was developed and tested using **Python 3.12**. Some combinations of **PyTorch**, **Transformers**, and **Sentence-Transformers** may exhibit compatibility issues on older Python versions. For reproducible results, we recommend running the project with **Python 3.12**.

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

The candidate dataset is intentionally **not included** in this repository because of its size.

Before execution, place the dataset (for example `candidates.jsonl`) in the repository root.

Run the ranking pipeline:

```bash
python rank_candidates.py \
    --candidates candidates.jsonl \
    --jd-index jd_hybrid_index.json \
    --output Real_RR.csv
```

Validate the generated submission:

```bash
python validate_submission.py Real_RR.csv
```

---

# Pipeline Overview

The ranking pipeline progressively reduces the candidate search space while applying increasingly sophisticated evaluation methods. Earlier stages efficiently remove obvious mismatches, whereas later stages perform computationally expensive semantic verification only on the strongest candidates.

```text
L0 Structured Triage
        │
        ▼
Behavior Validation Score (BVS)
        │
        ▼
Candidate Enrichment
        │
        ▼
Hybrid Retrieval
 ├── BM25 Lexical Retrieval
 └── Dense Vector Retrieval (Bi-Encoder)
        │
        ▼
Family-aware Candidate Recall
        │
        ▼
Adaptive Cross-Encoder Chunk Selection
        │
        ▼
Cross-Encoder Pairwise Reranking
        │
        ▼
Quality-based Coverage
        │
        ▼
Evidence Density
        │
        ▼
Negative Confidence
        │
        ▼
Final Score Fusion
        │
        ▼
Ranked Candidate List
```

---

# Methodology

## Stage 1 — L0 Structured Triage

The pipeline begins with a conservative structured filtering stage based entirely on explicit candidate metadata.

Only candidates with clear eligibility mismatches are removed using structured recruiter signals such as:

* notice period
* recruiter activity
* relocation preference
* location compatibility
* profile completeness
* structured hiring attributes

No semantic matching is performed at this stage, allowing the pipeline to significantly reduce the search space while minimizing false negatives.

---

## Stage 2 — Behavior Validation Score (BVS)

Each remaining candidate receives a **Behavior Validation Score (BVS)** that measures recruiter-oriented profile quality independently of semantic relevance.

The score combines structured hiring signals including:

* recruiter responsiveness
* assessment performance
* interview completion
* notice period
* experience alignment
* profile quality
* verification status
* market engagement
* recruiter activity

Rather than assigning fixed scores, BVS is calibrated using percentile normalization and smooth score shaping to preserve meaningful discrimination while preventing score saturation.

---

## Stage 3 — Candidate Enrichment

Before retrieval, candidate narratives are enriched using lightweight taxonomy expansion.

Relevant technologies, production terminology, deployment concepts, retrieval keywords, ranking terminology, and domain-specific concepts are appended as hidden retrieval tags.

This improves retrieval recall without modifying the original candidate information or requiring additional embedding models.

---

## Stage 4 — Hybrid Retrieval

The pipeline combines sparse lexical retrieval and dense semantic retrieval.

### BM25 Retrieval

BM25 retrieves candidates through exact lexical matching between the job description and candidate profiles.

This stage captures:

* exact technologies
* recruiter terminology
* product keywords
* implementation-specific skills
* domain vocabulary

---

### Dense Vector Retrieval (Bi-Encoder)

Candidate narratives and JD chunks are encoded using a Sentence Transformer Bi-Encoder.

Semantic similarity is computed using cosine similarity:

$$
\operatorname{CosSim}(x,y)=\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}
$$

allowing semantically similar concepts to match even when different terminology is used.

---

## Stage 5 — Family-aware Candidate Recall

Rather than selecting only globally highest-scoring retrieval results, candidate recall is diversified across multiple conceptual JD families:

* Retrieval
* Evaluation
* Systems
* Product
* Domain
* Culture
* Advanced Skills

This prevents a single topic from dominating retrieval while encouraging broad semantic coverage across the complete job description.

---

## Stage 6 — Adaptive Cross-Encoder Reranking

The strongest candidate evidence is first selected through adaptive chunk selection.

Instead of evaluating every available candidate passage, the pipeline retains only:

* up to **6 high-quality positive evidence chunks**
* up to **2 informative negative evidence chunks**

This substantially reduces Cross-Encoder computation while preserving semantic recall.

Each selected candidate chunk is then jointly encoded with its corresponding JD chunk using a Cross-Encoder, producing significantly more accurate pairwise relevance estimates than independent embeddings.

When the Cross-Encoder returns logits, confidence calibration is applied before conversion to probabilities:

$$
p = \sigma\!\left(\frac{z-b}{T}\right)
$$

where

* (z) is the raw Cross-Encoder logit,
* (b) is the median logit used for bias correction,
* (T) is the calibration temperature,
* (\sigma) denotes the sigmoid function.

Only a single sigmoid transformation is applied.

The strongest semantic evidence is then aggregated using a consistency-aware formulation:

$$
CE = 0.7\times\max(\text{Top3}) + 0.3\times\operatorname{mean}(\text{Top3})
$$

which rewards candidates demonstrating consistently strong evidence rather than relying on a single exceptional passage.

---

## Stage 7 — Coverage Quality

Coverage measures how completely a candidate satisfies the conceptual areas represented within the job description.

Rather than counting matched families, the pipeline evaluates the strongest semantic evidence within each family and computes a weighted quality score:

$$
Coverage = \frac{\sum_i w_i\cdot Best_i}{\sum_i w_i}
$$

where

* (Best_i) is the strongest semantic similarity observed for family (i),
* (w_i) is the corresponding JD importance weight.

This rewards both semantic quality and conceptual breadth while avoiding duplicated evidence.

---

## Stage 8 — Evidence Density

Evidence Density measures the consistency of supporting semantic evidence throughout the candidate profile.

Instead of rewarding numerous weak matches, only the strongest semantic evidence contributes:

$$
Evidence = 0.7\times\max(\text{Top3}) + 0.3\times\operatorname{mean}(\text{Top3})
$$

encouraging candidates who consistently demonstrate strong relevance across multiple supporting passages.

---

## Stage 9 — Negative Confidence

Potential recruiter risks—including wrapper-only experience, research-only backgrounds, or weak production evidence—are summarized into a calibrated Negative Confidence score.

Rather than applying hard penalties, multiple negative signals are aggregated into a confidence estimate, reducing false penalties from isolated keywords while maintaining explainability.

---

## Stage 10 — Final Score Fusion

Semantic relevance is computed as:

$$
Semantic = 0.55\times CE_{adj} + 0.25\times BE + 0.20\times BM25
$$

where

* (CE_{adj}) is the confidence-calibrated Cross-Encoder score,
* (BE) is the Bi-Encoder similarity,
* (BM25) is the lexical relevance score.

The final ranking score combines:

* semantic relevance
* quality-based coverage
* evidence density
* Behavior Validation Score (BVS)
* calibrated negative confidence

into a single explainable ranking function.

Each component measures a distinct aspect of candidate quality, minimizing duplicated signals while producing stable and interpretable rankings.

---

# Key Design Principles

The ranking system was designed around several production-inspired principles:

* **Progressive retrieval** — expensive neural models operate only on progressively smaller candidate pools.
* **Hybrid retrieval** — lexical and semantic retrieval complement each other rather than competing.
* **Family-aware diversification** — retrieval is balanced across multiple conceptual JD families.
* **Adaptive evidence selection** — only the strongest semantic evidence is passed to the Cross-Encoder.
* **Consistency-aware reranking** — candidates with multiple strong supporting signals are preferred over isolated matches.
* **Quality-weighted coverage** — breadth is measured using semantic quality rather than simple family counts.
* **Behavior-aware ranking** — structured recruiter signals refine semantic relevance without dominating it.
* **Explainable scoring** — every major score corresponds to an interpretable hiring dimension.

---

# Caching

To improve reproducibility and runtime, the pipeline caches intermediate computations including:

* Sentence Transformer models
* Cross-Encoder models
* JD embeddings
* Candidate embedding scores
* Cross-Encoder pairwise scores

The first execution downloads models and creates caches. Subsequent executions reuse these cached artifacts to significantly reduce runtime while preserving deterministic behaviour.

---

# Notes

* Candidate data is intentionally excluded from this repository because of its size.
* Place the candidate dataset (for example `candidates.jsonl`) in the repository root before execution.
* The first run may take longer due to model downloads and cache generation.
* Subsequent executions reuse cached models and intermediate computations for substantially faster execution.
* The pipeline is fully CPU compatible and designed to satisfy the Redrob Hackathon runtime constraints while maintaining explainability and reproducibility.
