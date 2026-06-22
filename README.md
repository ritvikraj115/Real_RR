# Redrob Hackathon Ranker

This repository contains our submission for the **Redrob Intelligent Candidate Discovery Challenge**.

The project implements a **production-inspired, multi-stage hybrid candidate ranking pipeline** that progressively narrows the candidate search space while increasing the sophistication of evaluation at each stage. It combines structured recruiter signals, hybrid information retrieval, semantic reranking, and explainable score fusion to rank candidates for a given job description under the hackathon's CPU-only runtime constraints.

Rather than relying on a single ranking signal, the system combines **structured eligibility**, **lexical retrieval**, **semantic retrieval**, **pairwise neural verification**, and **behavior-aware scoring** into one interpretable ranking framework. The latest version also adds agreement-aware Cross-Encoder calibration, JD-weighted evidence scoring, and a small adaptive semantic fusion layer that stays close to the original 0.55 / 0.25 / 0.20 blend while remaining bounded and stable.

---

# Repository Structure

```text
.
├── models/                         # Cached SentenceTransformer and Cross-Encoder models
├── app.py                          # Optional Streamlit demonstration
├── jd_hybrid_index.json            # Preprocessed hybrid JD index
├── rank_candidates.py              # Main ranking pipeline
├── validate_submission.py          # Submission validator
├── submission_metadata.yaml        # Challenge metadata
├── requirements.txt                # Python dependencies
└── README.md
```

Generated when you run the pipeline:

```text
Real_RR.csv                         # Final submission file
```

---

# Environment

The pipeline was developed and validated using the following environment.

| Component | Version |
| ---------------- | --------------- |
| Python | **3.12.x** |
| Execution | CPU Only |
| Operating System | Windows / Linux |
| GPU | Not Required |

> **Important**
>
> The project was developed and tested using **Python 3.12**. Some combinations of **PyTorch**, **Transformers**, and **Sentence-Transformers** may behave differently on older Python versions. For reproducible results, we recommend running the project with **Python 3.12**.

---

# Installation

We recommend using a clean virtual environment.

```bash
# Clone the repo
git clone 
cd Real_RR

# Create virtual environment (Python 3.12)
py -3.12 -m venv .venv

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
python rank_candidates.py --candidates candidates.jsonl --jd-index jd_hybrid_index.json --output Real_RR.csv
```

Validate the generated submission:

```bash
python validate_submission.py Real_RR.csv
```

If you want to inspect intermediate outputs, enable the optional report flags in `rank_candidates.py` and the pipeline will write additional CSVs alongside the final submission.

---



# Pipeline Overview

The ranking pipeline reduces the candidate pool stage by stage.

```text
L0 Structured Triage
        │
        ▼
Behavior Validation Score
        │
        ▼
Candidate Enrichment
        │
        ▼
Hybrid Retrieval
 ├── BM25 Lexical Retrieval
 └── Dense Vector Retrieval
        │
        ▼
Family aware Candidate Recall
        │
        ▼
Adaptive Cross Encoder Chunk Selection
        │
        ▼
Cross Encoder Pairwise Reranking
        │
        ▼
Quality based Coverage
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

This stage uses only explicit candidate metadata.

It removes only clear eligibility mismatches such as:

* notice period
* recruiter activity
* relocation preference
* location compatibility
* profile completeness
* other structured hiring signals

No semantic matching is used here.

The goal is to shrink the search space while keeping false negatives low.

---

## Stage 2 — Behavior Validation Score

Each surviving candidate receives a Behavior Validation Score.

The raw structured score is shaped as:

$$
\operatorname{shape\_bvs\_score}(x)
=
\operatorname{clip}_{0}^{1}
\left(
0.02
+
0.96\,\sigma\!\left((x-0.50)\times 4.20\right)
\right)
$$

where $\sigma$ is the sigmoid function.

The code then blends the shaped score with a percentile stretched version:

$$
r_i
=
\operatorname{percentile\_rank}(s)_i
=
1
-
\frac{p_i}{n-1}
$$

$$
t_i
=
\operatorname{stretch\_bvs\_percentile}(r_i)
=
1
-
(1-r_i)^{0.45}
$$

$$
\mathrm{BVS}
=
(1-0.34)\,s
+
0.34\,t
$$

The candidate is discarded if the shaped score falls below the L0 threshold:

$$
\mathrm{BVS} < 0.27
$$

---

## Stage 3 — Candidate Enrichment

Candidate text is enriched for retrieval only.

The code appends hidden retrieval tags from lightweight positive and negative taxonomy rules.

These tags improve retrieval recall.

They are not shown in the final explanation.

---

## Stage 4 — Hybrid Retrieval

The pipeline combines lexical retrieval and dense semantic retrieval.

### BM25 Lexical Retrieval

BM25 captures exact lexical overlap between the job description and the candidate narrative.

### Dense Vector Retrieval

Candidate narratives and JD chunks are encoded with a Sentence Transformer Bi Encoder.

Semantic similarity uses cosine similarity:

$$
\operatorname{CosSim}(x,y)
=
\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}
$$

The retrieval stage keeps both lexical and semantic signals.

---

## Stage 5 — Family aware Candidate Recall

Retrieval is diversified across JD families instead of taking only the global top scores.

The positive families are:

* Retrieval
* Evaluation
* Systems
* Product
* Domain
* Culture
* Advanced Skills

This prevents one topic from dominating the pool.

---

## Stage 6 — Adaptive Cross Encoder Reranking

The Cross Encoder sees only selected candidate evidence chunks.

The selection limits are:

* up to 6 positive chunks
* up to 2 negative chunks

Chunk selection is guided by earlier retrieval strength and by a light JD weight preference.

The code applies a smart multiplier using agreement between the three retrieval signals:

$$
agreement
=
1
-
\frac{
\lvert ce\_proxy-bi\_proxy\rvert
+
\lvert ce\_proxy-bm25\_proxy\rvert
+
\lvert bi\_proxy-bm25\_proxy\rvert
}{3}
$$

$$
smart\_multiplier
=
0.90
+
0.10 \times agreement
$$

The calibrated Cross Encoder contribution is:

$$
ce\_adjusted
=
ce\_tech
\times
smart\_multiplier
\times
e^{-2.4\,\max(0,cross\_neg)}
$$

The semantic fusion signal is then built from:

$$
semantic\_proxy
=
\operatorname{clamp01}(semantic\_boost)^{0.92}
$$

$$
fusion\_base
=
[0.55,\;0.25,\;0.20]
$$

$$
reliability_i
=
e^{-4\lvert s_i-\bar{s}\rvert}
$$

where

$$
s
=
\begin{bmatrix}
ce\_proxy\\
semantic\_proxy\\
bm25\_proxy
\end{bmatrix}
$$

and

$$
\bar{s}
=
\frac{ce\_proxy + semantic\_proxy + bm25\_proxy}{3}
$$

The final fusion weights are:

$$
fusion\_weights
=
\operatorname{normalize}
\left(
0.92 \cdot fusion\_base
+
0.08 \cdot reliability
\right)
$$

The semantic core is:

$$
semantic\_core
=
fusion\_weights \cdot [ce\_adjusted,\;semantic\_proxy,\;bm25\_proxy]
$$

The strongest semantic scores are then stretched only in the upper tail:

$$
semantic\_core
=
\operatorname{stretch\_upper\_tail}(semantic\_core)
$$

with

$$
threshold = 0.75
$$

$$
gamma = 0.92
$$

---

## Stage 7 — Quality based Coverage

Coverage measures how much of the JD is covered.

For each family, the code takes the strongest semantic score in that family and weights it by JD importance.

$$
quality
=
\frac{\sum_i w_i\,Best_i}{\sum_i w_i}
$$

$$
breadth
=
\frac{\lvert families\_hit\rvert}{7}
$$

$$
Coverage
=
0.06
\times
\sigma\!\left(
\left(
0.80\,quality
+
0.20\,breadth
-
0.45
\right)
\times
6.0
\right)
$$

---

## Stage 8 — Evidence Density

Evidence Density measures how consistently the candidate is supported by strong evidence.

The code keeps the strongest three positive chunks.

The selection order uses:

$$
S_i\sqrt{w_i}
$$

The aggregation is:

$$
E
=
\frac{\sum_{i=1}^{3}\alpha_i\,w_i\,S_i}{\sum_{i=1}^{3}\alpha_i\,w_i}
$$

where

$$
\alpha
=
[0.5,\;0.3,\;0.2]
$$

and

$$
Evidence
=
0.06
\times
\sigma\!\left(
\left(
E
-
0.45
\right)
\times
6.5
\right)
$$

---

## Stage 9 — Negative Confidence

Negative Confidence summarizes recruiter risk from the candidate narrative.

The negative signal is aggregated as:

$$
negative\_confidence
=
\frac{negative\_strength}{negative\_strength + positive\_strength + 0.20}
$$

The penalty is:

$$
negative\_confidence\_penalty(n)
=
\begin{cases}
1.0 & n \le 0.18 \\
e^{-2.40\,(n-0.18)} & n > 0.18
\end{cases}
$$

This keeps negative evidence bounded and explainable.

---

## Stage 10 — Final Score Fusion

The final score uses semantic relevance as the main signal.

The BVS contribution is centered:

$$
bvs\_bonus
=
0.17\,(BVS-0.5)
$$

The raw final score is:

$$
final\_raw
=
0.72\,semantic\_core
+
bvs\_bonus
+
Coverage
+
Evidence
$$

The final score is:

$$
final\_score
=
final\_raw
\times
negative\_confidence\_penalty(negative\_confidence)
$$

This keeps semantic relevance dominant while using coverage evidence BVS and negative confidence as bounded refinements.

---

## Outputs and Diagnostics

The main submission file is:

* `Real_RR.csv`

The pipeline can also write:

* `diagnostic_top20.csv`
* optional stage specific report CSVs
* cached model and embedding artifacts under `models/` and `cache/`

The main CSV contains these fields:

* `candidate_id`
* `semantic_score`
* `negative_score`
* `semantic_final_raw`
* `semantic_final_norm`
* `bvs_score`
* `final_score`

---

## Caching

To improve reproducibility and runtime the pipeline caches:

* Sentence Transformer models
* Cross Encoder models
* JD embeddings

The first execution downloads models and creates caches.

Later runs reuse the cached artifacts.


# Notes

* Candidate data is intentionally excluded from this repository because of its size.
* Place the candidate dataset (for example `candidates.jsonl`) in the repository root before execution.
* The first run may take longer due to model downloads and cache generation.
* Subsequent executions reuse cached models and intermediate computations for substantially faster execution.
* The pipeline is fully CPU compatible and designed to satisfy the Redrob Hackathon runtime constraints while maintaining explainability and reproducibility.
