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

The ranking pipeline reduces the candidate search space stage by stage.

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

This stage uses only explicit candidate metadata.

It removes only clear eligibility mismatches such as:

* notice period
* recruiter activity
* relocation preference
* location compatibility
* profile completeness
* structured hiring attributes

No semantic matching is used here.

The goal is to shrink the search space while keeping false negatives low.

---

## Stage 2 — Behavior Validation Score (BVS)

Each surviving candidate receives a Behavior Validation Score.

The shaped score is:

$$
\mathrm{shape\_bvs\_score}(x)=\mathrm{clamp01}\left(0.02+0.96\,\sigma\!\left((x-0.50)\cdot 4.20\right)\right)
$$

where $\sigma$ is the sigmoid function.

The percentile rank is:

$$
\mathrm{percentile\_rank}(s)_i=1-\frac{p_i}{n-1}
$$

The stretched percentile is:

$$
\mathrm{stretch\_bvs\_percentile}(r_i)=1-(1-r_i)^{0.45}
$$

The final BVS blend is:

$$
\mathrm{BVS}=(1-\lambda)\,s+\lambda\,t
$$

where:

$$
\lambda=0.34
$$

$$
s=\mathrm{raw\_bvs}
$$

$$
t=\mathrm{stretch\_bvs\_percentile}\!\left(\mathrm{percentile\_rank}(s)\right)
$$

The score combines structured hiring signals including recruiter responsiveness, assessment performance, interview completion, notice period, experience alignment, profile quality, verification status, market engagement, and recruiter activity.

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

### Dense Vector Retrieval (Bi-Encoder)

Candidate narratives and JD chunks are encoded with a Sentence Transformer Bi-Encoder.

Semantic similarity uses cosine similarity:

$$
\mathrm{CosSim}(x,y)=\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}
$$

The retrieval stage keeps both lexical and semantic signals.

---

## Stage 5 — Family-aware Candidate Recall

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

## Stage 6 — Adaptive Cross-Encoder Reranking

The Cross-Encoder sees only selected candidate evidence chunks.

The selection limits are:

* up to 6 positive chunks
* up to 2 negative chunks

Chunk selection is guided by earlier retrieval strength and by a light JD weight preference.

Each selected candidate chunk is then jointly encoded with its corresponding JD chunk using a Cross-Encoder.

The agreement score is:

$$
agreement=\mathrm{clamp01}\left(1-\frac{\lvert ce_{proxy}-bi_{proxy}\rvert+\lvert ce_{proxy}-bm25_{proxy}\rvert+\lvert bi_{proxy}-bm25_{proxy}\rvert}{3}\right)
$$

The smart multiplier is:

$$
smart\_multiplier=0.90+0.10\times agreement
$$

The calibrated Cross-Encoder contribution is:

$$
ce_{adjusted}=ce_{tech}\times smart\_multiplier\times e^{-2.4\max(0,cross_{neg})}
$$

The semantic proxy is:

$$
semantic_{proxy}=\mathrm{clamp01}(semantic_{boost})^{0.92}
$$

The base fusion weights are:

$$
b=[0.55,0.25,0.20]
$$

The signal vector is:

$$
s=[ce_{proxy},semantic_{proxy},bm25_{proxy}]
$$

The consensus is:

$$
\bar{s}=\frac{s_1+s_2+s_3}{3}
$$

The reliability terms are:

$$
r_i=e^{-4\lvert s_i-\bar{s}\rvert}
$$

The normalized reliability weights are:

$$
\hat{r}_i=\frac{r_i}{\sum_{j=1}^{3}r_j}
$$

The final fusion weights are:

$$
w_i=\frac{0.92\,b_i+0.08\,\hat{r}_i}{\sum_{j=1}^{3}\left(0.92\,b_j+0.08\,\hat{r}_j\right)}
$$

The semantic core is:

$$
semantic_{core}=\sum_{i=1}^{3}w_i x_i
$$

where:

$$
x=[ce_{adjusted},semantic_{proxy},bm25_{proxy}]
$$

The upper-tail stretch is:

$$
\mathrm{stretch\_upper\_tail}(x;t,\gamma)=
\begin{cases}
x, & x\le t \\
t+(1-t)\left(\frac{x-t}{1-t}\right)^{\gamma}, & x>t
\end{cases}
$$

with:

$$
t=0.75
$$

$$
\gamma=0.92
$$

The final semantic score is:

$$
semantic_{core}=\mathrm{stretch\_upper\_tail}(semantic_{core};0.75,0.92)
$$

---

## Stage 7 — Quality-based Coverage

Coverage measures how much of the JD is covered.

For each family, the code takes the strongest semantic score in that family and weights it by JD importance.

$$
quality=\frac{\sum_i w_i\,Best_i}{\sum_i w_i}
$$

$$
breadth=\frac{\lvert \mathrm{families\_hit}\rvert}{7}
$$

The coverage bonus is:

$$
Coverage=0.06\times\sigma\!\left(\left(0.80\,quality+0.20\,breadth-0.45\right)\times 6.0\right)
$$

This rewards both semantic quality and conceptual breadth while avoiding duplicated evidence.

---

## Stage 8 — Evidence Density

Evidence Density measures how consistently the candidate is supported by strong evidence.

The code keeps the strongest three positive chunks.

The selection order uses:

$$
rank_i=S_i\sqrt{w_i}
$$

The aggregation is:

$$
E=\frac{\sum_{i=1}^{3}\alpha_i\,w_i\,S_i}{\sum_{i=1}^{3}\alpha_i\,w_i}
$$

where:

$$
\alpha=[0.5,0.3,0.2]
$$

The evidence bonus is:

$$
Evidence=0.06\times\sigma\!\left((E-0.45)\times 6.5\right)
$$

This keeps the score focused on the strongest evidence while still respecting JD importance and consistency.

---

## Stage 9 — Negative Confidence

Negative Confidence summarizes recruiter risk from the candidate narrative.

The negative signal is aggregated as:

$$
\mathrm{negative\_confidence}=\frac{\mathrm{negative\_strength}}{\mathrm{negative\_strength}+\mathrm{positive\_strength}+0.20}
$$

The penalty is:

$$
\mathrm{negative\_confidence\_penalty}(n)=
\begin{cases}
1.0, & n\le 0.18 \\
e^{-2.40\,(n-0.18)}, & n>0.18
\end{cases}
$$

This keeps negative evidence bounded and explainable.

---

## Stage 10 — Final Score Fusion

The final score uses semantic relevance as the main signal.

The centered BVS contribution is:

$$
\mathrm{behavior\_boost}=BVS-0.5
$$

$$
\mathrm{bvs\_bonus}=0.17\times \mathrm{behavior\_boost}
$$

The raw final score is:

$$
\mathrm{final\_raw}=0.72\times semantic_{core}+\mathrm{bvs\_bonus}+Coverage+Evidence
$$

The final score is:

$$
\mathrm{final\_score}=\mathrm{final\_raw}\times \mathrm{negative\_confidence\_penalty}(\mathrm{negative\_confidence})
$$

This keeps semantic relevance dominant while using coverage, evidence, BVS, and negative confidence as bounded refinements.

---

# Outputs and Diagnostics

The main submission file is:

* `Real_RR.csv`

The pipeline can also write:

* `diagnostic_top20.csv` for manual inspection
* optional stage-specific report CSVs when enabled via CLI flags
* cached model/embedding artifacts under `models/` and `cache/`

The explanation text in the final output is generated from candidate narrative text, structured BVS signals, and the selected evidence passages. Internal retrieval tags are used only for retrieval and do not need to appear in the final explanation.

---

# Caching

To improve reproducibility and runtime, the pipeline caches intermediate computations including:

* Sentence Transformer models
* Cross-Encoder models
* JD embeddings

The first execution downloads models and creates caches. Subsequent executions reuse these cached artifacts to significantly reduce runtime while preserving deterministic behaviour.

---

# Notes

* Candidate data is intentionally excluded from this repository because of its size.
* Place the candidate dataset (for example `candidates.jsonl`) in the repository root before execution.
* The first run may take longer due to model downloads and cache generation.
* Subsequent executions reuse cached models and intermediate computations for substantially faster execution.
* The pipeline is fully CPU compatible and designed to satisfy the Redrob Hackathon runtime constraints while maintaining explainability and reproducibility.
