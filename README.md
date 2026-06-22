# Redrob Hackathon Ranker

This repository contains our submission for the **Redrob Intelligent Candidate Discovery Challenge**.

The project implements a **production-inspired, multi-stage hybrid candidate ranking pipeline** that progressively narrows the candidate search space while increasing the sophistication of evaluation at each stage. It combines structured recruiter signals, hybrid information retrieval, semantic reranking, and explainable score fusion to rank candidates for a given job description under the hackathon's CPU-only runtime constraints.

Rather than relying on a single ranking signal, the system combines **structured eligibility**, **lexical retrieval**, **semantic retrieval**, **pairwise neural verification**, and **behavior-aware scoring** into one interpretable ranking framework. The latest version also adds agreement-aware Cross-Encoder calibration, JD-weighted evidence scoring, and a small adaptive semantic fusion layer that stays close to the original 0.55 / 0.25 / 0.20 blend while remaining bounded and stable.

---

# Repository Structure

```text
.
├── models/                         # Cached SentenceTransformer and Cross-Encoder models
├── cache/                          # Cached JD embeddings and intermediate artifacts
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
diagnostic_top20.csv                # Optional manual inspection output
```

Optional diagnostic exports may also be written when the corresponding CLI flags are enabled.

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
git clone <repo-url>
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

No semantic matching is performed at this stage. That keeps the filter fast and conservative, so the system reduces the search space while minimizing false negatives.

---

## Stage 2 — Behavior Validation Score (BVS)

Each remaining candidate receives a **Behavior Validation Score (BVS)** that measures recruiter-oriented profile quality independently of semantic relevance.

The raw structured score is first shaped with a smooth non-saturating curve:

$$
r = 0.02 + 0.96\,\sigma\!\left(4.20(x-0.50)\right)
$$

where $\sigma$ is the sigmoid function.

The shaped score is then blended with a percentile-stretched version to keep the upper tail separable:

$$
\mathrm{BVS}=(1-\lambda)r+\lambda\left(1-\bigl(1-\pi(r)\bigr)^{0.45}\right)
$$

where $\pi(r)$ is the percentile rank of $r$ and $\lambda=0.34$.

This keeps the score spread meaningful while reducing top-end compression.

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

### Dense Vector Retrieval (Bi-Encoder)

Candidate narratives and JD chunks are encoded using a Sentence Transformer Bi-Encoder.

Semantic similarity is computed using cosine similarity:

$$
\mathrm{CosSim}(x,y)=\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}
$$

This allows semantically similar concepts to match even when different terminology is used.

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

When the Cross-Encoder returns logits, they are centered and converted to probabilities:

$$
p=\sigma\!\left(\frac{z-\beta}{T}\right)
$$

where:

* $z$ is the raw Cross-Encoder logit,
* $\beta$ is the median finite shortlist logit when logits are available,
* $T$ is the calibration temperature.

The temperature is chosen from the shortlist spread when enough finite logits exist, and the result is clipped to $[10^{-4},\,1-10^{-4}]$ for numerical stability.

The Cross-Encoder contribution is then softly adjusted using an agreement-based Smart Multiplier:

$$
a=\max\left(0,\min\left(1,1-\frac{|c-v|+|c-l|+|v-l|}{3}\right)\right)
$$

$$
m=0.90+0.10a
$$

$$
c_{\mathrm{adj}}=c\,m\,e^{-2.4\max(0,n)}
$$

where $c$ is the Cross-Encoder score, $v$ is the Bi-Encoder score, $l$ is the BM25 score, and $n$ is the negative evidence confidence.

The semantic fusion stage then applies a small consensus-aware calibration around the base retrieval blend:

$$
\mathbf{w}_0=[0.55,\,0.25,\,0.20]
$$

$$
\hat{\mathbf{r}}=\frac{[r_c,\,r_v,\,r_l]}{\lVert[r_c,\,r_v,\,r_l]\rVert_1}
$$

$$
\hat{\mathbf{w}}=\frac{0.92\,\mathbf{w}_0+0.08\,\hat{\mathbf{r}}}{\lVert0.92\,\mathbf{w}_0+0.08\,\hat{\mathbf{r}}\rVert_1}
$$

$$
S=\hat{w}_1 c_{\mathrm{adj}}+\hat{w}_2 v+\hat{w}_3 l
$$

The resulting semantic score is then stretched only in the upper tail so the strongest matches separate a little more without changing the ordering of the rest of the scale:

$$
S'=
\begin{cases}
S, & S\le 0.75 \\
0.75+0.25\left(\dfrac{S-0.75}{0.25}\right)^{0.92}, & S>0.75
\end{cases}
$$

---

## Stage 7 — Coverage Quality

Coverage measures how completely a candidate satisfies the conceptual areas represented within the job description.

Rather than counting matched families, the pipeline evaluates the strongest semantic evidence within each family and computes a weighted quality score:

$$
q=\frac{\sum_i w_i\,Best_i}{\sum_i w_i}
$$

where:

* $Best_i$ is the strongest semantic similarity observed for family $i$,
* $w_i$ is the corresponding JD importance weight.

That quality signal is blended with family breadth and then turned into a bounded bonus:

$$
b=\frac{|F|}{7}
$$

$$
C=0.06\times\sigma\!\left(6.0\,(0.80q+0.20b-0.45)\right)
$$

where $F$ is the set of families hit.

This rewards both semantic quality and conceptual breadth while avoiding duplicated evidence.

---

## Stage 8 — Evidence Density

Evidence Density measures the consistency of supporting semantic evidence throughout the candidate profile.

Instead of rewarding numerous weak matches, only the strongest semantic evidence contributes. The implementation also uses JD weights so high-importance evidence matters more than low-importance evidence.

The score is computed from the strongest three positive chunks using fixed emphasis coefficients $\alpha=[0.5,0.3,0.2]$:

$$
z=\frac{\sum_{i=1}^{3}\alpha_i w_i S_i}{\sum_{i=1}^{3}\alpha_i w_i}
$$

$$
D=0.06\times\sigma\!\left(6.5\,(z-0.45)\right)
$$

where:

* $S_i$ is the Cross-Encoder score for the selected evidence,
* $w_i$ is the corresponding JD chunk weight.

The three evidence chunks are selected with a light JD-weight-aware preference before aggregation, so more important chunks can edge out weaker ones when the scores are close.

This keeps the score focused on the strongest evidence while still respecting JD importance and consistency.

---

## Stage 9 — Negative Confidence

Potential recruiter risks—including wrapper-only experience, research-only backgrounds, or weak production evidence—are summarized into a calibrated Negative Confidence score.

The negative families are aggregated into a single confidence value:

$$
n=\frac{g}{g+h+0.20}
$$

where $g$ is negative strength and $h$ is positive strength.

The downstream penalty is then applied as:

$$
P_{\mathrm{neg}}=
\begin{cases}
1.0, & n\le 0.18 \\
e^{-2.4\,(n-0.18)}, & n>0.18
\end{cases}
$$

Rather than applying hard penalties, multiple negative signals are aggregated into a confidence estimate. That keeps the system explainable and reduces false penalties from isolated keywords.

---

## Stage 10 — Final Score Fusion

The final score combines semantic relevance, behavior signals, coverage, evidence, and a calibrated negative penalty.

The final ranking formula in the current code is:

$$
F_{\mathrm{raw}}=0.72\,S' + 0.17\,(BVS-0.5) + C + D
$$

$$
F_{\mathrm{score}}=F_{\mathrm{raw}}\times P_{\mathrm{neg}}
$$

This keeps semantic relevance dominant, while behavior, coverage, evidence, and negative confidence act as bounded refinements instead of replacing the core retrieval signal.

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
