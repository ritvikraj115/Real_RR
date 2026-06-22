# Redrob Hackathon Ranker

This repository contains our submission for the **Redrob Intelligent Candidate Discovery Challenge**.

The project implements a **production-inspired, multi-stage hybrid candidate ranking pipeline** that progressively narrows the candidate search space while increasing the sophistication of evaluation at each stage. It combines structured recruiter signals, hybrid information retrieval, semantic reranking, and explainable score fusion to rank candidates for a given job description under the hackathon's CPU-only runtime constraints.

Rather than relying on a single ranking signal, the system combines **structured eligibility**, **lexical retrieval**, **semantic retrieval**, **pairwise neural verification**, and **behavior-aware scoring** into one interpretable ranking framework. The latest version also adds agreement-aware Cross-Encoder calibration, JD-weighted evidence scoring, and a small adaptive semantic fusion layer that stays close to the original [0.55, 0.25, 0.20] blend.

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
├── README.md
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
git clone https://github.com/ritvikraj115/Real_RR.git

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

---

# Pipeline Overview

The ranking pipeline progressively reduces the candidate search space while applying increasingly sophisticated evaluation methods. Earlier stages remove obvious mismatches quickly, while later stages spend more compute only on the strongest candidates.

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

Each remaining candidate receives a **Behavior Validation Score (BVS)** from structured hiring signals.

The raw score is shaped with:

$$
\mathrm{shape\_bvs\_score}(x)=\mathrm{clip}\left(0.02 + 0.96\,\sigma\!\left((x-0.50)\times 4.20\right),\,0,\,1\right)
$$

Then the downstream BVS is blended with a percentile-stretched version of itself:

$$
\mathrm{BVS} = (1-0.34)\times raw\_bvs + 0.34\times stretch\_bvs\_percentile\bigl(percentile\_rank(raw\_bvs)\bigr)
$$

This keeps the score spread meaningful while reducing top-end compression.

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

Each selected candidate chunk is then jointly encoded with its corresponding JD chunk using a Cross-Encoder, producing significantly more accurate pairwise relevance estimates than independent embeddings.

When the Cross-Encoder returns logits, they are centered and calibrated before conversion to probabilities:

$$
p=\sigma\!\left(\frac{z-b}{T}\right)
$$

where:

* \(z\) is the raw Cross-Encoder logit,
* \(b\) is the median finite shortlist logit when logits are available,
* \(T\) is the calibration temperature,
* \(\sigma\) denotes the sigmoid function.

If the shortlist has at least five finite logits, the code uses \(T=\max(3.50,\,0.5\times(p90-p10))\); otherwise it uses \(T=3.50\). If the model already returns probabilities, the fallback is \(\sigma((z-0.0)/3.50)\). The result is clipped to \([10^{-4},\,1-10^{-4}]\).

The Cross-Encoder contribution is then softly adjusted using an agreement-based Smart Multiplier:

$$
agreement = \mathrm{clip}\!\left(1 - \frac{|ce-bi| + |ce-bm25| + |bi-bm25|}{3},\,0,\,1\right)
$$

$$
smart\_multiplier = 0.90 + 0.10\times agreement
$$

$$
ce\_adjusted = ce\_tech \times smart\_multiplier \times e^{-2.4\times cross\_neg}
$$

This keeps the Cross-Encoder close to the original score while gently discounting disagreement and risky negative evidence.

The strongest semantic evidence is then aggregated using a consistency-aware formulation:

$$
CE = 0.7\times\max(\text{Top3}) + 0.3\times\mathrm{mean}(\text{Top3})
$$

which rewards candidates demonstrating consistently strong evidence rather than relying on a single exceptional passage.

---


## Stage 7 — Coverage Quality

Coverage measures how completely a candidate satisfies the conceptual areas represented within the job description.

Rather than counting matched families, the pipeline evaluates the strongest semantic evidence within each family and computes a weighted quality score:

$$
quality = \frac{\sum_i w_i\cdot Best_i}{\sum_i w_i}
$$

where:

* \(Best_i\) is the strongest semantic similarity observed for family \(i\),
* \(w_i\) is the corresponding JD importance weight.

That quality signal is blended with family breadth and then turned into a bounded bonus:

$$
Coverage = 0.06\times \sigma\!\left((0.80\times quality + 0.20\times breadth - 0.45)\times 6.0\right)
$$

where \(breadth = |families\_hit| / 7\).

This rewards both semantic quality and conceptual breadth while avoiding duplicated evidence.

---


## Stage 8 — Evidence Density

Evidence Density measures the consistency of supporting semantic evidence throughout the candidate profile.

Instead of rewarding numerous weak matches, only the strongest semantic evidence contributes. The implementation also uses JD weights so high-importance evidence matters more than low-importance evidence.

The score is computed from the strongest three positive chunks using fixed emphasis coefficients \(\alpha=[0.5,0.3,0.2]\):

$$
E = \frac{\sum_{i=1}^{3}\alpha_i w_i S_i}{\sum_{i=1}^{3}\alpha_i w_i}
$$

$$
Evidence = 0.06\times \sigma\!\left((E - 0.45)\times 6.5\right)
$$

where:

* \(S_i\) is the Cross-Encoder score for the selected evidence,
* \(w_i\) is the corresponding JD chunk weight.

The three evidence chunks are selected with a light JD-weight-aware preference before aggregation, so more important chunks can edge out weaker ones when the scores are close.

This keeps the score focused on the strongest evidence while still respecting JD importance and consistency.

---


## Stage 9 — Negative Confidence

Potential recruiter risks—including wrapper-only experience, research-only backgrounds, or weak production evidence—are summarized into a calibrated Negative Confidence score.

The negative families are aggregated into a single confidence value:

$$
negative\_confidence = \frac{negative\_strength}{negative\_strength + positive\_strength + 0.20}
$$

The downstream penalty is then applied as:

$$
penalty =
\begin{cases}
1.0, & negative\_confidence \le 0.18 \\
e^{-2.40\,(negative\_confidence - 0.18)}, & negative\_confidence > 0.18
\end{cases}
$$

Rather than applying hard penalties, multiple negative signals are aggregated into a confidence estimate. That keeps the system explainable and reduces false penalties from isolated keywords.

---


## Stage 10 — Final Score Fusion

Semantic relevance is computed using an adaptive reliability-weighted fusion of the three retrieval signals. The weights shift only slightly around the original base blend while still staying normalized.

The base blend starts from:

$$
[0.55,\;0.25,\;0.20]
$$

The code then computes:

$$
semantic\_proxy = semantic\_boost^{0.92}
$$

$$
reliability\_weights = normalize\!\left(e^{-4\,|signal - mean(signal)|}\right)
$$

$$
fusion\_weights = normalize\!\left(0.92\times[0.55,0.25,0.20] + 0.08\times reliability\_weights\right)
$$

$$
semantic\_core = fusion\_weights \cdot [ce\_adjusted,\,semantic\_proxy,\,bm25\_proxy]
$$

$$
semantic\_core = stretch\_upper\_tail(semantic\_core,\,threshold=0.75,\,gamma=0.92)
$$

The final ranking score combines:

$$
final\_raw = 0.72\times semantic\_core + 0.17\times(BVS - 0.5) + coverage\_bonus + evidence\_bonus
$$

$$
final\_score = final\_raw \times negative\_confidence\_penalty(negative\_confidence)
$$

This keeps the ensemble close to the original design while giving slightly more trust to the signals that agree better on a given candidate.

---

# Key Design Principles

The ranking system was designed around several production-inspired principles:

* **Progressive retrieval** — expensive neural models operate only on progressively smaller candidate pools.
* **Hybrid retrieval** — lexical and semantic retrieval complement each other rather than competing.
* **Family-aware diversification** — retrieval is balanced across multiple conceptual JD families.
* **Adaptive evidence selection** — only the strongest semantic evidence is passed to the Cross-Encoder.
* **Agreement-aware CE calibration** — the Cross-Encoder is softly calibrated using agreement with BM25 and the Bi-Encoder, rather than being allowed to dominate blindly.
* **Consistency-aware reranking** — candidates with multiple strong supporting signals are preferred over isolated matches.
* **Quality-weighted coverage** — breadth is measured using semantic quality rather than simple family counts.
* **Weighted evidence density** — the strongest evidence is scored with JD importance awareness instead of raw hit count.
* **Behavior-aware ranking** — structured recruiter signals refine semantic relevance without dominating it.
* **Explainable scoring** — every major score corresponds to an interpretable hiring dimension.

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
