# Pipeline Overview

The ranking pipeline progressively narrows the candidate search space while increasing the sophistication of evaluation at each stage. Earlier stages are designed to efficiently eliminate obvious mismatches, whereas later stages perform increasingly expensive semantic verification only on the strongest candidates.

```
L0 Structured Triage
        │
        ▼
Behavior Validation Score (BVS)
        │
        ▼
Candidate Enrichment
        │
        ▼
BM25 Lexical Retrieval
        │
        ▼
Dense Vector Retrieval
(Bi-Encoder)
        │
        ▼
Family-aware Candidate Recall
        │
        ▼
Cross-Encoder Pairwise Reranking
        │
        ▼
Coverage Quality
+
Evidence Density
+
Negative Confidence
        │
        ▼
Final Score Fusion
        │
        ▼
Ranked Candidate List
```

## Stage 1 — L0 Structured Triage

The pipeline begins with a conservative structured filtering stage using only explicit candidate metadata such as location, relocation preference, notice period, recruiter activity, profile completeness, and other structured hiring signals.

This stage intentionally avoids semantic reasoning and removes only candidates with clear eligibility mismatches, reducing the search space while minimizing false negatives.

---

## Stage 2 — Behavior Validation Score (BVS)

Every remaining candidate receives a Behavior Validation Score (BVS), representing recruiter-oriented profile quality independent of semantic relevance.

BVS aggregates structured hiring signals including:

* recruiter responsiveness
* profile activity
* notice period
* experience alignment
* interview completion
* assessment performance
* profile quality
* market engagement
* verification signals

The score is calibrated using percentile normalization and smooth sigmoid shaping to prevent score saturation while preserving meaningful separation between candidates.

---

## Stage 3 — Candidate Enrichment

Candidate narratives are enriched before retrieval using lightweight taxonomy expansion.

Relevant technologies, production signals, retrieval terminology, deployment experience, and domain-specific concepts are appended as hidden retrieval tags.

This improves recall without modifying the original candidate information or requiring additional embedding models.

---

## Stage 4 — Hybrid Retrieval

Two complementary retrieval strategies operate in parallel.

### BM25 Retrieval

Sparse lexical matching identifies exact terminology overlap between the job description and candidate profile.

This stage captures:

* exact skills
* technologies
* product terminology
* recruiter keywords

---

### Dense Vector Retrieval

A Sentence Transformer Bi-Encoder converts candidate narratives and JD chunks into dense embeddings.

Semantic similarity is computed using cosine similarity

[
\mathrm{CosSim}(x,y)=
\frac{x\cdot y}
{|x||y|}
]

allowing semantically related concepts to match even when exact wording differs.

---

## Stage 5 — Family-aware Candidate Recall

Rather than selecting only globally highest scoring chunks, retrieval is diversified across multiple JD concept families such as:

* Retrieval
* Evaluation
* Systems
* Product
* Domain
* Culture
* Advanced Skills

This prevents a single topic from dominating retrieval and ensures broad coverage of the complete job description.

---

## Stage 6 — Cross-Encoder Reranking

The strongest candidate–JD evidence pairs are evaluated using a Cross-Encoder.

Unlike the Bi-Encoder, which embeds documents independently, the Cross-Encoder jointly processes both candidate evidence and JD text, producing substantially more accurate pairwise relevance estimates.

Recent roles receive higher influence through recency-aware weighting while historical roles contribute proportionally less.

The Cross-Encoder confidence is softly calibrated before fusion rather than directly altering overall ranking.

---

## Stage 7 — Coverage Quality

Coverage measures how completely a candidate satisfies the different conceptual areas of the job description.

Instead of counting matched families, the pipeline evaluates the strongest semantic evidence within each family and computes a weighted quality score

[
Coverage=
\frac{
\sum_i
w_i
\cdot
Best_i
}
{
\sum_i w_i
}
]

where

* (Best_i) is the strongest semantic similarity for family (i)
* (w_i) is the corresponding JD importance weight.

This rewards both breadth and quality while avoiding double counting.

---

## Stage 8 — Evidence Density

Evidence Density measures the consistency of strong supporting evidence across the candidate profile.

Instead of rewarding large numbers of weak matches, only the strongest semantic evidence contributes

[
Evidence=
0.5S_1+
0.3S_2+
0.2S_3
]

where

(S_1,S_2,S_3)

are the three strongest Cross-Encoder evidence scores.

This favors candidates demonstrating consistently strong alignment rather than isolated high-scoring passages.

---

## Stage 9 — Negative Confidence

Potential recruiter risks such as wrapper-only experience, research-only backgrounds, or weak production evidence are summarized into a calibrated Negative Confidence score.

Rather than hard penalties, confidence is smoothly normalized so isolated keywords do not disproportionately reduce ranking.

---

## Stage 10 — Final Score Fusion

The final ranking combines semantic relevance with structured recruiter signals while keeping each component responsible for a single aspect of candidate quality.

Semantic relevance is computed as

[
Semantic=
0.55\times CE_{adj}
+
0.25\times BE
+
0.20\times BM25
]

where

* (CE_{adj}) is the confidence-calibrated Cross-Encoder score,
* (BE) is the Bi-Encoder similarity,
* (BM25) is the lexical relevance score.

The final ranking score is then obtained by combining

* semantic relevance,
* coverage quality,
* evidence density,
* Behavior Validation Score (BVS),
* calibrated negative confidence,

into a single interpretable ranking function.

Each component measures a distinct property of candidate quality, reducing duplicated signals while producing stable, explainable rankings suitable for production-inspired recruiter workflows.
