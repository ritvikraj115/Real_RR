# Redrob Hackathon Ranker Package

This repository contains our submission-ready ranking system for the Redrob Hackathon.

## What this project does

We rank candidates for the job description using a compact offline pipeline built for the hackathon constraints:
BM25 → Bi-Encoder → Cross-Encoder → final top 100.

The goal is not only to find good candidates, but to rank them in a way that is fast, reproducible, and explainable under a 5-minute CPU-only limit.

## Why this ranker is different

Most rankers over-focus on one signal. Ours tries to balance the full problem:

- **Retrieval breadth** through BM25 family recall
- **Semantic matching** through a bi-encoder shortlist
- **Precision reranking** through a cross-encoder
- **Behavioral fit** through BVS
- **Coverage and evidence bonuses** so candidates who match more of the JD are rewarded
- **Negative confidence penalties** so wrapper-only, research-only, and demo-only profiles are pushed down
- **Overlap suppression** so repeated JD ideas do not inflate scores unfairly

That combination keeps us away from “keyword stuffing” and toward a more realistic hiring-style ranking.

## The math idea behind the model

The pipeline is designed like a funnel.

First, we keep the candidate pool broad enough to avoid missing strong profiles. Then we narrow it down using stronger scoring.

At a high level:

1. **L0 triage** removes only clearly invalid or direct-signal mismatches.
2. **BM25** retrieves candidates by lexical fit across JD families.
3. **Bi-Encoder** gives a semantic shortlist for deeper matching.
4. **Cross-Encoder** reranks only the best candidate-chunk pairs.
5. **Final score fusion** combines:
   - semantic match,
   - BVS,
   - coverage bonus,
   - evidence density bonus,
   - negative confidence penalty.

The scoring is intentionally not a simple one-signal ranking. It uses a mix of additive and multiplicative logic so one strong signal cannot completely hide weak overall fit.

## What we changed carefully

We kept the architecture the same, but improved it in ways that matter in a hackathon:

- We made CE chunk selection more focused so runtime stays safe.
- We improved breadth by rewarding candidates who cover more JD families.
- We extracted achievement sentences so the cross-encoder sees real impact, not only profile text.
- We made the negative side a confidence score instead of a blunt keyword hit.
- We kept BVS meaningful, but not overpowering.
- We avoided extra output files and keep the final submission clean.

## Step-by-step flow

The ranker follows this order:

1. Read `jd_hybrid_index.json`
2. Load candidates from `candidates.jsonl`
3. Apply L0 triage
4. Build BM25 recall over JD families
5. Build bi-encoder scores for the shortlist
6. Select the best 120 candidates for cross-encoding
7. Rerank with cross-encoder using only the strongest chunks
8. Apply coverage, evidence, BVS, and negative penalties
9. Write `Real_RR.csv`

## Caching strategy

We keep only the local caches that help runtime and reproducibility:

- `cache/jd_vector_embeddings.npz`
- `cache/pre_shortlist_vector_scores.npz`
- `cache/shortlist_cross_scores.npz`

These are local speedups. Candidate-level caches are intentionally not the focus, so the pipeline stays cleaner and safer for evaluation.

## Files to keep

- `rank_candidates.py` — full ranking pipeline
- `validate_submission.py` — CSV validator
- `requirements-ranker.txt` — dependencies
- `submission_metadata.yaml` — submission metadata
- `README.md` — setup and usage
- `cache/` — local cache directory

## Typical commands

```bash
python rank_candidates.py --candidates candidates.jsonl --jd-index jd_hybrid_index.json --output Real_RR.csv
python validate_submission.py Real_RR.csv