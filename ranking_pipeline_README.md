# Redrob Hackathon Ranker Package

This folder is the submission-friendly subset of the repo.

## Files to keep
- `rank_candidates.py` — the full ranking pipeline
- `validate_submission.py` — CSV format validator
- `requirements-ranker.txt` — runtime dependencies
- `submission_metadata.yaml` — submission metadata template
- `README.md` — quick usage notes
- `cache/` — local cache directory for JD embeddings and similarity scores

## What is cached
The ranker reuses:
- `cache/jd_vector_embeddings.npz`
- `cache/pre_shortlist_vector_scores.npz`
- `cache/shortlist_cross_scores.npz`

These caches are local speedups. They do not need to be committed to GitHub.
The repository should ignore the generated `.npz` files.

## Typical commands

```bash
python rank_candidates.py --candidates candidates.jsonl --jd-index jd_hybrid_index.json --output final_top100_submission.csv
python validate_submission.py final_top100_submission.csv
```

## Submission checks
The validator expects:
- CSV header: `candidate_id,rank,score,reasoning`
- Exactly 100 data rows
- `candidate_id` format `CAND_XXXXXXX`
- Non-increasing score order by rank
