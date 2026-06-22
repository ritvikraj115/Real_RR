#!/usr/bin/env python3
"""Hybrid candidate retrieval and reranking for the Redrob challenge.

The JD semantic layer is read from jd_hybrid_index.json and is never
regenerated here. Semantic retrieval uses only candidate narrative text:
profile.headline, profile.summary, and career_history[].description.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import glob
import gzip
import os

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#/-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
JD_EMBEDDING_CACHE_VERSION = 2
VECTOR_SCORE_CACHE_VERSION = 3
CROSS_SCORE_CACHE_VERSION = 17
BVS_MIN_THRESHOLD = 0.27

# Cross-encoder calibration controls
CROSS_ENCODER_LOGIT_BIAS = 0.0
CROSS_ENCODER_TEMPERATURE = 3.50
CE_SCORE_EPS = 1e-4

# BVS calibration controls
BVS_PERCENTILE_BLEND = 0.34

# Chunk-family coverage controls
CHUNK_FAMILIES: dict[str, str] = {
    "C01": "RETRIEVAL",
    "C02": "RETRIEVAL",
    "C03": "RETRIEVAL",
    "C04": "RETRIEVAL",
    "C13": "RETRIEVAL",
    "C06": "EVALUATION",
    "C07": "EVALUATION",
    "C16": "EVALUATION",
    "C08": "SYSTEMS",
    "C09": "SYSTEMS",
    "C17": "SYSTEMS",
    "C05": "PRODUCT",
    "C10": "PRODUCT",
    "C12": "PRODUCT",
    "C11": "DOMAIN",
    "C14": "CULTURE",
    "C15": "ADVANCED",
    "C18": "ADVANCED",
    "C19": "NEGATIVE",
    "C20": "NEGATIVE",
    "C21": "NEGATIVE",
    "C22": "NEGATIVE",
    "C23": "NEGATIVE",
    "C24": "NEGATIVE",
}
POSITIVE_FAMILY_ORDER = ("RETRIEVAL", "EVALUATION", "SYSTEMS", "PRODUCT", "DOMAIN", "CULTURE", "ADVANCED")
POSITIVE_FAMILY_TARGET = 10
CROSS_ENCODER_MAX_POSITIVE_CHUNKS = 6
CROSS_ENCODER_MAX_NEGATIVE_CHUNKS = 2

BONUS_CHUNK_WEIGHT_MULTIPLIERS: dict[str, float] = {
    "C05": 1.08,  # shipped systems
    "C09": 1.07,  # systems thinking
    "C10": 1.08,  # shipper mindset
    "C11": 1.05,  # recruiter workflow
    "C12": 1.06,  # ownership
    "C14": 1.05,  # async communication
    "C16": 1.10,  # learning-to-rank
    "C17": 1.08,  # distributed systems
}

BM25_FAMILY_RECALL_TARGETS: dict[str, int] = {
    "RETRIEVAL": 250,
    "EVALUATION": 150,
    "SYSTEMS": 150,
    "PRODUCT": 150,
    "DOMAIN": 100,
    "CULTURE": 100,
    "ADVANCED": 100,
}
VECTOR_FAMILY_RECALL_TARGETS: dict[str, int] = {
    "RETRIEVAL": 40,
    "EVALUATION": 25,
    "SYSTEMS": 25,
    "PRODUCT": 25,
    "DOMAIN": 15,
    "CULTURE": 15,
    "ADVANCED": 15,
}
FAMILY_RECALL_ORDER: tuple[str, ...] = POSITIVE_FAMILY_ORDER

# Minimum hybrid score required before a chunk is sent to CE
CROSS_ENCODER_POSITIVE_FLOOR = 0.08
CROSS_ENCODER_NEGATIVE_FLOOR = 0.03

ACHIEVEMENT_SNIPPET_MAX = 5
ACHIEVEMENT_SNIPPET_MAX_CHARS = 800
COVERAGE_BONUS_MAX = 0.06
EVIDENCE_DENSITY_BONUS_MAX = 0.06
COVERAGE_SUPPORT_THRESHOLD = CROSS_ENCODER_POSITIVE_FLOOR
EVIDENCE_DENSITY_THRESHOLD = 0.10
NEGATIVE_CONFIDENCE_RELEASE = 0.18
NEGATIVE_CONFIDENCE_PENALTY_SCALE = 2.40

WELCOME_CITIES = (
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
    "mumbai",
    "hyderabad",
)

PRIMARY_CITIES = ("pune", "noida")

RELEVANT_SKILL_TERMS = (
    "python",
    "machine learning",
    "ml",
    "ai",
    "nlp",
    "information retrieval",
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "embeddings",
    "sentence transformers",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "llm",
    "mlops",
    "mlflow",
    "xgboost",
    "lightgbm",
)


POSITIVE_TAG_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("airflow", "kafka", "spark", "flink", "beam", "etl", "dag", "orchestration", "pipeline"), "data engineering mlops streaming orchestration distributed systems"),
    (("retrieval", "ranking", "search", "recommendation", "relevance", "matching"), "search ranking retrieval recommendation matching"),
    (("embedding", "embeddings", "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch"), "embeddings vector database hybrid search dense retrieval"),
    (("ndcg", "mrr", "map", "a/b", "ab test", "offline", "online", "interleaving", "relevance benchmark"), "evaluation experimentation offline online correlation ab testing"),
    (("python", "pytorch", "sklearn", "scikit-learn", "fastapi", "service", "api", "ml systems", "deployed", "shipped"), "python production code ml systems"),
    (("xgboost", "lightgbm", "lambdamart", "lambdarank", "ranknet", "learning to rank"), "learning to rank xgboost lightgbm neural ranker"),
    (("lora", "qlora", "peft", "instruction tuning", "fine tuning"), "lora qlora peft fine tuning"),
    (("hr", "recruiting", "talent", "ats", "hiring", "candidate"), "hr tech recruiting tech candidate matching talent intelligence"),
    (("open source", "github", "talk", "talks", "blog", "blogs"), "open source public validation"),
    (("deployed", "shipped", "launched", "served", "users", "customers", "production", "live", "scaled", "scale", "latency", "throughput"), "production deployment user facing scale latency throughput"),
    (("owned", "owning", "led", "architected", "built from scratch", "end to end", "cross functional", "mentored", "stakeholder"), "ownership leadership end to end cross functional"),
    (("drift", "index refresh", "reindex", "monitoring", "observability", "rollback", "incident", "relevance regression"), "retrieval operations drift index refresh observability monitoring"),
    (("rerank", "reranking", "cross encoder", "pairwise ranking", "listwise", "lambdaMART", "lambdamart"), "reranking cross encoder ranking"),
    (("graphrag", "graph rag", "knowledge graph", "neo4j", "entity resolution", "graph search"), "graph rag knowledge graph entity resolution"),
    (("async", "asynchronous", "disagree", "disagreement", "documentation", "docs", "decision making", "communication", "writing"), "async writing open disagreement decision making"),
    (("shipper", "founding", "founder", "0 to 1", "v1", "iteration", "feedback loop", "fast iteration"), "shipper ownership fast iteration feedback loop"),
)

NEGATIVE_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (("research", "academic", "thesis", "publication", "paper", "benchmark", "lab"), ("production", "deployed", "shipped", "launched", "users", "live"), "research only academic only no production deployment"),
    (("langchain", "openai", "prompt engineering", "wrapper", "tutorial", "demo", "chatbot", "agent", "agents", "tool calling"), ("retrieval", "ranking", "search", "production", "evaluation", "vector", "embedding"), "wrapper only demo only no retrieval depth no ranking depth"),
    (("computer vision", "cv", "speech", "robotics"), ("retrieval", "ranking", "search", "nlp", "information retrieval"), "computer vision only speech only robotics only no nlp no ir"),
    (("framework", "frameworks", "notebook", "poc", "prototype", "tutorial", "demo"), ("system", "architecture", "observability", "monitoring", "production"), "framework only tutorial only demo only no systems thinking no production"),
    (("closed source", "proprietary"), ("open source", "github", "paper", "papers", "talk", "talks", "blog", "blogs"), "closed source only no external validation"),
)

ROLE_SIGNAL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("deployed", "shipped", "launched", "served", "users", "customers", "production", "live", "scaled", "scale", "latency", "throughput"), "production deployment user facing scale latency throughput"),
    (("owned", "owning", "led", "architected", "built from scratch", "end to end", "cross functional", "mentored", "stakeholder"), "ownership leadership end to end cross functional"),
    (("drift", "index refresh", "reindex", "monitoring", "observability", "rollback", "incident", "relevance regression"), "retrieval operations drift index refresh observability monitoring"),
    (("retrieval", "ranking", "search", "recommendation", "rerank", "reranking", "vector", "embedding", "hybrid search", "dense retrieval"), "retrieval ranking search embeddings hybrid search reranking"),
    (("ndcg", "mrr", "map", "a/b", "ab test", "offline", "online", "interleaving", "relevance benchmark", "evaluation"), "evaluation experimentation offline online correlation ab testing"),
    (("lambdamart", "lambdarank", "ranknet", "learning to rank", "xgboost", "lightgbm"), "learning to rank xgboost lightgbm neural ranker"),
    (("graphrag", "graph rag", "knowledge graph", "neo4j", "entity resolution", "graph search"), "graph rag knowledge graph entity resolution"),
    (("async", "asynchronous", "disagree", "disagreement", "documentation", "docs", "decision making", "communication", "writing"), "async writing open disagreement decision making"),
    (("shipper", "founding", "founder", "0 to 1", "v1", "iteration", "feedback loop", "fast iteration"), "shipper ownership fast iteration feedback loop"),
)




@dataclass(frozen=True)
class Chunk:
    id: str
    category: str
    polarity: str
    weight: float
    bm25_query: str
    vector_query: str
    expanded_text: str
    terms: tuple[str, ...]
    do_not_duplicate_with: tuple[str, ...]


@dataclass
class CandidateRecord:
    candidate_id: str
    candidate_texts: list[str]
    retrieval_texts: list[str] = field(default_factory=list)
    bvs_strengths: list[str] = field(default_factory=list)
    bvs_penalties: list[str] = field(default_factory=list)

    @property
    def candidate_text(self) -> str:
        return " ".join(text for text in self.candidate_texts if text).strip()

    @property
    def retrieval_text(self) -> str:
        texts = self.retrieval_texts or self.candidate_texts
        return " ".join(text for text in texts if text).strip()


def recombine_offline_models(base_dir="models"):
    """Scans for model chunks and glues them back together at runtime."""
    first_chunks = glob.glob(os.path.join(base_dir, "**", "*.part000"), recursive=True)
    
    for first_chunk in first_chunks:
        target_file = first_chunk.replace(".part000", "")
        
        if not os.path.exists(target_file):
            print(f"[ranker] Offline environment detected. Recombining chunks for {os.path.basename(target_file)}...")
            
            chunk_pattern = target_file + ".part*"
            chunks = sorted(glob.glob(chunk_pattern))
            
            with open(target_file, "wb") as outfile:
                for chunk in chunks:
                    with open(chunk, "rb") as infile:
                        outfile.write(infile.read())
            print(f"[ranker] Successfully rebuilt {os.path.basename(target_file)}")

# Trigger this immediately before your script loads the SentenceTransformers
recombine_offline_models("models")
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-arr))
    if np.isscalar(values):
        return float(out)
    return out.astype(np.float32)




def stretch_upper_tail(value: float, threshold: float = 0.75, gamma: float = 0.92) -> float:
    """Monotonic upper-tail calibration that only stretches high scores."""
    x = clamp01(float(value))
    threshold = clamp01(float(threshold))
    gamma = max(1e-6, float(gamma))
    if x <= threshold:
        return x
    span = max(1e-6, 1.0 - threshold)
    scaled = (x - threshold) / span
    stretched = threshold + span * float(np.power(clamp01(scaled), gamma))
    return clamp01(stretched)

def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    if values.shape[0] == 1:
        ranks[order[0]] = 1.0
        return ranks
    positions = np.arange(values.shape[0], dtype=np.float32)
    ranks[order] = 1.0 - (positions / float(values.shape[0] - 1))
    return np.clip(ranks, 0.0, 1.0).astype(np.float32)


def stretch_bvs_percentile(rank_values: np.ndarray) -> np.ndarray:
    """Spread the BVS upper tail so near-ties stay separable."""
    ranks = np.asarray(rank_values, dtype=np.float32)
    ranks = np.clip(ranks, 0.0, 1.0)
    # Power-law tail expansion: small percentile gaps near 1.0 get magnified.
    stretched = 1.0 - np.power(np.clip(1.0 - ranks, 0.0, 1.0), 0.45)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def shape_bvs_score(raw_bvs: float) -> float:
    """Map raw structured evidence into a smooth but non-saturating [0, 1] band."""
    if math.isnan(raw_bvs) or math.isinf(raw_bvs):
        return 0.0

    # Use a softer curve so the mid-band keeps meaningful spread.
    shaped = sigmoid((float(raw_bvs) - 0.50) * 4.20)
    return clamp01(0.02 + 0.96 * shaped)

def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            if fmt == "%Y-%m":
                return parsed.replace(day=1)
            if fmt == "%Y":
                return parsed.replace(month=1, day=1)
            return parsed
        except ValueError:
            continue
    return None



def has_any_term(text_l: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text_l for term in terms)


def collect_rule_phrases(text_l: str, rules: Iterable[tuple[tuple[str, ...], str]]) -> list[str]:
    phrases: list[str] = []
    for triggers, phrase in rules:
        if has_any_term(text_l, triggers):
            phrases.append(phrase)
    return phrases


def structured_skill_bonus(text_l: str, skills: Iterable[Any]) -> tuple[float, list[str]]:
    """Small bonus layer from structured skill-like fields."""
    bonus = 0.0
    signals: list[str] = []

    skill_text_parts: list[str] = []
    for skill in skills or []:
        if skill is None:
            continue
        if isinstance(skill, str):
            skill_text_parts.append(skill)
        elif isinstance(skill, dict):
            for key in ("name", "skill", "label", "title", "value"):
                val = skill.get(key)
                if isinstance(val, str) and val.strip():
                    skill_text_parts.append(val)

    skill_text = " ".join(skill_text_parts).lower()

    def hit(terms: tuple[str, ...]) -> bool:
        return any(term in skill_text for term in terms)

    clusters = [
        (("python", "pytorch", "sklearn", "scikit-learn"), 0.02, "python_skill"),
        (("retrieval", "ranking", "search", "recommendation"), 0.03, "retrieval_skill"),
        (("embeddings", "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch"), 0.03, "vector_db_skill"),
        (("ndcg", "mrr", "map", "a/b", "ab test", "offline", "online"), 0.03, "evaluation_skill"),
        (("xgboost", "lightgbm", "lambdamart", "lambdarank", "ranknet"), 0.02, "ltr_skill"),
        (("lora", "qlora", "peft"), 0.02, "llm_finetune_skill"),
        (("graphrag", "knowledge graph", "neo4j", "entity resolution"), 0.02, "graph_retrieval_skill"),
    ]
    for terms, weight, label in clusters:
        if hit(terms):
            bonus += weight
            signals.append(label)

    return bonus, signals


def chunk_family(chunk_id: str) -> str:
    return CHUNK_FAMILIES.get(chunk_id, "UNKNOWN")


def effective_chunk_weight(chunk: Chunk) -> float:
    return float(chunk.weight) * float(BONUS_CHUNK_WEIGHT_MULTIPLIERS.get(chunk.id, 1.0))


@dataclass(frozen=True)
class ChunkSelectionPlan:
    bi_positive_pool: list[int]
    ce_positive_indices: list[int]
    ce_negative_indices: list[int]
    bi_family_coverage: dict[str, int]
    ce_family_coverage: dict[str, int]

def family_chunk_indices(chunks: list[Chunk]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {family: [] for family in POSITIVE_FAMILY_ORDER}
    for idx, chunk in enumerate(chunks):
        family = chunk_family(chunk.id)
        if chunk.polarity != "positive" or family not in mapping:
            continue
        mapping[family].append(idx)
    return mapping


def family_score_matrix(score_matrix: np.ndarray, chunks: list[Chunk]) -> dict[str, np.ndarray]:
    mapping = family_chunk_indices(chunks)
    scores: dict[str, np.ndarray] = {}
    for family, indices in mapping.items():
        if not indices:
            continue
        family_scores = np.max(score_matrix[:, indices], axis=1)
        scores[family] = np.asarray(family_scores, dtype=np.float32)
    return scores


def family_recall_overlap(selected_family_hits: dict[int, int]) -> int:
    return sum(max(0, hit_count - 1) for hit_count in selected_family_hits.values())


def select_family_recall_candidates(
    candidate_indices: list[int],
    candidate_ids: list[str],
    global_scores: np.ndarray,
    family_scores: dict[str, np.ndarray],
    family_targets: dict[str, int],
    target_total: int,
) -> tuple[list[int], dict[str, int], int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    family_hits: Counter[str] = Counter()
    candidate_memberships: Counter[int] = Counter()

    for family in FAMILY_RECALL_ORDER:
        quota = int(family_targets.get(family, 0))
        if quota <= 0:
            continue
        scores = family_scores.get(family)
        if scores is None:
            continue
        ordered = sorted(
            candidate_indices,
            key=lambda idx: (-float(scores[idx]), candidate_ids[idx]),
        )
        taken = 0
        for idx in ordered:
            candidate_memberships[idx] += 1
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            family_hits[family] += 1
            taken += 1
            if taken >= quota:
                break

    if len(selected_set) > target_total:
        trimmed = sorted(
            selected_set,
            key=lambda idx: (-float(global_scores[idx]), candidate_ids[idx]),
        )[:target_total]
        selected_set = set(trimmed)

    if len(selected_set) < target_total:
        remaining = sorted(
            [idx for idx in candidate_indices if idx not in selected_set],
            key=lambda idx: (-float(global_scores[idx]), candidate_ids[idx]),
        )
        for idx in remaining:
            selected_set.add(idx)
            if len(selected_set) >= target_total:
                break

    selected = sorted(
        selected_set,
        key=lambda idx: (-float(global_scores[idx]), candidate_ids[idx]),
    )[:target_total]

    overlap = sum(max(0, hit_count - 1) for hit_count in candidate_memberships.values())
    return selected, dict(family_hits), overlap


def choose_bvs_threshold(raw_bvs_scores: np.ndarray) -> float:
    raw_bvs_scores = np.asarray(raw_bvs_scores, dtype=np.float32)
    if raw_bvs_scores.size < 10:
        return float(BVS_MIN_THRESHOLD)
    data_driven = float(np.quantile(raw_bvs_scores, 0.10))
    return max(float(BVS_MIN_THRESHOLD), data_driven)


def order_candidates_with_bvs_tiebreak(
    final_scores: np.ndarray,
    bvs_scores: np.ndarray,
    candidate_ids: list[str],
    close_gap: float = 0.0,
) -> list[int]:
    base_order = stable_order_desc(final_scores, candidate_ids)
    if not base_order:
        return []

    if close_gap <= 0.0:
        # Keep the public ranking strictly score-monotonic so the CSV rank and
        # score columns remain aligned. BVS already contributes through the
        # final score itself, so the display order should not overrule it.
        return base_order

    ordered: list[int] = []
    i = 0
    while i < len(base_order):
        group = [base_order[i]]
        group_leader = float(final_scores[base_order[i]])
        j = i + 1
        while j < len(base_order):
            idx = base_order[j]
            if group_leader - float(final_scores[idx]) < close_gap:
                group.append(idx)
                j += 1
            else:
                break
        if len(group) > 1:
            group = sorted(group, key=lambda idx: (-float(bvs_scores[idx]), candidate_ids[idx]))
        ordered.extend(group)
        i = j
    return ordered


def chunk_score_for_candidate(row: np.ndarray, chunk_idx: int, chunks: list[Chunk], chunk_focus_weights: np.ndarray) -> float:
    return float(row[chunk_idx]) * effective_chunk_weight(chunks[chunk_idx]) * float(chunk_focus_weights[chunk_idx])


def best_scoring_index(candidate_indices: Iterable[int], row: np.ndarray, chunks: list[Chunk], chunk_focus_weights: np.ndarray) -> int | None:
    best_idx: int | None = None
    best_score = float('-inf')
    for chunk_idx in candidate_indices:
        score = chunk_score_for_candidate(row, chunk_idx, chunks, chunk_focus_weights)
        if score > best_score or (score == best_score and best_idx is not None and chunks[chunk_idx].id < chunks[best_idx].id):
            best_score = score
            best_idx = chunk_idx
    return best_idx


def select_family_covered_positive_pool(
    row: np.ndarray,
    chunks: list[Chunk],
    chunk_focus_weights: np.ndarray,
    target_count: int = POSITIVE_FAMILY_TARGET,
) -> tuple[list[int], dict[str, int]]:
    positive_indices = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "positive"]
    scores = {idx: chunk_score_for_candidate(row, idx, chunks, chunk_focus_weights) for idx in positive_indices}

    selected: list[int] = []
    selected_set: set[int] = set()

    for family in POSITIVE_FAMILY_ORDER:
        family_indices = [idx for idx in positive_indices if chunk_family(chunks[idx].id) == family]
        best_idx = best_scoring_index(family_indices, row, chunks, chunk_focus_weights)
        if best_idx is None or best_idx in selected_set:
            continue
        selected.append(best_idx)
        selected_set.add(best_idx)

    remaining = sorted(
        ((score, idx) for idx, score in scores.items() if idx not in selected_set),
        key=lambda item: (-item[0], chunks[item[1]].id),
    )
    for _, idx in remaining:
        if len(selected) >= target_count:
            break
        selected.append(idx)
        selected_set.add(idx)

    family_coverage = Counter(chunk_family(chunks[idx].id) for idx in selected)
    return selected, dict(family_coverage)


def select_family_covered_ce_chunks(
    row: np.ndarray,
    chunks: list[Chunk],
    chunk_focus_weights: np.ndarray,
) -> ChunkSelectionPlan:
    bi_positive_pool, bi_family_coverage = select_family_covered_positive_pool(row, chunks, chunk_focus_weights)
    positive_indices = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "positive"]
    positive_scores = {
        idx: chunk_score_for_candidate(row, idx, chunks, chunk_focus_weights)
        for idx in positive_indices
    }
    negative_indices = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "negative"]
    negative_scores = {
        idx: chunk_score_for_candidate(row, idx, chunks, chunk_focus_weights)
        for idx in negative_indices
    }

    selected_positive: list[int] = []
    selected_set: set[int] = set()

    def take_best(candidate_indices: Iterable[int]) -> int | None:
        filtered = [idx for idx in candidate_indices if idx not in selected_set]
        return best_scoring_index(filtered, row, chunks, chunk_focus_weights)

    required_groups = [
        ("RETRIEVAL", "retrieval"),
        ("EVALUATION", "evaluation"),
        ("SYSTEMS", "systems"),
        ("PRODUCT", "product"),
        (("DOMAIN", "CULTURE"), "domain_culture"),
    ]

    for family_spec, _label in required_groups:
        if isinstance(family_spec, tuple):
            candidate_pool = [idx for idx in bi_positive_pool if chunk_family(chunks[idx].id) in family_spec]
            if not candidate_pool:
                candidate_pool = [idx for idx in positive_indices if chunk_family(chunks[idx].id) in family_spec]
        else:
            candidate_pool = [idx for idx in bi_positive_pool if chunk_family(chunks[idx].id) == family_spec]
            if not candidate_pool:
                candidate_pool = [idx for idx in positive_indices if chunk_family(chunks[idx].id) == family_spec]

        best_idx = take_best(candidate_pool)
        if best_idx is None:
            continue
        if positive_scores.get(best_idx, 0.0) < CROSS_ENCODER_POSITIVE_FLOOR:
            continue
        selected_positive.append(best_idx)
        selected_set.add(best_idx)

    wildcard_pool = [idx for idx in bi_positive_pool if idx not in selected_set]
    if not wildcard_pool:
        wildcard_pool = [idx for idx in positive_indices if idx not in selected_set]
    best_wildcard = take_best(wildcard_pool)
    if best_wildcard is not None and positive_scores.get(best_wildcard, 0.0) >= CROSS_ENCODER_POSITIVE_FLOOR:
        selected_positive.append(best_wildcard)
        selected_set.add(best_wildcard)

    remaining_positive = sorted(
        ((score, idx) for idx, score in positive_scores.items() if idx not in selected_set),
        key=lambda item: (-item[0], chunks[item[1]].id),
    )
    for score, idx in remaining_positive:
        if len(selected_positive) >= CROSS_ENCODER_MAX_POSITIVE_CHUNKS:
            break
        if score < CROSS_ENCODER_POSITIVE_FLOOR:
            break
        selected_positive.append(idx)
        selected_set.add(idx)

    positive_filled = list(dict.fromkeys(selected_positive))[:CROSS_ENCODER_MAX_POSITIVE_CHUNKS]

    negative_sorted = sorted(
        ((score, idx) for idx, score in negative_scores.items() if score >= CROSS_ENCODER_NEGATIVE_FLOOR),
        key=lambda item: (-item[0], chunks[item[1]].id),
    )
    ce_negative = [idx for _, idx in negative_sorted[:CROSS_ENCODER_MAX_NEGATIVE_CHUNKS]]

    ce_family_coverage = Counter(chunk_family(chunks[idx].id) for idx in positive_filled)
    return ChunkSelectionPlan(
        bi_positive_pool=bi_positive_pool,
        ce_positive_indices=positive_filled,
        ce_negative_indices=ce_negative,
        bi_family_coverage=dict(bi_family_coverage),
        ce_family_coverage=dict(ce_family_coverage),
    )


def role_alignment_score(role_text: str, chunks: list[Chunk], row: np.ndarray, selected_chunk_indices: list[int]) -> float:
    role_terms = set(tokenize(role_text))
    if not role_terms:
        return 0.0
    score = 0.0
    for chunk_idx in selected_chunk_indices:
        chunk = chunks[chunk_idx]
        chunk_terms = set(chunk.terms) or set(tokenize(chunk.bm25_query)) or set(tokenize(chunk.expanded_text))
        if not chunk_terms:
            continue
        overlap = len(role_terms & chunk_terms)
        if overlap:
            score += float(row[chunk_idx]) * (overlap / max(1, len(chunk_terms)))
    return score


def build_cross_encoder_candidate_text(
    candidate: CandidateRecord,
    latest_role: str | None,
    best_matching_role: str | None,
    achievement_snippets: list[str] | None = None,
) -> str:
    texts = [text.strip() for text in candidate.candidate_texts if text and text.strip()]
    if not texts:
        return ""

    profile_text = texts[0]
    headline = ""
    summary = ""
    if profile_text.startswith("Headline: "):
        body = profile_text[len("Headline: "):]
        marker = ". Summary: "
        if marker in body:
            headline, remainder = body.split(marker, 1)
            summary = remainder.strip()
            if summary.endswith("."):
                summary = summary[:-1].strip()
            if headline.endswith("."):
                headline = headline[:-1].strip()
        else:
            headline = body.strip().rstrip(".")
    else:
        headline = profile_text.strip().rstrip(".")

    parts: list[str] = []
    if headline:
        parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")
    if latest_role:
        parts.append(f"Latest role: {latest_role.strip()}")
    if best_matching_role:
        best_clean = best_matching_role.strip()
        if best_clean and best_clean != (latest_role or "").strip():
            parts.append(f"Best matching role: {best_clean}")

    existing_text = " ".join(parts).lower()
    cleaned_snippets: list[str] = []
    seen_snippets: set[str] = set()
    for snippet in achievement_snippets or []:
        clean = " ".join((snippet or "").split()).strip()
        if not clean:
            continue
        clean_l = clean.lower()
        if clean_l in seen_snippets or clean_l in existing_text:
            continue
        seen_snippets.add(clean_l)
        cleaned_snippets.append(clean)
        if len(cleaned_snippets) >= ACHIEVEMENT_SNIPPET_MAX:
            break
    if cleaned_snippets:
        parts.append("Key achievements: " + " | ".join(cleaned_snippets))

    return " ".join(parts).strip()[:1800]


ACHIEVEMENT_IMPACT_PATTERNS: tuple[str, ...] = (
    "improved",
    "reduced",
    "increased",
    "decreased",
    "shipped",
    "deployed",
    "launched",
    "scaled",
    "optimized",
    "optimised",
    "latency",
    "throughput",
    "ndcg",
    "mrr",
    "map",
    "lift",
    "revenue",
    "users",
    "customers",
    "faster",
)

NEGATIVE_CONFIDENCE_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "wrapper",
        ("langchain", "openai", "prompt engineering", "wrapper", "tutorial", "demo", "chatbot", "agent", "agents", "tool calling"),
        ("retrieval", "ranking", "search", "production", "evaluation", "vector", "embedding", "nlp", "information retrieval"),
    ),
    (
        "research",
        ("research", "academic", "thesis", "publication", "paper", "benchmark", "lab"),
        ("production", "deployed", "shipped", "launched", "users", "live"),
    ),
    (
        "framework",
        ("framework", "frameworks", "notebook", "poc", "prototype", "tutorial", "demo"),
        ("system", "architecture", "observability", "monitoring", "production"),
    ),
    (
        "wrong_domain",
        ("computer vision", "cv", "speech", "robotics"),
        ("retrieval", "ranking", "search", "nlp", "information retrieval"),
    ),
)

NEGATIVE_CONFIDENCE_POSITIVE_RULES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("production", "deployed", "shipped", "launched", "served", "users", "customers", "live", "scale", "latency", "throughput"), 0.28),
    (("retrieval", "ranking", "search", "recommendation", "rerank", "reranking", "embedding", "embeddings", "vector", "hybrid search", "dense retrieval"), 0.26),
    (("evaluation", "ndcg", "mrr", "map", "offline", "online", "a/b", "ab test", "interleaving", "relevance benchmark"), 0.22),
    (("system", "architecture", "pipeline", "observability", "monitoring", "rollback", "incident", "latency", "throughput"), 0.18),
    (("ownership", "owned", "led", "architected", "built from scratch", "end to end", "cross functional", "mentored", "stakeholder"), 0.14),
    (("open source", "github", "talk", "talks", "blog", "blogs"), 0.08),
)

def extract_achievement_snippets(text: str, max_snippets: int = ACHIEVEMENT_SNIPPET_MAX) -> list[str]:
    sentences = SENTENCE_RE.split(text or "")
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split()).strip()
        if not clean or len(clean) < 24:
            continue
        clean_l = clean.lower()
        if any(term in clean_l for term in ("role:", "headline:", "summary:", "current role", "recent role", "earlier role")):
            # Keep achievement snippets focused on impact statements rather than structural labels.
            pass
        impact_hits = sum(1 for term in ACHIEVEMENT_IMPACT_PATTERNS if term in clean_l)
        if impact_hits <= 0:
            continue
        metric_bonus = 0.0
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|s|sec|secs|seconds|minutes|hours|days|day|weeks|month|months|million|billion|k|m)\b", clean_l):
            metric_bonus += 1.0
        if re.search(r"\b(ndcg|mrr|map|lift|latency|throughput|conversion|ctr|revenue|users|customers)\b", clean_l):
            metric_bonus += 1.0
        if re.search(r"\b(reduced|improved|increased|decreased|optimized|optimised|scaled|shipped|deployed|launched)\b", clean_l):
            metric_bonus += 0.5
        score = float(impact_hits) + metric_bonus
        key = clean_l
        if key in seen:
            continue
        seen.add(key)
        scored.append((score, idx, clean))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    snippets: list[str] = []
    total_chars = 0
    for _, _, sentence in scored:
        if len(snippets) >= max_snippets:
            break
        if total_chars + len(sentence) > ACHIEVEMENT_SNIPPET_MAX_CHARS and snippets:
            break
        snippets.append(sentence)
        total_chars += len(sentence)
    return snippets


def positive_family_coverage_bonus(row: np.ndarray, chunks: list[Chunk]) -> tuple[float, float, list[str]]:
    """Reward quality-weighted breadth across JD families without raw count bias."""
    families_hit: list[str] = []
    weighted_quality_num = 0.0
    weighted_quality_den = 0.0

    for family in POSITIVE_FAMILY_ORDER:
        indices = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "positive" and chunk_family(chunk.id) == family]
        if not indices:
            continue
        family_peak = max(float(row[idx]) for idx in indices)
        if family_peak < COVERAGE_SUPPORT_THRESHOLD:
            continue

        family_importance = max(float(effective_chunk_weight(chunks[idx])) for idx in indices)
        families_hit.append(family)
        weighted_quality_num += family_peak * family_importance
        weighted_quality_den += family_importance

    if not families_hit or weighted_quality_den <= 0.0:
        return 0.0, 0.0, []

    quality = weighted_quality_num / weighted_quality_den
    breadth = len(families_hit) / float(len(POSITIVE_FAMILY_ORDER))
    combined = 0.80 * quality + 0.20 * breadth

    # Smoothly map quality to the configured bonus range.
    bonus = COVERAGE_BONUS_MAX * float(sigmoid((combined - 0.45) * 6.0))
    return float(bonus), float(combined), families_hit


def evidence_density_bonus(row: np.ndarray, chunks: list[Chunk]) -> tuple[float, float, int]:
    """Reward consistently strong evidence rather than many mediocre hits.

    The evidence score is quality-aware and JD-weight-aware, using the strongest
    three positive chunks with fixed emphasis coefficients [0.5, 0.3, 0.2].
    """
    hits: list[tuple[float, float]] = []
    positive_total = 0

    for idx, chunk in enumerate(chunks):
        if chunk.polarity != "positive":
            continue
        score = float(row[idx])
        if score < EVIDENCE_DENSITY_THRESHOLD:
            continue
        positive_total += 1
        hits.append((score, float(effective_chunk_weight(chunk))))

    if not hits:
        return 0.0, 0.0, 0

    # Keep the strongest three evidence chunks, but rank them with a softened
    # JD-weight-aware preference so important chunks help without overpowering CE quality.
    hits.sort(key=lambda item: (item[0] * math.sqrt(max(item[1], 1e-6)), item[0]), reverse=True)
    top_hits = hits[:3]

    alpha = (0.5, 0.3, 0.2)
    weighted_num = 0.0
    weighted_den = 0.0
    for coeff, (score, weight) in zip(alpha, top_hits):
        weighted_num += coeff * weight * score
        weighted_den += coeff * weight

    combined = weighted_num / max(weighted_den, 1e-6)
    combined = clamp01(combined)

    bonus = EVIDENCE_DENSITY_BONUS_MAX * float(sigmoid((combined - 0.45) * 6.5))
    return float(bonus), float(combined), positive_total


def negative_confidence_details(text: str) -> tuple[float, dict[str, float]]:
    text = text or ""
    sentences = [s for s in (seg.strip() for seg in SENTENCE_RE.split(text)) if s]
    if not sentences:
        return 0.0, {"wrapper": 0.0, "research": 0.0, "framework": 0.0, "wrong_domain": 0.0}

    family_weights = {
        "wrapper": 1.00,
        "research": 0.95,
        "framework": 0.85,
        "wrong_domain": 0.90,
    }

    def sentence_negative_score(sentence_l: str, triggers: tuple[str, ...], blockers: tuple[str, ...]) -> float:
        trig = sum(1 for term in triggers if term in sentence_l)
        if trig <= 0:
            return 0.0

        block = sum(1 for term in blockers if term in sentence_l)
        score = 0.28 + 0.18 * min(trig, 4) - 0.10 * min(block, 3)

        if any(term in sentence_l for term in ("only", "solely", "just", "primarily", "mostly")):
            score += 0.12
        if trig >= 2:
            score += 0.10
        if any(term in sentence_l for term in ("production", "deployed", "shipped", "launched", "users", "live")):
            score -= 0.14

        return clamp01(score)

    def sentence_positive_score(sentence_l: str) -> float:
        score = 0.0
        for triggers, weight in NEGATIVE_CONFIDENCE_POSITIVE_RULES:
            if any(term in sentence_l for term in triggers):
                score += weight

        if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|s|sec|secs|seconds|minutes|hours|days|day|weeks|month|months|million|billion|k|m)\b", sentence_l):
            score += 0.05
        return clamp01(score)

    family_scores: dict[str, float] = {}
    positive_strength = 0.0

    for sentence in sentences:
        sentence_l = " ".join(sentence.lower().split())
        positive_strength += sentence_positive_score(sentence_l)

    positive_strength = min(2.0, positive_strength)

    for label, triggers, blockers in NEGATIVE_CONFIDENCE_RULES:
        family_scores[label] = max(
            (sentence_negative_score(" ".join(sentence.lower().split()), triggers, blockers) for sentence in sentences),
            default=0.0,
        )

    if family_scores:
        weighted_num = 0.0
        weighted_den = 0.0
        ordered_family_scores = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        for label, score in ordered_family_scores:
            if score <= 0.0:
                continue
            weight = family_weights.get(label, 1.0)
            weighted_num += weight * score
            weighted_den += weight

        negative_strength = weighted_num / max(weighted_den, 1e-6)
        support_count = sum(1 for _, value in ordered_family_scores if value >= 0.35)
        if support_count >= 2:
            negative_strength *= 1.0 + 0.04 * min(3, support_count - 1)
    else:
        negative_strength = 0.0

    negative_confidence = negative_strength / (negative_strength + positive_strength + 0.20)
    return float(clamp01(negative_confidence)), family_scores


def negative_confidence_penalty(negative_confidence: float) -> float:
    negative_confidence = clamp01(negative_confidence)
    if negative_confidence <= NEGATIVE_CONFIDENCE_RELEASE:
        return 1.0
    return math.exp(-NEGATIVE_CONFIDENCE_PENALTY_SCALE * (negative_confidence - NEGATIVE_CONFIDENCE_RELEASE))

def minmax_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    clean = np.where(finite, values, 0.0)
    vmin = float(clean.min())
    vmax = float(clean.max())
    if vmax <= vmin:
        return np.zeros_like(clean, dtype=np.float32)
    return ((clean - vmin) / (vmax - vmin)).astype(np.float32)


def minmax_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    for col in range(matrix.shape[1]):
        out[:, col] = minmax_vector(matrix[:, col])
    return out


def stable_order_desc(scores: np.ndarray, candidate_ids: list[str]) -> list[int]:
    return sorted(range(len(candidate_ids)), key=lambda i: (-float(scores[i]), candidate_ids[i]))


def safe_model_dir_name(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("_")
    return safe or "model"


def configure_repo_model_cache(model_cache_dir: Path) -> Path:
    resolved = model_cache_dir.resolve()
    hf_home = resolved / "huggingface"
    transformers_cache = hf_home / "transformers"
    st_home = resolved / "sentence_transformers"
    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    st_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_home)
    return resolved


def repo_model_path(model_name: str, model_cache_dir: Path) -> Path:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path
    return model_cache_dir / safe_model_dir_name(model_name)


def load_sentence_transformer_model(model_name: str, model_cache_dir: Path) -> tuple[Any, Path]:
    model_cache_dir = configure_repo_model_cache(model_cache_dir)
    local_path = repo_model_path(model_name, model_cache_dir)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("SentenceTransformer is required for vector scoring") from exc

    try:
        if local_path.exists():
            return SentenceTransformer(str(local_path)), local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name, cache_folder=str(model_cache_dir / "sentence_transformers"))
        model.save(str(local_path))
        return model, local_path
    except Exception as exc:
        raise RuntimeError(f"Failed to load or store SentenceTransformer model {model_name!r}") from exc


def load_cross_encoder_model(model_name: str, model_cache_dir: Path) -> tuple[Any, Path]:
    model_cache_dir = configure_repo_model_cache(model_cache_dir)
    local_path = repo_model_path(model_name, model_cache_dir)

    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("CrossEncoder is required for reranking") from exc

    try:
        if local_path.exists():
            return CrossEncoder(str(local_path)), local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        model = CrossEncoder(model_name)
        model.save(str(local_path))
        return model, local_path
    except Exception as exc:
        raise RuntimeError(f"Failed to load or store CrossEncoder model {model_name!r}") from exc


def jd_embedding_metadata(chunks: list[Chunk], embedding_model: str) -> dict[str, Any]:
    return {
        "version": JD_EMBEDDING_CACHE_VERSION,
        "embedding_model": embedding_model,
        "chunk_ids": [chunk.id for chunk in chunks],
        "vector_queries": [chunk.vector_query for chunk in chunks],
        "normalize_embeddings": True,
    }


def load_cached_jd_embeddings(cache_path: Path, chunks: list[Chunk], embedding_model: str) -> np.ndarray | None:
    if not cache_path.exists():
        return None

    expected = jd_embedding_metadata(chunks, embedding_model)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    except Exception:
        return None

    if metadata != expected:
        return None
    if embeddings.shape[0] != len(chunks):
        return None
    return embeddings


def write_cached_jd_embeddings(
    cache_path: Path,
    chunks: list[Chunk],
    embedding_model: str,
    embeddings: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = jd_embedding_metadata(chunks, embedding_model)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def candidate_text_cache_key(candidate: CandidateRecord) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "candidate_texts": candidate.candidate_texts,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_retrieval_cache_key(candidate: CandidateRecord) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "retrieval_texts": candidate.retrieval_texts or candidate.candidate_texts,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_funnel_cache_key(candidate: CandidateRecord) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "candidate_texts": candidate.candidate_texts,
        "retrieval_texts": candidate.retrieval_texts or candidate.candidate_texts,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vector_score_metadata(
    chunks: list[Chunk],
    embedding_model: str,
    candidate_keys: list[str],
) -> dict[str, Any]:
    return {
        "version": VECTOR_SCORE_CACHE_VERSION,
        "embedding_model": embedding_model,
        "chunk_ids": [chunk.id for chunk in chunks],
        "vector_queries": [chunk.vector_query for chunk in chunks],
        "candidate_keys": candidate_keys,
        "score_normalization": "minmax_columns",
        "best_role_selection": "max_role_vector_similarity",
    }


def load_cached_vector_scores(
    cache_path: Path,
    chunks: list[Chunk],
    embedding_model: str,
    candidate_keys: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None

    expected = vector_score_metadata(chunks, embedding_model, candidate_keys)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            scores = np.asarray(data["scores"], dtype=np.float32)
            best_role_indices = np.asarray(data["best_role_indices"], dtype=np.int32)
    except Exception:
        return None

    if metadata != expected:
        return None
    if scores.shape != (len(candidate_keys), len(chunks)):
        return None
    if best_role_indices.shape != (len(candidate_keys),):
        return None
    return scores, best_role_indices


def write_cached_vector_scores(
    cache_path: Path,
    chunks: list[Chunk],
    embedding_model: str,
    candidate_keys: list[str],
    scores: np.ndarray,
    best_role_indices: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = vector_score_metadata(chunks, embedding_model, candidate_keys)
    np.savez_compressed(
        cache_path,
        scores=np.asarray(scores, dtype=np.float32),
        best_role_indices=np.asarray(best_role_indices, dtype=np.int32),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def cross_score_metadata(
    cross_encoder_model: str,
    candidate_keys: list[str],
    chunk_ids: list[str],
) -> dict[str, Any]:
    return {
        "version": CROSS_SCORE_CACHE_VERSION,
        "cross_encoder_model": cross_encoder_model,
        "candidate_keys": candidate_keys,
        "chunk_ids": chunk_ids,
        "score_transform": "sigmoid_then_epsilon_clip",
    }


def load_cached_cross_scores(
    cache_path: Path,
    cross_encoder_model: str,
    candidate_keys: list[str],
    chunk_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache_path.exists(): return None
    expected = cross_score_metadata(cross_encoder_model, candidate_keys, chunk_ids)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            scores = np.asarray(data["scores"], dtype=np.float32)
            neg_scores = np.asarray(data["neg_scores"], dtype=np.float32)
            adjusted = np.asarray(data["adjusted"], dtype=np.float32)
    except Exception: return None
    if metadata != expected or scores.shape != (len(candidate_keys),): return None
    if adjusted.shape != (len(candidate_keys), len(chunk_ids)): return None
    return scores, neg_scores, adjusted

def write_cached_cross_scores(
    cache_path: Path,
    cross_encoder_model: str,
    candidate_keys: list[str],
    chunk_ids: list[str],
    scores: np.ndarray,
    neg_scores: np.ndarray,
    adjusted: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cross_score_metadata(cross_encoder_model, candidate_keys, chunk_ids)
    np.savez_compressed(
        cache_path,
        scores=np.asarray(scores, dtype=np.float32),
        neg_scores=np.asarray(neg_scores, dtype=np.float32),
        adjusted=np.asarray(adjusted, dtype=np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def load_jd_index(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("doc_name") != "redrob_jd_hybrid_index":
        raise ValueError(f"{path} is not the expected Redrob JD index")

    chunks: list[Chunk] = []
    required = {
        "id",
        "category",
        "polarity",
        "weight",
        "bm25_query",
        "vector_query",
        "expanded_text",
        "terms",
        "do_not_duplicate_with",
        "source_type",
    }
    for raw in data.get("chunks", []):
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Chunk {raw.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        if raw["source_type"] != "jd_text_only":
            raise ValueError(f"Chunk {raw['id']} has unexpected source_type={raw['source_type']!r}")
        weight = as_float(raw["weight"])
        if weight <= 0.0:
            continue
        chunks.append(
            Chunk(
                id=raw["id"],
                category=raw["category"],
                polarity=raw["polarity"],
                weight=weight,
                bm25_query=raw["bm25_query"],
                vector_query=raw["vector_query"],
                expanded_text=raw["expanded_text"],
                terms=tuple(raw.get("terms") or ()),
                do_not_duplicate_with=tuple(raw.get("do_not_duplicate_with") or ()),
            )
        )

    expected = [f"C{i:02d}" for i in range(1, 25)]
    found = [chunk.id for chunk in chunks]
    if found != expected:
        raise ValueError(f"Expected scoreable chunks C01-C24, found {found}")
    return chunks


def iter_candidate_objects(path: Path) -> Iterable[dict[str, Any]]:
    """Yield candidate objects from JSON, JSONL, or gzipped JSON/JSONL.

    The hackathon sample tooling sometimes renames pretty-printed JSON to *.jsonl.
    We first try to parse the whole file as JSON, then fall back to line-delimited
    parsing so both formats are accepted safely.
    """
    suffix = path.suffix.lower()
    is_gz = path.name.lower().endswith(".gz")

    if is_gz:
        import gzip as _gzip  # local import keeps the dependency optional
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        with path.open("r", encoding="utf-8") as f:
            raw_text = f.read()

    raw_text = raw_text.strip()
    if not raw_text:
        return

    def yield_from_parsed(data: Any) -> Iterable[dict[str, Any]]:
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and isinstance(data.get("candidates"), list):
            yield from data["candidates"]
        elif isinstance(data, dict) and data.get("candidate_id"):
            yield data
        else:
            raise ValueError(f"Unsupported candidate JSON structure in {path}")

    # First try a single JSON object / array. This handles pretty-printed JSON
    # that may have been saved with a .jsonl extension by the sandbox UI.
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        yield from yield_from_parsed(parsed)
        return

    # Fall back to strict JSONL parsing.
    for line_no, line in enumerate(raw_text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def build_candidate_text(candidate: dict[str, Any], as_of_date: date | None = None) -> list[str]:
    profile = candidate.get("profile") or {}

    chunks: list[str] = []
    headline = str(profile.get("headline") or "").strip()
    summary = str(profile.get("summary") or "").strip()
    profile_chunk = " ".join(
        part
        for part in [
            f"Headline: {headline}." if headline else "",
            f"Summary: {summary}." if summary else "",
        ]
        if part
    ).strip()
    if profile_chunk:
        chunks.append(profile_chunk)

    history = candidate.get("career_history") or []
    indexed_history = list(enumerate(history))

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[date, int]:
        idx, role = item
        start = parse_date(role.get("start_date")) or date.min
        return start, idx

    sorted_history = sorted(indexed_history, key=sort_key)
    total_roles = len(sorted_history)

    def recency_label(sorted_pos: int) -> str:
        if total_roles <= 1 or sorted_pos == total_roles - 1:
            return "Current role. Most recent evidence. Production impact."
        if sorted_pos >= max(0, total_roles - 3):
            return "Recent role. Fresh evidence. Production impact."
        return "Earlier role. Historical context."

    for sorted_pos, (_, role) in enumerate(sorted_history):
        description = str(role.get("description") or "").strip()
        if not description:
            continue

        desc_l = " ".join(description.lower().split())
        role_tags = collect_rule_phrases(desc_l, ROLE_SIGNAL_RULES)
        prefix_bits = [recency_label(sorted_pos)]
        if as_of_date is not None and sorted_pos == total_roles - 1:
            prefix_bits.append("current role confirmation")
        if role_tags:
            prefix_bits.append("signals: " + " ; ".join(role_tags))

        role_chunk = ". ".join(prefix_bits) + ". " + description
        chunks.append(role_chunk.strip())

    return chunks


def build_candidate_enrichment_texts(narrative_text: str) -> list[str]:
    text_l = " ".join((narrative_text or "").lower().split())
    token_set = set(tokenize(text_l))
    enrichments: list[str] = []

    def term_matches(term: str) -> bool:
        term_l = " ".join(term.lower().split())
        if not term_l:
            return False
        if " " in term_l or "/" in term_l or "-" in term_l or "." in term_l:
            return term_l in text_l
        return term_l in token_set

    def filtered_phrase_tokens(phrase: str) -> list[str]:
        tokens = tokenize(phrase)
        filtered: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in token_set or token in seen:
                continue
            seen.add(token)
            filtered.append(token)
        return filtered

    def add_phrase(phrase: str) -> None:
        phrase = " ".join(phrase.split()).strip().lower()
        if not phrase:
            return
        if phrase in enrichments:
            return
        filtered_tokens = filtered_phrase_tokens(phrase)
        if not filtered_tokens:
            return
        enrichments.append(" ".join(filtered_tokens))

    for triggers, phrase in POSITIVE_TAG_RULES:
        if any(term_matches(term) for term in triggers):
            add_phrase(phrase)

    for triggers, blockers, phrase in NEGATIVE_TAG_RULES:
        if any(term_matches(term) for term in triggers) and not any(term_matches(blocker) for blocker in blockers):
            add_phrase(phrase)

    if not enrichments:
        return []

    positive_bits: list[str] = []
    negative_bits: list[str] = []
    for phrase in enrichments:
        if phrase.startswith(("research only", "wrapper only", "computer vision only", "framework only", "closed source only")):
            negative_bits.append(phrase)
        else:
            positive_bits.append(phrase)

    out: list[str] = []
    if positive_bits:
        out.append("candidate taxonomy tags: " + " ; ".join(positive_bits))
    if negative_bits:
        out.append("candidate anti-signal tags: " + " ; ".join(negative_bits))
    return out


def evaluate_l0_hard_drop_reasons(
    profile: dict[str, Any],
    signals: dict[str, Any],
    as_of_date: date,
) -> list[str]:
    """Return only direct-signal hard-drop reasons.

    Hard elimination is intentionally conservative and uses only structured
    candidate-side signals, not narrative semantics or keyword matching.
    """
    reasons: list[str] = []

    country = str(profile.get("country") or "").strip().lower()
    willing_relocate = bool(signals.get("willing_to_relocate"))
    open_to_work = bool(signals.get("open_to_work_flag"))
    last_active = parse_date(signals.get("last_active_date"))
    recruiter_response_rate = as_float(signals.get("recruiter_response_rate"), 0.0)
    notice_days = as_int(signals.get("notice_period_days"), 0)

    # Outside India + no relocation is a hard logistics mismatch here.
    if country and "india" not in country and not willing_relocate:
        reasons.append("outside_india_and_no_relocation")

    # Extremely long notice periods are effectively non-starters.
    if notice_days > 180:
        reasons.append("extreme_notice_period")

    # Truly stale + unresponsive + not open to work profiles are dropped.
    if last_active is not None:
        days_inactive = max(0, (as_of_date - last_active).days)
        if days_inactive >= 540 and recruiter_response_rate <= 0.05 and not open_to_work:
            reasons.append("stale_and_unresponsive")

    return reasons


def apply_l0_triage(
    candidates: list[dict[str, Any]],
    as_of_date: date,
) -> tuple[list[CandidateRecord], np.ndarray, list[dict[str, str]], float]:
    surviving: list[CandidateRecord] = []
    bvs_scores: list[float] = []
    discarded: list[dict[str, str]] = []
    staged: list[dict[str, Any]] = []

    def text_has_any(text: str, terms: Iterable[str]) -> bool:
        text_l = (text or "").lower()
        return any(term in text_l for term in terms)

    for candidate in candidates:
        cand_id = str(candidate.get("candidate_id") or "").strip()
        profile = candidate.get("profile") or {}
        signals = candidate.get("redrob_signals") or {}
        history = candidate.get("career_history") or []
        skills = candidate.get("skills") or []

        current_title = str(profile.get("current_title") or "").lower()
        current_industry = str(profile.get("current_industry") or "").lower()
        country = str(profile.get("country") or "").lower()
        location = str(profile.get("location") or "").lower()
        willing_relocate = bool(signals.get("willing_to_relocate"))
        yoe = as_float(profile.get("years_of_experience"), 0.0)
        last_active = parse_date(signals.get("last_active_date"))
        signup_date = parse_date(signals.get("signup_date"))
        recruiter_response_rate = as_float(signals.get("recruiter_response_rate"), 0.0)
        avg_response_hours = as_float(signals.get("avg_response_time_hours"), 0.0)
        notice_days = as_int(signals.get("notice_period_days"), 0)
        profile_completeness = as_float(signals.get("profile_completeness_score"), 0.0)

        hard_drop_reasons = evaluate_l0_hard_drop_reasons(profile, signals, as_of_date)
        if not hard_drop_reasons and profile_completeness <= 5.0 and not history and not skills:
            hard_drop_reasons.append("invalid_profile_quality")
        if hard_drop_reasons:
            discarded.append({"candidate_id": cand_id, "reason": "; ".join(hard_drop_reasons)})
            continue

        raw_bvs = 0.40
        strengths: list[str] = []
        penalties: list[str] = []

        # Tier constants for a symmetric, readable BVS framework.
        TIER3_EXCEPTIONAL_GREEN = 0.18   # strongest positive signals
        TIER3_STRONG_GREEN = 0.15        # very strong positives
        TIER2_GREEN = 0.08               # solid positives
        TIER1_GREEN = 0.03               # mild positives
        TIER2_MAJOR_WARNING = -0.18      # major warnings
        TIER1_FATAL_RED = -0.35          # severe soft penalty, not hard drop
        TIER1_SMALL_WARNING = -0.08      # mild-to-moderate warnings

        def add_strength(delta: float, reason: str) -> None:
            nonlocal raw_bvs
            raw_bvs += delta
            strengths.append(reason)

        def add_penalty(delta: float, reason: str) -> None:
            nonlocal raw_bvs
            raw_bvs += delta
            penalties.append(reason)

        # Geography / relocation fit
        if "india" not in country:
            if not willing_relocate and country:
                add_penalty(TIER2_MAJOR_WARNING, "geography_mismatch")
        else:
            if any(city in location for city in PRIMARY_CITIES):
                add_strength(TIER3_EXCEPTIONAL_GREEN, f"location={location or country}")
            elif any(city in location for city in WELCOME_CITIES):
                add_strength(TIER2_GREEN, f"location={location or country}")
            elif willing_relocate:
                add_strength(TIER2_GREEN, "willing_to_relocate")

        # Experience band
        if 6.0 <= yoe <= 8.0:
            add_strength(TIER3_EXCEPTIONAL_GREEN, f"target_yoe={yoe:.1f}")
        elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
            add_strength(TIER2_GREEN, f"acceptable_yoe={yoe:.1f}")
        elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
            add_strength(TIER1_GREEN, f"near_band_yoe={yoe:.1f}")
        elif yoe < 3.0 or yoe > 12.0:
            add_penalty(TIER1_FATAL_RED, f"outside_jd_band={yoe:.1f}")
        else:
            add_penalty(TIER2_MAJOR_WARNING, f"outside_jd_band={yoe:.1f}")

        # Availability / activity
        if signals.get("open_to_work_flag"):
            add_strength(TIER3_STRONG_GREEN, "open_to_work")

        if signup_date:
            signup_days = max(0, (as_of_date - signup_date).days)
            if signup_days <= 365:
                add_strength(TIER1_GREEN, "recent_platform_signup")
            elif signup_days <= 3650:
                add_strength(0.02, "established_platform_history")

        if last_active:
            days_inactive = max(0, (as_of_date - last_active).days)
            if days_inactive <= 14:
                add_strength(TIER3_STRONG_GREEN, "very_recent_activity")
            elif days_inactive <= 30:
                add_strength(TIER2_GREEN, "recent_activity")
            elif days_inactive <= 90:
                add_strength(TIER1_GREEN, "moderate_activity")
            elif days_inactive <= 180:
                add_penalty(TIER1_SMALL_WARNING, "aging_activity")
            else:
                add_penalty(TIER2_MAJOR_WARNING, "stale_activity")

        if recruiter_response_rate >= 0.80:
            add_strength(TIER3_EXCEPTIONAL_GREEN, f"recruiter_response_rate={recruiter_response_rate:.2f}")
        elif recruiter_response_rate >= 0.60:
            add_strength(TIER2_GREEN, f"recruiter_response_rate={recruiter_response_rate:.2f}")
        elif recruiter_response_rate < 0.20:
            add_penalty(TIER2_MAJOR_WARNING, f"low_recruiter_response_rate={recruiter_response_rate:.2f}")

        if avg_response_hours <= 24:
            add_strength(TIER3_EXCEPTIONAL_GREEN, f"avg_response_time={avg_response_hours:.1f}h")
        elif avg_response_hours <= 72:
            add_strength(TIER2_GREEN, f"avg_response_time={avg_response_hours:.1f}h")
        elif avg_response_hours > 168:
            add_penalty(TIER2_MAJOR_WARNING, f"slow_response={avg_response_hours:.1f}h")

        if notice_days <= 15 and notice_days > 0:
            add_strength(TIER1_GREEN, f"notice_period={notice_days}d")
        elif notice_days <= 30 and notice_days > 15:
            add_strength(TIER1_GREEN, f"notice_period={notice_days}d")
        elif 30 < notice_days <= 60:
            add_penalty(TIER1_SMALL_WARNING, f"notice_period={notice_days}d")
        elif 60 < notice_days <= 90:
            add_penalty(TIER2_MAJOR_WARNING, f"notice_period={notice_days}d")
        elif notice_days > 90:
            add_penalty(TIER2_MAJOR_WARNING, f"notice_period={notice_days}d")

        if as_int(signals.get("applications_submitted_30d"), 0) > 3:
            add_strength(TIER1_GREEN, "active_applications")

        views_30d = as_int(signals.get("profile_views_received_30d"), 0)
        search_appearance_30d = as_int(signals.get("search_appearance_30d"), 0)
        saved_by_recruiters_30d = as_int(signals.get("saved_by_recruiters_30d"), 0)
        connections = as_int(signals.get("connection_count"), 0)
        endorsements = as_int(signals.get("endorsements_received"), 0)

        if views_30d > 50 or search_appearance_30d > 50 or saved_by_recruiters_30d > 2 or connections > 100:
            add_strength(TIER1_GREEN, "market_interest")
            if views_30d > 50:
                strengths.append(f"profile_views_30d={views_30d}")
            if search_appearance_30d > 50:
                strengths.append(f"search_appearance_30d={search_appearance_30d}")
            if saved_by_recruiters_30d > 2:
                strengths.append(f"saved_by_recruiters_30d={saved_by_recruiters_30d}")
            if connections > 100:
                strengths.append(f"connection_count={connections}")

        if endorsements > 10:
            add_strength(TIER1_GREEN, f"endorsements_received={endorsements}")

        offer_rate = as_float(signals.get("offer_acceptance_rate"), -1.0)
        if offer_rate > 0.60:
            add_strength(TIER2_GREEN, f"offer_acceptance_rate={offer_rate:.2f}")

        interview_rate = as_float(signals.get("interview_completion_rate"), -1.0)
        if interview_rate >= 0.85:
            add_strength(TIER3_EXCEPTIONAL_GREEN, f"interview_completion_rate={interview_rate:.2f}")
        elif interview_rate > 0.70:
            add_strength(TIER2_GREEN, f"interview_completion_rate={interview_rate:.2f}")
        elif 0.0 <= interview_rate < 0.45:
            add_penalty(TIER2_MAJOR_WARNING, f"low_interview_completion_rate={interview_rate:.2f}")

        if profile_completeness > 80:
            add_strength(TIER1_GREEN, f"profile_completeness={profile_completeness:.1f}")

        if bool(signals.get("verified_email")) and bool(signals.get("verified_phone")) and bool(signals.get("linkedin_connected")):
            add_strength(TIER1_GREEN, "all_verifications_complete")

        assessments = signals.get("skill_assessment_scores") or {}
        relevant_assessment_scores: list[float] = []
        fallback_assessment_scores: list[float] = []
        for skill_name, raw_score in assessments.items():
            score = clamp01(as_float(raw_score, 0.0) / 100.0)
            fallback_assessment_scores.append(score)
            skill_l = str(skill_name).lower()
            if any(term in skill_l for term in RELEVANT_SKILL_TERMS):
                relevant_assessment_scores.append(score)

        if relevant_assessment_scores:
            avg_relevant = sum(relevant_assessment_scores) / len(relevant_assessment_scores)
        elif fallback_assessment_scores:
            avg_relevant = sum(fallback_assessment_scores) / len(fallback_assessment_scores)
        else:
            avg_relevant = 0.35

        if avg_relevant >= 0.85:
            add_strength(TIER3_EXCEPTIONAL_GREEN, f"high_assessment_avg={avg_relevant:.2f}")
        elif avg_relevant >= 0.65:
            add_strength(TIER2_GREEN, f"assessment_avg={avg_relevant:.2f}")
        elif avg_relevant < 0.45:
            add_penalty(TIER1_FATAL_RED, f"low_assessment_avg={avg_relevant:.2f}")

        github_score = as_float(signals.get("github_activity_score"), 0.0)
        if github_score > 50:
            add_strength(TIER2_GREEN, f"high_github_activity={github_score:.1f}")
        elif github_score < 0:
            add_penalty(TIER1_SMALL_WARNING, "missing_github_activity")

        role_durations = [
            as_int(role.get("duration_months"), 0)
            for role in history
            if as_int(role.get("duration_months"), 0) > 0
        ]
        sorted_history = sorted(
            enumerate(history),
            key=lambda item: (parse_date(item[1].get("start_date")) or date.min, item[0]),
        )

        if history and any(str(role.get("company_size") or "") in ("11-50", "51-200") for role in history):
            add_strength(TIER1_GREEN, "startup_exposure")

        if history and all(str(role.get("company_size") or "") == "10001+" for role in history):
            add_penalty(TIER1_SMALL_WARNING, "big_tech_only")

        if len(role_durations) >= 3:
            median_duration = float(np.median(np.asarray(role_durations, dtype=np.float32)))
            if median_duration < 12:
                add_penalty(TIER2_MAJOR_WARNING, f"job_hopping_median={median_duration:.1f}mo")
            elif median_duration < 18:
                add_penalty(TIER1_SMALL_WARNING, f"job_hopping_median={median_duration:.1f}mo")

        if len(sorted_history) >= 2:
            recent_two = [
                as_int(sorted_history[-1][1].get("duration_months"), 0),
                as_int(sorted_history[-2][1].get("duration_months"), 0),
            ]
            if all(duration > 0 for duration in recent_two):
                recent_avg = sum(recent_two) / 2.0
                if recent_avg < 12:
                    add_penalty(TIER2_MAJOR_WARNING, f"job_hopping_recent_avg={recent_avg:.1f}mo")
                elif recent_avg < 18:
                    add_penalty(TIER1_SMALL_WARNING, f"job_hopping_recent_avg={recent_avg:.1f}mo")

        if current_industry and current_industry not in ["it services", "consulting"]:
            add_strength(TIER1_GREEN, f"product_industry={current_industry}")
            if any(domain in current_industry for domain in ["hr", "recruiting", "marketplace", "talent"]):
                add_strength(TIER2_GREEN, "hr_marketplace_domain_expert")

        title_mismatch_terms = ("marketing", "sales", "hr", "recruiter", "people ops", "ops")
        tech_title_terms = ("engineer", "scientist", "ml", "ai", "search", "ranking", "retrieval", "nlp", "data")
        if current_title and any(term in current_title for term in title_mismatch_terms) and not any(term in current_title for term in tech_title_terms):
            add_penalty(TIER2_MAJOR_WARNING, "title_skill_mismatch")

        if 5.0 <= yoe <= 9.0:
            add_strength(TIER1_GREEN, "experience_in_band")

        # Queue candidate for the second, text-aware pass only if the cheap structured
        # signals are promising enough.
        staged.append(
            {
                "candidate": candidate,
                "candidate_id": cand_id,
                "raw_bvs": raw_bvs,
                "strengths": strengths,
                "penalties": penalties,
                "history": history,
                "skills": skills,
            }
        )

    if not staged:
        return [], np.zeros((0,), dtype=np.float32), discarded, BVS_MIN_THRESHOLD

    threshold = choose_bvs_threshold(np.asarray([item["raw_bvs"] for item in staged], dtype=np.float32))

    for item in staged:
        cand_id = item["candidate_id"]
        candidate = item["candidate"]
        raw_bvs = float(item["raw_bvs"])
        strengths = list(item["strengths"])
        penalties = list(item["penalties"])
        history = item["history"]
        skills = item["skills"]

        if raw_bvs < threshold:
            discarded.append({"candidate_id": cand_id, "reason": f"below_bvs_threshold={raw_bvs:.3f}"})
            continue

        narrative_chunks = build_candidate_text(candidate, as_of_date)
        narrative_text = " ".join(narrative_chunks).lower()
        if narrative_text:
            research_terms = ("research", "academic", "thesis", "publication", "paper", "benchmark", "lab")
            production_terms = ("production", "deployed", "ship", "shipped", "launched", "users", "customer", "live")
            wrapper_terms = ("langchain", "openai", "prompt engineering", "wrapper", "tutorial", "demo")
            retrieval_terms = ("retrieval", "ranking", "search", "embedding", "vector", "recommendation", "ir", "nlp")
            wrong_domain_terms = ("computer vision", "cv", "speech", "robotics")
            systems_terms = ("system", "architecture", "observability", "pipeline", "monitoring", "latency", "throughput")

            if text_has_any(narrative_text, production_terms):
                raw_bvs += 0.03
                strengths.append("production_evidence")
            if text_has_any(narrative_text, research_terms) and not text_has_any(narrative_text, production_terms):
                raw_bvs -= 0.18
                penalties.append("research_only_like")
            if text_has_any(narrative_text, wrapper_terms) and not text_has_any(narrative_text, retrieval_terms):
                raw_bvs -= 0.20
                penalties.append("wrapper_only_like")
            if text_has_any(narrative_text, wrong_domain_terms) and not text_has_any(narrative_text, retrieval_terms):
                raw_bvs -= 0.18
                penalties.append("wrong_domain_like")
            if text_has_any(narrative_text, ("framework", "frameworks", "notebook", "poc", "prototype", "tutorial", "demo")) and not text_has_any(narrative_text, systems_terms):
                raw_bvs -= 0.18
                penalties.append("framework_demo_only_like")

            if text_has_any(narrative_text, ("owned", "led", "architected", "built from scratch", "end to end", "cross functional")):
                raw_bvs += 0.08
                strengths.append("narrative_ownership")
        
        final_bvs = shape_bvs_score(clamp01(raw_bvs))
        if final_bvs < threshold:
            discarded.append({"candidate_id": cand_id, "reason": f"below_bvs_threshold={final_bvs:.3f}"})
            continue

        surviving.append(
            CandidateRecord(
                candidate_id=cand_id,
                candidate_texts=narrative_chunks,
                retrieval_texts=narrative_chunks,
                bvs_strengths=strengths[:8],
                bvs_penalties=penalties[:8],
            )
        )
        bvs_scores.append(final_bvs)

    return surviving, np.asarray(bvs_scores, dtype=np.float32), discarded, threshold

def load_candidates(
    path: Path,
    as_of: date,
    max_candidates: int | None,
) -> tuple[list[CandidateRecord], np.ndarray, list[dict[str, str]], float]:
    raw_candidates: list[dict[str, Any]] = []
    for idx, candidate in enumerate(iter_candidate_objects(path), 1):
        if max_candidates is not None and len(raw_candidates) >= max_candidates:
            break
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError(f"Candidate at input row {idx} has no candidate_id")
        raw_candidates.append(candidate)

    if not raw_candidates:
        raise ValueError(f"No candidates loaded from {path}")

    return apply_l0_triage(raw_candidates, as_of)


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.30):
        self.k1 = k1
        self.b = b
        self.doc_count = len(texts)
        self.doc_lengths = np.zeros(self.doc_count, dtype=np.float32)
        self.avg_doc_len = 1.0
        self.df: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._build(texts)

    def _build(self, texts: list[str]) -> None:
        total_len = 0
        df_counter: Counter[str] = Counter()
        doc_term_counts: list[Counter[str]] = []
        for idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_term_counts.append(counts)
            length = sum(counts.values())
            self.doc_lengths[idx] = length
            total_len += length
            df_counter.update(counts.keys())

        self.avg_doc_len = max(1.0, total_len / max(1, self.doc_count))
        self.df = dict(df_counter)
        for idx, counts in enumerate(doc_term_counts):
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df <= 0:
            return 0.0
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float32)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len)
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

    def score_queries(self, queries: list[str]) -> np.ndarray:
        matrix = np.zeros((self.doc_count, len(queries)), dtype=np.float32)
        for col, query in enumerate(queries):
            matrix[:, col] = self.score_query(query)
        return minmax_columns(matrix)


class SentenceTransformerVectorBackend:
    name = "sentence_transformers"

    def __init__(self, model_name: str, model_cache_dir: Path):
        self.model, self.model_path = load_sentence_transformer_model(model_name, model_cache_dir)
        self.name = f"sentence_transformers:{self.model_path}"

    def encode_queries(self, queries: list[str], batch_size: int) -> np.ndarray:
        try:
            query_emb = self.model.encode(
                queries,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError("Failed during JD query vector encoding") from exc
        return np.asarray(query_emb, dtype=np.float32)

    def score_queries(
        self,
        candidate_texts: list[list[str]],
        query_emb: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_lengths = np.asarray([len(texts) for texts in candidate_texts], dtype=np.int32)
        flat_texts = [text for texts in candidate_texts for text in texts if text]
        if not flat_texts:
            empty_scores = np.zeros((len(candidate_texts), query_emb.shape[0]), dtype=np.float32)
            empty_sims = np.zeros((0, query_emb.shape[0]), dtype=np.float32)
            return empty_scores, empty_sims, candidate_lengths

        try:
            doc_emb = self.model.encode(
                flat_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed during vector encoding") from exc

        doc_emb = np.asarray(doc_emb, dtype=np.float32)
        query_emb = np.asarray(query_emb, dtype=np.float32)
        sims = doc_emb @ query_emb.T
        out = np.full((len(candidate_texts), query_emb.shape[0]), -np.inf, dtype=np.float32)
        offset = 0
        for cand_idx, length in enumerate(candidate_lengths):
            if length <= 0:
                continue
            cand_sims = sims[offset : offset + length]
            out[cand_idx] = cand_sims.max(axis=0)
            offset += int(length)
        out[~np.isfinite(out)] = 0.0
        return minmax_columns(out), sims, candidate_lengths



def compute_vector_scores(
    texts: list[list[str]],
    chunks: list[Chunk],
    backend_name: str,
    model_name: str,
    batch_size: int,
    model_cache_dir: Path,
    jd_embeddings_cache: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    del backend_name
    backend = SentenceTransformerVectorBackend(model_name, model_cache_dir)
    query_emb = load_cached_jd_embeddings(jd_embeddings_cache, chunks, model_name)
    if query_emb is None:
        query_emb = backend.encode_queries([chunk.vector_query for chunk in chunks], batch_size)
        write_cached_jd_embeddings(jd_embeddings_cache, chunks, model_name, query_emb)

    scores, sims, candidate_lengths = backend.score_queries(texts, query_emb, batch_size)
    best_role_indices = np.full(len(texts), -1, dtype=np.int32)
    offset = 0
    for cand_idx, length in enumerate(candidate_lengths):
        if length <= 1:
            offset += int(length)
            continue
        role_sims = sims[offset + 1 : offset + length]
        if role_sims.size == 0:
            offset += int(length)
            continue
        role_scores = role_sims.max(axis=1)
        best_role_rel = int(np.argmax(role_scores))
        best_role_indices[cand_idx] = best_role_rel + 1
        offset += int(length)
    return scores, best_role_indices, backend.name


def prepare_repo_assets(
    chunks: list[Chunk],
    embedding_model: str,
    cross_encoder_model: str,
    batch_size: int,
    model_cache_dir: Path,
    jd_embeddings_cache: Path,
) -> tuple[str, str]:
    vector_backend = SentenceTransformerVectorBackend(embedding_model, model_cache_dir)
    query_emb = load_cached_jd_embeddings(jd_embeddings_cache, chunks, embedding_model)
    if query_emb is None:
        query_emb = vector_backend.encode_queries([chunk.vector_query for chunk in chunks], batch_size)
        write_cached_jd_embeddings(jd_embeddings_cache, chunks, embedding_model, query_emb)

    _, cross_encoder_path = load_cross_encoder_model(cross_encoder_model, model_cache_dir)
    return vector_backend.name, f"sentence_transformers_cross_encoder:{cross_encoder_path}"


def duplicate_neighbors(chunks: list[Chunk]) -> list[list[int]]:
    id_to_idx = {chunk.id: idx for idx, chunk in enumerate(chunks)}
    neighbors: list[set[int]] = [set() for _ in chunks]
    for idx, chunk in enumerate(chunks):
        for dup_id in chunk.do_not_duplicate_with:
            if dup_id not in id_to_idx:
                continue
            dup_idx = id_to_idx[dup_id]
            neighbors[idx].add(dup_idx)
            neighbors[dup_idx].add(idx)

    components: list[list[int]] = []
    visited: set[int] = set()
    for start_idx in range(len(chunks)):
        if start_idx in visited:
            continue
        stack = [start_idx]
        visited.add(start_idx)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def chunk_overlap_similarity_matrix(chunks: list[Chunk]) -> np.ndarray:
    signatures: list[set[str]] = []
    for chunk in chunks:
        signature = set(tokenize(" ".join([chunk.expanded_text, chunk.vector_query, chunk.bm25_query, " ".join(chunk.terms)])))
        signatures.append(signature)

    n = len(chunks)
    sims = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a = signatures[i]
            b = signatures[j]
            if not a or not b:
                sim = 0.0
            else:
                sim = float(len(a & b)) / float(len(a | b))
            sims[i, j] = sim
            sims[j, i] = sim
    return sims


def apply_overlap_adjustment(weighted_scores: np.ndarray, chunks: list[Chunk]) -> np.ndarray:
    adjusted = np.asarray(weighted_scores, dtype=np.float32).copy()
    clusters = duplicate_neighbors(chunks)
    similarity = chunk_overlap_similarity_matrix(chunks)

    # Only suppress near-duplicate JD chunks. Distinct concepts in the same
    # connected component keep their full credit.
    for row_idx in range(adjusted.shape[0]):
        row = adjusted[row_idx]
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            order = sorted(cluster, key=lambda i: (-float(row[i]), chunks[i].id))
            kept: list[int] = []
            for pos, col in enumerate(order):
                if pos == 0:
                    kept.append(col)
                    continue

                redundancy = max(float(similarity[col, kept_idx]) for kept_idx in kept) if kept else 0.0
                if redundancy >= 0.92:
                    multiplier = 0.0
                elif redundancy >= 0.80:
                    multiplier = 0.60
                else:
                    multiplier = 1.0

                row[col] = row[col] * multiplier
                if multiplier > 0.0:
                    kept.append(col)
    return adjusted


def aggregate_chunk_scores(match_matrix: np.ndarray, chunks: list[Chunk]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray([effective_chunk_weight(chunk) for chunk in chunks], dtype=np.float32)
    weighted = match_matrix * weights.reshape(1, -1)
    adjusted = apply_overlap_adjustment(weighted, chunks)
    pos_cols = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "positive"]
    neg_cols = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "negative"]
    semantic_score = adjusted[:, pos_cols].sum(axis=1) if pos_cols else np.zeros(match_matrix.shape[0], dtype=np.float32)
    negative_score = adjusted[:, neg_cols].sum(axis=1) if neg_cols else np.zeros(match_matrix.shape[0], dtype=np.float32)
    semantic_final_raw = semantic_score - negative_score
    return semantic_score, negative_score, semantic_final_raw, adjusted


def compute_chunk_focus_weights(
    bm25_scores: np.ndarray,
    vector_scores: np.ndarray,
    chunks: list[Chunk],
) -> np.ndarray:
    """Weight chunks that were less explored in earlier stages a little higher later."""
    n_chunks = len(chunks)
    if n_chunks == 0:
        return np.zeros((0,), dtype=np.float32)

    def chunk_strength(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros(n_chunks, dtype=np.float32)
        row_mask = np.any(matrix > 0.0, axis=1)
        active = matrix[row_mask] if row_mask.any() else matrix
        if active.size == 0:
            return np.zeros(n_chunks, dtype=np.float32)
        mean_strength = active.mean(axis=0)
        max_strength = active.max(axis=0)
        return (0.55 * mean_strength + 0.45 * max_strength).astype(np.float32)

    bm25_strength = chunk_strength(bm25_scores)
    vector_strength = chunk_strength(vector_scores)

    coverage = 0.62 * minmax_vector(bm25_strength) + 0.38 * minmax_vector(vector_strength)
    exploration_need = 1.0 - coverage
    focus = 0.92 + 0.28 * exploration_need

    # Preserve a light influence from the JD chunk weight so important chunks
    # still remain important even when their earlier-stage coverage is high.
    jd_weight = np.asarray([chunk.weight for chunk in chunks], dtype=np.float32)
    if jd_weight.size and float(jd_weight.max()) > float(jd_weight.min()):
        jd_weight = 0.96 + 0.08 * minmax_vector(jd_weight)
        focus = focus * jd_weight

    return np.clip(focus, 0.88, 1.20).astype(np.float32)



def select_cross_encoder_chunk_indices(
    cand_idx: int,
    chunks: list[Chunk],
    hybrid_scores: np.ndarray,
    chunk_focus_weights: np.ndarray,
) -> ChunkSelectionPlan:
    row = np.asarray(hybrid_scores[cand_idx], dtype=np.float32)
    return select_family_covered_ce_chunks(row, chunks, chunk_focus_weights)



def compute_cross_encoder_scores(
    candidates: list[CandidateRecord],
    chunks: list[Chunk],
    shortlist: list[int],
    hybrid_scores: np.ndarray,
    chunk_focus_weights: np.ndarray,
    best_role_indices_by_shortlist: list[int],
    backend_name: str,
    model_name: str,
    batch_size: int,
    require_cross_encoder: bool,
    model_cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    del backend_name, require_cross_encoder
    model, model_path = load_cross_encoder_model(model_name, model_cache_dir)

    positive_chunks = [c for c in chunks if c.polarity == "positive"]
    negative_chunks = [c for c in chunks if c.polarity == "negative"]

    all_pairs = []
    pair_meta = []
    ce_text_lengths: list[int] = []
    achievement_snippet_counts: list[int] = []
    selection_plans: dict[int, ChunkSelectionPlan] = {
        cand_idx: select_cross_encoder_chunk_indices(cand_idx, chunks, hybrid_scores, chunk_focus_weights)
        for cand_idx in shortlist
    }

    for shortlist_pos, cand_idx in enumerate(shortlist):
        plan = selection_plans.get(cand_idx)
        if plan is None:
            continue

        selected_indices = list(dict.fromkeys(plan.ce_positive_indices + plan.ce_negative_indices))
        if not selected_indices:
            continue

        best_role_idx = best_role_indices_by_shortlist[shortlist_pos] if shortlist_pos < len(best_role_indices_by_shortlist) else -1
        candidate_texts = [text.strip() for text in candidates[cand_idx].candidate_texts if text and text.strip()]
        latest_role = candidate_texts[-1] if len(candidate_texts) > 1 else ""
        best_matching_role = candidate_texts[best_role_idx] if 0 <= best_role_idx < len(candidate_texts) else ""
        achievement_snippets = extract_achievement_snippets(candidates[cand_idx].candidate_text)
        ce_text = build_cross_encoder_candidate_text(
            candidates[cand_idx],
            latest_role,
            best_matching_role,
            achievement_snippets=achievement_snippets,
        )
        clean_text = ce_text.strip()[:1600]
        if not clean_text:
            continue
        ce_text_lengths.append(len(clean_text))
        achievement_snippet_counts.append(len(achievement_snippets))

        for chunk_idx in selected_indices:
            chunk = chunks[chunk_idx]
            all_pairs.append((chunk.expanded_text, clean_text))
            pair_meta.append((cand_idx, chunk.id, chunk.weight, chunk.polarity))

    bi_family_totals: Counter[str] = Counter()
    ce_family_totals: Counter[str] = Counter()
    positive_selected = 0
    negative_selected = 0
    total_pool = 0
    for plan in selection_plans.values():
        bi_family_totals.update(plan.bi_family_coverage)
        ce_family_totals.update(plan.ce_family_coverage)
        positive_selected += len(plan.ce_positive_indices)
        negative_selected += len(plan.ce_negative_indices)
        total_pool += len(plan.bi_positive_pool)

    shortlist_count = max(1, len(shortlist))
    avg_pool = total_pool / shortlist_count
    avg_positive = positive_selected / shortlist_count
    avg_negative = negative_selected / shortlist_count
    avg_total = (positive_selected + negative_selected) / shortlist_count
    avg_text_len = float(sum(ce_text_lengths) / max(1, len(ce_text_lengths))) if ce_text_lengths else 0.0
    avg_snippets = float(sum(achievement_snippet_counts) / max(1, len(achievement_snippet_counts))) if achievement_snippet_counts else 0.0
    bi_family_summary = ", ".join(f"{family[:3]}={bi_family_totals.get(family, 0) / shortlist_count:.2f}" for family in POSITIVE_FAMILY_ORDER)
    ce_family_summary = ", ".join(f"{family[:3]}={ce_family_totals.get(family, 0) / shortlist_count:.2f}" for family in POSITIVE_FAMILY_ORDER)
    print(f"[ranker] CE family coverage avg/candidate: bi_pool({bi_family_summary}); ce_selected({ce_family_summary})", file=sys.stderr)
    print(
        f"[ranker] CE selected avg/candidate: pool={avg_pool:.2f} positive={avg_positive:.2f} negative={avg_negative:.2f} total={avg_total:.2f}; "
        f"avg_text_len={avg_text_len:.1f}; avg_achievement_snippets={avg_snippets:.2f}",
        file=sys.stderr,
    )
    print(f"[ranker] Cross-Encoder chunk matrix: {len(all_pairs)} pairs generated.", file=sys.stderr)

    raw_scores = model.predict(all_pairs, batch_size=batch_size, show_progress_bar=True) if all_pairs else np.asarray([], dtype=np.float32)
    raw_scores = np.asarray(raw_scores, dtype=np.float32).flatten()
    if raw_scores.size and (float(raw_scores.min()) < 0.0 or float(raw_scores.max()) > 1.0):
        finite_scores = raw_scores[np.isfinite(raw_scores)]
        if finite_scores.size:
            # Calibrate logits, not probabilities: center on the shortlist median and
            # use a wider spread so high-end evidence does not collapse to 1.0.
            ce_bias = float(np.median(finite_scores))
            if finite_scores.size >= 5:
                p10 = float(np.percentile(finite_scores, 10))
                p90 = float(np.percentile(finite_scores, 90))
                spread = max(1e-6, p90 - p10)
                ce_temp = max(CROSS_ENCODER_TEMPERATURE, 0.5 * spread)
            else:
                ce_temp = CROSS_ENCODER_TEMPERATURE
            raw_scores = sigmoid((raw_scores - ce_bias) / ce_temp).astype(np.float32)
        else:
            raw_scores = sigmoid((raw_scores - CROSS_ENCODER_LOGIT_BIAS) / CROSS_ENCODER_TEMPERATURE).astype(np.float32)
    if raw_scores.size:
        raw_scores = np.clip(raw_scores, CE_SCORE_EPS, 1.0 - CE_SCORE_EPS).astype(np.float32)

    cand_chunk_max = defaultdict(lambda: defaultdict(float))
    for meta, score in zip(pair_meta, raw_scores):
        c_idx, c_id, chunk_weight, pol = meta
        weighted_score = float(score)
        if weighted_score > cand_chunk_max[c_idx][c_id]:
            cand_chunk_max[c_idx][c_id] = weighted_score

    cross_matrix = np.zeros((len(shortlist), len(chunks)), dtype=np.float32)
    for row_idx, cand_idx in enumerate(shortlist):
        for col_idx, chunk in enumerate(chunks):
            cross_matrix[row_idx, col_idx] = float(cand_chunk_max[cand_idx][chunk.id])

    cross_weighted = cross_matrix * np.asarray([effective_chunk_weight(chunk) for chunk in chunks], dtype=np.float32).reshape(1, -1) * chunk_focus_weights.reshape(1, -1)
    cross_adjusted = apply_overlap_adjustment(cross_weighted, chunks)

    pos_cols = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "positive"]
    neg_cols = [idx for idx, chunk in enumerate(chunks) if chunk.polarity == "negative"]

    pos_scores = np.zeros(len(shortlist), dtype=np.float32)
    neg_scores = np.zeros(len(shortlist), dtype=np.float32)

    pos_weights = np.asarray([effective_chunk_weight(chunks[idx]) for idx in pos_cols], dtype=np.float32) if pos_cols else np.zeros((0,), dtype=np.float32)
    neg_weights = np.asarray([effective_chunk_weight(chunks[idx]) for idx in neg_cols], dtype=np.float32) if neg_cols else np.zeros((0,), dtype=np.float32)

    for row_idx in range(len(shortlist)):
        if pos_cols:
            row = np.asarray(cross_adjusted[row_idx, pos_cols], dtype=np.float32)
            if row.size:
                order = np.argsort(-row, kind="mergesort")
                top = order[:3]
                top_scores = row[top]
                max_score = float(top_scores[0]) if top_scores.size else 0.0
                mean_top = float(np.mean(top_scores)) if top_scores.size else 0.0
                pos_scores[row_idx] = float(np.clip(clamp01(0.70 * max_score + 0.30 * mean_top), CE_SCORE_EPS, 1.0 - CE_SCORE_EPS))
        if neg_cols:
            row = np.asarray(cross_adjusted[row_idx, neg_cols], dtype=np.float32)
            if row.size:
                order = np.argsort(-row, kind="mergesort")
                top = order[:2]
                top_scores = row[top]
                max_score = float(top_scores[0]) if top_scores.size else 0.0
                mean_top = float(np.mean(top_scores)) if top_scores.size else 0.0
                neg_scores[row_idx] = float(np.clip(clamp01(0.70 * max_score + 0.30 * mean_top), CE_SCORE_EPS, 1.0 - CE_SCORE_EPS))

    return pos_scores, neg_scores, cross_adjusted, f"sentence_transformers_cross_encoder:{model_path}"


def first_evidence_snippet(
    text: str,
    terms: Iterable[str],
    max_len: int = 140,
    reverse: bool = False,
) -> str | None:
    lowered_terms = [term.lower() for term in terms if term and len(term) >= 2]
    if not lowered_terms:
        return None
    sentences = SENTENCE_RE.split(text or "")
    if reverse:
        sentences = list(reversed(sentences))
    for sentence in sentences:
        clean = " ".join(sentence.split())
        if not clean:
            continue
        clean_l = clean.lower()
        if any(term in clean_l for term in lowered_terms):
            return clean[: max_len - 3] + "..." if len(clean) > max_len else clean
    return None


def chunk_label(chunk: Chunk) -> str:
    text = chunk.vector_query or chunk.expanded_text
    return text.rstrip(".")



def make_reasoning(
    candidate: CandidateRecord,
    chunks: list[Chunk],
    adjusted_scores: np.ndarray,
) -> str:
    pos = [
        (float(adjusted_scores[idx]), idx)
        for idx, chunk in enumerate(chunks)
        if chunk.polarity == "positive" and adjusted_scores[idx] > 0.05
    ]
    neg = [
        (float(adjusted_scores[idx]), idx)
        for idx, chunk in enumerate(chunks)
        if chunk.polarity == "negative" and adjusted_scores[idx] > 0.05
    ]
    pos.sort(reverse=True)
    neg.sort(reverse=True)

    candidate_text = candidate.candidate_text

    positive_bits: list[str] = []
    positive_families: list[str] = []
    for _, idx in pos[:4]:
        chunk = chunks[idx]
        snippet = first_evidence_snippet(
            candidate_text,
            list(chunk.terms) + tokenize(chunk.bm25_query),
            reverse=True,
        )
        if snippet:
            positive_bits.append(f"{chunk.id}: {snippet}")
        elif len(positive_bits) < 2:
            positive_bits.append(f"{chunk.id}: narrative match for {chunk_label(chunk)}")
        positive_families.append(chunk_family(chunk.id))
        if len(positive_bits) >= 3:
            break

    penalty_bits: list[str] = []
    for _, idx in neg[:3]:
        chunk = chunks[idx]
        penalty_bits.append(f"{chunk.id}: penalty for {chunk_label(chunk)}")
        break
    penalty_bits.extend(candidate.bvs_penalties[:2])

    strength_text = " ".join((candidate.bvs_strengths[:3] or [])).lower()
    top_family = positive_families[0] if positive_families else "UNKNOWN"
    family_count = len(set(positive_families))
    evidence_count = len(positive_bits)
    behavior_count = len(candidate.bvs_strengths)
    caution_count = len(penalty_bits)

    has_production_signal = top_family in {"SYSTEMS", "PRODUCT", "ADVANCED"} or any(
        term in strength_text for term in ("production", "deployed", "shipped", "launched", "latency", "throughput")
    )

    if positive_bits:
        if has_production_signal:
            opener = "The strongest signal is production-oriented execution, reinforced by the candidate's matching evidence."
        elif family_count >= 4:
            opener = "The strongest signal is breadth across several JD families, which makes the fit look durable."
        elif evidence_count >= 2:
            opener = "The strongest signal is repeated narrative evidence pointing to the same role fit."
        else:
            opener = "The strongest signal is direct semantic alignment in the candidate narrative."
    elif behavior_count > 0:
        opener = "The strongest signal is recruiter-side readiness, with structured cues supporting the profile."
    elif caution_count > 0:
        opener = "The strongest signal is negative-confidence risk, so the profile deserves caution."
    else:
        return "No concise evidence available from candidate narrative or structured fields."

    sentences: list[str] = [opener]

    if positive_bits:
        if has_production_signal:
            supporting_sentence = "Relevant evidence includes " + "; ".join(positive_bits[:3]) + "."
        elif family_count >= 4:
            supporting_sentence = "Supporting evidence spans " + str(family_count) + " JD families: " + "; ".join(positive_bits[:3]) + "."
        elif evidence_count >= 2:
            supporting_sentence = "Key supporting evidence includes " + "; ".join(positive_bits[:3]) + "."
        else:
            supporting_sentence = "The best supporting passage is " + positive_bits[0] + "."
        sentences.append(supporting_sentence)

    if family_count:
        if family_count >= 4:
            coverage_sentence = "Coverage spans several JD families, so the match is not isolated to one theme."
        elif family_count == 3:
            coverage_sentence = "Coverage is balanced across multiple JD families."
        elif family_count == 2:
            coverage_sentence = "Coverage reaches more than one JD family, which improves confidence."
        else:
            coverage_sentence = "Coverage is concentrated in one strong JD family."
        sentences.append(coverage_sentence)

    if candidate.bvs_strengths:
        strengths = [
            s.replace("_", " ").replace("=", ": ")
            for s in candidate.bvs_strengths[:3]
        ]
        if has_production_signal:
            behavior_sentence = "Recruiter-side signals add support through " + "; ".join(strengths) + "."
        else:
            behavior_sentence = "Behavioral signals add support through " + "; ".join(strengths) + "."
        sentences.append(behavior_sentence)

    if penalty_bits:
        penalties = [p.replace("_", " ") for p in penalty_bits[:3]]
        if positive_bits:
            caution_sentence = "A small caution remains around " + "; ".join(penalties) + "."
        else:
            caution_sentence = "The main caution remains around " + "; ".join(penalties) + "."
        sentences.append(caution_sentence)

    return " ".join(sentences)[:900]


def write_top_output(

    path: Path,
    ranked: list[int],
    final_scores: np.ndarray,
    candidates: list[CandidateRecord],
    chunks: list[Chunk],
    adjusted_scores: np.ndarray,
    top_k: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, cand_idx in enumerate(ranked[:top_k], 1):
            writer.writerow(
                {
                    "candidate_id": candidates[cand_idx].candidate_id,
                    "rank": rank,
                    "score": f"{float(final_scores[cand_idx]):.8f}",
                    "reasoning": make_reasoning(candidates[cand_idx], chunks, adjusted_scores[cand_idx]),
                }
            )


def write_scores_output(
    path: Path,
    candidates: list[CandidateRecord],
    semantic_score: np.ndarray,
    negative_score: np.ndarray,
    semantic_final_raw: np.ndarray,
    semantic_final_norm: np.ndarray,
    bvs_scores: np.ndarray,
    final_scores: np.ndarray | None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "candidate_id",
            "semantic_score",
            "negative_score",
            "semantic_final_raw",
            "semantic_final_norm",
            "bvs_score",
            "final_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, candidate in enumerate(candidates):
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "semantic_score": f"{float(semantic_score[idx]):.8f}",
                    "negative_score": f"{float(negative_score[idx]):.8f}",
                    "semantic_final_raw": f"{float(semantic_final_raw[idx]):.8f}",
                    "semantic_final_norm": f"{float(semantic_final_norm[idx]):.8f}",
                    "bvs_score": f"{float(bvs_scores[idx]):.8f}",
                    "final_score": "" if final_scores is None else f"{float(final_scores[idx]):.8f}",
                }
            )


def write_chunk_scores_output(
    path: Path,
    candidate_indices: list[int],
    candidates: list[CandidateRecord],
    chunks: list[Chunk],
    bm25: np.ndarray,
    vector: np.ndarray,
    hybrid: np.ndarray,
    adjusted: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "chunk_id",
                "polarity",
                "weight",
                "bm25_score",
                "vector_score",
                "hybrid_match",
                "overlap_adjusted_weighted_score",
            ],
        )
        writer.writeheader()
        for cand_idx in candidate_indices:
            for chunk_idx, chunk in enumerate(chunks):
                writer.writerow(
                    {
                        "candidate_id": candidates[cand_idx].candidate_id,
                        "chunk_id": chunk.id,
                        "polarity": chunk.polarity,
                        "weight": f"{chunk.weight:.4f}",
                        "bm25_score": f"{float(bm25[cand_idx, chunk_idx]):.8f}",
                        "vector_score": f"{float(vector[cand_idx, chunk_idx]):.8f}",
                        "hybrid_match": f"{float(hybrid[cand_idx, chunk_idx]):.8f}",
                        "overlap_adjusted_weighted_score": f"{float(adjusted[cand_idx, chunk_idx]):.8f}",
                    }
                )



def run(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    jd_path = Path(args.jd_index)
    candidate_path = Path(args.candidates)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Only the submission CSV should be produced during ranking.
    # Keep all diagnostics disabled so the repo stays hackathon-compliant.
    metadata_path = None
    l0_report_path = None
    stage1_report_path = None
    stage2_report_path = None
    args.scores_output = ""
    args.chunk_scores_output = ""
    model_cache_dir = Path(args.model_cache_dir)
    jd_embeddings_cache = Path(args.jd_embeddings_cache)

    chunks = load_jd_index(jd_path)
    args.cross_encoder_model = DEFAULT_CROSS_ENCODER_MODEL

    if args.validate_only:
        print(f"Validated {jd_path}: {len(chunks)} scoreable chunks (C01-C24).")
        return 0

    if args.prepare_cache_only:
        vector_backend, cross_backend = prepare_repo_assets(
            chunks,
            args.embedding_model,
            args.cross_encoder_model,
            args.batch_size,
            model_cache_dir,
            jd_embeddings_cache,
        )
        print(f"[ranker] prepared vector backend {vector_backend}", file=sys.stderr)
        print(f"[ranker] prepared cross-encoder backend {cross_backend}", file=sys.stderr)
        print(f"[ranker] prepared JD embeddings cache {jd_embeddings_cache}", file=sys.stderr)
        return 0

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidate_path}. Copy candidates.jsonl into {base_dir} "
            "or pass --candidates with a path inside this challenge folder."
        )

    as_of = parse_date(args.as_of_date)
    if not as_of:
        raise ValueError(f"Invalid --as-of-date {args.as_of_date!r}")

    print(f"[ranker] loading candidates from {candidate_path}", file=sys.stderr)
    candidates, bvs_scores, discarded, l0_threshold = load_candidates(candidate_path, as_of, args.max_candidates)
    raw_bvs_scores = np.asarray(bvs_scores, dtype=np.float32)
    bvs_scores = (
        (1.0 - BVS_PERCENTILE_BLEND) * raw_bvs_scores
        + BVS_PERCENTILE_BLEND * stretch_bvs_percentile(percentile_rank(raw_bvs_scores))
    ).astype(np.float32)
    print(f"[ranker] L0 kept={len(candidates)} discarded={len(discarded)} threshold={l0_threshold:.3f}", file=sys.stderr)

    # Optional diagnostics: off by default so the repo only emits the CSV the hackathon requires.
    if l0_report_path:
        with l0_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "status", "details"])
            writer.writeheader()
            for idx, candidate in enumerate(candidates):
                writer.writerow(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "Kept",
                        "details": f"{float(bvs_scores[idx]):.8f}",
                    }
                )
            for item in discarded:
                writer.writerow(
                    {
                        "candidate_id": item["candidate_id"],
                        "status": "Discarded",
                        "details": item["reason"],
                    }
                )

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    candidate_texts = [candidate.retrieval_text for candidate in candidates]

    print(f"[ranker] loaded {len(candidates)} candidates after L0 Triage", file=sys.stderr)
    print("[ranker] building BM25 scores", file=sys.stderr)
    bm25_index = BM25Index(candidate_texts)
    bm25_scores = bm25_index.score_queries([chunk.bm25_query for chunk in chunks])

    bm25_family_scores = family_score_matrix(bm25_scores, chunks)
    fast_semantic_raw, _, _, _ = aggregate_chunk_scores(bm25_scores, chunks)
    fast_triage_scores = (0.64 * minmax_vector(fast_semantic_raw)) + (0.36 * bvs_scores)
    pre_shortlist_size = min(args.pre_shortlist_size, len(candidates))
    pre_shortlist, bm25_family_hits, bm25_family_overlap = select_family_recall_candidates(
        list(range(len(candidates))),
        candidate_ids,
        fast_triage_scores,
        bm25_family_scores,
        BM25_FAMILY_RECALL_TARGETS,
        pre_shortlist_size,
    )
    pre_shortlist_set = set(pre_shortlist)
    print(
        "[ranker] BM25 family recall: "
        + ", ".join(f"{fam.lower()}={bm25_family_hits.get(fam, 0)}" for fam in FAMILY_RECALL_ORDER)
        + f"; overlap={bm25_family_overlap}",
        file=sys.stderr,
    )
    if stage1_report_path:
        with stage1_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "fast_triage_score", "status"])
            writer.writeheader()
            for idx, candidate in enumerate(candidates):
                writer.writerow(
                    {
                        "candidate_id": candidate.candidate_id,
                        "fast_triage_score": f"{float(fast_triage_scores[idx]):.8f}",
                        "status": "Advanced to Vector" if idx in pre_shortlist_set else "Dropped",
                    }
                )

    print("[ranker] building vector scores", file=sys.stderr)
    vector_scores = np.zeros((len(candidates), len(chunks)), dtype=np.float32)
    subset_vector_scores, subset_best_role_indices, vector_backend = compute_vector_scores(
        [candidates[idx].candidate_texts for idx in pre_shortlist],
        chunks,
        args.vector_backend,
        args.embedding_model,
        args.batch_size,
        model_cache_dir,
        jd_embeddings_cache,
    )
    vector_scores[np.asarray(pre_shortlist, dtype=np.int32)] = subset_vector_scores
    best_role_indices_map = {
        pre_shortlist[offset]: int(best_idx)
        for offset, best_idx in enumerate(subset_best_role_indices)
    }

    chunk_focus_weights = compute_chunk_focus_weights(bm25_scores, vector_scores, chunks)
    bm25_scores = bm25_scores * chunk_focus_weights.reshape(1, -1)
    vector_scores = vector_scores * chunk_focus_weights.reshape(1, -1)

    hybrid_scores = (0.35 * bm25_scores + 0.65 * vector_scores).astype(np.float32)
    semantic_score, negative_score, semantic_final_raw, adjusted_scores = aggregate_chunk_scores(hybrid_scores, chunks)
    semantic_final_norm = np.zeros_like(semantic_final_raw, dtype=np.float32)
    if pre_shortlist:
        subset_raw = semantic_final_raw[pre_shortlist]
        vmin, vmax = float(subset_raw.min()), float(subset_raw.max())
        if vmax > vmin:
            semantic_final_norm[pre_shortlist] = (subset_raw - vmin) / (vmax - vmin)

    first_pass_scores = np.full(len(candidates), -np.inf, dtype=np.float32)
    for cand_idx in pre_shortlist:
        first_pass_scores[cand_idx] = (0.64 * float(semantic_final_norm[cand_idx])) + (0.36 * float(bvs_scores[cand_idx]))

    shortlist_size = min(args.shortlist_size, len(candidates))
    vector_family_scores = family_score_matrix(vector_scores, chunks)
    shortlist_order, vector_family_hits, vector_family_overlap = select_family_recall_candidates(
        pre_shortlist,
        candidate_ids,
        first_pass_scores,
        vector_family_scores,
        VECTOR_FAMILY_RECALL_TARGETS,
        shortlist_size,
    )
    shortlist = shortlist_order
    shortlist_set = set(shortlist)
    print(
        "[ranker] Bi Encoder family recall: "
        + ", ".join(f"{fam.lower()}={vector_family_hits.get(fam, 0)}" for fam in FAMILY_RECALL_ORDER)
        + f"; overlap={vector_family_overlap}",
        file=sys.stderr,
    )
    if stage2_report_path:
        with stage2_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "first_pass_score", "status"])
            writer.writeheader()
            for idx in pre_shortlist:
                writer.writerow(
                    {
                        "candidate_id": candidates[idx].candidate_id,
                        "first_pass_score": f"{float(first_pass_scores[idx]):.8f}",
                        "status": "Advanced to CrossEncoder" if idx in shortlist_set else "Dropped",
                    }
                )

    print(f"[ranker] reranking shortlist of {len(shortlist)} candidates", file=sys.stderr)

    best_role_indices_by_shortlist = [best_role_indices_map.get(cand_idx, -1) for cand_idx in shortlist]
    cross_scores_shortlist, cross_neg_scores, cross_adjusted_shortlist, cross_backend = compute_cross_encoder_scores(
        candidates,
        chunks,
        shortlist,
        hybrid_scores,
        chunk_focus_weights,
        best_role_indices_by_shortlist,
        args.cross_encoder_backend,
        args.cross_encoder_model,
        args.batch_size,
        args.require_cross_encoder,
        model_cache_dir,
    )

    ce_family_counter: Counter[str] = Counter()
    ce_chunk_counts: list[int] = []
    for shortlist_pos, cand_idx in enumerate(shortlist):
        plan = select_cross_encoder_chunk_indices(cand_idx, chunks, hybrid_scores, chunk_focus_weights)
        ce_chunk_count = len(dict.fromkeys(plan.ce_positive_indices + plan.ce_negative_indices))
        ce_chunk_counts.append(ce_chunk_count)
        family_scores = family_score_matrix(vector_scores[[cand_idx]], chunks)
        if family_scores:
            best_family = max(family_scores.items(), key=lambda item: float(item[1][0]))[0]
            ce_family_counter[best_family] += 1
    avg_ce_chunks = float(sum(ce_chunk_counts) / max(1, len(ce_chunk_counts))) if ce_chunk_counts else 0.0
    print(
        "[ranker] CE family distribution: "
        + ", ".join(f"{fam.lower()}={ce_family_counter.get(fam, 0)}" for fam in FAMILY_RECALL_ORDER)
        + f"; avg_chunks={avg_ce_chunks:.2f}",
        file=sys.stderr,
    )

    cross_adjusted_full = np.zeros_like(adjusted_scores, dtype=np.float32)
    for offset, cand_idx in enumerate(shortlist):
        cross_adjusted_full[cand_idx] = cross_adjusted_shortlist[offset]


    final_scores = np.full(len(candidates), -np.inf, dtype=np.float32)
    coverage_bonus_values: list[float] = []
    evidence_bonus_values: list[float] = []
    negative_conf_values: list[float] = []
    family_counts: list[int] = []
    positive_evidence_hits: list[int] = []
    smart_multiplier_values: list[float] = []
    ce_penalty_values: list[float] = []

    for offset, cand_idx in enumerate(shortlist):
        ce_tech = float(cross_scores_shortlist[offset])
        semantic_boost = float(semantic_final_norm[cand_idx])
        behavior_boost = float(bvs_scores[cand_idx]) - 0.5

        adjusted_row = np.asarray(cross_adjusted_full[cand_idx], dtype=np.float32)
        coverage_bonus, coverage_quality, family_hits = positive_family_coverage_bonus(adjusted_row, chunks)
        evidence_bonus, evidence_quality, evidence_hits = evidence_density_bonus(adjusted_row, chunks)

        negative_conf, negative_breakdown = negative_confidence_details(candidates[cand_idx].candidate_text)

        # Smart multiplier is based on the spread between CE, Bi-Encoder, and BM25.
        # Lower disagreement means higher confidence in the Cross-Encoder signal.
        bm25_row = np.asarray(bm25_scores[cand_idx], dtype=np.float32).ravel()
        bi_row = np.asarray(vector_scores[cand_idx], dtype=np.float32).ravel()

        def _top3_mean(values: np.ndarray) -> float:
            vec = np.asarray(values, dtype=np.float32).ravel()
            if vec.size == 0:
                return 0.0
            k = min(3, int(vec.size))
            if k == vec.size:
                top = np.sort(vec)[-k:]
            else:
                top = np.partition(vec, -k)[-k:]
            return float(np.mean(top))

        bm25_proxy = clamp01(_top3_mean(bm25_row))
        bi_proxy = clamp01(_top3_mean(bi_row))
        ce_proxy = clamp01(float(ce_tech))

        agreement = 1.0 - (
            abs(ce_proxy - bi_proxy)
            + abs(ce_proxy - bm25_proxy)
            + abs(bi_proxy - bm25_proxy)
        ) / 3.0
        agreement = clamp01(agreement)
        smart_multiplier = 0.90 + 0.10 * agreement

        # Apply the smart multiplier only to the Cross-Encoder contribution.
        ce_risk_multiplier = math.exp(-max(0.0, float(cross_neg_scores[offset])) * 2.4)
        ce_adjusted = ce_tech * smart_multiplier * ce_risk_multiplier

        # Adaptive reliability-weighted semantic fusion: keep the base mix stable,
        # but apply only a small consensus-aware calibration to the three retrieval signals.
        semantic_proxy = float(math.pow(clamp01(semantic_boost), 0.92))
        fusion_base = np.asarray([0.55, 0.25, 0.20], dtype=np.float32)
        fusion_signals = np.asarray([ce_adjusted, semantic_proxy, bm25_proxy], dtype=np.float32)

        signal_proxies = np.asarray([
            ce_proxy,
            semantic_proxy,
            bm25_proxy,
        ], dtype=np.float32)
        consensus = float(np.mean(signal_proxies))
        deviation = np.abs(signal_proxies - consensus)
        raw_reliability = np.exp(-4.0 * deviation)
        raw_reliability = np.clip(raw_reliability, 1e-6, None)
        reliability_weights = raw_reliability / float(np.sum(raw_reliability))

        # Keep shifts small so the ensemble remains stable and close to the
        # original 0.55 / 0.25 / 0.20 blend.
        fusion_weights = (0.92 * fusion_base) + (0.08 * reliability_weights)
        fusion_weights = fusion_weights / float(np.sum(fusion_weights))
        semantic_core = float(np.dot(fusion_weights, fusion_signals))

        # Stretch only the upper tail so the strongest semantic matches separate
        # a little more without changing the ordering or the rest of the scale.
        semantic_core = stretch_upper_tail(semantic_core, threshold=0.75, gamma=0.92)

        # BVS remains a centered refinement signal, not a raw positive offset.
        bvs_bonus = 0.17 * behavior_boost

        # Coverage and evidence stay as small bounded refinements.
        final_raw = (0.72 * semantic_core) + bvs_bonus + coverage_bonus + evidence_bonus

        # Narrative negative confidence acts as a calibrated risk penalty.
        narrative_penalty = negative_confidence_penalty(negative_conf)
        final_scores[cand_idx] = final_raw * narrative_penalty

        coverage_bonus_values.append(coverage_bonus)
        evidence_bonus_values.append(evidence_bonus)
        negative_conf_values.append(negative_conf)
        family_counts.append(len(family_hits))
        positive_evidence_hits.append(evidence_hits)
        smart_multiplier_values.append(float(smart_multiplier))
        ce_penalty_values.append(float(ce_risk_multiplier))
    if shortlist:
        coverage_avg = float(np.mean(np.asarray(coverage_bonus_values, dtype=np.float32)))
        evidence_avg = float(np.mean(np.asarray(evidence_bonus_values, dtype=np.float32)))
        coverage_max = float(np.max(np.asarray(coverage_bonus_values, dtype=np.float32)))
        evidence_max = float(np.max(np.asarray(evidence_bonus_values, dtype=np.float32)))
        family_avg = float(np.mean(np.asarray(family_counts, dtype=np.float32)))
        hit_avg = float(np.mean(np.asarray(positive_evidence_hits, dtype=np.float32)))
        neg_arr = np.asarray(negative_conf_values, dtype=np.float32)
        neg_avg = float(np.mean(neg_arr))
        neg_p90 = float(np.quantile(neg_arr, 0.90)) if neg_arr.size else 0.0
        neg_max = float(np.max(neg_arr)) if neg_arr.size else 0.0
        strong_neg = int(np.sum(neg_arr >= 0.65))
        print(
            f"[ranker] final bonus stats: coverage_avg={coverage_avg:.4f} coverage_max={coverage_max:.4f} "
            f"evidence_avg={evidence_avg:.4f} evidence_max={evidence_max:.4f} "
            f"family_avg={family_avg:.2f} evidence_hits_avg={hit_avg:.2f}",
            file=sys.stderr,
        )
        print(
            f"[ranker] negative confidence stats: avg={neg_avg:.4f} p90={neg_p90:.4f} max={neg_max:.4f} strong={strong_neg}",
            file=sys.stderr,
        )

    ranked = order_candidates_with_bvs_tiebreak(final_scores, bvs_scores, candidate_ids)
    ranked = [idx for idx in ranked if np.isfinite(final_scores[idx])]
    if len(ranked) < args.top_k and not args.allow_fewer_than_top_k:
        raise ValueError(
            f"Only {len(ranked)} reranked candidates available; cannot write exactly {args.top_k}. "
            "Increase --shortlist-size or use --allow-fewer-than-top-k for smoke tests."
        )
    # -----------------------------------
    write_top_output(output_path, ranked, final_scores, candidates, chunks, cross_adjusted_full, min(args.top_k, len(ranked)))

    if args.scores_output:
        write_scores_output(
            Path(args.scores_output),
            candidates,
            semantic_score,
            negative_score,
            semantic_final_raw,
            semantic_final_norm,
            bvs_scores,
            final_scores,
        )

    if args.chunk_scores_output:
        chunk_indices = ranked[: min(args.chunk_scores_limit, len(ranked))]
        write_chunk_scores_output(
            Path(args.chunk_scores_output),
            chunk_indices,
            candidates,
            chunks,
            bm25_scores,
            vector_scores,
            hybrid_scores,
            adjusted_scores,
        )

    if metadata_path:
        metadata = {
            "jd_index": str(jd_path),
            "candidate_file": str(candidate_path),
            "candidate_count": len(candidates),
            "scoreable_chunk_count": len(chunks),
            "semantic_fields": ["profile.headline", "profile.summary", "career_history[].description"],
            "bm25_weight": 0.35,
            "vector_weight": 0.65,
            "bvs_transform": "percentile_rank_then_sigmoid_stretch",
            "embedding_model": args.embedding_model,
            "cross_encoder_model": args.cross_encoder_model,
            "model_cache_dir": str(model_cache_dir),
            "jd_embeddings_cache": str(jd_embeddings_cache),
            "final_weights": {
                "first_pass_hybrid": "Behavior-aware: 0.64 * semantic_norm + 0.36 * bvs_score",
                "cross_encoder_final": "Adaptive reliability-weighted fusion over CE, semantic_boost, and bvs_base; CE remains the dominant signal after smart calibration",
                "negative_penalty": "Calibrated narrative risk penalty: negative_confidence_penalty(negative_conf)",
            },
            "overlap_control": "connected-components near-duplicate suppression using chunk similarity thresholds",
            "vector_backend": vector_backend,
            "cross_encoder_backend": cross_backend,
            "pre_shortlist_size": len(pre_shortlist),
            "shortlist_size": len(shortlist),
            "top_k_written": min(args.top_k, len(ranked)),
            "as_of_date": args.as_of_date,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"[ranker] wrote {output_path}", file=sys.stderr)
    if metadata_path:
        print(f"[ranker] wrote {metadata_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Redrob hybrid candidate ranking.")
    parser.add_argument("--jd-index", default="jd_hybrid_index.json", help="Folder-local JD hybrid index JSON.")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Folder-local candidates JSONL/JSON input.")
    parser.add_argument("--output", default="Real_RR.csv", help="Top-K CSV output.")
    parser.add_argument("--metadata-output", default="", help="Optional run metadata JSON output.")
    parser.add_argument("--l0-report-output", default="", help="Optional L0 triage diagnostic CSV output.")
    parser.add_argument("--stage1-report-output", default="", help="Optional Stage 1 BM25 diagnostic CSV output.")
    parser.add_argument("--stage2-report-output", default="", help="Optional Stage 2 vector diagnostic CSV output.")
    parser.add_argument("--scores-output", default="", help="Optional per-candidate score CSV.")
    parser.add_argument("--chunk-scores-output", default="", help="Optional chunk score CSV for ranked candidates.")
    parser.add_argument("--chunk-scores-limit", type=int, default=1000, help="How many ranked candidates to include in chunk score export.")
    parser.add_argument("--pre-shortlist-size", type=int, default=1000, help="Candidates to send to Vector Search.")
    parser.add_argument("--shortlist-size", type=int, default=150, help="Candidates to send to Cross Encoder.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K rows to write.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--allow-fewer-than-top-k", action="store_true", help="Allow writing fewer than top_k rows for smoke tests.")
    parser.add_argument("--as-of-date", default="2026-06-09", help="Date used for recency logic (YYYY-MM-DD).")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformers embedding model.")
    parser.add_argument("--model-cache-dir", default="models", help="Repo-local model cache directory.")
    parser.add_argument("--jd-embeddings-cache", default="cache/jd_vector_embeddings.npz", help="Repo-local cache for precomputed JD vector embeddings.")
    parser.add_argument("--vector-backend", default="sentence-transformers", choices=["auto", "sentence-transformers", "sentence_transformers"], help="Vector backend.")
    parser.add_argument("--vector-scores-cache", default="", help="Disabled: candidate-level vector cache is not used.")
    parser.add_argument("--cross-encoder-backend", default="sentence-transformers", choices=["auto", "sentence-transformers", "sentence_transformers"], help="Cross-encoder backend.")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL, help="Sentence-transformers cross-encoder model (domain-adapted reranker).")
    parser.add_argument("--cross-scores-cache", default="", help="Disabled: candidate-level cross-encoder cache is not used.")
    parser.add_argument("--require-cross-encoder", action="store_true", help="Fail instead of using proxy reranking if cross-encoder is unavailable.")
    parser.add_argument("--batch-size", type=int, default=128, help="Model batch size when sentence-transformers is available.")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Download/save repo-local models and precompute JD embeddings, then exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the JD index and exit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[ranker] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
