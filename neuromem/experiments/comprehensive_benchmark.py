"""
Comprehensive multi-workload benchmark for NeuroMem versus traditional baselines.

The goal of this benchmark is to make the comparison more persuasive along three
dimensions:

1. Structural efficiency and retrieval quality
2. Associative/primed recall behavior
3. Long-conversation project handoff behavior under session scope

Methodological choices:
- Real package implementations are used, not mock classes.
- Trials run in isolated subprocesses for better repeatability.
- All systems share the same embedding backend within a run.
- Pairwise comparisons are reported with bootstrap confidence intervals and a
  simple sign-test p-value.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.memory_manager import MemoryManager, MemoryType
from ..core.text_embeddings import TextEmbedder
from ..core.traditional_rag import TraditionalRAGSystem
from .rigorous_benchmark import (
    ALL_EMBEDDING_BACKENDS,
    EMBEDDING_LEXICAL_HASHING,
    THREAD_ENV,
    CorpusItem,
    InMemoryNeuromemManager,
    build_association_pairs,
    build_corpus,
    build_embedder,
    database_footprint_bytes,
    ingest_item,
    result_contents,
    summarize_samples,
)


STRUCTURE_SUITE = "structure"
CONVERSATION_SUITE = "conversation"
ALL_SUITES = [STRUCTURE_SUITE, CONVERSATION_SUITE]

SYSTEM_TRADITIONAL = "traditional_rag"
SYSTEM_NEUROMEM_IN_MEMORY = "neuromem_in_memory"
SYSTEM_NEUROMEM_PERSISTENT = "neuromem_persistent"
STRUCTURE_SYSTEMS = [
    SYSTEM_TRADITIONAL,
    SYSTEM_NEUROMEM_IN_MEMORY,
    SYSTEM_NEUROMEM_PERSISTENT,
]

SYSTEM_TRADITIONAL_SCOPED = "traditional_scoped"
SYSTEM_NEUROMEM_SCOPED_IN_MEMORY = "neuromem_scoped_in_memory"
SYSTEM_NEUROMEM_SCOPED_PERSISTENT = "neuromem_scoped_persistent"
CONVERSATION_SYSTEMS = [
    SYSTEM_TRADITIONAL_SCOPED,
    SYSTEM_NEUROMEM_SCOPED_IN_MEMORY,
    SYSTEM_NEUROMEM_SCOPED_PERSISTENT,
]


@dataclass(frozen=True)
class ConversationCase:
    session_id: str
    decision_id: str
    decision_text: str
    followup_text: str
    handoff_query: str
    distractors: List[str]


DECISION_TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    (
        "choose sqlite with wal checkpoints for local project persistence",
        "keep the store file-backed so offline startup stays simple",
        "What did we decide for this workstream?",
    ),
    (
        "use candidate pruning before final ranking in the retrieval pipeline",
        "the search path should pre-filter before the expensive pass",
        "What did we decide for this workstream?",
    ),
    (
        "ship the local stack with docker compose for proxy and mcp services",
        "the whole setup should launch in one local stack command",
        "What did we decide for this workstream?",
    ),
    (
        "standardize on an openai compatible proxy so existing clients can reuse base_url",
        "we should keep current sdk workflows and only repoint the endpoint",
        "What did we decide for this workstream?",
    ),
    (
        "store conversation records as episodic memory and durable facts as semantic memory",
        "temporary chat turns and reusable knowledge should not share one bucket",
        "What did we decide for this workstream?",
    ),
    (
        "record observability metrics in json and prometheus formats",
        "the runtime should expose both machine readable and scraped monitoring output",
        "What did we decide for this workstream?",
    ),
    (
        "default to a local lexical embedder for local first development",
        "the offline path should work without calling an external embedding api",
        "What did we decide for this workstream?",
    ),
    (
        "separate project sessions with explicit scope tags to avoid leakage",
        "memories from unrelated projects must not bleed into each other",
        "What did we decide for this workstream?",
    ),
)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return float(lower_value + (upper_value - lower_value) * (position - lower))


def reciprocal_rank_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    for index, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    return 1.0 if any(item_id in relevant_ids for item_id in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    window = ranked_ids[:k]
    if not window:
        return 0.0
    hits = sum(1 for item_id in window if item_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for item_id in ranked_ids[:k] if item_id in relevant_ids)
    return hits / min(k, len(relevant_ids))


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for index, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        value = float(values[0])
        return (value, value)

    rng = random.Random(seed)
    means: List[float] = []
    values_list = list(values)
    sample_size = len(values_list)
    for _ in range(iterations):
        sample = [values_list[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(statistics.fmean(sample))

    alpha = (1.0 - confidence) / 2.0
    return (
        percentile(means, alpha),
        percentile(means, 1.0 - alpha),
    )


def sign_test_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(
        math.comb(trials, k)
        for k in range(max(wins, losses), trials + 1)
    ) / (2 ** trials)
    return min(1.0, 2.0 * tail)


def paired_advantage_summary(
    system_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    higher_is_better: bool,
    seed: int,
) -> Dict[str, float]:
    paired = list(zip(system_values, baseline_values))
    if not paired:
        return {
            "mean_system": 0.0,
            "mean_baseline": 0.0,
            "mean_advantage": 0.0,
            "advantage_ci_low": 0.0,
            "advantage_ci_high": 0.0,
            "mean_ratio": 0.0,
            "win_rate": 0.0,
            "tie_rate": 1.0,
            "p_value_sign_test": 1.0,
        }

    adjusted_diffs: List[float] = []
    ratios: List[float] = []
    wins = 0
    losses = 0
    ties = 0

    for system_value, baseline_value in paired:
        if higher_is_better:
            diff = system_value - baseline_value
            if baseline_value != 0.0:
                ratios.append(system_value / baseline_value)
        else:
            diff = baseline_value - system_value
            if system_value != 0.0:
                ratios.append(baseline_value / system_value)

        adjusted_diffs.append(diff)
        if diff > 1e-12:
            wins += 1
        elif diff < -1e-12:
            losses += 1
        else:
            ties += 1

    ci_low, ci_high = bootstrap_mean_ci(adjusted_diffs, seed=seed)
    total = len(paired)
    return {
        "mean_system": statistics.fmean(system_values),
        "mean_baseline": statistics.fmean(baseline_values),
        "mean_advantage": statistics.fmean(adjusted_diffs),
        "advantage_ci_low": ci_low,
        "advantage_ci_high": ci_high,
        "mean_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "p_value_sign_test": sign_test_pvalue(wins, losses),
    }


class ScopedTraditionalRAGSystem(TraditionalRAGSystem):
    """Traditional vector-store baseline with explicit session filtering."""

    def add_document(self, content: str, metadata: Dict[str, Any] | None = None) -> str:
        payload = dict(metadata or {})
        return super().add_document(content, metadata=payload)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        if not self.vector_store:
            return []

        similarities = []
        for index, stored_embedding in enumerate(self.vector_store):
            node_id = self.node_ids[index]
            node = self.memory_nodes[node_id]
            if session_id is not None and node.metadata.get("session_id") != session_id:
                continue

            dot_product = np.dot(query_embedding[0], stored_embedding)
            norm_query = np.linalg.norm(query_embedding[0])
            norm_stored = np.linalg.norm(stored_embedding)
            if norm_query == 0 or norm_stored == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_stored)
            similarities.append((node, float(similarity)))

        similarities.sort(key=lambda item: item[1], reverse=True)
        for node, _ in similarities[:top_k]:
            self.access_frequency[node.id] = self.access_frequency.get(node.id, 1) + 1
        return similarities[:top_k]


def _reset_access_frequency(system: Any):
    for key in list(system.access_frequency.keys()):
        system.access_frequency[key] = 1


def build_conversation_cases(size: int, rng: random.Random) -> List[ConversationCase]:
    session_count = max(4, min(10, max(1, size // 48)))
    records_per_session = max(12, size // session_count)
    cases_per_session = max(3, min(len(DECISION_TEMPLATES), records_per_session // 4))
    distractors_per_case = max(2, (records_per_session // cases_per_session) - 2)

    cases: List[ConversationCase] = []
    for session_index in range(session_count):
        session_id = f"project_{session_index:02d}"
        template_indices = list(range(len(DECISION_TEMPLATES)))
        rng.shuffle(template_indices)
        for case_index in range(cases_per_session):
            template = DECISION_TEMPLATES[template_indices[case_index]]
            decision_text = (
                f"{session_id} decision_{case_index} architecture note: {template[0]} "
                f"constraint_{session_index % 5} milestone_{case_index % 4}"
            )
            followup_text = (
                f"{session_id} followup_{case_index} recent discussion: {template[1]} "
                f"handoff_marker_{case_index} checkpoint_{session_index % 3}"
            )
            handoff_query = (
                f"{template[2]} session={session_id} workstream_{case_index}"
            )
            distractors = [
                (
                    f"{session_id} distractor_{case_index}_{distractor_index} progress update: "
                    f"general project cleanup, review notes, backlog sync, workstream_{distractor_index % 3}"
                )
                for distractor_index in range(distractors_per_case)
            ]
            cases.append(
                ConversationCase(
                    session_id=session_id,
                    decision_id=f"{session_id}_decision_{case_index}",
                    decision_text=decision_text,
                    followup_text=followup_text,
                    handoff_query=handoff_query,
                    distractors=distractors,
                )
            )
    return cases


def build_structure_system(
    system_name: str,
    *,
    capacity: int,
    temp_dir: Path,
    embedder: TextEmbedder,
) -> Tuple[Any, Optional[Path]]:
    if system_name == SYSTEM_TRADITIONAL:
        return TraditionalRAGSystem(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_IN_MEMORY:
        return InMemoryNeuromemManager(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_PERSISTENT:
        db_path = temp_dir / "structure_benchmark.db"
        return MemoryManager(capacity=capacity, db_path=str(db_path), embedder=embedder), db_path
    raise ValueError(f"Unsupported structure system: {system_name}")


def build_conversation_system(
    system_name: str,
    *,
    capacity: int,
    temp_dir: Path,
    embedder: TextEmbedder,
) -> Tuple[Any, Optional[Path]]:
    if system_name == SYSTEM_TRADITIONAL_SCOPED:
        return ScopedTraditionalRAGSystem(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_SCOPED_IN_MEMORY:
        return InMemoryNeuromemManager(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_SCOPED_PERSISTENT:
        db_path = temp_dir / "conversation_benchmark.db"
        return MemoryManager(capacity=capacity, db_path=str(db_path), embedder=embedder), db_path
    raise ValueError(f"Unsupported conversation system: {system_name}")


def _structure_ranked_doc_ids(
    system_name: str,
    results: Sequence[Any],
    doc_lookup: Dict[str, str],
) -> List[str]:
    contents = result_contents(system_name, results)
    return [doc_lookup[content] for content in contents if content in doc_lookup]


def run_structure_trial(
    *,
    system_name: str,
    size: int,
    query_count: int,
    warmup_count: int,
    association_degree: int,
    embedding_backend: str,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    topic_count = max(8, min(32, size // 8 if size >= 8 else 1))
    corpus = build_corpus(size=size, topic_count=topic_count)
    association_pairs = build_association_pairs(corpus, topic_count=topic_count, degree=association_degree)
    association_lookup: Dict[str, List[str]] = {}
    for left_doc_id, right_doc_id in association_pairs:
        association_lookup.setdefault(left_doc_id, []).append(right_doc_id)
        association_lookup.setdefault(right_doc_id, []).append(left_doc_id)

    exact_queries = rng.sample(corpus, k=min(query_count, len(corpus)))
    topic_queries: List[Tuple[int, str]] = []
    topics = list(range(topic_count))
    rng.shuffle(topics)
    for topic in topics[: min(query_count, topic_count)]:
        topic_queries.append(
            (
                topic,
                f"keyword_primary_{topic} bridge_{topic}_{(topic + 1) % topic_count}",
            )
        )
    primed_queries = rng.sample(corpus, k=min(query_count, len(corpus)))

    with tempfile.TemporaryDirectory(prefix="neuromem-comprehensive-structure-") as tmp_dir_name:
        temp_dir = Path(tmp_dir_name)
        embedder = build_embedder(embedding_backend, corpus)
        system, db_path = build_structure_system(
            system_name,
            capacity=size * 2,
            temp_dir=temp_dir,
            embedder=embedder,
        )

        tracemalloc.start()
        gc.collect()
        base_current, _ = tracemalloc.get_traced_memory()

        doc_id_map: Dict[str, str] = {}
        content_to_doc_id = {item.content: item.doc_id for item in corpus}
        ingest_start = time.perf_counter_ns()
        gc.disable()
        try:
            for item in corpus:
                doc_id_map[item.doc_id] = ingest_item(system_name=system_name, system=system, item=item)
        finally:
            gc.enable()
        ingest_ns = time.perf_counter_ns() - ingest_start

        association_ns = 0
        if system_name != SYSTEM_TRADITIONAL:
            association_start = time.perf_counter_ns()
            gc.disable()
            try:
                for left_doc_id, right_doc_id in association_pairs:
                    system.associate(doc_id_map[left_doc_id], doc_id_map[right_doc_id], strength=1.0)
            finally:
                gc.enable()
            association_ns = time.perf_counter_ns() - association_start

        current_after_ingest, peak_after_ingest = tracemalloc.get_traced_memory()
        python_bytes_after_ingest = current_after_ingest - base_current
        python_peak_bytes = peak_after_ingest - base_current

        for item in exact_queries[: min(warmup_count, len(exact_queries))]:
            system.retrieve(item.content, top_k=10)
        _reset_access_frequency(system)

        exact_latencies_ns: List[int] = []
        exact_top1_hits = 0
        exact_top5_hits = 0
        exact_mrr_scores: List[float] = []
        for item in exact_queries:
            _reset_access_frequency(system)
            started_at = time.perf_counter_ns()
            results = system.retrieve(item.content, top_k=10)
            exact_latencies_ns.append(time.perf_counter_ns() - started_at)
            ranked_doc_ids = _structure_ranked_doc_ids(system_name, results, content_to_doc_id)
            relevant_ids = {item.doc_id}
            exact_top1_hits += int(ranked_doc_ids[:1] == [item.doc_id])
            exact_top5_hits += int(item.doc_id in ranked_doc_ids[:5])
            exact_mrr_scores.append(reciprocal_rank_at_k(ranked_doc_ids, relevant_ids, 10))

        topic_latencies_ns: List[int] = []
        topic_hit_at_5 = 0
        topic_precision_scores: List[float] = []
        topic_recall_scores: List[float] = []
        topic_ndcg_scores: List[float] = []
        relevant_by_topic = {
            topic: {item.doc_id for item in corpus if item.topic == topic}
            for topic in range(topic_count)
        }
        for expected_topic, query in topic_queries:
            _reset_access_frequency(system)
            started_at = time.perf_counter_ns()
            results = system.retrieve(query, top_k=10)
            topic_latencies_ns.append(time.perf_counter_ns() - started_at)
            ranked_doc_ids = _structure_ranked_doc_ids(system_name, results, content_to_doc_id)
            relevant_ids = relevant_by_topic[expected_topic]
            topic_hit_at_5 += int(hit_rate_at_k(ranked_doc_ids, relevant_ids, 5))
            topic_precision_scores.append(precision_at_k(ranked_doc_ids, relevant_ids, 5))
            topic_recall_scores.append(recall_at_k(ranked_doc_ids, relevant_ids, 10))
            topic_ndcg_scores.append(ndcg_at_k(ranked_doc_ids, relevant_ids, 10))

        unprimed_latencies_ns: List[int] = []
        primed_latencies_ns: List[int] = []
        unprimed_neighbor_recall_scores: List[float] = []
        primed_neighbor_recall_scores: List[float] = []
        primed_neighbor_mrr_scores: List[float] = []
        for item in primed_queries:
            related_doc_ids = association_lookup.get(item.doc_id, [])
            if not related_doc_ids:
                continue
            relevant_ids = set(related_doc_ids)

            _reset_access_frequency(system)
            started_at = time.perf_counter_ns()
            results = system.retrieve(item.content, top_k=10)
            unprimed_latencies_ns.append(time.perf_counter_ns() - started_at)
            ranked_doc_ids = [
                doc_id
                for doc_id in _structure_ranked_doc_ids(system_name, results, content_to_doc_id)
                if doc_id != item.doc_id
            ]
            unprimed_neighbor_recall_scores.append(recall_at_k(ranked_doc_ids, relevant_ids, 10))

            _reset_access_frequency(system)
            for _ in range(3):
                system.retrieve(item.content, top_k=1)
            started_at = time.perf_counter_ns()
            primed_results = system.retrieve(item.content, top_k=10)
            primed_latencies_ns.append(time.perf_counter_ns() - started_at)
            primed_ranked_doc_ids = [
                doc_id
                for doc_id in _structure_ranked_doc_ids(system_name, primed_results, content_to_doc_id)
                if doc_id != item.doc_id
            ]
            primed_neighbor_recall_scores.append(recall_at_k(primed_ranked_doc_ids, relevant_ids, 10))
            primed_neighbor_mrr_scores.append(
                reciprocal_rank_at_k(primed_ranked_doc_ids, relevant_ids, 10)
            )

        tracemalloc.stop()

        stats = system.get_statistics()
        checkpoint = getattr(getattr(system, "db", None), "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        db_file_bytes = database_footprint_bytes(db_path)

        exact_p95_ms = percentile([value / 1e6 for value in exact_latencies_ns], 0.95)
        topic_p95_ms = percentile([value / 1e6 for value in topic_latencies_ns], 0.95)
        primed_p95_ms = percentile([value / 1e6 for value in primed_latencies_ns], 0.95)

        return {
            "suite": STRUCTURE_SUITE,
            "system": system_name,
            "embedding_backend": embedding_backend,
            "size": size,
            "seed": seed,
            "topic_count": topic_count,
            "doc_count": len(corpus),
            "association_edge_count": len(association_pairs),
            "query_count": len(exact_queries),
            "warmup_count": min(warmup_count, len(exact_queries)),
            "ingest_time_s": ingest_ns / 1e9,
            "association_time_s": association_ns / 1e9,
            "total_build_time_s": (ingest_ns + association_ns) / 1e9,
            "ingest_docs_per_s": len(corpus) / (ingest_ns / 1e9) if ingest_ns else 0.0,
            "exact_retrieval_latency_ns": exact_latencies_ns,
            "topic_retrieval_latency_ns": topic_latencies_ns,
            "unprimed_retrieval_latency_ns": unprimed_latencies_ns,
            "primed_retrieval_latency_ns": primed_latencies_ns,
            "exact_top1_accuracy": exact_top1_hits / len(exact_queries) if exact_queries else 0.0,
            "exact_top5_accuracy": exact_top5_hits / len(exact_queries) if exact_queries else 0.0,
            "exact_mrr_at_10": statistics.fmean(exact_mrr_scores) if exact_mrr_scores else 0.0,
            "topic_hit_rate_at_5": topic_hit_at_5 / len(topic_queries) if topic_queries else 0.0,
            "topic_precision_at_5": statistics.fmean(topic_precision_scores) if topic_precision_scores else 0.0,
            "topic_recall_at_10": statistics.fmean(topic_recall_scores) if topic_recall_scores else 0.0,
            "topic_ndcg_at_10": statistics.fmean(topic_ndcg_scores) if topic_ndcg_scores else 0.0,
            "unprimed_neighbor_recall_at_10": (
                statistics.fmean(unprimed_neighbor_recall_scores)
                if unprimed_neighbor_recall_scores
                else 0.0
            ),
            "primed_neighbor_recall_at_10": (
                statistics.fmean(primed_neighbor_recall_scores)
                if primed_neighbor_recall_scores
                else 0.0
            ),
            "priming_lift_neighbor_recall_at_10": (
                statistics.fmean(primed_neighbor_recall_scores)
                - statistics.fmean(unprimed_neighbor_recall_scores)
                if primed_neighbor_recall_scores and unprimed_neighbor_recall_scores
                else 0.0
            ),
            "primed_neighbor_mrr_at_10": (
                statistics.fmean(primed_neighbor_mrr_scores)
                if primed_neighbor_mrr_scores
                else 0.0
            ),
            "exact_p95_ms": exact_p95_ms,
            "topic_p95_ms": topic_p95_ms,
            "primed_p95_ms": primed_p95_ms,
            "python_bytes_after_ingest": python_bytes_after_ingest,
            "python_peak_bytes": python_peak_bytes,
            "estimated_size_bytes": stats.get("estimated_size_bytes", 0),
            "db_file_bytes": db_file_bytes,
        }


def _conversation_store_document(
    system_name: str,
    system: Any,
    content: str,
    session_id: str,
    memory_type: MemoryType,
    tags: Sequence[str],
) -> str:
    if system_name == SYSTEM_TRADITIONAL_SCOPED:
        return system.add_document(
            content,
            metadata={"session_id": session_id, "tags": list(tags)},
        )
    return system.encode(content, memory_type, tags=list(tags))


def _conversation_retrieve(
    system_name: str,
    system: Any,
    query: str,
    session_id: str,
    top_k: int,
) -> Sequence[Any]:
    if system_name == SYSTEM_TRADITIONAL_SCOPED:
        return system.retrieve(query, top_k=top_k, session_id=session_id)
    return system.retrieve(
        query,
        top_k=top_k,
        context={"tags": [f"session:{session_id}"], "required_tags": [f"session:{session_id}"]},
    )


def _conversation_ranked_ids(system_name: str, results: Sequence[Any], content_to_case_id: Dict[str, str]) -> List[str]:
    if system_name == SYSTEM_TRADITIONAL_SCOPED:
        contents = [node.content for node, _score in results]
    else:
        contents = [node.content for node in results]
    return [content_to_case_id[content] for content in contents if content in content_to_case_id]


def run_conversation_trial(
    *,
    system_name: str,
    size: int,
    query_count: int,
    warmup_count: int,
    embedding_backend: str,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    cases = build_conversation_cases(size=size, rng=rng)
    corpus_items = [
        CorpusItem(
            doc_id=f"{case.decision_id}_decision",
            topic=index % max(1, min(32, len(cases))),
            content=case.decision_text,
            tags=[f"session:{case.session_id}", "kind:decision"],
        )
        for index, case in enumerate(cases)
    ]
    corpus_items.extend(
        CorpusItem(
            doc_id=f"{case.decision_id}_followup",
            topic=index % max(1, min(32, len(cases))),
            content=case.followup_text,
            tags=[f"session:{case.session_id}", "kind:followup"],
        )
        for index, case in enumerate(cases)
    )
    for case in cases:
        for distractor_index, distractor in enumerate(case.distractors):
            corpus_items.append(
                CorpusItem(
                    doc_id=f"{case.decision_id}_distractor_{distractor_index}",
                    topic=distractor_index,
                    content=distractor,
                    tags=[f"session:{case.session_id}", "kind:distractor"],
                )
            )

    with tempfile.TemporaryDirectory(prefix="neuromem-comprehensive-conversation-") as tmp_dir_name:
        temp_dir = Path(tmp_dir_name)
        embedder = build_embedder(embedding_backend, corpus_items)
        system, db_path = build_conversation_system(
            system_name,
            capacity=max(len(corpus_items) * 2, size * 2),
            temp_dir=temp_dir,
            embedder=embedder,
        )

        tracemalloc.start()
        gc.collect()
        base_current, _ = tracemalloc.get_traced_memory()

        stored_ids: Dict[str, str] = {}
        content_to_case_id: Dict[str, str] = {}
        ingest_start = time.perf_counter_ns()
        gc.disable()
        try:
            for case in cases:
                decision_id = _conversation_store_document(
                    system_name,
                    system,
                    case.decision_text,
                    case.session_id,
                    MemoryType.SEMANTIC,
                    [f"session:{case.session_id}", "kind:decision", f"decision:{case.decision_id}"],
                )
                followup_id = _conversation_store_document(
                    system_name,
                    system,
                    case.followup_text,
                    case.session_id,
                    MemoryType.EPISODIC,
                    [f"session:{case.session_id}", "kind:followup", f"decision:{case.decision_id}"],
                )
                stored_ids[f"{case.decision_id}:decision"] = decision_id
                stored_ids[f"{case.decision_id}:followup"] = followup_id
                content_to_case_id[case.decision_text] = f"{case.decision_id}:decision"
                content_to_case_id[case.followup_text] = f"{case.decision_id}:followup"

                for distractor_index, distractor in enumerate(case.distractors):
                    distractor_id = _conversation_store_document(
                        system_name,
                        system,
                        distractor,
                        case.session_id,
                        MemoryType.EPISODIC,
                        [f"session:{case.session_id}", "kind:distractor"],
                    )
                    stored_ids[f"{case.decision_id}:distractor:{distractor_index}"] = distractor_id
                    content_to_case_id[distractor] = f"{case.decision_id}:distractor:{distractor_index}"

                if system_name in {SYSTEM_NEUROMEM_SCOPED_IN_MEMORY, SYSTEM_NEUROMEM_SCOPED_PERSISTENT}:
                    system.associate(decision_id, followup_id, strength=1.0)
        finally:
            gc.enable()
        ingest_ns = time.perf_counter_ns() - ingest_start

        current_after_ingest, peak_after_ingest = tracemalloc.get_traced_memory()
        python_bytes_after_ingest = current_after_ingest - base_current
        python_peak_bytes = peak_after_ingest - base_current

        evaluated_cases = cases[: min(query_count, len(cases))]
        for case in evaluated_cases[: min(warmup_count, len(evaluated_cases))]:
            _conversation_retrieve(system_name, system, case.followup_text, case.session_id, top_k=1)
        _reset_access_frequency(system)

        unprimed_latencies_ns: List[int] = []
        primed_latencies_ns: List[int] = []
        unprimed_hit_scores: List[float] = []
        primed_hit_scores: List[float] = []
        primed_mrr_scores: List[float] = []

        for case in evaluated_cases:
            relevant_id = {f"{case.decision_id}:decision"}

            _reset_access_frequency(system)
            started_at = time.perf_counter_ns()
            unprimed_results = _conversation_retrieve(
                system_name,
                system,
                case.handoff_query,
                case.session_id,
                top_k=5,
            )
            unprimed_latencies_ns.append(time.perf_counter_ns() - started_at)
            unprimed_ranked_ids = _conversation_ranked_ids(system_name, unprimed_results, content_to_case_id)
            unprimed_hit_scores.append(hit_rate_at_k(unprimed_ranked_ids, relevant_id, 5))

            _reset_access_frequency(system)
            for _ in range(3):
                _conversation_retrieve(system_name, system, case.followup_text, case.session_id, top_k=1)
            started_at = time.perf_counter_ns()
            primed_results = _conversation_retrieve(
                system_name,
                system,
                case.handoff_query,
                case.session_id,
                top_k=5,
            )
            primed_latencies_ns.append(time.perf_counter_ns() - started_at)
            primed_ranked_ids = _conversation_ranked_ids(system_name, primed_results, content_to_case_id)
            primed_hit_scores.append(hit_rate_at_k(primed_ranked_ids, relevant_id, 5))
            primed_mrr_scores.append(reciprocal_rank_at_k(primed_ranked_ids, relevant_id, 5))

        tracemalloc.stop()

        stats = system.get_statistics()
        checkpoint = getattr(getattr(system, "db", None), "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        db_file_bytes = database_footprint_bytes(db_path)

        return {
            "suite": CONVERSATION_SUITE,
            "system": system_name,
            "embedding_backend": embedding_backend,
            "size": size,
            "seed": seed,
            "session_count": len({case.session_id for case in cases}),
            "case_count": len(cases),
            "evaluated_case_count": len(evaluated_cases),
            "ingest_time_s": ingest_ns / 1e9,
            "unprimed_latency_ns": unprimed_latencies_ns,
            "primed_latency_ns": primed_latencies_ns,
            "handoff_unprimed_hit_at_5": (
                statistics.fmean(unprimed_hit_scores) if unprimed_hit_scores else 0.0
            ),
            "handoff_primed_hit_at_5": (
                statistics.fmean(primed_hit_scores) if primed_hit_scores else 0.0
            ),
            "handoff_primed_mrr_at_5": (
                statistics.fmean(primed_mrr_scores) if primed_mrr_scores else 0.0
            ),
            "handoff_priming_lift_at_5": (
                statistics.fmean(primed_hit_scores) - statistics.fmean(unprimed_hit_scores)
                if primed_hit_scores and unprimed_hit_scores
                else 0.0
            ),
            "unprimed_p95_ms": percentile([value / 1e6 for value in unprimed_latencies_ns], 0.95),
            "primed_p95_ms": percentile([value / 1e6 for value in primed_latencies_ns], 0.95),
            "python_bytes_after_ingest": python_bytes_after_ingest,
            "python_peak_bytes": python_peak_bytes,
            "estimated_size_bytes": stats.get("estimated_size_bytes", 0),
            "db_file_bytes": db_file_bytes,
        }


def aggregate_results(
    raw_trials: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    sizes: Sequence[int],
    *,
    scalar_metrics: Sequence[str],
    latency_metrics: Dict[str, str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for size in sizes:
        size_key = str(size)
        summary[size_key] = {}
        for system in systems:
            matching = [
                trial
                for trial in raw_trials
                if trial["system"] == system and trial["size"] == size
            ]
            if not matching:
                continue

            system_summary: Dict[str, Any] = {}
            for metric in scalar_metrics:
                system_summary[metric] = summarize_samples([float(trial[metric]) for trial in matching])

            for summary_key, raw_key in latency_metrics.items():
                combined = [
                    float(value)
                    for trial in matching
                    for value in trial.get(raw_key, [])
                ]
                system_summary[summary_key] = summarize_samples(combined)

            summary[size_key][system] = system_summary
    return summary


def build_pairwise_summary(
    raw_trials: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    sizes: Sequence[int],
    *,
    baseline: str,
    metric_directions: Dict[str, bool],
    seed: int,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for size in sizes:
        size_key = str(size)
        output[size_key] = {}
        baseline_trials = {
            int(trial["seed"]): trial
            for trial in raw_trials
            if trial["system"] == baseline and trial["size"] == size
        }
        for system in systems:
            if system == baseline:
                continue
            system_trials = {
                int(trial["seed"]): trial
                for trial in raw_trials
                if trial["system"] == system and trial["size"] == size
            }
            common_seeds = sorted(set(baseline_trials) & set(system_trials))
            if not common_seeds:
                continue

            comparison_key = f"{system}_vs_{baseline}"
            output[size_key][comparison_key] = {}
            for metric, higher_is_better in metric_directions.items():
                system_values = [float(system_trials[current_seed][metric]) for current_seed in common_seeds]
                baseline_values = [float(baseline_trials[current_seed][metric]) for current_seed in common_seeds]
                output[size_key][comparison_key][metric] = paired_advantage_summary(
                    system_values,
                    baseline_values,
                    higher_is_better=higher_is_better,
                    seed=seed + size,
                )
    return output


def build_markdown_report(results: Dict[str, Any]) -> str:
    lines = [
        "# Comprehensive Benchmark",
        "",
        "## Methodology",
        "- Real package implementations are benchmarked under a shared embedding backend.",
        "- Each condition runs in an isolated subprocess with deterministic thread settings.",
        "- Structural retrieval is measured separately from long-conversation handoff recall.",
        "- Pairwise summaries use bootstrap confidence intervals and a sign-test p-value.",
        "",
        "## Configuration",
        f"- Embedding backend: {results['config']['embedding_backend']}",
        f"- Sizes: {', '.join(str(size) for size in results['config']['sizes'])}",
        f"- Trials per condition: {results['config']['trials']}",
        f"- Query count per structural trial: {results['config']['query_count']}",
        f"- Warmup count per trial: {results['config']['warmup_count']}",
        f"- Association degree: {results['config']['association_degree']}",
        "",
    ]

    structure = results.get("structure_suite", {})
    if structure:
        lines.extend(
            [
                "## Structure Suite",
                "Measures exact retrieval, clustered-topic retrieval, and primed associative recall.",
                "",
            ]
        )
        for size in results["config"]["sizes"]:
            size_key = str(size)
            lines.append(f"### Corpus Size {size}")
            for system, metrics in structure["summary"].get(size_key, {}).items():
                lines.append(
                    "- `{0}`: exact top1 {1:.3f}, exact MRR@10 {2:.3f}, "
                    "topic nDCG@10 {3:.3f}, primed neighbor recall@10 {4:.3f}, "
                    "priming lift {5:.3f}, exact p95 {6:.3f} ms".format(
                        system,
                        metrics["exact_top1_accuracy"]["mean"],
                        metrics["exact_mrr_at_10"]["mean"],
                        metrics["topic_ndcg_at_10"]["mean"],
                        metrics["primed_neighbor_recall_at_10"]["mean"],
                        metrics["priming_lift_neighbor_recall_at_10"]["mean"],
                        metrics["exact_retrieval_latency_ns"]["p95"] / 1e6,
                    )
                )
            lines.append("")

    conversation = results.get("conversation_suite", {})
    if conversation:
        lines.extend(
            [
                "## Conversation Suite",
                "Measures project-scoped long-horizon handoff recall after a priming turn.",
                "",
            ]
        )
        for size in results["config"]["sizes"]:
            size_key = str(size)
            lines.append(f"### Approximate Record Budget {size}")
            for system, metrics in conversation["summary"].get(size_key, {}).items():
                lines.append(
                    "- `{0}`: unprimed handoff hit@5 {1:.3f}, primed handoff hit@5 {2:.3f}, "
                    "handoff priming lift {3:.3f}, primed MRR@5 {4:.3f}, primed p95 {5:.3f} ms".format(
                        system,
                        metrics["handoff_unprimed_hit_at_5"]["mean"],
                        metrics["handoff_primed_hit_at_5"]["mean"],
                        metrics["handoff_priming_lift_at_5"]["mean"],
                        metrics["handoff_primed_mrr_at_5"]["mean"],
                        metrics["primed_latency_ns"]["p95"] / 1e6,
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## Caveats",
            "- Shared lexical or TF-IDF embeddings keep the comparison focused on memory structure; they do not prove semantic generalization against a stronger encoder.",
            "- Primed handoff recall is deliberately a memory-centric workload: it tests whether a recent follow-up can steer retrieval back to an associated earlier decision.",
            "- Persistent NeuroMem variants include a real persistence cost that the in-memory baselines do not bear.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unavailable"


def run_worker_from_args(args: argparse.Namespace):
    if args.suite == STRUCTURE_SUITE:
        result = run_structure_trial(
            system_name=args.system,
            size=args.size,
            query_count=args.query_count,
            warmup_count=args.warmup_count,
            association_degree=args.association_degree,
            embedding_backend=args.embedding_backend,
            seed=args.seed,
        )
    elif args.suite == CONVERSATION_SUITE:
        result = run_conversation_trial(
            system_name=args.system,
            size=args.size,
            query_count=args.query_count,
            warmup_count=args.warmup_count,
            embedding_backend=args.embedding_backend,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported suite: {args.suite}")

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


def _launch_trials(
    *,
    suite_name: str,
    systems: Sequence[str],
    sizes: Sequence[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    raw_trials: List[Dict[str, Any]] = []
    for size in sizes:
        for system in systems:
            for trial_index in range(args.trials):
                seed = args.seed + size * 1000 + trial_index
                cmd = [
                    sys.executable,
                    "-m",
                    "neuromem.experiments.comprehensive_benchmark",
                    "--worker",
                    "--suite",
                    suite_name,
                    "--system",
                    system,
                    "--size",
                    str(size),
                    "--query-count",
                    str(args.query_count),
                    "--warmup-count",
                    str(args.warmup_count),
                    "--association-degree",
                    str(args.association_degree),
                    "--embedding-backend",
                    args.embedding_backend,
                    "--seed",
                    str(seed),
                ]
                env = os.environ.copy()
                env.update(THREAD_ENV)
                completed = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                )
                raw_trials.append(json.loads(completed.stdout))
    return raw_trials


def run_suite_from_args(args: argparse.Namespace):
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config": {
            "sizes": list(args.sizes),
            "trials": args.trials,
            "query_count": args.query_count,
            "warmup_count": args.warmup_count,
            "association_degree": args.association_degree,
            "embedding_backend": args.embedding_backend,
            "suites": list(args.suites),
            "seed": args.seed,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scikit_learn_version": _safe_package_version("sklearn"),
            "thread_env": THREAD_ENV,
        },
    }

    if STRUCTURE_SUITE in args.suites:
        raw_trials = _launch_trials(
            suite_name=STRUCTURE_SUITE,
            systems=STRUCTURE_SYSTEMS,
            sizes=args.sizes,
            args=args,
        )
        scalar_metrics = [
            "ingest_time_s",
            "association_time_s",
            "total_build_time_s",
            "ingest_docs_per_s",
            "exact_top1_accuracy",
            "exact_top5_accuracy",
            "exact_mrr_at_10",
            "topic_hit_rate_at_5",
            "topic_precision_at_5",
            "topic_recall_at_10",
            "topic_ndcg_at_10",
            "unprimed_neighbor_recall_at_10",
            "primed_neighbor_recall_at_10",
            "priming_lift_neighbor_recall_at_10",
            "primed_neighbor_mrr_at_10",
            "exact_p95_ms",
            "topic_p95_ms",
            "primed_p95_ms",
            "python_bytes_after_ingest",
            "python_peak_bytes",
            "estimated_size_bytes",
            "db_file_bytes",
        ]
        latency_metrics = {
            "exact_retrieval_latency_ns": "exact_retrieval_latency_ns",
            "topic_retrieval_latency_ns": "topic_retrieval_latency_ns",
            "unprimed_retrieval_latency_ns": "unprimed_retrieval_latency_ns",
            "primed_retrieval_latency_ns": "primed_retrieval_latency_ns",
        }
        results["structure_suite"] = {
            "systems": list(STRUCTURE_SYSTEMS),
            "raw_trials": raw_trials,
            "summary": aggregate_results(
                raw_trials,
                STRUCTURE_SYSTEMS,
                args.sizes,
                scalar_metrics=scalar_metrics,
                latency_metrics=latency_metrics,
            ),
            "pairwise": build_pairwise_summary(
                raw_trials,
                STRUCTURE_SYSTEMS,
                args.sizes,
                baseline=SYSTEM_TRADITIONAL,
                metric_directions={
                    "exact_top1_accuracy": True,
                    "exact_mrr_at_10": True,
                    "topic_ndcg_at_10": True,
                    "primed_neighbor_recall_at_10": True,
                    "priming_lift_neighbor_recall_at_10": True,
                    "exact_p95_ms": False,
                    "topic_p95_ms": False,
                    "primed_p95_ms": False,
                    "python_bytes_after_ingest": False,
                },
                seed=args.seed,
            ),
        }

    if CONVERSATION_SUITE in args.suites:
        raw_trials = _launch_trials(
            suite_name=CONVERSATION_SUITE,
            systems=CONVERSATION_SYSTEMS,
            sizes=args.sizes,
            args=args,
        )
        scalar_metrics = [
            "ingest_time_s",
            "handoff_unprimed_hit_at_5",
            "handoff_primed_hit_at_5",
            "handoff_primed_mrr_at_5",
            "handoff_priming_lift_at_5",
            "unprimed_p95_ms",
            "primed_p95_ms",
            "python_bytes_after_ingest",
            "python_peak_bytes",
            "estimated_size_bytes",
            "db_file_bytes",
        ]
        latency_metrics = {
            "unprimed_latency_ns": "unprimed_latency_ns",
            "primed_latency_ns": "primed_latency_ns",
        }
        results["conversation_suite"] = {
            "systems": list(CONVERSATION_SYSTEMS),
            "raw_trials": raw_trials,
            "summary": aggregate_results(
                raw_trials,
                CONVERSATION_SYSTEMS,
                args.sizes,
                scalar_metrics=scalar_metrics,
                latency_metrics=latency_metrics,
            ),
            "pairwise": build_pairwise_summary(
                raw_trials,
                CONVERSATION_SYSTEMS,
                args.sizes,
                baseline=SYSTEM_TRADITIONAL_SCOPED,
                metric_directions={
                    "handoff_unprimed_hit_at_5": True,
                    "handoff_primed_hit_at_5": True,
                    "handoff_primed_mrr_at_5": True,
                    "handoff_priming_lift_at_5": True,
                    "unprimed_p95_ms": False,
                    "primed_p95_ms": False,
                    "python_bytes_after_ingest": False,
                },
                seed=args.seed + 17,
            ),
        }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(build_markdown_report(results), encoding="utf-8")

    print(f"Wrote JSON results to {output_path}")
    print(f"Wrote Markdown summary to {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comprehensive multi-workload benchmark")
    parser.add_argument(
        "--embedding-backend",
        default=EMBEDDING_LEXICAL_HASHING,
        choices=ALL_EMBEDDING_BACKENDS,
        help="Shared embedding backend used by all systems in the benchmark",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=ALL_SUITES,
        choices=ALL_SUITES,
        help="Benchmark suites to run",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 512, 1024],
        help="Approximate corpus sizes / record budgets to test",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per condition")
    parser.add_argument("--query-count", type=int, default=64, help="Queries or evaluated cases per trial")
    parser.add_argument("--warmup-count", type=int, default=16, help="Warmup queries per trial")
    parser.add_argument(
        "--association-degree",
        type=int,
        default=3,
        help="Direct neighbors linked per item within a topic for the structure suite",
    )
    parser.add_argument("--seed", type=int, default=20260331, help="Base RNG seed")
    parser.add_argument(
        "--output",
        default="benchmark_results/comprehensive_benchmark.json",
        help="Path to the JSON output file",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=ALL_SUITES, help=argparse.SUPPRESS)
    parser.add_argument("--system", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        if not args.suite or not args.system or not args.size:
            parser.error("--worker requires --suite, --system, and --size")
        run_worker_from_args(args)
        return
    run_suite_from_args(args)


if __name__ == "__main__":
    main()
