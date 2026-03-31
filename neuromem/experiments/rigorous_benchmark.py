"""
Rigorous efficiency benchmark for NeuroMem versus a traditional memory baseline.

Methodological choices:
- Benchmarks use the real package implementations instead of mock classes.
- Trials run in isolated subprocesses to reduce cross-trial contamination.
- Environment is pinned for reproducibility (`PYTHONHASHSEED=0`, single-thread BLAS).
- All systems share the same embedding backend during a run so the measured
  difference is primarily memory structure cost rather than embedding variance.
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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.memory_manager import MemoryManager, MemoryType, SpikingNeuralNetwork
from ..core.text_embeddings import LexicalHashingEmbedder, TfidfEmbedder, TextEmbedder
from ..core.traditional_rag import TraditionalRAGSystem


THREAD_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

SYSTEM_TRADITIONAL = "traditional_rag"
SYSTEM_NEUROMEM_IN_MEMORY = "neuromem_in_memory"
SYSTEM_NEUROMEM_PERSISTENT = "neuromem_persistent"
ALL_SYSTEMS = [
    SYSTEM_TRADITIONAL,
    SYSTEM_NEUROMEM_IN_MEMORY,
    SYSTEM_NEUROMEM_PERSISTENT,
]
EMBEDDING_LEXICAL_HASHING = "lexical_hashing"
EMBEDDING_TFIDF = "tfidf"
ALL_EMBEDDING_BACKENDS = [EMBEDDING_LEXICAL_HASHING, EMBEDDING_TFIDF]


@dataclass(frozen=True)
class CorpusItem:
    doc_id: str
    topic: int
    content: str
    tags: List[str]


class NullMemoryDatabase:
    """No-op persistence layer used to isolate structural costs."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class InMemoryNeuromemManager(MemoryManager):
    """MemoryManager variant with persistence removed for fair structural timing."""

    def __init__(self, capacity: int = 10000, embedder: TextEmbedder | None = None):
        self.capacity = capacity
        self.memory_nodes = {}
        self.connections = {}
        self.working_memory_buffer = []
        self.long_term_memory = {}
        self.access_frequency = {}
        self.snn = SpikingNeuralNetwork()
        self.current_context = {}
        self.db = NullMemoryDatabase()
        self.embedder = embedder or LexicalHashingEmbedder()
        self._initialize_runtime_state()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    lower_value = values[lower]
    upper_value = values[upper]
    return float(lower_value + (upper_value - lower_value) * (position - lower))


def summarize_samples(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "ci95_half_width": 0.0,
        }

    if len(values) == 1:
        value = float(values[0])
        return {
            "mean": value,
            "stdev": 0.0,
            "min": value,
            "max": value,
            "p50": value,
            "p95": value,
            "ci95_half_width": 0.0,
        }

    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    ci95 = 1.96 * stdev / math.sqrt(len(values))
    return {
        "mean": mean,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "ci95_half_width": ci95,
    }


def build_corpus(size: int, topic_count: int) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    for index in range(size):
        topic = index % topic_count
        cluster = index // topic_count
        doc_id = f"doc_{index:05d}"
        content = (
            f"{doc_id} topic_{topic} cluster_{cluster} "
            f"keyword_primary_{topic} keyword_secondary_{cluster % 11} "
            f"bridge_{topic}_{(topic + 1) % topic_count} "
            f"fact_{index % 17} marker_exact_match_{index}"
        )
        tags = [
            f"topic_{topic}",
            f"cluster_{cluster}",
            f"bridge_{topic}_{(topic + 1) % topic_count}",
        ]
        items.append(CorpusItem(doc_id=doc_id, topic=topic, content=content, tags=tags))
    return items


def build_association_pairs(
    items: Sequence[CorpusItem],
    topic_count: int,
    degree: int,
) -> List[Tuple[str, str]]:
    by_topic: Dict[int, List[CorpusItem]] = {topic: [] for topic in range(topic_count)}
    for item in items:
        by_topic[item.topic].append(item)

    pairs: set[Tuple[str, str]] = set()
    for topic_items in by_topic.values():
        if len(topic_items) < 2:
            continue
        for idx, item in enumerate(topic_items):
            for offset in range(1, min(degree, len(topic_items) - 1) + 1):
                neighbor = topic_items[(idx + offset) % len(topic_items)]
                left, right = sorted((item.doc_id, neighbor.doc_id))
                pairs.add((left, right))
    return sorted(pairs)


def build_topic_queries(
    items: Sequence[CorpusItem],
    topic_count: int,
    sample_count: int,
    rng: random.Random,
) -> List[Tuple[int, str]]:
    topics = list(range(topic_count))
    rng.shuffle(topics)
    selected = topics[: min(sample_count, len(topics))]
    return [
        (
            topic,
            f"keyword_primary_{topic} bridge_{topic}_{(topic + 1) % topic_count}",
        )
        for topic in selected
    ]


def result_contents(system_name: str, results: Sequence[Any]) -> List[str]:
    if system_name == SYSTEM_TRADITIONAL:
        return [node.content for node, _score in results]
    return [node.content for node in results]


def build_embedder(embedding_backend: str, corpus: Sequence[CorpusItem]) -> TextEmbedder:
    if embedding_backend == EMBEDDING_LEXICAL_HASHING:
        return LexicalHashingEmbedder()
    if embedding_backend == EMBEDDING_TFIDF:
        return TfidfEmbedder.fit([item.content for item in corpus])
    raise ValueError(f"Unsupported embedding backend: {embedding_backend}")


def build_system(system_name: str, capacity: int, temp_dir: Path, embedder: TextEmbedder):
    if system_name == SYSTEM_TRADITIONAL:
        return TraditionalRAGSystem(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_IN_MEMORY:
        return InMemoryNeuromemManager(capacity=capacity, embedder=embedder), None
    if system_name == SYSTEM_NEUROMEM_PERSISTENT:
        db_path = temp_dir / "neuromem_benchmark.db"
        return MemoryManager(capacity=capacity, db_path=str(db_path), embedder=embedder), db_path
    raise ValueError(f"Unsupported system: {system_name}")


def database_footprint_bytes(db_path: Path | None) -> int:
    if db_path is None:
        return 0

    total_bytes = 0
    related_paths = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ]
    for path in related_paths:
        if path.exists():
            total_bytes += path.stat().st_size
    return total_bytes


def ingest_item(system_name: str, system: Any, item: CorpusItem) -> str:
    if system_name == SYSTEM_TRADITIONAL:
        return system.add_document(item.content, {"tags": item.tags, "topic": item.topic})
    return system.encode(item.content, MemoryType.SEMANTIC, tags=item.tags)


def run_trial(
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
    corpus_by_id = {item.doc_id: item for item in corpus}
    exact_queries = rng.sample(corpus, k=min(query_count, len(corpus)))
    primed_queries = rng.sample(corpus, k=min(query_count, len(corpus)))
    topic_queries = build_topic_queries(corpus, topic_count=topic_count, sample_count=query_count, rng=rng)

    with tempfile.TemporaryDirectory(prefix="neuromem-benchmark-") as tmp_dir_name:
        temp_dir = Path(tmp_dir_name)
        embedder = build_embedder(embedding_backend, corpus)
        system, db_path = build_system(
            system_name=system_name,
            capacity=size * 2,
            temp_dir=temp_dir,
            embedder=embedder,
        )

        tracemalloc.start()
        gc.collect()
        base_current, _ = tracemalloc.get_traced_memory()

        id_map: Dict[str, str] = {}
        ingest_start = time.perf_counter_ns()
        gc.disable()
        try:
            for item in corpus:
                id_map[item.doc_id] = ingest_item(system_name=system_name, system=system, item=item)
        finally:
            gc.enable()
        ingest_ns = time.perf_counter_ns() - ingest_start

        association_ns = 0
        if system_name != SYSTEM_TRADITIONAL:
            association_start = time.perf_counter_ns()
            gc.disable()
            try:
                for left_doc_id, right_doc_id in association_pairs:
                    system.associate(id_map[left_doc_id], id_map[right_doc_id], strength=1.0)
            finally:
                gc.enable()
            association_ns = time.perf_counter_ns() - association_start

        current_after_ingest, peak_after_ingest = tracemalloc.get_traced_memory()
        python_bytes_after_ingest = current_after_ingest - base_current
        python_peak_bytes = peak_after_ingest - base_current

        for item in exact_queries[: min(warmup_count, len(exact_queries))]:
            if system_name == SYSTEM_TRADITIONAL:
                system.retrieve(item.content, top_k=5)
            else:
                system.retrieve(item.content, top_k=5)

        for key in list(system.access_frequency.keys()):
            system.access_frequency[key] = 1

        exact_latencies_ns: List[int] = []
        exact_top1_hits = 0
        exact_top5_hits = 0
        for item in exact_queries:
            for key in list(system.access_frequency.keys()):
                system.access_frequency[key] = 1
            start_ns = time.perf_counter_ns()
            results = system.retrieve(item.content, top_k=5)
            latency_ns = time.perf_counter_ns() - start_ns
            exact_latencies_ns.append(latency_ns)

            contents = result_contents(system_name, results)
            if contents and contents[0] == item.content:
                exact_top1_hits += 1
            if item.content in contents[:5]:
                exact_top5_hits += 1

        topic_latencies_ns: List[int] = []
        topic_hit_at_5 = 0
        topic_precision_at_5_scores: List[float] = []
        content_to_topic = {item.content: item.topic for item in corpus}
        for expected_topic, query in topic_queries:
            for key in list(system.access_frequency.keys()):
                system.access_frequency[key] = 1
            start_ns = time.perf_counter_ns()
            results = system.retrieve(query, top_k=5)
            topic_latencies_ns.append(time.perf_counter_ns() - start_ns)
            contents = result_contents(system_name, results)
            returned_topics = [content_to_topic.get(content) for content in contents]
            same_topic_hits = sum(1 for topic in returned_topics if topic == expected_topic)
            if same_topic_hits > 0:
                topic_hit_at_5 += 1
            topic_precision_at_5_scores.append(same_topic_hits / 5 if contents else 0.0)

        neighbor_recall_scores: List[float] = []
        primed_latencies_ns: List[int] = []
        association_lookup: Dict[str, List[str]] = {}
        for left_doc_id, right_doc_id in association_pairs:
            association_lookup.setdefault(left_doc_id, []).append(right_doc_id)
            association_lookup.setdefault(right_doc_id, []).append(left_doc_id)

        prime_repetitions = 3
        for item in primed_queries:
            for key in list(system.access_frequency.keys()):
                system.access_frequency[key] = 1
            for _ in range(prime_repetitions):
                system.retrieve(item.content, top_k=1)

            start_ns = time.perf_counter_ns()
            results = system.retrieve(item.content, top_k=5)
            primed_latencies_ns.append(time.perf_counter_ns() - start_ns)

            related_doc_ids = association_lookup.get(item.doc_id, [])
            if not related_doc_ids:
                continue
            related_contents = {corpus_by_id[doc_id].content for doc_id in related_doc_ids}
            returned_contents = result_contents(system_name, results)
            returned_related = [content for content in returned_contents if content != item.content]
            hits = sum(1 for content in returned_related[:4] if content in related_contents)
            denominator = min(4, len(related_contents))
            neighbor_recall_scores.append(hits / denominator if denominator else 0.0)

        tracemalloc.stop()

        stats = system.get_statistics()
        checkpoint = getattr(getattr(system, "db", None), "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        db_file_bytes = database_footprint_bytes(db_path)

        return {
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
            "association_edges_per_s": (
                len(association_pairs) / (association_ns / 1e9)
                if association_ns and association_pairs
                else 0.0
            ),
            "exact_retrieval_latency_ns": exact_latencies_ns,
            "topic_retrieval_latency_ns": topic_latencies_ns,
            "primed_retrieval_latency_ns": primed_latencies_ns,
            "exact_top1_accuracy": exact_top1_hits / len(exact_queries) if exact_queries else 0.0,
            "exact_top5_accuracy": exact_top5_hits / len(exact_queries) if exact_queries else 0.0,
            "topic_hit_rate_at_5": topic_hit_at_5 / len(topic_queries) if topic_queries else 0.0,
            "topic_precision_at_5": (
                statistics.fmean(topic_precision_at_5_scores)
                if topic_precision_at_5_scores
                else 0.0
            ),
            "primed_neighbor_recall_at_5": (
                statistics.fmean(neighbor_recall_scores) if neighbor_recall_scores else 0.0
            ),
            "python_bytes_after_ingest": python_bytes_after_ingest,
            "python_peak_bytes": python_peak_bytes,
            "estimated_size_bytes": stats.get("estimated_size_bytes", 0),
            "db_file_bytes": db_file_bytes,
        }


def aggregate_results(
    raw_trials: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    sizes: Sequence[int],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    scalar_metrics = [
        "ingest_time_s",
        "association_time_s",
        "total_build_time_s",
        "ingest_docs_per_s",
        "association_edges_per_s",
        "exact_top1_accuracy",
        "exact_top5_accuracy",
        "topic_hit_rate_at_5",
        "topic_precision_at_5",
        "primed_neighbor_recall_at_5",
        "python_bytes_after_ingest",
        "python_peak_bytes",
        "estimated_size_bytes",
        "db_file_bytes",
    ]

    latency_metrics = {
        "exact_retrieval_latency_ns": "exact_retrieval_latency_ns",
        "topic_retrieval_latency_ns": "topic_retrieval_latency_ns",
        "primed_retrieval_latency_ns": "primed_retrieval_latency_ns",
    }

    for size in sizes:
        size_key = str(size)
        summary[size_key] = {}
        for system in systems:
            matching = [
                trial for trial in raw_trials if trial["system"] == system and trial["size"] == size
            ]
            if not matching:
                continue

            system_summary: Dict[str, Any] = {}
            for metric in scalar_metrics:
                values = [float(trial[metric]) for trial in matching]
                system_summary[metric] = summarize_samples(values)

            for result_key, raw_key in latency_metrics.items():
                combined = [
                    float(latency)
                    for trial in matching
                    for latency in trial.get(raw_key, [])
                ]
                system_summary[result_key] = summarize_samples(combined)

            summary[size_key][system] = system_summary
    return summary


def build_markdown_report(
    results: Dict[str, Any],
    systems: Sequence[str],
    sizes: Sequence[int],
) -> str:
    lines = [
        "# Rigorous Efficiency Benchmark",
        "",
        "## Methodology",
        "- Real package implementations were benchmarked, not mock classes.",
        "- Each trial ran in an isolated subprocess with `PYTHONHASHSEED=0` and single-thread BLAS settings.",
        "- Exact-match retrieval is used as a deterministic sanity check.",
        "- Topic queries test whether shared lexical/topic signals retrieve documents from the correct cluster.",
        "- A separate primed associative workload is kept as an exploratory signal for association dynamics.",
        "",
        "## Configuration",
        f"- Trials per condition: {results['config']['trials']}",
        f"- Sizes: {', '.join(str(size) for size in sizes)}",
        f"- Embedding backend: {results['config']['embedding_backend']}",
        f"- Query count per trial: {results['config']['query_count']}",
        f"- Warmup count per trial: {results['config']['warmup_count']}",
        f"- Association degree: {results['config']['association_degree']}",
        "",
        "## Summary",
    ]

    summary = results["summary"]
    for size in sizes:
        lines.append(f"### Corpus Size {size}")
        size_summary = summary.get(str(size), {})
        for system in systems:
            if system not in size_summary:
                continue
            metrics = size_summary[system]
            lines.append(f"- `{system}`")
            lines.append(
                "  "
                f"build {metrics['total_build_time_s']['mean']:.6f}s +/- "
                f"{metrics['total_build_time_s']['ci95_half_width']:.6f}s, "
                f"exact p95 {metrics['exact_retrieval_latency_ns']['p95'] / 1e6:.3f} ms, "
                f"top1 {metrics['exact_top1_accuracy']['mean']:.3f}, "
                f"topic-hit@5 {metrics['topic_hit_rate_at_5']['mean']:.3f}, "
                f"neighbor recall {metrics['primed_neighbor_recall_at_5']['mean']:.3f}, "
                f"heap {metrics['python_bytes_after_ingest']['mean']:.0f} B, "
                f"db {metrics['db_file_bytes']['mean']:.0f} B"
            )
        lines.append("")

    lines.extend(
        [
            "## Caveats",
            "- `tfidf` gives interpretable lexical retrieval, but it is still not a semantic encoder; paraphrase/generalization claims would require a stronger shared model.",
            "- The primed neighbor-recall metric is exploratory and is only useful for checking whether the association machinery changes ranking behavior at all.",
            "- TraditionalRAGSystem is currently an in-memory baseline without persistence; persistent NeuroMem measurements therefore include a feature cost that the baseline does not bear.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_worker_from_args(args: argparse.Namespace):
    result = run_trial(
        system_name=args.system,
        size=args.size,
        query_count=args.query_count,
        warmup_count=args.warmup_count,
        association_degree=args.association_degree,
        embedding_backend=args.embedding_backend,
        seed=args.seed,
    )
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


def run_suite_from_args(args: argparse.Namespace):
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_trials: List[Dict[str, Any]] = []
    for size in args.sizes:
        for system in args.systems:
            for trial_index in range(args.trials):
                seed = args.seed + size * 1000 + trial_index
                cmd = [
                    sys.executable,
                    "-m",
                    "neuromem.experiments.rigorous_benchmark",
                    "--worker",
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

    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except subprocess.CalledProcessError:
        git_commit = "unknown"

    results = {
        "config": {
            "systems": list(args.systems),
            "sizes": list(args.sizes),
            "trials": args.trials,
            "query_count": args.query_count,
            "warmup_count": args.warmup_count,
            "association_degree": args.association_degree,
            "embedding_backend": args.embedding_backend,
            "seed": args.seed,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scikit_learn_version": _safe_package_version("sklearn"),
            "thread_env": THREAD_ENV,
            "git_commit": git_commit,
        },
        "raw_trials": raw_trials,
        "summary": aggregate_results(raw_trials, args.systems, args.sizes),
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(
        build_markdown_report(results, args.systems, args.sizes),
        encoding="utf-8",
    )

    print(f"Wrote JSON results to {output_path}")
    print(f"Wrote Markdown summary to {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rigorous efficiency benchmark")
    parser.add_argument(
        "--embedding-backend",
        default=EMBEDDING_LEXICAL_HASHING,
        choices=ALL_EMBEDDING_BACKENDS,
        help="Shared embedding backend used by all systems in the benchmark",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=ALL_SYSTEMS,
        choices=ALL_SYSTEMS,
        help="Systems to benchmark",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
        help="Corpus sizes to test",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per condition")
    parser.add_argument("--query-count", type=int, default=64, help="Queries per trial")
    parser.add_argument("--warmup-count", type=int, default=16, help="Warmup queries per trial")
    parser.add_argument(
        "--association-degree",
        type=int,
        default=3,
        help="Direct neighbors linked per item within a topic",
    )
    parser.add_argument("--seed", type=int, default=20260331, help="Base RNG seed")
    parser.add_argument(
        "--output",
        default="benchmark_results/rigorous_efficiency_benchmark.json",
        help="Path to the JSON output file",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--system", choices=ALL_SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    return parser


def _safe_package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unavailable"


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        if not args.system or not args.size:
            parser.error("--worker requires --system and --size")
        run_worker_from_args(args)
        return
    run_suite_from_args(args)


if __name__ == "__main__":
    main()
