"""
Generate visualizations from rigorous benchmark JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = "benchmark_results/rigorous_efficiency_benchmark_tfidf_optimized.json"
SYSTEM_ORDER = [
    "traditional_rag",
    "neuromem_in_memory",
    "neuromem_persistent",
]
SYSTEM_LABELS = {
    "traditional_rag": "Traditional RAG",
    "neuromem_in_memory": "NeuroMem In-Memory",
    "neuromem_persistent": "NeuroMem Persistent",
}
SYSTEM_COLORS = {
    "traditional_rag": "#D65A31",
    "neuromem_in_memory": "#1F6FEB",
    "neuromem_persistent": "#2E8B57",
}


def latest_result_path(base_dir: Path) -> Path:
    results_dir = base_dir / "benchmark_results"
    preferred = results_dir / "rigorous_efficiency_benchmark_tfidf_optimized.json"
    if preferred.exists():
        return preferred

    candidates = sorted(results_dir.glob("rigorous_efficiency_benchmark*.json"))
    if not candidates:
        raise FileNotFoundError("No benchmark JSON files found under benchmark_results/")
    return candidates[-1]


def load_results(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_sizes(summary: Dict[str, Dict]) -> List[int]:
    return sorted(int(size) for size in summary.keys())


def available_systems(summary: Dict[str, Dict]) -> List[str]:
    present = set()
    for metrics in summary.values():
        present.update(metrics.keys())
    return [system for system in SYSTEM_ORDER if system in present]


def metric_series(
    summary: Dict[str, Dict],
    sizes: Sequence[int],
    system: str,
    metric: str,
    reducer: str = "mean",
    scale: float = 1.0,
) -> List[float]:
    values = []
    for size in sizes:
        metric_summary = summary[str(size)][system][metric]
        values.append(float(metric_summary[reducer]) * scale)
    return values


def annotate_line(ax, x_values: Sequence[float], y_values: Sequence[float], suffix: str = ""):
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            f"{y_value:.2f}{suffix}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )


def plot_absolute_metrics(summary: Dict[str, Dict], sizes: Sequence[int], systems: Sequence[str], output_path: Path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Rigorous Benchmark: Absolute Metrics", fontsize=16, fontweight="bold")

    charts = [
        ("exact_retrieval_latency_ns", "p95", 1e-6, "Exact Retrieval p95 (ms)"),
        ("topic_retrieval_latency_ns", "p95", 1e-6, "Topic Retrieval p95 (ms)"),
        ("total_build_time_s", "mean", 1.0, "Build Time Mean (s)"),
        ("primed_neighbor_recall_at_5", "mean", 1.0, "Primed Neighbor Recall@5"),
    ]

    for axis, (metric, reducer, scale, title) in zip(axes.flat, charts):
        for system in systems:
            values = metric_series(summary, sizes, system, metric, reducer=reducer, scale=scale)
            axis.plot(
                sizes,
                values,
                marker="o",
                linewidth=2.3,
                color=SYSTEM_COLORS[system],
                label=SYSTEM_LABELS[system],
            )
            annotate_line(axis, sizes, values, "" if scale == 1.0 else "")

        axis.set_title(title)
        axis.set_xlabel("Corpus Size")
        axis.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Milliseconds")
    axes[0, 1].set_ylabel("Milliseconds")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_ylim(0.0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ratio_metrics(summary: Dict[str, Dict], sizes: Sequence[int], systems: Sequence[str], output_path: Path):
    baseline = "traditional_rag"
    comparison_systems = [system for system in systems if system != baseline]
    if not comparison_systems:
        return

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Rigorous Benchmark: NeuroMem vs Traditional Ratios", fontsize=16, fontweight="bold")

    charts = [
        ("exact_retrieval_latency_ns", "p95", 1.0, "Exact Retrieval p95 Ratio"),
        ("topic_retrieval_latency_ns", "p95", 1.0, "Topic Retrieval p95 Ratio"),
        ("total_build_time_s", "mean", 1.0, "Build Time Ratio"),
        ("primed_neighbor_recall_at_5", "mean", -1.0, "Neighbor Recall Ratio"),
    ]

    baseline_color = "#6E7781"
    for axis, (metric, reducer, direction, title) in zip(axes.flat, charts):
        axis.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.0)
        for system in comparison_systems:
            baseline_values = metric_series(summary, sizes, baseline, metric, reducer=reducer)
            system_values = metric_series(summary, sizes, system, metric, reducer=reducer)
            if direction > 0:
                ratio_values = [
                    system_value / baseline_value if baseline_value else 0.0
                    for system_value, baseline_value in zip(system_values, baseline_values)
                ]
                better_text = "< 1 is better"
            else:
                ratio_values = [
                    system_value / baseline_value if baseline_value else 0.0
                    for system_value, baseline_value in zip(system_values, baseline_values)
                ]
                better_text = "> 1 is better"

            axis.plot(
                sizes,
                ratio_values,
                marker="o",
                linewidth=2.3,
                color=SYSTEM_COLORS[system],
                label=SYSTEM_LABELS[system],
            )
            annotate_line(axis, sizes, ratio_values)

        axis.set_title(f"{title} ({better_text})")
        axis.set_xlabel("Corpus Size")
        axis.set_ylabel("Ratio vs Traditional")
        axis.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_memory_footprint(summary: Dict[str, Dict], sizes: Sequence[int], systems: Sequence[str], output_path: Path):
    x_positions = np.arange(len(sizes))
    width = 0.25 if len(systems) >= 3 else 0.35

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    figure.suptitle("Rigorous Benchmark: Memory and Persistence Footprint", fontsize=16, fontweight="bold")

    for index, system in enumerate(systems):
        offset = (index - (len(systems) - 1) / 2) * width
        heap_values = metric_series(summary, sizes, system, "python_bytes_after_ingest", scale=1 / (1024 * 1024))
        db_values = metric_series(summary, sizes, system, "db_file_bytes", scale=1 / (1024 * 1024))

        axes[0].bar(
            x_positions + offset,
            heap_values,
            width=width,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )
        axes[1].bar(
            x_positions + offset,
            db_values,
            width=width,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )

    for axis, title in zip(
        axes,
        ["Python Heap After Ingest (MiB)", "Database Footprint (MiB)"],
    ):
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.set_xlabel("Corpus Size")
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("MiB")
    axes[1].set_ylabel("MiB")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_summary_text(results: Dict, input_path: Path, output_paths: Iterable[Path]) -> str:
    summary = results["summary"]
    sizes = ordered_sizes(summary)
    systems = available_systems(summary)
    lines = [
        "Rigorous Benchmark Visualization Summary",
        f"Source JSON: {input_path}",
        f"Embedding backend: {results['config']['embedding_backend']}",
        f"Sizes: {', '.join(str(size) for size in sizes)}",
        "",
    ]

    for size in sizes:
        lines.append(f"Corpus Size {size}")
        size_summary = summary[str(size)]
        for system in systems:
            metrics = size_summary[system]
            lines.append(
                f"- {SYSTEM_LABELS[system]}: build {metrics['total_build_time_s']['mean']:.3f}s, "
                f"exact p95 {metrics['exact_retrieval_latency_ns']['p95'] / 1e6:.3f}ms, "
                f"topic p95 {metrics['topic_retrieval_latency_ns']['p95'] / 1e6:.3f}ms, "
                f"neighbor recall {metrics['primed_neighbor_recall_at_5']['mean']:.3f}"
            )
        lines.append("")

    lines.append("Generated files:")
    for output_path in output_paths:
        lines.append(f"- {output_path}")
    lines.append("")
    return "\n".join(lines)


def default_output_stem(input_path: Path) -> Path:
    return input_path.with_suffix("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize rigorous benchmark results")
    parser.add_argument(
        "--input",
        help="Path to a rigorous benchmark JSON file. Defaults to the latest benchmark result.",
    )
    parser.add_argument(
        "--output-stem",
        help="Output path stem. Example: benchmark_results/my_run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_path = Path(args.input).resolve() if args.input else latest_result_path(base_dir)
    results = load_results(input_path)

    summary = results["summary"]
    sizes = ordered_sizes(summary)
    systems = available_systems(summary)

    output_stem = Path(args.output_stem).resolve() if args.output_stem else default_output_stem(input_path)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    absolute_path = output_stem.with_name(f"{output_stem.name}_absolute.png")
    ratios_path = output_stem.with_name(f"{output_stem.name}_ratios.png")
    footprint_path = output_stem.with_name(f"{output_stem.name}_footprint.png")
    summary_path = output_stem.with_name(f"{output_stem.name}_visualization_summary.txt")

    plot_absolute_metrics(summary, sizes, systems, absolute_path)
    plot_ratio_metrics(summary, sizes, systems, ratios_path)
    plot_memory_footprint(summary, sizes, systems, footprint_path)

    summary_text = build_summary_text(
        results,
        input_path=input_path,
        output_paths=[absolute_path, ratios_path, footprint_path],
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Wrote absolute metrics chart to {absolute_path}")
    print(f"Wrote ratio metrics chart to {ratios_path}")
    print(f"Wrote footprint chart to {footprint_path}")
    print(f"Wrote text summary to {summary_path}")


if __name__ == "__main__":
    main()
