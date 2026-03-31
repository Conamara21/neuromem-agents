"""
Lightweight observability for NeuroMem runtime services.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Dict


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * 0.95) - 1)
    return sorted_values[index]


class ProxyTelemetry:
    """Thread-safe counters and latency summaries for NeuroMem services."""

    def __init__(self, history_size: int = 512):
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._counters: Dict[str, int] = {
            "memory_writes_total": 0,
            "memory_duplicates_total": 0,
            "memory_search_total": 0,
            "memory_search_hits_total": 0,
            "memory_search_results_total": 0,
            "chat_requests_total": 0,
            "chat_requests_with_memory_hits_total": 0,
            "chat_messages_stored_total": 0,
            "associations_total": 0,
        }
        self._latencies: Dict[str, Deque[float]] = {
            "memory_search_latency_ms": deque(maxlen=history_size),
            "chat_request_latency_ms": deque(maxlen=history_size),
        }

    def record_memory_write(self, duplicate: bool) -> None:
        with self._lock:
            self._counters["memory_writes_total"] += 1
            if duplicate:
                self._counters["memory_duplicates_total"] += 1

    def record_memory_search(self, result_count: int, latency_ms: float) -> None:
        with self._lock:
            self._counters["memory_search_total"] += 1
            self._counters["memory_search_results_total"] += max(0, int(result_count))
            if result_count > 0:
                self._counters["memory_search_hits_total"] += 1
            self._latencies["memory_search_latency_ms"].append(float(latency_ms))

    def record_chat_request(
        self,
        latency_ms: float,
        retrieved_count: int,
        stored_message_count: int,
    ) -> None:
        with self._lock:
            self._counters["chat_requests_total"] += 1
            if retrieved_count > 0:
                self._counters["chat_requests_with_memory_hits_total"] += 1
            self._counters["chat_messages_stored_total"] += max(0, int(stored_message_count))
            self._latencies["chat_request_latency_ms"].append(float(latency_ms))

    def record_association(self) -> None:
        with self._lock:
            self._counters["associations_total"] += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            search_latencies = list(self._latencies["memory_search_latency_ms"])
            chat_latencies = list(self._latencies["chat_request_latency_ms"])

        search_total = max(1, counters["memory_search_total"])
        chat_total = max(1, counters["chat_requests_total"])
        return {
            "uptime_s": round(max(0.0, time.time() - self._started_at), 3),
            "counters": counters,
            "ratios": {
                "memory_search_hit_rate": counters["memory_search_hits_total"] / search_total,
                "average_results_per_search": counters["memory_search_results_total"] / search_total,
                "chat_request_memory_hit_rate": (
                    counters["chat_requests_with_memory_hits_total"] / chat_total
                ),
            },
            "latency_ms": {
                "memory_search_avg": round(_mean(search_latencies), 6),
                "memory_search_p95": round(_p95(search_latencies), 6),
                "chat_request_avg": round(_mean(chat_latencies), 6),
                "chat_request_p95": round(_p95(chat_latencies), 6),
            },
        }

    def prometheus_text(self) -> str:
        snapshot = self.snapshot()
        counters = snapshot["counters"]
        ratios = snapshot["ratios"]
        latency = snapshot["latency_ms"]
        lines = [
            "# HELP neuromem_uptime_seconds Process uptime in seconds.",
            "# TYPE neuromem_uptime_seconds gauge",
            f"neuromem_uptime_seconds {snapshot['uptime_s']}",
            "# HELP neuromem_memory_writes_total Total memory write attempts.",
            "# TYPE neuromem_memory_writes_total counter",
            f"neuromem_memory_writes_total {counters['memory_writes_total']}",
            "# HELP neuromem_memory_duplicates_total Duplicate memory write attempts.",
            "# TYPE neuromem_memory_duplicates_total counter",
            f"neuromem_memory_duplicates_total {counters['memory_duplicates_total']}",
            "# HELP neuromem_memory_search_total Total memory searches.",
            "# TYPE neuromem_memory_search_total counter",
            f"neuromem_memory_search_total {counters['memory_search_total']}",
            "# HELP neuromem_memory_search_hits_total Searches with at least one result.",
            "# TYPE neuromem_memory_search_hits_total counter",
            f"neuromem_memory_search_hits_total {counters['memory_search_hits_total']}",
            "# HELP neuromem_chat_requests_total Total chat requests.",
            "# TYPE neuromem_chat_requests_total counter",
            f"neuromem_chat_requests_total {counters['chat_requests_total']}",
            "# HELP neuromem_chat_requests_with_memory_hits_total Chat requests with retrieved memory.",
            "# TYPE neuromem_chat_requests_with_memory_hits_total counter",
            f"neuromem_chat_requests_with_memory_hits_total {counters['chat_requests_with_memory_hits_total']}",
            "# HELP neuromem_chat_messages_stored_total Total stored user/assistant turns.",
            "# TYPE neuromem_chat_messages_stored_total counter",
            f"neuromem_chat_messages_stored_total {counters['chat_messages_stored_total']}",
            "# HELP neuromem_associations_total Total associative links created.",
            "# TYPE neuromem_associations_total counter",
            f"neuromem_associations_total {counters['associations_total']}",
            "# HELP neuromem_memory_search_hit_rate Ratio of searches that returned results.",
            "# TYPE neuromem_memory_search_hit_rate gauge",
            f"neuromem_memory_search_hit_rate {ratios['memory_search_hit_rate']}",
            "# HELP neuromem_chat_request_memory_hit_rate Ratio of chat requests with memory hits.",
            "# TYPE neuromem_chat_request_memory_hit_rate gauge",
            f"neuromem_chat_request_memory_hit_rate {ratios['chat_request_memory_hit_rate']}",
            "# HELP neuromem_memory_search_latency_ms_avg Average memory search latency.",
            "# TYPE neuromem_memory_search_latency_ms_avg gauge",
            f"neuromem_memory_search_latency_ms_avg {latency['memory_search_avg']}",
            "# HELP neuromem_memory_search_latency_ms_p95 P95 memory search latency.",
            "# TYPE neuromem_memory_search_latency_ms_p95 gauge",
            f"neuromem_memory_search_latency_ms_p95 {latency['memory_search_p95']}",
            "# HELP neuromem_chat_request_latency_ms_avg Average chat request latency.",
            "# TYPE neuromem_chat_request_latency_ms_avg gauge",
            f"neuromem_chat_request_latency_ms_avg {latency['chat_request_avg']}",
            "# HELP neuromem_chat_request_latency_ms_p95 P95 chat request latency.",
            "# TYPE neuromem_chat_request_latency_ms_p95 gauge",
            f"neuromem_chat_request_latency_ms_p95 {latency['chat_request_p95']}",
        ]
        return "\n".join(lines) + "\n"
