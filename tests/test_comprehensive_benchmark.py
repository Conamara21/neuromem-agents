from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuromem.experiments import comprehensive_benchmark as benchmark


class ComprehensiveBenchmarkMetricTests(unittest.TestCase):
    def test_rank_metrics_behave_as_expected(self):
        ranked = ["a", "b", "c", "d"]
        relevant = {"b", "d"}

        self.assertAlmostEqual(benchmark.hit_rate_at_k(ranked, relevant, 1), 0.0)
        self.assertAlmostEqual(benchmark.hit_rate_at_k(ranked, relevant, 2), 1.0)
        self.assertAlmostEqual(benchmark.precision_at_k(ranked, relevant, 4), 0.5)
        self.assertAlmostEqual(benchmark.recall_at_k(ranked, relevant, 4), 1.0)
        self.assertAlmostEqual(benchmark.reciprocal_rank_at_k(ranked, relevant, 4), 0.5)
        self.assertGreater(benchmark.ndcg_at_k(ranked, relevant, 4), 0.6)

    def test_paired_advantage_summary_reports_wins(self):
        summary = benchmark.paired_advantage_summary(
            [2.0, 2.2, 2.1],
            [1.0, 1.1, 1.2],
            higher_is_better=True,
            seed=7,
        )
        self.assertGreater(summary["mean_advantage"], 0.0)
        self.assertGreater(summary["mean_ratio"], 1.0)
        self.assertGreaterEqual(summary["win_rate"], 1.0)
        self.assertLessEqual(summary["p_value_sign_test"], 1.0)


class ComprehensiveBenchmarkSmokeTests(unittest.TestCase):
    def test_structure_trial_emits_extended_metrics(self):
        result = benchmark.run_structure_trial(
            system_name=benchmark.SYSTEM_NEUROMEM_IN_MEMORY,
            size=32,
            query_count=8,
            warmup_count=2,
            association_degree=2,
            embedding_backend=benchmark.EMBEDDING_LEXICAL_HASHING,
            seed=42,
        )
        self.assertEqual(result["suite"], benchmark.STRUCTURE_SUITE)
        self.assertIn("exact_mrr_at_10", result)
        self.assertIn("topic_ndcg_at_10", result)
        self.assertIn("priming_lift_neighbor_recall_at_10", result)

    def test_conversation_trial_emits_handoff_metrics(self):
        result = benchmark.run_conversation_trial(
            system_name=benchmark.SYSTEM_NEUROMEM_SCOPED_IN_MEMORY,
            size=48,
            query_count=8,
            warmup_count=2,
            embedding_backend=benchmark.EMBEDDING_LEXICAL_HASHING,
            seed=99,
        )
        self.assertEqual(result["suite"], benchmark.CONVERSATION_SUITE)
        self.assertIn("handoff_primed_hit_at_5", result)
        self.assertIn("handoff_priming_lift_at_5", result)

    def test_cli_run_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comprehensive.json"
            args = benchmark.build_parser().parse_args(
                [
                    "--suites",
                    benchmark.STRUCTURE_SUITE,
                    "--sizes",
                    "32",
                    "--trials",
                    "1",
                    "--query-count",
                    "4",
                    "--warmup-count",
                    "1",
                    "--output",
                    str(output_path),
                ]
            )
            benchmark.run_suite_from_args(args)
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".md").exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("structure_suite", payload)
            self.assertIn("pairwise", payload["structure_suite"])


if __name__ == "__main__":
    unittest.main()
