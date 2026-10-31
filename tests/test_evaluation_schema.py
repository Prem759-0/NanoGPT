import os
import sys
import unittest
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_model import create_empty_summary, _save_reports, run_evaluation

class TestEvaluationSchema(unittest.TestCase):
    """
    Test suite verifying robust evaluation schema consistency across all outcomes:
    - Successful model runs
    - Missing checkpoints / failed loads
    - Empty benchmarks
    - Defensive report generation without KeyError
    """

    def test_01_create_empty_summary_contract(self):
        """Verify create_empty_summary provides all required schema keys."""
        summary = create_empty_summary(
            model_name="test_model",
            total_items=10,
            active_model_name="none",
            checkpoint_path="none",
            error_message="Checkpoint not found"
        )
        required_keys = [
            "model_name", "active_model_name", "checkpoint_path", "timestamp",
            "total_items", "verifiable_tasks_total", "verifiable_tasks_passed",
            "solution_correctness_pct", "code_syntax_pct", "coding_syntax_valid_pct",
            "keyword_coverage_pct", "coherence_pct", "non_empty_rate_pct",
            "empty_count", "error_count", "tokens_per_second", "total_tokens_generated",
            "total_time_seconds", "composite_quality_score", "status", "error", "results"
        ]
        for key in required_keys:
            self.assertIn(key, summary, f"Key '{key}' missing from summary schema!")

        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["composite_quality_score"], 0.0)
        self.assertEqual(summary["coherence_pct"], 0.0)
        self.assertEqual(summary["error"], "Checkpoint not found")

    def test_02_missing_checkpoint_fails_loudly_and_safely(self):
        """Verify requesting evaluation of an un-loadable or non-existent model returns failed schema."""
        summary = run_evaluation(model_name="non_existent_test_model", max_tokens=10)
        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["composite_quality_score"], 0.0)
        self.assertEqual(summary["coherence_pct"], 0.0)
        self.assertIsNotNone(summary["error"])
        self.assertIn("Failed to load model", summary["error"])

    def test_03_save_reports_handles_failed_summary_without_keyerror(self):
        """Verify _save_reports writes markdown and JSON without throwing KeyError: 'coherence_pct'."""
        summary = create_empty_summary(
            model_name="unit_test_failed_model",
            total_items=5,
            error_message="Simulated missing checkpoint"
        )
        try:
            _save_reports("unit_test_failed_model", summary)
        except KeyError as e:
            self.fail(f"_save_reports crashed with KeyError: {e}")

        # Check that reports were created on disk
        reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
        latest_md = os.path.join(reports_dir, "eval_unit_test_failed_model_latest.md")
        self.assertTrue(os.path.isfile(latest_md))

        with open(latest_md, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FAILED", content)
        self.assertIn("Simulated missing checkpoint", content)
        self.assertIn("Coherence (Non-Degenerate)**: **0.0%**", content)

        # Clean up test artifact
        try:
            os.remove(latest_md)
        except OSError:
            pass

    def test_04_save_reports_handles_sparse_dict_defensively(self):
        """Verify _save_reports handles completely sparse dictionaries without crashing."""
        sparse_summary = {
            "model_name": "sparse_model",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            _save_reports("sparse_model", sparse_summary)
        except Exception as e:
            self.fail(f"_save_reports failed on sparse summary: {e}")

        reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
        latest_md = os.path.join(reports_dir, "eval_sparse_model_latest.md")
        if os.path.isfile(latest_md):
            os.remove(latest_md)

if __name__ == "__main__":
    unittest.main()
