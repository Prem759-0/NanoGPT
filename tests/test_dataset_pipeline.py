"""
tests/test_dataset_pipeline.py
Unit tests for dataset auditing, deduplication, and splitting pipeline.
"""
import os
import sys
import json
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.audit_dataset import audit_file, normalize_text, check_python_code_blocks
from scripts.prepare_dataset import prepare_splits

class TestDatasetPipeline(unittest.TestCase):

    def test_01_normalize_text(self):
        """Verify normalization strips punctuation and excessive whitespace."""
        raw = "What is a DOG, really?!?"
        norm = normalize_text(raw)
        self.assertEqual(norm, "what is a dog really")

    def test_02_python_syntax_checker(self):
        """Verify syntax checker flags broken python code and passes valid code."""
        valid_response = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        valid_count, errors = check_python_code_blocks(valid_response)
        self.assertEqual(valid_count, 1)
        self.assertEqual(len(errors), 0)

        invalid_response = "Here is bad code:\n```python\ndef add(a, b)\n    return a +\n```"
        valid_count, errors = check_python_code_blocks(invalid_response)
        self.assertEqual(len(errors), 1)

    def test_03_prepare_splits_creates_files(self):
        """Verify prepare_splits creates valid train, val, and test files with manifest."""
        test_source = os.path.join(PROJECT_ROOT, "chatgpt_dataset.json")
        out_test_dir = os.path.join(PROJECT_ROOT, "data", "test_splits_temp")
        
        manifest = prepare_splits(
            source_files=[test_source],
            output_dir=out_test_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        self.assertIn("splits", manifest)
        self.assertGreater(manifest["total_records"], 0)
        self.assertTrue(os.path.isfile(os.path.join(out_test_dir, "train.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_test_dir, "val.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_test_dir, "test.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_test_dir, "manifest.json")))

        # Check for zero train/test leakage
        with open(os.path.join(out_test_dir, "train.json"), "r", encoding="utf-8") as f:
            train_items = json.load(f)
        with open(os.path.join(out_test_dir, "test.json"), "r", encoding="utf-8") as f:
            test_items = json.load(f)

        train_instructions = {t["instruction"] for t in train_items}
        test_instructions = {t["instruction"] for t in test_items}
        overlap = train_instructions.intersection(test_instructions)
        self.assertEqual(len(overlap), 0, "Train and test splits must have 0 overlapping instructions")

        # Cleanup temp dir
        import shutil
        shutil.rmtree(out_test_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
