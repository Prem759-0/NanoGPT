"""
tests/test_dataset_engineering.py
Unit tests for Automated Dataset Engineering, Duplicate Detection, and Split Leakage Prevention.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import unittest
import tempfile
import json
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.dataset_manager import audit_manifest, generate_splits, compute_sha256

class TestDatasetEngineering(unittest.TestCase):
    def test_duplicate_and_leakage_detection(self):
        """Verifies that audit_manifest detects exact duplicate media and train/val split leakage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample image file
            img_path = os.path.join(tmpdir, "sample.png")
            Image.new("RGB", (32, 32), color="red").save(img_path)

            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            # Record 1: train split
            r1 = {
                "id": "rec_01",
                "media_path": "sample.png",
                "caption": "A red square image for training",
                "split": "train",
                "license": "CC0"
            }
            # Record 2: val split with identical media -> leakage and duplicate!
            r2 = {
                "id": "rec_02",
                "media_path": "sample.png",
                "caption": "A red square image in validation",
                "split": "val",
                "license": "CC0"
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

            report = audit_manifest(manifest_path=manifest_path, base_dir=tmpdir, media_type="image")
            self.assertFalse(report["valid"])
            self.assertEqual(len(report["duplicates"]), 1)
            self.assertEqual(len(report["split_leakage"]), 1)

    def test_missing_and_corrupt_detection(self):
        """Verifies detection of missing file paths and corrupted media."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupted file
            corrupt_path = os.path.join(tmpdir, "corrupt.png")
            with open(corrupt_path, "wb") as f:
                f.write(b"NOT_A_PNG_HEADER_RANDOM_GARBAGE")

            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            r1 = {
                "id": "rec_missing",
                "media_path": "nonexistent.png",
                "caption": "Missing image file",
                "split": "train",
                "license": "MIT"
            }
            r2 = {
                "id": "rec_corrupt",
                "media_path": "corrupt.png",
                "caption": "Corrupted image file",
                "split": "train",
                "license": "MIT"
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

            report = audit_manifest(manifest_path=manifest_path, base_dir=tmpdir, media_type="image")
            self.assertFalse(report["valid"])
            self.assertEqual(len(report["missing_files"]), 1)
            self.assertEqual(len(report["corrupted_files"]), 1)

    def test_reproducible_split_generation(self):
        """Verifies deterministic split partitioning without leakage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "raw_manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as f:
                for i in range(10):
                    f.write(json.dumps({"id": f"item_{i}", "caption": f"Sample caption {i}"}) + "\n")

            splits_a = generate_splits(manifest_path, seed=42)
            splits_b = generate_splits(manifest_path, seed=42)

            self.assertEqual(len(splits_a), 10)
            self.assertEqual([x["split"] for x in splits_a], [x["split"] for x in splits_b])
            train_count = sum(1 for x in splits_a if x["split"] == "train")
            self.assertEqual(train_count, 8)

if __name__ == "__main__":
    unittest.main()

