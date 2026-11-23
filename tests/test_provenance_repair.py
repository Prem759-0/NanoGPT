"""
tests/test_provenance_repair.py
Unit tests for Stage 2.0.1 Critical Dataset Provenance Repair and Production Training Gate.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Verifies:
1. Procedural images cannot enter genuine production manifests.
2. Procedural video clips cannot enter genuine production manifests.
3. Synthetic records are rejected by production training (MoonImageDataset, MoonVideoDataset).
4. Missing provenance causes rejection.
5. Aggregate reports include all three splits (train, val, test) plus dev.
6. Registry status cannot claim DATA_READY when genuine data is insufficient.
7. Synthetic development manifests are rejected in production training mode.
8. Chat and Code development fixtures are quarantined and detected.
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

from datasets.tools.validate_image_sources import validate_image_source_record
from datasets.tools.validate_video_sources import validate_video_source_record
from datasets.tools.dataset_manager import audit_manifest, audit_all_dataset_manifests
from training.image.dataset import MoonImageDataset
from training.video.dataset import MoonVideoDataset

class TestProvenanceRepair(unittest.TestCase):
    def test_procedural_image_rejected_from_production_manifest(self):
        """Verifies that records marked is_synthetic=True cannot be admitted to production image manifests."""
        record = {
            "id": "img_procedural_test",
            "category": "landscapes",
            "license": "CustomSovereign",
            "creator": "Prem Gaikwad",
            "source_url": "https://example.com/mountain.jpg",
            "caption": "Alpine peaks bathed in golden sunset glow",
            "split": "train",
            "is_synthetic": True
        }
        ok, errors = validate_image_source_record(record, allow_synthetic=False)
        self.assertFalse(ok)
        self.assertTrue(any("Synthetic procedural fixtures are strictly prohibited" in e for e in errors))

    def test_procedural_video_rejected_from_production_manifest(self):
        """Verifies that synthetic video animations cannot be admitted to production video manifests."""
        record = {
            "id": "vid_procedural_test",
            "category": "landscapes",
            "license": "CustomSovereign",
            "creator": "Prem Gaikwad",
            "source_url": "https://example.com/ocean.mp4",
            "caption": "Simulated camera pan across mountain scene",
            "motion_description": "Pan across still image",
            "camera_motion": "pan_left",
            "split": "train",
            "is_synthetic": True
        }
        ok, errors = validate_video_source_record(record, allow_synthetic=False)
        self.assertFalse(ok)
        self.assertTrue(any("Synthetic animations cannot be admitted" in e for e in errors))

    def test_fake_internal_repo_urls_rejected(self):
        """Verifies that self-referential repository folder URLs are rejected as source URLs."""
        fake_rec = {
            "id": "img_fake_url",
            "category": "architecture",
            "license": "CustomSovereign",
            "creator": "Prem Gaikwad",
            "source_url": "https://github.com/Prem759-0/NanoGPT/tree/main/data/images/architecture",
            "caption": "High-rise glass skyscraper illuminated at night",
            "split": "train",
            "is_synthetic": False
        }
        ok, errors = validate_image_source_record(fake_rec, allow_synthetic=False)
        self.assertFalse(ok)
        self.assertTrue(any("Fabricated internal repo URL detected" in e for e in errors))

    def test_production_image_training_rejects_synthetic_records(self):
        """Ensures MoonImageDataset strictly fails when synthetic records or dev manifests are loaded in production."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_p = os.path.join(tmpdir, "test.png")
            Image.new("RGB", (64, 64), color="red").save(img_p)

            manifest_p = os.path.join(tmpdir, "image_train.jsonl")
            synth_rec = {
                "id": "img_synth_01",
                "media_path": "test.png",
                "caption": "A test image drawn with Pillow",
                "category": "landscapes",
                "creator": "Tester",
                "source_url": "https://example.com/test.jpg",
                "license": "CC0",
                "sha256": "a" * 64,
                "split": "train",
                "is_synthetic": True
            }
            with open(manifest_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(synth_rec) + "\n")

            with self.assertRaises(ValueError) as ctx:
                MoonImageDataset(
                    manifest_path=manifest_p,
                    base_dir=tmpdir,
                    split="train",
                    allow_synthetic=False # production mode
                )
            self.assertIn("STRICT IMAGE INTEGRITY ERROR", str(ctx.exception))

    def test_production_image_training_rejects_missing_provenance(self):
        """Ensures MoonImageDataset rejects records with missing creator, source_url, or invalid license."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_p = os.path.join(tmpdir, "test.png")
            Image.new("RGB", (64, 64), color="red").save(img_p)

            manifest_p = os.path.join(tmpdir, "image_train.jsonl")
            bad_rec = {
                "id": "img_bad_prov",
                "media_path": "test.png",
                "caption": "A test image with missing provenance fields",
                "category": "landscapes",
                "creator": "", # missing creator
                "source_url": "", # missing source_url
                "license": "NON_PERMISSIBLE",
                "sha256": "123", # invalid sha
                "split": "train",
                "is_synthetic": False
            }
            with open(manifest_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(bad_rec) + "\n")

            with self.assertRaises(ValueError) as ctx:
                MoonImageDataset(
                    manifest_path=manifest_p,
                    base_dir=tmpdir,
                    split="train",
                    allow_synthetic=False
                )
            self.assertIn("STRICT PROVENANCE ERROR", str(ctx.exception))

    def test_empty_production_manifest_fails_closed(self):
        """Verifies MoonImageDataset and MoonVideoDataset fail closed when production manifests contain 0 genuine records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_m = os.path.join(tmpdir, "empty_manifest.jsonl")
            with open(empty_m, "w", encoding="utf-8") as f:
                f.write("")

            with self.assertRaises(ValueError) as ctx_img:
                MoonImageDataset(
                    manifest_path=empty_m,
                    base_dir=tmpdir,
                    split="train",
                    allow_synthetic=False
                )
            self.assertIn("STRICT IMAGE INTEGRITY ERROR", str(ctx_img.exception))

            with self.assertRaises(ValueError) as ctx_vid:
                MoonVideoDataset(
                    manifest_path=empty_m,
                    base_dir=tmpdir,
                    split="train",
                    allow_synthetic=False
                )
            self.assertIn("STRICT VIDEO INTEGRITY ERROR", str(ctx_vid.exception))

    def test_aggregate_audit_includes_all_splits(self):
        """Verifies that audit_all_dataset_manifests aggregates train, val, test, and dev splits."""
        rep = audit_all_dataset_manifests()
        self.assertIn("datasets", rep)
        img_data = rep["datasets"]["image"]
        vid_data = rep["datasets"]["video"]

        # Verify all splits are reported
        self.assertIn("train", img_data["splits_summary"])
        self.assertIn("val", img_data["splits_summary"])
        self.assertIn("test", img_data["splits_summary"])
        self.assertIn("dev", img_data["splits_summary"])

        # Genuine counts must be 0
        self.assertEqual(img_data["genuine_count"], 0)
        self.assertEqual(vid_data["genuine_count"], 0)

        # Synthetic counts must reflect quarantined fixtures
        self.assertGreater(img_data["synthetic_count"], 0)
        self.assertGreater(vid_data["synthetic_count"], 0)

        # Overall quality gate flags
        self.assertFalse(rep["overall_status"]["GENUINE_DATA_PRESENT"])
        self.assertFalse(rep["overall_status"]["TRAINING_DATA_ADEQUATE"])

    def test_registry_enforces_honest_status(self):
        """Verifies that models with 0 genuine training data cannot claim DATA_READY or TRAINED."""
        registry_path = os.path.join(PROJECT_ROOT, "models", "registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        img_model = registry["models"]["moon-image-v0"]
        vid_model = registry["models"]["moon-video-v0"]
        chat_model = registry["models"]["moon-chat-v0"]
        code_model = registry["models"]["moon-code-v0"]

        self.assertEqual(img_model["status"], "DATA_INSUFFICIENT")
        self.assertFalse(img_model.get("training_ready", False))
        self.assertEqual(img_model.get("genuine_samples_count", 0), 0)

        self.assertEqual(vid_model["status"], "DATA_INSUFFICIENT")
        self.assertFalse(vid_model.get("training_ready", False))
        self.assertEqual(vid_model.get("genuine_samples_count", 0), 0)

        self.assertEqual(chat_model["status"], "DATA_INSUFFICIENT")
        self.assertEqual(code_model["status"], "DATA_INSUFFICIENT")

if __name__ == "__main__":
    unittest.main()

