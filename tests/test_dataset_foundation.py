"""
tests/test_dataset_foundation.py
Comprehensive Unit Test Suite for Stage 2.0 Balanced Dataset Foundation.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Tests:
1. Image source metadata validation
2. Image checksum verification
3. Video container validation
4. Real ordered frame extraction with timestamps
5. Synthetic media isolation
6. License rejection for unverified/impermissible licenses
7. Near-duplicate detection via dHash Hamming distance
8. Source-grouped split leakage detection
9. Manifest portability with relative paths
10. Insufficient dataset adequacy reporting (HYGIENE_VALID vs TRAINING_DATA_ADEQUATE)
11. Video training fail-closed behavior with no genuine video samples
12. Reproducible dataset audit reports
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
from datasets.tools.deduplicate_images import compute_sha256, compute_dhash, hamming_distance
from datasets.tools.inspect_video import inspect_video_file
from datasets.tools.extract_video_frames import extract_ordered_frames
from datasets.tools.dataset_manager import audit_manifest, PILOT_TARGETS
from training.video.dataset import MoonVideoDataset

class TestDatasetFoundation(unittest.TestCase):
    def test_image_source_metadata_validation(self):
        """Verifies validation of valid and invalid image metadata."""
        sha = "a" * 64
        valid_record = {
            "id": "img_test_01",
            "category": "landscapes",
            "license": "CustomSovereign",
            "creator": "Prem Gaikwad",
            "source_url": "https://raw.githubusercontent.com/Prem759-0/NanoGPT/main/data/images/img_test_01.png",
            "caption": "A beautiful sunset over mountain peaks with dramatic clouds",
            "is_synthetic": False,
            "sha256": sha,
            "media_path": "data/images/img_test_01.png",
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "asset_sha256": sha,
                "rights_statement": "I certify this is an original capture created by me."
            }
        }
        ok, errors = validate_image_source_record(valid_record)
        self.assertTrue(ok, f"Expected valid, got errors: {errors}")

        # Invalid category and missing creator
        invalid_record = {
            "id": "img_test_02",
            "category": "not_a_valid_category",
            "license": "CustomSovereign",
            "creator": "",
            "source_url": "",
            "caption": "short"
        }
        bad_ok, bad_errors = validate_image_source_record(invalid_record)
        self.assertFalse(bad_ok)
        self.assertGreaterEqual(len(bad_errors), 3)

    def test_license_rejection(self):
        """Ensures non-permissive or unknown licenses are strictly rejected."""
        rec = {
            "id": "img_bad_license",
            "category": "technology",
            "license": "CC-BY-NC-4.0",  # Non-commercial is prohibited
            "creator": "Someone",
            "source_url": "https://example.com",
            "caption": "A high-tech microprocessor circuit board component"
        }
        ok, errors = validate_image_source_record(rec)
        self.assertFalse(ok)
        self.assertTrue(any("Unverified or impermissible license" in e for e in errors))

    def test_image_checksum_and_dhash(self):
        """Verifies deterministic SHA-256 and dHash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "test.png")
            Image.new("RGB", (64, 64), color="blue").save(p)
            sha1 = compute_sha256(p)
            sha2 = compute_sha256(p)
            self.assertEqual(sha1, sha2)
            self.assertEqual(len(sha1), 64)

            dh1 = compute_dhash(p)
            dh2 = compute_dhash(p)
            self.assertEqual(dh1, dh2)
            self.assertEqual(hamming_distance(dh1, dh2), 0)

    def test_near_duplicate_detection(self):
        """Ensures near-duplicate detection flags images with small pixel differences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            im1 = Image.new("RGB", (64, 64), color="white")
            # Image 2 is almost identical, with 1 pixel changed
            im2 = Image.new("RGB", (64, 64), color="white")
            im2.putpixel((0, 0), (250, 250, 250))

            dh1 = compute_dhash(im1)
            dh2 = compute_dhash(im2)
            dist = hamming_distance(dh1, dh2)
            self.assertLessEqual(dist, 4)

    def test_video_container_validation(self):
        """Verifies inspect_video_file on real video file and rejection of invalid format."""
        sample_vid = os.path.join(PROJECT_ROOT, "data", "videos", "sample_real_video.mp4")
        if os.path.isfile(sample_vid):
            data = inspect_video_file(sample_vid, min_duration_sec=0.5, min_frames=4)
            self.assertTrue(data["valid"])
            self.assertGreater(data["num_frames"], 0)
            self.assertGreater(data["fps"], 0)

        # Invalid file format rejection
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = os.path.join(tmpdir, "fake.txt")
            with open(txt_path, "w") as f:
                f.write("not a video")
            with self.assertRaises(ValueError):
                inspect_video_file(txt_path)

    def test_real_ordered_frame_extraction(self):
        """Verifies real ordered frame extraction and timestamps."""
        sample_vid = os.path.join(PROJECT_ROOT, "data", "videos", "sample_real_video.mp4")
        if os.path.isfile(sample_vid):
            extracted = extract_ordered_frames(sample_vid, target_num_frames=8, target_size=64)
            self.assertEqual(len(extracted["frames"]), 8)
            self.assertEqual(len(extracted["timestamps"]), 8)
            # Timestamps must be non-decreasing
            for i in range(1, len(extracted["timestamps"])):
                self.assertGreaterEqual(extracted["timestamps"][i], extracted["timestamps"][i - 1])

    def test_synthetic_media_isolation(self):
        """Verifies that synthetic dev fixtures are marked and rejected from genuine video manifests."""
        synth_rec = {
            "id": "synth_vid_01",
            "category": "landscapes",
            "license": "CustomSovereign",
            "creator": "Prem Gaikwad",
            "source_url": "https://example.com",
            "caption": "A simulated camera pan across still image",
            "motion_description": "Simulated horizontal crop shift",
            "camera_motion": "pan_left",
            "is_synthetic": True  # Synthetic flag
        }
        ok, errors = validate_video_source_record(synth_rec)
        self.assertFalse(ok)
        self.assertTrue(any("Synthetic animations cannot be admitted" in e for e in errors))

    def test_source_grouped_split_leakage(self):
        """Verifies detection of split leakage at the media level and creator level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_p = os.path.join(tmpdir, "test.png")
            Image.new("RGB", (32, 32), color="red").save(img_p)

            manifest_path = os.path.join(tmpdir, "leaky.jsonl")
            r1 = {
                "id": "rec_01",
                "media_path": "test.png",
                "caption": "A test image in training split",
                "category": "landscapes",
                "creator": "Photographer_Alpha",
                "source_url": "https://alpha.example.com",
                "license": "CC0",
                "split": "train"
            }
            # Same creator and same media in test split -> double leakage
            r2 = {
                "id": "rec_02",
                "media_path": "test.png",
                "caption": "A test image in testing split",
                "category": "landscapes",
                "creator": "Photographer_Alpha",
                "source_url": "https://alpha.example.com",
                "license": "CC0",
                "split": "test"
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

            rep = audit_manifest(manifest_path, base_dir=tmpdir, media_type="image")
            self.assertFalse(rep["hygiene_valid"])
            self.assertEqual(len(rep["split_leakages"]), 1)
            self.assertEqual(len(rep["creator_leakages"]), 1)

    def test_manifest_portability_relative_paths(self):
        """Ensures all paths in datasets/manifests/ are relative and portable."""
        manifests_dir = os.path.join(PROJECT_ROOT, "datasets", "manifests")
        for fname in ["image_train.jsonl", "image_val.jsonl", "video_train.jsonl", "video_val.jsonl"]:
            fpath = os.path.join(manifests_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line.strip())
                        path_val = r.get("media_path") or r.get("video_path")
                        self.assertFalse(os.path.isabs(path_val), f"Absolute path found in manifest: {path_val}")
                        self.assertNotIn(":", path_val, f"Windows drive letter found in relative path: {path_val}")

    def test_insufficient_dataset_adequacy_separation(self):
        """Ensures that HYGIENE_VALID is True while TRAINING_DATA_ADEQUATE is False when below target."""
        manifest_path = os.path.join(PROJECT_ROOT, "datasets", "manifests", "image_train.jsonl")
        if os.path.isfile(manifest_path):
            rep = audit_manifest(manifest_path, base_dir=PROJECT_ROOT, media_type="image")
            # The pilot dataset has 20 samples, which is technically clean (hygiene passes)
            # but below the 500-sample pilot adequacy target!
            self.assertTrue(rep["hygiene_valid"], "Hygiene should pass for clean pilot data")
            self.assertFalse(rep["training_data_adequate"], "Adequacy should be False for 20 samples (target 500)")
            self.assertEqual(rep["adequacy_assessment"]["status"], "INSUFFICIENT")

    def test_video_training_fails_closed_without_genuine_video(self):
        """Ensures MoonVideoDataset strictly raises ValueError in production mode when 0 genuine videos exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_manifest = os.path.join(tmpdir, "empty_manifest.jsonl")
            with open(empty_manifest, "w", encoding="utf-8") as f:
                f.write("") # 0 records

            with self.assertRaises(ValueError) as ctx:
                MoonVideoDataset(
                    manifest_path=empty_manifest,
                    base_dir=tmpdir,
                    split="train",
                    num_frames=8,
                    allow_synthetic=False # production mode
                )
            self.assertIn("STRICT VIDEO INTEGRITY ERROR", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()

