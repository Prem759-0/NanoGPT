"""
tests/test_video_pipeline.py
Unit tests for Moon Video Training Pipeline, Real vs Synthetic Dataset Isolation, and Temporal Model.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import unittest
import tempfile
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_video.model import MoonVideoModel
from backend.engines.moon_video.config import MoonVideoConfig
from training.video.dataset import MoonVideoDataset, MoonRealVideoDataset, MoonSyntheticVideoDataset

class TestVideoPipeline(unittest.TestCase):
    def setUp(self):
        self.config = MoonVideoConfig(
            num_frames=4,
            frame_height=64,
            frame_width=64,
            timesteps=100
        )
        self.model = MoonVideoModel(config=self.config)

    def test_model_spatiotemporal_forward_shapes(self):
        """Verifies 3D spatiotemporal forward pass outputs matching shapes."""
        batch_size = 2
        frames = torch.randn(batch_size, 3, self.config.num_frames, 64, 64)
        input_ids = torch.randint(0, 500, (batch_size, self.config.text_max_length))
        camera_ids = torch.tensor([1, 2], dtype=torch.long)
        timesteps = torch.randint(0, 100, (batch_size,))

        pred_noise, target_noise = self.model(frames, input_ids, camera_ids, timesteps)
        self.assertEqual(pred_noise.shape, target_noise.shape)
        # Latent downsampled by 8: 64/8 = 8
        self.assertEqual(pred_noise.shape, (batch_size, 4, self.config.num_frames, 8, 8))

    def test_production_rejects_synthetic_data(self):
        """Ensures production video training strictly raises ValueError if no real videos exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "fake_synth_manifest.jsonl")
            # Create a manifest with only synthetic records
            record = {
                "id": "synthetic_01",
                "media_path": "data/images/test.png",
                "caption": "A moving landscape",
                "camera_motion": "orbit_left",
                "split": "train",
                "license": "CustomSovereign",
                "is_synthetic": True
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Production mode (allow_synthetic=False) MUST fail
            with self.assertRaises(ValueError) as ctx:
                MoonVideoDataset(
                    manifest_path=manifest_path,
                    base_dir=tmpdir,
                    split="train",
                    num_frames=4,
                    allow_synthetic=False
                )
            self.assertIn("STRICT VIDEO INTEGRITY ERROR", str(ctx.exception))

    def test_dev_synthetic_dataset_isolation(self):
        """Ensures synthetic dataset marks items as is_synthetic=True for dev isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image
            img_path = os.path.join(tmpdir, "base_img.png")
            Image.new("RGB", (64, 64), color="blue").save(img_path)

            records = [{
                "id": "dev_synth_01",
                "media_path": "base_img.png",
                "caption": "Camera orbiting",
                "camera_motion": "orbit_left"
            }]
            dataset = MoonSyntheticVideoDataset(
                records=records,
                base_dir=tmpdir,
                num_frames=4,
                frame_size=64
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertTrue(item["is_synthetic"])
            self.assertEqual(item["video_values"].shape, (3, 4, 64, 64))

if __name__ == "__main__":
    unittest.main()
