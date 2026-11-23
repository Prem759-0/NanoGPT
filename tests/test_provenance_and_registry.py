"""
tests/test_provenance_and_registry.py
Unit tests for Strict Sovereign Provenance Verification and Dynamic Model Registry.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import unittest
import tempfile
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.common.checkpoint import load_checkpoint, save_checkpoint, TRAINING_MODE_DEVELOPMENT
from models.registry_manager import verify_checkpoint_provenance, get_dynamic_model_stats
from backend.engines.moon_image.inference import MoonImageInferenceService
from backend.engines.moon_video.inference import MoonVideoInferenceService

class TestProvenanceAndRegistry(unittest.TestCase):
    def test_reject_unverified_provenance(self):
        """Verifies that load_checkpoint raises ValueError when provenance is missing or external."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_ckpt_path = os.path.join(tmpdir, "foreign_model.pt")
            # Save dummy weights with external or missing provenance
            torch.save({
                "model_state_dict": {},
                "provenance": "external_huggingface_pretrained"
            }, bad_ckpt_path)

            dummy_model = torch.nn.Linear(10, 10)
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(bad_ckpt_path, dummy_model)
            self.assertIn("STRICT OWNERSHIP VIOLATION", str(ctx.exception))

            with self.assertRaises(ValueError):
                verify_checkpoint_provenance(bad_ckpt_path)

    def test_accept_sovereign_provenance(self):
        """Verifies that load_checkpoint succeeds when provenance is 'moon_ai_random_init'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            good_ckpt_path = os.path.join(tmpdir, "sovereign_model.pt")
            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={"test": True},
                output_path=good_ckpt_path,
                training_mode=TRAINING_MODE_DEVELOPMENT
            )

            new_model = torch.nn.Linear(10, 10)
            meta = load_checkpoint(good_ckpt_path, new_model, mode=TRAINING_MODE_DEVELOPMENT)
            self.assertEqual(meta["provenance"], "moon_ai_random_init")
            self.assertTrue(verify_checkpoint_provenance(good_ckpt_path))

    def test_dynamic_parameter_counts(self):
        """Verifies that model stats are calculated dynamically without hardcoding."""
        stats = get_dynamic_model_stats()
        self.assertIn("moon-image-v0", stats)
        self.assertIn("moon-video-v0", stats)
        self.assertIn("moon-llm-scratch", stats)

        img_params = stats["moon-image-v0"]["trainable_parameters"]
        vid_params = stats["moon-video-v0"]["trainable_parameters"]
        self.assertGreater(img_params, 20_000_000)
        self.assertGreater(vid_params, 15_000_000)

    def test_truthful_inference_service_without_checkpoint(self):
        """Ensures inference services report truthful MODEL_NOT_TRAINED when no checkpoint exists."""
        img_service = MoonImageInferenceService(checkpoint_path="nonexistent_checkpoint.pt")
        res = img_service.generate(prompt="A test prompt")
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "MODEL_NOT_TRAINED")
        self.assertFalse(res["external_fallback_used"])

        vid_service = MoonVideoInferenceService(checkpoint_path="nonexistent_checkpoint.pt")
        v_res = vid_service.generate_video(prompt="A test prompt")
        self.assertFalse(v_res["success"])
        self.assertEqual(v_res["status"], "MODEL_NOT_TRAINED")
        self.assertFalse(v_res["external_fallback_used"])

if __name__ == "__main__":
    unittest.main()
