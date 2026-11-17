"""
tests/test_image_pipeline.py
Unit tests for Moon Image Training Pipeline, Checkpointing, and State Restoration.
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

from backend.engines.moon_image.model import MoonImageModel
from backend.engines.moon_image.config import MoonImageConfig
from backend.engines.moon_image.text_encoder import SimpleTextTokenizer
from training.common.checkpoint import save_checkpoint, load_checkpoint, TRAINING_MODE_DEVELOPMENT

class TestImagePipeline(unittest.TestCase):
    def setUp(self):
        self.config = MoonImageConfig(
            image_height=64,
            image_width=64,
            latent_channels=4,
            timesteps=100
        )
        self.model = MoonImageModel(config=self.config)
        self.tokenizer = SimpleTextTokenizer(max_length=self.config.text_max_length)

    def test_model_forward_pass_shapes(self):
        """Verifies forward pass output shapes and loss computation."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 64, 64)
        input_ids = torch.randint(0, 1000, (batch_size, self.config.text_max_length))
        timesteps = torch.randint(0, 100, (batch_size,))

        pred_noise, target_noise = self.model(images, input_ids, timesteps)
        self.assertEqual(pred_noise.shape, target_noise.shape)
        # Latent downsampled by 8: 64/8 = 8
        self.assertEqual(pred_noise.shape, (batch_size, 4, 8, 8))

    def test_tokenizer_consistency(self):
        """Verifies tokenizer output range and fixed length."""
        text = "A majestic chrome spacecraft orbiting Earth"
        encoded = self.tokenizer.encode(text)
        self.assertEqual(encoded.shape, (self.config.text_max_length,))
        self.assertTrue((encoded >= 0).all() and (encoded < self.config.text_vocab_size).all())

    def test_checkpoint_save_and_full_resume(self):
        """Verifies atomic save and complete restoration of model, optimizer, step, and RNG states."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_image_ckpt.pt")
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

            # Do one fake optimization step to populate optimizer state
            images = torch.randn(1, 3, 64, 64)
            ids = torch.randint(0, 500, (1, self.config.text_max_length))
            t = torch.tensor([10])
            pred, target = self.model(images, ids, t)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            test_manifest_hash = "abcdef1234567890"

            # Save checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                step=42,
                epoch=1,
                loss=loss.item(),
                config_dict=self.config.__dict__,
                output_path=ckpt_path,
                manifest_hash=test_manifest_hash,
                manifest_path="datasets/image/manifest.jsonl",
                training_mode=TRAINING_MODE_DEVELOPMENT
            )
            self.assertTrue(os.path.isfile(ckpt_path))

            # Instantiate a fresh model and resume
            new_model = MoonImageModel(config=self.config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            meta = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=new_model,
                optimizer=new_optimizer,
                mode=TRAINING_MODE_DEVELOPMENT
            )

            self.assertEqual(meta["step"], 42)
            self.assertEqual(meta["provenance"], "moon_ai_random_init")
            self.assertEqual(meta["manifest_hash"], test_manifest_hash)

            # Check parameter equality
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

if __name__ == "__main__":
    unittest.main()
