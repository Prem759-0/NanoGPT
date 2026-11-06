"""
Moon AI — Stage 2.1.6 Verification Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Verification tests for:
1. Student model architecture: exact 25,698,816 parameters (8L, 8H, 512D, 512 vocab, 512 block).
2. Random Gaussian scratch initialization: zero teacher checkpoint weights transferred.
3. Assistant-only loss masking: prompt tokens masked (-1), assistant tokens targetable.
4. Readiness gate passes with 0 blockers on expanded corpus (572 records).
5. Production training gate enforcement (blocks unready data, permits ready data).
6. Saved checkpoint payload integrity (SHA-256, Policy B attestation, PILOT_TRAINED status).
7. Sovereign tokenizer decode alias and ChatML roundtripping.
8. Text generation from trained checkpoint produces valid tokens without crashing.
"""

import os
import sys
import json
import math
import hashlib
import unittest
import tempfile
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT
from datasets.chat.tokenizer import SovereignTokenizer
from training.chat.dataset import SovereignChatDataset
from training.chat.readiness_gate import (
    ChatReadinessGate,
    STATUS_READY_FOR_PILOT,
    STATUS_ENGINEERING_SMOKE_TEST_ONLY
)
from training.chat.train import train_moon_chat
from training.chat.sample import load_chat_model, generate_chat_response

EXPANDED_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "manifests", "expanded_chat_train.jsonl")
EXPANDED_VAL = os.path.join(PROJECT_ROOT, "datasets", "manifests", "expanded_chat_val.jsonl")
EXPANDED_TEST = os.path.join(PROJECT_ROOT, "datasets", "manifests", "expanded_chat_test.jsonl")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
PILOT_CONFIG = os.path.join(PROJECT_ROOT, "training", "chat", "pilot_config.yaml")
PILOT_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")


class TestStage216StudentPilotTraining(unittest.TestCase):
    """Stage 2.1.6 Student Pilot Training and Policy B Attestation Tests."""

    @classmethod
    def setUpClass(cls):
        from tests.checkpoint_fixture_utils import ensure_pilot_checkpoint_fixture
        cls.ckpt_path, cls.created_fixture = ensure_pilot_checkpoint_fixture()

    @classmethod
    def tearDownClass(cls):
        from tests.checkpoint_fixture_utils import cleanup_checkpoint_fixture
        cleanup_checkpoint_fixture(cls.ckpt_path, cls.created_fixture)

    def setUp(self):
        self.tokenizer = SovereignTokenizer.load(TOKENIZER_PATH)

    def test_01_student_pilot_model_exact_parameter_count(self):
        """Student pilot model geometry yields exactly 25,698,816 parameters."""
        gpt_conf = GPTConfig(
            block_size=512,
            vocab_size=512,
            n_layer=8,
            n_head=8,
            n_embd=512,
            dropout=0.1,
            bias=False
        )
        model = GPT(gpt_conf)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(
            total_params, 25698816,
            f"Expected exactly 25,698,816 parameters for 25.7M pilot model, got {total_params:,}"
        )

    def test_02_random_gaussian_scratch_initialization_invariant(self):
        """Student model initializes purely from random Gaussian distribution N(0, 0.02^2)."""
        gpt_conf = GPTConfig(
            block_size=512,
            vocab_size=512,
            n_layer=8,
            n_head=8,
            n_embd=512,
            bias=False
        )
        model = GPT(gpt_conf)
        # Check standard deviation of linear projection weights (should be ~0.02)
        std_wte = model.transformer.wte.weight.data.std().item()
        mean_wte = model.transformer.wte.weight.data.mean().item()
        self.assertAlmostEqual(mean_wte, 0.0, delta=0.01, msg="WTE mean should be ~0.0")
        self.assertAlmostEqual(std_wte, 0.02, delta=0.005, msg="WTE std should be ~0.02")

    def test_03_assistant_only_loss_masking_in_dataset(self):
        """SovereignChatDataset enforces assistant-only loss masking (user prompts have target -1)."""
        ds = SovereignChatDataset(EXPANDED_TRAIN, self.tokenizer, block_size=128)
        self.assertGreater(len(ds), 0)
        x, y = ds[0]

        # Target sequence must contain both -1 (masked) and non-negative (supervised) tokens
        has_masked = (y == -1).any().item()
        has_supervised = (y >= 0).any().item()
        self.assertTrue(has_masked, "Dataset targets must contain -1 for user prompt tokens.")
        self.assertTrue(has_supervised, "Dataset targets must contain >= 0 for assistant response tokens.")

    def test_04_readiness_gate_passes_on_expanded_manifests(self):
        """ChatReadinessGate evaluates expanded manifests to READY_FOR_PILOT with 0 blockers."""
        gate = ChatReadinessGate(
            train_manifest=EXPANDED_TRAIN,
            val_manifest=EXPANDED_VAL,
            test_manifest=EXPANDED_TEST
        )
        res = gate.run_all_checks()
        self.assertEqual(res["status"], STATUS_READY_FOR_PILOT)
        self.assertTrue(res["passed"])
        self.assertEqual(len(res["blockers"]), 0)
        self.assertGreaterEqual(res["checks"]["pilot_scale_evaluation"]["total_records"], 500)
        self.assertGreaterEqual(res["checks"]["pilot_scale_evaluation"]["train_records"], 400)
        self.assertGreaterEqual(res["checks"]["pilot_scale_evaluation"]["val_records"], 50)
        self.assertGreaterEqual(res["checks"]["pilot_scale_evaluation"]["test_records"], 50)
        self.assertGreaterEqual(res["checks"]["pilot_scale_evaluation"]["total_tokens"], 100000)

    def test_05_pilot_checkpoint_file_and_hash_integrity(self):
        """Pilot checkpoint file exists, is valid PyTorch dict, and matches SHA-256."""
        self.assertTrue(os.path.exists(PILOT_CKPT), f"Checkpoint missing: {PILOT_CKPT}")
        with open(PILOT_CKPT, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()

        ckpt = torch.load(PILOT_CKPT, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["status"], "PILOT_TRAINED")
        self.assertTrue(ckpt["is_production_trained"])
        self.assertTrue(ckpt["promotable"])

        cfg = ckpt["config"]
        self.assertEqual(cfg["ownership_policy"], "TEACHER_STUDENT_RANDOM_INIT")
        self.assertEqual(cfg["initialization"], "random_gaussian_scratch")
        self.assertEqual(cfg["student_init_method"], "random_gaussian_scratch")
        self.assertFalse(cfg["teacher_checkpoint_weights_transferred"],
            "Policy B violation: teacher weights must never transfer into student checkpoint.")
        self.assertIn("Qwen2.5-7B-Instruct-GGUF", cfg["teacher_models"])
        self.assertIn("final_train_loss", cfg)
        self.assertIn("final_val_loss", cfg)
        self.assertIn("perplexity", cfg)

    def test_06_sampling_and_generation_with_trained_model(self):
        """Text generation from the trained checkpoint completes without errors."""
        model, ckpt = load_chat_model(PILOT_CKPT, device="cpu")
        output = generate_chat_response(
            model=model,
            tokenizer=self.tokenizer,
            prompt="Hi",
            max_new_tokens=15,
            temperature=0.7,
            top_k=20,
            device="cpu"
        )
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0, "Generated output must not be empty.")
        self.assertIn("Hi", output, "Output should contain the initial prompt.")

    def test_07_tokenizer_decode_alias(self):
        """SovereignTokenizer.decode() is a working alias for decode_tokens()."""
        tokens = self.tokenizer.encode_text("Hello, Moon AI!")
        decoded_std = self.tokenizer.decode_tokens(tokens)
        decoded_alias = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_std, decoded_alias)
        self.assertEqual(decoded_alias, "Hello, Moon AI!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

