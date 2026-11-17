"""
tests/test_phase_2_6_2_8_authorization_closure.py
Phase 2.6.2.8: Comprehensive Authorization Gate Closure Verification Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Static & safe isolated unit tests verifying that all training entry points across
the entire platform fail closed when AUTHORIZED_BY_PREM is False.
Zero training, zero optimizer steps, zero backward passes, zero checkpoint writes.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.policy import (
    AUTHORIZED_BY_PREM,
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    is_valid_policy,
    validate_production_policy,
)


class TestPhase2628AuthorizationGateClosure(unittest.TestCase):
    """
    Verifies that all training entry points across the repository fail closed
    under AUTHORIZED_BY_PREM = False.
    """

    def setUp(self):
        # Guarantee invariant at start of every test
        self.assertFalse(AUTHORIZED_BY_PREM, "Invariant violation: AUTHORIZED_BY_PREM must be False!")

    def test_01_core_policy_invariant(self):
        """AUTHORIZED_BY_PREM must be False and STRICT_SOVEREIGN_RANDOM_INIT_ONLY must be active."""
        from core.policy import AUTHORIZED_BY_PREM, get_active_policy
        self.assertFalse(AUTHORIZED_BY_PREM)
        self.assertEqual(get_active_policy(), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(is_valid_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT))
        self.assertFalse(is_valid_policy("TEACHER_STUDENT_RANDOM_INIT"))

    def test_02_image_train_fails_closed(self):
        """training/image/train.py: train() must raise PermissionError when unauthorized."""
        from training.image.train import train as image_train
        dummy_args = MagicMock()
        dummy_args.allow_synthetic = False
        dummy_args.manifest = "datasets/manifests/image_train.jsonl"
        with self.assertRaises(PermissionError) as ctx:
            image_train(dummy_args)
        self.assertIn("IMAGE TRAINING BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_03_video_train_fails_closed(self):
        """training/video/train.py: train() must raise PermissionError when unauthorized."""
        from training.video.train import train as video_train
        dummy_args = MagicMock()
        dummy_args.allow_synthetic = False
        dummy_args.manifest = "datasets/manifests/video_train.jsonl"
        with self.assertRaises(PermissionError) as ctx:
            video_train(dummy_args)
        self.assertIn("VIDEO TRAINING BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_04_train_assistant_fails_closed(self):
        """scripts/train_assistant.py: train() must raise PermissionError when unauthorized."""
        from scripts.train_assistant import train as assistant_train
        dummy_args = MagicMock()
        with self.assertRaises(PermissionError) as ctx:
            assistant_train(dummy_args)
        self.assertIn("ASSISTANT TRAINING BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_05_colab_train_runner_fails_closed(self):
        """scripts/colab_train.py: main() must raise PermissionError when unauthorized."""
        from scripts.colab_train import main as colab_main
        with self.assertRaises(PermissionError) as ctx:
            colab_main()
        self.assertIn("COLAB TRAINING RUNNER BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_06_empirical_t4_validation_fails_closed(self):
        """scripts/validate_t4_empirical.py: run_empirical_t4_validation() must raise PermissionError when unauthorized."""
        from scripts.validate_t4_empirical import run_empirical_t4_validation
        with self.assertRaises(PermissionError) as ctx:
            run_empirical_t4_validation()
        self.assertIn("EMPIRICAL T4 VALIDATION BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_07_bench_py_fails_closed_statically(self):
        """bench.py must contain top-level AUTHORIZED_BY_PREM gate before GPT/optimizer."""
        bench_path = os.path.join(PROJECT_ROOT, "bench.py")
        self.assertTrue(os.path.isfile(bench_path))
        with open(bench_path, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("from core.policy import AUTHORIZED_BY_PREM", code)
        self.assertIn("if not AUTHORIZED_BY_PREM:", code)
        self.assertIn("raise PermissionError", code)
        # Verify the gate appears before optimizer
        gate_pos = code.find("if not AUTHORIZED_BY_PREM:")
        opt_pos = code.find("model.configure_optimizers")
        self.assertGreater(opt_pos, gate_pos, "Safeguard gate must appear before optimizer instantiation!")

    def test_08_root_train_py_fails_closed_statically(self):
        """train.py must contain top-level AUTHORIZED_BY_PREM gate before GPT/optimizer."""
        train_path = os.path.join(PROJECT_ROOT, "train.py")
        self.assertTrue(os.path.isfile(train_path))
        with open(train_path, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("from core.policy import AUTHORIZED_BY_PREM", code)
        self.assertIn("if not AUTHORIZED_BY_PREM:", code)
        self.assertIn("raise PermissionError", code)
        gate_pos = code.find("if not AUTHORIZED_BY_PREM:")
        opt_pos = code.find("configure_optimizers")
        self.assertGreater(opt_pos, gate_pos, "Safeguard gate must appear before optimizer instantiation!")

    def test_09_chat_train_production_fails_closed(self):
        """training/chat/train.py: train_moon_chat() must raise PermissionError for production runs."""
        from training.chat.train import train_moon_chat
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(
                config_path="training/chat/pilot_config.yaml",
                dry_run=False,
                smoke_test=False
            )
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_10_webapp_self_train_endpoint_blocked(self):
        """webapp/main.py: POST /api/self-train must return HTTP 403 Forbidden."""
        from fastapi.testclient import TestClient
        from webapp.main import app
        client = TestClient(app)
        response = client.post("/api/self-train", json={"iterations": 10})
        self.assertEqual(response.status_code, 403)
        self.assertIn("permanently disabled", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
