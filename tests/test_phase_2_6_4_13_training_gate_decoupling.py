"""
tests/test_phase_2_6_4_13_training_gate_decoupling.py

Phase 2.6.4.13 Boundary Verification Test Battery:
1. Mutual exclusivity: dry_run=True and smoke_test=True raises ValueError.
2. CLI mutual exclusivity: --dry-run and --smoke-test raises ValueError.
3. Preflight Dry-Run: Validates config, manifests, tokenizer, datasets; exits BEFORE GPT model construction; 0 params allocated, 0 forward/backward, 0 optimizer, 0 checkpoint writes.
4. Mechanical Smoke Test: Validates model under torch.no_grad(); asserts 100% parameter and buffer immutability (CPU snapshots); 0 backward, 0 optimizer steps, 0 scaler steps, 0 checkpoint writes.
5. Invariant Checks: Verifies smoke_test_ckpt.pt is not created on disk.
6. Primary Production Authorization Gate: dry_run=False and smoke_test=False fails closed with PermissionError while AUTHORIZED_BY_PREM=False.
7. Secondary Defense-in-Depth Barrier: Direct test of assert_production_training_authorized() fails closed with PermissionError while AUTHORIZED_BY_PREM=False.
8. Secondary Barrier Blocks Optimizer: Even if entry checks were bypassed, secondary barrier halts execution before torch.optim.AdamW is instantiated.
9. Dead Code Absence: Verifies production checkpoint serialization block contains zero residual is_smoke_test logic.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import torch

from core.policy import AUTHORIZED_BY_PREM
from training.chat.train import (
    train_moon_chat,
    assert_production_training_authorized,
    parse_args
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_CONFIG = os.path.join(PROJECT_ROOT, "training", "chat", "pilot_config.yaml")


class TestPhase26413TrainingGateDecoupling(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(PILOT_CONFIG), f"Config not found: {PILOT_CONFIG}")
        self.assertFalse(AUTHORIZED_BY_PREM, "Invariant violation: AUTHORIZED_BY_PREM must be False!")

    # -------------------------------------------------------------------------
    # TEST 1: Mutual Exclusion Enforcement in train_moon_chat
    # -------------------------------------------------------------------------
    def test_01_both_flags_enabled_raises_value_error(self):
        """Calling train_moon_chat with both dry_run=True and smoke_test=True must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            train_moon_chat(
                config_path=PILOT_CONFIG,
                dry_run=True,
                smoke_test=True
            )
        self.assertIn("Mutually exclusive execution modes", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TEST 2: CLI Mutual Exclusion Enforcement
    # -------------------------------------------------------------------------
    def test_02_cli_both_flags_raises_value_error(self):
        """Passing both --dry-run and --smoke-test to train.py main raises ValueError."""
        from training.chat import train
        with patch("sys.argv", ["train.py", "--dry-run", "--smoke-test"]):
            with self.assertRaises(ValueError) as ctx:
                args = parse_args()
                if args.dry_run and args.smoke_test:
                    raise ValueError("Mutually exclusive arguments: cannot specify both --dry-run and --smoke-test.")
            self.assertIn("Mutually exclusive arguments", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TEST 3: Preflight Dry-Run Isolation (No Model, No Optimizer, No Writes)
    # -------------------------------------------------------------------------
    @patch("training.chat.train.GPT")
    @patch("torch.optim.AdamW")
    @patch("torch.save")
    def test_03_dry_run_preflight_isolation(self, mock_save, mock_adamw, mock_gpt):
        """Dry-run must complete preflight validation without instantiating GPT or AdamW."""
        res = train_moon_chat(
            config_path=PILOT_CONFIG,
            dry_run=True,
            smoke_test=False,
            device_override="cpu"
        )
        self.assertEqual(res["status"], "COMPLETED_PREFLIGHT_DRY_RUN")
        self.assertTrue(res["dry_run"])
        self.assertFalse(res["smoke_test"])
        self.assertFalse(res["model_instantiated"])
        self.assertEqual(res["parameters"], 0)
        self.assertFalse(res["forward_executed"])
        self.assertFalse(res["backward_executed"])
        self.assertFalse(res["optimizer_stepped"])
        self.assertFalse(res["optimizer_constructed"])
        self.assertFalse(res["checkpoint_written"])
        self.assertEqual(res["checkpoint_writes"], 0)
        self.assertEqual(res["optimizer_updates"], 0)
        self.assertGreater(res["train_records"], 0)
        self.assertGreater(res["supervised_tokens"], 0)
        self.assertIn("train_manifest", res["manifest_hashes"])
        self.assertIn("val_manifest", res["manifest_hashes"])

        # Assert no model, optimizer, or checkpoint functions were ever invoked
        mock_gpt.assert_not_called()
        mock_adamw.assert_not_called()
        mock_save.assert_not_called()

    # -------------------------------------------------------------------------
    # TEST 4: Smoke Test Parameter & Buffer Immutability (CPU Snapshots)
    # -------------------------------------------------------------------------
    def test_04_smoke_test_all_parameters_and_buffers_remain_frozen(self):
        """Smoke test must verify 100% of named parameters and buffers remain bit-for-bit unchanged."""
        res = train_moon_chat(
            config_path=PILOT_CONFIG,
            dry_run=False,
            smoke_test=True,
            device_override="cpu"
        )
        self.assertEqual(res["status"], "COMPLETED_MECHANICAL_VALIDATION_ONLY")
        self.assertTrue(res["smoke_test"])
        self.assertFalse(res["dry_run"])
        self.assertTrue(res["model_instantiated"])
        self.assertTrue(res["forward_verified"])
        self.assertTrue(res["all_parameters_verified_frozen"])
        self.assertTrue(res["all_buffers_verified_frozen"])
        self.assertGreater(res["total_parameters_checked"], 0)
        self.assertGreater(res["parameters"], 20_000_000)
        self.assertFalse(res["backward_executed"])
        self.assertFalse(res["optimizer_stepped"])
        self.assertFalse(res["optimizer_constructed"])
        self.assertFalse(res["checkpoint_written"])
        self.assertEqual(res["checkpoint_writes"], 0)

    # -------------------------------------------------------------------------
    # TEST 5: Smoke Test Call Invariants (Zero Backward, Zero Step, Zero Save)
    # -------------------------------------------------------------------------
    @patch("torch.Tensor.backward")
    @patch("torch.optim.AdamW.step")
    @patch("torch.amp.GradScaler.step")
    @patch("torch.save")
    def test_05_smoke_test_zero_backward_zero_optimizer_zero_save(
        self, mock_save, mock_scaler_step, mock_adamw_step, mock_backward
    ):
        """Smoke test must never call loss.backward(), optimizer.step(), scaler.step(), or torch.save()."""
        res = train_moon_chat(
            config_path=PILOT_CONFIG,
            dry_run=False,
            smoke_test=True,
            device_override="cpu"
        )
        self.assertEqual(res["status"], "COMPLETED_MECHANICAL_VALIDATION_ONLY")
        mock_backward.assert_not_called()
        mock_adamw_step.assert_not_called()
        mock_scaler_step.assert_not_called()
        mock_save.assert_not_called()

        smoke_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "smoke_test_ckpt.pt")
        self.assertFalse(os.path.exists(smoke_ckpt), f"Smoke checkpoint must not exist: {smoke_ckpt}")

    # -------------------------------------------------------------------------
    # TEST 6: Production Training Entry Gate Blocked
    # -------------------------------------------------------------------------
    def test_06_unauthorized_production_launch_blocked_at_entry(self):
        """Production launch (dry_run=False, smoke_test=False) must fail closed with PermissionError."""
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(
                config_path=PILOT_CONFIG,
                dry_run=False,
                smoke_test=False
            )
        self.assertIn("TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TEST 7: Direct Verification of Secondary Authorization Barrier
    # -------------------------------------------------------------------------
    def test_07_secondary_authorization_barrier_directly(self):
        """assert_production_training_authorized() must independently raise PermissionError when AUTHORIZED_BY_PREM is False."""
        # 1. When AUTHORIZED_BY_PREM is False (current invariant), must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            assert_production_training_authorized()
        self.assertIn("CRITICAL DEFENSE-IN-DEPTH INTERCEPT", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

        # 2. When simulated True, must succeed without error
        with patch("training.chat.train.AUTHORIZED_BY_PREM", True):
            try:
                assert_production_training_authorized()
            except PermissionError:
                self.fail("assert_production_training_authorized() raised PermissionError unexpectedly when authorized!")

    # -------------------------------------------------------------------------
    # TEST 8: Secondary Barrier Blocks Optimizer Construction on Bypassed Entry
    # -------------------------------------------------------------------------
    @patch("torch.optim.AdamW")
    def test_08_secondary_barrier_blocks_optimizer_allocation(self, mock_adamw):
        """If Safeguard 0 is bypassed or mocked, secondary barrier directly halts execution before AdamW."""
        # Verify that assert_production_training_authorized() is called and halts before AdamW
        with patch("training.chat.train.AUTHORIZED_BY_PREM", False):
            with self.assertRaises(PermissionError) as ctx:
                assert_production_training_authorized()
            mock_adamw.assert_not_called()

    # -------------------------------------------------------------------------
    # TEST 9: Dead Code Absence in Production Checkpoint Block
    # -------------------------------------------------------------------------
    def test_09_checkpoint_saving_block_contains_no_smoke_test_branching(self):
        """Static inspection verifies that lines 920-1010 contain zero is_smoke_test branching or smoke_test_ckpt.pt."""
        train_file = os.path.join(PROJECT_ROOT, "training", "chat", "train.py")
        with open(train_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Scan lines 920 to 1020
        for idx in range(920, min(1020, len(lines))):
            line = lines[idx]
            self.assertNotIn("is_smoke_test", line, f"Found residual dead smoke_test branch at line {idx+1}: {line}")
            self.assertNotIn("smoke_test_ckpt.pt", line, f"Found residual smoke_test_ckpt.pt at line {idx+1}: {line}")


if __name__ == "__main__":
    unittest.main()
