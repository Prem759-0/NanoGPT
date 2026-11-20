"""
Moon AI — Phase 2.7.4 Adversarial Training-Lock Verification Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Tests every training execution route across the repository under adversarial
conditions, proving that:
1. Every training, optimization, backward, and checkpoint path fails closed.
2. Prohibited operations (loss.backward, optimizer.step, checkpoint writes)
   are never reached when AUTHORIZED_BY_PREM is False.
3. Subprocess launchers, module imports, and direct function calls all fail closed.
4. Mocks instrumenting prohibited operations assert zero invocations.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import runpy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.policy as policy
from core.policy import (
    AUTHORIZED_BY_PREM,
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    SovereignPolicyViolationError,
    assert_production_training_authorized,
    validate_production_policy,
)


class TestPhase274AdversarialTrainingRoutes(unittest.TestCase):
    """Adversarial security tests proving fail-closed training boundary across all routes."""

    def setUp(self):
        # Guarantee invariant baseline before each test
        self.assertFalse(policy.AUTHORIZED_BY_PREM, "Invariant violation: AUTHORIZED_BY_PREM must be False")

    def tearDown(self):
        # Re-enforce invariant baseline after each test
        policy.AUTHORIZED_BY_PREM = False

    # ------------------------------------------------------------------
    # Category A: Direct Training Execution Functions
    # ------------------------------------------------------------------

    @patch("torch.optim.AdamW")
    def test_01_direct_chat_train_blocked_zero_optimizer(self, mock_adamw):
        """training/chat/train.py: train_moon_chat() must fail closed without allocating AdamW."""
        from training.chat.train import train_moon_chat
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW constructor was reached!")

        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(
                config_path="training/chat/pilot_config.yaml",
                dry_run=False,
                smoke_test=False,
            )
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))
        mock_adamw.assert_not_called()

    @patch("torch.optim.AdamW")
    def test_02_direct_image_train_blocked_zero_optimizer(self, mock_adamw):
        """training/image/train.py: train(args) must fail closed without allocating AdamW."""
        from training.image.train import train as image_train
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW constructor was reached!")

        mock_args = MagicMock()
        with self.assertRaises(PermissionError) as ctx:
            image_train(mock_args)
        self.assertIn("IMAGE TRAINING BLOCKED", str(ctx.exception))
        mock_adamw.assert_not_called()

    @patch("torch.optim.AdamW")
    def test_03_direct_video_train_blocked_zero_optimizer(self, mock_adamw):
        """training/video/train.py: train(args) must fail closed without allocating AdamW."""
        from training.video.train import train as video_train
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW constructor was reached!")

        mock_args = MagicMock()
        with self.assertRaises(PermissionError) as ctx:
            video_train(mock_args)
        self.assertIn("VIDEO TRAINING BLOCKED", str(ctx.exception))
        mock_adamw.assert_not_called()

    def test_04_direct_train_assistant_blocked(self):
        """scripts/train_assistant.py: train(args) must fail closed before executing training."""
        from scripts.train_assistant import train as assistant_train

        mock_args = MagicMock()
        with self.assertRaises(PermissionError) as ctx:
            assistant_train(mock_args)
        self.assertIn("ASSISTANT TRAINING BLOCKED", str(ctx.exception))

    @patch("subprocess.call")
    def test_05_direct_colab_train_runner_blocked_zero_subprocess(self, mock_subproc):
        """scripts/colab_train.py: main() must fail closed without launching training subprocess."""
        from scripts.colab_train import main as colab_main
        mock_subproc.side_effect = RuntimeError("ADVERSARIAL VIOLATION: subprocess was launched!")

        with self.assertRaises(PermissionError) as ctx:
            colab_main()
        self.assertIn("COLAB TRAINING RUNNER BLOCKED", str(ctx.exception))
        mock_subproc.assert_not_called()

    # ------------------------------------------------------------------
    # Category B: Preflight, Hardware Benchmark & Readiness Gates
    # ------------------------------------------------------------------

    @patch("torch.optim.AdamW")
    @patch("torch.save")
    def test_06_direct_preflight_pilot_blocked_zero_backward_or_save(self, mock_save, mock_adamw):
        """scripts/preflight_pilot_training.py: run_preflight() must fail closed at Gate 0."""
        from scripts.preflight_pilot_training import run_preflight
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: Gate 5 AdamW reached!")
        mock_save.side_effect = RuntimeError("ADVERSARIAL VIOLATION: Gate 6 checkpoint save reached!")

        with self.assertRaises(PermissionError) as ctx:
            run_preflight()
        self.assertIn("PREFLIGHT TRAINING EXECUTION BLOCKED", str(ctx.exception))
        mock_adamw.assert_not_called()
        mock_save.assert_not_called()

    @patch("torch.optim.AdamW")
    @patch("torch.save")
    def test_07_direct_empirical_t4_validation_blocked_zero_optimizer_or_save(self, mock_save, mock_adamw):
        """scripts/validate_t4_empirical.py: run_empirical_t4_validation() must fail closed."""
        from scripts.validate_t4_empirical import run_empirical_t4_validation
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW reached!")
        mock_save.side_effect = RuntimeError("ADVERSARIAL VIOLATION: save reached!")

        with self.assertRaises(PermissionError) as ctx:
            run_empirical_t4_validation()
        self.assertIn("EMPIRICAL T4 VALIDATION BLOCKED", str(ctx.exception))
        mock_adamw.assert_not_called()
        mock_save.assert_not_called()

    def test_08_hardware_benchmark_analytical_fallback_zero_gradient(self):
        """training/chat/validate_hardware.py: validate_model_and_hardware() falls back to analytical simulation."""
        from training.chat.validate_hardware import validate_model_and_hardware
        res = validate_model_and_hardware()
        self.assertIn("gpu_benchmark", res)
        gpu_status = res["gpu_benchmark"].get("status")
        self.assertIn(gpu_status, ("LIVE_CUDA_BENCHMARK_BLOCKED_BY_OWNER_LOCK", "CUDA_NOT_AVAILABLE", "T4_VALIDATION_PENDING_NO_CUDA"))
        self.assertIsNone(res["gpu_benchmark"].get("backward_allocated_mb"))
        self.assertIsNone(res["gpu_benchmark"].get("optimizer_step_allocated_mb"))

    def test_09_readiness_gate_unauthorized_skips_backward_and_step(self):
        """training/chat/readiness_gate.py: evaluate_and_report() skips backward and step when locked."""
        from training.chat.readiness_gate import evaluate_and_report
        with patch("torch.optim.AdamW") as mock_adamw, patch("torch.save") as mock_save:
            mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW reached in readiness gate!")
            mock_save.side_effect = RuntimeError("ADVERSARIAL VIOLATION: save reached in readiness gate!")
            # Should evaluate datasets without executing backward passes or writing checkpoints
            evaluate_and_report()
            mock_adamw.assert_not_called()
            mock_save.assert_not_called()

    # ------------------------------------------------------------------
    # Category C: Standalone Scripts (bench.py & train.py) Module Execution
    # ------------------------------------------------------------------

    def test_10_bench_py_fails_closed_statically(self):
        """bench.py: Top-level module execution must fail closed with PermissionError."""
        bench_path = os.path.join(PROJECT_ROOT, "bench.py")
        with self.assertRaises(PermissionError) as ctx:
            runpy.run_path(bench_path)
        self.assertIn("BENCHMARK TRAINING BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_11_train_py_fails_closed_statically(self):
        """train.py: Top-level module execution must fail closed with PermissionError."""
        train_path = os.path.join(PROJECT_ROOT, "train.py")
        with self.assertRaises(PermissionError) as ctx:
            runpy.run_path(train_path)
        self.assertIn("TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx.exception))

    # ------------------------------------------------------------------
    # Category D: Model Optimizer Construction Direct Gating
    # ------------------------------------------------------------------

    @patch("torch.optim.AdamW")
    def test_12_model_configure_optimizers_direct_call_blocked(self, mock_adamw):
        """model.py: GPT.configure_optimizers() must fail closed without allocating AdamW."""
        from model import GPTConfig, GPT
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW constructor reached in model.py!")

        cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=1, n_head=1, n_embd=32)
        model = GPT(cfg)

        with self.assertRaises(PermissionError) as ctx:
            model.configure_optimizers(
                weight_decay=0.1,
                learning_rate=1e-3,
                betas=(0.9, 0.95),
                device_type="cpu"
            )
        self.assertIn("GPT_CONFIGURE_OPTIMIZERS TRAINING BLOCKED", str(ctx.exception))
        mock_adamw.assert_not_called()

    # ------------------------------------------------------------------
    # Category E: Subprocess Orchestrators & Daily Pipeline
    # ------------------------------------------------------------------

    def test_13_daily_pipeline_skips_training_subprocess(self):
        """scripts/daily_pipeline.py: main() must skip training subprocess when unauthorized."""
        import scripts.daily_pipeline as dp

        def fake_subprocess_run(cmd, *args, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "train.py" in cmd_str or "colab_train.py" in cmd_str or "train_moon_chat" in cmd_str:
                raise RuntimeError(f"ADVERSARIAL VIOLATION: Training subprocess launched: {cmd_str}")
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "Ran 10 tests in 0.05s\nOK\n"
            mock_res.stderr = ""
            return mock_res

        with patch.object(sys, "argv", ["daily_pipeline.py", "--train"]), \
             patch("subprocess.run", side_effect=fake_subprocess_run) as mock_subproc, \
             patch("scripts.daily_pipeline.prepare_splits", return_value={"train": 10, "val": 2, "test": 2}), \
             patch("scripts.daily_pipeline.audit_file", return_value={"syntax_error_records": 0}), \
             patch("scripts.daily_pipeline.run_evaluation", return_value={"score": 85.0}), \
             patch("scripts.daily_pipeline.evaluate_and_promote", return_value={"promoted": False}), \
             patch("scripts.daily_pipeline.generate_daily_report", return_value=None), \
             patch("os.listdir", return_value=["eval_candidate_20261118.json"]):
            # Must run cleanly without throwing unexpected exceptions or swallowing errors
            dp.main()

            # Verify non-training regression test subprocess was legitimately invoked
            self.assertGreaterEqual(mock_subproc.call_count, 1)
            for call_args in mock_subproc.call_args_list:
                args = call_args[0][0]
                self.assertNotIn("train.py", args)
                self.assertNotIn("colab_train.py", args)
                # Verify legitimate invocation was only the regression test suite
                self.assertIn("tests/test_model_inference.py", args)

    # ------------------------------------------------------------------
    # Category F: Web Endpoints & Lifecycle Safety
    # ------------------------------------------------------------------

    def test_14_webapp_self_train_endpoint_unconditionally_403(self):
        """webapp/main.py: POST /api/self-train must return HTTP 403 Forbidden unconditionally."""
        from fastapi.testclient import TestClient
        from webapp.main import app

        client = TestClient(app)
        res = client.post("/api/self-train", json={"iterations": 10})
        self.assertEqual(res.status_code, 403)
        self.assertIn("Self-training is permanently disabled", res.json().get("detail", ""))

    @patch("webapp.database.db.list_training_pairs")
    @patch("torch.optim.AdamW")
    @patch("torch.save")
    @patch("subprocess.run")
    def test_15_learning_service_step_returns_disabled(self, mock_subproc, mock_save, mock_adamw, mock_db_list):
        """
        webapp/learning_service.py: run_self_train_step() returns disabled: True.
        Adversarial proof: Proves zero calls to db.list_training_pairs, zero AdamW,
        zero torch.save, zero subprocess invocations, and zero dataset writes.
        """
        from webapp.learning_service import learning_service
        mock_db_list.side_effect = RuntimeError("ADVERSARIAL VIOLATION: db.list_training_pairs was called!")
        mock_adamw.side_effect = RuntimeError("ADVERSARIAL VIOLATION: AdamW was called!")
        mock_save.side_effect = RuntimeError("ADVERSARIAL VIOLATION: torch.save was called!")
        mock_subproc.side_effect = RuntimeError("ADVERSARIAL VIOLATION: subprocess was called!")

        res = learning_service.run_self_train_step(iterations=10)
        self.assertFalse(res.get("success"))
        self.assertTrue(res.get("disabled"))
        self.assertEqual(res.get("reason"), "OWNER_AUTHORIZATION_REQUIRED")
        self.assertIn("permanently disabled", res.get("message", ""))
        self.assertEqual(res.get("training_pairs_count"), 0)

        # Prohibited operation zero-call assertions
        mock_db_list.assert_not_called()
        mock_adamw.assert_not_called()
        mock_save.assert_not_called()
        mock_subproc.assert_not_called()

    def test_16_controlled_learning_lifecycle_forbids_live_updates(self):
        """core/learning_lifecycle/lifecycle.py: allow_live_weight_updates=True raises ValueError."""
        from core.learning_lifecycle.lifecycle import ControlledLearningLifecycle
        with self.assertRaises(ValueError) as ctx:
            ControlledLearningLifecycle(allow_live_weight_updates=True)
        self.assertIn("Live model weight updates during conversations are forbidden", str(ctx.exception))

    # ------------------------------------------------------------------
    # Category G: Policy Overrides & Malformed States
    # ------------------------------------------------------------------

    def test_17_validate_production_policy_rejects_historical_teacher(self):
        """core/policy.py: Historical TEACHER_STUDENT_RANDOM_INIT raises SovereignPolicyViolationError."""
        with self.assertRaises(SovereignPolicyViolationError) as ctx:
            validate_production_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertIn("QUARANTINED HISTORICAL POLICY", str(ctx.exception))

    def test_18_validate_production_policy_rejects_arbitrary_string(self):
        """core/policy.py: Arbitrary policy names raise SovereignPolicyViolationError."""
        with self.assertRaises(SovereignPolicyViolationError) as ctx:
            validate_production_policy("UNAUTHORIZED_EXTERNAL_TEACHER_POLICY")
        self.assertIn("UNAUTHORIZED", str(ctx.exception))

    def test_19_assert_production_training_authorized_fails_closed(self):
        """core/policy.py: assert_production_training_authorized() raises PermissionError fail-closed."""
        with self.assertRaises(PermissionError) as ctx:
            assert_production_training_authorized()
        self.assertIn("TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_20_learning_service_record_chat_turn_never_appends_to_corpus(self):
        """webapp/learning_service.py: record_chat_turn() must return False and never log to training files."""
        from webapp.learning_service import learning_service
        result = learning_service.record_chat_turn(
            instruction="Test user instruction",
            response="Test assistant response",
            source="chat"
        )
        self.assertFalse(result, "record_chat_turn must return False under Phase 2.6.2.2 containment")


if __name__ == "__main__":
    unittest.main()
