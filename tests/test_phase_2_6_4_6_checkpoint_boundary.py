"""
tests/test_phase_2_6_4_6_checkpoint_boundary.py
Phase 2.6.4.6: Production Checkpoint Boundary, Historical Fixture Isolation & Security Audit
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Comprehensive security test suite validating:
1. Label-alone insufficiency: Strict policy string alone CANNOT certify a checkpoint.
2. Explicit negative transfer requirement: teacher_checkpoint_weights_transferred must be literal False.
3. Contradictory base weights & teacher references fail closed.
4. Historical fixture isolation: is_test_fixture=True and is_production_trained=False cannot be bypassed.
5. Promotion gate protection: evaluate_promotion rejects missing/None policy, non-strict policy, fixtures.
6. Training resume isolation: production training resume is blocked, non-strict resume is blocked.
"""

import os
import sys
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
)
from webapp.model_service import (
    ModelService,
    is_verified_genuine_checkpoint,
    is_historical_pilot_checkpoint,
    is_checkpoint_fixture,
    validate_checkpoint_structure,
)
from evaluation.teacher_student_eval import PromotionGate
from scripts.promote_best_model import evaluate_and_promote


class TestPhase2646CheckpointBoundary(unittest.TestCase):
    """Deep verification of production checkpoint boundary and historical fixture isolation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_checkpoint(self, filename: str, status="PILOT_TRAINED", is_fixture=False,
                           is_prod_trained=False, config_overrides=None, omit_model=False,
                           omit_model_args=False, model_args_override=None):
        path = os.path.join(self.temp_dir.name, filename)
        cfg = {
            "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
            "initialization": "random_gaussian_scratch",
            "student_init_method": "random_gaussian_scratch",
            "teacher_checkpoint_weights_transferred": False,
        }
        if config_overrides:
            cfg.update(config_overrides)

        m_args = {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "vocab_size": 512,
            "block_size": 512
        }
        if model_args_override:
            m_args.update(model_args_override)

        payload = {
            "status": status,
            "is_test_fixture": is_fixture,
            "is_production_trained": is_prod_trained,
            "config": cfg,
        }
        if not omit_model:
            payload["model"] = {"transformer.wte.weight": torch.randn(512, 512)}
        if not omit_model_args:
            payload["model_args"] = m_args

        torch.save(payload, path)
        return path

    # =========================================================================
    # 1. LABEL ALONE INSUFFICIENCY TESTS
    # =========================================================================

    def test_01_strict_label_with_omitted_transfer_flag_rejected(self):
        """Strict policy string + omitted teacher_checkpoint_weights_transferred fails closed."""
        p = self._create_checkpoint(
            "strict_no_transfer_flag.pt",
            config_overrides={"teacher_checkpoint_weights_transferred": None}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p))

    def test_02_strict_label_with_true_transfer_flag_rejected(self):
        """Strict policy string + teacher_checkpoint_weights_transferred=True is rejected."""
        p = self._create_checkpoint(
            "strict_with_transfer.pt",
            config_overrides={"teacher_checkpoint_weights_transferred": True}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p))

    def test_03_strict_label_with_external_base_weights_rejected(self):
        """Strict policy string + declared external base_weights (e.g. 'gpt2') fails closed."""
        p1 = self._create_checkpoint(
            "strict_gpt2_base.pt",
            config_overrides={"base_weights": "gpt2"}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p1))

        p2 = self._create_checkpoint(
            "strict_llama_source.pt",
            config_overrides={"pretrained_source": "llama-3.2-1b"}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p2))

    def test_04_strict_label_with_teacher_references_rejected(self):
        """Strict policy string + teacher_checkpoint / teacher_model metadata fails closed."""
        p = self._create_checkpoint(
            "strict_teacher_ref.pt",
            config_overrides={"teacher_checkpoint": "models/teacher_v1.pt"}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p))

    def test_05_strict_label_with_empty_or_non_scratch_init_rejected(self):
        """Strict policy string + non-scratch initialization fails closed."""
        p1 = self._create_checkpoint(
            "strict_empty_cfg.pt",
            config_overrides={"initialization": None, "student_init_method": None}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p1))

        p2 = self._create_checkpoint(
            "strict_pretrained_init.pt",
            config_overrides={"initialization": "pretrained"}
        )
        self.assertFalse(is_verified_genuine_checkpoint(p2))

    def test_06_strict_label_with_missing_model_or_args_rejected(self):
        """Strict policy string + missing model tensor state or model_args fails closed."""
        p_no_model = self._create_checkpoint("no_model.pt", omit_model=True)
        self.assertFalse(is_verified_genuine_checkpoint(p_no_model))

        p_no_args = self._create_checkpoint("no_args.pt", omit_model_args=True)
        self.assertFalse(is_verified_genuine_checkpoint(p_no_args))

    # =========================================================================
    # 2. HISTORICAL FIXTURE ISOLATION TESTS
    # =========================================================================

    def test_07_historical_pilot_fixture_cannot_pass_genuine_verification(self):
        """Historical Stage 2.1.6 pilot checkpoint fails is_verified_genuine_checkpoint."""
        hist_p = self._create_checkpoint(
            "hist_pilot.pt",
            config_overrides={"ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT}
        )
        # Structural check succeeds
        self.assertTrue(validate_checkpoint_structure(hist_p)[0])
        # Historical check succeeds
        self.assertTrue(is_historical_pilot_checkpoint(hist_p))
        # Genuine production verification MUST fail
        self.assertFalse(is_verified_genuine_checkpoint(hist_p))

    def test_08_historical_pilot_claiming_production_trained_still_fails_genuine_check(self):
        """Historical pilot checkpoint claiming is_production_trained=True fails genuine production check."""
        p = self._create_checkpoint(
            "fake_prod_hist.pt",
            is_prod_trained=True,
            config_overrides={"ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT}
        )
        # Genuine production verification MUST fail
        self.assertFalse(is_verified_genuine_checkpoint(p))
        # When loaded in ModelService, is_test_fixture is enforced True regardless of checkpoint metadata
        ms = ModelService(base_dir=self.temp_dir.name)
        self.assertIsNone(ms.find_pilot_checkpoint(allow_fixture=False))

    def test_09_model_service_production_catalog_isolates_historical_pilot(self):
        """ModelService marks historical pilot available=False, is_test_fixture=True in production catalog."""
        # Setup simulated chat checkpoint path in temp base dir
        chat_dir = os.path.join(self.temp_dir.name, "models", "checkpoints", "chat")
        os.makedirs(chat_dir, exist_ok=True)
        ckpt_path = os.path.join(chat_dir, "moon_chat_pilot_v0.pt")
        from model import GPT, GPTConfig
        dummy_gpt = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=16, vocab_size=512, block_size=512))
        # Save historical teacher-student checkpoint
        torch.save({
            "status": "PILOT_TRAINED",
            "is_test_fixture": False,
            "is_production_trained": False,
            "promotable": False,
            "model": dummy_gpt.state_dict(),
            "model_args": {"n_layer": 1, "n_head": 1, "n_embd": 16, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT,
                "initialization": "random_gaussian_scratch",
                "student_init_method": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False
            }
        }, ckpt_path)

        ms = ModelService(base_dir=self.temp_dir.name)
        # allow_fixture=False must return None
        self.assertIsNone(ms.find_pilot_checkpoint(allow_fixture=False))
        # allow_fixture=True locates it for retrospective inspection
        self.assertIsNotNone(ms.find_pilot_checkpoint(allow_fixture=True))

        models = ms.get_available_models()
        pilot_entry = next((m for m in models if m["id"] == "moon-chat-pilot-v0"), None)
        self.assertIsNotNone(pilot_entry)
        # Production catalog must declare available=False and is_test_fixture=True
        self.assertFalse(pilot_entry["available"])
        self.assertTrue(pilot_entry["is_test_fixture"])

        # Default loading without allow_fixture=True must fail closed
        self.assertFalse(ms.load_model("moon-chat-pilot-v0"))
        # Explicit retrospective inspection mode enables fixture loading
        loaded = ms.load_model("moon-chat-pilot-v0", allow_fixture=True)
        self.assertTrue(loaded)
        self.assertTrue(getattr(ms, "is_test_fixture", False))
        info = ms.get_info()
        self.assertTrue(info["is_test_fixture"])
        self.assertFalse(info["is_production_trained"])
        self.assertFalse(info["promotable"])
        self.assertEqual(info["status"], "CI_TEST_FIXTURE")

    # =========================================================================
    # 3. PROMOTION GATE PROTECTION TESTS
    # =========================================================================

    def test_10_promotion_gate_mandates_strict_policy(self):
        """PromotionGate strictly requires STRICT_SOVEREIGN_RANDOM_INIT_ONLY; rejects missing/None."""
        base_cand = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0, "math_reasoning": 90.0, "programming": 90.0, "logical_reasoning": 90.0}
        }

        # Case A: Missing policy
        cand_no_policy = dict(base_cand)
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_no_policy)
        self.assertFalse(promoted)
        self.assertIn("STRICT_SOVEREIGN_RANDOM_INIT_ONLY", reason)

        # Case B: None policy
        cand_none_policy = dict(base_cand)
        cand_none_policy["ownership_policy"] = None
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_none_policy)
        self.assertFalse(promoted)
        self.assertIn("STRICT_SOVEREIGN_RANDOM_INIT_ONLY", reason)

        # Case C: Teacher-student policy
        cand_ts_policy = dict(base_cand)
        cand_ts_policy["ownership_policy"] = POLICY_TEACHER_STUDENT_RANDOM_INIT
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_ts_policy)
        self.assertFalse(promoted)
        self.assertIn("STRICT_SOVEREIGN_RANDOM_INIT_ONLY", reason)

        # Case D: Strict policy passes
        cand_strict = dict(base_cand)
        cand_strict["ownership_policy"] = POLICY_STRICT_SOVEREIGN_RANDOM_INIT
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_strict)
        self.assertTrue(promoted)

    def test_11_promotion_gate_rejects_fixtures_and_unpromotable_candidates(self):
        """PromotionGate strictly rejects candidates with is_test_fixture=True or is_production_trained=False."""
        cand_fixture = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
            "is_test_fixture": True
        }
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_fixture)
        self.assertFalse(promoted)
        self.assertIn("unpromotable", reason.lower())

        cand_unpromoted = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
            "is_production_trained": False
        }
        promoted2, reason2, _ = PromotionGate.evaluate_promotion(cand_unpromoted)
        self.assertFalse(promoted2)
        self.assertIn("unpromotable", reason2.lower())

    def test_12_promotion_gate_rejects_candidate_pointing_to_disk_fixture(self):
        """PromotionGate rejects promotion when candidate checkpoint file on disk fails genuine verification."""
        fixture_p = self._create_checkpoint(
            "fixture_on_disk.pt",
            status="CI_TEST_FIXTURE",
            is_fixture=True
        )
        cand = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0, "math_reasoning": 90.0, "programming": 90.0, "logical_reasoning": 90.0},
            "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT
        }
        promoted, reason, _ = PromotionGate.evaluate_promotion(
            candidate_eval=cand,
            candidate_checkpoint_path=fixture_p
        )
        self.assertFalse(promoted)
        self.assertIn("failed genuine checkpoint verification", reason)

    # =========================================================================
    # 4. TRAINING RESUME ISOLATION TESTS
    # =========================================================================

    def test_13_training_resume_production_mode_strictly_blocked(self):
        """Production pilot training (smoke_test=False) strictly blocks training and resume_from_ckpt."""
        from unittest.mock import patch
        from training.chat.train import train_moon_chat
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "pilot_config.yaml")

        # Case A: Safeguard 0 blocks execution when AUTHORIZED_BY_PREM=False
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(
                config_path=cfg_path,
                smoke_test=False,
                resume_from_ckpt="dummy_path.pt"
            )
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))

        # Case B: Even if authorized, resuming in production is strictly blocked by sovereign policy
        with patch("training.chat.train.AUTHORIZED_BY_PREM", True):
            with self.assertRaises(RuntimeError) as ctx2:
                train_moon_chat(
                    config_path=cfg_path,
                    smoke_test=False,
                    resume_from_ckpt="dummy_path.pt"
                )
            self.assertIn("Production pilot training blocked: resuming from arbitrary checkpoints is strictly prohibited", str(ctx2.exception))

    def test_14_training_resume_rejects_historical_or_non_strict_checkpoint(self):
        """In smoke test mode, resuming from a non-strict or historical checkpoint raises ValueError."""
        from training.chat.train import train_moon_chat
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "pilot_config.yaml")

        hist_p = self._create_checkpoint(
            "hist_resume.pt",
            config_overrides={"ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT}
        )

        with self.assertRaises(Exception) as ctx:
            train_moon_chat(
                config_path=cfg_path,
                smoke_test=True,
                resume_from_ckpt=hist_p
            )
        self.assertTrue(
            "QUARANTINED HISTORICAL POLICY" in str(ctx.exception)
            or "Strict Sovereign Policy Violation" in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
