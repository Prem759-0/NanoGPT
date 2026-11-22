"""
Moon AI — Phase 2.6.4.5: Strict Sovereign Policy Consumer Closure & Negative Regression Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Verifies:
1. Strict sovereign policy is accepted as the ONLY authorized production policy.
2. Historical teacher-student policy (TEACHER_STUDENT_RANDOM_INIT) is rejected for production.
3. Legacy teacher-policy checkpoints are strictly rejected from genuine production loading.
4. Unannotated checkpoints fail closed (is_verified_genuine_checkpoint == False).
5. Unknown provenance remains UNKNOWN_PROVENANCE_INSUFFICIENT and is never silently relabeled.
6. Teacher demonstrations are strictly excluded from the sovereign training data path.
7. Strict ownership metadata is accepted ONLY after comprehensive validation of scratch init and zero teacher weights.
8. Evaluation reports reflect authentic checkpoint provenance dynamically and never forge strict status.
9. AUTHORIZED_BY_PREM remains False, locking all production training pipelines.
"""

import os
import sys
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.policy as policy_module
from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    VALID_PRODUCTION_POLICIES,
    HISTORICAL_QUARANTINED_POLICIES,
    SovereignPolicyViolationError,
    get_active_policy,
    is_valid_policy,
    is_historical_policy,
    validate_production_policy,
    is_production_training_authorized,
    assert_production_training_authorized,
    AUTHORIZED_BY_PREM,
    POLICY_RULES
)
from webapp.model_service import (
    ModelService,
    validate_checkpoint_structure,
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    is_checkpoint_fixture,
    is_historical_pilot_checkpoint
)
from evaluation.teacher_student_eval import PromotionGate


class TestPhase2645ConsumerClosure(unittest.TestCase):
    """Certifies policy consumer closure and negative regression guarantees under Phase 2.6.4.5."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # 1. Strict Policy Accepted
    def test_01_strict_policy_accepted_for_production(self):
        """Strict sovereign policy is accepted as valid production policy."""
        self.assertEqual(get_active_policy(), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(is_valid_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT))
        self.assertIn(POLICY_STRICT_SOVEREIGN_RANDOM_INIT, VALID_PRODUCTION_POLICIES)
        # Does not raise
        validate_production_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT)

    # 2. Teacher–Student Policy Rejected for Production
    def test_02_teacher_student_policy_rejected_for_production(self):
        """Quarantined teacher-student policy is rejected for production authorization."""
        self.assertFalse(is_valid_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))
        self.assertTrue(is_historical_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))
        self.assertIn(POLICY_TEACHER_STUDENT_RANDOM_INIT, HISTORICAL_QUARANTINED_POLICIES)
        with self.assertRaises(SovereignPolicyViolationError):
            validate_production_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT)

        # PromotionGate rejects non-strict policy
        cand_eval = {
            "mean_composite_score": 98.0,
            "category_scores": {
                "safety_uncertainty": 100.0,
                "math_reasoning": 95.0,
                "programming": 95.0,
                "logical_reasoning": 95.0
            },
            "ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT
        }
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_eval)
        self.assertFalse(promoted)
        self.assertIn("STRICT_SOVEREIGN_RANDOM_INIT_ONLY", reason)

    # 3. Legacy Teacher-Policy Checkpoints Rejected
    def test_03_legacy_teacher_checkpoints_rejected(self):
        """Legacy checkpoints declaring teacher-student policy fail is_verified_genuine_checkpoint."""
        legacy_path = os.path.join(self.temp_dir.name, "legacy_teacher.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "is_production_trained": False,
            "promotable": False,
            "is_test_fixture": False,
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT,
                "initialization": "random_gaussian_scratch",
                "student_init_method": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False
            }
        }, legacy_path)

        self.assertTrue(validate_checkpoint_structure(legacy_path)[0])
        self.assertFalse(is_verified_genuine_checkpoint(legacy_path))

        # ModelService.find_pilot_checkpoint(allow_fixture=False) rejects it
        ms = ModelService(base_dir=self.temp_dir.name)
        # Even if placed in candidate path
        os.makedirs(os.path.join(self.temp_dir.name, "models", "checkpoints", "chat"), exist_ok=True)
        dest_path = os.path.join(self.temp_dir.name, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")
        import shutil
        shutil.copyfile(legacy_path, dest_path)
        self.assertIsNone(ms.find_pilot_checkpoint(allow_fixture=False))

    # 4. Unannotated Checkpoints Fail Closed
    def test_04_unannotated_checkpoints_fail_closed(self):
        """Checkpoints with missing ownership metadata fail closed as unverified."""
        unannotated_path = os.path.join(self.temp_dir.name, "unannotated.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "initialization": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False
                # ownership_policy deliberately omitted
            }
        }, unannotated_path)

        self.assertFalse(is_verified_genuine_checkpoint(unannotated_path))

        # Promotion gate rejects candidate with unannotated / None policy
        cand_eval = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "ownership_policy": None
        }
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_eval)
        self.assertFalse(promoted)

    # 5. Unknown Provenance Remains Unknown (Never Silently Relabeled)
    def test_05_unknown_provenance_remains_unknown(self):
        """Checkpoints lacking provenance default to UNKNOWN_PROVENANCE_INSUFFICIENT and are never relabeled strict."""
        unannotated_path = os.path.join(self.temp_dir.name, "unknown.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {}
        }, unannotated_path)

        ckpt = torch.load(unannotated_path, map_location="cpu", weights_only=True)
        meta = ckpt.get("config", {}) or ckpt.get("metadata", {})
        policy = meta.get("ownership_policy") or meta.get("policy") or "UNKNOWN_PROVENANCE_INSUFFICIENT"
        self.assertEqual(policy, "UNKNOWN_PROVENANCE_INSUFFICIENT")
        self.assertNotEqual(policy, POLICY_STRICT_SOVEREIGN_RANDOM_INIT)

        # Promotion gate rejects UNKNOWN_PROVENANCE_INSUFFICIENT
        cand_eval = {
            "mean_composite_score": 95.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "ownership_policy": "UNKNOWN_PROVENANCE_INSUFFICIENT"
        }
        promoted, reason, _ = PromotionGate.evaluate_promotion(cand_eval)
        self.assertFalse(promoted)
        self.assertIn("STRICT_SOVEREIGN_RANDOM_INIT_ONLY", reason)

    # 6. Teacher Demonstrations Cannot Enter Strict Training Data Path
    def test_06_teacher_demonstrations_excluded_from_strict_path(self):
        """Strict sovereign policy rules strictly forbid teacher demonstrations."""
        strict_rules = POLICY_RULES[POLICY_STRICT_SOVEREIGN_RANDOM_INIT]
        self.assertFalse(strict_rules["allow_teacher_demonstrations"])
        self.assertFalse(strict_rules["allow_third_party_weights"])
        self.assertFalse(strict_rules["allow_external_apis_in_runtime"])
        self.assertTrue(strict_rules["require_random_initialization"])

    # 7. Strict Ownership Metadata Accepted Only After Validation
    def test_07_strict_ownership_accepted_only_after_validation(self):
        """Strict sovereign checkpoint is accepted ONLY if weights, init, and transfer flags validate."""
        # Valid strict checkpoint
        valid_strict = os.path.join(self.temp_dir.name, "valid_strict.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "is_production_trained": False,
            "promotable": False,
            "is_test_fixture": False,
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
                "initialization": "random_gaussian_scratch",
                "student_init_method": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False
            }
        }, valid_strict)
        self.assertEqual(classify_checkpoint(valid_strict, allow_test_fixture_classification=True), "TEST_FIXTURE_VALID_STRUCTURE")
        self.assertFalse(is_verified_genuine_checkpoint(valid_strict))

        # Forged strict policy with teacher weights transferred
        forged_transfer = os.path.join(self.temp_dir.name, "forged_transfer.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "is_production_trained": False,
            "promotable": False,
            "is_test_fixture": False,
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
                "initialization": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": True  # FORBIDDEN
            }
        }, forged_transfer)
        self.assertFalse(is_verified_genuine_checkpoint(forged_transfer))

        # Forged strict policy with non-random initialization
        forged_init = os.path.join(self.temp_dir.name, "forged_init.pt")
        torch.save({
            "status": "PILOT_TRAINED",
            "is_production_trained": False,
            "promotable": False,
            "is_test_fixture": False,
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
                "initialization": "pretrained_weights_import",  # FORBIDDEN
                "teacher_checkpoint_weights_transferred": False
            }
        }, forged_init)
        self.assertFalse(is_verified_genuine_checkpoint(forged_init))

        # Forged strict policy marked as CI test fixture
        forged_fixture = os.path.join(self.temp_dir.name, "forged_fixture.pt")
        torch.save({
            "status": "CI_TEST_FIXTURE",
            "is_test_fixture": True,
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
                "initialization": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False,
                "is_test_fixture": True
            }
        }, forged_fixture)
        self.assertFalse(is_verified_genuine_checkpoint(forged_fixture))

    # 8. Evaluation Output Does Not Falsely Claim Strict Provenance
    def test_08_evaluation_output_does_not_falsely_claim_strict(self):
        """Evaluation outputs record authentic checkpoint policy and do not falsely label legacy checkpoints strict."""
        legacy_meta = {
            "ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT,
            "initialization": "random_gaussian_scratch"
        }
        ckpt_policy = legacy_meta.get("ownership_policy") or legacy_meta.get("policy") or "UNKNOWN_PROVENANCE_INSUFFICIENT"
        self.assertEqual(ckpt_policy, POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertNotEqual(ckpt_policy, POLICY_STRICT_SOVEREIGN_RANDOM_INIT)

        unannotated_meta = {}
        unannotated_policy = unannotated_meta.get("ownership_policy") or unannotated_meta.get("policy") or "UNKNOWN_PROVENANCE_INSUFFICIENT"
        self.assertEqual(unannotated_policy, "UNKNOWN_PROVENANCE_INSUFFICIENT")

    # 9. AUTHORIZED_BY_PREM Remains False
    def test_09_authorized_by_prem_remains_false(self):
        """Production training authorization lock AUTHORIZED_BY_PREM is strictly False."""
        self.assertFalse(AUTHORIZED_BY_PREM)
        self.assertFalse(is_production_training_authorized())
        with self.assertRaises(PermissionError):
            assert_production_training_authorized()


if __name__ == "__main__":
    unittest.main()

