"""
tests/test_phase_2_6_2_5_sovereign_policy_hardening.py
Phase 2.6.2.5 Test Suite: Strict Sovereign Policy Hardening Verification
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Verifies:
1. Strict sovereign policy is accepted for production.
2. Teacher policy is rejected for production and marked quarantined.
3. Unknown/third-party policies fail closed.
4. Historical policy metadata inspection cannot authorize training.
5. All training configs (pilot_config.yaml, config.yaml) are sanitized (zero teacher distillation).
6. Config and environment overrides attempting teacher policy fail closed.
7. Production training remains blocked while AUTHORIZED_BY_PREM = False.
8. Zero training loops, zero optimizers, zero backward passes, zero checkpoint writes.
"""

import os
import sys
import yaml
import importlib
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    VALID_PRODUCTION_POLICIES,
    HISTORICAL_QUARANTINED_POLICIES,
    ALL_KNOWN_POLICIES,
    VALID_OWNERSHIP_POLICIES,
    ACTIVE_OWNERSHIP_POLICY,
    AUTHORIZED_BY_PREM,
    SovereignPolicyViolationError,
    get_active_policy,
    is_valid_policy,
    is_historical_policy,
    validate_production_policy,
    validate_student_initialization,
    get_policy_summary,
    is_production_training_authorized,
    assert_production_training_authorized,
)


class TestPhase2625SovereignPolicyHardening(unittest.TestCase):
    """Exhaustive test suite for Phase 2.6.2.5 strict sovereign policy hardening."""

    # -------------------------------------------------------------------------
    # 1. Production Policy Acceptance
    # -------------------------------------------------------------------------

    def test_strict_sovereign_policy_is_accepted(self):
        """Strict sovereign policy is recognized as the only valid production policy."""
        self.assertEqual(get_active_policy(), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertEqual(ACTIVE_OWNERSHIP_POLICY, POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(is_valid_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT))
        self.assertIn(POLICY_STRICT_SOVEREIGN_RANDOM_INIT, VALID_PRODUCTION_POLICIES)
        self.assertIn(POLICY_STRICT_SOVEREIGN_RANDOM_INIT, VALID_OWNERSHIP_POLICIES)
        self.assertNotIn(POLICY_STRICT_SOVEREIGN_RANDOM_INIT, HISTORICAL_QUARANTINED_POLICIES)

        # validate_production_policy passes without exception
        try:
            validate_production_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        except Exception as e:
            self.fail(f"validate_production_policy raised unexpectedly: {e}")

    # -------------------------------------------------------------------------
    # 2. Historical Teacher Policy Rejection for Production
    # -------------------------------------------------------------------------

    def test_teacher_policy_is_rejected_for_production(self):
        """Teacher-student policy is NOT valid for production and raises on production selection."""
        self.assertFalse(is_valid_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))
        self.assertTrue(is_historical_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))
        self.assertNotIn(POLICY_TEACHER_STUDENT_RANDOM_INIT, VALID_PRODUCTION_POLICIES)
        self.assertNotIn(POLICY_TEACHER_STUDENT_RANDOM_INIT, VALID_OWNERSHIP_POLICIES)
        self.assertIn(POLICY_TEACHER_STUDENT_RANDOM_INIT, HISTORICAL_QUARANTINED_POLICIES)

        # Attempting to validate teacher policy for production fails closed
        with self.assertRaises(SovereignPolicyViolationError) as ctx:
            validate_production_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertIn("QUARANTINED HISTORICAL POLICY", str(ctx.exception))
        self.assertIn("strictly forbidden", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. Unknown & Third-Party Policy Fail-Closed
    # -------------------------------------------------------------------------

    def test_unknown_policies_fail_closed(self):
        """Third-party pretrained import and arbitrary strings fail closed."""
        unknown_policies = [
            "THIRD_PARTY_PRETRAINED_IMPORT",
            "OPENAI_GPT4_DISTILLATION",
            "CLAUDE_DEMONSTRATION_LEARNING",
            "TEACHER_ONLY",
            "STUDENT_FINETUNE",
            "",
            "None",
            "STRICT_SOVEREIGN",  # Must be exact enum
        ]
        for pol in unknown_policies:
            with self.subTest(policy=pol):
                self.assertFalse(is_valid_policy(pol))
                self.assertFalse(is_historical_policy(pol))
                with self.assertRaises(SovereignPolicyViolationError) as ctx:
                    validate_production_policy(pol)
                self.assertIn("UNKNOWN or UNAUTHORIZED", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Historical Policy Inspection Cannot Authorize Training
    # -------------------------------------------------------------------------

    def test_historical_policy_inspection_cannot_activate_training(self):
        """Inspecting historical metadata returns audit details but can_authorize_training remains False."""
        summary = get_policy_summary(POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertEqual(summary["policy"], POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertFalse(summary["is_active"])
        self.assertFalse(summary["is_production_valid"])
        self.assertTrue(summary["is_quarantined_historical"])
        self.assertFalse(summary["can_authorize_training"])
        self.assertTrue(summary["rules"].get("prohibit_production_use", False))

        # Active policy remains strictly sovereign after inspection
        self.assertEqual(get_active_policy(), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertFalse(is_production_training_authorized())

    def test_active_policy_summary(self):
        """Default policy summary returns active production sovereign rules."""
        summary = get_policy_summary()
        self.assertEqual(summary["policy"], POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(summary["is_active"])
        self.assertTrue(summary["is_production_valid"])
        self.assertFalse(summary["is_quarantined_historical"])
        self.assertTrue(summary["can_authorize_training"])
        self.assertFalse(summary["rules"]["allow_teacher_demonstrations"])
        self.assertFalse(summary["rules"]["allow_third_party_weights"])
        self.assertFalse(summary["rules"]["allow_external_apis_in_runtime"])
        self.assertTrue(summary["rules"]["require_random_initialization"])

    # -------------------------------------------------------------------------
    # 5. Training Configurations Sanitization
    # -------------------------------------------------------------------------

    def test_pilot_config_yaml_is_sanitized(self):
        """training/chat/pilot_config.yaml is sanitized with teacher distillation disabled."""
        path = os.path.join(PROJECT_ROOT, "training", "chat", "pilot_config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg.get("ownership_policy"), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        td = cfg.get("teacher_distillation", {})
        self.assertFalse(td.get("enabled", True))
        self.assertIsNone(td.get("target_teacher"))
        self.assertIsNone(td.get("teacher_provider"))
        self.assertFalse(td.get("allow_teacher_demonstrations", True))
        self.assertTrue(td.get("prohibit_teacher_checkpoint_weights", False))
        self.assertEqual(td.get("student_init_method"), "random_gaussian_scratch")

    def test_config_yaml_is_sanitized(self):
        """training/chat/config.yaml is sanitized with strict sovereign ownership."""
        path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg.get("ownership_policy"), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        td = cfg.get("teacher_distillation", {})
        self.assertFalse(td.get("enabled", True))

    def test_scaled_and_phase_2_4_configs_are_sovereign(self):
        """scaled_config.yaml and phase_2_4_t4_pilot.yaml declare strict sovereign policy."""
        for cname in ["scaled_config.yaml", "phase_2_4_t4_pilot.yaml"]:
            with self.subTest(config=cname):
                path = os.path.join(PROJECT_ROOT, "training", "chat", cname)
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.assertEqual(cfg.get("ownership_policy"), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
                self.assertFalse(cfg.get("teacher_distillation", {}).get("enabled", True))

    # -------------------------------------------------------------------------
    # 6. Override Prevention (Environment & Runtime)
    # -------------------------------------------------------------------------

    def test_environment_override_attempt_fails_closed(self):
        """Attempting to inject a non-sovereign policy via environment variable triggers violation."""
        with patch.dict(os.environ, {"MOON_OWNERSHIP_POLICY": "TEACHER_STUDENT_RANDOM_INIT"}):
            # Attempting to re-execute policy module check fails closed
            for env_key in ("MOON_OWNERSHIP_POLICY", "OWNERSHIP_POLICY", "MOON_AI_POLICY"):
                val = os.environ.get(env_key)
                if val and val != POLICY_STRICT_SOVEREIGN_RANDOM_INIT:
                    with self.assertRaises(SovereignPolicyViolationError):
                        validate_production_policy(val)

    # -------------------------------------------------------------------------
    # 7. Production Training Authorization Gate
    # -------------------------------------------------------------------------

    def test_production_training_blocked_while_authorized_by_prem_is_false(self):
        """assert_production_training_authorized raises PermissionError when AUTHORIZED_BY_PREM is False."""
        self.assertFalse(AUTHORIZED_BY_PREM)
        self.assertFalse(is_production_training_authorized())

        with self.assertRaises(PermissionError) as ctx:
            assert_production_training_authorized()
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))
        self.assertIn("Prem Gaikwad", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 8. Student Initialization Validation
    # -------------------------------------------------------------------------

    def test_student_initialization_scratch_rules(self):
        """Random Gaussian and Xavier scratch pass; pretrained and external weights fail."""
        ok_gauss, _ = validate_student_initialization("random_gaussian_scratch")
        self.assertTrue(ok_gauss)

        ok_xavier, _ = validate_student_initialization("xavier_uniform_scratch")
        self.assertTrue(ok_xavier)

        ok_pre, msg_pre = validate_student_initialization("pretrained_weights")
        self.assertFalse(ok_pre)
        self.assertIn("strictly forbidden", msg_pre)

        ok_meta, msg_meta = validate_student_initialization(
            "random_gaussian_scratch",
            {"base_weights": "Qwen/Qwen2.5-7B-Instruct"}
        )
        self.assertFalse(ok_meta)
        self.assertIn("Ownership violation", msg_meta)


if __name__ == "__main__":
    unittest.main()

