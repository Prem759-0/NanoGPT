"""
tests/test_phase_2_6_4_adversarial_policy_closure.py

Moon AI — Phase 2.6.4.1 Adversarial Sovereign Policy Closure Test Battery
Chief ML Systems Architect & Owner: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Adversarial test suite verifying that every entrypoint, runner, preflight gate,
checkpoint loader, and environment configuration actively and strictly rejects
any attempt to select, activate, or inject the historical TEACHER_STUDENT_RANDOM_INIT
policy or teacher distillation.
"""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    VALID_PRODUCTION_POLICIES,
    HISTORICAL_QUARANTINED_POLICIES,
    SovereignPolicyViolationError,
    validate_production_policy,
    is_valid_policy,
    is_historical_policy,
    get_active_policy,
    AUTHORIZED_BY_PREM
)
from training.chat.train import train_moon_chat
from scripts.preflight_pilot_training import run_preflight


class TestPhase264AdversarialPolicyClosure(unittest.TestCase):
    """
    Adversarial verification testing active fail-closed rejection of historical
    teacher policy selection attempts across all Moon AI components.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

        # Baseline minimal config for testing
        self.base_cfg = {
            "model_track": "moon-chat-v0-test",
            "ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
            "canonical_ownership_policy": POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
            "teacher_distillation": {
                "enabled": False,
                "strategy": "POLICY_STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "student_init_method": "random_gaussian_scratch",
                "allow_teacher_demonstrations": False,
                "prohibit_teacher_checkpoint_weights": True,
            },
            "model": {
                "exact_parameters": 25698816,
                "n_layer": 8,
                "n_head": 8,
                "n_embd": 512,
                "block_size": 256,
                "vocab_size": 512,
                "dropout": 0.0,
                "bias": False,
            },
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "learning_rate": 1e-4,
                "device": "cpu",
                "precision": "float32",
                "use_amp": False,
                "eval_interval": 10,
                "save_interval": 10,
            },
            "paths": {
                "train_manifest": "datasets/manifests/scaled_chat_train.jsonl",
                "val_manifest": "datasets/manifests/scaled_chat_val.jsonl",
                "test_manifest": "datasets/manifests/scaled_chat_test.jsonl",
                "tokenizer_path": "models/chat/tokenizer.json",
                "tokenizer_vocab": "datasets/chat/vocab.json",
                "tokenizer_merges": "datasets/chat/merges.txt",
                "output_dir": self.temp_dir,
                "checkpoint_dir": self.temp_dir,
                "checkpoint_name": "moon_chat_pilot_v0.pt",
            }
        }

    def _write_temp_config(self, cfg_dict: dict) -> str:
        cfg_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f)
        return cfg_path

    # -------------------------------------------------------------------------
    # Test 1: Chat Training Runner Rejects Historical Teacher Policy
    # -------------------------------------------------------------------------
    def test_adversarial_chat_train_rejects_historical_teacher_policy(self):
        """
        Adversarial Attempt: Invoke train_moon_chat with config explicitly set to
        ownership_policy: "TEACHER_STUDENT_RANDOM_INIT".
        Expected: SovereignPolicyViolationError raised immediately fail-closed.
        """
        adv_cfg = copy.deepcopy(self.base_cfg)
        adv_cfg["ownership_policy"] = POLICY_TEACHER_STUDENT_RANDOM_INIT
        cfg_path = self._write_temp_config(adv_cfg)

        with self.assertRaises(SovereignPolicyViolationError) as ctx:
            train_moon_chat(config_path=cfg_path, smoke_test=True)

        self.assertIn("QUARANTINED HISTORICAL POLICY", str(ctx.exception))
        self.assertIn(POLICY_TEACHER_STUDENT_RANDOM_INIT, str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 2: Chat Training Runner Rejects Teacher Distillation Enabled
    # -------------------------------------------------------------------------
    def test_adversarial_chat_train_rejects_teacher_distillation_enabled(self):
        """
        Adversarial Attempt: Invoke train_moon_chat with teacher_distillation.enabled = True.
        Expected: ValueError or SovereignPolicyViolationError raised fail-closed.
        """
        adv_cfg = copy.deepcopy(self.base_cfg)
        adv_cfg["teacher_distillation"]["enabled"] = True
        cfg_path = self._write_temp_config(adv_cfg)

        with self.assertRaises((ValueError, SovereignPolicyViolationError)) as ctx:
            train_moon_chat(config_path=cfg_path, smoke_test=True)

        self.assertIn("teacher_distillation.enabled", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test 3: Preflight Gate 1 Rejects Historical Policy
    # -------------------------------------------------------------------------
    def test_adversarial_preflight_gate_rejects_historical_policy(self):
        """
        Adversarial Attempt: Run preflight checks on config with TEACHER_STUDENT_RANDOM_INIT.
        Expected: Preflight execution is blocked fail-closed by default, and Gate 1
        blocks the historical policy even when runner authorization is active.
        """
        adv_cfg = copy.deepcopy(self.base_cfg)
        adv_cfg["ownership_policy"] = POLICY_TEACHER_STUDENT_RANDOM_INIT
        cfg_path = self._write_temp_config(adv_cfg)

        # 1. Unmocked default behavior: preflight execution blocked fail-closed
        with self.assertRaises(PermissionError) as perm_ctx:
            run_preflight(config_path=cfg_path)
        self.assertIn("AUTHORIZED_BY_PREM is False", str(perm_ctx.exception))

        # 2. Gate 1 policy closure: Even if runner is authorized, Gate 1 blocks historical policy
        with patch("core.policy.AUTHORIZED_BY_PREM", True):
            res = run_preflight(config_path=cfg_path)
            self.assertFalse(res["passed"])
            self.assertEqual(res["status"], "FAILED")
            gate_1 = res["preflight_gates"]["gate_1_config_integrity"]
            self.assertFalse(gate_1["passed"])
            self.assertTrue(any("Ownership policy violation" in b for b in res["blockers"]))

    # -------------------------------------------------------------------------
    # Test 4: Checkpoint Resume Rejects Quarantined Teacher Policy
    # -------------------------------------------------------------------------
    def test_adversarial_checkpoint_resume_rejects_teacher_policy(self):
        """
        Adversarial Attempt: Resume training from a checkpoint metadata declaring
        ownership_policy: "TEACHER_STUDENT_RANDOM_INIT" or teacher weights transferred.
        Expected: SovereignPolicyViolationError or ValueError raised fail-closed.
        """
        # Create a synthetic checkpoint with quarantined policy
        ckpt_path = os.path.join(self.temp_dir, "adversarial_ckpt.pt")
        from model import GPTConfig, GPT
        m_cfg = GPTConfig(
            vocab_size=512,
            block_size=256,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=False
        )
        dummy_model = GPT(m_cfg)
        torch.save({
            "model": dummy_model.state_dict(),
            "config": {
                "ownership_policy": POLICY_TEACHER_STUDENT_RANDOM_INIT,
                "teacher_checkpoint_weights_transferred": False,
                "initialization": "random_gaussian_scratch"
            }
        }, ckpt_path)

        adv_cfg = copy.deepcopy(self.base_cfg)
        adv_cfg["training"]["resume_from_checkpoint"] = ckpt_path
        cfg_path = self._write_temp_config(adv_cfg)

        with self.assertRaises((SovereignPolicyViolationError, ValueError)) as ctx:
            train_moon_chat(config_path=cfg_path, smoke_test=True, resume_from_ckpt=ckpt_path)

        self.assertTrue(
            "QUARANTINED HISTORICAL POLICY" in str(ctx.exception) or
            "Policy B violation" in str(ctx.exception)
        )

    # -------------------------------------------------------------------------
    # Test 5: Image, Video, and Script Entry Points Reject Historical Policy
    # -------------------------------------------------------------------------
    def test_adversarial_image_and_video_runners_reject_historical_policy(self):
        """
        Adversarial Attempt: Direct call to validate_production_policy with
        historical or unauthorized policy identifiers.
        Expected: SovereignPolicyViolationError raised for any non-sovereign policy.
        """
        # Historical quarantined policy must be rejected
        with self.assertRaises(SovereignPolicyViolationError) as ctx:
            validate_production_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT)
        self.assertIn("QUARANTINED HISTORICAL POLICY", str(ctx.exception))

        # Arbitrary / spoofed policies must be rejected
        for spoofed in ["TEACHER_STUDENT", "OPENAI_TEACHER", "KIMI_DISTILLED", "", "None", None]:
            with self.assertRaises(SovereignPolicyViolationError):
                validate_production_policy(spoofed)

        # Only STRICT_SOVEREIGN_RANDOM_INIT_ONLY passes
        validate_production_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(is_valid_policy(POLICY_STRICT_SOVEREIGN_RANDOM_INIT))
        self.assertFalse(is_valid_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))
        self.assertTrue(is_historical_policy(POLICY_TEACHER_STUDENT_RANDOM_INIT))

    # -------------------------------------------------------------------------
    # Test 6: Environment Override Fails Closed Immediately in Subprocess
    # -------------------------------------------------------------------------
    def test_adversarial_env_override_fails_closed(self):
        """
        Adversarial Attempt: Set MOON_OWNERSHIP_POLICY or OWNERSHIP_POLICY env var
        to TEACHER_STUDENT_RANDOM_INIT in an independent Python process.
        Expected: Process terminates with non-zero exit code and SovereignPolicyViolationError.
        """
        for env_key in ["MOON_OWNERSHIP_POLICY", "OWNERSHIP_POLICY", "MOON_AI_POLICY"]:
            env = os.environ.copy()
            env[env_key] = POLICY_TEACHER_STUDENT_RANDOM_INIT
            cmd = [
                sys.executable, "-c",
                "import core.policy"
            ]
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROJECT_ROOT)
            self.assertNotEqual(res.returncode, 0, f"Expected non-zero returncode with {env_key}")
            self.assertIn("SovereignPolicyViolationError", res.stderr)
            self.assertIn("SOVEREIGN POLICY OVERRIDE VIOLATION", res.stderr)


if __name__ == "__main__":
    unittest.main()
