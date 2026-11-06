"""
tests/test_stage_2_4_checkpoint_isolation_and_integrity.py
Unit & Integration Verification for Moon AI Phase 2.4:
1. Training Checkpoint Isolation (Phase 2.4 vs Stage 2.2).
2. Configurable best-checkpoint filenames (zero cross-stage collisions).
3. Manifest hashes and effective configuration embedded in checkpoint metadata.
4. CI Test Fixture integrity & anti-promotion enforcement.
5. ModelService fixture discovery rejection (fixtures never treated as available trained models).

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)
"""

import os
import sys
import json
import yaml
import torch
import hashlib
import tempfile
import unittest
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.chat.train import train_moon_chat
from webapp.model_service import ModelService
from scripts.promote_best_model import evaluate_and_promote
from tests.checkpoint_fixture_utils import (
    ensure_pilot_checkpoint_fixture,
    cleanup_checkpoint_fixture,
    is_fixture_checkpoint
)


class TestStage24CheckpointIsolationAndIntegrity(unittest.TestCase):
    """Test suite certifying Phase 2.4 checkpoint isolation and fixture integrity."""

    def test_01_phase_2_4_config_checkpoint_isolation(self):
        """Phase 2.4 checkpoint filenames are distinct and isolated from Stage 2.2."""
        pilot_yaml_path = os.path.join(PROJECT_ROOT, "training", "chat", "phase_2_4_t4_pilot.yaml")
        scaled_yaml_path = os.path.join(PROJECT_ROOT, "training", "chat", "scaled_config.yaml")

        self.assertTrue(os.path.isfile(pilot_yaml_path), f"Missing {pilot_yaml_path}")
        self.assertTrue(os.path.isfile(scaled_yaml_path), f"Missing {scaled_yaml_path}")

        with open(pilot_yaml_path, "r", encoding="utf-8") as f:
            pilot_cfg = yaml.safe_load(f)
        with open(scaled_yaml_path, "r", encoding="utf-8") as f:
            scaled_cfg = yaml.safe_load(f)

        # Final checkpoint names
        p24_final = pilot_cfg["paths"]["checkpoint_name"]
        s22_final = scaled_cfg["paths"]["checkpoint_name"]
        self.assertEqual(p24_final, "moon_chat_phase_2_4_t4_pilot_v0.pt")
        self.assertEqual(s22_final, "moon_chat_scaled_v1.pt")
        self.assertNotEqual(p24_final, s22_final)

        # Best checkpoint names
        p24_best = pilot_cfg["paths"]["best_checkpoint_name"]
        s22_best = scaled_cfg["paths"]["best_checkpoint_name"]
        self.assertEqual(p24_best, "moon_chat_phase_2_4_t4_pilot_best.pt")
        self.assertEqual(s22_best, "moon_chat_scaled_best.pt")
        self.assertNotEqual(p24_best, s22_best)

        # Zero collision guarantee
        all_names = {p24_final, p24_best, s22_final, s22_best}
        self.assertEqual(len(all_names), 4, "Checkpoint names must be pairwise distinct.")

    def test_02_effective_config_and_manifest_hashes_saved_in_checkpoint(self):
        """Training checkpoint metadata contains effective_config and manifest SHA-256 hashes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_ckpt = os.path.join(tmp_dir, "test_effective_meta.pt")
            pilot_yaml_path = os.path.join(PROJECT_ROOT, "training", "chat", "phase_2_4_t4_pilot.yaml")

            # Run 1-step smoke test with custom checkpoint path
            res = train_moon_chat(
                config_path=pilot_yaml_path,
                max_iters_override=1,
                smoke_test=True,
                custom_ckpt_path=temp_ckpt
            )
            self.assertTrue(os.path.isfile(temp_ckpt))

            ckpt = torch.load(temp_ckpt, map_location="cpu", weights_only=False)
            cfg_meta = ckpt.get("config", {})

            # Effective config saved
            self.assertIn("effective_config", cfg_meta)
            self.assertEqual(cfg_meta["effective_config"]["model"]["name"], "moon-chat-t4-pilot-25m")

            # Manifest hashes saved
            self.assertIn("manifest_hashes", cfg_meta)
            m_hashes = cfg_meta["manifest_hashes"]
            for m_key in ["train_manifest", "val_manifest", "test_manifest"]:
                self.assertIn(m_key, m_hashes)
                self.assertIn("path", m_hashes[m_key])
                self.assertIn("sha256", m_hashes[m_key])
                sha = m_hashes[m_key]["sha256"]
                self.assertIsInstance(sha, str)
                self.assertEqual(len(sha), 64, f"Invalid SHA-256 for {m_key}: {sha}")

    def test_03_smoke_test_checkpoints_isolated_and_unpromotable(self):
        """Smoke test checkpoints have status=ENGINEERING_SMOKE_TEST_ONLY and promotable=False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_ckpt = os.path.join(tmp_dir, "smoke_test_ckpt.pt")
            scaled_yaml_path = os.path.join(PROJECT_ROOT, "training", "chat", "scaled_config.yaml")

            res = train_moon_chat(
                config_path=scaled_yaml_path,
                max_iters_override=1,
                smoke_test=True,
                custom_ckpt_path=temp_ckpt
            )
            ckpt = torch.load(temp_ckpt, map_location="cpu", weights_only=False)

            self.assertEqual(ckpt["status"], "ENGINEERING_SMOKE_TEST_ONLY")
            self.assertFalse(ckpt["is_production_trained"])
            self.assertFalse(ckpt["promotable"])

    def test_04_ci_fixtures_cannot_be_promoted(self):
        """Model promotion gate strictly rejects any candidate marked as a CI test fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            cand_path = os.path.join(tmp_dir, "candidate_report.json")
            cand_report = {
                "model_name": "candidate_fixture",
                "status": "CI_TEST_FIXTURE",
                "training_mode": "CI_TEST_FIXTURE",
                "is_test_fixture": True,
                "is_production_trained": False,
                "is_production_eligible": False,
                "promotable": False,
                "composite_quality_score": 95.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0
            }
            with open(cand_path, "w", encoding="utf-8") as f:
                json.dump(cand_report, f)

            promoted = evaluate_and_promote(cand_path, registry_path=reg_path, base_dir=tmp_dir)
            self.assertFalse(promoted, "evaluate_and_promote must return False for CI test fixtures.")

    def test_05_ci_fixtures_not_available_in_model_service(self):
        """ModelService does not mark a CI test fixture as an available trained model."""
        fixture_path, was_created = ensure_pilot_checkpoint_fixture()
        try:
            ms = ModelService(base_dir=PROJECT_ROOT)
            models = ms.get_available_models()
            pilot_entries = [m for m in models if m["id"] == "moon-chat-pilot-v0"]
            self.assertEqual(len(pilot_entries), 1)
            pilot = pilot_entries[0]

            # If it's a fixture, available must be False
            if is_fixture_checkpoint(fixture_path):
                self.assertFalse(pilot["available"], "CI fixture must not be marked available as a trained production model.")
                self.assertTrue(pilot.get("is_test_fixture", False))
        finally:
            cleanup_checkpoint_fixture(fixture_path, was_created)

    def test_06_genuine_checkpoint_protection(self):
        """cleanup_checkpoint_fixture and is_fixture_checkpoint protect genuine checkpoints from deletion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_genuine = os.path.join(tmp_dir, "mock_genuine_checkpoint.pt")
            genuine_payload = {
                "status": "PILOT_TRAINED",
                "is_production_trained": True,
                "promotable": True,
                "is_test_fixture": False,
                "model": {},
                "config": {"run_id": "real_training_run"}
            }
            torch.save(genuine_payload, mock_genuine)

            self.assertFalse(is_fixture_checkpoint(mock_genuine), "Genuine checkpoint must not be flagged as a fixture.")

            # Attempt cleanup on genuine checkpoint: must safely refuse to delete
            cleanup_checkpoint_fixture(mock_genuine, was_created=True)
            self.assertTrue(os.path.isfile(mock_genuine), "Genuine checkpoint was deleted! Safe guard failed.")


if __name__ == "__main__":
    unittest.main()
