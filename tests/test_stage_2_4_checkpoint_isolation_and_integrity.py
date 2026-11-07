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
from webapp.model_service import (
    ModelService,
    validate_checkpoint_structure,
    is_verified_genuine_checkpoint,
    is_checkpoint_fixture
)
from scripts.promote_best_model import evaluate_and_promote
from scripts.validate_t4_empirical import verify_checkpoint_roundtrip_isolated
from model import GPTConfig, GPT
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

    def test_02_effective_config_manifest_hashes_and_unpromoted_pilot_metadata(self):
        """Training checkpoint metadata contains effective_config, manifest hashes, and strict unpromoted metadata."""
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

            # Top-level unpromoted pilot status invariants
            self.assertEqual(ckpt.get("status"), "ENGINEERING_SMOKE_TEST_ONLY")
            self.assertFalse(ckpt.get("is_production_trained", True))
            self.assertFalse(ckpt.get("promotable", True))
            self.assertFalse(ckpt.get("is_production_eligible", True))
            self.assertFalse(ckpt.get("is_test_fixture", True))

            # Config unpromoted pilot status invariants
            self.assertEqual(cfg_meta.get("status"), "ENGINEERING_SMOKE_TEST_ONLY")
            self.assertFalse(cfg_meta.get("is_production_trained", True))
            self.assertFalse(cfg_meta.get("promotable", True))
            self.assertFalse(cfg_meta.get("is_production_eligible", True))

    def test_03_scaled_config_isolation_and_policy_b_invariants(self):
        """scaled_config.yaml enforces strict Policy B and isolated output destinations."""
        scaled_yaml_path = os.path.join(PROJECT_ROOT, "training", "chat", "scaled_config.yaml")
        with open(scaled_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg.get("ownership_policy"), "TEACHER_STUDENT_RANDOM_INIT")
        self.assertEqual(cfg["teacher_distillation"]["student_init_method"], "random_gaussian_scratch")
        self.assertTrue(cfg["teacher_distillation"]["prohibit_teacher_checkpoint_weights"])
        self.assertEqual(cfg["paths"]["checkpoint_name"], "moon_chat_scaled_v1.pt")
        self.assertEqual(cfg["paths"]["best_checkpoint_name"], "moon_chat_scaled_best.pt")

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
                "is_production_trained": False,
                "promotable": False,
                "is_test_fixture": False,
                "model": {},
                "config": {"run_id": "real_training_run"}
            }
            torch.save(genuine_payload, mock_genuine)

            self.assertFalse(is_fixture_checkpoint(mock_genuine), "Genuine checkpoint must not be flagged as a fixture.")

            # Attempt cleanup on genuine checkpoint: must safely refuse to delete
            cleanup_checkpoint_fixture(mock_genuine, was_created=True)
            self.assertTrue(os.path.isfile(mock_genuine), "Genuine checkpoint was deleted! Safe guard failed.")

    def test_07_model_service_fail_closed_validation(self):
        """ModelService checkpoint validation fails closed on invalid, corrupted, or unverified checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Non-existent path
            non_existent = os.path.join(tmp_dir, "non_existent.pt")
            self.assertFalse(validate_checkpoint_structure(non_existent))
            self.assertFalse(is_verified_genuine_checkpoint(non_existent))

            # 2. Corrupted file (random non-pickle text)
            corrupt_file = os.path.join(tmp_dir, "corrupt.pt")
            with open(corrupt_file, "w", encoding="utf-8") as f:
                f.write("NOT_A_VALID_PYTORCH_CHECKPOINT_DATA_CORRUPTION")
            self.assertFalse(validate_checkpoint_structure(corrupt_file))
            self.assertFalse(is_verified_genuine_checkpoint(corrupt_file))
            self.assertFalse(is_checkpoint_fixture(corrupt_file))

            # 3. Checkpoint missing required model_args
            missing_args_file = os.path.join(tmp_dir, "missing_args.pt")
            torch.save({"model": {"dummy": torch.tensor([1.0])}}, missing_args_file)
            self.assertFalse(validate_checkpoint_structure(missing_args_file))

            # 4. Checkpoint missing model tensors
            missing_model_file = os.path.join(tmp_dir, "missing_model.pt")
            torch.save({
                "model": {},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512}
            }, missing_model_file)
            self.assertFalse(validate_checkpoint_structure(missing_model_file))

            # 5. Policy B violation (teacher weights transferred)
            violating_file = os.path.join(tmp_dir, "violating_policy_b.pt")
            torch.save({
                "status": "PILOT_TRAINED",
                "model": {"transformer.wte.weight": torch.randn(512, 512)},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
                "config": {
                    "teacher_checkpoint_weights_transferred": True,
                    "ownership_policy": "TEACHER_TRANSFER_FORBIDDEN"
                }
            }, violating_file)
            self.assertTrue(validate_checkpoint_structure(violating_file))
            self.assertFalse(is_verified_genuine_checkpoint(violating_file))

            # 6. Genuine unpromoted checkpoint passes structure and genuine check
            genuine_file = os.path.join(tmp_dir, "genuine_pilot.pt")
            torch.save({
                "status": "PILOT_TRAINED",
                "is_production_trained": False,
                "promotable": False,
                "is_test_fixture": False,
                "model": {"transformer.wte.weight": torch.randn(512, 512)},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
                "config": {
                    "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
                    "initialization": "random_gaussian_scratch",
                    "student_init_method": "random_gaussian_scratch",
                    "teacher_checkpoint_weights_transferred": False
                }
            }, genuine_file)
            self.assertTrue(validate_checkpoint_structure(genuine_file))
            self.assertTrue(is_verified_genuine_checkpoint(genuine_file))
            self.assertFalse(is_checkpoint_fixture(genuine_file))

    def test_08_t4_benchmark_tempfile_isolation_and_sentinel_protection(self):
        """verify_checkpoint_roundtrip_isolated safely roundtrips without touching existing checkpoint files."""
        ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat")
        os.makedirs(ckpt_dir, exist_ok=True)
        sentinel_path = os.path.join(ckpt_dir, "sentinel_t4_safety_test.pt")
        sentinel_content = b"CRITICAL_STAGE_CHECKPOINT_DATA_DO_NOT_DELETE"

        with open(sentinel_path, "wb") as f:
            f.write(sentinel_content)

        try:
            # Construct a minimal GPT model for roundtrip testing
            gpt_conf = GPTConfig(
                block_size=64,
                vocab_size=128,
                n_layer=2,
                n_head=2,
                n_embd=64,
                bias=False
            )
            device = torch.device("cpu")
            model = GPT(gpt_conf).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = None

            # Run isolated roundtrip
            res = verify_checkpoint_roundtrip_isolated(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                gpt_conf=gpt_conf,
                device=device,
                device_name="CPU_TEST"
            )
            self.assertTrue(res.get("verified"))

            # Verify sentinel is untouched
            self.assertTrue(os.path.isfile(sentinel_path), "Sentinel file was deleted!")
            with open(sentinel_path, "rb") as f:
                self.assertEqual(f.read(), sentinel_content, "Sentinel file content was modified!")

            # Verify no temporary files matching t4_empirical were leaked in ckpt_dir
            leaked_files = [fn for fn in os.listdir(ckpt_dir) if "t4_empirical" in fn or "tmp_t4_" in fn]
            self.assertEqual(len(leaked_files), 0, f"Leaked temporary files found in {ckpt_dir}: {leaked_files}")
        finally:
            if os.path.exists(sentinel_path):
                os.remove(sentinel_path)

    def test_09_pilot_checkpoint_anti_promotion_guard(self):
        """Unpromoted pilot checkpoints and raw training modes cannot be promoted to production."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")

            # Candidate report claiming raw pilot status cannot be promoted
            for disallowed_status in ["PILOT_TRAINED", "PILOT_TRAINED_BEST", "ENGINEERING_SMOKE_TEST_ONLY"]:
                cand_path = os.path.join(tmp_dir, f"cand_{disallowed_status}.json")
                with open(cand_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model_name": "moon-chat-pilot-candidate",
                        "status": disallowed_status,
                        "training_mode": disallowed_status,
                        "is_production_trained": False,
                        "promotable": False,
                        "composite_quality_score": 98.5,
                        "non_empty_rate_pct": 100.0,
                        "code_syntax_pct": 100.0
                    }, f)
                promoted = evaluate_and_promote(cand_path, registry_path=reg_path, base_dir=tmp_dir)
                self.assertFalse(promoted, f"Candidate with status {disallowed_status} must not be promoted!")


if __name__ == "__main__":
    unittest.main()
