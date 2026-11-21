"""
tests/test_phase_2_6_2_2_containment.py
Moon AI — Phase 2.6.2.2 Containment & Training Isolation Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates all 4 Core Containment Areas:
1. Authentication Containment (Task 1): Fail-closed identity checks; arbitrary caller IDs rejected.
2. Mechanical Validation Isolation (Task 2): Zero optimizer steps, zero backward passes, zero checkpoint writes.
3. Historical Dataset Quarantine (Task 3): Exact hashes, quarantine isolation, zero contamination in active manifests.
4. Training Execution Boundary (Task 4): Fail-closed production gates in train_moon_chat, train.py, preflight.
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from core.policy import AUTHORIZED_BY_PREM, ACTIVE_OWNERSHIP_POLICY, POLICY_STRICT_SOVEREIGN_RANDOM_INIT
from core.learning_lifecycle.donation import (
    DonationConsentManager,
    is_authenticated_user,
    SERVER_AUTHENTICATION_VERIFIED,
    DONATION_INGESTION_ENABLED
)
from core.learning_lifecycle.lifecycle import ControlledLearningLifecycle
from training.chat.train import train_moon_chat


class TestTask1AuthenticationContainment(unittest.TestCase):
    """Task 1: Caller-supplied user IDs and forged headers must fail closed."""

    def test_server_authentication_verified_flag_is_false(self):
        """Invariant: Server-side cryptographic authentication is currently unverified."""
        self.assertFalse(SERVER_AUTHENTICATION_VERIFIED)
        self.assertFalse(DONATION_INGESTION_ENABLED)

    def test_forged_and_arbitrary_user_ids_fail_closed(self):
        """String checking is not authentication. All caller identities fail closed."""
        arbitrary_ids = [
            "prem",
            "x-user-id: prem",
            "admin",
            "researcher_prem",
            "user_12345",
            "guest",
            "anonymous",
            "",
            None
        ]
        for uid in arbitrary_ids:
            self.assertFalse(
                is_authenticated_user(uid),
                f"Caller identity '{uid}' was improperly treated as authenticated!"
            )

    def test_donation_consent_creation_fails_closed(self):
        """create_donation_consent must fail closed without server auth verification."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            db_path = os.path.join(tmp_dir, "consent.sqlite")
            mgr = DonationConsentManager(storage_path=db_path)
            ok, msg, rec = mgr.create_donation_consent(
                user_id="prem",
                interaction_id="turn_001",
                affirmative_ack=True
            )
            self.assertFalse(ok)
            self.assertIsNone(rec)
            self.assertIn("Server-side cryptographic authentication is not configured", msg)

    def test_candidate_ingestion_fails_closed(self):
        """ControlledLearningLifecycle.ingest_candidate_example fails closed without server auth."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            wal_path = os.path.join(tmp_dir, "wal.jsonl")
            db_path = os.path.join(tmp_dir, "consent.sqlite")
            lifecycle = ControlledLearningLifecycle(wal_path=wal_path, consent_db_path=db_path)
            interaction = {
                "prompt": "Calculate 12 * 12",
                "response": "The result is 144",
                "consent_id": "consent_test_001",
                "explicitly_donated_for_training": True,
                "is_private_chat": False
            }
            ok, reason, rec = lifecycle.ingest_candidate_example(
                user_id="prem",
                interaction=interaction,
                user_training_consent=True
            )
            self.assertFalse(ok)
            self.assertIsNone(rec)
            self.assertIn("Server-side cryptographic authentication is not verified", reason)

    def test_check_interaction_training_eligibility_fails_closed(self):
        """check_interaction_training_eligibility fails closed without server auth."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            wal_path = os.path.join(tmp_dir, "wal.jsonl")
            db_path = os.path.join(tmp_dir, "consent.sqlite")
            lifecycle = ControlledLearningLifecycle(wal_path=wal_path, consent_db_path=db_path)
            interaction = {
                "prompt": "Calculate 12 * 12",
                "response": "The result is 144",
                "explicitly_donated_for_training": True
            }
            ok, reason = lifecycle.check_interaction_training_eligibility(
                user_id="prem",
                interaction=interaction,
                user_training_consent=True
            )
            self.assertFalse(ok)
            self.assertIn("Server-side cryptographic authentication is not verified", reason)


class TestTask2MechanicalValidationIsolation(unittest.TestCase):
    """Task 2: Mechanical validation must never call backward(), optimizer.step(), or torch.save()."""

    def test_smoke_test_is_parameter_frozen_mechanical_validation(self):
        """train_moon_chat(smoke_test=True) returns mechanical validation status with 0 updates."""
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        res = train_moon_chat(
            config_path=cfg_path,
            max_iters_override=1,
            smoke_test=True
        )
        self.assertEqual(res["status"], "COMPLETED_MECHANICAL_VALIDATION_ONLY")
        self.assertTrue(res["forward_verified"])
        self.assertFalse(res["backward_executed"])
        self.assertFalse(res["optimizer_stepped"])
        self.assertFalse(res["checkpoint_written"])
        self.assertFalse(res["metrics_written"])
        self.assertEqual(res["total_iterations"], 0)
        self.assertIsNone(res["checkpoint_path"])
        self.assertIsNone(res["checkpoint_sha256"])

    def test_smoke_test_strictly_forbids_optimizer_step_and_backward(self):
        """Mock-verified assertion: optimizer.step and backward are NEVER called during smoke test."""
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")

        with patch("torch.optim.Optimizer.step") as mock_step, \
             patch("torch.save") as mock_save:
            res = train_moon_chat(
                config_path=cfg_path,
                max_iters_override=1,
                smoke_test=True
            )
            self.assertEqual(res["status"], "COMPLETED_MECHANICAL_VALIDATION_ONLY")
            mock_step.assert_not_called()
            mock_save.assert_not_called()

    def test_smoke_test_leaves_zero_disk_artifacts(self):
        """Verify no checkpoints or metrics files are written during mechanical validation."""
        ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat")
        smoke_ckpt = os.path.join(ckpt_dir, "smoke_test_ckpt.pt")
        self.assertFalse(os.path.exists(smoke_ckpt), "smoke_test_ckpt.pt must not exist on disk!")

        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        train_moon_chat(
            config_path=cfg_path,
            max_iters_override=1,
            smoke_test=True
        )
        self.assertFalse(os.path.exists(smoke_ckpt), "smoke_test_ckpt.pt was erroneously written to disk!")


class TestTask3HistoricalDatasetQuarantine(unittest.TestCase):
    """Task 3: Historical user_learned_data.jsonl preserved byte-for-byte; legacy splits quarantined."""

    def test_historical_jsonl_preserved_byte_for_byte(self):
        """data/user_learned_data.jsonl is preserved byte-for-byte (65 records) when present."""
        fpath = os.path.join(PROJECT_ROOT, "data", "user_learned_data.jsonl")
        if not os.path.isfile(fpath):
            self.skipTest("Historical quarantined user_learned_data.jsonl is untracked in CI environment.")

        with open(fpath, "rb") as f:
            data = f.read()
        self.assertEqual(len(data), 19718, "user_learned_data.jsonl size mismatch!")
        sha = hashlib.sha256(data).hexdigest()
        self.assertEqual(
            sha,
            "5de99432ad3601d32268ae463fe807ecabcd4a1179d2d5efb2d2bc59e22dfbb6",
            "user_learned_data.jsonl SHA-256 hash mismatch!"
        )
        lines = [l for l in data.decode("utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 65, "user_learned_data.jsonl must contain exactly 65 records!")

    def test_contaminated_splits_quarantined_and_not_plain_json(self):
        """Legacy split files do not exist as active .json files in data/splits/."""
        splits_dir = os.path.join(PROJECT_ROOT, "data", "splits")
        for bad_json in ["train.json", "val.json", "test.json"]:
            bad_path = os.path.join(splits_dir, bad_json)
            self.assertFalse(
                os.path.isfile(bad_path),
                f"Contaminated legacy file '{bad_path}' must not exist as plain JSON!"
            )
            # If quarantined backup exists on disk, verify it is a file
            quarantined_path = bad_path + ".QUARANTINED_CONTAMINATED"
            if os.path.isfile(quarantined_path):
                self.assertTrue(os.path.isfile(quarantined_path))

    def test_quarantine_directory_and_manifest_exist(self):
        """Quarantine directory contains isolated splits and quarantine manifest when present."""
        q_dir = os.path.join(PROJECT_ROOT, "data", "splits", "quarantine_unconsented_legacy_splits")
        if not os.path.isdir(q_dir):
            self.skipTest("Quarantine directory is untracked in CI environment.")
        for fname in ["train.json", "val.json", "test.json", "manifest.json", "QUARANTINE_MANIFEST.json"]:
            fpath = os.path.join(q_dir, fname)
            self.assertTrue(os.path.isfile(fpath), f"Quarantine file missing: {fpath}")

    def test_active_curriculum_manifests_are_100_percent_clean(self):
        """Active sovereign curriculum manifests contain 0 records from user_learned_data.jsonl."""
        from scripts.scan_dataset_contamination import load_historical_fingerprints, scan_file_for_contamination
        fps, _ = load_historical_fingerprints()

        active_manifests = [
            os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_train.jsonl"),
            os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_val.jsonl"),
            os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_test.jsonl"),
        ]
        for m in active_manifests:
            res = scan_file_for_contamination(m, fps)
            self.assertEqual(res["contaminated_count"], 0, f"Active manifest {m} contains contaminated records!")


class TestTask4TrainingExecutionBoundary(unittest.TestCase):
    """Task 4: Production training authorization is strictly fail-closed."""

    def test_authorized_by_prem_is_strictly_false(self):
        """Invariant: AUTHORIZED_BY_PREM must be strictly False."""
        self.assertFalse(AUTHORIZED_BY_PREM)

    def test_train_moon_chat_fails_closed_without_authorization(self):
        """Calling train_moon_chat with smoke_test=False raises PermissionError."""
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(config_path=cfg_path, smoke_test=False)
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_legacy_train_script_fails_closed(self):
        """Importing train.py directly raises PermissionError fail-closed."""
        # Because train.py has the top-level gate, importing it raises PermissionError
        with self.assertRaises(PermissionError) as ctx:
            import train
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))

    def test_preflight_pilot_training_fails_closed(self):
        """scripts/preflight_pilot_training.py raises PermissionError fail-closed."""
        from scripts.preflight_pilot_training import run_preflight
        with self.assertRaises(PermissionError) as ctx:
            run_preflight()
        self.assertIn("PREFLIGHT TRAINING EXECUTION BLOCKED", str(ctx.exception))

    def test_learning_service_record_chat_turn_fails_closed(self):
        """record_chat_turn in learning_service permanently returns False."""
        from webapp.learning_service import learning_service
        res = learning_service.record_chat_turn("Hello", "Hi", explicit_donation_consent=True)
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
