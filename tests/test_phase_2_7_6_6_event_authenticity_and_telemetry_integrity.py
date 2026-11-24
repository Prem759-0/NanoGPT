"""
tests/test_phase_2_7_6_6_event_authenticity_and_telemetry_integrity.py
Adversarial Verification Suite for Phase 2.7.6.6:
- Mandatory Hash Completeness & Format Validation (env, optim, arch, data, code, logs, ckpt)
- Optimizer Telemetry Completeness (non-empty step_records, finite lr/grad_norm, final step equality)
- Training Telemetry Log Integrity ([RUN_INIT], ordered [STEP_TELEMETRY], valid finite loss, [RUN_COMPLETE] status=SUCCESS)
- Independent Training-Event Evidence Boundary Verification
- Production Inference Isolation & Invariant Protection (AUTHORIZED_BY_PREM = False)

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import hashlib
import math
from datetime import datetime, timezone, timedelta
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import policy
from core.security.trust_provider import (
    BaseTrustRootProvider,
    ExternalTrustRootProvider,
    TestIsolatedTrustProvider,
)
from core.security.training_evidence import (
    EvidenceArtifactStore,
    TrainingEvidenceValidator,
    create_sample_training_artifacts,
    create_sample_training_evidence,
    compute_file_sha256,
)
from core.security.training_attestation import (
    verify_training_attestation,
    create_test_training_attestation,
    generate_training_authority_keypair,
    sign_training_attestation,
    compute_bytes_sha256,
)
from webapp.model_service import (
    ModelService,
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    ModelNotTrainedException,
)


class TestPhase2766EventAuthenticityAndTelemetryIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.priv_hex, self.pub_hex = generate_training_authority_keypair()
        self.verifier_id = "phase_2_7_6_6_auditor"
        self.authority_entry = {
            self.verifier_id: {
                "name": "Phase 2.7.6.6 Authority",
                "public_key_hex": self.pub_hex,
                "status": "active",
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        self.test_provider = TestIsolatedTrustProvider(self.authority_entry)

        # Create external keystore file
        self.ext_keystore_path = os.path.join(self.tmpdir, "trusted_authorities.json")
        with open(self.ext_keystore_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.1",
                "policy": "ASYMMETRIC_ED25519_ONLY",
                "trusted_authorities": self.authority_entry,
            }, f)
        self.production_provider = ExternalTrustRootProvider(
            external_path=self.ext_keystore_path, base_dir=PROJECT_ROOT
        )

        # Create valid synthetic checkpoint fixture file on disk
        self.ckpt_path = os.path.join(self.tmpdir, "candidate_scratch.pt")
        dummy_state = {
            "model": {"transformer.wte.weight": torch.randn(512, 128)},
            "model_args": {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
                "vocab_size": 512,
                "block_size": 256,
            },
            "config": {
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "initialization": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False,
                "iter_num": 1000,
                "val_loss": 2.21,
                "step": 1000,
            },
            "iter_num": 1000,
            "best_val_loss": 2.21,
        }
        torch.save(dummy_state, self.ckpt_path)
        with open(self.ckpt_path, "rb") as f:
            self.ckpt_bytes = f.read()
        self.ckpt_hash = compute_bytes_sha256(self.ckpt_bytes)

        # Create valid evidence artifacts in separate store directory
        self.store_dir = os.path.join(self.tmpdir, "production_evidence_store")
        self.artifact_paths, self.evidence_manifest = create_sample_training_artifacts(
            self.store_dir,
            run_id="run_p2766",
            step_count=1000,
            checkpoint_file_sha256=self.ckpt_hash,
        )
        self.evidence_store = EvidenceArtifactStore(self.store_dir, base_dir=PROJECT_ROOT)

        # Generate base valid attestation
        self.valid_att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_p2766",
            step_count=1000,
            evidence_manifest=self.evidence_manifest,
        )

        # Write attestation sibling
        with open(self.ckpt_path + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(self.valid_att, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 01: Missing Environment Hash Rejected
    # -------------------------------------------------------------------------
    def test_01_missing_environment_hash_rejected(self):
        """Manifest missing environment dependencies_hash is rejected."""
        ev = dict(self.evidence_manifest)
        ev["environment"] = dict(ev["environment"])
        ev["environment"].pop("dependencies_hash", None)
        ev["environment"].pop("environment_manifest_hash", None)
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("environment", reason.lower())

    # -------------------------------------------------------------------------
    # Test 02: Empty or Malformed Environment Hash Rejected
    # -------------------------------------------------------------------------
    def test_02_empty_or_malformed_environment_hash_rejected(self):
        """Manifest declaring empty or non-64-hex environment hash is rejected."""
        ev = dict(self.evidence_manifest)
        ev["environment"] = dict(ev["environment"])
        ev["environment"]["dependencies_hash"] = "invalid_short_hash"
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("environment", reason.lower())

    # -------------------------------------------------------------------------
    # Test 03: Missing Optimizer Hash Rejected
    # -------------------------------------------------------------------------
    def test_03_missing_optimizer_hash_rejected(self):
        """Manifest missing optimizer_telemetry_hash is rejected."""
        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"].pop("optimizer_telemetry_hash", None)
        ev["optimizer"].pop("optimizer_hash", None)
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("optimizer", reason.lower())

    # -------------------------------------------------------------------------
    # Test 04: Empty or Malformed Optimizer Hash Rejected
    # -------------------------------------------------------------------------
    def test_04_empty_or_malformed_optimizer_hash_rejected(self):
        """Manifest declaring empty or non-64-hex optimizer hash is rejected."""
        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = ""
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("optimizer", reason.lower())

    # -------------------------------------------------------------------------
    # Test 05: Missing Optimizer Step Records Rejected
    # -------------------------------------------------------------------------
    def test_05_missing_optimizer_step_records_rejected(self):
        """Optimizer telemetry on disk missing step_records list is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("step_records", None)
        data.pop("optimizer_steps", None)
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("step_records", reason.lower())

    # -------------------------------------------------------------------------
    # Test 06: Empty Optimizer Step Records Rejected
    # -------------------------------------------------------------------------
    def test_06_empty_optimizer_step_records_rejected(self):
        """Optimizer telemetry declaring empty step_records: [] is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["step_records"] = []
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("step_records", reason.lower())

    # -------------------------------------------------------------------------
    # Test 07: Optimizer Step Count Exceeds Total Steps Rejected
    # -------------------------------------------------------------------------
    def test_07_optimizer_step_count_exceeds_total_steps_rejected(self):
        """More records than declared total_steps is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["total_steps"] = 2
        data["step_records"] = [
            {"step": 1, "lr": 0.0006},
            {"step": 2, "lr": 0.0003},
            {"step": 3, "lr": 0.0001},
        ]
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["total_steps"] = 2
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["step_count"] = 2
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("exceeds total_steps", reason.lower())

    # -------------------------------------------------------------------------
    # Test 08: Optimizer Final Step Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_08_optimizer_final_step_mismatch_rejected(self):
        """Final observed step in optimizer telemetry not reaching total_steps is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["total_steps"] = 1000
        data["step_records"] = [
            {"step": 100, "lr": 0.0006},
            {"step": 500, "lr": 0.0003},  # Stops at 500, not 1000!
        ]
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("final observed step", reason.lower())

    # -------------------------------------------------------------------------
    # Test 09: Optimizer Non-Monotonic or Duplicate Steps Rejected
    # -------------------------------------------------------------------------
    def test_09_optimizer_non_monotonic_steps_rejected(self):
        """Optimizer records with out-of-order or duplicate step numbers are rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["step_records"] = [
            {"step": 100, "lr": 0.0006},
            {"step": 90, "lr": 0.0005},  # regressed step!
            {"step": 1000, "lr": 0.0001},
        ]
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("not strictly monotonic", reason.lower())

    # -------------------------------------------------------------------------
    # Test 10: Optimizer Non-Positive or Non-Finite Learning Rate Rejected
    # -------------------------------------------------------------------------
    def test_10_optimizer_invalid_learning_rate_rejected(self):
        """Optimizer record with lr=0.0 or negative/NaN is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["step_records"] = [
            {"step": 100, "lr": -0.0006},  # negative lr!
            {"step": 1000, "lr": 0.0001},
        ]
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("learning rate", reason.lower())

    # -------------------------------------------------------------------------
    # Test 11: Optimizer Negative or Non-Finite Gradient Norm Rejected
    # -------------------------------------------------------------------------
    def test_11_optimizer_invalid_gradient_norm_rejected(self):
        """Optimizer record with grad_norm < 0.0 is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["step_records"] = [
            {"step": 100, "lr": 0.0006, "grad_norm": -1.5},  # negative norm!
            {"step": 1000, "lr": 0.0001, "grad_norm": 0.2},
        ]
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("gradient norm", reason.lower())

    # -------------------------------------------------------------------------
    # Test 12: Optimizer Telemetry Run ID Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_12_optimizer_telemetry_run_id_mismatch_rejected(self):
        """Optimizer telemetry containing conflicting run_id is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["run_id"] = "forged_run_p2766_alt"
        content = json.dumps(data, indent=2).encode("utf-8")
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["optimizer"] = dict(ev["optimizer"])
        ev["optimizer"]["optimizer_telemetry_hash"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("run_id mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Test 13: Telemetry Log Missing [RUN_INIT] Rejected
    # -------------------------------------------------------------------------
    def test_13_telemetry_log_missing_run_init_rejected(self):
        """Telemetry log that does not start with [RUN_INIT] is rejected."""
        lines = [
            f"[STEP_TELEMETRY] run_id=run_p2766 step=500 train_loss=2.65 val_loss=2.70",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("[run_init]", reason.lower())

    # -------------------------------------------------------------------------
    # Test 14: Telemetry Log Duplicate [RUN_INIT] Rejected
    # -------------------------------------------------------------------------
    def test_14_telemetry_log_duplicate_run_init_rejected(self):
        """Telemetry log with duplicate [RUN_INIT] mid-run is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=3.85 val_loss=3.91",
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",  # Duplicate init!
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("duplicate", reason.lower())

    # -------------------------------------------------------------------------
    # Test 15: Telemetry Log Missing [RUN_COMPLETE] Rejected
    # -------------------------------------------------------------------------
    def test_15_telemetry_log_missing_run_complete_rejected(self):
        """Telemetry log missing terminal [RUN_COMPLETE] is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("[run_complete]", reason.lower())

    # -------------------------------------------------------------------------
    # Test 16: Telemetry Log [RUN_COMPLETE] Status Not SUCCESS Rejected
    # -------------------------------------------------------------------------
    def test_16_telemetry_log_run_complete_status_not_success_rejected(self):
        """Telemetry log ending with status=ABORTED or FAILED is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=FAILED",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("status=success", reason.lower())

    # -------------------------------------------------------------------------
    # Test 17: Telemetry Log Step Count Mismatch with Total Steps Rejected
    # -------------------------------------------------------------------------
    def test_17_telemetry_log_step_count_mismatch_rejected(self):
        """[RUN_COMPLETE] total_steps not matching manifest is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=99999 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("step count", reason.lower())

    # -------------------------------------------------------------------------
    # Test 18: Telemetry Log Non-Monotonic Steps Rejected
    # -------------------------------------------------------------------------
    def test_18_telemetry_log_non_monotonic_steps_rejected(self):
        """Intermediate step events with decreasing or duplicate steps are rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=500 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=200 train_loss=2.65 val_loss=2.70",  # regressed!
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("not strictly monotonic", reason.lower())

    # -------------------------------------------------------------------------
    # Test 19: Telemetry Log Negative or Non-Finite Train Loss Rejected
    # -------------------------------------------------------------------------
    def test_19_telemetry_log_invalid_train_loss_rejected(self):
        """Step event with train_loss <= 0.0 or non-finite is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=-2.50 val_loss=3.91",  # negative!
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("train_loss", reason.lower())

    # -------------------------------------------------------------------------
    # Test 20: Telemetry Log Run ID Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_20_telemetry_log_run_id_mismatch_rejected(self):
        """Step event declaring conflicting run_id is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=forged_alien_run step=100 train_loss=3.85 val_loss=3.91",  # wrong run_id!
            f"[STEP_TELEMETRY] run_id=run_p2766 step=1000 train_loss=2.15 val_loss=2.21",
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("run_id", reason.lower())

    # -------------------------------------------------------------------------
    # Test 21: Telemetry Log Final Step Mismatch with Total Steps Rejected
    # -------------------------------------------------------------------------
    def test_21_telemetry_log_final_step_mismatch_rejected(self):
        """Observed final step in [STEP_TELEMETRY] not reaching total_steps is rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2766 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=100 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2766 step=750 train_loss=2.30 val_loss=2.40",  # Stops at 750, not 1000!
            f"[RUN_COMPLETE] run_id=run_p2766 total_steps=1000 status=SUCCESS",
        ]
        content = "\n".join(lines).encode("utf-8")
        with open(self.artifact_paths["telemetry_logs"], "wb") as f:
            f.write(content)
        new_hash = hashlib.sha256(content).hexdigest().lower()

        ev = dict(self.evidence_manifest)
        ev["telemetry_logs_sha256"] = new_hash
        att = dict(self.valid_att)
        att["evidence_manifest"] = ev
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("final step", reason.lower())

    # -------------------------------------------------------------------------
    # Test 22: Candidate Checkpoint Bytes Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_22_candidate_checkpoint_bytes_mismatch_rejected(self):
        """Tampering with candidate checkpoint file bytes triggers hash mismatch."""
        tampered_bytes = self.ckpt_bytes + b"\x00\xde\xad\xbe\xef"
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            tampered_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_CHECKPOINT_HASH_MISMATCH")

    # -------------------------------------------------------------------------
    # Test 23: Complete Authentic Evidence Package Verified
    # -------------------------------------------------------------------------
    def test_23_complete_authentic_evidence_package_verified(self):
        """Valid external evidence store with authentic artifacts passes production verification."""
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
            require_production_provider=True,
            require_artifacts=True,
        )
        self.assertTrue(valid, f"Verification failed unexpectedly: {reason}")
        self.assertEqual(tier, "GENUINE_TRAINED_SOVEREIGN")

    # -------------------------------------------------------------------------
    # Test 24: Test Provider Cannot Produce Genuine Production Classification
    # -------------------------------------------------------------------------
    def test_24_test_provider_cannot_produce_genuine_production_classification(self):
        """Test isolated provider cannot verify genuine production checkpoint."""
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=self.evidence_store,
            require_production_provider=True,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_UNTRUSTED_SIGNER")

    # -------------------------------------------------------------------------
    # Test 25: allow_fixture Cannot Produce Genuine Production Classification
    # -------------------------------------------------------------------------
    def test_25_allow_fixture_cannot_produce_genuine_production_classification(self):
        """allow_test_fixture_classification produces TEST_FIXTURE_VALID_STRUCTURE, never genuine."""
        tier = classify_checkpoint(
            self.ckpt_path,
            trust_provider=self.test_provider,
            evidence_store=self.evidence_store,
            allow_test_fixture_classification=True,
        )
        self.assertEqual(tier, "TEST_FIXTURE_VALID_STRUCTURE")

        # Confirm is_verified_genuine_checkpoint strictly returns False
        self.assertFalse(
            is_verified_genuine_checkpoint(
                self.ckpt_path,
                trust_provider=self.test_provider,
                evidence_store=self.evidence_store,
            )
        )

    # -------------------------------------------------------------------------
    # Test 26: Direct ModelService.generate() Fails Closed
    # -------------------------------------------------------------------------
    def test_26_direct_model_service_generate_fails_closed(self):
        """Direct ModelService.generate() raises ModelNotTrainedException when model unverified."""
        service = ModelService(base_dir=PROJECT_ROOT)
        with self.assertRaises(ModelNotTrainedException):
            service.generate("Hello Moon AI", allow_fixture=False)

    # -------------------------------------------------------------------------
    # Test 27: DEBUG_INFERENCE Cannot Bypass Production Gate
    # -------------------------------------------------------------------------
    def test_27_debug_inference_cannot_bypass_production_gate(self):
        """DEBUG_INFERENCE=1 cannot bypass production evidence gate."""
        os.environ["DEBUG_INFERENCE"] = "1"
        try:
            service = ModelService(base_dir=PROJECT_ROOT)
            with self.assertRaises(ModelNotTrainedException):
                service.generate("Test prompt with debug flag", allow_fixture=False)
        finally:
            os.environ.pop("DEBUG_INFERENCE", None)

    # -------------------------------------------------------------------------
    # Test 28: AUTHORIZED_BY_PREM Strictly Preserved as False
    # -------------------------------------------------------------------------
    def test_28_authorized_by_prem_strictly_preserved(self):
        """Policy invariant AUTHORIZED_BY_PREM remains False."""
        self.assertFalse(policy.AUTHORIZED_BY_PREM)
        self.assertEqual(policy.ACTIVE_OWNERSHIP_POLICY, "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")


if __name__ == "__main__":
    unittest.main()
