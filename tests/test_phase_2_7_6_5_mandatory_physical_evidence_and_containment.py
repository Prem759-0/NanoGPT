"""
tests/test_phase_2_7_6_5_mandatory_physical_evidence_and_containment.py
Adversarial Verification Suite for Phase 2.7.6.5:
- Mandatory Physical Evidence at Production Boundaries
- Artifact Store Path Containment & Traversal Prevention (symlinks, .., absolute, non-regular)
- External Production Store Strict Invariant (In-repo store fails closed)
- Mandatory 6-Artifact Verification (architecture, dataset, code, env, optimizer, logs + checkpoint bytes)
- Monotonic Telemetry Logs & Optimizer Step Records Validation
- Fail-Closed Gate Enforcement & Invariant Protection (AUTHORIZED_BY_PREM = False)

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


class TestPhase2765MandatoryPhysicalEvidenceAndContainment(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.priv_hex, self.pub_hex = generate_training_authority_keypair()
        self.verifier_id = "phase_2_7_6_5_auditor"
        self.authority_entry = {
            self.verifier_id: {
                "name": "Phase 2.7.6.5 Authority",
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
            run_id="run_p2765",
            step_count=1000,
            checkpoint_file_sha256=self.ckpt_hash,
        )
        self.evidence_store = EvidenceArtifactStore(self.store_dir, base_dir=PROJECT_ROOT)

        # Create matching signed attestation
        self.valid_att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_p2765",
            step_count=1000,
            evidence_manifest=self.evidence_manifest,
        )

        # Write attestation sibling
        with open(self.ckpt_path + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(self.valid_att, f)

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Defaults Require Physical Evidence Artifacts in Production
    # -------------------------------------------------------------------------
    def test_01_defaults_require_artifacts_in_production(self):
        """classify_checkpoint and verify_training_attestation default to require_artifacts=True."""
        # When called without evidence_store, defaults must fail closed
        tier = classify_checkpoint(
            self.ckpt_path,
            trust_provider=self.production_provider,
        )
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertFalse(is_verified_genuine_checkpoint(self.ckpt_path, trust_provider=self.production_provider))

    # -------------------------------------------------------------------------
    # Test 2: Unconfigured Evidence Store Fails Closed in Production
    # -------------------------------------------------------------------------
    def test_02_unconfigured_evidence_store_fails_closed(self):
        """Unconfigured EvidenceArtifactStore returns ATTESTATION_EVIDENCE_INVALID."""
        empty_store = EvidenceArtifactStore(store_dir=os.path.join(self.tmpdir, "nonexistent_dir"))
        self.assertFalse(empty_store.is_configured())
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=empty_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("external evidence store", reason.lower())

    # -------------------------------------------------------------------------
    # Test 3: In-Repository Evidence Store Fails Closed
    # -------------------------------------------------------------------------
    def test_03_in_repo_evidence_store_fails_closed(self):
        """EvidenceArtifactStore inside PROJECT_ROOT is rejected as in-repo security violation."""
        in_repo_dir = os.path.join(PROJECT_ROOT, "core", "security")
        in_repo_store = EvidenceArtifactStore(store_dir=in_repo_dir, base_dir=PROJECT_ROOT)
        self.assertFalse(in_repo_store.is_production_store())
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=in_repo_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("strictly outside the repository", reason.lower())

    # -------------------------------------------------------------------------
    # Test 4: Path Traversal (..) in Artifact Path is Rejected
    # -------------------------------------------------------------------------
    def test_04_path_traversal_double_dot_rejected(self):
        """Evidence declaring '../escaped.json' is strictly rejected and returns None."""
        escaped_file = os.path.join(self.tmpdir, "escaped.json")
        with open(escaped_file, "w") as f:
            f.write("{}")

        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("../escaped.json"))
        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("sub/../../escaped.json"))

        ev_tampered = dict(self.evidence_manifest)
        ev_tampered["model_architecture"] = dict(ev_tampered["model_architecture"])
        ev_tampered["model_architecture"]["artifact_file"] = "../escaped.json"

        att_tampered = dict(self.valid_att)
        att_tampered["evidence_manifest"] = ev_tampered
        signed = sign_training_attestation(att_tampered, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # -------------------------------------------------------------------------
    # Test 5: Absolute Paths (/ or C:\) in Artifact Path are Rejected
    # -------------------------------------------------------------------------
    def test_05_absolute_paths_rejected(self):
        """Evidence declaring absolute paths (/etc/passwd or C:\\windows) is rejected."""
        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("/etc/passwd"))
        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("C:\\target\\file.json"))
        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("\\\\server\\share\\file.json"))

        ev_tampered = dict(self.evidence_manifest)
        ev_tampered["model_architecture"] = dict(ev_tampered["model_architecture"])
        ev_tampered["model_architecture"]["artifact_file"] = "/absolute/arch.json"

        att_tampered = dict(self.valid_att)
        att_tampered["evidence_manifest"] = ev_tampered
        signed = sign_training_attestation(att_tampered, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # -------------------------------------------------------------------------
    # Test 6: Symlink Escaping Root is Rejected
    # -------------------------------------------------------------------------
    def test_06_symlink_escaping_root_rejected(self):
        """Symlinks pointing outside store_dir fail containment checks."""
        outside_file = os.path.join(self.tmpdir, "outside_target.json")
        with open(outside_file, "w") as f:
            f.write('{"outside": true}')

        symlink_path = os.path.join(self.store_dir, "symlink_escape.json")
        try:
            os.symlink(outside_file, symlink_path)
            self.assertIsNone(self.evidence_store._sanitize_and_contain_path("symlink_escape.json"))
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported in this test environment.")

    # -------------------------------------------------------------------------
    # Test 7: Non-Regular File (Directory) in Artifact Path is Rejected
    # -------------------------------------------------------------------------
    def test_07_non_regular_file_rejected(self):
        """Targeting a directory instead of a regular file is rejected."""
        sub_dir = os.path.join(self.store_dir, "subdir_artifact")
        os.makedirs(sub_dir, exist_ok=True)
        self.assertIsNone(self.evidence_store._sanitize_and_contain_path("subdir_artifact"))

    # -------------------------------------------------------------------------
    # Test 8: Missing Architecture Artifact Rejected
    # -------------------------------------------------------------------------
    def test_08_missing_architecture_artifact_rejected(self):
        """Missing architecture_spec.json on disk fails verification."""
        os.remove(self.artifact_paths["architecture_spec"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("model_architecture", reason)

    # -------------------------------------------------------------------------
    # Test 9: Missing Dataset Artifact Rejected
    # -------------------------------------------------------------------------
    def test_09_missing_dataset_artifact_rejected(self):
        """Missing dataset_manifest.json on disk fails verification."""
        os.remove(self.artifact_paths["dataset_manifest"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("dataset_manifest", reason)

    # -------------------------------------------------------------------------
    # Test 10: Missing Training Code Artifact Rejected
    # -------------------------------------------------------------------------
    def test_10_missing_training_code_artifact_rejected(self):
        """Missing code_manifest.json on disk fails verification."""
        os.remove(self.artifact_paths["training_code"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("training_code", reason)

    # -------------------------------------------------------------------------
    # Test 11: Missing Environment Artifact Rejected
    # -------------------------------------------------------------------------
    def test_11_missing_environment_artifact_rejected(self):
        """Missing environment_manifest.json on disk fails verification."""
        os.remove(self.artifact_paths["environment"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("environment", reason)

    # -------------------------------------------------------------------------
    # Test 12: Missing Optimizer Telemetry Artifact Rejected
    # -------------------------------------------------------------------------
    def test_12_missing_optimizer_telemetry_artifact_rejected(self):
        """Missing optimizer_telemetry.json on disk fails verification."""
        os.remove(self.artifact_paths["optimizer"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("optimizer", reason)

    # -------------------------------------------------------------------------
    # Test 13: Missing Telemetry Logs Artifact Rejected
    # -------------------------------------------------------------------------
    def test_13_missing_telemetry_logs_artifact_rejected(self):
        """Missing telemetry_logs.raw on disk fails verification."""
        os.remove(self.artifact_paths["telemetry_logs"])
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("telemetry_logs", reason)

    # -------------------------------------------------------------------------
    # Test 14: Environment Manifest Hash Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_14_environment_manifest_hash_mismatch_rejected(self):
        """Corrupted environment_manifest.json on disk triggers SHA-256 mismatch."""
        with open(self.artifact_paths["environment"], "wb") as f:
            f.write(b'{"corrupted_env": true}')
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Test 15: Optimizer Telemetry Hash Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_15_optimizer_telemetry_hash_mismatch_rejected(self):
        """Corrupted optimizer_telemetry.json on disk triggers SHA-256 mismatch."""
        with open(self.artifact_paths["optimizer"], "wb") as f:
            f.write(b'{"corrupted_optim": true}')
        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            self.ckpt_bytes,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Test 16: Optimizer Telemetry Run ID Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_16_optimizer_telemetry_run_id_mismatch_rejected(self):
        """Optimizer telemetry containing a conflicting run_id is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["run_id"] = "forged_run_999"
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
    # Test 17: Optimizer Telemetry Total Steps Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_17_optimizer_telemetry_total_steps_mismatch_rejected(self):
        """Optimizer telemetry total_steps not matching manifest is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["total_steps"] = 99999
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
        self.assertIn("total_steps mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Test 18: Non-Monotonic or Duplicate Optimizer Steps Rejected
    # -------------------------------------------------------------------------
    def test_18_non_monotonic_optimizer_steps_rejected(self):
        """Optimizer telemetry with duplicate or decreasing steps is rejected."""
        with open(self.artifact_paths["optimizer"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["step_records"] = [
            {"step": 100, "lr": 0.0006},
            {"step": 100, "lr": 0.0006},  # duplicate!
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
    # Test 19: Telemetry Log Missing [RUN_INIT] Rejected
    # -------------------------------------------------------------------------
    def test_19_telemetry_log_missing_run_init_rejected(self):
        """Telemetry logs lacking [RUN_INIT] header are rejected."""
        lines = [
            f"[STEP_TELEMETRY] run_id=run_p2765 step=100 train_loss=3.85 val_loss=3.91",
            f"[RUN_COMPLETE] run_id=run_p2765 total_steps=1000 status=SUCCESS",
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
    # Test 20: Telemetry Log Duplicate or Non-Monotonic Steps Rejected
    # -------------------------------------------------------------------------
    def test_20_telemetry_log_non_monotonic_steps_rejected(self):
        """Telemetry logs with decreasing or duplicate steps are rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2765 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2765 step=500 train_loss=3.85 val_loss=3.91",
            f"[STEP_TELEMETRY] run_id=run_p2765 step=200 train_loss=3.00 val_loss=3.10",  # regressed step!
            f"[RUN_COMPLETE] run_id=run_p2765 total_steps=1000 status=SUCCESS",
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
    # Test 21: Telemetry Log Missing [RUN_COMPLETE] Rejected
    # -------------------------------------------------------------------------
    def test_21_telemetry_log_missing_run_complete_rejected(self):
        """Telemetry logs lacking [RUN_COMPLETE] status=SUCCESS are rejected."""
        lines = [
            f"[RUN_INIT] run_id=run_p2765 policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
            f"[STEP_TELEMETRY] run_id=run_p2765 step=1000 train_loss=2.15 val_loss=2.21",
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
    # Test 22: Candidate Checkpoint Bytes Mismatch Rejected
    # -------------------------------------------------------------------------
    def test_22_candidate_checkpoint_bytes_mismatch_rejected(self):
        """Candidate checkpoint on disk with tampered bytes fails verification."""
        tampered_ckpt = os.path.join(self.tmpdir, "tampered_candidate.pt")
        with open(tampered_ckpt, "wb") as f:
            f.write(self.ckpt_bytes + b"\xde\xad\xbe\xef")

        valid, tier, reason = verify_training_attestation(
            self.valid_att,
            tampered_ckpt,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_CHECKPOINT_HASH_MISMATCH")

    # -------------------------------------------------------------------------
    # Test 23: Complete Authentic Evidence Package Verified as GENUINE_TRAINED_SOVEREIGN
    # -------------------------------------------------------------------------
    def test_23_authentic_evidence_package_verified_as_genuine(self):
        """All 6 physical artifacts + checkpoint verified yields GENUINE_TRAINED_SOVEREIGN."""
        tier = classify_checkpoint(
            self.ckpt_path,
            trust_provider=self.production_provider,
            evidence_store=self.evidence_store,
        )
        self.assertEqual(tier, "GENUINE_TRAINED_SOVEREIGN")
        self.assertTrue(
            is_verified_genuine_checkpoint(
                self.ckpt_path,
                trust_provider=self.production_provider,
                evidence_store=self.evidence_store,
            )
        )

    # -------------------------------------------------------------------------
    # Test 24: Test Provider Cannot Produce Genuine Production Classification
    # -------------------------------------------------------------------------
    def test_24_test_provider_cannot_produce_genuine_production_classification(self):
        """Even with all valid physical artifacts, TestIsolatedTrustProvider fails production check."""
        tier = classify_checkpoint(
            self.ckpt_path,
            trust_provider=self.test_provider,
            evidence_store=self.evidence_store,
        )
        self.assertNotEqual(tier, "GENUINE_TRAINED_SOVEREIGN")
        self.assertNotEqual(tier, "PROMOTED_PRODUCTION")
        self.assertFalse(
            is_verified_genuine_checkpoint(
                self.ckpt_path,
                trust_provider=self.test_provider,
                evidence_store=self.evidence_store,
            )
        )

    # -------------------------------------------------------------------------
    # Test 25: Direct ModelService.generate Fails Closed Without Production Verification
    # -------------------------------------------------------------------------
    def test_25_model_service_generate_fails_closed(self):
        """Unverified checkpoints or unconfigured evidence store raises ModelNotTrainedException."""
        ms = ModelService(base_dir=self.tmpdir, preferred_checkpoint=self.ckpt_path)
        with self.assertRaises(ModelNotTrainedException):
            ms.generate("Hello Moon AI", allow_fixture=False)

    # -------------------------------------------------------------------------
    # Test 26: DEBUG_INFERENCE=1 Cannot Bypass Evidence Gate
    # -------------------------------------------------------------------------
    def test_26_debug_inference_cannot_bypass_evidence_gate(self):
        """DEBUG_INFERENCE=1 cannot bypass physical evidence gate."""
        os.environ["DEBUG_INFERENCE"] = "1"
        try:
            ms = ModelService(base_dir=self.tmpdir, preferred_checkpoint=self.ckpt_path)
            ms.active_model_name = "scratch"
            ms.is_test_fixture = True
            with self.assertRaises(ModelNotTrainedException):
                ms.generate("Bypass attempt", allow_fixture=False)
        finally:
            os.environ.pop("DEBUG_INFERENCE", None)

    # -------------------------------------------------------------------------
    # Test 27: AUTHORIZED_BY_PREM Strict Invariant
    # -------------------------------------------------------------------------
    def test_27_authorized_by_prem_strictly_locked_false(self):
        """AUTHORIZED_BY_PREM must be strictly False and policy unchanged."""
        self.assertFalse(policy.AUTHORIZED_BY_PREM)
        self.assertEqual(policy.ACTIVE_OWNERSHIP_POLICY, "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")


if __name__ == "__main__":
    unittest.main()
