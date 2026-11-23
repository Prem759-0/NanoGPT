"""
tests/test_phase_2_7_6_4_mandatory_evidence_and_trust_isolation.py
Comprehensive Adversarial & Integrity Test Suite for Phase 2.7.6.4:
Mandatory Evidence Enforcement, Artifact Verification & Production Trust Boundary Closure.

24 Adversarial Scenarios Covered:
1. Missing evidence manifest rejected (ATTESTATION_EVIDENCE_INVALID).
2. Empty evidence manifest rejected (ATTESTATION_EVIDENCE_INVALID).
3. Unknown evidence schema rejected (e.g. schema "1.0").
4. Missing dataset artifact rejected when require_artifacts=True.
5. Dataset artifact hash mismatch rejected when require_artifacts=True.
6. Missing architecture artifact rejected when require_artifacts=True.
7. Architecture artifact hash mismatch rejected when require_artifacts=True.
8. Missing training code artifact rejected when require_artifacts=True.
9. Training code artifact hash mismatch rejected when require_artifacts=True.
10. Missing telemetry logs artifact rejected when require_artifacts=True.
11. Telemetry logs artifact hash mismatch rejected when require_artifacts=True.
12. Fabricated optimizer step count rejected when run records mismatch.
13. Run ID mismatch between attestation and evidence rejected.
14. Attestation/evidence timestamp mismatch rejected.
15. Checkpoint byte modification rejected (ATTESTATION_CHECKPOINT_HASH_MISMATCH).
16. Test provider rejected by production verification (require_production_provider=True).
17. Test provider cannot authorize direct ModelService.generate().
18. allow_untrusted_for_test cannot produce production genuine classification.
19. Missing external trust root fails closed (UnconfiguredFailClosed, is_production_provider=False).
20. Repository trust-root injection rejected (InRepoSecurityViolationFailClosed).
21. Revoked signer rejected (ATTESTATION_REVOKED_AUTHORITY).
22. Expired signer rejected (ATTESTATION_EXPIRED_AUTHORITY).
23. DEBUG_INFERENCE=1 cannot bypass production gate.
24. Zero training side effects (AUTHORIZED_BY_PREM is False, zero checkpoints written).

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""

import os
import sys
import json
import shutil
import tempfile
import hashlib
import unittest
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import core.policy as policy
from core.security.trust_provider import (
    BaseTrustRootProvider,
    ExternalTrustRootProvider,
    TestIsolatedTrustProvider,
    get_default_trust_provider,
)
from core.security.training_evidence import (
    EvidenceArtifactStore,
    TrainingEvidenceValidator,
    create_sample_training_artifacts,
    create_sample_training_evidence,
)
from core.security.training_attestation import (
    generate_training_authority_keypair,
    sign_training_attestation,
    verify_training_attestation,
    create_test_training_attestation,
    compute_file_sha256,
    compute_bytes_sha256,
)
from webapp.model_service import (
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    ModelService,
    ModelNotTrainedException,
)


class TestPhase2764MandatoryEvidenceAndTrustIsolation(unittest.TestCase):
    """24 Adversarial Vectors for Mandatory Evidence & Production Trust Boundary Closure."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="moon_phase_2_7_6_4_test_")
        self.priv_hex, self.pub_hex = generate_training_authority_keypair()
        self.verifier_id = "test_auth_phase_2_7_6_4"
        self.authority_entry = {
            self.verifier_id: {
                "name": "Phase 2.7.6.4 Test Authority",
                "public_key_hex": self.pub_hex,
                "status": "active",
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        self.test_provider = TestIsolatedTrustProvider(self.authority_entry)

        # Create a valid synthetic checkpoint fixture file on disk
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

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Scenario 1: Missing evidence manifest rejected
    # -------------------------------------------------------------------------
    def test_01_missing_evidence_manifest_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            include_evidence=False,
        )
        self.assertNotIn("evidence_manifest", att)
        valid, tier, reason = verify_training_attestation(
            att, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mandatory", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 2: Empty evidence manifest rejected
    # -------------------------------------------------------------------------
    def test_02_empty_evidence_manifest_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            include_evidence=False,
            evidence_manifest={},
        )
        valid, tier, reason = verify_training_attestation(
            att, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # -------------------------------------------------------------------------
    # Scenario 3: Unknown evidence schema rejected
    # -------------------------------------------------------------------------
    def test_03_unknown_evidence_schema_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            include_evidence=True,
        )
        att["evidence_manifest"]["schema_version"] = "1.0"
        # Re-sign with the modified payload
        signed = sign_training_attestation(att, self.priv_hex)
        valid, tier, reason = verify_training_attestation(
            signed, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("schema_version", reason)

    # -------------------------------------------------------------------------
    # Scenario 4: Missing dataset artifact rejected when require_artifacts=True
    # -------------------------------------------------------------------------
    def test_04_missing_dataset_artifact_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_04")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_04", step_count=1000)
        os.remove(paths["dataset_manifest"])

        store = EvidenceArtifactStore(art_dir)
        self.assertTrue(store.is_configured())

        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_04",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        # Update checkpoint_sha256 in evidence to match
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("dataset_manifest", reason)

    # -------------------------------------------------------------------------
    # Scenario 5: Dataset artifact hash mismatch rejected
    # -------------------------------------------------------------------------
    def test_05_dataset_artifact_hash_mismatch_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_05")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_05", step_count=1000)
        with open(paths["dataset_manifest"], "wb") as f:
            f.write(b'{"corrupted": true}')

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_05",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 6: Missing architecture artifact rejected
    # -------------------------------------------------------------------------
    def test_06_missing_architecture_artifact_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_06")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_06", step_count=1000)
        os.remove(paths["architecture_spec"])

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_06",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("model_architecture", reason)

    # -------------------------------------------------------------------------
    # Scenario 7: Architecture artifact hash mismatch rejected
    # -------------------------------------------------------------------------
    def test_07_architecture_artifact_hash_mismatch_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_07")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_07", step_count=1000)
        with open(paths["architecture_spec"], "wb") as f:
            f.write(b'{"tampered_arch": true}')

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_07",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 8: Missing training code artifact rejected
    # -------------------------------------------------------------------------
    def test_08_missing_training_code_artifact_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_08")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_08", step_count=1000)
        os.remove(paths["training_code"])

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_08",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("training_code", reason)

    # -------------------------------------------------------------------------
    # Scenario 9: Training code artifact hash mismatch rejected
    # -------------------------------------------------------------------------
    def test_09_training_code_artifact_hash_mismatch_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_09")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_09", step_count=1000)
        with open(paths["training_code"], "wb") as f:
            f.write(b'{"forged_commit": "evil"}')

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_09",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 10: Missing telemetry logs artifact rejected
    # -------------------------------------------------------------------------
    def test_10_missing_telemetry_logs_artifact_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_10")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_10", step_count=1000)
        os.remove(paths["telemetry_logs"])

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_10",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("telemetry_logs", reason)

    # -------------------------------------------------------------------------
    # Scenario 11: Telemetry logs artifact hash mismatch rejected
    # -------------------------------------------------------------------------
    def test_11_telemetry_logs_artifact_hash_mismatch_rejected(self):
        art_dir = os.path.join(self.tmpdir, "artifacts_11")
        paths, ev = create_sample_training_artifacts(art_dir, run_id="run_11", step_count=1000)
        with open(paths["telemetry_logs"], "wb") as f:
            f.write(b"[RUN_LOGS_TAMPERED]\n")

        store = EvidenceArtifactStore(art_dir)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_11",
            step_count=1000,
            start_time=ev["timestamps"]["start_time"],
            completion_time=ev["timestamps"]["completion_time"],
            model_architecture_hash=ev["model_architecture"]["model_architecture_hash"],
            dataset_manifest_hash=ev["dataset_manifest"]["dataset_manifest_hash"],
            training_code_hash=ev["training_code"]["training_code_hash"],
            evidence_manifest=ev,
        )
        att["evidence_manifest"]["checkpoint_sha256"] = self.ckpt_hash
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            evidence_store=store,
            require_artifacts=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 12: Fabricated optimizer step count rejected
    # -------------------------------------------------------------------------
    def test_12_fabricated_optimizer_step_count_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            step_count=1000,
        )
        # Fabricate optimizer total_steps = 99999 while attestation claimed 1000
        att["evidence_manifest"]["optimizer"]["total_steps"] = 99999
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("step count mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 13: Run ID mismatch between attestation and evidence rejected
    # -------------------------------------------------------------------------
    def test_13_run_id_mismatch_between_attestation_and_evidence_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
            run_id="run_authentic",
        )
        att["evidence_manifest"]["run_id"] = "run_forged"
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("run id mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 14: Attestation/evidence timestamp mismatch rejected
    # -------------------------------------------------------------------------
    def test_14_attestation_evidence_timestamp_mismatch_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        # Tamper with evidence start time
        att["evidence_manifest"]["timestamps"]["start_time"] = "2020-01-01T00:00:00+00:00"
        signed = sign_training_attestation(att, self.priv_hex)

        valid, tier, reason = verify_training_attestation(
            signed, self.ckpt_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")
        self.assertIn("timestamp mismatch", reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 15: Checkpoint byte modification rejected
    # -------------------------------------------------------------------------
    def test_15_checkpoint_byte_modification_rejected(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        tampered_bytes = self.ckpt_bytes + b"\x00\xff"
        valid, tier, reason = verify_training_attestation(
            att, tampered_bytes, trust_provider=self.test_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_CHECKPOINT_HASH_MISMATCH")

    # -------------------------------------------------------------------------
    # Scenario 16: Test provider rejected by production verification
    # -------------------------------------------------------------------------
    def test_16_test_provider_rejected_by_production_verification(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        # Direct verification with require_production_provider=True
        valid, tier, reason = verify_training_attestation(
            att,
            self.ckpt_bytes,
            trust_provider=self.test_provider,
            require_production_provider=True,
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_UNTRUSTED_SIGNER")
        self.assertIn("production", reason.lower())

        # is_verified_genuine_checkpoint defaults to require_production=True
        with open(self.ckpt_path + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)
        self.assertFalse(
            is_verified_genuine_checkpoint(self.ckpt_path, trust_provider=self.test_provider)
        )

    # -------------------------------------------------------------------------
    # Scenario 17: Test provider cannot authorize direct ModelService.generate()
    # -------------------------------------------------------------------------
    def test_17_test_provider_cannot_authorize_direct_model_service_generate(self):
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        with open(self.ckpt_path + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)

        svc = ModelService(base_dir=self.tmpdir, preferred_checkpoint=self.ckpt_path)
        # Attempting generate without allow_fixture=True must fail closed
        with self.assertRaises(ModelNotTrainedException):
            svc.generate("Hello Moon AI", allow_fixture=False)

    # -------------------------------------------------------------------------
    # Scenario 18: allow_untrusted_for_test cannot produce production genuine classification
    # -------------------------------------------------------------------------
    def test_18_allow_untrusted_for_test_cannot_produce_production_genuine_classification(self):
        tier = classify_checkpoint(
            self.ckpt_path,
            allow_test_fixture_classification=True,
            allow_unattested_for_test=True,
        )
        self.assertEqual(tier, "TEST_FIXTURE_VALID_STRUCTURE")
        self.assertNotEqual(tier, "GENUINE_TRAINED_SOVEREIGN")
        self.assertNotEqual(tier, "PROMOTED_PRODUCTION")

    # -------------------------------------------------------------------------
    # Scenario 19: Missing external trust root fails closed
    # -------------------------------------------------------------------------
    def test_19_missing_external_trust_root_fails_closed(self):
        # Empty provider with clean environment
        provider = ExternalTrustRootProvider(external_path="/nonexistent/trust.json")
        auths = provider.get_authorities()
        self.assertEqual(auths, {})
        self.assertFalse(provider.is_configured())
        self.assertFalse(provider.is_production_provider())
        status = provider.get_configuration_status()
        self.assertEqual(status["authority_count"], 0)
        self.assertFalse(status["is_configured"])

    # -------------------------------------------------------------------------
    # Scenario 20: Repository trust-root injection rejected
    # -------------------------------------------------------------------------
    def test_20_repository_trust_root_injection_rejected(self):
        in_repo_keystore = os.path.join(
            PROJECT_ROOT, "core", "security", "trusted_training_authorities.json"
        )
        provider = ExternalTrustRootProvider(
            external_path=in_repo_keystore, base_dir=PROJECT_ROOT
        )
        auths = provider.get_authorities()
        self.assertEqual(auths, {})
        self.assertEqual(provider.get_provider_type(), "InRepoSecurityViolationFailClosed")
        self.assertFalse(provider.is_configured())
        self.assertFalse(provider.is_production_provider())

    # -------------------------------------------------------------------------
    # Scenario 21: Revoked signer rejected
    # -------------------------------------------------------------------------
    def test_21_revoked_signer_rejected(self):
        revoked_authority_entry = {
            self.verifier_id: {
                "name": "Revoked Authority",
                "public_key_hex": self.pub_hex,
                "status": "revoked",
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        revoked_provider = TestIsolatedTrustProvider(revoked_authority_entry)
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        valid, tier, reason = verify_training_attestation(
            att, self.ckpt_bytes, trust_provider=revoked_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_REVOKED_AUTHORITY")

    # -------------------------------------------------------------------------
    # Scenario 22: Expired signer rejected
    # -------------------------------------------------------------------------
    def test_22_expired_signer_rejected(self):
        past_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        expired_authority_entry = {
            self.verifier_id: {
                "name": "Expired Authority",
                "public_key_hex": self.pub_hex,
                "status": "active",
                "valid_until": past_time,
                "registered_at": (datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
            }
        }
        # External trust provider validates expiration timestamps
        ext_store_path = os.path.join(self.tmpdir, "expired_ext_store.json")
        with open(ext_store_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.1",
                "policy": "ASYMMETRIC_ED25519_ONLY",
                "trusted_authorities": expired_authority_entry,
            }, f)
        ext_provider = ExternalTrustRootProvider(
            external_path=ext_store_path, base_dir=PROJECT_ROOT
        )
        att, _ = create_test_training_attestation(
            self.ckpt_bytes,
            private_key_hex=self.priv_hex,
            verifier_id=self.verifier_id,
        )
        valid, tier, reason = verify_training_attestation(
            att, self.ckpt_bytes, trust_provider=ext_provider
        )
        self.assertFalse(valid)
        self.assertEqual(tier, "ATTESTATION_EXPIRED_AUTHORITY")

    # -------------------------------------------------------------------------
    # Scenario 23: DEBUG_INFERENCE=1 cannot bypass production gate
    # -------------------------------------------------------------------------
    def test_23_debug_inference_cannot_bypass_production_gate(self):
        old_debug = os.environ.get("DEBUG_INFERENCE")
        try:
            os.environ["DEBUG_INFERENCE"] = "1"
            svc = ModelService(base_dir=self.tmpdir)
            self.assertTrue(svc.debug_mode)
            # Cannot generate or return READY when genuine model does not exist
            self.assertEqual(svc.get_status(), "MODEL_NOT_READY")
            with self.assertRaises(ModelNotTrainedException):
                svc.generate("Adversarial prompt bypassing lock", allow_fixture=False)
        finally:
            if old_debug is None:
                os.environ.pop("DEBUG_INFERENCE", None)
            else:
                os.environ["DEBUG_INFERENCE"] = old_debug

    # -------------------------------------------------------------------------
    # Scenario 24: Zero training side effects and policy lock preservation
    # -------------------------------------------------------------------------
    def test_24_no_training_or_checkpoint_writing_side_effects(self):
        self.assertFalse(
            policy.AUTHORIZED_BY_PREM,
            "CRITICAL: AUTHORIZED_BY_PREM must remain strictly False.",
        )
        self.assertEqual(
            policy.ACTIVE_OWNERSHIP_POLICY,
            "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "Active ownership policy must remain STRICT_SOVEREIGN_RANDOM_INIT_ONLY.",
        )
        prod_ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints")
        if os.path.isdir(prod_ckpt_dir):
            for fname in os.listdir(prod_ckpt_dir):
                self.assertFalse(
                    fname.endswith(".tmp"),
                    f"Found temporary checkpoint leak in models/checkpoints: {fname}",
                )


if __name__ == "__main__":
    unittest.main()
