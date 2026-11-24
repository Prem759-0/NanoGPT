"""
tests/test_phase_2_7_6_3_external_trust_and_evidence.py
Phase 2.7.6.3 Adversarial Test Suite: External Trust Root, Evidence Validation & Production Isolation.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Specification: MRDC-JSON-2.1
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple

import torch
from cryptography.hazmat.primitives.asymmetric import ed25519

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.policy as policy
from core.security.trust_provider import (
    BaseTrustRootProvider,
    ExternalTrustRootProvider,
    TestIsolatedTrustProvider,
    get_default_trust_provider,
)
from core.security.training_evidence import (
    TrainingEvidenceValidator,
    create_sample_training_evidence,
    is_valid_hex64,
    parse_iso_utc_timestamp,
)
from core.security.training_attestation import (
    compute_file_sha256,
    generate_training_authority_keypair,
    create_test_training_attestation,
    verify_training_attestation,
)
from webapp.model_service import (
    ModelService,
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    validate_checkpoint_structure,
    ModelNotTrainedException,
)


class TestPhase2763ExternalTrustAndEvidence(unittest.TestCase):
    """
    27-Vector Adversarial Verification Suite for Phase 2.7.6.3:
    - Closure of test-only provenance bypass
    - Strict separation of repository keystore and external trust roots
    - Deep training evidence validation (MRDC-JSON-2.1)
    - Production isolation and fail-closed defense across all serving entry points
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_minimal_checkpoint(self, name: str, **kwargs) -> str:
        """Helper creating minimal structurally valid strict sovereign checkpoint."""
        ckpt_path = os.path.join(self.tmp_path, name)
        ckpt_data = {
            "model": {"transformer.wte.weight": torch.randn(512, 512)},
            "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512},
            "config": {
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "initialization": "random_gaussian_scratch",
                "student_init_method": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False,
                "step": 1000,
                "val_loss": 2.14,
            },
            "iter_num": 1000,
            "best_val_loss": 2.14,
            "status": "PILOT_TRAINED",
            "is_test_fixture": False,
            "promotable": False,
        }
        for k, v in kwargs.items():
            if k == "config" and isinstance(v, dict):
                ckpt_data["config"].update(v)
            else:
                ckpt_data[k] = v

        torch.save(ckpt_data, ckpt_path)
        return ckpt_path

    # Vector 1: Test-only bypass cannot return genuine
    def test_01_test_fixture_classification_cannot_return_genuine(self):
        """allow_test_fixture_classification returns TEST_FIXTURE_VALID_STRUCTURE, NEVER GENUINE_TRAINED_SOVEREIGN."""
        ckpt = self._create_minimal_checkpoint("unattested_strict.pt")
        tier = classify_checkpoint(ckpt, allow_test_fixture_classification=True)
        self.assertEqual(tier, "TEST_FIXTURE_VALID_STRUCTURE")
        self.assertNotEqual(tier, "GENUINE_TRAINED_SOVEREIGN")
        self.assertNotEqual(tier, "PROMOTED_PRODUCTION")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Vector 2: Test-only bypass cannot enable production inference
    def test_02_test_fixture_cannot_enable_production_inference(self):
        """Checkpoints classified as TEST_FIXTURE_VALID_STRUCTURE fail production generate()."""
        ckpt = self._create_minimal_checkpoint("fixture_for_infer.pt")
        ms = ModelService(base_dir=self.tmp_path, preferred_checkpoint=ckpt)
        ms.active_model_name = "scratch"
        ms.is_test_fixture = True
        ms.model = "mock_model_obj"
        with self.assertRaises(ModelNotTrainedException):
            ms.generate("Hello world", allow_fixture=False)

    # Vector 3: Repository-added public key cannot become a production trust root
    def test_03_in_repo_trust_store_cannot_be_production_root(self):
        """ExternalTrustRootProvider fails closed if trust store path is inside PROJECT_ROOT."""
        in_repo_path = os.path.join(PROJECT_ROOT, "core", "security", "trusted_training_authorities.json")
        provider = ExternalTrustRootProvider(trust_store_path=in_repo_path)
        authorities = provider.get_authorities()
        self.assertEqual(authorities, {})

    # Vector 4: Missing external trust root fails closed
    def test_04_missing_external_trust_root_fails_closed(self):
        """Nonexistent external trust store path fails closed to empty dict."""
        provider = ExternalTrustRootProvider(trust_store_path=os.path.join(self.tmp_path, "missing_store.json"))
        authorities = provider.get_authorities()
        self.assertEqual(authorities, {})

    # Vector 5: Invalid external trust root fails closed
    def test_05_invalid_external_trust_root_fails_closed(self):
        """Corrupt or non-JSON external trust store fails closed."""
        bad_store = os.path.join(self.tmp_path, "corrupt_store.json")
        with open(bad_store, "w") as f:
            f.write("not json {{{")
        provider = ExternalTrustRootProvider(trust_store_path=bad_store)
        self.assertEqual(provider.get_authorities(), {})

    # Vector 6: Revoked authority is rejected (ATTESTATION_REVOKED_AUTHORITY)
    def test_06_revoked_authority_is_rejected(self):
        """Attestation signed by a revoked authority is rejected with ATTESTATION_REVOKED_AUTHORITY."""
        ckpt = self._create_minimal_checkpoint("revoked_auth_test.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="authority_01")
        auth["authority_01"]["status"] = "revoked"

        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(
            att, ckpt, trust_provider=provider,
            require_production_provider=False, require_artifacts=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_REVOKED_AUTHORITY")

    # Vector 7: Expired authority is rejected (ATTESTATION_EXPIRED_AUTHORITY)
    def test_07_expired_authority_is_rejected(self):
        """Attestation signed by an expired authority is rejected with ATTESTATION_EXPIRED_AUTHORITY."""
        ckpt = self._create_minimal_checkpoint("expired_auth_test.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="authority_01")
        auth["authority_01"]["status"] = "expired"

        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(
            att, ckpt, trust_provider=provider,
            require_production_provider=False, require_artifacts=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_EXPIRED_AUTHORITY")

    # Vector 8: Test key is rejected in production
    def test_08_test_key_is_rejected_in_production(self):
        """TestIsolatedTrustProvider reports is_production_provider() == False."""
        provider = TestIsolatedTrustProvider({})
        self.assertFalse(provider.is_production_provider())

    # Vector 9: Unknown signer is rejected
    def test_09_unknown_signer_is_rejected(self):
        """Attestation with unknown verifier_id returns ATTESTATION_UNTRUSTED_SIGNER."""
        ckpt = self._create_minimal_checkpoint("unknown_signer.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="unknown_verifier")
        provider = TestIsolatedTrustProvider({"trusted_authority_99": {"public_key_hex": pub_hex, "status": "active"}})
        is_valid, tier, reason = verify_training_attestation(
            att, ckpt, trust_provider=provider,
            require_production_provider=False, require_artifacts=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_UNTRUSTED_SIGNER")

    # Vector 10: Tampered signature is rejected
    def test_10_tampered_signature_is_rejected(self):
        """Tampered Ed25519 signature fails verification with ATTESTATION_INVALID."""
        ckpt = self._create_minimal_checkpoint("tampered_sig.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="auth_01")
        att["signature"] = "deadbeef" * 16
        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(
            att, ckpt, trust_provider=provider,
            require_production_provider=False, require_artifacts=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_INVALID")

    # Vector 11: Tampered payload is rejected
    def test_11_tampered_payload_is_rejected(self):
        """Altering payload after signing invalidates Ed25519 signature."""
        ckpt = self._create_minimal_checkpoint("tampered_payload.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="auth_01")
        att["step_count"] = 500000  # modified after signature
        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(
            att, ckpt, trust_provider=provider,
            require_production_provider=False, require_artifacts=False,
        )
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_INVALID")

    # Vector 12: Checkpoint byte modification is detected
    def test_12_checkpoint_byte_modification_detected(self):
        """Modifying checkpoint file bytes after attestation generation triggers hash mismatch."""
        ckpt = self._create_minimal_checkpoint("byte_tamper.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="auth_01")
        # Append byte to checkpoint file
        with open(ckpt, "ab") as f:
            f.write(b"\x00\x01\x02")
        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(att, ckpt, trust_provider=provider)
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_CHECKPOINT_HASH_MISMATCH")

    # Vector 13: Invalid timestamp ordering is rejected (start >= completion)
    def test_13_chronological_ordering_invalid_rejected(self):
        """Training start time not before completion time is rejected."""
        ckpt = self._create_minimal_checkpoint("timestamp_order.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="auth_01")
        # Invert timestamps
        att["training_start_time"] = "2026-11-20T10:00:00Z"
        att["training_completion_time"] = "2026-11-20T08:00:00Z"
        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(att, ckpt, trust_provider=provider)
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_INVALID")

    # Vector 14: Future timestamp is rejected
    def test_14_future_timestamp_rejected(self):
        """Attestation with future completion timestamp is rejected by evidence validator."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        future_time = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat().replace("+00:00", "Z")
        evidence["training_completion_time"] = future_time
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 15: Invalid hash formats rejected (non-64 hex chars)
    def test_15_invalid_hash_formats_rejected(self):
        """Attestation with invalid hash lengths or characters is rejected."""
        ckpt = self._create_minimal_checkpoint("bad_hash.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="auth_01")
        att["model_architecture_hash"] = "short_hash_not_64"
        provider = TestIsolatedTrustProvider(auth)
        is_valid, tier, reason = verify_training_attestation(att, ckpt, trust_provider=provider)
        self.assertFalse(is_valid)
        self.assertEqual(tier, "ATTESTATION_INVALID")

    # Vector 16: Missing training evidence is rejected when validated
    def test_16_missing_training_evidence_rejected(self):
        """Evidence validator fails closed on None or non-dict manifest."""
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(None, "aa" * 32)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 17: Empty training metrics are rejected
    def test_17_empty_training_metrics_rejected(self):
        """Evidence manifest with empty metrics list is rejected."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        evidence["metrics_history"] = []
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 18: Non-monotonic step numbers rejected
    def test_18_non_monotonic_step_numbers_rejected(self):
        """Metrics history with non-monotonic steps (e.g. 50 then 25) is rejected."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        evidence["metrics_history"] = [
            {"step": 100, "val_loss": 3.0, "timestamp": "2026-11-20T08:10:00Z"},
            {"step": 50, "val_loss": 2.8, "timestamp": "2026-11-20T08:20:00Z"},
        ]
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 19: Dataset hash mismatch rejected
    def test_19_dataset_hash_mismatch_rejected(self):
        """Evidence manifest dataset hash conflicting with attestation hash is rejected."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        att = {"dataset_manifest_hash": "bb" * 32}
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32, attestation=att)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 20: Architecture hash mismatch rejected
    def test_20_architecture_hash_mismatch_rejected(self):
        """Evidence manifest architecture hash conflicting with attestation hash is rejected."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        att = {"model_architecture_hash": "cc" * 32}
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32, attestation=att)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 21: Training code hash mismatch rejected
    def test_21_training_code_hash_mismatch_rejected(self):
        """Evidence manifest code tree hash conflicting with attestation hash is rejected."""
        evidence = create_sample_training_evidence("sample_run", "aa" * 32)
        att = {"training_code_hash": "dd" * 32}
        ok, tier, reason = TrainingEvidenceValidator.validate_evidence_manifest(evidence, "aa" * 32, attestation=att)
        self.assertFalse(ok)
        self.assertEqual(tier, "ATTESTATION_EVIDENCE_INVALID")

    # Vector 22: Direct ModelService.generate() cannot bypass production gate
    def test_22_direct_modelservice_generate_blocks_unverified(self):
        """Direct ModelService.generate() raises ModelNotTrainedException if checkpoint lacks genuine verification."""
        ms = ModelService(base_dir=self.tmp_path)
        ms.active_model_name = "scratch"
        with self.assertRaises(ModelNotTrainedException):
            ms.generate("test prompt", allow_fixture=False)

    # Vector 23: API /chat production gate
    def test_23_api_chat_production_gate(self):
        """ModelService reports MODEL_NOT_READY when unpromoted / unverified."""
        ms = ModelService(base_dir=self.tmp_path)
        ms.active_model_name = "moon-chat-pilot-v0"
        self.assertEqual(ms.get_status(), "MODEL_NOT_READY")

    # Vector 24: Startup / discovery blocks unverified
    def test_24_startup_discovery_blocks_unverified(self):
        """find_pilot_checkpoint(allow_fixture=False) returns None for unverified checkpoints."""
        ckpt = self._create_minimal_checkpoint("unverified_pilot.pt")
        ms = ModelService(base_dir=self.tmp_path)
        res = ms.find_pilot_checkpoint(allow_fixture=False)
        self.assertIsNone(res)

    # Vector 25: DEBUG_INFERENCE=1 cannot bypass production gate
    def test_25_debug_inference_cannot_bypass_production_gate(self):
        """DEBUG_INFERENCE=1 environment variable does NOT bypass production gate."""
        os.environ["DEBUG_INFERENCE"] = "1"
        try:
            ckpt = self._create_minimal_checkpoint("debug_infer_test.pt")
            ms = ModelService(base_dir=self.tmp_path, preferred_checkpoint=ckpt)
            ms.active_model_name = "scratch"
            ms.is_test_fixture = True
            with self.assertRaises(ModelNotTrainedException):
                ms.generate("Hello from debug", allow_fixture=False)
        finally:
            os.environ.pop("DEBUG_INFERENCE", None)

    # Vector 26: AUTHORIZED_BY_PREM invariant
    def test_26_authorized_by_prem_invariant(self):
        """AUTHORIZED_BY_PREM must be strictly False."""
        self.assertFalse(policy.AUTHORIZED_BY_PREM)

    # Vector 27: Strict sovereign policy invariant
    def test_27_strict_sovereign_policy_invariant(self):
        """ACTIVE_OWNERSHIP_POLICY must strictly be STRICT_SOVEREIGN_RANDOM_INIT_ONLY."""
        self.assertEqual(policy.ACTIVE_OWNERSHIP_POLICY, "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")


if __name__ == "__main__":
    unittest.main()
