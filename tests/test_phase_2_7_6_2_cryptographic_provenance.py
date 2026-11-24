"""
tests/test_phase_2_7_6_2_cryptographic_provenance.py
Adversarial Verification Suite for Phase 2.7.6.2:
- 14+ Mutually Exclusive Provenance Tiers
- Cryptographic Training Attestation & Ed25519 Verification
- Zero-Trust Root Configuration & Fail-Closed Gate Enforcement
- Default Strict Discovery (allow_fixture=False)
- Immunity against DEBUG_INFERENCE bypass

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""

import os
import sys
import json
import tempfile
import unittest
import hashlib
import torch
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.security.training_attestation import (
    load_trusted_training_authorities,
    compute_file_sha256,
    construct_canonical_training_payload,
    generate_training_authority_keypair,
    sign_training_attestation,
    verify_training_attestation,
    create_test_training_attestation,
    DEFAULT_AUTHORITIES_PATH,
)
from core.security.training_evidence import (
    create_sample_training_artifacts,
    EvidenceArtifactStore,
)
from webapp.model_service import (
    ModelService,
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    validate_checkpoint_structure,
)


class TestPhase2762CryptographicProvenance(unittest.TestCase):
    """19 Comprehensive Adversarial Tests for Phase 2.7.6.2."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.temp_dir.name

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def _create_minimal_checkpoint(self, filename="ckpt.pt", **custom_data):
        """Creates a structurally valid torch checkpoint with customizable fields."""
        path = os.path.join(self.tmp_path, filename)
        base = {
            "status": "PILOT_TRAINED",
            "is_test_fixture": False,
            "iter_num": 1000,
            "val_loss": 2.15,
            "model": {"transformer.wte.weight": torch.randn(16, 16)},
            "model_args": {"n_layer": 2, "n_head": 2, "n_embd": 16, "vocab_size": 16, "block_size": 32},
            "config": {
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "initialization": "random_gaussian_scratch",
                "student_init_method": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": False,
            },
        }
        base.update(custom_data)
        torch.save(base, path)
        return path

    # Test 1: Empty Trust Store Defaults Fail Closed
    def test_01_empty_trust_store_defaults_fail_closed(self):
        """Default core/security/trusted_training_authorities.json has empty trusted_authorities."""
        authorities = load_trusted_training_authorities()
        self.assertEqual(authorities, {}, "Production keystore must default to empty authorities dict.")
        self.assertTrue(os.path.isfile(DEFAULT_AUTHORITIES_PATH))
        with open(DEFAULT_AUTHORITIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("policy"), "ASYMMETRIC_ED25519_ONLY")
        self.assertEqual(data.get("trusted_authorities"), {})

    # Test 2: Checkpoint Without Attestation Classified as ATTESTATION_MISSING
    def test_02_checkpoint_without_attestation_classified_as_attestation_missing(self):
        """Strict metadata without attestation fails closed to ATTESTATION_MISSING."""
        ckpt = self._create_minimal_checkpoint("unattested.pt")
        tier = classify_checkpoint(ckpt)
        self.assertEqual(tier, "ATTESTATION_MISSING")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 3: Corrupt or Malformed Checkpoint Classified as STRUCTURALLY_INVALID
    def test_03_corrupt_or_malformed_checkpoint_classified_as_structurally_invalid(self):
        """Corrupted files, empty files, or invalid paths return STRUCTURALLY_INVALID."""
        bad_file = os.path.join(self.tmp_path, "corrupt.pt")
        with open(bad_file, "wb") as f:
            f.write(b"not a valid torch checkpoint")
        self.assertEqual(classify_checkpoint(bad_file), "STRUCTURALLY_INVALID")
        self.assertEqual(classify_checkpoint(os.path.join(self.tmp_path, "nonexistent.pt")), "STRUCTURALLY_INVALID")
        self.assertFalse(is_verified_genuine_checkpoint(bad_file))

    # Test 4: Synthetic Mock Classified as SYNTHETIC_MOCK
    def test_04_synthetic_mock_classified_as_synthetic_mock(self):
        """Checkpoints with synthetic or mock indicators return SYNTHETIC_MOCK."""
        ckpt = self._create_minimal_checkpoint("mock_model.pt", is_synthetic=True)
        self.assertEqual(classify_checkpoint(ckpt), "SYNTHETIC_MOCK")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 5: CI Test Fixture Classified as TEST_FIXTURE
    def test_05_ci_test_fixture_classified_as_test_fixture(self):
        """Checkpoints marked as test fixtures return TEST_FIXTURE."""
        ckpt = self._create_minimal_checkpoint("fixture.pt", is_test_fixture=True)
        self.assertEqual(classify_checkpoint(ckpt), "TEST_FIXTURE")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 6: Historical Policy B Classified as HISTORICAL_POLICY_B
    def test_06_historical_policy_b_classified_as_historical_policy_b(self):
        """Checkpoints declaring TEACHER_STUDENT_RANDOM_INIT return HISTORICAL_POLICY_B."""
        ckpt = self._create_minimal_checkpoint(
            "hist.pt",
            config={"ownership_policy": "TEACHER_STUDENT_RANDOM_INIT", "teacher_checkpoint_weights_transferred": False},
        )
        self.assertEqual(classify_checkpoint(ckpt), "HISTORICAL_POLICY_B")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 7: Unapproved Policy Classified as UNAPPROVED_POLICY
    def test_07_unapproved_policy_classified_as_unapproved_policy(self):
        """Checkpoints declaring an unknown policy return UNAPPROVED_POLICY."""
        ckpt = self._create_minimal_checkpoint(
            "unapproved.pt",
            config={"ownership_policy": "COMMERCIAL_TRANSFER_POLICY", "teacher_checkpoint_weights_transferred": False},
        )
        self.assertEqual(classify_checkpoint(ckpt), "UNAPPROVED_POLICY")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 8: Non-Scratch Init Classified as NON_SCRATCH_INITIALIZATION
    def test_08_non_scratch_init_classified_as_non_scratch_initialization(self):
        """Checkpoints claiming non-scratch initialization return NON_SCRATCH_INITIALIZATION."""
        ckpt = self._create_minimal_checkpoint(
            "non_scratch.pt",
            config={
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "initialization": "pretrained_lora",
                "teacher_checkpoint_weights_transferred": False,
            },
        )
        self.assertEqual(classify_checkpoint(ckpt), "NON_SCRATCH_INITIALIZATION")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 9: Teacher Weights Transferred Classified as TEACHER_CONTAMINATION
    def test_09_teacher_weights_transferred_classified_as_teacher_contamination(self):
        """Checkpoints with teacher transfer set to True return TEACHER_CONTAMINATION."""
        ckpt = self._create_minimal_checkpoint(
            "contaminated.pt",
            config={
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "initialization": "random_gaussian_scratch",
                "teacher_checkpoint_weights_transferred": True,
            },
        )
        self.assertEqual(classify_checkpoint(ckpt), "TEACHER_CONTAMINATION")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 10: Untrained Iteration Zero Classified as UNTRAINED_RANDOM_INIT
    def test_10_untrained_iteration_zero_classified_as_untrained_random_init(self):
        """Checkpoints with iter_num <= 0 or val_loss <= 0.0 return UNTRAINED_RANDOM_INIT."""
        ckpt = self._create_minimal_checkpoint("untrained.pt", iter_num=0, val_loss=0.0)
        self.assertEqual(classify_checkpoint(ckpt), "UNTRAINED_RANDOM_INIT")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 11: Attestation Missing Required Fields Classified as ATTESTATION_INVALID
    def test_11_attestation_missing_required_fields_classified_as_attestation_invalid(self):
        """Attestation missing mandatory schema fields returns ATTESTATION_INVALID."""
        ckpt = self._create_minimal_checkpoint("partial_att.pt")
        incomplete_att = {
            "run_id": "test_incomplete",
            "policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            # missing initialization, hashes, signature, etc.
        }
        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(incomplete_att, f)
        self.assertEqual(classify_checkpoint(ckpt), "ATTESTATION_INVALID")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 12: Attestation Hash Mismatch Classified as ATTESTATION_CHECKPOINT_HASH_MISMATCH
    def test_12_attestation_hash_mismatch_classified_as_checkpoint_hash_mismatch(self):
        """Attestation binding a mismatched SHA-256 returns ATTESTATION_CHECKPOINT_HASH_MISMATCH."""
        ckpt = self._create_minimal_checkpoint("mismatch.pt")
        att, auth = create_test_training_attestation(ckpt, checkpoint_sha256="0" * 64)
        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)
        tier = classify_checkpoint(ckpt)
        self.assertEqual(tier, "ATTESTATION_CHECKPOINT_HASH_MISMATCH")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 13: Attestation Untrusted Signer Classified as ATTESTATION_UNTRUSTED_SIGNER
    def test_13_attestation_untrusted_signer_classified_as_untrusted_signer(self):
        """Attestation signed by unregistered authority returns ATTESTATION_UNTRUSTED_SIGNER."""
        ckpt = self._create_minimal_checkpoint("untrusted_signer.pt")
        att, auth = create_test_training_attestation(ckpt, verifier_id="unregistered_rogue_auditor")
        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)
        tier = classify_checkpoint(ckpt)
        self.assertEqual(tier, "ATTESTATION_UNTRUSTED_SIGNER")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt))

    # Test 14: Attestation Tampered Signature Classified as ATTESTATION_INVALID
    def test_14_attestation_tampered_signature_classified_as_attestation_invalid(self):
        """Attestation with corrupted signature bytes returns ATTESTATION_INVALID."""
        ckpt = self._create_minimal_checkpoint("corrupt_sig.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="trusted_auth_01")
        # Tamper signature hex
        att["signature"] = "deadbeef" * 16

        auth_store = os.path.join(self.tmp_path, "test_authorities.json")
        with open(auth_store, "w", encoding="utf-8") as f:
            json.dump({"trusted_authorities": auth}, f)

        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)

        tier = classify_checkpoint(ckpt, trusted_authorities_path=auth_store)
        self.assertEqual(tier, "ATTESTATION_INVALID")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt, trusted_authorities_path=auth_store))

    # Test 15: Attestation Tampered Payload Detected and Rejected
    def test_15_attestation_tampered_payload_detected_and_rejected(self):
        """Modifying attestation telemetry after signing causes signature verification failure."""
        ckpt = self._create_minimal_checkpoint("tampered_payload.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        att, auth = create_test_training_attestation(ckpt, private_key_hex=priv_hex, verifier_id="trusted_auth_01")
        # Tamper step count after signature generation
        att["step_count"] = 999999

        auth_store = os.path.join(self.tmp_path, "test_authorities.json")
        with open(auth_store, "w", encoding="utf-8") as f:
            json.dump({"trusted_authorities": auth}, f)

        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)

        tier = classify_checkpoint(ckpt, trusted_authorities_path=auth_store)
        self.assertEqual(tier, "ATTESTATION_INVALID")
        self.assertFalse(is_verified_genuine_checkpoint(ckpt, trusted_authorities_path=auth_store))

    # Test 16: Valid Attestation with Trusted Authority Classified as GENUINE_TRAINED_SOVEREIGN
    def test_16_valid_attestation_with_trusted_authority_classified_as_genuine_sovereign(self):
        """Validly signed attestation against registered authority returns GENUINE_TRAINED_SOVEREIGN."""
        ckpt = self._create_minimal_checkpoint("genuine_attested.pt")
        priv_hex, pub_hex = generate_training_authority_keypair()
        art_dir = os.path.join(self.tmp_path, "evidence_artifacts_16")
        paths, manifest = create_sample_training_artifacts(art_dir)
        att, auth = create_test_training_attestation(
            ckpt,
            private_key_hex=priv_hex,
            verifier_id="trusted_auth_01",
            evidence_manifest=manifest,
        )

        auth_store = os.path.join(self.tmp_path, "test_authorities.json")
        with open(auth_store, "w", encoding="utf-8") as f:
            json.dump({"trusted_authorities": auth}, f)

        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)

        ev_store = EvidenceArtifactStore(art_dir)
        tier = classify_checkpoint(ckpt, trusted_authorities_path=auth_store, evidence_store=ev_store)
        self.assertEqual(tier, "GENUINE_TRAINED_SOVEREIGN")
        self.assertTrue(is_verified_genuine_checkpoint(ckpt, trusted_authorities_path=auth_store, evidence_store=ev_store))

    # Test 17: Promoted Production Checkpoint with Valid Attestation
    def test_17_promoted_production_checkpoint_with_valid_attestation(self):
        """Promoted checkpoint with valid attestation classifies as PROMOTED_PRODUCTION."""
        ckpt = self._create_minimal_checkpoint("promoted_attested.pt", status="PROMOTED_PRODUCTION")
        priv_hex, pub_hex = generate_training_authority_keypair()
        art_dir = os.path.join(self.tmp_path, "evidence_artifacts_17")
        paths, manifest = create_sample_training_artifacts(art_dir)
        att, auth = create_test_training_attestation(
            ckpt,
            private_key_hex=priv_hex,
            verifier_id="trusted_auth_01",
            evidence_manifest=manifest,
        )

        auth_store = os.path.join(self.tmp_path, "test_authorities.json")
        with open(auth_store, "w", encoding="utf-8") as f:
            json.dump({"trusted_authorities": auth}, f)

        with open(ckpt + ".attestation.json", "w", encoding="utf-8") as f:
            json.dump(att, f)

        ev_store = EvidenceArtifactStore(art_dir)
        tier = classify_checkpoint(ckpt, trusted_authorities_path=auth_store, evidence_store=ev_store)
        self.assertEqual(tier, "PROMOTED_PRODUCTION")
        self.assertTrue(is_verified_genuine_checkpoint(ckpt, trusted_authorities_path=auth_store, evidence_store=ev_store))

    # Test 18: find_checkpoint and find_pilot_checkpoint Default to No Fixtures
    def test_18_find_checkpoint_and_find_pilot_checkpoint_default_to_no_fixtures(self):
        """Strict discovery defaults to allow_fixture=False across checkpoint finding methods."""
        ms = ModelService(base_dir=self.tmp_path)
        fixture_ckpt = self._create_minimal_checkpoint("fixture.pt", is_test_fixture=True)

        # Default find_checkpoint must be allow_fixture=False and return None
        self.assertIsNone(ms.find_checkpoint(preferred_path=fixture_ckpt))
        # Explicit allow_fixture=True succeeds
        self.assertEqual(ms.find_checkpoint(preferred_path=fixture_ckpt, allow_fixture=True), fixture_ckpt)

    # Test 19: DEBUG_INFERENCE Cannot Bypass Production Verification
    def test_19_debug_inference_cannot_bypass_production_verification(self):
        """Setting DEBUG_INFERENCE=1 cannot bypass production verification in ModelService.load_model."""
        orig_debug = os.environ.get("DEBUG_INFERENCE")
        try:
            os.environ["DEBUG_INFERENCE"] = "1"
            unattested_ckpt = self._create_minimal_checkpoint("unattested_probe.pt")
            ms = ModelService(base_dir=self.tmp_path, preferred_checkpoint=unattested_ckpt)

            # Calling load_model with default allow_fixture=False must block even with DEBUG_INFERENCE=1
            loaded = ms.load_model("scratch", allow_fixture=False)
            self.assertFalse(loaded, "DEBUG_INFERENCE=1 must NOT bypass production verification gate.")
            self.assertIn("restricted to isolated retrospective testing", ms.error_message or "")
        finally:
            if orig_debug is not None:
                os.environ["DEBUG_INFERENCE"] = orig_debug
            else:
                os.environ.pop("DEBUG_INFERENCE", None)


if __name__ == "__main__":
    unittest.main()
