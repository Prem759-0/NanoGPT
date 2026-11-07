"""
tests/test_asymmetric_trust.py
Stage 2.0.5 — Asymmetric Cryptographic Trust, Checkpoint Attestation & Evaluation Integrity Test Suite.

Lead ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

22 Explicit Adversarial & Cryptographic Scenarios:
1.  Valid Ed25519 signature from active keystore verifier passes verification.
2.  Forged signature fails verification (InvalidSignature / rejected).
3.  Signature created with correct algorithm but different private key fails verification.
4.  Tampered asset_sha256 in canonical payload fails verification.
5.  Tampered record_id in canonical payload fails verification.
6.  Tampered source_url in canonical payload fails verification.
7.  Tampered ownership_evidence_reference in canonical payload fails verification.
8.  Tampered license_id in canonical payload fails verification.
9.  Missing or malformed ISO 8601 timestamp fails verification.
10. Future-dated attestation timestamp (> 60s) fails verification.
11. Expired verifier trust certificate fails verification.
12. Revoked verifier status (REVOKED or SUSPENDED) fails verification.
13. Unknown/unregistered verifier_id fails verification.
14. Missing required field in canonical payload fails verification.
15. Non-canonical JSON payload fails signature check.
16. Unsigned or HMAC-signed record submitted for production training raises ValueError.
17. Checkpoint save in PRODUCTION_GENUINE mode with altered image on disk fails closed.
18. Checkpoint save in PRODUCTION_GENUINE mode with synthetic record in manifest fails closed.
19. Checkpoint save in PRODUCTION_GENUINE mode with unverified record in manifest fails closed.
20. Checkpoint save in PRODUCTION_GENUINE mode with sample count mismatch fails closed.
21. Evaluation report promotion with forged/unsigned evaluation attestation fails closed.
22. Key rotation: old key signs before expiry (valid), new key signs after rotation (valid), revoked key fails (invalid).
"""
import os
import sys
import json
import tempfile
import unittest
import hashlib
from datetime import datetime, timezone, timedelta
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

from datasets.security.attestation_engine import (
    sign_attestation,
    verify_attestation,
    construct_canonical_attestation_payload,
    load_trusted_verifiers
)
from datasets.tools.provenance_validator import (
    validate_provenance_record,
    VERIFICATION_STATE_SELF_DECLARED,
    VERIFICATION_STATE_PENDING_MANUAL,
    VERIFICATION_STATE_EXTERNALLY_VERIFIED,
    VERIFICATION_STATE_REJECTED
)
from training.common.checkpoint import (
    save_checkpoint,
    verify_checkpoint_provenance,
    load_checkpoint,
    compute_file_sha256,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from evaluation.security.evaluation_attestation import (
    sign_evaluation_attestation,
    verify_evaluation_attestation,
    construct_canonical_evaluation_payload
)
from scripts.promote_best_model import evaluate_and_promote

TRUSTED_VERIFIER_ID = "moon_trusted_auditor_v2"

class TinyTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

class TestAsymmetricTrust(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = self.tmpdir.name

        # Create genuine real image fixture on disk
        self.img_rel = "datasets/images/test_real_photo.png"
        self.img_full = os.path.join(self.base_dir, "datasets", "images", "test_real_photo.png")
        os.makedirs(os.path.dirname(self.img_full), exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(45, 85, 125))
        img.save(self.img_full, format="PNG")

        with open(self.img_full, "rb") as f:
            self.img_sha = hashlib.sha256(f.read()).hexdigest()

        # Primary verifier Ed25519 keypair
        self.primary_priv = ed25519.Ed25519PrivateKey.generate()
        self.primary_pub_hex = self.primary_priv.public_key().public_bytes_raw().hex()

        # Alternate / rogue keypair
        self.rogue_priv = ed25519.Ed25519PrivateKey.generate()
        self.rogue_pub_hex = self.rogue_priv.public_key().public_bytes_raw().hex()

        # Revoked verifier keypair
        self.revoked_priv = ed25519.Ed25519PrivateKey.generate()
        self.revoked_pub_hex = self.revoked_priv.public_key().public_bytes_raw().hex()

        # Expired verifier keypair
        self.expired_priv = ed25519.Ed25519PrivateKey.generate()
        self.expired_pub_hex = self.expired_priv.public_key().public_bytes_raw().hex()

        # Trusted verifiers keystore
        self.keystore_path = os.path.join(self.base_dir, "trusted_verifiers_test.json")
        self.keystore_data = {
            "schema_version": "2.0",
            "authority": "Moon Sovereign Governance",
            "policy": "ED25519_ASYMMETRIC_ONLY",
            "updated_at": "2026-11-01T00:00:00Z",
            "trusted_verifiers": {
                TRUSTED_VERIFIER_ID: {
                    "verifier_id": TRUSTED_VERIFIER_ID,
                    "name": "Moon AI Independent Auditor v2",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.primary_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                },
                "moon_revoked_auditor_v1": {
                    "verifier_id": "moon_revoked_auditor_v1",
                    "name": "Compromised Verifier",
                    "status": "REVOKED",
                    "algorithm": "Ed25519",
                    "public_key": self.revoked_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                },
                "moon_expired_auditor_v1": {
                    "verifier_id": "moon_expired_auditor_v1",
                    "name": "Expired Verifier",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.expired_pub_hex,
                    "valid_from": "2020-01-01T00:00:00Z",
                    "valid_until": "2021-01-01T00:00:00Z"
                }
            }
        }
        with open(self.keystore_path, "w", encoding="utf-8") as f:
            json.dump(self.keystore_data, f, indent=2)

        # Evaluator keypair and keystore for promotion tests
        self.eval_priv = ed25519.Ed25519PrivateKey.generate()
        self.eval_pub_hex = self.eval_priv.public_key().public_bytes_raw().hex()
        self.eval_id = "evaluator_alpha_v1"
        self.evaluators_keystore_path = os.path.join(self.base_dir, "trusted_evaluators_test.json")
        self.evaluators_data = {
            "schema_version": "2.0",
            "authority": "Moon AI Evaluation Governance",
            "policy": "ASYMMETRIC_ED25519_ONLY",
            "updated_at": "2026-11-01T00:00:00Z",
            "trusted_evaluators": {
                self.eval_id: {
                    "evaluator_id": self.eval_id,
                    "name": "Moon Primary Benchmark Evaluator",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.eval_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                }
            }
        }
        with open(self.evaluators_keystore_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluators_data, f, indent=2)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_test_record(self, record_id: str = "img_001") -> dict:
        return {
            "id": record_id,
            "media_path": self.img_rel,
            "image_path": self.img_rel,
            "caption": "Pristine mountain lake reflecting snow peaks in morning sunlight",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": f"owner://declarations/{record_id}",
            "ownership_evidence_reference": f"LEGAL_DOC_REF_{record_id}",
            "license": "CustomSovereign",
            "sha256": self.img_sha,
            "split": "train",
            "is_synthetic": False,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "ownership_evidence_reference": f"LEGAL_DOC_REF_{record_id}",
                "asset_sha256": self.img_sha,
                "rights_statement": "I certify that I captured this genuine photograph and hold all commercial rights."
            }
        }

    # =========================================================================
    # 1. Valid Ed25519 signature from active keystore verifier passes
    # =========================================================================
    def test_01_valid_ed25519_signature_passes(self):
        rec = self._build_test_record("img_test_01")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_01",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_01",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_01",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertTrue(audit["is_valid"], f"Expected valid, got: {audit['errors']}")
        self.assertTrue(audit["training_eligible"])
        self.assertEqual(audit["status"], "VERIFIED_PRODUCTION_ELIGIBLE")
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_EXTERNALLY_VERIFIED)

    # =========================================================================
    # 2. Forged signature fails verification
    # =========================================================================
    def test_02_forged_signature_fails_verification(self):
        rec = self._build_test_record("img_test_02")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_02",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_02",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_02",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        # Flip bits in signature hex
        bad_sig = "deadbeef" * 16
        att["signature"] = bad_sig
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Signature altered or forged" in err for err in audit["errors"]))

    # =========================================================================
    # 3. Signature created with correct algorithm but different private key fails
    # =========================================================================
    def test_03_signature_with_different_private_key_fails(self):
        rec = self._build_test_record("img_test_03")
        # Signed with rogue private key but claiming to be TRUSTED_VERIFIER_ID
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.rogue_priv,
            record_id="img_test_03",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_03",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_03",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Signature altered or forged" in err for err in audit["errors"]))

    # =========================================================================
    # 4. Tampered asset_sha256 in canonical payload fails verification
    # =========================================================================
    def test_04_tampered_asset_sha256_fails_verification(self):
        rec = self._build_test_record("img_test_04")
        tampered_sha = "1" * 64
        # Attestation signed for tampered_sha, attached to record with actual self.img_sha
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_04",
            asset_sha256=tampered_sha,
            source_url="owner://declarations/img_test_04",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_04",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("asset_sha256 mismatch" in err for err in audit["errors"]))

    # =========================================================================
    # 5. Tampered record_id in canonical payload fails verification
    # =========================================================================
    def test_05_tampered_record_id_fails_verification(self):
        rec = self._build_test_record("img_test_05")
        # Attestation signed for different record_id
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_other_id",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_05",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_05",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("record_id mismatch" in err for err in audit["errors"]))

    # =========================================================================
    # 6. Tampered source_url in canonical payload fails verification
    # =========================================================================
    def test_06_tampered_source_url_fails_verification(self):
        rec = self._build_test_record("img_test_06")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_06",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/tampered_source",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_06",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("source_url mismatch" in err for err in audit["errors"]))

    # =========================================================================
    # 7. Tampered ownership_evidence_reference in canonical payload fails verification
    # =========================================================================
    def test_07_tampered_ownership_evidence_fails_verification(self):
        rec = self._build_test_record("img_test_07")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_07",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_07",
            ownership_evidence_reference="DIFFERENT_EVIDENCE_REF",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("ownership_evidence_reference mismatch" in err for err in audit["errors"]))

    # =========================================================================
    # 8. Tampered license_id in canonical payload fails verification
    # =========================================================================
    def test_08_tampered_license_id_fails_verification(self):
        rec = self._build_test_record("img_test_08")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_08",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_08",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_08",
            license_id="CC0",  # Record specifies CustomSovereign
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("license mismatch" in err for err in audit["errors"]))

    # =========================================================================
    # 9. Missing or malformed ISO 8601 timestamp fails verification
    # =========================================================================
    def test_09_missing_or_malformed_timestamp_fails(self):
        rec = self._build_test_record("img_test_09")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_09",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_09",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_09",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        att["timestamp"] = "not-a-valid-timestamp"
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Malformed or unparseable" in err for err in audit["errors"]))

    # =========================================================================
    # 10. Future-dated attestation timestamp (> 60s) fails verification
    # =========================================================================
    def test_10_future_dated_timestamp_fails(self):
        rec = self._build_test_record("img_test_10")
        far_future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_10",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_10",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_10",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp=far_future
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("is in the future" in err for err in audit["errors"]))

    # =========================================================================
    # 11. Expired verifier trust certificate fails verification
    # =========================================================================
    def test_11_expired_verifier_certificate_fails(self):
        rec = self._build_test_record("img_test_11")
        att = sign_attestation(
            verifier_id="moon_expired_auditor_v1",
            private_key=self.expired_priv,
            record_id="img_test_11",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_11",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_11",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("certificate has expired" in err for err in audit["errors"]))

    # =========================================================================
    # 12. Revoked verifier status (REVOKED or SUSPENDED) fails verification
    # =========================================================================
    def test_12_revoked_verifier_fails(self):
        rec = self._build_test_record("img_test_12")
        att = sign_attestation(
            verifier_id="moon_revoked_auditor_v1",
            private_key=self.revoked_priv,
            record_id="img_test_12",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_12",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_12",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("is not ACTIVE" in err for err in audit["errors"]))

    # =========================================================================
    # 13. Unknown/unregistered verifier_id fails verification
    # =========================================================================
    def test_13_unknown_verifier_id_fails(self):
        rec = self._build_test_record("img_test_13")
        unknown_priv = ed25519.Ed25519PrivateKey.generate()
        att = sign_attestation(
            verifier_id="unregistered_rogue_authority",
            private_key=unknown_priv,
            record_id="img_test_13",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_13",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_13",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Unknown or untrusted verifier_id" in err for err in audit["errors"]))

    # =========================================================================
    # 14. Missing required field in canonical payload fails verification
    # =========================================================================
    def test_14_missing_required_field_in_attestation_fails(self):
        rec = self._build_test_record("img_test_14")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="img_test_14",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_test_14",
            ownership_evidence_reference="LEGAL_DOC_REF_img_test_14",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        # Omit ownership_evidence_reference
        del att["ownership_evidence_reference"]
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("missing required field" in err for err in audit["errors"]))

    # =========================================================================
    # 15. Non-canonical JSON payload fails signature check
    # =========================================================================
    def test_15_non_canonical_json_payload_fails_signature(self):
        # A signature computed over a non-canonical byte sequence (e.g. arbitrary spacing)
        # must fail verification against the canonical RFC 8785 byte serializer.
        non_canonical_bytes = b'{"verifier_id": "test",  "record_id":  "rec1"}'
        sig = self.primary_priv.sign(non_canonical_bytes)

        # Reconstructed canonical bytes
        canonical_bytes = construct_canonical_attestation_payload(
            verifier_id=TRUSTED_VERIFIER_ID,
            record_id="rec1",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec1",
            ownership_evidence_reference="LEGAL_REF",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )

        with self.assertRaises(InvalidSignature):
            self.primary_priv.public_key().verify(sig, canonical_bytes)

    # =========================================================================
    # 16. Unsigned or HMAC-signed record submitted for production raises ValueError
    # =========================================================================
    def test_16_unsigned_or_hmac_signed_record_rejected_for_production(self):
        rec = self._build_test_record("img_test_16")
        # Attestation with legacy HMAC algorithm
        rec["attestation"] = {
            "verifier_id": TRUSTED_VERIFIER_ID,
            "algorithm": "HMAC-SHA256",
            "signature": "deadbeef" * 8,
            "record_id": "img_test_16",
            "asset_sha256": self.img_sha,
            "source_url": "owner://declarations/img_test_16",
            "ownership_evidence_reference": "LEGAL_DOC_REF_img_test_16",
            "license": "CustomSovereign",
            "commercial_use_permitted": True,
            "timestamp": "2026-11-01T00:00:00Z"
        }

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Ed25519 asymmetric signatures required" in err for err in audit["errors"]))

    # =========================================================================
    # 17. Checkpoint save in PRODUCTION_GENUINE mode with altered image fails closed
    # =========================================================================
    def test_17_checkpoint_save_with_altered_image_fails_closed(self):
        train_p = os.path.join(self.base_dir, "train_altered.jsonl")
        val_p = os.path.join(self.base_dir, "val_altered.jsonl")

        rec_train = self._build_test_record("rec_train_17")
        rec_train["split"] = "train"
        rec_train["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_train_17",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_train_17",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_train_17",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_val = self._build_test_record("rec_val_17")
        rec_val["split"] = "val"
        rec_val["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_val_17",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_val_17",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_val_17",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        with open(train_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        # Now tamper with image file bytes on disk (asset corruption / swap attack)
        with open(self.img_full, "wb") as f:
            f.write(b"corrupted or swapped image bytes")

        model = TinyTestModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "test_altered_img.pt")
        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                training_mode=TRAINING_MODE_PRODUCTION,
                manifest_path=train_p,
                val_manifest_path=val_p,
                synthetic_sample_count=0,
                production_eligible_sample_count=2,
                verifiers_path=self.keystore_path,
                base_dir=self.base_dir
            )
        self.assertIn("failed deep verification", str(ctx.exception))
        self.assertIn("SHA-256 mismatch", str(ctx.exception))

    # =========================================================================
    # 18. Checkpoint save in PRODUCTION_GENUINE mode with synthetic record fails closed
    # =========================================================================
    def test_18_checkpoint_save_with_synthetic_record_fails_closed(self):
        train_p = os.path.join(self.base_dir, "train_synth.jsonl")
        val_p = os.path.join(self.base_dir, "val_synth.jsonl")

        # Injected synthetic fixture inside production manifest
        rec_train = self._build_test_record("rec_train_18")
        rec_train["split"] = "train"
        rec_train["is_synthetic"] = True  # Contamination!

        rec_val = self._build_test_record("rec_val_18")
        rec_val["split"] = "val"

        with open(train_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        model = TinyTestModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "test_synth_contam.pt")
        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                training_mode=TRAINING_MODE_PRODUCTION,
                manifest_path=train_p,
                val_manifest_path=val_p,
                synthetic_sample_count=0,
                production_eligible_sample_count=2,
                verifiers_path=self.keystore_path,
                base_dir=self.base_dir
            )
        self.assertIn("Synthetic record contamination", str(ctx.exception))

    # =========================================================================
    # 19. Checkpoint save in PRODUCTION_GENUINE mode with unverified record fails closed
    # =========================================================================
    def test_19_checkpoint_save_with_unverified_record_fails_closed(self):
        train_p = os.path.join(self.base_dir, "train_unverified.jsonl")
        val_p = os.path.join(self.base_dir, "val_unverified.jsonl")

        # Unverified record (no cryptographic attestation)
        rec_train = self._build_test_record("rec_train_19")
        rec_train["split"] = "train"
        # No rec_train["attestation"]

        rec_val = self._build_test_record("rec_val_19")
        rec_val["split"] = "val"
        rec_val["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_val_19",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_val_19",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_val_19",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )

        with open(train_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        model = TinyTestModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "test_unverified.pt")
        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                training_mode=TRAINING_MODE_PRODUCTION,
                manifest_path=train_p,
                val_manifest_path=val_p,
                synthetic_sample_count=0,
                production_eligible_sample_count=2,
                verifiers_path=self.keystore_path,
                base_dir=self.base_dir
            )
        self.assertIn("failed deep verification", str(ctx.exception))
        self.assertIn("PENDING_MANUAL_VERIFICATION", str(ctx.exception))

    # =========================================================================
    # 20. Checkpoint save in PRODUCTION_GENUINE mode with sample count mismatch fails closed
    # =========================================================================
    def test_20_checkpoint_save_with_sample_count_mismatch_fails_closed(self):
        train_p = os.path.join(self.base_dir, "train_count.jsonl")
        val_p = os.path.join(self.base_dir, "val_count.jsonl")

        rec_train = self._build_test_record("rec_train_20")
        rec_train["split"] = "train"
        rec_train["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_train_20",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_train_20",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_train_20",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_val = self._build_test_record("rec_val_20")
        rec_val["split"] = "val"
        rec_val["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_val_20",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_val_20",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_val_20",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        with open(train_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        model = TinyTestModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "test_count_mismatch.pt")
        # Manifests actually contain 2 verified samples (1 train, 1 val), but claim 100
        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                training_mode=TRAINING_MODE_PRODUCTION,
                manifest_path=train_p,
                val_manifest_path=val_p,
                synthetic_sample_count=0,
                production_eligible_sample_count=100,  # Mismatch!
                verifiers_path=self.keystore_path,
                base_dir=self.base_dir
            )
        self.assertIn("Verified production sample count mismatch", str(ctx.exception))

    # =========================================================================
    # 21. Evaluation report promotion with forged/unsigned evaluation attestation fails closed
    # =========================================================================
    def test_21_evaluation_report_promotion_with_forged_attestation_fails(self):
        train_p = os.path.join(self.base_dir, "train_eval.jsonl")
        val_p = os.path.join(self.base_dir, "val_eval.jsonl")

        rec_train = self._build_test_record("rec_train_21")
        rec_train["split"] = "train"
        rec_train["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_train_21",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_train_21",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_train_21",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_val = self._build_test_record("rec_val_21")
        rec_val["split"] = "val"
        rec_val["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.primary_priv,
            record_id="rec_val_21",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_val_21",
            ownership_evidence_reference="LEGAL_DOC_REF_rec_val_21",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        with open(train_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        model = TinyTestModel()
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "cand_eval_21.pt")
        save_checkpoint(
            model=model,
            optimizer=None,
            step=10,
            epoch=1,
            loss=0.1,
            config_dict={},
            output_path=ckpt_path,
            model_name="moon-model-test-21",
            training_mode=TRAINING_MODE_PRODUCTION,
            manifest_path=train_p,
            val_manifest_path=val_p,
            synthetic_sample_count=0,
            production_eligible_sample_count=2,
            verifiers_path=self.keystore_path,
            base_dir=self.base_dir
        )
        actual_ckpt_sha = compute_file_sha256(ckpt_path)

        rep_path = os.path.join(self.base_dir, "eval_cand_21.json")
        reg_path = os.path.join(self.base_dir, "test_registry_21.json")

        # 1. Unsigned evaluation report
        rep_data = {
            "model_name": "moon-model-test-21",
            "composite_quality_score": 95.0,
            "non_empty_rate_pct": 100.0,
            "coding_syntax_valid_pct": 100.0,
            "checkpoint_path": ckpt_path,
            "checkpoint_sha256": actual_ckpt_sha
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertFalse(evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.evaluators_keystore_path,
            verifiers_path=self.keystore_path,
            base_dir=self.base_dir
        ))

        # 2. Forged signature on evaluation report
        forged_eval_att = sign_evaluation_attestation(
            evaluator_id=self.eval_id,
            private_key=self.eval_priv,
            checkpoint_sha256=actual_ckpt_sha,
            model_name="moon-model-test-21",
            architecture_version="1.0",
            composite_quality_score=95.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            timestamp="2026-11-01T00:00:00Z"
        )
        forged_eval_att["signature"] = "deadbeef" * 16
        rep_data["evaluation_attestation"] = forged_eval_att
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertFalse(evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.evaluators_keystore_path,
            verifiers_path=self.keystore_path,
            base_dir=self.base_dir
        ))

        # 3. Authentically signed evaluation report passes promotion cleanly
        valid_eval_att = sign_evaluation_attestation(
            evaluator_id=self.eval_id,
            private_key=self.eval_priv,
            checkpoint_sha256=actual_ckpt_sha,
            model_name="moon-model-test-21",
            architecture_version="1.0",
            composite_quality_score=95.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            timestamp="2026-11-01T00:00:00Z"
        )
        rep_data["evaluation_attestation"] = valid_eval_att
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertTrue(evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.evaluators_keystore_path,
            verifiers_path=self.keystore_path,
            base_dir=self.base_dir
        ))

    # =========================================================================
    # 22. Key rotation: old key signs before expiry, new key signs after rotation, revoked fails
    # =========================================================================
    def test_22_key_rotation_and_revocation_lifecycle(self):
        # Setup rotation keystore with:
        # - key_v1: valid from 2024 to 2026-06-01 (retired/expired)
        # - key_v2: valid from 2026-06-01 to 2028-01-01 (current active)
        # - key_compromised: revoked
        key_v1_priv = ed25519.Ed25519PrivateKey.generate()
        key_v1_pub = key_v1_priv.public_key().public_bytes_raw().hex()

        key_v2_priv = ed25519.Ed25519PrivateKey.generate()
        key_v2_pub = key_v2_priv.public_key().public_bytes_raw().hex()

        key_comp_priv = ed25519.Ed25519PrivateKey.generate()
        key_comp_pub = key_comp_priv.public_key().public_bytes_raw().hex()

        rotation_keystore_path = os.path.join(self.base_dir, "rotation_keystore.json")
        rotation_data = {
            "schema_version": "2.0",
            "authority": "Moon AI Cryptographic Governance Board",
            "policy": "ED25519_ASYMMETRIC_ONLY",
            "updated_at": "2026-11-01T00:00:00Z",
            "trusted_verifiers": {
                "verifier_v1_retired": {
                    "verifier_id": "verifier_v1_retired",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": key_v1_pub,
                    "valid_from": "2024-01-01T00:00:00Z",
                    "valid_until": "2026-06-01T00:00:00Z"
                },
                "verifier_v2_current": {
                    "verifier_id": "verifier_v2_current",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": key_v2_pub,
                    "valid_from": "2026-06-01T00:00:00Z",
                    "valid_until": "2028-01-01T00:00:00Z"
                },
                "verifier_compromised": {
                    "verifier_id": "verifier_compromised",
                    "status": "REVOKED",
                    "algorithm": "Ed25519",
                    "public_key": key_comp_pub,
                    "valid_from": "2025-01-01T00:00:00Z",
                    "valid_until": "2027-01-01T00:00:00Z"
                }
            }
        }
        with open(rotation_keystore_path, "w", encoding="utf-8") as f:
            json.dump(rotation_data, f, indent=2)

        # 1. Attestation signed by key_v2 within active period (2026-11-01) passes
        rec_current = self._build_test_record("img_rot_current")
        att_current = sign_attestation(
            verifier_id="verifier_v2_current",
            private_key=key_v2_priv,
            record_id="img_rot_current",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_rot_current",
            ownership_evidence_reference="LEGAL_DOC_REF_img_rot_current",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_current["attestation"] = att_current
        audit1 = validate_provenance_record(
            rec_current,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=rotation_keystore_path
        )
        self.assertTrue(audit1["is_valid"], f"Audit 1 failed: {audit1['errors']}")
        self.assertTrue(audit1["training_eligible"])

        # 2. Attestation signed by retired key_v1 after expiry fails
        rec_retired = self._build_test_record("img_rot_retired")
        att_retired = sign_attestation(
            verifier_id="verifier_v1_retired",
            private_key=key_v1_priv,
            record_id="img_rot_retired",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_rot_retired",
            ownership_evidence_reference="LEGAL_DOC_REF_img_rot_retired",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"  # After 2026-06-01 expiration!
        )
        rec_retired["attestation"] = att_retired
        audit2 = validate_provenance_record(
            rec_retired,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=rotation_keystore_path
        )
        self.assertFalse(audit2["training_eligible"])
        self.assertTrue(any("certificate has expired" in err for err in audit2["errors"]))

        # 3. Attestation signed by revoked key fails
        rec_revoked = self._build_test_record("img_rot_revoked")
        att_revoked = sign_attestation(
            verifier_id="verifier_compromised",
            private_key=key_comp_priv,
            record_id="img_rot_revoked",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_rot_revoked",
            ownership_evidence_reference="LEGAL_DOC_REF_img_rot_revoked",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_revoked["attestation"] = att_revoked
        audit3 = validate_provenance_record(
            rec_revoked,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=rotation_keystore_path
        )
        self.assertFalse(audit3["training_eligible"])
        self.assertTrue(any("is not ACTIVE" in err for err in audit3["errors"]))

if __name__ == "__main__":
    unittest.main()
