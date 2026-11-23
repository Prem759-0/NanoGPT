"""
tests/test_zero_trust_release_gate.py
Zero-Trust Cryptographic Dataset Attestation, Checkpoint Provenance,
and Production Release Gate Security Audit Test Suite for Moon AI.

Architect: Prem Gaikwad (India 🇮🇳)
Stage 2.0.4:
20 Adversarial & Cryptographic Test Scenarios:
1.  Self-declared EXTERNALLY_VERIFIED without signed attestation rejected.
2.  Self-declared independent_verification=True boolean rejected.
3.  Valid signed cryptographic attestation from active keystore verifier passes.
4.  Forged / tampered HMAC-SHA256 attestation signature fails closed.
5.  Unknown / unregistered verifier_id fails closed.
6.  Inactive / revoked verifier fails closed.
7.  Expired verifier trust certificate fails closed.
8.  Attestation asset SHA-256 mismatch (asset-swapping attack) fails closed.
9.  Attestation record ID mismatch (record-swapping attack) fails closed.
10. Attestation license ID mismatch fails closed.
11. Attestation with commercial_use_permitted=False rejected for production.
12. Saving production checkpoint without manifest_path fails closed.
13. Saving production checkpoint with non-existent manifest file fails closed.
14. Saving production checkpoint with empty manifest (0 records) fails closed.
15. Saving production checkpoint with synthetic_sample_count > 0 fails closed.
16. Saving production checkpoint with production_eligible_sample_count == 0 fails closed.
17. Manifest drift detection: altering manifest after training invalidates checkpoint provenance.
18. Candidate model promotion without checkpoint_path is rejected.
19. Candidate model promotion without checkpoint_sha256 or with checksum mismatch is rejected.
20. Candidate model promotion referencing development synthetic checkpoint is rejected.
"""
import os
import sys
import json
import tempfile
import unittest
import hashlib
from datetime import datetime, timezone
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from cryptography.hazmat.primitives.asymmetric import ed25519

from datasets.security.attestation_engine import (
    sign_attestation,
    verify_attestation,
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
    TRAINING_MODE_DEVELOPMENT,
    PROVENANCE_TAG
)
from scripts.promote_best_model import evaluate_and_promote
from evaluation.security.evaluation_attestation import sign_evaluation_attestation

TRUSTED_VERIFIER_ID = "moon_compliance_authority_v1"

class DummyTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

class TestZeroTrustReleaseGate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = self.tmpdir.name

        # Create genuine dummy image fixture
        self.img_rel = "datasets/images/zero_trust_real.png"
        self.img_full = os.path.join(self.base_dir, "datasets", "images", "zero_trust_real.png")
        os.makedirs(os.path.dirname(self.img_full), exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(12, 34, 56))
        img.save(self.img_full, format="PNG")

        with open(self.img_full, "rb") as f:
            self.img_sha = hashlib.sha256(f.read()).hexdigest()

        # Keys for Ed25519 verifiers
        self.trusted_priv = ed25519.Ed25519PrivateKey.generate()
        self.trusted_pub_hex = self.trusted_priv.public_key().public_bytes_raw().hex()

        self.revoked_priv = ed25519.Ed25519PrivateKey.generate()
        self.revoked_pub_hex = self.revoked_priv.public_key().public_bytes_raw().hex()

        self.expired_priv = ed25519.Ed25519PrivateKey.generate()
        self.expired_pub_hex = self.expired_priv.public_key().public_bytes_raw().hex()

        # Custom trusted verifiers keystore for adversarial tests
        self.custom_keystore_path = os.path.join(self.base_dir, "test_trusted_verifiers.json")
        self.custom_keystore = {
            "schema_version": "2.0",
            "authority": "Test Security Board",
            "policy": "ED25519_ASYMMETRIC_ONLY",
            "updated_at": "2026-11-01T00:00:00Z",
            "trusted_verifiers": {
                TRUSTED_VERIFIER_ID: {
                    "verifier_id": TRUSTED_VERIFIER_ID,
                    "name": "Moon AI Independent Compliance & Rights Authority",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.trusted_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                },
                "revoked_verifier_v1": {
                    "verifier_id": "revoked_verifier_v1",
                    "name": "Compromised Authority",
                    "status": "REVOKED",
                    "algorithm": "Ed25519",
                    "public_key": self.revoked_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                },
                "expired_verifier_v1": {
                    "verifier_id": "expired_verifier_v1",
                    "name": "Expired Auditor",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.expired_pub_hex,
                    "valid_from": "2020-01-01T00:00:00Z",
                    "valid_until": "2021-01-01T00:00:00Z"
                }
            }
        }
        with open(self.custom_keystore_path, "w", encoding="utf-8") as f:
            json.dump(self.custom_keystore, f, indent=2)

        # Custom trusted evaluators keystore for promotion tests
        self.eval_priv = ed25519.Ed25519PrivateKey.generate()
        self.eval_pub_hex = self.eval_priv.public_key().public_bytes_raw().hex()
        self.eval_id = "test_evaluator_v1"
        self.custom_evaluators_path = os.path.join(self.base_dir, "test_trusted_evaluators.json")
        self.custom_evaluators = {
            "schema_version": "2.0",
            "authority": "Test Evaluation Board",
            "policy": "ASYMMETRIC_ED25519_ONLY",
            "updated_at": "2026-11-01T00:00:00Z",
            "trusted_evaluators": {
                self.eval_id: {
                    "evaluator_id": self.eval_id,
                    "name": "Test Independent Evaluator",
                    "status": "ACTIVE",
                    "algorithm": "Ed25519",
                    "public_key": self.eval_pub_hex,
                    "valid_from": "2026-01-01T00:00:00Z",
                    "valid_until": "2030-01-01T00:00:00Z"
                }
            }
        }
        with open(self.custom_evaluators_path, "w", encoding="utf-8") as f:
            json.dump(self.custom_evaluators, f, indent=2)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_base_record(self, record_id: str = "img_zt_001") -> dict:
        return {
            "id": record_id,
            "media_path": self.img_rel,
            "image_path": self.img_rel,
            "caption": "A crystal clear mountain creek flowing through dense cedar pine forest",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": f"owner://declarations/{record_id}",
            "ownership_evidence_reference": f"LEGAL_EVIDENCE_REF_{record_id}",
            "license": "CustomSovereign",
            "sha256": self.img_sha,
            "split": "train",
            "is_synthetic": False,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "ownership_evidence_reference": f"LEGAL_EVIDENCE_REF_{record_id}",
                "asset_sha256": self.img_sha,
                "rights_statement": "I certify that I captured this genuine photograph and hold all commercial rights."
            }
        }

    # =========================================================================
    # TASK 3 & FIX A: ZERO-TRUST ATTESTATION ADVERSARIAL TESTS (1-11)
    # =========================================================================

    def test_01_self_declared_externally_verified_without_attestation_rejected(self):
        """1. Submitters cannot self-declare EXTERNALLY_VERIFIED without signed cryptographic attestation."""
        rec = self._build_base_record("img_self_ext")
        rec["verification_state"] = VERIFICATION_STATE_EXTERNALLY_VERIFIED
        rec["owner_declaration"]["verification_state"] = VERIFICATION_STATE_EXTERNALLY_VERIFIED

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_SELF_DECLARED)
        self.assertEqual(audit["status"], VERIFICATION_STATE_PENDING_MANUAL)
        self.assertTrue(any("without cryptographic attestation" in err for err in audit["errors"]))

    def test_02_self_declared_independent_verification_bool_rejected(self):
        """2. Self-declared independent_verification=True flag without cryptographic attestation fails closed."""
        rec = self._build_base_record("img_self_bool")
        rec["independent_verification"] = True
        rec["owner_declaration"]["independent_verification"] = True

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_SELF_DECLARED)

    def test_03_valid_signed_attestation_passes(self):
        """3. Authentically signed cryptographic attestation from active keystore verifier passes cleanly."""
        rec = self._build_base_record("img_valid_att")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_valid_att",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_valid_att",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_valid_att",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertTrue(audit["is_valid"], f"Audit errors: {audit['errors']}")
        self.assertTrue(audit["training_eligible"])
        self.assertEqual(audit["status"], "VERIFIED_PRODUCTION_ELIGIBLE")
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_EXTERNALLY_VERIFIED)

    def test_04_forged_attestation_signature_fails_closed(self):
        """4. Forged Ed25519 signature on an attestation bundle fails closed with REJECTED."""
        rec = self._build_base_record("img_forged_sig")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_forged_sig",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_forged_sig",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_forged_sig",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        # Tamper with signature
        att["signature"] = "deadbeef" * 16
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Signature altered or forged" in err for err in audit["errors"]))

    def test_05_unknown_verifier_id_fails_closed(self):
        """5. Attestation signed by unknown or unregistered verifier_id fails closed."""
        rec = self._build_base_record("img_unknown_verifier")
        rogue_priv = ed25519.Ed25519PrivateKey.generate()
        att = sign_attestation(
            verifier_id="unregistered_adversary_org",
            private_key=rogue_priv,
            record_id="img_unknown_verifier",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_unknown_verifier",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_unknown_verifier",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Unknown or untrusted verifier_id" in err for err in audit["errors"]))

    def test_06_revoked_inactive_verifier_fails_closed(self):
        """6. Attestation signed by revoked or inactive verifier fails closed."""
        rec = self._build_base_record("img_revoked_verifier")
        att = sign_attestation(
            verifier_id="revoked_verifier_v1",
            private_key=self.revoked_priv,
            record_id="img_revoked_verifier",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_revoked_verifier",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_revoked_verifier",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("is not ACTIVE" in err for err in audit["errors"]))

    def test_07_expired_verifier_certificate_fails_closed(self):
        """7. Attestation signed by a verifier whose certificate has expired fails closed."""
        rec = self._build_base_record("img_expired_verifier")
        att = sign_attestation(
            verifier_id="expired_verifier_v1",
            private_key=self.expired_priv,
            record_id="img_expired_verifier",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_expired_verifier",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_expired_verifier",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("certificate has expired" in err for err in audit["errors"]))

    def test_08_attestation_asset_sha256_mismatch_fails_closed(self):
        """8. Attestation cryptographic binding detects asset-swapping attack (SHA mismatch)."""
        rec = self._build_base_record("img_asset_swap")
        other_sha = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
        # Signed for other_sha, attached to rec with self.img_sha
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_asset_swap",
            asset_sha256=other_sha,
            source_url="owner://declarations/img_asset_swap",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_asset_swap",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("asset_sha256 mismatch" in err for err in audit["errors"]))

    def test_09_attestation_record_id_mismatch_fails_closed(self):
        """9. Attestation cryptographic binding detects record-swapping attack (ID mismatch)."""
        rec = self._build_base_record("img_record_swap_a")
        # Attestation was issued for record_swap_b
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_record_swap_b",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_record_swap_a",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_record_swap_a",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("record_id mismatch" in err for err in audit["errors"]))

    def test_10_attestation_license_mismatch_fails_closed(self):
        """10. Attestation cryptographic binding detects license mismatch."""
        rec = self._build_base_record("img_license_swap")
        # Attestation was issued for CC0, but record declares CustomSovereign
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_license_swap",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_license_swap",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_license_swap",
            license_id="CC0",
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
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("license mismatch" in err for err in audit["errors"]))

    def test_11_attestation_commercial_use_forbidden_fails_closed(self):
        """11. Attestation with commercial_use_permitted=False rejected for production."""
        rec = self._build_base_record("img_non_commercial")
        att = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="img_non_commercial",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_non_commercial",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_non_commercial",
            license_id="CustomSovereign",
            commercial_use_permitted=False, # Commercial forbidden!
            timestamp="2026-11-01T00:00:00Z"
        )
        rec["attestation"] = att

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_REJECTED)
        self.assertTrue(any("Production requires commercial suitability" in err for err in audit["errors"]))

    # =========================================================================
    # TASK 4 & FIX B/D: CHECKPOINT PROVENANCE & MANIFEST INTEGRITY (12-17)
    # =========================================================================

    def _create_valid_manifests(self):
        train_manifest = os.path.join(self.base_dir, "train.jsonl")
        val_manifest = os.path.join(self.base_dir, "val.jsonl")
        rec_train = self._build_base_record("rec_train_001")
        rec_train["split"] = "train"
        rec_train["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="rec_train_001",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_train_001",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_rec_train_001",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec_val = self._build_base_record("rec_val_001")
        rec_val["split"] = "val"
        rec_val["attestation"] = sign_attestation(
            verifier_id=TRUSTED_VERIFIER_ID,
            private_key=self.trusted_priv,
            record_id="rec_val_001",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/rec_val_001",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_rec_val_001",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        with open(train_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")
        return train_manifest, val_manifest

    def test_12_save_production_checkpoint_without_manifest_fails_closed(self):
        """12. save_checkpoint in PRODUCTION_GENUINE mode strictly requires manifest_path."""
        model = DummyTinyModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "test_ckpt.pt")
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
                manifest_path=None
            )
        self.assertIn("manifest_path is strictly required for PRODUCTION_GENUINE", str(ctx.exception))

    def test_13_save_production_checkpoint_with_missing_manifest_file_fails_closed(self):
        """13. save_checkpoint in PRODUCTION_GENUINE mode rejects non-existent manifest files."""
        model = DummyTinyModel()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "test_ckpt.pt")
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
                manifest_path="datasets/nonexistent_manifest.jsonl",
                val_manifest_path="datasets/nonexistent_val.jsonl"
            )
        self.assertIn("does not exist on disk", str(ctx.exception))

    def test_14_save_production_checkpoint_with_empty_manifest_fails_closed(self):
        """14. save_checkpoint in PRODUCTION_GENUINE mode rejects empty manifest files (0 records)."""
        model = DummyTinyModel()
        train_p = os.path.join(self.base_dir, "empty_train.jsonl")
        val_p = os.path.join(self.base_dir, "empty_val.jsonl")
        with open(train_p, "w", encoding="utf-8") as f:
            f.write("# comment only\n\n")
        with open(val_p, "w", encoding="utf-8") as f:
            f.write("# comment only\n\n")

        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "test_ckpt.pt")
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
                production_eligible_sample_count=10
            )
        self.assertIn("contains 0 records", str(ctx.exception))

    def test_15_save_production_checkpoint_with_synthetic_samples_fails_closed(self):
        """15. save_checkpoint in PRODUCTION_GENUINE mode strictly rejects synthetic samples > 0."""
        model = DummyTinyModel()
        train_p, val_p = self._create_valid_manifests()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "test_ckpt.pt")

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
                synthetic_sample_count=5, # Forbidden in production!
                production_eligible_sample_count=50
            )
        self.assertIn("Cannot emit PRODUCTION_GENUINE checkpoint with synthetic samples", str(ctx.exception))

    def test_16_save_production_checkpoint_with_zero_production_eligible_fails_closed(self):
        """16. save_checkpoint in PRODUCTION_GENUINE mode rejects production_eligible_sample_count == 0."""
        model = DummyTinyModel()
        train_p, val_p = self._create_valid_manifests()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "test_ckpt.pt")

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
                production_eligible_sample_count=0 # Forbidden!
            )
        self.assertIn("production_eligible_sample_count must be > 0", str(ctx.exception))

    def test_17_manifest_drift_detection_invalidates_checkpoint(self):
        """17. Manifest drift / modification after training invalidates checkpoint provenance on disk."""
        model = DummyTinyModel()
        train_p, val_p = self._create_valid_manifests()
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_image", "drift_ckpt.pt")

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
            verifiers_path=self.custom_keystore_path,
            base_dir=self.base_dir
        )

        # Baseline verify: should pass
        valid, meta, errs = verify_checkpoint_provenance(
            out_ckpt,
            mode=TRAINING_MODE_PRODUCTION,
            recompute_manifests=True,
            base_dir=self.base_dir,
            verifiers_path=self.custom_keystore_path
        )
        self.assertTrue(valid, f"Initial verification failed: {errs}")

        # Adversarially alter the manifest on disk (manifest drift attack)
        with open(train_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "injected_unverified_record", "split": "train"}) + "\n")

        # Re-verify: must fail closed because actual hash drifted from checkpoint manifest_hash!
        valid2, meta2, errs2 = verify_checkpoint_provenance(
            out_ckpt,
            mode=TRAINING_MODE_PRODUCTION,
            recompute_manifests=True,
            base_dir=self.base_dir,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(valid2, "Manifest drift must fail checkpoint verification")
        self.assertTrue(any("manifest hash drift detected" in err for err in errs2))

    # =========================================================================
    # TASK 5 & FIX C: MODEL PROMOTION SECURITY TESTS (18-20)
    # =========================================================================

    def test_18_promote_candidate_without_checkpoint_path_rejected(self):
        """18. Model promotion strictly rejects candidate evaluation report without checkpoint_path."""
        rep_path = os.path.join(self.base_dir, "eval_no_ckpt.json")
        reg_path = os.path.join(self.base_dir, "test_registry.json")
        rep_data = {
            "model_name": "moon-promoted-test",
            "composite_quality_score": 98.0,
            "non_empty_rate_pct": 100.0,
            "coding_syntax_valid_pct": 100.0
            # checkpoint_path omitted!
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)

        promoted = evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.custom_evaluators_path,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(promoted, "Promotion must be rejected when checkpoint_path is missing")

    def test_19_promote_candidate_without_checkpoint_sha256_rejected(self):
        """19. Model promotion strictly rejects report without checkpoint_sha256 or with checksum mismatch."""
        model = DummyTinyModel()
        train_p, val_p = self._create_valid_manifests()
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "moon_image", "cand_ckpt.pt")
        reg_path = os.path.join(self.base_dir, "test_registry.json")

        save_checkpoint(
            model=model,
            optimizer=None,
            step=10,
            epoch=1,
            loss=0.1,
            config_dict={},
            output_path=ckpt_path,
            model_name="moon-promoted-test",
            training_mode=TRAINING_MODE_PRODUCTION,
            manifest_path=train_p,
            val_manifest_path=val_p,
            synthetic_sample_count=0,
            production_eligible_sample_count=2,
            verifiers_path=self.custom_keystore_path,
            base_dir=self.base_dir
        )

        rep_path = os.path.join(self.base_dir, "eval_bad_sha.json")
        # 1. Missing checkpoint_sha256
        rep_data = {
            "model_name": "moon-promoted-test",
            "composite_quality_score": 98.0,
            "non_empty_rate_pct": 100.0,
            "coding_syntax_valid_pct": 100.0,
            "checkpoint_path": ckpt_path
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertFalse(evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.custom_evaluators_path,
            verifiers_path=self.custom_keystore_path,
            base_dir=self.base_dir
        ))

        # 2. Checkpoint SHA-256 mismatch (tampered checksum)
        rep_data["checkpoint_sha256"] = "f" * 64
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertFalse(evaluate_and_promote(
            rep_path,
            registry_path=reg_path,
            evaluators_path=self.custom_evaluators_path,
            verifiers_path=self.custom_keystore_path,
            base_dir=self.base_dir
        ))

        # 3. Correct SHA-256 and valid signed evaluation attestation passes promotion
        actual_sha = compute_file_sha256(ckpt_path)
        rep_data["checkpoint_sha256"] = actual_sha
        rep_data["evaluation_attestation"] = sign_evaluation_attestation(
            evaluator_id=self.eval_id,
            private_key=self.eval_priv,
            checkpoint_sha256=actual_sha,
            model_name="moon-promoted-test",
            architecture_version="1.0",
            composite_quality_score=98.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            timestamp="2026-11-01T00:00:00Z"
        )
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)
        self.assertTrue(
            evaluate_and_promote(
                rep_path,
                registry_path=reg_path,
                evaluators_path=self.custom_evaluators_path,
                verifiers_path=self.custom_keystore_path,
                base_dir=self.base_dir
            ),
            "Valid candidate with matching SHA-256 and signed attestation must pass promotion"
        )

    def test_20_promote_candidate_with_development_checkpoint_rejected(self):
        """20. Model promotion strictly rejects reports pointing to DEVELOPMENT_SYNTHETIC checkpoints."""
        model = DummyTinyModel()
        dev_ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "moon_image", "dev_ckpt.pt")
        reg_path = os.path.join(self.base_dir, "test_registry.json")

        save_checkpoint(
            model=model,
            optimizer=None,
            step=5,
            epoch=1,
            loss=0.2,
            config_dict={},
            output_path=dev_ckpt_path,
            model_name="moon-dev-candidate",
            training_mode=TRAINING_MODE_DEVELOPMENT,
            synthetic_sample_count=10
        )

        dev_sha = compute_file_sha256(dev_ckpt_path)
        rep_path = os.path.join(self.base_dir, "eval_dev_ckpt.json")
        rep_data = {
            "model_name": "moon-dev-candidate",
            "composite_quality_score": 99.0,
            "non_empty_rate_pct": 100.0,
            "coding_syntax_valid_pct": 100.0,
            "checkpoint_path": dev_ckpt_path,
            "checkpoint_sha256": dev_sha
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep_data, f)

        promoted = evaluate_and_promote(rep_path, registry_path=reg_path)
        self.assertFalse(promoted, "Promotion must strictly reject candidate built from development synthetic checkpoint")

if __name__ == "__main__":
    unittest.main()
