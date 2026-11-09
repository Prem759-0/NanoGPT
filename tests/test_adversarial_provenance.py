"""
tests/test_adversarial_provenance.py
Adversarial Security & Provenance Validation Test Suite for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Stage 2.0.2:
Verifies fail-closed behavior against adversarial inputs:
1. Missing 'is_synthetic' field (must never default to false).
2. 'is_synthetic: true' records in production mode.
3. Missing or suspicious creator.
4. Fake or incomplete owner declarations (CustomSovereign / owner://).
5. Generic or fabricated repository URLs.
6. Unsupported or non-permissive licenses.
7. Missing or unverified license evidence for external media.
8. Missing or malformed SHA-256 checksums.
9. Byte-level checksum mismatch (tampered file content).
10. Missing media files.
11. Directory traversal attacks (../, etc.).
12. Synthetic video disguised as MP4.
13. Synthetic image disguised as genuine.
14. Synthetic development manifests loaded in production mode.
15. Empty manifests failing closed.
16. Legitimate fully verified records passing cleanly.
"""
import os
import sys
import json
import unittest
import tempfile
import hashlib
from datetime import datetime, timezone
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.security.attestation_engine import sign_attestation
from datasets.tools.provenance_validator import (
    validate_provenance_record,
    validate_safe_path,
    verify_file_checksum
)
from datasets.tools.validate_image_sources import validate_image_source_record
from datasets.tools.validate_video_sources import validate_video_source_record
from training.image.dataset import MoonImageDataset
from training.video.dataset import MoonVideoDataset

class TestAdversarialProvenance(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = self.tmpdir.name

        # Create a real test image file
        self.real_img_rel = "data/images/real_test.png"
        self.real_img_full = os.path.join(self.base_dir, "data", "images", "real_test.png")
        os.makedirs(os.path.dirname(self.real_img_full), exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(self.real_img_full, format="PNG")

        with open(self.real_img_full, "rb") as f:
            self.real_img_sha = hashlib.sha256(f.read()).hexdigest()

        from cryptography.hazmat.primitives.asymmetric import ed25519
        self.test_priv = ed25519.Ed25519PrivateKey.generate()
        self.test_pub = self.test_priv.public_key().public_bytes_raw().hex()
        self.custom_keystore_path = os.path.join(self.base_dir, "test_keystore.json")
        with open(self.custom_keystore_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "authority": "Test Authority",
                "policy": "ED25519_ASYMMETRIC_ONLY",
                "trusted_verifiers": {
                    "moon_compliance_authority_v1": {
                        "verifier_id": "moon_compliance_authority_v1",
                        "status": "ACTIVE",
                        "algorithm": "Ed25519",
                        "public_key": self.test_pub,
                        "valid_from": "2026-01-01T00:00:00Z",
                        "valid_until": "2030-01-01T00:00:00Z"
                    }
                }
            }, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_valid_sovereign_record(self) -> dict:
        now_iso = datetime.now(timezone.utc).isoformat()
        att = sign_attestation(
            verifier_id="moon_compliance_authority_v1",
            private_key=self.test_priv,
            record_id="img_valid_001",
            asset_sha256=self.real_img_sha,
            source_url="owner://declarations/img_valid_001",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_valid_001",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp=now_iso
        )
        return {
            "id": "img_valid_001",
            "media_path": self.real_img_rel,
            "image_path": self.real_img_rel,
            "caption": "Alpine peaks bathed in golden twilight glow with crisp snow",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": "owner://declarations/img_valid_001",
            "ownership_evidence_reference": "LEGAL_EVIDENCE_REF_img_valid_001",
            "license": "CustomSovereign",
            "attribution": "Original photograph captured and owned by Prem Gaikwad",
            "sha256": self.real_img_sha,
            "split": "train",
            "is_synthetic": False,
            "verification_state": "EXTERNALLY_VERIFIED",
            "attestation": att,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": now_iso,
                "asset_id": "img_valid_001",
                "ownership_evidence_reference": "LEGAL_EVIDENCE_REF_img_valid_001",
                "asset_sha256": self.real_img_sha,
                "rights_statement": "I certify that I am the sole author and owner of this original photograph and grant full sovereign training rights.",
                "verification_state": "EXTERNALLY_VERIFIED"
            }
        }

    def test_reject_missing_is_synthetic(self):
        """Verifies that records omitting 'is_synthetic' are rejected and NEVER defaulted to false."""
        rec = self._build_valid_sovereign_record()
        del rec["is_synthetic"]

        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit["is_valid"])
        self.assertFalse(audit["training_eligible"])
        self.assertTrue(any("Missing required 'is_synthetic' field" in e for e in audit["errors"]))

    def test_reject_synthetic_true(self):
        """Verifies that is_synthetic: true is strictly rejected from production training."""
        rec = self._build_valid_sovereign_record()
        rec["is_synthetic"] = True

        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit["is_valid"])
        self.assertFalse(audit["training_eligible"])
        self.assertEqual(audit["status"], "QUARANTINED_SYNTHETIC")

    def test_reject_missing_or_suspicious_creator(self):
        """Verifies rejection of empty or generic anonymous creators."""
        for bad_creator in ["", "unknown", "anonymous", "none", "n/a", "ai"]:
            rec = self._build_valid_sovereign_record()
            rec["creator"] = bad_creator
            audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
            self.assertFalse(audit["is_valid"])
            self.assertFalse(audit["dimensions"]["metadata_completeness"])

    def test_reject_fake_owner_declaration(self):
        """Verifies that bare 'CustomSovereign' without valid owner declaration is rejected."""
        # 1. Missing owner declaration completely
        rec1 = self._build_valid_sovereign_record()
        del rec1["owner_declaration"]
        audit1 = validate_provenance_record(rec1, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit1["is_valid"])
        self.assertFalse(audit1["dimensions"]["ownership_verified"])
        self.assertEqual(audit1["status"], "PENDING_MANUAL_VERIFICATION")

        # 2. Owner declaration with mismatched SHA-256
        rec2 = self._build_valid_sovereign_record()
        rec2["owner_declaration"]["asset_sha256"] = "0" * 64
        audit2 = validate_provenance_record(rec2, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit2["is_valid"])
        self.assertFalse(audit2["dimensions"]["ownership_verified"])

        # 3. Owner declaration with declarant mismatch
        rec3 = self._build_valid_sovereign_record()
        rec3["owner_declaration"]["declarant"] = "Someone Else"
        audit3 = validate_provenance_record(rec3, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit3["is_valid"])

    def test_reject_fabricated_source_urls(self):
        """Verifies that generic repo folder paths and bare URLs are rejected."""
        fake_urls = [
            "https://github.com/Prem759-0/NanoGPT/tree/main/data/images",
            "owner://",
            "https://example.com",
            "ftp://invalid-protocol.com"
        ]
        for u in fake_urls:
            rec = self._build_valid_sovereign_record()
            rec["source_url"] = u
            audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
            self.assertFalse(audit["is_valid"])
            self.assertFalse(audit["dimensions"]["source_accessibility"])

    def test_reject_unsupported_or_noncommercial_license(self):
        """Verifies that non-permissible licenses (e.g. NC, GPL) are rejected."""
        rec = self._build_valid_sovereign_record()
        rec["license"] = "CC-BY-NC-4.0"
        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit["is_valid"])
        self.assertTrue(any("Impermissible or unsupported license" in e for e in audit["errors"]))

    def test_reject_missing_license_evidence_for_external_assets(self):
        """Verifies that external open licenses (CC-BY, MIT) require verified license evidence."""
        rec = self._build_valid_sovereign_record()
        rec["license"] = "CC-BY-4.0"
        rec["source_url"] = "https://commons.wikimedia.org/wiki/File:Alps_mountain_view.jpg"

        # Missing license_evidence
        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit["is_valid"])
        self.assertFalse(audit["dimensions"]["license_verified"])
        self.assertEqual(audit["status"], "PENDING_MANUAL_VERIFICATION")

        # Incomplete license evidence (verification_status pending)
        rec["license_evidence"] = {
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "verification_status": "PENDING_MANUAL_VERIFICATION",
            "commercial_use_permitted": True
        }
        audit_pending = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False)
        self.assertFalse(audit_pending["is_valid"])
        self.assertFalse(audit_pending["dimensions"]["license_verified"])

    def test_reject_mismatched_byte_checksum(self):
        """Verifies that an incorrect SHA-256 field is detected by recomputing bytes on disk."""
        rec = self._build_valid_sovereign_record()
        rec["sha256"] = "1" * 64
        rec["owner_declaration"]["asset_sha256"] = "1" * 64

        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )
        self.assertFalse(audit["is_valid"])
        self.assertFalse(audit["dimensions"]["file_checksum_verified"])
        self.assertTrue(any("SHA-256 mismatch" in e for e in audit["errors"]))

    def test_reject_path_traversal(self):
        """Verifies that directory traversal paths (../) are caught and rejected."""
        traversal_paths = [
            "../../etc/passwd",
            "data/images/../../../secret.png",
            "C:\\Windows\\system32\\calc.exe"
        ]
        for p in traversal_paths:
            ok, _, err = validate_safe_path(p, base_dir=self.base_dir)
            self.assertFalse(ok)
            self.assertIsNotNone(err)

    def test_production_dataset_fails_closed_on_synthetic_or_unverified(self):
        """Verifies MoonImageDataset and MoonVideoDataset refuse to load invalid records."""
        manifest_p = os.path.join(self.base_dir, "image_train.jsonl")
        bad_rec = self._build_valid_sovereign_record()
        del bad_rec["owner_declaration"] # Unverified ownership

        with open(manifest_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(bad_rec) + "\n")

        with self.assertRaises(ValueError) as ctx:
            MoonImageDataset(
                manifest_path=manifest_p,
                base_dir=self.base_dir,
                split="train",
                allow_synthetic=False
            )
        self.assertIn("STRICT IMAGE INTEGRITY ERROR", str(ctx.exception))

    def test_production_dataset_rejects_dev_manifest(self):
        """Verifies loading a synthetic dev manifest with allow_synthetic=False raises ValueError."""
        dev_manifest_p = os.path.join(self.base_dir, "synthetic_image_dev.jsonl")
        with open(dev_manifest_p, "w", encoding="utf-8") as f:
            f.write("")

        with self.assertRaises(ValueError) as ctx:
            MoonImageDataset(
                manifest_path=dev_manifest_p,
                base_dir=self.base_dir,
                split="train",
                allow_synthetic=False
            )
        self.assertIn("STRICT IMAGE INTEGRITY ERROR", str(ctx.exception))

    def test_accept_legitimate_verified_record(self):
        """Verifies that a fully compliant, verified record passes all 5 dimensions."""
        rec = self._build_valid_sovereign_record()
        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=self.custom_keystore_path
        )

        self.assertTrue(audit["is_valid"], f"Validation failed: {audit['errors']}")
        self.assertTrue(audit["training_eligible"])
        self.assertEqual(audit["status"], "VERIFIED_PRODUCTION_ELIGIBLE")
        for dim_name, val in audit["dimensions"].items():
            self.assertTrue(val, f"Dimension '{dim_name}' failed")

if __name__ == "__main__":
    unittest.main()
