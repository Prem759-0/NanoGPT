"""
tests/test_checkpoint_security_and_provenance.py
Comprehensive Unit Test Suite for Stage 2.0.3:
1. Training Isolation (PRODUCTION_GENUINE vs DEVELOPMENT_SYNTHETIC).
2. Checkpoint Provenance & Tampering Protection.
3. Production Model Promotion Rejection of Dev Models.
4. Symlink Escape & Path Traversal Rejection.
5. Malformed Manifest JSON Error Reporting with Line Number.
6. Honest Ownership Verification States (SELF_DECLARED vs EXTERNALLY_VERIFIED).
7. External Licence Evidence Verification.
8. Production Training Fail-Closed on Unverified Media.
9. Genuine Fully Verified Fixture Admittance.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import json
import unittest
import tempfile
import hashlib
import random
import numpy as np
from datetime import datetime, timezone
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.common.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint_provenance,
    get_current_rng_state,
    restore_rng_state,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT,
    PROVENANCE_TAG
)
from datasets.security.attestation_engine import sign_attestation
from datasets.tools.provenance_validator import (
    validate_safe_path,
    validate_provenance_record,
    VERIFICATION_STATE_SELF_DECLARED,
    VERIFICATION_STATE_PENDING_MANUAL,
    VERIFICATION_STATE_EXTERNALLY_VERIFIED
)
from training.image.dataset import MoonImageDataset
from training.video.dataset import MoonVideoDataset
from scripts.promote_best_model import evaluate_and_promote

class TinyDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

class TestCheckpointSecurityAndProvenance(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = self.tmpdir.name

        # Create dummy real image
        self.img_rel = "data/images/test_image.png"
        self.img_full = os.path.join(self.base_dir, "data", "images", "test_image.png")
        os.makedirs(os.path.dirname(self.img_full), exist_ok=True)
        img = Image.new("RGB", (64, 64), color="green")
        img.save(self.img_full)
        with open(self.img_full, "rb") as f:
            self.img_sha = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_synthetic_dev_training_cannot_produce_production_checkpoint(self):
        """Ensures DEVELOPMENT_SYNTHETIC training cannot emit a production-eligible checkpoint."""
        model = TinyDummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "dev", "model.pt")

        # 1. Reject explicit is_production_eligible=True in dev mode
        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=10,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                training_mode=TRAINING_MODE_DEVELOPMENT,
                is_production_eligible=True
            )
        self.assertIn("cannot emit a production-eligible checkpoint", str(ctx.exception))

        # 2. Reject saving dev checkpoint directly into production path
        prod_ckpt_path = os.path.join(self.base_dir, "checkpoints", "moon_image", "best_moon_image.pt")
        with self.assertRaises(ValueError) as ctx2:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=10,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=prod_ckpt_path,
                training_mode=TRAINING_MODE_DEVELOPMENT
            )
        self.assertIn("Cannot save DEVELOPMENT_SYNTHETIC checkpoint to production path", str(ctx2.exception))

        # 3. Reject saving PRODUCTION_GENUINE checkpoint if synthetic samples exist
        with self.assertRaises(ValueError) as ctx3:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=10,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=os.path.join(self.base_dir, "checkpoints", "moon_image", "best_moon_image.pt"),
                training_mode=TRAINING_MODE_PRODUCTION,
                synthetic_sample_count=25
            )
        self.assertIn("Cannot emit PRODUCTION_GENUINE checkpoint with synthetic samples", str(ctx3.exception))

    def test_production_loader_rejects_synthetic_checkpoints(self):
        """Verifies that load_checkpoint in PRODUCTION_GENUINE mode rejects DEVELOPMENT_SYNTHETIC checkpoints."""
        model = TinyDummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dev_ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "dev_model.pt")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=5,
            epoch=1,
            loss=0.2,
            config_dict={},
            output_path=dev_ckpt_path,
            training_mode=TRAINING_MODE_DEVELOPMENT,
            synthetic_sample_count=10
        )

        fresh_model = TinyDummyModel()
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(
                checkpoint_path=dev_ckpt_path,
                model=fresh_model,
                mode=TRAINING_MODE_PRODUCTION
            )
        self.assertIn("Attempted to load DEVELOPMENT_SYNTHETIC checkpoint in PRODUCTION_GENUINE mode", str(ctx.exception))

    def test_production_model_promotion_rejects_development_checkpoints(self):
        """Verifies that model promotion strictly rejects candidate reports marked DEVELOPMENT_SYNTHETIC."""
        dev_report_path = os.path.join(self.base_dir, "eval_dev_report.json")
        report_data = {
            "model_name": "moon-image-dev-test",
            "training_mode": "DEVELOPMENT_SYNTHETIC",
            "is_production_eligible": False,
            "composite_quality_score": 95.0,
            "non_empty_rate_pct": 100.0,
            "coding_syntax_valid_pct": 100.0,
            "checkpoint_path": "checkpoints/dev/moon_image/best_moon_image.pt"
        }
        with open(dev_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f)

        promoted = evaluate_and_promote(dev_report_path)
        self.assertFalse(promoted, "Promotion should be rejected for development synthetic candidate")

    def test_missing_checkpoint_training_mode_rejected(self):
        """Verifies that checkpoints missing 'training_mode' are rejected upon loading."""
        model = TinyDummyModel()
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "legacy_ckpt.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        payload = {
            "provenance": PROVENANCE_TAG,
            "model_state_dict": model.state_dict(),
            "step": 1,
            "epoch": 1,
            "loss": 0.1
            # Missing training_mode!
        }
        torch.save(payload, ckpt_path)

        fresh_model = TinyDummyModel()
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(checkpoint_path=ckpt_path, model=fresh_model, enforce_provenance=True)
        self.assertIn("Missing required 'training_mode' provenance metadata", str(ctx.exception))

    def test_tampered_checkpoint_metadata_rejected(self):
        """Verifies that metadata tampering (e.g. claiming production eligibility on dev model) is rejected."""
        model = TinyDummyModel()
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "tampered_ckpt.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # Adversarially claim is_production_eligible=True while training_mode is DEVELOPMENT_SYNTHETIC
        payload = {
            "provenance": PROVENANCE_TAG,
            "training_mode": TRAINING_MODE_DEVELOPMENT,
            "is_production_eligible": True, # Tampered!
            "synthetic_sample_count": 50,
            "model_state_dict": model.state_dict(),
            "step": 1,
            "epoch": 1,
            "loss": 0.1
        }
        torch.save(payload, ckpt_path)

        fresh_model = TinyDummyModel()
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(checkpoint_path=ckpt_path, model=fresh_model, enforce_provenance=True)
        self.assertIn("Metadata tampering detected", str(ctx.exception))

    def test_symlink_escape_rejected(self):
        """Verifies that symlinks escaping the dataset root are detected and rejected."""
        outside_file = os.path.join(self.base_dir, "secret_outside.txt")
        with open(outside_file, "w") as f:
            f.write("confidential data")

        dataset_root = os.path.join(self.base_dir, "datasets", "images")
        os.makedirs(dataset_root, exist_ok=True)
        symlink_path = os.path.join(dataset_root, "symlink_escape.png")

        try:
            os.symlink(outside_file, symlink_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported in this environment")

        safe_ok, resolved, err = validate_safe_path("symlink_escape.png", base_dir=dataset_root)
        self.assertFalse(safe_ok, "Symlink escaping base directory must be rejected")
        self.assertIn("escapes base directory", str(err))

    def test_corrupt_video_manifest_json_fails_with_line_number(self):
        """Verifies that corrupt JSON in video manifests reports the exact manifest path and line number."""
        manifest_path = os.path.join(self.base_dir, "corrupt_video.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("# Video training manifest split=train\n")
            f.write('{bad_json_line_here\n') # Corrupt line 2

        with self.assertRaises(ValueError) as ctx:
            MoonVideoDataset(
                manifest_path=manifest_path,
                base_dir=self.base_dir,
                split="train",
                allow_synthetic=False
            )
        err_msg = str(ctx.exception)
        self.assertIn("JSON decode error", err_msg)
        self.assertIn("line 2", err_msg)
        self.assertIn("corrupt_video.jsonl", err_msg)

    def test_self_declared_ownership_remains_unverified(self):
        """Verifies that self-declared ownership is marked SELF_DECLARED / PENDING_MANUAL and rejected from production."""
        rec = {
            "id": "img_self_declared",
            "media_path": self.img_rel,
            "caption": "A sunset photograph captured on a smartphone camera",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": "owner://declarations/img_self_declared",
            "license": "CustomSovereign",
            "sha256": self.img_sha,
            "is_synthetic": False,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "asset_sha256": self.img_sha,
                "rights_statement": "I certify that I created this image and own all rights."
                # Note: No independent external verification!
            }
        }
        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False, verify_file=True)
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_SELF_DECLARED)
        self.assertEqual(audit["status"], VERIFICATION_STATE_PENDING_MANUAL)
        self.assertFalse(audit["training_eligible"], "Self-declared records must not be eligible for production training")

    def test_external_licence_claims_without_evidence_remain_blocked(self):
        """Verifies that external licence claims lacking verification evidence remain blocked in PENDING_MANUAL."""
        rec = {
            "id": "img_cc0_no_ev",
            "media_path": self.img_rel,
            "caption": "An open source photograph of city architecture skyline",
            "category": "architecture",
            "creator": "Photographer",
            "source_url": "https://example.com/photos/city.jpg",
            "license": "CC0",
            "sha256": self.img_sha,
            "is_synthetic": False
            # Missing license_evidence dictionary!
        }
        audit = validate_provenance_record(rec, modality="image", base_dir=self.base_dir, allow_synthetic=False, verify_file=True)
        self.assertEqual(audit["licence_verification_state"], VERIFICATION_STATE_PENDING_MANUAL)
        self.assertEqual(audit["status"], VERIFICATION_STATE_PENDING_MANUAL)
        self.assertFalse(audit["training_eligible"])

    def test_production_training_rejects_any_unverified_dataset_record(self):
        """Verifies that MoonImageDataset strictly raises ValueError when loading unverified records in production mode."""
        manifest_path = os.path.join(self.base_dir, "image_train.jsonl")
        rec = {
            "id": "img_unverified",
            "media_path": self.img_rel,
            "caption": "An unverified image in production manifest",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": "owner://declarations/img_unverified",
            "license": "CustomSovereign",
            "sha256": self.img_sha,
            "is_synthetic": False,
            "split": "train",
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "asset_sha256": self.img_sha,
                "rights_statement": "I certify that I created this image and own all rights."
            }
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        with self.assertRaises(ValueError) as ctx:
            MoonImageDataset(
                manifest_path=manifest_path,
                base_dir=self.base_dir,
                split="train",
                allow_synthetic=False
            )
        self.assertIn("failed production verification (PENDING_MANUAL_VERIFICATION)", str(ctx.exception))

    def test_genuine_fully_verified_fixture_passes_production_validation(self):
        """Verifies that an authentic, externally verified record passes the production validation path cleanly."""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        test_priv = ed25519.Ed25519PrivateKey.generate()
        test_pub = test_priv.public_key().public_bytes_raw().hex()
        keystore_path = os.path.join(self.base_dir, "test_keystore.json")
        with open(keystore_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "authority": "Test Authority",
                "policy": "ED25519_ASYMMETRIC_ONLY",
                "trusted_verifiers": {
                    "moon_compliance_authority_v1": {
                        "verifier_id": "moon_compliance_authority_v1",
                        "status": "ACTIVE",
                        "algorithm": "Ed25519",
                        "public_key": test_pub,
                        "valid_from": "2026-01-01T00:00:00Z",
                        "valid_until": "2030-01-01T00:00:00Z"
                    }
                }
            }, f)

        att = sign_attestation(
            verifier_id="moon_compliance_authority_v1",
            private_key=test_priv,
            record_id="img_fully_verified",
            asset_sha256=self.img_sha,
            source_url="owner://declarations/img_fully_verified",
            ownership_evidence_reference="LEGAL_EVIDENCE_REF_img_fully_verified",
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp="2026-11-01T00:00:00Z"
        )
        rec = {
            "id": "img_fully_verified",
            "media_path": self.img_rel,
            "caption": "A genuine photograph of snow-capped mountains at dawn",
            "category": "landscapes",
            "creator": "Prem Gaikwad",
            "source_url": "owner://declarations/img_fully_verified",
            "ownership_evidence_reference": "LEGAL_EVIDENCE_REF_img_fully_verified",
            "license": "CustomSovereign",
            "sha256": self.img_sha,
            "is_synthetic": False,
            "split": "train",
            "verification_state": VERIFICATION_STATE_EXTERNALLY_VERIFIED,
            "attestation": att,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": "2026-11-01T00:00:00Z",
                "ownership_evidence_reference": "LEGAL_EVIDENCE_REF_img_fully_verified",
                "asset_sha256": self.img_sha,
                "rights_statement": "I certify this is an original capture created by me with full commercial rights.",
                "verification_state": VERIFICATION_STATE_EXTERNALLY_VERIFIED
            }
        }
        audit = validate_provenance_record(
            rec,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=True,
            verifiers_path=keystore_path
        )
        self.assertTrue(audit["is_valid"], f"Errors: {audit['errors']}")
        self.assertTrue(audit["training_eligible"])
        self.assertEqual(audit["status"], "VERIFIED_PRODUCTION_ELIGIBLE")
        self.assertEqual(audit["ownership_verification_state"], VERIFICATION_STATE_EXTERNALLY_VERIFIED)

    def test_checkpoint_rng_state_pure_primitives_roundtrip(self):
        """
        Verifies that checkpoint save serializes NumPy RNG state as pure Python primitives,
        loads securely under PyTorch weights_only=True without WeightsUnpickler errors,
        and accurately restores reproducible RNG sequences across Python, NumPy, and PyTorch.
        """
        # 1. Establish deterministic seeds
        random.seed(1337)
        np.random.seed(4242)
        torch.manual_seed(9999)

        model = TinyDummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "rng_test_ckpt.pt")

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            epoch=5,
            loss=0.05,
            config_dict={"test": "rng_validation"},
            output_path=ckpt_path,
            training_mode=TRAINING_MODE_DEVELOPMENT
        )

        # 2. Inspect serialized payload on disk under strict weights_only=True
        raw_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        rng = raw_payload.get("rng_state", {})
        self.assertIn("numpy", rng)
        self.assertIn("python", rng)
        self.assertIn("torch_cpu", rng)

        # 3. Assert NumPy state is serialized as pure Python primitives (no ndarray)
        np_state = rng["numpy"]
        self.assertIsInstance(np_state, tuple)
        self.assertEqual(len(np_state), 5)
        self.assertIsInstance(np_state[0], str)
        self.assertIsInstance(np_state[1], list)  # Pure Python list, not np.ndarray
        self.assertTrue(all(isinstance(x, int) for x in np_state[1]))
        self.assertIsInstance(np_state[2], int)
        self.assertIsInstance(np_state[3], int)
        self.assertIsInstance(np_state[4], float)

        # Generate reference outputs from current post-save RNG state
        ref_py = random.random()
        ref_np = np.random.rand(5)
        ref_torch = torch.rand(5)

        # Scramble RNG states completely
        random.seed(999999)
        np.random.seed(888888)
        torch.manual_seed(777777)

        # Restore RNG states from checkpoint via restore_rng_state
        restore_rng_state(rng)

        # Verify exact reproducibility
        restored_py = random.random()
        restored_np = np.random.rand(5)
        restored_torch = torch.rand(5)

        self.assertEqual(ref_py, restored_py)
        self.assertTrue(np.allclose(ref_np, restored_np))
        self.assertTrue(torch.allclose(ref_torch, restored_torch))

    def test_checkpoint_rng_state_legacy_safe_handling(self):
        """Verifies that checkpoints with missing, empty, or None RNG state load safely without errors."""
        model = TinyDummyModel()
        fresh_model = TinyDummyModel()
        ckpt_path = os.path.join(self.base_dir, "checkpoints", "dev", "no_rng_ckpt.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        payload = {
            "provenance": PROVENANCE_TAG,
            "training_mode": TRAINING_MODE_DEVELOPMENT,
            "is_production_eligible": False,
            "synthetic_sample_count": 0,
            "model_state_dict": model.state_dict(),
            "step": 1,
            "epoch": 1,
            "loss": 0.1,
            "rng_state": None  # Missing/None RNG
        }
        torch.save(payload, ckpt_path)

        # Must load cleanly in DEVELOPMENT mode without raising exceptions
        loaded = load_checkpoint(
            checkpoint_path=ckpt_path,
            model=fresh_model,
            mode=TRAINING_MODE_DEVELOPMENT,
            enforce_provenance=True
        )
        self.assertEqual(loaded["step"], 1)

if __name__ == "__main__":
    unittest.main()
