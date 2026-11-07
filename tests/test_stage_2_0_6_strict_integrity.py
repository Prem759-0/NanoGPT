"""
tests/test_stage_2_0_6_strict_integrity.py
Stage 2.0.6 — Strict Attestation Policy, Evaluation Reproducibility & CI Repair Test Suite.

Lead ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

22 Explicit Adversarial & Cryptographic Scenarios:
1.  Verifier missing valid_from rejected.
2.  Verifier missing valid_until rejected.
3.  Timezone-naive validity timestamp rejected.
4.  Future validity timestamp (not yet valid) rejected.
5.  Inverted validity window (valid_from >= valid_until) rejected.
6.  Attestation with version != "2.0" rejected.
7.  Attestation with scope != "PRODUCTION_DATASET" rejected.
8.  Attestation with non-boolean commercial_use_permitted (e.g. "true", 1, "false") rejected.
9.  Attestation with malformed signature length (!= 128 hex chars) rejected.
10. Attestation with malformed asset SHA-256 (!= 64 hex chars) rejected.
11. Attestation older than maximum age (> 730 days) rejected.
12. Attestation with future timestamp beyond 60s rejected.
13. Verifier marked REVOKED rejected.
14. Verifier marked SUSPENDED rejected.
15. Expired verifier certificate rejected.
16. Canonical JSON serialization deterministic across platforms.
17. Evaluation attestation with placeholder dataset_manifest_sha256 ('none') rejected in promotion.
18. Evaluation attestation with placeholder test_results_summary_sha256 ('none') rejected in promotion.
19. Evaluation report with tampered aggregate metrics vs individual test results rejected.
20. Checkpoint save with duplicate record IDs in manifest rejected.
21. Checkpoint save with non-dict record in manifest rejected.
22. Full promotion workflow with valid independent evaluation attestation succeeds in test environment.
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

from datasets.security.attestation_engine import (
    sign_attestation,
    verify_attestation,
    construct_canonical_attestation_payload,
    load_trusted_verifiers,
    extract_ownership_evidence_reference
)
from datasets.tools.provenance_validator import (
    validate_provenance_record,
    compute_actual_file_sha256
)
from evaluation.security.evaluation_attestation import (
    sign_evaluation_attestation,
    verify_evaluation_attestation,
    compute_test_results_summary_sha256,
    recompute_aggregate_metrics
)
from training.common.checkpoint import (
    save_checkpoint,
    verify_checkpoint_provenance,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from scripts.promote_best_model import evaluate_and_promote, load_registry

class TestStage206StrictIntegrity(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name

        # Generate Ed25519 test keys
        self.verifier_priv = ed25519.Ed25519PrivateKey.generate()
        self.verifier_pub = self.verifier_priv.public_key()
        self.verifier_pub_hex = self.verifier_pub.public_bytes_raw().hex()

        self.evaluator_priv = ed25519.Ed25519PrivateKey.generate()
        self.evaluator_pub = self.evaluator_priv.public_key()
        self.evaluator_pub_hex = self.evaluator_pub.public_bytes_raw().hex()

        # Generate genuine test image
        self.img_rel_path = "test_asset.png"
        self.img_abs_path = os.path.join(self.base_dir, self.img_rel_path)
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(self.img_abs_path)
        self.asset_sha256 = compute_actual_file_sha256(self.img_abs_path)

        # Baseline verifier validity window
        self.now_utc = datetime.now(timezone.utc)
        self.valid_from_iso = (self.now_utc - timedelta(days=10)).isoformat()
        self.valid_until_iso = (self.now_utc + timedelta(days=365)).isoformat()

        # Standard trusted verifiers keystore
        self.verifiers_path = os.path.join(self.base_dir, "trusted_verifiers.json")
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Independent Auditor Alpha",
                "organization": "Sovereign AI Verification Body",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": self.valid_from_iso,
                "valid_until": self.valid_until_iso,
                "verification_scope": "PRODUCTION_DATASET"
            }
        })

        # Standard trusted evaluators keystore
        self.evaluators_path = os.path.join(self.base_dir, "trusted_evaluators.json")
        self.update_evaluators_keystore({
            "test_evaluator_1": {
                "name": "Autonomous Model Benchmark Lab",
                "organization": "Independent AI Evaluation Council",
                "algorithm": "Ed25519",
                "public_key": self.evaluator_pub_hex,
                "status": "ACTIVE",
                "valid_from": self.valid_from_iso,
                "valid_until": self.valid_until_iso,
                "evaluation_scope": "MODEL_PROMOTION"
            }
        })

        # Baseline valid record
        self.record = {
            "id": "img_pilot_001",
            "is_synthetic": False,
            "creator": "Prem Gaikwad",
            "source_url": "owner://prem759-0/moon-data/pilot/img_001",
            "license": "CustomSovereign",
            "sha256": self.asset_sha256,
            "media_path": self.img_rel_path,
            "category": "technology",
            "caption": "Authentic sovereign laboratory equipment captured locally by Prem Gaikwad.",
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": self.now_utc.isoformat(),
                "asset_sha256": self.asset_sha256,
                "rights_statement": "Full commercial sovereign ownership explicitly declared for Moon AI training."
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def update_verifiers_keystore(self, verifiers_dict):
        with open(self.verifiers_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "trusted_verifiers": verifiers_dict
            }, f, indent=2)

    def update_evaluators_keystore(self, evaluators_dict):
        with open(self.evaluators_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "trusted_evaluators": evaluators_dict
            }, f, indent=2)

    def generate_signed_attestation(self, **kwargs):
        rec = kwargs.get("record", self.record)
        evidence_ref = kwargs.get("evidence_ref") or extract_ownership_evidence_reference(rec)
        rec_id = kwargs.get("record_id", rec["id"])
        source_url = kwargs.get("source_url", rec["source_url"])
        lic_id = kwargs.get("license_id", rec["license"])
        return sign_attestation(
            verifier_id=kwargs.get("verifier_id", "test_verifier_1"),
            private_key=kwargs.get("private_key", self.verifier_priv),
            record_id=rec_id,
            asset_sha256=kwargs.get("asset_sha256", self.asset_sha256),
            source_url=source_url,
            ownership_evidence_reference=evidence_ref,
            license_id=lic_id,
            commercial_use_permitted=kwargs.get("commercial_use_permitted", True),
            timestamp=kwargs.get("timestamp", self.now_utc.isoformat()),
            attestation_version=kwargs.get("attestation_version", "2.0"),
            verification_scope=kwargs.get("verification_scope", "PRODUCTION_DATASET")
        )

    # -------------------------------------------------------------------------
    # SCENARIOS 1-5: VERIFIER VALIDITY WINDOW INTEGRITY
    # -------------------------------------------------------------------------
    def test_01_verifier_missing_valid_from_rejected(self):
        """Scenario 1: Verifier entry missing valid_from is rejected fail-closed."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_until": self.valid_until_iso
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("missing required 'valid_from'", err.lower())

    def test_02_verifier_missing_valid_until_rejected(self):
        """Scenario 2: Verifier entry missing valid_until is rejected fail-closed."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": self.valid_from_iso
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("missing required 'valid_until'", err.lower())

    def test_03_timezone_naive_validity_timestamp_rejected(self):
        """Scenario 3: Timezone-naive validity timestamps in keystore are rejected."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": "2026-01-01T00:00:00",  # Naive, no Z or +00:00
                "valid_until": self.valid_until_iso
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("lacks explicit utc timezone", err.lower())

    def test_04_future_validity_timestamp_rejected(self):
        """Scenario 4: Verifier whose valid_from is in the future is not yet valid and rejected."""
        future_from = (self.now_utc + timedelta(days=10)).isoformat()
        future_until = (self.now_utc + timedelta(days=365)).isoformat()
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": future_from,
                "valid_until": future_until
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("not yet valid", err.lower())

    def test_05_inverted_validity_window_rejected(self):
        """Scenario 5: Inverted validity window (valid_from >= valid_until) is rejected."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": (self.now_utc + timedelta(days=10)).isoformat(),
                "valid_until": (self.now_utc - timedelta(days=10)).isoformat()
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("inverted validity window", err.lower())

    # -------------------------------------------------------------------------
    # SCENARIOS 6-12: ATTESTATION ENVELOPE & STRICT TYPING
    # -------------------------------------------------------------------------
    def test_06_attestation_version_unsupported_rejected(self):
        """Scenario 6: Attestation version != '2.0' is rejected."""
        att = self.generate_signed_attestation(attestation_version="1.0")
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("unsupported", err.lower())

    def test_07_attestation_scope_unsupported_rejected(self):
        """Scenario 7: Attestation scope != 'PRODUCTION_DATASET' is rejected."""
        att = self.generate_signed_attestation(verification_scope="DEVELOPMENT_SYNTHETIC")
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("unsupported for production", err.lower())

    def test_08_non_boolean_commercial_use_permitted_rejected(self):
        """Scenario 8: Non-boolean commercial_use_permitted ('true', 1, 'false') is strictly rejected."""
        # Case A: String 'true'
        att_str = self.generate_signed_attestation()
        att_str["commercial_use_permitted"] = "true"
        ok, state, err = verify_attestation(att_str, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("must be a strict json boolean", err.lower())

        # Case B: Integer 1 (in Python isinstance(1, int) is True, isinstance(True, int) is True)
        att_int = self.generate_signed_attestation()
        att_int["commercial_use_permitted"] = 1
        ok, state, err = verify_attestation(att_int, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("must be a strict json boolean", err.lower())

    def test_09_malformed_signature_length_rejected(self):
        """Scenario 9: Malformed signature length (!= 128 hex chars) is rejected."""
        att = self.generate_signed_attestation()
        att["signature"] = att["signature"][:64]  # Truncate from 128 to 64
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("invalid length", err.lower())

    def test_10_malformed_asset_sha256_rejected(self):
        """Scenario 10: Malformed asset SHA-256 (!= 64 hex chars) is rejected."""
        att = self.generate_signed_attestation()
        att["asset_sha256"] = "bad_sha_12345"
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("malformed asset_sha256", err.lower())

    def test_11_attestation_older_than_max_age_rejected(self):
        """Scenario 11: Attestation older than maximum age (> 730 days) is expired and rejected."""
        old_ts = (self.now_utc - timedelta(days=735)).isoformat()
        att = self.generate_signed_attestation(timestamp=old_ts)
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("expired", err.lower())

    def test_12_future_timestamp_beyond_clock_skew_rejected(self):
        """Scenario 12: Attestation timestamp in future beyond 60s is rejected."""
        future_ts = (self.now_utc + timedelta(seconds=120)).isoformat()
        att = self.generate_signed_attestation(timestamp=future_ts)
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("is in the future", err.lower())

    # -------------------------------------------------------------------------
    # SCENARIOS 13-16: VERIFIER LIFECYCLE & CANONICAL FORMAT
    # -------------------------------------------------------------------------
    def test_13_verifier_marked_revoked_rejected(self):
        """Scenario 13: Verifier marked REVOKED in keystore is rejected."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "REVOKED",
                "valid_from": self.valid_from_iso,
                "valid_until": self.valid_until_iso
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("revoked", err.lower())

    def test_14_verifier_marked_suspended_rejected(self):
        """Scenario 14: Verifier marked SUSPENDED in keystore is rejected."""
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "SUSPENDED",
                "valid_from": self.valid_from_iso,
                "valid_until": self.valid_until_iso
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("suspended", err.lower())

    def test_15_expired_verifier_certificate_rejected(self):
        """Scenario 15: Verifier whose valid_until is in the past is rejected."""
        past_from = (self.now_utc - timedelta(days=400)).isoformat()
        past_until = (self.now_utc - timedelta(days=30)).isoformat()
        self.update_verifiers_keystore({
            "test_verifier_1": {
                "name": "Auditor",
                "algorithm": "Ed25519",
                "public_key": self.verifier_pub_hex,
                "status": "ACTIVE",
                "valid_from": past_from,
                "valid_until": past_until
            }
        })
        att = self.generate_signed_attestation()
        ok, state, err = verify_attestation(att, self.record, self.asset_sha256, self.verifiers_path)
        self.assertFalse(ok)
        self.assertEqual(state, "REJECTED")
        self.assertIn("expired", err.lower())

    def test_16_canonical_json_serialization_deterministic(self):
        """Scenario 16: Canonical JSON serialization is strictly deterministic."""
        payload1 = construct_canonical_attestation_payload(
            verifier_id="v1", record_id="r1", asset_sha256=self.asset_sha256,
            source_url="owner://test", ownership_evidence_reference="rights",
            license_id="CustomSovereign", commercial_use_permitted=True,
            timestamp=self.now_utc.isoformat()
        )
        payload2 = construct_canonical_attestation_payload(
            verifier_id="v1", record_id="r1", asset_sha256=self.asset_sha256,
            source_url="owner://test", ownership_evidence_reference="rights",
            license_id="CustomSovereign", commercial_use_permitted=True,
            timestamp=self.now_utc.isoformat()
        )
        self.assertEqual(payload1, payload2)
        # Check no whitespace separators
        self.assertNotIn(b": ", payload1)
        self.assertNotIn(b", ", payload1)

    # -------------------------------------------------------------------------
    # SCENARIOS 17-19: EVALUATION REPRODUCIBILITY & ARTIFACT TAMPERING
    # -------------------------------------------------------------------------
    def test_17_promotion_rejects_placeholder_dataset_manifest_hash(self):
        """Scenario 17: Production promotion rejects placeholder dataset_manifest_sha256 ('none')."""
        candidate_ckpt = os.path.join(self.base_dir, "cand_ckpt.pt")
        torch.save({"provenance": "moon_ai_random_init", "training_mode": TRAINING_MODE_PRODUCTION}, candidate_ckpt)
        cand_ckpt_sha = compute_actual_file_sha256(candidate_ckpt)

        report = {
            "model_name": "moon-promoted-test",
            "checkpoint_path": candidate_ckpt,
            "checkpoint_sha256": cand_ckpt_sha,
            "dataset_manifest_sha256": "none",  # Placeholder!
            "test_results_summary_sha256": "a" * 64,
            "composite_quality_score": 90.0,
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0
        }
        report["evaluation_attestation"] = sign_evaluation_attestation(
            evaluator_id="test_evaluator_1",
            private_key=self.evaluator_priv,
            checkpoint_sha256=cand_ckpt_sha,
            model_name="moon-promoted-test",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=90.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            dataset_manifest_sha256="none",
            test_results_summary_sha256="a" * 64
        )
        report_path = os.path.join(self.base_dir, "cand_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        res = evaluate_and_promote(
            candidate_report_path=report_path,
            evaluators_path=self.evaluators_path,
            verifiers_path=self.verifiers_path,
            registry_path=os.path.join(self.base_dir, "test_reg.json")
        )
        self.assertFalse(res)

    def test_18_promotion_rejects_placeholder_test_results_hash(self):
        """Scenario 18: Production promotion rejects placeholder test_results_summary_sha256 ('none')."""
        candidate_ckpt = os.path.join(self.base_dir, "cand_ckpt.pt")
        torch.save({"provenance": "moon_ai_random_init", "training_mode": TRAINING_MODE_PRODUCTION}, candidate_ckpt)
        cand_ckpt_sha = compute_actual_file_sha256(candidate_ckpt)

        report = {
            "model_name": "moon-promoted-test",
            "checkpoint_path": candidate_ckpt,
            "checkpoint_sha256": cand_ckpt_sha,
            "dataset_manifest_sha256": "b" * 64,
            "test_results_summary_sha256": "none",  # Placeholder!
            "composite_quality_score": 90.0,
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0
        }
        report["evaluation_attestation"] = sign_evaluation_attestation(
            evaluator_id="test_evaluator_1",
            private_key=self.evaluator_priv,
            checkpoint_sha256=cand_ckpt_sha,
            model_name="moon-promoted-test",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=90.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            dataset_manifest_sha256="b" * 64,
            test_results_summary_sha256="none"
        )
        report_path = os.path.join(self.base_dir, "cand_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        res = evaluate_and_promote(
            candidate_report_path=report_path,
            evaluators_path=self.evaluators_path,
            verifiers_path=self.verifiers_path,
            registry_path=os.path.join(self.base_dir, "test_reg.json")
        )
        self.assertFalse(res)

    def test_19_tampered_aggregate_metrics_vs_individual_results_rejected(self):
        """Scenario 19: Candidate report with inflated aggregate score vs individual results rejected."""
        # Individual results have score 50.0 and 60.0 -> average is 55.0
        test_results = [
            {"test_id": "t1", "output": "print(1)", "syntax_valid": True, "score": 50.0},
            {"test_id": "t2", "output": "print(2)", "syntax_valid": True, "score": 60.0}
        ]
        test_sha = compute_test_results_summary_sha256(test_results)

        # But report claims 99.0 composite_quality_score!
        report = {
            "model_name": "moon-promoted-test",
            "checkpoint_sha256": "c" * 64,
            "dataset_manifest_sha256": "d" * 64,
            "test_results_summary_sha256": test_sha,
            "composite_quality_score": 99.0,  # Tampered inflated score!
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0,
            "test_results": test_results
        }
        report["evaluation_attestation"] = sign_evaluation_attestation(
            evaluator_id="test_evaluator_1",
            private_key=self.evaluator_priv,
            checkpoint_sha256="c" * 64,
            model_name="moon-promoted-test",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=99.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            dataset_manifest_sha256="d" * 64,
            test_results_summary_sha256=test_sha
        )
        ok, err = verify_evaluation_attestation(report, self.evaluators_path, enforce_production_hashes=True)
        self.assertFalse(ok)
        self.assertIn("does not match individual test results", err.lower())

    # -------------------------------------------------------------------------
    # SCENARIOS 20-21: CHECKPOINT SAVE MANIFEST INTEGRITY
    # -------------------------------------------------------------------------
    def test_20_checkpoint_save_rejects_duplicate_record_ids_in_manifest(self):
        """Scenario 20: Checkpoint save in PRODUCTION_GENUINE rejects manifest with duplicate IDs."""
        rec1 = dict(self.record, split="train")
        rec1["attestation"] = self.generate_signed_attestation(record=rec1)

        rec2 = dict(self.record, split="train")
        rec2["attestation"] = self.generate_signed_attestation(record=rec2)

        train_manifest = os.path.join(self.base_dir, "train_dup.jsonl")
        val_manifest = os.path.join(self.base_dir, "val.jsonl")

        with open(train_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")

        with open(val_manifest, "w", encoding="utf-8") as f:
            val_rec = dict(self.record, id="img_val_001", split="val")
            val_rec["attestation"] = self.generate_signed_attestation(record=val_rec, record_id="img_val_001")
            f.write(json.dumps(val_rec) + "\n")

        model = torch.nn.Linear(10, 2)
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_model.pt")

        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=10,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                manifest_path=train_manifest,
                val_manifest_path=val_manifest,
                training_mode=TRAINING_MODE_PRODUCTION,
                production_eligible_sample_count=2,
                base_dir=self.base_dir,
                verifiers_path=self.verifiers_path
            )
        self.assertIn("duplicate record id", str(ctx.exception).lower())

    def test_21_checkpoint_save_rejects_non_dict_record_in_manifest(self):
        """Scenario 21: Checkpoint save in PRODUCTION_GENUINE rejects manifest with non-dict records."""
        train_manifest = os.path.join(self.base_dir, "train_nondict.jsonl")
        val_manifest = os.path.join(self.base_dir, "val.jsonl")

        with open(train_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps(["not", "a", "dict"]) + "\n")

        with open(val_manifest, "w", encoding="utf-8") as f:
            val_rec = dict(self.record, id="img_val_001", split="val")
            val_att = self.generate_signed_attestation(record_id="img_val_001")
            val_rec["attestation"] = val_att
            f.write(json.dumps(val_rec) + "\n")

        model = torch.nn.Linear(10, 2)
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "moon_model.pt")

        with self.assertRaises(ValueError) as ctx:
            save_checkpoint(
                model=model,
                optimizer=None,
                step=10,
                epoch=1,
                loss=0.5,
                config_dict={},
                output_path=out_ckpt,
                manifest_path=train_manifest,
                val_manifest_path=val_manifest,
                training_mode=TRAINING_MODE_PRODUCTION,
                production_eligible_sample_count=1,
                base_dir=self.base_dir,
                verifiers_path=self.verifiers_path
            )
        self.assertIn("not a json object", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # SCENARIO 22: FULL END-TO-END PROMOTION WORKFLOW
    # -------------------------------------------------------------------------
    def test_22_full_promotion_workflow_with_valid_attestation_succeeds(self):
        """Scenario 22: Full promotion succeeds when independent evaluation attestation is genuine."""
        # 1. Create genuine production manifests
        rec_train = dict(self.record, id="img_tr_001", split="train")
        att_train = self.generate_signed_attestation(record=rec_train, record_id="img_tr_001")
        rec_train["attestation"] = att_train

        rec_val = dict(self.record, id="img_val_001", split="val")
        att_val = self.generate_signed_attestation(record=rec_val, record_id="img_val_001")
        rec_val["attestation"] = att_val

        train_m = os.path.join(self.base_dir, "train_p22.jsonl")
        val_m = os.path.join(self.base_dir, "val_p22.jsonl")
        with open(train_m, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_train) + "\n")
        with open(val_m, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_val) + "\n")

        # 2. Save verifiable production checkpoint
        model = torch.nn.Linear(10, 2)
        out_ckpt = os.path.join(self.base_dir, "checkpoints", "best_moon_model.pt")
        save_checkpoint(
            model=model,
            optimizer=None,
            step=100,
            epoch=5,
            loss=0.2,
            config_dict={},
            output_path=out_ckpt,
            manifest_path=train_m,
            val_manifest_path=val_m,
            model_name="moon-candidate-sovereign",
            training_mode=TRAINING_MODE_PRODUCTION,
            production_eligible_sample_count=2,
            base_dir=self.base_dir,
            verifiers_path=self.verifiers_path
        )
        ckpt_sha = compute_actual_file_sha256(out_ckpt)
        val_m_sha = compute_actual_file_sha256(val_m)

        # 3. Create genuine individual test results and compute summary hash
        test_results = [
            {"test_id": "t1", "output": "print('hello')", "syntax_valid": True, "score": 95.0},
            {"test_id": "t2", "output": "x = 42", "syntax_valid": True, "score": 97.0}
        ]
        test_summary_sha = compute_test_results_summary_sha256(test_results)
        metrics = recompute_aggregate_metrics(test_results)

        # 4. Independent evaluator signs evaluation attestation
        eval_att = sign_evaluation_attestation(
            evaluator_id="test_evaluator_1",
            private_key=self.evaluator_priv,
            checkpoint_sha256=ckpt_sha,
            model_name="moon-candidate-sovereign",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=metrics["composite_quality_score"],
            non_empty_rate_pct=metrics["non_empty_rate_pct"],
            code_syntax_pct=metrics["code_syntax_pct"],
            dataset_manifest_sha256=val_m_sha,
            test_results_summary_sha256=test_summary_sha
        )

        report = {
            "model_name": "moon-candidate-sovereign",
            "checkpoint_path": out_ckpt,
            "checkpoint_sha256": ckpt_sha,
            "dataset_manifest_path": val_m,
            "dataset_manifest_sha256": val_m_sha,
            "test_results_summary_sha256": test_summary_sha,
            "composite_quality_score": metrics["composite_quality_score"],
            "non_empty_rate_pct": metrics["non_empty_rate_pct"],
            "code_syntax_pct": metrics["code_syntax_pct"],
            "test_results": test_results,
            "evaluation_attestation": eval_att
        }
        report_path = os.path.join(self.base_dir, "valid_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        reg_path = os.path.join(self.base_dir, "test_registry.json")
        promoted = evaluate_and_promote(
            candidate_report_path=report_path,
            min_score_threshold=60.0,
            registry_path=reg_path,
            evaluators_path=self.evaluators_path,
            verifiers_path=self.verifiers_path,
            base_dir=self.base_dir
        )
        self.assertTrue(promoted)
        reg = load_registry(reg_path)
        self.assertEqual(reg["current_best_model"]["name"], "moon-candidate-sovereign")

if __name__ == "__main__":
    unittest.main()
