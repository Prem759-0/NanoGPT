"""
tests/test_stage_2_0_7_integrity_hardening.py
Stage 2.0.7 Integrity Hardening Test Suite for Moon AI.

Covers:
1. Database isolation & environment variable leakage protection (SEC-01).
2. Empty and malformed evaluation results validation (SEC-02).
3. Strict metric validation: NaN, Infinity, booleans, out-of-range, and boundaries [0.0, 100.0] (SEC-03).
4. Promotion path containment: directory traversal, absolute paths outside root, symlinks (SEC-04).
5. Restricted Deterministic Canonical JSON documentation & compatibility (SEC-05).
6. Sovereign ownership & synthetic data isolation gates.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""
import os
import sys
import json
import math
import shutil
import tempfile
import unittest
import hashlib
import sqlite3
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from cryptography.hazmat.primitives.asymmetric import ed25519
from PIL import Image

from webapp.database import Database, _DEFAULT_DB_URL
from evaluation.security.evaluation_attestation import (
    sign_evaluation_attestation,
    verify_evaluation_attestation,
    compute_test_results_summary_sha256,
    recompute_aggregate_metrics
)
from datasets.security.attestation_engine import (
    sign_attestation,
    verify_attestation,
    load_trusted_verifiers
)
from datasets.tools.provenance_validator import compute_actual_file_sha256
from training.common.checkpoint import (
    save_checkpoint,
    verify_checkpoint_provenance,
    deep_verify_production_manifest,
    compute_file_sha256,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from scripts.promote_best_model import (
    evaluate_and_promote,
    verify_path_containment,
    safe_read_json_contained,
    safe_compute_file_sha256_contained,
    load_registry
)

class TestStage207IntegrityHardening(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name

        # Generate ephemeral Ed25519 test keys
        self.evaluator_priv = ed25519.Ed25519PrivateKey.generate()
        self.evaluator_pub = self.evaluator_priv.public_key()
        self.evaluator_pub_hex = self.evaluator_pub.public_bytes_raw().hex()

        self.verifier_priv = ed25519.Ed25519PrivateKey.generate()
        self.verifier_pub = self.verifier_priv.public_key()
        self.verifier_pub_hex = self.verifier_pub.public_bytes_raw().hex()

        # Ephemeral keystores
        self.now_utc = datetime.now(timezone.utc)
        self.valid_from_iso = (self.now_utc - timedelta(days=10)).isoformat()
        self.valid_until_iso = (self.now_utc + timedelta(days=365)).isoformat()

        self.evaluators_path = os.path.join(self.base_dir, "trusted_evaluators.json")
        with open(self.evaluators_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "trusted_evaluators": {
                    "eval_hardened_01": {
                        "public_key": self.evaluator_pub_hex,
                        "status": "ACTIVE",
                        "algorithm": "Ed25519",
                        "valid_from": self.valid_from_iso,
                        "valid_until": self.valid_until_iso
                    }
                }
            }, f)

        self.verifiers_path = os.path.join(self.base_dir, "trusted_verifiers.json")
        with open(self.verifiers_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "2.0",
                "trusted_verifiers": {
                    "verifier_hardened_01": {
                        "public_key": self.verifier_pub_hex,
                        "status": "ACTIVE",
                        "algorithm": "Ed25519",
                        "valid_from": self.valid_from_iso,
                        "valid_until": self.valid_until_iso
                    }
                }
            }, f)

        # Baseline genuine asset & manifest
        self.img_rel = "asset.png"
        self.img_abs = os.path.join(self.base_dir, self.img_rel)
        img = Image.new("RGB", (32, 32), color=(80, 120, 160))
        img.save(self.img_abs)
        self.asset_sha256 = compute_actual_file_sha256(self.img_abs)

        from datasets.security.attestation_engine import extract_ownership_evidence_reference

        self.manifest_record = {
            "id": "rec_h01",
            "is_synthetic": False,
            "creator": "Prem Gaikwad",
            "source_url": "owner://prem759-0/moon-data/pilot/h01",
            "license": "CustomSovereign",
            "sha256": self.asset_sha256,
            "media_path": self.img_rel,
            "category": "technology",
            "caption": "Authentic sovereign equipment captured locally by Prem Gaikwad.",
            "split": "train",
            "commercial_use_permitted": True,
            "owner_declaration": {
                "declarant": "Prem Gaikwad",
                "declaration_timestamp": self.now_utc.isoformat(),
                "asset_sha256": self.asset_sha256,
                "rights_statement": "Full commercial sovereign ownership explicitly declared for Moon AI training."
            }
        }
        evidence_ref = extract_ownership_evidence_reference(self.manifest_record)

        self.attestation = sign_attestation(
            verifier_id="verifier_hardened_01",
            private_key=self.verifier_priv,
            record_id=self.manifest_record["id"],
            asset_sha256=self.asset_sha256,
            source_url=self.manifest_record["source_url"],
            ownership_evidence_reference=evidence_ref,
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp=self.now_utc.isoformat()
        )
        self.manifest_record["attestation"] = self.attestation

        self.manifest_rel = "manifest_train.jsonl"
        self.manifest_abs = os.path.join(self.base_dir, self.manifest_rel)
        with open(self.manifest_abs, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.manifest_record) + "\n")
        self.manifest_sha256 = compute_file_sha256(self.manifest_abs)

        # Baseline validation manifest with split="val"
        self.val_record = dict(self.manifest_record)
        self.val_record["id"] = "rec_h01_val"
        self.val_record["split"] = "val"
        val_evidence_ref = extract_ownership_evidence_reference(self.val_record)
        self.val_attestation = sign_attestation(
            verifier_id="verifier_hardened_01",
            private_key=self.verifier_priv,
            record_id=self.val_record["id"],
            asset_sha256=self.asset_sha256,
            source_url=self.val_record["source_url"],
            ownership_evidence_reference=val_evidence_ref,
            license_id="CustomSovereign",
            commercial_use_permitted=True,
            timestamp=self.now_utc.isoformat()
        )
        self.val_record["attestation"] = self.val_attestation
        self.val_manifest_abs = os.path.join(self.base_dir, "manifest_val.jsonl")
        with open(self.val_manifest_abs, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.val_record) + "\n")

        # Baseline production checkpoint
        from model import GPTConfig, GPT
        gpt_conf = GPTConfig(block_size=32, vocab_size=64, n_layer=1, n_head=1, n_embd=32)
        model = GPT(gpt_conf)
        self.ckpt_rel = "checkpoint.pt"
        self.ckpt_abs = os.path.join(self.base_dir, self.ckpt_rel)
        save_checkpoint(
            model=model,
            optimizer=None,
            step=1,
            epoch=1,
            loss=1.5,
            config_dict={},
            output_path=self.ckpt_abs,
            model_name="MoonSovereignModel",
            training_mode=TRAINING_MODE_PRODUCTION,
            manifest_path=self.manifest_abs,
            val_manifest_path=self.val_manifest_abs,
            production_eligible_sample_count=2,
            base_dir=self.base_dir,
            verifiers_path=self.verifiers_path
        )
        self.ckpt_sha256 = compute_file_sha256(self.ckpt_abs)

    def tearDown(self):
        self.temp_dir.cleanup()

    # =========================================================================
    # PART 1: DATABASE ENVIRONMENT ISOLATION (SEC-01)
    # =========================================================================

    def test_01_db_explicit_sqlite_selection_with_none_or_empty(self):
        """Explicit db_url=None or db_url='' forces SQLite even if DATABASE_URL is set in environment."""
        old_env = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = "postgresql://user:pass@remote.host:5432/live_db"
            db_file = os.path.join(self.base_dir, "isolated_sqlite1.db")
            
            db1 = Database(db_path=db_file, db_url=None)
            self.assertFalse(db1.is_postgres, "db_url=None must force local SQLite mode.")
            self.assertIsNone(db1.db_url, "db_url must resolve to None.")
            
            db2 = Database(db_path=db_file, db_url="")
            self.assertFalse(db2.is_postgres, "db_url='' must force local SQLite mode.")
            self.assertIsNone(db2.db_url, "db_url must resolve to None.")
        finally:
            if old_env is not None:
                os.environ["DATABASE_URL"] = old_env
            else:
                os.environ.pop("DATABASE_URL", None)

    def test_02_db_default_behavior_reads_environment(self):
        """When db_url is omitted, default behavior correctly respects DATABASE_URL from environment."""
        old_env = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = "postgresql://user:pass@remote.host:5432/test_env_db"
            # allow_sqlite_fallback=True so __init__ doesn't raise while inspecting connection
            db = Database(db_path=os.path.join(self.base_dir, "test_env.db"), allow_sqlite_fallback=True)
            self.assertEqual(db.db_url, "postgresql://user:pass@remote.host:5432/test_env_db")
        finally:
            if old_env is not None:
                os.environ["DATABASE_URL"] = old_env
            else:
                os.environ.pop("DATABASE_URL", None)

    def test_03_db_failed_postgresql_fails_closed_without_silent_fallback(self):
        """Failed PostgreSQL connection raises RuntimeError when allow_sqlite_fallback=False."""
        invalid_pg_url = "postgresql://invalid_user:invalid_pass@127.0.0.1:54329/nonexistent_db"
        with self.assertRaises(RuntimeError) as ctx:
            Database(
                db_path=os.path.join(self.base_dir, "fail_closed.db"),
                db_url=invalid_pg_url,
                allow_sqlite_fallback=False
            )
        self.assertIn("Silent SQLite fallback is disabled", str(ctx.exception))

    def test_04_db_sqlite_commit_rollback_and_foreign_keys(self):
        """Verified local SQLite transactions commit on success, rollback on error, enforce FKs."""
        db_file = os.path.join(self.base_dir, "tx_test.db")
        db = Database(db_path=db_file, db_url="")

        # Commit test
        sid = db.create_session(title="Hardened Session")
        db.add_message(sid, "user", "Hello SQLite")
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT content FROM messages WHERE session_id = ?", (sid,))
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "Hello SQLite")

        # Foreign key integrity test
        with self.assertRaises(sqlite3.IntegrityError):
            with db._get_connection() as conn:
                conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('missing_id', 'user', 'fail')")

        # Rollback test
        with self.assertRaises(RuntimeError):
            with db._get_connection() as conn:
                conn.execute("INSERT INTO sessions (id, title) VALUES ('rollback_id', 'Rollback Session')")
                raise RuntimeError("Forced rollback exception")

        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT id FROM sessions WHERE id = 'rollback_id'")
        self.assertIsNone(cur.fetchone())
        conn.close()

    # =========================================================================
    # PART 2: EMPTY & MALFORMED EVALUATION RESULTS (SEC-02)
    # =========================================================================

    def test_05_recompute_aggregate_metrics_rejects_empty_and_null(self):
        """recompute_aggregate_metrics raises controlled ValueError on None, empty list, or non-list."""
        with self.assertRaises(ValueError) as ctx1:
            recompute_aggregate_metrics([])
        self.assertIn("test_results collection is empty", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            recompute_aggregate_metrics(None)
        self.assertIn("test_results cannot be None", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            recompute_aggregate_metrics("not_a_list")
        self.assertIn("must be a list", str(ctx3.exception))

    def test_06_verify_evaluation_attestation_rejects_empty_test_results_list(self):
        """verify_evaluation_attestation fails closed when test_results is empty []."""
        report = {
            "model_name": "TestModel",
            "checkpoint_sha256": "a" * 64,
            "composite_quality_score": 85.0,
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0,
            "test_results": [],
            "evaluation_attestation": {
                "algorithm": "Ed25519",
                "evaluator_id": "eval_hardened_01",
                "checkpoint_sha256": "a" * 64,
                "model_name": "TestModel",
                "architecture_version": "moon_arch_v1.0",
                "composite_quality_score": 85.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0,
                "timestamp": self.now_utc.isoformat(),
                "signature": "b" * 128
            }
        }
        valid, err = verify_evaluation_attestation(report, evaluators_path=self.evaluators_path)
        self.assertFalse(valid)
        self.assertIn("must be a non-empty list", err)

    def test_07_promotion_rejects_empty_and_malformed_test_results(self):
        """evaluate_and_promote rejects reports with empty test_results or malformed test_results_path."""
        reg_file = os.path.join(self.base_dir, "registry.json")
        rep_file = os.path.join(self.base_dir, "report_empty_tr.json")

        # Case A: test_results is []
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "MoonSovereignModel",
                "checkpoint_path": self.ckpt_abs,
                "checkpoint_sha256": self.ckpt_sha256,
                "composite_quality_score": 90.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0,
                "test_results": [],
                "test_results_summary_sha256": "c" * 64
            }, f)
        promoted = evaluate_and_promote(
            candidate_report_path=rep_file,
            registry_path=reg_file,
            evaluators_path=self.evaluators_path,
            base_dir=self.base_dir
        )
        self.assertFalse(promoted)

        # Case B: test_results_path points to invalid JSON
        bad_json_path = os.path.join(self.base_dir, "bad.json")
        with open(bad_json_path, "w", encoding="utf-8") as f:
            f.write("NOT_VALID_JSON{")
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "MoonSovereignModel",
                "checkpoint_path": self.ckpt_abs,
                "checkpoint_sha256": self.ckpt_sha256,
                "composite_quality_score": 90.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0,
                "test_results_path": bad_json_path,
                "test_results_summary_sha256": "c" * 64
            }, f)
        promoted_bad_json = evaluate_and_promote(
            candidate_report_path=rep_file,
            registry_path=reg_file,
            evaluators_path=self.evaluators_path,
            base_dir=self.base_dir
        )
        self.assertFalse(promoted_bad_json)

    # =========================================================================
    # PART 3: STRICT METRIC VALIDATION: NAN, INF, BOUNDS (SEC-03)
    # =========================================================================

    def test_08_metric_validation_rejects_nan_and_infinity(self):
        """recompute_aggregate_metrics rejects float and string NaN and +/- Infinity."""
        bad_cases = [
            [{"score": float("nan"), "output": "ok"}],
            [{"score": "NaN", "output": "ok"}],
            [{"score": float("inf"), "output": "ok"}],
            [{"score": "Infinity", "output": "ok"}],
            [{"score": float("-inf"), "output": "ok"}],
            [{"score": "-Infinity", "output": "ok"}]
        ]
        for case in bad_cases:
            with self.assertRaises(ValueError) as ctx:
                recompute_aggregate_metrics(case)
            self.assertIn("non-finite", str(ctx.exception))

    def test_09_metric_validation_rejects_booleans(self):
        """recompute_aggregate_metrics strictly rejects boolean scores (True, False)."""
        with self.assertRaises(ValueError) as ctx1:
            recompute_aggregate_metrics([{"score": True, "output": "ok"}])
        self.assertIn("boolean value", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            recompute_aggregate_metrics([{"score": False, "output": "ok"}])
        self.assertIn("boolean value", str(ctx2.exception))

    def test_10_metric_validation_rejects_non_numeric_strings_and_null(self):
        """recompute_aggregate_metrics rejects arbitrary strings, missing keys, and null scores."""
        with self.assertRaises(ValueError) as ctx1:
            recompute_aggregate_metrics([{"score": "excellent", "output": "ok"}])
        self.assertIn("non-numeric string value", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            recompute_aggregate_metrics([{"score": None, "output": "ok"}])
        self.assertIn("null", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            recompute_aggregate_metrics([{"output": "ok"}])
        self.assertIn("Missing required 'score' field", str(ctx3.exception))

    def test_11_metric_validation_bounds_and_valid_limits(self):
        """recompute_aggregate_metrics rejects < 0.0 or > 100.0, accepts valid 0.0 and 100.0."""
        # Out of bounds
        with self.assertRaises(ValueError) as ctx1:
            recompute_aggregate_metrics([{"score": -0.01, "output": "ok"}])
        self.assertIn("out of permitted range", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            recompute_aggregate_metrics([{"score": 100.01, "output": "ok"}])
        self.assertIn("out of permitted range", str(ctx2.exception))

        # Valid boundaries 0.0 and 100.0
        m_zero = recompute_aggregate_metrics([{"score": 0.0, "output": "ok", "syntax_valid": True}])
        self.assertEqual(m_zero["composite_quality_score"], 0.0)

        m_hundred = recompute_aggregate_metrics([{"score": 100.0, "output": "ok", "syntax_valid": True}])
        self.assertEqual(m_hundred["composite_quality_score"], 100.0)

    # =========================================================================
    # PART 4: PROMOTION PATH CONTAINMENT (SEC-04)
    # =========================================================================

    def test_12_path_containment_rejects_raw_traversal(self):
        """verify_path_containment rejects raw '..' traversal patterns."""
        ok, resolved, err = verify_path_containment("../outside/target.pt", base_dir=self.base_dir)
        self.assertFalse(ok)
        self.assertIn("traversal attempt", err)

        ok2, resolved2, err2 = verify_path_containment("sub/../../escape.pt", base_dir=self.base_dir)
        self.assertFalse(ok2)
        self.assertIn("traversal attempt", err2)

    def test_13_path_containment_rejects_absolute_path_outside_root(self):
        """verify_path_containment rejects absolute paths outside base_dir."""
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = os.path.join(outside_dir, "outside.pt")
            with open(outside_file, "w") as f:
                f.write("OUTSIDE")

            ok, resolved, err = verify_path_containment(outside_file, base_dir=self.base_dir)
            self.assertFalse(ok)
            self.assertTrue("escapes approved directory root" in err or "separate filesystem drive" in err)

    def test_14_path_containment_rejects_symlink_escape(self):
        """verify_path_containment detects and rejects symlinks that point outside base_dir."""
        if sys.platform == "win32":
            # On Windows, creating symlinks may require SeCreateSymbolicLinkPrivilege.
            # Test if symlink creation is permitted; if not, skip this specific test.
            try:
                test_link = os.path.join(self.base_dir, "test_perm.lnk")
                os.symlink(self.manifest_abs, test_link)
                os.unlink(test_link)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation not permitted in this Windows test environment.")

        with tempfile.TemporaryDirectory() as outside_dir:
            outside_target = os.path.join(outside_dir, "secret_outside.pt")
            with open(outside_target, "w") as f:
                f.write("SECRET")
            
            symlink_in_root = os.path.join(self.base_dir, "symlink_escape.pt")
            try:
                os.symlink(outside_target, symlink_in_root)
            except OSError:
                self.skipTest("Symlink creation not permitted.")

            ok, resolved, err = verify_path_containment(symlink_in_root, base_dir=self.base_dir)
            self.assertFalse(ok)
            self.assertTrue("escapes approved directory root" in err or "separate filesystem drive" in err)

    def test_15_path_containment_accepts_valid_in_root_paths(self):
        """verify_path_containment accepts relative and absolute paths contained within base_dir."""
        sub_dir = os.path.join(self.base_dir, "checkpoints")
        os.makedirs(sub_dir, exist_ok=True)
        valid_file = os.path.join(sub_dir, "model.pt")
        with open(valid_file, "w") as f:
            f.write("VALID")

        ok_rel, res_rel, err_rel = verify_path_containment("checkpoints/model.pt", base_dir=self.base_dir)
        self.assertTrue(ok_rel)
        self.assertEqual(res_rel, os.path.realpath(valid_file))

        ok_abs, res_abs, err_abs = verify_path_containment(valid_file, base_dir=self.base_dir)
        self.assertTrue(ok_abs)
        self.assertEqual(res_abs, os.path.realpath(valid_file))

    # =========================================================================
    # PART 5: RESTRICTED CANONICAL JSON COMPATIBILITY & COMPLETE PROMOTION WORKFLOW
    # =========================================================================

    def test_16_end_to_end_hardened_promotion_workflow_succeeds(self):
        """Valid evaluation report with non-empty results and safe paths passes promotion."""
        valid_results = [
            {"score": 92.0, "output": "result 1", "syntax_valid": True},
            {"score": 98.0, "output": "result 2", "syntax_valid": True}
        ]
        test_sha = compute_test_results_summary_sha256(valid_results)
        metrics = recompute_aggregate_metrics(valid_results)
        self.assertEqual(metrics["composite_quality_score"], 95.0)

        att = sign_evaluation_attestation(
            evaluator_id="eval_hardened_01",
            private_key=self.evaluator_priv,
            checkpoint_sha256=self.ckpt_sha256,
            model_name="MoonSovereignModel",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=95.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0,
            dataset_manifest_sha256=self.manifest_sha256,
            test_results_summary_sha256=test_sha,
            timestamp=self.now_utc.isoformat()
        )

        candidate_report = {
            "model_name": "MoonSovereignModel",
            "checkpoint_path": self.ckpt_abs,
            "checkpoint_sha256": self.ckpt_sha256,
            "dataset_manifest_path": self.manifest_abs,
            "dataset_manifest_sha256": self.manifest_sha256,
            "composite_quality_score": 95.0,
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0,
            "test_results": valid_results,
            "test_results_summary_sha256": test_sha,
            "evaluation_attestation": att
        }
        rep_file = os.path.join(self.base_dir, "candidate_valid.json")
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(candidate_report, f)

        reg_file = os.path.join(self.base_dir, "registry.json")
        promoted = evaluate_and_promote(
            candidate_report_path=rep_file,
            registry_path=reg_file,
            evaluators_path=self.evaluators_path,
            verifiers_path=self.verifiers_path,
            base_dir=self.base_dir
        )
        self.assertTrue(promoted, "Valid candidate report must successfully pass promotion.")

        reg = load_registry(reg_file)
        self.assertEqual(reg["current_best_model"]["name"], "MoonSovereignModel")
        self.assertEqual(reg["current_best_model"]["composite_score"], 95.0)

    def test_17_numeric_overflow_large_integers_and_floats(self):
        """Verify recompute_aggregate_metrics produces controlled ValueError on huge ints and overflow floats."""
        overflow_inputs = [
            10**1000,
            -10**1000,
            2**2048,
            -2**2048,
            float("inf"),
            float("-inf"),
            float("nan"),
            -0.0001,
            100.0001,
            10**20,
            -10**20
        ]
        for val in overflow_inputs:
            with self.subTest(val=val):
                with self.assertRaises(ValueError, msg=f"Value {val} must produce a controlled ValueError"):
                    recompute_aggregate_metrics([{"score": val, "output": "ok", "syntax_valid": True}])

    def test_18_numeric_validation_string_overflow_and_boundaries(self):
        """Verify numeric string overflow, invalid strings, booleans, and valid boundary values [0, 100]."""
        invalid_strings = [
            "1e500",
            "-1e500",
            "1" + "0" * 1000,
            "nan",
            "NaN",
            "infinity",
            "-infinity",
            "inf",
            "-inf",
            "+inf",
            "abc",
            "",
            "   ",
            "1.2.3"
        ]
        for s in invalid_strings:
            with self.subTest(string=s):
                with self.assertRaises(ValueError, msg=f"String '{s}' must produce a controlled ValueError"):
                    recompute_aggregate_metrics([{"score": s, "output": "ok", "syntax_valid": True}])

        # Booleans must raise ValueError
        with self.assertRaises(ValueError):
            recompute_aggregate_metrics([{"score": True, "output": "ok", "syntax_valid": True}])
        with self.assertRaises(ValueError):
            recompute_aggregate_metrics([{"score": False, "output": "ok", "syntax_valid": True}])

        # Valid boundaries 0 and 100 must succeed
        res_0 = recompute_aggregate_metrics([{"score": 0, "output": "ok", "syntax_valid": True}])
        self.assertEqual(res_0["composite_quality_score"], 0.0)

        res_100 = recompute_aggregate_metrics([{"score": 100, "output": "ok", "syntax_valid": True}])
        self.assertEqual(res_100["composite_quality_score"], 100.0)

        res_0_f = recompute_aggregate_metrics([{"score": 0.0, "output": "ok", "syntax_valid": True}])
        self.assertEqual(res_0_f["composite_quality_score"], 0.0)

        res_100_f = recompute_aggregate_metrics([{"score": 100.0, "output": "ok", "syntax_valid": True}])
        self.assertEqual(res_100_f["composite_quality_score"], 100.0)

        res_str = recompute_aggregate_metrics([{"score": "95.5", "output": "ok", "syntax_valid": True}])
        self.assertEqual(res_str["composite_quality_score"], 95.5)

    def test_19_database_url_scheme_validation_and_rejection(self):
        """Verify database URL scheme validation: unsupported schemes rejected, sqlite:// and postgresql:// accepted."""
        unsupported_schemes = [
            "mysql://user:pass@localhost:3306/db",
            "ftp://files.internal/db",
            "http://malicious.com/db",
            "https://malicious.com/db",
            "mongodb://localhost:27017/db",
            "redis://127.0.0.1:6379/0",
            "oracle://user:pass@host/xe",
            "random_string_not_a_url",
            "custom://scheme/path"
        ]
        for url in unsupported_schemes:
            with self.subTest(url=url):
                with self.assertRaises(ValueError, msg=f"Scheme in '{url}' must be rejected with ValueError"):
                    Database(db_path=os.path.join(self.base_dir, "test.db"), db_url=url)

        # sqlite:// accepted
        sqlite_custom = os.path.join(self.base_dir, "custom_sqlite.db")
        if os.name == "nt":
            test_sqlite_url = f"sqlite:///{sqlite_custom}"
        else:
            test_sqlite_url = f"sqlite:////{sqlite_custom.lstrip('/')}"
        db_sqlite = Database(db_url=test_sqlite_url)
        self.assertFalse(db_sqlite.is_postgres)
        self.assertEqual(os.path.realpath(db_sqlite.db_path), os.path.realpath(sqlite_custom))

        # db_url=None and db_url="" select SQLite
        db_none = Database(db_path=os.path.join(self.base_dir, "test_none.db"), db_url=None)
        self.assertFalse(db_none.is_postgres)
        self.assertIsNone(db_none.db_url)

        db_empty = Database(db_path=os.path.join(self.base_dir, "test_empty.db"), db_url="")
        self.assertFalse(db_empty.is_postgres)
        self.assertIsNone(db_empty.db_url)

    def test_20_safe_contained_file_reading_and_toctou_mitigation(self):
        """Verify safe_read_json_contained and safe_compute_file_sha256_contained enforce path & symlink containment."""
        valid_json_path = os.path.join(self.base_dir, "valid_data.json")
        sample_data = {"key": "value", "score": 90.0}
        with open(valid_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        ok, data, err = safe_read_json_contained(valid_json_path, base_dir=self.base_dir)
        self.assertTrue(ok)
        self.assertEqual(data["key"], "value")
        self.assertIsNone(err)

        ok_h, hash_val, err_h = safe_compute_file_sha256_contained(valid_json_path, base_dir=self.base_dir)
        self.assertTrue(ok_h)
        self.assertEqual(len(hash_val), 64)
        self.assertIsNone(err_h)

        # Reject path outside base_dir
        outside_dir = tempfile.mkdtemp()
        outside_path = os.path.join(outside_dir, "outside.json")
        with open(outside_path, "w", encoding="utf-8") as f:
            json.dump({"leak": True}, f)
        try:
            ok_out, _, err_out = safe_read_json_contained(outside_path, base_dir=self.base_dir)
            self.assertFalse(ok_out)
            self.assertTrue(
                "escapes approved directory root" in err_out or "separate filesystem drive" in err_out,
                f"Unexpected error reason: {err_out}"
            )
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
