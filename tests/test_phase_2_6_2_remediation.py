"""
Moon AI — Phase 2.6.2 Targeted Security & Governance Remediation Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates all 8 Security Remediations (SEC-01 through SEC-08):
- SEC-01: Untrusted code execution disabled; AST check + HUMAN_REVIEW_REQUIRED.
- SEC-02: Durable SQLite consent and revocation persistence across restarts.
- SEC-03: Lifecycle ingestion requires valid consent; rejects synthetic auto_consent IDs.
- SEC-04: Legacy training quarantine for user_learned_data.jsonl; /api/self-train 403.
- SEC-05: AUTHORIZED_BY_PREM = False enforced fail-closed in train.py.
- SEC-06: WAL SHA-256 corruption detection and thread safety.
- SEC-07: Consent revocation propagation to candidate WAL (STATE_WITHDRAWN).
- SEC-08: Caller identity authentication validation.
"""

import os
import sys
import json
import sqlite3
import tempfile
import shutil
import unittest
from unittest.mock import patch
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force isolated local SQLite in tests to ensure offline hermetic test execution
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from core.policy import AUTHORIZED_BY_PREM, get_active_policy, POLICY_STRICT_SOVEREIGN_RANDOM_INIT
from core.learning_lifecycle.donation import (
    DonationConsentManager,
    is_authenticated_user,
    DONATION_POLICY_VERSION
)
from core.learning_lifecycle.verifier import (
    ObjectiveVerifier,
    SafeMathEvaluator,
    VERIFIED_CODE,
    VERIFIED_MATH,
    HUMAN_REVIEW_REQUIRED,
    REJECTED
)
from core.learning_lifecycle.candidate_store import (
    CandidateWALStore,
    STATE_OBJECTIVE_VERIFICATION,
    STATE_PENDING_HUMAN_REVIEW,
    STATE_APPROVED_CANDIDATE,
    STATE_WITHDRAWN
)
from core.learning_lifecycle.lifecycle import ControlledLearningLifecycle
from sandbox.code_sandbox import SovereignCodeSandbox


class TestSEC01UntrustedCodeExecutionDisabled(unittest.TestCase):
    """SEC-01: Verify untrusted code execution is disabled pending OS-level isolation."""

    def setUp(self):
        self.verifier = ObjectiveVerifier(sandbox_timeout=3.0)

    def test_verify_code_never_executes_untrusted_subprocess(self):
        """Code is parsed with AST, but subprocess execution is halted closed."""
        code = "result = 40 + 2\n"
        ok, reason, details = self.verifier.verify_code(code)
        self.assertFalse(ok)
        self.assertEqual(reason, "UNTRUSTED_CODE_EXECUTION_DISABLED")
        self.assertTrue(details["syntax_valid"])
        self.assertFalse(details["sandbox_executed"])
        self.assertEqual(details["security_status"], "UNTRUSTED_CODE_EXECUTION_DISABLED")

    def test_ast_syntax_error_caught(self):
        """Invalid syntax is still caught and rejected immediately."""
        bad_code = "def incomplete(:\n    pass"
        ok, reason, details = self.verifier.verify_code(bad_code)
        self.assertFalse(ok)
        self.assertEqual(reason, "SYNTAX_ERROR")
        self.assertFalse(details["syntax_valid"])

    def test_ast_security_violations_caught(self):
        """Forbidden modules like socket/subprocess are caught during AST screening."""
        forbidden_code = "import socket\ns = socket.socket()"
        ok, reason, details = self.verifier.verify_code(forbidden_code)
        self.assertFalse(ok)
        self.assertIn("SECURITY_BLOCK", reason)

    def test_classify_and_verify_never_returns_verified_code(self):
        """Code block in response routes to HUMAN_REVIEW_REQUIRED, NEVER VERIFIED_CODE."""
        prompt = "Write a function that doubles a number"
        response = "Here is the code:\n```python\ndef double(x):\n    return x * 2\n```"
        classification, reason, meta = self.verifier.classify_and_verify(prompt, response)
        self.assertEqual(classification, HUMAN_REVIEW_REQUIRED)
        self.assertNotEqual(classification, VERIFIED_CODE)
        self.assertIn("HUMAN_REVIEW_REQUIRED", classification)

    def test_sovereign_sandbox_gate_blocks_untrusted_flag(self):
        """SovereignCodeSandbox blocks execution if is_untrusted=True and ALLOW_UNTRUSTED_EXECUTION=False."""
        sandbox = SovereignCodeSandbox()
        self.assertFalse(sandbox.ALLOW_UNTRUSTED_EXECUTION)
        res = sandbox.execute("print('hello')", is_untrusted=True)
        self.assertFalse(res["success"])
        self.assertEqual(res["exit_code"], -4)
        self.assertIn("REJECTED_UNTRUSTED_EXECUTION_DISABLED", res["ast_security"])


class TestSEC02DurableConsentPersistence(unittest.TestCase):
    """SEC-02: Verify durable SQLite consent and revocation tracking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_consent.sqlite")
        self.patch_auth = patch("core.learning_lifecycle.donation.SERVER_AUTHENTICATION_VERIFIED", True)
        self.patch_ingest = patch("core.learning_lifecycle.donation.DONATION_INGESTION_ENABLED", True)
        self.patch_auth.start()
        self.patch_ingest.start()
        self.manager = DonationConsentManager(storage_path=self.db_path)

    def tearDown(self):
        self.patch_ingest.stop()
        self.patch_auth.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_consent_persists_to_sqlite(self):
        """Consent written by manager can be queried from SQLite directly."""
        ok, msg, rec = self.manager.create_donation_consent(
            user_id="researcher_prem",
            interaction_id="turn_001",
            affirmative_ack=True
        )
        self.assertTrue(ok)
        cid = rec["consent_id"]

        # Query raw SQLite file
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT consent_id, user_id, status FROM donation_consents WHERE consent_id = ?", (cid,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], cid)
        self.assertEqual(row[1], "researcher_prem")
        self.assertEqual(row[2], "ACTIVE")

    def test_consent_reconstructs_on_restart(self):
        """A new manager instance reloads all active and revoked consents from SQLite."""
        ok1, _, rec1 = self.manager.create_donation_consent(
            user_id="researcher_prem",
            interaction_id="turn_001",
            affirmative_ack=True
        )
        cid1 = rec1["consent_id"]

        ok2, _, rec2 = self.manager.create_donation_consent(
            user_id="researcher_prem",
            interaction_id="turn_002",
            affirmative_ack=True
        )
        cid2 = rec2["consent_id"]

        # Revoke one
        self.manager.revoke_donation_consent(cid2, "researcher_prem", reason="User changed mind")

        # Simulate restart by creating brand new manager pointing to same SQLite path
        new_manager = DonationConsentManager(storage_path=self.db_path)
        c1 = new_manager.get_consent(cid1)
        c2 = new_manager.get_consent(cid2)

        self.assertIsNotNone(c1)
        self.assertEqual(c1["status"], "ACTIVE")
        self.assertIsNotNone(c2)
        self.assertEqual(c2["status"], "REVOKED")
        self.assertEqual(c2["revocation_reason"], "User changed mind")


class TestSEC03AndSEC08LifecycleGatingAndAuthentication(unittest.TestCase):
    """SEC-03 & SEC-08: Ingestion gating, synthetic ID rejection, and caller authentication."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.temp_dir, "wal.jsonl")
        self.db_path = os.path.join(self.temp_dir, "consent.sqlite")
        self.lifecycle = ControlledLearningLifecycle(wal_path=self.wal_path, consent_db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unauthenticated_callers_rejected(self):
        """Anonymous, guest, or empty identities are rejected, and caller IDs fail closed without server auth."""
        # Unverified server identity fails closed
        self.assertFalse(is_authenticated_user(""))
        self.assertFalse(is_authenticated_user("guest"))
        self.assertFalse(is_authenticated_user("anonymous"))
        self.assertFalse(is_authenticated_user("null"))
        self.assertFalse(is_authenticated_user(None))
        self.assertFalse(is_authenticated_user("user_prem_01"))  # Fails closed under Phase 2.6.2.2 Task 1

        # Ingestion rejection for unauthenticated
        interaction = {
            "prompt": "Explain transformer self-attention mechanisms in detail.",
            "response": "Self-attention computes dynamic attention weights across key, query, and value vectors.",
            "explicitly_donated_for_training": True,
            "is_private_chat": False,
            "consent_id": "consent_123"
        }
        ok, reason, _ = self.lifecycle.ingest_candidate_example("guest", interaction, user_training_consent=True)
        self.assertFalse(ok)
        self.assertIn("Server-side cryptographic authentication is not verified", reason)

    def test_synthetic_auto_consent_rejected(self):
        """Synthetic auto_consent IDs are blocked fail-closed."""
        with patch("core.learning_lifecycle.donation.SERVER_AUTHENTICATION_VERIFIED", True), \
             patch("core.learning_lifecycle.donation.DONATION_INGESTION_ENABLED", True):
            interaction = {
                "prompt": "Calculate 12 * 12 + 6",
                "response": "The calculation result is 150.",
                "explicitly_donated_for_training": True,
                "is_private_chat": False,
                "consent_id": "auto_consent_researcher_prem"
            }
            ok, reason, _ = self.lifecycle.ingest_candidate_example("researcher_prem", interaction, user_training_consent=True)
            self.assertFalse(ok)
            self.assertIn("Synthetic", reason)

    def test_missing_consent_id_rejected(self):
        """Interaction with missing consent_id is blocked."""
        with patch("core.learning_lifecycle.donation.SERVER_AUTHENTICATION_VERIFIED", True), \
             patch("core.learning_lifecycle.donation.DONATION_INGESTION_ENABLED", True):
            interaction = {
                "prompt": "Calculate 12 * 12 + 6",
                "response": "The calculation result is 150.",
                "explicitly_donated_for_training": True,
                "is_private_chat": False
            }
            ok, reason, _ = self.lifecycle.ingest_candidate_example("researcher_prem", interaction, user_training_consent=True)
            self.assertFalse(ok)
            self.assertIn("Missing explicit consent_id", reason)

    def test_valid_consent_admitted_to_wal(self):
        """Valid consent allows candidate to pass into WAL candidate store when simulated auth is active."""
        with patch("core.learning_lifecycle.donation.SERVER_AUTHENTICATION_VERIFIED", True), \
             patch("core.learning_lifecycle.donation.DONATION_INGESTION_ENABLED", True):
            # 1. Create valid consent
            ok_c, _, c_rec = self.lifecycle.consent_manager.create_donation_consent(
                user_id="researcher_prem",
                interaction_id="math_turn_01",
                affirmative_ack=True
            )
            self.assertTrue(ok_c)
            cid = c_rec["consent_id"]

            interaction = {
                "prompt": "calculate 12 * 12 + 6",
                "response": "The answer is 150.0",
                "explicitly_donated_for_training": True,
                "is_private_chat": False,
                "consent_id": cid
            }
            ok_ingest, msg, cand_rec = self.lifecycle.ingest_candidate_example(
                user_id="researcher_prem",
                interaction=interaction,
                user_training_consent=True
            )
            self.assertTrue(ok_ingest)
            self.assertIsNotNone(cand_rec)
            self.assertEqual(cand_rec["verification_status"], VERIFIED_MATH)
            self.assertEqual(cand_rec["state"], STATE_OBJECTIVE_VERIFICATION)


class TestSEC04LegacyTrainingAndQuarantine(unittest.TestCase):
    """SEC-04: Quarantine user_learned_data.jsonl and disable self-train endpoint."""

    def test_prepare_splits_quarantines_user_learned_data(self):
        """prepare_splits explicitly skips any file named user_learned_data.jsonl."""
        from scripts.prepare_dataset import prepare_splits
        temp_dir = tempfile.mkdtemp()
        try:
            # Create dummy source and dummy user_learned_data.jsonl
            safe_source = os.path.join(temp_dir, "safe_data.json")
            with open(safe_source, "w", encoding="utf-8") as f:
                json.dump([
                    {"instruction": "What is Python?", "response": "Python is a high-level programming language."}
                ], f)

            unconsented_file = os.path.join(temp_dir, "user_learned_data.jsonl")
            with open(unconsented_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"instruction": "unconsented", "response": "chat"}) + "\n")

            manifest = prepare_splits(
                source_files=[safe_source, unconsented_file],
                output_dir=os.path.join(temp_dir, "splits"),
                seed=42
            )

            # Check that only safe_data was ingested
            train_path = os.path.join(temp_dir, "splits", "train.json")
            if os.path.exists(train_path):
                with open(train_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    for r in records:
                        self.assertNotEqual(r.get("source"), "user_learned_data.jsonl")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_run_self_train_step_is_disabled(self):
        """LearningService.run_self_train_step returns disabled status."""
        from webapp.learning_service import learning_service
        res = learning_service.run_self_train_step()
        self.assertFalse(res["success"])
        self.assertTrue(res.get("disabled", False))
        self.assertIn("permanently disabled", res["message"])

    def test_api_self_train_endpoint_returns_403(self):
        """Webapp /api/self-train route raises HTTPException(status_code=403)."""
        import asyncio
        from webapp.main import self_train_endpoint
        from webapp.schemas import SelfTrainRequest
        from fastapi import HTTPException

        req = SelfTrainRequest(iterations=10)
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self_train_endpoint(req))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("permanently disabled", ctx.exception.detail)


class TestSEC05StandaloneTrainingAuthorizationGate(unittest.TestCase):
    """SEC-05: Enforce AUTHORIZED_BY_PREM = False locks production training."""

    def test_policy_authorized_by_prem_is_false(self):
        """Module constant AUTHORIZED_BY_PREM must be strictly False."""
        self.assertFalse(AUTHORIZED_BY_PREM)

    def test_train_moon_chat_fails_closed_without_authorization(self):
        """Calling train_moon_chat without smoke_test raises PermissionError."""
        from training.chat.train import train_moon_chat
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(
                config_path="training/chat/config.yaml",
                smoke_test=False,
                dry_run=False
            )
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))
        self.assertIn("AUTHORIZED_BY_PREM is False", str(ctx.exception))

    def test_smoke_test_bypasses_production_gate_for_verification(self):
        """Smoke test flag bypasses production lock to permit dry-run pipeline testing."""
        # We verify that Safeguard 0 specifically allows is_smoke_test=True
        # We pass a non-existent config to verify it passed Safeguard 0 and reached config loading
        from training.chat.train import train_moon_chat
        try:
            train_moon_chat(
                config_path="non_existent_smoke_config.yaml",
                smoke_test=True
            )
        except Exception as e:
            # Must fail with FileNotFoundError (config loading), NOT PermissionError!
            self.assertIsInstance(e, FileNotFoundError)


class TestSEC06CandidateWALStoreIntegrityAndCorruptionDetection(unittest.TestCase):
    """SEC-06: Candidate WAL integrity, SHA-256 corruption detection, and concurrency."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.temp_dir, "test_wal.jsonl")
        self.store = CandidateWALStore(wal_path=self.wal_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_wal_stores_sha256_hash(self):
        """Candidate records store content_integrity_hash computed over prompt and response."""
        ok, _, rec = self.store.stage_candidate(
            candidate_id="cand_test_01",
            user_id="user_prem",
            consent_ref="consent_01",
            prompt="Prompt text",
            response="Response text",
            policy_version="moon-learning-donation-v1",
            verification_status="VERIFIED_MATH",
            verification_method="ast_eval"
        )
        self.assertTrue(ok)
        self.assertIn("content_integrity_hash", rec)
        self.assertEqual(len(rec["content_integrity_hash"]), 64)

    def test_tampered_wal_record_is_quarantined_on_reload(self):
        """If a record's prompt/response is tampered with on disk, reload rejects it."""
        self.store.stage_candidate(
            candidate_id="cand_valid",
            user_id="user_prem",
            consent_ref="consent_01",
            prompt="Original prompt",
            response="Original response",
            policy_version="moon-learning-donation-v1",
            verification_status="VERIFIED_MATH",
            verification_method="ast_eval"
        )

        # Manually tamper with the WAL file by modifying the response but keeping original hash
        with open(self.wal_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        tampered_entry = lines[0]
        tampered_entry["record"]["response"] = "MALICIOUSLY TAMPERED RESPONSE"

        with open(self.wal_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(tampered_entry) + "\n")

        # Reload store from disk
        reloaded_store = CandidateWALStore(wal_path=self.wal_path)
        # Tampered record must NOT be in active candidates
        self.assertIsNone(reloaded_store.get_candidate("cand_valid"))
        self.assertEqual(len(reloaded_store.list_candidates()), 0)
        # Must be captured in corrupted records
        corrupted = reloaded_store.get_corrupted_records()
        self.assertEqual(len(corrupted), 1)
        self.assertEqual(corrupted[0]["candidate_id"], "cand_valid")
        self.assertIn("mismatch", corrupted[0]["reason"])

    def test_thread_safe_wal_appends(self):
        """Multiple concurrent threads appending to WAL do not corrupt JSONL structure."""
        def append_task(thread_id: int):
            for i in range(10):
                self.store.stage_candidate(
                    candidate_id=f"t_{thread_id}_item_{i}",
                    user_id=f"user_{thread_id}",
                    consent_ref=f"c_{thread_id}",
                    prompt=f"Prompt {thread_id} {i}",
                    response=f"Response {thread_id} {i}",
                    policy_version="moon-learning-donation-v1",
                    verification_status="HUMAN_REVIEW_REQUIRED",
                    verification_method="manual"
                )

        threads = [threading.Thread(target=append_task, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Reload from disk and verify all 50 items reloaded cleanly with 0 corruption
        reloaded = CandidateWALStore(wal_path=self.wal_path)
        self.assertEqual(len(reloaded.list_candidates()), 50)
        self.assertEqual(len(reloaded.get_corrupted_records()), 0)


class TestSEC07ConsentRevocationPropagation(unittest.TestCase):
    """SEC-07: Consent revocation propagates to candidate store, setting state to WITHDRAWN."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.temp_dir, "wal.jsonl")
        self.db_path = os.path.join(self.temp_dir, "consent.sqlite")
        self.patch_auth = patch("core.learning_lifecycle.donation.SERVER_AUTHENTICATION_VERIFIED", True)
        self.patch_ingest = patch("core.learning_lifecycle.donation.DONATION_INGESTION_ENABLED", True)
        self.patch_auth.start()
        self.patch_ingest.start()
        self.consent_mgr = DonationConsentManager(storage_path=self.db_path)
        self.store = CandidateWALStore(wal_path=self.wal_path)

    def tearDown(self):
        self.patch_ingest.stop()
        self.patch_auth.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_revocation_transitions_candidates_to_withdrawn(self):
        """When user revokes consent, all associated staged candidates are marked WITHDRAWN."""
        # 1. Create consent
        ok_c, _, c_rec = self.consent_mgr.create_donation_consent(
            user_id="researcher_prem",
            interaction_id="turn_100",
            affirmative_ack=True
        )
        cid = c_rec["consent_id"]

        # 2. Stage candidate linked to this consent
        self.store.stage_candidate(
            candidate_id="cand_100",
            user_id="researcher_prem",
            consent_ref=cid,
            prompt="How to build an offline LLM?",
            response="An offline LLM requires local weights, tokenizers, and sovereign training code.",
            policy_version="moon-learning-donation-v1",
            verification_status="HUMAN_REVIEW_REQUIRED",
            verification_method="syntax_check"
        )
        c_before = self.store.get_candidate("cand_100")
        self.assertEqual(c_before["state"], STATE_PENDING_HUMAN_REVIEW)

        # 3. Revoke consent with candidate_store propagation
        ok_r, r_msg = self.consent_mgr.revoke_donation_consent(
            consent_id=cid,
            user_id="researcher_prem",
            reason="User exercised privacy right to withdraw",
            candidate_store=self.store
        )
        self.assertTrue(ok_r)
        self.assertIn("WITHDRAWN", r_msg)

        # 4. Verify candidate is now WITHDRAWN in memory and in WAL
        c_after = self.store.get_candidate("cand_100")
        self.assertEqual(c_after["state"], STATE_WITHDRAWN)

        # Verify state transition persisted on WAL reload
        reloaded_store = CandidateWALStore(wal_path=self.wal_path)
        reloaded_cand = reloaded_store.get_candidate("cand_100")
        self.assertEqual(reloaded_cand["state"], STATE_WITHDRAWN)


if __name__ == "__main__":
    unittest.main()

