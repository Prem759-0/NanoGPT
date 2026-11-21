"""
Moon AI — Conditional User Conversation Donation & Governance Module
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Data Governance Policy: moon-learning-donation-v1

Implements strict affirmative, double opt-in conversation donation.
Separate from live session memory consent.
Zero automatic ingestion: users must explicitly select and donate specific turns.
Private conversations and personal memories are permanently forbidden from donation.
Durable SQLite consent storage and revocation tracking.
"""

import os
import json
import sqlite3
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LEARNING_DIR = os.path.join(PROJECT_ROOT, "data", "learning")
DEFAULT_CONSENT_DB_PATH = os.path.join(DEFAULT_LEARNING_DIR, "donation_consent.sqlite")

DONATION_POLICY_VERSION = "moon-learning-donation-v1"

# Server-side authentication status flag (Phase 2.6.2.2 Task 1)
# Must remain False until server-side cryptographic token verification is active.
SERVER_AUTHENTICATION_VERIFIED: bool = False

# Global donation ingestion status gate
# Invariant: Must remain False until genuine server-side authentication is deployed.
DONATION_INGESTION_ENABLED: bool = False


def is_authenticated_user(user_id: Optional[str]) -> bool:
    """
    Validates caller identity against a trusted server-side authentication provider.

    SECURITY ARCHITECTURE NOTE (Phase 2.6.2.2 Task 1):
    String checking (e.g. rejecting 'guest' or 'anonymous') is NOT authentication.
    A caller can supply arbitrary headers (e.g. 'x-user-id: prem') or forged function arguments.
    True authentication requires server-side cryptographic token verification (e.g. Clerk JWT
    signature verification, session HMAC, or OAuth introspection).
    In the absence of a verified server-side cryptographic session validator,
    SERVER_AUTHENTICATION_VERIFIED is False, and all caller-supplied identities fail closed.
    """
    if not SERVER_AUTHENTICATION_VERIFIED:
        return False
    if not user_id or not isinstance(user_id, str):
        return False
    cleaned = user_id.strip().lower()
    if cleaned in ("", "guest", "anonymous", "null", "undefined", "none", "unknown"):
        return False
    return True


class DonationConsentManager:
    """
    Manages explicit, affirmative user conversation donation consents and revocations
    with durable SQLite backing and memory indexing.
    Guarantees strict privacy boundaries and zero automatic data harvesting.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or DEFAULT_CONSENT_DB_PATH
        self._consents: Dict[str, Dict[str, Any]] = {}
        self._user_consents: Dict[str, Set[str]] = {}
        self._init_db()
        self._reload_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes the SQLite schema for consent tracking."""
        if self.storage_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)

        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS donation_consents (
                        consent_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        interaction_id TEXT NOT NULL,
                        is_private_chat INTEGER NOT NULL,
                        declared_topic TEXT,
                        consent_timestamp TEXT NOT NULL,
                        policy_version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        revoked_at TEXT,
                        revocation_reason TEXT,
                        notice TEXT,
                        raw_record TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_consents_user_id
                    ON donation_consents(user_id)
                """)
        finally:
            conn.close()

    def _reload_from_db(self) -> None:
        """Reconstructs in-memory consent index from SQLite database."""
        self._consents.clear()
        self._user_consents.clear()

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT consent_id, user_id, raw_record FROM donation_consents")
            for row in cur.fetchall():
                cid = row["consent_id"]
                uid = row["user_id"]
                try:
                    rec = json.loads(row["raw_record"])
                    self._consents[cid] = rec
                    if uid not in self._user_consents:
                        self._user_consents[uid] = set()
                    self._user_consents[uid].add(cid)
                except Exception:
                    continue
        finally:
            conn.close()

    def create_donation_consent(
        self,
        user_id: str,
        interaction_id: str,
        is_private_chat: bool = False,
        declared_topic: str = "",
        affirmative_ack: bool = False
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Creates a new, explicit conversation donation consent record and persists to SQLite.
        Requires affirmative acknowledgment of training purpose.
        Strictly rejects unauthenticated users, private chats, and missing acknowledgments.
        """
        # 1. Server-side cryptographic authentication check (Phase 2.6.2.2 Task 1)
        if not DONATION_INGESTION_ENABLED or not SERVER_AUTHENTICATION_VERIFIED:
            return False, (
                "Donation rejected: Server-side cryptographic authentication is not configured. "
                "Caller-supplied user IDs and request headers cannot authorize conversation donation. "
                "Donation ingestion remains permanently disabled."
            ), None

        if not is_authenticated_user(user_id):
            return False, "Donation rejected: User must be authenticated to donate conversation data.", None

        # 2. Affirmative acknowledgment requirement
        if not affirmative_ack:
            return False, "Donation rejected: Explicit affirmative acknowledgment of training purpose required.", None

        # 3. Private conversation invariant
        if is_private_chat:
            return False, "Donation rejected: Private chat sessions cannot be donated for model training.", None

        # 4. Generate verifiable consent record
        consent_id = f"consent_{uuid.uuid4().hex}"
        now_iso = datetime.now(timezone.utc).isoformat()

        record: Dict[str, Any] = {
            "consent_id": consent_id,
            "user_id": user_id.strip(),
            "interaction_id": interaction_id.strip(),
            "explicitly_donated_for_training": True,
            "is_private_chat": False,
            "declared_topic": declared_topic.strip(),
            "consent_timestamp": now_iso,
            "policy_version": DONATION_POLICY_VERSION,
            "status": "ACTIVE",
            "revoked_at": None,
            "revocation_reason": None,
            "notice": (
                "Donation authorized voluntarily by user for Moon AI sovereign offline research. "
                "Data is subject to PII scrubbing, injection filtering, and deterministic verification."
            )
        }

        # 5. Persist to SQLite (SEC-02)
        raw_json = json.dumps(record, ensure_ascii=False)
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO donation_consents (
                        consent_id, user_id, interaction_id, is_private_chat,
                        declared_topic, consent_timestamp, policy_version,
                        status, revoked_at, revocation_reason, notice, raw_record
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    consent_id, record["user_id"], record["interaction_id"], 0,
                    record["declared_topic"], now_iso, DONATION_POLICY_VERSION,
                    "ACTIVE", None, None, record["notice"], raw_json
                ))
        finally:
            conn.close()

        # Update in-memory index
        self._consents[consent_id] = record
        if user_id not in self._user_consents:
            self._user_consents[user_id] = set()
        self._user_consents[user_id].add(consent_id)

        return True, "Donation consent recorded successfully.", record

    def revoke_donation_consent(
        self,
        consent_id: str,
        user_id: str,
        reason: str = "User requested withdrawal",
        candidate_store: Optional[Any] = None
    ) -> Tuple[bool, str]:
        """
        Revokes a previously granted donation consent in SQLite and memory.
        If candidate_store is provided, propagates revocation to all associated
        candidate records in the WAL, transitioning them to WITHDRAWN (SEC-07).
        """
        if consent_id not in self._consents:
            return False, f"Consent record '{consent_id}' not found."

        rec = self._consents[consent_id]
        if rec["user_id"] != user_id:
            return False, "Authorization error: Cannot revoke consent belonging to another user."

        if rec["status"] == "REVOKED":
            return True, "Consent is already revoked."

        now_iso = datetime.now(timezone.utc).isoformat()
        rec["status"] = "REVOKED"
        rec["revoked_at"] = now_iso
        rec["revocation_reason"] = reason

        # Persist revocation to SQLite
        raw_json = json.dumps(rec, ensure_ascii=False)
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    UPDATE donation_consents
                    SET status = 'REVOKED', revoked_at = ?, revocation_reason = ?, raw_record = ?
                    WHERE consent_id = ?
                """, (now_iso, reason, raw_json, consent_id))
        finally:
            conn.close()

        # Revocation propagation to candidate store (SEC-07)
        withdrawn_count = 0
        if candidate_store is not None:
            for cand in candidate_store.list_candidates():
                if cand.get("consent_ref") == consent_id:
                    cand_id = cand.get("candidate_id")
                    candidate_store.withdraw_candidate(
                        cand_id,
                        reason=f"Consent '{consent_id}' revoked by user: {reason}"
                    )
                    withdrawn_count += 1

        msg = (
            f"Donation consent successfully revoked in durable storage. "
            f"{withdrawn_count} associated candidate records transitioned to WITHDRAWN. "
            f"(Note: historical checkpoints already compiled cannot have weight adjustments reversed retrospectively)."
        )
        return True, msg

    def verify_donation_eligibility(
        self,
        user_id: str,
        interaction: Dict[str, Any],
        consent_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Strictly verifies that an interaction possesses valid, active, non-revoked donation consent.
        Missing, synthetic, or revoked consent immediately causes rejection (SEC-03, SEC-08).
        """
        # 1. Server-side authentication check (Phase 2.6.2.2 Task 1)
        if not DONATION_INGESTION_ENABLED or not SERVER_AUTHENTICATION_VERIFIED:
            return False, (
                "Ineligible: Server-side cryptographic authentication is not verified. "
                "Caller-supplied identities cannot authorize candidate donation. "
                "Donation ingestion remains permanently disabled."
            ), None

        if not is_authenticated_user(user_id):
            return False, "Ineligible: Unauthenticated user.", None

        # 2. Check memory isolation invariant: memory items can NEVER be donated
        if interaction.get("eligible_for_training") is False and interaction.get("source") == "memory":
            return False, "Ineligible: User personal memories cannot be converted to training candidates.", None

        # 3. Check private conversation flag
        if interaction.get("is_private_chat", True):
            return False, "Ineligible: Private conversations cannot be donated.", None

        # 4. Explicit donation flag must be True
        if not interaction.get("explicitly_donated_for_training", False):
            return False, "Ineligible: Missing explicit donation flag ('explicitly_donated_for_training').", None

        # 5. Check consent record (reject synthetic/auto_consent IDs) (SEC-03)
        cid = consent_id or interaction.get("consent_id")
        if not cid:
            return False, "Ineligible: Missing consent_id.", None

        if cid.startswith("auto_consent_"):
            return False, "Ineligible: Synthetic auto_consent IDs are strictly prohibited.", None

        if cid not in self._consents:
            return False, f"Ineligible: No matching active consent record found for ID '{cid}'.", None

        consent_rec = self._consents[cid]
        if consent_rec["user_id"] != user_id:
            return False, "Ineligible: Consent user_id mismatch.", None

        if consent_rec["status"] != "ACTIVE":
            return False, f"Ineligible: Consent status is '{consent_rec['status']}'.", None

        if consent_rec.get("policy_version") != DONATION_POLICY_VERSION:
            return False, f"Ineligible: Unrecognized policy version '{consent_rec.get('policy_version')}'.", None

        return True, "Donation verified and eligible for security screening.", consent_rec

    def get_consent(self, consent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a consent record by ID."""
        return self._consents.get(consent_id)

    def list_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Lists all consent records for a user."""
        cids = self._user_consents.get(user_id, set())
        return [self._consents[cid] for cid in cids if cid in self._consents]
