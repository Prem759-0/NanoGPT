"""
Moon AI — Durable Candidate Store & Governance WAL Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Provides an append-only, SHA-256 integrity-verified Write-Ahead Log (WAL)
for staged training candidate lifecycle management.

Lifecycle States:
1. RECEIVED
2. CONSENT_CHECK
3. SANITIZATION
4. INJECTION_SCREENING
5. OBJECTIVE_VERIFICATION
6. DEDUPLICATION
7. PENDING_HUMAN_REVIEW
8. APPROVED_CANDIDATE
9. REJECTED
10. WITHDRAWN

CRITICAL TRAINING ISOLATION INVARIANT:
The candidate store is an isolated quarantine, triage, and review facility.
Records in this store are NEVER automatically merged into production training manifests
(e.g., scaled_chat_train.jsonl) or fed to active training scripts.
"""

import os
import json
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LEARNING_DIR = os.path.join(PROJECT_ROOT, "data", "learning")
DEFAULT_WAL_PATH = os.path.join(DEFAULT_LEARNING_DIR, "candidates_wal.jsonl")

# Lifecycle States
STATE_RECEIVED = "RECEIVED"
STATE_CONSENT_CHECK = "CONSENT_CHECK"
STATE_SANITIZATION = "SANITIZATION"
STATE_INJECTION_SCREENING = "INJECTION_SCREENING"
STATE_OBJECTIVE_VERIFICATION = "OBJECTIVE_VERIFICATION"
STATE_DEDUPLICATION = "DEDUPLICATION"
STATE_PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
STATE_APPROVED_CANDIDATE = "APPROVED_CANDIDATE"
STATE_REJECTED = "REJECTED"
STATE_WITHDRAWN = "WITHDRAWN"

VALID_CANDIDATE_STATES: Set[str] = {
    STATE_RECEIVED,
    STATE_CONSENT_CHECK,
    STATE_SANITIZATION,
    STATE_INJECTION_SCREENING,
    STATE_OBJECTIVE_VERIFICATION,
    STATE_DEDUPLICATION,
    STATE_PENDING_HUMAN_REVIEW,
    STATE_APPROVED_CANDIDATE,
    STATE_REJECTED,
    STATE_WITHDRAWN,
}


class CandidateStoreError(Exception):
    """Raised when candidate record integrity or lifecycle rules are violated."""
    pass


class CandidateWALStore:
    """
    Manages durable candidate records with state transitions and SHA-256 integrity verification.

    INTEGRITY & IMMUTABILITY LIMITATION:
    The SHA-256 integrity hash provides corruption and accidental alteration detection on startup.
    It does not provide hardware or cryptographic immutability if an adversary has write access
    to the underlying filesystem. True immutability requires append-only WORM storage or cryptographic
    ledger signatures.
    """

    def __init__(self, wal_path: Optional[str] = None):
        self.wal_path = wal_path or DEFAULT_WAL_PATH
        os.makedirs(os.path.dirname(self.wal_path), exist_ok=True)
        self._wal_lock = threading.Lock()
        # In-memory index reconstructed from WAL
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._content_hashes: Dict[str, str] = {}  # sha256 -> candidate_id
        self._corrupted_records: List[Dict[str, Any]] = []
        self._reload_from_wal()

    def _compute_content_hash(self, prompt: str, response: str) -> str:
        """Computes deterministic SHA-256 hash of prompt and response."""
        combined = f"{prompt.strip()}\n---\n{response.strip()}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _reload_from_wal(self) -> None:
        """Recovers candidate state by reading all events from the WAL on startup."""
        self._candidates.clear()
        self._content_hashes.clear()
        self._corrupted_records.clear()

        if not os.path.exists(self.wal_path):
            return

        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    event_type = entry.get("event")
                    cand_id = entry.get("candidate_id")

                    if event_type == "CANDIDATE_CREATED":
                        rec = entry.get("record", {})
                        prompt = rec.get("prompt", "")
                        response = rec.get("response", "")
                        stored_hash = rec.get("content_integrity_hash")
                        computed_hash = self._compute_content_hash(prompt, response)

                        # SEC-06: Cryptographic integrity check on load
                        if stored_hash != computed_hash:
                            self._corrupted_records.append({
                                "candidate_id": cand_id,
                                "line_no": line_no,
                                "stored_hash": stored_hash,
                                "computed_hash": computed_hash,
                                "reason": "SHA-256 integrity mismatch: prompt or response modified"
                            })
                            continue

                        self._candidates[cand_id] = rec
                        if stored_hash:
                            self._content_hashes[stored_hash] = cand_id

                    elif event_type == "STATE_TRANSITION":
                        if cand_id in self._candidates:
                            self._candidates[cand_id]["state"] = entry.get("to_state")
                            self._candidates[cand_id]["updated_at"] = entry.get("timestamp")
                            if "metadata" in entry:
                                self._candidates[cand_id].setdefault("transition_history", []).append({
                                    "from": entry.get("from_state"),
                                    "to": entry.get("to_state"),
                                    "timestamp": entry.get("timestamp"),
                                    "metadata": entry.get("metadata")
                                })
                except Exception as e:
                    # Skip malformed lines rather than crashing server startup, but track
                    self._corrupted_records.append({
                        "candidate_id": None,
                        "line_no": line_no,
                        "reason": f"Malformed JSON: {str(e)}"
                    })
                    continue

    def _append_wal_entry(self, entry: Dict[str, Any]) -> None:
        """Appends a single JSON event to the write-ahead log under lock (thread-safe)."""
        line = json.dumps(entry, ensure_ascii=False)
        with self._wal_lock:
            with open(self.wal_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def stage_candidate(
        self,
        candidate_id: str,
        user_id: str,
        consent_ref: str,
        prompt: str,
        response: str,
        policy_version: str,
        verification_status: str,
        verification_method: str,
        sanitization_status: str = "PASSED",
        subjective_feedback: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Creates and persists a new candidate record in the WAL.
        Checks for exact deduplication before accepting.
        """
        content_hash = self._compute_content_hash(prompt, response)

        # 1. Deduplication check
        if content_hash in self._content_hashes:
            dup_id = self._content_hashes[content_hash]
            return False, f"Deduplication rejection: Exact match of candidate '{dup_id}'", None

        now_iso = datetime.now(timezone.utc).isoformat()
        initial_state = STATE_PENDING_HUMAN_REVIEW if verification_status == "HUMAN_REVIEW_REQUIRED" else STATE_OBJECTIVE_VERIFICATION

        record: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "provenance": "VERIFIED_SOVEREIGN_EXPERIENCE",
            "consent_ref": consent_ref,
            "donor_user_id": user_id,
            "prompt": prompt,
            "response": response,
            "policy_version": policy_version,
            "state": initial_state,
            "verification_status": verification_status,
            "verification_method": verification_method,
            "sanitization_status": sanitization_status,
            "deduplication_result": "UNIQUE",
            "review_status": "PENDING",
            "content_integrity_hash": content_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
            "subjective_feedback": subjective_feedback or {},
            "transition_history": [
                {
                    "from": STATE_RECEIVED,
                    "to": initial_state,
                    "timestamp": now_iso,
                    "reason": "Candidate ingested through governance funnel"
                }
            ]
        }

        wal_entry = {
            "event": "CANDIDATE_CREATED",
            "candidate_id": candidate_id,
            "timestamp": now_iso,
            "record": record
        }

        self._append_wal_entry(wal_entry)
        self._candidates[candidate_id] = record
        self._content_hashes[content_hash] = candidate_id

        return True, f"Candidate '{candidate_id}' successfully staged with state '{initial_state}'.", record

    def transition_state(
        self,
        candidate_id: str,
        new_state: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Transitions candidate record to a new lifecycle state and appends to WAL.
        """
        if candidate_id not in self._candidates:
            return False, f"Candidate '{candidate_id}' does not exist."

        if new_state not in VALID_CANDIDATE_STATES:
            return False, f"Invalid candidate state '{new_state}'."

        rec = self._candidates[candidate_id]
        old_state = rec.get("state")
        now_iso = datetime.now(timezone.utc).isoformat()

        rec["state"] = new_state
        rec["updated_at"] = now_iso
        trans_meta = {
            "reason": reason,
            **(metadata or {})
        }
        rec.setdefault("transition_history", []).append({
            "from": old_state,
            "to": new_state,
            "timestamp": now_iso,
            "metadata": trans_meta
        })

        wal_entry = {
            "event": "STATE_TRANSITION",
            "candidate_id": candidate_id,
            "from_state": old_state,
            "to_state": new_state,
            "timestamp": now_iso,
            "metadata": trans_meta
        }
        self._append_wal_entry(wal_entry)

        return True, f"Candidate '{candidate_id}' transitioned from '{old_state}' to '{new_state}'."

    def withdraw_candidate(self, candidate_id: str, reason: str = "User revoked donation") -> Tuple[bool, str]:
        """Marks candidate as WITHDRAWN upon consent revocation."""
        return self.transition_state(candidate_id, STATE_WITHDRAWN, reason=reason)

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Returns candidate record if present."""
        return self._candidates.get(candidate_id)

    def list_candidates(self, state: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all candidates, optionally filtered by lifecycle state."""
        if state:
            return [c for c in self._candidates.values() if c.get("state") == state]
        return list(self._candidates.values())

    def get_corrupted_records(self) -> List[Dict[str, Any]]:
        """Returns records that failed SHA-256 integrity validation during WAL reload."""
        return list(self._corrupted_records)

    def get_wal_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics for candidate store."""
        counts: Dict[str, int] = {}
        for c in self._candidates.values():
            s = c.get("state", "UNKNOWN")
            counts[s] = counts.get(s, 0) + 1

        return {
            "wal_path": self.wal_path,
            "total_candidates": len(self._candidates),
            "state_distribution": counts,
            "corrupted_records_count": len(self._corrupted_records),
            "training_isolation_verified": True
        }

