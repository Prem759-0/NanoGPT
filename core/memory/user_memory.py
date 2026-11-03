"""
Moon AI — Persistent User-Approved Memory Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Provides user-consented, persistent context memory across sessions.
Critical Architecture Separation:
- User Memory is stored in private key-value / semantic storage for live session personalization.
- User Memory is NEVER automatically fed into model training manifests or neural network weights.
- Users have full sovereignty: explicit opt-in, per-item deletion, and complete account purge.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import json

class UserMemoryManager:
    """Manages consented persistent user memory with strict privacy controls."""

    def __init__(self):
        # In-memory store: user_id -> {key: {"value": ..., "created_at": ..., "consent": True}}
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._opt_in_users: Dict[str, bool] = {}

    def set_user_memory_consent(self, user_id: str, consent: bool):
        """Sets the user's explicit consent for persistent memory retention."""
        self._opt_in_users[user_id] = consent
        if not consent and user_id in self._store:
            # If consent revoked, purge existing memory
            self._store.pop(user_id, None)

    def has_consent(self, user_id: str) -> bool:
        """Returns True if the user has explicitly consented to persistent memory."""
        return self._opt_in_users.get(user_id, False)

    def store_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        description: str = ""
    ) -> Tuple[bool, str]:
        """Stores a persistent memory item if user consent is active."""
        if not self.has_consent(user_id):
            return False, f"Memory storage rejected: User '{user_id}' has not opted in to persistent memory."

        if user_id not in self._store:
            self._store[user_id] = {}

        self._store[user_id][key] = {
            "value": value,
            "description": description,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "eligible_for_training": False  # Invariant: User memory is never eligible for training
        }
        return True, f"Memory '{key}' stored successfully."

    def retrieve_memory(self, user_id: str, key: str) -> Optional[Any]:
        """Retrieves a specific memory item."""
        if not self.has_consent(user_id):
            return None
        return self._store.get(user_id, {}).get(key, {}).get("value")

    def list_memories(self, user_id: str) -> Dict[str, Any]:
        """Lists all stored memory keys and values for a user."""
        if not self.has_consent(user_id):
            return {}
        return {k: v["value"] for k, v in self._store.get(user_id, {k: v}).items()}

    def delete_memory(self, user_id: str, key: str) -> bool:
        """Deletes a specific memory item."""
        if user_id in self._store and key in self._store[user_id]:
            del self._store[user_id][key]
            return True
        return False

    def purge_all_user_memories(self, user_id: str) -> int:
        """Purges all memory for a user (GDPR / right to be forgotten)."""
        count = len(self._store.get(user_id, {}))
        self._store.pop(user_id, None)
        return count

    def export_user_memories(self, user_id: str) -> Dict[str, Any]:
        """Exports user memory in a portable JSON format."""
        return {
            "user_id": user_id,
            "consent_active": self.has_consent(user_id),
            "memories": self._store.get(user_id, {})
        }

    def verify_training_isolation(self, user_id: str, key: str) -> bool:
        """Asserts that user memory items are never marked eligible for training."""
        item = self._store.get(user_id, {}).get(key)
        if not item:
            return True
        return item.get("eligible_for_training") is False

