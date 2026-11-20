import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from webapp.database import db

logger = logging.getLogger("nanogpt-lab.learning")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LEARNED_JSONL_PATH = os.path.join(DATA_DIR, "user_learned_data.jsonl")

class LearningService:
    """
    Dual Self-Learning Engine for NanoGPT Lab:
    1. Persistent Long-Term Memory (In-Context RAG / Fact Bank)
    2. Continuous Chat Training Dataset Collector & Local Fine-Tuning Pipeline
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # =========================================================================
    # 1. LONG-TERM MEMORY (Fact & Rule Learning)
    # =========================================================================

    def detect_memory_intent(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Detects if the user is explicitly or implicitly teaching the AI a fact or rule.
        Returns (fact, category) or None.
        """
        t = text.strip()

        # Explicit commands: /learn <fact>, /remember <fact>
        cmd_match = re.match(r"^/(?:learn|remember)\s+(.+)$", t, re.IGNORECASE)
        if cmd_match:
            return (cmd_match.group(1).strip(), "explicit_rule")

        # Natural phrasing: "remember that ...", "remember: ...", "please remember that ..."
        rem_match = re.match(r"^(?:please\s+)?remember(?:\s+that|:)?\s+(.+)$", t, re.IGNORECASE)
        if rem_match:
            return (rem_match.group(1).strip(), "user_fact")

        # "keep in mind that ..."
        kim_match = re.match(r"^(?:please\s+)?keep in mind(?:\s+that)?\s+(.+)$", t, re.IGNORECASE)
        if kim_match:
            return (kim_match.group(1).strip(), "instruction")

        # Personal identity / preference learning
        name_match = re.match(r"^(?:my name is|i am called|call me)\s+([A-Za-z0-9_ -]+)$", t, re.IGNORECASE)
        if name_match:
            return (f"The user's name is {name_match.group(1).strip()}", "user_identity")

        pref_match = re.match(r"^(?:i prefer|my favorite\s+.+?\s+is|always use|never use)\s+(.+)$", t, re.IGNORECASE)
        if pref_match:
            return (t, "preference")

        return None

    def store_memory(self, fact: str, category: str = "general", session_id: Optional[str] = None) -> Dict[str, Any]:
        """Saves a permanent memory to SQLite."""
        mem_id = db.add_memory(fact, category, session_id)
        return {
            "success": True,
            "id": mem_id,
            "fact": fact,
            "category": category,
            "already_existed": (mem_id is None)
        }

    def get_memories_prompt_injection(self) -> str:
        """
        Formats all saved memories into a clean markdown block to inject into the LLM system prompt.
        """
        memories = db.list_memories(limit=30)
        if not memories:
            return ""

        lines = ["[LONG-TERM MEMORY & USER PROFILE]:"]
        for m in memories:
            lines.append(f"• {m['fact']}")
        lines.append("(Strictly adhere to and utilize these remembered facts in all responses.)\n")
        return "\n".join(lines)

    # =========================================================================
    # 2. CHAT DATASET COLLECTOR & SELF-TRAINING
    # =========================================================================

    def record_chat_turn(
        self,
        instruction: str,
        response: str,
        source: str = "chat",
        explicit_donation_consent: bool = False
    ) -> bool:
        """
        Phase 2.6.2.2 Containment Guard:
        Appending to user_learned_data.jsonl is permanently disabled.
        Historical files are quarantined and preserved byte-for-byte.
        Direct conversation logging to training datasets is strictly prohibited.
        """
        # Under Moon AI Phase 2.6.2.2 containment, appending to user_learned_data.jsonl is permanently disabled
        return False

    def get_learning_stats(self) -> Dict[str, Any]:
        """Returns statistics on learned memories and collected training pairs."""
        memories = db.list_memories(limit=100)
        pair_count = db.count_training_pairs()
        return {
            "total_memories": len(memories),
            "memories": memories,
            "total_training_pairs": pair_count,
            "jsonl_path": LEARNED_JSONL_PATH
        }

    def run_self_train_step(self, iterations: int = 50) -> Dict[str, Any]:
        """
        Self-training endpoint is permanently disabled under Moon AI learning governance policy (SEC-04).
        Automatic or unconsented integration of live chat data into training corpora is strictly prohibited.
        Fails closed immediately while AUTHORIZED_BY_PREM is False.
        """
        return {
            "success": False,
            "disabled": True,
            "reason": "OWNER_AUTHORIZATION_REQUIRED",
            "message": (
                "🔒 Self-training is permanently disabled: unconsented automatic integration of "
                "live chat data into training corpora is prohibited under Moon AI sovereign policy."
            ),
            "training_pairs_count": 0,
        }

learning_service = LearningService()
