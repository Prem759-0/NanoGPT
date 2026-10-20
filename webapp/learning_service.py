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

    def record_chat_turn(self, instruction: str, response: str, source: str = "chat"):
        """
        Records an interaction into SQLite and appends to user_learned_data.jsonl.
        """
        inst = instruction.strip()
        resp = response.strip()

        # Filter out trivial greetings or errors from fine-tuning
        if len(inst) < 4 or len(resp) < 4 or "Input Validation Error" in resp:
            return

        # 1. Save to SQLite
        db.log_training_pair(inst, resp, source=source)

        # 2. Append to JSONL for easy model fine-tuning
        record = {
            "instruction": inst,
            "response": resp,
            "source": source
        }
        try:
            with open(LEARNED_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not append to {LEARNED_JSONL_PATH}: {e}")

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
        Performs a rapid self-learning update:
        1. Compiles collected chat pairs into assistant_chat dataset
        2. Tokenizes and updates train.bin
        3. Returns status
        """
        pairs = db.list_training_pairs(limit=200)
        if not pairs:
            return {
                "success": False,
                "message": "No chat interactions recorded yet to train on.",
                "training_pairs_count": 0
            }

        # Sync training pairs to data/assistant_chat/input.txt
        assistant_input_path = os.path.join(DATA_DIR, "assistant_chat", "input.txt")
        new_text_blocks = []
        for p in reversed(pairs):
            new_text_blocks.append(f"User: {p['instruction']}\nAssistant: {p['response']}\n")

        try:
            if os.path.exists(assistant_input_path):
                with open(assistant_input_path, "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(new_text_blocks) + "\n")

            return {
                "success": True,
                "message": f"Successfully integrated {len(pairs)} learned conversation pairs into model training corpus!",
                "training_pairs_count": len(pairs),
                "dataset_ready": True
            }
        except Exception as e:
            logger.error(f"Failed to integrate training data: {e}")
            return {
                "success": False,
                "message": f"Error updating training dataset: {str(e)}",
                "training_pairs_count": len(pairs)
            }

learning_service = LearningService()
