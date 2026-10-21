import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chats.db"))

class Database:
    """
    Secure Local SQLite Database for NanoGPT Lab.
    Provides session history, context memory, and data protection with zero external leaks.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode = WAL;")
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT 'guest',
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT DEFAULT '{}',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'guest',
                    fact TEXT NOT NULL UNIQUE,
                    category TEXT DEFAULT 'general',
                    source_session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS training_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction TEXT NOT NULL,
                    response TEXT NOT NULL,
                    source TEXT DEFAULT 'chat',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Safe column migration for existing databases
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'guest';")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN user_id TEXT DEFAULT 'guest';")
            except Exception:
                pass

            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_training_created ON training_pairs (created_at DESC);
            """)

    def create_session(self, title: str = "New Chat", session_id: Optional[str] = None, user_id: str = "guest") -> str:
        s_id = session_id or str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (s_id, user_id or "guest", title, now, now)
            )
        return s_id

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if user_id:
                if user_id in ["guest", "guest_user"]:
                    cursor = conn.execute(
                        "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE user_id IN ('guest', 'guest_user') ORDER BY updated_at DESC"
                    )
                else:
                    cursor = conn.execute(
                        "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                        (user_id,)
                    )
            else:
                cursor = conn.execute(
                    "SELECT id, user_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, session_id, role, content, meta_json, timestamp 
                FROM (
                    SELECT id, session_id, role, content, meta_json, timestamp 
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY id DESC 
                    LIMIT ?
                ) 
                ORDER BY id ASC
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["meta"] = json.loads(item.get("meta_json", "{}"))
                except Exception:
                    item["meta"] = {}
                results.append(item)
            return results

    def add_message(self, session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None, user_id: str = "guest") -> int:
        now = datetime.utcnow().isoformat()
        meta_str = json.dumps(meta or {})
        
        # Ensure session exists
        if not self.get_session(session_id):
            title = content[:30] + "..." if len(content) > 30 else content
            self.create_session(title=title, session_id=session_id, user_id=user_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, meta_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_str, now)
            )
            msg_id = cursor.lastrowid
            # Update session timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id)
            )
            return msg_id

    def delete_session(self, session_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def rename_session(self, session_id: str, new_title: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, datetime.utcnow().isoformat(), session_id)
            )

    # --- Long-Term Memory Storage ---
    def add_memory(self, fact: str, category: str = "general", session_id: Optional[str] = None, user_id: str = "guest") -> Optional[int]:
        """Saves a permanent memory or learned fact to the database."""
        fact_clean = fact.strip()
        if not fact_clean:
            return None
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO memories (user_id, fact, category, source_session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id or "guest", fact_clean, category, session_id, now)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Already remembered
                return None

    def list_memories(self, limit: int = 100, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns stored long-term memories, optionally filtered by user."""
        with self._get_connection() as conn:
            if user_id:
                cursor = conn.execute(
                    "SELECT id, user_id, fact, category, source_session_id, created_at FROM memories WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, user_id, fact, category, source_session_id, created_at FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def delete_memory(self, memory_id: int) -> bool:
        """Deletes a specific memory by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def clear_all_memories(self) -> int:
        """Clears all stored memories."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories")
            return cursor.rowcount

    # --- Self-Training Dataset Storage ---
    def log_training_pair(self, instruction: str, response: str, source: str = "chat") -> Optional[int]:
        """Logs an instruction-response turn into the self-learning training dataset."""
        inst_clean = instruction.strip()
        resp_clean = response.strip()
        if len(inst_clean) < 3 or len(resp_clean) < 3:
            return None
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO training_pairs (instruction, response, source, created_at) VALUES (?, ?, ?, ?)",
                (inst_clean, resp_clean, source, now)
            )
            return cursor.lastrowid

    def list_training_pairs(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Retrieves collected training pairs."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, instruction, response, source, created_at FROM training_pairs ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_training_pairs(self) -> int:
        """Returns the number of collected training pairs."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM training_pairs")
            row = cursor.fetchone()
            return row["count"] if row else 0

# Global database instance
db = Database()
