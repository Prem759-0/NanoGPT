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
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
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

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, timestamp);
            """)

    def create_session(self, title: str = "New Chat", session_id: Optional[str] = None) -> str:
        s_id = session_id or str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (s_id, title, now, now)
            )
        return s_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
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

    def add_message(self, session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> int:
        now = datetime.utcnow().isoformat()
        meta_str = json.dumps(meta or {})
        
        # Ensure session exists
        if not self.get_session(session_id):
            title = content[:30] + "..." if len(content) > 30 else content
            self.create_session(title=title, session_id=session_id)

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

# Global database instance
db = Database()
