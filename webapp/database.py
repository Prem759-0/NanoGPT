import os
import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nanogpt-lab")

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chats.db"))
DATABASE_URL = os.environ.get("DATABASE_URL")

class Database:
    """
    Sovereign Dual Database Engine for Grok / NanoGPT Lab.
    Supports:
    1. Neon PostgreSQL / Supabase (Serverless Cloud Postgres) when DATABASE_URL is set.
    2. Local SQLite with WAL mode and zero external dependencies when offline.
    """

    def __init__(self, db_path: str = DB_PATH, db_url: Optional[str] = DATABASE_URL):
        self.db_path = db_path
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.is_postgres = bool(self.db_url and (self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")))
        
        if not self.is_postgres:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        self._init_db()

    def _get_connection(self):
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
                # Handle postgres:// vs postgresql:// compatibility
                conn_str = self.db_url
                if conn_str.startswith("postgres://"):
                    conn_str = "postgresql://" + conn_str[len("postgres://"):]
                conn = psycopg.connect(conn_str, row_factory=dict_row)
                return conn
            except Exception as e:
                logger.warning(f"PostgreSQL connection to Neon failed ({e}). Falling back to local SQLite.")
                self.is_postgres = False
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                return self._get_sqlite_connection()
        else:
            return self._get_sqlite_connection()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        if self.is_postgres:
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS sessions (
                                id VARCHAR(64) PRIMARY KEY,
                                user_id VARCHAR(128) DEFAULT 'guest',
                                title TEXT NOT NULL,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );

                            CREATE TABLE IF NOT EXISTS messages (
                                id SERIAL PRIMARY KEY,
                                session_id VARCHAR(64) NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
                                role VARCHAR(32) NOT NULL,
                                content TEXT NOT NULL,
                                meta_json TEXT DEFAULT '{}',
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );

                            CREATE TABLE IF NOT EXISTS memories (
                                id SERIAL PRIMARY KEY,
                                user_id VARCHAR(128) DEFAULT 'guest',
                                fact TEXT NOT NULL UNIQUE,
                                category VARCHAR(64) DEFAULT 'general',
                                source_session_id VARCHAR(64),
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );

                            CREATE TABLE IF NOT EXISTS training_pairs (
                                id SERIAL PRIMARY KEY,
                                instruction TEXT NOT NULL,
                                response TEXT NOT NULL,
                                source VARCHAR(64) DEFAULT 'chat',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """)
                    conn.commit()
                return
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL schema: {e}. Falling back to SQLite.")
                self.is_postgres = False

        # SQLite Schema
        with self._get_sqlite_connection() as conn:
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

    def _execute(self, query: str, params: tuple = ()):
        """Normalizes and executes a query across PostgreSQL and SQLite."""
        if self.is_postgres:
            pg_query = query.replace("?", "%s")
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    return cur
        else:
            with self._get_connection() as conn:
                return conn.execute(query, params)

    def create_session(self, title: str = "New Chat", session_id: Optional[str] = None, user_id: str = "guest") -> str:
        s_id = session_id or str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                        (s_id, user_id or "guest", title, now, now)
                    )
                conn.commit()
        else:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (s_id, user_id or "guest", title, now, now)
                )
        return s_id

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        if user_id in ["guest", "guest_user"]:
                            cur.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE user_id IN ('guest', 'guest_user') ORDER BY updated_at DESC")
                        else:
                            cur.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE user_id = %s ORDER BY updated_at DESC", (user_id,))
                    else:
                        cur.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
        else:
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
                    cursor = conn.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
                return [dict(row) for row in cursor.fetchall()]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = %s", (session_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, session_id, role, content, meta_json, timestamp 
                        FROM (
                            SELECT id, session_id, role, content, meta_json, timestamp 
                            FROM messages 
                            WHERE session_id = %s 
                            ORDER BY id DESC 
                            LIMIT %s
                        ) sub 
                        ORDER BY id ASC
                        """,
                        (session_id, limit)
                    )
                    rows = cur.fetchall()
        else:
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

        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO messages (session_id, role, content, meta_json, timestamp) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                        (session_id, role, content, meta_str, now)
                    )
                    msg_id = cur.fetchone()["id"]
                    cur.execute("UPDATE sessions SET updated_at = %s WHERE id = %s", (now, session_id))
                conn.commit()
                return msg_id
        else:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, role, content, meta_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, meta_str, now)
                )
                msg_id = cursor.lastrowid
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
                return msg_id

    def delete_session(self, session_id: str):
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                conn.commit()
        else:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def rename_session(self, session_id: str, new_title: str):
        now = datetime.utcnow().isoformat()
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE sessions SET title = %s, updated_at = %s WHERE id = %s", (new_title, now, session_id))
                conn.commit()
        else:
            with self._get_connection() as conn:
                conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (new_title, now, session_id))

    def add_memory(self, fact: str, category: str = "general", session_id: Optional[str] = None, user_id: str = "guest") -> Optional[int]:
        fact_clean = fact.strip()
        if not fact_clean:
            return None
        now = datetime.utcnow().isoformat()
        if self.is_postgres:
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO memories (user_id, fact, category, source_session_id, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (fact) DO NOTHING RETURNING id",
                            (user_id or "guest", fact_clean, category, session_id, now)
                        )
                        res = cur.fetchone()
                        conn.commit()
                        return res["id"] if res else None
            except Exception:
                return None
        else:
            with self._get_connection() as conn:
                try:
                    cursor = conn.execute(
                        "INSERT INTO memories (user_id, fact, category, source_session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                        (user_id or "guest", fact_clean, category, session_id, now)
                    )
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    return None

    def list_memories(self, limit: int = 100, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute("SELECT id, user_id, fact, category, source_session_id, created_at FROM memories WHERE user_id = %s ORDER BY id DESC LIMIT %s", (user_id, limit))
                    else:
                        cur.execute("SELECT id, user_id, fact, category, source_session_id, created_at FROM memories ORDER BY id DESC LIMIT %s", (limit,))
                    return [dict(r) for r in cur.fetchall()]
        else:
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
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                    rc = cur.rowcount
                conn.commit()
                return rc > 0
        else:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0

    def get_user_chat_stats(self, user_id: str) -> Dict[str, Any]:
        if self.is_postgres:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as total_sessions FROM sessions WHERE user_id = %s", (user_id,))
                    s_count = cur.fetchone()["total_sessions"]
                    cur.execute("""
                        SELECT COUNT(m.id) as total_messages 
                        FROM messages m 
                        JOIN sessions s ON m.session_id = s.id 
                        WHERE s.user_id = %s
                    """, (user_id,))
                    m_count = cur.fetchone()["total_messages"]
                    cur.execute("SELECT COUNT(*) as total_memories FROM memories WHERE user_id = %s", (user_id,))
                    mem_count = cur.fetchone()["total_memories"]
                    return {
                        "user_id": user_id,
                        "engine": "Neon PostgreSQL",
                        "total_sessions": s_count,
                        "total_messages": m_count,
                        "total_memories": mem_count
                    }
        else:
            with self._get_connection() as conn:
                s_count = conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)).fetchone()[0]
                m_count = conn.execute("""
                    SELECT COUNT(m.id) 
                    FROM messages m 
                    JOIN sessions s ON m.session_id = s.id 
                    WHERE s.user_id = ?
                """, (user_id,)).fetchone()[0]
                mem_count = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()[0]
                return {
                    "user_id": user_id,
                    "engine": "SQLite WAL",
                    "total_sessions": s_count,
                    "total_messages": m_count,
                    "total_memories": mem_count
                }

# Global singleton database instance
db = Database()
