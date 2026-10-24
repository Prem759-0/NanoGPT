import os
import psycopg

# Parse .env directly without third-party dependencies
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_mTred7vp8DfY@ep-tiny-moon-b4hw7tzr-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require")

print(f"Connecting to Neon DB at: {DATABASE_URL.split('@')[-1]}...")
with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        ver = cur.fetchone()[0]
        print(f"Connected to Neon PostgreSQL:\n  {ver}\n")
        
        # Create schema
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
        print("Schema verified and initialized on Neon PostgreSQL.")

        # Test insert session and message
        test_session_id = "test-neon-init-001"
        cur.execute("""
            INSERT INTO sessions (id, user_id, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP;
        """, (test_session_id, "test_user_neon", "Neon Production Verification Session"))

        cur.execute("""
            INSERT INTO messages (session_id, role, content, meta_json)
            VALUES (%s, %s, %s, %s);
        """, (test_session_id, "user", "Neon DB + Clerk Live Test", '{"model": "qwen-2.5-coder-7b"}'))

        cur.execute("""
            INSERT INTO messages (session_id, role, content, meta_json)
            VALUES (%s, %s, %s, %s);
        """, (test_session_id, "assistant", "Verified Neon Postgres connection and chat message persistence.", '{"status": "ok"}'))
        conn.commit()

        # Query back
        cur.execute("SELECT id, user_id, title FROM sessions WHERE id = %s;", (test_session_id,))
        session_row = cur.fetchone()
        print(f"Verified Session in Neon: {session_row}")

        cur.execute("SELECT role, content FROM messages WHERE session_id = %s;", (test_session_id,))
        msg_rows = cur.fetchall()
        print(f"Verified Messages in Neon ({len(msg_rows)} rows):")
        for r in msg_rows:
            print(f"  [{r[0]}] {r[1]}")

        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [r[0] for r in cur.fetchall()]
        print(f"\nAll Public Tables in Neon Database: {tables}")

print("\nNEON POSTGRESQL VERIFICATION COMPLETE & SUCCESSFUL!")
