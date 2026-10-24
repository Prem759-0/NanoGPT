import urllib.request
import json
import psycopg
import os

print("--- 1. Testing FastAPI Session Creation ---")
req_data = json.dumps({'title': 'Neon + Clerk E2E Verification', 'user_id': 'clerk_user_live_test'}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/api/sessions', data=req_data, headers={'Content-Type': 'application/json'}, method='POST')

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    session_id = res['session_id']
    print(f"Created Session via FastAPI: id={session_id}, title={res['title']}")

print("\n--- 2. Testing FastAPI Session Fetch ---")
with urllib.request.urlopen('http://127.0.0.1:8000/api/sessions?user_id=clerk_user_live_test') as resp:
    sessions = json.loads(resp.read().decode('utf-8'))
    print(f"Fetched {len(sessions)} session(s) via FastAPI for 'clerk_user_live_test':")
    for s in sessions:
        print(f"  - {s['id']}: {s['title']}")

print("\n--- 3. Direct Query from Neon PostgreSQL Cloud ---")
db_url = 'postgresql://neondb_owner:npg_mTred7vp8DfY@ep-tiny-moon-b4hw7tzr-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require'
with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT id, user_id, title, created_at FROM sessions WHERE id = %s;', (session_id,))
        row = cur.fetchone()
        print(f"Direct proof from Neon PostgreSQL:")
        print(f"  ID:        {row[0]}")
        print(f"  User ID:   {row[1]}")
        print(f"  Title:     {row[2]}")
        print(f"  Timestamp: {row[3]}")

        cur.execute("SELECT count(*) FROM sessions;")
        total_sessions = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM messages;")
        total_messages = cur.fetchone()[0]
        print(f"\nNeon Cloud DB Stats:")
        print(f"  Total sessions stored in Neon: {total_sessions}")
        print(f"  Total messages stored in Neon: {total_messages}")

print("\n=== ALL END-TO-END TESTS PASSED ON NEON POSTGRESQL! ===")
