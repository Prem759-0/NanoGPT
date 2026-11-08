"""
Moon AI — Daily GitHub Streak & Architecture Synchronization Tool
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Usage:
    python scripts/daily_streak_update.py
    python scripts/daily_streak_update.py --note "Completed Stage 2.5 benchmark verification"

This script automatically:
1. Records a verified engineering milestone with timestamp in docs/DAILY_DEVELOPMENT_LOG.md.
2. Formats a high-standard conventional commit message.
3. Stages and commits the update under Prem Gaikwad's authenticated author credentials.
4. Pushes cleanly to origin main to maintain an unbroken GitHub contribution streak.
"""

import os
import sys
import argparse
import datetime
import subprocess

DAILY_LOG_FILE = os.path.join("docs", "DAILY_DEVELOPMENT_LOG.md")


def get_current_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_current_date() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def ensure_log_file_exists():
    os.makedirs("docs", exist_ok=True)
    if not os.path.isfile(DAILY_LOG_FILE):
        header = (
            "# 🌙 MOON AI — DAILY ARCHITECTURAL DEVELOPMENT & STREAK LOG\n"
            "**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)\n"
            "**Tracking**: Sovereign AI milestones, zero-teacher scratch initialization, and active GitHub development streak.\n\n"
            "---\n\n"
            "| Date (UTC) | Milestone / Focus | Status | Git Sync |\n"
            "| :--- | :--- | :--- | :--- |\n"
        )
        with open(DAILY_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)


def append_streak_entry(note: str) -> str:
    ensure_log_file_exists()
    ts = get_current_timestamp()
    date_str = get_current_date()
    entry_line = f"| {date_str} | {note} | `ACTIVE_VERIFIED` | `{ts}` |\n"
    
    with open(DAILY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry_line)
        
    return date_str


def run_git_push(commit_msg: str) -> bool:
    print(f"[*] Staging {DAILY_LOG_FILE}...")
    subprocess.run(["git", "add", DAILY_LOG_FILE], check=True)
    
    print(f"[*] Committing: {commit_msg}")
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    
    print("[*] Pushing to origin main...")
    res = subprocess.run(["git", "push", "origin", "main"])
    if res.returncode == 0:
        print("  GitHub Streak successfully synchronized to origin/main!")
        return True
    else:
        print("  Push failed. Check network and git credentials.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Moon AI Daily GitHub Streak Synchronizer")
    parser.add_argument(
        "--note",
        type=str,
        default="Daily sovereign pipeline audit, README refinement, and architecture synchronization",
        help="Description of the engineering milestone completed today."
    )
    args = parser.parse_args()

    date_str = append_streak_entry(args.note)
    commit_msg = f"docs(streak): {date_str} sovereign progress - {args.note}"
    success = run_git_push(commit_msg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
