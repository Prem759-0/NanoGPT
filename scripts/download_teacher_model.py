#!/usr/bin/env python3
"""
Moon AI — Public Teacher Model Downloader CLI
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Usage:
  python scripts/download_teacher_model.py --check-only
  python scripts/download_teacher_model.py --dry-run
  python scripts/download_teacher_model.py
"""

import os
import sys
import argparse
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from teacher.download.downloader import (
    PublicModelDownloader,
    QUALIFIED_TEACHER_MODELS,
    PRIMARY_TEACHER_KEY,
    DEFAULT_TEACHER_DIR
)


def main():
    parser = argparse.ArgumentParser(description="Moon AI Public Teacher Model Downloader")
    parser.add_argument("--model-key", default=PRIMARY_TEACHER_KEY, choices=list(QUALIFIED_TEACHER_MODELS.keys()), help="Model key")
    parser.add_argument("--dest-dir", default=DEFAULT_TEACHER_DIR, help="Destination directory for teacher weights")
    parser.add_argument("--check-only", action="store_true", help="Check local status and disk space without downloading")
    parser.add_argument("--dry-run", action="store_true", help="Verify remote URLs and headers without downloading payload")
    args = parser.parse_args()

    print("=" * 65)
    print("   MOON AI -- PUBLIC TEACHER MODEL DOWNLOADER")
    print("   Chief ML Systems Architect: Prem Gaikwad (India)")
    print("   Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)")
    print("=" * 65)

    downloader = PublicModelDownloader(
        dest_dir=args.dest_dir,
        model_key=args.model_key,
        allow_network_download=not args.check_only
    )

    info = downloader.model_info
    print(f"\nTarget Model:   {info['display_name']}")
    print(f"Repository:     {info['repo_id']}")
    print(f"License:        {info['license']}")
    print(f"Commit SHA:     {info['commit']}")
    print(f"Quantization:   {info['quantization']}")
    print(f"Total Size:     {info['total_size_bytes'] / (1024**3):.2f} GB ({info['total_size_bytes']} bytes)")
    print(f"Target Dir:     {downloader.dest_dir}")

    # Check storage
    has_space, avail, req = downloader.check_disk_space()
    print(f"\nStorage Assessment:")
    print(f"  • Available Disk Space: {avail / (1024**3):.2f} GB")
    print(f"  • Required Capacity:    {req / (1024**3):.2f} GB (incl. 2 GB safety buffer)")
    print(f"  • Status:               {'SUFFICIENT' if has_space else 'INSUFFICIENT'}")

    if not has_space:
        print("\n[ERROR] Insufficient free space to download teacher model.")
        sys.exit(1)

    # Check local shards
    existing = downloader.verify_existing_shards()
    print(f"\nLocal Checkpoint Status:")
    for s in existing["shards"]:
        status_label = "VERIFIED" if (s["exists"] and s["size_match"] and s["sha256_match"]) else ("MISSING" if not s["exists"] else "CORRUPTED")
        print(f"  * {s['filename']}: {status_label}")

    if existing["complete"]:
        print("\n[OK] Teacher model is already completely downloaded and cryptographically verified!")
        sys.exit(0)

    if args.check_only:
        print("\n[INFO] Check-only mode completed. Shards are pending download.")
        sys.exit(0)

    if args.dry_run:
        print("\nVerifying remote URLs (HTTP HEAD)...")
        for shard in info["shards"]:
            req = urllib.request.Request(shard["url"], method="HEAD", headers={"User-Agent": "MoonAI-Downloader/2.1.5"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    print(f"  * {shard['filename']}: HTTP {resp.status} (Length: {resp.headers.get('Content-Length')})")
            except Exception as e:
                print(f"  * {shard['filename']}: FAILED ({e})")
                sys.exit(1)
        print("\n[OK] Dry run successful: Remote files accessible without authentication.")
        sys.exit(0)

    # Start download
    print("\nStarting unauthenticated download...")
    last_reported = 0

    def progress(shard_name, cur, total):
        nonlocal last_reported
        pct = (cur / total) * 100
        if int(pct) >= last_reported + 10 or cur == total:
            last_reported = int(pct)
            print(f"[{shard_name}] {pct:.1f}% ({cur / (1024**2):.1f} MB / {total / (1024**2):.1f} MB)")

    paths = downloader.download_model(progress_callback=progress)
    print(f"\n[OK] Download completed! Verified shards saved to:")
    for p in paths:
        print(f"  * {p}")


if __name__ == "__main__":
    main()
