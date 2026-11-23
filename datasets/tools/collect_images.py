"""
datasets/tools/collect_images.py
Image Ingestion and Manifest Collection Engine for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Ingestion Workflow:
1. Validates metadata (licensing, creator, source URL, caption, category).
2. Verifies image integrity and extracts perceptual & SHA-256 hashes.
3. Checks for exact or near-duplicates.
4. Archives original raw file into data/raw/images/.
5. Normalizes processed training copy into data/images/.
6. Generates version 2.0.0 JSONL manifest records.
7. Produces category taxonomy distribution report.
"""
import os
import sys
import json
import shutil
import argparse
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.validate_image_sources import validate_image_source_record, VALID_IMAGE_CATEGORIES
from datasets.tools.normalize_images import normalize_single_image
from datasets.tools.deduplicate_images import compute_sha256, compute_dhash, hamming_distance

SCHEMA_VERSION = "2.0.0"

class ImageCollector:
    def __init__(
        self,
        base_dir: str = PROJECT_ROOT,
        raw_dir: str = "data/raw/images",
        processed_dir: str = "data/images",
        target_resolution: int = 256
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, raw_dir)
        self.processed_dir = os.path.join(self.base_dir, processed_dir)
        self.target_resolution = target_resolution

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.existing_sha: Dict[str, str] = {}
        self.existing_dhash: Dict[str, str] = {}

    def register_existing_manifest(self, manifest_path: str):
        """Loads existing hashes to prevent duplicate ingestion."""
        if not os.path.isfile(manifest_path):
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                    sha = data.get("sha256")
                    dh = data.get("dhash")
                    rec_id = data.get("id")
                    if sha:
                        self.existing_sha[sha] = rec_id
                    if dh:
                        self.existing_dhash[dh] = rec_id
                except Exception:
                    pass

    def ingest_image(
        self,
        src_file_path: str,
        caption: str,
        category: str,
        creator: str,
        source_url: str,
        license_id: str,
        attribution: Optional[str] = None,
        split: str = "train",
        phash_threshold: int = 4
    ) -> Dict[str, Any]:
        """
        Ingests a single image file with strict validation.
        Returns a valid manifest record dict or raises an exception.
        """
        if not os.path.isfile(src_file_path):
            raise FileNotFoundError(f"Source image not found: {src_file_path}")

        # Compute preliminary raw hashes
        raw_sha = compute_sha256(src_file_path)
        if raw_sha in self.existing_sha:
            raise ValueError(f"Exact duplicate detected! Matches existing record '{self.existing_sha[raw_sha]}' (SHA256: {raw_sha})")

        # Preliminary dHash check
        raw_dhash = compute_dhash(src_file_path)
        for ex_dh, ex_id in self.existing_dhash.items():
            dist = hamming_distance(raw_dhash, ex_dh)
            if dist <= phash_threshold:
                raise ValueError(
                    f"Near-duplicate detected! dHash distance={dist} (threshold={phash_threshold}) "
                    f"to existing record '{ex_id}'"
                )

        # Build candidate record for validation
        candidate = {
            "id": f"img_{category}_{raw_sha[:10]}",
            "caption": caption.strip(),
            "category": category.strip().lower(),
            "creator": creator.strip(),
            "source_url": source_url.strip(),
            "license": license_id.strip(),
            "attribution": (attribution or f"Image by {creator} via {source_url} ({license_id})").strip(),
            "split": split.strip().lower()
        }

        # Validate metadata
        is_valid, errors = validate_image_source_record(candidate)
        if not is_valid:
            raise ValueError(f"Metadata validation failed: {'; '.join(errors)}")

        # Archive raw copy
        ext = os.path.splitext(src_file_path)[1].lower() or ".jpg"
        raw_filename = f"{candidate['id']}_raw{ext}"
        archived_raw_path = os.path.join(self.raw_dir, raw_filename)
        if os.path.abspath(src_file_path) != os.path.abspath(archived_raw_path):
            shutil.copy2(src_file_path, archived_raw_path)

        # Normalize and save processed copy
        proc_filename = f"{candidate['id']}.png"
        proc_full_path = os.path.join(self.processed_dir, proc_filename)
        norm_meta = normalize_single_image(
            src_path=src_file_path,
            dest_path=proc_full_path,
            target_size=self.target_resolution
        )

        # Relative path for manifest portability
        rel_media_path = os.path.relpath(proc_full_path, self.base_dir).replace("\\", "/")

        final_record = {
            "schema_version": SCHEMA_VERSION,
            "id": candidate["id"],
            "media_path": rel_media_path,
            "caption": candidate["caption"],
            "category": candidate["category"],
            "source_url": candidate["source_url"],
            "creator": candidate["creator"],
            "license": candidate["license"],
            "attribution": candidate["attribution"],
            "sha256": norm_meta["sha256"],
            "dhash": norm_meta["dhash"],
            "split": candidate["split"],
            "width": norm_meta["normalized_width"],
            "height": norm_meta["normalized_height"],
            "channels": norm_meta["channels"],
            "original_width": norm_meta["original_width"],
            "original_height": norm_meta["original_height"],
            "is_synthetic": False
        }

        # Update cache
        self.existing_sha[norm_meta["sha256"]] = candidate["id"]
        self.existing_dhash[norm_meta["dhash"]] = candidate["id"]

        return final_record

def generate_category_distribution_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates distribution report across the 10 balanced categories."""
    dist = {cat: 0 for cat in VALID_IMAGE_CATEGORIES}
    splits = {"train": 0, "val": 0, "test": 0, "other": 0}

    for r in records:
        cat = r.get("category", "unknown")
        if cat in dist:
            dist[cat] += 1
        s = r.get("split", "train").lower()
        if s in splits:
            splits[s] += 1
        else:
            splits["other"] += 1

    total = len(records)
    pcts = {cat: round((count / total) * 100, 1) if total > 0 else 0.0 for cat, count in dist.items()}

    return {
        "total_images": total,
        "category_counts": dist,
        "category_percentages": pcts,
        "splits": splits
    }

