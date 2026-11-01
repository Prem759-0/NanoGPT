"""
datasets/tools/collect_images.py
Automated Collection, Normalization, Deduplication, and Hardened Ingestion for Moon Image.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
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

from datasets.tools.normalize_images import normalize_single_image
from datasets.tools.deduplicate_images import compute_sha256, compute_dhash, hamming_distance
from datasets.tools.provenance_validator import (
    validate_provenance_record,
    VALID_IMAGE_CATEGORIES,
    PERMISSIBLE_LICENSES
)

SCHEMA_VERSION = "2.0.2"

class ImageCollector:
    def __init__(
        self,
        base_dir: str = PROJECT_ROOT,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        target_resolution: int = 256
    ):
        self.base_dir = base_dir
        self.raw_dir = raw_dir or os.path.join(base_dir, "data", "raw", "images")
        self.processed_dir = processed_dir or os.path.join(base_dir, "data", "images")
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
        owner_declaration: Optional[Dict[str, Any]] = None,
        license_evidence: Optional[Dict[str, Any]] = None,
        attribution: Optional[str] = None,
        split: str = "train",
        phash_threshold: int = 4
    ) -> Dict[str, Any]:
        """
        Ingests a single image file with fail-closed provenance validation.
        """
        if not os.path.isfile(src_file_path):
            raise FileNotFoundError(f"Source image not found: {src_file_path}")

        # Compute preliminary raw hashes
        raw_sha = compute_sha256(src_file_path)
        if raw_sha in self.existing_sha:
            raise ValueError(f"Exact duplicate detected! Matches existing record '{self.existing_sha[raw_sha]}' (SHA256: {raw_sha})")

        raw_dhash = compute_dhash(src_file_path)
        for ex_dh, ex_id in self.existing_dhash.items():
            dist = hamming_distance(raw_dhash, ex_dh)
            if dist <= phash_threshold:
                raise ValueError(
                    f"Near-duplicate detected! dHash distance={dist} (threshold={phash_threshold}) "
                    f"to existing record '{ex_id}'"
                )

        # Build candidate record for preliminary schema validation
        rec_id = f"img_{category}_{raw_sha[:10]}"
        candidate = {
            "id": rec_id,
            "caption": caption.strip(),
            "category": category.strip().lower(),
            "creator": creator.strip(),
            "source_url": source_url.strip(),
            "license": license_id.strip(),
            "attribution": (attribution or f"Image by {creator} via {source_url} ({license_id})").strip(),
            "split": split.strip().lower(),
            "is_synthetic": False,
            "sha256": raw_sha,
            "media_path": src_file_path
        }
        if owner_declaration:
            candidate["owner_declaration"] = owner_declaration
        if license_evidence:
            candidate["license_evidence"] = license_evidence

        # Validate provenance dimensions
        audit = validate_provenance_record(
            record=candidate,
            modality="image",
            base_dir=self.base_dir,
            allow_synthetic=False,
            verify_file=False
        )
        if not audit["is_valid"]:
            raise ValueError(f"Image provenance validation failed: {'; '.join(audit['errors'])}")
        if not audit["dimensions"]["license_verified"] or not audit["dimensions"]["ownership_verified"]:
            raise ValueError(f"Ingestion blocked: {audit['status']} ({'; '.join(audit['errors'])})")

        # Archive raw copy
        ext = os.path.splitext(src_file_path)[1].lower() or ".jpg"
        raw_filename = f"{rec_id}_raw{ext}"
        archived_raw_path = os.path.join(self.raw_dir, raw_filename)
        if os.path.abspath(src_file_path) != os.path.abspath(archived_raw_path):
            shutil.copy2(src_file_path, archived_raw_path)

        # Normalize and save processed copy
        proc_filename = f"{rec_id}.png"
        proc_full_path = os.path.join(self.processed_dir, proc_filename)
        norm_meta = normalize_single_image(
            src_path=src_file_path,
            dest_path=proc_full_path,
            target_size=self.target_resolution
        )

        rel_media_path = os.path.relpath(proc_full_path, self.base_dir).replace("\\", "/")

        final_record = {
            "schema_version": SCHEMA_VERSION,
            "id": rec_id,
            "media_path": rel_media_path,
            "image_path": rel_media_path,
            "caption": candidate["caption"],
            "category": candidate["category"],
            "source_url": candidate["source_url"],
            "creator": candidate["creator"],
            "license": candidate["license"],
            "attribution": candidate["attribution"],
            "sha256": norm_meta["sha256"],
            "dhash": norm_meta["dhash"],
            "split": candidate["split"],
            "width": norm_meta["width"],
            "height": norm_meta["height"],
            "channels": norm_meta["channels"],
            "original_width": norm_meta["original_width"],
            "original_height": norm_meta["original_height"],
            "is_synthetic": False,
            "provenance_status": "VERIFIED_PRODUCTION_ELIGIBLE"
        }
        if owner_declaration:
            final_decl = dict(owner_declaration)
            final_decl["asset_sha256"] = norm_meta["sha256"]
            final_record["owner_declaration"] = final_decl
        if license_evidence:
            final_record["license_evidence"] = license_evidence

        self.existing_sha[norm_meta["sha256"]] = rec_id
        self.existing_dhash[norm_meta["dhash"]] = rec_id

        return final_record

def append_to_manifest(record: Dict[str, Any], manifest_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
