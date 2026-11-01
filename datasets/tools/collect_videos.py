"""
datasets/tools/collect_videos.py
Video Ingestion and Manifest Collection Engine for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Ingests genuine video files (.mp4, .webm) into version 2.0.0 manifests.
Guarantees real temporal frames, timestamps, and verifiable licensing.
"""
import os
import sys
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.inspect_video import inspect_video_file
from datasets.tools.extract_video_frames import extract_ordered_frames
from datasets.tools.validate_video_sources import validate_video_source_record
from datasets.tools.deduplicate_images import compute_sha256

SCHEMA_VERSION = "2.0.0"

class VideoCollector:
    def __init__(
        self,
        base_dir: str = PROJECT_ROOT,
        raw_dir: str = "data/raw/videos",
        processed_dir: str = "data/videos"
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, raw_dir)
        self.processed_dir = os.path.join(self.base_dir, processed_dir)

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.existing_sha: Dict[str, str] = {}

    def register_existing_manifest(self, manifest_path: str):
        """Loads existing hashes to prevent duplicate video ingestion."""
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
                    rec_id = data.get("id")
                    if sha:
                        self.existing_sha[sha] = rec_id
                except Exception:
                    pass

    def ingest_video(
        self,
        src_video_path: str,
        caption: str,
        motion_description: str,
        category: str,
        creator: str,
        source_url: str,
        license_id: str,
        camera_motion: str = "static",
        attribution: Optional[str] = None,
        split: str = "train",
        min_duration_sec: float = 1.0,
        min_frames: int = 8
    ) -> Dict[str, Any]:
        """
        Ingests a genuine video clip with full validation.
        """
        if not os.path.isfile(src_video_path):
            raise FileNotFoundError(f"Source video not found: {src_video_path}")

        # 1. Exact SHA-256 duplicate check
        video_sha = compute_sha256(src_video_path)
        if video_sha in self.existing_sha:
            raise ValueError(f"Duplicate video! Matches existing record '{self.existing_sha[video_sha]}' (SHA-256: {video_sha})")

        # 2. Inspect video technical specs
        inspect_data = inspect_video_file(src_video_path, min_duration_sec=min_duration_sec, min_frames=min_frames)

        # 3. Test frame extraction to verify readable stream and timestamps
        frame_data = extract_ordered_frames(src_video_path, target_num_frames=min_frames)

        # 4. Build candidate record for metadata validation
        rec_id = f"vid_{category}_{video_sha[:10]}"
        candidate = {
            "id": rec_id,
            "caption": caption.strip(),
            "motion_description": motion_description.strip(),
            "category": category.strip().lower(),
            "creator": creator.strip(),
            "source_url": source_url.strip(),
            "license": license_id.strip(),
            "attribution": (attribution or f"Video by {creator} via {source_url} ({license_id})").strip(),
            "camera_motion": camera_motion.strip().lower(),
            "split": split.strip().lower(),
            "is_synthetic": False
        }

        is_valid, errors = validate_video_source_record(candidate)
        if not is_valid:
            raise ValueError(f"Video metadata validation failed: {'; '.join(errors)}")

        # 5. Archive raw original video
        ext = os.path.splitext(src_video_path)[1].lower() or ".mp4"
        raw_dest = os.path.join(self.raw_dir, f"{rec_id}_raw{ext}")
        if os.path.abspath(src_video_path) != os.path.abspath(raw_dest):
            shutil.copy2(src_video_path, raw_dest)

        # 6. Copy to processed videos directory
        proc_dest = os.path.join(self.processed_dir, f"{rec_id}{ext}")
        if os.path.abspath(src_video_path) != os.path.abspath(proc_dest):
            shutil.copy2(src_video_path, proc_dest)

        rel_video_path = os.path.relpath(proc_dest, self.base_dir).replace("\\", "/")

        final_record = {
            "schema_version": SCHEMA_VERSION,
            "id": rec_id,
            "media_path": rel_video_path,
            "video_path": rel_video_path,
            "caption": candidate["caption"],
            "motion_description": candidate["motion_description"],
            "category": candidate["category"],
            "source_url": candidate["source_url"],
            "creator": candidate["creator"],
            "license": candidate["license"],
            "attribution": candidate["attribution"],
            "camera_motion": candidate["camera_motion"],
            "sha256": video_sha,
            "split": candidate["split"],
            "width": inspect_data["width"],
            "height": inspect_data["height"],
            "fps": inspect_data["fps"],
            "duration_sec": inspect_data["duration_sec"],
            "num_frames": inspect_data["num_frames"],
            "timestamps": frame_data["timestamps"][:min_frames],
            "codec": inspect_data["codec"],
            "is_synthetic": False
        }

        self.existing_sha[video_sha] = rec_id
        return final_record

