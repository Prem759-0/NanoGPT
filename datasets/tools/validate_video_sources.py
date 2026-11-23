"""
datasets/tools/validate_video_sources.py
Licensing, Provenance, and Motion Metadata Validation for Moon Video.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Enforces:
1. Permissible open video licenses.
2. Verified creator and verifiable source URL.
3. Informative scene caption + motion description.
4. Valid camera motion label matching architectural vocabulary.
5. Rejection of synthetic still-image animations in genuine manifests.
"""
import os
import sys
from typing import Dict, Any, List, Tuple

VALID_VIDEO_LICENSES = {
    "CC0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "OpenData",
    "PublicDomain",
    "CustomSovereign"
}

VALID_CAMERA_MOTIONS = {
    "static",
    "orbit_left",
    "orbit_right",
    "dolly_in",
    "dolly_out",
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "crane_up",
    "crane_down"
}

VALID_VIDEO_CATEGORIES = {
    "landscapes",
    "architecture",
    "products",
    "technology",
    "people",
    "animals",
    "vehicles",
    "food",
    "illustration_design",
    "abstract_scientific"
}

SUSPICIOUS_CREATORS = {"unknown", "anonymous", "none", "n/a", "internet", "various", "ai"}

def validate_video_source_record(record: Dict[str, Any], allow_synthetic: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates a single video metadata record.
    Returns (is_valid, list_of_errors).
    """
    errors = []

    # 1. ID Check
    rec_id = str(record.get("id", "")).strip()
    if not rec_id:
        errors.append("Missing required 'id' field.")

    # 2. Category Check
    category = str(record.get("category", "")).strip().lower()
    if not category:
        errors.append("Missing required 'category' field.")
    elif category not in VALID_VIDEO_CATEGORIES:
        errors.append(f"Invalid category '{category}'. Must be one of: {sorted(list(VALID_VIDEO_CATEGORIES))}")

    # 3. License Check
    license_id = str(record.get("license", "")).strip()
    if not license_id:
        errors.append("Missing required 'license' field.")
    elif license_id not in VALID_VIDEO_LICENSES:
        errors.append(
            f"Unverified or impermissible video license '{license_id}'. "
            f"Permitted: {sorted(list(VALID_VIDEO_LICENSES))}"
        )

    # 4. Creator & Source URL
    creator = str(record.get("creator", "")).strip()
    if not creator:
        errors.append("Missing required 'creator' field.")
    elif creator.lower() in SUSPICIOUS_CREATORS and license_id != "PublicDomain":
        errors.append(f"Suspicious generic creator '{creator}'. Must provide verifiable attribution.")

    source_url = str(record.get("source_url", "")).strip()
    if not source_url:
        errors.append("Missing required 'source_url' field.")
    elif not (source_url.startswith("http://") or source_url.startswith("https://") or source_url.startswith("owner://")):
        errors.append(f"Invalid source_url protocol '{source_url}'. Must start with http://, https://, or owner://.")
    elif "github.com/prem759-0/nanogpt/tree/main/data" in source_url.lower():
        errors.append(
            f"Fabricated internal repo URL detected: '{source_url}'. "
            "Self-referential local data folders cannot serve as asset-level source URLs."
        )

    # 5. Scene Caption & Motion Description
    caption = str(record.get("caption", "")).strip()
    if not caption or len(caption) < 10:
        errors.append("Caption is missing or shorter than 10 characters.")

    motion_desc = str(record.get("motion_description", "")).strip()
    if not motion_desc:
        errors.append("Missing required 'motion_description' field.")

    # 6. Camera Motion
    cam_motion = str(record.get("camera_motion", "static")).strip().lower()
    if cam_motion not in VALID_CAMERA_MOTIONS:
        errors.append(f"Invalid camera_motion '{cam_motion}'. Allowed: {sorted(list(VALID_CAMERA_MOTIONS))}")

    # 7. Synthetic Check
    is_synth = record.get("is_synthetic", False)
    if is_synth and not allow_synthetic:
        split = record.get("split", "train").lower()
        if split in ("train", "val", "test"):
            errors.append(
                "Record is marked as synthetic. Synthetic animations cannot be admitted to genuine video training manifests."
            )

    return len(errors) == 0, errors
