"""
datasets/tools/validate_image_sources.py
Licensing and Provenance Validation for Moon AI Image Datasets.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Enforces strict compliance:
1. Permissible open licenses only.
2. Verified creator and verifiable source URL attribution.
3. Quality captions (rejects empty/generic prefixes).
4. Balanced category taxonomy validation.
5. Strict rejection of synthetic fixtures from genuine production manifests.
"""
import os
import sys
import re
from typing import Dict, Any, List, Tuple

VALID_IMAGE_LICENSES = {
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

VALID_IMAGE_CATEGORIES = {
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

GENERIC_CAPTION_PATTERNS = [
    r"^(?:an?\s+)?image(?:\s+of)?$",
    r"^(?:an?\s+)?photo(?:graph)?(?:\s+of)?$",
    r"^(?:an?\s+)?picture(?:\s+of)?$",
    r"^test(?:\s+image)?$",
    r"^untitled$",
    r"^sample(?:\s+image)?$"
]

SUSPICIOUS_CREATORS = {"unknown", "anonymous", "none", "n/a", "internet", "various", "ai"}

def validate_image_source_record(record: Dict[str, Any], allow_synthetic: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates a single image metadata record against licensing and hygiene rules.
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
    elif category not in VALID_IMAGE_CATEGORIES:
        errors.append(f"Invalid category '{category}'. Must be one of: {sorted(list(VALID_IMAGE_CATEGORIES))}")

    # 3. License Check
    license_id = str(record.get("license", "")).strip()
    if not license_id:
        errors.append("Missing required 'license' field.")
    elif license_id not in VALID_IMAGE_LICENSES:
        errors.append(
            f"Unverified or impermissible license '{license_id}'. "
            f"Permitted licenses: {sorted(list(VALID_IMAGE_LICENSES))}"
        )

    # 4. Creator & Provenance Check
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

    # 5. Caption Check
    caption = str(record.get("caption", "")).strip()
    if not caption:
        errors.append("Caption is empty.")
    elif len(caption) < 10:
        errors.append(f"Caption too short ({len(caption)} chars; min 10 chars required): '{caption}'")
    else:
        for pat in GENERIC_CAPTION_PATTERNS:
            if re.match(pat, caption, re.IGNORECASE):
                errors.append(f"Generic uninformative caption detected: '{caption}'")
                break

    # 6. Strict Synthetic Segregation
    is_synth = record.get("is_synthetic", False)
    if is_synth and not allow_synthetic:
        split = record.get("split", "train").lower()
        if split in ("train", "val", "test"):
            errors.append(
                "Record is marked as synthetic. Synthetic procedural fixtures are strictly prohibited "
                "from genuine production training/validation/test manifests."
            )

    return len(errors) == 0, errors
