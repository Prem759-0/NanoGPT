"""
datasets/tools/provenance_validator.py
Unified Hardened Provenance, Ownership, and Cryptographic File Validation Engine for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Stage 2.0.2:
Implements fail-closed adversarial security checks:
1. Metadata Completeness (explicit is_synthetic=false, no missing fields, no defaults).
2. Source Accessibility & Asset-Level Verification (rejects generic repos & fake URLs).
3. Licence Verification (requires explicit license_evidence; marks unverified as PENDING_MANUAL_VERIFICATION).
4. Ownership Verification (rejects bare 'CustomSovereign' / 'owner://' strings without full owner declaration).
5. Actual Byte Checksum & Safe Path Verification (recomputes SHA-256 from disk; prevents directory traversal).
6. Training Eligibility Gate (fails closed if any check fails).
"""
import os
import sys
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PERMISSIBLE_LICENSES = {
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

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm"}

VALID_IMAGE_CATEGORIES = {
    "landscapes", "architecture", "products", "technology", "people",
    "animals", "vehicles", "food", "illustration_design", "abstract_scientific"
}

VALID_CAMERA_MOTIONS = {
    "static", "orbit_left", "orbit_right", "dolly_in", "dolly_out",
    "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down",
    "crane_up", "crane_down"
}

SUSPICIOUS_CREATORS = {"unknown", "anonymous", "none", "n/a", "internet", "various", "ai"}

GENERIC_CAPTION_PATTERNS = [
    r"^(?:an?\s+)?image(?:\s+of)?$",
    r"^(?:an?\s+)?photo(?:graph)?(?:\s+of)?$",
    r"^(?:an?\s+)?picture(?:\s+of)?$",
    r"^test(?:\s+image)?$",
    r"^untitled$",
    r"^sample(?:\s+image)?$"
]

def validate_safe_path(
    rel_path: str,
    base_dir: str = PROJECT_ROOT,
    allowed_extensions: Optional[Set[str]] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Validates that rel_path resolves strictly within base_dir without directory traversal.
    Ensures safe extension.
    """
    if not rel_path or not isinstance(rel_path, str):
        return False, "", "Media path is empty or invalid."

    # Reject explicit traversal tokens
    if ".." in rel_path or "\\.." in rel_path or "/.." in rel_path:
        return False, "", f"Directory traversal detected in path: '{rel_path}'"

    # Reject absolute Windows paths or root paths
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":"):
        return False, "", f"Absolute paths are forbidden in manifests: '{rel_path}'"

    resolved = os.path.normpath(os.path.join(base_dir, rel_path))
    norm_base = os.path.normpath(base_dir)

    try:
        common = os.path.commonpath([norm_base, resolved])
        if common != norm_base:
            return False, "", f"Path escapes base directory: '{rel_path}'"
    except Exception as e:
        return False, "", f"Path resolution error: {e}"

    if allowed_extensions:
        ext = os.path.splitext(resolved)[1].lower()
        if ext not in allowed_extensions:
            return False, resolved, f"File extension '{ext}' is not permitted. Allowed: {sorted(list(allowed_extensions))}"

    return True, resolved, None

def compute_actual_file_sha256(filepath: str) -> str:
    """Reads actual file bytes from disk and computes SHA-256."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()

def verify_file_checksum(filepath: str, expected_sha256: str) -> Tuple[bool, str, Optional[str]]:
    """
    Verifies that the file at filepath exists and its actual byte SHA-256
    matches expected_sha256.
    """
    if not os.path.isfile(filepath):
        return False, "", f"Media file not found on disk: '{filepath}'"

    actual_sha = compute_actual_file_sha256(filepath)
    expected_norm = expected_sha256.strip().lower()

    if actual_sha != expected_norm:
        return False, actual_sha, f"SHA-256 mismatch! Manifest specifies '{expected_norm}', but actual file bytes yield '{actual_sha}'."

    return True, actual_sha, None

def validate_iso_timestamp(ts_str: str) -> bool:
    """Validates that a string is a valid ISO 8601 timestamp."""
    try:
        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return True
    except Exception:
        return False

def validate_provenance_record(
    record: Dict[str, Any],
    modality: str = "image",
    base_dir: str = PROJECT_ROOT,
    allow_synthetic: bool = False,
    verify_file: bool = True
) -> Dict[str, Any]:
    """
    Performs comprehensive 5-dimensional security, provenance, and integrity audit on a record.
    Returns audit status dictionary.
    """
    errors: List[str] = []
    dims = {
        "metadata_completeness": True,
        "source_accessibility": True,
        "license_verified": True,
        "ownership_verified": True,
        "safe_path_verified": True,
        "file_checksum_verified": True
    }

    # =========================================================================
    # DIMENSION 1: METADATA COMPLETENESS & SCHEMA INTEGRITY
    # =========================================================================
    rec_id = str(record.get("id", "")).strip()
    if not rec_id:
        errors.append("Missing required 'id' field.")
        dims["metadata_completeness"] = False

    # CRITICAL: is_synthetic MUST be explicitly present. Never default to False!
    if "is_synthetic" not in record:
        errors.append("Missing required 'is_synthetic' field. Production records must explicitly set 'is_synthetic: false'. Never defaulted.")
        dims["metadata_completeness"] = False
    elif record["is_synthetic"] is not False and not allow_synthetic:
        errors.append(
            "Synthetic procedural fixtures are strictly prohibited from production manifests. "
            "Synthetic animations cannot be admitted into production manifests. "
            "Record has 'is_synthetic: true' or non-boolean value. Synthetic fixtures must be quarantined in dev manifests."
        )
        dims["metadata_completeness"] = False

    creator = str(record.get("creator", "")).strip()
    if not creator:
        errors.append("Missing required 'creator' field.")
        dims["metadata_completeness"] = False
    elif creator.lower() in SUSPICIOUS_CREATORS:
        errors.append(f"Suspicious generic creator identity '{creator}'. Must provide traceable creator.")
        dims["metadata_completeness"] = False

    source_url = str(record.get("source_url", "")).strip()
    if not source_url:
        errors.append("Missing required 'source_url' field.")
        dims["metadata_completeness"] = False

    license_id = str(record.get("license", "")).strip()
    if not license_id:
        errors.append("Missing required 'license' field.")
        dims["metadata_completeness"] = False
    elif license_id not in PERMISSIBLE_LICENSES:
        errors.append(f"Unverified or impermissible license. Impermissible or unsupported license '{license_id}'. Permitted: {sorted(list(PERMISSIBLE_LICENSES))}")
        dims["metadata_completeness"] = False

    sha256_val = str(record.get("sha256", "")).strip().lower()
    if not sha256_val:
        errors.append("Missing required 'sha256' checksum field.")
        dims["metadata_completeness"] = False
    elif len(sha256_val) != 64 or not re.fullmatch(r"[0-9a-f]{64}", sha256_val):
        errors.append(f"Malformed SHA-256 checksum '{sha256_val}'. Must be exactly 64 lowercase hex characters.")
        dims["metadata_completeness"] = False

    rel_path = record.get("media_path") or record.get("image_path") or record.get("video_path")
    if not rel_path:
        errors.append("Missing required media path field ('media_path', 'image_path', or 'video_path').")
        dims["metadata_completeness"] = False

    # Modality specific checks
    caption = str(record.get("caption", "")).strip()
    if not caption:
        errors.append("Caption is missing or empty.")
        dims["metadata_completeness"] = False
    elif len(caption) < 10:
        errors.append(f"Caption too short ({len(caption)} chars; min 10 chars required): '{caption}'")
        dims["metadata_completeness"] = False
    else:
        for pat in GENERIC_CAPTION_PATTERNS:
            if re.match(pat, caption, re.IGNORECASE):
                errors.append(f"Generic uninformative caption detected: '{caption}'")
                dims["metadata_completeness"] = False
                break

    if modality == "image":
        cat = str(record.get("category", "")).strip().lower()
        if not cat:
            errors.append("Missing required 'category' field for image.")
            dims["metadata_completeness"] = False
        elif cat not in VALID_IMAGE_CATEGORIES:
            errors.append(f"Invalid image category '{cat}'. Allowed: {sorted(list(VALID_IMAGE_CATEGORIES))}")
            dims["metadata_completeness"] = False

    elif modality == "video":
        cam_motion = str(record.get("camera_motion", "static")).strip().lower()
        if cam_motion not in VALID_CAMERA_MOTIONS:
            errors.append(f"Invalid camera_motion '{cam_motion}'. Allowed: {sorted(list(VALID_CAMERA_MOTIONS))}")
            dims["metadata_completeness"] = False

        motion_desc = str(record.get("motion_description", "")).strip()
        if not motion_desc:
            errors.append("Missing required 'motion_description' field for video.")
            dims["metadata_completeness"] = False

        fps = record.get("fps")
        if fps is None or float(fps) < 8.0:
            errors.append(f"Video FPS ({fps}) below minimum architectural standard (8.0 FPS).")
            dims["metadata_completeness"] = False

        dur = record.get("duration_sec")
        if dur is None or float(dur) < 1.0:
            errors.append(f"Video duration ({dur}s) below minimum standard (1.0s).")
            dims["metadata_completeness"] = False

        nframes = record.get("num_frames")
        if nframes is None or int(nframes) < 8:
            errors.append(f"Video frame count ({nframes}) below minimum standard (8 frames).")
            dims["metadata_completeness"] = False

    # =========================================================================
    # DIMENSION 2: SOURCE ACCESSIBILITY & ASSET-LEVEL VERIFICATION
    # =========================================================================
    if source_url:
        low_url = source_url.lower()
        # Reject generic github data directories
        if "github.com/prem759-0/nanogpt/tree/main/data" in low_url or "github.com/prem759-0/nanogpt/tree" in low_url:
            errors.append(f"Fabricated internal repo URL detected: '{source_url}'. Self-referential folders cannot serve as asset-level source URLs.")
            dims["source_accessibility"] = False
        elif low_url.startswith("owner://") and len(source_url.strip()) <= 8:
            errors.append("Bare 'owner://' URI provided without asset-level identifier.")
            dims["source_accessibility"] = False
        elif not (low_url.startswith("http://") or low_url.startswith("https://") or low_url.startswith("owner://")):
            errors.append(f"Invalid URL protocol '{source_url}'. Must start with http://, https://, or owner://.")
            dims["source_accessibility"] = False
        elif low_url in {"https://example.com", "http://example.com", "https://unsplash.com", "https://pixabay.com"}:
            errors.append(f"Generic domain root URL without asset-level identifier: '{source_url}'.")
            dims["source_accessibility"] = False

    # =========================================================================
    # DIMENSION 3: LICENCE VERIFICATION
    # =========================================================================
    # Do not treat mere string 'CustomSovereign' as proof of ownership or permission
    if license_id == "CustomSovereign":
        # Sovereign license is verified via Dimension 4 (owner_declaration)
        pass
    else:
        # Third-party / external open licenses (CC0, CC-BY, MIT, etc.)
        license_ev = record.get("license_evidence")
        if not license_ev or not isinstance(license_ev, dict):
            errors.append(
                f"Missing required 'license_evidence' dictionary for external license '{license_id}'. "
                "Must provide license URL, verification status, and commercial suitability evidence."
            )
            dims["license_verified"] = False
        else:
            ver_status = license_ev.get("verification_status")
            if ver_status != "VERIFIED_PERMISSIBLE":
                errors.append(f"Licence verification status is '{ver_status}'. Must be 'VERIFIED_PERMISSIBLE' for production eligibility.")
                dims["license_verified"] = False
            if not license_ev.get("commercial_use_permitted", False):
                errors.append("Licence does not permit commercial/unrestricted use.")
                dims["license_verified"] = False

    # =========================================================================
    # DIMENSION 4: OWNERSHIP VERIFICATION (FOR SELF-CREATED / SOVEREIGN ASSETS)
    # =========================================================================
    if license_id == "CustomSovereign" or (source_url and source_url.lower().startswith("owner://")):
        owner_decl = record.get("owner_declaration")
        if not owner_decl or not isinstance(owner_decl, dict):
            errors.append(
                "Missing required 'owner_declaration' dictionary for CustomSovereign/owner media. "
                "Strings 'CustomSovereign', 'owner://', or creator names alone do NOT prove ownership. "
                "Explicit declaration with declarant, timestamp, matching asset SHA-256, and rights statement is required."
            )
            dims["ownership_verified"] = False
        else:
            declarant = str(owner_decl.get("declarant", "")).strip()
            decl_ts = str(owner_decl.get("declaration_timestamp", "")).strip()
            decl_sha = str(owner_decl.get("asset_sha256", "")).strip().lower()
            decl_rights = str(owner_decl.get("rights_statement", "")).strip()

            if not declarant or declarant.lower() != creator.lower():
                errors.append(f"Owner declaration declarant '{declarant}' does not match creator '{creator}'.")
                dims["ownership_verified"] = False

            if not decl_ts or not validate_iso_timestamp(decl_ts):
                errors.append(f"Owner declaration timestamp '{decl_ts}' is missing or not a valid ISO 8601 timestamp.")
                dims["ownership_verified"] = False

            if not decl_sha or decl_sha != sha256_val:
                errors.append(f"Owner declaration asset_sha256 ('{decl_sha}') does not match manifest SHA-256 ('{sha256_val}').")
                dims["ownership_verified"] = False

            if not decl_rights or len(decl_rights) < 20:
                errors.append("Owner declaration rights_statement is missing or insufficient.")
                dims["ownership_verified"] = False

    # =========================================================================
    # DIMENSION 5: ACTUAL FILE CHECKSUM & SAFE PATH RESOLUTION (TASK 3)
    # =========================================================================
    resolved_file_path = ""
    computed_sha = ""
    if rel_path:
        allowed_exts = ALLOWED_IMAGE_EXTENSIONS if modality == "image" else ALLOWED_VIDEO_EXTENSIONS
        safe_ok, resolved, path_err = validate_safe_path(rel_path, base_dir=base_dir, allowed_extensions=allowed_exts)
        if not safe_ok:
            errors.append(f"Path traversal or security violation: {path_err}")
            dims["safe_path_verified"] = False
        else:
            resolved_file_path = resolved

        if verify_file and safe_ok:
            chk_ok, comp_sha, chk_err = verify_file_checksum(resolved, sha256_val)
            computed_sha = comp_sha
            if not chk_ok:
                errors.append(f"Cryptographic file verification failed: {chk_err}")
                dims["file_checksum_verified"] = False

    # =========================================================================
    # OVERALL TRAINING ELIGIBILITY EVALUATION
    # =========================================================================
    all_dims_pass = all(dims.values())
    is_synth = record.get("is_synthetic", True)

    training_eligible = (
        all_dims_pass
        and (is_synth is False)
        and (not allow_synthetic)
        and len(errors) == 0
    )

    if training_eligible:
        status = "VERIFIED_PRODUCTION_ELIGIBLE"
    elif is_synth:
        status = "QUARANTINED_SYNTHETIC"
    elif not dims["license_verified"] or not dims["ownership_verified"]:
        status = "PENDING_MANUAL_VERIFICATION"
    else:
        status = "REJECTED_INTEGRITY"

    return {
        "is_valid": len(errors) == 0,
        "training_eligible": training_eligible,
        "status": status,
        "errors": errors,
        "dimensions": dims,
        "computed_sha256": computed_sha,
        "file_path": resolved_file_path
    }
