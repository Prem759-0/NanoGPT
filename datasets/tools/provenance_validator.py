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

from datasets.security.attestation_engine import verify_attestation, DEFAULT_VERIFIERS_PATH

# Explicit Verification States (Task 4)
VERIFICATION_STATE_SELF_DECLARED = "SELF_DECLARED"
VERIFICATION_STATE_PENDING_MANUAL = "PENDING_MANUAL_VERIFICATION"
VERIFICATION_STATE_EXTERNALLY_VERIFIED = "EXTERNALLY_VERIFIED"
VERIFICATION_STATE_REJECTED = "REJECTED"
VALID_VERIFICATION_STATES = {
    VERIFICATION_STATE_SELF_DECLARED,
    VERIFICATION_STATE_PENDING_MANUAL,
    VERIFICATION_STATE_EXTERNALLY_VERIFIED,
    VERIFICATION_STATE_REJECTED
}

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
    allowed_extensions: Optional[Set[str]] = None,
    must_exist: bool = False
) -> Tuple[bool, str, Optional[str]]:
    """
    Validates that rel_path resolves strictly within base_dir without directory traversal
    or symlink escape.
    Uses real path resolution, including symlink resolution, before containment checks.
    """
    if not rel_path or not isinstance(rel_path, str):
        return False, "", "Media path is empty or invalid."

    # Reject explicit traversal tokens
    if ".." in rel_path or "\\.." in rel_path or "/.." in rel_path:
        return False, "", f"Directory traversal detected in path: '{rel_path}'"

    # Reject absolute Windows paths or root paths
    if os.path.isabs(rel_path) or (len(rel_path) > 1 and rel_path[1] == ":") or rel_path.startswith("/") or rel_path.startswith("\\"):
        return False, "", f"Absolute paths are forbidden in manifests: '{rel_path}'"

    # Preliminary path normalization
    raw_joined = os.path.normpath(os.path.join(base_dir, rel_path))
    real_base = os.path.realpath(os.path.abspath(base_dir))
    real_resolved = os.path.realpath(raw_joined)

    # 1. Structural containment check on normpath
    norm_base = os.path.normpath(base_dir)
    try:
        common_norm = os.path.commonpath([norm_base, raw_joined])
        if common_norm != norm_base:
            return False, "", f"Path escapes base directory: '{rel_path}'"
    except Exception as e:
        return False, "", f"Path resolution error: {e}"

    # 2. Real path resolution (including symlink resolution) containment check
    try:
        common_real = os.path.commonpath([real_base, real_resolved])
        if common_real != real_base:
            return False, "", f"Path or symlink escapes base directory: '{rel_path}' resolves outside root -> '{real_resolved}'"
    except Exception as e:
        return False, "", f"Path resolution error: {e}"

    # 3. Explicit symlink inspection if link exists
    if os.path.islink(raw_joined):
        try:
            target = os.readlink(raw_joined)
            target_full = target if os.path.isabs(target) else os.path.normpath(os.path.join(os.path.dirname(raw_joined), target))
            target_real = os.path.realpath(target_full)
            common_sym = os.path.commonpath([real_base, target_real])
            if common_sym != real_base:
                return False, "", f"Symlink escapes base directory: '{rel_path}' points to '{target_real}' outside '{real_base}'"
        except Exception as e:
            return False, "", f"Symlink check error: {e}"

    # 4. Allowed extension check
    if allowed_extensions:
        ext = os.path.splitext(real_resolved)[1].lower()
        if ext not in allowed_extensions:
            return False, real_resolved, f"File extension '{ext}' is not permitted. Allowed: {sorted(list(allowed_extensions))}"

    # 5. File existence check if requested
    if must_exist and not os.path.exists(real_resolved):
        return False, "", f"Media file does not exist on disk: '{real_resolved}'"

    return True, real_resolved, None

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
    verify_file: bool = True,
    verifiers_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive 5-dimensional security, provenance, and integrity audit on a record.
    Returns audit status dictionary.
    """
    errors: List[str] = []
    dims = {
        "metadata_completeness": True,
        "url_syntax_valid": True,
        "asset_level_source_identified": True,
        "url_accessibility": VERIFICATION_STATE_PENDING_MANUAL,
        "source_accessibility": True,
        "license_verified": True,
        "licence_verification_state": VERIFICATION_STATE_PENDING_MANUAL,
        "ownership_verified": True,
        "ownership_verification_state": VERIFICATION_STATE_PENDING_MANUAL,
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
    # DIMENSION 2: SOURCE ACCESSIBILITY & ASSET-LEVEL VERIFICATION (TASKS 4 & 5)
    # =========================================================================
    if source_url:
        low_url = source_url.lower()
        if not (low_url.startswith("http://") or low_url.startswith("https://") or low_url.startswith("owner://")):
            errors.append(f"Invalid URL protocol '{source_url}'. Must start with http://, https://, or owner://.")
            dims["url_syntax_valid"] = False
            dims["source_accessibility"] = False
        else:
            dims["url_syntax_valid"] = True

        if "github.com/prem759-0/nanogpt/tree/main/data" in low_url or "github.com/prem759-0/nanogpt/tree" in low_url:
            errors.append(f"Fabricated internal repo URL detected: '{source_url}'. Self-referential folders cannot serve as asset-level source URLs.")
            dims["asset_level_source_identified"] = False
            dims["source_accessibility"] = False
        elif low_url.startswith("owner://") and len(source_url.strip()) <= 8:
            errors.append("Bare 'owner://' URI provided without asset-level identifier.")
            dims["asset_level_source_identified"] = False
            dims["source_accessibility"] = False
        elif low_url in {"https://example.com", "http://example.com", "https://unsplash.com", "https://pixabay.com"}:
            errors.append(f"Generic domain root URL without asset-level identifier: '{source_url}'.")
            dims["asset_level_source_identified"] = False
            dims["source_accessibility"] = False
        else:
            dims["asset_level_source_identified"] = True

        # Task 5: Distinguish URL syntax from real network accessibility.
        # Do not call a URL verified merely because it starts with HTTPS.
        # In offline/no-network mode, record PENDING_MANUAL_VERIFICATION.
        dims["url_accessibility"] = VERIFICATION_STATE_PENDING_MANUAL
        dims["source_accessibility"] = dims["url_syntax_valid"] and dims["asset_level_source_identified"]

    # =========================================================================
    # DIMENSION 3: LICENCE VERIFICATION (TASK 4 & 5)
    # =========================================================================
    # Extract optional cryptographic attestation bundle from record, license_evidence, or owner_declaration
    att_bundle = record.get("attestation")
    if not att_bundle and isinstance(record.get("license_evidence"), dict):
        att_bundle = record["license_evidence"].get("attestation")
    if not att_bundle and isinstance(record.get("owner_declaration"), dict):
        att_bundle = record["owner_declaration"].get("attestation")

    if license_id == "CustomSovereign":
        # Governed by ownership verification
        dims["license_verified"] = True
        dims["licence_verification_state"] = VERIFICATION_STATE_SELF_DECLARED
    else:
        # Third-party / external open licenses (CC0, CC-BY, MIT, etc.)
        license_ev = record.get("license_evidence")
        if not license_ev or not isinstance(license_ev, dict):
            errors.append(
                f"Missing required 'license_evidence' dictionary for external license '{license_id}'. "
                "Must provide license URL, verification status, and commercial suitability evidence."
            )
            dims["license_verified"] = False
            dims["licence_verification_state"] = VERIFICATION_STATE_PENDING_MANUAL
        else:
            ver_status = license_ev.get("verification_status")
            comm_ok = bool(license_ev.get("commercial_use_permitted", False))
            if not comm_ok:
                dims["license_verified"] = False
                dims["licence_verification_state"] = VERIFICATION_STATE_PENDING_MANUAL
                errors.append(
                    f"External licence '{license_id}' has commercial_use_permitted=False. Commercial suitability is required for production."
                )
            elif not att_bundle:
                # No cryptographic attestation provided! Even if submitter wrote "EXTERNALLY_VERIFIED" or "VERIFIED_PERMISSIBLE"
                dims["license_verified"] = False
                dims["licence_verification_state"] = VERIFICATION_STATE_PENDING_MANUAL
                if ver_status in (VERIFICATION_STATE_EXTERNALLY_VERIFIED, "VERIFIED_PERMISSIBLE") or record.get("verification_state") == VERIFICATION_STATE_EXTERNALLY_VERIFIED:
                    errors.append(
                        f"External licence claim '{license_id}' self-declared external verification ('{ver_status}') "
                        "without a valid cryptographic attestation bundle from a trusted verifier in trusted_verifiers.json."
                    )
                else:
                    errors.append(
                        f"External licence claim remains pending manual verification (status: '{ver_status}'). "
                        "Requires cryptographic attestation from a trusted verifier."
                    )
            else:
                # Verify cryptographic attestation
                is_valid_att, att_state, att_reason = verify_attestation(
                    attestation=att_bundle,
                    record_id=rec_id,
                    actual_asset_sha256=sha256_val,
                    license_id=license_id,
                    commercial_use_permitted=True,
                    verifiers_path=verifiers_path
                )
                if is_valid_att:
                    dims["license_verified"] = True
                    dims["licence_verification_state"] = VERIFICATION_STATE_EXTERNALLY_VERIFIED
                else:
                    dims["license_verified"] = False
                    dims["licence_verification_state"] = att_state
                    errors.append(f"External licence cryptographic attestation failed: {att_reason}")

    # =========================================================================
    # DIMENSION 4: OWNERSHIP VERIFICATION (TASKS 4 & 5)
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
            dims["ownership_verification_state"] = VERIFICATION_STATE_PENDING_MANUAL
        else:
            declarant = str(owner_decl.get("declarant", "")).strip()
            decl_ts = str(owner_decl.get("declaration_timestamp", "")).strip()
            decl_sha = str(owner_decl.get("asset_sha256", "")).strip().lower()
            decl_rights = str(owner_decl.get("rights_statement", "")).strip()

            decl_valid = True
            if not declarant or declarant.lower() != creator.lower():
                errors.append(f"Owner declaration declarant '{declarant}' does not match creator '{creator}'.")
                dims["ownership_verified"] = False
                dims["ownership_verification_state"] = VERIFICATION_STATE_REJECTED
                decl_valid = False

            if not decl_ts or not validate_iso_timestamp(decl_ts):
                errors.append(f"Owner declaration timestamp '{decl_ts}' is missing or not a valid ISO 8601 timestamp.")
                dims["ownership_verified"] = False
                dims["ownership_verification_state"] = VERIFICATION_STATE_REJECTED
                decl_valid = False

            if not decl_sha or decl_sha != sha256_val:
                errors.append(f"Owner declaration asset_sha256 ('{decl_sha}') does not match manifest SHA-256 ('{sha256_val}').")
                dims["ownership_verified"] = False
                dims["ownership_verification_state"] = VERIFICATION_STATE_REJECTED
                decl_valid = False

            if not decl_rights or len(decl_rights) < 20:
                errors.append("Owner declaration rights_statement is missing or insufficient.")
                dims["ownership_verified"] = False
                dims["ownership_verification_state"] = VERIFICATION_STATE_REJECTED
                decl_valid = False

            if decl_valid:
                # Task 4 & Stage 2.0.4 Fix A:
                # Never convert a self-declaration into independent ownership verification automatically.
                # A self-declaration is by definition SELF_DECLARED unless verified by cryptographic attestation.
                if not att_bundle:
                    dims["ownership_verified"] = False
                    dims["ownership_verification_state"] = VERIFICATION_STATE_SELF_DECLARED
                    claimed_ext = (
                        owner_decl.get("verification_state") == VERIFICATION_STATE_EXTERNALLY_VERIFIED
                        or record.get("verification_state") == VERIFICATION_STATE_EXTERNALLY_VERIFIED
                        or bool(record.get("independent_verification"))
                        or bool(owner_decl.get("independent_verification"))
                    )
                    if claimed_ext:
                        errors.append(
                            "Self-declared EXTERNALLY_VERIFIED or independent_verification without cryptographic attestation. "
                            "Submitters cannot self-declare external verification. "
                            "Requires cryptographic attestation signed by a trusted verifier in trusted_verifiers.json."
                        )
                    else:
                        errors.append(
                            "Self-declared ownership remains unverified (SELF_DECLARED). "
                            "Self-attestation alone does not prove independent ownership. "
                            "Asset remains blocked pending independent external verification."
                        )
                else:
                    # Verify cryptographic attestation
                    is_valid_att, att_state, att_reason = verify_attestation(
                        attestation=att_bundle,
                        record_id=rec_id,
                        actual_asset_sha256=sha256_val,
                        license_id=license_id,
                        commercial_use_permitted=True,
                        verifiers_path=verifiers_path
                    )
                    if is_valid_att:
                        dims["ownership_verified"] = True
                        dims["ownership_verification_state"] = VERIFICATION_STATE_EXTERNALLY_VERIFIED
                    else:
                        dims["ownership_verified"] = False
                        dims["ownership_verification_state"] = att_state
                        errors.append(f"Ownership cryptographic attestation failed: {att_reason}")
    else:
        # For third-party assets, ownership is tied to valid licence verification with external attestation
        if dims["license_verified"] and dims["licence_verification_state"] == VERIFICATION_STATE_EXTERNALLY_VERIFIED:
            dims["ownership_verified"] = True
            dims["ownership_verification_state"] = VERIFICATION_STATE_EXTERNALLY_VERIFIED
        else:
            dims["ownership_verified"] = False
            dims["ownership_verification_state"] = VERIFICATION_STATE_PENDING_MANUAL

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
    all_dims_pass = (
        dims["metadata_completeness"]
        and dims["url_syntax_valid"]
        and dims["asset_level_source_identified"]
        and dims["license_verified"]
        and dims["ownership_verified"]
        and dims["safe_path_verified"]
        and dims["file_checksum_verified"]
    )
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
    elif (dims["ownership_verification_state"] == VERIFICATION_STATE_SELF_DECLARED
          or dims["ownership_verification_state"] == VERIFICATION_STATE_PENDING_MANUAL
          or dims["licence_verification_state"] == VERIFICATION_STATE_PENDING_MANUAL
          or not dims["license_verified"]
          or not dims["ownership_verified"]):
        status = "PENDING_MANUAL_VERIFICATION"
    else:
        status = "REJECTED_INTEGRITY"

    return {
        "is_valid": len(errors) == 0,
        "training_eligible": training_eligible,
        "status": status,
        "ownership_verification_state": dims["ownership_verification_state"],
        "licence_verification_state": dims["licence_verification_state"],
        "url_accessibility": dims["url_accessibility"],
        "errors": errors,
        "dimensions": dims,
        "computed_sha256": computed_sha,
        "file_path": resolved_file_path
    }
