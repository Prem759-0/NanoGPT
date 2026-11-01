"""
datasets/security/attestation_engine.py
Zero-Trust Cryptographic Dataset Attestation Engine for Moon AI.
Enforces that dataset submitters cannot self-promote records to EXTERNALLY_VERIFIED.
Attestations must be cryptographically signed by registered, active trusted verifiers
stored in a keystore outside ordinary dataset manifests.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import json
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VERIFIERS_PATH = os.path.join(PROJECT_ROOT, "datasets", "security", "trusted_verifiers.json")

def load_trusted_verifiers(verifiers_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads registered trusted verifiers from secure keystore."""
    path = verifiers_path or DEFAULT_VERIFIERS_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("trusted_verifiers", {})
    except Exception:
        return {}

def construct_canonical_attestation_payload(
    verifier_id: str,
    record_id: str,
    asset_sha256: str,
    license_id: str,
    commercial_use_permitted: bool,
    timestamp: str
) -> bytes:
    """
    Constructs deterministic canonical byte payload for cryptographic binding.
    Format: VERIFIER_ID|RECORD_ID|ASSET_SHA256|LICENSE|COMMERCIAL_PERMITTED|TIMESTAMP
    """
    v_id = str(verifier_id).strip()
    r_id = str(record_id).strip()
    sha = str(asset_sha256).strip().lower()
    lic = str(license_id).strip()
    comm = str(bool(commercial_use_permitted)).lower()
    ts = str(timestamp).strip()
    payload_str = f"{v_id}|{r_id}|{sha}|{lic}|{comm}|{ts}"
    return payload_str.encode("utf-8")

def sign_attestation(
    verifier_id: str,
    secret_key: str,
    record_id: str,
    asset_sha256: str,
    license_id: str,
    commercial_use_permitted: bool = True,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a cryptographically signed attestation bundle.
    Used exclusively by authorized independent verifiers.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    payload_bytes = construct_canonical_attestation_payload(
        verifier_id=verifier_id,
        record_id=record_id,
        asset_sha256=asset_sha256,
        license_id=license_id,
        commercial_use_permitted=commercial_use_permitted,
        timestamp=ts
    )
    sig = hmac.new(secret_key.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

    return {
        "verifier_id": verifier_id,
        "record_id": record_id,
        "asset_sha256": asset_sha256.lower().strip(),
        "license": license_id,
        "commercial_use_permitted": commercial_use_permitted,
        "timestamp": ts,
        "algorithm": "HMAC-SHA256",
        "signature": sig
    }

def verify_attestation(
    attestation: Dict[str, Any],
    record_id: str,
    actual_asset_sha256: str,
    license_id: str,
    commercial_use_permitted: bool = True,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Cryptographically verifies an attestation bundle against the trusted verifier keystore.
    Returns: (is_valid, verification_state, error_reason)
    """
    if not attestation or not isinstance(attestation, dict):
        return False, "PENDING_MANUAL_VERIFICATION", "Missing or invalid attestation dictionary."

    # 1. Required structure check
    required_fields = ["verifier_id", "record_id", "asset_sha256", "license", "signature", "timestamp"]
    for f in required_fields:
        if f not in attestation or not str(attestation[f]).strip():
            return False, "REJECTED", f"Attestation is missing required field: '{f}'"

    v_id = str(attestation["verifier_id"]).strip()
    att_rec_id = str(attestation["record_id"]).strip()
    att_sha = str(attestation["asset_sha256"]).strip().lower()
    att_lic = str(attestation["license"]).strip()
    att_comm = bool(attestation.get("commercial_use_permitted", True))
    att_ts = str(attestation["timestamp"]).strip()
    att_sig = str(attestation["signature"]).strip().lower()

    # 2. Strict bindings check against target record and disk asset
    if att_rec_id != str(record_id).strip():
        return False, "REJECTED", f"Attestation record_id mismatch: attestation specifies '{att_rec_id}', record is '{record_id}'."

    norm_actual_sha = str(actual_asset_sha256).strip().lower()
    if att_sha != norm_actual_sha:
        return False, "REJECTED", f"Attestation asset_sha256 mismatch: attestation specifies '{att_sha}', actual file bytes yield '{norm_actual_sha}'."

    if att_lic != str(license_id).strip():
        return False, "REJECTED", f"Attestation license mismatch: attestation specifies '{att_lic}', record specifies '{license_id}'."

    if bool(commercial_use_permitted) and not att_comm:
        return False, "REJECTED", "Attestation commercial_use_permitted is False. Production requires commercial suitability."

    # 3. Lookup trusted verifier outside dataset manifests
    trusted_registry = load_trusted_verifiers(verifiers_path)
    if not trusted_registry:
        return False, "PENDING_MANUAL_VERIFICATION", "Trusted verifier keystore is empty or unconfigured. External verification blocked."

    verifier_info = trusted_registry.get(v_id)
    if not verifier_info:
        return False, "REJECTED", f"Unknown or untrusted verifier_id: '{v_id}'. Not present in trusted verifiers keystore."

    if verifier_info.get("status") != "ACTIVE":
        return False, "REJECTED", f"Verifier '{v_id}' is not ACTIVE (status: '{verifier_info.get('status')}')."

    # Check verifier validity window
    valid_until = verifier_info.get("valid_until")
    if valid_until:
        try:
            exp_dt = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > exp_dt:
                return False, "REJECTED", f"Verifier '{v_id}' trust certificate has expired on {valid_until}."
        except Exception:
            pass

    # 4. Cryptographic signature verification
    ver_key = verifier_info.get("verification_key")
    if not ver_key:
        return False, "REJECTED", f"Verifier '{v_id}' has no verification_key in keystore."

    expected_payload = construct_canonical_attestation_payload(
        verifier_id=v_id,
        record_id=att_rec_id,
        asset_sha256=att_sha,
        license_id=att_lic,
        commercial_use_permitted=att_comm,
        timestamp=att_ts
    )
    expected_sig = hmac.new(ver_key.encode("utf-8"), expected_payload, hashlib.sha256).hexdigest().lower()

    if not hmac.compare_digest(expected_sig, att_sig):
        return False, "REJECTED", f"Cryptographic attestation signature verification failed for verifier '{v_id}'. Signature altered or forged."

    return True, "EXTERNALLY_VERIFIED", None
