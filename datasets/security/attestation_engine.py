"""
datasets/security/attestation_engine.py
Asymmetric Cryptographic Dataset Attestation Engine for Moon AI.
Enforces Ed25519 asymmetric digital signatures for dataset provenance.
Authorized independent verifiers hold private keys strictly outside this repository.
The repository stores only public verification keys in a trusted keystore.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VERIFIERS_PATH = os.path.join(PROJECT_ROOT, "datasets", "security", "trusted_verifiers.json")

ATTESTATION_VERSION_CURRENT = "2.0"
VERIFICATION_SCOPE_PRODUCTION = "PRODUCTION_DATASET"

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

def extract_ownership_evidence_reference(record: Dict[str, Any]) -> str:
    """Extracts the canonical ownership or licensing evidence reference from a metadata record."""
    if "ownership_evidence_reference" in record and record["ownership_evidence_reference"]:
        return str(record["ownership_evidence_reference"]).strip()

    if record.get("license") == "CustomSovereign" or str(record.get("source_url", "")).lower().startswith("owner://"):
        owner_decl = record.get("owner_declaration") or {}
        return str(
            owner_decl.get("ownership_evidence_reference")
            or owner_decl.get("declaration_reference")
            or owner_decl.get("declaration_timestamp")
            or owner_decl.get("rights_statement")
            or ""
        ).strip()
    else:
        lic_ev = record.get("license_evidence") or {}
        return str(
            lic_ev.get("ownership_evidence_reference")
            or lic_ev.get("evidence_reference")
            or lic_ev.get("license_url")
            or ""
        ).strip()

def construct_canonical_attestation_payload(
    verifier_id: str,
    record_id: str,
    asset_sha256: str,
    source_url: str,
    ownership_evidence_reference: str,
    license_id: str,
    commercial_use_permitted: bool,
    timestamp: str,
    attestation_version: str = ATTESTATION_VERSION_CURRENT,
    verification_scope: str = VERIFICATION_SCOPE_PRODUCTION
) -> bytes:
    """
    Constructs deterministic canonical JSON byte payload for cryptographic binding.
    Canonical format: RFC 8785 style sorted keys, no whitespace separators, UTF-8.
    Never uses ambiguous delimiter concatenation.
    """
    canonical_dict = {
        "asset_sha256": str(asset_sha256).strip().lower(),
        "attestation_version": str(attestation_version).strip(),
        "commercial_use_permitted": bool(commercial_use_permitted),
        "license_id": str(license_id).strip(),
        "ownership_evidence_reference": str(ownership_evidence_reference).strip(),
        "record_id": str(record_id).strip(),
        "source_url": str(source_url).strip(),
        "timestamp": str(timestamp).strip(),
        "verification_scope": str(verification_scope).strip(),
        "verifier_id": str(verifier_id).strip()
    }
    return json.dumps(canonical_dict, sort_keys=True, separators=(',', ':')).encode('utf-8')

def sign_attestation(
    verifier_id: str,
    private_key: ed25519.Ed25519PrivateKey,
    record_id: str,
    asset_sha256: str,
    source_url: str,
    ownership_evidence_reference: str,
    license_id: str,
    commercial_use_permitted: bool = True,
    timestamp: Optional[str] = None,
    attestation_version: str = ATTESTATION_VERSION_CURRENT,
    verification_scope: str = VERIFICATION_SCOPE_PRODUCTION
) -> Dict[str, Any]:
    """
    Generates an Ed25519 cryptographically signed attestation bundle.
    Used by authorized independent verifiers (or in isolated test fixtures).
    Private signing keys are held strictly outside this repository.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    payload_bytes = construct_canonical_attestation_payload(
        verifier_id=verifier_id,
        record_id=record_id,
        asset_sha256=asset_sha256,
        source_url=source_url,
        ownership_evidence_reference=ownership_evidence_reference,
        license_id=license_id,
        commercial_use_permitted=commercial_use_permitted,
        timestamp=ts,
        attestation_version=attestation_version,
        verification_scope=verification_scope
    )
    sig_bytes = private_key.sign(payload_bytes)

    return {
        "attestation_version": attestation_version,
        "verifier_id": verifier_id,
        "record_id": record_id,
        "asset_sha256": asset_sha256.lower().strip(),
        "source_url": source_url.strip(),
        "ownership_evidence_reference": ownership_evidence_reference.strip(),
        "license": license_id,
        "commercial_use_permitted": commercial_use_permitted,
        "timestamp": ts,
        "verification_scope": verification_scope,
        "algorithm": "Ed25519",
        "signature": sig_bytes.hex()
    }

def verify_attestation(
    attestation: Dict[str, Any],
    record: Dict[str, Any],
    actual_asset_sha256: str,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Cryptographically verifies an attestation bundle against trusted public keys in the keystore.
    Returns: (is_valid, verification_state, error_reason)
    """
    if not attestation or not isinstance(attestation, dict):
        return False, "PENDING_MANUAL_VERIFICATION", "Missing or invalid attestation dictionary."

    # 1. Algorithm check: HMAC is strictly rejected; only Ed25519 asymmetric signatures permitted
    algo = str(attestation.get("algorithm", "")).strip()
    if algo != "Ed25519":
        return False, "REJECTED", f"Algorithm '{algo}' is strictly rejected. Ed25519 asymmetric signatures required."

    # 2. Required structure check
    required_fields = [
        "verifier_id", "record_id", "asset_sha256", "source_url",
        "ownership_evidence_reference", "license", "commercial_use_permitted",
        "timestamp", "signature", "algorithm"
    ]
    for f in required_fields:
        if f not in attestation or attestation[f] is None or (isinstance(attestation[f], str) and not attestation[f].strip()):
            return False, "REJECTED", f"Attestation is missing required field: '{f}'"

    v_id = str(attestation["verifier_id"]).strip()
    att_rec_id = str(attestation["record_id"]).strip()
    att_sha = str(attestation["asset_sha256"]).strip().lower()
    att_source = str(attestation["source_url"]).strip()
    att_evidence = str(attestation["ownership_evidence_reference"]).strip()
    att_lic = str(attestation["license"]).strip()
    att_comm = bool(attestation.get("commercial_use_permitted", False))
    att_ts = str(attestation["timestamp"]).strip()
    att_sig_hex = str(attestation["signature"]).strip().lower()

    # 3. Strict timestamp validation (ISO 8601 UTC parsing and future timestamp rejection)
    try:
        att_dt = datetime.fromisoformat(att_ts.replace("Z", "+00:00"))
        if att_dt.tzinfo is None:
            return False, "REJECTED", f"Attestation timestamp '{att_ts}' lacks explicit UTC timezone."
    except Exception as e:
        return False, "REJECTED", f"Malformed or unparseable attestation timestamp '{att_ts}': {e}"

    now_utc = datetime.now(timezone.utc)
    # Reject future timestamps beyond 60s clock skew allowance
    if att_dt > (now_utc + timedelta(seconds=60)):
        return False, "REJECTED", f"Attestation timestamp '{att_ts}' is in the future."

    # 4. Lookup trusted verifier public key from keystore
    trusted_registry = load_trusted_verifiers(verifiers_path)
    if not trusted_registry:
        return False, "PENDING_MANUAL_VERIFICATION", "Trusted verifier keystore is empty or unconfigured. Production verification blocked."

    verifier_info = trusted_registry.get(v_id)
    if not verifier_info:
        return False, "REJECTED", f"Unknown or untrusted verifier_id: '{v_id}'. Not present in trusted verifiers keystore."

    if verifier_info.get("status") != "ACTIVE":
        return False, "REJECTED", f"Verifier '{v_id}' is not ACTIVE (status: '{verifier_info.get('status')}')."

    if verifier_info.get("algorithm") != "Ed25519":
        return False, "REJECTED", f"Verifier '{v_id}' algorithm '{verifier_info.get('algorithm')}' is invalid. Ed25519 required."

    # Check verifier validity window
    valid_from = verifier_info.get("valid_from")
    if valid_from:
        try:
            from_dt = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
            if att_dt < from_dt:
                return False, "REJECTED", f"Attestation timestamp '{att_ts}' predates verifier validity start '{valid_from}'."
        except Exception:
            return False, "REJECTED", f"Verifier '{v_id}' has malformed valid_from timestamp in keystore."

    valid_until = verifier_info.get("valid_until")
    if valid_until:
        try:
            until_dt = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            if att_dt > until_dt or now_utc > until_dt:
                return False, "REJECTED", f"Verifier '{v_id}' trust certificate has expired on {valid_until}."
        except Exception:
            return False, "REJECTED", f"Verifier '{v_id}' has malformed valid_until timestamp in keystore."

    # 5. Load Ed25519 Public Key
    pub_hex = verifier_info.get("public_key") or verifier_info.get("public_key_hex") or verifier_info.get("verification_key")
    if not pub_hex or len(pub_hex) != 64:
        return False, "REJECTED", f"Verifier '{v_id}' has invalid or missing 64-char public_key in keystore."

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
    except Exception as e:
        return False, "REJECTED", f"Failed to initialize Ed25519 public key for verifier '{v_id}': {e}"

    # 6. Asymmetric Signature Verification
    try:
        sig_bytes = bytes.fromhex(att_sig_hex)
    except Exception as e:
        return False, "REJECTED", f"Attestation signature is not valid hex: {e}"

    payload_bytes = construct_canonical_attestation_payload(
        verifier_id=v_id,
        record_id=att_rec_id,
        asset_sha256=att_sha,
        source_url=att_source,
        ownership_evidence_reference=att_evidence,
        license_id=att_lic,
        commercial_use_permitted=att_comm,
        timestamp=att_ts,
        attestation_version=attestation.get("attestation_version", ATTESTATION_VERSION_CURRENT),
        verification_scope=attestation.get("verification_scope", VERIFICATION_SCOPE_PRODUCTION)
    )

    try:
        public_key.verify(sig_bytes, payload_bytes)
    except InvalidSignature:
        return False, "REJECTED", "Signature altered or forged! Ed25519 cryptographic verification failed."
    except Exception as e:
        return False, "REJECTED", f"Signature verification error: {e}"

    # 7. Strict bindings check against target record and disk asset
    record_id = str(record.get("id", "")).strip()
    if att_rec_id != record_id:
        return False, "REJECTED", f"Attestation record_id mismatch: attestation specifies '{att_rec_id}', record is '{record_id}'."

    norm_actual_sha = str(actual_asset_sha256).strip().lower()
    if att_sha != norm_actual_sha:
        return False, "REJECTED", f"Attestation asset_sha256 mismatch: attestation specifies '{att_sha}', actual file bytes yield '{norm_actual_sha}'."

    record_sha = str(record.get("sha256", "")).strip().lower()
    if att_sha != record_sha:
        return False, "REJECTED", f"Attestation asset_sha256 mismatch with record sha256: attestation '{att_sha}', record '{record_sha}'."

    record_source = str(record.get("source_url", "")).strip()
    if att_source != record_source:
        return False, "REJECTED", f"Attestation source_url mismatch: attestation specifies '{att_source}', record specifies '{record_source}'."

    record_lic = str(record.get("license", "")).strip()
    if att_lic != record_lic:
        return False, "REJECTED", f"Attestation license mismatch: attestation specifies '{att_lic}', record specifies '{record_lic}'."

    expected_evidence_ref = extract_ownership_evidence_reference(record)
    if att_evidence != expected_evidence_ref:
        return False, "REJECTED", f"Attestation ownership_evidence_reference mismatch: attestation specifies '{att_evidence}', record specifies '{expected_evidence_ref}'."

    if not att_comm:
        return False, "REJECTED", "Attestation commercial_use_permitted is False. Production requires commercial suitability."

    return True, "EXTERNALLY_VERIFIED", None
