"""
datasets/security/attestation_engine.py
Strict Asymmetric Cryptographic Dataset Attestation Engine for Moon AI.
Enforces Ed25519 asymmetric digital signatures for dataset provenance.
Authorized independent verifiers hold private keys strictly outside this repository.
The repository stores only public verification keys in a trusted keystore.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: RFC 8785 Compatible Canonical JSON Ed25519 Attestation v2.0
"""
import os
import sys
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VERIFIERS_PATH = os.path.join(PROJECT_ROOT, "datasets", "security", "trusted_verifiers.json")

ATTESTATION_VERSION_CURRENT = "2.0"
VERIFICATION_SCOPE_PRODUCTION = "PRODUCTION_DATASET"
MAX_ATTESTATION_AGE_DAYS = 730
MAX_CLOCK_SKEW_SECONDS = 60

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
    Canonical format: Restricted RFC 8785 JSON representation:
    - Lexicographically sorted dictionary keys (sort_keys=True).
    - No whitespace separators (separators=(',', ':')).
    - UTF-8 encoding.
    - Strict boolean representation for commercial_use_permitted.
    - Normalized lowercase SHA-256 hex string.
    Never uses ambiguous delimiter concatenation.
    """
    canonical_dict = {
        "asset_sha256": str(asset_sha256).strip().lower(),
        "attestation_version": str(attestation_version).strip(),
        "commercial_use_permitted": True if commercial_use_permitted is True else False,
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
    Strictly enforces:
    1. Algorithm: Ed25519 asymmetric signatures only. HMAC or symmetric signing strictly rejected.
    2. Version & Scope: attestation_version must be '2.0', verification_scope must be 'PRODUCTION_DATASET'.
    3. Strict Typing: commercial_use_permitted must be genuine bool and True (never string or int).
    4. Signature: Exactly 128 lowercase hex characters (64-byte Ed25519 raw signature).
    5. Hashes: Exactly 64 lowercase hex characters matching ^[0-9a-f]{64}$.
    6. Timestamps: Explicit UTC timestamp required (ISO 8601 with Z or +00:00). Timezone-naive rejected.
       Future timestamps beyond 60s rejected. Max attestation age 730 days.
    7. Verifier Policy: valid_from and valid_until required, must be explicit UTC, valid_from < valid_until,
       status must be ACTIVE (REVOKED or SUSPENDED rejected).
    8. Bindings: record_id, asset_sha256, source_url, license, ownership_evidence_reference match record and actual bytes.

    Returns: (is_valid, verification_state, error_reason)
    """
    if not attestation or not isinstance(attestation, dict):
        return False, "PENDING_MANUAL_VERIFICATION", "Missing or invalid attestation dictionary."

    # 1. Algorithm check: HMAC is strictly rejected; only Ed25519 asymmetric signatures permitted
    algo = str(attestation.get("algorithm", "")).strip()
    if algo != "Ed25519":
        return False, "REJECTED", f"Algorithm '{algo}' is strictly rejected. Ed25519 asymmetric signatures required."

    # 2. Attestation Version check (Task 1)
    att_ver = attestation.get("attestation_version")
    if att_ver != ATTESTATION_VERSION_CURRENT:
        return False, "REJECTED", f"Attestation version '{att_ver}' is unsupported. Required: '{ATTESTATION_VERSION_CURRENT}'."

    # 3. Verification Scope check (Task 1)
    att_scope = attestation.get("verification_scope")
    if att_scope != VERIFICATION_SCOPE_PRODUCTION:
        return False, "REJECTED", f"Verification scope '{att_scope}' is unsupported for production datasets. Required: '{VERIFICATION_SCOPE_PRODUCTION}'."

    # 4. Strict Boolean Typing on commercial_use_permitted (Task 1)
    comm_val = attestation.get("commercial_use_permitted")
    if type(comm_val) is not bool:
        return False, "REJECTED", f"Attestation 'commercial_use_permitted' must be a strict JSON boolean, got {type(comm_val).__name__}."
    if comm_val is not True:
        return False, "REJECTED", "Attestation commercial_use_permitted is False. Production requires commercial suitability."
    att_comm = comm_val

    # 5. Required structure check
    required_fields = [
        "verifier_id", "record_id", "asset_sha256", "source_url",
        "ownership_evidence_reference", "license", "commercial_use_permitted",
        "timestamp", "signature", "algorithm", "attestation_version", "verification_scope"
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
    att_ts = str(attestation["timestamp"]).strip()
    att_sig_hex = str(attestation["signature"]).strip().lower()

    # 6. Signature format & length check (Task 1: exactly 128 lowercase hex characters)
    if len(att_sig_hex) != 128:
        return False, "REJECTED", f"Attestation signature has invalid length ({len(att_sig_hex)} chars; exactly 128 hex chars required for Ed25519)."
    if not re.fullmatch(r"[0-9a-f]{128}", att_sig_hex):
        return False, "REJECTED", "Attestation signature contains invalid non-hex characters."

    # 7. Asset SHA-256 format check (Task 1: exactly 64 lowercase hex characters)
    if len(att_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", att_sha):
        return False, "REJECTED", f"Malformed asset_sha256 in attestation '{att_sha}' (must be 64-char lowercase hex)."

    # 8. Strict timestamp validation (ISO 8601 UTC parsing, naive rejection, clock skew, max age)
    try:
        # Check explicit UTC notation
        if not (att_ts.endswith("Z") or "+00:00" in att_ts or att_ts.endswith("+00")):
            return False, "REJECTED", f"Malformed or unparseable attestation timestamp '{att_ts}': lacks explicit UTC timezone."
        att_dt = datetime.fromisoformat(att_ts.replace("Z", "+00:00"))
        if att_dt.tzinfo is None or att_dt.utcoffset() != timedelta(0):
            return False, "REJECTED", f"Malformed or unparseable attestation timestamp '{att_ts}': lacks explicit UTC timezone."
    except Exception as e:
        return False, "REJECTED", f"Malformed or unparseable attestation timestamp '{att_ts}': {e}"

    now_utc = datetime.now(timezone.utc)

    # Reject future timestamps beyond 60s clock skew allowance
    if att_dt > (now_utc + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)):
        return False, "REJECTED", f"Attestation timestamp '{att_ts}' is in the future."

    # Reject expired attestations (> 730 days lifespan)
    if (now_utc - att_dt) > timedelta(days=MAX_ATTESTATION_AGE_DAYS):
        return False, "REJECTED", f"Attestation has expired: age ({((now_utc - att_dt).days)} days) exceeds maximum allowed lifespan (730 days)."

    # 9. Lookup trusted verifier public key from keystore
    trusted_registry = load_trusted_verifiers(verifiers_path)
    if not trusted_registry:
        return False, "PENDING_MANUAL_VERIFICATION", "Trusted verifier keystore is empty or unconfigured. Production verification blocked."

    verifier_info = trusted_registry.get(v_id)
    if not verifier_info:
        return False, "REJECTED", f"Unknown or untrusted verifier_id: '{v_id}'. Not present in trusted verifiers keystore."

    # Verifier status check
    v_status = str(verifier_info.get("status", "")).strip().upper()
    if v_status != "ACTIVE":
        return False, "REJECTED", f"Verifier '{v_id}' is not ACTIVE (status: '{v_status}')."

    if verifier_info.get("algorithm") != "Ed25519":
        return False, "REJECTED", f"Verifier '{v_id}' algorithm '{verifier_info.get('algorithm')}' is invalid. Ed25519 required."

    # Verifier validity window checks (Task 1: valid_from and valid_until strictly required)
    valid_from = verifier_info.get("valid_from")
    if not valid_from or not isinstance(valid_from, str) or not valid_from.strip():
        return False, "REJECTED", f"Verifier '{v_id}' is missing required 'valid_from' timestamp in keystore."

    valid_until = verifier_info.get("valid_until")
    if not valid_until or not isinstance(valid_until, str) or not valid_until.strip():
        return False, "REJECTED", f"Verifier '{v_id}' is missing required 'valid_until' timestamp in keystore."

    try:
        if not (valid_from.endswith("Z") or "+00:00" in valid_from or valid_from.endswith("+00")):
            return False, "REJECTED", f"Verifier '{v_id}' valid_from timestamp '{valid_from}' lacks explicit UTC timezone."
        from_dt = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
        if from_dt.tzinfo is None or from_dt.utcoffset() != timedelta(0):
            return False, "REJECTED", f"Verifier '{v_id}' valid_from timestamp '{valid_from}' lacks explicit UTC timezone."
    except Exception as e:
        return False, "REJECTED", f"Verifier '{v_id}' has malformed valid_from timestamp in keystore: {e}"

    try:
        if not (valid_until.endswith("Z") or "+00:00" in valid_until or valid_until.endswith("+00")):
            return False, "REJECTED", f"Verifier '{v_id}' valid_until timestamp '{valid_until}' lacks explicit UTC timezone."
        until_dt = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
        if until_dt.tzinfo is None or until_dt.utcoffset() != timedelta(0):
            return False, "REJECTED", f"Verifier '{v_id}' valid_until timestamp '{valid_until}' lacks explicit UTC timezone."
    except Exception as e:
        return False, "REJECTED", f"Verifier '{v_id}' has malformed valid_until timestamp in keystore: {e}"

    if from_dt >= until_dt:
        return False, "REJECTED", f"Verifier '{v_id}' has inverted validity window: valid_from '{valid_from}' >= valid_until '{valid_until}'."

    if now_utc < from_dt:
        return False, "REJECTED", f"Verifier '{v_id}' trust certificate is not yet valid (valid_from: '{valid_from}')."

    if now_utc > until_dt:
        return False, "REJECTED", f"Verifier '{v_id}' trust certificate has expired on {valid_until}."

    if att_dt < from_dt:
        return False, "REJECTED", f"Attestation timestamp '{att_ts}' predates verifier validity start '{valid_from}'."

    if att_dt > until_dt:
        return False, "REJECTED", f"Attestation timestamp '{att_ts}' exceeds verifier validity end '{valid_until}'."

    # 10. Load Ed25519 Public Key
    pub_hex = verifier_info.get("public_key") or verifier_info.get("public_key_hex") or verifier_info.get("verification_key")
    if not pub_hex or len(pub_hex) != 64 or not re.fullmatch(r"[0-9a-f]{64}", str(pub_hex).lower()):
        return False, "REJECTED", f"Verifier '{v_id}' has invalid or missing 64-char public_key in keystore."

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
    except Exception as e:
        return False, "REJECTED", f"Failed to initialize Ed25519 public key for verifier '{v_id}': {e}"

    # 11. Asymmetric Signature Verification
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
        attestation_version=att_ver,
        verification_scope=att_scope
    )

    try:
        public_key.verify(sig_bytes, payload_bytes)
    except InvalidSignature:
        return False, "REJECTED", "Signature altered or forged! Ed25519 cryptographic verification failed."
    except Exception as e:
        return False, "REJECTED", f"Signature verification error: {e}"

    # 12. Strict bindings check against target record and disk asset
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

    return True, "EXTERNALLY_VERIFIED", None
