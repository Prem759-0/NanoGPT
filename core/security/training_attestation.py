"""
core/security/training_attestation.py
Strict Asymmetric Cryptographic Training Provenance & Tamper-Proof Attestation Engine for Moon AI.

Binds candidate model checkpoint SHA-256, random Gaussian scratch initialization proof,
strict sovereign policy, architecture specification, dataset manifest hashes, step counts,
and verifiable training telemetry directly to independent training authorities via Ed25519 signatures.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.0
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union, List

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_AUTHORITIES_PATH = os.path.join(
    PROJECT_ROOT, "core", "security", "trusted_training_authorities.json"
)

REQUIRED_ATTESTATION_FIELDS = [
    "run_id",
    "policy",
    "initialization",
    "model_architecture_hash",
    "dataset_manifest_hash",
    "training_code_hash",
    "training_start_time",
    "training_completion_time",
    "step_count",
    "metrics",
    "checkpoint_sha256",
    "verifier_id",
    "signature",
]

VALID_TRAINING_PROVENANCE_TIERS = {
    "STRUCTURALLY_INVALID",
    "STRUCTURALLY_VALID_UNVERIFIED",
    "TEST_FIXTURE",
    "HISTORICAL_POLICY_B",
    "UNAPPROVED_POLICY",
    "NON_SCRATCH_INITIALIZATION",
    "TEACHER_CONTAMINATION",
    "UNTRAINED_RANDOM_INIT",
    "SYNTHETIC_MOCK",
    "ATTESTATION_MISSING",
    "ATTESTATION_INVALID",
    "ATTESTATION_CHECKPOINT_HASH_MISMATCH",
    "ATTESTATION_UNTRUSTED_SIGNER",
    "GENUINE_TRAINED_SOVEREIGN",
    "PROMOTED_PRODUCTION",
}


def load_trusted_training_authorities(
    authorities_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads registered trusted training authorities from secure keystore."""
    path = authorities_path or DEFAULT_AUTHORITIES_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("trusted_authorities", {})
    except Exception:
        return {}


def compute_file_sha256(filepath: str) -> str:
    """Computes exact raw-bytes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of raw byte buffer."""
    return hashlib.sha256(data).hexdigest().lower()


def construct_canonical_training_payload(attestation_dict: Dict[str, Any]) -> bytes:
    """
    Constructs deterministic canonical byte payload for signing and verification.
    Follows Restricted Deterministic Canonical JSON standard:
    - Keys sorted alphabetically.
    - No whitespace separators (',', ':').
    - Strict UTF-8 encoding.
    - Omits signature field during payload calculation.
    """
    if not isinstance(attestation_dict, dict):
        raise ValueError("Attestation must be a dictionary.")

    payload_fields = {
        "checkpoint_sha256": str(attestation_dict.get("checkpoint_sha256", "")).lower().strip(),
        "dataset_manifest_hash": str(attestation_dict.get("dataset_manifest_hash", "")).strip(),
        "initialization": str(attestation_dict.get("initialization", "")).strip(),
        "metrics": attestation_dict.get("metrics", {}) if isinstance(attestation_dict.get("metrics"), dict) else {},
        "model_architecture_hash": str(attestation_dict.get("model_architecture_hash", "")).strip(),
        "policy": str(attestation_dict.get("policy", "")).strip(),
        "run_id": str(attestation_dict.get("run_id", "")).strip(),
        "step_count": int(attestation_dict.get("step_count", 0)),
        "training_code_hash": str(attestation_dict.get("training_code_hash", "")).strip(),
        "training_completion_time": str(attestation_dict.get("training_completion_time", "")).strip(),
        "training_start_time": str(attestation_dict.get("training_start_time", "")).strip(),
        "verifier_id": str(attestation_dict.get("verifier_id", "")).strip(),
    }
    return json.dumps(payload_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_training_authority_keypair() -> Tuple[str, str]:
    """Generates a fresh Ed25519 signing keypair for testing or authority registration."""
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_hex = priv.private_bytes_raw().hex()
    pub_hex = priv.public_key().public_bytes_raw().hex()
    return priv_hex, pub_hex


def sign_training_attestation(
    attestation_dict: Dict[str, Any], private_key_hex: str
) -> Dict[str, Any]:
    """
    Cryptographically signs a training attestation payload using an Ed25519 private key.
    Attaches the hex-encoded signature directly to the attestation dictionary.
    """
    if not private_key_hex or len(private_key_hex) != 64:
        raise ValueError("Invalid private key: expected 64-character (32-byte) hex string.")

    canonical_bytes = construct_canonical_training_payload(attestation_dict)
    priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    sig = priv_key.sign(canonical_bytes)

    out = dict(attestation_dict)
    out["signature"] = sig.hex().lower()
    return out


def verify_training_attestation(
    attestation: Optional[Dict[str, Any]],
    checkpoint_bytes_or_path: Union[str, bytes],
    trusted_authorities: Optional[Dict[str, Any]] = None,
    authorities_path: Optional[str] = None,
    allow_untrusted_for_test: bool = False,
) -> Tuple[bool, str, str]:
    """
    Performs comprehensive cryptographic verification of training attestation against trust roots:
    1. Validates presence and structure of attestation object.
    2. Enforces presence and non-null values for all 13 required fields.
    3. Recomputes exact raw-bytes SHA-256 of candidate checkpoint file or bytes.
    4. Validates strict policy and Gaussian scratch initialization constraints.
    5. Checks verifier identity against configured trusted training authorities.
    6. Asymmetrically verifies Ed25519 signature over canonical JSON payload.

    Returns: (is_valid: bool, tier: str, reason: str)
    """
    if attestation is None:
        return False, "ATTESTATION_MISSING", "Training attestation is missing."
    if not isinstance(attestation, dict):
        return False, "ATTESTATION_INVALID", "Training attestation must be a JSON dictionary."

    # Verify required fields
    missing_fields = [f for f in REQUIRED_ATTESTATION_FIELDS if f not in attestation or attestation[f] is None]
    if missing_fields:
        return (
            False,
            "ATTESTATION_INVALID",
            f"Training attestation is missing required fields: {', '.join(missing_fields)}",
        )

    # Validate policy constraint
    policy = attestation.get("policy")
    if policy != "STRICT_SOVEREIGN_RANDOM_INIT_ONLY":
        return (
            False,
            "UNAPPROVED_POLICY",
            f"Attestation policy '{policy}' violates STRICT_SOVEREIGN_RANDOM_INIT_ONLY.",
        )

    # Validate initialization method
    init_method = attestation.get("initialization")
    if init_method != "random_gaussian_scratch":
        return (
            False,
            "NON_SCRATCH_INITIALIZATION",
            f"Attestation initialization '{init_method}' violates random_gaussian_scratch.",
        )

    # Validate step count
    try:
        step_count = int(attestation.get("step_count", 0))
    except (ValueError, TypeError):
        return False, "ATTESTATION_INVALID", "Attestation 'step_count' must be an integer."
    if step_count <= 0:
        return (
            False,
            "UNTRAINED_RANDOM_INIT",
            f"Attestation step count ({step_count}) indicates untrained model state.",
        )

    # Compute actual checkpoint SHA-256
    try:
        if isinstance(checkpoint_bytes_or_path, (bytes, bytearray)):
            actual_sha256 = compute_bytes_sha256(bytes(checkpoint_bytes_or_path))
        elif isinstance(checkpoint_bytes_or_path, str):
            if not os.path.isfile(checkpoint_bytes_or_path):
                return False, "STRUCTURALLY_INVALID", f"Checkpoint file '{checkpoint_bytes_or_path}' not found on disk."
            actual_sha256 = compute_file_sha256(checkpoint_bytes_or_path)
        else:
            return False, "STRUCTURALLY_INVALID", "Checkpoint must be a file path string or byte buffer."
    except Exception as e:
        return False, "STRUCTURALLY_INVALID", f"Failed to compute checkpoint SHA-256: {e}"

    expected_sha256 = str(attestation.get("checkpoint_sha256", "")).lower().strip()
    if not expected_sha256 or expected_sha256 != actual_sha256:
        return (
            False,
            "ATTESTATION_CHECKPOINT_HASH_MISMATCH",
            f"Attestation SHA-256 '{expected_sha256}' does not match checkpoint file SHA-256 '{actual_sha256}'.",
        )

    # Resolve trusted authorities
    verifier_id = str(attestation.get("verifier_id", "")).strip()
    authorities = (
        trusted_authorities
        if trusted_authorities is not None
        else load_trusted_training_authorities(authorities_path)
    )

    if not isinstance(authorities, dict):
        authorities = {}

    auth_entry = authorities.get(verifier_id)
    if not auth_entry:
        if not allow_untrusted_for_test:
            return (
                False,
                "ATTESTATION_UNTRUSTED_SIGNER",
                f"Verifier '{verifier_id}' is not registered in trusted training authorities.",
            )
        pub_key_hex = None
    else:
        if not isinstance(auth_entry, dict) or auth_entry.get("status") != "active":
            return (
                False,
                "ATTESTATION_UNTRUSTED_SIGNER",
                f"Authority '{verifier_id}' is not in active status.",
            )
        pub_key_hex = auth_entry.get("public_key_hex")

    sig_hex = str(attestation.get("signature", "")).strip()
    if not sig_hex:
        return False, "ATTESTATION_INVALID", "Attestation signature field is empty."

    # If pub_key_hex is known, verify Ed25519 signature
    if pub_key_hex:
        try:
            pub_bytes = bytes.fromhex(pub_key_hex)
            sig_bytes = bytes.fromhex(sig_hex)
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            canonical_bytes = construct_canonical_training_payload(attestation)
            pub_key.verify(sig_bytes, canonical_bytes)
        except InvalidSignature:
            return (
                False,
                "ATTESTATION_INVALID",
                "Ed25519 cryptographic signature verification failed: signature mismatch.",
            )
        except Exception as e:
            return (
                False,
                "ATTESTATION_INVALID",
                f"Ed25519 cryptographic signature verification error: {str(e)}",
            )
    elif not allow_untrusted_for_test:
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            f"No public key available for verifier '{verifier_id}'.",
        )

    return (
        True,
        "GENUINE_TRAINED_SOVEREIGN",
        "Cryptographic training attestation successfully verified against trusted authority.",
    )


def create_test_training_attestation(
    checkpoint_bytes_or_path: Union[str, bytes],
    private_key_hex: Optional[str] = None,
    verifier_id: str = "test_authority_01",
    run_id: str = "moon_test_run_001",
    step_count: int = 1000,
    **overrides,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Creates a valid signed training attestation and corresponding authority registration entry.
    Utility helper for test suites and isolated offline verification tests.
    """
    if private_key_hex is None:
        priv_hex, pub_hex = generate_training_authority_keypair()
    else:
        priv_hex = private_key_hex
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
        pub_hex = priv.public_key().public_bytes_raw().hex()

    if isinstance(checkpoint_bytes_or_path, (bytes, bytearray)):
        ckpt_hash = compute_bytes_sha256(bytes(checkpoint_bytes_or_path))
    else:
        ckpt_hash = compute_file_sha256(checkpoint_bytes_or_path)

    now_iso = datetime.now(timezone.utc).isoformat()
    raw_attestation = {
        "run_id": run_id,
        "policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
        "initialization": "random_gaussian_scratch",
        "model_architecture_hash": hashlib.sha256(b"moon_arch_test").hexdigest(),
        "dataset_manifest_hash": hashlib.sha256(b"moon_dataset_test").hexdigest(),
        "training_code_hash": hashlib.sha256(b"moon_code_test").hexdigest(),
        "training_start_time": now_iso,
        "training_completion_time": now_iso,
        "step_count": step_count,
        "metrics": {"train_loss": 2.14, "val_loss": 2.21},
        "checkpoint_sha256": ckpt_hash,
        "verifier_id": verifier_id,
    }
    raw_attestation.update(overrides)

    signed_attestation = sign_training_attestation(raw_attestation, priv_hex)

    authority_entry = {
        verifier_id: {
            "name": f"Test Training Authority ({verifier_id})",
            "public_key_hex": pub_hex,
            "status": "active",
            "registered_at": now_iso,
        }
    }
    return signed_attestation, authority_entry
