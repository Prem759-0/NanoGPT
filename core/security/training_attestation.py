"""
core/security/training_attestation.py
Strict Asymmetric Cryptographic Training Provenance & Tamper-Proof Attestation Engine for Moon AI.

Binds candidate model checkpoint SHA-256, random Gaussian scratch initialization proof,
strict sovereign policy, architecture specification, dataset manifest hashes, step counts,
and verifiable training telemetry directly to independent training authorities via Ed25519 signatures.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: Moon Restricted Deterministic Canonical JSON Training Attestation Specification v2.1 (MRDC-JSON-2.1)
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

from core.security.trust_provider import (
    BaseTrustRootProvider,
    ExternalTrustRootProvider,
    TestIsolatedTrustProvider,
    get_default_trust_provider,
)
from core.security.training_evidence import (
    is_valid_hex64,
    parse_iso_utc_timestamp,
    TrainingEvidenceValidator,
    create_sample_training_evidence,
    EvidenceArtifactStore,
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
    "TEST_FIXTURE_VALID_STRUCTURE",
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
    "ATTESTATION_REVOKED_AUTHORITY",
    "ATTESTATION_EXPIRED_AUTHORITY",
    "ATTESTATION_EVIDENCE_INVALID",
    "EVIDENCE_MISSING",
    "EVIDENCE_INVALID",
    "GENUINE_TRAINED_SOVEREIGN",
    "PROMOTED_PRODUCTION",
}

DEFAULT_AUTHORITIES_PATH = os.path.join(
    PROJECT_ROOT, "core", "security", "trusted_training_authorities.json"
)


def load_trusted_training_authorities(
    path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads trusted training authorities from external trust store provider or specified path.
    """
    if path is not None:
        provider = ExternalTrustRootProvider(trust_store_path=path)
        return provider.get_authorities()
    return get_default_trust_provider().get_authorities()


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
    Follows Moon Restricted Deterministic Canonical JSON Specification v2.1 (MRDC-JSON-2.1):
    - Keys sorted lexicographically by UTF-16 code units.
    - Zero whitespace separators (',', ':').
    - Strict UTF-8 encoding.
    - Omits signature field during payload calculation.
    - Floats in metrics formatted deterministically.
    """
    if not isinstance(attestation_dict, dict):
        raise ValueError("Attestation must be a dictionary.")

    payload_fields = {
        "checkpoint_sha256": str(attestation_dict.get("checkpoint_sha256", "")).lower().strip(),
        "dataset_manifest_hash": str(attestation_dict.get("dataset_manifest_hash", "")).lower().strip(),
        "initialization": str(attestation_dict.get("initialization", "")).strip(),
        "metrics": attestation_dict.get("metrics", {}) if isinstance(attestation_dict.get("metrics"), dict) else {},
        "model_architecture_hash": str(attestation_dict.get("model_architecture_hash", "")).lower().strip(),
        "policy": str(attestation_dict.get("policy", "")).strip(),
        "run_id": str(attestation_dict.get("run_id", "")).strip(),
        "step_count": int(attestation_dict.get("step_count", 0)),
        "training_code_hash": str(attestation_dict.get("training_code_hash", "")).lower().strip(),
        "training_completion_time": str(attestation_dict.get("training_completion_time", "")).strip(),
        "training_start_time": str(attestation_dict.get("training_start_time", "")).strip(),
        "verifier_id": str(attestation_dict.get("verifier_id", "")).strip(),
    }
    return json.dumps(payload_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_training_authority_keypair() -> Tuple[str, str]:
    """Generates a fresh Ed25519 signing keypair for testing or authority registration."""
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_hex = priv.private_bytes_raw().hex().lower()
    pub_hex = priv.public_key().public_bytes_raw().hex().lower()
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
    trust_provider: Optional[BaseTrustRootProvider] = None,
    evidence_manifest: Optional[Dict[str, Any]] = None,
    authorities_path: Optional[str] = None,
    allow_untrusted_for_test: bool = False,
    require_production_provider: bool = True,
    evidence_store: Optional[Any] = None,
    require_artifacts: bool = True,
    require_production: Optional[bool] = None,
) -> Tuple[bool, str, str]:
    """
    Performs comprehensive cryptographic verification of training attestation against trust roots:
    1. Validates presence and structure of attestation object.
    2. Enforces presence and non-null values for all 13 required fields.
    3. Recomputes exact raw-bytes SHA-256 of candidate checkpoint file or bytes.
    4. Validates strict policy and Gaussian scratch initialization constraints.
    5. Validates 64-char lowercase hexadecimal hash formatting for architecture, dataset, code, and checkpoint.
    6. Checks chronological timestamp ordering and rejects future timestamps.
    7. Checks verifier identity against configured trust provider (fails closed if unconfigured or revoked).
    8. Asymmetrically verifies Ed25519 signature over MRDC-JSON-2.1 canonical payload.
    9. Enforces presence and containment of physical evidence artifacts in production boundaries.
    10. If evidence manifest is attached or provided: executes deep evidence and disk artifact validation.

    Returns: (is_valid: bool, tier: str, reason: str)
    """
    if require_production is not None:
        require_production_provider = require_production

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

    # Validate hash formats (64-char lowercase hex)
    for hash_field in ("model_architecture_hash", "dataset_manifest_hash", "training_code_hash", "checkpoint_sha256"):
        val = attestation.get(hash_field)
        if not is_valid_hex64(val):
            return (
                False,
                "ATTESTATION_INVALID",
                f"Field '{hash_field}' is not a valid 64-character lowercase hexadecimal hash.",
            )

    # Validate timestamps
    start_dt = parse_iso_utc_timestamp(attestation.get("training_start_time"))
    completion_dt = parse_iso_utc_timestamp(attestation.get("training_completion_time"))
    if not start_dt or not completion_dt:
        return False, "ATTESTATION_INVALID", "Training start and completion timestamps must be valid ISO-8601 UTC strings."
    if start_dt >= completion_dt:
        return (
            False,
            "ATTESTATION_INVALID",
            f"Chronological violation: training start_time ({start_dt}) is not before completion_time ({completion_dt}).",
        )

    # Compute actual checkpoint SHA-256 from raw disk bytes
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

    # Resolve trusted authorities from trust root provider
    if trust_provider is not None:
        resolved_provider = trust_provider
    elif authorities_path is not None:
        resolved_provider = ExternalTrustRootProvider(trust_store_path=authorities_path)
    else:
        resolved_provider = get_default_trust_provider()

    if require_production_provider:
        if not resolved_provider.is_production_provider() or not resolved_provider.is_configured():
            return (
                False,
                "ATTESTATION_UNTRUSTED_SIGNER",
                f"Production verification requires a configured production trust root provider. "
                f"Provider '{resolved_provider.get_provider_type()}' is not an active production provider.",
            )

    authorities = resolved_provider.get_authorities()
    if not isinstance(authorities, dict) or len(authorities) == 0:
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            "No trusted training authorities configured in trust root provider (fail-closed).",
        )

    verifier_id = str(attestation.get("verifier_id", "")).strip()
    auth_entry = authorities.get(verifier_id)
    if not auth_entry:
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            f"Verifier '{verifier_id}' is not registered in trusted training authorities.",
        )

    if not isinstance(auth_entry, dict):
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            f"Authority entry for '{verifier_id}' is malformed.",
        )

    status = str(auth_entry.get("status", "")).strip().lower()
    if status == "revoked":
        return (
            False,
            "ATTESTATION_REVOKED_AUTHORITY",
            f"Authority '{verifier_id}' has been revoked.",
        )
    if status == "expired":
        return (
            False,
            "ATTESTATION_EXPIRED_AUTHORITY",
            f"Authority '{verifier_id}' registration has expired.",
        )

    # Check expiration if valid_until is specified on authority
    valid_until_str = auth_entry.get("valid_until")
    if valid_until_str:
        valid_until_dt = parse_iso_utc_timestamp(valid_until_str)
        now = datetime.now(timezone.utc)
        if valid_until_dt and now > valid_until_dt:
            return (
                False,
                "ATTESTATION_EXPIRED_AUTHORITY",
                f"Authority '{verifier_id}' registration expired at {valid_until_dt}.",
            )

    if status != "active":
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            f"Authority '{verifier_id}' status '{status}' is not active.",
        )

    pub_key_hex = auth_entry.get("public_key_hex")
    if not is_valid_hex64(pub_key_hex):
        return (
            False,
            "ATTESTATION_UNTRUSTED_SIGNER",
            f"Authority '{verifier_id}' has an invalid public key format.",
        )

    sig_hex = str(attestation.get("signature", "")).strip().lower()
    if len(sig_hex) != 128 or not all(c in "0123456789abcdef" for c in sig_hex):
        return False, "ATTESTATION_INVALID", "Attestation signature must be a 128-character (64-byte) hex string."

    # Verify Ed25519 digital signature
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

    # 10. Deep Evidence Manifest Validation (Mandatory for genuine verification)
    ev_data = evidence_manifest or attestation.get("evidence_manifest")
    if ev_data is None:
        return (
            False,
            "ATTESTATION_EVIDENCE_INVALID",
            "Training evidence manifest is strictly mandatory. Unattested or evidence-free models cannot be verified.",
        )
    if not isinstance(ev_data, dict) or not ev_data:
        return (
            False,
            "ATTESTATION_EVIDENCE_INVALID",
            "Training evidence manifest must be a non-empty JSON dictionary.",
        )

    # Production Boundary Enforcement: Physical evidence is strictly mandatory in production
    if require_production_provider:
        require_artifacts = True

    if require_artifacts:
        if evidence_store is None:
            evidence_store = EvidenceArtifactStore()
        if not evidence_store.is_configured():
            msg = (
                "Production verification requires a configured, independently trusted external evidence store (MOON_EVIDENCE_STORE_DIR)."
                if require_production_provider
                else "Physical evidence artifact store is unconfigured or directory does not exist."
            )
            return False, "ATTESTATION_EVIDENCE_INVALID", msg
        if require_production_provider and not evidence_store.is_production_store():
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                "Production verification requires an external evidence store located strictly outside the repository.",
            )

    ckpt_path_str = checkpoint_bytes_or_path if isinstance(checkpoint_bytes_or_path, str) else None
    ev_ok, ev_tier, ev_reason = TrainingEvidenceValidator.validate_evidence_manifest(
        ev_data,
        actual_sha256,
        attestation=attestation,
        evidence_store=evidence_store,
        require_artifacts=require_artifacts,
        checkpoint_file_path=ckpt_path_str,
    )
    if not ev_ok:
        return False, ev_tier, ev_reason

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
    include_evidence: bool = True,
    **overrides,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Creates a valid signed training attestation and corresponding authority registration entry.
    Utility helper for test suites and isolated offline verification tests.
    """
    if private_key_hex is None:
        priv_hex, pub_hex = generate_training_authority_keypair()
    else:
        priv_hex = private_key_hex.lower().strip()
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
        pub_hex = priv.public_key().public_bytes_raw().hex().lower()

    if isinstance(checkpoint_bytes_or_path, (bytes, bytearray)):
        ckpt_hash = compute_bytes_sha256(bytes(checkpoint_bytes_or_path))
    else:
        ckpt_hash = compute_file_sha256(checkpoint_bytes_or_path)

    now = datetime.now(timezone.utc)
    from datetime import timedelta
    start_iso = (now - timedelta(hours=2)).isoformat()
    completion_iso = (now - timedelta(minutes=5)).isoformat()

    ev_m = overrides.get("evidence_manifest")
    st_from_ev = ev_m.get("timestamps", {}).get("start_time") if isinstance(ev_m, dict) else None
    ct_from_ev = ev_m.get("timestamps", {}).get("completion_time") if isinstance(ev_m, dict) else None
    arch_from_ev = ev_m.get("model_architecture", {}).get("model_architecture_hash") if isinstance(ev_m, dict) else None
    data_from_ev = ev_m.get("dataset_manifest", {}).get("dataset_manifest_hash") if isinstance(ev_m, dict) else None
    code_from_ev = ev_m.get("training_code", {}).get("training_code_hash") if isinstance(ev_m, dict) else None
    run_from_ev = ev_m.get("run_id") if isinstance(ev_m, dict) else None

    st = overrides.pop("training_start_time", None) or overrides.pop("start_time", None) or st_from_ev or start_iso
    ct = overrides.pop("training_completion_time", None) or overrides.pop("completion_time", None) or ct_from_ev or completion_iso

    raw_attestation = {
        "run_id": run_from_ev or run_id,
        "policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
        "initialization": "random_gaussian_scratch",
        "model_architecture_hash": arch_from_ev or hashlib.sha256(b"moon_arch_test").hexdigest(),
        "dataset_manifest_hash": data_from_ev or hashlib.sha256(b"moon_dataset_test").hexdigest(),
        "training_code_hash": code_from_ev or hashlib.sha256(b"moon_code_test").hexdigest(),
        "training_start_time": st,
        "training_completion_time": ct,
        "step_count": step_count,
        "metrics": {"train_loss": 2.14, "val_loss": 2.21},
        "checkpoint_sha256": ckpt_hash,
        "verifier_id": verifier_id,
    }
    raw_attestation.update(overrides)

    if "evidence_manifest" in raw_attestation and isinstance(raw_attestation["evidence_manifest"], dict):
        if not raw_attestation["evidence_manifest"].get("checkpoint_sha256") or raw_attestation["evidence_manifest"]["checkpoint_sha256"] == "00" * 32:
            raw_attestation["evidence_manifest"]["checkpoint_sha256"] = ckpt_hash

    if include_evidence and "evidence_manifest" not in raw_attestation:
        ev = create_sample_training_evidence(
            ckpt_hash,
            run_id=run_id,
            step_count=step_count,
            start_time=raw_attestation["training_start_time"],
            completion_time=raw_attestation["training_completion_time"],
            model_architecture={"model_architecture_hash": raw_attestation["model_architecture_hash"]},
            dataset_manifest={"dataset_manifest_hash": raw_attestation["dataset_manifest_hash"]},
            training_code={"training_code_hash": raw_attestation["training_code_hash"]},
        )
        raw_attestation["evidence_manifest"] = ev

    signed_attestation = sign_training_attestation(raw_attestation, priv_hex)

    authority_entry = {
        verifier_id: {
            "name": f"Test Training Authority ({verifier_id})",
            "public_key_hex": pub_hex,
            "status": "active",
            "registered_at": start_iso,
        }
    }
    return signed_attestation, authority_entry
