"""
core/security package
Sovereign security, integrity, filtering, external trust, and evidence engines for Moon AI.
"""

from core.security.filtering import (
    SensitiveDataFilter,
    PromptInjectionFilter,
    InputIntegrityValidator
)
from core.security.trust_provider import (
    BaseTrustRootProvider,
    ExternalTrustRootProvider,
    TestIsolatedTrustProvider,
    get_default_trust_provider,
)
from core.security.training_evidence import (
    TrainingEvidenceValidator,
    create_sample_training_evidence,
    is_valid_hex64,
    parse_iso_utc_timestamp,
)
from core.security.training_attestation import (
    compute_file_sha256,
    compute_bytes_sha256,
    construct_canonical_training_payload,
    generate_training_authority_keypair,
    sign_training_attestation,
    verify_training_attestation,
    create_test_training_attestation,
    VALID_TRAINING_PROVENANCE_TIERS,
)

__all__ = [
    "SensitiveDataFilter",
    "PromptInjectionFilter",
    "InputIntegrityValidator",
    "BaseTrustRootProvider",
    "ExternalTrustRootProvider",
    "TestIsolatedTrustProvider",
    "get_default_trust_provider",
    "TrainingEvidenceValidator",
    "create_sample_training_evidence",
    "is_valid_hex64",
    "parse_iso_utc_timestamp",
    "compute_file_sha256",
    "compute_bytes_sha256",
    "construct_canonical_training_payload",
    "generate_training_authority_keypair",
    "sign_training_attestation",
    "verify_training_attestation",
    "create_test_training_attestation",
    "VALID_TRAINING_PROVENANCE_TIERS",
]
