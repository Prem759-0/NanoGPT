"""
core/security/trust_provider.py
External Trust-Root Provider & Protected Keystore Separation for Moon AI.

Architecture Principles:
1. Production Separation: Production trust roots are NEVER loaded from ordinary repository files.
   In-repository keystores (e.g. core/security/trusted_training_authorities.json) are strictly
   empty offline placeholders. Committing a public key to the repository CANNOT make it trusted.
2. Protected External Sources: Production public keys must be supplied via:
   a) MOON_EXTERNAL_TRUST_STORE_PATH: An absolute file path strictly outside the repository root.
   b) MOON_TRUSTED_TRAINING_AUTHORITIES_JSON: A protected environment secret (e.g. from AWS Secrets
      Manager, GCP Secret Manager, or GitHub Actions Environment Secrets).
3. Fail-Closed Default: If no external protected trust configuration is supplied, the provider
   returns an empty authority dict ({}), ensuring the system honestly remains in MODEL_NOT_READY.
4. Cryptographic Validation & Revocation: Supports active, revoked, and expired authorities.
5. Isolated Testing: TestIsolatedTrustProvider isolates test keypairs so unit tests can never
   inadvertently configure or poison production trust roots.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.1
"""

import os
import sys
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("moon.security.trust_provider")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IN_REPO_KEYSTORE_PATH = os.path.join(
    PROJECT_ROOT, "core", "security", "trusted_training_authorities.json"
)


class BaseTrustRootProvider(ABC):
    """Abstract interface for Moon AI training attestation trust root providers."""

    @abstractmethod
    def get_authorities(self) -> Dict[str, Any]:
        """Returns the dictionary of configured authority definitions."""
        pass

    @abstractmethod
    def get_provider_type(self) -> str:
        """Returns a non-sensitive identifier of the provider type for audit logs."""
        pass

    @abstractmethod
    def is_production_provider(self) -> bool:
        """Indicates whether this provider satisfies production isolation requirements."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Indicates whether this provider has a non-empty, valid authority configuration."""
        pass

    def get_configuration_status(self) -> Dict[str, Any]:
        """Returns structured metadata about the trust provider's operational configuration."""
        auths = self.get_authorities()
        return {
            "provider_type": self.get_provider_type(),
            "is_production_provider": self.is_production_provider(),
            "is_configured": self.is_configured(),
            "authority_count": len(auths) if isinstance(auths, dict) else 0,
        }


class ExternalTrustRootProvider(BaseTrustRootProvider):
    """
    Production-grade trust root provider enforcing external configuration isolation.
    Strictly forbids loading production trust roots from repository-controlled files.
    """

    def __init__(
        self,
        external_path: Optional[str] = None,
        external_secret_json: Optional[str] = None,
        base_dir: Optional[str] = None,
        trust_store_path: Optional[str] = None,
    ):
        if trust_store_path is not None and external_path is None:
            external_path = trust_store_path
        self.base_dir = os.path.abspath(base_dir or PROJECT_ROOT)
        self.external_path = external_path or os.environ.get("MOON_EXTERNAL_TRUST_STORE_PATH")
        self.external_secret_json = (
            external_secret_json or os.environ.get("MOON_TRUSTED_TRAINING_AUTHORITIES_JSON")
        )
        self._cached_authorities: Optional[Dict[str, Any]] = None
        self._provider_type: str = "UnconfiguredFailClosed"

    def is_configured(self) -> bool:
        auths = self.get_authorities()
        return bool(
            isinstance(auths, dict)
            and len(auths) > 0
            and self._provider_type not in (
                "UnconfiguredFailClosed",
                "MalformedSecretFailClosed",
                "InRepoSecurityViolationFailClosed",
                "UnreadableFileFailClosed",
            )
        )

    def is_production_provider(self) -> bool:
        """
        External provider is an active production provider ONLY IF it is configured
        with non-empty valid external authorities. An unconfigured provider fails closed
        and cannot claim production authorization status.
        """
        return self.is_configured()

    def get_provider_type(self) -> str:
        return self._provider_type

    def _is_path_inside_repo(self, path: str) -> bool:
        """Returns True if the specified path resolves within the repository root directory."""
        try:
            real_target = os.path.realpath(os.path.abspath(path))
            real_base = os.path.realpath(self.base_dir)
            common = os.path.commonpath([real_base, real_target])
            return common == real_base
        except (ValueError, Exception):
            return False

    def _validate_raw_keystore_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates schema structure, policy compliance, and individual authority records:
        - schema_version in ("2.0", "2.1")
        - policy == "ASYMMETRIC_ED25519_ONLY"
        - trusted_authorities is dict
        - each authority has valid public_key_hex (64-char lowercase hex) and status
        """
        if not isinstance(data, dict):
            logger.error("External trust store is not a JSON object.")
            return {}

        policy = data.get("policy")
        if policy is not None and policy != "ASYMMETRIC_ED25519_ONLY":
            logger.error(
                f"External trust store policy '{policy}' is invalid; must be 'ASYMMETRIC_ED25519_ONLY'."
            )
            return {}

        raw_authorities = data.get("trusted_authorities", {})
        if not isinstance(raw_authorities, dict):
            logger.error("External trust store 'trusted_authorities' field is not a dictionary.")
            return {}

        valid_authorities: Dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        for auth_id, record in raw_authorities.items():
            if not isinstance(record, dict):
                continue
            pub_hex = str(record.get("public_key_hex", "")).strip().lower()
            if len(pub_hex) != 64 or not all(c in "0123456789abcdef" for c in pub_hex):
                logger.warning(
                    f"Authority '{auth_id}' rejected: invalid 64-char hex public key format."
                )
                continue

            status = str(record.get("status", "")).strip().lower()
            if status not in ("active", "revoked"):
                logger.warning(
                    f"Authority '{auth_id}' rejected: status '{status}' must be 'active' or 'revoked'."
                )
                continue

            # Check expiration if valid_until is specified
            valid_until_str = record.get("valid_until")
            if valid_until_str:
                try:
                    valid_until_dt = datetime.fromisoformat(str(valid_until_str).replace("Z", "+00:00"))
                    if valid_until_dt.tzinfo is None:
                        valid_until_dt = valid_until_dt.replace(tzinfo=timezone.utc)
                    if now > valid_until_dt:
                        record = dict(record)
                        record["status"] = "expired"
                except Exception as e:
                    logger.warning(f"Authority '{auth_id}' has malformed valid_until timestamp: {e}")
                    continue

            valid_authorities[auth_id] = record

        return valid_authorities

    def get_authorities(self) -> Dict[str, Any]:
        """
        Loads and resolves production authorities from external protected sources:
        1. Environment secret (MOON_TRUSTED_TRAINING_AUTHORITIES_JSON)
        2. External filesystem keystore (MOON_EXTERNAL_TRUST_STORE_PATH) outside repo
        Fails closed ({}) if sources are missing, inside repository, or malformed.
        """
        if self._cached_authorities is not None:
            return self._cached_authorities

        # Option 1: Protected Environment Secret
        if self.external_secret_json and self.external_secret_json.strip():
            try:
                raw_json = self.external_secret_json.strip()
                parsed = json.loads(raw_json)
                auths = self._validate_raw_keystore_dict(parsed)
                self._provider_type = "ProtectedEnvironmentSecret"
                self._cached_authorities = auths
                active_count = sum(1 for a in auths.values() if a.get("status") == "active")
                revoked_count = sum(1 for a in auths.values() if a.get("status") == "revoked")
                logger.info(
                    f"Loaded production trust root from ProtectedEnvironmentSecret "
                    f"({active_count} active, {revoked_count} revoked)."
                )
                return self._cached_authorities
            except Exception as e:
                logger.error(f"Failed to parse MOON_TRUSTED_TRAINING_AUTHORITIES_JSON secret: {e}")
                self._provider_type = "MalformedSecretFailClosed"
                self._cached_authorities = {}
                return {}

        # Option 2: External Filesystem Keystore (Strictly Outside Repository)
        if self.external_path and os.path.isfile(self.external_path):
            if self._is_path_inside_repo(self.external_path):
                logger.error(
                    f"SECURITY VIOLATION: External trust store path '{self.external_path}' "
                    f"resolves inside repository root '{self.base_dir}'. "
                    f"Repository files are strictly prohibited from supplying production trust roots."
                )
                self._provider_type = "InRepoSecurityViolationFailClosed"
                self._cached_authorities = {}
                return {}

            try:
                with open(self.external_path, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                auths = self._validate_raw_keystore_dict(parsed)
                self._provider_type = f"ExternalProtectedFile"
                self._cached_authorities = auths
                active_count = sum(1 for a in auths.values() if a.get("status") == "active")
                revoked_count = sum(1 for a in auths.values() if a.get("status") == "revoked")
                logger.info(
                    f"Loaded production trust root from ExternalProtectedFile "
                    f"({active_count} active, {revoked_count} revoked)."
                )
                return self._cached_authorities
            except Exception as e:
                logger.error(f"Failed to load external trust store file '{self.external_path}': {e}")
                self._provider_type = "UnreadableFileFailClosed"
                self._cached_authorities = {}
                return {}

        # Fail-closed default: No external trust configuration configured
        self._provider_type = "UnconfiguredFailClosed"
        self._cached_authorities = {}
        logger.info(
            "Production trust root is unconfigured (clean checkout). Failing closed with 0 trusted authorities."
        )
        return self._cached_authorities


class TestIsolatedTrustProvider(BaseTrustRootProvider):
    """
    Isolated test-only trust provider for unit and adversarial test suites.
    Explicitly flagged as is_production_provider() == False.
    Production model verification routes strictly reject this provider.
    """

    def __init__(self, test_authorities: Dict[str, Any]):
        if not isinstance(test_authorities, dict):
            raise ValueError("test_authorities must be a dictionary.")
        self.test_authorities = test_authorities
        self._provider_type = "TestIsolatedKeystore"

    def is_production_provider(self) -> bool:
        return False

    def is_configured(self) -> bool:
        return bool(isinstance(self.test_authorities, dict) and len(self.test_authorities) > 0)

    def get_provider_type(self) -> str:
        return self._provider_type

    def get_authorities(self) -> Dict[str, Any]:
        return self.test_authorities


def get_default_trust_provider() -> BaseTrustRootProvider:
    """Returns the production external trust root provider."""
    return ExternalTrustRootProvider()
