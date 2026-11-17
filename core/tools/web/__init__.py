"""
Moon AI — Sovereign Web Intelligence Subsystem
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Training Gate: AUTHORIZED_BY_PREM = False (LOCKED)

Phase 2.7 Milestone 1: Secure Networking & SSRF Defense Foundation.
"""

from .ssrf_guard import (
    SSRFSecurityError,
    MaxRedirectsExceededError,
    is_ip_allowed,
    resolve_and_validate_host,
    validate_url,
    validate_redirect,
    ValidatedURL,
)
from .client import (
    SafeHTTPClient,
    FetchResult,
    FetchStatus,
)

__all__ = [
    "SSRFSecurityError",
    "MaxRedirectsExceededError",
    "is_ip_allowed",
    "resolve_and_validate_host",
    "validate_url",
    "validate_redirect",
    "ValidatedURL",
    "SafeHTTPClient",
    "FetchResult",
    "FetchStatus",
]
