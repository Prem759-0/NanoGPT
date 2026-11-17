"""
Moon AI — Core Engine & Platform Services
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Active Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""

from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    ACTIVE_OWNERSHIP_POLICY,
    VALID_PRODUCTION_POLICIES,
    HISTORICAL_QUARANTINED_POLICIES,
    ALL_KNOWN_POLICIES,
    VALID_OWNERSHIP_POLICIES,
    SovereignPolicyViolationError,
    get_active_policy,
    is_valid_policy,
    is_historical_policy,
    validate_production_policy,
    validate_student_initialization,
    get_policy_summary,
    assert_production_training_authorized,
)

__all__ = [
    "POLICY_STRICT_SOVEREIGN_RANDOM_INIT",
    "POLICY_TEACHER_STUDENT_RANDOM_INIT",
    "ACTIVE_OWNERSHIP_POLICY",
    "VALID_PRODUCTION_POLICIES",
    "HISTORICAL_QUARANTINED_POLICIES",
    "ALL_KNOWN_POLICIES",
    "VALID_OWNERSHIP_POLICIES",
    "SovereignPolicyViolationError",
    "get_active_policy",
    "is_valid_policy",
    "is_historical_policy",
    "validate_production_policy",
    "validate_student_initialization",
    "get_policy_summary",
    "assert_production_training_authorized",
]
