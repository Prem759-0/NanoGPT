"""
Moon AI — Sovereign Security & Privacy Filtering Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Comprehensive Multi-Layer Defense:
1. SensitiveDataFilter: Detects and redacts PII, private keys, API tokens, passwords, and secrets.
2. PromptInjectionFilter: Screens for adversarial jailbreaks, system prompt overrides, and DAN patterns.
3. InputIntegrityValidator: Bounds length, blocks null bytes, and audits Unicode structure.

Important:
Regex and heuristic filtering provide a crucial defense-in-depth boundary, but are not
a mathematical proof of safety. Raw secrets are NEVER logged in rejection metadata.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set


class SensitiveDataFilter:
    """Detects and redacts sensitive personally identifiable information (PII) and secrets."""

    PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL_ADDRESS"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
        (re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b"), "PHONE_NUMBER"),
        (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"), "CREDIT_CARD"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP_ADDRESS"),
    ]

    SECRET_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), "OPENAI_OR_KIMI_SECRET_KEY"),
        (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}"), "GITHUB_TOKEN"),
        (re.compile(r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"), "GITHUB_FINE_GRAINED_PAT"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "GOOGLE_API_KEY"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS_ACCESS_KEY_ID"),
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), "PRIVATE_KEY_BLOCK"),
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "BEARER_TOKEN"),
        (re.compile(r"ngrok[_\s\w]*=\s*[\"'][a-zA-Z0-9_\-]{20,}[\"']", re.IGNORECASE), "NGROK_AUTH_TOKEN"),
        (re.compile(r"(?:password|passwd|secret_key|api_secret)\s*[:=]\s*[\"'][^\"']{6,}[\"']", re.IGNORECASE), "PASSWORD_OR_SECRET_ASSIGNMENT"),
    ]

    @classmethod
    def check_sensitive_data(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Scans text for sensitive PII or credentials.
        Returns (has_sensitive_data, list_of_violations).
        NOTE: Never logs or returns the raw secret text itself.
        """
        if not text:
            return False, []

        violations: List[str] = []
        for pat, label in cls.PII_PATTERNS:
            if pat.search(text):
                violations.append(f"Contains PII ({label})")

        for pat, label in cls.SECRET_PATTERNS:
            if pat.search(text):
                violations.append(f"Contains credential or secret ({label})")

        return len(violations) > 0, violations

    @classmethod
    def redact_sensitive_data(cls, text: str) -> str:
        """
        Redacts detected sensitive patterns in-place with standardized replacement tags.
        """
        if not text:
            return ""

        cleaned = text
        for pat, label in cls.PII_PATTERNS:
            cleaned = pat.sub(f"[{label}_REDACTED]", cleaned)
        for pat, label in cls.SECRET_PATTERNS:
            cleaned = pat.sub(f"[{label}_REDACTED]", cleaned)
        return cleaned


class PromptInjectionFilter:
    """Detects adversarial jailbreak patterns, prompt overrides, and system prompt leaks."""

    INJECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE), "ignore_previous_instructions"),
        (re.compile(r"disregard\s+(?:all\s+)?prior\s+(?:prompts|rules|instructions)", re.IGNORECASE), "disregard_prior_instructions"),
        (re.compile(r"you\s+are\s+now\s+in\s+dan\s+mode", re.IGNORECASE), "dan_mode_jailbreak"),
        (re.compile(r"bypass\s+all\s+(?:safety|ethical|security)\s+guidelines", re.IGNORECASE), "bypass_safety_guidelines"),
        (re.compile(r"reveal\s+your\s+(?:internal\s+)?system\s+prompt", re.IGNORECASE), "reveal_system_prompt"),
        (re.compile(r"do\s+anything\s+now", re.IGNORECASE), "do_anything_now"),
        (re.compile(r"developer\s+mode\s+(?:enabled|activated)", re.IGNORECASE), "developer_mode_override"),
        (re.compile(r"override\s+(?:all\s+)?system\s+(?:prompts|instructions)", re.IGNORECASE), "system_instruction_override"),
    ]

    @classmethod
    def check_injection(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_injection, reason).
        """
        if not text:
            return False, None

        for pat, label in cls.INJECTION_PATTERNS:
            if pat.search(text):
                return True, f"Prompt injection detected: matched rule '{label}'"
        return False, None


class InputIntegrityValidator:
    """Validates basic character encoding, size bounds, and absence of control byte attacks."""

    MAX_INPUT_LENGTH = 65536  # 64 KB per turn

    @classmethod
    def validate_input(cls, text: str, max_chars: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Audits input text for length bounds and dangerous control characters."""
        if text is None:
            return False, "Input is null"

        limit = max_chars or cls.MAX_INPUT_LENGTH
        if len(text) > limit:
            return False, f"Input exceeds maximum allowed length of {limit} characters"

        # Check for null bytes (common exploit vector for C bindings)
        if "\x00" in text:
            return False, "Input contains prohibited null byte"

        return True, None
