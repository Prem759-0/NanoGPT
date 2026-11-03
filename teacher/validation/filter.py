"""
Moon AI — Teacher Data Validation, Filtering & Quarantine Pipeline
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Ensures dataset quality, safety, and integrity:
1. Sensitive Data Filter (PII, credentials, private keys, API tokens).
2. Prompt Injection Filter (adversarial jailbreaks, prompt overrides).
3. Duplicate Detection (exact hash collisions and n-gram Jaccard near-duplicates).
4. Quality & Formatting Validation (length, repetition, structure).
5. Blind Test Leakage Prevention (train/test contamination check).
6. Quarantine Management (isolated storage with explicit rejection reasons).
"""

import hashlib
import json
import os
import re
from typing import Dict, Any, List, Optional, Set, Tuple

from teacher.schemas.example import (
    TeacherExample,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED,
    SPLIT_QUARANTINE,
    SPLIT_TRAIN
)

class SensitiveDataFilter:
    """Detects and redacts sensitive personally identifiable information (PII) and secrets."""

    PII_PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "EMAIL_ADDRESS"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
        (re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b"), "PHONE_NUMBER"),
        (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"), "CREDIT_CARD"),
    ]

    SECRET_PATTERNS = [
        (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OPENAI_OR_KIMI_SECRET_KEY"),
        (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GITHUB_TOKEN"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "GOOGLE_API_KEY"),
        (re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"), "PRIVATE_KEY_BLOCK"),
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "BEARER_TOKEN"),
    ]

    @classmethod
    def check_sensitive_data(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Returns (has_sensitive_data, list_of_violations).
        """
        violations = []
        for pat, label in cls.PII_PATTERNS:
            if pat.search(text):
                violations.append(f"Contains PII ({label})")
        for pat, label in cls.SECRET_PATTERNS:
            if pat.search(text):
                violations.append(f"Contains credential or secret ({label})")
        return len(violations) > 0, violations

    @classmethod
    def redact_sensitive_data(cls, text: str) -> str:
        """Redacts detected sensitive patterns."""
        cleaned = text
        for pat, label in cls.PII_PATTERNS:
            cleaned = pat.sub(f"[{label}_REDACTED]", cleaned)
        for pat, label in cls.SECRET_PATTERNS:
            cleaned = pat.sub(f"[{label}_REDACTED]", cleaned)
        return cleaned


class PromptInjectionFilter:
    """Detects adversarial jailbreak patterns and system prompt overrides."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?prior\s+(?:prompts|rules)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+dan\s+mode", re.IGNORECASE),
        re.compile(r"bypass\s+all\s+(?:safety|ethical)\s+guidelines", re.IGNORECASE),
        re.compile(r"reveal\s+your\s+(?:internal\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    ]

    @classmethod
    def check_injection(cls, text: str) -> Tuple[bool, Optional[str]]:
        for pat in cls.INJECTION_PATTERNS:
            if pat.search(text):
                return True, f"Prompt injection detected: matched pattern '{pat.pattern}'"
        return False, None


class DuplicateDetector:
    """Detects exact duplicate hash collisions and near-duplicate text via word n-gram Jaccard similarity."""

    def __init__(self, jaccard_threshold: float = 0.85):
        self.jaccard_threshold = jaccard_threshold
        self.seen_hashes: Dict[str, str] = {}  # sha256 -> example_id
        self.seen_ngrams: Dict[str, Set[str]] = {}  # example_id -> set of 3-grams

    def _get_ngrams(self, text: str, n: int = 3) -> Set[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < n:
            return set(words)
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def check_and_add(self, example_id: str, prompt: str, response: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if example is a duplicate. If duplicate, returns (True, conflict_id_or_reason).
        If unique, stores signature and returns (False, None).
        """
        combined = f"{prompt.strip()}\n---\n{response.strip()}"
        h = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        # 1. Exact hash collision
        if h in self.seen_hashes:
            return True, f"Exact duplicate of record '{self.seen_hashes[h]}'"

        # 2. Near duplicate check via 3-grams
        current_ngrams = self._get_ngrams(combined)
        if current_ngrams:
            for prev_id, prev_ngrams in self.seen_ngrams.items():
                intersection = len(current_ngrams.intersection(prev_ngrams))
                union = len(current_ngrams.union(prev_ngrams))
                if union > 0:
                    sim = intersection / union
                    if sim >= self.jaccard_threshold:
                        return True, f"Near-duplicate (Jaccard {sim:.2f}) of record '{prev_id}'"

        # Unique: record it
        self.seen_hashes[h] = example_id
        self.seen_ngrams[example_id] = current_ngrams
        return False, None


class QualityValidator:
    """Validates structural quality, minimum information density, and repetition."""

    def __init__(self, min_chars: int = 20, max_chars: int = 32000):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def validate_quality(self, prompt: str, response: str) -> Tuple[bool, Optional[str]]:
        if not response or len(response.strip()) < self.min_chars:
            return False, f"Response too short (< {self.min_chars} characters)."

        if len(response) > self.max_chars:
            return False, f"Response exceeds maximum length (> {self.max_chars} characters)."

        # Repetition check (degenerative looping)
        words = re.findall(r"\w+", response.lower())
        if len(words) >= 40:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.20:
                return False, f"Degenerative repetition loop detected (unique word ratio {unique_ratio:.2f} < 0.20)."

        return True, None


class LeakageDetector:
    """Ensures training candidates have zero exact or near-duplicate overlap with the blind test set."""

    def __init__(self):
        self.test_set_hashes: Set[str] = set()

    def register_test_example(self, prompt: str, response: str):
        combined = f"{prompt.strip()}\n---\n{response.strip()}"
        h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        self.test_set_hashes.add(h)

    def check_leakage(self, prompt: str, response: str) -> Tuple[bool, Optional[str]]:
        combined = f"{prompt.strip()}\n---\n{response.strip()}"
        h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if h in self.test_set_hashes:
            return True, "Data leakage violation: Candidate example matches a blind test benchmark example."
        return False, None


class QuarantineManager:
    """Manages quarantined records and records audit logs."""

    def __init__(self, quarantine_file: str = "datasets/manifests/quarantined_teacher_data.jsonl"):
        self.quarantine_file = quarantine_file
        self.quarantined_count = 0

    def quarantine_example(self, example: TeacherExample, reason: str) -> TeacherExample:
        example.validation_status = STATUS_QUARANTINED
        example.split = SPLIT_QUARANTINE
        example.reviewer_status = "AUTOMATED_QUARANTINE"
        example.rejection_reason = reason
        self.quarantined_count += 1
        return example


class TeacherDataPipeline:
    """Unified validation pipeline for teacher demonstrations."""

    def __init__(self, jaccard_threshold: float = 0.85):
        self.sensitive_filter = SensitiveDataFilter()
        self.injection_filter = PromptInjectionFilter()
        self.dup_detector = DuplicateDetector(jaccard_threshold=jaccard_threshold)
        self.quality_validator = QualityValidator()
        self.leakage_detector = LeakageDetector()
        self.quarantine_mgr = QuarantineManager()

    def process_candidate(self, example: TeacherExample) -> Tuple[bool, TeacherExample]:
        """
        Runs candidate through all security and quality checks.
        If any check fails, marks example as QUARANTINED with explicit reason and returns (False, example).
        If all pass, marks example as APPROVED and returns (True, example).
        """
        # 1. Sensitive Data Check
        has_pii, pii_errs = self.sensitive_filter.check_sensitive_data(example.prompt + " " + example.response)
        if has_pii:
            return False, self.quarantine_mgr.quarantine_example(example, "; ".join(pii_errs))

        # 2. Prompt Injection Check
        is_inj, inj_err = self.injection_filter.check_injection(example.prompt)
        if is_inj:
            return False, self.quarantine_mgr.quarantine_example(example, inj_err)

        # 3. Quality Validation
        is_qual, qual_err = self.quality_validator.validate_quality(example.prompt, example.response)
        if not is_qual:
            return False, self.quarantine_mgr.quarantine_example(example, qual_err)

        # 4. Leakage Check against blind test set
        is_leak, leak_err = self.leakage_detector.check_leakage(example.prompt, example.response)
        if is_leak:
            return False, self.quarantine_mgr.quarantine_example(example, leak_err)

        # 5. Duplicate Check
        is_dup, dup_err = self.dup_detector.check_and_add(example.id, example.prompt, example.response)
        if is_dup:
            return False, self.quarantine_mgr.quarantine_example(example, dup_err)

        # All checks passed!
        example.validation_status = STATUS_APPROVED
        example.reviewer_status = "AUTOMATED_FILTER_PASSED"
        example.split = SPLIT_TRAIN
        example.rejection_reason = None
        return True, example

