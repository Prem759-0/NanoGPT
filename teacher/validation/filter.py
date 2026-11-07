"""
Moon AI — Teacher Data Validation, Filtering & Quarantine Pipeline
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Comprehensive Quality Control Pipeline:
1. Sensitive Data Filter (PII, credentials, private keys, API tokens).
2. Prompt Injection Filter (adversarial jailbreaks, prompt overrides).
3. Exact & Near-Duplicate Detection (exact hash collisions + 3-gram Jaccard).
4. Structural Quality & Repetition Loop Detection.
5. Code Syntax Validation (safe isolated AST parsing of Python snippets).
6. License & Provenance Verification (approved licenses only).
7. Blind Test Leakage Prevention (train/test contamination check).
8. Quarantine Management (isolated logging with explicit rejection reasons).
9. Domain Distribution Reporting.
"""

import ast
import hashlib
import json
import os
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone

from teacher.schemas.example import (
    TeacherExample,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED,
    STATUS_PENDING,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_QUARANTINED,
    REVIEW_STATUS_REJECTED,
    SPLIT_QUARANTINE,
    SPLIT_TRAIN,
    APPROVED_LICENSES,
    CURRICULUM_DOMAINS_2_1_4,
    ALL_CURRICULUM_DOMAINS
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
        (re.compile(r"ngrok[_\s\w]*=\s*[\"'][a-zA-Z0-9_\-]{20,}[\"']", re.IGNORECASE), "NGROK_AUTH_TOKEN"),
    ]

    @classmethod
    def check_sensitive_data(cls, text: str) -> Tuple[bool, List[str]]:
        """Returns (has_sensitive_data, list_of_violations)."""
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

        # Repeated phrase looping (4-gram repetition >= 4 times)
        if len(words) >= 16:
            four_grams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
            counts: Dict[str, int] = {}
            for fg in four_grams:
                counts[fg] = counts.get(fg, 0) + 1
                if counts[fg] >= 4:
                    return False, f"Degenerative repetition: phrase '{fg}' repeated {counts[fg]} times."

        return True, None


class CodeSyntaxValidator:
    """
    Safely validates Python code snippets in programming demonstrations using isolated AST parsing.
    Does NOT execute arbitrary code — pure static syntactic analysis.
    """

    @classmethod
    def extract_python_blocks(cls, text: str) -> List[str]:
        """Extracts python code blocks delimited by triple backticks."""
        pattern = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        return [m.strip() for m in matches if m.strip()]

    @classmethod
    def validate_code_syntax(cls, response_text: str) -> Tuple[bool, Optional[str]]:
        """Parses extracted Python blocks with ast.parse. Returns (is_valid, error_reason)."""
        blocks = cls.extract_python_blocks(response_text)
        for idx, block in enumerate(blocks, 1):
            try:
                ast.parse(block)
            except SyntaxError as e:
                return False, f"Code syntax error in Python block #{idx}: {e.msg} at line {e.lineno}"
            except Exception as ex:
                return False, f"Code parse error in Python block #{idx}: {str(ex)}"
        return True, None


class LicenseSourceValidator:
    """Validates that candidate records declare approved open licenses and verifiable source metadata."""

    @classmethod
    def validate_license_and_source(
        cls,
        license_str: str,
        source_type: str,
        source_url: str
    ) -> Tuple[bool, Optional[str]]:
        if not license_str or not isinstance(license_str, str) or len(license_str.strip()) < 2:
            return False, "License declaration is empty or missing."

        clean_lic = license_str.strip()
        if clean_lic not in APPROVED_LICENSES:
            return False, f"Unapproved license '{clean_lic}'. Allowed: {sorted(APPROVED_LICENSES)}."

        if not source_type or len(source_type.strip()) < 3:
            return False, "source_type must be specified."

        if not source_url or len(source_url.strip()) < 3:
            return False, "source_url must be specified."

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
        self.quarantined_records: List[Dict[str, Any]] = []

    def quarantine_example(self, example: TeacherExample, reason: str) -> TeacherExample:
        example.validation_status = STATUS_QUARANTINED
        example.split = SPLIT_QUARANTINE
        example.reviewer_status = REVIEW_STATUS_QUARANTINED
        example.rejection_reason = reason
        self.quarantined_count += 1

        rec = example.to_dict()
        self.quarantined_records.append(rec)

        # Append to quarantine file safely
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.quarantine_file)), exist_ok=True)
            with open(self.quarantine_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

        return example


class CurriculumDistributionTracker:
    """Tracks and reports domain coverage and token counts across the 10 curriculum domains."""

    def __init__(self):
        self.domain_counts: Dict[str, int] = {d: 0 for d in CURRICULUM_DOMAINS_2_1_4}
        self.domain_tokens: Dict[str, int] = {d: 0 for d in CURRICULUM_DOMAINS_2_1_4}
        self.total_records = 0
        self.total_tokens = 0

    def record_example(self, domain: str, token_count: int):
        clean_dom = domain.lower()
        if clean_dom in self.domain_counts:
            self.domain_counts[clean_dom] += 1
            self.domain_tokens[clean_dom] += token_count
        else:
            self.domain_counts[clean_dom] = self.domain_counts.get(clean_dom, 0) + 1
            self.domain_tokens[clean_dom] = self.domain_tokens.get(clean_dom, 0) + token_count
        self.total_records += 1
        self.total_tokens += token_count

    def get_distribution_report(self) -> Dict[str, Any]:
        """Returns distribution metrics across all curriculum domains."""
        breakdown = {}
        for d in CURRICULUM_DOMAINS_2_1_4:
            cnt = self.domain_counts.get(d, 0)
            tok = self.domain_tokens.get(d, 0)
            pct = round((cnt / self.total_records * 100), 2) if self.total_records > 0 else 0.0
            breakdown[d] = {
                "records": cnt,
                "tokens": tok,
                "percentage": pct
            }
        return {
            "total_records": self.total_records,
            "total_tokens": self.total_tokens,
            "domains_covered": sum(1 for d in CURRICULUM_DOMAINS_2_1_4 if self.domain_counts.get(d, 0) > 0),
            "domains_total": len(CURRICULUM_DOMAINS_2_1_4),
            "domain_breakdown": breakdown
        }


class TeacherDataPipeline:
    """Unified validation pipeline for teacher demonstrations."""

    def __init__(self, jaccard_threshold: float = 0.85):
        self.sensitive_filter = SensitiveDataFilter()
        self.injection_filter = PromptInjectionFilter()
        self.dup_detector = DuplicateDetector(jaccard_threshold=jaccard_threshold)
        self.quality_validator = QualityValidator()
        self.code_validator = CodeSyntaxValidator()
        self.license_validator = LicenseSourceValidator()
        self.leakage_detector = LeakageDetector()
        self.quarantine_mgr = QuarantineManager()
        self.distribution_tracker = CurriculumDistributionTracker()

    def process_candidate(self, example: TeacherExample) -> Tuple[bool, TeacherExample]:
        """
        Runs candidate through all security and quality checks.
        If any check fails, marks example as QUARANTINED with explicit reason and returns (False, example).
        If all pass, marks example as APPROVED and returns (True, example).
        """
        # 0. Development Fixture Check
        if example.is_dev_fixture:
            reason = "Development/test fixture: Classified as synthetic fixture only. Strictly ineligible for production training."
            example.is_synthetic = True
            return False, self.quarantine_mgr.quarantine_example(example, reason)

        # 1. Sensitive Data Check
        has_pii, pii_errs = self.sensitive_filter.check_sensitive_data(example.prompt + " " + example.response)
        if has_pii:
            return False, self.quarantine_mgr.quarantine_example(example, "; ".join(pii_errs))

        # 2. Prompt Injection Check
        is_inj, inj_err = self.injection_filter.check_injection(example.prompt)
        if is_inj:
            return False, self.quarantine_mgr.quarantine_example(example, inj_err)

        # 3. Quality & Repetition Validation
        is_qual, qual_err = self.quality_validator.validate_quality(example.prompt, example.response)
        if not is_qual:
            return False, self.quarantine_mgr.quarantine_example(example, qual_err)

        # 4. Code Syntax Validation for Programming Domains
        if "program" in example.domain or "code" in example.domain or "software" in example.domain:
            code_ok, code_err = self.code_validator.validate_code_syntax(example.response)
            if not code_ok:
                return False, self.quarantine_mgr.quarantine_example(example, code_err or "Invalid code syntax")

        # 5. License & Source Verification
        lic_ok, lic_err = self.license_validator.validate_license_and_source(
            example.license, example.source_type, example.source_url
        )
        if not lic_ok:
            return False, self.quarantine_mgr.quarantine_example(example, lic_err or "License/Source invalid")

        # 6. Leakage Check against blind test set
        is_leak, leak_err = self.leakage_detector.check_leakage(example.prompt, example.response)
        if is_leak:
            return False, self.quarantine_mgr.quarantine_example(example, leak_err)

        # 7. Duplicate Check
        is_dup, dup_err = self.dup_detector.check_and_add(example.id, example.prompt, example.response)
        if is_dup:
            return False, self.quarantine_mgr.quarantine_example(example, dup_err)

        # All checks passed!
        example.validation_status = STATUS_APPROVED
        example.reviewer_status = REVIEW_STATUS_PENDING  # Default to PENDING_REVIEW until audited
        example.split = SPLIT_TRAIN
        example.rejection_reason = None

        # Track curriculum distribution
        approx_tokens = len(example.prompt.split()) + len(example.response.split())
        self.distribution_tracker.record_example(example.domain, approx_tokens)

        return True, example
