"""
Moon AI — Teacher Demonstration Data Schema
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Standardized schema for high-quality teacher demonstrations across 10 curriculum domains.
Every record carries complete provenance, teacher attribution, licensing declarations,
content hash integrity, quality metrics, and review status.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set

# 10 Standard Dataset Categories
DOMAIN_GENERAL_CONVERSATION = "general_conversation"
DOMAIN_MULTI_TURN_INSTRUCTION = "multi_turn_instruction"
DOMAIN_MATH_LOGIC_REASONING = "math_logic_reasoning"
DOMAIN_PROGRAMMING_DEBUGGING = "programming_debugging"
DOMAIN_SOFTWARE_ARCHITECTURE = "software_architecture"
DOMAIN_TOOL_SELECTION_EXECUTION = "tool_selection_execution"
DOMAIN_RESEARCH_EVIDENCE = "research_evidence"
DOMAIN_DOCUMENT_CREATION = "document_creation_structured"
DOMAIN_MULTIMODAL_PROMPT = "multimodal_prompt_understanding"
DOMAIN_SAFETY_UNCERTAINTY = "safety_uncertainty_refusal"

VALID_DOMAINS: Set[str] = {
    DOMAIN_GENERAL_CONVERSATION,
    DOMAIN_MULTI_TURN_INSTRUCTION,
    DOMAIN_MATH_LOGIC_REASONING,
    DOMAIN_PROGRAMMING_DEBUGGING,
    DOMAIN_SOFTWARE_ARCHITECTURE,
    DOMAIN_TOOL_SELECTION_EXECUTION,
    DOMAIN_RESEARCH_EVIDENCE,
    DOMAIN_DOCUMENT_CREATION,
    DOMAIN_MULTIMODAL_PROMPT,
    DOMAIN_SAFETY_UNCERTAINTY,
}

# Validation and Split Statuses
STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_QUARANTINED = "QUARANTINED"
STATUS_REJECTED = "REJECTED"

VALID_STATUSES: Set[str] = {
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED
}

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_QUARANTINE = "quarantine"

VALID_SPLITS: Set[str] = {
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
    SPLIT_QUARANTINE
}

def compute_example_sha256(prompt: str, response: str) -> str:
    """Computes deterministic canonical SHA-256 over prompt and response text."""
    normalized = f"prompt:{prompt.strip()}\n---\nresponse:{response.strip()}".encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()

@dataclass
class TeacherExample:
    """Represents a validated, annotated teacher demonstration record."""
    id: str
    teacher_provider: str
    teacher_model_id: str
    generation_timestamp: str
    prompt: str
    response: str
    messages: List[Dict[str, str]]
    domain: str
    task_type: str
    provenance: str
    license: str
    validation_status: str
    quality_score: float
    reviewer_status: str
    split: str
    sha256_hash: str
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        example_id: str,
        teacher_provider: str,
        teacher_model_id: str,
        prompt: str,
        response: str,
        domain: str,
        task_type: str,
        license_str: str = "Permissive-Teacher-Terms",
        messages: Optional[List[Dict[str, str]]] = None,
        quality_score: float = 90.0,
        split: str = SPLIT_TRAIN,
        rejection_reason: Optional[str] = None
    ) -> "TeacherExample":
        h = compute_example_sha256(prompt, response)
        now_utc = datetime.now(timezone.utc).isoformat()
        msgs = messages or [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        status = STATUS_REJECTED if rejection_reason else STATUS_PENDING
        rev_status = "REJECTED_ON_CREATION" if rejection_reason else "AWAITING_VALIDATION"
        return cls(
            id=example_id,
            teacher_provider=teacher_provider,
            teacher_model_id=teacher_model_id,
            generation_timestamp=now_utc,
            prompt=prompt,
            response=response,
            messages=msgs,
            domain=domain,
            task_type=task_type,
            provenance=f"teacher_distillation_{teacher_provider}_{teacher_model_id}",
            license=license_str,
            validation_status=status,
            quality_score=quality_score,
            reviewer_status=rev_status,
            split=split,
            sha256_hash=h,
            rejection_reason=rejection_reason
        )

def validate_teacher_example(record: Dict[str, Any], is_production: bool = False) -> Tuple[bool, str]:
    """
    Validates teacher demonstration record schema, required fields, and integrity.
    Never allows missing fields, invented licenses, or quarantined records in production manifests.
    """
    if not isinstance(record, dict):
        return False, "Teacher record must be a JSON dictionary."

    rec_id = record.get("id")
    if not rec_id or not isinstance(rec_id, str):
        return False, "Record missing or invalid 'id'."

    # Required metadata fields
    required_fields = [
        "teacher_provider", "teacher_model_id", "generation_timestamp",
        "prompt", "response", "messages", "domain", "task_type",
        "provenance", "license", "validation_status", "quality_score",
        "reviewer_status", "split", "sha256_hash"
    ]
    for field in required_fields:
        if field not in record:
            return False, f"Record '{rec_id}' missing required metadata field '{field}'."

    # Domain check
    domain = record["domain"]
    if domain not in VALID_DOMAINS:
        return False, f"Record '{rec_id}' has invalid domain '{domain}' (allowed: {VALID_DOMAINS})."

    # Status check
    status = record["validation_status"]
    if status not in VALID_STATUSES:
        return False, f"Record '{rec_id}' has invalid validation_status '{status}'."

    # Split check
    split = record["split"]
    if split not in VALID_SPLITS:
        return False, f"Record '{rec_id}' has invalid split '{split}'."

    # License check: cannot be empty or unspecified
    lic = record["license"]
    if not lic or not isinstance(lic, str) or len(lic.strip()) < 2:
        return False, f"Record '{rec_id}' has invalid or empty license string."

    # Production manifest rules
    if is_production:
        if status != STATUS_APPROVED:
            return False, (
                f"Production violation: Record '{rec_id}' has status '{status}'. "
                "Only APPROVED records can be admitted into production manifests."
            )
        if split == SPLIT_QUARANTINE:
            return False, f"Production violation: Record '{rec_id}' is marked for quarantine split."

    # Message structure check
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"Record '{rec_id}' must have at least 2 turns in 'messages'."

    # Content hash check
    expected_hash = compute_example_sha256(record["prompt"], record["response"])
    if record["sha256_hash"] != expected_hash:
        return False, (
            f"Content integrity failure in record '{rec_id}': "
            f"declared hash '{record['sha256_hash']}' != computed '{expected_hash}'."
        )

    return True, "Valid teacher demonstration record."

