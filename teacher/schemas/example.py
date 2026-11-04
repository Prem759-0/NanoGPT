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

# 10 Standard Dataset Categories (Stage 2.1.3 Base)
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

# 10 Stage 2.1.4 Curriculum Domains
DOMAIN_214_GENERAL_CONVERSATION = "general_conversation"
DOMAIN_214_INSTRUCTION_FOLLOWING = "instruction_following"
DOMAIN_214_MATH_LOGIC_REASONING = "math_logic_reasoning"
DOMAIN_214_PROGRAMMING_FUNDAMENTALS = "programming_fundamentals"
DOMAIN_214_ADVANCED_PROGRAMMING_DEBUGGING = "advanced_programming_debugging"
DOMAIN_214_SOFTWARE_ARCHITECTURE = "software_architecture"
DOMAIN_214_COMPUTER_SCIENCE = "computer_science"
DOMAIN_214_RESEARCH_EVIDENCE = "research_evidence"
DOMAIN_214_TOOL_SELECTION_USAGE = "tool_selection_usage"
DOMAIN_214_SAFETY_UNCERTAINTY = "safety_uncertainty"

CURRICULUM_DOMAINS_2_1_4: List[str] = [
    DOMAIN_214_GENERAL_CONVERSATION,
    DOMAIN_214_INSTRUCTION_FOLLOWING,
    DOMAIN_214_MATH_LOGIC_REASONING,
    DOMAIN_214_PROGRAMMING_FUNDAMENTALS,
    DOMAIN_214_ADVANCED_PROGRAMMING_DEBUGGING,
    DOMAIN_214_SOFTWARE_ARCHITECTURE,
    DOMAIN_214_COMPUTER_SCIENCE,
    DOMAIN_214_RESEARCH_EVIDENCE,
    DOMAIN_214_TOOL_SELECTION_USAGE,
    DOMAIN_214_SAFETY_UNCERTAINTY
]

DOMAIN_ALIASES: Dict[str, str] = {
    "instruction_following": DOMAIN_MULTI_TURN_INSTRUCTION,
    "programming_fundamentals": "programming_fundamentals",
    "advanced_programming_debugging": DOMAIN_PROGRAMMING_DEBUGGING,
    "computer_science": "computer_science",
    "tool_selection_usage": DOMAIN_TOOL_SELECTION_EXECUTION,
    "safety_uncertainty": DOMAIN_SAFETY_UNCERTAINTY,
    DOMAIN_MULTI_TURN_INSTRUCTION: "instruction_following",
    DOMAIN_PROGRAMMING_DEBUGGING: "advanced_programming_debugging",
    DOMAIN_TOOL_SELECTION_EXECUTION: "tool_selection_usage",
    DOMAIN_SAFETY_UNCERTAINTY: "safety_uncertainty"
}

ALL_CURRICULUM_DOMAINS: Set[str] = VALID_DOMAINS.union(set(CURRICULUM_DOMAINS_2_1_4))

# Approved Open & Permissive Licenses (Verified SPDX Licenses)
# Note (Stage 2.1.5 Audit): 'Permissive-Teacher-Terms' was rejected and removed as an unverified non-SPDX identifier.
APPROVED_LICENSES: Set[str] = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MIT",
    "Apache-2.0",
    "CustomSovereign"
}

# Validation and Split Statuses
STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_QUARANTINED = "QUARANTINED"
STATUS_REJECTED = "REJECTED"

# Review Statuses (Stage 2.1.4)
REVIEW_STATUS_PENDING = "PENDING_REVIEW"
REVIEW_STATUS_APPROVED = "APPROVED"
REVIEW_STATUS_QUARANTINED = "QUARANTINED"
REVIEW_STATUS_REJECTED = "REJECTED"

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
    # Stage 2.1.4 Metadata Extensions
    source_type: str = "OFFLINE_TEACHER_DEMONSTRATION"
    source_url: str = "local://offline_curator"
    teacher_checkpoint_hash: Optional[str] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    is_dev_fixture: bool = False
    is_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["content_hash"] = self.sha256_hash
        d["review_status"] = self.reviewer_status
        d["provenance_type"] = self.provenance
        if d.get("generation_metadata") is None:
            d["generation_metadata"] = {
                "timestamp": self.generation_timestamp,
                "teacher_provider": self.teacher_provider,
                "teacher_model_id": self.teacher_model_id,
                "task_type": self.task_type
            }
        return d

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
        license_str: str = "CC-BY-4.0",
        messages: Optional[List[Dict[str, str]]] = None,
        quality_score: float = 90.0,
        split: str = SPLIT_TRAIN,
        rejection_reason: Optional[str] = None,
        source_type: str = "OFFLINE_TEACHER_DEMONSTRATION",
        source_url: str = "local://offline_curator",
        teacher_checkpoint_hash: Optional[str] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
        is_dev_fixture: bool = False,
        is_synthetic: bool = False,
        reviewer_status: Optional[str] = None
    ) -> "TeacherExample":
        h = compute_example_sha256(prompt, response)
        now_utc = datetime.now(timezone.utc).isoformat()
        msgs = messages or [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        status = STATUS_REJECTED if rejection_reason else STATUS_PENDING
        rev_status = reviewer_status or ("REJECTED_ON_CREATION" if rejection_reason else "PENDING_REVIEW")
        
        # If dev fixture, mark as synthetic
        if is_dev_fixture:
            is_synthetic = True
            if not rejection_reason:
                rev_status = "DEV_FIXTURE_MOCK"

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
            rejection_reason=rejection_reason,
            source_type=source_type,
            source_url=source_url,
            teacher_checkpoint_hash=teacher_checkpoint_hash,
            generation_metadata=generation_metadata or {
                "timestamp": now_utc,
                "teacher_provider": teacher_provider,
                "teacher_model_id": teacher_model_id,
                "task_type": task_type
            },
            is_dev_fixture=is_dev_fixture,
            is_synthetic=is_synthetic
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
    if domain not in ALL_CURRICULUM_DOMAINS:
        return False, f"Record '{rec_id}' has invalid domain '{domain}' (allowed: {ALL_CURRICULUM_DOMAINS})."

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
        if record.get("is_dev_fixture") is True:
            return False, (
                f"Production violation: Record '{rec_id}' is marked is_dev_fixture=True. "
                "Development and mock fixtures are strictly prohibited from production manifests."
            )
        if split == SPLIT_QUARANTINE:
            return False, f"Production violation: Record '{rec_id}' is marked for quarantine split."
        rev = record.get("review_status") or record.get("reviewer_status")
        if rev and rev in (REVIEW_STATUS_REJECTED, REVIEW_STATUS_QUARANTINED, "DEV_FIXTURE_MOCK", "REJECTED_ON_CREATION"):
            return False, f"Production violation: Record '{rec_id}' has review status '{rev}'."
        if status != STATUS_APPROVED:
            rev_label = rev or "PENDING"
            return False, (
                f"Production violation: Record '{rec_id}' has review status '{rev_label}'. "
                f"Status '{STATUS_APPROVED}' required before production manifest entry."
            )
        if lic not in APPROVED_LICENSES:
            return False, f"Production violation: Record '{rec_id}' has unapproved license '{lic}'."

    # Message structure check
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"Record '{rec_id}' must have at least 2 turns in 'messages'."

    # Content hash check
    expected_hash = compute_example_sha256(record["prompt"], record["response"])
    if record.get("sha256_hash") and record.get("sha256_hash") != expected_hash:
        return False, (
            f"Content integrity failure in record '{rec_id}': "
            f"declared hash '{record['sha256_hash']}' != computed '{expected_hash}'."
        )
    if record.get("content_hash") and record.get("content_hash") != expected_hash:
        return False, (
            f"Content integrity failure in record '{rec_id}': "
            f"declared hash '{record['content_hash']}' != computed '{expected_hash}'."
        )

    return True, "Valid teacher demonstration record."

