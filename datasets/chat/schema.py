"""
Moon AI — Sovereign Chat Dataset Schema & Validation Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates multi-turn conversational records with verifiable provenance,
strict open licensing, explicit attribution, source revisions, and zero synthetic fixture contamination.
"""

import hashlib
import json
import re
from typing import Dict, List, Any, Tuple, Optional

APPROVED_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-SA-3.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "0BSD",
    "PSF-2.0",
    "Public Domain",
    "PublicDomain",
    "CustomSovereign",
}

PROHIBITED_LICENSES = {
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "CC-BY-NC-ND",
    "CC-BY-ND",
    "ALL_RIGHTS_RESERVED",
    "PROPRIETARY",
    "UNKNOWN",
    "UNVERIFIED",
}

SUPPORTED_CHAT_CATEGORIES: Dict[str, str] = {
    "general_knowledge": "General knowledge",
    "basic_science": "Basic science",
    "mathematics": "Mathematics and beginner explanations",
    "programming_fundamentals": "Programming fundamentals",
    "python": "Python",
    "javascript": "JavaScript",
    "html_css": "HTML and CSS",
    "web_development": "Web development",
    "computer_science": "Computer science",
    "ai_ml_fundamentals": "AI and machine learning fundamentals",
    "instruction_following": "Instruction following",
    "reasoning_problem_solving": "Reasoning and problem-solving",
    "helpful_conversational": "Helpful conversational responses",
}

CATEGORY_ALIASES: Dict[str, str] = {
    "science": "basic_science",
    "physics": "basic_science",
    "biology": "basic_science",
    "chemistry": "basic_science",
    "beginner_educational": "mathematics",
    "math": "mathematics",
    "software_engineering": "programming_fundamentals",
    "coding": "programming_fundamentals",
    "operating_systems": "computer_science",
    "cryptography": "computer_science",
    "distributed_systems": "computer_science",
    "algorithms": "computer_science",
    "sovereign_identity": "ai_ml_fundamentals",
    "ai_ml": "ai_ml_fundamentals",
    "reasoning_explanations": "reasoning_problem_solving",
    "reasoning": "reasoning_problem_solving",
    "general_conversation": "helpful_conversational",
    "conversational": "helpful_conversational",
    "html": "html_css",
    "css": "html_css",
    "web_dev": "web_development",
    "js": "javascript",
}


def canonicalize_category(category: Optional[str]) -> str:
    """Maps a record category string to one of the 13 canonical Phase 2.7.8 categories or returns normalized raw slug."""
    raw = re.sub(r"[^a-z0-9]+", "_", str(category or "").strip().lower()).strip("_")
    if raw in SUPPORTED_CHAT_CATEGORIES:
        return raw
    return CATEGORY_ALIASES.get(raw, raw or "general_knowledge")


APPROVAL_STATUS_PENDING = "PENDING_REVIEW"
APPROVAL_STATUS_APPROVED = "APPROVED"
APPROVAL_STATUS_REJECTED = "REJECTED"
VALID_APPROVAL_STATUSES = {
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_REJECTED,
}

# Granular provenance taxonomy
PROVENANCE_ORIGINAL_SELF_AUTHORED_QA = "ORIGINAL_SELF_AUTHORED_QA"
PROVENANCE_SOURCE_DERIVED_SUMMARY = "SOURCE_DERIVED_EDUCATIONAL_SUMMARY"
PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY = "SOURCE_DERIVED_EDUCATIONAL_SUMMARY"
PROVENANCE_PUBLIC_SOURCE_CURATED = "PUBLIC_SOURCE_CURATED"
PROVENANCE_GENUINE_HUMAN_CONVERSATION = "GENUINE_HUMAN_CONVERSATION"
PROVENANCE_TEACHER_DISTILLATION_QUALIFIED = "TEACHER_DISTILLATION_OFFLINE_QUALIFIED"
# Stage 2.1.5.1 Audit: Records produced by deterministic Python template expansion
# (not by actual live Qwen2.5-7B-Instruct inference). Content is educationally valid
# and eligible for training but MUST NOT be labelled as Qwen teacher outputs.
# Distinct from SYNTHETIC_DEV_FIXTURE, which is strictly ineligible for production training.
PROVENANCE_DETERMINISTIC_CURRICULUM_TEMPLATE = "DETERMINISTIC_CURRICULUM_TEMPLATE"
PROVENANCE_SYNTHETIC_DEV_FIXTURE = "SYNTHETIC_DEV_FIXTURE"
PROVENANCE_PENDING = "PROVENANCE_PENDING"

ALLOWED_PROVENANCE_TYPES = {
    PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
    PROVENANCE_SOURCE_DERIVED_SUMMARY,
    PROVENANCE_PUBLIC_SOURCE_CURATED,
    PROVENANCE_GENUINE_HUMAN_CONVERSATION,
    PROVENANCE_TEACHER_DISTILLATION_QUALIFIED,
    PROVENANCE_DETERMINISTIC_CURRICULUM_TEMPLATE,
    PROVENANCE_SYNTHETIC_DEV_FIXTURE,
    PROVENANCE_PENDING,
    # Backward compatibility aliases from Stage 2.1.1:
    "ORIGINAL_SELF_AUTHORED",
    "GENUINE_SOURCED_EDUCATIONAL"
}

ALLOWED_ROLES = {"system", "user", "assistant"}

def compute_conversation_sha256(messages: List[Dict[str, str]]) -> str:
    """Computes a deterministic canonical SHA-256 over normalized conversation turns for content integrity."""
    norm_parts = []
    for turn in messages:
        role = turn.get("role", "").strip().lower()
        content = " ".join(turn.get("content", "").split())
        norm_parts.append(f"{role}:{content}")
    canonical_repr = "\n".join(norm_parts)
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    """Normalizes text for fuzzy deduplication."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def validate_chat_record(record: Dict[str, Any], is_production: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validates a chat dataset record against strict sovereign production policies.
    
    Rules:
    1. Must contain 'id', 'messages', 'source', 'source_url', 'license', 'attribution', 'provenance_type', 'category'.
    2. 'provenance_type' must be in ALLOWED_PROVENANCE_TYPES. Missing provenance NEVER silently defaults to genuine data.
    3. 'attribution' must be explicitly provided and non-empty. Missing attribution NEVER silently falls back to source title.
    4. In production mode (is_production=True):
       - 'provenance_type' cannot be 'SYNTHETIC_DEV_FIXTURE' or 'PROVENANCE_PENDING'.
       - 'is_dev_fixture' must not be True.
       - 'source_type' cannot be 'dev_example_fixture'.
       - 'category' cannot contain 'synthetic'.
       - 'source_url' must be a valid URL string starting with http://, https://, or internal repo path.
    5. License validation:
       - Must be in APPROVED_LICENSES.
       - Wikipedia sources MUST NOT be labeled as CC0, MIT, or Public Domain. Wikipedia is CC-BY-SA-4.0 / CC-BY-SA-3.0.
    6. 'messages' must be a list of at least 2 turns (must have >= 1 user and >= 1 assistant turn).
    7. Each turn must have role in {'system', 'user', 'assistant'} and non-empty content (>= 5 chars).
    8. Content integrity: 'sha256' if present must match canonical computed SHA-256.
    """
    if not isinstance(record, dict):
        return False, "Record must be a JSON dictionary."

    rec_id = record.get("id")
    if not rec_id or not isinstance(rec_id, str):
        return False, "Record missing or invalid 'id' string."

    # Synthetic fixture checks (check is_dev_fixture first for clear diagnostics)
    if is_production and record.get("is_dev_fixture") is True:
        return False, f"Record '{rec_id}' flagged as is_dev_fixture=True; forbidden in production."
    if is_production and record.get("source_type") == "dev_example_fixture":
        return False, f"Record '{rec_id}' has source_type='dev_example_fixture'; forbidden in production."

    # Strict provenance validation: missing provenance NEVER automatically assumes genuine data
    prov_type = record.get("provenance_type")
    if not prov_type or not isinstance(prov_type, str):
        return False, f"Record '{rec_id}' missing required 'provenance_type'. Missing provenance must never automatically assume genuine data."

    if prov_type not in ALLOWED_PROVENANCE_TYPES:
        return False, f"Record '{rec_id}' has invalid provenance_type '{prov_type}' (allowed: {ALLOWED_PROVENANCE_TYPES})."

    # Production gate: reject synthetic fixtures and unverified provenance
    if is_production:
        if prov_type == PROVENANCE_SYNTHETIC_DEV_FIXTURE:
            return False, f"Record '{rec_id}' has provenance_type='SYNTHETIC_DEV_FIXTURE'; forbidden in production."
        if prov_type == PROVENANCE_PENDING:
            return False, f"Record '{rec_id}' has provenance_type='PROVENANCE_PENDING'; forbidden in production until verified."
        if "synthetic" in str(record.get("category", "")).lower():
            return False, f"Record '{rec_id}' has synthetic category; forbidden in production."

    license_name = record.get("license")
    if not license_name or license_name not in APPROVED_LICENSES:
        return False, f"Record '{rec_id}' has unapproved or missing license: '{license_name}'."

    source = record.get("source")
    if not source or not isinstance(source, str) or len(source.strip()) < 3:
        return False, f"Record '{rec_id}' missing verifiable 'source'."

    source_url = record.get("source_url")
    if not source_url or not isinstance(source_url, str):
        return False, f"Record '{rec_id}' missing 'source_url'."
    if not (source_url.startswith("http://") or source_url.startswith("https://") or source_url.startswith("file://") or source_url.startswith("internal://") or source_url.startswith("docs/") or source_url.startswith("local://")):
        return False, f"Record '{rec_id}' has invalid 'source_url' format ('{source_url}'). Must be an HTTP(S), file, docs, local, or internal URL."

    # Strict Wikipedia Licensing Check
    if "wikipedia.org" in source_url.lower():
        if license_name in {"CC0-1.0", "Public Domain", "PublicDomain", "MIT"}:
            return False, (
                f"Record '{rec_id}' references Wikipedia ('{source_url}') but declares license '{license_name}'. "
                "Wikipedia text is licensed under CC-BY-SA-4.0 (or CC-BY-SA-3.0), never CC0, MIT, or Public Domain."
            )

    # Approval status check (Phase 2.7.8: unapproved records cannot pass production validation)
    approval_status = record.get("approval_status")
    if is_production and approval_status is not None:
        if approval_status != APPROVAL_STATUS_APPROVED:
            return False, (
                f"Record '{rec_id}' has approval_status='{approval_status}'; "
                "only 'APPROVED' (or verified legacy approved) records may enter production."
            )

    # Missing attribution must NEVER silently fall back to source title
    attribution = record.get("attribution")
    if not attribution or not isinstance(attribution, str) or len(attribution.strip()) < 3:
        return False, f"Record '{rec_id}' missing required 'attribution' field (missing attribution must not silently fall back to source title)."

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"Record '{rec_id}' must contain 'messages' list with >= 2 turns."

    has_user = False
    has_assistant = False

    for i, turn in enumerate(messages):
        if not isinstance(turn, dict):
            return False, f"Record '{rec_id}' turn {i} is not a dictionary."
        role = turn.get("role")
        if role not in ALLOWED_ROLES:
            return False, f"Record '{rec_id}' turn {i} has invalid role '{role}'."
        content = turn.get("content")
        if not content or not isinstance(content, str) or len(content.strip()) < 5:
            return False, f"Record '{rec_id}' turn {i} has empty or trivial content (< 5 characters)."
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    if not has_user:
        return False, f"Record '{rec_id}' missing at least one 'user' message."
    if not has_assistant:
        return False, f"Record '{rec_id}' missing at least one 'assistant' message."

    # Content integrity hash verification
    expected_hash = compute_conversation_sha256(messages)
    provided_hash = record.get("sha256")
    if provided_hash and provided_hash != expected_hash:
        return False, f"Record '{rec_id}' SHA-256 mismatch (provided: {provided_hash}, expected: {expected_hash})."

    return True, None


def validate_conversation_structure_and_safety(
    messages: Any,
    require_strict_alternation: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validates strict conversation UTF-8 integrity, turn roles, alternating structure,
    non-empty content, and end-of-conversation assistant turn.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "Conversation 'messages' must be a list of at least 2 turns."

    idx_start = 0
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        sys_content = messages[0].get("content")
        if not isinstance(sys_content, str) or not sys_content.strip():
            return False, "Leading 'system' message must have non-empty string content."
        idx_start = 1

    conv_turns = messages[idx_start:]
    if len(conv_turns) < 2:
        return False, "Conversation must have at least one user turn and one assistant turn."

    expected_role = "user"
    for offset, turn in enumerate(conv_turns):
        turn_idx = idx_start + offset
        if not isinstance(turn, dict):
            return False, f"Turn[{turn_idx}] must be a dictionary."
        role = turn.get("role")
        if role not in {"user", "assistant"}:
            return False, f"Turn[{turn_idx}] has invalid non-system role '{role}'."
        if require_strict_alternation and role != expected_role:
            return False, (
                f"Turn[{turn_idx}] violates role alternation: expected '{expected_role}', got '{role}'."
            )
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, f"Turn[{turn_idx}] ({role}) has empty or non-string content."
        if "\x00" in content or "\ufffd" in content:
            return False, f"Turn[{turn_idx}] ({role}) contains NUL byte or corrupted UTF-8 replacement character."
        try:
            content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            return False, f"Turn[{turn_idx}] ({role}) contains invalid UTF-8 surrogate/encoding: {exc}"
        expected_role = "assistant" if role == "user" else "user"

    if require_strict_alternation and conv_turns[-1].get("role") != "assistant":
        return False, "Conversation must end with an 'assistant' response turn."

    return True, None
