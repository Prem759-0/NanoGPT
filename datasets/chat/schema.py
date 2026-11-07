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
    "Public Domain",
    "CustomSovereign"
}

# Granular provenance taxonomy
PROVENANCE_ORIGINAL_SELF_AUTHORED_QA = "ORIGINAL_SELF_AUTHORED_QA"
PROVENANCE_SOURCE_DERIVED_SUMMARY = "SOURCE_DERIVED_EDUCATIONAL_SUMMARY"
PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY = "SOURCE_DERIVED_EDUCATIONAL_SUMMARY"
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
        if license_name in {"CC0-1.0", "Public Domain", "MIT"}:
            return False, (
                f"Record '{rec_id}' references Wikipedia ('{source_url}') but declares license '{license_name}'. "
                "Wikipedia text is licensed under CC-BY-SA-4.0 (or CC-BY-SA-3.0), never CC0, MIT, or Public Domain."
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
