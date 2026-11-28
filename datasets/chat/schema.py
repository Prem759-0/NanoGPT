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

# Immutable canonical conversation SHA-256 fingerprints of the 22 pre-Phase-2.7.8
# verified historical baseline records in `datasets/manifests/chat_{train,val,test}.jsonl`.
# Used strictly by the dedicated historical baseline validation path so historical records
# lacking `approval_status` remain preserved without permitting unapproved new records.
IMMUTABLE_BASELINE_SHA256_BY_ID: Dict[str, str] = {
    "moon_chat_gen_001": "799ffc242e07957715caca087d0a3d09742d093f10e4d450add5dcf42433c4c1",
    "moon_chat_gen_002": "77094b299b4a638f4a9c35ab7579f07b3a7b0ae3ee91d6a5bdb37241ddb5916d",
    "moon_chat_gen_003": "43ea793f647eb718d05944e0ab5c841ba756884395785bfc37928e053c3db39a",
    "moon_chat_gen_004": "9eb280ec2e38e6cec3ac95a44a7064e45d8d1077b1acf95c6dd4b2769aa61a2d",
    "moon_chat_gen_005": "0979770f7e3148fec2b779729264d0b03071439f12c0422d1864330b56e21259",
    "moon_chat_gen_006": "976cca80283a39b5ccf70716770530af9800ed4d8a835f1c3f3f92de21d517c6",
    "moon_chat_gen_007": "8b191b0e29371caca44329e9fedddcab1270c184979fcc34fdd45b9b113b45cc",
    "moon_chat_gen_008": "c159444eab897c1abb564c84f141e8790788591207d045508e026717d116685d",
    "moon_chat_gen_009": "774acb8e4477068eef7b76ee6a3ba0feea93c050bccfb736034d2e538febbd05",
    "moon_chat_gen_010": "9cd91e18903f0dc5ad6a0080f11932ec8bbb8b008193ea9e9356201120c4a1d1",
    "moon_chat_gen_011": "f8abdf56e6da52c57abb850441c9b4fb6ed314834b19d62e7efa71d975d484d0",
    "moon_chat_gen_012": "ca65b13b11cf797383fb540368534327ad1f46048ae45b73f2d9c0a6836391ef",
    "moon_chat_gen_013": "c5433ad1220af807abd95931a15aca7bef3f32c401539908b436b5a012523e6c",
    "moon_chat_gen_014": "8544ee149ad4713d0b9d231fe620f516bb65cf3b3ae5d2ab580e74d8145e8f09",
    "moon_chat_gen_015": "a47bb02ba9d215a897c0e5d11ad220aa0498399574334d8d6d957a4ef5f78969",
    "moon_chat_gen_016": "170eee0a426b9ef7841200cf3b223a2f1216169e4f73b31ee2c9e00c9627fa66",
    "moon_chat_gen_017": "e0919b1453d7da0016938d42a4e865163fba5ffc6814e90994b9ca697e9bda90",
    "moon_chat_gen_018": "4310e6f9613a9cdf94b96492cb6901b4b557a14a6d97d55848749cfa9212820f",
    "moon_chat_gen_019": "0a9f7c7568b5bc3e2a01f1733685e3e1a139eea1dbad7682a56b4d68a93236ec",
    "moon_chat_gen_020": "527bc8e0567c19d58f5c35bf0001bd83be5407f215ca1b1cddc946bb21a51f54",
    "moon_chat_gen_021": "a22e6e289a6e0890353d0972e20b552f0bc11a324d176649b1cc1313da987a7b",
    "moon_chat_gen_022": "a07575f9e02c5abbc035f5e302e06071412cade2ccfed954b58dce1afef89508",
}


def is_immutable_baseline_record(record: Any) -> bool:
    """
    Returns True ONLY for the 22 pre-Phase-2.7.8 verified baseline records whose ID,
    canonical conversation SHA-256, and historical provenance match the immutable baseline corpus.
    Rejects tampered approval statuses (e.g., PENDING_REVIEW, REJECTED, None, empty).
    """
    if not isinstance(record, dict):
        return False
    rec_id = str(record.get("id") or "")
    expected_sha = IMMUTABLE_BASELINE_SHA256_BY_ID.get(rec_id)
    if not expected_sha:
        return False
    if record.get("is_dev_fixture") is True or record.get("is_synthetic") is True:
        return False
    prov_type = str(record.get("provenance_type") or "")
    if prov_type not in {PROVENANCE_ORIGINAL_SELF_AUTHORED_QA, PROVENANCE_SOURCE_DERIVED_SUMMARY}:
        return False
    if "approval_status" in record and record.get("approval_status") != APPROVAL_STATUS_APPROVED:
        return False
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    try:
        return compute_conversation_sha256(messages) == expected_sha
    except Exception:
        return False


def validate_chat_record(
    record: Dict[str, Any],
    is_production: bool = True,
    *,
    require_explicit_approval: bool = False,
    _allow_pending_staging: bool = False,
) -> Tuple[bool, Optional[str]]:
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
       - If 'approval_status' key is present (except staging mode), non-'APPROVED' values (including None/'') fail closed.
       - When require_explicit_approval=True, explicit approval_status='APPROVED' is strictly mandatory.
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

    # Approval status check (Phase 2.7.9 hardened fail-closed semantics)
    has_approval_key = "approval_status" in record
    approval_status = record.get("approval_status")
    if is_production and not _allow_pending_staging:
        if require_explicit_approval:
            if not has_approval_key or not isinstance(approval_status, str) or approval_status != APPROVAL_STATUS_APPROVED:
                got = repr(approval_status) if has_approval_key else "<MISSING>"
                return False, (
                    f"Record '{rec_id}' has approval_status={got}; "
                    f"newly curated production records MUST explicitly declare approval_status='{APPROVAL_STATUS_APPROVED}'."
                )
        elif has_approval_key:
            if not isinstance(approval_status, str) or approval_status != APPROVAL_STATUS_APPROVED:
                return False, (
                    f"Record '{rec_id}' has approval_status='{approval_status}'; "
                    "only 'APPROVED' (or verified immutable baseline) records may enter production."
                )
    elif is_production and _allow_pending_staging and has_approval_key:
        if not isinstance(approval_status, str) or approval_status not in {APPROVAL_STATUS_PENDING, APPROVAL_STATUS_APPROVED}:
            return False, (
                f"Record '{rec_id}' has invalid staging approval_status='{approval_status}'; "
                "must be 'PENDING_REVIEW' or 'APPROVED'."
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


def validate_curated_production_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Strict schema validator for newly curated production records.
    Requires explicit `approval_status == "APPROVED"` and fails closed on missing/None/PENDING_REVIEW/REJECTED.
    """
    return validate_chat_record(record, is_production=True, require_explicit_approval=True)


def validate_immutable_baseline_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Dedicated historical compatibility validator strictly limited to the 22 pre-Phase-2.7.8
    verified baseline records (`moon_chat_gen_001`..`moon_chat_gen_022`).
    Rejects non-baseline records lacking `approval_status` as well as tampered approval statuses.
    """
    if not is_immutable_baseline_record(record):
        rec_id = record.get("id") if isinstance(record, dict) else None
        return False, (
            f"Record '{rec_id}' is not an immutable historical verified baseline record and cannot use "
            "the historical baseline compatibility path (explicit approval_status='APPROVED' required)."
        )
    return validate_chat_record(record, is_production=True, require_explicit_approval=False)


def validate_staging_candidate_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates a pre-approval candidate record (`PENDING_REVIEW` or `APPROVED`) without mutating
    or silently assigning `approval_status = "APPROVED"`.
    """
    return validate_chat_record(
        record,
        is_production=True,
        require_explicit_approval=False,
        _allow_pending_staging=True,
    )


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
