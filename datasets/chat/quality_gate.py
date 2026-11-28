"""
datasets/chat/quality_gate.py
Phase 2.7.8 Quality Control, License, Structure, Security & Tokenizer Validation Gate.

Provides unified record-level validation used across source acquisition, owner
contribution staging (`PENDING_REVIEW`), owner approval promotion, and production
manifest generation.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.chat.schema import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    APPROVED_LICENSES,
    PROHIBITED_LICENSES,
    SUPPORTED_CHAT_CATEGORIES,
    canonicalize_category,
    compute_conversation_sha256,
    normalize_text,
    validate_chat_record,
    validate_conversation_structure_and_safety,
    validate_curated_production_record,
    validate_immutable_baseline_record,
    validate_staging_candidate_record,
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    QUALIFYING_PRODUCTION_TAXONOMIES,
    TAXONOMY_HUMAN_AUTHORED_VERIFIED,
    TAXONOMY_PUBLIC_SOURCE_VERIFIED,
    classify_chat_record,
    extract_first_assistant_reply,
    extract_first_user_prompt,
)

PYTHON_FENCE_RE = re.compile(r"```(?:python|py)[^\n\r]*(?:\r?\n)(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```json[^\n\r]*(?:\r?\n)(.*?)```", re.DOTALL | re.IGNORECASE)

SECRET_AND_PII_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI/API secret key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub classic token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("Private key PEM block", re.compile(r"-----BEGIN\s+(?:RSA|EC|OPENSSH|PGP|PRIVATE)\s+PRIVATE\s+KEY-----")),
    ("Bearer authorization header", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{24,}=*\b")),
    ("Hardcoded password literal", re.compile(r"(?i)\b(?:password|passwd|secret_key|api_key)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']")),
    ("US SSN pattern", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
SAFE_ILLUSTRATIVE_EMAIL_DOMAINS = frozenset({"example.com", "example.org", "example.net"})

PLACEHOLDER_TEXT_MARKERS = (
    "[todo",
    "todo:",
    "fill in your",
    "insert answer here",
    "placeholder answer",
    "lorem ipsum",
)

MIN_USER_CONTENT_CHARS = 5
MIN_ASSISTANT_CONTENT_CHARS = 20


def check_license_permissibility(license_name: Any, source_url: Any = "") -> Tuple[bool, Optional[str]]:
    """Strict license check distinguishing approved permissive licenses from blocked/unknown licenses."""
    lic = str(license_name or "").strip()
    url = str(source_url or "").strip().lower()

    if not lic:
        return False, "Missing 'license' field."
    upper = lic.upper()
    if upper in PROHIBITED_LICENSES or "NC" in upper.split("-") or "ND" in upper.split("-"):
        return False, f"License '{lic}' is explicitly prohibited (non-commercial, no-derivatives, or unverified)."
    if lic not in APPROVED_LICENSES:
        return False, f"License '{lic}' is not in APPROVED_LICENSES {sorted(APPROVED_LICENSES)}."
    if "wikipedia.org" in url and lic in {"CC0-1.0", "Public Domain", "PublicDomain", "MIT"}:
        return False, f"Wikipedia sources require CC-BY-SA-4.0/3.0, never '{lic}'."
    return True, None


def check_code_and_safety(messages: Sequence[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Validates role ordering, UTF-8 cleanliness, secret/PII absence, balanced code fences,
    Python AST syntax, and JSON fenced block validity.
    """
    struct_ok, struct_err = validate_conversation_structure_and_safety(messages, require_strict_alternation=True)
    if not struct_ok:
        return False, struct_err

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        raw_content = str(msg.get("content", ""))
        norm_len = len(normalize_text(raw_content))

        if role == "user" and norm_len < MIN_USER_CONTENT_CHARS:
            return False, f"Turn[{idx}] user prompt too short (< {MIN_USER_CONTENT_CHARS} normalized chars)."
        if role == "assistant":
            if norm_len < MIN_ASSISTANT_CONTENT_CHARS:
                return False, f"Turn[{idx}] assistant answer too short (< {MIN_ASSISTANT_CONTENT_CHARS} normalized chars)."
            low = raw_content.lower()
            for marker in PLACEHOLDER_TEXT_MARKERS:
                if marker in low:
                    return False, f"Turn[{idx}] contains unresolved placeholder marker '{marker}'."

        for rule_name, pat in SECRET_AND_PII_RULES:
            if pat.search(raw_content):
                return False, f"Potential secret or PII detected in turn[{idx}]: {rule_name} matched."

        for em in EMAIL_RE.finditer(raw_content):
            domain = em.group(1).lower()
            if domain not in SAFE_ILLUSTRATIVE_EMAIL_DOMAINS:
                return False, f"Personal/external email PII detected in turn[{idx}] ('@{domain}')."

        fence_count = raw_content.count("```")
        if fence_count % 2 != 0:
            return False, f"Unbalanced markdown code fence ('```') in turn[{idx}]."

        for code_match in PYTHON_FENCE_RE.finditer(raw_content):
            code_body = code_match.group(1).strip()
            if not code_body:
                continue
            try:
                ast.parse(code_body)
            except SyntaxError as exc:
                return False, f"Invalid Python code block syntax in turn[{idx}] (line {exc.lineno}): {exc.msg}"

        for json_match in JSON_FENCE_RE.finditer(raw_content):
            json_body = json_match.group(1).strip()
            if not json_body:
                continue
            try:
                json.loads(json_body)
            except json.JSONDecodeError as exc:
                return False, f"Invalid JSON syntax in turn[{idx}] code block: {exc.msg}"

    return True, None


def check_tokenizer_compatibility(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Optional[SovereignTokenizer],
    block_size: int = 512,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Checks tokenization round-trip stability, assistant-only loss masking, and window count."""
    if tokenizer is None:
        return True, {"checked": False}, None

    ids, mask = tokenizer.encode_chat_with_loss_mask(list(messages))
    if len(ids) != len(mask) or len(ids) < 2:
        return False, {}, "Tokenization produced invalid sequence length or mask mismatch."
    sup_tokens = sum(1 for m in mask if m)
    if sup_tokens <= 0:
        return False, {}, "Conversation yielded 0 supervised assistant tokens."

    decoded = tokenizer.decode(ids)
    # Verify decoded ChatML preserves end-of-turn markers and content cleanly
    if "<|im_end|>" not in decoded:
        return False, {}, "Tokenizer round-trip lost <|im_end|> delimiter."

    stride = max(1, block_size)
    windows = 1 if len(ids) <= block_size + 1 else len(range(0, max(1, len(ids) - 1), stride))
    stats = {
        "checked": True,
        "total_tokens": len(ids),
        "supervised_tokens": sup_tokens,
        "windows_at_block_512": windows,
        "exceeds_single_window": len(ids) > block_size + 1,
    }
    return True, stats, None


def check_draft_summary_originality(rec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Ensures source-derived conversion drafts are not completed by copying raw reference extracts
    verbatim or still flagged as needing original human educational synthesis.
    """
    if not isinstance(rec, dict):
        return False, "Record must be a JSON dictionary."
    if rec.get("needs_original_human_summary") is True:
        return False, (
            "Conversion draft is still marked 'needs_original_human_summary=True'; "
            "complete original educational synthesis and mark it reviewed before approval."
        )
    ref_extract = normalize_text(str(rec.get("source_reference_extract") or ""))
    msgs = rec.get("messages") or []
    asst_reply = extract_first_assistant_reply(msgs) if isinstance(msgs, list) else ""
    if ref_extract and len(ref_extract) >= 40 and asst_reply:
        if asst_reply == ref_extract or (len(asst_reply) <= len(ref_extract) + 30 and ref_extract[:80] in asst_reply):
            return False, (
                "Assistant answer copies 'source_reference_extract' nearly verbatim; "
                "write original educational explanation in your own words."
            )
    return True, None


def validate_candidate_record(
    rec: Dict[str, Any],
    *,
    source_file: str = "<memory>",
    line_no: int = 0,
    require_approved: bool = True,
    tokenizer: Optional[SovereignTokenizer] = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Comprehensive Phase 2.7.9 validation gate returning (is_valid, diagnostics_dict, error_list).

    - Production/curated mode (`require_approved=True`, default): fails closed unless the record
      explicitly carries `approval_status == "APPROVED"`. Missing, None, empty, `PENDING_REVIEW`,
      or `REJECTED` statuses are strictly rejected. Never mutates or defaults `approval_status`.
    - Staging mode (`require_approved=False`): validates pre-approval candidate schema/safety/taxonomy
      without ever rewriting `PENDING_REVIEW` to `APPROVED`.
    """
    errors: List[str] = []
    rec_id = str(rec.get("id") or f"line_{line_no}") if isinstance(rec, dict) else f"line_{line_no}"
    src_url = str(rec.get("source_url") or "") if isinstance(rec, dict) else ""

    def _fmt(field: str, msg: str) -> str:
        return f"[{source_file}:{line_no} id={rec_id} field={field} source={src_url}] {msg}"

    if not isinstance(rec, dict):
        return False, {"id": rec_id, "source_file": source_file, "line_no": line_no}, [_fmt("record", "Expected JSON object.")]

    lic_ok, lic_err = check_license_permissibility(rec.get("license"), src_url)
    if not lic_ok:
        errors.append(_fmt("license", lic_err or "Invalid license."))

    msgs = rec.get("messages") or []
    safe_ok, safe_err = check_code_and_safety(msgs)
    if not safe_ok:
        errors.append(_fmt("messages", safe_err or "Conversation quality/safety validation failed."))

    has_approval_key = "approval_status" in rec
    approval_status = rec.get("approval_status")
    if require_approved:
        if not has_approval_key or not isinstance(approval_status, str) or approval_status != APPROVAL_STATUS_APPROVED:
            got = repr(approval_status) if has_approval_key else "<MISSING>"
            errors.append(
                _fmt(
                    "approval_status",
                    f"Production requires explicit approval_status='{APPROVAL_STATUS_APPROVED}', got {got}.",
                )
            )
        schema_ok, schema_err = validate_curated_production_record(rec)
    else:
        if has_approval_key and (
            not isinstance(approval_status, str)
            or approval_status not in {APPROVAL_STATUS_PENDING, APPROVAL_STATUS_APPROVED}
        ):
            errors.append(
                _fmt(
                    "approval_status",
                    f"Staging rejects approval_status='{approval_status}' (must be PENDING_REVIEW or APPROVED).",
                )
            )
        schema_ok, schema_err = validate_staging_candidate_record(rec)

    if not schema_ok:
        errors.append(_fmt("schema", schema_err or "Failed schema validation."))

    tax, tax_reasons = classify_chat_record(rec, allow_pending_staging=(not require_approved))
    if tax not in QUALIFYING_PRODUCTION_TAXONOMIES:
        errors.append(_fmt("provenance_type", f"Disallowed truth taxonomy '{tax}': {'; '.join(tax_reasons)}"))

    if tax == TAXONOMY_PUBLIC_SOURCE_VERIFIED and not str(rec.get("transformation_performed") or rec.get("modification_summary") or "").strip():
        errors.append(_fmt("transformation_performed", "Source-derived records must document what transformation/editing was performed."))

    if require_approved:
        orig_ok, orig_err = check_draft_summary_originality(rec)
        if not orig_ok:
            errors.append(_fmt("draft_completeness", orig_err or "Incomplete source conversion draft."))

    tok_stats: Dict[str, Any] = {}
    if safe_ok and tokenizer is not None:
        tok_ok, tok_stats, tok_err = check_tokenizer_compatibility(msgs, tokenizer)
        if not tok_ok:
            errors.append(_fmt("tokenizer", tok_err or "Tokenizer incompatibility."))

    raw_cat = str(rec.get("category") or "").strip()
    canon_cat = canonicalize_category(raw_cat)
    conv_sha = compute_conversation_sha256(msgs) if isinstance(msgs, list) and msgs else ""
    prompt_norm = extract_first_user_prompt(msgs).lower() if isinstance(msgs, list) else ""
    reply_norm = extract_first_assistant_reply(msgs).lower() if isinstance(msgs, list) else ""

    diagnostics = {
        "id": rec_id,
        "source_file": source_file,
        "line_no": line_no,
        "source_url": src_url,
        "taxonomy": tax,
        "raw_category": raw_cat,
        "canonical_category": canon_cat,
        "canonical_category_supported": canon_cat in SUPPORTED_CHAT_CATEGORIES,
        "conversation_sha256": conv_sha,
        "prompt_norm": prompt_norm,
        "reply_norm": reply_norm,
        "token_stats": tok_stats,
    }
    return len(errors) == 0, diagnostics, errors


def validate_baseline_record(
    rec: Dict[str, Any],
    *,
    source_file: str = "<baseline>",
    line_no: int = 0,
    tokenizer: Optional[SovereignTokenizer] = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Dedicated, fail-closed validation path strictly for the 22 pre-Phase-2.7.8 immutable
    verified baseline records (`moon_chat_gen_001`..`moon_chat_gen_022`).
    Never permits newly curated records missing `approval_status`.
    """
    errors: List[str] = []
    rec_id = str(rec.get("id") or f"line_{line_no}") if isinstance(rec, dict) else f"line_{line_no}"
    src_url = str(rec.get("source_url") or "") if isinstance(rec, dict) else ""

    def _fmt(field: str, msg: str) -> str:
        return f"[{source_file}:{line_no} id={rec_id} field={field} source={src_url}] {msg}"

    if not isinstance(rec, dict):
        return False, {"id": rec_id, "source_file": source_file, "line_no": line_no}, [_fmt("record", "Expected JSON object.")]

    base_ok, base_err = validate_immutable_baseline_record(rec)
    if not base_ok:
        return False, {"id": rec_id, "source_file": source_file, "line_no": line_no}, [_fmt("baseline", base_err or "Not a verified immutable baseline record.")]

    lic_ok, lic_err = check_license_permissibility(rec.get("license"), src_url)
    if not lic_ok:
        errors.append(_fmt("license", lic_err or "Invalid license."))

    msgs = rec.get("messages") or []
    safe_ok, safe_err = check_code_and_safety(msgs)
    if not safe_ok:
        errors.append(_fmt("messages", safe_err or "Conversation quality/safety validation failed."))

    tax, tax_reasons = classify_chat_record(rec, allow_pending_staging=False)
    if tax not in QUALIFYING_PRODUCTION_TAXONOMIES:
        errors.append(_fmt("provenance_type", f"Disallowed baseline taxonomy '{tax}': {'; '.join(tax_reasons)}"))

    tok_stats: Dict[str, Any] = {}
    if safe_ok and tokenizer is not None:
        tok_ok, tok_stats, tok_err = check_tokenizer_compatibility(msgs, tokenizer)
        if not tok_ok:
            errors.append(_fmt("tokenizer", tok_err or "Tokenizer incompatibility."))

    raw_cat = str(rec.get("category") or "").strip()
    canon_cat = canonicalize_category(raw_cat)
    conv_sha = compute_conversation_sha256(msgs) if isinstance(msgs, list) and msgs else ""
    prompt_norm = extract_first_user_prompt(msgs).lower() if isinstance(msgs, list) else ""
    reply_norm = extract_first_assistant_reply(msgs).lower() if isinstance(msgs, list) else ""

    diagnostics = {
        "id": rec_id,
        "source_file": source_file,
        "line_no": line_no,
        "source_url": src_url,
        "taxonomy": tax,
        "raw_category": raw_cat,
        "canonical_category": canon_cat,
        "canonical_category_supported": canon_cat in SUPPORTED_CHAT_CATEGORIES,
        "conversation_sha256": conv_sha,
        "prompt_norm": prompt_norm,
        "reply_norm": reply_norm,
        "token_stats": tok_stats,
    }
    return len(errors) == 0, diagnostics, errors
