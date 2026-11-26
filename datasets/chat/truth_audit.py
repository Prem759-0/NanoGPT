"""
datasets/chat/truth_audit.py
Phase 2.7.7 Dataset Truth Classification & Contamination Audit Engine for Moon AI.

Classifies every chat record into an explicit truth taxonomy:
  - HUMAN_AUTHORED_VERIFIED
  - PUBLIC_SOURCE_VERIFIED
  - DETERMINISTIC_TEMPLATE
  - SYNTHETIC_FIXTURE
  - AI_GENERATED
  - UNVERIFIED

Provides rigorous single-manifest and train/val/test group audits covering:
  - Unique vs duplicate SHA-256 records
  - Prompt and assistant-answer uniqueness & template repetition
  - Cross-split leakage across hashes, normalized user prompts, and normalized assistant replies
  - Exact ChatML token counts, user tokens, and supervised assistant tokens (`loss_mask=True`)
  - Honest `DATASET_READY` vs `DATASET_NOT_READY` production readiness evaluation
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.chat.schema import (
    APPROVED_LICENSES,
    compute_conversation_sha256,
    normalize_text,
    validate_chat_record,
)
from datasets.chat.tokenizer import SovereignTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "reports", "moon_chat_dataset_truth_audit.json")

# Explicit Phase 2.7.7 Taxonomy
TAXONOMY_HUMAN_AUTHORED_VERIFIED = "HUMAN_AUTHORED_VERIFIED"
TAXONOMY_PUBLIC_SOURCE_VERIFIED = "PUBLIC_SOURCE_VERIFIED"
TAXONOMY_DETERMINISTIC_TEMPLATE = "DETERMINISTIC_TEMPLATE"
TAXONOMY_SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
TAXONOMY_AI_GENERATED = "AI_GENERATED"
TAXONOMY_UNVERIFIED = "UNVERIFIED"

ALL_TAXONOMY_CLASSES: Tuple[str, ...] = (
    TAXONOMY_HUMAN_AUTHORED_VERIFIED,
    TAXONOMY_PUBLIC_SOURCE_VERIFIED,
    TAXONOMY_DETERMINISTIC_TEMPLATE,
    TAXONOMY_SYNTHETIC_FIXTURE,
    TAXONOMY_AI_GENERATED,
    TAXONOMY_UNVERIFIED,
)

QUALIFYING_PRODUCTION_TAXONOMIES = frozenset({
    TAXONOMY_HUMAN_AUTHORED_VERIFIED,
    TAXONOMY_PUBLIC_SOURCE_VERIFIED,
})

DISALLOWED_PRODUCTION_TAXONOMIES = frozenset({
    TAXONOMY_DETERMINISTIC_TEMPLATE,
    TAXONOMY_SYNTHETIC_FIXTURE,
    TAXONOMY_AI_GENERATED,
    TAXONOMY_UNVERIFIED,
})

PRODUCTION_PILOT_TARGETS: Dict[str, int] = {
    "min_total_records": 500,
    "min_train_records": 400,
    "min_val_records": 50,
    "min_test_records": 50,
    "min_supervised_tokens": 100000,
    "min_total_tokens": 100000,
}

TEMPLATE_PROMPT_PREFIX_RE = re.compile(r"^Variant\s+\d+\s*:\s*", re.IGNORECASE)
TEMPLATE_ANSWER_PREFIX_RE = re.compile(r"^Overview of\s+.+?:\s*\n\n", re.IGNORECASE | re.DOTALL)
TEMPLATE_KEY_TAKEAWAY_MARKERS = (
    "Key Takeaway: Understanding these principles provides foundational insight",
    "Key Takeaway:",
    "In practice, structuring implementations around",
)
VALID_PUBLIC_LICENSES = set(APPROVED_LICENSES) | {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "PSF-2.0",
    "PublicDomain",
}
VALID_HUMAN_LICENSES = VALID_PUBLIC_LICENSES | {"CustomSovereign"}


def file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_first_user_prompt(messages: Sequence[Dict[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return normalize_text(str(m.get("content", "")))
    return ""


def extract_first_assistant_reply(messages: Sequence[Dict[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return normalize_text(str(m.get("content", "")))
    return ""


def is_external_http_url(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    u = url.strip().lower()
    if not (u.startswith("https://") or u.startswith("http://")):
        return False
    if "example.com" in u or "localhost" in u or "127.0.0.1" in u:
        return False
    return len(u) > 10


def _looks_like_deterministic_template(rec: Dict[str, Any]) -> bool:
    prov = str(rec.get("provenance_type") or "").strip().upper()
    src = str(rec.get("source") or "").lower()
    src_url = str(rec.get("source_url") or "").lower()
    mod_sum = str(rec.get("modification_summary") or "").lower()
    gen_meta = rec.get("generation_metadata") or {}
    gen_method = str(gen_meta.get("generation_method") or "").upper() if isinstance(gen_meta, dict) else ""

    if "DETERMINISTIC" in prov or "TEMPLATE" in prov:
        return True
    if src_url.startswith("local://") or "curriculum_generator" in src_url or "template" in src:
        return True
    if gen_method == "DETERMINISTIC_PYTHON_TEMPLATE" or "template expansion" in mod_sum:
        return True

    messages = rec.get("messages") or []
    prompt = extract_first_user_prompt(messages)
    reply = extract_first_assistant_reply(messages)
    if TEMPLATE_PROMPT_PREFIX_RE.search(prompt):
        return True
    if TEMPLATE_ANSWER_PREFIX_RE.search(reply):
        for marker in TEMPLATE_KEY_TAKEAWAY_MARKERS:
            if marker in reply:
                return True
    return False


def classify_chat_record(rec: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Classifies a single chat JSONL record into the Phase 2.7.7 taxonomy.
    Returns (taxonomy_class, reasons).
    Fail-closed: uncertain or missing metadata defaults to UNVERIFIED (or SYN/AI/TEMPLATE).
    """
    if not isinstance(rec, dict):
        return TAXONOMY_UNVERIFIED, ["Record is not a JSON object."]

    reasons: List[str] = []
    prov = str(rec.get("provenance_type") or "").strip().upper()
    is_dev_fixture = bool(rec.get("is_dev_fixture", False))
    is_synthetic = bool(rec.get("is_synthetic", False))

    # 1. Synthetic fixtures first
    if is_dev_fixture or is_synthetic or "SYNTHETIC" in prov or prov == "TEST_FIXTURE":
        if is_dev_fixture:
            reasons.append("is_dev_fixture=True")
        if is_synthetic:
            reasons.append("is_synthetic=True")
        if "SYNTHETIC" in prov:
            reasons.append(f"provenance_type={prov}")
        return TAXONOMY_SYNTHETIC_FIXTURE, reasons

    # 2. AI / Teacher-generated data
    teacher_id = rec.get("teacher_model_id") or rec.get("teacher_provider") or rec.get("teacher_checkpoint_hash")
    gen_meta = rec.get("generation_metadata") if isinstance(rec.get("generation_metadata"), dict) else {}
    teacher_meta = gen_meta.get("teacher_model")
    if (
        prov == "AI_GENERATED"
        or "TEACHER" in prov
        or bool(teacher_id)
        or bool(teacher_meta)
        or str(rec.get("source") or "").strip().lower().startswith("mock ")
    ):
        reasons.append(f"AI/Teacher metadata present (provenance_type={prov or 'MISSING'})")
        return TAXONOMY_AI_GENERATED, reasons

    # 3. Deterministic programmatic templates
    if _looks_like_deterministic_template(rec):
        reasons.append(f"Deterministic template indicators detected (provenance_type={prov or 'MISSING'}, source_url={rec.get('source_url')})")
        return TAXONOMY_DETERMINISTIC_TEMPLATE, reasons

    # 4. Schema validation for production-grade human or public-source records
    schema_ok, schema_err = validate_chat_record(rec, is_production=True)
    if not schema_ok:
        return TAXONOMY_UNVERIFIED, [f"Failed production schema validation: {schema_err}"]

    attribution = str(rec.get("attribution") or "").strip()
    license_name = str(rec.get("license") or "").strip()
    source_url = str(rec.get("source_url") or "").strip()
    if len(attribution) < 3:
        return TAXONOMY_UNVERIFIED, ["Missing or trivial attribution."]

    # 5. Human-authored verified
    if prov in ("ORIGINAL_SELF_AUTHORED_QA", "HUMAN_AUTHORED_VERIFIED"):
        if license_name not in VALID_HUMAN_LICENSES:
            return TAXONOMY_UNVERIFIED, [f"Unsupported license '{license_name}' for human-authored record."]
        if not is_external_http_url(source_url) and not source_url.startswith("repo://"):
            return TAXONOMY_UNVERIFIED, [f"Missing verifiable source_url '{source_url}'."]
        return TAXONOMY_HUMAN_AUTHORED_VERIFIED, ["Verified self-authored human Q&A with attribution and source reference."]

    # 6. Public-source verified
    if prov in ("SOURCE_DERIVED_EDUCATIONAL_SUMMARY", "PUBLIC_SOURCE_CURATED", "PUBLIC_SOURCE_VERIFIED"):
        if license_name not in VALID_PUBLIC_LICENSES:
            return TAXONOMY_UNVERIFIED, [f"Unsupported public license '{license_name}'."]
        if not is_external_http_url(source_url):
            return TAXONOMY_UNVERIFIED, [f"Public-source record requires external http(s) source_url, got '{source_url}'."]
        if not str(rec.get("license_url") or "").strip():
            return TAXONOMY_UNVERIFIED, ["Missing license_url for public-source record."]
        return TAXONOMY_PUBLIC_SOURCE_VERIFIED, ["Verified external public source with permissible license and attribution."]

    return TAXONOMY_UNVERIFIED, [f"Unclassified or unverified provenance_type '{prov}'."]


def load_jsonl_records(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Loads JSONL records and counts malformed JSON lines."""
    records: List[Dict[str, Any]] = []
    malformed = 0
    if not os.path.isfile(path):
        return records, malformed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                malformed += 1
    return records, malformed


def _compute_token_stats(
    records: Sequence[Dict[str, Any]],
    tokenizer: Optional[SovereignTokenizer],
    block_size: int = 512,
    _encode_cache: Optional[Dict[str, int]] = None,
    _chat_cache: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    if tokenizer is None or not records:
        return {
            "total_chatml_tokens": 0,
            "user_content_tokens": 0,
            "assistant_content_tokens": 0,
            "assistant_supervised_tokens": 0,
            "avg_tokens_per_record": 0.0,
            "min_tokens_per_record": 0,
            "max_tokens_per_record": 0,
            "block_size": block_size,
            "total_windows": 0,
        }

    enc_cache = _encode_cache if _encode_cache is not None else {}
    chat_cache = _chat_cache if _chat_cache is not None else {}

    def _cached_text_len(text: str) -> int:
        val = enc_cache.get(text)
        if val is None:
            val = len(tokenizer.encode_text(text))
            enc_cache[text] = val
        return val

    total_chatml = 0
    total_user = 0
    total_asst_content = 0
    total_asst_supervised = 0
    total_windows = 0
    lengths: List[int] = []

    for rec in records:
        msgs = rec.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            continue
        cache_key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        cached = chat_cache.get(cache_key)
        if cached is None:
            ids, mask = tokenizer.encode_chat_with_loss_mask(msgs)
            cached = (len(ids), sum(1 for m in mask if m))
            chat_cache[cache_key] = cached
        seq_len, sup_len = cached

        lengths.append(seq_len)
        total_chatml += seq_len
        total_asst_supervised += sup_len
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = normalize_text(str(m.get("content", "")))
            c_toks = _cached_text_len(c)
            if m.get("role") == "user":
                total_user += c_toks
            elif m.get("role") == "assistant":
                total_asst_content += c_toks

        if seq_len > 1:
            stride = max(1, block_size // 2)
            if seq_len <= block_size + 1:
                total_windows += 1
            else:
                total_windows += len(range(0, max(1, seq_len - 1), stride))

    avg_len = round(total_chatml / len(lengths), 2) if lengths else 0.0
    return {
        "total_chatml_tokens": total_chatml,
        "user_content_tokens": total_user,
        "assistant_content_tokens": total_asst_content,
        "assistant_supervised_tokens": total_asst_supervised,
        "avg_tokens_per_record": avg_len,
        "min_tokens_per_record": min(lengths) if lengths else 0,
        "max_tokens_per_record": max(lengths) if lengths else 0,
        "block_size": block_size,
        "total_windows": total_windows,
    }


def _aggregate_split_token_stats(split_reports: Sequence[Dict[str, Any]], block_size: int = 512) -> Dict[str, Any]:
    total_recs = sum(sr.get("records_count", 0) for sr in split_reports)
    stats_list = [sr.get("token_stats") or {} for sr in split_reports if sr.get("records_count", 0) > 0]
    if not stats_list or total_recs == 0:
        return {
            "total_chatml_tokens": 0,
            "user_content_tokens": 0,
            "assistant_content_tokens": 0,
            "assistant_supervised_tokens": 0,
            "avg_tokens_per_record": 0.0,
            "min_tokens_per_record": 0,
            "max_tokens_per_record": 0,
            "block_size": block_size,
            "total_windows": 0,
        }
    tot_chatml = sum(s["total_chatml_tokens"] for s in stats_list)
    return {
        "total_chatml_tokens": tot_chatml,
        "user_content_tokens": sum(s["user_content_tokens"] for s in stats_list),
        "assistant_content_tokens": sum(s["assistant_content_tokens"] for s in stats_list),
        "assistant_supervised_tokens": sum(s["assistant_supervised_tokens"] for s in stats_list),
        "avg_tokens_per_record": round(tot_chatml / total_recs, 2),
        "min_tokens_per_record": min(s["min_tokens_per_record"] for s in stats_list),
        "max_tokens_per_record": max(s["max_tokens_per_record"] for s in stats_list),
        "block_size": block_size,
        "total_windows": sum(s["total_windows"] for s in stats_list),
    }


def audit_single_manifest(
    path: str,
    tokenizer: Optional[SovereignTokenizer] = None,
    block_size: int = 512,
    base_dir: str = PROJECT_ROOT,
    _encode_cache: Optional[Dict[str, int]] = None,
    _chat_cache: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Audits a single JSONL manifest file in detail."""
    rel_path = os.path.relpath(path, base_dir).replace("\\", "/") if os.path.isabs(path) else path.replace("\\", "/")
    exists = os.path.isfile(path)
    if not exists:
        return {
            "exists": False,
            "path": rel_path,
            "sha256": None,
            "records_count": 0,
            "malformed_lines": 0,
            "taxonomy_counts": {k: 0 for k in ALL_TAXONOMY_CLASSES},
            "qualifying_production_genuine_records": 0,
            "token_stats": _compute_token_stats([], tokenizer, block_size),
        }

    records, malformed = load_jsonl_records(path)
    taxonomy_counts: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})
    provenance_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()

    sha_counter: Counter[str] = Counter()
    prompt_counter: Counter[str] = Counter()
    reply_counter: Counter[str] = Counter()
    external_url_count = 0
    local_template_url_count = 0
    valid_attribution_count = 0

    for rec in records:
        tax, _ = classify_chat_record(rec)
        taxonomy_counts[tax] += 1
        provenance_counts[str(rec.get("provenance_type") or "MISSING")] += 1
        license_counts[str(rec.get("license") or "MISSING")] += 1

        msgs = rec.get("messages") or []
        conv_sha = str(rec.get("sha256") or "") or compute_conversation_sha256(msgs)
        sha_counter[conv_sha] += 1
        prompt = extract_first_user_prompt(msgs)
        reply = extract_first_assistant_reply(msgs)
        if prompt:
            prompt_counter[prompt.lower()] += 1
        if reply:
            reply_counter[reply.lower()] += 1

        url = str(rec.get("source_url") or "")
        if is_external_http_url(url):
            external_url_count += 1
        elif url.startswith("local://"):
            local_template_url_count += 1

        attr = str(rec.get("attribution") or "").strip()
        if len(attr) >= 3:
            valid_attribution_count += 1

    repeated_replies = {r: c for r, c in reply_counter.items() if c > 1}
    qualifying = (
        taxonomy_counts[TAXONOMY_HUMAN_AUTHORED_VERIFIED]
        + taxonomy_counts[TAXONOMY_PUBLIC_SOURCE_VERIFIED]
    )

    return {
        "exists": True,
        "path": rel_path,
        "sha256": file_sha256(path),
        "records_count": len(records),
        "malformed_lines": malformed,
        "unique_sha256_records": len(sha_counter),
        "duplicate_sha256_count": len(records) - len(sha_counter),
        "unique_prompts": len(prompt_counter),
        "unique_first_assistant_replies": len(reply_counter),
        "repeated_answer_templates_count": len(repeated_replies),
        "max_answer_template_repetition": max(reply_counter.values()) if reply_counter else 0,
        "taxonomy_counts": {k: taxonomy_counts[k] for k in ALL_TAXONOMY_CLASSES},
        "raw_provenance_counts": dict(provenance_counts),
        "license_counts": dict(license_counts),
        "external_http_url_count": external_url_count,
        "local_template_url_count": local_template_url_count,
        "valid_attribution_count": valid_attribution_count,
        "qualifying_production_genuine_records": qualifying,
        "token_stats": _compute_token_stats(
            records, tokenizer, block_size, _encode_cache=_encode_cache, _chat_cache=_chat_cache
        ),
    }


def audit_manifest_group(
    name: str,
    split_paths: Dict[str, str],
    tokenizer: Optional[SovereignTokenizer] = None,
    block_size: int = 512,
    base_dir: str = PROJECT_ROOT,
    _encode_cache: Optional[Dict[str, int]] = None,
    _chat_cache: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Audits a train/val/test group including cross-split leakage and global uniqueness."""
    splits_report: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []
    split_hashes: Dict[str, set[str]] = {}
    split_prompts: Dict[str, set[str]] = {}
    split_replies: Dict[str, set[str]] = {}
    enc_cache = _encode_cache if _encode_cache is not None else {}
    chat_cache = _chat_cache if _chat_cache is not None else {}

    for split_name, path in split_paths.items():
        splits_report[split_name] = audit_single_manifest(
            path, tokenizer, block_size, base_dir=base_dir, _encode_cache=enc_cache, _chat_cache=chat_cache
        )
        recs, _ = load_jsonl_records(path)
        all_records.extend(recs)
        hashes = set()
        prompts = set()
        replies = set()
        for r in recs:
            msgs = r.get("messages") or []
            hashes.add(compute_conversation_sha256(msgs) if isinstance(msgs, list) and msgs else str(r.get("sha256") or ""))
            p = extract_first_user_prompt(msgs)
            a = extract_first_assistant_reply(msgs)
            if p:
                prompts.add(p.lower())
            if a:
                replies.add(a.lower())
        split_hashes[split_name] = hashes
        split_prompts[split_name] = prompts
        split_replies[split_name] = replies

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    sha_leakage: Dict[str, int] = {}
    prompt_leakage: Dict[str, int] = {}
    reply_leakage: Dict[str, int] = {}
    for a, b in pairs:
        key = f"{a}_{b}"
        sha_leakage[key] = len(split_hashes.get(a, set()) & split_hashes.get(b, set()))
        prompt_leakage[key] = len(split_prompts.get(a, set()) & split_prompts.get(b, set()))
        reply_leakage[key] = len(split_replies.get(a, set()) & split_replies.get(b, set()))

    taxonomy_totals: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})
    global_hashes: Counter[str] = Counter()
    global_prompts: Counter[str] = Counter()
    global_replies: Counter[str] = Counter()
    raw_prov_totals: Counter[str] = Counter()
    license_totals: Counter[str] = Counter()
    ext_url_total = 0
    local_url_total = 0
    valid_attr_total = 0

    for rec in all_records:
        tax, _ = classify_chat_record(rec)
        taxonomy_totals[tax] += 1
        raw_prov_totals[str(rec.get("provenance_type") or "MISSING")] += 1
        license_totals[str(rec.get("license") or "MISSING")] += 1
        msgs = rec.get("messages") or []
        conv_sha = compute_conversation_sha256(msgs) if isinstance(msgs, list) and msgs else str(rec.get("sha256") or "")
        global_hashes[conv_sha] += 1
        p = extract_first_user_prompt(msgs)
        a = extract_first_assistant_reply(msgs)
        if p:
            global_prompts[p.lower()] += 1
        if a:
            global_replies[a.lower()] += 1
        url = str(rec.get("source_url") or "")
        if is_external_http_url(url):
            ext_url_total += 1
        elif url.startswith("local://"):
            local_url_total += 1
        if len(str(rec.get("attribution") or "").strip()) >= 3:
            valid_attr_total += 1

    repeated_global_replies = {k: v for k, v in global_replies.items() if v > 1}
    top_repeated = [
        {"count": c, "reply_preview": (txt[:100] + "...") if len(txt) > 100 else txt}
        for txt, c in sorted(repeated_global_replies.items(), key=lambda x: -x[1])[:5]
    ]

    token_totals = _aggregate_split_token_stats(list(splits_report.values()), block_size=block_size)
    qualifying_total = (
        taxonomy_totals[TAXONOMY_HUMAN_AUTHORED_VERIFIED]
        + taxonomy_totals[TAXONOMY_PUBLIC_SOURCE_VERIFIED]
    )
    disallowed_total = sum(taxonomy_totals[k] for k in DISALLOWED_PRODUCTION_TAXONOMIES)

    total_records = len(all_records)
    train_count = splits_report.get("train", {}).get("records_count", 0)
    val_count = splits_report.get("val", {}).get("records_count", 0)
    test_count = splits_report.get("test", {}).get("records_count", 0)
    sup_tokens = token_totals["assistant_supervised_tokens"]
    tot_tokens = token_totals["total_chatml_tokens"]

    leakage_total = (
        sum(sha_leakage.values())
        + sum(prompt_leakage.values())
        + sum(reply_leakage.values())
    )

    blockers: List[str] = []
    if total_records == 0:
        blockers.append("Manifest group contains 0 records.")
    if disallowed_total > 0:
        breakdown = {k: taxonomy_totals[k] for k in DISALLOWED_PRODUCTION_TAXONOMIES if taxonomy_totals[k] > 0}
        blockers.append(f"Contains {disallowed_total} disallowed non-production records: {breakdown}.")
    if qualifying_total < PRODUCTION_PILOT_TARGETS["min_total_records"]:
        blockers.append(
            f"Qualifying genuine records {qualifying_total} < required minimum {PRODUCTION_PILOT_TARGETS['min_total_records']}."
        )
    if train_count < PRODUCTION_PILOT_TARGETS["min_train_records"]:
        blockers.append(
            f"Train records {train_count} < required minimum {PRODUCTION_PILOT_TARGETS['min_train_records']}."
        )
    if val_count < PRODUCTION_PILOT_TARGETS["min_val_records"]:
        blockers.append(
            f"Val records {val_count} < required minimum {PRODUCTION_PILOT_TARGETS['min_val_records']}."
        )
    if test_count < PRODUCTION_PILOT_TARGETS["min_test_records"]:
        blockers.append(
            f"Test records {test_count} < required minimum {PRODUCTION_PILOT_TARGETS['min_test_records']}."
        )
    if sup_tokens < PRODUCTION_PILOT_TARGETS["min_supervised_tokens"]:
        blockers.append(
            f"Supervised assistant tokens {sup_tokens} < required minimum {PRODUCTION_PILOT_TARGETS['min_supervised_tokens']}."
        )
    if leakage_total > 0:
        blockers.append(
            f"Cross-split leakage detected: sha={sha_leakage}, prompt={prompt_leakage}, reply={reply_leakage}."
        )

    is_production_ready = len(blockers) == 0
    return {
        "group_name": name,
        "total_records": total_records,
        "unique_sha256_records": len(global_hashes),
        "duplicate_sha256_count": total_records - len(global_hashes),
        "unique_prompts": len(global_prompts),
        "unique_first_assistant_replies": len(global_replies),
        "repeated_answer_templates_count": len(repeated_global_replies),
        "max_answer_template_repetition": max(global_replies.values()) if global_replies else 0,
        "top_repeated_answer_samples": top_repeated,
        "taxonomy_counts": {k: taxonomy_totals[k] for k in ALL_TAXONOMY_CLASSES},
        "raw_provenance_counts": dict(raw_prov_totals),
        "license_counts": dict(license_totals),
        "external_http_url_count": ext_url_total,
        "local_template_url_count": local_url_total,
        "valid_attribution_count": valid_attr_total,
        "qualifying_production_genuine_records": qualifying_total,
        "disallowed_production_records": disallowed_total,
        "cross_split_leakage": {
            "sha256_overlap": sha_leakage,
            "normalized_prompt_overlap": prompt_leakage,
            "normalized_assistant_reply_overlap": reply_leakage,
            "total_leakage_events": leakage_total,
        },
        "token_stats": token_totals,
        "splits": splits_report,
        "production_readiness": {
            "status": "DATASET_READY" if is_production_ready else "DATASET_NOT_READY",
            "eligible_for_production_training": is_production_ready,
            "targets": dict(PRODUCTION_PILOT_TARGETS),
            "shortfall": {
                "qualifying_records_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_total_records"] - qualifying_total),
                "train_records_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_train_records"] - train_count),
                "val_records_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_val_records"] - val_count),
                "test_records_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_test_records"] - test_count),
                "supervised_tokens_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_supervised_tokens"] - sup_tokens),
                "total_chatml_tokens_shortfall": max(0, PRODUCTION_PILOT_TARGETS["min_total_tokens"] - tot_tokens),
            },
            "blockers": blockers,
        },
    }


def verify_production_manifest_group(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    require_scale: bool = True,
    min_total_records: int = PRODUCTION_PILOT_TARGETS["min_total_records"],
    min_train_records: int = PRODUCTION_PILOT_TARGETS["min_train_records"],
    min_val_records: int = PRODUCTION_PILOT_TARGETS["min_val_records"],
    min_test_records: int = PRODUCTION_PILOT_TARGETS["min_test_records"],
    min_supervised_tokens: int = PRODUCTION_PILOT_TARGETS["min_supervised_tokens"],
) -> Dict[str, Any]:
    """
    Fail-closed production dataset gate for Moon Chat v0.1+.
    Rejects DETERMINISTIC_TEMPLATE, SYNTHETIC_FIXTURE, AI_GENERATED, UNVERIFIED,
    cross-split prompt/hash/assistant-reply leakage, or insufficient genuine scale.
    """
    tok = SovereignTokenizer.load(tokenizer_path) if os.path.isfile(tokenizer_path) else None
    split_paths = {"train": train_path, "val": val_path}
    if test_path:
        split_paths["test"] = test_path

    report = audit_manifest_group("production_candidate", split_paths, tokenizer=tok)
    blockers: List[str] = []

    for sname, spath in split_paths.items():
        if not os.path.isfile(spath):
            blockers.append(f"Missing required {sname} manifest: {spath}")
        elif report["splits"][sname].get("malformed_lines", 0) > 0:
            blockers.append(f"Malformed JSON lines in {sname} manifest.")

    if report["disallowed_production_records"] > 0:
        disallowed = {
            k: v for k, v in report["taxonomy_counts"].items()
            if k in DISALLOWED_PRODUCTION_TAXONOMIES and v > 0
        }
        blockers.append(f"Disallowed non-genuine records present: {disallowed}")

    if report["cross_split_leakage"]["total_leakage_events"] > 0:
        blockers.append(f"Cross-split leakage present: {report['cross_split_leakage']}")

    if report["repeated_answer_templates_count"] > 0:
        blockers.append(
            f"Repeated boilerplate assistant replies detected ({report['repeated_answer_templates_count']} groups, "
            f"max repetition={report['max_answer_template_repetition']})."
        )

    if require_scale:
        qual = report["qualifying_production_genuine_records"]
        tr = report["splits"].get("train", {}).get("records_count", 0)
        va = report["splits"].get("val", {}).get("records_count", 0)
        te = report["splits"].get("test", {}).get("records_count", 0) if test_path else min_test_records
        sup = report["token_stats"]["assistant_supervised_tokens"]

        if qual < min_total_records:
            blockers.append(f"Insufficient genuine records: {qual} < {min_total_records}")
        if tr < min_train_records:
            blockers.append(f"Insufficient genuine train records: {tr} < {min_train_records}")
        if va < min_val_records:
            blockers.append(f"Insufficient genuine val records: {va} < {min_val_records}")
        if test_path and te < min_test_records:
            blockers.append(f"Insufficient genuine test records: {te} < {min_test_records}")
        if sup < min_supervised_tokens:
            blockers.append(f"Insufficient supervised assistant tokens: {sup} < {min_supervised_tokens}")

    passed = len(blockers) == 0
    return {
        "passed": passed,
        "dataset_status": "DATASET_READY" if passed else "DATASET_NOT_READY",
        "blockers": blockers,
        "audit": report,
    }


def run_full_truth_audit(
    base_dir: str = PROJECT_ROOT,
    tokenizer_path: Optional[str] = None,
    output_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Executes or loads the complete Phase 2.7.7 Dataset Truth Audit across all repository chat manifests."""
    out_target = output_path or DEFAULT_REPORT_PATH
    if use_cache and os.path.isfile(out_target):
        try:
            with open(out_target, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data.get("schema_version") == "phase_2_7_7_dataset_truth_audit_v1" and "manifest_groups" in cached_data:
                return cached_data
        except Exception:
            pass
    tok_path = tokenizer_path or os.path.join(base_dir, "models", "chat", "tokenizer.json")
    tokenizer = SovereignTokenizer.load(tok_path) if os.path.isfile(tok_path) else None
    manifests_dir = os.path.join(base_dir, "datasets", "manifests")

    groups = {
        "genuine_baseline": {
            "train": os.path.join(manifests_dir, "chat_train.jsonl"),
            "val": os.path.join(manifests_dir, "chat_val.jsonl"),
            "test": os.path.join(manifests_dir, "chat_test.jsonl"),
        },
        "expanded_pilot": {
            "train": os.path.join(manifests_dir, "expanded_chat_train.jsonl"),
            "val": os.path.join(manifests_dir, "expanded_chat_val.jsonl"),
            "test": os.path.join(manifests_dir, "expanded_chat_test.jsonl"),
        },
        "scaled_pilot": {
            "train": os.path.join(manifests_dir, "scaled_chat_train.jsonl"),
            "val": os.path.join(manifests_dir, "scaled_chat_val.jsonl"),
            "test": os.path.join(manifests_dir, "scaled_chat_test.jsonl"),
        },
    }

    prod_paths = {
        "train": os.path.join(manifests_dir, "prod_chat_train.jsonl"),
        "val": os.path.join(manifests_dir, "prod_chat_val.jsonl"),
        "test": os.path.join(manifests_dir, "prod_chat_test.jsonl"),
    }
    if any(os.path.isfile(p) for p in prod_paths.values()):
        groups["production_v0_1_candidate"] = prod_paths

    audited_groups = {
        gname: audit_manifest_group(gname, spaths, tokenizer=tokenizer, base_dir=base_dir)
        for gname, spaths in groups.items()
    }

    standalone_candidates = {
        "synthetic_chat_dev": os.path.join(manifests_dir, "synthetic_chat_dev.jsonl"),
        "teacher_pilot_demonstrations": os.path.join(manifests_dir, "teacher_pilot_demonstrations.jsonl"),
        "quarantined_teacher_data": os.path.join(manifests_dir, "quarantined_teacher_data.jsonl"),
    }
    standalone_reports = {
        name: audit_single_manifest(path, tokenizer=tokenizer, base_dir=base_dir)
        for name, path in standalone_candidates.items()
    }

    genuine_summary = audited_groups["genuine_baseline"]
    scaled_summary = audited_groups["scaled_pilot"]
    expanded_summary = audited_groups["expanded_pilot"]

    overall_genuine_records = genuine_summary["qualifying_production_genuine_records"]
    overall_genuine_sup_tokens = genuine_summary["token_stats"]["assistant_supervised_tokens"]
    overall_genuine_tot_tokens = genuine_summary["token_stats"]["total_chatml_tokens"]

    report: Dict[str, Any] = {
        "schema_version": "phase_2_7_7_dataset_truth_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy": list(ALL_TAXONOMY_CLASSES),
        "tokenizer": {
            "path": os.path.relpath(tok_path, base_dir).replace("\\", "/"),
            "sha256": file_sha256(tok_path),
            "vocab_size": tokenizer.vocab_size if tokenizer else None,
            "merges_count": len(tokenizer.merges) if tokenizer else None,
        },
        "executive_summary": {
            "dataset_status": "DATASET_NOT_READY"
            if overall_genuine_records < PRODUCTION_PILOT_TARGETS["min_total_records"]
            else "DATASET_READY",
            "true_qualifying_baseline_records": overall_genuine_records,
            "true_qualifying_taxonomy_breakdown": genuine_summary["taxonomy_counts"],
            "true_qualifying_chatml_tokens": overall_genuine_tot_tokens,
            "true_qualifying_supervised_assistant_tokens": overall_genuine_sup_tokens,
            "minimum_genuine_pilot_targets": dict(PRODUCTION_PILOT_TARGETS),
            "exact_shortfall": genuine_summary["production_readiness"]["shortfall"],
            "scaled_chat_total_records": scaled_summary["total_records"],
            "scaled_chat_deterministic_template_records": scaled_summary["taxonomy_counts"][
                TAXONOMY_DETERMINISTIC_TEMPLATE
            ],
            "scaled_chat_unique_assistant_replies": scaled_summary["unique_first_assistant_replies"],
            "scaled_chat_cross_split_reply_leakage": scaled_summary["cross_split_leakage"][
                "normalized_assistant_reply_overlap"
            ],
            "scaled_chat_eligible_for_production": False,
            "expanded_chat_total_records": expanded_summary["total_records"],
            "expanded_chat_deterministic_template_records": expanded_summary["taxonomy_counts"][
                TAXONOMY_DETERMINISTIC_TEMPLATE
            ],
            "expanded_chat_eligible_for_production": False,
        },
        "manifest_groups": audited_groups,
        "standalone_quarantined_and_dev_manifests": standalone_reports,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return report


def main() -> None:
    report = run_full_truth_audit(output_path=DEFAULT_REPORT_PATH)
    summary = report["executive_summary"]
    print(f"[TruthAudit] dataset_status={summary['dataset_status']}")
    print(
        f"[TruthAudit] genuine_baseline={summary['true_qualifying_baseline_records']} records, "
        f"supervised_tokens={summary['true_qualifying_supervised_assistant_tokens']}, "
        f"shortfall={summary['exact_shortfall']}"
    )
    print(
        f"[TruthAudit] scaled_pilot={summary['scaled_chat_total_records']} records "
        f"(DETERMINISTIC_TEMPLATE={summary['scaled_chat_deterministic_template_records']}, "
        f"unique_replies={summary['scaled_chat_unique_assistant_replies']}, "
        f"reply_leakage={summary['scaled_chat_cross_split_reply_leakage']})"
    )


if __name__ == "__main__":
    main()
