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
    SUPPORTED_CHAT_CATEGORIES,
    canonicalize_category,
    compute_conversation_sha256,
    normalize_text,
    validate_chat_record,
)
from datasets.chat.tokenizer import SovereignTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "reports", "moon_chat_dataset_truth_audit.json")
DEFAULT_PROGRESS_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "reports", "moon_chat_dataset_progress.json")
DEFAULT_AUDIT_CACHE_PATH = os.path.join(PROJECT_ROOT, "datasets", "sources", ".manifest_audit_cache.json")
AUDIT_CACHE_SCHEMA_VERSION = "phase_2_7_8_file_audit_cache_v1"

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


def iter_jsonl_records(path: str):
    """Streams JSONL records line-by-line yielding (line_no, record_or_none, parse_error_or_none)."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"{os.path.basename(path)} line {line_no}: {exc.msg}"


def load_jsonl_records(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Loads JSONL records and counts malformed JSON lines."""
    records: List[Dict[str, Any]] = []
    malformed = 0
    for _, rec, err in iter_jsonl_records(path):
        if err is not None:
            malformed += 1
        elif rec is not None:
            records.append(rec)
    return records, malformed


def _sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_disk_audit_cache(cache_path: str = DEFAULT_AUDIT_CACHE_PATH) -> Dict[str, Any]:
    if not cache_path or not os.path.isfile(cache_path):
        return {"schema_version": AUDIT_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") == AUDIT_CACHE_SCHEMA_VERSION and isinstance(data.get("entries"), dict):
            return data
    except Exception:
        pass
    return {"schema_version": AUDIT_CACHE_SCHEMA_VERSION, "entries": {}}


def _save_disk_audit_cache(cache_data: Dict[str, Any], cache_path: str = DEFAULT_AUDIT_CACHE_PATH) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _compute_record_token_metrics(
    rec: Dict[str, Any],
    tokenizer: Optional[SovereignTokenizer],
    block_size: int,
    enc_cache: Dict[str, int],
    chat_cache: Dict[str, Tuple[int, int]],
) -> Tuple[int, int, int, int, int]:
    """Computes (seq_len, user_toks, asst_content_toks, asst_sup_toks, windows) streaming per record."""
    if tokenizer is None:
        return 0, 0, 0, 0, 0
    msgs = rec.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return 0, 0, 0, 0, 0
    cache_key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
    cached = chat_cache.get(cache_key)
    if cached is None:
        ids, mask = tokenizer.encode_chat_with_loss_mask(msgs)
        cached = (len(ids), sum(1 for m in mask if m))
        chat_cache[cache_key] = cached
    seq_len, sup_len = cached

    u_toks = 0
    a_toks = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = normalize_text(str(m.get("content", "")))
        val = enc_cache.get(c)
        if val is None:
            val = len(tokenizer.encode_text(c))
            enc_cache[c] = val
        if m.get("role") == "user":
            u_toks += val
        elif m.get("role") == "assistant":
            a_toks += val

    windows = 0
    if seq_len > 1:
        stride = max(1, block_size // 2)
        if seq_len <= block_size + 1:
            windows = 1
        else:
            windows = len(range(0, max(1, seq_len - 1), stride))

    return seq_len, u_toks, a_toks, sup_len, windows


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


def _aggregate_category_counters(counters: Sequence[Dict[str, int]]) -> Dict[str, int]:
    merged: Counter[str] = Counter()
    for c in counters:
        merged.update(c or {})
    return dict(merged)


def _aggregate_13_category_targets(counters: Sequence[Dict[str, int]]) -> Dict[str, int]:
    totals = {k: 0 for k in SUPPORTED_CHAT_CATEGORIES}
    for c in counters:
        for k, v in (c or {}).items():
            totals[canonicalize_category(k)] = totals.get(canonicalize_category(k), 0) + v
    return totals


def _safe_relpath(path: str, base_dir: str) -> str:
    if not os.path.isabs(path):
        return path.replace("\\", "/")
    try:
        return os.path.relpath(path, base_dir).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def audit_single_manifest(
    path: str,
    tokenizer: Optional[SovereignTokenizer] = None,
    block_size: int = 512,
    base_dir: str = PROJECT_ROOT,
    _encode_cache: Optional[Dict[str, int]] = None,
    _chat_cache: Optional[Dict[str, Tuple[int, int]]] = None,
    use_disk_cache: bool = True,
    _disk_cache: Optional[Dict[str, Any]] = None,
    _tokenizer_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Audits a single JSONL manifest file via line-by-line streaming and optional file-SHA caching."""
    rel_path = _safe_relpath(path, base_dir)
    exists = os.path.isfile(path)
    if not exists:
        return {
            "exists": False,
            "path": rel_path,
            "sha256": None,
            "records_count": 0,
            "malformed_lines": 0,
            "taxonomy_counts": {k: 0 for k in ALL_TAXONOMY_CLASSES},
            "category_counts_raw": {},
            "category_counts_13_targets": {k: 0 for k in SUPPORTED_CHAT_CATEGORIES},
            "qualifying_production_genuine_records": 0,
            "token_stats": _compute_token_stats([], tokenizer, block_size),
            "_fingerprints": {
                "conv_shas": {},
                "prompt_counts": {},
                "reply_counts": {},
                "public_source_urls": [],
                "malformed_errors": [],
            },
        }

    f_sha = file_sha256(path) or ""
    tok_sha = _tokenizer_sha256 if _tokenizer_sha256 is not None else (file_sha256(DEFAULT_TOKENIZER_PATH) if tokenizer else "none")
    cache_key = f"{rel_path}::{f_sha}::{tok_sha}::{block_size}"

    local_disk_cache = _disk_cache
    own_disk_cache = False
    if use_disk_cache and local_disk_cache is None:
        local_disk_cache = _load_disk_audit_cache()
        own_disk_cache = True

    if use_disk_cache and local_disk_cache is not None:
        cached_entry = local_disk_cache.get("entries", {}).get(cache_key)
        if isinstance(cached_entry, dict) and cached_entry.get("sha256") == f_sha:
            return cached_entry

    enc_cache = _encode_cache if _encode_cache is not None else {}
    chat_cache = _chat_cache if _chat_cache is not None else {}

    taxonomy_counts: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})
    provenance_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    category_counts_raw: Counter[str] = Counter()
    category_counts_13: Counter[str] = Counter({k: 0 for k in SUPPORTED_CHAT_CATEGORIES})

    sha_counter: Counter[str] = Counter()
    prompt_counter: Counter[str] = Counter()
    reply_counter: Counter[str] = Counter()
    public_source_urls: set[str] = set()
    external_url_count = 0
    local_template_url_count = 0
    valid_attribution_count = 0
    records_count = 0
    malformed = 0
    malformed_errors: List[str] = []

    tot_chatml = 0
    tot_user = 0
    tot_asst_content = 0
    tot_asst_sup = 0
    tot_windows = 0
    lengths: List[int] = []

    for line_no, rec, parse_err in iter_jsonl_records(path):
        if parse_err is not None or rec is None:
            malformed += 1
            if parse_err and len(malformed_errors) < 20:
                malformed_errors.append(parse_err)
            continue

        records_count += 1
        tax, _ = classify_chat_record(rec)
        taxonomy_counts[tax] += 1
        provenance_counts[str(rec.get("provenance_type") or "MISSING")] += 1
        license_counts[str(rec.get("license") or "MISSING")] += 1

        raw_cat = str(rec.get("category") or "uncategorized").strip()
        category_counts_raw[raw_cat] += 1
        canon_cat = canonicalize_category(raw_cat)
        category_counts_13[canon_cat] = category_counts_13.get(canon_cat, 0) + 1

        msgs = rec.get("messages") or []
        conv_sha = str(rec.get("sha256") or "") or compute_conversation_sha256(msgs)
        sha_counter[conv_sha] += 1
        prompt = extract_first_user_prompt(msgs)
        reply = extract_first_assistant_reply(msgs)
        if prompt:
            prompt_counter[prompt.lower()] += 1
        if reply:
            reply_counter[reply.lower()] += 1

        url = str(rec.get("source_url") or "").strip()
        if is_external_http_url(url):
            external_url_count += 1
            if tax == TAXONOMY_PUBLIC_SOURCE_VERIFIED:
                public_source_urls.add(url.lower())
        elif url.startswith("local://"):
            local_template_url_count += 1

        attr = str(rec.get("attribution") or "").strip()
        if len(attr) >= 3:
            valid_attribution_count += 1

        seq_len, u_toks, a_toks, sup_toks, windows = _compute_record_token_metrics(
            rec, tokenizer, block_size, enc_cache, chat_cache
        )
        if seq_len > 0:
            lengths.append(seq_len)
            tot_chatml += seq_len
            tot_user += u_toks
            tot_asst_content += a_toks
            tot_asst_sup += sup_toks
            tot_windows += windows

    repeated_replies = {r: c for r, c in reply_counter.items() if c > 1}
    qualifying = (
        taxonomy_counts[TAXONOMY_HUMAN_AUTHORED_VERIFIED]
        + taxonomy_counts[TAXONOMY_PUBLIC_SOURCE_VERIFIED]
    )

    token_stats = {
        "total_chatml_tokens": tot_chatml,
        "user_content_tokens": tot_user,
        "assistant_content_tokens": tot_asst_content,
        "assistant_supervised_tokens": tot_asst_sup,
        "avg_tokens_per_record": round(tot_chatml / len(lengths), 2) if lengths else 0.0,
        "min_tokens_per_record": min(lengths) if lengths else 0,
        "max_tokens_per_record": max(lengths) if lengths else 0,
        "block_size": block_size,
        "total_windows": tot_windows,
    }

    result = {
        "exists": True,
        "path": rel_path,
        "sha256": f_sha,
        "records_count": records_count,
        "malformed_lines": malformed,
        "unique_sha256_records": len(sha_counter),
        "duplicate_sha256_count": records_count - len(sha_counter),
        "unique_prompts": len(prompt_counter),
        "unique_first_assistant_replies": len(reply_counter),
        "repeated_answer_templates_count": len(repeated_replies),
        "max_answer_template_repetition": max(reply_counter.values()) if reply_counter else 0,
        "taxonomy_counts": {k: taxonomy_counts[k] for k in ALL_TAXONOMY_CLASSES},
        "raw_provenance_counts": dict(provenance_counts),
        "license_counts": dict(license_counts),
        "category_counts_raw": dict(category_counts_raw),
        "category_counts_13_targets": {k: category_counts_13.get(k, 0) for k in SUPPORTED_CHAT_CATEGORIES},
        "external_http_url_count": external_url_count,
        "local_template_url_count": local_template_url_count,
        "valid_attribution_count": valid_attribution_count,
        "qualifying_production_genuine_records": qualifying,
        "token_stats": token_stats,
        "_fingerprints": {
            "conv_shas": dict(sha_counter),
            "prompt_hashes": {_sha256_str(k): v for k, v in prompt_counter.items()},
            "reply_hashes": {_sha256_str(k): v for k, v in reply_counter.items()},
            "reply_previews": {_sha256_str(k): (k[:100] + "..." if len(k) > 100 else k) for k, v in reply_counter.items() if v > 1},
            "public_source_urls": sorted(public_source_urls),
            "malformed_errors": malformed_errors,
        },
    }

    if use_disk_cache and local_disk_cache is not None:
        local_disk_cache.setdefault("entries", {})[cache_key] = result
        if own_disk_cache:
            _save_disk_audit_cache(local_disk_cache)

    return result


def _strip_internal_fingerprints(split_report: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(split_report)
    cleaned.pop("_fingerprints", None)
    return cleaned


def audit_manifest_group(
    name: str,
    split_paths: Dict[str, str],
    tokenizer: Optional[SovereignTokenizer] = None,
    block_size: int = 512,
    base_dir: str = PROJECT_ROOT,
    _encode_cache: Optional[Dict[str, int]] = None,
    _chat_cache: Optional[Dict[str, Tuple[int, int]]] = None,
    use_disk_cache: bool = True,
    _disk_cache: Optional[Dict[str, Any]] = None,
    _tokenizer_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Audits a train/val/test group including cross-split leakage, 13 categories, and global uniqueness."""
    raw_splits_report: Dict[str, Any] = {}
    split_hashes: Dict[str, set[str]] = {}
    split_prompts: Dict[str, set[str]] = {}
    split_replies: Dict[str, set[str]] = {}
    split_pub_urls: Dict[str, set[str]] = {}
    enc_cache = _encode_cache if _encode_cache is not None else {}
    chat_cache = _chat_cache if _chat_cache is not None else {}

    local_disk_cache = _disk_cache
    own_disk_cache = False
    if use_disk_cache and local_disk_cache is None:
        local_disk_cache = _load_disk_audit_cache()
        own_disk_cache = True

    for split_name, path in split_paths.items():
        sr = audit_single_manifest(
            path,
            tokenizer,
            block_size,
            base_dir=base_dir,
            _encode_cache=enc_cache,
            _chat_cache=chat_cache,
            use_disk_cache=use_disk_cache,
            _disk_cache=local_disk_cache,
            _tokenizer_sha256=_tokenizer_sha256,
        )
        raw_splits_report[split_name] = sr
        fps = sr.get("_fingerprints") or {}
        split_hashes[split_name] = set((fps.get("conv_shas") or {}).keys())
        split_prompts[split_name] = set((fps.get("prompt_hashes") or {}).keys())
        split_replies[split_name] = set((fps.get("reply_hashes") or {}).keys())
        split_pub_urls[split_name] = set(fps.get("public_source_urls") or [])

    if use_disk_cache and own_disk_cache and local_disk_cache is not None:
        _save_disk_audit_cache(local_disk_cache)

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    sha_leakage: Dict[str, int] = {}
    prompt_leakage: Dict[str, int] = {}
    reply_leakage: Dict[str, int] = {}
    public_source_url_leakage: Dict[str, int] = {}
    for a, b in pairs:
        key = f"{a}_{b}"
        sha_leakage[key] = len(split_hashes.get(a, set()) & split_hashes.get(b, set()))
        prompt_leakage[key] = len(split_prompts.get(a, set()) & split_prompts.get(b, set()))
        reply_leakage[key] = len(split_replies.get(a, set()) & split_replies.get(b, set()))
        public_source_url_leakage[key] = len(split_pub_urls.get(a, set()) & split_pub_urls.get(b, set()))

    taxonomy_totals: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})
    global_hashes: Counter[str] = Counter()
    global_prompts: Counter[str] = Counter()
    global_replies: Counter[str] = Counter()
    reply_previews: Dict[str, str] = {}
    raw_prov_totals: Counter[str] = Counter()
    license_totals: Counter[str] = Counter()
    cat_raw_totals: Counter[str] = Counter()
    cat_13_totals: Counter[str] = Counter({k: 0 for k in SUPPORTED_CHAT_CATEGORIES})
    ext_url_total = 0
    local_url_total = 0
    valid_attr_total = 0
    total_records = 0

    for sr in raw_splits_report.values():
        total_records += sr.get("records_count", 0)
        for k, v in (sr.get("taxonomy_counts") or {}).items():
            taxonomy_totals[k] += v
        raw_prov_totals.update(sr.get("raw_provenance_counts") or {})
        license_totals.update(sr.get("license_counts") or {})
        cat_raw_totals.update(sr.get("category_counts_raw") or {})
        cat_13_totals.update(sr.get("category_counts_13_targets") or {})
        ext_url_total += sr.get("external_http_url_count", 0)
        local_url_total += sr.get("local_template_url_count", 0)
        valid_attr_total += sr.get("valid_attribution_count", 0)
        fps = sr.get("_fingerprints") or {}
        global_hashes.update(fps.get("conv_shas") or {})
        global_prompts.update(fps.get("prompt_hashes") or {})
        global_replies.update(fps.get("reply_hashes") or {})
        reply_previews.update(fps.get("reply_previews") or {})

    repeated_global_replies = {k: v for k, v in global_replies.items() if v > 1}
    top_repeated = [
        {"count": c, "reply_preview": reply_previews.get(h, f"reply_sha:{h[:16]}")}
        for h, c in sorted(repeated_global_replies.items(), key=lambda x: -x[1])[:5]
    ]

    splits_report = {k: _strip_internal_fingerprints(v) for k, v in raw_splits_report.items()}
    token_totals = _aggregate_split_token_stats(list(splits_report.values()), block_size=block_size)
    qualifying_total = (
        taxonomy_totals[TAXONOMY_HUMAN_AUTHORED_VERIFIED]
        + taxonomy_totals[TAXONOMY_PUBLIC_SOURCE_VERIFIED]
    )
    disallowed_total = sum(taxonomy_totals[k] for k in DISALLOWED_PRODUCTION_TAXONOMIES)

    train_count = splits_report.get("train", {}).get("records_count", 0)
    val_count = splits_report.get("val", {}).get("records_count", 0)
    test_count = splits_report.get("test", {}).get("records_count", 0)
    sup_tokens = token_totals["assistant_supervised_tokens"]
    tot_tokens = token_totals["total_chatml_tokens"]

    leakage_total = (
        sum(sha_leakage.values())
        + sum(prompt_leakage.values())
        + sum(reply_leakage.values())
        + sum(public_source_url_leakage.values())
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
            f"Cross-split leakage detected: sha={sha_leakage}, prompt={prompt_leakage}, "
            f"reply={reply_leakage}, public_source_url={public_source_url_leakage}."
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
        "category_counts_raw": dict(cat_raw_totals),
        "category_counts_13_targets": {k: cat_13_totals.get(k, 0) for k in SUPPORTED_CHAT_CATEGORIES},
        "external_http_url_count": ext_url_total,
        "local_template_url_count": local_url_total,
        "valid_attribution_count": valid_attr_total,
        "qualifying_production_genuine_records": qualifying_total,
        "disallowed_production_records": disallowed_total,
        "cross_split_leakage": {
            "sha256_overlap": sha_leakage,
            "normalized_prompt_overlap": prompt_leakage,
            "normalized_assistant_reply_overlap": reply_leakage,
            "public_source_url_overlap": public_source_url_leakage,
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
    use_disk_cache: bool = True,
) -> Dict[str, Any]:
    """Executes the complete Dataset Truth Audit across all repository chat manifests using safe SHA-keyed file caching."""
    tok_path = tokenizer_path or os.path.join(base_dir, "models", "chat", "tokenizer.json")
    tok_sha = file_sha256(tok_path) if os.path.isfile(tok_path) else None
    tokenizer = SovereignTokenizer.load(tok_path) if os.path.isfile(tok_path) else None
    manifests_dir = os.path.join(base_dir, "datasets", "manifests")

    enc_cache: Dict[str, int] = {}
    chat_cache: Dict[str, Tuple[int, int]] = {}
    cache_path = os.path.join(base_dir, "datasets", "sources", ".manifest_audit_cache.json")
    disk_cache = _load_disk_audit_cache(cache_path) if (use_cache and use_disk_cache) else {"schema_version": AUDIT_CACHE_SCHEMA_VERSION, "entries": {}}

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
        gname: audit_manifest_group(
            gname,
            spaths,
            tokenizer=tokenizer,
            base_dir=base_dir,
            _encode_cache=enc_cache,
            _chat_cache=chat_cache,
            use_disk_cache=(use_cache and use_disk_cache),
            _disk_cache=disk_cache,
            _tokenizer_sha256=tok_sha,
        )
        for gname, spaths in groups.items()
    }

    standalone_candidates = {
        "synthetic_chat_dev": os.path.join(manifests_dir, "synthetic_chat_dev.jsonl"),
        "teacher_pilot_demonstrations": os.path.join(manifests_dir, "teacher_pilot_demonstrations.jsonl"),
        "quarantined_teacher_data": os.path.join(manifests_dir, "quarantined_teacher_data.jsonl"),
    }
    standalone_reports = {
        name: _strip_internal_fingerprints(
            audit_single_manifest(
                path,
                tokenizer=tokenizer,
                base_dir=base_dir,
                _encode_cache=enc_cache,
                _chat_cache=chat_cache,
                use_disk_cache=(use_cache and use_disk_cache),
                _disk_cache=disk_cache,
                _tokenizer_sha256=tok_sha,
            )
        )
        for name, path in standalone_candidates.items()
    }

    if use_disk_cache:
        _save_disk_audit_cache(disk_cache, cache_path)

    genuine_summary = audited_groups["genuine_baseline"]
    prod_summary = audited_groups.get("production_v0_1_candidate", genuine_summary)
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
            "path": _safe_relpath(tok_path, base_dir),
            "sha256": tok_sha,
            "vocab_size": tokenizer.vocab_size if tokenizer else None,
            "merges_count": len(tokenizer.merges) if tokenizer else None,
        },
        "executive_summary": {
            "dataset_status": prod_summary["production_readiness"]["status"],
            "true_qualifying_baseline_records": overall_genuine_records,
            "true_qualifying_taxonomy_breakdown": genuine_summary["taxonomy_counts"],
            "true_qualifying_chatml_tokens": overall_genuine_tot_tokens,
            "true_qualifying_supervised_assistant_tokens": overall_genuine_sup_tokens,
            "production_candidate_qualifying_records": prod_summary["qualifying_production_genuine_records"],
            "production_candidate_supervised_assistant_tokens": prod_summary["token_stats"]["assistant_supervised_tokens"],
            "production_candidate_category_13_distribution": prod_summary.get("category_counts_13_targets", {}),
            "production_candidate_license_distribution": prod_summary.get("license_counts", {}),
            "minimum_genuine_pilot_targets": dict(PRODUCTION_PILOT_TARGETS),
            "exact_shortfall": prod_summary["production_readiness"]["shortfall"],
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
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return report


def generate_progress_report(
    base_dir: str = PROJECT_ROOT,
    output_path: Optional[str] = DEFAULT_PROGRESS_REPORT_PATH,
    pending_queue_path: Optional[str] = None,
    rejected_queue_path: Optional[str] = None,
    source_registry_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Generates machine-readable Phase 2.7.8 progress tracking report across approved, pending, and target thresholds."""
    audit = run_full_truth_audit(base_dir=base_dir, output_path=None, use_cache=use_cache)
    prod = audit["manifest_groups"].get("production_v0_1_candidate") or audit["manifest_groups"]["genuine_baseline"]
    splits = prod.get("splits", {})
    train_n = splits.get("train", {}).get("records_count", 0)
    val_n = splits.get("val", {}).get("records_count", 0)
    test_n = splits.get("test", {}).get("records_count", 0)
    qual = prod["qualifying_production_genuine_records"]
    sup_toks = prod["token_stats"]["assistant_supervised_tokens"]
    tot_toks = prod["token_stats"]["total_chatml_tokens"]

    p_path = pending_queue_path or os.path.join(base_dir, "datasets", "staging", "pending_review_queue.jsonl")
    r_path = rejected_queue_path or os.path.join(base_dir, "datasets", "staging", "rejected_queue.jsonl")
    reg_path = source_registry_path or os.path.join(base_dir, "datasets", "sources", "source_registry.json")

    pending_recs, pending_malformed = load_jsonl_records(p_path)
    rejected_recs, _ = load_jsonl_records(r_path)
    pending_by_cat: Counter[str] = Counter({k: 0 for k in SUPPORTED_CHAT_CATEGORIES})
    pending_by_license: Counter[str] = Counter()
    pending_by_prov: Counter[str] = Counter()
    for pr in pending_recs:
        pending_by_cat[canonicalize_category(pr.get("category"))] += 1
        pending_by_license[str(pr.get("license") or "MISSING")] += 1
        pending_by_prov[str(pr.get("provenance_type") or "MISSING")] += 1

    sources_count = 0
    if os.path.isfile(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as sf:
                reg = json.load(sf)
            sources_count = len(reg.get("sources", {}))
        except Exception:
            sources_count = 0

    progress_doc: Dict[str, Any] = {
        "schema_version": "phase_2_7_8_dataset_progress_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": prod["production_readiness"]["status"],
        "eligible_for_production_training": prod["production_readiness"]["eligible_for_production_training"],
        "targets": dict(PRODUCTION_PILOT_TARGETS),
        "current_approved_production": {
            "total_records": qual,
            "train_records": train_n,
            "val_records": val_n,
            "test_records": test_n,
            "total_chatml_tokens": tot_toks,
            "supervised_assistant_tokens": sup_toks,
            "taxonomy_counts": prod["taxonomy_counts"],
            "license_distribution": prod["license_counts"],
            "category_13_distribution": prod["category_counts_13_targets"],
            "category_raw_distribution": prod["category_counts_raw"],
            "cross_split_leakage": prod["cross_split_leakage"],
        },
        "review_queue": {
            "pending_review_count": len(pending_recs),
            "pending_malformed_lines": pending_malformed,
            "rejected_count": len(rejected_recs),
            "registered_sources_count": sources_count,
            "pending_category_13_distribution": dict(pending_by_cat),
            "pending_license_distribution": dict(pending_by_license),
            "pending_provenance_distribution": dict(pending_by_prov),
        },
        "shortfall": prod["production_readiness"]["shortfall"],
        "completion_pct": {
            "records_pct": round(min(100.0, (qual / PRODUCTION_PILOT_TARGETS["min_total_records"]) * 100.0), 2),
            "supervised_tokens_pct": round(min(100.0, (sup_toks / PRODUCTION_PILOT_TARGETS["min_supervised_tokens"]) * 100.0), 2),
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(progress_doc, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return progress_doc


def main() -> None:
    report = run_full_truth_audit(output_path=DEFAULT_REPORT_PATH)
    generate_progress_report()
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
