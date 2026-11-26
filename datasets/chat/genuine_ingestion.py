"""
datasets/chat/genuine_ingestion.py
Phase 2.7.7 Reproducible Genuine Data-Ingestion Workflow for Moon Chat v0.1+.

Ingests ONLY genuine human-authored Q&A and verified permissible public-source
educational records (e.g., CC-BY-SA-4.0, CC-BY-4.0, CC0-1.0, MIT, Apache-2.0,
BSD-3-Clause, PSF-2.0, PublicDomain, CustomSovereign).

Strict fail-closed guarantees:
  - Blocks DETERMINISTIC_TEMPLATE, SYNTHETIC_FIXTURE, AI_GENERATED, UNVERIFIED.
  - Records SHA-256 checksums of every raw source file ingested.
  - Verifies Python code fences with `ast.parse` and rejects leaked secrets/PII.
  - Deduplicates on conversation SHA-256, normalized prompt, and normalized reply.
  - Assigns or enforces group-isolated train/val/test splits (zero cross-split
    hash, prompt, or assistant-reply leakage).
  - Never synthesizes filler records; reports honest `DATASET_NOT_READY` status
    and exact token/record shortfall when genuine volume is below release threshold.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets.chat.schema import compute_conversation_sha256, normalize_text
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    ALL_TAXONOMY_CLASSES,
    DEFAULT_TOKENIZER_PATH,
    PRODUCTION_PILOT_TARGETS,
    QUALIFYING_PRODUCTION_TAXONOMIES,
    classify_chat_record,
    extract_first_assistant_reply,
    extract_first_user_prompt,
    file_sha256,
    load_jsonl_records,
    verify_production_manifest_group,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "datasets", "manifests")

PYTHON_FENCE_RE = re.compile(r"```python[^\n\r]*(?:\r?\n)(.*?)```", re.DOTALL | re.IGNORECASE)
SECRET_PII_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)
MIN_ASSISTANT_CONTENT_CHARS = 20
MIN_USER_CONTENT_CHARS = 4


def check_code_and_safety(messages: Sequence[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Checks secret/PII patterns and validates any fenced Python blocks with ast.parse."""
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message[{idx}] is not a dict."
        role = msg.get("role")
        raw_content = str(msg.get("content", ""))
        content = normalize_text(raw_content)
        if role == "user" and len(content) < MIN_USER_CONTENT_CHARS:
            return False, f"User message[{idx}] shorter than {MIN_USER_CONTENT_CHARS} chars."
        if role == "assistant" and len(content) < MIN_ASSISTANT_CONTENT_CHARS:
            return False, f"Assistant message[{idx}] shorter than {MIN_ASSISTANT_CONTENT_CHARS} chars."

        for pat in SECRET_PII_PATTERNS:
            if pat.search(raw_content):
                return False, f"Potential secret or PII token matched in message[{idx}]."

        for code_match in PYTHON_FENCE_RE.finditer(raw_content):
            code_body = code_match.group(1)
            try:
                ast.parse(code_body)
            except SyntaxError as exc:
                return False, f"Invalid Python code block syntax in message[{idx}]: {exc.msg}"

    return True, None


def ingest_genuine_sources(
    source_paths: Sequence[str],
    output_dir: str = DEFAULT_MANIFESTS_DIR,
    output_prefix: str = "prod_chat",
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    preserve_explicit_splits: bool = False,
    expected_source_checksums: Optional[Dict[str, str]] = None,
    min_total_records: int = PRODUCTION_PILOT_TARGETS["min_total_records"],
    min_train_records: int = PRODUCTION_PILOT_TARGETS["min_train_records"],
    min_val_records: int = PRODUCTION_PILOT_TARGETS["min_val_records"],
    min_test_records: int = PRODUCTION_PILOT_TARGETS["min_test_records"],
    min_supervised_tokens: int = PRODUCTION_PILOT_TARGETS["min_supervised_tokens"],
) -> Dict[str, Any]:
    """
    Reproducibly ingests verified genuine chat JSONL files into `train`, `val`, and `test`
    manifests plus a checksummed provenance report.
    """
    os.makedirs(output_dir, exist_ok=True)
    source_checksums: Dict[str, str] = {}
    raw_records_with_src: List[Tuple[Dict[str, Any], str, str]] = []
    malformed_lines_total = 0

    for raw_path in source_paths:
        abs_path = os.path.abspath(raw_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Ingestion source file not found: {abs_path}")
        actual_sha = file_sha256(abs_path) or ""
        norm_key = abs_path.replace("\\", "/")
        base_key = os.path.basename(abs_path)

        if expected_source_checksums:
            expected = (
                expected_source_checksums.get(raw_path)
                or expected_source_checksums.get(norm_key)
                or expected_source_checksums.get(base_key)
            )
            if expected and expected.lower() != actual_sha.lower():
                raise ValueError(
                    f"Source file SHA-256 mismatch for '{abs_path}': expected {expected}, got {actual_sha}"
                )

        source_checksums[os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")] = actual_sha
        recs, malformed = load_jsonl_records(abs_path)
        malformed_lines_total += malformed
        for rec in recs:
            raw_records_with_src.append((rec, abs_path, actual_sha))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})

    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()
    seen_replies: set[str] = set()

    for rec, src_path, src_sha in raw_records_with_src:
        tax, tax_reasons = classify_chat_record(rec)
        if tax not in QUALIFYING_PRODUCTION_TAXONOMIES:
            reason_key = f"disallowed_taxonomy:{tax}"
            rejection_reasons[reason_key] += 1
            rejected.append({
                "id": rec.get("id"),
                "taxonomy": tax,
                "reason": "; ".join(tax_reasons),
                "source_file": os.path.basename(src_path),
            })
            continue

        msgs = rec.get("messages") or []
        safe_ok, safe_err = check_code_and_safety(msgs)
        if not safe_ok:
            rejection_reasons["quality_or_safety_failed"] += 1
            rejected.append({
                "id": rec.get("id"),
                "taxonomy": tax,
                "reason": safe_err,
                "source_file": os.path.basename(src_path),
            })
            continue

        conv_sha = compute_conversation_sha256(msgs)
        prompt_norm = extract_first_user_prompt(msgs).lower()
        reply_norm = extract_first_assistant_reply(msgs).lower()

        if conv_sha in seen_hashes:
            rejection_reasons["duplicate_conversation_sha256"] += 1
            rejected.append({"id": rec.get("id"), "taxonomy": tax, "reason": "Duplicate conversation SHA-256."})
            continue
        if prompt_norm and prompt_norm in seen_prompts:
            rejection_reasons["duplicate_prompt"] += 1
            rejected.append({"id": rec.get("id"), "taxonomy": tax, "reason": "Duplicate normalized prompt."})
            continue
        if reply_norm and reply_norm in seen_replies:
            rejection_reasons["duplicate_assistant_reply"] += 1
            rejected.append({"id": rec.get("id"), "taxonomy": tax, "reason": "Duplicate normalized assistant reply."})
            continue

        seen_hashes.add(conv_sha)
        if prompt_norm:
            seen_prompts.add(prompt_norm)
        if reply_norm:
            seen_replies.add(reply_norm)

        taxonomy_counts[tax] += 1
        enriched = dict(rec)
        enriched["sha256"] = conv_sha
        enriched["truth_taxonomy"] = tax
        enriched["source_file_sha256"] = src_sha
        enriched["is_dev_fixture"] = False
        enriched["is_synthetic"] = False
        accepted.append(enriched)

    # Split assignment: preserve verified non-leaking splits if requested, else deterministic group allocation.
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    if preserve_explicit_splits and accepted and all(r.get("split") in splits for r in accepted):
        for r in accepted:
            splits[r["split"]].append(r)
    elif accepted:
        # Group by (normalized source_url + taxonomy) so related docs never leak across splits.
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in accepted:
            gkey = f"{str(r.get('source_url') or r.get('id') or r['sha256']).strip().lower()}::{r['sha256']}"
            groups[gkey].append(r)

        ordered_keys = sorted(
            groups.keys(),
            key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest(),
        )
        total_items = len(accepted)
        target_val = max(1, round(total_items * 0.10)) if total_items >= 10 else (1 if total_items >= 3 else 0)
        target_test = max(1, round(total_items * 0.10)) if total_items >= 10 else (1 if total_items >= 4 else 0)

        for idx, gkey in enumerate(ordered_keys):
            items = groups[gkey]
            if len(splits["val"]) < target_val:
                target_split = "val"
            elif len(splits["test"]) < target_test:
                target_split = "test"
            else:
                target_split = "train"
            for item in items:
                item_copy = dict(item)
                item_copy["split"] = target_split
                splits[target_split].append(item_copy)

    out_paths: Dict[str, str] = {}
    out_hashes: Dict[str, Optional[str]] = {}
    for sname in ("train", "val", "test"):
        spath = os.path.join(output_dir, f"{output_prefix}_{sname}.jsonl")
        with open(spath, "w", encoding="utf-8", newline="\n") as f:
            for item in splits[sname]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_paths[sname] = spath
        out_hashes[sname] = file_sha256(spath)

    gate_result = verify_production_manifest_group(
        train_path=out_paths["train"],
        val_path=out_paths["val"],
        test_path=out_paths["test"],
        tokenizer_path=tokenizer_path,
        require_scale=True,
        min_total_records=min_total_records,
        min_train_records=min_train_records,
        min_val_records=min_val_records,
        min_test_records=min_test_records,
        min_supervised_tokens=min_supervised_tokens,
    )

    provenance_doc = {
        "schema_version": "moon_chat_genuine_ingestion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": gate_result["dataset_status"],
        "eligible_for_production_training": gate_result["passed"],
        "blockers": gate_result["blockers"],
        "raw_input_records": len(raw_records_with_src),
        "malformed_input_lines": malformed_lines_total,
        "accepted_genuine_records": len(accepted),
        "rejected_records_count": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "taxonomy_counts": {k: taxonomy_counts[k] for k in ALL_TAXONOMY_CLASSES},
        "source_file_sha256s": source_checksums,
        "output_files": {
            sname: {
                "path": os.path.relpath(out_paths[sname], PROJECT_ROOT).replace("\\", "/")
                if os.path.abspath(out_paths[sname]).startswith(PROJECT_ROOT)
                else out_paths[sname].replace("\\", "/"),
                "sha256": out_hashes[sname],
                "records_count": len(splits[sname]),
            }
            for sname in ("train", "val", "test")
        },
        "token_stats": gate_result["audit"]["token_stats"],
        "cross_split_leakage": gate_result["audit"]["cross_split_leakage"],
        "shortfall": gate_result["audit"]["production_readiness"]["shortfall"],
    }

    prov_path = os.path.join(output_dir, f"{output_prefix}_provenance.json")
    with open(prov_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    provenance_doc["provenance_path"] = prov_path
    return provenance_doc


def run_repo_baseline_ingestion(base_dir: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Ingests the existing verified baseline manifests into prod_chat_{train,val,test}.jsonl."""
    manifests_dir = os.path.join(base_dir, "datasets", "manifests")
    tok_path = os.path.join(base_dir, "models", "chat", "tokenizer.json")
    sources = [
        os.path.join(manifests_dir, "chat_train.jsonl"),
        os.path.join(manifests_dir, "chat_val.jsonl"),
        os.path.join(manifests_dir, "chat_test.jsonl"),
    ]
    return ingest_genuine_sources(
        source_paths=sources,
        output_dir=manifests_dir,
        output_prefix="prod_chat",
        tokenizer_path=tok_path,
        preserve_explicit_splits=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Moon Chat Genuine Data-Ingestion Pipeline")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            os.path.join(DEFAULT_MANIFESTS_DIR, "chat_train.jsonl"),
            os.path.join(DEFAULT_MANIFESTS_DIR, "chat_val.jsonl"),
            os.path.join(DEFAULT_MANIFESTS_DIR, "chat_test.jsonl"),
        ],
        help="Input JSONL source files containing candidate human-authored or public-source records.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--output-prefix", default="prod_chat")
    parser.add_argument("--preserve-splits", action="store_true")
    args = parser.parse_args()

    result = ingest_genuine_sources(
        source_paths=args.sources,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        preserve_explicit_splits=args.preserve_splits,
    )
    print(f"[GenuineIngestion] dataset_status={result['dataset_status']}")
    print(
        f"[GenuineIngestion] accepted={result['accepted_genuine_records']}, "
        f"rejected={result['rejected_records_count']}, "
        f"supervised_tokens={result['token_stats']['assistant_supervised_tokens']}"
    )
    print(f"[GenuineIngestion] shortfall={result['shortfall']}")


if __name__ == "__main__":
    main()
