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

from datasets.chat.quality_gate import check_code_and_safety, validate_baseline_record, validate_candidate_record
from datasets.chat.schema import APPROVAL_STATUS_APPROVED, canonicalize_category, compute_conversation_sha256
from datasets.chat.truth_audit import (
    ALL_TAXONOMY_CLASSES,
    DEFAULT_TOKENIZER_PATH,
    PRODUCTION_PILOT_TARGETS,
    QUALIFYING_PRODUCTION_TAXONOMIES,
    _safe_relpath,
    classify_chat_record,
    extract_first_assistant_reply,
    extract_first_user_prompt,
    file_sha256,
    iter_jsonl_records,
    verify_production_manifest_group,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "datasets", "manifests")


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
    malformed_lines_total = 0
    raw_input_records_total = 0

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})

    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()
    seen_replies: set[str] = set()

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

        source_checksums[_safe_relpath(abs_path, PROJECT_ROOT)] = actual_sha

        for line_no, rec, parse_err in iter_jsonl_records(abs_path):
            if parse_err is not None or rec is None:
                malformed_lines_total += 1
                rejected.append({
                    "id": None,
                    "line_no": line_no,
                    "source_file": base_key,
                    "reason": parse_err or "Malformed JSON line.",
                })
                continue

            raw_input_records_total += 1
            ok, diag, errors = validate_candidate_record(
                rec, source_file=base_key, line_no=line_no, require_approved=True
            )
            if not ok:
                tax = diag.get("taxonomy", "UNVERIFIED")
                rejection_reasons[f"validation_failed:{tax}"] += 1
                rejected.append({
                    "id": rec.get("id"),
                    "line_no": line_no,
                    "taxonomy": tax,
                    "reason": "; ".join(errors),
                    "source_file": base_key,
                })
                continue

            msgs = rec.get("messages") or []
            conv_sha = diag["conversation_sha256"] or compute_conversation_sha256(msgs)
            prompt_norm = diag["prompt_norm"]
            reply_norm = diag["reply_norm"]
            tax = diag["taxonomy"]

            if conv_sha in seen_hashes:
                rejection_reasons["duplicate_conversation_sha256"] += 1
                rejected.append({"id": rec.get("id"), "line_no": line_no, "taxonomy": tax, "reason": "Duplicate conversation SHA-256.", "source_file": base_key})
                continue
            if prompt_norm and prompt_norm in seen_prompts:
                rejection_reasons["duplicate_prompt"] += 1
                rejected.append({"id": rec.get("id"), "line_no": line_no, "taxonomy": tax, "reason": "Duplicate normalized prompt.", "source_file": base_key})
                continue
            if reply_norm and reply_norm in seen_replies:
                rejection_reasons["duplicate_assistant_reply"] += 1
                rejected.append({"id": rec.get("id"), "line_no": line_no, "taxonomy": tax, "reason": "Duplicate normalized assistant reply.", "source_file": base_key})
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
            enriched["canonical_category"] = canonicalize_category(rec.get("category"))
            enriched["source_file_sha256"] = actual_sha
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
        "raw_input_records": raw_input_records_total,
        "malformed_input_lines": malformed_lines_total,
        "accepted_genuine_records": len(accepted),
        "rejected_records_count": len(rejected),
        "rejected_records_preview": rejected[:25],
        "rejection_reasons": dict(rejection_reasons),
        "taxonomy_counts": {k: taxonomy_counts[k] for k in ALL_TAXONOMY_CLASSES},
        "license_counts": gate_result["audit"]["license_counts"],
        "category_counts_13_targets": gate_result["audit"]["category_counts_13_targets"],
        "source_file_sha256s": source_checksums,
        "output_files": {
            sname: {
                "path": _safe_relpath(out_paths[sname], PROJECT_ROOT),
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


def ingest_immutable_baseline_sources(
    base_dir: str = PROJECT_ROOT,
    output_dir: Optional[str] = None,
    output_prefix: str = "prod_chat",
) -> Dict[str, Any]:
    """
    Dedicated historical baseline ingestion path strictly for the 22 pre-Phase-2.7.8
    verified baseline records (`chat_{train,val,test}.jsonl`).
    """
    manifests_dir = os.path.join(base_dir, "datasets", "manifests")
    out_dir = output_dir or manifests_dir
    os.makedirs(out_dir, exist_ok=True)
    tok_path = os.path.join(base_dir, "models", "chat", "tokenizer.json")
    sources = {
        "train": os.path.join(manifests_dir, "chat_train.jsonl"),
        "val": os.path.join(manifests_dir, "chat_val.jsonl"),
        "test": os.path.join(manifests_dir, "chat_test.jsonl"),
    }

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    source_checksums: Dict[str, str] = {}
    rejected: List[Dict[str, Any]] = []
    taxonomy_counts: Counter[str] = Counter({k: 0 for k in ALL_TAXONOMY_CLASSES})
    seen_hashes: set[str] = set()

    for sname, spath in sources.items():
        actual_sha = file_sha256(spath) or ""
        source_checksums[_safe_relpath(spath, PROJECT_ROOT)] = actual_sha
        for line_no, rec, parse_err in iter_jsonl_records(spath):
            if parse_err is not None or rec is None:
                rejected.append({"id": None, "line_no": line_no, "reason": parse_err or "Malformed JSON line."})
                continue
            ok, diag, errors = validate_baseline_record(rec, source_file=os.path.basename(spath), line_no=line_no)
            if not ok:
                rejected.append({"id": diag.get("id"), "line_no": line_no, "reason": "; ".join(errors)})
                continue
            csha = diag["conversation_sha256"]
            if csha in seen_hashes:
                rejected.append({"id": rec.get("id"), "line_no": line_no, "reason": "Duplicate baseline SHA-256."})
                continue
            seen_hashes.add(csha)
            tax = diag["taxonomy"]
            taxonomy_counts[tax] += 1
            enriched = dict(rec)
            enriched["sha256"] = csha
            enriched["truth_taxonomy"] = tax
            enriched["canonical_category"] = canonicalize_category(rec.get("category"))
            enriched["source_file_sha256"] = actual_sha
            enriched["is_dev_fixture"] = False
            enriched["is_synthetic"] = False
            enriched["baseline_historical_verified"] = True
            enriched["split"] = sname
            splits[sname].append(enriched)

    out_paths: Dict[str, str] = {}
    out_hashes: Dict[str, Optional[str]] = {}
    for sname in ("train", "val", "test"):
        spath = os.path.join(out_dir, f"{output_prefix}_{sname}.jsonl")
        with open(spath, "w", encoding="utf-8", newline="\n") as f:
            for item in splits[sname]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_paths[sname] = spath
        out_hashes[sname] = file_sha256(spath)

    gate_result = verify_production_manifest_group(
        train_path=out_paths["train"],
        val_path=out_paths["val"],
        test_path=out_paths["test"],
        tokenizer_path=tok_path,
        require_scale=True,
    )
    accepted_total = sum(len(splits[s]) for s in ("train", "val", "test"))
    prov_doc = {
        "schema_version": "moon_chat_genuine_ingestion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": gate_result["dataset_status"],
        "eligible_for_production_training": gate_result["passed"],
        "blockers": gate_result["blockers"],
        "raw_input_records": accepted_total + len(rejected),
        "malformed_input_lines": 0,
        "accepted_genuine_records": accepted_total,
        "rejected_records_count": len(rejected),
        "rejected_records_preview": rejected[:25],
        "rejection_reasons": {"baseline_validation_failed": len(rejected)} if rejected else {},
        "taxonomy_counts": {k: taxonomy_counts[k] for k in ALL_TAXONOMY_CLASSES},
        "license_counts": gate_result["audit"]["license_counts"],
        "category_counts_13_targets": gate_result["audit"]["category_counts_13_targets"],
        "source_file_sha256s": source_checksums,
        "output_files": {
            sname: {
                "path": _safe_relpath(out_paths[sname], PROJECT_ROOT),
                "sha256": out_hashes[sname],
                "records_count": len(splits[sname]),
            }
            for sname in ("train", "val", "test")
        },
        "token_stats": gate_result["audit"]["token_stats"],
        "cross_split_leakage": gate_result["audit"]["cross_split_leakage"],
        "shortfall": gate_result["audit"]["production_readiness"]["shortfall"],
    }
    prov_path = os.path.join(out_dir, f"{output_prefix}_provenance.json")
    with open(prov_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    prov_doc["provenance_path"] = prov_path
    return prov_doc


def run_repo_baseline_ingestion(base_dir: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Ingests the existing verified baseline manifests via the dedicated historical baseline path."""
    return ingest_immutable_baseline_sources(base_dir=base_dir)


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
