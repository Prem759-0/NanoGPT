"""
datasets/chat/review_queue.py
Phase 2.7.8 Owner Review, Approval/Rejection & Reproducible Manifest Regeneration.

Enforces strict human/owner verification barrier between staged candidates (`PENDING_REVIEW`)
and production manifests (`prod_chat_train/val/test.jsonl`).

No candidate record can enter `prod_chat_*` without explicit verification and promotion
through `approve_pending_records(...)`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.chat.acquisition import DEFAULT_REGISTRY_PATH, load_source_registry, save_source_registry
from datasets.chat.contribute import DEFAULT_PENDING_QUEUE_PATH
from datasets.chat.quality_gate import (
    check_draft_summary_originality,
    validate_baseline_record,
    validate_candidate_record,
)
from datasets.chat.schema import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    compute_conversation_sha256,
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    ALL_TAXONOMY_CLASSES,
    DEFAULT_PROGRESS_REPORT_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_TOKENIZER_PATH,
    TAXONOMY_PUBLIC_SOURCE_VERIFIED,
    _safe_relpath,
    extract_first_assistant_reply,
    extract_first_user_prompt,
    file_sha256,
    generate_progress_report,
    iter_jsonl_records,
    run_full_truth_audit,
    verify_production_manifest_group,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "datasets", "manifests")
DEFAULT_CURATED_APPROVED_PATH = os.path.join(DEFAULT_MANIFESTS_DIR, "curated_approved_records.jsonl")
DEFAULT_REJECTED_QUEUE_PATH = os.path.join(PROJECT_ROOT, "datasets", "staging", "rejected_queue.jsonl")


def _load_records_clean(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return recs
    for _, rec, err in iter_jsonl_records(path):
        if err is None and isinstance(rec, dict):
            recs.append(rec)
    return recs


def _write_jsonl_atomic(path: str, records: Sequence[Dict[str, Any]]) -> Optional[str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    return file_sha256(path)


def list_pending_records(queue_path: str = DEFAULT_PENDING_QUEUE_PATH) -> List[Dict[str, Any]]:
    return _load_records_clean(queue_path)


def rebuild_production_manifests(
    *,
    base_dir: str = PROJECT_ROOT,
    manifests_dir: Optional[str] = None,
    curated_approved_path: Optional[str] = None,
    pending_queue_path: Optional[str] = None,
    rejected_queue_path: Optional[str] = None,
    source_registry_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    output_prefix: str = "prod_chat",
    write_audit_reports: bool = True,
) -> Dict[str, Any]:
    """
    Deterministically rebuilds `prod_chat_{train,val,test}.jsonl` and `prod_chat_provenance.json`
    exclusively from verified baseline records (`chat_{train,val,test}.jsonl`) plus explicitly
    APPROVED owner/source records in `curated_approved_records.jsonl`.
    """
    m_dir = manifests_dir or os.path.join(base_dir, "datasets", "manifests")
    tok_path = tokenizer_path or os.path.join(base_dir, "models", "chat", "tokenizer.json")
    cur_path = curated_approved_path or os.path.join(m_dir, "curated_approved_records.jsonl")
    p_path = pending_queue_path or os.path.join(base_dir, "datasets", "staging", "pending_review_queue.jsonl")
    r_path = rejected_queue_path or os.path.join(base_dir, "datasets", "staging", "rejected_queue.jsonl")
    reg_path = source_registry_path or os.path.join(base_dir, "datasets", "sources", "source_registry.json")

    tok = SovereignTokenizer.load(tok_path) if os.path.isfile(tok_path) else None

    baseline_sources = {
        "train": os.path.join(m_dir, "chat_train.jsonl"),
        "val": os.path.join(m_dir, "chat_val.jsonl"),
        "test": os.path.join(m_dir, "chat_test.jsonl"),
    }

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()
    seen_replies: set[str] = set()
    source_checksums: Dict[str, str] = {}
    taxonomy_counts = {k: 0 for k in ALL_TAXONOMY_CLASSES}
    rejected_on_rebuild: List[Dict[str, Any]] = []

    def _admit_verified(
        rec: Dict[str, Any],
        src_label: str,
        src_sha: str,
        line_no: int,
        *,
        is_baseline: bool,
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        if is_baseline:
            ok, diag, errors = validate_baseline_record(
                rec, source_file=src_label, line_no=line_no, tokenizer=tok
            )
        else:
            ok, diag, errors = validate_candidate_record(
                rec, source_file=src_label, line_no=line_no, require_approved=True, tokenizer=tok
            )
            if ok:
                reviewer_str = str(rec.get("reviewer") or "").strip()
                if len(reviewer_str) < 3:
                    ok = False
                    errors = errors + [f"[{src_label}:{line_no} id={diag.get('id')}] Curated APPROVED record missing reviewer identity."]

        if not ok:
            rejected_on_rebuild.append({"id": diag.get("id"), "source": src_label, "reasons": errors})
            return None

        rec_id = str(rec["id"])
        csha = diag["conversation_sha256"]
        p_norm = diag["prompt_norm"]
        a_norm = diag["reply_norm"]
        if rec_id in seen_ids or csha in seen_hashes or (p_norm and p_norm in seen_prompts) or (a_norm and a_norm in seen_replies):
            rejected_on_rebuild.append({"id": rec_id, "source": src_label, "reasons": ["Duplicate id/hash/prompt/reply."]})
            return None

        seen_ids.add(rec_id)
        seen_hashes.add(csha)
        if p_norm:
            seen_prompts.add(p_norm)
        if a_norm:
            seen_replies.add(a_norm)

        enriched = dict(rec)
        enriched["sha256"] = csha
        enriched["truth_taxonomy"] = diag["taxonomy"]
        enriched["canonical_category"] = diag["canonical_category"]
        enriched["source_file_sha256"] = enriched.get("source_file_sha256") or src_sha
        enriched["is_dev_fixture"] = False
        enriched["is_synthetic"] = False
        if is_baseline:
            enriched["baseline_historical_verified"] = True
        taxonomy_counts[diag["taxonomy"]] += 1
        return enriched, diag["taxonomy"]

    baseline_count = 0
    curated_count = 0

    # 1. Load immutable baseline verified splits via dedicated historical baseline validation path
    for sname, spath in baseline_sources.items():
        if not os.path.isfile(spath):
            continue
        ssha = file_sha256(spath) or ""
        rel_src = _safe_relpath(spath, base_dir)
        source_checksums[rel_src] = ssha
        for line_no, rec, err in iter_jsonl_records(spath):
            if err is not None or rec is None:
                continue
            res = _admit_verified(rec, os.path.basename(spath), ssha, line_no, is_baseline=True)
            if res is not None:
                item, _ = res
                item["split"] = sname
                splits[sname].append(item)
                baseline_count += 1

    # 2. Load newly approved curated records via strict APPROVED-only validation path
    extra_approved: List[Tuple[Dict[str, Any], str]] = []
    if os.path.isfile(cur_path):
        csha_file = file_sha256(cur_path) or ""
        source_checksums[_safe_relpath(cur_path, base_dir)] = csha_file
        for line_no, rec, err in iter_jsonl_records(cur_path):
            if err is not None or rec is None:
                continue
            res = _admit_verified(rec, os.path.basename(cur_path), csha_file, line_no, is_baseline=False)
            if res is not None:
                extra_approved.append(res)
                curated_count += 1

    if extra_approved:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Ensure public sources never straddle splits if already present in baseline train/val/test
        url_existing_split: Dict[str, str] = {}
        for sname in ("train", "val", "test"):
            for item in splits[sname]:
                if item.get("truth_taxonomy") == TAXONOMY_PUBLIC_SOURCE_VERIFIED:
                    u = str(item.get("source_url") or "").strip().lower()
                    if u:
                        url_existing_split[u] = sname

        for item, tax in extra_approved:
            u = str(item.get("source_url") or "").strip().lower()
            if tax == TAXONOMY_PUBLIC_SOURCE_VERIFIED and u in url_existing_split:
                assigned = url_existing_split[u]
                item_copy = dict(item)
                item_copy["split"] = assigned
                splits[assigned].append(item_copy)
            else:
                gkey = u if tax == TAXONOMY_PUBLIC_SOURCE_VERIFIED and u else f"{u}::{item['sha256']}"
                groups[gkey].append(item)

        ordered_groups = sorted(groups.keys(), key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
        for gkey in ordered_groups:
            total_cur = sum(len(splits[s]) for s in splits)
            val_frac = len(splits["val"]) / max(1, total_cur)
            test_frac = len(splits["test"]) / max(1, total_cur)
            if val_frac < 0.10:
                target_split = "val"
            elif test_frac < 0.10:
                target_split = "test"
            else:
                target_split = "train"
            for item in groups[gkey]:
                item_copy = dict(item)
                item_copy["split"] = target_split
                splits[target_split].append(item_copy)

    out_paths: Dict[str, str] = {}
    out_hashes: Dict[str, Optional[str]] = {}
    for sname in ("train", "val", "test"):
        spath = os.path.join(m_dir, f"{output_prefix}_{sname}.jsonl")
        out_paths[sname] = spath
        out_hashes[sname] = _write_jsonl_atomic(spath, splits[sname])

    gate_result = verify_production_manifest_group(
        train_path=out_paths["train"],
        val_path=out_paths["val"],
        test_path=out_paths["test"],
        tokenizer_path=tok_path,
        require_scale=True,
    )

    total_approved = sum(len(splits[s]) for s in ("train", "val", "test"))
    provenance_doc = {
        "schema_version": "moon_chat_genuine_ingestion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": gate_result["dataset_status"],
        "eligible_for_production_training": gate_result["passed"],
        "blockers": gate_result["blockers"],
        "raw_input_records": total_approved + len(rejected_on_rebuild),
        "malformed_input_lines": 0,
        "accepted_genuine_records": total_approved,
        "approved_breakdown": {
            "immutable_baseline_records": baseline_count,
            "curated_explicitly_approved_records": curated_count,
        },
        "rejected_records_count": len(rejected_on_rebuild),
        "rejection_reasons": {"rebuild_rejected": len(rejected_on_rebuild)} if rejected_on_rebuild else {},
        "taxonomy_counts": taxonomy_counts,
        "license_counts": gate_result["audit"]["license_counts"],
        "category_counts_13_targets": gate_result["audit"]["category_counts_13_targets"],
        "source_file_sha256s": source_checksums,
        "output_files": {
            sname: {
                "path": _safe_relpath(out_paths[sname], base_dir),
                "sha256": out_hashes[sname],
                "records_count": len(splits[sname]),
            }
            for sname in ("train", "val", "test")
        },
        "token_stats": gate_result["audit"]["token_stats"],
        "cross_split_leakage": gate_result["audit"]["cross_split_leakage"],
        "shortfall": gate_result["audit"]["production_readiness"]["shortfall"],
    }

    prov_path = os.path.join(m_dir, f"{output_prefix}_provenance.json")
    os.makedirs(os.path.dirname(prov_path), exist_ok=True)
    with open(prov_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if write_audit_reports and base_dir == PROJECT_ROOT:
        run_full_truth_audit(base_dir=base_dir, output_path=DEFAULT_REPORT_PATH, use_cache=True)
        generate_progress_report(
            base_dir=base_dir,
            output_path=DEFAULT_PROGRESS_REPORT_PATH,
            pending_queue_path=p_path,
            rejected_queue_path=r_path,
            source_registry_path=reg_path,
            use_cache=True,
        )

    provenance_doc["provenance_path"] = prov_path
    return provenance_doc


def complete_pending_draft(
    record_id: str,
    *,
    answer: str,
    prompt: Optional[str] = None,
    attribution: Optional[str] = None,
    transformation_performed: Optional[str] = None,
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
) -> Dict[str, Any]:
    """
    Completes a staged PENDING_REVIEW conversion draft with owner-authored educational prose,
    preserving all source attribution/license metadata and keeping `approval_status = "PENDING_REVIEW"`
    until explicit reviewer verification and promotion.
    """
    pending = _load_records_clean(queue_path)
    tok = SovereignTokenizer.load(tokenizer_path) if os.path.isfile(tokenizer_path) else None
    updated_queue: List[Dict[str, Any]] = []
    target_updated: Optional[Dict[str, Any]] = None
    now_ts = datetime.now(timezone.utc).isoformat()

    for rec in pending:
        if str(rec.get("id")) != record_id or target_updated is not None:
            updated_queue.append(rec)
            continue

        candidate = dict(rec)
        orig_msgs = list(candidate.get("messages") or [])
        user_text = str(prompt).strip() if prompt is not None else ""
        if not user_text:
            for turn in orig_msgs:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    user_text = str(turn.get("content") or "").strip()
                    break
        candidate["messages"] = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": str(answer).strip()},
        ]
        candidate["sha256"] = compute_conversation_sha256(candidate["messages"])
        if attribution and attribution.strip():
            candidate["attribution"] = attribution.strip()
        if transformation_performed and transformation_performed.strip():
            candidate["transformation_performed"] = transformation_performed.strip()
        elif candidate.get("record_kind") == "SOURCE_CONVERSION_DRAFT":
            candidate["transformation_performed"] = (
                f"Original educational Q&A explanation synthesized from verified source metadata "
                f"({candidate.get('source_url', '')}); source attribution and license preserved."
            )
            candidate["modification_summary"] = "Owner-authored educational synthesis ready for reviewer inspection."
        candidate["needs_original_human_summary"] = False
        if candidate.get("record_kind") == "SOURCE_CONVERSION_DRAFT":
            candidate["record_kind"] = "SOURCE_DERIVED_COMPLETED"
        candidate["approval_status"] = APPROVAL_STATUS_PENDING
        candidate["reviewer"] = None
        candidate["draft_completed_at_utc"] = now_ts

        st_ok, diag, st_errs = validate_candidate_record(
            candidate, source_file=os.path.basename(queue_path), line_no=0, require_approved=False, tokenizer=tok
        )
        orig_ok, orig_err = check_draft_summary_originality(candidate)
        if not st_ok or not orig_ok:
            errs = list(st_errs) + ([orig_err] if orig_err else [])
            return {"completed": False, "id": record_id, "errors": errs}

        target_updated = candidate
        updated_queue.append(candidate)

    if target_updated is None:
        return {"completed": False, "id": record_id, "errors": [f"Record '{record_id}' not found in pending queue."]}

    _write_jsonl_atomic(queue_path, updated_queue)
    return {
        "completed": True,
        "id": record_id,
        "approval_status": APPROVAL_STATUS_PENDING,
        "record": target_updated,
        "diagnostics": diag,
    }


def approve_pending_records(
    record_ids: Sequence[str],
    *,
    reviewer: str,
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    curated_approved_path: str = DEFAULT_CURATED_APPROVED_PATH,
    source_registry_path: str = DEFAULT_REGISTRY_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    rebuild_manifests: bool = True,
    base_dir: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    """
    Promotes explicitly verified records from PENDING_REVIEW to APPROVED after validating:
      1. Reviewer identity & timestamp.
      2. Pre-approval quality, license, attribution, AST syntax, and PII checks.
      3. Draft completeness & non-verbatim originality (blocks empty or incomplete source drafts).
      4. Strict production validation with explicit `approval_status == "APPROVED"`.
    """
    if not reviewer or len(reviewer.strip()) < 3:
        raise ValueError("Approval requires non-empty reviewer identity (--reviewer).")

    pending = _load_records_clean(queue_path)
    curated = _load_records_clean(curated_approved_path)
    tok = SovereignTokenizer.load(tokenizer_path) if os.path.isfile(tokenizer_path) else None
    target_set = set(record_ids)

    remaining_pending: List[Dict[str, Any]] = []
    approved_now: List[Dict[str, Any]] = []
    failed_checks: List[Dict[str, Any]] = []

    reg = load_source_registry(source_registry_path) if os.path.isfile(source_registry_path) else None
    now_ts = datetime.now(timezone.utc).isoformat()

    for rec in pending:
        rid = str(rec.get("id") or "")
        if rid not in target_set:
            remaining_pending.append(rec)
            continue

        pre_ok, _, pre_errors = validate_candidate_record(
            rec, source_file=os.path.basename(queue_path), line_no=0, require_approved=False, tokenizer=tok
        )
        orig_ok, orig_err = check_draft_summary_originality(rec)
        if not pre_ok or not orig_ok:
            combined_errs = list(pre_errors) + ([orig_err] if orig_err else [])
            failed_checks.append({"id": rid, "errors": combined_errs})
            remaining_pending.append(rec)
            continue

        promoted_rec = dict(rec)
        promoted_rec["approval_status"] = APPROVAL_STATUS_APPROVED
        promoted_rec["reviewer"] = reviewer.strip()
        promoted_rec["reviewed_at_utc"] = now_ts
        promoted_rec["needs_original_human_summary"] = False

        ok, diag, errors = validate_candidate_record(
            promoted_rec, source_file=os.path.basename(queue_path), line_no=0, require_approved=True, tokenizer=tok
        )
        if not ok:
            failed_checks.append({"id": rid, "errors": errors})
            remaining_pending.append(rec)
            continue

        promoted_rec["sha256"] = diag["conversation_sha256"]
        curated.append(promoted_rec)
        approved_now.append(promoted_rec)

        src_id = promoted_rec.get("source_id")
        if reg and src_id and src_id in reg.get("sources", {}):
            reg["sources"][src_id]["approval_status"] = APPROVAL_STATUS_APPROVED
            reg["sources"][src_id]["reviewer"] = reviewer.strip()
            reg["sources"][src_id]["reviewed_at_utc"] = now_ts

    not_found = sorted(target_set - ({str(r.get("id")) for r in approved_now} | {f["id"] for f in failed_checks}))

    if approved_now:
        _write_jsonl_atomic(queue_path, remaining_pending)
        _write_jsonl_atomic(curated_approved_path, curated)
        if reg:
            save_source_registry(reg, source_registry_path)

    rebuild_res = None
    if rebuild_manifests and approved_now:
        rebuild_res = rebuild_production_manifests(
            base_dir=base_dir,
            curated_approved_path=curated_approved_path,
            pending_queue_path=queue_path,
            source_registry_path=source_registry_path,
            tokenizer_path=tokenizer_path,
        )

    return {
        "approved_count": len(approved_now),
        "approved_ids": [r["id"] for r in approved_now],
        "failed_validation": failed_checks,
        "not_found_ids": not_found,
        "remaining_pending_count": len(remaining_pending),
        "rebuild_result": rebuild_res,
    }


def reject_pending_record(
    record_id: str,
    *,
    reviewer: str,
    reason: str,
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    rejected_queue_path: str = DEFAULT_REJECTED_QUEUE_PATH,
    source_registry_path: str = DEFAULT_REGISTRY_PATH,
) -> Dict[str, Any]:
    if not reviewer or len(reviewer.strip()) < 3:
        raise ValueError("Rejection requires reviewer identity.")
    if not reason or len(reason.strip()) < 3:
        raise ValueError("Rejection requires explicit reason.")

    pending = _load_records_clean(queue_path)
    rejected = _load_records_clean(rejected_queue_path)
    remaining: List[Dict[str, Any]] = []
    rejected_item: Optional[Dict[str, Any]] = None

    for rec in pending:
        if str(rec.get("id")) == record_id and rejected_item is None:
            copy_r = dict(rec)
            copy_r["approval_status"] = APPROVAL_STATUS_REJECTED
            copy_r["reviewer"] = reviewer.strip()
            copy_r["rejection_reason"] = reason.strip()
            copy_r["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            rejected.append(copy_r)
            rejected_item = copy_r
        else:
            remaining.append(rec)

    if rejected_item is None:
        return {"rejected": False, "reason": f"Record '{record_id}' not found in pending queue."}

    _write_jsonl_atomic(queue_path, remaining)
    _write_jsonl_atomic(rejected_queue_path, rejected)

    if os.path.isfile(source_registry_path):
        reg = load_source_registry(source_registry_path)
        src_id = rejected_item.get("source_id")
        if src_id and src_id in reg.get("sources", {}):
            reg["sources"][src_id]["approval_status"] = APPROVAL_STATUS_REJECTED
            reg["sources"][src_id]["reviewer"] = reviewer.strip()
            reg["sources"][src_id]["reviewed_at_utc"] = rejected_item["reviewed_at_utc"]
            save_source_registry(reg, source_registry_path)

    return {"rejected": True, "id": record_id, "remaining_pending_count": len(remaining)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Moon Chat Review Queue & Manifest Builder")
    parser.add_argument("--status", action="store_true", help="Print progress summary and refresh progress JSON.")
    parser.add_argument("--list", action="store_true", help="List items currently in PENDING_REVIEW.")
    parser.add_argument("--show", type=str, default=None, help="Inspect full details, source extract, and status for one pending record.")
    parser.add_argument("--complete-draft", type=str, default=None, help="Record ID in pending queue to fill in with owner-authored answer.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional revised user question for --complete-draft.")
    parser.add_argument("--answer", type=str, default=None, help="Owner-authored original educational answer for --complete-draft.")
    parser.add_argument("--author", type=str, default=None, help="Optional updated attribution for --complete-draft.")
    parser.add_argument("--approve", nargs="+", default=None, help="Record ID(s) to approve and promote.")
    parser.add_argument("--reject", type=str, default=None, help="Record ID to reject.")
    parser.add_argument("--reviewer", type=str, default="", help="Reviewer name/identity.")
    parser.add_argument("--reason", type=str, default="", help="Rejection reason.")
    parser.add_argument("--rebuild-manifests", action="store_true", help="Rebuild production manifests deterministically from APPROVED/baseline records.")
    args = parser.parse_args()

    if args.list:
        pending = list_pending_records()
        print(f"Pending review queue ({len(pending)} records):")
        for r in pending:
            ready_ok, _, errs = validate_candidate_record(r, require_approved=False)
            orig_ok, orig_err = check_draft_summary_originality(r)
            if ready_ok and orig_ok:
                state = "READY_FOR_REVIEW"
            else:
                reason = errs[0] if errs else (orig_err or "incomplete draft")
                state = f"NEEDS_EDIT ({reason})"
            print(f"  - {r.get('id')} | {r.get('category')} | {r.get('license')} | {state}")
        return

    if args.show:
        pending = list_pending_records()
        match = next((r for r in pending if r.get("id") == args.show), None)
        if match is None:
            print(f"[ReviewQueue] Record '{args.show}' not found in pending queue.")
            return
        print(json.dumps(match, indent=2, ensure_ascii=False))
        return

    if args.complete_draft:
        if not args.answer:
            print("[ReviewQueue] Error: --complete-draft requires --answer \"...\".")
            return
        res = complete_pending_draft(
            args.complete_draft,
            answer=args.answer,
            prompt=args.prompt,
            attribution=args.author,
        )
        print(f"[ReviewQueue] complete_draft={res.get('completed')} id={res.get('id')} status={res.get('approval_status')}")
        if not res.get("completed"):
            print(f"  Errors: {res.get('errors')}")
        return

    if args.approve:
        res = approve_pending_records(args.approve, reviewer=args.reviewer)
        print(
            f"[ReviewQueue] approved={res['approved_count']} failed={len(res['failed_validation'])} "
            f"not_found={res['not_found_ids']} remaining_pending={res['remaining_pending_count']}"
        )
        for fc in res["failed_validation"]:
            print(f"  [FAILED id={fc['id']}] {fc['errors']}")
        return

    if args.reject:
        res = reject_pending_record(args.reject, reviewer=args.reviewer, reason=args.reason)
        print(f"[ReviewQueue] reject result={res}")
        return

    if args.rebuild_manifests:
        res = rebuild_production_manifests()
        print(
            f"[RebuildManifests] status={res['dataset_status']} accepted={res['accepted_genuine_records']} "
            f"supervised_tokens={res['token_stats']['assistant_supervised_tokens']}"
        )
        return

    prog = generate_progress_report()
    app = prog["current_approved_production"]
    rq = prog["review_queue"]
    print(f"[DatasetProgress] status={prog['dataset_status']}")
    print(
        f"  Approved: {app['total_records']} (train={app['train_records']}, val={app['val_records']}, "
        f"test={app['test_records']}), supervised_tokens={app['supervised_assistant_tokens']}, "
        f"approval_breakdown={app.get('approval_status_distribution', {})}"
    )
    print(
        f"  Pending Review: {rq['pending_review_count']} | Rejected: {rq['rejected_count']} | "
        f"Sources Registered: {rq['registered_sources_count']}"
    )
    print(f"  Shortfall: {prog['shortfall']}")


if __name__ == "__main__":
    main()

