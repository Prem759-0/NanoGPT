"""
tests/test_phase_2_7_9_approval_gate_and_dataset_integrity.py
Phase 2.7.9 Regression Suite: Approval Gate Hardening, Baseline Path Isolation,
Draft Completeness Enforcement, Owner Contribution Workflow, and Dataset Truth.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from typing import Any, Dict

import core.policy as policy
from datasets.chat.acquisition import register_local_permitted_source
from datasets.chat.contribute import (
    CATEGORY_AUTHORING_GUIDANCE,
    DEFAULT_TEMPLATES_PATH,
    build_empty_owner_templates,
    import_contributions,
    interactive_submit,
    list_category_guidance,
    submit_single_qa,
)
from datasets.chat.genuine_ingestion import ingest_genuine_sources, run_repo_baseline_ingestion
from datasets.chat.quality_gate import (
    validate_baseline_record,
    validate_candidate_record,
)
from datasets.chat.review_queue import (
    DEFAULT_PENDING_QUEUE_PATH,
    approve_pending_records,
    complete_pending_draft,
    list_pending_records,
    rebuild_production_manifests,
    reject_pending_record,
)
from datasets.chat.schema import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    IMMUTABLE_BASELINE_SHA256_BY_ID,
    SUPPORTED_CHAT_CATEGORIES,
    is_immutable_baseline_record,
    validate_chat_record,
    validate_curated_production_record,
    validate_immutable_baseline_record,
    validate_staging_candidate_record,
)
from datasets.chat.truth_audit import (
    DEFAULT_PROGRESS_REPORT_PATH,
    TAXONOMY_UNVERIFIED,
    classify_chat_record,
    run_full_truth_audit,
    verify_production_manifest_group,
)
from training.chat.dataset import SovereignChatDataset
from datasets.chat.tokenizer import SovereignTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")


def _new_curated_record(
    rec_id: str = "curated_py_test_001",
    *,
    include_approval: bool = True,
    approval_status: Any = APPROVAL_STATUS_APPROVED,
    reviewer: Any = "Prem Gaikwad",
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": rec_id,
        "category": "python",
        "provenance_type": "ORIGINAL_SELF_AUTHORED_QA",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Moon AI Chief ML Systems Architect)",
        "source": "Owner Original Contribution",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions",
        "transformation_performed": "Original owner-authored Python Q&A explanation.",
        "is_dev_fixture": False,
        "is_synthetic": False,
        "messages": [
            {"role": "user", "content": "How does a Python context manager guarantee resource cleanup?"},
            {
                "role": "assistant",
                "content": (
                    "A Python context manager implements `__enter__` and `__exit__`, ensuring cleanup logic "
                    "runs reliably when leaving a `with` block even if an exception occurs.\n\n"
                    "```python\nwith open('notes.txt', 'w', encoding='utf-8') as handle:\n    handle.write('done')\n```\n"
                ),
            },
        ],
    }
    if include_approval:
        rec["approval_status"] = approval_status
    if reviewer is not None:
        rec["reviewer"] = reviewer
    return rec


def _write_jsonl(path: str, records: list[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestPhase279ApprovalGateHardening(unittest.TestCase):
    """Proves missing/pending/rejected approval statuses fail closed across all public gates."""

    def test_01_missing_none_pending_and_rejected_fail_closed(self):
        tok = SovereignTokenizer.load(TOKENIZER_PATH)

        # 1. Missing approval_status key -> MUST fail default and explicit require_approved=True
        rec_missing = _new_curated_record(rec_id="no_app_key", include_approval=False)
        snapshot = copy.deepcopy(rec_missing)
        ok_default, _, errs_default = validate_candidate_record(rec_missing, tokenizer=tok)
        self.assertFalse(ok_default)
        self.assertTrue(any("approval_status" in e for e in errs_default))

        ok_explicit, _, errs_explicit = validate_candidate_record(rec_missing, require_approved=True, tokenizer=tok)
        self.assertFalse(ok_explicit)
        self.assertEqual(rec_missing, snapshot, "Validation must never mutate input record.")

        schema_ok, schema_err = validate_curated_production_record(rec_missing)
        self.assertFalse(schema_ok)
        self.assertIn("APPROVED", str(schema_err))

        # 2. Explicit None / empty / wrong case / PENDING_REVIEW / REJECTED -> MUST fail closed
        for bad_val in (None, "", "approved", APPROVAL_STATUS_PENDING, APPROVAL_STATUS_REJECTED, "BYPASS"):
            bad_rec = _new_curated_record(rec_id=f"bad_{bad_val}", include_approval=True, approval_status=bad_val)
            ok, _, errs = validate_candidate_record(bad_rec, require_approved=True, tokenizer=tok)
            self.assertFalse(ok, f"Expected approval_status={bad_val!r} to fail closed.")
            self.assertTrue(any("approval_status" in e or "APPROVED" in e for e in errs))

            ok_schema, _ = validate_curated_production_record(bad_rec)
            self.assertFalse(ok_schema)

        # 3. Valid explicit APPROVED record passes strictly
        good_rec = _new_curated_record(rec_id="valid_approved_01")
        ok_good, diag_good, errs_good = validate_candidate_record(good_rec, require_approved=True, tokenizer=tok)
        self.assertTrue(ok_good, f"Expected explicit APPROVED curated record to pass, got {errs_good}")
        self.assertEqual(diag_good["canonical_category"], "python")
        ok_cur, err_cur = validate_curated_production_record(good_rec)
        self.assertTrue(ok_cur, err_cur)

    def test_02_staging_validation_never_silently_marks_approved(self):
        pending_rec = _new_curated_record(rec_id="pending_check_01", approval_status=APPROVAL_STATUS_PENDING, reviewer=None)
        st_ok, st_err = validate_staging_candidate_record(pending_rec)
        self.assertTrue(st_ok, st_err)
        self.assertEqual(pending_rec["approval_status"], APPROVAL_STATUS_PENDING)

        # Direct production validation or classification (non-staging) must fail closed on PENDING/REJECTED
        ok_chat, _ = validate_chat_record(pending_rec, is_production=True)
        self.assertFalse(ok_chat)
        tax_pending, _ = classify_chat_record(pending_rec)
        self.assertEqual(tax_pending, TAXONOMY_UNVERIFIED)

        rej_rec = _new_curated_record(rec_id="rej_check_01", approval_status=APPROVAL_STATUS_REJECTED, reviewer="Prem")
        ok_rej, _, _ = validate_candidate_record(rej_rec, require_approved=False)
        self.assertFalse(ok_rej)
        tax_rej, _ = classify_chat_record(rej_rec)
        self.assertEqual(tax_rej, TAXONOMY_UNVERIFIED)

    def test_03_baseline_path_preserves_22_and_blocks_curated_spoofing(self):
        tok = SovereignTokenizer.load(TOKENIZER_PATH)
        self.assertEqual(len(IMMUTABLE_BASELINE_SHA256_BY_ID), 22)

        baseline_records = []
        for split in ("train", "val", "test"):
            p = os.path.join(PROJECT_ROOT, "datasets", "manifests", f"chat_{split}.jsonl")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        baseline_records.append(json.loads(line))
        self.assertEqual(len(baseline_records), 22)

        for br in baseline_records:
            self.assertNotIn("approval_status", br)
            self.assertTrue(is_immutable_baseline_record(br))
            ok_b, _, errs_b = validate_baseline_record(br, tokenizer=tok)
            self.assertTrue(ok_b, f"Baseline record {br.get('id')} failed validate_baseline_record: {errs_b}")

        # Spoofing attempt 1: brand-new curated record lacking approval_status via baseline path MUST fail
        rogue_new = _new_curated_record(rec_id="rogue_new_001", include_approval=False)
        self.assertFalse(is_immutable_baseline_record(rogue_new))
        ok_rogue, _, _ = validate_baseline_record(rogue_new, tokenizer=tok)
        self.assertFalse(ok_rogue)
        ok_imm, _ = validate_immutable_baseline_record(rogue_new)
        self.assertFalse(ok_imm)

        # Spoofing attempt 2: reusing baseline ID with modified message text MUST fail SHA match
        spoof_id = copy.deepcopy(baseline_records[0])
        spoof_id["messages"][1]["content"] = "Tampered response text reusing baseline record identifier cleanly."
        spoof_id.pop("sha256", None)
        ok_spoof, _, _ = validate_baseline_record(spoof_id, tokenizer=tok)
        self.assertFalse(ok_spoof)

        # Spoofing attempt 3: valid baseline record tampered with PENDING_REVIEW status MUST fail
        pending_base = copy.deepcopy(baseline_records[0])
        pending_base["approval_status"] = APPROVAL_STATUS_PENDING
        self.assertFalse(is_immutable_baseline_record(pending_base))
        ok_pb, _, _ = validate_baseline_record(pending_base, tokenizer=tok)
        self.assertFalse(ok_pb)


class TestPhase279PublicManifestAndIngestionGates(unittest.TestCase):
    """Exercises public ingestion, rebuild_production_manifests, and truth verification gates."""

    def test_01_ingest_genuine_sources_blocks_missing_pending_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = os.path.join(tmp, "tr_in.jsonl")
            va = os.path.join(tmp, "va_in.jsonl")
            te = os.path.join(tmp, "te_in.jsonl")
            out_dir = os.path.join(tmp, "out")

            missing_app = _new_curated_record("ing_missing", include_approval=False)
            pending_app = _new_curated_record("ing_pending", approval_status=APPROVAL_STATUS_PENDING)
            rejected_app = _new_curated_record("ing_rejected", approval_status=APPROVAL_STATUS_REJECTED)
            valid_app = _new_curated_record("ing_approved_ok", approval_status=APPROVAL_STATUS_APPROVED)

            _write_jsonl(tr, [missing_app, pending_app, rejected_app])
            _write_jsonl(va, [valid_app])
            _write_jsonl(te, [])

            res = ingest_genuine_sources(
                [tr, va],
                output_dir=out_dir,
                output_prefix="test_prod",
                tokenizer_path=TOKENIZER_PATH,
            )
            self.assertEqual(res["accepted_genuine_records"], 1)
            self.assertEqual(res["rejected_records_count"], 3)

            # Verify accepted record preserved explicit APPROVED without mutating rejected records in
            with open(os.path.join(out_dir, "test_prod_train.jsonl"), "r", encoding="utf-8") as f:
                written = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["id"], "ing_approved_ok")
            self.assertEqual(written[0]["approval_status"], APPROVAL_STATUS_APPROVED)

    def test_02_rebuild_and_manifest_gate_block_unapproved_curated_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            m_dir = os.path.join(tmp, "datasets", "manifests")
            os.makedirs(m_dir, exist_ok=True)
            for split in ("train", "val", "test"):
                src = os.path.join(PROJECT_ROOT, "datasets", "manifests", f"chat_{split}.jsonl")
                with open(src, "r", encoding="utf-8") as sf, open(os.path.join(m_dir, f"{os.path.basename(src)}"), "w", encoding="utf-8", newline="\n") as df:
                    df.write(sf.read())

            curated_path = os.path.join(m_dir, "curated_approved_records.jsonl")
            pending_path = os.path.join(tmp, "datasets", "staging", "pending_review_queue.jsonl")
            rej_path = os.path.join(tmp, "datasets", "staging", "rejected_queue.jsonl")
            reg_path = os.path.join(tmp, "datasets", "sources", "source_registry.json")

            # Curated ledger tampered with:
            # 1) missing approval_status, 2) PENDING_REVIEW, 3) REJECTED, 4) APPROVED missing reviewer, 5) valid APPROVED
            bad_missing = _new_curated_record("cur_missing_status", include_approval=False)
            bad_pending = _new_curated_record("cur_pending_status", approval_status=APPROVAL_STATUS_PENDING)
            bad_rejected = _new_curated_record("cur_rejected_status", approval_status=APPROVAL_STATUS_REJECTED)
            bad_no_rev = _new_curated_record("cur_no_reviewer", approval_status=APPROVAL_STATUS_APPROVED, reviewer="")
            good_cur = _new_curated_record("cur_valid_01", approval_status=APPROVAL_STATUS_APPROVED, reviewer="Prem Gaikwad")
            good_cur["messages"][0]["content"] = "Explain Python decorators with a concise example."
            good_cur["messages"][1]["content"] = "A decorator wraps a callable to extend behavior cleanly without modifying its source definition."

            _write_jsonl(curated_path, [bad_missing, bad_pending, bad_rejected, bad_no_rev, good_cur])

            res = rebuild_production_manifests(
                base_dir=tmp,
                manifests_dir=m_dir,
                curated_approved_path=curated_path,
                pending_queue_path=pending_path,
                rejected_queue_path=rej_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                write_audit_reports=False,
            )
            self.assertEqual(res["approved_breakdown"]["immutable_baseline_records"], 22)
            self.assertEqual(res["approved_breakdown"]["curated_explicitly_approved_records"], 1)
            self.assertEqual(res["accepted_genuine_records"], 23)
            self.assertEqual(res["rejected_records_count"], 4)

            # Verify prod manifest group gate rejects tampered unapproved non-baseline record inserted directly on disk
            tampered_val = os.path.join(m_dir, "tampered_val.jsonl")
            with open(os.path.join(m_dir, "prod_chat_val.jsonl"), "r", encoding="utf-8") as vf:
                val_lines = vf.read()
            with open(tampered_val, "w", encoding="utf-8", newline="\n") as tvf:
                tvf.write(val_lines)
                tvf.write(json.dumps(bad_missing, ensure_ascii=False) + "\n")

            gate = verify_production_manifest_group(
                train_path=os.path.join(m_dir, "prod_chat_train.jsonl"),
                val_path=tampered_val,
                test_path=os.path.join(m_dir, "prod_chat_test.jsonl"),
                tokenizer_path=TOKENIZER_PATH,
                require_scale=False,
            )
            self.assertFalse(gate["passed"])
            self.assertTrue(any("unapproved or missing-approval-status" in b for b in gate["blockers"]))


class TestPhase279ReviewWorkflowAndOwnerContribution(unittest.TestCase):
    """Verifies real pending drafts integrity, draft completion barrier, owner CLI, and repo truth."""

    def test_01_repo_20_pending_drafts_preserved_and_cannot_auto_promote(self):
        pending = list_pending_records(DEFAULT_PENDING_QUEUE_PATH)
        self.assertGreaterEqual(len(pending), 20)
        for rec in pending:
            self.assertEqual(rec["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertTrue(rec.get("attribution"))
            self.assertTrue(rec.get("license"))
            self.assertTrue(rec.get("source_url"))
            self.assertTrue(rec.get("original_file_sha256"))
            # Every staged raw source draft without completed educational response must fail staging/prod validation
            ok_prod, _, _ = validate_candidate_record(rec, require_approved=True)
            self.assertFalse(ok_prod)

    def test_02_source_draft_completion_review_and_rejection_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            m_dir = os.path.join(tmp, "datasets", "manifests")
            os.makedirs(m_dir, exist_ok=True)
            for split in ("train", "val", "test"):
                src = os.path.join(PROJECT_ROOT, "datasets", "manifests", f"chat_{split}.jsonl")
                with open(src, "r", encoding="utf-8") as sf, open(os.path.join(m_dir, os.path.basename(src)), "w", encoding="utf-8", newline="\n") as df:
                    df.write(sf.read())

            queue_path = os.path.join(tmp, "datasets", "staging", "pending_review_queue.jsonl")
            curated_path = os.path.join(m_dir, "curated_approved_records.jsonl")
            rej_path = os.path.join(tmp, "datasets", "staging", "rejected_queue.jsonl")
            reg_path = os.path.join(tmp, "datasets", "sources", "source_registry.json")
            src_file = os.path.join(tmp, "src.txt")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write("Binary search educational reference extract.\n")

            extract_text = (
                "Binary search is a search algorithm that finds the position of a target value "
                "within a sorted array by repeatedly halving the search interval."
            )
            reg_res = register_local_permitted_source(
                file_path=src_file,
                source_id="wiki_binary_search_test",
                source_url="https://en.wikipedia.org/w/index.php?title=Binary_search&oldid=123456",
                title="Wikipedia: Binary search",
                category="computer_science",
                author_or_publisher="Wikipedia contributors / Wikimedia Foundation",
                license_name="CC-BY-SA-4.0",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                attribution="Wikipedia contributors, 'Binary search' (CC-BY-SA-4.0)",
                transformation_performed="Staged conversion draft awaiting educational summary.",
                suggested_question="How does binary search work and what precondition does it require?",
                summary_excerpt=extract_text,
                registry_path=reg_path,
                queue_path=queue_path,
            )
            draft_id = reg_res["staged_id"]

            # 1. Empty draft CANNOT be approved
            app_empty = approve_pending_records(
                [draft_id],
                reviewer="Prem Gaikwad",
                queue_path=queue_path,
                curated_approved_path=curated_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                rebuild_manifests=True,
                base_dir=tmp,
            )
            self.assertEqual(app_empty["approved_count"], 0)

            # 2. Verbatim extract copy CANNOT complete draft
            comp_copy = complete_pending_draft(
                draft_id,
                answer=extract_text,
                queue_path=queue_path,
                tokenizer_path=TOKENIZER_PATH,
            )
            self.assertFalse(comp_copy["completed"])

            # 3. Draft manually modified without clearing needs_original_human_summary still blocked on approve
            pending_items = list_pending_records(queue_path)
            patched = dict(pending_items[0])
            patched["messages"][1]["content"] = (
                "Binary search requires a sorted collection and inspects the middle item at each step "
                "to eliminate half of the remaining candidates, achieving O(log n) worst-case runtime."
            )
            _write_jsonl(queue_path, [patched])
            app_unreviewed = approve_pending_records(
                [draft_id],
                reviewer="Prem Gaikwad",
                queue_path=queue_path,
                curated_approved_path=curated_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                rebuild_manifests=False,
                base_dir=tmp,
            )
            self.assertEqual(app_unreviewed["approved_count"], 0)

            # Reset draft cleanly, complete with complete_pending_draft (stays PENDING_REVIEW!), then explicitly approve
            _write_jsonl(queue_path, [pending_items[0]])
            comp_ok = complete_pending_draft(
                draft_id,
                answer=(
                    "Binary search finds an item in a sorted array by comparing the target to the middle element "
                    "and discarding half of the search range each step, running in O(log n) time."
                ),
                queue_path=queue_path,
                tokenizer_path=TOKENIZER_PATH,
            )
            self.assertTrue(comp_ok["completed"])
            self.assertEqual(comp_ok["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertEqual(comp_ok["record"]["license"], "CC-BY-SA-4.0")
            self.assertIn("Wikipedia contributors", comp_ok["record"]["attribution"])

            app_ok = approve_pending_records(
                [draft_id],
                reviewer="Prem Gaikwad",
                queue_path=queue_path,
                curated_approved_path=curated_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                rebuild_manifests=True,
                base_dir=tmp,
            )
            self.assertEqual(app_ok["approved_count"], 1)
            self.assertEqual(app_ok["rebuild_result"]["approved_breakdown"]["immutable_baseline_records"], 22)
            self.assertEqual(app_ok["rebuild_result"]["approved_breakdown"]["curated_explicitly_approved_records"], 1)
            self.assertEqual(app_ok["rebuild_result"]["accepted_genuine_records"], 23)
            with open(curated_path, "r", encoding="utf-8") as cf:
                promoted = json.loads(cf.readline())
            self.assertEqual(promoted["approval_status"], APPROVAL_STATUS_APPROVED)
            self.assertEqual(promoted["reviewer"], "Prem Gaikwad")
            self.assertTrue(promoted.get("reviewed_at_utc"))
            self.assertEqual(promoted["license"], "CC-BY-SA-4.0")

            # 4. Rejected records remain excluded from production manifests even after rebuild
            sub_rej = submit_single_qa(
                category="mathematics",
                prompt="What is the Pythagorean theorem?",
                answer="For a right triangle with legs a and b and hypotenuse c, the squared sides satisfy a^2 + b^2 = c^2.",
                attribution="Prem Gaikwad",
                record_id="owner_math_rej_01",
                queue_path=queue_path,
                approved_paths=(),
                tokenizer_path=TOKENIZER_PATH,
            )
            self.assertTrue(sub_rej["accepted"])
            rej_res = reject_pending_record(
                "owner_math_rej_01",
                reviewer="Prem Gaikwad",
                reason="Excluded for duplicate concept check.",
                queue_path=queue_path,
                rejected_queue_path=rej_path,
                source_registry_path=reg_path,
            )
            self.assertTrue(rej_res["rejected"])
            rebuilt_after_rej = rebuild_production_manifests(
                base_dir=tmp,
                manifests_dir=m_dir,
                curated_approved_path=curated_path,
                pending_queue_path=queue_path,
                rejected_queue_path=rej_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                write_audit_reports=False,
            )
            self.assertEqual(rebuilt_after_rej["accepted_genuine_records"], 23)

    def test_03_owner_contribution_interactive_guidance_and_13_categories(self):
        self.assertEqual(set(CATEGORY_AUTHORING_GUIDANCE.keys()), set(SUPPORTED_CHAT_CATEGORIES.keys()))
        guidance_rows = list_category_guidance(DEFAULT_PROGRESS_REPORT_PATH)
        self.assertEqual(len(guidance_rows), 13)
        templates = build_empty_owner_templates()
        self.assertEqual(len(templates), 13)
        self.assertTrue(os.path.isfile(DEFAULT_TEMPLATES_PATH))
        for t in templates:
            self.assertEqual(t["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertEqual(t["messages"][0]["content"], "")
            self.assertEqual(t["messages"][1]["content"], "")
            self.assertTrue(t.get("_category_guidance"))

        with tempfile.TemporaryDirectory() as tmp:
            queue_path = os.path.join(tmp, "pending.jsonl")
            answers_iter = iter([
                "5",  # Python category
                "Prem Gaikwad",
                "How do Python dataclasses reduce boilerplate code?",
                "Dataclasses automatically generate `__init__`, `__repr__`, and equality methods from typed class attributes.",
                "Let's see an example:",
                "```python",
                "from dataclasses import dataclass",
                "@dataclass",
                "class Point:",
                "    x: float",
                "    y: float",
                "```",
                "END",
            ])
            outputs: list[str] = []
            res = interactive_submit(
                queue_path=queue_path,
                approved_paths=(),
                tokenizer_path=TOKENIZER_PATH,
                input_fn=lambda _p="": next(answers_iter),
                print_fn=outputs.append,
            )
            self.assertTrue(res["accepted"], res.get("reasons"))
            self.assertEqual(res["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertEqual(res["record"]["category"], "python")
            self.assertIsNone(res["record"]["reviewer"])

            # Import attempting self-approval is still forced PENDING_REVIEW
            imp_file = os.path.join(tmp, "imp.jsonl")
            self_app = _new_curated_record("owner_imp_01", approval_status=APPROVAL_STATUS_APPROVED, reviewer="Fake")
            _write_jsonl(imp_file, [self_app])
            imp_res = import_contributions(imp_file, queue_path=queue_path, approved_paths=(), tokenizer_path=TOKENIZER_PATH)
            self.assertEqual(imp_res["accepted_to_pending_review_count"], 1)
            self.assertEqual(imp_res["approval_status_assigned"], APPROVAL_STATUS_PENDING)
            queued = list_pending_records(queue_path)
            self.assertEqual(len(queued), 2)
            self.assertTrue(all(r["approval_status"] == APPROVAL_STATUS_PENDING and r["reviewer"] is None for r in queued))

    def test_04_repo_state_honesty_policy_and_baseline_ingestion(self):
        self.assertEqual(policy.ACTIVE_OWNERSHIP_POLICY, "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
        self.assertFalse(policy.AUTHORIZED_BY_PREM)

        audit = run_full_truth_audit(use_cache=False)
        self.assertEqual(audit["executive_summary"]["dataset_status"], "DATASET_NOT_READY")
        self.assertEqual(audit["executive_summary"]["true_qualifying_baseline_records"], 22)
        self.assertEqual(audit["executive_summary"]["production_candidate_qualifying_records"], 22)
        self.assertEqual(audit["executive_summary"]["production_candidate_supervised_assistant_tokens"], 7199)
        self.assertEqual(audit["executive_summary"]["exact_shortfall"]["qualifying_records_shortfall"], 478)
        self.assertEqual(audit["executive_summary"]["exact_shortfall"]["supervised_tokens_shortfall"], 92801)

        with open(DEFAULT_PROGRESS_REPORT_PATH, "r", encoding="utf-8") as f:
            prog = json.load(f)
        self.assertEqual(prog["current_approved_production"]["total_records"], 22)
        self.assertEqual(prog["current_approved_production"]["supervised_assistant_tokens"], 7199)
        self.assertEqual(prog["review_queue"]["pending_review_count"], 20)
        self.assertEqual(prog["review_queue"]["rejected_count"], 0)

        # Verify dedicated baseline ingestion helper produces valid prod_chat manifests in isolated temp workspace
        with tempfile.TemporaryDirectory() as tmp:
            tmp_m = os.path.join(tmp, "datasets", "manifests")
            tmp_tok_dir = os.path.join(tmp, "models", "chat")
            os.makedirs(tmp_m, exist_ok=True)
            os.makedirs(tmp_tok_dir, exist_ok=True)
            for s in ("train", "val", "test"):
                with open(os.path.join(PROJECT_ROOT, "datasets", "manifests", f"chat_{s}.jsonl"), "r", encoding="utf-8") as sf, open(os.path.join(tmp_m, f"chat_{s}.jsonl"), "w", encoding="utf-8", newline="\n") as df:
                    df.write(sf.read())
            with open(TOKENIZER_PATH, "r", encoding="utf-8") as tf, open(os.path.join(tmp_tok_dir, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as dtf:
                dtf.write(tf.read())
            base_res = run_repo_baseline_ingestion(base_dir=tmp)
            self.assertEqual(base_res["accepted_genuine_records"], 22)
            self.assertEqual(base_res["rejected_records_count"], 0)
            self.assertEqual(base_res["token_stats"]["assistant_supervised_tokens"], 7199)


if __name__ == "__main__":
    unittest.main()
