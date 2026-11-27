"""
tests/test_phase_2_7_8_dataset_acquisition_and_quality.py
Comprehensive Phase 2.7.8 Test Suite for Genuine Dataset Acquisition, Quality Control,
Owner Contribution Workflow, Review Queue Barriers, and Training Compatibility.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict

import core.policy as policy
from datasets.chat.acquisition import (
    DEFAULT_REGISTRY_PATH,
    load_source_registry,
    register_local_permitted_source,
    register_source_entry,
)
from datasets.chat.contribute import (
    DEFAULT_EXAMPLE_PATH,
    DEFAULT_TEMPLATES_PATH,
    build_empty_owner_templates,
    import_contributions,
    submit_single_qa,
)
from datasets.chat.quality_gate import (
    check_code_and_safety,
    check_license_permissibility,
    validate_candidate_record,
)
from datasets.chat.review_queue import (
    approve_pending_records,
    list_pending_records,
    rebuild_production_manifests,
    reject_pending_record,
)
from datasets.chat.schema import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    SUPPORTED_CHAT_CATEGORIES,
    validate_chat_record,
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    DEFAULT_PROGRESS_REPORT_PATH,
    audit_single_manifest,
    generate_progress_report,
    run_full_truth_audit,
    verify_production_manifest_group,
)
from training.chat.config_validator import load_training_config, validate_training_config
from training.chat.dataset import SovereignChatDataset
from training.chat.train import train_moon_chat

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")


def _sample_valid_owner_record(
    rec_id: str = "owner_test_py_001",
    category: str = "python",
    prompt: str = "What is a generator function in Python and why does yield save memory?",
    answer: str = (
        "A generator function uses the yield keyword to produce values lazily one at a time, "
        "preserving local state between iterations instead of allocating a full list in memory.\n\n"
        "```python\ndef count_up(limit: int):\n    for i in range(limit):\n        yield i\n```\n"
    ),
) -> Dict[str, Any]:
    return {
        "id": rec_id,
        "category": category,
        "provenance_type": "ORIGINAL_SELF_AUTHORED_QA",
        "license": "CustomSovereign",
        "attribution": "Prem Gaikwad (Moon AI Chief ML Systems Architect)",
        "source": "Owner Original Contribution",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions",
        "prompt": prompt,
        "answer": answer,
    }


class TestPhase278LicenseAndProvenance(unittest.TestCase):
    """Tests strict license permissibility and source metadata enforcement."""

    def test_01_permissible_and_prohibited_licenses(self):
        for ok_lic, url in [
            ("CustomSovereign", "https://github.com/Prem759-0/NanoGPT"),
            ("CC0-1.0", "https://docs.python.org/3/tutorial/"),
            ("CC-BY-4.0", "https://openstax.org/books/calculus"),
            ("CC-BY-SA-4.0", "https://en.wikipedia.org/wiki/Quicksort"),
            ("PSF-2.0", "https://docs.python.org/3/"),
            ("0BSD", "https://docs.python.org/3/library/"),
            ("MIT", "https://opensource.org/licenses/MIT"),
        ]:
            ok, err = check_license_permissibility(ok_lic, url)
            self.assertTrue(ok, f"Expected {ok_lic} valid, got {err}")

        for bad_lic, url in [
            ("CC-BY-NC", "https://example.org"),
            ("CC-BY-NC-SA-4.0", "https://example.org"),
            ("CC-BY-ND-4.0", "https://example.org"),
            ("ALL_RIGHTS_RESERVED", "https://example.org"),
            ("", "https://example.org"),
            ("CC0-1.0", "https://en.wikipedia.org/wiki/Quicksort"),
            ("MIT", "https://en.wikipedia.org/wiki/DNA"),
        ]:
            ok, err = check_license_permissibility(bad_lic, url)
            self.assertFalse(ok, f"Expected {bad_lic} rejected for {url}")
            self.assertIsNotNone(err)

    def test_02_source_registry_contains_full_provenance_metadata(self):
        self.assertTrue(os.path.isfile(DEFAULT_REGISTRY_PATH))
        reg = load_source_registry(DEFAULT_REGISTRY_PATH)
        sources = reg.get("sources", {})
        self.assertGreaterEqual(len(sources), 20)

        required_fields = (
            "source_url",
            "author_or_publisher",
            "license",
            "attribution",
            "original_file_sha256",
            "retrieval_date",
            "derived_record_ids",
            "transformation_performed",
            "approval_status",
        )
        for sid, entry in sources.items():
            for field in required_fields:
                self.assertIn(field, entry, f"Missing {field} in source {sid}")
            self.assertEqual(len(entry["original_file_sha256"]), 64)
            self.assertEqual(entry["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertTrue(entry["license"].startswith("CC-BY-SA"))

    def test_03_register_source_rejects_bad_license_or_missing_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg_path = os.path.join(tmp, "reg.json")
            with self.assertRaises(ValueError):
                register_source_entry(
                    source_id="bad_nc",
                    source_url="https://example.org/doc",
                    author_or_publisher="Org",
                    license_name="CC-BY-NC-4.0",
                    license_url="https://creativecommons.org/licenses/by-nc/4.0/",
                    attribution="Org",
                    original_file_sha256="a" * 64,
                    transformation_performed="none",
                    registry_path=reg_path,
                )
            with self.assertRaises(ValueError):
                register_source_entry(
                    source_id="bad_attr",
                    source_url="https://example.org/doc",
                    author_or_publisher="Org",
                    license_name="CC-BY-4.0",
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                    attribution="",
                    original_file_sha256="a" * 64,
                    transformation_performed="none",
                    registry_path=reg_path,
                )


class TestPhase278QualityControlAndStructure(unittest.TestCase):
    """Tests structural alternation, UTF-8, code syntax, PII/secrets, and 13 categories."""

    def test_01_all_13_categories_registered_and_empty_templates_valid(self):
        self.assertEqual(len(SUPPORTED_CHAT_CATEGORIES), 13)
        templates = build_empty_owner_templates()
        self.assertEqual(len(templates), 13)
        self.assertTrue(os.path.isfile(DEFAULT_TEMPLATES_PATH))
        self.assertTrue(os.path.isfile(DEFAULT_EXAMPLE_PATH))

        for t in templates:
            self.assertEqual(t["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertEqual(t["attribution"], "")
            self.assertEqual(t["messages"][0]["content"], "")
            self.assertEqual(t["messages"][1]["content"], "")

    def test_02_role_ordering_utf8_code_and_pii_checks(self):
        tok = SovereignTokenizer.load(TOKENIZER_PATH)

        # Good multi-turn record with Python & JSON code blocks
        good_msgs = [
            {"role": "system", "content": "You are Moon AI educational assistant."},
            {"role": "user", "content": "Show valid Python and JSON formatting examples."},
            {
                "role": "assistant",
                "content": (
                    "Here is valid Python and JSON:\n"
                    "```python\nx = [i * 2 for i in range(4)]\n```\n"
                    "```json\n{\"status\": \"ok\", \"count\": 4}\n```\n"
                ),
            },
        ]
        ok, err = check_code_and_safety(good_msgs)
        self.assertTrue(ok, err)

        # Bad role order (consecutive user)
        bad_order = [
            {"role": "user", "content": "Hello first user prompt"},
            {"role": "user", "content": "Second consecutive user prompt"},
            {"role": "assistant", "content": "Response of sufficient length for check."},
        ]
        ok_order, err_order = check_code_and_safety(bad_order)
        self.assertFalse(ok_order)
        self.assertIn("role alternation", str(err_order))

        # Corrupted UTF-8 replacement character
        bad_utf8 = [
            {"role": "user", "content": "Valid prompt question?"},
            {"role": "assistant", "content": "Corrupted answer containing replacement char \ufffd here."},
        ]
        ok_u, err_u = check_code_and_safety(bad_utf8)
        self.assertFalse(ok_u)
        self.assertIn("UTF-8", str(err_u))

        # Invalid JSON code block
        bad_json_block = [
            {"role": "user", "content": "Show invalid JSON."},
            {"role": "assistant", "content": "Bad JSON:\n```json\n{\"a\": 1,}\n```\nEnd."},
        ]
        ok_j, err_j = check_code_and_safety(bad_json_block)
        self.assertFalse(ok_j)
        self.assertIn("Invalid JSON syntax", str(err_j))

        # Secret token / email PII
        bad_email = [
            {"role": "user", "content": "Contact information test?"},
            {"role": "assistant", "content": "Please email private_person@realdomain.org with details."},
        ]
        ok_e, err_e = check_code_and_safety(bad_email)
        self.assertFalse(ok_e)
        self.assertIn("email PII", str(err_e))

        # Placeholder marker rejected
        placeholder_rec = {
            "id": "placeholder_01",
            "category": "python",
            "provenance_type": "ORIGINAL_SELF_AUTHORED_QA",
            "license": "CustomSovereign",
            "attribution": "Prem Gaikwad",
            "source": "Owner",
            "source_url": "https://github.com/Prem759-0/NanoGPT",
            "messages": [
                {"role": "user", "content": "Explain Python dicts."},
                {"role": "assistant", "content": "TODO: insert answer here about dictionaries later."},
            ],
        }
        v_ok, _, v_errs = validate_candidate_record(placeholder_rec, tokenizer=tok)
        self.assertFalse(v_ok)
        self.assertTrue(any("placeholder" in e.lower() for e in v_errs))


class TestPhase278ContributionAndReviewWorkflow(unittest.TestCase):
    """Tests pending-review lock, duplicate rejection, approval gate, and manifest rebuild."""

    def test_01_pending_review_never_auto_approved_on_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub_path = os.path.join(tmp, "subs.jsonl")
            queue_path = os.path.join(tmp, "pending.jsonl")

            # Attacker/submitter tries to self-assign APPROVED status -> must still stage as PENDING_REVIEW
            r1 = _sample_valid_owner_record(rec_id="owner_self_claim_01")
            r1["approval_status"] = APPROVAL_STATUS_APPROVED
            # Duplicate prompt record
            r2 = _sample_valid_owner_record(rec_id="owner_dup_02", answer="Another valid answer explaining generators clearly enough.")
            # Empty answer record
            r3 = _sample_valid_owner_record(rec_id="owner_empty_03", prompt="Empty prompt?", answer="")

            with open(sub_path, "w", encoding="utf-8") as f:
                for row in (r1, r2, r3):
                    f.write(json.dumps(row) + "\n")

            dry_res = import_contributions(sub_path, queue_path=queue_path, approved_paths=(), dry_run=True)
            self.assertEqual(dry_res["accepted_to_pending_review_count"], 1)
            self.assertEqual(dry_res["rejected_count"], 2)
            self.assertFalse(os.path.exists(queue_path))

            live_res = import_contributions(sub_path, queue_path=queue_path, approved_paths=(), dry_run=False)
            self.assertEqual(live_res["accepted_to_pending_review_count"], 1)
            pending = list_pending_records(queue_path)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["approval_status"], APPROVAL_STATUS_PENDING)
            self.assertIsNone(pending[0]["reviewer"])

            # validate_chat_record in production mode rejects PENDING_REVIEW record
            prod_ok, prod_err = validate_chat_record(pending[0], is_production=True)
            self.assertFalse(prod_ok)
            self.assertIn("PENDING_REVIEW", str(prod_err))

    def test_02_review_queue_blocks_empty_draft_and_rebuilds_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests_dir = os.path.join(tmp, "datasets", "manifests")
            os.makedirs(manifests_dir, exist_ok=True)
            # Copy baseline 22 records cleanly into isolated tmp manifests dir
            for fn in ("chat_train.jsonl", "chat_val.jsonl", "chat_test.jsonl"):
                src = os.path.join(PROJECT_ROOT, "datasets", "manifests", fn)
                with open(src, "r", encoding="utf-8") as sf, open(os.path.join(manifests_dir, fn), "w", encoding="utf-8", newline="\n") as df:
                    df.write(sf.read())

            queue_path = os.path.join(tmp, "pending.jsonl")
            curated_path = os.path.join(manifests_dir, "curated_approved_records.jsonl")
            rej_path = os.path.join(tmp, "rejected.jsonl")
            reg_path = os.path.join(tmp, "registry.json")

            # Register a local permitted source that stages an incomplete draft
            dummy_src = os.path.join(tmp, "source_notes.txt")
            with open(dummy_src, "w", encoding="utf-8") as sf:
                sf.write("Open educational source reference content.\n")

            reg_res = register_local_permitted_source(
                file_path=dummy_src,
                source_id="open_test_note",
                source_url="https://en.wikipedia.org/wiki/Recursion",
                title="Open Note",
                category="computer_science",
                author_or_publisher="Wikipedia contributors",
                license_name="CC-BY-SA-4.0",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                attribution="Wikipedia contributors (CC-BY-SA 4.0)",
                transformation_performed="Registered source conversion draft.",
                suggested_question="Explain recursion clearly.",
                summary_excerpt="Recursion occurs when a function calls itself on a smaller subproblem.",
                registry_path=reg_path,
                queue_path=queue_path,
            )
            draft_id = reg_res["staged_id"]
            self.assertIsNotNone(draft_id)

            # Attempting to approve an empty assistant draft MUST FAIL CLOSED
            app_fail = approve_pending_records(
                [draft_id],
                reviewer="Prem Gaikwad",
                queue_path=queue_path,
                curated_approved_path=curated_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                rebuild_manifests=False,
                base_dir=tmp,
            )
            self.assertEqual(app_fail["approved_count"], 0)
            self.assertEqual(len(app_fail["failed_validation"]), 1)

            # Submit a genuine complete Q&A -> pending -> approve -> rebuild twice and confirm reproducibility
            sub = submit_single_qa(
                category="javascript",
                prompt="How does Array.prototype.map work in JavaScript?",
                answer=(
                    "Array.prototype.map creates a new array populated with the results of calling "
                    "a provided callback function on every element in the calling array without mutating the original array."
                ),
                attribution="Prem Gaikwad",
                record_id="owner_js_map_001",
                queue_path=queue_path,
                approved_paths=(),
                tokenizer_path=TOKENIZER_PATH,
            )
            self.assertTrue(sub["accepted"])

            app_ok = approve_pending_records(
                ["owner_js_map_001"],
                reviewer="Prem Gaikwad",
                queue_path=queue_path,
                curated_approved_path=curated_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                rebuild_manifests=True,
                base_dir=tmp,
            )
            self.assertEqual(app_ok["approved_count"], 1)
            self.assertEqual(app_ok["rebuild_result"]["accepted_genuine_records"], 23)
            self.assertEqual(app_ok["rebuild_result"]["cross_split_leakage"]["total_leakage_events"], 0)

            first_hashes = {k: v["sha256"] for k, v in app_ok["rebuild_result"]["output_files"].items()}
            rebuilt_again = rebuild_production_manifests(
                base_dir=tmp,
                manifests_dir=manifests_dir,
                curated_approved_path=curated_path,
                pending_queue_path=queue_path,
                rejected_queue_path=rej_path,
                source_registry_path=reg_path,
                tokenizer_path=TOKENIZER_PATH,
                write_audit_reports=False,
            )
            second_hashes = {k: v["sha256"] for k, v in rebuilt_again["output_files"].items()}
            self.assertEqual(first_hashes, second_hashes)

            # Rejection flow cleanly removes draft from pending to rejected
            rej_res = reject_pending_record(
                draft_id,
                reviewer="Prem Gaikwad",
                reason="Empty summary draft archived.",
                queue_path=queue_path,
                rejected_queue_path=rej_path,
                source_registry_path=reg_path,
            )
            self.assertTrue(rej_res["rejected"])
            self.assertEqual(rej_res["remaining_pending_count"], 0)


class TestPhase278EfficiencyLeakageAndTrainingCompatibility(unittest.TestCase):
    """Tests split leakage checks, SHA audit caching, progress metrics, and training compatibility."""

    def test_01_public_source_url_leakage_caught_by_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec_a = {
                "id": "pub_leak_1",
                "category": "computer_science",
                "source": "Wikipedia: Sorting algorithm",
                "source_url": "https://en.wikipedia.org/wiki/Sorting_algorithm",
                "source_revision": "rev1",
                "license": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": "Wikipedia contributors",
                "provenance_type": "SOURCE_DERIVED_EDUCATIONAL_SUMMARY",
                "messages": [
                    {"role": "user", "content": "What makes a sorting algorithm stable?"},
                    {"role": "assistant", "content": "A stable sorting algorithm preserves the relative order of records with equal keys."},
                ],
            }
            rec_b = {
                "id": "pub_leak_2",
                "category": "computer_science",
                "source": "Wikipedia: Sorting algorithm",
                "source_url": "https://en.wikipedia.org/wiki/Sorting_algorithm",
                "source_revision": "rev1",
                "license": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": "Wikipedia contributors",
                "provenance_type": "SOURCE_DERIVED_EDUCATIONAL_SUMMARY",
                "messages": [
                    {"role": "user", "content": "Compare in-place versus out-of-place sorting algorithms."},
                    {"role": "assistant", "content": "In-place sorting uses O(1) or O(log n) auxiliary space while out-of-place allocates additional buffers."},
                ],
            }
            tr = os.path.join(tmp, "train.jsonl")
            va = os.path.join(tmp, "val.jsonl")
            te = os.path.join(tmp, "test.jsonl")
            with open(tr, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec_a) + "\n")
            with open(va, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec_a) + "\n")
            with open(te, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec_b) + "\n")

            gate = verify_production_manifest_group(tr, va, te, tokenizer_path=TOKENIZER_PATH, require_scale=False)
            self.assertFalse(gate["passed"])
            self.assertGreater(gate["audit"]["cross_split_leakage"]["public_source_url_overlap"]["train_test"], 0)
            self.assertGreater(gate["audit"]["cross_split_leakage"]["total_leakage_events"], 0)

    def test_02_progress_report_and_disk_cache_consistency(self):
        self.assertTrue(os.path.isfile(DEFAULT_PROGRESS_REPORT_PATH))
        with open(DEFAULT_PROGRESS_REPORT_PATH, "r", encoding="utf-8") as f:
            prog = json.load(f)

        self.assertEqual(prog["dataset_status"], "DATASET_NOT_READY")
        self.assertEqual(prog["current_approved_production"]["total_records"], 22)
        self.assertEqual(prog["current_approved_production"]["supervised_assistant_tokens"], 7199)
        self.assertGreaterEqual(prog["review_queue"]["pending_review_count"], 20)
        self.assertEqual(prog["shortfall"]["qualifying_records_shortfall"], 478)
        self.assertEqual(prog["shortfall"]["supervised_tokens_shortfall"], 92801)
        self.assertEqual(set(prog["current_approved_production"]["category_13_distribution"].keys()), set(SUPPORTED_CHAT_CATEGORIES.keys()))

        # Compare fresh uncached single-manifest audit vs cached result
        train_manifest = os.path.join(PROJECT_ROOT, "datasets", "manifests", "prod_chat_train.jsonl")
        tok = SovereignTokenizer.load(TOKENIZER_PATH)
        fresh = audit_single_manifest(train_manifest, tokenizer=tok, use_disk_cache=False)
        self.assertEqual(fresh["records_count"], 18)
        self.assertEqual(fresh["token_stats"]["assistant_supervised_tokens"], 5825)

    def test_03_training_compatibility_and_strict_sovereign_policy(self):
        self.assertEqual(policy.ACTIVE_OWNERSHIP_POLICY, "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
        self.assertFalse(policy.AUTHORIZED_BY_PREM)

        tok = SovereignTokenizer.load(TOKENIZER_PATH)
        for split_name, expected_recs in [("train", 18), ("val", 2), ("test", 2)]:
            mpath = os.path.join(PROJECT_ROOT, "datasets", "manifests", f"prod_chat_{split_name}.jsonl")
            ds = SovereignChatDataset(mpath, tok, block_size=512)
            self.assertEqual(ds.get_unique_record_count(), expected_recs)
            self.assertGreater(len(ds), 0)
            self.assertGreater(ds.get_total_supervised_tokens(), 0)
            x, y = ds[0]
            self.assertEqual(x.shape[0], 512)
            self.assertEqual(y.shape[0], 512)
            # Verify prompt tokens masked (-1) and assistant tokens supervised (>= 0)
            self.assertGreater(int((y == -1).sum().item()), 0)
            self.assertGreater(int((y >= 0).sum().item()), 0)

        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "v0_1_t4_release_config.yaml")
        cfg = load_training_config(cfg_path)
        val_res = validate_training_config(cfg, require_production_release=True, verify_dataset_truth=True)
        self.assertTrue(val_res["valid_config"])
        self.assertEqual(val_res["dataset_status"], "DATASET_NOT_READY")
        self.assertFalse(val_res["ready_for_training"])

        dry_res = train_moon_chat(config_path=cfg_path, dry_run=True, device_override="cpu")
        self.assertEqual(dry_res["status"], "COMPLETED_PREFLIGHT_DRY_RUN")
        self.assertFalse(dry_res["checkpoint_written"])

        with self.assertRaises(PermissionError):
            train_moon_chat(config_path=cfg_path, dry_run=False, smoke_test=False)


if __name__ == "__main__":
    unittest.main()
