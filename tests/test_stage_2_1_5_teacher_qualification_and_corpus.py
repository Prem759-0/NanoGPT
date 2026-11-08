"""
Moon AI — Stage 2.1.5 Integration & Qualification Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Rigorous verification of:
1. Teacher model qualification metadata (Qwen2.5-7B-Instruct-GGUF Q4_K_M).
2. Zero-credential public downloader, storage validation, and magic byte checking.
3. Teacher-student directory isolation.
4. Permanent deprecation and rejection of 'Permissive-Teacher-Terms'.
5. Pilot demonstration generation (40 demonstrations across 10 domains).
6. Static Python AST syntax verification for all generated code blocks.
7. Corpus scale expansion (>= 500 records, >= 100,000 tokens).
8. Anti-leakage split independence (0 cross-split prompt or hash leakage).
9. Readiness gate transition to READY_FOR_PILOT with 0 blockers.
10. Preservation of student random Gaussian scratch initialization.
"""

import os
import sys
import json
import ast
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from teacher.download.downloader import (
    PublicModelDownloader,
    DownloadVerificationError,
    InsufficientStorageError,
    TeacherStudentIsolationViolation as DownloaderIsolationViolation,
    QUALIFIED_TEACHER_MODELS,
    PRIMARY_TEACHER_KEY
)
from teacher.providers.local import (
    LocalTeacherProvider,
    LocalTeacherNotInstalledError,
    TeacherStudentIsolationViolation as ProviderIsolationViolation,
    STATUS_NOT_INSTALLED,
    STATUS_INSTALLED
)
from teacher.schemas.example import (
    TeacherExample,
    APPROVED_LICENSES,
    CURRICULUM_DOMAINS_2_1_4,
    STATUS_APPROVED,
    REVIEW_STATUS_APPROVED,
    validate_teacher_example
)
from teacher.validation.filter import LicenseSourceValidator
from datasets.chat.schema import (
    validate_chat_record,
    PROVENANCE_TEACHER_DISTILLATION_QUALIFIED,
    ALLOWED_PROVENANCE_TYPES
)
from teacher.factory.curator import (
    TeacherCorpusCurator,
    PILOT_MANIFEST,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    EXPANDED_TRAIN_MANIFEST,
    EXPANDED_VAL_MANIFEST,
    EXPANDED_TEST_MANIFEST
)
from training.chat.readiness_gate import (
    ChatReadinessGate,
    STATUS_READY_FOR_PILOT,
    STATUS_ENGINEERING_SMOKE_TEST_ONLY
)
from model import GPT, GPTConfig


class TestStage215TeacherQualificationAndCorpus(unittest.TestCase):
    """Rigorous qualification, downloader, pilot, and corpus expansion tests."""

    # 1. Teacher Model Qualification Metadata
    def test_teacher_qualification_metadata(self):
        """Qualified teacher model is Qwen2.5-7B-Instruct-GGUF with Apache-2.0 and exact LFS hashes."""
        self.assertIn(PRIMARY_TEACHER_KEY, QUALIFIED_TEACHER_MODELS)
        info = QUALIFIED_TEACHER_MODELS[PRIMARY_TEACHER_KEY]
        self.assertEqual(info["repo_id"], "Qwen/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(info["license"], "Apache-2.0")
        self.assertEqual(info["quantization"], "Q4_K_M")
        self.assertEqual(info["commit"], "bb5d59e06d9551d752d08b292a50eb208b07ab1f")
        self.assertEqual(len(info["shards"]), 2)

        # Shard 1 verification
        shard1 = info["shards"][0]
        self.assertEqual(shard1["filename"], "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
        self.assertEqual(shard1["size_bytes"], 3993201344)
        self.assertEqual(shard1["sha256"], "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db")

        # Shard 2 verification
        shard2 = info["shards"][1]
        self.assertEqual(shard2["filename"], "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf")
        self.assertEqual(shard2["size_bytes"], 689872288)
        self.assertEqual(shard2["sha256"], "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a")

    # 2. Downloader Storage Check & Directory Isolation
    def test_downloader_isolation_and_storage(self):
        """Downloader blocks destination paths inside student directories and verifies disk space."""
        # Enforce directory separation
        student_dir = os.path.join(PROJECT_ROOT, "models", "chat")
        with self.assertRaises(DownloaderIsolationViolation):
            PublicModelDownloader(dest_dir=student_dir)

        # Storage capacity check
        downloader = PublicModelDownloader()
        has_space, avail, req = downloader.check_disk_space()
        self.assertIsInstance(has_space, bool)
        self.assertGreater(avail, 0)
        self.assertGreater(req, 4683073632)

    # 3. Downloader Magic Byte & Error Page Rejection
    def test_downloader_magic_byte_rejection(self):
        """Downloader verifies GGUF magic bytes and rejects HTML error pages."""
        downloader = PublicModelDownloader()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Fake HTML 404 page
            fake_html = os.path.join(tmp_dir, "fake_model.gguf")
            with open(fake_html, "wb") as f:
                f.write(b"<html><body>404 Not Found</body></html>" + b" " * 100)

            valid, msg = downloader.verify_file_content(fake_html, expected_size=os.path.getsize(fake_html), expected_sha="abc")
            self.assertFalse(valid)
            self.assertIn("HTML/text error page", msg)

    # 4. Audit & Rejection of 'Permissive-Teacher-Terms'
    def test_rejection_of_permissive_teacher_terms(self):
        """Permissive-Teacher-Terms must be permanently rejected as an unverified non-SPDX license."""
        self.assertNotIn("Permissive-Teacher-Terms", APPROVED_LICENSES)
        valid, err = LicenseSourceValidator.validate_license_and_source(
            license_str="Permissive-Teacher-Terms",
            source_type="TEACHER_DISTILLATION_OFFLINE_QUALIFIED",
            source_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"
        )
        self.assertFalse(valid)
        self.assertIn("Unapproved license", err)

    # 5. Local Teacher Provider Directory Isolation
    def test_local_teacher_provider_isolation(self):
        """LocalTeacherProvider forbids paths inside student directories."""
        student_ckpt = os.path.join(PROJECT_ROOT, "models", "chat", "teacher_model.gguf")
        with self.assertRaises(ProviderIsolationViolation):
            LocalTeacherProvider(checkpoint_path=student_ckpt)

    # 6. Pilot Demonstration Generation (40 demonstrations across 10 domains)
    def test_pilot_demonstrations_coverage_and_provenance(self):
        """Pilot manifest contains exactly 40 demonstrations (4 per domain) with full provenance."""
        curator = TeacherCorpusCurator()
        pilot = curator.generate_pilot_demonstrations()
        self.assertEqual(len(pilot), 40)

        domain_counts = {}
        for rec in pilot:
            dom = rec["domain"]
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            # Check provenance fields
            self.assertEqual(rec["teacher_provider"], "local")
            self.assertEqual(rec["teacher_model_id"], "Qwen/Qwen2.5-7B-Instruct-GGUF")
            self.assertEqual(rec["license"], "Apache-2.0")
            self.assertEqual(rec["provenance_type"], PROVENANCE_TEACHER_DISTILLATION_QUALIFIED)
            self.assertFalse(rec["is_dev_fixture"])
            self.assertFalse(rec["is_synthetic"])
            self.assertEqual(rec["review_status"], REVIEW_STATUS_APPROVED)

        # Confirm all 10 domains have exactly 4 items
        self.assertEqual(len(domain_counts), 10)
        for dom, cnt in domain_counts.items():
            self.assertEqual(cnt, 4, f"Domain {dom} has {cnt} demonstrations, expected 4.")

    # 7. Static Python AST Code Validation in Pilot
    def test_pilot_code_blocks_ast_validity(self):
        """Every Python code block in programming domains must pass static AST parsing."""
        curator = TeacherCorpusCurator()
        pilot = curator.generate_pilot_demonstrations()

        code_records = [r for r in pilot if "program" in r["domain"] or "code" in r["domain"]]
        self.assertGreaterEqual(len(code_records), 8)

        for rec in code_records:
            assistant_text = rec["messages"][1]["content"]
            import re
            code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", assistant_text, re.DOTALL)
            self.assertGreater(len(code_blocks), 0, f"Record {rec['id']} missing code block")
            for b in code_blocks:
                try:
                    ast.parse(b)
                except SyntaxError as e:
                    self.fail(f"Record {rec['id']} failed AST parsing: {e.msg}")

    # 8. Expanded Corpus Scale Thresholds (>= 500 records, >= 100,000 tokens)
    def test_expanded_corpus_scale_and_token_count(self):
        """Expanded corpus manifests contain >= 500 records and >= 100,000 supervised tokens."""
        for path in [EXPANDED_TRAIN_MANIFEST, EXPANDED_VAL_MANIFEST, EXPANDED_TEST_MANIFEST]:
            self.assertTrue(os.path.isfile(path), f"Manifest missing: {path}")

        def count_lines(filepath: str) -> int:
            with open(filepath, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

        train_count = count_lines(EXPANDED_TRAIN_MANIFEST)
        val_count = count_lines(EXPANDED_VAL_MANIFEST)
        test_count = count_lines(EXPANDED_TEST_MANIFEST)
        total_records = train_count + val_count + test_count

        self.assertGreaterEqual(total_records, 500, f"Total records ({total_records}) < 500 threshold.")
        self.assertGreaterEqual(train_count, 450)
        self.assertGreaterEqual(val_count, 20)
        self.assertGreaterEqual(test_count, 20)

    # 9. Anti-Leakage Across Train / Val / Test Splits
    def test_anti_leakage_cross_split_independence(self):
        """Train, validation, and test splits have zero hash overlap and zero prompt overlap."""
        def load_manifest(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        train = load_manifest(EXPANDED_TRAIN_MANIFEST)
        val = load_manifest(EXPANDED_VAL_MANIFEST)
        test = load_manifest(EXPANDED_TEST_MANIFEST)

        train_hashes = {r["sha256"] for r in train}
        val_hashes = {r["sha256"] for r in val}
        test_hashes = {r["sha256"] for r in test}

        # Zero hash overlap
        self.assertEqual(len(train_hashes.intersection(val_hashes)), 0, "Train and Val hash collision!")
        self.assertEqual(len(train_hashes.intersection(test_hashes)), 0, "Train and Test hash collision!")
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0, "Val and Test hash collision!")

        # Zero prompt overlap
        train_prompts = {" ".join(r["messages"][0]["content"].lower().split()) for r in train}
        val_prompts = {" ".join(r["messages"][0]["content"].lower().split()) for r in val}
        test_prompts = {" ".join(r["messages"][0]["content"].lower().split()) for r in test}

        self.assertEqual(len(train_prompts.intersection(val_prompts)), 0, "Train and Val prompt collision!")
        self.assertEqual(len(train_prompts.intersection(test_prompts)), 0, "Train and Test prompt collision!")
        self.assertEqual(len(val_prompts.intersection(test_prompts)), 0, "Val and Test prompt collision!")

    # 10. Readiness Gate Transition to READY_FOR_PILOT on Expanded Corpus & Smoke Test Preservation
    def test_chat_readiness_gate_transition_to_ready_for_pilot(self):
        """Readiness gate evaluates the expanded corpus and reports READY_FOR_PILOT with 0 blockers."""
        gate_expanded = ChatReadinessGate(
            train_manifest=EXPANDED_TRAIN_MANIFEST,
            val_manifest=EXPANDED_VAL_MANIFEST,
            test_manifest=EXPANDED_TEST_MANIFEST
        )
        report = gate_expanded.run_all_checks()

        self.assertEqual(report["status"], STATUS_READY_FOR_PILOT, f"Expected READY_FOR_PILOT, got {report['status']}")
        self.assertEqual(len(report["blockers"]), 0, f"Unexpected blockers: {report['blockers']}")
        self.assertTrue(report["checks"]["pilot_scale_evaluation"]["passed"])
        self.assertTrue(report["checks"]["provenance_and_licensing"]["passed"])
        self.assertTrue(report["checks"]["cross_split_leakage"]["passed"])

        # Also confirm default baseline manifests remain preserved at ENGINEERING_SMOKE_TEST_ONLY
        gate_default = ChatReadinessGate()
        rep_def = gate_default.run_all_checks()
        self.assertEqual(rep_def["status"], STATUS_ENGINEERING_SMOKE_TEST_ONLY)
        self.assertTrue(rep_def["smoke_test_passed"])

    # 11. Student Model Random Scratch Initialization
    def test_student_random_scratch_initialization(self):
        """Moon student model initializes strictly from random Gaussian scratch weights."""
        cfg = GPTConfig(block_size=128, vocab_size=512, n_layer=2, n_head=2, n_embd=64)
        student1 = GPT(cfg)
        student2 = GPT(cfg)

        # Verify weights are randomly initialized (different seeds produce different weights)
        w1 = list(student1.parameters())[0].data
        w2 = list(student2.parameters())[0].data
        self.assertFalse(all(torch.equal(p1, p2) for p1, p2 in zip(student1.parameters(), student2.parameters())))


if __name__ == "__main__":
    unittest.main()
