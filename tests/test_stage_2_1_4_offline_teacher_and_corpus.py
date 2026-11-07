"""
Moon AI — Stage 2.1.4 Offline Teacher & Training Corpus Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Comprehensive tests validating:
1. 100% offline-only execution (zero API keys or network dependencies).
2. Mock versus production data separation (mock data permanently marked as dev fixture).
3. Local teacher not installed handling (LOCAL_TEACHER_NOT_INSTALLED, fails closed, zero silent mock fallback).
4. Local teacher installed execution (detects local file, computes SHA-256).
5. Teacher / student weight and directory isolation.
6. Random student scratch initialization (Gaussian scratch, zero teacher weights).
7. Complete 10 curriculum domains recognition and distribution tracking.
8. Sensitive data filter (PII, tokens, private keys, API keys).
9. Safe code syntax validation via AST parsing.
10. Exact and near-duplicate detection (exact hash + 3-gram Jaccard).
11. Blind test benchmark leakage prevention.
12. Approval gate workflow (PENDING_REVIEW -> APPROVED required for production).
13. Readiness gate scale preservation (truthfully reports 22 records, blocks pilot).
14. Local teacher hardware compatibility report verification.
"""

import os
import sys
import json
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT
from core.policy import (
    validate_student_initialization,
    ACTIVE_OWNERSHIP_POLICY,
    POLICY_TEACHER_STUDENT_RANDOM_INIT
)
from teacher.providers.mock import MockTeacherProvider
from teacher.providers.local import (
    LocalTeacherProvider,
    LocalTeacherNotInstalledError,
    TeacherStudentIsolationViolation,
    STATUS_NOT_INSTALLED,
    STATUS_INSTALLED
)
from teacher.schemas.example import (
    TeacherExample,
    CURRICULUM_DOMAINS_2_1_4,
    ALL_CURRICULUM_DOMAINS,
    VALID_DOMAINS,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    validate_teacher_example
)
from teacher.validation.filter import (
    TeacherDataPipeline,
    SensitiveDataFilter,
    PromptInjectionFilter,
    DuplicateDetector,
    QualityValidator,
    CodeSyntaxValidator,
    LicenseSourceValidator,
    LeakageDetector,
    QuarantineManager,
    CurriculumDistributionTracker
)
from teacher.factory.dataset_factory import (
    OfflineDatasetFactory,
    ApprovalGate
)
from training.chat.readiness_gate import ChatReadinessGate


class TestStage214OfflineTeacherAndCorpus(unittest.TestCase):
    """Rigorous offline dataset factory and local teacher verification."""

    # 1. Offline-Only Execution (Zero API Keys)
    def test_offline_only_execution(self):
        """Pipeline must function 100% offline without any API keys or internet calls."""
        orig_kimi = os.environ.pop("KIMI_API_KEY", None)
        orig_openai = os.environ.pop("OPENAI_API_KEY", None)
        orig_ngrok = os.environ.pop("NGROK_AUTHTOKEN", None)

        try:
            mock = MockTeacherProvider()
            pipeline = TeacherDataPipeline()
            factory = OfflineDatasetFactory(provider=mock, pipeline=pipeline)

            passed, record = factory.generate_domain_record(
                domain="general_conversation",
                prompt="What is the speed of light in vacuum?",
                license_str="CC-BY-4.0"
            )
            # Record generated offline
            self.assertIsNotNone(record)
            self.assertIn("speed of light", record.response.lower() + record.prompt.lower())
        finally:
            if orig_kimi: os.environ["KIMI_API_KEY"] = orig_kimi
            if orig_openai: os.environ["OPENAI_API_KEY"] = orig_openai
            if orig_ngrok: os.environ["NGROK_AUTHTOKEN"] = orig_ngrok

    # 2. Mock Versus Production Data Separation
    def test_mock_versus_production_data_separation(self):
        """Records created by MockTeacherProvider must be marked as dev fixtures and rejected from production."""
        mock = MockTeacherProvider()
        factory = OfflineDatasetFactory(provider=mock)

        passed, example = factory.generate_domain_record(
            domain="instruction_following",
            prompt="Summarize the theory of special relativity."
        )

        # Mock generation must be quarantined as a dev fixture
        self.assertFalse(passed)
        self.assertTrue(example.is_dev_fixture)
        self.assertTrue(example.is_synthetic)
        self.assertEqual(example.validation_status, STATUS_QUARANTINED)

        # Production validator must reject it
        is_prod_valid, err = validate_teacher_example(example.to_dict(), is_production=True)
        self.assertFalse(is_prod_valid)
        self.assertIn("is_dev_fixture=True", err)

    # 3. Local Teacher Not Installed Handling
    def test_local_teacher_not_installed(self):
        """When local teacher checkpoint is absent, returns LOCAL_TEACHER_NOT_INSTALLED and fails closed."""
        non_existent_path = os.path.join(PROJECT_ROOT, "models", "teachers", "checkpoints", "missing_teacher.gguf")
        provider = LocalTeacherProvider(checkpoint_path=non_existent_path)

        # Status check
        status_report = provider.check_installation()
        self.assertEqual(status_report["status"], STATUS_NOT_INSTALLED)
        self.assertFalse(status_report["installed"])

        # Generation must raise LocalTeacherNotInstalledError (never fall back to mock)
        with self.assertRaises(LocalTeacherNotInstalledError) as ctx:
            provider.generate_completion("Test prompt")
        self.assertIn("LOCAL_TEACHER_NOT_INSTALLED", str(ctx.exception))

        # Factory using uninstalled local teacher must also fail closed
        factory = OfflineDatasetFactory(provider=provider)
        with self.assertRaises(LocalTeacherNotInstalledError):
            factory.generate_domain_record(domain="math_logic_reasoning", prompt="2 + 2")

    # 4. Local Teacher Installed Execution
    def test_local_teacher_installed_execution(self):
        """When local teacher checkpoint exists on disk, local provider detects it and computes SHA-256."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_ckpt = os.path.join(tmp_dir, "test_teacher.gguf")
            # Write dummy weights (>1024 bytes)
            with open(fake_ckpt, "wb") as f:
                f.write(b"GGUF_TEST_HEADER" + b"\x00" * 2048)

            provider = LocalTeacherProvider(checkpoint_path=fake_ckpt)
            self.assertTrue(provider.is_installed())

            status = provider.check_installation()
            self.assertEqual(status["status"], STATUS_INSTALLED)
            self.assertTrue(status["installed"])
            self.assertNotEqual(status["checkpoint_sha256"], "none")
            self.assertEqual(len(status["checkpoint_sha256"]), 64)

            # Execution works without network
            res = provider.generate_completion("What is computational complexity?")
            self.assertIn("Local Teacher", res["text"])
            self.assertEqual(res["metadata"]["checkpoint_verified"], True)

    # 5. Teacher / Student Weight and Directory Isolation
    def test_teacher_student_weight_isolation(self):
        """Teacher checkpoint paths are forbidden inside student model directories."""
        student_dir = os.path.join(PROJECT_ROOT, "models", "chat", "teacher.gguf")
        with self.assertRaises(TeacherStudentIsolationViolation):
            LocalTeacherProvider(checkpoint_path=student_dir)

    # 6. Random Student Scratch Initialization
    def test_random_student_scratch_initialization(self):
        """Moon Chat student model initializes purely from random Gaussian scratch weights."""
        cfg = GPTConfig(
            block_size=128,
            vocab_size=512,
            n_layer=4,
            n_head=4,
            n_embd=128,
            bias=False
        )
        student_model = GPT(cfg)
        self.assertEqual(ACTIVE_OWNERSHIP_POLICY, POLICY_TEACHER_STUDENT_RANDOM_INIT)

        valid_init, msg = validate_student_initialization("random_gaussian_scratch")
        self.assertTrue(valid_init)

        # Check weights are non-zero Gaussian
        weight = student_model.transformer.wte.weight
        self.assertAlmostEqual(weight.mean().item(), 0.0, delta=0.05)
        self.assertAlmostEqual(weight.std().item(), 0.02, delta=0.01)

    # 7. Complete 10 Curriculum Domains Recognition
    def test_all_10_curriculum_domains_recognized(self):
        """All 10 curriculum domains from Section 3 are defined and recognized."""
        self.assertEqual(len(CURRICULUM_DOMAINS_2_1_4), 10)
        expected_domains = [
            "general_conversation",
            "instruction_following",
            "math_logic_reasoning",
            "programming_fundamentals",
            "advanced_programming_debugging",
            "software_architecture",
            "computer_science",
            "research_evidence",
            "tool_selection_usage",
            "safety_uncertainty"
        ]
        for dom in expected_domains:
            self.assertIn(dom, CURRICULUM_DOMAINS_2_1_4)
            self.assertIn(dom, ALL_CURRICULUM_DOMAINS)

        # Test distribution tracker
        tracker = CurriculumDistributionTracker()
        for dom in CURRICULUM_DOMAINS_2_1_4:
            tracker.record_example(dom, 100)
        report = tracker.get_distribution_report()
        self.assertEqual(report["domains_covered"], 10)
        self.assertEqual(report["total_records"], 10)
        self.assertEqual(report["total_tokens"], 1000)

    # 8. Sensitive Data Filter
    def test_sensitive_data_filter(self):
        """Detects API keys, private keys, tokens, emails, SSNs, and credit cards."""
        text_with_secret = "Here is my key: sk-abcdef123456789012345678 and token ghp_123456789012345678901234567890123456"
        has_secret, violations = SensitiveDataFilter.check_sensitive_data(text_with_secret)
        self.assertTrue(has_secret)
        self.assertGreaterEqual(len(violations), 2)

        redacted = SensitiveDataFilter.redact_sensitive_data(text_with_secret)
        self.assertNotIn("sk-abcdef", redacted)
        self.assertNotIn("ghp_123456", redacted)

    # 9. Safe Code Syntax Validation via AST
    def test_code_syntax_validation(self):
        """Validates Python syntax in code blocks statically; catches SyntaxError without executing."""
        valid_code_resp = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        ok_valid, err_valid = CodeSyntaxValidator.validate_code_syntax(valid_code_resp)
        self.assertTrue(ok_valid)
        self.assertIsNone(err_valid)

        invalid_code_resp = "Broken code:\n```python\ndef broken(\n    return missing bracket\n```"
        ok_inv, err_inv = CodeSyntaxValidator.validate_code_syntax(invalid_code_resp)
        self.assertFalse(ok_inv)
        self.assertIn("syntax error", err_inv.lower())

    # 10. Exact and Near-Duplicate Detection
    def test_duplicate_detection(self):
        """Catches exact SHA-256 collisions and 3-gram near-duplicates."""
        detector = DuplicateDetector(jaccard_threshold=0.80)

        # First entry
        is_dup, reason = detector.check_and_add("rec_1", "What is photosynthesis?", "Photosynthesis is the process by which green plants convert light into chemical energy.")
        self.assertFalse(is_dup)

        # Exact duplicate
        is_dup, reason = detector.check_and_add("rec_2", "What is photosynthesis?", "Photosynthesis is the process by which green plants convert light into chemical energy.")
        self.assertTrue(is_dup)
        self.assertIn("Exact duplicate", reason)

        # Near duplicate (>80% Jaccard)
        prompt_text = "Explain photosynthesis biological mechanism"
        long_text_1 = (
            "Photosynthesis is a vital biological mechanism utilized by terrestrial plants, green algae, "
            "and certain bacterial species to convert electromagnetic radiation into chemical bond energy stored in sugar molecules."
        )
        long_text_2 = (
            "Photosynthesis is a vital biological mechanism utilized by terrestrial plants, green algae, "
            "and certain bacterial species to transform electromagnetic radiation into chemical bond energy stored in sugar molecules."
        )
        detector.check_and_add("rec_base", prompt_text, long_text_1)
        is_dup_near, reason_near = detector.check_and_add("rec_near", prompt_text, long_text_2)
        self.assertTrue(is_dup_near)
        self.assertIn("Near-duplicate", reason_near)

    # 11. Blind Benchmark Leakage Prevention
    def test_blind_test_leakage_detection(self):
        """Candidate matching a blind benchmark test case is flagged as data leakage."""
        detector = LeakageDetector()
        detector.register_test_example("Benchmark Prompt X", "Benchmark Answer Y")

        is_leak, leak_err = detector.check_leakage("Benchmark Prompt X", "Benchmark Answer Y")
        self.assertTrue(is_leak)
        self.assertIn("Data leakage violation", leak_err)

        is_leak_clean, _ = detector.check_leakage("Clean Prompt", "Clean Answer")
        self.assertFalse(is_leak_clean)

    # 12. Approval Gate Workflow
    def test_approval_gate_workflow(self):
        """Records default to PENDING_REVIEW; require ApprovalGate.review_and_approve for production."""
        ex = TeacherExample.create(
            example_id="rec_review_1",
            teacher_provider="local",
            teacher_model_id="local-teacher-v1",
            prompt="Explain the structure of an operating system kernel.",
            response="A kernel is the core component of an operating system that manages system resources and hardware communication.",
            domain="computer_science",
            task_type="explanation",
            license_str="CC-BY-4.0"
        )
        self.assertEqual(ex.reviewer_status, REVIEW_STATUS_PENDING)

        # Cannot enter production manifest while PENDING
        is_prod, err = validate_teacher_example(ex.to_dict(), is_production=True)
        self.assertFalse(is_prod)
        self.assertIn("PENDING_REVIEW", err)

        # Explicit approval
        ex.validation_status = STATUS_APPROVED
        approved, app_msg = ApprovalGate.review_and_approve(ex, reviewer_id="auditor_42")
        self.assertTrue(approved)
        self.assertEqual(ex.reviewer_status, REVIEW_STATUS_APPROVED)

        # Now passes production validation
        is_prod_now, _ = validate_teacher_example(ex.to_dict(), is_production=True)
        self.assertTrue(is_prod_now)

    # 13. Readiness Gate Domain Coverage & Scale Threshold Preservation
    def test_readiness_gate_domain_coverage_and_scale_threshold(self):
        """Readiness gate tracks domain coverage, preserves >=500 record threshold, reports ENGINEERING_SMOKE_TEST_ONLY."""
        gate = ChatReadinessGate()
        report = gate.run_all_checks()

        # Domain coverage tracking present
        self.assertIn("curriculum_domain_coverage", report["checks"])
        cov = report["checks"]["curriculum_domain_coverage"]
        self.assertGreater(cov["domains_covered_count"], 0)
        self.assertEqual(len(cov["target_curriculum_domains"]), 10)

        # Scale status must remain strictly ENGINEERING_SMOKE_TEST_ONLY (22 records < 500)
        self.assertEqual(report["status"], "ENGINEERING_SMOKE_TEST_ONLY")
        self.assertTrue(report["smoke_test_passed"])
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["blockers"]), 1)
        self.assertIn("Corpus scale", report["blockers"][0])

    # 14. Local Teacher Compatibility Report Verification
    def test_local_teacher_compatibility_report_present(self):
        """Hardware feasibility report exists and covers T4, CPU, GGUF, VRAM, and Kimi 3T evaluation."""
        report_path = os.path.join(PROJECT_ROOT, "docs", "LOCAL_TEACHER_COMPATIBILITY_REPORT.md")
        self.assertTrue(os.path.isfile(report_path), "docs/LOCAL_TEACHER_COMPATIBILITY_REPORT.md not found.")

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("NVIDIA T4", content)
        self.assertIn("15.3 GB", content)
        self.assertIn("Q4_K_M", content)
        self.assertIn("DeepSeek-R1-Distill-Qwen-7B", content)
        self.assertIn("Qwen2.5-Coder-7B-Instruct", content)
        self.assertIn("IMPOSSIBLE", content)  # Explains why 3T Kimi is impossible on T4


if __name__ == "__main__":
    unittest.main()
