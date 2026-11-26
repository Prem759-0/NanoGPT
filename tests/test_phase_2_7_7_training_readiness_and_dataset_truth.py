"""
tests/test_phase_2_7_7_training_readiness_and_dataset_truth.py
Comprehensive Phase 2.7.7 Test Suite for Moon AI.

Verifies:
  1. Dataset truth taxonomy classification (human, public-source, deterministic template, synthetic fixture, AI-generated, unverified)
  2. Dataset truth audit metrics and contamination reporting across all manifests
  3. Genuine ingestion pipeline (reproducibility, AST validation, deduplication, split isolation, provenance manifest)
  4. Tokenizer symmetry and assistant loss masking
  5. Training configuration validator and T4 release configuration (exact 25.7M parameters, VRAM headroom, step budgets, split anti-collision)
  6. Learning rate cosine decay schedule robustness (warmup edge cases, zero division immunity)
  7. Sovereign training locks and dry-run preflight verification
  8. Colab T4 training notebook structure and safety gates
  9. Webapp and model metadata honesty (quarantined external routes, DATASET_NOT_READY, MODEL_NOT_READY)
"""
import copy
import json
import os
import unittest
from typing import Any, Dict

import yaml

import core.policy as policy
from datasets.chat.schema import compute_conversation_sha256
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    ALL_TAXONOMY_CLASSES,
    TAXONOMY_AI_GENERATED,
    TAXONOMY_DETERMINISTIC_TEMPLATE,
    TAXONOMY_HUMAN_AUTHORED_VERIFIED,
    TAXONOMY_PUBLIC_SOURCE_VERIFIED,
    TAXONOMY_SYNTHETIC_FIXTURE,
    TAXONOMY_UNVERIFIED,
    audit_single_manifest,
    classify_chat_record,
    run_full_truth_audit,
    verify_production_manifest_group,
)
from datasets.chat.genuine_ingestion import (
    check_code_and_safety,
    ingest_genuine_sources,
)
from training.chat.config_validator import (
    load_training_config,
    validate_training_config,
)
from training.chat.train import get_lr, train_moon_chat
from webapp.model_service import ModelService

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestPhase277DatasetTruthAndTaxonomy(unittest.TestCase):
    """Verifies fail-closed dataset classification and audit truthfulness."""

    def test_01_classify_human_authored_record(self):
        rec = {
            "id": "moon_human_001",
            "category": "architecture",
            "source": "Moon AI Architecture Whitepaper",
            "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/ARCHITECTURE.md",
            "source_revision": "v1.0",
            "license": "CustomSovereign",
            "attribution": "Prem Gaikwad (Chief ML Systems Architect)",
            "provenance_type": "ORIGINAL_SELF_AUTHORED_QA",
            "messages": [
                {"role": "user", "content": "Explain the sovereign random initialization policy."},
                {"role": "assistant", "content": "Under STRICT_SOVEREIGN_RANDOM_INIT_ONLY, all model weights originate from Gaussian random distributions."},
            ],
        }
        tax, reasons = classify_chat_record(rec)
        self.assertEqual(tax, TAXONOMY_HUMAN_AUTHORED_VERIFIED)

    def test_02_classify_public_source_record(self):
        rec = {
            "id": "moon_public_001",
            "category": "algorithms",
            "source": "Wikipedia: Quicksort",
            "source_url": "https://en.wikipedia.org/wiki/Quicksort",
            "source_revision": "2024 snapshot",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "Wikipedia contributors",
            "provenance_type": "SOURCE_DERIVED_EDUCATIONAL_SUMMARY",
            "messages": [
                {"role": "user", "content": "How does Quicksort choose a pivot element?"},
                {"role": "assistant", "content": "Quicksort can choose a pivot using strategies such as median-of-three or uniform random selection."},
            ],
        }
        tax, reasons = classify_chat_record(rec)
        self.assertEqual(tax, TAXONOMY_PUBLIC_SOURCE_VERIFIED)

    def test_03_classify_deterministic_template_record(self):
        rec = {
            "id": "scaled_gene_0001",
            "source": "Moon AI Deterministic Curriculum Template (Scaled Engine v1)",
            "source_url": "local://scaled_curriculum_generator",
            "license": "Apache-2.0",
            "attribution": "Prem Gaikwad",
            "provenance_type": "DETERMINISTIC_CURRICULUM_TEMPLATE",
            "messages": [
                {"role": "user", "content": "Variant 1: Can you explain the mechanics of photosynthesis?"},
                {"role": "assistant", "content": "Overview of photosynthesis:\n\nKey Takeaway: Understanding these principles provides foundational insight."},
            ],
        }
        tax, reasons = classify_chat_record(rec)
        self.assertEqual(tax, TAXONOMY_DETERMINISTIC_TEMPLATE)

    def test_04_classify_synthetic_fixture_and_ai_generated(self):
        dev_rec = {
            "id": "dev_01",
            "is_dev_fixture": True,
            "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        }
        tax_dev, _ = classify_chat_record(dev_rec)
        self.assertEqual(tax_dev, TAXONOMY_SYNTHETIC_FIXTURE)

        teacher_rec = {
            "id": "teacher_01",
            "teacher_model_id": "Qwen2.5-Coder-7B",
            "provenance_type": "TEACHER_DISTILLATION_OFFLINE_QUALIFIED",
            "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        }
        tax_teacher, _ = classify_chat_record(teacher_rec)
        self.assertEqual(tax_teacher, TAXONOMY_AI_GENERATED)

    def test_05_unverified_record_fails_closed(self):
        bad_rec = {
            "id": "bad_01",
            "license": "UnapprovedLicense",
            "provenance_type": "RANDOM_UNVERIFIED",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        tax, reasons = classify_chat_record(bad_rec)
        self.assertEqual(tax, TAXONOMY_UNVERIFIED)
        self.assertTrue(len(reasons) > 0)

    def test_06_truth_audit_reports_exact_manifest_numbers(self):
        audit = run_full_truth_audit()
        exec_sum = audit["executive_summary"]
        self.assertEqual(exec_sum["dataset_status"], "DATASET_NOT_READY")
        self.assertEqual(exec_sum["true_qualifying_baseline_records"], 22)
        self.assertEqual(exec_sum["true_qualifying_chatml_tokens"], 9445)
        self.assertEqual(exec_sum["true_qualifying_supervised_assistant_tokens"], 7199)

        scaled_group = audit["manifest_groups"]["scaled_pilot"]
        self.assertEqual(scaled_group["total_records"], 1100)
        self.assertEqual(scaled_group["taxonomy_counts"][TAXONOMY_DETERMINISTIC_TEMPLATE], 1100)
        self.assertEqual(scaled_group["unique_first_assistant_replies"], 261)
        self.assertEqual(scaled_group["cross_split_leakage"]["total_leakage_events"], 105)

        expanded_group = audit["manifest_groups"]["expanded_pilot"]
        self.assertEqual(expanded_group["total_records"], 572)
        self.assertEqual(expanded_group["taxonomy_counts"][TAXONOMY_DETERMINISTIC_TEMPLATE], 550)
        self.assertEqual(expanded_group["qualifying_production_genuine_records"], 22)


class TestPhase277GenuineIngestionPipeline(unittest.TestCase):
    """Verifies genuine ingestion, AST code safety, and blind split isolation."""

    def test_01_ast_code_and_pii_check(self):
        safe_msgs = [
            {"role": "user", "content": "How do I calculate factorial in Python?"},
            {"role": "assistant", "content": "Here is a Python function:\n```python\ndef fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n```\nIt is recursive."},
        ]
        ok, err = check_code_and_safety(safe_msgs)
        self.assertTrue(ok)
        self.assertIsNone(err)

        syntax_error_msgs = [
            {"role": "user", "content": "Broken Python code test"},
            {"role": "assistant", "content": "Here is broken syntax:\n```python\ndef fact(n\n    return 1\n```\n"},
        ]
        bad_syntax_ok, bad_syntax_err = check_code_and_safety(syntax_error_msgs)
        self.assertFalse(bad_syntax_ok)
        self.assertIn("Invalid Python code block", str(bad_syntax_err))

        leaked_secret_msgs = [
            {"role": "user", "content": "Test secret leak"},
            {"role": "assistant", "content": "Use your secret key sk-123456789012345678901234 to authenticate."},
        ]
        secret_ok, secret_err = check_code_and_safety(leaked_secret_msgs)
        self.assertFalse(secret_ok)
        self.assertIn("Potential secret or PII", str(secret_err))

    def test_02_production_manifest_group_and_provenance(self):
        prov_path = os.path.join(PROJECT_ROOT, "datasets", "manifests", "prod_chat_provenance.json")
        self.assertTrue(os.path.isfile(prov_path))
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertEqual(prov["dataset_status"], "DATASET_NOT_READY")
        self.assertEqual(prov["accepted_genuine_records"], 22)
        self.assertEqual(prov["rejected_records_count"], 0)
        self.assertEqual(prov["cross_split_leakage"]["total_leakage_events"], 0)
        self.assertEqual(prov["shortfall"]["qualifying_records_shortfall"], 478)
        self.assertEqual(prov["shortfall"]["supervised_tokens_shortfall"], 92801)


class TestPhase277TokenizerSymmetryAndLossMasking(unittest.TestCase):
    """Verifies tokenizer symmetry, merges, and loss masking."""

    def setUp(self):
        tok_path = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
        self.tokenizer = SovereignTokenizer.load(tok_path)

    def test_01_vocab_size_and_merges(self):
        self.assertEqual(self.tokenizer.vocab_size, 512)
        self.assertEqual(len(self.tokenizer.merges), 249)

    def test_02_roundtrip_symmetry(self):
        test_strings = [
            "Hello, Moon AI!",
            "def quicksort(arr): return arr if len(arr) <= 1 else arr",
            "Universal constant: c = 299,792,458 m/s (Refractive index n = c/v)",
            "Prem Gaikwad engineered Moon AI with zero paid APIs.",
        ]
        for s in test_strings:
            encoded = self.tokenizer.encode_text(s)
            decoded = self.tokenizer.decode(encoded)
            self.assertEqual(decoded, s)

    def test_03_encode_chat_loss_masking(self):
        turns = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
        ]
        ids, mask = self.tokenizer.encode_chat_with_loss_mask(turns)
        self.assertEqual(len(ids), len(mask))
        # User prompt tokens must be unmasked (False)
        user_tokens_count = len(self.tokenizer.encode_text("What is Python?"))
        self.assertFalse(all(mask[:user_tokens_count]))
        # Assistant supervised tokens must be present and True
        self.assertTrue(any(mask))
        self.assertTrue(sum(mask) > 10)


class TestPhase277TrainingConfigAndHardwareReadiness(unittest.TestCase):
    """Verifies training configuration validation, T4 VRAM sizing, and step budgets."""

    def setUp(self):
        self.config_path = os.path.join(PROJECT_ROOT, "training", "chat", "v0_1_t4_release_config.yaml")
        self.cfg = load_training_config(self.config_path)

    def test_01_v0_1_t4_release_config_structure(self):
        self.assertEqual(self.cfg["ownership_policy"], "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
        self.assertTrue(self.cfg["is_production_release"])
        self.assertFalse(self.cfg["teacher_distillation"]["enabled"])
        self.assertEqual(self.cfg["model"]["n_layer"], 8)
        self.assertEqual(self.cfg["model"]["n_head"], 8)
        self.assertEqual(self.cfg["model"]["n_embd"], 512)
        self.assertEqual(self.cfg["model"]["block_size"], 512)
        self.assertEqual(self.cfg["model"]["vocab_size"], 512)
        self.assertEqual(self.cfg["training"]["batch_size"], 8)
        self.assertEqual(self.cfg["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(self.cfg["training"]["max_iters"], 600)

    def test_02_config_validation_clean_pass(self):
        res = validate_training_config(self.cfg, require_production_release=True, verify_dataset_truth=False)
        self.assertTrue(res["valid_config"])
        self.assertEqual(len(res["config_blockers"]), 0)
        self.assertEqual(res["parameters"]["total_parameters"], 25698816)
        self.assertEqual(res["tokens_per_optimizer_step"], 16384)
        self.assertEqual(res["total_training_token_budget"], 9830400)
        self.assertTrue(res["t4_vram_estimate"]["estimated_peak_vram_mb"] < 2000.0)

    def test_03_config_validation_with_dataset_gate_reports_deficit(self):
        res = validate_training_config(self.cfg, require_production_release=True, verify_dataset_truth=True)
        self.assertTrue(res["valid_config"])
        self.assertEqual(res["dataset_status"], "DATASET_NOT_READY")
        self.assertFalse(res["ready_for_training"])
        self.assertTrue(any("Insufficient genuine records" in b for b in res["dataset_blockers"]))

    def test_04_config_validator_rejects_smoke_iters_in_production(self):
        bad_cfg = copy.deepcopy(self.cfg)
        bad_cfg["training"]["max_iters"] = 5
        res = validate_training_config(bad_cfg, require_production_release=True, verify_dataset_truth=False)
        self.assertFalse(res["valid_config"])
        self.assertTrue(any("minimum meaningful training budget" in b for b in res["config_blockers"]))

    def test_05_config_validator_rejects_split_collision(self):
        bad_cfg = copy.deepcopy(self.cfg)
        bad_cfg["paths"]["test_manifest"] = bad_cfg["paths"]["train_manifest"]
        res = validate_training_config(bad_cfg, require_production_release=True, verify_dataset_truth=False)
        self.assertFalse(res["valid_config"])
        self.assertTrue(any("Split collision" in b for b in res["config_blockers"]))

    def test_06_get_lr_cosine_decay_and_edge_cases(self):
        # Normal schedule
        lr_0 = get_lr(0, warmup_iters=10, max_iters=100, lr=1e-3, min_lr=1e-4)
        self.assertEqual(lr_0, 0.0)
        lr_10 = get_lr(10, warmup_iters=10, max_iters=100, lr=1e-3, min_lr=1e-4)
        self.assertAlmostEqual(lr_10, 1e-3, places=6)
        lr_100 = get_lr(100, warmup_iters=10, max_iters=100, lr=1e-3, min_lr=1e-4)
        self.assertAlmostEqual(lr_100, 1e-4, places=6)

        # Zero warmup edge case
        lr_no_warmup = get_lr(50, warmup_iters=0, max_iters=100, lr=1e-3, min_lr=1e-4)
        self.assertTrue(1e-4 <= lr_no_warmup <= 1e-3)

        # Max iters <= warmup iters edge case (must not divide by zero)
        lr_tied = get_lr(10, warmup_iters=10, max_iters=10, lr=1e-3, min_lr=1e-4)
        self.assertEqual(lr_tied, 1e-4)


class TestPhase277SovereigntyAndExecutionGates(unittest.TestCase):
    """Verifies that training execution is fail-closed while preflight dry-run passes."""

    def test_01_training_blocked_when_unauthorized(self):
        self.assertFalse(policy.AUTHORIZED_BY_PREM)
        config_path = os.path.join(PROJECT_ROOT, "training", "chat", "v0_1_t4_release_config.yaml")
        with self.assertRaises(PermissionError) as ctx:
            train_moon_chat(config_path=config_path, dry_run=False, smoke_test=False)
        self.assertIn("TRAINING EXECUTION BLOCKED", str(ctx.exception))

    def test_02_dry_run_preflight_verifies_without_training(self):
        config_path = os.path.join(PROJECT_ROOT, "training", "chat", "v0_1_t4_release_config.yaml")
        res = train_moon_chat(config_path=config_path, dry_run=True, device_override="cpu")
        self.assertEqual(res["status"], "COMPLETED_PREFLIGHT_DRY_RUN")
        self.assertTrue(res["dry_run"])
        self.assertFalse(res["model_instantiated"])
        self.assertFalse(res["forward_executed"])
        self.assertFalse(res["backward_executed"])
        self.assertFalse(res["optimizer_stepped"])
        self.assertFalse(res["checkpoint_written"])


class TestPhase277ColabNotebook(unittest.TestCase):
    """Verifies that the reproducible Colab training notebook is valid and safety-gated."""

    def test_01_notebook_json_and_cells(self):
        nb_path = os.path.join(PROJECT_ROOT, "notebooks", "moon_chat_v0_1_t4_training.ipynb")
        self.assertTrue(os.path.isfile(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        self.assertEqual(nb["nbformat"], 4)
        cells = nb.get("cells", [])
        self.assertTrue(len(cells) >= 6)

        sources = ["".join(c.get("source", [])) for c in cells]
        combined = "\n".join(sources)
        self.assertIn("check_t4_hardware_identity", combined)
        self.assertIn("v0_1_t4_release_config.yaml", combined)
        self.assertIn("compute_file_sha256", combined)
        self.assertIn("validate_training_config", combined)
        self.assertIn("AUTHORIZED_BY_PREM", combined)
        self.assertIn("dry_run=True", combined)


class TestPhase277ModelMetadataHonesty(unittest.TestCase):
    """Verifies honest metadata and status reporting across ModelService."""

    def setUp(self):
        self.ms = ModelService()

    def test_01_get_status_honest_when_no_checkpoint(self):
        self.assertEqual(self.ms.get_status(), "MODEL_NOT_READY")

    def test_02_get_info_does_not_contain_misleading_labels(self):
        info = self.ms.get_info()
        self.assertNotIn("Fine-Tuned by Prem Gaikwad", str(info))
        self.assertNotIn("Frontier Engine", str(info))
        self.assertNotIn("Moon AI Sovereign (7.6B v1", str(info))

    def test_03_moon_chat_pilot_metadata_is_honest(self):
        self.ms.active_model_name = "moon-chat-pilot-v0"
        info = self.ms.get_info()
        self.assertEqual(info["model_name"], "Moon AI Pilot (25.7M Sovereign Chat — Pure Scratch Init)")
        self.assertIn("DATASET_NOT_READY", info["dataset"])
        self.assertEqual(info["dataset_status"], "DATASET_NOT_READY")
        self.assertEqual(info["status"], "MODEL_NOT_READY")
        self.assertFalse(info["is_production_trained"])

    def test_04_quarantined_reference_metadata_is_honest(self):
        self.ms.active_model_name = "qwen-2.5-coder-7b"
        qwen_info = self.ms.get_info()
        self.assertIn("Quarantined", qwen_info["model_name"])
        self.assertFalse(qwen_info["sovereign_owned"])
        self.assertEqual(qwen_info["model_status"], "BLOCKED_EXTERNAL_PRETRAINED")

        self.ms.active_model_name = "llama-3.2"
        llama_info = self.ms.get_info()
        self.assertIn("Quarantined", llama_info["model_name"])
        self.assertFalse(llama_info["sovereign_owned"])
        self.assertEqual(llama_info["model_status"], "BLOCKED_EXTERNAL_PRETRAINED")


if __name__ == "__main__":
    unittest.main()
