"""
Moon AI — Stage 2.0.8 / Stage 2.1 Test Suite
Registry Reconciliation & Moon Chat Dataset Preparation (moon-chat-v0)
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import yaml

from datasets.chat.schema import (
    validate_chat_record,
    compute_conversation_sha256,
    normalize_text,
    APPROVED_LICENSES
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.curator import deduplicate_and_validate_records, split_chat_dataset
from datasets.chat.corpus import get_genuine_corpus
from datasets.chat.audit import generate_chat_audit_report
from training.chat.dataset import SovereignChatDataset
from scripts.promote_best_model import evaluate_and_promote, load_registry
from scripts.daily_report import generate_daily_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TestStage208ChatDatasetAndRegistry(unittest.TestCase):
    """
    Test suite verifying:
    1. Truthful registry reconciliation (current_best_model: null, legacy third-party disqualified).
    2. Graceful promotion gate and daily report handling of current_best_model: null.
    3. Pure-sovereign Byte-Level BPE tokenizer from scratch (zero pretrained vocabs).
    4. Conversational record schema validation and synthetic fixture rejection.
    5. Exact and normalized deduplication.
    6. Deterministic 80/10/10 splitting with zero cross-split leakage.
    7. Production manifests containing verified genuine records with zero synthetic contamination.
    8. Audit report generation with 100% hash integrity.
    9. Hardware feasibility and VRAM budget for Colab T4.
    10. SovereignChatDataset PyTorch data loading and causal next-token chunking.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_registry_reconciliation_truthfulness(self):
        """Verify models/registry.json and models/model_registry.json are reconciled truthfully."""
        model_reg_path = os.path.join(PROJECT_ROOT, "models", "model_registry.json")
        reg_path = os.path.join(PROJECT_ROOT, "models", "registry.json")

        self.assertTrue(os.path.exists(model_reg_path), "models/model_registry.json must exist")
        self.assertTrue(os.path.exists(reg_path), "models/registry.json must exist")

        with open(model_reg_path, "r", encoding="utf-8") as f:
            m_reg = json.load(f)
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        # current_best_model must be null (no unverified third-party checkpoints promoted)
        self.assertIsNone(m_reg.get("current_best_model"), "model_registry current_best_model must be null")
        self.assertIsNone(reg.get("current_best_model"), "registry current_best_model must be null")

        # Moon AI Sovereign (7.6B v1) must be isolated under unverified_legacy_artifacts
        legacy_m = m_reg.get("unverified_legacy_artifacts", [])
        legacy = reg.get("unverified_legacy_artifacts", [])
        self.assertTrue(len(legacy_m) > 0, "model_registry must document unverified_legacy_artifacts")
        self.assertTrue(len(legacy) > 0, "registry must document unverified_legacy_artifacts")

        artifact = legacy_m[0]
        self.assertEqual(artifact["name"], "Moon AI Sovereign (7.6B v1)")
        self.assertEqual(artifact["policy_compliance"], "NON_COMPLIANT")
        self.assertTrue(len(artifact["disqualification_reasons"]) >= 3)
        self.assertTrue(any("Qwen" in r for r in artifact["disqualification_reasons"]))

    def test_02_promote_best_model_handles_null_best(self):
        """Verify promote_best_model handles current_best_model: null gracefully."""
        test_reg_path = os.path.join(self.temp_dir, "test_registry.json")
        initial_reg = {
            "version": "2.0.8",
            "current_best_model": None,
            "promotion_history": [],
            "unverified_legacy_artifacts": []
        }
        with open(test_reg_path, "w", encoding="utf-8") as f:
            json.dump(initial_reg, f)

        loaded = load_registry(test_reg_path)
        self.assertIsNone(loaded["current_best_model"])

        # Create candidate report lacking attestation (must fail closed cleanly without crashing on None best)
        cand_report_path = os.path.join(self.temp_dir, "eval_cand.json")
        with open(cand_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "candidate_test",
                "composite_quality_score": 80.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0
            }, f)

        res = evaluate_and_promote(
            candidate_report_path=cand_report_path,
            registry_path=test_reg_path,
            base_dir=self.temp_dir
        )
        self.assertFalse(res, "Unattested candidate must be rejected")

        # Verify registry was not corrupted
        updated_reg = load_registry(test_reg_path)
        self.assertIsNone(updated_reg["current_best_model"])

    def test_03_daily_report_handles_null_best(self):
        """Verify generate_daily_report executes without crashing when current_best_model is null."""
        out_file = generate_daily_report()
        self.assertTrue(os.path.exists(out_file), "Daily report markdown file must exist")
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertTrue(len(content) > 100)
        self.assertIn("Active Best Model Status", content)
        self.assertIn("None (Awaiting Sovereign Training)", content)

    def test_04_sovereign_tokenizer_from_scratch(self):
        """Verify SovereignTokenizer trains from scratch, expands vocab, and roundtrips UTF-8."""
        tok = SovereignTokenizer()
        self.assertEqual(tok.vocab_size, 263) # 7 special + 256 bytes

        corpus = [
            "Moon AI is an autonomous, sovereign artificial intelligence system.",
            "Trained strictly from random Gaussian initialization by Prem Gaikwad.",
            "Byte-level BPE ensures 100% vocabulary coverage without <unk> tokens."
        ]
        tok.train(corpus, target_vocab_size=300, min_frequency=2)
        self.assertGreater(tok.vocab_size, 263)

        # Lossless round-trip test
        test_strings = [
            "Hello World!",
            "Prem Gaikwad (India 🇮🇳)",
            "Special characters: !@#$%^&*()_+~`|}{[]:;?><,./-=",
            "Mathematical notation: ∀x ∈ ℝ, x² ≥ 0",
            "Multi-line\nChatML\tTokens"
        ]
        for s in test_strings:
            enc = tok.encode_text(s)
            dec = tok.decode_tokens(enc)
            self.assertEqual(s, dec, f"Lossless roundtrip failed for: {s}")

        # Serialization test
        tok_file = os.path.join(self.temp_dir, "test_tok.json")
        tok.save(tok_file)
        loaded_tok = SovereignTokenizer.load(tok_file)
        self.assertEqual(loaded_tok.vocab_size, tok.vocab_size)
        self.assertEqual(loaded_tok.encode_text("Moon AI"), tok.encode_text("Moon AI"))

    def test_05_chat_record_schema_validation(self):
        """Verify schema validation enforces mandatory fields and rejects synthetic fixtures."""
        valid_rec = {
            "id": "test_rec_01",
            "category": "science",
            "source": "Open Science",
            "source_url": "https://example.org",
            "license": "CC0-1.0",
            "attribution": "Open Science Foundation",
            "provenance_type": "SOURCE_DERIVED_EDUCATIONAL_SUMMARY",
            "is_dev_fixture": False,
            "messages": [
                {"role": "user", "content": "What is gravity?"},
                {"role": "assistant", "content": "Gravity is the fundamental natural phenomenon."}
            ]
        }
        ok, err = validate_chat_record(valid_rec, is_production=True)
        self.assertTrue(ok, f"Expected valid record: {err}")

        # Synthetic fixture in production must fail
        syn_rec = dict(valid_rec)
        syn_rec["is_dev_fixture"] = True
        ok, err = validate_chat_record(syn_rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("is_dev_fixture=True", err)

        # Missing user turn must fail
        no_user_rec = dict(valid_rec)
        no_user_rec["messages"] = [
            {"role": "assistant", "content": "Only assistant message."},
            {"role": "system", "content": "Only system message."}
        ]
        ok, err = validate_chat_record(no_user_rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("missing at least one 'user'", err)

        # Unapproved license must fail
        bad_lic_rec = dict(valid_rec)
        bad_lic_rec["license"] = "UnknownNonCommercialProprietary"
        ok, err = validate_chat_record(bad_lic_rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("unapproved or missing license", err)

    def test_06_exact_and_normalized_deduplication(self):
        """Verify deduplication detects both exact hash duplicates and normalized prompt duplicates."""
        corpus = get_genuine_corpus()
        self.assertGreaterEqual(len(corpus), 5)

        base_rec = corpus[0]
        exact_dup = dict(base_rec)
        exact_dup["id"] = "exact_dup_id"

        norm_dup = {
            "id": "norm_dup_id",
            "category": base_rec["category"],
            "source": base_rec["source"],
            "source_url": base_rec["source_url"],
            "license": base_rec["license"],
            "is_dev_fixture": False,
            "messages": [
                {"role": "user", "content": "  " + base_rec["messages"][0]["content"].upper() + "  "},
                {"role": "assistant", "content": "Different response content for testing."}
            ]
        }

        test_records = [base_rec, exact_dup, norm_dup]
        valid, rejections = deduplicate_and_validate_records(test_records, is_production=True)
        self.assertEqual(len(valid), 1, "Only 1 record should survive exact + normalized deduplication")
        self.assertEqual(len(rejections), 2)

    def test_07_deterministic_splitting_and_zero_leakage(self):
        """Verify dataset splitting is deterministic and has strictly zero cross-split leakage."""
        records = get_genuine_corpus()
        train, val, test = split_chat_dataset(records, seed=42)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(val) + len(test), len(records))

        train_hashes = {r["sha256"] for r in train}
        val_hashes = {r["sha256"] for r in val}
        test_hashes = {r["sha256"] for r in test}

        # Assert zero hash overlap
        self.assertEqual(len(train_hashes.intersection(val_hashes)), 0)
        self.assertEqual(len(train_hashes.intersection(test_hashes)), 0)
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0)

        # Assert zero prompt text overlap
        train_prompts = {normalize_text(r["messages"][0]["content"]) for r in train}
        val_prompts = {normalize_text(r["messages"][0]["content"]) for r in val}
        test_prompts = {normalize_text(r["messages"][0]["content"]) for r in test}

        self.assertEqual(len(train_prompts.intersection(val_prompts)), 0)
        self.assertEqual(len(train_prompts.intersection(test_prompts)), 0)
        self.assertEqual(len(val_prompts.intersection(test_prompts)), 0)

    def test_08_production_manifests_contain_genuine_records(self):
        """Verify datasets/manifests/chat_*.jsonl files contain genuine records and 0 dev fixtures."""
        manifest_dir = os.path.join(PROJECT_ROOT, "datasets", "manifests")
        train_path = os.path.join(manifest_dir, "chat_train.jsonl")
        val_path = os.path.join(manifest_dir, "chat_val.jsonl")
        test_path = os.path.join(manifest_dir, "chat_test.jsonl")

        for p in [train_path, val_path, test_path]:
            self.assertTrue(os.path.exists(p), f"Manifest {p} must exist")

        with open(train_path, "r", encoding="utf-8") as f:
            train_recs = [json.loads(line) for line in f if line.strip()]
        with open(val_path, "r", encoding="utf-8") as f:
            val_recs = [json.loads(line) for line in f if line.strip()]
        with open(test_path, "r", encoding="utf-8") as f:
            test_recs = [json.loads(line) for line in f if line.strip()]

        self.assertGreaterEqual(len(train_recs), 8, "chat_train.jsonl must have >= 8 records")
        self.assertGreaterEqual(len(val_recs), 1, "chat_val.jsonl must have >= 1 record")
        self.assertGreaterEqual(len(test_recs), 1, "chat_test.jsonl must have >= 1 record")

        # Zero dev fixtures in any production split
        for rec in train_recs + val_recs + test_recs:
            self.assertFalse(rec.get("is_dev_fixture", False))
            self.assertNotEqual(rec.get("source_type"), "dev_example_fixture")
            self.assertIn(rec.get("license"), APPROVED_LICENSES)
            self.assertTrue(rec.get("sha256"))
            # Validate hash integrity
            expected_hash = compute_conversation_sha256(rec["messages"])
            self.assertEqual(rec["sha256"], expected_hash)

    def test_09_audit_report_verification(self):
        """Verify datasets/chat/audit_report.json integrity and metrics."""
        audit_path = os.path.join(PROJECT_ROOT, "datasets", "chat", "audit_report.json")
        audit_md_path = os.path.join(PROJECT_ROOT, "datasets", "chat", "audit_report.md")

        self.assertTrue(os.path.exists(audit_path), "audit_report.json must exist")
        self.assertTrue(os.path.exists(audit_md_path), "audit_report.md must exist")

        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)

        self.assertIn(audit["report_metadata"]["status"], ["READY_FOR_PILOT_TRAINING", "ENGINEERING_SMOKE_TEST_ONLY", "READY_FOR_PILOT"])
        self.assertEqual(audit["report_metadata"]["ownership_policy"], "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
        self.assertFalse(audit["integrity_verification"]["cross_split_leakage_detected"])
        self.assertEqual(audit["integrity_verification"]["hash_mismatches"], 0)
        self.assertGreaterEqual(audit["summary"]["total_tokens"], 3000)

    def test_10_colab_t4_training_config_and_budget(self):
        """Verify training/chat/config.yaml allocations match NVIDIA T4 15.3 GB VRAM limits."""
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        self.assertTrue(os.path.exists(cfg_path), "config.yaml must exist")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["ownership_policy"], "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
        self.assertEqual(cfg["model"]["name"], "moon-chat-pilot-25m")
        self.assertEqual(cfg["model"]["vocab_size"], 512)

        hw = cfg["hardware_budget"]
        self.assertEqual(hw["vram_available_mb"], 15360)
        self.assertLess(hw["estimated_peak_vram_mb"], 2500) # Must stay under 2.5 GB peak
        self.assertGreater(hw["vram_safety_margin_mb"], 12000) # > 12 GB headroom
        self.assertIn(hw["oom_risk"], ("LOW", "ZERO"))

    def test_11_chat_dataset_pytorch_loader(self):
        """Verify SovereignChatDataset produces properly formatted (x, y) next-token chunks."""
        tok_path = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
        self.assertTrue(os.path.exists(tok_path), "models/chat/tokenizer.json must exist")
        tokenizer = SovereignTokenizer.load(tok_path)

        train_path = os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl")
        dataset = SovereignChatDataset(train_path, tokenizer, block_size=128)
        self.assertGreater(len(dataset), 0)

        x, y = dataset[0]
        self.assertEqual(x.shape, (128,))
        self.assertEqual(y.shape, (128,))

        # Verify causal alignment for non-padded tokens
        valid_mask = y != -1
        if valid_mask.sum() > 1:
            indices = valid_mask.nonzero().squeeze(-1)
            first_idx = indices[0].item()
            last_idx = indices[-1].item()
            if last_idx > first_idx:
                # Target y[i] is next token after x[i]
                self.assertEqual(x[first_idx + 1].item(), y[first_idx].item())

if __name__ == "__main__":
    unittest.main()
