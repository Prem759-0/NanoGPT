"""
Moon AI — Stage 2.1.2 Dataset Authenticity, Corpus Adequacy & Training Safety Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates:
1. Schema & Provenance Validation:
   - Missing provenance_type fails validation without silent fallback.
   - Missing attribution fails validation without silent fallback.
   - URL protocol validation rejects malformed/unsupported schemes.
   - Production validation strictly rejects PROVENANCE_PENDING and SYNTHETIC_DEV_FIXTURE.
   - Content hash (sha256) strictly isolates conversation text from metadata.
2. Corpus Authenticity Reclassification:
   - All 22 records strictly categorized as ORIGINAL_SELF_AUTHORED_QA or SOURCE_DERIVED_EDUCATIONAL_SUMMARY.
   - Zero newly constructed records falsely labeled as GENUINE_HUMAN_CONVERSATION.
   - Provenance metadata complete: source_revision, license_url, modification_summary.
3. Multi-Window Chunking & Assistant-Only Loss Masking:
   - Long conversations (> block_size + 1) sliced into sequential windows (zero silent truncation).
   - Assistant-only loss masking preserved across every window (prompt/pad targets = -1).
   - Granular token and window accounting: record_id, window_idx, supervised_tokens.
4. Readiness Gate 6-Status Taxonomy:
   - Evaluates current 22-record corpus as ENGINEERING_SMOKE_TEST_ONLY.
   - Production pilot training blocked (passed: False).
   - Scale thresholds enforced: pilot requires >= 500 records and >= 100k tokens.
5. Training Loop Safety:
   - Production invocation without --smoke-test raises RuntimeError and aborts.
   - Smoke test mode runs 1 iteration, outputs unpromotable smoke_test_ckpt.pt, and touches no registry.
6. Hardware Validation Realism:
   - Exact parameter calculation: 25,698,816 for pilot, 86,133,504 for baseline.
   - GPU validation truthfully reports T4_VALIDATION_PENDING_NO_CUDA when run on CPU.
   - config.yaml declares realistic oom_risk: "LOW".
"""

import copy
import json
import os
import sys
import tempfile
import unittest
import yaml
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.chat.schema import (
    validate_chat_record,
    compute_conversation_sha256,
    PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
    PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY,
    PROVENANCE_GENUINE_HUMAN_CONVERSATION,
    PROVENANCE_SYNTHETIC_DEV_FIXTURE,
    PROVENANCE_PENDING,
    ALLOWED_PROVENANCE_TYPES,
)
from datasets.chat.corpus import get_genuine_corpus
from datasets.chat.tokenizer import SovereignTokenizer
from model import GPTConfig, GPT
from training.chat.dataset import SovereignChatDataset
from training.chat.readiness_gate import (
    ChatReadinessGate,
    STATUS_DATA_INSUFFICIENT,
    STATUS_PROVENANCE_PENDING,
    STATUS_ENGINEERING_SMOKE_TEST_ONLY,
    STATUS_READY_FOR_PILOT,
    STATUS_PILOT_TRAINED,
    STATUS_EVALUATION_PASSED,
    PILOT_MIN_TOTAL_RECORDS,
    PILOT_MIN_TOTAL_TOKENS,
    SMOKE_TEST_MIN_TOTAL_RECORDS,
    SMOKE_TEST_MIN_TOTAL_TOKENS,
)
from training.chat.validate_hardware import (
    calculate_exact_parameters,
    estimate_t4_vram,
    run_live_t4_benchmark,
)
from training.chat.train import train_moon_chat


class TestStage212AuthenticityAndSafety(unittest.TestCase):
    """Exhaustive test suite for Stage 2.1.2 requirements."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tokenizer_path = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
        self.assertTrue(os.path.exists(self.tokenizer_path), "Tokenizer must exist.")
        self.tokenizer = SovereignTokenizer.load(self.tokenizer_path)

        self.valid_record = {
            "id": "test_record_001",
            "source": "Wikipedia - Binary Search",
            "source_url": "https://en.wikipedia.org/wiki/Binary_search_algorithm",
            "source_revision": "123456789",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
            "provenance_type": PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY,
            "is_modified": True,
            "modification_summary": "Adapted educational explanation for conversational teaching.",
            "category": "algorithms",
            "messages": [
                {"role": "user", "content": "How does binary search work?"},
                {"role": "assistant", "content": "Binary search is an efficient search algorithm on sorted arrays with O(log n) time complexity."}
            ]
        }

    def tearDown(self):
        self.tmp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Schema & Provenance Validation
    # -------------------------------------------------------------------------

    def test_missing_provenance_fails_closed(self):
        """Missing provenance_type must fail validation with zero silent fallback."""
        rec = copy.deepcopy(self.valid_record)
        del rec["provenance_type"]
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("missing required 'provenance_type'", err)

        # Even with empty string
        rec["provenance_type"] = ""
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("missing required 'provenance_type'", err)

    def test_missing_attribution_fails_closed(self):
        """Missing attribution must fail validation with zero silent fallback to source title."""
        rec = copy.deepcopy(self.valid_record)
        del rec["attribution"]
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("missing required 'attribution' field", err)

        # Empty whitespace attribution
        rec["attribution"] = "   "
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("missing required 'attribution' field", err)

    def test_url_protocol_validation(self):
        """Source URL must use an approved protocol (https, http, file, internal, docs)."""
        rec = copy.deepcopy(self.valid_record)
        # Unsupported scheme
        rec["source_url"] = "ftp://files.example.com/data.txt"
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("invalid 'source_url' format", err)

        # Malformed string without scheme
        rec["source_url"] = "just_a_string_no_protocol"
        ok, err = validate_chat_record(rec, is_production=True)
        self.assertFalse(ok)
        self.assertIn("invalid 'source_url' format", err)

        # Valid schemes
        valid_urls = [
            "https://en.wikipedia.org/wiki/Quicksort",
            "http://example.com/algo",
            "file:///opt/corpora/docs.txt",
            "internal://original/chat/cs_basics",
            "docs/original/curriculum.md",
        ]
        for url in valid_urls:
            rec["source_url"] = url
            ok, err = validate_chat_record(rec, is_production=True)
            self.assertTrue(ok, f"Failed on valid URL {url}: {err}")

    def test_production_rejects_pending_and_synthetic(self):
        """Production validation strictly rejects PROVENANCE_PENDING and SYNTHETIC_DEV_FIXTURE."""
        rec_pending = copy.deepcopy(self.valid_record)
        rec_pending["provenance_type"] = PROVENANCE_PENDING
        ok, err = validate_chat_record(rec_pending, is_production=True)
        self.assertFalse(ok)
        self.assertIn("provenance_type='PROVENANCE_PENDING'", err)

        rec_synth = copy.deepcopy(self.valid_record)
        rec_synth["provenance_type"] = PROVENANCE_SYNTHETIC_DEV_FIXTURE
        ok, err = validate_chat_record(rec_synth, is_production=True)
        self.assertFalse(ok)
        self.assertIn("SYNTHETIC_DEV_FIXTURE", err)

    def test_content_hash_vs_provenance_separation(self):
        """Content hash (sha256) is computed solely over conversation messages."""
        rec1 = copy.deepcopy(self.valid_record)
        rec2 = copy.deepcopy(self.valid_record)
        rec2["attribution"] = "Different Attribution String"
        rec2["source_revision"] = "99999999"

        h1 = compute_conversation_sha256(rec1["messages"])
        h2 = compute_conversation_sha256(rec2["messages"])
        self.assertEqual(h1, h2, "Metadata change must not affect content sha256.")

        # Modifying message content MUST change sha256
        rec3 = copy.deepcopy(self.valid_record)
        rec3["messages"][1]["content"] = "A modified answer."
        h3 = compute_conversation_sha256(rec3["messages"])
        self.assertNotEqual(h1, h3, "Message modification must alter sha256.")

    # -------------------------------------------------------------------------
    # 2. Corpus Authenticity Reclassification
    # -------------------------------------------------------------------------

    def test_corpus_authenticity_categories(self):
        """All 22 corpus records must be rigorously classified without false claims."""
        corpus = get_genuine_corpus()
        self.assertEqual(len(corpus), 22, "Corpus must contain exactly 22 curated records.")

        self_authored_count = 0
        source_derived_count = 0

        for r in corpus:
            ptype = r.get("provenance_type")
            self.assertIn(
                ptype,
                [PROVENANCE_ORIGINAL_SELF_AUTHORED_QA, PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY],
                f"Record {r.get('id')} has unacceptable provenance_type: {ptype}",
            )
            self.assertNotEqual(
                ptype,
                PROVENANCE_GENUINE_HUMAN_CONVERSATION,
                f"Record {r.get('id')} falsely claimed to be GENUINE_HUMAN_CONVERSATION.",
            )
            if ptype == PROVENANCE_ORIGINAL_SELF_AUTHORED_QA:
                self_authored_count += 1
                self.assertIn("prem gaikwad", r.get("attribution", "").lower())
                self.assertIn("self-authored", r.get("modification_summary", "").lower())
            elif ptype == PROVENANCE_SOURCE_DERIVED_EDUCATIONAL_SUMMARY:
                source_derived_count += 1
                self.assertTrue(r.get("source_revision"), f"Missing source_revision in {r.get('id')}")
                self.assertTrue(r.get("license_url"), f"Missing license_url in {r.get('id')}")
                self.assertTrue(r.get("is_modified"), f"is_modified should be True for {r.get('id')}")
                self.assertTrue(r.get("modification_summary"), f"Missing modification_summary in {r.get('id')}")

            # All records must pass strict production validation
            ok, err = validate_chat_record(r, is_production=True)
            self.assertTrue(ok, f"Record {r.get('id')} failed production validation: {err}")

        self.assertEqual(self_authored_count, 6)
        self.assertEqual(source_derived_count, 16)

    # -------------------------------------------------------------------------
    # 3. Dataset Multi-Window Chunking & Loss Masking
    # -------------------------------------------------------------------------

    def test_multi_window_chunking_long_conversation(self):
        """Long conversations (> block_size + 1) are chunked sequentially with zero tail loss."""
        # Create a conversation with > 200 tokens using short block_size=64
        block_size = 64
        long_turns = []
        for i in range(8):
            long_turns.append({"role": "user", "content": f"User question turn number {i} with several words."})
            long_turns.append({"role": "assistant", "content": f"Assistant response turn number {i} explaining concepts in detail."})

        record = {
            "id": "long_convo_001",
            "provenance_type": "ORIGINAL_SELF_AUTHORED",
            "messages": long_turns
        }

        # Write to temporary jsonl manifest
        manifest_path = os.path.join(self.tmp_dir.name, "long_manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        dataset = SovereignChatDataset(
            manifest_path=manifest_path,
            tokenizer=self.tokenizer,
            block_size=block_size,
            stride=32,
        )

        # Must have created multiple windows
        self.assertGreater(len(dataset), 1, "Long conversation must produce > 1 window.")
        self.assertEqual(dataset.get_unique_record_count(), 1)

        # Verify window metadata
        for idx in range(len(dataset)):
            meta = dataset.get_sample_metadata(idx)
            self.assertEqual(meta["record_id"], "long_convo_001")
            self.assertEqual(meta["window_idx"], idx)
            self.assertEqual(meta["total_windows"], len(dataset))
            self.assertGreater(meta["supervised_tokens"], 0)

    def test_assistant_only_loss_masking_all_windows(self):
        """Assistant-only loss masking is preserved across all windows."""
        # Verify on real training manifest
        train_manifest = os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl")
        self.assertTrue(os.path.exists(train_manifest))

        dataset = SovereignChatDataset(
            manifest_path=train_manifest,
            tokenizer=self.tokenizer,
            block_size=512,
        )

        total_supervised = dataset.get_total_supervised_tokens()
        self.assertGreater(total_supervised, 3000, "Should have several thousand supervised tokens.")

        # Inspect each window
        for i in range(len(dataset)):
            x, y = dataset[i]
            self.assertEqual(x.shape, (512,))
            self.assertEqual(y.shape, (512,))

            # Find unmasked targets (assistant response tokens)
            active_targets = y[y != -1]
            self.assertGreater(len(active_targets), 0, f"Window {i} must have active supervision tokens.")

            # All active targets must be valid token IDs
            self.assertTrue(torch.all(active_targets >= 0))
            self.assertTrue(torch.all(active_targets < self.tokenizer.vocab_size))

    # -------------------------------------------------------------------------
    # 4. Readiness Gate 6-Status Taxonomy
    # -------------------------------------------------------------------------

    def test_readiness_gate_evaluates_smoke_test_only(self):
        """22-record corpus evaluates to ENGINEERING_SMOKE_TEST_ONLY and blocks pilot training."""
        gate = ChatReadinessGate(root_dir=PROJECT_ROOT)
        report = gate.run_all_checks()

        self.assertEqual(report["status"], STATUS_ENGINEERING_SMOKE_TEST_ONLY)
        self.assertFalse(report["passed"], "Readiness gate must NOT pass for 22 records.")
        self.assertTrue(report["smoke_test_passed"], "Smoke test criteria must pass for 22 records.")
        self.assertGreater(len(report["blockers"]), 0)
        self.assertIn("ENGINEERING_SMOKE_TEST_ONLY", report["blockers"][0])

    def test_readiness_scale_thresholds(self):
        """Configurable thresholds distinguish smoke test from pilot scale."""
        self.assertEqual(SMOKE_TEST_MIN_TOTAL_RECORDS, 10)
        self.assertEqual(SMOKE_TEST_MIN_TOTAL_TOKENS, 2000)
        self.assertEqual(PILOT_MIN_TOTAL_RECORDS, 500)
        self.assertEqual(PILOT_MIN_TOTAL_TOKENS, 100000)

    # -------------------------------------------------------------------------
    # 5. Training Loop Safety
    # -------------------------------------------------------------------------

    def test_production_training_blocked_without_readiness(self):
        """Calling train_moon_chat without smoke_test=True fails closed and blocks production training."""
        with self.assertRaises((ValueError, RuntimeError)) as ctx:
            train_moon_chat(
                config_path=os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml"),
                smoke_test=False,
            )
        err = str(ctx.exception)
        is_manifest_security_block = "Production manifest security violation" in err
        is_readiness_block = "Production pilot training blocked by readiness gate" in err
        self.assertTrue(
            is_manifest_security_block or is_readiness_block,
            f"Expected fail-closed production training block, got: {err}"
        )

    def test_smoke_test_mode_creates_unpromotable_checkpoint(self):
        """Smoke test mode executes 1 iteration and produces unpromotable checkpoint."""
        ckpt_path = os.path.join(self.tmp_dir.name, "smoke_test_ckpt.pt")
        res = train_moon_chat(
            config_path=os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml"),
            smoke_test=True,
            custom_ckpt_path=ckpt_path,
        )

        self.assertEqual(res["status"], "COMPLETED_SMOKE_TEST")
        self.assertEqual(res["total_iterations"], 1)
        self.assertEqual(res["unique_records"], 18)
        self.assertGreater(res["chunks_generated"], 0)
        self.assertGreater(res["supervised_tokens"], 0)

        # Inspect checkpoint
        self.assertTrue(os.path.exists(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["status"], STATUS_ENGINEERING_SMOKE_TEST_ONLY)
        self.assertFalse(checkpoint["is_production_trained"])
        self.assertFalse(checkpoint["promotable"])

        # Verify models/registry.json was NOT modified
        registry_path = os.path.join(PROJECT_ROOT, "models", "registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        self.assertEqual(registry["models"]["moon-chat-v0"]["status"], STATUS_DATA_INSUFFICIENT)

    # -------------------------------------------------------------------------
    # 6. Hardware Validation & Sizing
    # -------------------------------------------------------------------------

    def test_exact_parameter_counts(self):
        """Exact parameter counting accounts for weight tying, biases, embeddings, and pre-LN."""
        pilot_cfg = GPTConfig(
            block_size=512,
            vocab_size=512,
            n_layer=8,
            n_head=8,
            n_embd=512,
            bias=False
        )
        pilot_calc = calculate_exact_parameters(pilot_cfg)
        self.assertEqual(pilot_calc["total_parameters"], 25698816, "Pilot model must have exactly 25,698,816 parameters.")

        base_cfg = GPTConfig(
            block_size=1024,
            vocab_size=512,
            n_layer=12,
            n_head=12,
            n_embd=768,
            bias=False
        )
        base_calc = calculate_exact_parameters(base_cfg)
        self.assertEqual(base_calc["total_parameters"], 86133504, "Baseline model must have exactly 86,133,504 parameters.")

    def test_hardware_status_pending_on_cpu(self):
        """run_live_t4_benchmark reports PENDING when CUDA is not present."""
        if not torch.cuda.is_available():
            res = run_live_t4_benchmark(batch_size=4, block_size=512)
            self.assertEqual(res["status"], "T4_VALIDATION_PENDING_NO_CUDA")
            self.assertFalse(res["live_measured"])
            self.assertIn("requires a CUDA GPU host", res["message"])

    def test_config_oom_risk_is_low(self):
        """config.yaml must declare realistic oom_risk: 'LOW'."""
        config_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hw = cfg.get("hardware_budget", {})
        self.assertEqual(hw.get("oom_risk"), "LOW")


if __name__ == "__main__":
    unittest.main()

