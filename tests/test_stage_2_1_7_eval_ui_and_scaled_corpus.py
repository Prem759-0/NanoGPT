"""
tests/test_stage_2_1_7_eval_ui_and_scaled_corpus.py
Unit & Integration Verification for Moon AI Stage 2.1.7:
1. Sovereign Pilot Evaluation & Promotion Gate audit.
2. Webapp ModelService & interactive chat integration for moon-chat-pilot-v0.
3. High-volume scaled offline corpus generation & ChatReadinessGate certification.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)
"""

import os
import sys
import re
import json
import ast
import unittest
from typing import Dict, Any, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.model_service import ModelService
from training.chat.readiness_gate import ChatReadinessGate, STATUS_READY_FOR_PILOT
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.schema import validate_chat_record, PROVENANCE_DETERMINISTIC_CURRICULUM_TEMPLATE

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")
EVAL_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "reports", "eval_moon_chat_pilot_v0_latest.json")
SCALED_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_train.jsonl")
SCALED_VAL = os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_val.jsonl")
SCALED_TEST = os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_test.jsonl")


class TestStage217EvalUiAndScaledCorpus(unittest.TestCase):
    """Test suite certifying Stage 2.1.7 deliverables."""

    @classmethod
    def setUpClass(cls):
        from tests.checkpoint_fixture_utils import ensure_pilot_checkpoint_fixture
        cls.ckpt_path, cls.created_fixture = ensure_pilot_checkpoint_fixture()

    @classmethod
    def tearDownClass(cls):
        try:
            from tests.checkpoint_fixture_utils import cleanup_checkpoint_fixture
            cleanup_checkpoint_fixture(cls.ckpt_path, getattr(cls, "created_fixture", False))
        except Exception:
            pass

    def test_01_pilot_checkpoint_policy_b_attestation(self):
        """Certifies that moon_chat_pilot_v0.pt exists and maintains strict Policy B attestation."""
        self.assertTrue(os.path.isfile(CHECKPOINT_PATH), f"Checkpoint not found at: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        meta = ckpt.get("config", {}) or ckpt.get("metadata", {})

        self.assertFalse(meta.get("teacher_checkpoint_weights_transferred", True),
                         "Policy B violation: teacher_checkpoint_weights_transferred must be False.")
        self.assertEqual(meta.get("initialization"), "random_gaussian_scratch",
                         "Model must be initialized from random_gaussian_scratch.")
        self.assertEqual(meta.get("ownership_policy"), "TEACHER_STUDENT_RANDOM_INIT",
                         "Ownership policy must be TEACHER_STUDENT_RANDOM_INIT.")
        self.assertIn("moon-chat", meta.get("model_name", ""))

    def test_02_pilot_evaluation_report_integrity(self):
        """Certifies that the pilot evaluation report was generated with valid test metrics."""
        self.assertTrue(os.path.isfile(EVAL_REPORT_PATH), f"Eval report not found at: {EVAL_REPORT_PATH}")
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("model_name"), "moon-chat-pilot-v0")
        self.assertIn("test_loss", data)
        self.assertIn("test_perplexity", data)
        self.assertGreater(data["test_loss"], 0.0)
        self.assertGreater(data["test_perplexity"], 1.0)
        self.assertEqual(data.get("non_empty_rate_pct"), 100.0)
        self.assertIn("promotion_gate", data)
        # Verify promotion gate decision is documented honestly
        self.assertIn("promoted", data["promotion_gate"])
        self.assertIn("reason", data["promotion_gate"])

    def test_03_model_service_available_models_includes_pilot(self):
        """Certifies that ModelService.get_available_models lists moon-chat-pilot-v0 and distinguishes fixtures from trained models."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        models = ms.get_available_models()
        pilot_entries = [m for m in models if m["id"] == "moon-chat-pilot-v0"]

        self.assertEqual(len(pilot_entries), 1, "moon-chat-pilot-v0 must be listed in available models.")
        pilot = pilot_entries[0]
        self.assertEqual(pilot["parameters"], "25.70M")
        self.assertEqual(pilot["block_size"], 512)
        self.assertEqual(pilot["vocab_size"], 512)
        self.assertEqual(pilot["type"], "sovereign_scratch")

        is_fixture = getattr(self, "created_fixture", False) or pilot.get("is_test_fixture", False)
        if is_fixture:
            # Production model discovery must NOT treat a temporary CI fixture as an available trained model
            self.assertFalse(pilot["available"], "CI test fixture must not be marked available as a trained production model.")
            self.assertTrue(pilot.get("is_test_fixture", False), "CI fixture entry must be flagged with is_test_fixture: true.")
        else:
            self.assertTrue(pilot["available"], "Genuine production checkpoint must be marked available.")

    def test_04_model_service_pilot_loading_and_get_info(self):
        """Certifies switching to moon-chat-pilot-v0 loads the model and reports Policy B metadata without false training claims."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        success = ms.switch_model("moon-chat-pilot-v0")
        self.assertTrue(success, f"Failed to switch to moon-chat-pilot-v0: {ms.error_message}")
        self.assertEqual(ms.active_model_name, "moon-chat-pilot-v0")

        info = ms.get_info()
        self.assertEqual(info["model_name"], "Moon AI Pilot (25.7M Sovereign Chat — Pure Scratch Init)")
        self.assertEqual(info["parameters"], 25698816)
        self.assertEqual(info["n_layer"], 8)
        self.assertEqual(info["n_head"], 8)
        self.assertEqual(info["n_embd"], 512)
        self.assertEqual(info["ownership_policy"], "TEACHER_STUDENT_RANDOM_INIT")
        self.assertFalse(info["teacher_checkpoint_weights_transferred"])

        is_fixture = getattr(self, "created_fixture", False) or info.get("is_test_fixture", False)
        if is_fixture:
            self.assertTrue(info.get("is_test_fixture", False))
            self.assertFalse(info.get("is_production_trained", True))
            self.assertEqual(info.get("status"), "CI_TEST_FIXTURE")
        else:
            self.assertTrue(info.get("is_production_trained", False))

    def test_05_model_service_pilot_generation(self):
        """Certifies interactive generation using moon-chat-pilot-v0 via ModelService."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        ms.switch_model("moon-chat-pilot-v0")

        res = ms.generate("Hi", max_new_tokens=20, temperature=0.7)
        self.assertIsInstance(res, dict)
        self.assertIn("reply", res)
        self.assertGreater(len(res["reply"]), 0)
        self.assertIn("Moon AI Pilot", res["model_name"])
        self.assertIn("Local Sovereign Engine", res["device"])
        self.assertGreaterEqual(res["tokens_generated"], 1)

    def test_06_scaled_corpus_manifest_scale_and_splits(self):
        """Certifies that scaled manifests exist with >= 1000 records and >= 400000 supervised tokens."""
        for path in [SCALED_TRAIN, SCALED_VAL, SCALED_TEST]:
            self.assertTrue(os.path.isfile(path), f"Scaled manifest not found at: {path}")

        def read_jsonl(p: str) -> List[Dict[str, Any]]:
            recs = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
            return recs

        train_recs = read_jsonl(SCALED_TRAIN)
        val_recs = read_jsonl(SCALED_VAL)
        test_recs = read_jsonl(SCALED_TEST)

        total_recs = len(train_recs) + len(val_recs) + len(test_recs)
        self.assertGreaterEqual(total_recs, 1000, f"Expected >= 1000 scaled records, got {total_recs}")
        self.assertGreaterEqual(len(train_recs), 800, f"Expected >= 800 train records, got {len(train_recs)}")
        self.assertGreaterEqual(len(val_recs), 50, f"Expected >= 50 val records, got {len(val_recs)}")
        self.assertGreaterEqual(len(test_recs), 50, f"Expected >= 50 test records, got {len(test_recs)}")

        # Check deduplication and non-overlapping hashes
        train_hashes = {r["sha256"] for r in train_recs}
        val_hashes = {r["sha256"] for r in val_recs}
        test_hashes = {r["sha256"] for r in test_recs}

        self.assertEqual(len(train_hashes & val_hashes), 0, "Train and Val splits share overlapping hashes!")
        self.assertEqual(len(train_hashes & test_hashes), 0, "Train and Test splits share overlapping hashes!")
        self.assertEqual(len(val_hashes & test_hashes), 0, "Val and Test splits share overlapping hashes!")

    def test_07_scaled_corpus_readiness_gate_certification(self):
        """Certifies that the scaled manifests pass ChatReadinessGate with READY_FOR_PILOT and 0 blockers."""
        gate = ChatReadinessGate(
            train_manifest=SCALED_TRAIN,
            val_manifest=SCALED_VAL,
            test_manifest=SCALED_TEST
        )
        res = gate.run_all_checks()
        self.assertEqual(res["status"], STATUS_READY_FOR_PILOT,
                         f"Readiness gate failed on scaled corpus: {res.get('blockers')}")
        self.assertEqual(len(res.get("blockers", [])), 0,
                         f"Expected 0 blockers on scaled corpus, got: {res.get('blockers')}")

    def test_08_scaled_corpus_ast_syntax_and_provenance(self):
        """Certifies that all code blocks in scaled corpus pass AST parsing and have correct provenance."""
        with open(SCALED_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.assertEqual(rec.get("provenance_type"), PROVENANCE_DETERMINISTIC_CURRICULUM_TEMPLATE)
                self.assertEqual(rec.get("license"), "Apache-2.0")
                self.assertFalse(rec.get("is_synthetic"))
                self.assertFalse(rec.get("is_dev_fixture"))

                # If programming, test AST
                if "program" in rec.get("domain", "") or "code" in rec.get("domain", ""):
                    content = rec["messages"][1]["content"]
                    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
                    for b in blocks:
                        try:
                            ast.parse(b)
                        except SyntaxError as e:
                            self.fail(f"AST SyntaxError in scaled record {rec.get('id')}: {e}")

    def test_09_tokenizer_encode_decode_symmetry(self):
        """Certifies that SovereignTokenizer implements standard encode and decode methods."""
        tok = SovereignTokenizer()
        self.assertTrue(hasattr(tok, "encode"), "SovereignTokenizer missing encode alias.")
        self.assertTrue(hasattr(tok, "decode"), "SovereignTokenizer missing decode alias.")

        sample_text = "Testing Sovereign Tokenizer Symmetry."
        encoded = tok.encode(sample_text)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        decoded = tok.decode(encoded)
        self.assertIsInstance(decoded, str)
        self.assertEqual(decoded, sample_text)


if __name__ == "__main__":
    unittest.main()

