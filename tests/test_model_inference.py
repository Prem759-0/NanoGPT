"""
tests/test_model_inference.py
Deterministic inference regression test suite for NanoGPT Lab.
Tests:
- Checkpoint loading integrity and error handling
- Tokenizer mapping and vocab size validation
- Context window boundary and truncation
- Multi-turn conversation handling
- Empty prompt handling
- 10-prompt regression benchmark (raw output recorded without masking)
"""
import os
import sys
import unittest
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.model_service import ModelService, ModelNotTrainedException

class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["DEBUG_INFERENCE"] = "1"
        cls.service = ModelService(base_dir=PROJECT_ROOT)

    def test_01_deterministic_scratch_loading(self):
        """Verify scratch model loads correctly with valid vocab and metadata."""
        loaded = self.service.load_model("scratch")
        self.assertTrue(loaded, f"Scratch model failed to load: {self.service.error_message}")
        self.assertEqual(self.service.active_model_name, "scratch")
        self.assertIsNotNone(self.service.model)
        self.assertFalse(self.service.model.training, "Model should be in eval() mode")
        self.assertEqual(self.service.vocab_size, 97, "Expected 97-char vocabulary from assistant_chat")
        self.assertIsNotNone(self.service.encode_fn)
        self.assertIsNotNone(self.service.decode_fn)

    def test_02_tokenizer_roundtrip(self):
        """Verify tokenizer encode and decode roundtrip."""
        self.service.load_model("scratch")
        sample_text = "User: Hello\nAssistant: Hi!"
        token_ids = self.service.encode_fn(sample_text)
        self.assertIsInstance(token_ids, list)
        self.assertEqual(len(token_ids), len(sample_text))
        decoded = self.service.decode_fn(token_ids)
        self.assertEqual(sample_text, decoded)

    def test_03_missing_checkpoint_fails_loudly(self):
        """Verify requesting a non-existent checkpoint fails cleanly without silent fallbacks."""
        bogus_service = ModelService(base_dir=PROJECT_ROOT, preferred_checkpoint="/nonexistent/ckpt.pt")
        # Ensure it does not find a bogus path
        ckpt = bogus_service.find_checkpoint(preferred_path="/nonexistent/ckpt.pt")
        self.assertIsNone(ckpt)
        loaded = bogus_service.load_model("scratch")
        self.assertFalse(loaded)
        self.assertIn("No local scratch assistant checkpoint found", bogus_service.error_message)

    def test_04_missing_tokenizer_metadata_fails_loudly(self):
        """Verify _setup_scratch_tokenizer fails loudly if meta.pkl is absent."""
        service = ModelService(base_dir=PROJECT_ROOT)
        with self.assertRaises(FileNotFoundError):
            service._setup_scratch_tokenizer(
                ckpt_path=os.path.join(PROJECT_ROOT, "nonexistent_dir", "ckpt.pt"),
                dataset="nonexistent_dataset_12345"
            )

    def test_05_context_window_truncation(self):
        """Verify prompt exceeding block_size is safely truncated without overflow."""
        self.service.load_model("scratch")
        block_size = getattr(self.service.model_config, "block_size", 128)
        # Create prompt exceeding block_size
        long_prompt = "A" * (block_size + 50)
        res = self.service.generate(prompt=long_prompt, max_new_tokens=10)
        self.assertIn("reply", res)
        # Prompt length should have been capped to <= block_size - 1
        self.assertLess(res["prompt_length"], block_size)

    def test_06_empty_prompt_handling(self):
        """Verify empty prompt returns clean diagnostic message instead of crashing."""
        self.service.load_model("scratch")
        res = self.service.generate(prompt="   ")
        self.assertEqual(res["stop_reason"], "empty_prompt")
        self.assertEqual(res["tokens_generated"], 0)
        self.assertIn("Empty prompt", res["reply"])

    def test_07_ten_prompt_regression_suite(self):
        """
        Execute the 10-prompt regression benchmark on scratch model.
        Records raw outputs, tokens generated, and stop reasons without masking.
        """
        self.service.load_model("scratch")
        prompts = [
            ("1. Hi", "Hi", None),
            ("2. What is a dog?", "What is a dog?", None),
            ("3. Explain Sun and Moon", "Explain the Sun and Moon in simple English.", None),
            ("4. Math 2 + 3", "What is 2 + 3?", None),
            ("5. Python add function", "Write a Python function that adds two numbers.", None),
            ("6. HTML/CSS/JS page", "Write a basic HTML page with CSS and JavaScript.", None),
            ("7. Python syntax error", "Explain a Python syntax error.", None),
            ("8. Multi-turn conversation", "Can you explain more?", [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]),
            ("9. Empty prompt", "", None),
            ("10. Missing checkpoint", "__CHECK_INVALID_CKPT__", None),
        ]

        results = []
        print("\n" + "=" * 70)
        print("   SCRATCH NANOGPT 10-PROMPT REGRESSION BENCHMARK")
        print("=" * 70)

        for label, p_text, hist in prompts:
            if p_text == "__CHECK_INVALID_CKPT__":
                bogus = ModelService(base_dir=PROJECT_ROOT, preferred_checkpoint="fake.pt")
                loaded = bogus.load_model("scratch")
                status = "PASSED (Clean Failure)" if not loaded else "FAILED"
                print(f"[{label}] -> Status: {status} | Error: {bogus.error_message}")
                results.append((label, status, bogus.error_message or ""))
                self.assertFalse(loaded)
                continue

            res = self.service.generate(prompt=p_text, max_new_tokens=60, temperature=0.7, history=hist)
            status = f"Tokens: {res['tokens_generated']} | Stop: {res['stop_reason']}"
            raw_preview = repr(res.get("raw_reply", "")[:60])
            print(f"[{label}] Prompt: {repr(p_text)}")
            print(f"   -> {status}")
            print(f"   -> Raw: {raw_preview}")
            print(f"   -> Clean Reply: {repr(res['reply'][:60])}\n")
            results.append((label, status, res["reply"]))

            self.assertIn("reply", res)
            self.assertIn("stop_reason", res)
            self.assertIn("tokens_generated", res)

        print("=" * 70)
        print(f"All {len(results)} regression cases completed.")
        print("=" * 70)

if __name__ == "__main__":
    unittest.main()
