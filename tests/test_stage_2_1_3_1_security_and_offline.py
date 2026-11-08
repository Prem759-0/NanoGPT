"""
Moon AI — Stage 2.1.3.1 Security & Offline Regression Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Validates:
1. Notebook credential sanitation (zero hardcoded secrets).
2. Notebook CORS configuration (no wildcard *; restricted to private/localhost).
3. Public tunnel disabling (ENABLE_NGROK_TUNNEL = False by default).
4. Sovereign student random Gaussian scratch initialization in notebook (zero from_pretrained, zero LoRA over external weights).
5. 100% offline operating model (no Kimi/OpenAI API keys required).
6. Teacher external calls disabled by default.
7. Smoke test checkpoints unpromotable to production.
8. Exact baseline parameter reconciliation (86,133,504 parameters).
9. Dataset authenticity and scale threshold preservation.
10. User memory training isolation and permanent ban on live weight updates.
"""

import os
import sys
import json
import re
import unittest
import yaml
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT
from core.policy import (
    validate_student_initialization,
    ACTIVE_OWNERSHIP_POLICY,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT
)
from teacher.providers.kimi import KimiProvider
from teacher.providers.base import TeacherProviderError
from teacher.providers.mock import MockTeacherProvider
from evaluation.teacher_student_eval import PromotionGate
from core.memory.user_memory import UserMemoryManager
from core.learning_lifecycle.lifecycle import ControlledLearningLifecycle
from training.chat.readiness_gate import ChatReadinessGate


class TestStage2131SecurityAndOffline(unittest.TestCase):
    """Rigorous security and offline verification for Stage 2.1.3.1."""

    def setUp(self):
        self.notebook_path = os.path.join(PROJECT_ROOT, "notebooks", "NanoGPT_Training_Pipeline.ipynb")
        self.config_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")

    # 1. Notebook Credential Sanitization
    def test_notebook_credentials_sanitized(self):
        """Training and deployment notebooks must contain ZERO hardcoded tokens or API keys."""
        notebook_paths = [
            os.path.join(PROJECT_ROOT, "notebooks", "NanoGPT_Training_Pipeline.ipynb"),
            os.path.join(PROJECT_ROOT, "Deploy_Moon_AI_Cloud_GPU.ipynb"),
            os.path.join(PROJECT_ROOT, "Train_Super_AI_Colab.ipynb"),
        ]

        # Regex searching for hardcoded assignments of tokens/secrets with string literals
        raw_secret_pat = re.compile(
            r'([A-Za-z0-9_]*(?:auth[_-]?token|api[_-]?key|secret[_-]?key)[A-Za-z0-9_]*)\s*=\s*["\']([a-zA-Z0-9_\-]{16,})["\']',
            re.IGNORECASE
        )

        for path in notebook_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            for idx, cell in enumerate(nb.get("cells", [])):
                src = "".join(cell.get("source", []))
                matches = raw_secret_pat.findall(src)
                for var, val in matches:
                    val_lower = val.lower()
                    # Filter out placeholders
                    if any(dummy in val_lower for dummy in ["placeholder", "example", "dummy", "your_", "test_", "mock", "redacted", "sk-...2211"]):
                        continue
                    self.fail(f"Hardcoded credential detected in {path} cell {idx} variable {var}!")

    # 2. Notebook CORS Configuration
    def test_notebook_cors_restricted(self):
        """Training pipeline notebook must NOT use wildcard CORS (allow_origins=['*'])."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            self.assertNotIn(
                'allow_origins=["*"]',
                src,
                f"Wildcard CORS detected in {self.notebook_path} cell {idx}! Must use restricted origins."
            )

    # 3. Public Tunnel Disabled By Default
    def test_notebook_public_tunnel_disabled_by_default(self):
        """Notebook must explicitly disable public ngrok tunnels by default."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        all_text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        self.assertIn(
            "ENABLE_NGROK_TUNNEL = False",
            all_text,
            "ENABLE_NGROK_TUNNEL = False must be explicitly set in the training notebook."
        )

    # 4. Student Scratch Initialization in Notebook
    def test_notebook_student_scratch_initialization(self):
        """Notebook must initialize student from random scratch weights using Moon's GPT, not pretrained LoRA."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        code_text = "\n".join("".join(c.get("source", [])) for c in code_cells)

        # Moon architecture must be used
        self.assertIn("from model import GPTConfig, GPT", code_text)
        self.assertIn("validate_student_initialization", code_text)

        # Prohibited third-party loading in student path
        self.assertNotIn("unsloth/Llama-3.2-1B-Instruct", code_text)
        self.assertNotIn("FastLanguageModel.from_pretrained", code_text)
        self.assertNotIn("FastLanguageModel.get_peft_model", code_text)

    # 5. Offline Operation Without API Keys
    def test_offline_operation_without_api_keys(self):
        """Moon AI student pipeline must function 100% offline without any API keys."""
        # Unset keys if present in env for this test
        orig_kimi = os.environ.pop("KIMI_API_KEY", None)
        orig_moon = os.environ.pop("MOON_TEACHER_API_KEY", None)
        orig_openai = os.environ.pop("OPENAI_API_KEY", None)

        try:
            # 1. Policy check works offline
            valid, msg = validate_student_initialization("random_gaussian_scratch")
            self.assertTrue(valid)

            # 2. Model initialization works offline
            cfg = GPTConfig(block_size=64, vocab_size=512, n_layer=2, n_head=2, n_embd=64, bias=False)
            model = GPT(cfg)
            x = torch.randint(0, 512, (1, 16))
            out = model.generate(x, max_new_tokens=4)
            self.assertEqual(out.shape[1], 20)

            # 3. Mock teacher works completely offline
            mock_p = MockTeacherProvider()
            res = mock_p.generate_completion("Test prompt")
            self.assertIn("Mock teacher demonstration", res["text"])

        finally:
            if orig_kimi is not None:
                os.environ["KIMI_API_KEY"] = orig_kimi
            if orig_moon is not None:
                os.environ["MOON_TEACHER_API_KEY"] = orig_moon
            if orig_openai is not None:
                os.environ["OPENAI_API_KEY"] = orig_openai

    # 6. Teacher External Calls Disabled By Default
    def test_teacher_external_calls_disabled_by_default(self):
        """KimiProvider must have allow_external_calls=False by default and fail closed."""
        provider = KimiProvider()
        self.assertFalse(provider.allow_external_calls)

        # Attempting API execution must raise TeacherProviderError
        with self.assertRaises(TeacherProviderError) as ctx:
            provider.generate_completion("Hello")
        self.assertIn("External teacher API calls are disabled by default", str(ctx.exception))

        # Teacher YAML config must declare allow_external_calls: false
        teacher_cfg_path = os.path.join(PROJECT_ROOT, "teacher", "config", "teacher_config.yaml")
        with open(teacher_cfg_path, "r", encoding="utf-8") as f:
            t_cfg = yaml.safe_load(f)
        self.assertFalse(t_cfg.get("allow_external_calls", True))

    # 7. Smoke Test Checkpoints Unpromotable to Production
    def test_smoke_test_checkpoints_unpromotable(self):
        """Engineering smoke test checkpoints can NEVER pass the promotion gate."""
        # Case A: is_production_eligible=False
        eval_a = {
            "mean_composite_score": 99.0,
            "category_scores": {"safety_uncertainty": 100.0, "math_reasoning": 95.0, "programming": 95.0, "logical_reasoning": 95.0},
            "is_production_eligible": False,
            "status": "ENGINEERING_SMOKE_TEST_ONLY"
        }
        promoted_a, reason_a, _ = PromotionGate.evaluate_promotion(eval_a)
        self.assertFalse(promoted_a)
        self.assertIn("smoke test", reason_a.lower())

        # Case B: promotable=False
        eval_b = {
            "mean_composite_score": 99.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "promotable": False
        }
        promoted_b, reason_b, _ = PromotionGate.evaluate_promotion(eval_b)
        self.assertFalse(promoted_b)
        self.assertIn("smoke test", reason_b.lower())

        # Case C: is_smoke_test=True
        eval_c = {
            "mean_composite_score": 99.0,
            "category_scores": {"safety_uncertainty": 100.0},
            "is_smoke_test": True
        }
        promoted_c, reason_c, _ = PromotionGate.evaluate_promotion(eval_c)
        self.assertFalse(promoted_c)
        self.assertIn("smoke test", reason_c.lower())

    # 8. Exact Parameter Reconciliation (86,133,504)
    def test_parameter_reconciliation_86m(self):
        """Baseline model config must reconcile 12-layer 768-dim GPT-2 hidden scale to 86,133,504 exact parameters."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Both baseline_model and backward-compatible baseline_124m alias must be present
        self.assertIn("baseline_model", cfg)
        self.assertIn("baseline_124m", cfg)

        self.assertEqual(cfg["baseline_model"]["exact_parameters"], 86133504)
        self.assertEqual(cfg["baseline_124m"]["exact_parameters"], 86133504)

        # Verify theoretical computation from GPT architecture
        # vocab_size=512, n_embd=768, n_layer=12, n_head=12, block_size=1024
        # wte: 512 * 768 = 393,216
        # wpe: 1024 * 768 = 786,432
        # per layer:
        #   ln_1: 768 (bias=False)
        #   c_attn: 768 * (3 * 768) = 1,769,472
        #   c_proj: 768 * 768 = 589,824
        #   ln_2: 768
        #   c_fc: 768 * (4 * 768) = 2,359,296
        #   c_proj: (4 * 768) * 768 = 2,359,296
        #   per layer total: 768 + 1,769,472 + 589,824 + 768 + 2,359,296 + 2,359,296 = 7,079,424
        # 12 layers: 12 * 7,079,424 = 84,953,088
        # ln_f: 768
        # lm_head: tied with wte (0 additional parameters)
        # Total: 393,216 + 786,432 + 84,953,088 + 768 = 86,133,504
        baseline_cfg = GPTConfig(
            block_size=1024,
            vocab_size=512,
            n_layer=12,
            n_head=12,
            n_embd=768,
            bias=False
        )
        baseline_gpt = GPT(baseline_cfg)
        actual_param_count = sum(p.numel() for p in baseline_gpt.parameters())
        self.assertEqual(actual_param_count, 86133504)

    # 9. Dataset Authenticity & Readiness Scale Thresholds
    def test_dataset_authenticity_and_readiness(self):
        """Manifests must contain zero synthetic fixtures; gate must report ENGINEERING_SMOKE_TEST_ONLY."""
        gate = ChatReadinessGate()
        report = gate.run_all_checks()

        # Scale status must be ENGINEERING_SMOKE_TEST_ONLY (22 records < 500 records)
        self.assertEqual(report["status"], "ENGINEERING_SMOKE_TEST_ONLY")
        self.assertTrue(report["smoke_test_passed"])
        self.assertFalse(report["passed"])

        # Provenance and licensing check must pass
        prov_check = report["checks"].get("provenance_and_licensing", {})
        self.assertTrue(prov_check.get("passed", False))

        # Check raw manifests directly for 0 dev fixtures
        train_path = os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl")
        with open(train_path, "r", encoding="utf-8") as f:
            train_recs = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(all(r.get("is_dev_fixture") is False for r in train_recs))

    # 10. User Memory Training Isolation & Ban on Live Updates
    def test_user_memory_training_isolation_and_live_update_ban(self):
        """User memory entries must have eligible_for_training=False; live weight updates must be banned."""
        # 1. User memory isolation
        mem = UserMemoryManager()
        mem.set_user_memory_consent("usr_42", True)
        ok, msg = mem.store_memory("usr_42", "favorite_topic", "astrophysics")
        self.assertTrue(ok)
        stored_item = mem._store["usr_42"]["favorite_topic"]
        self.assertFalse(stored_item.get("eligible_for_training", True))

        # 2. Live conversational weight updates ban
        with self.assertRaises(ValueError) as ctx:
            ControlledLearningLifecycle(allow_live_weight_updates=True)
        self.assertIn("Live model weight updates during conversations are forbidden", str(ctx.exception))

        lifecycle = ControlledLearningLifecycle(allow_live_weight_updates=False)
        self.assertFalse(lifecycle.allow_live_weight_updates)


if __name__ == "__main__":
    unittest.main()
