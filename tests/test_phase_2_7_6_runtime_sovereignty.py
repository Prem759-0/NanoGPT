"""
tests/test_phase_2_7_6_runtime_sovereignty.py
Phase 2.7.6 & Phase 2.7.6.1: Runtime Sovereignty, Gate & Checkpoint Provenance Audit Suite
Chief ML Systems Architect & Repository Owner: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Owner Authorization Lock: AUTHORIZED_BY_PREM = False (STRICT INVARIANT)

Validates 18 Core Sovereignty & Provenance Scenarios:
1. Qwen-2.5-Coder-7B cannot become default Moon production model.
2. LLaMA-3.2 cannot become default Moon production model.
3. GPT-2 cannot become default Moon production model.
4. Unverified, malformed, or incomplete checkpoints fail verification.
5. CI test fixtures are rejected for production inference.
6. Historical Policy B checkpoints are rejected for production inference.
7. Synthetic checkpoint with random weights fails genuine training provenance validation.
8. /api/health reports exact MODEL_NOT_READY status when no eligible checkpoint exists.
9. Checkpoint file's mere presence does not enable scratch production inference.
10. Switching to scratch is blocked without a verified genuine checkpoint (HTTP 403).
11. Environment-variable checkpoint overrides (NANOGPT_CHECKPOINT_PATH) do not bypass provenance.
12. Qwen, LLaMA, and GPT-2 remain strictly blocked.
13. Greetings and creator reflexes report zero neural tokens (tokens_generated = 0).
14. Creator identity text does not claim an untrained model has completed training.
15. Deterministic utilities, games, and visual generators report zero neural tokens.
16. General chat with no eligible checkpoint returns honest MODEL_NOT_READY notice.
17. AUTHORIZED_BY_PREM remains strictly False.
18. Instrumented check confirms 0 optimizer instantiations, 0 backward passes, 0 checkpoint writes.
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.policy
from webapp.model_service import (
    ModelService,
    ModelNotTrainedException,
    classify_checkpoint,
    is_verified_genuine_checkpoint,
    is_historical_pilot_checkpoint,
    is_checkpoint_fixture
)
from webapp.schemas import ChatResponse, HealthResponse
from webapp.main import app, model_service


class TestPhase276RuntimeSovereignty(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import webapp.database
        import webapp.main
        cls.test_db_dir = tempfile.mkdtemp(prefix="phase_2_7_6_db_")
        cls.test_db_path = os.path.join(cls.test_db_dir, "test.sqlite").replace("\\", "/")
        webapp.database.db = webapp.database.Database(db_url=f"sqlite:///{cls.test_db_path}")
        webapp.main.db = webapp.database.db
        webapp.main.db._init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "test_db_dir") and os.path.isdir(cls.test_db_dir):
            try:
                import shutil
                shutil.rmtree(cls.test_db_dir, ignore_errors=True)
            except Exception:
                pass

    def _create_mock_checkpoint(
        self,
        directory: str,
        filename: str,
        policy: str = "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
        teacher_transferred: bool = False,
        init_method: str = "random_gaussian_scratch",
        status: str = "PILOT_TRAINED",
        is_fixture: bool = False,
        is_synthetic: bool = False,
        iter_num: Any = None,
        val_loss: Any = None,
        training_completed: bool = False,
        run_id: Any = None,
        dataset: str = "test_corpus",
        omit_model: bool = False,
        omit_model_args: bool = False
    ) -> str:
        """Helper to create deterministic isolated mock checkpoints for testing."""
        path = os.path.join(directory, filename)
        model_args = {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 64,
            "vocab_size": 256,
            "block_size": 128
        }
        cfg: Dict[str, Any] = {
            "ownership_policy": policy,
            "teacher_checkpoint_weights_transferred": teacher_transferred,
            "student_init_method": init_method,
            "initialization": init_method,
            "dataset": dataset
        }
        if iter_num is not None:
            cfg["iter_num"] = iter_num
        if val_loss is not None:
            cfg["best_val_loss"] = val_loss
            cfg["val_loss"] = val_loss
        if training_completed:
            cfg["training_completed"] = True
        if run_id:
            cfg["training_run_id"] = run_id

        payload: Dict[str, Any] = {
            "status": status,
            "is_test_fixture": is_fixture,
            "is_synthetic": is_synthetic,
            "config": cfg
        }
        if iter_num is not None:
            payload["iter_num"] = iter_num
        if val_loss is not None:
            payload["best_val_loss"] = val_loss
            payload["val_loss"] = val_loss
        if training_completed:
            payload["training_completed"] = True

        if not omit_model:
            payload["model"] = {
                "transformer.wte.weight": torch.randn(256, 64)
            }
        if not omit_model_args:
            payload["model_args"] = model_args

        torch.save(payload, path)
        return path

    # 1. Qwen cannot become default Moon production model
    def test_01_qwen_cannot_become_default_moon_production_model(self):
        """ModelService MUST default strictly to 'moon-chat-pilot-v0', never Qwen-2.5-Coder-7B."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        self.assertNotEqual(
            ms.active_model_name,
            "qwen-2.5-coder-7b",
            "Violation: Qwen must not be selected as the default Moon AI inference model."
        )
        self.assertEqual(
            ms.active_model_name,
            "moon-chat-pilot-v0",
            "Moon AI must default to its sovereign student model 'moon-chat-pilot-v0'."
        )

    # 2. Llama cannot become default Moon production model
    def test_02_llama_cannot_become_default_moon_production_model(self):
        """ModelService MUST NOT default to LLaMA-3.2 under any circumstances."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        self.assertNotEqual(
            ms.active_model_name,
            "llama-3.2",
            "Violation: LLaMA-3.2 must not be selected as the default Moon AI inference model."
        )

    # 3. GPT-2 cannot become default Moon production model
    def test_03_gpt2_cannot_become_default_moon_production_model(self):
        """ModelService MUST NOT default to GPT-2 pretrained weights."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        self.assertNotEqual(
            ms.active_model_name,
            "gpt2",
            "Violation: GPT-2 pretrained weights must not be selected as default Moon AI engine."
        )

    # 4. Unverified checkpoint is rejected
    def test_04_unverified_checkpoint_rejected(self):
        """Incomplete, corrupted, or non-conforming checkpoints fail positive verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Missing model dictionary
            bad_path1 = self._create_mock_checkpoint(tmpdir, "bad1.pt", omit_model=True)
            self.assertFalse(is_verified_genuine_checkpoint(bad_path1))
            self.assertEqual(classify_checkpoint(bad_path1), "STRUCTURALLY_INVALID")

            # Missing model_args
            bad_path2 = self._create_mock_checkpoint(tmpdir, "bad2.pt", omit_model_args=True)
            self.assertFalse(is_verified_genuine_checkpoint(bad_path2))
            self.assertEqual(classify_checkpoint(bad_path2), "STRUCTURALLY_INVALID")

            # Nonexistent path
            self.assertFalse(is_verified_genuine_checkpoint(os.path.join(tmpdir, "nonexistent.pt")))
            self.assertEqual(classify_checkpoint(os.path.join(tmpdir, "nonexistent.pt")), "STRUCTURALLY_INVALID")

    # 5. Test fixture is rejected for production inference
    def test_05_test_fixture_rejected_for_production_inference(self):
        """Checkpoints marked as test fixtures must be rejected by is_verified_genuine_checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = self._create_mock_checkpoint(tmpdir, "fixture.pt", is_fixture=True)
            self.assertTrue(is_checkpoint_fixture(fixture_path))
            self.assertFalse(
                is_verified_genuine_checkpoint(fixture_path),
                "CI test fixture must fail closed on positive genuine verification."
            )
            self.assertEqual(classify_checkpoint(fixture_path), "TEST_FIXTURE")

    # 6. Historical Policy B checkpoint is rejected for production
    def test_06_historical_policy_b_checkpoint_rejected(self):
        """Historical Policy B checkpoints (TEACHER_STUDENT_RANDOM_INIT) must be rejected for strict production."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = self._create_mock_checkpoint(
                tmpdir, "historical.pt",
                policy="TEACHER_STUDENT_RANDOM_INIT",
                status="PILOT_TRAINED"
            )
            self.assertTrue(is_historical_pilot_checkpoint(hist_path))
            self.assertFalse(
                is_verified_genuine_checkpoint(hist_path),
                "Historical Policy B checkpoint must NOT pass strict sovereign production verification."
            )
            self.assertEqual(classify_checkpoint(hist_path), "HISTORICAL_POLICY_B")

    # 7. Synthetic checkpoint fails genuine provenance validation
    def test_07_synthetic_checkpoint_fails_genuine_provenance_validation(self):
        """A synthetic checkpoint made with torch.randn() and mock labels must NOT count as genuine training provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Synthetic checkpoint with random tensors but lacking training telemetry
            synthetic_path = self._create_mock_checkpoint(
                tmpdir, "synthetic_mock.pt",
                policy="STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                teacher_transferred=False,
                init_method="random_gaussian_scratch",
                status="PILOT_TRAINED",
                iter_num=None,  # No training telemetry
                val_loss=None,
                training_completed=False,
                is_synthetic=True
            )
            # Must fail closed: random tensors are not evidence of training provenance
            self.assertFalse(
                is_verified_genuine_checkpoint(synthetic_path),
                "Synthetic checkpoint with random tensors must NOT pass genuine training provenance."
            )
            self.assertIn(classify_checkpoint(synthetic_path), ("UNTRAINED_RANDOM_INIT", "SYNTHETIC_MOCK"))

            # 2. SHA-256 immutability check: inspecting a checkpoint must never mutate bytes
            with open(synthetic_path, "rb") as f:
                sha_before = hashlib.sha256(f.read()).hexdigest()

            is_valid, reason, data = ModelService.validate_checkpoint_structure(synthetic_path)
            self.assertTrue(is_valid)

            with open(synthetic_path, "rb") as f:
                sha_after = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(sha_before, sha_after, "Checkpoint inspection must never modify bytes on disk.")

    # 8. Exact MODEL_NOT_READY health reporting
    def test_08_health_endpoint_reports_exact_model_not_ready(self):
        """When no eligible checkpoint exists on disk, /api/health must report EXACTLY MODEL_NOT_READY."""
        with tempfile.TemporaryDirectory() as empty_dir:
            ms = ModelService(base_dir=empty_dir)
            self.assertEqual(
                ms.get_status(),
                "MODEL_NOT_READY",
                "ModelService must honestly report MODEL_NOT_READY when no genuine checkpoint exists."
            )

        # Query live /api/health
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            data["model_status"],
            "MODEL_NOT_READY",
            "Violation: /api/health must report strictly 'MODEL_NOT_READY' (no ambiguous READY / ON DEMAND permitted)."
        )
        self.assertFalse(data["is_sovereign_ready"], "is_sovereign_ready must be False when no promoted checkpoint exists.")
        self.assertIsNone(data["sovereign_checkpoint_path"], "sovereign_checkpoint_path must be None.")
        self.assertTrue(data["checkpoint_exists"], "checkpoint_exists is True when unpromoted fixtures exist on disk.")

    # 9. Scratch checkpoint mere presence does not enable production inference
    def test_09_scratch_checkpoint_mere_presence_does_not_enable_production_inference(self):
        """A checkpoint file's mere presence on disk must NOT make 'scratch' production-eligible without provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "out-assistant")
            os.makedirs(out_dir, exist_ok=True)
            # Create an unverified dummy ckpt.pt
            self._create_mock_checkpoint(
                out_dir, "ckpt.pt",
                policy="UNAPPROVED_LEGACY_POLICY",
                status="UNVERIFIED_SCRATCH"
            )

            ms = ModelService(base_dir=tmpdir)
            # Positive verification gate must reject it for production
            self.assertIsNone(
                ms.find_checkpoint(allow_fixture=False),
                "Unverified scratch checkpoint must be rejected when allow_fixture=False."
            )

            # Available models list must report scratch as MODEL_NOT_READY
            models = {m["id"]: m for m in ms.get_available_models()}
            self.assertIn("scratch", models)
            self.assertFalse(models["scratch"]["available"])
            self.assertEqual(models["scratch"]["status"], "MODEL_NOT_READY")

    # 10. Switching to scratch blocked without verified checkpoint
    def test_10_switching_to_scratch_blocked_without_verified_checkpoint(self):
        """Attempting to switch to 'scratch' via /api/switch-model must return HTTP 403 when unverified."""
        resp = self.client.post("/api/switch-model", json={"model_name": "scratch"})
        self.assertEqual(
            resp.status_code, 403,
            "Switching to scratch model without a verified genuine checkpoint must be rejected with HTTP 403."
        )

    # 11. Environment variable override does not bypass provenance
    def test_11_environment_variable_override_does_not_bypass_provenance(self):
        """Setting NANOGPT_CHECKPOINT_PATH to an unverified checkpoint must not bypass provenance verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unverified_path = self._create_mock_checkpoint(
                tmpdir, "env_ckpt.pt",
                policy="UNAPPROVED_EXPERIMENTAL",
                status="UNVERIFIED"
            )
            old_env = os.environ.get("NANOGPT_CHECKPOINT_PATH")
            try:
                os.environ["NANOGPT_CHECKPOINT_PATH"] = unverified_path
                ms = ModelService(base_dir=tmpdir)
                self.assertIsNone(
                    ms.find_checkpoint(allow_fixture=False),
                    "NANOGPT_CHECKPOINT_PATH override must NOT bypass strict provenance verification."
                )
            finally:
                if old_env is not None:
                    os.environ["NANOGPT_CHECKPOINT_PATH"] = old_env
                else:
                    os.environ.pop("NANOGPT_CHECKPOINT_PATH", None)

    # 12. Qwen, LLaMA, and GPT-2 remain strictly blocked
    def test_12_qwen_llama_gpt2_remain_blocked(self):
        """Third-party pretrained models remain strictly blocked from production inference."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        for model_id in ["qwen-2.5-coder-7b", "llama-3.2", "gpt2"]:
            loaded = ms.load_model(model_id, allow_fixture=False)
            self.assertFalse(loaded, f"Loading {model_id} must fail closed.")
            self.assertIn("Runtime Sovereignty Violation", ms.error_message or "")

    # 13. Greetings and creator reflexes report zero neural tokens
    def test_13_greetings_and_creator_reflexes_report_zero_neural_tokens(self):
        """Greetings and creator reflexes must report tokens_generated = 0 and is_neural_generation = False."""
        # Greeting
        res_greet = self.client.post("/api/chat", json={"message": "hello", "mode": "chat"})
        self.assertEqual(res_greet.status_code, 200)
        data_greet = res_greet.json()
        self.assertEqual(data_greet["engine_type"], "system_reflex")
        self.assertFalse(data_greet["is_neural_generation"])
        self.assertTrue(data_greet["is_deterministic_utility"])
        self.assertEqual(data_greet["tokens_generated"], 0, "Greeting reflex must report 0 neural tokens.")

        # Creator query
        res_creator = self.client.post("/api/chat", json={"message": "who created you", "mode": "chat"})
        self.assertEqual(res_creator.status_code, 200)
        data_creator = res_creator.json()
        self.assertEqual(data_creator["engine_type"], "system_reflex")
        self.assertFalse(data_creator["is_neural_generation"])
        self.assertTrue(data_creator["is_deterministic_utility"])
        self.assertEqual(data_creator["tokens_generated"], 0, "Creator reflex must report 0 neural tokens.")

    # 14. Creator identity does not claim untrained model is trained
    def test_14_creator_identity_does_not_claim_untrained_model_is_trained(self):
        """Creator identity response must NOT claim the model is 'trained purely from scratch' when untrained."""
        res = self.client.post("/api/chat", json={"message": "who is your developer", "mode": "chat"})
        self.assertEqual(res.status_code, 200)
        reply = res.json()["reply"]
        self.assertNotIn(
            "trained purely from scratch",
            reply.lower(),
            "Violation: Identity response must not claim student core is already trained from scratch."
        )
        self.assertIn(
            "model_not_ready",
            reply.lower(),
            "Identity response must honestly reflect MODEL_NOT_READY status."
        )

    # 15. Deterministic utilities report zero neural tokens
    def test_15_deterministic_utilities_report_zero_neural_tokens(self):
        """Procedural games and visual generator routes must report tokens_generated = 0 and is_neural_generation = False."""
        # Snake game
        res_snake = self.client.post("/api/chat", json={"message": "play snake game in html", "mode": "chat"})
        self.assertEqual(res_snake.status_code, 200)
        d_snake = res_snake.json()
        self.assertEqual(d_snake["engine_type"], "deterministic_utility")
        self.assertFalse(d_snake["is_neural_generation"])
        self.assertEqual(d_snake["tokens_generated"], 0)

        # Procedural image intent
        res_img = self.client.post("/api/chat", json={"message": "generate an image of a cybernetic tiger", "mode": "chat"})
        self.assertEqual(res_img.status_code, 200)
        d_img = res_img.json()
        self.assertFalse(d_img["is_neural_generation"])
        self.assertEqual(d_img["tokens_generated"], 0)

        # Procedural video inquiry
        res_vid = self.client.post("/api/chat", json={"message": "what about video", "mode": "chat"})
        self.assertEqual(res_vid.status_code, 200)
        d_vid = res_vid.json()
        self.assertFalse(d_vid["is_neural_generation"])
        self.assertEqual(d_vid["tokens_generated"], 0)

    # 16. General chat returns MODEL_NOT_READY for scratch and pilot
    def test_16_general_chat_returns_model_not_ready_for_scratch_and_pilot(self):
        """Unrecognized / reasoning questions must return truthful MODEL_NOT_READY notice."""
        queries = [
            "Explain quantum computing in detail",
            "Write a poem about the ocean"
        ]
        for q in queries:
            res = self.client.post("/api/chat", json={"message": q, "mode": "chat"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine_type"], "system_notice")
            self.assertEqual(data["model_status"], "MODEL_NOT_READY")
            self.assertFalse(data["is_neural_generation"])
            self.assertEqual(data["tokens_generated"], 0)
            self.assertIn("MODEL_NOT_READY", data["reply"])

    # 17. AUTHORIZED_BY_PREM remains strictly False
    def test_17_authorized_by_prem_remains_strictly_false(self):
        """The owner authorization lock MUST remain False."""
        self.assertFalse(
            core.policy.AUTHORIZED_BY_PREM,
            "CRITICAL INVARIANT VIOLATION: AUTHORIZED_BY_PREM must remain strictly False."
        )
        self.assertEqual(
            core.policy.ACTIVE_OWNERSHIP_POLICY,
            "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "ACTIVE_OWNERSHIP_POLICY must remain STRICT_SOVEREIGN_RANDOM_INIT_ONLY."
        )

    # 18. Instrumented zero optimizers, zero backward, zero checkpoint writes
    def test_18_instrumented_zero_optimizers_zero_backward_zero_checkpoint_writes(self):
        """Instrumented audit test proves 0 optimizers, 0 backward passes, and 0 checkpoint writes occur."""
        ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat")
        initial_files = set(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else set()

        with patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("Optimizer initialization attempted during audit!")) as mock_opt, \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("Backward pass attempted during audit!")) as mock_bwd:

            # Execute comprehensive suite of API requests
            self.client.get("/api/health")
            self.client.get("/api/models")
            self.client.get("/api/model-info")
            self.client.post("/api/switch-model", json={"model_name": "scratch"})
            self.client.post("/api/chat", json={"message": "hello", "mode": "chat"})
            self.client.post("/api/chat", json={"message": "who created you", "mode": "chat"})
            self.client.post("/api/chat", json={"message": "play snake game in html", "mode": "chat"})
            self.client.post("/api/chat", json={"message": "explain neural networks", "mode": "chat"})

            self.assertEqual(mock_opt.call_count, 0, "Zero optimizers must be constructed.")
            self.assertEqual(mock_bwd.call_count, 0, "Zero backward passes must be executed.")

        final_files = set(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else set()
        new_files = final_files - initial_files
        self.assertEqual(len(new_files), 0, f"Violation: Checkpoint files were written: {new_files}")


if __name__ == "__main__":
    unittest.main()
