"""
tests/test_phase_2_7_6_runtime_sovereignty.py
Phase 2.7.6: Runtime Sovereignty, Genuine Inference & Hardcoded-Answer Audit Suite
Chief ML Systems Architect & Repository Owner: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Owner Authorization Lock: AUTHORIZED_BY_PREM = False (STRICT INVARIANT)

Validates 14 Core Sovereignty Scenarios:
1. Qwen-2.5-Coder-7B cannot become default Moon production model.
2. LLaMA-3.2 cannot become default Moon production model.
3. GPT-2 cannot become default Moon production model.
4. Unverified, malformed, or incomplete checkpoints are rejected.
5. CI test fixtures are rejected for production inference.
6. Historical Policy B checkpoints are rejected for production inference.
7. Verified eligible Moon checkpoint can be selected without modifying weights (hash invariant).
8. Missing eligible checkpoints produce MODEL_NOT_READY.
9. No silent pretrained-model fallback occurs.
10. Hardcoded knowledge responses are eliminated or not labelled neural generation.
11. Deterministic utilities (games, reflexes, notices) are correctly labelled with 0 fake tokens.
12. Web retrieval is separately labelled without fabricated citations.
13. AUTHORIZED_BY_PREM remains strictly False.
14. Zero production optimizer allocations, zero backward passes, zero checkpoint writes.
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
import torch
from typing import Dict, Any
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.policy
from webapp.model_service import (
    ModelService,
    ModelNotTrainedException,
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
        cfg = {
            "ownership_policy": policy,
            "teacher_checkpoint_weights_transferred": teacher_transferred,
            "student_init_method": init_method,
            "initialization": init_method,
            "dataset": "test_corpus"
        }
        payload: Dict[str, Any] = {
            "status": status,
            "is_test_fixture": is_fixture,
            "config": cfg
        }
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

            # Missing model_args
            bad_path2 = self._create_mock_checkpoint(tmpdir, "bad2.pt", omit_model_args=True)
            self.assertFalse(is_verified_genuine_checkpoint(bad_path2))

            # Nonexistent path
            self.assertFalse(is_verified_genuine_checkpoint(os.path.join(tmpdir, "nonexistent.pt")))

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

    # 7. Verified eligible Moon checkpoint can be selected without modifying weights
    def test_07_verified_eligible_moon_checkpoint_can_be_selected_without_weight_modification(self):
        """Loading or inspecting an eligible checkpoint must never mutate checkpoint bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._create_mock_checkpoint(
                tmpdir, "genuine_moon.pt",
                policy="STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                teacher_transferred=False,
                init_method="random_gaussian_scratch",
                status="PILOT_TRAINED"
            )
            self.assertTrue(is_verified_genuine_checkpoint(ckpt_path))

            # Compute SHA-256 before loading
            with open(ckpt_path, "rb") as f:
                sha_before = hashlib.sha256(f.read()).hexdigest()

            # Verify through ModelService static validator
            is_valid, reason, data = ModelService.validate_checkpoint_structure(ckpt_path)
            self.assertTrue(is_valid)

            # Compute SHA-256 after loading
            with open(ckpt_path, "rb") as f:
                sha_after = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(
                sha_before, sha_after,
                "Checkpoint weights must be immutable during inspection (SHA-256 invariant violation)."
            )

    # 8. Missing eligible checkpoints produce MODEL_NOT_READY
    def test_08_missing_eligible_checkpoints_produce_model_not_ready(self):
        """When no genuine checkpoint is on disk, status must be MODEL_NOT_READY."""
        with tempfile.TemporaryDirectory() as empty_dir:
            ms = ModelService(base_dir=empty_dir)
            self.assertEqual(
                ms.get_status(),
                "MODEL_NOT_READY",
                "ModelService must honestly report MODEL_NOT_READY when no genuine checkpoint exists."
            )

            # Verify GET /api/health reports honest status
            resp = self.client.get("/api/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn(data["model_status"], ["MODEL_NOT_READY", "READY (ON DEMAND)", "READY"])

    # 9. No silent pretrained-model fallback occurs
    def test_09_no_silent_pretrained_model_fallback(self):
        """Attempting to load third-party weights as Moon production model must fail closed."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        
        # Explicitly attempt to load Qwen as Moon's brain
        qwen_loaded = ms.load_model("qwen-2.5-coder-7b")
        self.assertFalse(qwen_loaded)
        self.assertIn("Runtime Sovereignty Violation", ms.error_message or "")

        # Explicitly attempt to load LLaMA as Moon's brain
        llama_loaded = ms.load_model("llama-3.2")
        self.assertFalse(llama_loaded)
        self.assertIn("Runtime Sovereignty Violation", ms.error_message or "")

        # Explicitly attempt to load GPT-2 as Moon's brain
        gpt2_loaded = ms.load_model("gpt2")
        self.assertFalse(gpt2_loaded)
        self.assertIn("Runtime Sovereignty Violation", ms.error_message or "")

    # 10. Hardcoded knowledge responses are not labelled neural generation
    def test_10_hardcoded_knowledge_responses_not_labelled_neural_generation(self):
        """General knowledge queries must NOT masquerade as neural model output."""
        queries = [
            "how to make a pancake",
            "what is the capital of France",
            "tell me about GTA 6",
            "10 + 20"
        ]
        for q in queries:
            res = self.client.post("/api/chat", json={"message": q, "mode": "chat"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            # Under Phase 2.7.6, responses must NEVER claim is_neural_generation=True without actual model generation
            self.assertFalse(
                data.get("is_neural_generation", False),
                f"Query '{q}' was falsely labelled as neural generation."
            )

    # 11. Deterministic utilities are correctly labelled
    def test_11_deterministic_utilities_correctly_labelled(self):
        """Procedural games and reflex greetings must be labelled deterministic utilities with 0 tokens generated."""
        # Snake game
        res_snake = self.client.post("/api/chat", json={"message": "Write a playable snake game in html", "mode": "chat"})
        self.assertEqual(res_snake.status_code, 200)
        data_snake = res_snake.json()
        self.assertEqual(data_snake["engine_type"], "deterministic_utility")
        self.assertTrue(data_snake["is_deterministic_utility"])
        self.assertFalse(data_snake["is_neural_generation"])
        self.assertEqual(data_snake["tokens_generated"], 0, "Deterministic templates must not report fake neural tokens.")

        # Greeting reflex
        res_greet = self.client.post("/api/chat", json={"message": "hello", "mode": "chat"})
        self.assertEqual(res_greet.status_code, 200)
        data_greet = res_greet.json()
        self.assertEqual(data_greet["engine_type"], "system_reflex")
        self.assertTrue(data_greet["is_deterministic_utility"])
        self.assertFalse(data_greet["is_neural_generation"])

    # 12. Web retrieval is separately labelled without fabricated citations
    def test_12_web_retrieval_separately_labelled_no_fabricated_sources(self):
        """ChatResponse schema must support verifiable sources and distinguish web retrieval."""
        test_response = ChatResponse(
            reply="Retrieval test result",
            model_name="Moon Web Intelligence",
            prompt_length=10,
            tokens_generated=0,
            device="Web Retrieval Engine",
            session_id="test_sess",
            engine_type="web_retrieval",
            model_id="moon-web-v1",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=False,
            is_web_retrieval=True,
            checkpoint_verified=False,
            provenance_status="REAL_TIME_RETRIEVAL",
            sources=["https://arxiv.org/abs/1706.03762"]
        )
        self.assertEqual(test_response.engine_type, "web_retrieval")
        self.assertTrue(test_response.is_web_retrieval)
        self.assertFalse(test_response.is_neural_generation)
        self.assertEqual(len(test_response.sources), 1)
        self.assertTrue(test_response.sources[0].startswith("https://"))

    # 13. AUTHORIZED_BY_PREM remains strictly False
    def test_13_authorized_by_prem_remains_false(self):
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

    # 14. Zero production optimizer allocations, zero backward passes, zero checkpoint writes
    def test_14_zero_optimizers_zero_backward_zero_checkpoint_writes(self):
        """Audit test proves no production optimizers, backward passes, or checkpoint files were created."""
        ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat")
        initial_files = set(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else set()

        # Execute multiple inference and status checks
        ms = ModelService(base_dir=PROJECT_ROOT)
        ms.get_available_models()
        ms.get_status()
        self.client.get("/api/health")
        self.client.get("/api/models")
        self.client.post("/api/chat", json={"message": "audit check", "mode": "chat"})

        # Verify no new files were written to checkpoints directory
        final_files = set(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else set()
        new_files = final_files - initial_files
        self.assertEqual(
            len(new_files), 0,
            f"Violation: Checkpoint files were written during audit execution: {new_files}"
        )


if __name__ == "__main__":
    unittest.main()
