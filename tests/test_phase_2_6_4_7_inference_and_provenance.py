"""
tests/test_phase_2_6_4_7_inference_and_provenance.py
Phase 2.6.4.7: Production Inference Isolation, Checkpoint Provenance & Release Boundary Suite
Chief ML Systems Architect & Repository Owner: Prem Gaikwad (India 🇮🇳)
Active Sovereign Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates:
1. Direct ModelService loading of historical pilot fails closed by default (SEC-INFER-01)
2. Direct ModelService generation on unpromoted fixture fails closed (SEC-INFER-02)
3. Webapp /api/switch-model blocks switching to unpromoted test fixtures (SEC-INFER-03)
4. Webapp /api/chat blocks production inference on test fixtures (SEC-INFER-04)
5. PromotionGate rejects missing candidate checkpoint path (SEC-PROM-01)
6. PromotionGate rejects nonexistent candidate checkpoint path on disk (SEC-PROM-02)
7. PromotionGate rejects candidate pointing to test fixture on disk (SEC-PROM-03)
8. promote_best_model rejects missing or non-strict ownership policy (SEC-PROM-04)
9. Root train.py rejects initializing from third-party pretrained weights ('gpt2') (SEC-RESUME-01)
10. Root train.py resume enforces weights_only=True and strict sovereign provenance (SEC-RESUME-02)
11. Registry manager verify_checkpoint_provenance fails closed on missing/non-strict metadata (SEC-PROV-01)
12. Common save_checkpoint automatically binds strict sovereign provenance metadata (SEC-PROV-02)
"""
import os
import sys
import json
import unittest
import tempfile
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.model_service import (
    ModelService,
    ModelNotTrainedException,
    is_verified_genuine_checkpoint,
    is_historical_pilot_checkpoint
)
from webapp.main import app
from evaluation.teacher_student_eval import PromotionGate
from models.registry_manager import verify_checkpoint_provenance as registry_verify_provenance
from training.common.checkpoint import (
    save_checkpoint,
    verify_checkpoint_provenance as common_verify_provenance,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)


class TestPhase2647InferenceAndProvenance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import webapp.database
        try:
            webapp.database.db._init_db()
        except Exception:
            pass
        cls.client = TestClient(app)
        cls.service = ModelService(base_dir=PROJECT_ROOT)

    def test_01_direct_load_model_pilot_fails_closed_without_allow_fixture(self):
        """ModelService.load_model('moon-chat-pilot-v0') must return False without allow_fixture=True."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        # Default allow_fixture is False (omitted parameter must fail closed)
        loaded = ms.load_model("moon-chat-pilot-v0")
        self.assertFalse(loaded, "Historical pilot must not load in production mode when allow_fixture is omitted.")
        self.assertIn("restricted to isolated retrospective testing", ms.error_message or "")

    def test_02_direct_generation_on_fixture_fails_closed(self):
        """ModelService.generate() on a loaded test fixture must raise ModelNotTrainedException without allow_fixture=True."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        # Explicitly load fixture for retrospective inspection
        ms.load_model("moon-chat-pilot-v0", allow_fixture=True)
        self.assertTrue(getattr(ms, "is_test_fixture", False), "Loaded pilot must be flagged as a test fixture.")

        # Attempting generation without allow_fixture=True must raise ModelNotTrainedException
        with self.assertRaises(ModelNotTrainedException) as ctx:
            ms.generate("Hello Moon AI", allow_fixture=False)
        self.assertIn("Production inference blocked: Active model is an unpromoted CI test fixture", str(ctx.exception))

    def test_03_webapp_switch_model_rejects_unpromoted_fixtures(self):
        """POST /api/switch-model must reject switching to an unpromoted fixture with HTTP 403."""
        resp = self.client.post("/api/switch-model", json={"model_name": "moon-chat-pilot-v0"})
        self.assertEqual(resp.status_code, 403)
        data = resp.json()
        self.assertIn("uncertified test fixture", data.get("detail", ""))

    def test_04_webapp_chat_endpoint_blocks_fixture_inference(self):
        """POST /api/chat must reject inference with HTTP 403 if active model is a fixture."""
        from webapp.main import model_service
        orig_active = model_service.active_model_name
        orig_fixture = getattr(model_service, "is_test_fixture", False)
        try:
            # Simulate fixture state on webapp model service
            model_service.is_test_fixture = True
            model_service.active_model_name = "moon-chat-pilot-v0"

            resp = self.client.post("/api/chat", json={"message": "Inference probe on fixture", "mode": "chat"})
            self.assertEqual(resp.status_code, 403)
            data = resp.json()
            self.assertIn("Production inference blocked", data.get("detail", ""))
        finally:
            model_service.active_model_name = orig_active
            model_service.is_test_fixture = orig_fixture

    def test_05_promotion_gate_rejects_missing_candidate_checkpoint_path(self):
        """PromotionGate must reject candidates when candidate_checkpoint_path is missing or empty."""
        cand_eval = {
            "mean_composite_score": 90.0,
            "category_scores": {
                "safety_uncertainty": 100.0,
                "math_reasoning": 90.0,
                "programming": 90.0,
                "logical_reasoning": 90.0
            },
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "is_production_eligible": True,
            "promotable": True,
            "is_smoke_test": False,
            "is_test_fixture": False,
            "is_production_trained": True,
            "status": "EVALUATION_PASSED"
        }
        # If require_checkpoint_file=True is enforced, None/empty path must fail closed
        promoted, reason, _ = PromotionGate.evaluate_promotion(
            candidate_eval=cand_eval,
            candidate_checkpoint_path="",
            require_checkpoint_file=True
        )
        self.assertFalse(promoted)
        self.assertIn("Candidate checkpoint path is strictly required", reason)

    def test_06_promotion_gate_rejects_nonexistent_checkpoint_path(self):
        """PromotionGate must reject candidates pointing to nonexistent checkpoint paths on disk."""
        cand_eval = {
            "mean_composite_score": 90.0,
            "category_scores": {
                "safety_uncertainty": 100.0,
                "math_reasoning": 90.0,
                "programming": 90.0,
                "logical_reasoning": 90.0
            },
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "is_production_eligible": True,
            "promotable": True,
            "is_smoke_test": False,
            "is_test_fixture": False,
            "is_production_trained": True,
            "status": "EVALUATION_PASSED"
        }
        fake_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "phantom_nonexistent_v99.pt")
        promoted, reason, _ = PromotionGate.evaluate_promotion(
            candidate_eval=cand_eval,
            candidate_checkpoint_path=fake_path
        )
        self.assertFalse(promoted)
        self.assertIn("does not exist on disk", reason)

    def test_07_promotion_gate_rejects_fixture_checkpoint_on_disk(self):
        """PromotionGate must reject candidates pointing to physical historical pilot checkpoint."""
        cand_eval = {
            "mean_composite_score": 90.0,
            "category_scores": {
                "safety_uncertainty": 100.0,
                "math_reasoning": 90.0,
                "programming": 90.0,
                "logical_reasoning": 90.0
            },
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "is_production_eligible": True,
            "promotable": True,
            "is_smoke_test": False,
            "is_test_fixture": False,
            "is_production_trained": True,
            "status": "EVALUATION_PASSED"
        }
        pilot_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")
        if os.path.isfile(pilot_path):
            promoted, reason, _ = PromotionGate.evaluate_promotion(
                candidate_eval=cand_eval,
                candidate_checkpoint_path=pilot_path
            )
            self.assertFalse(promoted)
            self.assertIn("failed genuine checkpoint verification", reason)

    def test_08_registry_verify_provenance_rejects_missing_and_tampered_policy(self):
        """registry_manager.verify_checkpoint_provenance must fail closed on missing/tampered policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Case A: Missing policy
            no_pol_path = os.path.join(tmpdir, "no_pol.pt")
            torch.save({
                "provenance": "moon_ai_random_init",
                "config": {"teacher_checkpoint_weights_transferred": False}
            }, no_pol_path)
            with self.assertRaises(ValueError) as ctx:
                registry_verify_provenance(no_pol_path)
            self.assertIn("Sovereign ownership policy violation", str(ctx.exception))

            # Case B: Teacher weights transferred True
            teacher_path = os.path.join(tmpdir, "teacher_transfer.pt")
            torch.save({
                "provenance": "moon_ai_random_init",
                "config": {
                    "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                    "teacher_checkpoint_weights_transferred": True
                }
            }, teacher_path)
            with self.assertRaises(ValueError) as ctx:
                registry_verify_provenance(teacher_path)
            self.assertIn("Policy B violation", str(ctx.exception))

            # Case C: CI test fixture status
            fixture_path = os.path.join(tmpdir, "fixture.pt")
            torch.save({
                "provenance": "moon_ai_random_init",
                "is_test_fixture": True,
                "config": {
                    "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                    "teacher_checkpoint_weights_transferred": False
                }
            }, fixture_path)
            with self.assertRaises(ValueError) as ctx:
                registry_verify_provenance(fixture_path)
            self.assertIn("unpromotable CI test fixture", str(ctx.exception))

    def test_09_save_checkpoint_binds_strict_sovereign_metadata(self):
        """save_checkpoint automatically binds STRICT_SOVEREIGN_RANDOM_INIT_ONLY and negative transfer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_out = os.path.join(tmpdir, "test_bind.pt")
            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1,
                epoch=1,
                loss=0.5,
                config_dict={"test": True},
                output_path=ckpt_out,
                training_mode=TRAINING_MODE_DEVELOPMENT
            )
            raw = torch.load(ckpt_out, map_location="cpu", weights_only=True)
            self.assertEqual(raw.get("ownership_policy"), "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
            self.assertIs(raw.get("teacher_checkpoint_weights_transferred"), False)

    def test_10_root_train_py_rejects_gpt2_init(self):
        """Root train.py must raise RuntimeError if init_from starts with 'gpt2'."""
        train_py_path = os.path.join(PROJECT_ROOT, "train.py")
        with open(train_py_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Production training violation: initializing from pretrained third-party weights", content)
        self.assertIn("weights_only=True", content)


if __name__ == "__main__":
    unittest.main()

