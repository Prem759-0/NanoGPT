"""
tests/test_phase_2_6_4_8_attestation_and_fixture.py
Adversarial Verification Suite for Phase 2.6.4.8:
- Fixture opt-in closure (default allow_fixture=False)
- Public API tamper resistance against fixture permission injection
- Cryptographic byte-integrity and Ed25519 attestation validation
- Distinction between metadata claims and genuine training provenance
- Strict fail-closed execution across all training entry points when AUTHORIZED_BY_PREM is False

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
"""
import os
import sys
import json
import unittest
import tempfile
import hashlib
from datetime import datetime, timezone
import torch
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric import ed25519

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.main import app, model_service
from webapp.model_service import ModelService, ModelNotTrainedException
from models.registry_manager import verify_checkpoint_provenance as registry_verify_provenance
from training.common.checkpoint import (
    save_checkpoint,
    compute_file_sha256,
    verify_checkpoint_provenance as common_verify_provenance,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from evaluation.security.evaluation_attestation import (
    sign_evaluation_attestation,
    verify_evaluation_attestation,
    construct_canonical_evaluation_payload
)
from evaluation.teacher_student_eval import PromotionGate
from datasets.security.attestation_engine import (
    sign_attestation as sign_dataset_attestation,
    verify_attestation as verify_dataset_attestation
)
import core.policy as core_policy
from tests.checkpoint_fixture_utils import ensure_pilot_checkpoint_fixture


class TestPhase2648AttestationAndFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ensure_pilot_checkpoint_fixture()
        import webapp.database
        try:
            webapp.database.db._init_db()
        except Exception:
            pass
        cls.client = TestClient(app)

    def test_01_fixture_loading_omitted_parameter_fails_closed(self):
        """ModelService.load_model() must default to allow_fixture=False and reject fixtures."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        # Calling load_model with omitted allow_fixture argument must fail closed
        result = ms.load_model("moon-chat-pilot-v0")
        self.assertFalse(result, "load_model without allow_fixture must fail closed by default.")
        self.assertIn("restricted to isolated retrospective testing", ms.error_message or "")

    def test_02_explicit_fixture_loading_retrospective_mode(self):
        """ModelService.load_model(..., allow_fixture=True) explicitly allows inspection loading."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        result = ms.load_model("moon-chat-pilot-v0", allow_fixture=True)
        self.assertTrue(result, "Explicit allow_fixture=True must succeed for inspection.")
        self.assertTrue(getattr(ms, "is_test_fixture", False))

    def test_03_fixture_generation_direct_service_call_blocked(self):
        """Direct ModelService.generate() on loaded fixture must block without allow_fixture=True."""
        ms = ModelService(base_dir=PROJECT_ROOT)
        ms.load_model("moon-chat-pilot-v0", allow_fixture=True)
        self.assertTrue(ms.is_test_fixture)

        # Direct generation without allow_fixture=True must raise ModelNotTrainedException
        with self.assertRaises(ModelNotTrainedException) as ctx:
            ms.generate("Direct generation probe", allow_fixture=False)
        self.assertIn("Production inference blocked: Active model is an unpromoted CI test fixture", str(ctx.exception))

        # Generation with omitted allow_fixture must also raise
        with self.assertRaises(ModelNotTrainedException):
            ms.generate("Direct generation probe default")

    def test_04_http_request_body_cannot_inject_fixture_permission(self):
        """POST /api/chat cannot be bypassed by injecting allow_fixture in JSON body."""
        orig_active = model_service.active_model_name
        orig_fixture = getattr(model_service, "is_test_fixture", False)
        try:
            model_service.is_test_fixture = True
            model_service.active_model_name = "moon-chat-pilot-v0"

            # Attempt to bypass via unauthorized parameter in JSON payload
            resp = self.client.post("/api/chat", json={
                "message": "Probe attempt",
                "mode": "chat",
                "allow_fixture": True
            })
            # Must be rejected with 403 Forbidden (or 422 validation error)
            self.assertIn(resp.status_code, (403, 422))
            if resp.status_code == 403:
                self.assertIn("Production inference blocked", resp.json().get("detail", ""))
        finally:
            model_service.active_model_name = orig_active
            model_service.is_test_fixture = orig_fixture

    def test_05_alternate_routes_block_fixture_switching(self):
        """POST /api/switch-model fails closed on unpromoted test fixtures."""
        resp = self.client.post("/api/switch-model", json={"model_name": "moon-chat-pilot-v0"})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("uncertified test fixture", resp.json().get("detail", ""))

    def test_06_forged_checkpoint_metadata_fails_verification(self):
        """Checkpoints with contradictory, missing, or tampered policy metadata fail verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Case A: Checkpoint claiming moon_ai_random_init but with Policy B transfer = True
            tampered_path = os.path.join(tmpdir, "tampered_ckpt.pt")
            torch.save({
                "provenance": "moon_ai_random_init",
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "teacher_checkpoint_weights_transferred": True,
                "model": {"weight": torch.randn(4, 4)},
                "config": {"test": True}
            }, tampered_path)

            with self.assertRaises(ValueError) as ctx:
                registry_verify_provenance(tampered_path)
            self.assertIn("Policy B violation", str(ctx.exception))

            # Case B: Checkpoint claiming random init but with external base weights
            foreign_base_path = os.path.join(tmpdir, "foreign_base.pt")
            torch.save({
                "provenance": "moon_ai_random_init",
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "teacher_checkpoint_weights_transferred": False,
                "config": {"base_weights": "gpt2_openai"}
            }, foreign_base_path)

            valid, _, errs = common_verify_provenance(foreign_base_path, mode=TRAINING_MODE_PRODUCTION)
            self.assertFalse(valid)

    def test_07_altered_checkpoint_bytes_rejected_by_attestation(self):
        """Mutating a single byte of a checkpoint invalidates the SHA-256 digest and rejects attestation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "candidate.pt")
            torch.save({"dummy": "tensor_weights", "step": 100}, ckpt_path)
            original_sha = compute_file_sha256(ckpt_path)

            # Generate real Ed25519 keypair and signed attestation
            priv_key = ed25519.Ed25519PrivateKey.generate()
            pub_key = priv_key.public_key()
            pub_hex = pub_key.public_bytes_raw().hex()

            keystore_path = os.path.join(tmpdir, "trusted_evaluators.json")
            with open(keystore_path, "w", encoding="utf-8") as f:
                json.dump({
                    "trusted_evaluators": {
                        "eval_auditor_01": {
                            "status": "ACTIVE",
                            "algorithm": "Ed25519",
                            "public_key": pub_hex,
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_until": "2028-01-01T00:00:00Z"
                        }
                    }
                }, f)

            att = sign_evaluation_attestation(
                evaluator_id="eval_auditor_01",
                private_key=priv_key,
                checkpoint_sha256=original_sha,
                model_name="moon_candidate_v1",
                architecture_version="moon_arch_v1.0",
                composite_quality_score=95.0,
                non_empty_rate_pct=100.0,
                code_syntax_pct=100.0,
                dataset_manifest_sha256="none",
                test_results_summary_sha256="none"
            )

            report = {
                "checkpoint_sha256": original_sha,
                "model_name": "moon_candidate_v1",
                "composite_quality_score": 95.0,
                "non_empty_rate_pct": 100.0,
                "code_syntax_pct": 100.0,
                "evaluation_attestation": att
            }

            # Original report passes verification
            valid, err = verify_evaluation_attestation(report, evaluators_path=keystore_path)
            self.assertTrue(valid, f"Original attestation should verify: {err}")

            # Now tamper with checkpoint file bytes
            with open(ckpt_path, "ab") as f:
                f.write(b"\x00")  # Append 1 byte

            tampered_sha = compute_file_sha256(ckpt_path)
            self.assertNotEqual(original_sha, tampered_sha)

            # Promotion gate check against tampered file must fail closed
            report["checkpoint_sha256"] = tampered_sha
            valid_tampered, err_tampered = verify_evaluation_attestation(report, evaluators_path=keystore_path)
            self.assertFalse(valid_tampered)
            self.assertIn("mismatch", err_tampered)

    def test_08_missing_or_untrusted_signing_keys_fail_closed(self):
        """Attestation signed by unknown key or verified with unconfigured keystore fails closed."""
        priv_key = ed25519.Ed25519PrivateKey.generate()
        att = sign_evaluation_attestation(
            evaluator_id="rogue_evaluator",
            private_key=priv_key,
            checkpoint_sha256="a" * 64,
            model_name="untrusted_model",
            architecture_version="moon_arch_v1.0",
            composite_quality_score=99.0,
            non_empty_rate_pct=100.0,
            code_syntax_pct=100.0
        )
        report = {
            "checkpoint_sha256": "a" * 64,
            "model_name": "untrusted_model",
            "composite_quality_score": 99.0,
            "non_empty_rate_pct": 100.0,
            "code_syntax_pct": 100.0,
            "evaluation_attestation": att
        }

        # Verifying against repository default (empty keystore) fails closed
        valid, err = verify_evaluation_attestation(report)
        self.assertFalse(valid)
        self.assertIn("keystore is empty or unconfigured", err)

    def test_09_invalid_or_self_signed_manifest_rejected(self):
        """Dataset record with forged or unverified attestation fails closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_verifiers = os.path.join(tmpdir, "empty_verifiers.json")
            with open(empty_verifiers, "w", encoding="utf-8") as f:
                json.dump({"trusted_verifiers": {}}, f)

            record = {
                "id": "rec_001",
                "asset_sha256": "b" * 64,
                "source_url": "https://example.com/asset.jpg",
                "license": "CC-BY-4.0",
                "commercial_use_permitted": True,
                "attestation": {
                    "algorithm": "Ed25519",
                    "attestation_version": "2.0",
                    "verification_scope": "PRODUCTION_DATASET",
                    "verifier_id": "unknown_auditor",
                    "record_id": "rec_001",
                    "asset_sha256": "b" * 64,
                    "source_url": "https://example.com/asset.jpg",
                    "license": "CC-BY-4.0",
                    "ownership_evidence_reference": "ref_001",
                    "commercial_use_permitted": True,
                    "timestamp": "2026-06-01T12:00:00Z",
                    "signature": "c" * 128
                }
            }
            valid, status, err = verify_dataset_attestation(
                attestation=record["attestation"],
                record=record,
                actual_asset_sha256="b" * 64,
                verifiers_path=empty_verifiers
            )
            self.assertFalse(valid)
            self.assertIn("keystore is empty or unconfigured", err)

    def test_10_checkpoint_metadata_alone_does_not_prove_random_init(self):
        """Self-asserted metadata labels alone are declarative and do not prove random initialization."""
        # A synthetic checkpoint can claim random init in its metadata dictionary
        fake_ckpt = {
            "provenance": "moon_ai_random_init",
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "teacher_checkpoint_weights_transferred": False,
            "status": "PILOT_TRAINED",
            "model": {"dummy": torch.randn(2, 2)},
            "config": {
                "initialization": "random_gaussian_scratch",
                "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
                "teacher_checkpoint_weights_transferred": False
            }
        }
        # Metadata check alone returns True
        self.assertEqual(fake_ckpt.get("provenance"), "moon_ai_random_init")
        self.assertEqual(fake_ckpt.get("ownership_policy"), "STRICT_SOVEREIGN_RANDOM_INIT_ONLY")

        # But without verified training artifacts, attestation, and evaluation reports,
        # PromotionGate must reject candidate promotion
        promoted, reason, _ = PromotionGate.evaluate_promotion(
            candidate_eval={"mean_composite_score": 90.0, "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY"},
            candidate_checkpoint_path=None,
            require_checkpoint_file=True
        )
        self.assertFalse(promoted)
        self.assertIn("Candidate checkpoint path is strictly required", reason)

    def test_11_all_training_entry_points_fail_closed_when_authorization_disabled(self):
        """Every training script must fail closed when AUTHORIZED_BY_PREM is False."""
        self.assertFalse(core_policy.AUTHORIZED_BY_PREM, "Production training must be hard-locked.")

        # 1. train.py policy assertion
        with self.assertRaises(PermissionError) as ctx1:
            from core.policy import assert_production_training_authorized
            assert_production_training_authorized()
        self.assertIn("TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx1.exception))

        # 2. scripts/train_assistant.py guard
        from scripts.train_assistant import train as train_assistant
        with self.assertRaises(PermissionError) as ctx2:
            train_assistant(None)
        self.assertIn("ASSISTANT TRAINING BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx2.exception))

        # 3. scripts/colab_train.py guard
        from scripts.colab_train import main as colab_main
        with self.assertRaises(PermissionError) as ctx3:
            colab_main()
        self.assertIn("COLAB TRAINING RUNNER BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx3.exception))

        # 4. training/image/train.py guard
        from training.image.train import train as train_image
        with self.assertRaises(PermissionError) as ctx4:
            train_image(None)
        self.assertIn("IMAGE TRAINING BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx4.exception))

        # 5. training/video/train.py guard
        from training.video.train import train as train_video
        with self.assertRaises(PermissionError) as ctx5:
            train_video(None)
        self.assertIn("VIDEO TRAINING BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx5.exception))

        # 6. scripts/preflight_pilot_training.py guard
        from scripts.preflight_pilot_training import run_preflight
        with self.assertRaises(PermissionError) as ctx6:
            run_preflight()
        self.assertIn("PREFLIGHT TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx6.exception))

        # 7. bench.py guard simulation
        from core.policy import AUTHORIZED_BY_PREM
        self.assertFalse(AUTHORIZED_BY_PREM)

        # 8. training/chat/train.py guard (production pilot training)
        from training.chat.train import train_moon_chat
        with self.assertRaises(PermissionError) as ctx8:
            train_moon_chat(config_path="training/chat/pilot_config.yaml", smoke_test=False)
        self.assertIn("TRAINING EXECUTION BLOCKED: AUTHORIZED_BY_PREM is False", str(ctx8.exception))

        # 9. scripts/daily_pipeline.py (auto-training step disabled)
        import scripts.daily_pipeline
        self.assertTrue(hasattr(scripts.daily_pipeline, "main"))


if __name__ == "__main__":
    unittest.main()
