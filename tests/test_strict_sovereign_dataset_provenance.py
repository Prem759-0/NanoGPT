"""
tests/test_strict_sovereign_dataset_provenance.py
Exhaustive verification of Strict Sovereign Ownership & Dataset Provenance Enforcement.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Verifies:
1. Valid original curriculum records load cleanly.
2. TEACHER_DISTILLATION_OFFLINE_QUALIFIED is rejected at record level.
3. Non-null teacher_provider is rejected immediately.
4. Non-null teacher_model_id is rejected immediately.
5. Nested teacher-generation metadata in generation_metadata is rejected.
6. Rejection occurs strictly BEFORE tokenization (encode_chat_with_loss_mask is never called).
7. Quarantined records and manifests (teacher_pilot_demonstrations.jsonl, quarantined_teacher_data.jsonl)
   cannot enter training or dataset loading.
8. Central policy engine exports and active rules enforce STRICT_SOVEREIGN_RANDOM_INIT_ONLY.
"""

import os
import sys
import json
import tempfile
import unittest
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.policy import (
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
    ACTIVE_OWNERSHIP_POLICY,
    get_active_policy,
    is_valid_policy,
    get_policy_summary,
    validate_student_initialization
)
from datasets.chat.tokenizer import SovereignTokenizer
from training.chat.dataset import (
    SovereignChatDataset,
    validate_chat_record_provenance,
    SAFE_SOVEREIGN_CHAT_PROVENANCES
)

class TrackingMockTokenizer:
    """Mock tokenizer that records whether encode_chat_with_loss_mask was invoked."""
    def __init__(self):
        self.encode_call_count = 0
        self.vocab_size = 512

    def encode_chat_with_loss_mask(self, messages: List[Dict[str, str]]):
        self.encode_call_count += 1
        return [1, 2, 3, 4, 5], [False, False, True, True, False]


class TestStrictSovereignDatasetProvenance(unittest.TestCase):
    """Test suite verifying fail-closed record-level teacher data rejection."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tokenizer_path = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
        self.tokenizer = SovereignTokenizer.load(self.tokenizer_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_temp_manifest(self, records: List[Dict[str, Any]]) -> str:
        manifest_path = os.path.join(self.tmp_dir.name, "test_manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return manifest_path

    # 1. Valid curriculum record loads cleanly
    def test_valid_original_curriculum_record_loads(self):
        """Valid deterministic curriculum record loads cleanly and creates proper targets."""
        record = {
            "id": "valid_curriculum_001",
            "provenance_type": "DETERMINISTIC_CURRICULUM_TEMPLATE",
            "source": "Moon AI Scaled Curriculum Generator",
            "license": "Apache-2.0",
            "attribution": "Prem Gaikwad (Moon AI Chief ML Systems Architect, India)",
            "teacher_provider": None,
            "teacher_model_id": None,
            "teacher_checkpoint_hash": None,
            "messages": [
                {"role": "user", "content": "Explain Newton's third law of motion."},
                {"role": "assistant", "content": "For every action in nature there is an equal and opposite reaction."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        ds = SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.get_unique_record_count(), 1)
        x, y = ds[0]
        self.assertEqual(x.shape, (64,))
        self.assertEqual(y.shape, (64,))
        self.assertGreater(ds.get_total_supervised_tokens(), 0)

    # 2. TEACHER_DISTILLATION_OFFLINE_QUALIFIED is rejected
    def test_teacher_distillation_provenance_rejected(self):
        """Record with TEACHER_DISTILLATION_OFFLINE_QUALIFIED is rejected with ValueError."""
        record = {
            "id": "teacher_rec_001",
            "provenance_type": "TEACHER_DISTILLATION_OFFLINE_QUALIFIED",
            "source": "Teacher Model",
            "license": "Apache-2.0",
            "attribution": "Qwen Team",
            "messages": [
                {"role": "user", "content": "Explain quantum entanglement."},
                {"role": "assistant", "content": "Quantum entanglement is a physical phenomenon..."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        with self.assertRaises(ValueError) as ctx:
            SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        self.assertIn("Strict Sovereign violation", str(ctx.exception))
        self.assertIn("teacher provenance_type", str(ctx.exception))

    # 3. Non-null teacher_provider is rejected
    def test_non_null_teacher_provider_rejected(self):
        """Record with non-null teacher_provider is rejected immediately."""
        record = {
            "id": "teacher_prov_001",
            "provenance_type": "ORIGINAL_SELF_AUTHORED_QA",
            "source": "Self",
            "license": "Apache-2.0",
            "attribution": "Prem Gaikwad",
            "teacher_provider": "local",
            "messages": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Greetings."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        with self.assertRaises(ValueError) as ctx:
            SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        self.assertIn("Strict Sovereign violation", str(ctx.exception))
        self.assertIn("non-null teacher_provider", str(ctx.exception))

    # 4. Non-null teacher_model_id is rejected
    def test_non_null_teacher_model_id_rejected(self):
        """Record with non-null teacher_model_id is rejected immediately."""
        record = {
            "id": "teacher_model_001",
            "provenance_type": "DETERMINISTIC_CURRICULUM_TEMPLATE",
            "source": "Curriculum",
            "license": "Apache-2.0",
            "attribution": "Prem Gaikwad",
            "teacher_model_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "messages": [
                {"role": "user", "content": "What is gravity?"},
                {"role": "assistant", "content": "Gravity is the universal force of attraction."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        with self.assertRaises(ValueError) as ctx:
            SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        self.assertIn("Strict Sovereign violation", str(ctx.exception))
        self.assertIn("non-null teacher_model_id", str(ctx.exception))

    # 5. Nested teacher-generation metadata is rejected
    def test_nested_teacher_generation_metadata_rejected(self):
        """Record with teacher_model in generation_metadata is rejected."""
        record = {
            "id": "nested_teacher_001",
            "provenance_type": "DETERMINISTIC_CURRICULUM_TEMPLATE",
            "source": "Curriculum",
            "license": "Apache-2.0",
            "attribution": "Prem Gaikwad",
            "generation_metadata": {
                "teacher_model": "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "timestamp": "2026-11-04T00:00:00Z"
            },
            "messages": [
                {"role": "user", "content": "Test prompt."},
                {"role": "assistant", "content": "Test response."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        with self.assertRaises(ValueError) as ctx:
            SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        self.assertIn("Strict Sovereign violation", str(ctx.exception))
        self.assertIn("generation_metadata.teacher_model", str(ctx.exception))

    # 6. Rejection occurs strictly before tokenization
    def test_rejection_occurs_before_tokenization(self):
        """Proof that tokenizer is NEVER called when a teacher record is processed."""
        record = {
            "id": "early_reject_001",
            "provenance_type": "TEACHER_DISTILLATION_OFFLINE_QUALIFIED",
            "source": "Teacher",
            "messages": [
                {"role": "user", "content": "Sample prompt."},
                {"role": "assistant", "content": "Sample response."}
            ]
        }
        manifest_path = self._create_temp_manifest([record])
        tracking_tok = TrackingMockTokenizer()

        with self.assertRaises(ValueError):
            SovereignChatDataset(manifest_path, tracking_tok, block_size=64)

        # Invariant: encode_chat_with_loss_mask must have been called exactly 0 times
        self.assertEqual(
            tracking_tok.encode_call_count,
            0,
            "Security invariant broken: tokenizer was invoked before provenance validation!"
        )

    # 7. Quarantined records and manifests cannot enter training loaders
    def test_quarantined_records_and_manifests_rejected(self):
        """Quarantined manifests fail closed when loaded into SovereignChatDataset."""
        qwen_manifest = os.path.join(PROJECT_ROOT, "datasets", "manifests", "teacher_pilot_demonstrations.jsonl")
        mock_manifest = os.path.join(PROJECT_ROOT, "datasets", "manifests", "quarantined_teacher_data.jsonl")

        self.assertTrue(os.path.exists(qwen_manifest))
        self.assertTrue(os.path.exists(mock_manifest))

        # 1. Qwen manifest must fail closed
        with self.assertRaises(ValueError) as ctx1:
            SovereignChatDataset(qwen_manifest, self.tokenizer, block_size=64)
        self.assertIn("QUARANTINED", str(ctx1.exception))

        # 2. Mock quarantined manifest must fail closed
        with self.assertRaises(ValueError) as ctx2:
            SovereignChatDataset(mock_manifest, self.tokenizer, block_size=64)
        self.assertIn("QUARANTINED", str(ctx2.exception))

    # 8. Central ownership policy engine consistency
    def test_central_ownership_policy_strictly_sovereign(self):
        """Active policy is STRICT_SOVEREIGN_RANDOM_INIT_ONLY with zero teacher allowance."""
        self.assertEqual(get_active_policy(), POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertEqual(ACTIVE_OWNERSHIP_POLICY, POLICY_STRICT_SOVEREIGN_RANDOM_INIT)

        summary = get_policy_summary()
        self.assertEqual(summary["policy"], POLICY_STRICT_SOVEREIGN_RANDOM_INIT)
        self.assertTrue(summary["is_active"])
        self.assertFalse(summary["rules"]["allow_teacher_demonstrations"])
        self.assertTrue(summary["rules"]["require_random_initialization"])
        self.assertFalse(summary["rules"]["allow_third_party_weights"])
        self.assertFalse(summary["rules"]["allow_external_apis_in_runtime"])

        # Random Gaussian init passes; teacher foundation import fails
        ok, msg = validate_student_initialization("random_gaussian_scratch")
        self.assertTrue(ok)
        ok_fail, msg_fail = validate_student_initialization("kimi_k3_1_foundation_weights")
        self.assertFalse(ok_fail)

if __name__ == "__main__":
    unittest.main()

