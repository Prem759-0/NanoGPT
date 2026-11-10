"""
Moon AI — Stage 2.1.1 Training Readiness Repair Test Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates:
1. Wikipedia licensing rules (CC-BY-SA-4.0 requirement, CC0/PD rejection).
2. Provenance categorization (ORIGINAL_SELF_AUTHORED vs GENUINE_SOURCED_EDUCATIONAL vs SYNTHETIC_DEV_FIXTURE).
3. Assistant-only loss masking in tokenizer and dataset (user/system tokens masked with -1).
4. Loss target invariance (user prompt positions have zero gradient contribution).
5. Fail-closed behavior on empty validation sets (no fake 0.0 validation loss).
6. Architecture consistency across registry, config, training plan, and model.py (Pre-LayerNorm GELU).
7. Strict readiness gate behavior (scale thresholds, anti-leakage, pass/fail states).
8. Realistic hardware VRAM sizing and non-zero OOM risk reporting.
9. Forward/backward/optimizer dry run with assistant-only loss masking.
10. Checkpoint save, reload, and bit-for-bit tensor identity.
"""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
import yaml
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.chat.schema import (
    validate_chat_record,
    compute_conversation_sha256,
    normalize_text,
    APPROVED_LICENSES,
    ALLOWED_PROVENANCE_TYPES
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.curator import deduplicate_and_validate_records, split_chat_dataset
from datasets.chat.corpus import get_genuine_corpus
from model import GPTConfig, GPT
from training.chat.dataset import SovereignChatDataset
from training.chat.readiness_gate import (
    ChatReadinessGate,
    STATUS_DATA_INSUFFICIENT,
    STATUS_PROVENANCE_PENDING,
    STATUS_READY_FOR_PILOT,
    MIN_TOTAL_RECORDS,
    MIN_TRAIN_RECORDS,
    MIN_VAL_RECORDS,
    MIN_TEST_RECORDS
)
from training.chat.validate_hardware import calculate_exact_parameters, estimate_t4_vram
from training.chat.train import train_moon_chat

class TestStage211ReadinessRepair(unittest.TestCase):
    """Exhaustive test suite for Moon Chat Stage 2.1.1 training readiness."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tokenizer_path = os.path.join(PROJECT_ROOT, "models", "chat", "tokenizer.json")
        self.assertTrue(os.path.exists(self.tokenizer_path), "Tokenizer must exist.")
        self.tokenizer = SovereignTokenizer.load(self.tokenizer_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # 1. Wikipedia Licensing Verification
    def test_wikipedia_licensing_enforcement(self):
        """Wikipedia sources MUST declare CC-BY-SA-4.0 (or CC-BY-SA-3.0), never CC0, PD, or MIT."""
        # Valid CC-BY-SA-4.0 Wikipedia record
        valid_wiki = {
            "id": "wiki_valid_01",
            "source": "Wikipedia - QuickSort",
            "source_url": "https://en.wikipedia.org/wiki/Quicksort",
            "license": "CC-BY-SA-4.0",
            "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
            "provenance_type": "GENUINE_SOURCED_EDUCATIONAL",
            "category": "computer_science",
            "messages": [
                {"role": "user", "content": "Explain Quicksort."},
                {"role": "assistant", "content": "Quicksort is a divide-and-conquer algorithm."}
            ]
        }
        ok, err = validate_chat_record(valid_wiki, is_production=True)
        self.assertTrue(ok, f"Valid Wikipedia record failed: {err}")

        # Invalid: Wikipedia claimed as CC0-1.0
        invalid_wiki_cc0 = copy.deepcopy(valid_wiki)
        invalid_wiki_cc0["license"] = "CC0-1.0"
        ok, err = validate_chat_record(invalid_wiki_cc0, is_production=True)
        self.assertFalse(ok)
        self.assertIn("Wikipedia text is licensed under CC-BY-SA", err)

        # Invalid: Wikipedia claimed as Public Domain
        invalid_wiki_pd = copy.deepcopy(valid_wiki)
        invalid_wiki_pd["license"] = "Public Domain"
        ok, err = validate_chat_record(invalid_wiki_pd, is_production=True)
        self.assertFalse(ok)
        self.assertIn("Wikipedia text is licensed under CC-BY-SA", err)

        # Invalid: Wikipedia claimed as MIT
        invalid_wiki_mit = copy.deepcopy(valid_wiki)
        invalid_wiki_mit["license"] = "MIT"
        ok, err = validate_chat_record(invalid_wiki_mit, is_production=True)
        self.assertFalse(ok)
        self.assertIn("Wikipedia text is licensed under CC-BY-SA", err)

    # 2. Provenance Categorization Verification
    def test_provenance_type_isolation(self):
        """Production manifests must accept genuine records and reject synthetic fixtures."""
        base_record = {
            "id": "prov_test_01",
            "source": "Moon AI Architecture Specification",
            "source_url": "https://github.com/Prem759-0/NanoGPT/docs/ARCH.md",
            "license": "CustomSovereign",
            "attribution": "Prem Gaikwad (Chief ML Systems Architect)",
            "provenance_type": "ORIGINAL_SELF_AUTHORED",
            "category": "sovereign_identity",
            "messages": [
                {"role": "user", "content": "Who built Moon AI?"},
                {"role": "assistant", "content": "Moon AI is developed by Prem Gaikwad."}
            ]
        }
        # ORIGINAL_SELF_AUTHORED is accepted
        ok, err = validate_chat_record(base_record, is_production=True)
        self.assertTrue(ok, f"Valid self-authored record failed: {err}")

        # GENUINE_SOURCED_EDUCATIONAL is accepted
        sourced_record = copy.deepcopy(base_record)
        sourced_record["provenance_type"] = "GENUINE_SOURCED_EDUCATIONAL"
        sourced_record["license"] = "CC-BY-SA-4.0"
        ok, err = validate_chat_record(sourced_record, is_production=True)
        self.assertTrue(ok)

        # SYNTHETIC_DEV_FIXTURE must be rejected in production
        synthetic_record = copy.deepcopy(base_record)
        synthetic_record["provenance_type"] = "SYNTHETIC_DEV_FIXTURE"
        ok, err = validate_chat_record(synthetic_record, is_production=True)
        self.assertFalse(ok)
        self.assertIn("forbidden in production", err)

        # SYNTHETIC_DEV_FIXTURE is permitted in non-production development mode
        ok_dev, _ = validate_chat_record(synthetic_record, is_production=False)
        self.assertTrue(ok_dev)

        # Missing or invalid provenance_type must fail
        invalid_prov = copy.deepcopy(base_record)
        invalid_prov["provenance_type"] = "INTERNET_SCRAPED_UNKNOWN"
        ok, err = validate_chat_record(invalid_prov, is_production=True)
        self.assertFalse(ok)
        self.assertIn("invalid provenance_type", err.lower())

    # 3. Assistant-Only Loss Masking in Tokenizer & Dataset
    def test_assistant_only_loss_masking_tokens(self):
        """Tokens predicting user/system prompt must have mask=False; assistant tokens have mask=True."""
        messages = [
            {"role": "system", "content": "You are Moon AI."},
            {"role": "user", "content": "Hello world!"},
            {"role": "assistant", "content": "Greetings human."}
        ]
        tokens, mask = self.tokenizer.encode_chat_with_loss_mask(messages)
        self.assertEqual(len(tokens), len(mask))

        assistant_token_count = sum(1 for is_active in mask if is_active)
        prompt_token_count = sum(1 for is_active in mask if not is_active)

        self.assertGreater(assistant_token_count, 0, "Must have active assistant tokens.")
        self.assertGreater(prompt_token_count, 0, "Must have masked prompt tokens.")

        # Test dataset target tensor creation
        manifest_path = os.path.join(self.tmp_dir.name, "test_mask_manifest.jsonl")
        rec = {
            "id": "rec_mask_01",
            "source": "Self",
            "license": "CustomSovereign",
            "attribution": "Prem Gaikwad",
            "provenance_type": "ORIGINAL_SELF_AUTHORED",
            "messages": messages
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        ds = SovereignChatDataset(manifest_path, self.tokenizer, block_size=128)
        x, y = ds[0]

        # Verify y contains -1 for prompt tokens and padding, and non-negative for assistant tokens
        masked_count = (y == -1).sum().item()
        unmasked_count = (y >= 0).sum().item()
        self.assertGreater(masked_count, 0)
        self.assertGreater(unmasked_count, 0)

    # 4. Target Invariance: Mathematical proof that user prompt tokens do not affect supervised loss
    def test_loss_target_invariance(self):
        """Assistant-only masked loss matches loss over active tokens only."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 is 4."}
        ]
        manifest_path = os.path.join(self.tmp_dir.name, "test_invariance_manifest.jsonl")
        rec = {
            "id": "rec_inv_01",
            "source": "Self",
            "license": "CustomSovereign",
            "attribution": "Prem Gaikwad",
            "provenance_type": "ORIGINAL_SELF_AUTHORED",
            "messages": messages
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        ds = SovereignChatDataset(manifest_path, self.tokenizer, block_size=64)
        x, y = ds[0]

        cfg = GPTConfig(block_size=64, vocab_size=self.tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=64)
        model = GPT(cfg)

        logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))

        masked_positions = (y == -1).nonzero(as_tuple=True)[0]
        unmasked_positions = (y >= 0).nonzero(as_tuple=True)[0]

        self.assertGreater(len(masked_positions), 0)
        self.assertGreater(len(unmasked_positions), 0)

        # Show that only assistant token predictions are evaluated in cross_entropy
        active_logits = logits[0, unmasked_positions]
        active_targets = y[unmasked_positions]
        expected_loss = F.cross_entropy(active_logits, active_targets)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    # 5. Fail-Closed Empty Validation Verification
    def test_empty_validation_fails_closed(self):
        """Empty validation dataset must fail closed with an explicit exception, never reporting 0.0 loss."""
        empty_manifest = os.path.join(self.tmp_dir.name, "empty_val.jsonl")
        with open(empty_manifest, "w", encoding="utf-8") as f:
            pass  # Empty file

        val_ds = SovereignChatDataset(empty_manifest, self.tokenizer, block_size=64)
        self.assertEqual(len(val_ds), 0)

        # Build dummy config pointing to empty val manifest
        tmp_cfg = {
            "model_track": "moon-chat-v0",
            "paths": {
                "tokenizer_path": self.tokenizer_path,
                "train_manifest": os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl"),
                "val_manifest": empty_manifest,
                "checkpoint_dir": self.tmp_dir.name,
                "checkpoint_name": "test_ckpt.pt"
            },
            "model": {
                "name": "moon-chat-test",
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "block_size": 64,
                "dropout": 0.0,
                "bias": False
            },
            "training": {
                "batch_size": 2,
                "max_iters": 2,
                "learning_rate": 1e-4,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95
            }
        }
        cfg_file = os.path.join(self.tmp_dir.name, "test_empty_cfg.yaml")
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(tmp_cfg, f)

        with self.assertRaises(ValueError) as ctx:
            train_moon_chat(config_path=cfg_file, max_iters_override=1, device_override="cpu", dry_run=True)
        self.assertIn("Validation dataset is empty! Failing closed", str(ctx.exception))

    def test_windows_cross_drive_path_handling_regression(self):
        """
        Regression test for Windows cross-drive tempfile paths.
        Ensures safe_relpath normalizes paths without ValueError across different drive mounts,
        while maintaining fail-closed boundary enforcement.
        """
        from training.chat.train import safe_relpath
        # 1. Cross-drive path formatting succeeds without exception
        cross_drive_path = "C:\\Users\\Temp\\dataset.jsonl" if PROJECT_ROOT.startswith("D:") else "D:\\Temp\\dataset.jsonl"
        norm = safe_relpath(cross_drive_path, start=PROJECT_ROOT)
        self.assertTrue(os.path.isabs(norm))
        self.assertEqual(norm, os.path.normpath(cross_drive_path))

        # 2. Within-repo relative path is preserved
        local_rel = os.path.join("datasets", "manifests", "scaled_chat_train.jsonl")
        local_full = os.path.join(PROJECT_ROOT, local_rel)
        computed_rel = safe_relpath(local_full, start=PROJECT_ROOT)
        self.assertEqual(os.path.normpath(computed_rel), os.path.normpath(local_rel))

    # 6. Architecture Consistency Verification
    def test_architecture_consistency(self):
        """Registry, config.yaml, training plan, and model.py must agree on Pre-LayerNorm GELU architecture."""
        # 1. Config.yaml
        cfg_path = os.path.join(PROJECT_ROOT, "training", "chat", "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get("architecture_type"), "Causal_Decoder_Only_Transformer_PreLayerNorm_GELU")
        self.assertEqual(cfg["model"]["exact_parameters"], 25698816)
        self.assertEqual(cfg["baseline_124m"]["exact_parameters"], 86133504)
        self.assertEqual(cfg["training"]["loss_masking"], "assistant_only")

        # 2. Registry
        reg_path = os.path.join(PROJECT_ROOT, "models", "registry.json")
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        chat_track = reg.get("models", {}).get("moon-chat-v0") or reg.get("production_tracks", {}).get("moon-chat-v0")
        self.assertIsNotNone(chat_track)
        self.assertIn("Pre-LayerNorm", chat_track["architecture"])
        self.assertIn("GELU", chat_track["architecture"])
        self.assertEqual(chat_track["trainable_parameters"], 25698816)

        # 3. Model.py PyTorch implementation parameters
        pilot_cfg = GPTConfig(block_size=512, vocab_size=512, n_layer=8, n_head=8, n_embd=512, bias=False)
        pilot_model = GPT(pilot_cfg)
        self.assertEqual(sum(p.numel() for p in pilot_model.parameters()), 25698816)

        base_cfg = GPTConfig(block_size=1024, vocab_size=512, n_layer=12, n_head=12, n_embd=768, bias=False)
        base_model = GPT(base_cfg)
        self.assertEqual(sum(p.numel() for p in base_model.parameters()), 86133504)

    # 7. Strict Readiness Gate Evaluation
    def test_strict_readiness_gate(self):
        """Readiness gate must pass with verified corpus and fail closed when scale is insufficient."""
        # Configured with pilot threshold
        gate = ChatReadinessGate(pilot_min_records=20, pilot_min_tokens=5000)
        res = gate.run_all_checks()
        self.assertTrue(res["passed"], f"Readiness gate failed: {res['blockers']}")
        self.assertEqual(res["status"], STATUS_READY_FOR_PILOT)
        self.assertEqual(len(res["blockers"]), 0)

        # Default strict gate marks small 22-record corpus as smoke-test ready only
        gate_default = ChatReadinessGate()
        res_default = gate_default.run_all_checks()
        self.assertEqual(res_default["status"], "ENGINEERING_SMOKE_TEST_ONLY")
        self.assertTrue(res_default["smoke_test_passed"])
        self.assertFalse(res_default["passed"])

        # Scale check: must fail if total records < 20
        with tempfile.TemporaryDirectory() as empty_dir:
            gate_empty = ChatReadinessGate(root_dir=empty_dir)
            res_empty = gate_empty.run_all_checks()
            self.assertFalse(res_empty["passed"])
            self.assertEqual(res_empty["status"], STATUS_DATA_INSUFFICIENT)

    # 8. Realistic Hardware Feasibility & OOM Risk Reporting
    def test_hardware_feasibility_and_oom_risk(self):
        """VRAM estimation must return realistic peak memory and LOW risk assessment, never ZERO."""
        calc = calculate_exact_parameters(GPTConfig(block_size=512, vocab_size=512, n_layer=8, n_head=8, n_embd=512, bias=False))
        self.assertEqual(calc["total_parameters"], 25698816)

        vram = estimate_t4_vram(25698816, batch_size=4, block_size=512, n_embd=512, n_layer=8)
        self.assertGreater(vram["estimated_peak_vram_mb"], 1000.0)
        self.assertLess(vram["estimated_peak_vram_mb"], 4000.0)
        self.assertGreater(vram["headroom_mb"], 10000.0)
        self.assertIn("LOW", vram["oom_risk_assessment"])
        self.assertNotIn("ZERO", vram["oom_risk_assessment"])

    # 9. Forward/Backward Dry Run on CPU
    def test_forward_backward_dry_run(self):
        """Performs a 1-step forward, backward, and AdamW optimizer step with assistant-only loss masking."""
        train_manifest = os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl")
        ds = SovereignChatDataset(train_manifest, self.tokenizer, block_size=128)
        self.assertGreater(len(ds), 0)

        x, y = ds[0]
        cfg = GPTConfig(block_size=128, vocab_size=self.tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=64)
        model = GPT(cfg)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Verify gradients exist and are non-zero
        has_grads = any(p.grad is not None and torch.norm(p.grad).item() > 0 for p in model.parameters())
        self.assertTrue(has_grads, "Active parameters must have non-zero gradients.")

        optimizer.step()
        optimizer.zero_grad()

    # 10. Checkpoint Bit-for-Bit Identity
    def test_checkpoint_reload_identity(self):
        """Model weights saved and reloaded must match bit-for-bit with identical tensor values."""
        cfg = GPTConfig(block_size=64, vocab_size=self.tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=64)
        model1 = GPT(cfg)

        ckpt_file = os.path.join(self.tmp_dir.name, "ckpt_identity.pt")
        torch.save({"model": model1.state_dict(), "step": 42}, ckpt_file)

        with open(ckpt_file, "rb") as f:
            h1 = hashlib.sha256(f.read()).hexdigest()

        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        model2 = GPT(cfg)
        model2.load_state_dict(loaded["model"])

        # Compare state dict tensors
        for k in model1.state_dict():
            self.assertTrue(
                torch.equal(model1.state_dict()[k], model2.state_dict()[k]),
                f"Tensor {k} mismatch after reload!"
            )

        ckpt_file2 = os.path.join(self.tmp_dir.name, "ckpt_identity2.pt")
        torch.save({"model": model2.state_dict(), "step": 42}, ckpt_file2)
        with open(ckpt_file2, "rb") as f:
            h2 = hashlib.sha256(f.read()).hexdigest()

        # Both state dicts are bit-for-bit identical
        self.assertEqual(loaded["step"], 42)

if __name__ == "__main__":
    unittest.main()
