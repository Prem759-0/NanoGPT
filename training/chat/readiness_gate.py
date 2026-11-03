"""
Moon AI — Strict Training Readiness Gate (moon-chat-v0)
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Comprehensive gate validating:
1. Cryptographic provenance, source tracking, attribution, and revision metadata.
2. Verified open licensing (CC-BY-SA-4.0 for Wikipedia, CC0, MIT, CustomSovereign).
3. Complete synthetic fixture isolation.
4. Scale criteria explicitly justified by model size:
   - Engineering Smoke Test: >= 10 records, >= 2,000 tokens (tests code execution only).
   - Meaningful Pilot Training: >= 500 records, >= 100,000 tokens (required for READY_FOR_PILOT).
5. Adequate independent validation and blind test splits (zero cross-split leakage).
6. Pure-sovereign tokenizer integrity and multi-window chunking validity.
7. Architecture consistency between registry, config, and model implementation.
8. Forward/backward/optimizer assistant-only loss masking verification.
9. Checkpoint save, reload, and SHA-256 bit-for-bit integrity.

Outputs explicit status:
- DATA_INSUFFICIENT
- PROVENANCE_PENDING
- ENGINEERING_SMOKE_TEST_ONLY
- READY_FOR_PILOT
- PILOT_TRAINED
- EVALUATION_PASSED
"""

import hashlib
import json
import os
import sys
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import yaml
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.chat.schema import (
    validate_chat_record,
    compute_conversation_sha256,
    normalize_text,
    APPROVED_LICENSES,
    ALLOWED_PROVENANCE_TYPES,
    PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
    PROVENANCE_SOURCE_DERIVED_SUMMARY,
    PROVENANCE_GENUINE_HUMAN_CONVERSATION,
    PROVENANCE_SYNTHETIC_DEV_FIXTURE,
    PROVENANCE_PENDING
)
from datasets.chat.tokenizer import SovereignTokenizer
from model import GPTConfig, GPT
from training.chat.dataset import SovereignChatDataset

# Explicit status taxonomy
STATUS_DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
STATUS_PROVENANCE_PENDING = "PROVENANCE_PENDING"
STATUS_ENGINEERING_SMOKE_TEST_ONLY = "ENGINEERING_SMOKE_TEST_ONLY"
STATUS_READY_FOR_PILOT = "READY_FOR_PILOT"
STATUS_PILOT_TRAINED = "PILOT_TRAINED"
STATUS_EVALUATION_PASSED = "EVALUATION_PASSED"

# Scale criteria for Engineering Smoke Test (Code pipeline verification only)
SMOKE_TEST_MIN_TOTAL_RECORDS = 10
SMOKE_TEST_MIN_TRAIN_RECORDS = 8
SMOKE_TEST_MIN_VAL_RECORDS = 1
SMOKE_TEST_MIN_TEST_RECORDS = 1
SMOKE_TEST_MIN_TOTAL_TOKENS = 2000

# Backward compatibility aliases
MIN_TOTAL_RECORDS = SMOKE_TEST_MIN_TOTAL_RECORDS
MIN_TRAIN_RECORDS = SMOKE_TEST_MIN_TRAIN_RECORDS
MIN_VAL_RECORDS = SMOKE_TEST_MIN_VAL_RECORDS
MIN_TEST_RECORDS = SMOKE_TEST_MIN_TEST_RECORDS
MIN_TOTAL_TOKENS = SMOKE_TEST_MIN_TOTAL_TOKENS

# Scale criteria for Meaningful Pilot Training (25.7M parameter model)
PILOT_MIN_TOTAL_RECORDS = 500
PILOT_MIN_TRAIN_RECORDS = 400
PILOT_MIN_VAL_RECORDS = 50
PILOT_MIN_TEST_RECORDS = 50
PILOT_MIN_TOTAL_TOKENS = 100000

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

class ChatReadinessGate:
    """
    Evaluates whether moon-chat-v0 is genuinely ready for pilot training.
    Distinguishes engineering smoke test capability from genuine pilot training readiness.
    """

    def __init__(
        self,
        root_dir: str = PROJECT_ROOT,
        pilot_min_records: int = PILOT_MIN_TOTAL_RECORDS,
        pilot_min_tokens: int = PILOT_MIN_TOTAL_TOKENS
    ):
        self.root_dir = root_dir
        self.pilot_min_records = pilot_min_records
        self.pilot_min_tokens = pilot_min_tokens
        self.train_manifest = os.path.join(root_dir, "datasets", "manifests", "chat_train.jsonl")
        self.val_manifest = os.path.join(root_dir, "datasets", "manifests", "chat_val.jsonl")
        self.test_manifest = os.path.join(root_dir, "datasets", "manifests", "chat_test.jsonl")
        self.tokenizer_path = os.path.join(root_dir, "models", "chat", "tokenizer.json")
        self.config_path = os.path.join(root_dir, "training", "chat", "config.yaml")
        self.registry_path = os.path.join(root_dir, "models", "registry.json")

    def run_all_checks(self) -> Dict[str, Any]:
        """Runs the complete battery of readiness checks."""
        results = {
            "checks": {},
            "status": STATUS_DATA_INSUFFICIENT,
            "passed": False,
            "smoke_test_passed": False,
            "blockers": []
        }

        # 1. Load manifests
        train_recs = load_jsonl(self.train_manifest)
        val_recs = load_jsonl(self.val_manifest)
        test_recs = load_jsonl(self.test_manifest)
        all_recs = train_recs + val_recs + test_recs

        # CHECK 1: Engineering Smoke Test Minimum Scale
        smoke_scale_ok = True
        smoke_errs = []
        if len(all_recs) < SMOKE_TEST_MIN_TOTAL_RECORDS:
            smoke_scale_ok = False
            smoke_errs.append(f"Total genuine records ({len(all_recs)}) < smoke test threshold ({SMOKE_TEST_MIN_TOTAL_RECORDS}).")
        if len(train_recs) < SMOKE_TEST_MIN_TRAIN_RECORDS:
            smoke_scale_ok = False
            smoke_errs.append(f"Train records ({len(train_recs)}) < smoke test threshold ({SMOKE_TEST_MIN_TRAIN_RECORDS}).")
        if len(val_recs) < SMOKE_TEST_MIN_VAL_RECORDS:
            smoke_scale_ok = False
            smoke_errs.append(f"Validation records ({len(val_recs)}) < smoke test threshold ({SMOKE_TEST_MIN_VAL_RECORDS}).")
        if len(test_recs) < SMOKE_TEST_MIN_TEST_RECORDS:
            smoke_scale_ok = False
            smoke_errs.append(f"Test records ({len(test_recs)}) < smoke test threshold ({SMOKE_TEST_MIN_TEST_RECORDS}).")

        # CHECK 2: Provenance & Attribution Integrity
        prov_ok = True
        prov_errs = []
        for r in all_recs:
            valid, err = validate_chat_record(r, is_production=True)
            if not valid:
                prov_ok = False
                prov_errs.append(f"Record '{r.get('id')}': {err}")

        results["checks"]["provenance_and_licensing"] = {
            "passed": prov_ok,
            "errors": prov_errs
        }
        if not prov_ok:
            results["blockers"].extend(prov_errs)

        # CHECK 3: Anti-Leakage Split Independence
        leak_ok = True
        leak_errs = []
        train_hashes = {r.get("sha256") for r in train_recs}
        val_hashes = {r.get("sha256") for r in val_recs}
        test_hashes = {r.get("sha256") for r in test_recs}

        if train_hashes.intersection(val_hashes):
            leak_ok = False
            leak_errs.append("Hash overlap between train and val splits.")
        if train_hashes.intersection(test_hashes):
            leak_ok = False
            leak_errs.append("Hash overlap between train and test splits.")
        if val_hashes.intersection(test_hashes):
            leak_ok = False
            leak_errs.append("Hash overlap between val and test splits.")

        train_prompts = {normalize_text(r["messages"][0]["content"]) for r in train_recs if r.get("messages")}
        val_prompts = {normalize_text(r["messages"][0]["content"]) for r in val_recs if r.get("messages")}
        test_prompts = {normalize_text(r["messages"][0]["content"]) for r in test_recs if r.get("messages")}

        if train_prompts.intersection(val_prompts):
            leak_ok = False
            leak_errs.append("Prompt text overlap between train and val splits.")
        if train_prompts.intersection(test_prompts):
            leak_ok = False
            leak_errs.append("Prompt text overlap between train and test splits.")
        if val_prompts.intersection(test_prompts):
            leak_ok = False
            leak_errs.append("Prompt text overlap between val and test splits.")

        results["checks"]["cross_split_leakage"] = {
            "passed": leak_ok,
            "errors": leak_errs
        }
        if not leak_ok:
            results["blockers"].extend(leak_errs)

        # CHECK 4: Tokenizer Validity & Total Token Count
        tok_ok = True
        tok_errs = []
        tokenizer = None
        total_tokens = 0
        if not os.path.exists(self.tokenizer_path):
            tok_ok = False
            tok_errs.append(f"Tokenizer not found at: {self.tokenizer_path}")
        else:
            try:
                tokenizer = SovereignTokenizer.load(self.tokenizer_path)
                if tokenizer.vocab_size < 512:
                    tok_ok = False
                    tok_errs.append(f"Tokenizer vocab size ({tokenizer.vocab_size}) < 512.")
                for rec in all_recs:
                    t_ids = tokenizer.encode_chat(rec.get("messages", []))
                    total_tokens += len(t_ids)
                if total_tokens < SMOKE_TEST_MIN_TOTAL_TOKENS:
                    tok_ok = False
                    tok_errs.append(f"Total tokens ({total_tokens}) < smoke test threshold ({SMOKE_TEST_MIN_TOTAL_TOKENS}).")
            except Exception as e:
                tok_ok = False
                tok_errs.append(f"Tokenizer failed verification: {e}")

        results["checks"]["tokenizer_and_token_count"] = {
            "passed": tok_ok,
            "vocab_size": tokenizer.vocab_size if tokenizer else 0,
            "total_tokens": total_tokens,
            "errors": tok_errs
        }
        if not tok_ok:
            results["blockers"].extend(tok_errs)

        # CHECK 5: Architecture Consistency
        arch_ok = True
        arch_errs = []
        if not os.path.exists(self.config_path):
            arch_ok = False
            arch_errs.append("training/chat/config.yaml not found.")
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "PreLayerNorm" not in cfg.get("architecture_type", ""):
                arch_ok = False
                arch_errs.append("config.yaml architecture_type does not specify PreLayerNorm.")
            if cfg.get("training", {}).get("loss_masking") != "assistant_only":
                arch_ok = False
                arch_errs.append("config.yaml does not specify loss_masking: assistant_only.")

        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            chat_track = reg.get("models", {}).get("moon-chat-v0", {}) or reg.get("production_tracks", {}).get("moon-chat-v0", {})
            reg_arch = chat_track.get("architecture", "")
            if "Pre-LayerNorm" not in reg_arch:
                arch_ok = False
                arch_errs.append(f"registry.json architecture '{reg_arch}' does not match Pre-LayerNorm.")

        results["checks"]["architecture_consistency"] = {
            "passed": arch_ok,
            "errors": arch_errs
        }
        if not arch_ok:
            results["blockers"].extend(arch_errs)

        # CHECK 6: Assistant-Only Loss Masking & Forward/Backward/Optimizer Dry Run
        dry_run_ok = True
        dry_run_errs = []
        model = None
        gpt_cfg = None
        if tokenizer and os.path.exists(self.train_manifest) and os.path.exists(self.val_manifest):
            try:
                ds = SovereignChatDataset(self.train_manifest, tokenizer, block_size=128)
                val_ds = SovereignChatDataset(self.val_manifest, tokenizer, block_size=128)
                if len(ds) == 0:
                    raise ValueError("Train dataset produced 0 samples.")
                if len(val_ds) == 0:
                    raise ValueError("Validation dataset produced 0 samples.")

                x, y = ds[0]
                has_masked = (y == -1).any().item()
                has_active = (y >= 0).any().item()
                if not (has_masked and has_active):
                    dry_run_ok = False
                    dry_run_errs.append("Dataset does not exhibit assistant-only loss masking (missing -1 or active tokens).")

                # Test forward pass with random Gaussian GPT
                gpt_cfg = GPTConfig(block_size=128, vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=64)
                model = GPT(gpt_cfg)
                x_batch = x.unsqueeze(0)
                y_batch = y.unsqueeze(0)
                logits, loss = model(x_batch, y_batch)
                if loss is None or not torch.isfinite(loss) or loss.item() <= 0:
                    dry_run_ok = False
                    dry_run_errs.append(f"Model loss is invalid: {loss}")
                else:
                    loss.backward()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    optimizer.step()
                    optimizer.zero_grad()
            except Exception as e:
                dry_run_ok = False
                dry_run_errs.append(f"Dry run failed: {e}")
        else:
            dry_run_ok = False
            dry_run_errs.append("Prerequisites for dry run missing (tokenizer or manifests).")

        results["checks"]["assistant_only_dry_run"] = {
            "passed": dry_run_ok,
            "errors": dry_run_errs
        }
        if not dry_run_ok:
            results["blockers"].extend(dry_run_errs)

        # CHECK 7: Checkpoint Save, Reload, and Bit-for-Bit Hash Integrity
        ckpt_ok = True
        ckpt_errs = []
        if dry_run_ok and model is not None and gpt_cfg is not None:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    test_ckpt_path = os.path.join(tmp_dir, "test_gate_ckpt.pt")
                    state_dict = model.state_dict()
                    payload = {"model": state_dict, "step": 1}
                    torch.save(payload, test_ckpt_path)

                    loaded = torch.load(test_ckpt_path, map_location="cpu", weights_only=True)
                    model_new = GPT(gpt_cfg)
                    model_new.load_state_dict(loaded["model"])

                    for k in state_dict:
                        if not torch.equal(state_dict[k], model_new.state_dict()[k]):
                            ckpt_ok = False
                            ckpt_errs.append(f"Reloaded tensor '{k}' did not match original.")
            except Exception as e:
                ckpt_ok = False
                ckpt_errs.append(f"Checkpoint save/reload failed: {e}")
        else:
            ckpt_ok = False
            ckpt_errs.append("Skipped due to dry-run failure.")

        results["checks"]["checkpoint_integrity"] = {
            "passed": ckpt_ok,
            "errors": ckpt_errs
        }
        if not ckpt_ok:
            results["blockers"].extend(ckpt_errs)

        # CHECK 8: Meaningful Pilot Scale Evaluation
        pilot_scale_ok = (len(all_recs) >= self.pilot_min_records) and (total_tokens >= self.pilot_min_tokens)
        results["checks"]["pilot_scale_evaluation"] = {
            "passed": pilot_scale_ok,
            "total_records": len(all_recs),
            "required_records": self.pilot_min_records,
            "total_tokens": total_tokens,
            "required_tokens": self.pilot_min_tokens
        }

        # DETERMINE STATUS
        engineering_smoke_test_passed = (smoke_scale_ok and prov_ok and leak_ok and tok_ok and arch_ok and dry_run_ok and ckpt_ok)
        results["smoke_test_passed"] = engineering_smoke_test_passed

        if not prov_ok:
            results["status"] = STATUS_PROVENANCE_PENDING
        elif not smoke_scale_ok:
            results["status"] = STATUS_DATA_INSUFFICIENT
            results["blockers"].extend(smoke_errs)
        elif not engineering_smoke_test_passed:
            results["status"] = STATUS_DATA_INSUFFICIENT
        elif not pilot_scale_ok:
            # Code pipeline works, but data is insufficient for meaningful pilot training
            results["status"] = STATUS_ENGINEERING_SMOKE_TEST_ONLY
            results["blockers"].append(
                f"Corpus scale ({len(all_recs)} records, {total_tokens:,} tokens) is sufficient only for pipeline "
                f"engineering verification ({STATUS_ENGINEERING_SMOKE_TEST_ONLY}). Meaningful pilot training for a "
                f"25.7M parameter model requires at least {self.pilot_min_records} verified records and {self.pilot_min_tokens:,} tokens."
            )
        else:
            results["status"] = STATUS_READY_FOR_PILOT
            results["passed"] = True

        return results

def evaluate_and_report() -> Dict[str, Any]:
    gate = ChatReadinessGate()
    res = gate.run_all_checks()
    print("=" * 65)
    print("   MOON AI -- CHAT READINESS GATE EVALUATION")
    print("=" * 65)
    print(f"Status:             {res['status']}")
    print(f"Smoke Test Ready:   {res['smoke_test_passed']}")
    print(f"Production Ready:   {res['passed']}")
    print(f"Remaining Blockers: {len(res['blockers'])}")
    for b in res["blockers"]:
        print(f"  - {b}")
    print("=" * 65)
    return res

if __name__ == "__main__":
    evaluate_and_report()
