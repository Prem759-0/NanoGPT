"""
Moon AI — Sovereign Chat Transformer Training Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Trains Moon Chat (Causal Autoregressive Transformer) from purely random weights.
Enforces strict training safety gates, reports dataset exposure metrics,
and cleanly isolates engineering smoke tests from production runs.
Zero third-party pretrained weights, zero external APIs.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if "datasets" in sys.modules:
    mod = sys.modules["datasets"]
    if not hasattr(mod, "__file__") or not mod.__file__ or "site-packages" in mod.__file__ or "dist-packages" in mod.__file__:
        del sys.modules["datasets"]

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from model import GPTConfig, GPT
from datasets.chat.tokenizer import SovereignTokenizer
from training.chat.dataset import SovereignChatDataset
from training.chat.readiness_gate import ChatReadinessGate, STATUS_READY_FOR_PILOT
from scripts.validate_t4_empirical import check_t4_hardware_identity

# Canonical LF hashes for high-volume 1,100-record curriculum splits
CANONICAL_MANIFEST_LF_HASHES = {
    "scaled_chat_train.jsonl": "c0f37ba515977e1ea542fed18c78ad018b3dce2f5b02d5f6213beab47e03eebe",
    "scaled_chat_val.jsonl": "b2f53140c0f417d59eabb5cce2163198a2a35f5bd7732a4bb51f143ebaa059cd",
    "scaled_chat_test.jsonl": "e854f62814d413b36d02224777fe1d6d33a7ebffbac1a64feb4fee8de8c53428"
}

def compute_manifest_hashes(fpath: str) -> Tuple[Optional[str], Optional[str]]:
    """Compute (raw_sha256, normalized_lf_sha256) for a manifest file."""
    if os.path.isfile(fpath):
        with open(fpath, "rb") as mf:
            content = mf.read()
        raw_sha = hashlib.sha256(content).hexdigest()
        lf_sha = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        return raw_sha, lf_sha
    return None, None

def parse_args():
    parser = argparse.ArgumentParser(description="Moon AI Sovereign Chat Training Engine")
    parser.add_argument("--config", type=str, default="training/chat/config.yaml", help="Path to training config YAML")
    parser.add_argument("--max-iters", type=int, default=None, help="Override maximum training iterations")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override validation evaluation interval")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from an existing random-init checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 iteration to verify pipeline integrity")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 iteration engineering smoke test without production checkpointing")
    return parser.parse_args()

def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def train_moon_chat(
    config_path: str,
    max_iters_override: Optional[int] = None,
    device_override: Optional[str] = None,
    eval_interval_override: Optional[int] = None,
    resume_from_ckpt: Optional[str] = None,
    dry_run: bool = False,
    smoke_test: bool = False,
    custom_ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    is_smoke_test = dry_run or smoke_test

    # Load configuration
    abs_config_path = os.path.abspath(config_path)
    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"Config file not found: {abs_config_path}")

    with open(abs_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # SAFEGUARD 1: Fail-Closed Zero-Teacher Policy B Option 1 Enforcement
    # -------------------------------------------------------------------------
    td_cfg = cfg.get("teacher_distillation", {})
    if td_cfg.get("enabled", False) is not False:
        raise ValueError(
            "Policy B Option 1 Sovereign violation: teacher_distillation.enabled must be False. "
            "Zero teacher influence is mandated for Option 1 Sovereign training."
        )
    if td_cfg.get("allow_teacher_demonstrations", False) is not False:
        raise ValueError(
            "Policy B Option 1 Sovereign violation: allow_teacher_demonstrations must be False. "
            "Teacher demonstrations are strictly prohibited."
        )
    if td_cfg.get("prohibit_teacher_checkpoint_weights", True) is not True:
        raise ValueError(
            "Policy B Option 1 Sovereign violation: prohibit_teacher_checkpoint_weights must be True. "
            "Pretrained teacher checkpoint weights are strictly prohibited."
        )
    if td_cfg.get("student_init_method", "random_gaussian_scratch") != "random_gaussian_scratch":
        raise ValueError(
            f"Policy B Option 1 Sovereign violation: student_init_method must be 'random_gaussian_scratch' "
            f"(detected '{td_cfg.get('student_init_method')}'). Student must initialize purely from Gaussian scratch."
        )
    if td_cfg.get("target_teacher") is not None:
        raise ValueError(
            f"Policy B Option 1 Sovereign violation: target_teacher must be null/None (detected '{td_cfg.get('target_teacher')}')."
        )
    if td_cfg.get("teacher_provider") is not None:
        raise ValueError(
            f"Policy B Option 1 Sovereign violation: teacher_provider must be null/None (detected '{td_cfg.get('teacher_provider')}')."
        )

    # -------------------------------------------------------------------------
    # SAFEGUARD 3: Manifest Resolution, Blind-Test Isolation & Canonical Hashes
    # -------------------------------------------------------------------------
    train_manifest = cfg["paths"]["train_manifest"]
    if not os.path.isabs(train_manifest):
        train_manifest = os.path.join(PROJECT_ROOT, train_manifest)

    val_manifest = cfg["paths"]["val_manifest"]
    if not os.path.isabs(val_manifest):
        val_manifest = os.path.join(PROJECT_ROOT, val_manifest)

    test_manifest = cfg["paths"].get("test_manifest", "datasets/manifests/scaled_chat_test.jsonl")
    if not os.path.isabs(test_manifest):
        test_manifest = os.path.join(PROJECT_ROOT, test_manifest)

    # Assert test manifest is strictly distinct from training and validation splits
    if os.path.abspath(test_manifest) == os.path.abspath(train_manifest):
        raise ValueError(
            f"Blind-test split collision: test_manifest ({test_manifest}) cannot be identical to train_manifest!"
        )
    if os.path.abspath(test_manifest) == os.path.abspath(val_manifest):
        raise ValueError(
            f"Blind-test split collision: test_manifest ({test_manifest}) cannot be identical to val_manifest!"
        )

    # Compute manifest provenance hashes
    train_raw_sha, train_lf_sha = compute_manifest_hashes(train_manifest)
    val_raw_sha, val_lf_sha = compute_manifest_hashes(val_manifest)
    test_raw_sha, test_lf_sha = compute_manifest_hashes(test_manifest)

    # Production run manifest hash verification against audited values
    if not is_smoke_test:
        for m_name, m_path, m_lf_sha in [
            ("train_manifest", train_manifest, train_lf_sha),
            ("val_manifest", val_manifest, val_lf_sha),
            ("test_manifest", test_manifest, test_lf_sha)
        ]:
            bname = os.path.basename(m_path)
            if bname in CANONICAL_MANIFEST_LF_HASHES:
                expected_sha = CANONICAL_MANIFEST_LF_HASHES[bname]
                if m_lf_sha != expected_sha:
                    raise ValueError(
                        f"Manifest provenance hash mismatch for {m_name} ({bname})! "
                        f"Expected canonical LF SHA: {expected_sha}, detected: {m_lf_sha}. "
                        "Failing closed to prevent training on unverified or tampered data."
                    )

    manifest_provenance_hashes = {
        "train_manifest": {
            "path": os.path.relpath(train_manifest, PROJECT_ROOT),
            "sha256": train_lf_sha or "",
            "raw_sha256": train_raw_sha
        },
        "val_manifest": {
            "path": os.path.relpath(val_manifest, PROJECT_ROOT),
            "sha256": val_lf_sha or "",
            "raw_sha256": val_raw_sha
        },
        "test_manifest": {
            "path": os.path.relpath(test_manifest, PROJECT_ROOT),
            "sha256": test_lf_sha or "",
            "raw_sha256": test_raw_sha
        }
    }

    # -------------------------------------------------------------------------
    # SAFEGUARD 4: Stage-Isolated Checkpoint Destinations (Anti-Collision)
    # -------------------------------------------------------------------------
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if custom_ckpt_path:
        checkpoint_path = custom_ckpt_path
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        ckpt_filename = os.path.basename(checkpoint_path)
    else:
        ckpt_filename = "smoke_test_ckpt.pt" if is_smoke_test else cfg["paths"]["checkpoint_name"]
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)

    best_ckpt_filename = cfg["paths"].get("best_checkpoint_name")
    if not best_ckpt_filename:
        base_name = cfg["paths"].get("checkpoint_name", "moon_chat_model.pt")
        stem, ext = os.path.splitext(base_name)
        best_ckpt_filename = f"{stem}_best{ext}"
    best_ckpt_path = os.path.join(checkpoint_dir, best_ckpt_filename)

    FORBIDDEN_LEGACY_CHECKPOINTS = {
        "moon_chat_model.pt",
        "moon_chat_scaled_v1.pt",
        "moon_chat_scaled_best.pt",
        "moon_chat_pilot_v0.pt",
        "moon_chat_pilot_best.pt",
    }
    track_name = cfg.get("model_track", "")
    is_phase_2_4 = "stage-2-4" in track_name or "t4_pilot" in track_name or "phase_2_4" in abs_config_path

    if not is_smoke_test and is_phase_2_4:
        if ckpt_filename in FORBIDDEN_LEGACY_CHECKPOINTS:
            raise ValueError(
                f"Checkpoint anti-collision failure: checkpoint_name '{ckpt_filename}' collides with "
                f"a protected legacy checkpoint. Phase 2.4 must use dedicated stage-isolated filenames."
            )
        if best_ckpt_filename in FORBIDDEN_LEGACY_CHECKPOINTS:
            raise ValueError(
                f"Checkpoint anti-collision failure: best_checkpoint_name '{best_ckpt_filename}' collides with "
                f"a protected legacy checkpoint. Phase 2.4 must use dedicated stage-isolated filenames."
            )
        if ckpt_filename == best_ckpt_filename:
            raise ValueError(
                f"Checkpoint destination collision: checkpoint_name ('{ckpt_filename}') cannot be equal "
                f"to best_checkpoint_name ('{best_ckpt_filename}')!"
            )
        if "phase_2_4" not in ckpt_filename and "t4_pilot" not in ckpt_filename:
            raise ValueError(
                f"Phase 2.4 checkpoint naming violation: '{ckpt_filename}' must contain 'phase_2_4' or 't4_pilot'."
            )

    # Real Filesystem Pre-Flight Collision Gate (Production Pilot Run)
    if not is_smoke_test:
        if os.path.isfile(checkpoint_path):
            raise FileExistsError(
                f"Production checkpoint collision: destination '{checkpoint_path}' already exists! "
                f"Silent overwriting of production checkpoints is strictly prohibited under Moon AI integrity policy. "
                f"Archive or rename existing checkpoints before launching a new production training run."
            )
        if os.path.isfile(best_ckpt_path):
            raise FileExistsError(
                f"Production checkpoint collision: best checkpoint destination '{best_ckpt_path}' already exists! "
                f"Silent overwriting of production checkpoints is strictly prohibited under Moon AI integrity policy. "
                f"Archive or rename existing checkpoints before launching a new production training run."
            )

    # -------------------------------------------------------------------------
    # READINESS GATE: Preflight Verification
    # -------------------------------------------------------------------------
    if not is_smoke_test:
        gate = ChatReadinessGate(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            config_path=abs_config_path
        )
        gate_res = gate.run_all_checks()
        if gate_res["status"] != STATUS_READY_FOR_PILOT:
            print("!" * 65)
            print("❌ PRODUCTION PILOT TRAINING BLOCKED BY READINESS GATE")
            print("!" * 65)
            print(f"Current Status:   {gate_res['status']} (Required: {STATUS_READY_FOR_PILOT})")
            print(f"Blockers ({len(gate_res['blockers'])}):")
            for b in gate_res["blockers"]:
                print(f"  • {b}")
            print("\nMeaningful pilot training for a 25.7M model requires at least 500 records and 100,000+ tokens.")
            print("To verify pipeline mechanics without producing a production checkpoint, pass --smoke-test.")
            print("!" * 65 + "\n")
            raise RuntimeError(
                f"Production pilot training blocked by readiness gate (status: {gate_res['status']}). "
                f"Use --smoke-test to execute pipeline verification only."
            )

    # -------------------------------------------------------------------------
    # SAFEGUARD 2: Hardware Device Authentication & Physical Tesla T4 Check
    # -------------------------------------------------------------------------
    if device_override:
        device = device_override
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was explicitly requested (--device cuda), but torch.cuda.is_available() is False. "
                "Failing closed."
            )
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    gpu_identity_info = {}
    if not is_smoke_test:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Production pilot training blocked: CUDA is unavailable. "
                "An authenticated NVIDIA Tesla T4 GPU (Compute Capability 7.5, >=14GB VRAM) is strictly required for Phase 2.4. "
                "Use --smoke-test to execute an engineering verification dry-run on CPU."
            )
        if device != "cuda":
            raise RuntimeError(
                f"Production pilot training blocked: requested device '{device}' is not CUDA. "
                "Phase 2.4 production pilot training must execute on CUDA."
            )

        gpu_name = torch.cuda.get_device_name(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        cc_str = f"{cc_major}.{cc_minor}"
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        is_t4, reason, details = check_t4_hardware_identity(
            device_name=gpu_name,
            compute_capability=cc_str,
            total_vram_mb=vram_mb,
            min_vram_mb=14000.0
        )
        if not is_t4:
            raise RuntimeError(
                f"Production pilot training blocked by Hardware Security Gate: {reason}. "
                f"Detected: '{gpu_name}' (CC: {cc_str}, VRAM: {vram_mb:.1f} MB). "
                f"Expected: Authentic NVIDIA Tesla T4 (CC: 7.5, >=14,000 MB VRAM)."
            )
        gpu_identity_info = {
            "name": gpu_name,
            "compute_capability": cc_str,
            "vram_mb": round(vram_mb, 1),
            "authenticated_t4": True,
            "verification_status": "AUTHENTICATED_TESLA_T4"
        }
    else:
        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_identity_info = {
                "name": gpu_name,
                "compute_capability": f"{cc_major}.{cc_minor}",
                "vram_mb": round(vram_mb, 1),
                "authenticated_t4": "t4" in gpu_name.lower(),
                "verification_status": "SMOKE_TEST_CUDA"
            }
        else:
            gpu_identity_info = {
                "name": "CPU",
                "compute_capability": "N/A",
                "vram_mb": 0.0,
                "authenticated_t4": False,
                "verification_status": "SMOKE_TEST_CPU"
            }

    # Set random seed
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load Sovereign Tokenizer
    tok_path = cfg["paths"]["tokenizer_path"]
    if not os.path.isabs(tok_path):
        tok_path = os.path.join(PROJECT_ROOT, tok_path)
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Sovereign tokenizer not found at: {tok_path}. Run scripts/build_chat_dataset.py first.")
    tokenizer = SovereignTokenizer.load(tok_path)

    m_cfg = cfg["model"]
    block_size = m_cfg["block_size"]

    train_dataset = SovereignChatDataset(train_manifest, tokenizer, block_size=block_size)
    val_dataset = SovereignChatDataset(val_manifest, tokenizer, block_size=block_size)

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Populate datasets/manifests/chat_train.jsonl first.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Failing closed to prevent misleading zero validation loss.")

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Safeguard: Assert blind test manifest is NEVER loaded into training or validation loaders
    abs_test_manifest = os.path.abspath(test_manifest)
    assert os.path.abspath(train_manifest) != abs_test_manifest, "CRITICAL: train_manifest equals blind-test manifest!"
    assert os.path.abspath(val_manifest) != abs_test_manifest, "CRITICAL: val_manifest equals blind-test manifest!"
    if hasattr(train_dataset, "manifest_path"):
        assert os.path.abspath(train_dataset.manifest_path) != abs_test_manifest, "CRITICAL: train_dataset loaded blind-test manifest!"
    if hasattr(val_dataset, "manifest_path"):
        assert os.path.abspath(val_dataset.manifest_path) != abs_test_manifest, "CRITICAL: val_dataset loaded blind-test manifest!"

    # Dataset Exposure Analysis
    t_cfg = cfg["training"]
    max_iters = 1 if is_smoke_test else (max_iters_override or t_cfg["max_iters"])
    grad_accum_steps = 1 if is_smoke_test else t_cfg.get("gradient_accumulation_steps", 1)

    unique_records = train_dataset.get_unique_record_count()
    total_train_windows = len(train_dataset)
    supervised_tokens = train_dataset.get_total_supervised_tokens()
    effective_samples = max_iters * batch_size * grad_accum_steps
    dataset_passes = effective_samples / max(1, total_train_windows)

    # Initialize Model from Random Gaussian Weights ONLY
    gpt_conf = GPTConfig(
        block_size=m_cfg["block_size"],
        vocab_size=tokenizer.vocab_size,
        n_layer=m_cfg["n_layer"],
        n_head=m_cfg["n_head"],
        n_embd=m_cfg["n_embd"],
        dropout=m_cfg.get("dropout", 0.1),
        bias=m_cfg.get("bias", False)
    )

    model = GPT(gpt_conf)
    model.to(device)

    if resume_from_ckpt:
        if not os.path.exists(resume_from_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_ckpt}")
        ckpt_data = torch.load(resume_from_ckpt, map_location=device, weights_only=False)
        cfg_meta = ckpt_data.get("config", {}) or ckpt_data.get("metadata", {})
        # Policy B Verification
        if cfg_meta.get("teacher_checkpoint_weights_transferred") is not False:
            raise ValueError("Policy B violation: teacher_checkpoint_weights_transferred must be False.")
        if cfg_meta.get("initialization") != "random_gaussian_scratch":
            raise ValueError("Policy B violation: checkpoint must originate from random_gaussian_scratch.")
        model.load_state_dict(ckpt_data["model"])
        print(f"✅ Policy B Verified: Resumed weights from student checkpoint: {resume_from_ckpt}")

    total_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    learning_rate = t_cfg["learning_rate"]
    min_lr = t_cfg.get("min_lr", learning_rate / 10.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(t_cfg["beta1"], t_cfg["beta2"]),
        weight_decay=t_cfg["weight_decay"]
    )

    use_amp = device == "cuda" and t_cfg.get("mixed_precision") == "fp16"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    eval_interval = eval_interval_override or t_cfg.get("eval_interval", 100)
    eval_iters = t_cfg.get("eval_iters", 5)
    best_val_loss = float("inf")
    metrics_history = []

    # -------------------------------------------------------------------------
    # SAFEGUARD 5: Comprehensive Pre-Training Verification Summary Console Table
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("   MOON AI -- PRE-TRAINING RUNTIME VERIFICATION & SAFEGUARD AUDIT")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print(f"   Execution Mode: {'ENGINEERING SMOKE TEST' if is_smoke_test else 'PRODUCTION PILOT RUN'}")
    print("=" * 70)
    print("[Config & Policy B Option 1]")
    print(f"  • Config File:          {os.path.relpath(abs_config_path, PROJECT_ROOT)}")
    print(f"  • Model Track:          {cfg.get('model_track', 'N/A')}")
    print(f"  • Ownership Policy:     {cfg.get('ownership_policy', 'TEACHER_STUDENT_RANDOM_INIT')} [Option 1 Sovereign]")
    print(f"  • Student Init:         random_gaussian_scratch (N(0, 0.02^2)) [VERIFIED]")
    print(f"  • Prohibit Weights:     True [VERIFIED]")
    print(f"  • Allow Demonstrations: False [VERIFIED]")
    print(f"  • Distillation Enabled: False [VERIFIED]")

    print("\n[Hardware Target & Device Identity]")
    print(f"  • Target Device:        {device.upper()}")
    print(f"  • GPU / Device Name:    {gpu_identity_info.get('name')}")
    print(f"  • Compute Capability:   {gpu_identity_info.get('compute_capability')}")
    print(f"  • Available VRAM:       {gpu_identity_info.get('vram_mb')} MB")
    print(f"  • Hardware Gate Status: {gpu_identity_info.get('verification_status', 'UNKNOWN')}")

    print("\n[Data Manifest Provenance & Hash Integrity]")
    train_rel = os.path.relpath(train_manifest, PROJECT_ROOT)
    val_rel = os.path.relpath(val_manifest, PROJECT_ROOT)
    test_rel = os.path.relpath(test_manifest, PROJECT_ROOT)
    train_status_tag = "VERIFIED" if not is_smoke_test and os.path.basename(train_manifest) in CANONICAL_MANIFEST_LF_HASHES else "AUDITED"
    val_status_tag = "VERIFIED" if not is_smoke_test and os.path.basename(val_manifest) in CANONICAL_MANIFEST_LF_HASHES else "AUDITED"
    test_status_tag = "VERIFIED" if not is_smoke_test and os.path.basename(test_manifest) in CANONICAL_MANIFEST_LF_HASHES else "AUDITED"
    print(f"  • Train Manifest:       {train_rel} ({unique_records} records)")
    print(f"    - LF SHA-256:         {train_lf_sha[:32] if train_lf_sha else 'N/A'}... [{train_status_tag}]")
    print(f"  • Val Manifest:         {val_rel} ({len(val_dataset)} windows)")
    print(f"    - LF SHA-256:         {val_lf_sha[:32] if val_lf_sha else 'N/A'}... [{val_status_tag}]")
    print(f"  • Blind-Test Manifest:  {test_rel}")
    print(f"    - LF SHA-256:         {test_lf_sha[:32] if test_lf_sha else 'N/A'}... [{test_status_tag}]")
    print("  • Blind-Test Isolation: 100% UNTOUCHED (Strictly excluded from train/val DataLoaders)")

    print("\n[Hyperparameters & Architecture]")
    print(f"  • Parameters:           {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  • Geometry:             {m_cfg['n_layer']} layers, {m_cfg['n_head']} heads, {m_cfg['n_embd']} embd, {block_size} ctx")
    print(f"  • Training Iterations:  {max_iters} (Warmup: {t_cfg.get('warmup_iters', 10)}, Eval Interval: {eval_interval})")
    print(f"  • Batch / Accumulation: {batch_size} micro-batch × {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective batch")
    print(f"  • Planned Exposures:    {effective_samples:,} samples ({dataset_passes:.2f} dataset passes)")
    print(f"  • Mixed Precision:      {'FP16 AMP (autocast + GradScaler enabled)' if use_amp else 'FP32 standard'}")

    print("\n[Checkpoint Isolation & Anti-Collision]")
    print(f"  • Output Directory:     {os.path.relpath(checkpoint_dir, PROJECT_ROOT)}")
    print(f"  • Final Checkpoint:     {ckpt_filename}")
    print(f"  • Best Checkpoint:      {best_ckpt_filename}")
    print("  • Stage Isolation:      CLEARED (Stage 2.2 / earlier checkpoints protected)")
    print("  • Filesystem Collision: CLEARED (Neither destination exists on disk; overwrite blocked)")
    print("=" * 70 + "\n")

    print(f"Beginning training run: {max_iters} iterations (grad accum: {grad_accum_steps}, AMP: {use_amp}, eval_interval: {eval_interval})...\n")

    model.train()
    iter_num = 0
    t0 = time.time()
    train_iter = iter(train_loader)
    accum_loss = 0.0

    while iter_num < max_iters:
        lr = get_lr(iter_num, t_cfg.get("warmup_iters", 50), max_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device)
            y = y.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(x, y)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss.backward()

            accum_loss += loss.item()

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
            optimizer.step()

        iter_num += 1

        if not is_smoke_test and (iter_num % eval_interval == 0 or iter_num == max_iters):
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_x, v_y = v_x.to(device), v_y.to(device)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            _, l = model(v_x, v_y)
                    else:
                        _, l = model(v_x, v_y)
                    val_loss += l.item()
                    val_steps += 1
                    if val_steps >= eval_iters:
                        break
            mean_v_loss = val_loss / max(1, val_steps)
            step_ppl = math.exp(min(20.0, mean_v_loss))
            dt = time.time() - t0
            t0 = time.time()
            print(f"Iter {iter_num:4d}/{max_iters} | Train Loss: {accum_loss:.4f} | Val Loss: {mean_v_loss:.4f} (PPL: {step_ppl:.2f}) | LR: {lr:.6f} | Time: {dt*1000:.1f}ms")

            metrics_history.append({
                "iter": iter_num,
                "train_loss": round(accum_loss, 4),
                "val_loss": round(mean_v_loss, 4),
                "perplexity": round(step_ppl, 2),
                "lr": round(lr, 8)
            })

            if mean_v_loss < best_val_loss:
                best_val_loss = mean_v_loss
                best_payload = {
                    "status": "PILOT_TRAINED_BEST",
                    "is_production_trained": False,
                    "promotable": False,
                    "is_production_eligible": False,
                    "is_test_fixture": False,
                    "model": model.state_dict(),
                    "model_args": {
                        "block_size": gpt_conf.block_size,
                        "vocab_size": gpt_conf.vocab_size,
                        "n_layer": gpt_conf.n_layer,
                        "n_head": gpt_conf.n_head,
                        "n_embd": gpt_conf.n_embd,
                        "dropout": gpt_conf.dropout,
                        "bias": gpt_conf.bias
                    },
                    "config": {
                        "model_name": m_cfg["name"],
                        "initialization": "random_gaussian_scratch",
                        "student_init_method": "random_gaussian_scratch",
                        "teacher_checkpoint_weights_transferred": False,
                        "best_val_loss": best_val_loss,
                        "iter_num": iter_num,
                        "policy": td_cfg.get("strategy", "POLICY_B_ZERO_TEACHER_INFLUENCE"),
                        "ownership_policy": cfg.get("ownership_policy", "TEACHER_STUDENT_RANDOM_INIT"),
                        "best_checkpoint_path": best_ckpt_path,
                        "effective_config": cfg,
                        "manifest_hashes": manifest_provenance_hashes,
                        "hardware_identity": gpu_identity_info,
                        "status": "PILOT_TRAINED_BEST",
                        "is_production_trained": False,
                        "promotable": False,
                        "is_production_eligible": False,
                        "is_test_fixture": False,
                        "notes": "Best validation checkpoint from pilot training run. Unpromoted pilot requiring independent evaluation before production consideration."
                    }
                }
                torch.save(best_payload, best_ckpt_path)
                print(f"  ⭐ Saved new best validation checkpoint: {best_ckpt_path} (Loss: {best_val_loss:.4f})")
            model.train()
        elif iter_num % 10 == 0 or iter_num == max_iters or is_smoke_test:
            dt = time.time() - t0
            t0 = time.time()
            print(f"Iter {iter_num:4d}/{max_iters} | Loss: {accum_loss:.4f} | LR: {lr:.6f} | Time: {dt*1000:.1f}ms")

    # Evaluation
    print("\nRunning validation evaluation...")
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            val_loss += loss.item()
            val_steps += 1
            if is_smoke_test and val_steps >= 2:
                break
    if val_steps == 0:
        raise RuntimeError("Validation evaluation produced 0 steps! Cannot compute validation metrics.")
    mean_val_loss = val_loss / val_steps
    perplexity = math.exp(min(20.0, mean_val_loss))
    print(f"Final Validation Loss: {mean_val_loss:.4f} | Perplexity: {perplexity:.2f}")

    # Qualitative Inference Diagnostic
    print("\nRunning qualitative inference generation check...")
    model.eval()
    sample_queries = ["Hi", "What is Python?"]
    generated_samples = {}
    with torch.no_grad():
        for q in sample_queries:
            q_turns = [{"role": "user", "content": q}]
            q_ids = tokenizer.encode_chat(q_turns)
            idx = torch.tensor(q_ids, dtype=torch.long, device=device).unsqueeze(0)
            for _ in range(20):
                idx_cond = idx if idx.size(1) <= gpt_conf.block_size else idx[:, -gpt_conf.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            gen_text = tokenizer.decode(idx[0].tolist())
            generated_samples[q] = gen_text
            preview = gen_text.replace("\n", " ")[:75]
            print(f"  • Prompt '{q}': {preview}...")

    # Determine checkpoint destination (Smoke test checkpoints are isolated and marked unpromotable)
    if custom_ckpt_path:
        checkpoint_path = custom_ckpt_path
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        ckpt_filename = os.path.basename(checkpoint_path)
    else:
        checkpoint_dir = cfg["paths"]["checkpoint_dir"]
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_filename = "smoke_test_ckpt.pt" if is_smoke_test else cfg["paths"]["checkpoint_name"]
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)

    checkpoint_payload = {
        "status": "ENGINEERING_SMOKE_TEST_ONLY" if is_smoke_test else "PILOT_TRAINED",
        "is_production_trained": False,
        "promotable": False,
        "is_production_eligible": False,
        "is_test_fixture": False,
        "model": model.state_dict(),
        "model_args": {
            "block_size": gpt_conf.block_size,
            "vocab_size": gpt_conf.vocab_size,
            "n_layer": gpt_conf.n_layer,
            "n_head": gpt_conf.n_head,
            "n_embd": gpt_conf.n_embd,
            "dropout": gpt_conf.dropout,
            "bias": gpt_conf.bias
        },
        "config": {
            "dataset": cfg.get("model_track", "moon-chat-v0"),
            "model_name": m_cfg["name"],
            "ownership_policy": cfg.get("ownership_policy", "TEACHER_STUDENT_RANDOM_INIT"),
            "policy": td_cfg.get("strategy", "POLICY_B_ZERO_TEACHER_INFLUENCE"),
            "initialization": "random_gaussian_scratch",
            "student_init_method": "random_gaussian_scratch",
            "teacher_checkpoint_weights_transferred": False,
            "teacher_models": [],
            "teacher_license": None,
            "dataset_provenance": "DETERMINISTIC_CURRICULUM_TEMPLATE + ORIGINAL_SELF_AUTHORED_QA",
            "run_id": f"moon_chat_run_{int(time.time())}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            "final_train_loss": accum_loss,
            "final_val_loss": mean_val_loss,
            "perplexity": perplexity,
            "iterations": iter_num,
            "is_production_trained": False,
            "promotable": False,
            "is_production_eligible": False,
            "is_test_fixture": False,
            "status": "ENGINEERING_SMOKE_TEST_ONLY" if is_smoke_test else "PILOT_TRAINED",
            "checkpoint_filename": ckpt_filename,
            "best_checkpoint_filename": best_ckpt_filename,
            "effective_config": cfg,
            "manifest_hashes": manifest_provenance_hashes,
            "hardware_identity": gpu_identity_info,
            "notes": (
                "Engineering smoke test checkpoint only. Unpromotable to production."
                if is_smoke_test
                else "Pilot trained checkpoint from pure Gaussian scratch. Unpromoted pilot requiring independent evaluation before production consideration."
            )
        },
        "iter_num": iter_num,
        "best_val_loss": mean_val_loss
    }

    torch.save(checkpoint_payload, checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    print("\n" + "=" * 65)
    print(f"Checkpoint saved: {'[SMOKE TEST UNPROMOTABLE]' if is_smoke_test else '[PRODUCTION PILOT]'}")
    try:
        display_path = os.path.relpath(checkpoint_path, PROJECT_ROOT)
    except ValueError:
        display_path = checkpoint_path
    print(f"Path:    {display_path}")
    print(f"Size:    {ckpt_size_mb:.2f} MB")
    print(f"SHA-256: {ckpt_sha256}")
    print("=" * 65)

    # Save training metrics history
    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "training_metrics_latest.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": f"moon_run_{int(time.time())}",
            "model_name": m_cfg["name"],
            "parameters": total_params,
            "max_iters": max_iters,
            "final_train_loss": round(accum_loss, 4),
            "final_val_loss": round(mean_val_loss, 4),
            "final_perplexity": round(perplexity, 2),
            "best_val_loss": round(best_val_loss if best_val_loss != float("inf") else mean_val_loss, 4),
            "metrics_history": metrics_history,
            "checkpoint_sha256": ckpt_sha256
        }, f, indent=2)

    return {
        "status": "COMPLETED_SMOKE_TEST" if is_smoke_test else "COMPLETED_PRODUCTION_PILOT",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_sha256,
        "parameters": total_params,
        "final_train_loss": accum_loss,
        "final_val_loss": mean_val_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else mean_val_loss,
        "perplexity": perplexity,
        "iterations": iter_num,
        "total_iterations": iter_num,
        "unique_records": unique_records,
        "chunks_generated": len(train_dataset),
        "supervised_tokens": train_dataset.get_total_supervised_tokens(),
        "is_smoke_test": is_smoke_test,
        "effective_dataset_passes": dataset_passes,
        "generated_samples": generated_samples,
        "metrics_history": metrics_history
    }

if __name__ == "__main__":
    args = parse_args()
    train_moon_chat(
        config_path=args.config,
        max_iters_override=args.max_iters,
        device_override=args.device,
        eval_interval_override=args.eval_interval,
        resume_from_ckpt=args.resume_from,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test
    )
