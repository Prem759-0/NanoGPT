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
import sys
import time
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from model import GPTConfig, GPT
from datasets.chat.tokenizer import SovereignTokenizer
from training.chat.dataset import SovereignChatDataset
from training.chat.readiness_gate import ChatReadinessGate, STATUS_READY_FOR_PILOT

def parse_args():
    parser = argparse.ArgumentParser(description="Moon AI Sovereign Chat Training Engine")
    parser.add_argument("--config", type=str, default="training/chat/config.yaml", help="Path to training config YAML")
    parser.add_argument("--max-iters", type=int, default=None, help="Override maximum training iterations")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps)")
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

    print("=" * 65)
    print("   MOON AI -- SOVEREIGN CHAT MODEL TRAINING (moon-chat-v0)")
    print("=" * 65)
    policy = cfg.get("ownership_policy", "TEACHER_STUDENT_RANDOM_INIT")
    print(f"Ownership Policy:   {policy}")
    print("Student Init:       RANDOM GAUSSIAN SCRATCH (Zero Pretrained Checkpoint Weights)")
    print(f"Execution Mode:     {'ENGINEERING SMOKE TEST' if is_smoke_test else 'PRODUCTION PILOT RUN'}\n")

    # Resolve manifest paths
    train_manifest = cfg["paths"]["train_manifest"]
    if not os.path.isabs(train_manifest):
        train_manifest = os.path.join(PROJECT_ROOT, train_manifest)

    val_manifest = cfg["paths"]["val_manifest"]
    if not os.path.isabs(val_manifest):
        val_manifest = os.path.join(PROJECT_ROOT, val_manifest)

    test_manifest = cfg["paths"].get("test_manifest", "datasets/manifests/chat_test.jsonl")
    if not os.path.isabs(test_manifest):
        test_manifest = os.path.join(PROJECT_ROOT, test_manifest)

    # Readiness Gate Check: Block production training if not READY_FOR_PILOT
    if not is_smoke_test:
        gate = ChatReadinessGate(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest
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

    # Determine device
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Target compute device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"CUDA GPU: {gpu_name} ({gpu_mem_gb} GB VRAM)")

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
    print(f"Loaded Sovereign Tokenizer: {tokenizer.vocab_size} tokens")

    # Instantiate datasets with multi-window chunking
    train_manifest = cfg["paths"]["train_manifest"]
    if not os.path.isabs(train_manifest):
        train_manifest = os.path.join(PROJECT_ROOT, train_manifest)

    val_manifest = cfg["paths"]["val_manifest"]
    if not os.path.isabs(val_manifest):
        val_manifest = os.path.join(PROJECT_ROOT, val_manifest)

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

    # Dataset Exposure Analysis
    t_cfg = cfg["training"]
    max_iters = 1 if is_smoke_test else (max_iters_override or t_cfg["max_iters"])
    grad_accum_steps = 1 if is_smoke_test else t_cfg.get("gradient_accumulation_steps", 1)

    unique_records = train_dataset.get_unique_record_count()
    total_train_windows = len(train_dataset)
    supervised_tokens = train_dataset.get_total_supervised_tokens()
    effective_samples = max_iters * batch_size * grad_accum_steps
    dataset_passes = effective_samples / max(1, total_train_windows)

    print("\nDataset Exposure & Accounting Metrics:")
    print(f"  • Unique conversation records:       {unique_records}")
    print(f"  • Multi-window training chunks:      {total_train_windows} (val: {len(val_dataset)})")
    print(f"  • Supervised assistant target tokens: {supervised_tokens:,}")
    print(f"  • Planned sample exposures:          {effective_samples:,} (iters: {max_iters}, batch: {batch_size}, accum: {grad_accum_steps})")
    print(f"  • Effective dataset passes (epochs): {dataset_passes:.1f}")
    if dataset_passes > 10 and not is_smoke_test:
        print(f"  ⚠️ HIGH OVERFITTING RISK: {dataset_passes:.1f} passes over {unique_records} unique records!")

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

    print("\nInitializing model from random Gaussian initialization (zero pretrained weights)...")
    model = GPT(gpt_conf)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture initialized: {total_params:,} parameters ({total_params / 1e6:.2f}M)")

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

    print(f"\nBeginning training run: {max_iters} iterations (grad accum: {grad_accum_steps}, AMP: {use_amp})...\n")

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

        if iter_num % 10 == 0 or iter_num == max_iters or is_smoke_test:
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
    else:
        checkpoint_dir = cfg["paths"]["checkpoint_dir"]
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_filename = "smoke_test_ckpt.pt" if is_smoke_test else cfg["paths"]["checkpoint_name"]
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)

    checkpoint_payload = {
        "status": "ENGINEERING_SMOKE_TEST_ONLY" if is_smoke_test else "PILOT_TRAINED",
        "is_production_trained": not is_smoke_test,
        "promotable": not is_smoke_test,
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
            "dataset": "moon-chat-v0",
            "model_name": m_cfg["name"],
            "ownership_policy": cfg.get("ownership_policy", "TEACHER_STUDENT_RANDOM_INIT"),
            "initialization": "random_gaussian_scratch",
            "student_init_method": "random_gaussian_scratch",
            "teacher_checkpoint_weights_transferred": False,
            "teacher_models": ["Qwen2.5-7B-Instruct-GGUF"],
            "teacher_license": "Apache-2.0",
            "dataset_provenance": "DETERMINISTIC_CURRICULUM_TEMPLATE + ORIGINAL_SELF_AUTHORED_QA",
            "run_id": f"moon_chat_run_{int(time.time())}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            "final_train_loss": accum_loss,
            "final_val_loss": mean_val_loss,
            "perplexity": perplexity,
            "iterations": iter_num,
            "is_production_trained": not is_smoke_test,
            "status": "ENGINEERING_SMOKE_TEST_ONLY" if is_smoke_test else "PILOT_TRAINED",
            "notes": (
                "Engineering smoke test checkpoint only. Unpromotable to production."
                if is_smoke_test
                else "Pilot trained checkpoint from pure Gaussian scratch."
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

    return {
        "status": "COMPLETED_SMOKE_TEST" if is_smoke_test else "COMPLETED_PRODUCTION_PILOT",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_sha256,
        "parameters": total_params,
        "final_train_loss": accum_loss,
        "final_val_loss": mean_val_loss,
        "perplexity": perplexity,
        "iterations": iter_num,
        "total_iterations": iter_num,
        "unique_records": unique_records,
        "chunks_generated": len(train_dataset),
        "supervised_tokens": train_dataset.get_total_supervised_tokens(),
        "is_smoke_test": is_smoke_test,
        "effective_dataset_passes": dataset_passes,
        "generated_samples": generated_samples
    }

if __name__ == "__main__":
    args = parse_args()
    train_moon_chat(
        config_path=args.config,
        max_iters_override=args.max_iters,
        device_override=args.device,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test
    )
