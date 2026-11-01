"""
Moon AI — Sovereign Chat Transformer Training Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Trains Moon Chat (Causal Autoregressive Transformer) from purely random weights.
Compatible with Google Colab NVIDIA T4 (15.3 GB VRAM).
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

def parse_args():
    parser = argparse.ArgumentParser(description="Moon AI Sovereign Chat Training Engine")
    parser.add_argument("--config", type=str, default="training/chat/config.yaml", help="Path to training config YAML")
    parser.add_argument("--max-iters", type=int, default=None, help="Override maximum training iterations")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 iterations to verify pipeline integrity")
    return parser.parse_args()

def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0 to 1
    return min_lr + coeff * (lr - min_lr)

def train_moon_chat(config_path: str, max_iters_override: Optional[int] = None, device_override: Optional[str] = None, dry_run: bool = False):
    print("=" * 65)
    print("   MOON AI — SOVEREIGN CHAT MODEL TRAINING (moon-chat-v0)")
    print("=" * 65)
    print("Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY")
    print("Pretrained Weights: STRICTLY PROHIBITED (Random Gaussian Init Only)\n")

    # Load configuration
    abs_config_path = os.path.abspath(config_path)
    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"Config file not found: {abs_config_path}")

    with open(abs_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    tok_path = os.path.join(PROJECT_ROOT, cfg["paths"]["tokenizer_path"])
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Sovereign tokenizer not found at: {tok_path}. Run scripts/build_chat_dataset.py first.")
    tokenizer = SovereignTokenizer.load(tok_path)
    print(f"Loaded Sovereign Tokenizer: {tokenizer.vocab_size} tokens")

    # Instantiate datasets
    train_manifest = os.path.join(PROJECT_ROOT, cfg["paths"]["train_manifest"])
    val_manifest = os.path.join(PROJECT_ROOT, cfg["paths"]["val_manifest"])

    m_cfg = cfg["model"]
    block_size = m_cfg["block_size"]

    train_dataset = SovereignChatDataset(train_manifest, tokenizer, block_size=block_size)
    val_dataset = SovereignChatDataset(val_manifest, tokenizer, block_size=block_size)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Populate datasets/manifests/chat_train.jsonl first.")

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
    t_cfg = cfg["training"]
    learning_rate = t_cfg["learning_rate"]
    min_lr = t_cfg.get("min_lr", learning_rate / 10.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(t_cfg["beta1"], t_cfg["beta2"]),
        weight_decay=t_cfg["weight_decay"]
    )

    max_iters = max_iters_override or t_cfg["max_iters"]
    if dry_run:
        max_iters = 1
        grad_accum_steps = 1
        print("Dry run requested: executing 1 iteration with grad_accum_steps=1 to verify pipeline.", flush=True)
    else:
        grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 1)
    use_amp = device == "cuda" and t_cfg.get("mixed_precision") == "fp16"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nBeginning training run: {max_iters} iterations (grad accum: {grad_accum_steps}, AMP: {use_amp})...\n")

    model.train()
    iter_num = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    while iter_num < max_iters:
        # Determine learning rate
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

        if iter_num % 10 == 0 or iter_num == max_iters or dry_run:
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
    mean_val_loss = val_loss / max(1, val_steps)
    print(f"Final Validation Loss: {mean_val_loss:.4f}")

    # Save Sovereign Checkpoint
    checkpoint_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, cfg["paths"]["checkpoint_name"])

    checkpoint_payload = {
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
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "initialization": "random_gaussian_scratch",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            "final_train_loss": accum_loss,
            "final_val_loss": mean_val_loss,
            "iterations": iter_num
        },
        "iter_num": iter_num,
        "best_val_loss": mean_val_loss
    }

    torch.save(checkpoint_payload, checkpoint_path)

    # Compute SHA-256 of checkpoint
    with open(checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    print(f"\n🎉 Checkpoint saved successfully!")
    print(f"Path:    {os.path.relpath(checkpoint_path, PROJECT_ROOT)}")
    print(f"Size:    {ckpt_size_mb:.2f} MB")
    print(f"SHA-256: {ckpt_sha256}")
    print("=" * 65)

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_sha256,
        "parameters": total_params,
        "final_val_loss": mean_val_loss,
        "iterations": iter_num
    }

if __name__ == "__main__":
    args = parse_args()
    train_moon_chat(
        config_path=args.config,
        max_iters_override=args.max_iters,
        device_override=args.device,
        dry_run=args.dry_run
    )
