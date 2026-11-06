"""
scripts/validate_t4_empirical.py
Moon AI — Empirical T4 Hardware Validation Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Rigorous empirical benchmark script for execution in NVIDIA T4 GPU environments (e.g. Google Colab).
Validates:
1. torch.cuda.is_available() & compute capability
2. Device name & total available physical VRAM
3. FP16 automatic mixed precision (AMP) autocast
4. torch.amp.GradScaler operational integrity
5. Forward and backward passes
6. 4-step gradient accumulation
7. Gradient norm clipping (max_norm=1.0)
8. AdamW optimizer parameter update
9. NaN / Inf numerical stability detection
10. Checkpoint save & load bitwise verification roundtrip
11. Empirical peak allocated and reserved VRAM measurements

Outputs:
- "T4_EMPIRICAL_BENCHMARK_VERIFIED" upon complete physical validation pass.
- Halts with "CUDA_UNAVAILABLE" blocker if executed without a CUDA GPU.
"""

import os
import sys
import time
import json
import math
import hashlib
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from model import GPTConfig, GPT


def run_empirical_t4_validation() -> Dict[str, Any]:
    print("=" * 70)
    print("   MOON AI — EMPIRICAL T4 GPU HARDWARE BENCHMARK & VALIDATION")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("   Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)")
    print("=" * 70)

    # 1. CUDA Availability Check
    if not torch.cuda.is_available():
        print("\n❌ CRITICAL BLOCKER: torch.cuda.is_available() returned False.")
        print("Hardware Status: CUDA is NOT available on this machine/runtime.")
        print("Execution Result: CUDA_UNAVAILABLE")
        print("\nExplanation:")
        print("Live T4 GPU validation requires a physical NVIDIA T4 GPU environment")
        print("(e.g., Google Colab with T4 GPU runtime enabled).")
        print("Simulated or analytical GPU benchmarks are strictly prohibited by Moon AI policy.\n")
        print("=" * 70)
        return {
            "status": "CUDA_UNAVAILABLE",
            "error": "CUDA is not available on this host.",
            "device": "cpu",
            "empirical_benchmark_verified": False
        }

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_vram_mb = round(device_props.total_memory / (1024 ** 2), 2)
    major, minor = device_props.major, device_props.minor
    compute_cap = f"{major}.{minor}"

    print(f"\n[1/8] CUDA Device Detected: {device_name}")
    print(f"      • Compute Capability:     {compute_cap}")
    print(f"      • Total Physical VRAM:    {total_vram_mb:,.1f} MB ({total_vram_mb/1024:.2f} GB)")
    print(f"      • Multi-Processor Count:   {device_props.multi_processor_count}")

    # 2. Reset peak memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_alloc_mb = round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2)

    # 3. Model Architecture Instantiation (Student: 25.7M Moon Chat GPT)
    print("\n[2/8] Instantiating 25.7M Moon Chat Student Model (Policy B)...")
    gpt_conf = GPTConfig(
        block_size=512,
        vocab_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=False
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = GPT(gpt_conf).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      • Exact Parameter Count:   {total_params:,} (25.7M target: 25,698,816)")
    assert total_params == 25698816, f"Expected 25,698,816 parameters, got {total_params}"

    model_weight_mb = round(torch.cuda.memory_allocated(device) / (1024 ** 2) - initial_alloc_mb, 2)
    print(f"      • Model VRAM Footprint:    {model_weight_mb} MB")

    # 4. Optimizer and FP16 GradScaler Initialization
    print("\n[3/8] Initializing AdamW Optimizer and FP16 GradScaler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # 5. 4-Step Gradient Accumulation Simulation with FP16 Autocast
    micro_batch_size = 4
    seq_len = 512
    grad_accum_steps = 4
    print(f"\n[4/8] Executing {grad_accum_steps}-Step Gradient Accumulation (micro_batch={micro_batch_size}, seq_len={seq_len}, FP16 AMP)...")

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0

    t_start = time.perf_counter()
    for micro_step in range(grad_accum_steps):
        # Generate representative random batch of token IDs
        x = torch.randint(0, 512, (micro_batch_size, seq_len), device=device, dtype=torch.long)
        y = torch.randint(0, 512, (micro_batch_size, seq_len), device=device, dtype=torch.long)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, loss = model(x, y)
            loss_scaled = loss / grad_accum_steps

        # Check for NaN / Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(f"NaN/Inf detected in loss at micro-step {micro_step}")

        scaler.scale(loss_scaled).backward()
        accumulated_loss += loss.item()
        print(f"      • Micro-step {micro_step + 1}/{grad_accum_steps}: loss = {loss.item():.4f}")

    # 6. Unscaling, Gradient Norm Clipping, and Optimizer Step
    print("\n[5/8] Unscaling gradients, clipping norm, and advancing optimizer step...")
    scaler.unscale_(optimizer)

    # Check for NaN / Inf in gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        raise FloatingPointError(f"NaN/Inf detected in gradient norm: {grad_norm}")
    print(f"      • Clipped Gradient Norm:   {grad_norm.item():.4f}")

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    print(f"      • 4-Step Accumulation Time: {t_elapsed_ms} ms")

    # 7. VRAM Measurement
    peak_allocated_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    peak_reserved_mb = round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2)
    headroom_mb = round(total_vram_mb - peak_reserved_mb, 2)
    vram_utilization_pct = round((peak_reserved_mb / total_vram_mb) * 100, 2)

    print("\n[6/8] Physical VRAM Accounting on Live GPU:")
    print(f"      • Peak VRAM Allocated:     {peak_allocated_mb:,.1f} MB")
    print(f"      • Peak VRAM Reserved:      {peak_reserved_mb:,.1f} MB")
    print(f"      • VRAM Headroom Remaining: {headroom_mb:,.1f} MB ({headroom_mb/1024:.2f} GB)")
    print(f"      • VRAM Utilization:        {vram_utilization_pct:.2f}% of {total_vram_mb/1024:.1f} GB")

    # 8. Checkpoint Save/Load Bitwise Integrity Roundtrip
    print("\n[7/8] Testing Checkpoint Save & Load Bitwise Integrity Roundtrip...")
    temp_ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat")
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    temp_ckpt_path = os.path.join(temp_ckpt_dir, "moon_chat_t4_empirical_test.pt")

    checkpoint_payload = {
        "status": "EMPIRICAL_T4_BENCHMARK_CHECKPOINT",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": {
            "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
            "teacher_checkpoint_weights_transferred": False,
            "initialization": "random_gaussian_scratch",
            "device": device_name,
            "peak_vram_mb": peak_reserved_mb
        }
    }
    torch.save(checkpoint_payload, temp_ckpt_path)
    file_size_mb = round(os.path.getsize(temp_ckpt_path) / (1024 ** 2), 2)

    with open(temp_ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Load back into fresh model
    fresh_model = GPT(gpt_conf).to(device)
    loaded_ckpt = torch.load(temp_ckpt_path, map_location=device, weights_only=False)
    fresh_model.load_state_dict(loaded_ckpt["model"])

    # Bitwise comparison
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        if not torch.equal(v1, v2):
            raise AssertionError(f"Bitwise parameter mismatch in key '{k1}' during checkpoint reload!")

    print(f"      • Checkpoint Size:         {file_size_mb} MB")
    print(f"      • Checkpoint SHA-256:      {ckpt_sha256[:16]}...{ckpt_sha256[-8:]}")
    print("      • Bitwise Verification:    100% IDENTICAL (All 25.7M parameters restored bit-for-bit)")

    # Cleanup temporary checkpoint
    if os.path.exists(temp_ckpt_path):
        os.remove(temp_ckpt_path)

    # 9. Final Benchmark Summary
    summary = {
        "status": "T4_EMPIRICAL_BENCHMARK_VERIFIED",
        "device_name": device_name,
        "compute_capability": compute_cap,
        "total_vram_mb": total_vram_mb,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "vram_headroom_mb": headroom_mb,
        "vram_utilization_pct": vram_utilization_pct,
        "model_parameters": total_params,
        "fp16_amp_verified": True,
        "grad_scaler_verified": True,
        "grad_accumulation_steps": grad_accum_steps,
        "step_latency_ms": t_elapsed_ms,
        "checkpoint_roundtrip_verified": True,
        "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
        "teacher_checkpoint_weights_transferred": False
    }

    report_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "t4_empirical_benchmark.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[8/8] Benchmark Results Exported:")
    print(f"      • JSON Report:             {report_path}")
    print("\n" + "=" * 70)
    print("   STATUS: T4_EMPIRICAL_BENCHMARK_VERIFIED")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    result = run_empirical_t4_validation()
    if result["status"] != "T4_EMPIRICAL_BENCHMARK_VERIFIED":
        sys.exit(1)
    sys.exit(0)
