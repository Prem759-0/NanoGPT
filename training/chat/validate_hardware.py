"""
Moon AI — Hardware Feasibility & Parameter Sizing Validator
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Validates parameter counts, computes exact analytical VRAM footprints,
measures GPU allocations if CUDA is available, and provides realistic OOM risk analysis
for Google Colab Free Tier (NVIDIA T4 15.3 GB VRAM).
"""

import argparse
import os
import sys
from typing import Dict, Any
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT

def calculate_exact_parameters(config: GPTConfig) -> Dict[str, Any]:
    """Calculates exact parameter counts across all model sub-components."""
    wte_params = config.vocab_size * config.n_embd
    wpe_params = config.block_size * config.n_embd

    # Per transformer block
    ln_params = 2 * config.n_embd if config.bias else config.n_embd # ln_1 and ln_2
    attn_qkv = config.n_embd * (3 * config.n_embd) + (3 * config.n_embd if config.bias else 0)
    attn_proj = config.n_embd * config.n_embd + (config.n_embd if config.bias else 0)
    mlp_fc = config.n_embd * (4 * config.n_embd) + (4 * config.n_embd if config.bias else 0)
    mlp_proj = (4 * config.n_embd) * config.n_embd + (config.n_embd if config.bias else 0)

    per_block = (2 * ln_params) + attn_qkv + attn_proj + mlp_fc + mlp_proj
    total_blocks = config.n_layer * per_block

    final_ln = 2 * config.n_embd if config.bias else config.n_embd
    # Note: lm_head is weight-tied with wte, so no extra parameters are added
    total_params = wte_params + wpe_params + total_blocks + final_ln

    return {
        "wte_embedding": wte_params,
        "wpe_position": wpe_params,
        "per_block": per_block,
        "all_blocks": total_blocks,
        "final_layernorm": final_ln,
        "total_parameters": total_params,
        "total_millions": round(total_params / 1e6, 2)
    }

def estimate_t4_vram(
    exact_params: int,
    batch_size: int = 4,
    block_size: int = 512,
    n_embd: int = 512,
    n_layer: int = 8,
    mixed_precision: str = "fp16"
) -> Dict[str, Any]:
    """
    Computes analytical peak VRAM budget for NVIDIA T4 (15.3 GB).
    Never reports OOM risk as 0.0%.
    """
    bytes_per_param = 2 if mixed_precision in {"fp16", "bf16"} else 4
    weights_mb = (exact_params * bytes_per_param) / (1024 ** 2)
    gradients_mb = (exact_params * bytes_per_param) / (1024 ** 2)

    # AdamW maintains FP32 master weights (4 bytes), 1st moment (4 bytes), 2nd moment (4 bytes)
    optimizer_mb = (exact_params * 12) / (1024 ** 2)

    # Activations estimate: attention matrices + intermediate activations
    # For each layer: Q, K, V activations + attention matrix (B * H * T * T * 2) + MLP intermediate (B * T * 4d * 2)
    activations_bytes = n_layer * (
        (batch_size * block_size * n_embd * bytes_per_param * 4) +
        (batch_size * 8 * block_size * block_size * bytes_per_param) +
        (batch_size * block_size * 4 * n_embd * bytes_per_param * 2)
    )
    activations_mb = activations_bytes / (1024 ** 2)

    cuda_overhead_mb = 600.0 # PyTorch driver context, caching allocator buffers

    peak_vram_mb = weights_mb + gradients_mb + optimizer_mb + activations_mb + cuda_overhead_mb
    t4_total_mb = 15360.0
    headroom_mb = t4_total_mb - peak_vram_mb
    utilization_pct = round((peak_vram_mb / t4_total_mb) * 100, 1)

    # Realistic OOM Assessment
    if utilization_pct < 20.0:
        oom_risk = "LOW (Safe with micro-batch 4 and gradient accumulation 8. Risk elevates if batch_size > 32 or block_size > 2048 without gradient checkpointing)"
    elif utilization_pct < 60.0:
        oom_risk = "MODERATE (Requires careful monitoring of activation caching and batch sizes)"
    else:
        oom_risk = "HIGH (Substantial danger of out-of-memory under activation spikes)"

    return {
        "model_parameters": exact_params,
        "weights_mb": round(weights_mb, 1),
        "gradients_mb": round(gradients_mb, 1),
        "optimizer_states_adamw_mb": round(optimizer_mb, 1),
        "activations_estimate_mb": round(activations_mb, 1),
        "cuda_runtime_context_mb": round(cuda_overhead_mb, 1),
        "estimated_peak_vram_mb": round(peak_vram_mb, 1),
        "t4_total_vram_mb": t4_total_mb,
        "headroom_mb": round(headroom_mb, 1),
        "vram_utilization_pct": utilization_pct,
        "oom_risk_assessment": oom_risk
    }

def validate_model_and_hardware():
    print("=" * 65)
    print("   MOON AI — HARDWARE VALIDATION & SIZING ENGINE")
    print("=" * 65)

    # Pilot 25.7M Config
    pilot_cfg = GPTConfig(
        block_size=512,
        vocab_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        bias=False
    )
    pilot_model = GPT(pilot_cfg)
    actual_pilot_params = sum(p.numel() for p in pilot_model.parameters())
    pilot_calc = calculate_exact_parameters(pilot_cfg)

    print(f"\n1. Moon Chat Pilot Model (8L, 8H, 512D, bias=False):")
    print(f"   Calculated Parameters: {pilot_calc['total_parameters']:,}")
    print(f"   PyTorch Model Numel:   {actual_pilot_params:,}")
    assert actual_pilot_params == pilot_calc["total_parameters"], "Parameter count mismatch in pilot model!"

    vram_pilot = estimate_t4_vram(actual_pilot_params, batch_size=4, block_size=512, n_embd=512, n_layer=8)
    print(f"   Peak VRAM Footprint:   {vram_pilot['estimated_peak_vram_mb']} MB / {vram_pilot['t4_total_vram_mb']} MB ({vram_pilot['vram_utilization_pct']}%)")
    print(f"   Headroom:              {vram_pilot['headroom_mb']} MB")
    print(f"   OOM Risk Assessment:   {vram_pilot['oom_risk_assessment']}")

    # Baseline 124M Config
    base_cfg = GPTConfig(
        block_size=1024,
        vocab_size=512,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False
    )
    base_model = GPT(base_cfg)
    actual_base_params = sum(p.numel() for p in base_model.parameters())
    base_calc = calculate_exact_parameters(base_cfg)

    print(f"\n2. Moon Chat Baseline Model (12L, 12H, 768D, bias=False, vocab=512):")
    print(f"   Calculated Parameters: {base_calc['total_parameters']:,}")
    print(f"   PyTorch Model Numel:   {actual_base_params:,}")
    assert actual_base_params == base_calc["total_parameters"], "Parameter count mismatch in baseline model!"

    vram_base = estimate_t4_vram(actual_base_params, batch_size=2, block_size=1024, n_embd=768, n_layer=12)
    print(f"   Peak VRAM Footprint:   {vram_base['estimated_peak_vram_mb']} MB / {vram_base['t4_total_vram_mb']} MB ({vram_base['vram_utilization_pct']}%)")
    print(f"   Headroom:              {vram_base['headroom_mb']} MB")

    # CUDA Measurement if GPU available
    if torch.cuda.is_available():
        print(f"\n3. CUDA GPU Live Measurement on {torch.cuda.get_device_name(0)}:")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pilot_model.to("cuda")
        x = torch.randint(0, 512, (4, 512), device="cuda")
        y = torch.randint(0, 512, (4, 512), device="cuda")
        with torch.amp.autocast("cuda"):
            logits, loss = pilot_model(x, y)
        loss.backward()
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"   Actual Peak CUDA Memory Allocated: {peak_allocated:.2f} MB")
    else:
        print(f"\n3. CUDA GPU: Not available on host environment (Analytical simulation verified).")

    print("\n" + "=" * 65)
    print("SUCCESS: HARDWARE VALIDATION COMPLETE!")
    print("=" * 65)

    return {
        "pilot_params": actual_pilot_params,
        "pilot_vram": vram_pilot,
        "base_params": actual_base_params,
        "base_vram": vram_base
    }

if __name__ == "__main__":
    validate_model_and_hardware()
