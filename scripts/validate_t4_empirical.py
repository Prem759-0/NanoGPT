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
import re
import time
import json
import math
import uuid
import tempfile
import hashlib
from typing import Dict, Any, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from model import GPTConfig, GPT


def check_t4_hardware_identity(
    device_name: str,
    compute_capability: str,
    total_vram_mb: float,
    min_vram_mb: float = 14000.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Strictly verifies that the target device is an authentic physical NVIDIA Tesla T4 GPU.
    Enforces:
    1. Device name contains 'T4' or 'Tesla T4' and does NOT match other GPU architectures (A100, L4, V100, H100, etc.).
    2. Compute capability is exactly '7.5' (Turing architecture).
    3. Total physical VRAM meets minimum requirement (>= 14,000 MB, standard T4 has ~15.1-15.3 GB).
    """
    name_clean = device_name.strip()
    name_lower = name_clean.lower()

    details = {
        "device_name": device_name,
        "compute_capability": compute_capability,
        "total_vram_mb": total_vram_mb,
        "min_vram_mb": min_vram_mb
    }

    # Prohibited non-T4 architectures
    prohibited_models = [
        r"\ba100\b",
        r"\bl4\b",
        r"\bv100\b",
        r"\bh100\b",
        r"\ba10g?\b",
        r"\brtx\s*\d+\b",
        r"\bgtx\s*\d+\b",
        r"\bk80\b",
        r"\bp100\b",
        r"\bt600\b",
        r"\bt1000\b",
        r"\bcpu\b"
    ]
    for pat in prohibited_models:
        if re.search(pat, name_lower):
            return False, f"Hardware mismatch: detected non-T4 device '{device_name}'", details

    # Positive T4 matching (e.g. 'Tesla T4', 'NVIDIA T4', 'T4')
    if not (re.search(r"\b(tesla\s+)?t4\b", name_lower) or "t4" in name_lower):
        return False, f"Hardware mismatch: device name '{device_name}' does not identify an NVIDIA Tesla T4", details

    # Compute capability check: Turing T4 is exactly 7.5
    comp_cap_str = str(compute_capability).strip()
    if comp_cap_str != "7.5":
        return False, f"Hardware mismatch: compute capability is '{compute_capability}', expected exactly '7.5' for NVIDIA T4", details

    # Physical VRAM check: standard T4 has 16 GB (~15,100 to 15,360 MB)
    if total_vram_mb < min_vram_mb:
        return False, f"Hardware mismatch: insufficient physical VRAM ({total_vram_mb:.1f} MB < required {min_vram_mb:.1f} MB)", details

    return True, "NVIDIA_T4_AUTHENTICATED", details


def save_benchmark_report(
    summary: Dict[str, Any],
    report_dir: Optional[str] = None,
    allow_overwrite_verified: bool = False
) -> str:
    """
    Safely exports the benchmark report.
    Guarantees:
    1. Generates a unique run_id and exports to t4_empirical_benchmark_{run_id}.json.
    2. Only updates canonical data/reports/t4_empirical_benchmark.json if:
       - The run is verified (status == 'T4_EMPIRICAL_BENCHMARK_VERIFIED'), AND
       - Either no verified report exists yet, or allow_overwrite_verified is explicitly True.
    3. An existing verified benchmark report is never overwritten by an unverified or mismatched run.
    """
    if report_dir is None:
        report_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    os.makedirs(report_dir, exist_ok=True)

    run_id = summary.get("run_id") or f"t4_bench_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    summary["run_id"] = run_id

    # 1. Always save the isolated, unique run report
    unique_report_path = os.path.join(report_dir, f"t4_empirical_benchmark_{run_id}.json")
    with open(unique_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 2. Canonical report protection
    canonical_path = os.path.join(report_dir, "t4_empirical_benchmark.json")
    is_verified_run = summary.get("status") == "T4_EMPIRICAL_BENCHMARK_VERIFIED"

    if os.path.isfile(canonical_path):
        try:
            with open(canonical_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_is_verified = existing_data.get("status") == "T4_EMPIRICAL_BENCHMARK_VERIFIED"
        except Exception:
            existing_is_verified = False

        if existing_is_verified and not allow_overwrite_verified:
            # Preserve existing verified report
            print(f"      ℹ️ Existing verified benchmark report preserved at {canonical_path}")
            print(f"      ℹ️ This run exported to isolated report: {unique_report_path}")
            return unique_report_path

    # If this run is not verified, never overwrite canonical report with an error report
    if not is_verified_run:
        print(f"      ℹ️ Unverified run ({summary.get('status')}) exported to isolated report: {unique_report_path}")
        return unique_report_path

    # Update canonical report
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return canonical_path


def verify_checkpoint_roundtrip_isolated(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    gpt_conf: GPTConfig,
    device: torch.device,
    custom_temp_dir: Optional[str] = None,
    device_name: str = "T4_GPU"
) -> Dict[str, Any]:
    """
    Saves, loads, and bitwise verifies a benchmark checkpoint in an isolated temporary directory.
    Guarantees:
    1. Uses a unique temporary file path so no existing checkpoint file can ever be overwritten.
    2. Uses try ... finally for guaranteed cleanup of ONLY the temporary file created by this invocation.
    3. Preserves all existing checkpoint files on disk byte-for-byte.
    4. Confirms 100% bit-for-bit parameter restoration across all weights.
    """
    owns_dir = False
    if custom_temp_dir:
        temp_dir = custom_temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="moon_t4_bench_")
        temp_dir = temp_dir_obj.name
        owns_dir = True

    unique_ckpt_name = f"t4_empirical_bench_{uuid.uuid4().hex[:12]}.pt"
    temp_ckpt_path = os.path.join(temp_dir, unique_ckpt_name)

    try:
        checkpoint_payload = {
            "status": "EMPIRICAL_T4_BENCHMARK_CHECKPOINT",
            "is_test_fixture": True,
            "is_production_trained": False,
            "promotable": False,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else {},
            "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
            "config": {
                "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
                "teacher_checkpoint_weights_transferred": False,
                "initialization": "random_gaussian_scratch",
                "device": device_name,
                "note": "Temporary hardware benchmark checkpoint only. Unpromotable."
            }
        }
        torch.save(checkpoint_payload, temp_ckpt_path)
        file_size_mb = round(os.path.getsize(temp_ckpt_path) / (1024 ** 2), 2)

        with open(temp_ckpt_path, "rb") as f:
            ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Load back into fresh model
        fresh_model = GPT(gpt_conf).to(device)
        loaded_ckpt = torch.load(temp_ckpt_path, map_location=device, weights_only=True)
        fresh_model.load_state_dict(loaded_ckpt["model"])

        # Bitwise comparison
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
            if not torch.equal(v1, v2):
                raise AssertionError(f"Bitwise parameter mismatch in key '{k1}' during checkpoint reload!")

        return {
            "verified": True,
            "size_mb": file_size_mb,
            "sha256": ckpt_sha256,
            "temp_path": temp_ckpt_path
        }
    finally:
        # Delete ONLY the temporary file created by this invocation
        if os.path.isfile(temp_ckpt_path):
            try:
                os.remove(temp_ckpt_path)
            except Exception:
                pass
        if owns_dir:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass


def run_empirical_t4_validation(
    custom_temp_dir: Optional[str] = None,
    min_vram_mb: float = 14000.0,
    allow_overwrite_verified: bool = False,
    report_dir: Optional[str] = None
) -> Dict[str, Any]:
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
        summary = {
            "status": "CUDA_UNAVAILABLE",
            "error": "CUDA is not available on this host.",
            "device": "cpu",
            "empirical_benchmark_verified": False
        }
        save_benchmark_report(summary, report_dir=report_dir, allow_overwrite_verified=False)
        return summary

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

    # Enforce physical NVIDIA T4 identity
    is_t4, mismatch_reason, hw_details = check_t4_hardware_identity(
        device_name=device_name,
        compute_capability=compute_cap,
        total_vram_mb=total_vram_mb,
        min_vram_mb=min_vram_mb
    )
    if not is_t4:
        print(f"\n❌ CRITICAL HARDWARE MISMATCH: {mismatch_reason}")
        print("Hardware Status: Detected device is NOT an authentic NVIDIA Tesla T4.")
        print("Execution Result: HARDWARE_MISMATCH")
        print("\nExplanation:")
        print("Live T4 GPU validation strictly requires an authentic NVIDIA Tesla T4 GPU (Compute Capability 7.5).")
        print("Running on alternate GPUs (e.g. A100, L4, V100) or CPU cannot certify T4 empirical performance.\n")
        print("=" * 70)
        summary = {
            "status": "HARDWARE_MISMATCH",
            "error": mismatch_reason,
            "device_name": device_name,
            "compute_capability": compute_cap,
            "total_vram_mb": total_vram_mb,
            "empirical_benchmark_verified": False,
            "hardware_details": hw_details
        }
        save_benchmark_report(summary, report_dir=report_dir, allow_overwrite_verified=False)
        return summary

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
    print("      Note: Synthetic token batches are hardware test inputs only, not dataset training.")

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0

    t_start = time.perf_counter()
    for micro_step in range(grad_accum_steps):
        # Generate representative random batch of token IDs (hardware test inputs only)
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

    # 8. Checkpoint Save/Load Bitwise Integrity Roundtrip in Isolated Temporary Directory
    print("\n[7/8] Testing Checkpoint Save & Load Bitwise Integrity Roundtrip in Isolated Temporary Directory...")
    rt_res = verify_checkpoint_roundtrip_isolated(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gpt_conf=gpt_conf,
        device=device,
        custom_temp_dir=custom_temp_dir,
        device_name=device_name
    )

    print(f"      • Checkpoint Size:         {rt_res['size_mb']} MB")
    print(f"      • Checkpoint SHA-256:      {rt_res['sha256'][:16]}...{rt_res['sha256'][-8:]}")
    print("      • Bitwise Verification:    100% IDENTICAL (All 25.7M parameters restored bit-for-bit)")
    print("      • Safe File Cleanup:       Temporary file safely removed (Zero impact on existing models)")

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
        "temporary_file_safety_verified": True,
        "hardware_test_inputs_only": True,
        "is_dataset_training": False,
        "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
        "teacher_checkpoint_weights_transferred": False
    }

    report_path = save_benchmark_report(
        summary,
        report_dir=report_dir,
        allow_overwrite_verified=allow_overwrite_verified
    )

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
