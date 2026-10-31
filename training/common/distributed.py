"""
training/common/distributed.py
Hardware and Precision Detection for Moon AI Training.
Supports CUDA (NVIDIA T4, A10G, A100, RTX), Apple MPS, and CPU with mixed precision (fp16 / bf16 / fp32).
"""
import os
import torch
from typing import Tuple, Dict, Any
from contextlib import nullcontext

def setup_device(requested_device: str = "auto", precision: str = "auto") -> Tuple[torch.device, torch.dtype, bool]:
    """
    Detects available hardware accelerator and selects optimal compute precision.
    Returns: (device, dtype, use_amp)
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # Resolve precision
    use_amp = False
    if precision == "fp16":
        dtype = torch.float16
        use_amp = device.type == "cuda"
    elif precision == "bf16":
        dtype = torch.bfloat16
        use_amp = (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "cpu"
    elif precision == "fp32":
        dtype = torch.float32
        use_amp = False
    else:
        # Auto mode
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            use_amp = True
        else:
            dtype = torch.float32
            use_amp = False

    return device, dtype, use_amp

def get_amp_context(device: torch.device, dtype: torch.dtype, enabled: bool = True):
    """
    Returns an appropriate autocast context manager for mixed precision training.
    """
    if not enabled:
        return nullcontext()
    device_type = device.type if device.type in ["cuda", "cpu"] else "cpu"
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    except Exception:
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
        return nullcontext()

def create_grad_scaler(device: torch.device, dtype: torch.dtype, enabled: bool = True):
    """
    Creates a GradScaler for FP16 training to prevent underflow.
    GradScaler is typically only needed for float16 on CUDA.
    """
    is_fp16_cuda = (device.type == "cuda" and dtype == torch.float16 and enabled)
    try:
        return torch.amp.GradScaler(device=device.type, enabled=is_fp16_cuda)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=is_fp16_cuda)

def get_vram_usage_mb() -> float:
    """Returns allocated GPU VRAM memory in megabytes, or 0.0 if running on CPU."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0

def get_hardware_info() -> Dict[str, Any]:
    """Returns comprehensive metadata regarding host hardware and accelerators."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cpu_count": os.cpu_count() or 1,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info
