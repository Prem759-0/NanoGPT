"""
training/common/distributed.py
Hardware and Precision Detection for Moon AI Training.
Supports CUDA, Apple MPS, and CPU with mixed precision (fp16 / bf16 / fp32).
"""
import torch
from typing import Tuple

def setup_device(requested_device: str = "auto") -> Tuple[torch.device, torch.dtype]:
    """Detects available hardware accelerator and selects optimal compute precision."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Prefer bfloat16 on Ampere/Ada/Hopper architectures, otherwise float16
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32
    else:
        device = torch.device(requested_device)
        dtype = torch.float32

    return device, dtype

def get_vram_usage_mb() -> float:
    """Returns allocated GPU VRAM memory in megabytes, or 0.0 if running on CPU."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0
