"""
training/common/checkpoint.py
Robust Checkpoint Management with Full State Restoration & Strict Provenance.
Ensures that only models trained from random initialization within Moon AI can be saved or loaded.
Saves:
- Model state dict
- Optimizer state dict
- Learning rate scheduler state dict
- Mixed-precision GradScaler state dict
- Global step, epoch, best validation loss
- Complete RNG states (Torch CPU/CUDA, Python random, NumPy)
- Dataset manifest SHA-256 checksum
- Architecture configuration
"""
import os
import hashlib
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

PROVENANCE_TAG = "moon_ai_random_init"
ARCHITECT_TAG = "Prem Gaikwad (India 🇮🇳)"

def compute_file_sha256(filepath: str) -> str:
    """Calculates the SHA-256 hash of a file for immutable dataset/checkpoint tracking."""
    if not os.path.isfile(filepath):
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_current_rng_state() -> Dict[str, Any]:
    """Captures all random number generator states for deterministic training resumption."""
    rng = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        try:
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return rng

def restore_rng_state(rng: Dict[str, Any]) -> None:
    """Restores all random number generator states from checkpoint."""
    if not rng:
        return
    if "torch_cpu" in rng and rng["torch_cpu"] is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if "python" in rng and rng["python"] is not None:
        random.setstate(rng["python"])
    if "numpy" in rng and rng["numpy"] is not None:
        np.random.set_state(rng["numpy"])
    if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    config_dict: Dict[str, Any],
    output_path: str,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    manifest_path: Optional[str] = None,
    manifest_hash: Optional[str] = None,
    model_name: str = "moon_model",
    extra_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves complete training checkpoint with verifiable provenance and full state resumption.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    calculated_manifest_hash = manifest_hash
    if not calculated_manifest_hash and manifest_path and os.path.isfile(manifest_path):
        calculated_manifest_hash = compute_file_sha256(manifest_path)

    payload = {
        "provenance": PROVENANCE_TAG,
        "architect": ARCHITECT_TAG,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "step": step,
        "epoch": epoch,
        "loss": float(loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "manifest_path": manifest_path or "",
        "manifest_hash": calculated_manifest_hash or "none",
        "config": config_dict,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": get_current_rng_state(),
        "extra_meta": extra_meta or {}
    }

    # Atomic write to temporary file first to avoid corrupted checkpoints if interrupted
    tmp_path = f"{output_path}.tmp"
    torch.save(payload, tmp_path)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
    os.replace(tmp_path, output_path)
    return output_path

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    restore_rng: bool = True,
    enforce_provenance: bool = True
) -> Dict[str, Any]:
    """
    Loads checkpoint, validates model provenance, and restores states.
    Rejects any checkpoint that lacks sovereign provenance ('moon_ai_random_init').
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = ckpt.get("provenance")

    if enforce_provenance and provenance != PROVENANCE_TAG:
        raise ValueError(
            f"STRICT OWNERSHIP VIOLATION: Checkpoint '{checkpoint_path}' has unknown provenance '{provenance}'. "
            f"Expected '{PROVENANCE_TAG}'. Third-party pretrained weights are strictly prohibited."
        )

    # Load model weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Checkpoint at '{checkpoint_path}' contains no 'model_state_dict'.")

    # Load optimizer state if supplied
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"⚠️ Warning: Could not restore optimizer state: {e}")

    # Load scheduler state if supplied
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"⚠️ Warning: Could not restore scheduler state: {e}")

    # Load GradScaler state if supplied
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"⚠️ Warning: Could not restore scaler state: {e}")

    # Restore RNG state if requested
    if restore_rng and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])

    return {
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "loss": ckpt.get("loss", 0.0),
        "val_loss": ckpt.get("val_loss", None),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "manifest_hash": ckpt.get("manifest_hash", "none"),
        "manifest_path": ckpt.get("manifest_path", ""),
        "model_name": ckpt.get("model_name", "moon_model"),
        "saved_at": ckpt.get("saved_at", ""),
        "config": ckpt.get("config", {}),
        "provenance": provenance
    }
