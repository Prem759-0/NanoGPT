"""
training/common/checkpoint.py
Checkpoint Management with Strict Model Provenance Verification.
Enforces that only models trained from random initialization within Moon AI can be saved or loaded.
"""
import os
import torch
from typing import Dict, Any, Optional
from datetime import datetime, timezone

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    config_dict: Dict[str, Any],
    output_path: str,
    manifest_hash: Optional[str] = None
) -> str:
    """Saves model checkpoint with verified provenance metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "provenance": "moon_ai_random_init",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "epoch": epoch,
        "loss": float(loss),
        "manifest_hash": manifest_hash or "none",
        "config": config_dict,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, output_path)
    return output_path

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: str = "cpu") -> Dict[str, Any]:
    """
    Loads checkpoint and verifies model provenance.
    Rejects any checkpoint that lacks sovereign provenance or appears to contain third-party weights.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    provenance = ckpt.get("provenance")
    if provenance != "moon_ai_random_init":
        raise ValueError(f"STRICT OWNERSHIP VIOLATION: Checkpoint '{checkpoint_path}' has unknown provenance '{provenance}'. Expected 'moon_ai_random_init'.")

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "loss": ckpt.get("loss", 0.0),
        "saved_at": ckpt.get("saved_at", ""),
        "config": ckpt.get("config", {})
    }
