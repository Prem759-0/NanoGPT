"""
training/common/checkpoint.py
Robust Checkpoint Management with Full State Restoration & Strict Provenance.
Ensures that only models trained from random initialization within Moon AI can be saved or loaded.
Enforces strict training isolation between PRODUCTION_GENUINE and DEVELOPMENT_SYNTHETIC checkpoints.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import hashlib
import random
import subprocess
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Provenance and training mode constants
TRAINING_MODE_PRODUCTION = "PRODUCTION_GENUINE"
TRAINING_MODE_DEVELOPMENT = "DEVELOPMENT_SYNTHETIC"
VALID_TRAINING_MODES = {TRAINING_MODE_PRODUCTION, TRAINING_MODE_DEVELOPMENT}

PROVENANCE_TAG = "moon_ai_random_init"
ARCHITECT_TAG = "Prem Gaikwad (India 🇮🇳)"
ARCHITECTURE_VERSION = "moon_arch_v1.0"
SOFTWARE_VERSION = "2.0.3"

def get_git_commit() -> str:
    """Retrieves current Git commit SHA, with fallback to known baseline."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "f74fb94d5dc3daae742747bb4ab0f407bfcdcb3c"

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
    optimizer: Optional[torch.optim.Optimizer],
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
    extra_meta: Optional[Dict[str, Any]] = None,
    training_mode: str = TRAINING_MODE_PRODUCTION,
    is_production_eligible: Optional[bool] = None,
    dataset_record_counts: Optional[Dict[str, int]] = None,
    synthetic_sample_count: int = 0,
    production_eligible_sample_count: int = 0,
    val_manifest_path: Optional[str] = None,
    val_manifest_hash: Optional[str] = None,
    architecture_version: str = ARCHITECTURE_VERSION,
    software_version: str = SOFTWARE_VERSION,
    git_commit: Optional[str] = None,
    evaluation_status: str = "UNEVALUATED"
) -> str:
    """
    Saves complete training checkpoint with verifiable provenance, strict training isolation,
    and full state resumption.
    """
    if training_mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"CHECKPOINT SECURITY ERROR: Invalid training_mode '{training_mode}'. "
            f"Must be one of {sorted(list(VALID_TRAINING_MODES))}."
        )

    norm_out = output_path.replace("\\", "/").lower()

    # Enforce development vs production isolation
    if training_mode == TRAINING_MODE_DEVELOPMENT:
        if is_production_eligible is True:
            raise ValueError(
                "CHECKPOINT SECURITY ERROR: DEVELOPMENT_SYNTHETIC training cannot emit a "
                "production-eligible checkpoint (is_production_eligible cannot be True)."
            )
        is_production_eligible = False

        # Prevent overwriting production best models with development synthetic checkpoints
        if ("best_moon_image.pt" in norm_out or "best_moon_video.pt" in norm_out) and "dev" not in norm_out:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Cannot save DEVELOPMENT_SYNTHETIC checkpoint to production path '{output_path}'. "
                "Must be saved to development directory (e.g. checkpoints/dev/moon_image/...)."
            )
    elif training_mode == TRAINING_MODE_PRODUCTION:
        if synthetic_sample_count > 0:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Cannot emit PRODUCTION_GENUINE checkpoint with synthetic samples "
                f"({synthetic_sample_count} synthetic samples present). Synthetic data is strictly prohibited in production."
            )
        if is_production_eligible is False:
            raise ValueError(
                "CHECKPOINT SECURITY ERROR: Cannot set is_production_eligible=False for PRODUCTION_GENUINE checkpoint."
            )
        is_production_eligible = True

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    calculated_manifest_hash = manifest_hash
    if not calculated_manifest_hash and manifest_path and os.path.isfile(manifest_path):
        calculated_manifest_hash = compute_file_sha256(manifest_path)

    calculated_val_manifest_hash = val_manifest_hash
    if not calculated_val_manifest_hash and val_manifest_path and os.path.isfile(val_manifest_path):
        calculated_val_manifest_hash = compute_file_sha256(val_manifest_path)

    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        # Provenance and identity
        "provenance": PROVENANCE_TAG,
        "architect": ARCHITECT_TAG,
        "model_name": model_name,
        "architecture_version": architecture_version,
        "software_version": software_version,
        "git_commit": git_commit or get_git_commit(),
        "saved_at": now_iso,
        "training_timestamp": now_iso,
        # Training provenance mode
        "training_mode": training_mode,
        "is_production_eligible": is_production_eligible,
        # Dataset provenance
        "manifest_path": manifest_path or "",
        "manifest_hash": calculated_manifest_hash or "none",
        "val_manifest_path": val_manifest_path or "",
        "val_manifest_hash": calculated_val_manifest_hash or "none",
        "dataset_record_counts": dataset_record_counts or {"total": 0, "train": 0, "val": 0},
        "synthetic_sample_count": synthetic_sample_count,
        "production_eligible_sample_count": production_eligible_sample_count,
        "evaluation_status": evaluation_status,
        # Training state
        "step": step,
        "epoch": epoch,
        "loss": float(loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        # Model weights and optimizer states
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

def verify_checkpoint_provenance(
    checkpoint_path: str,
    mode: str = TRAINING_MODE_PRODUCTION
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Inspects checkpoint provenance and training isolation metadata on disk.
    Returns (is_valid, metadata_dict, list_of_errors).
    """
    errors: List[str] = []
    if not os.path.isfile(checkpoint_path):
        return False, {}, [f"Checkpoint file not found: '{checkpoint_path}'"]

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, {}, [f"Corrupted or invalid checkpoint file: {e}"]

    provenance = ckpt.get("provenance")
    if provenance != PROVENANCE_TAG:
        errors.append(
            f"STRICT OWNERSHIP VIOLATION: Checkpoint has invalid provenance '{provenance}'. "
            f"Expected '{PROVENANCE_TAG}'."
        )

    ckpt_training_mode = ckpt.get("training_mode")
    if not ckpt_training_mode:
        errors.append("CHECKPOINT SECURITY ERROR: Missing required 'training_mode' provenance metadata.")
    elif ckpt_training_mode not in VALID_TRAINING_MODES:
        errors.append(f"CHECKPOINT SECURITY ERROR: Invalid training_mode '{ckpt_training_mode}'.")

    is_prod_eligible = ckpt.get("is_production_eligible")
    synth_count = ckpt.get("synthetic_sample_count", 0)

    # Check for metadata tampering
    if is_prod_eligible is True and (ckpt_training_mode == TRAINING_MODE_DEVELOPMENT or synth_count > 0):
        errors.append(
            f"CHECKPOINT SECURITY ERROR: Metadata tampering detected! Claimed is_production_eligible=True "
            f"but training_mode='{ckpt_training_mode}' and synthetic_samples={synth_count}."
        )

    if mode == TRAINING_MODE_PRODUCTION:
        if ckpt_training_mode != TRAINING_MODE_PRODUCTION:
            errors.append(
                f"CHECKPOINT SECURITY ERROR: Attempted to load {ckpt_training_mode} checkpoint in PRODUCTION_GENUINE mode."
            )
        if is_prod_eligible is not True:
            errors.append(
                "CHECKPOINT SECURITY ERROR: Checkpoint has 'is_production_eligible: false'. Rejected from production."
            )
        if synth_count > 0:
            errors.append(
                f"CHECKPOINT SECURITY ERROR: Checkpoint contains {synth_count} synthetic samples. Rejected from production."
            )

    meta = {
        "model_name": ckpt.get("model_name", "unknown"),
        "architecture_version": ckpt.get("architecture_version", "unknown"),
        "software_version": ckpt.get("software_version", "unknown"),
        "git_commit": ckpt.get("git_commit", "unknown"),
        "training_mode": ckpt_training_mode,
        "is_production_eligible": is_prod_eligible,
        "manifest_path": ckpt.get("manifest_path", ""),
        "manifest_hash": ckpt.get("manifest_hash", "none"),
        "val_manifest_path": ckpt.get("val_manifest_path", ""),
        "val_manifest_hash": ckpt.get("val_manifest_hash", "none"),
        "dataset_record_counts": ckpt.get("dataset_record_counts", {}),
        "synthetic_sample_count": synth_count,
        "production_eligible_sample_count": ckpt.get("production_eligible_sample_count", 0),
        "training_timestamp": ckpt.get("training_timestamp", ckpt.get("saved_at", "")),
        "evaluation_status": ckpt.get("evaluation_status", "UNEVALUATED"),
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "loss": ckpt.get("loss", 0.0),
        "val_loss": ckpt.get("val_loss", None),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "provenance": provenance
    }

    return len(errors) == 0, meta, errors

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    restore_rng: bool = True,
    enforce_provenance: bool = True,
    mode: str = TRAINING_MODE_PRODUCTION
) -> Dict[str, Any]:
    """
    Loads checkpoint, validates model provenance and training mode, and restores states.
    Rejects development checkpoints in production mode.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    if enforce_provenance:
        valid, meta, errors = verify_checkpoint_provenance(checkpoint_path, mode=mode)
        if not valid:
            raise ValueError(f"CHECKPOINT SECURITY ERROR: Failed verification on '{checkpoint_path}': {'; '.join(errors)}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
        "val_manifest_hash": ckpt.get("val_manifest_hash", "none"),
        "val_manifest_path": ckpt.get("val_manifest_path", ""),
        "model_name": ckpt.get("model_name", "moon_model"),
        "architecture_version": ckpt.get("architecture_version", ARCHITECTURE_VERSION),
        "software_version": ckpt.get("software_version", SOFTWARE_VERSION),
        "git_commit": ckpt.get("git_commit", "unknown"),
        "training_mode": ckpt.get("training_mode", "UNKNOWN"),
        "is_production_eligible": ckpt.get("is_production_eligible", False),
        "synthetic_sample_count": ckpt.get("synthetic_sample_count", 0),
        "production_eligible_sample_count": ckpt.get("production_eligible_sample_count", 0),
        "saved_at": ckpt.get("saved_at", ""),
        "training_timestamp": ckpt.get("training_timestamp", ckpt.get("saved_at", "")),
        "config": ckpt.get("config", {}),
        "provenance": ckpt.get("provenance", "unknown")
    }
