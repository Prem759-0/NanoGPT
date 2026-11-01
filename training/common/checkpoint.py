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
from typing import Dict, Any, Optional, Tuple, List, Set
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Provenance and training mode constants
TRAINING_MODE_PRODUCTION = "PRODUCTION_GENUINE"
TRAINING_MODE_DEVELOPMENT = "DEVELOPMENT_SYNTHETIC"
VALID_TRAINING_MODES = {TRAINING_MODE_PRODUCTION, TRAINING_MODE_DEVELOPMENT}

PROVENANCE_TAG = "moon_ai_random_init"
ARCHITECT_TAG = "Prem Gaikwad (India 🇮🇳)"
ARCHITECTURE_VERSION = "moon_arch_v1.0"
SOFTWARE_VERSION = "2.0.5"

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

def count_manifest_records(filepath: str) -> int:
    """Counts non-empty, non-comment records in a JSONL manifest."""
    if not os.path.isfile(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str and not line_str.startswith("#"):
                count += 1
    return count

def deep_verify_production_manifest(
    manifest_path: str,
    expected_split: str,
    base_dir: str = PROJECT_ROOT,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, int, List[str]]:
    """
    Parses and cryptographically audits every single record in a production manifest:
    1. Parse JSON line-by-line with exact line number error reporting.
    2. Check split matches expected_split ('train' or 'val').
    3. Reject any synthetic record (is_synthetic must be strictly False).
    4. Call validate_provenance_record:
       - Checks safe path and symlink containment (rejects directory traversal and escaping symlinks).
       - Recomputes actual file byte SHA-256 on disk.
       - Cryptographically validates Ed25519 attestation.
       - Validates source URL and license evidence.
    5. Returns (is_valid, production_eligible_count, list_of_errors).
    """
    resolved_path = os.path.normpath(os.path.join(base_dir, manifest_path)) if not os.path.isabs(manifest_path) else manifest_path
    if not os.path.isfile(resolved_path):
        return False, 0, [f"Manifest file not found on disk: '{manifest_path}'"]

    import json
    from datasets.tools.provenance_validator import validate_provenance_record

    errors: List[str] = []
    eligible_count = 0
    seen_ids: Set[str] = set()

    with open(resolved_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                errors.append(f"JSON decode error in '{manifest_path}' line {line_no}: {e}")
                return False, 0, errors

            if not isinstance(record, dict):
                errors.append(f"Record in '{manifest_path}' line {line_no} is not a JSON object: {type(record).__name__}.")
                return False, 0, errors

            rec_id = str(record.get("id", "")).strip()
            if not rec_id:
                errors.append(f"Record in '{manifest_path}' line {line_no} is missing required 'id'.")
                return False, 0, errors

            if rec_id in seen_ids:
                errors.append(f"Duplicate record id '{rec_id}' detected in '{manifest_path}' line {line_no}.")
                return False, 0, errors
            seen_ids.add(rec_id)

            rec_split = str(record.get("split", "")).strip().lower()
            if rec_split != expected_split.lower():
                errors.append(
                    f"Split mismatch in '{manifest_path}' line {line_no} for record '{record.get('id')}': "
                    f"record has split='{rec_split}', expected '{expected_split}'."
                )
                return False, 0, errors

            is_synth = record.get("is_synthetic")
            if is_synth is not False:
                errors.append(
                    f"Synthetic record contamination in '{manifest_path}' line {line_no}: "
                    f"record '{record.get('id')}' has is_synthetic={is_synth}. Production manifests strictly require is_synthetic: false."
                )
                return False, 0, errors

            modality = "video" if ("video_path" in record or str(record.get("media_path", "")).endswith((".mp4", ".webm"))) else "image"

            audit = validate_provenance_record(
                record=record,
                modality=modality,
                base_dir=base_dir,
                allow_synthetic=False,
                verify_file=True,
                verifiers_path=verifiers_path
            )

            if not audit["training_eligible"]:
                errors.append(
                    f"Record '{record.get('id')}' in '{manifest_path}' line {line_no} failed production verification "
                    f"({audit['status']}): {'; '.join(audit['errors'])}"
                )
                return False, 0, errors

            eligible_count += 1

    if eligible_count == 0:
        errors.append(f"Production manifest '{manifest_path}' contains 0 production-eligible records.")
        return False, 0, errors

    return len(errors) == 0, eligible_count, errors

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
    evaluation_status: str = "UNEVALUATED",
    verifiers_path: Optional[str] = None,
    base_dir: str = PROJECT_ROOT
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

        calculated_manifest_hash = manifest_hash
        if not calculated_manifest_hash and manifest_path and os.path.isfile(manifest_path):
            calculated_manifest_hash = compute_file_sha256(manifest_path)

        calculated_val_manifest_hash = val_manifest_hash
        if not calculated_val_manifest_hash and val_manifest_path and os.path.isfile(val_manifest_path):
            calculated_val_manifest_hash = compute_file_sha256(val_manifest_path)

    elif training_mode == TRAINING_MODE_PRODUCTION:
        # Task 4 & 5: Fail-closed verification on production manifests and data integrity
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

        # Reject saving production checkpoints in development directories
        if "/dev/" in norm_out or "\\dev\\" in norm_out or norm_out.startswith("dev/") or norm_out.startswith("checkpoints/dev/"):
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Cannot save PRODUCTION_GENUINE checkpoint to development directory '{output_path}'."
            )

        if not manifest_path or not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ValueError(
                "CHECKPOINT SECURITY ERROR: manifest_path is strictly required for PRODUCTION_GENUINE checkpoints."
            )
        resolved_manifest = os.path.normpath(os.path.join(base_dir, manifest_path)) if not os.path.isabs(manifest_path) else manifest_path
        if not os.path.isfile(resolved_manifest):
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production training manifest does not exist on disk: '{manifest_path}'."
            )

        if not val_manifest_path or not isinstance(val_manifest_path, str) or not val_manifest_path.strip():
            raise ValueError(
                "CHECKPOINT SECURITY ERROR: val_manifest_path is strictly required for PRODUCTION_GENUINE checkpoints."
            )
        resolved_val_manifest = os.path.normpath(os.path.join(base_dir, val_manifest_path)) if not os.path.isabs(val_manifest_path) else val_manifest_path
        if not os.path.isfile(resolved_val_manifest):
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production validation manifest does not exist on disk: '{val_manifest_path}'."
            )

        # Recompute SHA-256 of manifest files directly from disk bytes
        actual_manifest_sha = compute_file_sha256(resolved_manifest)
        if manifest_hash and manifest_hash != "none" and manifest_hash.strip().lower() != actual_manifest_sha.lower():
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Supplied manifest_hash '{manifest_hash}' does not match actual disk hash '{actual_manifest_sha}'."
            )
        calculated_manifest_hash = actual_manifest_sha

        actual_val_sha = compute_file_sha256(resolved_val_manifest)
        if val_manifest_hash and val_manifest_hash != "none" and val_manifest_hash.strip().lower() != actual_val_sha.lower():
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Supplied val_manifest_hash '{val_manifest_hash}' does not match actual disk hash '{actual_val_sha}'."
            )
        calculated_val_manifest_hash = actual_val_sha

        # Verify manifests are not empty
        train_rec_count = count_manifest_records(resolved_manifest)
        if train_rec_count == 0:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production manifest '{manifest_path}' contains 0 records."
            )

        val_rec_count = count_manifest_records(resolved_val_manifest)
        if val_rec_count == 0:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production validation manifest '{val_manifest_path}' contains 0 records."
            )

        if production_eligible_sample_count <= 0:
            raise ValueError(
                "CHECKPOINT SECURITY ERROR: production_eligible_sample_count must be > 0 for PRODUCTION_GENUINE checkpoint."
            )

        # Deep verify train manifest
        train_valid, train_count, train_errors = deep_verify_production_manifest(
            resolved_manifest, expected_split="train", base_dir=base_dir, verifiers_path=verifiers_path
        )
        if not train_valid:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production training manifest failed deep verification: {'; '.join(train_errors)}"
            )

        # Deep verify val manifest
        val_valid, val_count, val_errors = deep_verify_production_manifest(
            resolved_val_manifest, expected_split="val", base_dir=base_dir, verifiers_path=verifiers_path
        )
        if not val_valid:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Production validation manifest failed deep verification: {'; '.join(val_errors)}"
            )

        if (train_count + val_count) != production_eligible_sample_count:
            raise ValueError(
                f"CHECKPOINT SECURITY ERROR: Verified production sample count mismatch! "
                f"Manifests contain {train_count + val_count} verified samples ({train_count} train, {val_count} val), "
                f"but production_eligible_sample_count was declared as {production_eligible_sample_count}."
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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
    mode: str = TRAINING_MODE_PRODUCTION,
    recompute_manifests: bool = True,
    base_dir: str = PROJECT_ROOT,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Inspects checkpoint provenance and training isolation metadata on disk.
    In PRODUCTION_GENUINE mode, independently verifies and recomputes actual manifest
    hashes on disk to prevent manifest drift and silent dataset tampering.
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
    prod_sample_count = ckpt.get("production_eligible_sample_count", 0)

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
        if prod_sample_count <= 0:
            errors.append(
                "CHECKPOINT SECURITY ERROR: Checkpoint has 0 production_eligible_sample_count. Rejected from production."
            )

        if recompute_manifests:
            m_path = ckpt.get("manifest_path")
            m_hash = ckpt.get("manifest_hash", "")
            v_path = ckpt.get("val_manifest_path")
            v_hash = ckpt.get("val_manifest_hash", "")

            m_verified = False
            v_verified = False
            m_count = 0
            v_count = 0

            if not m_path:
                errors.append("CHECKPOINT SECURITY ERROR: Production checkpoint is missing 'manifest_path'.")
            else:
                res_m = os.path.normpath(os.path.join(base_dir, m_path)) if not os.path.isabs(m_path) else m_path
                if not os.path.isfile(res_m):
                    errors.append(f"CHECKPOINT INTEGRITY ERROR: Training manifest not found on disk: '{m_path}'")
                else:
                    actual_m_sha = compute_file_sha256(res_m)
                    if m_hash.lower() != actual_m_sha.lower():
                        errors.append(
                            f"CHECKPOINT INTEGRITY ERROR: Training manifest hash drift detected! Checkpoint expected manifest_hash '{m_hash}', "
                            f"but actual disk file '{m_path}' has hash '{actual_m_sha}'."
                        )
                    m_valid, m_count, m_errors = deep_verify_production_manifest(
                        res_m, expected_split="train", base_dir=base_dir, verifiers_path=verifiers_path
                    )
                    if not m_valid:
                        errors.extend([f"Training manifest verification failed: {err}" for err in m_errors])
                    else:
                        m_verified = True

            if not v_path:
                errors.append("CHECKPOINT SECURITY ERROR: Production checkpoint is missing 'val_manifest_path'.")
            else:
                res_v = os.path.normpath(os.path.join(base_dir, v_path)) if not os.path.isabs(v_path) else v_path
                if not os.path.isfile(res_v):
                    errors.append(f"CHECKPOINT INTEGRITY ERROR: Validation manifest not found on disk: '{v_path}'")
                else:
                    actual_v_sha = compute_file_sha256(res_v)
                    if v_hash.lower() != actual_v_sha.lower():
                        errors.append(
                            f"CHECKPOINT INTEGRITY ERROR: Validation manifest hash drift detected! Checkpoint expected val_manifest_hash '{v_hash}', "
                            f"but actual disk file '{v_path}' has hash '{actual_v_sha}'."
                        )
                    v_valid, v_count, v_errors = deep_verify_production_manifest(
                        res_v, expected_split="val", base_dir=base_dir, verifiers_path=verifiers_path
                    )
                    if not v_valid:
                        errors.extend([f"Validation manifest verification failed: {err}" for err in v_errors])
                    else:
                        v_verified = True

            if m_verified and v_verified:
                if (m_count + v_count) != prod_sample_count:
                    errors.append(
                        f"CHECKPOINT INTEGRITY ERROR: Verified sample count mismatch! "
                        f"Manifests contain {m_count + v_count} verified samples ({m_count} train, {v_count} val), "
                        f"but checkpoint recorded production_eligible_sample_count={prod_sample_count}."
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
        "production_eligible_sample_count": prod_sample_count,
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
    mode: str = TRAINING_MODE_PRODUCTION,
    recompute_manifests: Optional[bool] = None,
    verifiers_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads checkpoint, validates model provenance and training mode, and restores states.
    Rejects development checkpoints in production mode.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    if enforce_provenance:
        do_recompute = (mode == TRAINING_MODE_PRODUCTION) if recompute_manifests is None else recompute_manifests
        valid, meta, errors = verify_checkpoint_provenance(
            checkpoint_path,
            mode=mode,
            recompute_manifests=do_recompute,
            verifiers_path=verifiers_path
        )
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
