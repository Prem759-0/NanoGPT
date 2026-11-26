"""
training/chat/config_validator.py
Phase 2.7.7 Validation & Readiness Engine for Moon Chat v0.1 Training Configs.

Single source of truth imported by `training/chat/train.py`, the reproducible
Google Colab T4 notebook, and CI verification tests.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

import yaml

from core.policy import ACTIVE_OWNERSHIP_POLICY, AUTHORIZED_BY_PREM
from datasets.chat.truth_audit import (
    PRODUCTION_PILOT_TARGETS,
    verify_production_manifest_group,
)
from model import GPTConfig
from training.chat.validate_hardware import calculate_exact_parameters, estimate_t4_vram

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DISALLOWED_PRODUCTION_MANIFEST_MARKERS = (
    "scaled_chat_",
    "expanded_chat_",
    "synthetic_chat_dev",
    "teacher_",
    "quarantined_",
)

FORBIDDEN_PRODUCTION_CHECKPOINT_NAMES = {
    "moon_chat_model.pt",
    "moon_chat_scaled_v1.pt",
    "moon_chat_scaled_best.pt",
    "moon_chat_pilot_v0.pt",
    "moon_chat_pilot_best.pt",
    "moon_chat_phase_2_4_t4_pilot_v0.pt",
    "moon_chat_phase_2_4_t4_pilot_best.pt",
}

MIN_PRODUCTION_MAX_ITERS = 300
MIN_NON_SMOKE_MAX_ITERS = 50


def compute_file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Training config at '{config_path}' must parse to a mapping.")
    return data


def _resolve_repo_path(path: str, project_root: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))


def validate_training_config(
    cfg: Dict[str, Any],
    require_production_release: Optional[bool] = None,
    verify_dataset_truth: bool = False,
    project_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    """
    Validates Moon Chat training configuration for policy compliance, architecture integrity,
    learning-rate schedule validity, step-budget sanity, split isolation, checkpoint isolation,
    and optional production dataset truth verification.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    is_prod_release = bool(cfg.get("is_production_release", False))
    if require_production_release is True:
        is_prod_release = True

    policy = cfg.get("ownership_policy")
    if policy != "STRICT_SOVEREIGN_RANDOM_INIT_ONLY":
        blockers.append(
            f"ownership_policy must be 'STRICT_SOVEREIGN_RANDOM_INIT_ONLY', got '{policy}'."
        )

    teacher_cfg = cfg.get("teacher_distillation") or {}
    if not isinstance(teacher_cfg, dict):
        blockers.append("teacher_distillation must be a mapping.")
    else:
        if teacher_cfg.get("enabled", False) is not False:
            blockers.append("teacher_distillation.enabled must be false under strict sovereign policy.")
        if teacher_cfg.get("allow_teacher_demonstrations", False) is not False:
            blockers.append("teacher_distillation.allow_teacher_demonstrations must be false.")
        if teacher_cfg.get("prohibit_teacher_checkpoint_weights", True) is not True:
            blockers.append("teacher_distillation.prohibit_teacher_checkpoint_weights must be true.")
        if teacher_cfg.get("target_teacher") is not None or teacher_cfg.get("teacher_provider") is not None:
            blockers.append("teacher_distillation target_teacher/teacher_provider must be null.")

    model_cfg = cfg.get("model") or {}
    init_from = str(model_cfg.get("init_from", "scratch")).strip().lower()
    if is_prod_release and "init_from" not in model_cfg:
        blockers.append("Production release config must explicitly set model.init_from='scratch'.")
    elif init_from != "scratch":
        blockers.append(f"model.init_from must be 'scratch', got '{init_from}'.")
    if model_cfg.get("pretrained_weights"):
        blockers.append("model.pretrained_weights must not be set for sovereign scratch initialization.")

    n_layer = int(model_cfg.get("n_layer", 0) or 0)
    n_head = int(model_cfg.get("n_head", 0) or 0)
    n_embd = int(model_cfg.get("n_embd", 0) or 0)
    block_size = int(model_cfg.get("block_size", 0) or 0)
    vocab_size = int(model_cfg.get("vocab_size", 0) or 0)

    if n_layer <= 0 or n_head <= 0 or n_embd <= 0 or block_size <= 0 or vocab_size <= 0:
        blockers.append("All model architecture dimensions must be positive integers.")
    elif n_embd % n_head != 0:
        blockers.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")

    # Support both canonical `paths` section (used by train.py) and `data` fallback
    paths_cfg = cfg.get("paths") or cfg.get("data") or {}
    train_manifest = str(paths_cfg.get("train_manifest", "")).strip()
    val_manifest = str(paths_cfg.get("val_manifest", "")).strip()
    test_manifest = str(paths_cfg.get("test_manifest", "")).strip()
    tokenizer_path = str(paths_cfg.get("tokenizer_path", "")).strip()
    ckpt_name = str(paths_cfg.get("checkpoint_name", "")).strip()
    best_ckpt_name = str(paths_cfg.get("best_checkpoint_name", "")).strip()

    if not train_manifest or not val_manifest:
        blockers.append("Both train_manifest and val_manifest are required.")
    norm_paths = [
        p.replace("\\", "/").lower()
        for p in (train_manifest, val_manifest, test_manifest)
        if p
    ]
    if len(norm_paths) != len(set(norm_paths)):
        blockers.append("Split collision detected: train/val/test manifests must point to distinct files.")

    if is_prod_release:
        if not test_manifest:
            blockers.append("Production release config must specify a blind test_manifest.")
        if not paths_cfg.get("production_genuine_only", False):
            blockers.append("Production release config must set production_genuine_only=true.")
        for spath in (train_manifest, val_manifest, test_manifest):
            low = spath.replace("\\", "/").lower()
            for marker in DISALLOWED_PRODUCTION_MANIFEST_MARKERS:
                if marker in low:
                    blockers.append(
                        f"Production release config cannot point to quarantined/template/dev manifest '{spath}'."
                    )
                    break
        if ckpt_name in FORBIDDEN_PRODUCTION_CHECKPOINT_NAMES or best_ckpt_name in FORBIDDEN_PRODUCTION_CHECKPOINT_NAMES:
            blockers.append(
                f"Production release checkpoint filenames ({ckpt_name}, {best_ckpt_name}) collide with protected legacy checkpoints."
            )
        if ckpt_name and best_ckpt_name and ckpt_name == best_ckpt_name:
            blockers.append("checkpoint_name and best_checkpoint_name must be distinct.")

    train_cfg = cfg.get("training") or {}
    is_smoke_test = bool(train_cfg.get("is_smoke_test", False))
    max_iters = int(train_cfg.get("max_iters", 0) or 0)
    warmup_iters = int(train_cfg.get("warmup_iters", 0) or 0)
    lr_decay_iters = int(train_cfg.get("lr_decay_iters", max_iters) or max_iters)
    eval_interval = int(train_cfg.get("eval_interval", 0) or 0)
    save_interval = int(train_cfg.get("save_interval", 0) or 0)
    batch_size = int(train_cfg.get("batch_size", 0) or 0)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 0) or 0)
    lr = float(train_cfg.get("learning_rate", 0.0) or 0.0)
    min_lr = float(train_cfg.get("min_lr", 0.0) or 0.0)

    if batch_size <= 0 or grad_accum <= 0:
        blockers.append("training.batch_size and training.gradient_accumulation_steps must be >= 1.")
    if lr <= 0 or min_lr < 0 or min_lr > lr:
        blockers.append(f"Invalid learning rate range: learning_rate={lr}, min_lr={min_lr}.")
    if max_iters <= 0:
        blockers.append(f"training.max_iters must be positive, got {max_iters}.")
    if warmup_iters < 0 or warmup_iters >= max_iters:
        blockers.append(
            f"training.warmup_iters ({warmup_iters}) must be >= 0 and strictly < max_iters ({max_iters})."
        )
    if lr_decay_iters < max_iters:
        warnings.append(f"lr_decay_iters ({lr_decay_iters}) < max_iters ({max_iters}); tail runs at min_lr.")

    if is_prod_release:
        if is_smoke_test:
            blockers.append("Production release config cannot enable training.is_smoke_test=true.")
        if max_iters < MIN_PRODUCTION_MAX_ITERS:
            blockers.append(
                f"Production release max_iters ({max_iters}) < minimum meaningful training budget ({MIN_PRODUCTION_MAX_ITERS})."
            )
        if eval_interval <= 0 or eval_interval >= max_iters:
            blockers.append(
                f"Production release eval_interval ({eval_interval}) must be > 0 and < max_iters ({max_iters})."
            )
        if save_interval <= 0:
            blockers.append("Production release save_interval must be positive.")
    elif not is_smoke_test and max_iters < MIN_NON_SMOKE_MAX_ITERS:
        warnings.append(
            f"Non-smoke training config has low max_iters={max_iters} (< {MIN_NON_SMOKE_MAX_ITERS}); treat as pilot/smoke run only."
        )

    param_summary = None
    vram_summary = None
    if n_layer > 0 and n_head > 0 and n_embd > 0 and block_size > 0 and vocab_size > 0 and batch_size > 0 and n_embd % n_head == 0:
        gpt_cfg = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            bias=bool(model_cfg.get("bias", False)),
        )
        param_summary = calculate_exact_parameters(gpt_cfg)
        mixed_prec = str(train_cfg.get("mixed_precision", train_cfg.get("dtype", "fp16"))).lower()
        if "16" in mixed_prec:
            mixed_prec = "fp16"
        vram_summary = estimate_t4_vram(
            exact_params=param_summary["total_parameters"],
            batch_size=batch_size,
            block_size=block_size,
            n_embd=n_embd,
            n_layer=n_layer,
            mixed_precision=mixed_prec,
        )
        if vram_summary["estimated_peak_vram_mb"] >= 14000.0:
            blockers.append(
                f"Estimated T4 peak VRAM ({vram_summary['estimated_peak_vram_mb']} MB) exceeds safe T4 margin."
            )

    config_blockers = list(blockers)
    dataset_blockers: List[str] = []
    dataset_gate: Optional[Dict[str, Any]] = None
    if verify_dataset_truth:
        abs_tr = _resolve_repo_path(train_manifest, project_root)
        abs_va = _resolve_repo_path(val_manifest, project_root)
        abs_te = _resolve_repo_path(test_manifest, project_root) if test_manifest else None
        abs_tok = _resolve_repo_path(tokenizer_path, project_root) if tokenizer_path else ""
        dataset_gate = verify_production_manifest_group(
            train_path=abs_tr,
            val_path=abs_va,
            test_path=abs_te,
            tokenizer_path=abs_tok or os.path.join(project_root, "models", "chat", "tokenizer.json"),
            require_scale=True,
            min_total_records=PRODUCTION_PILOT_TARGETS["min_total_records"],
            min_train_records=PRODUCTION_PILOT_TARGETS["min_train_records"],
            min_val_records=PRODUCTION_PILOT_TARGETS["min_val_records"],
            min_test_records=PRODUCTION_PILOT_TARGETS["min_test_records"],
            min_supervised_tokens=PRODUCTION_PILOT_TARGETS["min_supervised_tokens"],
        )
        if not dataset_gate["passed"]:
            dataset_blockers.extend(dataset_gate["blockers"])

    all_blockers = config_blockers + dataset_blockers
    tokens_per_step = batch_size * grad_accum * block_size if batch_size > 0 and grad_accum > 0 and block_size > 0 else 0
    total_token_budget = tokens_per_step * max_iters

    dataset_ready = dataset_gate["passed"] if dataset_gate is not None else None
    ready_for_training = bool(len(all_blockers) == 0 and (dataset_ready is not False) and AUTHORIZED_BY_PREM)

    return {
        "valid_config": len(config_blockers) == 0,
        "dataset_ready": dataset_ready,
        "dataset_status": dataset_gate["dataset_status"] if dataset_gate is not None else "NOT_EVALUATED",
        "ready_for_training": ready_for_training,
        "is_production_release": is_prod_release,
        "owner_authorized": bool(AUTHORIZED_BY_PREM),
        "ownership_policy": ACTIVE_OWNERSHIP_POLICY,
        "config_blockers": config_blockers,
        "dataset_blockers": dataset_blockers,
        "blockers": all_blockers,
        "warnings": warnings,
        "tokens_per_optimizer_step": tokens_per_step,
        "total_training_token_budget": total_token_budget,
        "parameters": param_summary,
        "t4_vram_estimate": vram_summary,
        "dataset_gate": dataset_gate,
    }
