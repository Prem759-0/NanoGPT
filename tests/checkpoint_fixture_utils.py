"""
tests/checkpoint_fixture_utils.py
Utilities to manage transient Policy B test fixture checkpoints
for environments (such as clean CI runners) where binary model weights are not committed to Git.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: Policy B (Random Gaussian Scratch Initialization, Zero Teacher Weights)

CRITICAL PROVENANCE AND INTEGRITY RULES:
1. Status is strictly "CI_TEST_FIXTURE" (Never "PILOT_TRAINED").
2. is_production_trained is strictly False.
3. promotable is strictly False.
4. is_test_fixture is strictly True; run_id is "ci_test_fixture".
5. No fabricated training metrics (loss/perplexity are clearly labelled test-schema-only).
6. Never overwrites or deletes genuine production checkpoints.
"""

import os
from typing import Tuple
import torch
from model import GPTConfig, GPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")
SCALED_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_scaled_v1.pt")


def is_fixture_checkpoint(ckpt_path: str) -> bool:
    """Returns True if the checkpoint at ckpt_path is an unpromotable test fixture."""
    if not os.path.isfile(ckpt_path):
        return False
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            return False
        if data.get("is_test_fixture") is True:
            return True
        if data.get("status") in ("CI_TEST_FIXTURE", "ENGINEERING_SMOKE_TEST_ONLY"):
            return True
        cfg = data.get("config", {}) or data.get("metadata", {})
        if cfg.get("is_test_fixture") is True or cfg.get("run_id") == "ci_test_fixture":
            return True
        return False
    except Exception:
        return False


def ensure_pilot_checkpoint_fixture() -> Tuple[str, bool]:
    """
    Ensures moon_chat_pilot_v0.pt exists for testing.
    If missing (e.g. clean CI checkout), synthesizes a transient Policy B test fixture.
    Never overwrites an existing genuine checkpoint.
    Returns (path, was_created).
    """
    os.makedirs(os.path.dirname(PILOT_CKPT_PATH), exist_ok=True)
    if os.path.isfile(PILOT_CKPT_PATH):
        # Genuine or existing checkpoint found: do NOT overwrite
        return PILOT_CKPT_PATH, False

    conf = GPTConfig(
        block_size=512,
        vocab_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=False
    )
    model = GPT(conf)

    payload = {
        "status": "CI_TEST_FIXTURE",
        "is_production_trained": False,
        "promotable": False,
        "is_test_fixture": True,
        "model": model.state_dict(),
        "model_args": {
            "block_size": 512,
            "vocab_size": 512,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "dropout": 0.1,
            "bias": False
        },
        "config": {
            "dataset": "moon-chat-v0",
            "model_name": "moon-chat-pilot-25m",
            "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
            "initialization": "random_gaussian_scratch",
            "student_init_method": "random_gaussian_scratch",
            "teacher_checkpoint_weights_transferred": False,
            "teacher_models": ["Qwen2.5-7B-Instruct-GGUF"],
            "teacher_license": "Apache-2.0",
            "dataset_provenance": "DETERMINISTIC_CURRICULUM_TEMPLATE + ORIGINAL_SELF_AUTHORED_QA",
            "run_id": "ci_test_fixture",
            "is_test_fixture": True,
            "is_production_trained": False,
            "promotable": False,
            "status": "CI_TEST_FIXTURE",
            "created_at": "2026-11-06T00:00:00Z",
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            # Explicit test-only schema placeholders (never interpreted as genuine training metrics)
            "test_schema_val_loss": 99.0,
            "test_schema_perplexity": 999.0,
            "notes": "CI test fixture only with random Gaussian weights. NOT a trained production model."
        }
    }
    torch.save(payload, PILOT_CKPT_PATH)
    return PILOT_CKPT_PATH, True


def ensure_scaled_checkpoint_fixture() -> Tuple[str, bool]:
    """
    Ensures moon_chat_scaled_v1.pt exists for testing.
    If missing (e.g. clean CI checkout), synthesizes a transient Policy B test fixture.
    Never overwrites an existing genuine checkpoint.
    Returns (path, was_created).
    """
    os.makedirs(os.path.dirname(SCALED_CKPT_PATH), exist_ok=True)
    if os.path.isfile(SCALED_CKPT_PATH):
        # Genuine or existing checkpoint found: do NOT overwrite
        return SCALED_CKPT_PATH, False

    conf = GPTConfig(
        block_size=512,
        vocab_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=False
    )
    model = GPT(conf)

    payload = {
        "status": "CI_TEST_FIXTURE",
        "is_production_trained": False,
        "promotable": False,
        "is_test_fixture": True,
        "model": model.state_dict(),
        "model_args": {
            "block_size": 512,
            "vocab_size": 512,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "dropout": 0.1,
            "bias": False
        },
        "config": {
            "dataset": "moon-chat-v1-scaled",
            "model_name": "moon-chat-scaled-25m",
            "ownership_policy": "TEACHER_STUDENT_RANDOM_INIT",
            "initialization": "random_gaussian_scratch",
            "student_init_method": "random_gaussian_scratch",
            "teacher_checkpoint_weights_transferred": False,
            "teacher_models": ["Qwen2.5-7B-Instruct-GGUF"],
            "teacher_license": "Apache-2.0",
            "dataset_provenance": "DETERMINISTIC_CURRICULUM_TEMPLATE + ORIGINAL_SELF_AUTHORED_QA",
            "run_id": "ci_test_fixture",
            "is_test_fixture": True,
            "is_production_trained": False,
            "promotable": False,
            "status": "CI_TEST_FIXTURE",
            "created_at": "2026-11-06T00:00:00Z",
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            # Explicit test-only schema placeholders (never interpreted as genuine training metrics)
            "test_schema_val_loss": 99.0,
            "test_schema_perplexity": 999.0,
            "notes": "CI test fixture only with random Gaussian weights. NOT a trained production model."
        }
    }
    torch.save(payload, SCALED_CKPT_PATH)
    return SCALED_CKPT_PATH, True


def cleanup_checkpoint_fixture(ckpt_path: str, was_created: bool) -> None:
    """
    Safely removes transient test fixture checkpoint only if it was created during testing
    and verifies that the file is indeed a test fixture before deletion.
    Never overwrites or deletes genuine checkpoints.
    """
    if not was_created or not os.path.isfile(ckpt_path):
        return
    try:
        # Extra safety check: confirm it's actually marked as a fixture
        if is_fixture_checkpoint(ckpt_path):
            os.remove(ckpt_path)
    except Exception:
        pass
