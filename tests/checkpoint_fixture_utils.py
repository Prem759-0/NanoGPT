"""
tests/checkpoint_fixture_utils.py
Utilities to manage transient Policy B test fixture checkpoints
for environments (such as clean CI runners) where binary model weights are not committed to Git.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: Policy B (Random Gaussian Scratch Initialization, Zero Teacher Weights)
"""

import os
from typing import Tuple
import torch
from model import GPTConfig, GPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt")
SCALED_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_scaled_v1.pt")


def ensure_pilot_checkpoint_fixture() -> Tuple[str, bool]:
    """
    Ensures moon_chat_pilot_v0.pt exists for testing.
    If missing (e.g. clean CI checkout), synthesizes a transient Policy B test fixture.
    Returns (path, was_created).
    """
    os.makedirs(os.path.dirname(PILOT_CKPT_PATH), exist_ok=True)
    if os.path.isfile(PILOT_CKPT_PATH):
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
        "status": "PILOT_TRAINED",
        "is_production_trained": True,
        "promotable": True,
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
            "run_id": "ci_test_pilot_fixture",
            "trained_at": "2026-11-06T00:00:00Z",
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            "final_train_loss": 5.82,
            "final_val_loss": 5.82,
            "perplexity": 337.0,
            "iterations": 1,
            "is_production_trained": True,
            "status": "PILOT_TRAINED",
            "notes": "CI test fixture checkpoint synthesized for automated testing."
        },
        "iter_num": 1,
        "best_val_loss": 5.82
    }
    torch.save(payload, PILOT_CKPT_PATH)
    return PILOT_CKPT_PATH, True


def ensure_scaled_checkpoint_fixture() -> Tuple[str, bool]:
    """
    Ensures moon_chat_scaled_v1.pt exists for testing.
    If missing (e.g. clean CI checkout), synthesizes a transient Policy B test fixture.
    Returns (path, was_created).
    """
    os.makedirs(os.path.dirname(SCALED_CKPT_PATH), exist_ok=True)
    if os.path.isfile(SCALED_CKPT_PATH):
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
        "status": "PILOT_TRAINED",
        "is_production_trained": True,
        "promotable": True,
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
            "run_id": "ci_test_scaled_fixture",
            "trained_at": "2026-11-06T00:00:00Z",
            "architect": "Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)",
            "final_train_loss": 5.7510,
            "final_val_loss": 5.6672,
            "perplexity": 289.22,
            "iterations": 5,
            "is_production_trained": True,
            "status": "PILOT_TRAINED",
            "notes": "CI test fixture checkpoint synthesized for automated testing."
        },
        "iter_num": 5,
        "best_val_loss": 5.6672
    }
    torch.save(payload, SCALED_CKPT_PATH)
    return SCALED_CKPT_PATH, True


def cleanup_checkpoint_fixture(ckpt_path: str, was_created: bool) -> None:
    """Removes transient test fixture checkpoint if it was created during testing."""
    if was_created and os.path.isfile(ckpt_path):
        try:
            os.remove(ckpt_path)
        except Exception:
            pass
