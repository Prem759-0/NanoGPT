"""
Moon AI — Conversational Dataset Curator & Splitter
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Deduplicates, validates, and partitions genuine chat records into
train/val/test splits with zero cross-split prompt/context leakage.
"""

import json
import random
from typing import List, Dict, Any, Tuple, Set
from datasets.chat.schema import validate_chat_record, normalize_text, compute_conversation_sha256

def deduplicate_and_validate_records(
    records: List[Dict[str, Any]],
    is_production: bool = True
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validates and deduplicates conversational records.
    Filters out duplicates based on:
    1. Exact canonical SHA-256 hash.
    2. Normalized prompt & response text.
    3. Production policy violations (e.g. synthetic dev fixtures).
    
    Returns:
    (valid_unique_records, rejection_reasons)
    """
    seen_hashes: Set[str] = set()
    seen_normalized_prompts: Set[str] = set()
    valid_records: List[Dict[str, Any]] = []
    rejections: List[str] = []

    for idx, rec in enumerate(records):
        rec_id = rec.get("id", f"record_{idx}")

        # Validate schema and production policy
        is_valid, err_msg = validate_chat_record(rec, is_production=is_production)
        if not is_valid:
            rejections.append(f"Rejected '{rec_id}': {err_msg}")
            continue

        # Exact SHA-256 deduplication
        canonical_hash = rec.get("sha256") or compute_conversation_sha256(rec["messages"])
        if canonical_hash in seen_hashes:
            rejections.append(f"Rejected '{rec_id}': Duplicate exact conversation hash ({canonical_hash[:12]}...).")
            continue

        # Normalized prompt deduplication
        first_user_prompt = ""
        for turn in rec["messages"]:
            if turn.get("role") == "user":
                first_user_prompt = turn.get("content", "")
                break

        norm_prompt = normalize_text(first_user_prompt)
        if norm_prompt in seen_normalized_prompts:
            rejections.append(f"Rejected '{rec_id}': Duplicate normalized user prompt ('{first_user_prompt[:30]}...').")
            continue

        # Add to accepted set
        seen_hashes.add(canonical_hash)
        seen_normalized_prompts.add(norm_prompt)
        rec_copy = dict(rec)
        rec_copy["sha256"] = canonical_hash
        valid_records.append(rec_copy)

    return valid_records, rejections

def split_chat_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically partitions records into train, val, and test splits.
    Ensures absolute zero prompt/context leakage across splits.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0.")

    # Shuffle deterministically
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total == 0:
        return [], [], []

    # Calculate split sizes
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0
    n_test = max(1, int(n_total * test_ratio)) if n_total >= 3 else 0
    n_train = n_total - n_val - n_test

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    # Verify zero leakage
    train_hashes = {r["sha256"] for r in train_set}
    val_hashes = {r["sha256"] for r in val_set}
    test_hashes = {r["sha256"] for r in test_set}

    leakage_train_val = train_hashes.intersection(val_hashes)
    leakage_train_test = train_hashes.intersection(test_hashes)
    leakage_val_test = val_hashes.intersection(test_hashes)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError("Data leakage detected between dataset splits!")

    # Assign split attribute
    for r in train_set:
        r["split"] = "train"
    for r in val_set:
        r["split"] = "val"
    for r in test_set:
        r["split"] = "test"

    return train_set, val_set, test_set
