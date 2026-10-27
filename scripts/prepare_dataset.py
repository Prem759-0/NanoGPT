#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
Reproducible dataset preparation and splitting pipeline for NanoGPT Lab.
Features:
- Reads all master datasets without modifying originals.
- Deduplicates exact and near-duplicate instructions.
- Verifies python code syntax in examples.
- Standardizes into ChatML / OpenAI conversational messages format and instruction/response.
- Generates reproducible, non-leaking train/val/test splits with a fixed seed.
- Produces versioned split manifest with checksums and category counts.
"""
import os
import sys
import json
import random
import hashlib
import ast
import re
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_SYSTEM_PROMPT = (
    "You are Moon AI, the world-class coding, reasoning, and conversational AI assistant "
    "created and deployed by developer Prem Gaikwad from India. You provide robust, accurate, "
    "and complete code and explanations with zero external paid APIs."
)

def normalize_key(text: str) -> str:
    """Normalize text for near-duplicate collision checking."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t

def calculate_sha256(filepath: str) -> str:
    """Computes SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def categorize_record(instruction: str, response: str) -> str:
    """Categorizes example based on keywords."""
    inst_low = instruction.lower()
    resp_low = response.lower()
    if any(k in inst_low for k in ["python", "javascript", "html", "css", "code", "function", "def ", "class ", "algorithm", "debug", "react"]):
        return "coding"
    elif any(k in inst_low for k in ["dna", "cell", "biology", "planet", "sun", "moon", "physics", "gravity", "chemistry", "anatomy", "organ"]):
        return "science"
    elif any(k in inst_low for k in ["prem gaikwad", "who made", "who created", "who are you", "your name"]):
        return "persona"
    elif re.search(r'[\u0900-\u097F]', inst_low) or any(k in inst_low for k in ["kya", "kaise", "namaste", "dhanyawad"]):
        return "multilingual"
    return "general_conversation"

def prepare_splits(
    source_files: List[str],
    output_dir: str,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """Processes source datasets, cleans, deduplicates, and splits data."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    cleaned_records: List[Dict[str, Any]] = []
    seen_exact = set()
    seen_near = set()
    skipped_empty = 0
    skipped_duplicates = 0
    skipped_syntax_errors = 0

    for fpath in source_files:
        if not os.path.isfile(fpath):
            continue

        raw_items = []
        if fpath.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_items.append(json.loads(line))
                        except Exception:
                            pass
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_items = data
                except Exception:
                    pass

        source_name = os.path.basename(fpath)
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            # Extract instruction & response
            if "messages" in item and isinstance(item["messages"], list):
                inst = next((m.get("content", "") for m in item["messages"] if m.get("role") == "user"), "")
                resp = next((m.get("content", "") for m in item["messages"] if m.get("role") == "assistant"), "")
            else:
                inst = item.get("instruction", "")
                resp = item.get("response", "")

            if not inst or not isinstance(inst, str) or not inst.strip():
                skipped_empty += 1
                continue
            if not resp or not isinstance(resp, str) or not resp.strip():
                skipped_empty += 1
                continue

            inst_clean = inst.strip()
            resp_clean = resp.strip()

            # Exact duplicate filter
            if inst_clean in seen_exact:
                skipped_duplicates += 1
                continue

            # Near duplicate filter
            norm_k = normalize_key(inst_clean)
            if norm_k in seen_near:
                skipped_duplicates += 1
                continue

            # Code syntax verification
            py_blocks = re.findall(r'```(?:python|py)\s*([\s\S]*?)```', resp_clean, re.I)
            has_syntax_err = False
            for code_block in py_blocks:
                try:
                    ast.parse(code_block.strip())
                except SyntaxError:
                    has_syntax_err = True
                    break
            if has_syntax_err:
                skipped_syntax_errors += 1
                continue

            seen_exact.add(inst_clean)
            seen_near.add(norm_k)

            category = categorize_record(inst_clean, resp_clean)
            record_id = f"moon_ai_{len(cleaned_records) + 1:05d}"

            # Standard ChatML conversational format
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": inst_clean},
                {"role": "assistant", "content": resp_clean}
            ]

            cleaned_records.append({
                "id": record_id,
                "instruction": inst_clean,
                "response": resp_clean,
                "messages": messages,
                "category": category,
                "source": source_name
            })

    # Shuffle deterministically
    random.shuffle(cleaned_records)
    total_count = len(cleaned_records)

    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_data = cleaned_records[:train_end]
    val_data = cleaned_records[train_end:val_end]
    test_data = cleaned_records[val_end:]

    # Write split files
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")

    for path, data in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Categories breakdown
    cat_counts: Dict[str, int] = {}
    for r in cleaned_records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    manifest = {
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "total_records": total_count,
        "splits": {
            "train": {
                "count": len(train_data),
                "file": "train.json",
                "sha256": calculate_sha256(train_path)
            },
            "val": {
                "count": len(val_data),
                "file": "val.json",
                "sha256": calculate_sha256(val_path)
            },
            "test": {
                "count": len(test_data),
                "file": "test.json",
                "sha256": calculate_sha256(test_path)
            }
        },
        "filtering_stats": {
            "skipped_empty": skipped_empty,
            "skipped_duplicates": skipped_duplicates,
            "skipped_syntax_errors": skipped_syntax_errors
        },
        "category_distribution": cat_counts,
        "sources_scanned": [os.path.basename(s) for s in source_files]
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest

def main():
    parser = argparse.ArgumentParser(description="Prepare and split NanoGPT datasets.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    default_sources = [
        os.path.join(PROJECT_ROOT, "moon_ai_master_dataset.json"),
        os.path.join(PROJECT_ROOT, "dataset.json"),
        os.path.join(PROJECT_ROOT, "chatgpt_dataset.json"),
        os.path.join(PROJECT_ROOT, "data", "user_learned_data.jsonl")
    ]
    sources = [s for s in default_sources if os.path.isfile(s)]

    print("=" * 70)
    print("   NANOGPT REPRODUCIBLE DATASET PREPARATION PIPELINE")
    print("=" * 70)
    print(f"Sources: {[os.path.basename(s) for s in sources]}")
    print(f"Output:  {args.output_dir}")
    print(f"Seed:    {args.seed}\n")

    manifest = prepare_splits(
        source_files=sources,
        output_dir=args.output_dir,
        seed=args.seed
    )

    print("✅ Splits Generated Successfully:")
    print(f"  • Total Clean Records:  {manifest['total_records']}")
    print(f"  • Train Set (90%):      {manifest['splits']['train']['count']} records")
    print(f"  • Validation Set (5%):  {manifest['splits']['val']['count']} records")
    print(f"  • Test Set (5%):        {manifest['splits']['test']['count']} records")
    print(f"  • Duplicates Filtered:  {manifest['filtering_stats']['skipped_duplicates']}")
    print(f"\nCategory Distribution:   {json.dumps(manifest['category_distribution'], indent=4)}")
    print(f"\nManifest saved to: {os.path.join(args.output_dir, 'manifest.json')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
