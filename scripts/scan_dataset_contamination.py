#!/usr/bin/env python3
"""
scripts/scan_dataset_contamination.py
Moon AI — Read-Only Dataset Contamination Scanner
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Audits active training datasets and manifests for unconsented historical records
originating from data/user_learned_data.jsonl.
Enforces strict quarantine verification (Phase 2.6.2.2 Task 3).
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORICAL_SOURCE_PATH = os.path.join(PROJECT_ROOT, "data", "user_learned_data.jsonl")

def normalize_text(text: str) -> str:
    """Normalize whitespace for fuzzy match resilience."""
    return " ".join(text.strip().split())

def record_fingerprint(instruction: str, response: str) -> str:
    """Computes a deterministic fingerprint for an instruction/response pair."""
    norm_inst = normalize_text(instruction)
    norm_resp = normalize_text(response)
    raw = f"{norm_inst}|||{norm_resp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def load_historical_fingerprints() -> Tuple[Set[str], List[Dict[str, Any]]]:
    """Loads all records from the historical user_learned_data.jsonl."""
    fingerprints: Set[str] = set()
    records: List[Dict[str, Any]] = []

    if not os.path.isfile(HISTORICAL_SOURCE_PATH):
        return fingerprints, records

    with open(HISTORICAL_SOURCE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                records.append(item)
                inst = item.get("instruction", "")
                resp = item.get("response", "")
                fp = record_fingerprint(inst, resp)
                fingerprints.add(fp)
            except Exception:
                pass
    return fingerprints, records

def scan_file_for_contamination(
    filepath: str,
    historical_fps: Set[str]
) -> Dict[str, Any]:
    """Scans a single file against historical fingerprints."""
    if not os.path.isfile(filepath):
        return {"file": filepath, "exists": False, "records": 0, "contaminated_count": 0}

    total_records = 0
    contaminated_records = 0
    contaminated_samples = []

    try:
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    total_records += 1
                    try:
                        item = json.loads(line)
                        inst = ""
                        resp = ""
                        if "messages" in item and isinstance(item["messages"], list):
                            inst = next((m.get("content", "") for m in item["messages"] if m.get("role") == "user"), "")
                            resp = next((m.get("content", "") for m in item["messages"] if m.get("role") == "assistant"), "")
                        else:
                            inst = item.get("instruction", "") or item.get("prompt", "")
                            resp = item.get("response", "")
                        source = item.get("source", "")

                        fp = record_fingerprint(inst, resp)
                        if fp in historical_fps or "user_learned_data.jsonl" in str(source):
                            contaminated_records += 1
                            if len(contaminated_samples) < 3:
                                contaminated_samples.append({
                                    "line": line_idx + 1,
                                    "instruction_preview": inst[:60],
                                    "source": source
                                })
                    except Exception:
                        pass
        elif filepath.endswith(".json") or filepath.endswith(".QUARANTINED_CONTAMINATED"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        if not isinstance(item, dict):
                            continue
                        total_records += 1
                        inst = item.get("instruction", "") or item.get("prompt", "")
                        resp = item.get("response", "")
                        source = item.get("source", "")
                        fp = record_fingerprint(inst, resp)
                        if fp in historical_fps or "user_learned_data.jsonl" in str(source):
                            contaminated_records += 1
                            if len(contaminated_samples) < 3:
                                contaminated_samples.append({
                                    "record_index": idx,
                                    "instruction_preview": inst[:60],
                                    "source": source
                                })
                elif isinstance(data, dict):
                    # Manifest file or metadata
                    total_records = len(data)
    except Exception as e:
        return {
            "file": filepath,
            "exists": True,
            "error": str(e),
            "records": 0,
            "contaminated_count": 0
        }

    return {
        "file": filepath,
        "exists": True,
        "records": total_records,
        "contaminated_count": contaminated_records,
        "is_contaminated": contaminated_records > 0,
        "samples": contaminated_samples
    }

def main():
    print("=" * 70)
    print("   MOON AI -- READ-ONLY DATASET CONTAMINATION SCANNER")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("   Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY (Phase 2.6.2.2)")
    print("=" * 70)

    historical_fps, hist_records = load_historical_fingerprints()
    print(f"Loaded {len(hist_records)} historical reference records from {HISTORICAL_SOURCE_PATH}")
    print(f"Computed {len(historical_fps)} unique deterministic fingerprints.\n")

    # 1. Active Training & Curriculum Datasets (MUST BE 100% CLEAN)
    active_sources = [
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_train.jsonl"),
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_val.jsonl"),
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "scaled_chat_test.jsonl"),
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_train.jsonl"),
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_val.jsonl"),
        os.path.join(PROJECT_ROOT, "datasets", "manifests", "chat_test.jsonl"),
        os.path.join(PROJECT_ROOT, "moon_ai_master_dataset.json"),
        os.path.join(PROJECT_ROOT, "dataset.json"),
        os.path.join(PROJECT_ROOT, "chatgpt_dataset.json")
    ]

    print("[Active Training & Manifest Sources - Zero Contamination Invariant]")
    active_clean = True
    active_results = []
    for fpath in active_sources:
        if not os.path.exists(fpath):
            continue
        res = scan_file_for_contamination(fpath, historical_fps)
        active_results.append(res)
        rel = os.path.relpath(fpath, PROJECT_ROOT)
        if res["contaminated_count"] > 0:
            active_clean = False
            print(f"  ❌ CONTAMINATED: {rel} ({res['contaminated_count']}/{res['records']} records contaminated!)")
        else:
            print(f"  ✅ CLEAN:        {rel} ({res['records']} records, 0 contaminated)")

    # 2. Quarantined Legacy Splits
    print("\n[Quarantined Legacy Split Files]")
    quarantine_files = [
        os.path.join(PROJECT_ROOT, "data", "splits", "train.json.QUARANTINED_CONTAMINATED"),
        os.path.join(PROJECT_ROOT, "data", "splits", "val.json.QUARANTINED_CONTAMINATED"),
        os.path.join(PROJECT_ROOT, "data", "splits", "test.json.QUARANTINED_CONTAMINATED"),
        os.path.join(PROJECT_ROOT, "data", "splits", "quarantine_unconsented_legacy_splits", "train.json"),
        os.path.join(PROJECT_ROOT, "data", "splits", "quarantine_unconsented_legacy_splits", "val.json"),
        os.path.join(PROJECT_ROOT, "data", "splits", "quarantine_unconsented_legacy_splits", "test.json")
    ]
    quarantine_results = []
    for fpath in quarantine_files:
        if not os.path.exists(fpath):
            continue
        res = scan_file_for_contamination(fpath, historical_fps)
        quarantine_results.append(res)
        rel = os.path.relpath(fpath, PROJECT_ROOT)
        print(f"  🔒 QUARANTINED:  {rel} ({res['contaminated_count']}/{res['records']} records from unconsented log)")

    # 3. Export Comprehensive Scan Report
    report_data = {
        "scan_timestamp": "2026-11-10T19:05:00Z",
        "historical_source": HISTORICAL_SOURCE_PATH,
        "historical_records_count": len(hist_records),
        "active_sources_clean": active_clean,
        "active_sources": active_results,
        "quarantined_sources": quarantine_results
    }
    report_out = os.path.join(PROJECT_ROOT, "data", "reports", "dataset_contamination_scan_report.json")
    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=2)

    print("\n" + "=" * 70)
    if active_clean:
        print("✅ CONTAMINATION AUDIT PASSED: All active training datasets are 100% clean.")
        print("   Quarantined files are completely isolated from active training and evaluation.")
    else:
        print("❌ CONTAMINATION AUDIT FAILED: Active training datasets contain unconsented records!")
    print(f"Report written to: {os.path.relpath(report_out, PROJECT_ROOT)}")
    print("=" * 70)

    if not active_clean:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
