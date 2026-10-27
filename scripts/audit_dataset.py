#!/usr/bin/env python3
"""
scripts/audit_dataset.py
Audits dataset files without modifying originals.
Checks:
- Schema validity (instruction/response or messages)
- Empty or whitespace-only records
- Exact duplicate detection
- Near-duplicate detection (normalized text similarity)
- Python code syntax validation (via ast.parse)
- Language distribution (English, Hindi, Hinglish)
- Persona & creator attribution check (Prem Gaikwad)
- Length distributions (min, max, mean chars)
Outputs structured audit reports to data/reports/ and console.
"""
import os
import sys
import json
import ast
import re
import argparse
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

def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t

def is_hindi_or_hinglish(text: str) -> bool:
    """Detects Devanagari script or common Hinglish phonetic words."""
    # Check for Devanagari Unicode block (\u0900-\u097F)
    if re.search(r'[\u0900-\u097F]', text):
        return True
    # Common Hinglish terms
    hinglish_markers = {
        "kya", "hai", "kaise", "karen", "dhanyawad", "namaste", "bhai",
        "mujhe", "batao", "sikhao", "samajh", "aaya", "karo", "kyun", "hoga"
    }
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(words.intersection(hinglish_markers)) >= 2

def check_python_code_blocks(response: str) -> Tuple[int, List[str]]:
    """Extracts explicit python code blocks from response and tests with ast.parse."""
    code_blocks = re.findall(r'```(?:python|py)\s*([\s\S]*?)```', response, re.I)
    errors = []
    valid_count = 0
    for idx, code in enumerate(code_blocks):
        clean_code = code.strip()
        if not clean_code:
            continue
        try:
            ast.parse(clean_code)
            valid_count += 1
        except SyntaxError as e:
            errors.append(f"Block #{idx+1} syntax error: {e.msg} at line {e.lineno}")
        except Exception as e:
            errors.append(f"Block #{idx+1} parse error: {str(e)}")
    return valid_count, errors

def audit_file(filepath: str) -> Dict[str, Any]:
    """Audits a single JSON or JSONL dataset file."""
    if not os.path.isfile(filepath):
        return {"error": f"File not found: {filepath}"}

    records = []
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        return {"error": f"JSONL decode error on line {line_no}: {str(e)}"}
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    return {"error": f"Expected JSON list of records, got {type(data)}"}
            except Exception as e:
                return {"error": f"JSON decode error: {str(e)}"}

    total_records = len(records)
    valid_records = 0
    empty_instructions = 0
    empty_responses = 0
    exact_duplicates = 0
    near_duplicates = 0
    syntax_error_records = 0
    multilingual_count = 0
    creator_attribution_count = 0

    seen_instructions = set()
    seen_normalized = set()
    syntax_issues: List[Dict[str, Any]] = []

    inst_lengths = []
    resp_lengths = []

    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            continue

        # Handle both {"instruction": ..., "response": ...} and {"messages": [...]}
        if "messages" in r and isinstance(r["messages"], list):
            inst = next((m.get("content", "") for m in r["messages"] if m.get("role") == "user"), "")
            resp = next((m.get("content", "") for m in r["messages"] if m.get("role") == "assistant"), "")
        else:
            inst = r.get("instruction", "")
            resp = r.get("response", "")

        if not isinstance(inst, str) or not inst.strip():
            empty_instructions += 1
            continue

        if not isinstance(resp, str) or not resp.strip():
            empty_responses += 1
            continue

        inst_str = inst.strip()
        resp_str = resp.strip()

        inst_lengths.append(len(inst_str))
        resp_lengths.append(len(resp_str))

        # Duplicate detection
        if inst_str in seen_instructions:
            exact_duplicates += 1
        else:
            seen_instructions.add(inst_str)

        norm_inst = normalize_text(inst_str)
        if norm_inst in seen_normalized:
            near_duplicates += 1
        else:
            seen_normalized.add(norm_inst)

        # Code syntax check
        valid_blocks, errors = check_python_code_blocks(resp_str)
        if errors:
            syntax_error_records += 1
            syntax_issues.append({
                "record_index": idx,
                "instruction_preview": inst_str[:60],
                "errors": errors
            })

        # Multilingual check
        if is_hindi_or_hinglish(inst_str) or is_hindi_or_hinglish(resp_str):
            multilingual_count += 1

        # Creator attribution check
        if "prem gaikwad" in inst_str.lower() or "prem gaikwad" in resp_str.lower():
            creator_attribution_count += 1

        valid_records += 1

    return {
        "file": os.path.basename(filepath),
        "total_records": total_records,
        "valid_records": valid_records,
        "empty_instructions": empty_instructions,
        "empty_responses": empty_responses,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "syntax_error_records": syntax_error_records,
        "multilingual_records": multilingual_count,
        "creator_attribution_records": creator_attribution_count,
        "avg_instruction_chars": round(sum(inst_lengths) / len(inst_lengths), 1) if inst_lengths else 0,
        "avg_response_chars": round(sum(resp_lengths) / len(resp_lengths), 1) if resp_lengths else 0,
        "syntax_issues_sample": syntax_issues[:5]
    }

def main():
    parser = argparse.ArgumentParser(description="Audit dataset files for NanoGPT Lab.")
    parser.add_argument("--file", type=str, help="Path to specific dataset file to audit")
    args = parser.parse_args()

    target_files = []
    if args.file:
        target_files.append(args.file)
    else:
        # Default standard dataset files
        candidates = [
            os.path.join(PROJECT_ROOT, "moon_ai_master_dataset.json"),
            os.path.join(PROJECT_ROOT, "dataset.json"),
            os.path.join(PROJECT_ROOT, "chatgpt_dataset.json"),
            os.path.join(PROJECT_ROOT, "data", "user_learned_data.jsonl")
        ]
        target_files = [c for c in candidates if os.path.isfile(c)]

    print("=" * 70)
    print("   NANOGPT DATASET INTEGRITY & SYNTAX AUDIT REPORT")
    print("=" * 70)

    summary_reports = []
    for fpath in target_files:
        report = audit_file(fpath)
        summary_reports.append(report)
        print(f"\n[File]: {report.get('file', fpath)}")
        if "error" in report:
            print(f"  ❌ Error: {report['error']}")
            continue

        print(f"  • Total Records:              {report['total_records']}")
        print(f"  • Clean & Valid:              {report['valid_records']}")
        print(f"  • Exact Duplicates:           {report['exact_duplicates']}")
        print(f"  • Near Duplicates:            {report['near_duplicates']}")
        print(f"  • Python Syntax Errors:       {report['syntax_error_records']}")
        print(f"  • Multilingual (Hindi/Hing):  {report['multilingual_records']}")
        print(f"  • Prem Gaikwad Attribution:   {report['creator_attribution_records']}")
        print(f"  • Avg Prompt Length:          {report['avg_instruction_chars']} chars")
        print(f"  • Avg Response Length:        {report['avg_response_chars']} chars")

    # Save to data/reports/dataset_audit_report.json
    out_report_path = os.path.join(PROJECT_ROOT, "data", "reports", "dataset_audit_report.json")
    os.makedirs(os.path.dirname(out_report_path), exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as out_f:
        json.dump(summary_reports, out_f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Audit report saved: {os.path.relpath(out_report_path, PROJECT_ROOT)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
