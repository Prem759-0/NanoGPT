#!/usr/bin/env python3
"""
scripts/evaluate_model.py
Rigorously verified, objective benchmark evaluation suite for Moon AI.

Evaluates 100 benchmark items across:
- Mathematics (exact expected answer verification)
- Python Coding (real unit test execution in restricted sandbox)
- Web/JavaScript/HTML/CSS (syntax and instruction adherence)
- Code Debugging & Correction
- Structured JSON Generation (strict json.loads & schema validation)
- Hindi & Hinglish Multilingual Reasoning
- Multi-Turn Conversational Context Continuation
- Science, Reasoning & General Knowledge
- Safety, Persona & Prompt-Injection Resistance

Separately measures:
- Solution correctness (test execution, math verification, JSON parsing)
- Code syntax validity (AST analysis)
- Keyword & concept coverage
- Degenerate repetition penalty
- Non-empty rate vs Exception/Error rate
- Latency and tokens/second throughput
"""
import os
import sys
import json
import time
import ast
import re
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from webapp.model_service import ModelService

BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "data", "benchmark", "benchmark_100.json")

def load_benchmark(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads benchmark items from JSON file or creates it if missing."""
    target_path = filepath or BENCHMARK_FILE
    if not os.path.isfile(target_path):
        from scripts.create_benchmark_100 import build_benchmark
        items = build_benchmark()
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        return items

    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================================
# Verifiers for Each Domain
# =========================================================================

def extract_python_code(text: str, entry_point: Optional[str] = None) -> str:
    """Extracts raw python code from markdown blocks or raw text."""
    blocks = re.findall(r'```(?:python|py)?\s*([\s\S]*?)```', text, re.I)
    if blocks:
        # If multiple blocks, find the one containing entry_point or the longest
        if entry_point:
            for b in blocks:
                if f"def {entry_point}" in b:
                    return b.strip()
        return blocks[0].strip()

    # If no markdown block, search for def entry_point block
    if entry_point and f"def {entry_point}" in text:
        lines = text.split("\n")
        start_idx = None
        code_lines = []
        for i, line in enumerate(lines):
            if f"def {entry_point}" in line:
                start_idx = i
                break
        if start_idx is not None:
            for line in lines[start_idx:]:
                if line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("def") and not line.startswith("#"):
                    break
                code_lines.append(line)
            return "\n".join(code_lines).strip()

    return text.strip()

def check_python_syntax(code: str) -> bool:
    """Checks whether python code parses into a valid AST without SyntaxError."""
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

import concurrent.futures

def execute_python_tests(code: str, entry_point: str, test_cases: str) -> Tuple[bool, bool, str]:
    """
    Safely executes code + unit tests in a restricted namespace with timeout protection.
    Returns: (is_syntax_valid, is_test_passed, diagnostic_message)
    """
    clean_code = extract_python_code(code, entry_point)
    if not check_python_syntax(clean_code):
        return (False, False, "SyntaxError: Code failed AST parse.")

    # Restricted builtins for evaluation security
    safe_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k in {
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
            "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "getattr", "hasattr", "hash", "hex", "int",
            "isinstance", "issubclass", "iter", "len", "list", "map", "max",
            "min", "next", "oct", "ord", "pow", "print", "range", "repr",
            "reversed", "round", "set", "slice", "sorted", "str", "sum",
            "tuple", "type", "zip", "True", "False", "None"
        }
    }
    env = {"__builtins__": safe_builtins}

    try:
        compiled_code = compile(clean_code, "<benchmark_code>", "exec")
        compiled_tests = compile(test_cases, "<benchmark_tests>", "exec")
    except Exception as e:
        return (False, False, f"CompileError: {str(e)}")

    def _eval_runner():
        exec(compiled_code, env)
        if entry_point not in env or not callable(env[entry_point]):
            return (True, False, f"EntryPointNotFound: '{entry_point}' function was not defined.")
        exec(compiled_tests, env)
        return (True, True, "All unit tests passed successfully.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_eval_runner)
        try:
            return future.result(timeout=1.5)
        except concurrent.futures.TimeoutError:
            return (True, False, "Timeout: Code execution timed out after 1.5s (likely infinite loop).")
        except AssertionError:
            return (True, False, "AssertionError: Unit test assertion failed.")
        except Exception as e:
            return (True, False, f"Runtime error: {type(e).__name__}: {str(e)}")


def evaluate_math_answer(reply: str, expected_num: float) -> Tuple[bool, str]:
    """
    Extracts numerical candidates from the reply and checks if expected_num is present.
    """
    if not reply or not reply.strip():
        return False, "Empty reply"

    # Search for fractions like 3/4
    frac_match = re.findall(r'(\d+)\s*/\s*(\d+)', reply)
    for num, den in frac_match:
        if int(den) != 0 and abs((int(num) / int(den)) - expected_num) < 0.01:
            return True, f"Matched fraction {num}/{den} == {expected_num}"

    # Extract all numbers (integers, floats, negatives)
    num_matches = re.findall(r'[-+]?\d*\.?\d+', reply)
    candidates = []
    for m in num_matches:
        try:
            val = float(m)
            candidates.append(val)
        except ValueError:
            pass

    # Check equality within tolerance
    for val in candidates:
        if abs(val - expected_num) < 0.01:
            return True, f"Found correct number: {val}"

    return False, f"Expected {expected_num}, candidates found: {candidates[:6]}"

def evaluate_json_schema(reply: str, required_keys: List[str]) -> Tuple[bool, float, str]:
    """
    Extracts and parses JSON object from reply and verifies required keys.
    Returns: (is_valid_json, key_match_ratio, diagnostic)
    """
    if not reply or not reply.strip():
        return False, 0.0, "Empty reply"

    # Try extracting markdown json block
    json_block = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', reply)
    candidate_str = json_block.group(1) if json_block else None

    # Fallback to finding first '{' and matching '}'
    if not candidate_str:
        start_idx = reply.find('{')
        end_idx = reply.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate_str = reply[start_idx:end_idx + 1]

    if not candidate_str:
        return False, 0.0, "No JSON object markers {...} found."

    try:
        data = json.loads(candidate_str)
        if not isinstance(data, dict):
            return False, 0.0, "Parsed JSON is not an object/dictionary."

        matched_keys = [k for k in required_keys if k in data]
        key_ratio = len(matched_keys) / len(required_keys) if required_keys else 1.0
        is_complete = (len(matched_keys) == len(required_keys))

        return True, key_ratio, f"Parsed JSON. Found {len(matched_keys)}/{len(required_keys)} required keys."
    except json.JSONDecodeError as je:
        return False, 0.0, f"JSONDecodeError: {str(je)[:60]}"

def detect_repetition_degeneracy(reply: str) -> float:
    """
    Computes a repetition penalty factor (0.0 to 1.0) where 1.0 means degenerate looping gibberish.
    e.g. "an an an an an an..."
    """
    if not reply or len(reply) < 30:
        return 0.0

    words = re.findall(r'\b\w+\b', reply.lower())
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        # If unique ratio is extremely low (< 0.20), it is a degenerate loop
        if unique_ratio < 0.20:
            return 1.0
        if unique_ratio < 0.35:
            return 0.5

    # Check 3-gram repetition loops
    if len(words) >= 18:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_ratio = len(set(trigrams)) / len(trigrams)
        if trigram_ratio < 0.25:
            return 0.8

    return 0.0

# =========================================================================
# Main Evaluation Engine
# =========================================================================

def run_evaluation(model_name: str, max_tokens: int = 60, benchmark_path: Optional[str] = None) -> Dict[str, Any]:
    """Runs the verified benchmark suite against the requested model."""
    benchmark_items = load_benchmark(benchmark_path)
    service = ModelService(base_dir=PROJECT_ROOT)
    success = service.load_model(model_name)
    if not success:
        err_msg = f"Failed to load model '{model_name}': {service.error_message}"
        print(f"⚠️ {err_msg}")
        summary = {
            "model_name": model_name,
            "active_model_name": "none",
            "checkpoint_path": "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": len(benchmark_items),
            "total_tokens_generated": 0,
            "total_time_seconds": 0.0,
            "tokens_per_second": 0.0,
            "non_empty_rate_pct": 0.0,
            "solution_correctness_pct": 0.0,
            "code_syntax_pct": 0.0,
            "keyword_coverage_pct": 0.0,
            "composite_quality_score": 0.0,
            "empty_count": len(benchmark_items),
            "error_count": len(benchmark_items),
            "error": err_msg,
            "results": []
        }
        _save_reports(model_name, summary)
        return summary

    results = []
    total_tokens = 0
    total_time_s = 0.0
    passed_non_empty = 0
    empty_count = 0
    error_count = 0

    verifiable_total = 0
    verifiable_passed = 0

    code_total = 0
    code_syntax_passed = 0

    keyword_hit_ratios = []
    degeneracy_penalties = []

    print("\n" + "=" * 70)
    print(f"   MOON AI BENCHMARK EVALUATION: {model_name} ({service.active_model_name})")
    print(f"   Items: {len(benchmark_items)} verified evaluation tasks")
    print("=" * 70)

    for idx, item in enumerate(benchmark_items, 1):
        t0 = time.time()
        eval_type = item.get("eval_type", "text_keywords")
        is_empty_test = (eval_type == "safety_empty" or not item["prompt"].strip())

        try:
            res = service.generate(
                prompt=item["prompt"],
                max_new_tokens=max_tokens,
                temperature=0.2,
                history=item.get("history")
            )
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            res = {
                "reply": f"[Exception: {str(e)}]",
                "raw_reply": "",
                "stop_reason": "exception",
                "tokens_generated": 0,
                "prompt_length": 0
            }

        total_time_s += elapsed
        reply = res.get("reply", "").strip()
        tokens_gen = res.get("tokens_generated", 0)
        total_tokens += tokens_gen
        stop_reason = res.get("stop_reason", "unknown")

        # Emptiness & Exception tracking
        is_exception = (stop_reason == "exception" or "[Exception:" in reply)
        is_empty = (not reply or "emitted 0 readable tokens" in reply or "empty response" in reply.lower())

        if is_exception:
            error_count += 1
        elif is_empty and not is_empty_test:
            empty_count += 1
        else:
            passed_non_empty += 1

        # Check degenerate repetition loop
        degeneracy = detect_repetition_degeneracy(reply)
        degeneracy_penalties.append(degeneracy)

        # Domain-specific verification
        item_correct = False
        item_syntax_valid = False
        diag_msg = ""
        hit_ratio = 0.0

        # 1. Math evaluation
        if eval_type == "math":
            verifiable_total += 1
            expected_num = float(item["expected_number"])
            is_correct, diag_msg = evaluate_math_answer(reply, expected_num)
            item_correct = is_correct
            if is_correct:
                verifiable_passed += 1

        # 2. Python Code execution evaluation
        elif eval_type == "code_exec":
            verifiable_total += 1
            code_total += 1
            entry_point = item["entry_point"]
            test_cases = item["test_cases"]
            syntax_ok, tests_ok, diag_msg = execute_python_tests(reply, entry_point, test_cases)
            item_syntax_valid = syntax_ok
            item_correct = tests_ok
            if syntax_ok:
                code_syntax_passed += 1
            if tests_ok:
                verifiable_passed += 1

        # 3. JSON Schema evaluation
        elif eval_type == "json_schema":
            verifiable_total += 1
            required_keys = item.get("required_keys", [])
            json_ok, key_ratio, diag_msg = evaluate_json_schema(reply, required_keys)
            item_correct = json_ok and (key_ratio >= 1.0)
            if item_correct:
                verifiable_passed += 1
            hit_ratio = key_ratio

        # 4. Text / Keyword matching (general, science, multilingual, multiturn, web, debugging)
        expected_kw = item.get("expected_keywords", [])
        if expected_kw and eval_type != "json_schema":
            reply_lower = reply.lower()
            matches = [kw for kw in expected_kw if kw.lower() in reply_lower]
            hit_ratio = len(matches) / len(expected_kw) if expected_kw else 1.0
            if eval_type not in ["math", "code_exec"]:
                item_correct = (hit_ratio >= 0.60)
            if not diag_msg:
                diag_msg = f"Keywords matched: {len(matches)}/{len(expected_kw)}"

        # If it was an empty prompt test, verify clean diagnostic reply
        if is_empty_test:
            item_correct = ("empty" in reply.lower() or "prompt" in reply.lower() or is_empty)
            diag_msg = "Handled empty prompt correctly."

        # Apply degeneracy penalty to keyword score
        adjusted_keyword_score = max(0.0, hit_ratio * (1.0 - degeneracy))
        keyword_hit_ratios.append(adjusted_keyword_score)

        t_ms = round(elapsed * 1000, 1)
        status_sym = "✅" if item_correct else ("⚠️" if hit_ratio > 0.3 else "❌")
        preview = repr(reply[:60].replace("\n", " "))
        
        # Log first 12 or every 15th item for clean output
        if idx <= 12 or idx % 15 == 0 or idx == len(benchmark_items):
            print(f"[{idx:03d}/100] {status_sym} [{item['id']}] ({t_ms}ms, {tokens_gen} tok) -> {diag_msg}")
            print(f"        Output: {preview}\n")
            sys.stdout.flush()

        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "eval_type": eval_type,
            "prompt": item["prompt"],
            "reply": reply,
            "correct": item_correct,
            "syntax_valid": item_syntax_valid if eval_type == "code_exec" else True,
            "tokens_generated": tokens_gen,
            "latency_ms": t_ms,
            "keyword_hit_ratio": round(hit_ratio, 2),
            "adjusted_score": round(adjusted_keyword_score, 2),
            "degeneracy_penalty": degeneracy,
            "diagnostic": diag_msg
        })

    num_items = len(benchmark_items)
    tokens_per_sec = round(total_tokens / max(0.001, total_time_s), 2)
    non_empty_rate = round((passed_non_empty / num_items) * 100, 1)

    sol_correct_pct = round((verifiable_passed / max(1, verifiable_total)) * 100, 1)
    code_syntax_pct = round((code_syntax_passed / max(1, code_total)) * 100, 1)
    avg_keyword_pct = round((sum(keyword_hit_ratios) / num_items) * 100, 1)
    avg_degeneracy_pct = round((sum(degeneracy_penalties) / num_items) * 100, 1)

    # Coherence score: 100% minus average degeneracy
    coherence_pct = max(0.0, round(100.0 - (avg_degeneracy_pct * 100), 1))

    # Rigorous Composite Quality Score Formula:
    # 40% Solution Correctness (Code Unit Tests + Math Calculation + JSON Validation)
    # 25% Keyword & Semantic Coverage (penalized for degeneracy)
    # 15% Code Syntax Validity
    # 10% Coherence (Freedom from token looping)
    # 10% Non-Empty Completion Rate
    composite_score = round(
        (0.40 * sol_correct_pct) +
        (0.25 * avg_keyword_pct) +
        (0.15 * code_syntax_pct) +
        (0.10 * coherence_pct) +
        (0.10 * non_empty_rate),
        1
    )

    summary = {
        "model_name": model_name,
        "active_model_name": service.active_model_name,
        "checkpoint_path": service.current_checkpoint_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_items": num_items,
        "verifiable_tasks_total": verifiable_total,
        "verifiable_tasks_passed": verifiable_passed,
        "solution_correctness_pct": sol_correct_pct,
        "code_syntax_pct": code_syntax_pct,
        "keyword_coverage_pct": avg_keyword_pct,
        "coherence_pct": coherence_pct,
        "non_empty_rate_pct": non_empty_rate,
        "empty_count": empty_count,
        "error_count": error_count,
        "tokens_per_second": tokens_per_sec,
        "total_tokens_generated": total_tokens,
        "total_time_seconds": round(total_time_s, 2),
        "composite_quality_score": composite_score,
        "results": results
    }

    _save_reports(model_name, summary)

    print("=" * 70)
    print(f"BENCHMARK COMPLETED: {model_name}")
    print(f"  • Verifiable Tasks Correct:   {verifiable_passed}/{verifiable_total} ({sol_correct_pct}%)")
    print(f"  • Python Code Syntax Passed:  {code_syntax_passed}/{code_total} ({code_syntax_pct}%)")
    print(f"  • Keyword / Semantic Score:   {avg_keyword_pct}%")
    print(f"  • Model Coherence:            {coherence_pct}%")
    print(f"  • Non-Empty Output Rate:      {non_empty_rate}% (Empty: {empty_count}, Errors: {error_count})")
    print(f"  • Tokens Per Second:          {tokens_per_sec}")
    print(f"  • COMPOSITE QUALITY SCORE:    {composite_score} / 100")
    print("=" * 70)

    return summary

def _save_reports(model_name: str, summary: Dict[str, Any]):
    """Saves both timestamped JSON and latest Markdown evaluation reports."""
    report_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Save JSON report
    report_file = os.path.join(report_dir, f"eval_{model_name}_{int(time.time())}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save latest markdown summary
    md_file = os.path.join(report_dir, f"eval_{model_name}_latest.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Evaluation Report: {model_name}\n\n")
        f.write(f"- **Evaluated Model**: `{model_name}` ({summary.get('active_model_name', 'N/A')})\n")
        f.write(f"- **Timestamp**: {summary['timestamp']}\n")
        f.write(f"- **Checkpoint**: `{summary.get('checkpoint_path', 'N/A')}`\n")
        f.write(f"- **Total Items Evaluated**: {summary['total_items']}\n\n")
        f.write("## Core Quality Metrics\n\n")
        f.write(f"- **Solution Correctness**: **{summary['solution_correctness_pct']}%** ({summary.get('verifiable_tasks_passed', 0)}/{summary.get('verifiable_tasks_total', 0)} math/code/JSON tasks passed)\n")
        f.write(f"- **Python Code Syntax Validity**: **{summary['code_syntax_pct']}%**\n")
        f.write(f"- **Keyword & Concept Coverage**: **{summary['keyword_coverage_pct']}%**\n")
        f.write(f"- **Coherence (Non-Degenerate)**: **{summary['coherence_pct']}%**\n")
        f.write(f"- **Non-Empty Output Rate**: **{summary['non_empty_rate_pct']}%** (Empty: {summary.get('empty_count', 0)}, Errors: {summary.get('error_count', 0)})\n")
        f.write(f"- **Tokens / Second**: {summary['tokens_per_second']}\n")
        f.write(f"- **Composite Quality Score**: **{summary['composite_quality_score']} / 100**\n\n")

        f.write("## Per-Item Results Sample\n\n")
        f.write("| ID | Domain | Correct | Syntax | Keywords | Output Preview |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in summary.get("results", [])[:30]:
            clean_p = r["reply"][:40].replace("\n", " ").replace("|", "/")
            f.write(f"| {r['id']} | {r['domain']} | {'✅' if r['correct'] else '❌'} | {'✅' if r['syntax_valid'] else '❌'} | {int(r['keyword_hit_ratio']*100)}% | `{clean_p}` |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Moon AI Objective Benchmark Evaluation.")
    parser.add_argument("--model", type=str, default="scratch", choices=["scratch", "gpt2", "llama-3.2", "qwen-2.5-coder-7b"])
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(model_name=args.model, max_tokens=args.tokens, benchmark_path=args.benchmark)
