#!/usr/bin/env python3
"""
scripts/evaluate_model.py
Objective benchmark evaluation suite for NanoGPT Lab.
Evaluates:
- General conversation
- Basic factual and science questions
- English, Hindi, and Hinglish
- Python, JavaScript, HTML, and CSS
- Debugging and code correction
- Creator attribution (Prem Gaikwad)
- Multi-turn conversation continuation
- Latency, token generation rate, and non-empty rate
Supports comparing models and generating versioned evaluation reports.
"""
import os
import sys
import json
import time
import ast
import re
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List

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

# Fixed, versioned benchmark evaluation set (15 items)
BENCHMARK_PROMPTS = [
    {
        "id": "bench_01_greeting",
        "domain": "general",
        "prompt": "Hi, how are you today?",
        "expected_keywords": ["hello", "hi", "assist", "help", "good", "today"],
        "history": None
    },
    {
        "id": "bench_02_dog",
        "domain": "science",
        "prompt": "What is a dog?",
        "expected_keywords": ["animal", "mammal", "canine", "pet", "domestic", "four"],
        "history": None
    },
    {
        "id": "bench_03_sun_moon",
        "domain": "science",
        "prompt": "Explain the Sun and Moon in simple English.",
        "expected_keywords": ["sun", "star", "moon", "earth", "light", "orbit"],
        "history": None
    },
    {
        "id": "bench_04_math",
        "domain": "math",
        "prompt": "What is 2 + 3?",
        "expected_keywords": ["5", "five"],
        "history": None
    },
    {
        "id": "bench_05_py_add",
        "domain": "coding_python",
        "prompt": "Write a Python function that adds two numbers.",
        "expected_keywords": ["def ", "return", "+"],
        "history": None
    },
    {
        "id": "bench_06_py_reverse",
        "domain": "coding_python",
        "prompt": "Write a Python function to reverse a string.",
        "expected_keywords": ["def ", "[::-1]", "reverse"],
        "history": None
    },
    {
        "id": "bench_07_web_button",
        "domain": "coding_web",
        "prompt": "Write an HTML button with JavaScript that shows an alert when clicked.",
        "expected_keywords": ["<button", "onclick", "alert(", "</button>"],
        "history": None
    },
    {
        "id": "bench_08_css_center",
        "domain": "coding_web",
        "prompt": "How do you center a div using CSS flexbox?",
        "expected_keywords": ["display: flex", "justify-content: center", "align-items: center"],
        "history": None
    },
    {
        "id": "bench_09_debug_python",
        "domain": "debugging",
        "prompt": "Fix this broken Python code:\n```python\ndef greet(name)\nprint('Hello' name)\n```",
        "expected_keywords": [":", ",", "+"],
        "history": None
    },
    {
        "id": "bench_10_multilingual_hindi",
        "domain": "multilingual",
        "prompt": "कंप्यूटर क्या है?",
        "expected_keywords": ["कंप्यूटर", "मशीन", "डेटा", "इलेक्ट्रॉनिक"],
        "history": None
    },
    {
        "id": "bench_11_multilingual_hinglish",
        "domain": "multilingual",
        "prompt": "Python programming kaise sikhein?",
        "expected_keywords": ["python", "practice", "code", "basic", "sikhein"],
        "history": None
    },
    {
        "id": "bench_12_creator",
        "domain": "persona",
        "prompt": "Who developed you?",
        "expected_keywords": ["prem gaikwad", "prem", "india"],
        "history": None
    },
    {
        "id": "bench_13_json_format",
        "domain": "instruction_following",
        "prompt": "Output a JSON object with keys 'status' and 'version'.",
        "expected_keywords": ["{", "}", "\"status\"", "\"version\""],
        "history": None
    },
    {
        "id": "bench_14_multiturn",
        "domain": "conversation",
        "prompt": "Can you explain that more simply?",
        "expected_keywords": ["language", "code", "programming", "simple"],
        "history": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level, general-purpose programming language known for readability."}
        ]
    },
    {
        "id": "bench_15_empty_safety",
        "domain": "safety",
        "prompt": "",
        "expected_keywords": ["empty", "prompt", "error", "valid"],
        "history": None
    }
]

def check_code_syntax(text: str) -> bool:
    """Verifies whether python code blocks in text are syntactically valid."""
    blocks = re.findall(r'```(?:python|py)\s*([\s\S]*?)```', text, re.I)
    if not blocks:
        return True # No python blocks to fail
    for block in blocks:
        try:
            ast.parse(block.strip())
        except SyntaxError:
            return False
    return True

def run_evaluation(model_name: str, max_tokens: int = 150) -> Dict[str, Any]:
    """Runs the full benchmark against the requested model."""
    service = ModelService(base_dir=PROJECT_ROOT)
    success = service.load_model(model_name)
    if not success:
        return {"error": f"Failed to load model '{model_name}': {service.error_message}"}

    results = []
    total_tokens = 0
    total_time_s = 0.0
    passed_non_empty = 0
    keyword_hits_total = 0
    valid_syntax_count = 0
    coding_count = 0

    print(f"\nEvaluating Model: {model_name} ({service.active_model_name})")
    print("-" * 65)

    for item in BENCHMARK_PROMPTS:
        t0 = time.time()
        try:
            res = service.generate(
                prompt=item["prompt"],
                max_new_tokens=max_tokens,
                temperature=0.2, # Low temperature for deterministic evaluation
                history=item["history"]
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

        # Check non-emptiness
        is_empty = not reply or "emitted 0 readable tokens" in reply or "empty response" in reply.lower()
        if not is_empty:
            passed_non_empty += 1

        # Check keywords
        expected = item["expected_keywords"]
        matches = [kw for kw in expected if kw.lower() in reply.lower()]
        hit_ratio = len(matches) / len(expected) if expected else 1.0
        keyword_hits_total += hit_ratio

        # Check syntax for coding items
        is_syntax_valid = True
        if "coding" in item["domain"]:
            coding_count += 1
            is_syntax_valid = check_code_syntax(reply)
            if is_syntax_valid:
                valid_syntax_count += 1

        t_ms = round(elapsed * 1000, 1)
        print(f"[{item['id']}] Time: {t_ms}ms | Tokens: {tokens_gen} | Keywords: {len(matches)}/{len(expected)}")
        print(f"  Prompt: {repr(item['prompt'][:50])}")
        print(f"  Reply:  {repr(reply[:80])}\n")

        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "prompt": item["prompt"],
            "reply": reply,
            "tokens_generated": tokens_gen,
            "latency_ms": t_ms,
            "stop_reason": res.get("stop_reason", "unknown"),
            "keyword_hit_ratio": round(hit_ratio, 2),
            "is_syntax_valid": is_syntax_valid
        })

    num_items = len(BENCHMARK_PROMPTS)
    avg_keyword_score = round((keyword_hits_total / num_items) * 100, 1)
    non_empty_rate = round((passed_non_empty / num_items) * 100, 1)
    syntax_rate = round((valid_syntax_count / max(1, coding_count)) * 100, 1)
    tokens_per_sec = round(total_tokens / max(0.001, total_time_s), 2)

    # Composite score formula: 40% Keyword coverage, 30% Non-emptiness, 20% Code syntax, 10% Consistency
    composite_score = round(
        (0.40 * avg_keyword_score) + (0.30 * non_empty_rate) + (0.20 * syntax_rate) + (0.10 * (100 if passed_non_empty == num_items else 50)),
        1
    )

    summary = {
        "model_name": model_name,
        "active_model_name": service.active_model_name,
        "checkpoint_path": service.current_checkpoint_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_items": num_items,
        "total_tokens_generated": total_tokens,
        "total_time_seconds": round(total_time_s, 2),
        "tokens_per_second": tokens_per_sec,
        "non_empty_rate_pct": non_empty_rate,
        "keyword_coverage_pct": avg_keyword_score,
        "coding_syntax_valid_pct": syntax_rate,
        "composite_quality_score": composite_score,
        "results": results
    }

    # Save reports
    report_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"eval_{model_name}_{int(time.time())}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save latest markdown summary
    md_file = os.path.join(report_dir, f"eval_{model_name}_latest.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Evaluation Report: {model_name}\n\n")
        f.write(f"- **Timestamp**: {summary['timestamp']}\n")
        f.write(f"- **Checkpoint**: `{summary['checkpoint_path']}`\n")
        f.write(f"- **Tokens / Sec**: {summary['tokens_per_second']}\n")
        f.write(f"- **Non-Empty Output Rate**: {summary['non_empty_rate_pct']}%\n")
        f.write(f"- **Keyword Coverage**: {summary['keyword_coverage_pct']}%\n")
        f.write(f"- **Code Syntax Accuracy**: {summary['coding_syntax_valid_pct']}%\n")
        f.write(f"- **Composite Quality Score**: **{summary['composite_quality_score']} / 100**\n\n")
        f.write("## Per-Item Results\n\n")
        f.write("| ID | Domain | Tokens | Latency | Keywords | Syntax Valid | Sample Output |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            preview = r["reply"][:40].replace("\n", " ").replace("|", "/")
            f.write(f"| {r['id']} | {r['domain']} | {r['tokens_generated']} | {r['latency_ms']}ms | {int(r['keyword_hit_ratio']*100)}% | {'Yes' if r['is_syntax_valid'] else 'No'} | `{preview}` |\n")

    print("=" * 65)
    print(f"Evaluation Completed for {model_name}:")
    print(f"  • Tokens/Sec:               {tokens_per_sec}")
    print(f"  • Non-Empty Rate:           {non_empty_rate}%")
    print(f"  • Keyword Match Score:      {avg_keyword_score}%")
    print(f"  • Code Syntax Validity:     {syntax_rate}%")
    print(f"  • Composite Quality Score:  {composite_score} / 100")
    print(f"  • Report: {os.path.relpath(report_file, PROJECT_ROOT)}")
    print("=" * 65)

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate NanoGPT / Moon AI models against fixed benchmark.")
    parser.add_argument("--model", type=str, default="scratch", choices=["scratch", "gpt2", "llama-3.2", "qwen-2.5-coder-7b"])
    parser.add_argument("--max_tokens", type=int, default=80)
    args = parser.parse_args()

    run_evaluation(model_name=args.model, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
