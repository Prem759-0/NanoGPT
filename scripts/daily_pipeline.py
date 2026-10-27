#!/usr/bin/env python3
"""
scripts/daily_pipeline.py
Automated Daily Improvement Pipeline for NanoGPT Lab.
Workflow:
1. Check for new training examples (e.g. from user_learned_data.jsonl)
2. Run data validation, deduplication, and update splits
3. Run dataset audit checks (AST syntax and length distributions)
4. Execute deterministic regression test suite
5. (Optional) Run controlled fine-tuning if GPU is available and sufficient new data exists
6. Run benchmark evaluation on active/candidate model
7. Check promotion criteria and update model registry
8. Generate comprehensive daily report
"""
import os
import sys
import json
import subprocess
import argparse
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.prepare_dataset import prepare_splits
from scripts.audit_dataset import audit_file
from scripts.evaluate_model import run_evaluation
from scripts.promote_best_model import evaluate_and_promote
from scripts.daily_report import generate_daily_report

def run_step(title: str, func, *args, **kwargs):
    """Executes a pipeline step with timing and error handling."""
    print("\n" + "=" * 65)
    print(f"▶️ STEP: {title}")
    print("=" * 65)
    t0 = datetime.now()
    try:
        res = func(*args, **kwargs)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"✅ {title} completed successfully in {elapsed:.2f}s")
        return res
    except Exception as e:
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"❌ {title} failed after {elapsed:.2f}s: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run NanoGPT Daily Improvement Pipeline.")
    parser.add_argument("--eval_model", type=str, default="scratch", choices=["scratch", "gpt2", "llama-3.2", "qwen-2.5-coder-7b"])
    parser.add_argument("--train", action="store_true", help="Launch fine-tuning if GPU is available")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("*" * 65)
    print("   NANOGPT LAB — DAILY AUTONOMOUS AI IMPROVEMENT PIPELINE")
    print("*" * 65)
    print(f"Date:       {datetime.now(timezone.utc).isoformat()}")
    print(f"Evaluator:  {args.eval_model}")
    print(f"Auto-Train: {args.train}\n")

    # Step 1: Update & Validate Dataset Splits
    sources = [
        os.path.join(PROJECT_ROOT, "moon_ai_master_dataset.json"),
        os.path.join(PROJECT_ROOT, "dataset.json"),
        os.path.join(PROJECT_ROOT, "chatgpt_dataset.json"),
        os.path.join(PROJECT_ROOT, "data", "user_learned_data.jsonl")
    ]
    existing_sources = [s for s in sources if os.path.isfile(s)]
    splits_dir = os.path.join(PROJECT_ROOT, "data", "splits")

    manifest = run_step(
        "1. Dataset Preparation & Deduplication",
        prepare_splits,
        source_files=existing_sources,
        output_dir=splits_dir,
        seed=args.seed
    )

    # Step 2: Audit Splits
    train_path = os.path.join(splits_dir, "train.json")
    audit_res = run_step(
        "2. Dataset Quality & AST Syntax Audit",
        audit_file,
        filepath=train_path
    )
    if audit_res.get("syntax_error_records", 0) > 0:
        print(f"⚠️ Warning: Found {audit_res['syntax_error_records']} records with syntax errors!")

    # Step 3: Run Inference Regression Test Suite
    def run_regression_tests():
        cmd = [sys.executable, "-m", "unittest", "tests/test_model_inference.py"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise RuntimeError(f"Regression tests failed with exit code {res.returncode}")
        return True

    run_step("3. Regression Test Suite", run_regression_tests)

    # Step 4: Optional Fine-Tuning
    if args.train:
        import torch
        if torch.cuda.is_available():
            from scripts.train_assistant import train as run_train
            train_args = argparse.Namespace(
                base_model="unsloth/Llama-3.2-1B-Instruct",
                train_path=os.path.join(splits_dir, "train.json"),
                val_path=os.path.join(splits_dir, "val.json"),
                max_steps=100,
                batch_size=2,
                grad_accum=4,
                learning_rate=2e-4,
                lora_r=16,
                max_seq_length=1024,
                load_in_4bit=True,
                output_dir=os.path.join(PROJECT_ROOT, "lora_output"),
                export_gguf=False,
                seed=args.seed
            )
            run_step("4. Fine-Tuning Run", run_train, train_args)
        else:
            print("ℹ️ Skipping training step: CUDA GPU not detected on local machine. (Use Colab GPU for training)")

    # Step 5: Benchmark Evaluation
    eval_summary = run_step(
        "5. Objective Benchmark Evaluation",
        run_evaluation,
        model_name=args.eval_model,
        max_tokens=60
    )

    # Step 6: Model Promotion Gate
    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    latest_eval_file = os.path.join(reports_dir, f"eval_{args.eval_model}_latest.md")
    eval_json_file = [f for f in os.listdir(reports_dir) if f.startswith(f"eval_{args.eval_model}_") and f.endswith(".json")]
    if eval_json_file:
        eval_json_path = os.path.join(reports_dir, sorted(eval_json_file)[-1])
        run_step(
            "6. Model Registry & Promotion Verification",
            evaluate_and_promote,
            candidate_report_path=eval_json_path,
            min_score_threshold=60.0
        )

    # Step 7: Daily Report Generation
    run_step(
        "7. Daily Report Generation",
        generate_daily_report
    )

    print("\n" + "=" * 65)
    print("🎉 DAILY PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
