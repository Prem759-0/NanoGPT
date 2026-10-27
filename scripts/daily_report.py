#!/usr/bin/env python3
"""
scripts/daily_report.py
Generates a comprehensive daily status report summarizing:
- Dataset split counts and recent additions
- Active model status from model_registry.json
- Latest evaluation benchmark results
- Pipeline health and open recommendations
Outputs Markdown report to data/reports/daily_report_<YYYY-MM-DD>.md and prints to console.
"""
import os
import sys
import json
import glob
from datetime import datetime, timezone
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def generate_daily_report() -> str:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest_path = os.path.join(PROJECT_ROOT, "data", "splits", "manifest.json")
    registry_path = os.path.join(PROJECT_ROOT, "models", "model_registry.json")
    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")

    # Load dataset manifest
    manifest: Dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    # Load model registry
    registry: Dict[str, Any] = {}
    if os.path.isfile(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            pass

    # Find latest eval report
    eval_files = glob.glob(os.path.join(reports_dir, "eval_*.json"))
    latest_eval: Optional[Dict[str, Any]] = None
    if eval_files:
        eval_files.sort(key=os.path.getmtime, reverse=True)
        try:
            with open(eval_files[0], "r", encoding="utf-8") as f:
                latest_eval = json.load(f)
        except Exception:
            pass

    current_best = registry.get("current_best_model", {
        "name": "Moon AI Sovereign (7.6B v1)",
        "checkpoint": "Moon-AI-v1.Q4_K_M.gguf",
        "composite_score": 75.0
    })

    splits = manifest.get("splits", {})
    categories = manifest.get("category_distribution", {})

    md = []
    md.append(f"# Sovereign AI Assistant — Daily Pipeline Report")
    md.append(f"**Date**: {today_str}  ")
    md.append(f"**Architect & Creator**: Prem Gaikwad (India 🇮🇳)  \n")
    md.append("---")

    md.append("## 1. Active Best Model Status")
    md.append(f"- **Active Model**: **{current_best.get('name')}**")
    md.append(f"- **Checkpoint Artifact**: `{current_best.get('checkpoint')}`")
    md.append(f"- **Composite Quality Score**: **{current_best.get('composite_score', 'N/A')} / 100**")
    md.append(f"- **Last Promotion Timestamp**: {current_best.get('promoted_at', 'Initial setup')}")
    md.append("")

    md.append("## 2. Dataset Split Statistics")
    md.append(f"- **Total Clean Verified Records**: {manifest.get('total_records', 'N/A')}")
    md.append(f"- **Train Split (90%)**: {splits.get('train', {}).get('count', 0)} examples")
    md.append(f"- **Validation Split (5%)**: {splits.get('val', {}).get('count', 0)} examples")
    md.append(f"- **Blind Test Split (5%)**: {splits.get('test', {}).get('count', 0)} examples")
    md.append(f"- **Filtered Duplicates**: {manifest.get('filtering_stats', {}).get('skipped_duplicates', 0)}")
    md.append("")
    md.append("### Domain Category Breakdown")
    for cat, count in categories.items():
        md.append(f"- **{cat.title()}**: {count} records")
    md.append("")

    md.append("## 3. Latest Benchmark Evaluation Results")
    if latest_eval:
        md.append(f"- **Evaluated Model**: `{latest_eval.get('model_name')}`")
        md.append(f"- **Tokens / Sec**: {latest_eval.get('tokens_per_second')}")
        md.append(f"- **Non-Empty Output Rate**: {latest_eval.get('non_empty_rate_pct')}%")
        md.append(f"- **Keyword Coverage**: {latest_eval.get('keyword_coverage_pct')}%")
        md.append(f"- **Code Syntax Accuracy**: {latest_eval.get('coding_syntax_valid_pct')}%")
        md.append(f"- **Composite Benchmark Score**: **{latest_eval.get('composite_quality_score')} / 100**")
    else:
        md.append("No automated evaluation runs recorded today.")
    md.append("")

    md.append("## 4. Promotion History & Integrity Log")
    prom_hist = registry.get("promotion_history", [])
    if prom_hist:
        md.append("| Timestamp | Candidate | Score | Status | Notes |")
        md.append("|---|---|---|---|---|")
        for h in prom_hist[-5:]:
            ts = h.get("evaluated_at", "")[:19]
            name = h.get("candidate_name", "")
            score = h.get("score", "")
            status = h.get("status", "")
            notes = h.get("replaced_model", ", ".join(h.get("reasons", [])))
            md.append(f"| {ts} | `{name}` | {score} | **{status}** | {notes} |")
    else:
        md.append("No candidate models evaluated yet.")
    md.append("")

    md.append("## 5. Daily Recommendations")
    md.append("1. **Data Additions**: Expand domain-specific test cases in `data/splits/test.json`.")
    md.append("2. **Colab Training**: When new verified data reaches 50+ samples, run `notebooks/NanoGPT_Training_Pipeline.ipynb` on Colab T4 GPU.")
    md.append("3. **Direct GPU Serving**: Utilize Step 9 ngrok URL to connect local webapp to Colab GPU.")

    report_content = "\n".join(md)
    out_file = os.path.join(reports_dir, f"daily_report_{today_str}.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print(f"\n✅ Report written to: {os.path.relpath(out_file, PROJECT_ROOT)}")
    return out_file

def main():
    generate_daily_report()

if __name__ == "__main__":
    main()
