#!/usr/bin/env python3
"""
scripts/promote_best_model.py
Evaluates candidate model against current best model and handles promotion.
Prevents quality regression and maintains a versioned model registry.
"""
import os
import sys
import json
import argparse
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

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "model_registry.json")

def load_registry() -> Dict[str, Any]:
    """Loads or initializes model registry."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    if os.path.isfile(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "version": "1.0.0",
        "current_best_model": {
            "name": "Moon AI Sovereign (7.6B v1)",
            "checkpoint": "Moon-AI-v1.Q4_K_M.gguf",
            "composite_score": 75.0,
            "promoted_at": datetime.now(timezone.utc).isoformat()
        },
        "promotion_history": []
    }

def save_registry(data: Dict[str, Any]):
    """Saves model registry to disk."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def evaluate_and_promote(candidate_report_path: str, min_score_threshold: float = 60.0) -> bool:
    """
    Evaluates candidate model evaluation report against promotion criteria.
    Acceptance Criteria:
    1. Non-empty rate must be 100% (zero silent blank responses).
    2. Coding syntax validity must be 100%.
    3. Composite quality score must be >= current best score AND >= min_score_threshold.
    """
    if not os.path.isfile(candidate_report_path):
        print(f"❌ Evaluation report not found: {candidate_report_path}")
        return False

    with open(candidate_report_path, "r", encoding="utf-8") as f:
        cand = json.load(f)

    registry = load_registry()
    current_best = registry.get("current_best_model", {})
    current_best_score = current_best.get("composite_score", 0.0)

    cand_name = cand.get("model_name", "Unknown Candidate")
    cand_score = cand.get("composite_quality_score", 0.0)
    cand_non_empty = cand.get("non_empty_rate_pct", 0.0)
    cand_syntax = cand.get("coding_syntax_valid_pct", 0.0)

    print("=" * 65)
    print("   MODEL PROMOTION GATE & REGRESSION VERIFICATION")
    print("=" * 65)
    print(f"Current Best Model:   {current_best.get('name')} (Score: {current_best_score})")
    print(f"Candidate Model:      {cand_name} (Score: {cand_score})")
    print(f"Candidate Non-Empty:  {cand_non_empty}% (Required: 100%)")
    print(f"Candidate Syntax:     {cand_syntax}% (Required: 100%)")
    print("-" * 65)

    rejection_reasons = []
    if cand_non_empty < 100.0:
        rejection_reasons.append(f"Non-empty rate ({cand_non_empty}%) is below 100% threshold.")
    if cand_syntax < 100.0:
        rejection_reasons.append(f"Coding syntax validity ({cand_syntax}%) is below 100% threshold.")
    if cand_score < min_score_threshold:
        rejection_reasons.append(f"Composite score ({cand_score}) is below minimum floor ({min_score_threshold}).")
    if cand_score < current_best_score:
        rejection_reasons.append(f"Candidate score ({cand_score}) regressed compared to current best ({current_best_score}).")

    if rejection_reasons:
        print("❌ PROMOTION REJECTED:")
        for r in rejection_reasons:
            print(f"  • {r}")
        print("\nPrevious best model remains active.")
        # Log to registry history
        registry["promotion_history"].append({
            "candidate_name": cand_name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "status": "REJECTED",
            "score": cand_score,
            "reasons": rejection_reasons
        })
        save_registry(registry)
        print("=" * 65)
        return False

    print("🎉 PROMOTION ACCEPTED! Candidate meets all quality and non-regression criteria.")
    # Promote
    previous_model = registry["current_best_model"]
    registry["current_best_model"] = {
        "name": cand_name,
        "checkpoint": cand.get("checkpoint_path", "unknown"),
        "composite_score": cand_score,
        "tokens_per_second": cand.get("tokens_per_second", 0),
        "promoted_at": datetime.now(timezone.utc).isoformat()
    }
    registry["promotion_history"].append({
        "candidate_name": cand_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROMOTED",
        "score": cand_score,
        "replaced_model": previous_model.get("name")
    })
    save_registry(registry)
    print(f"Model registry updated: {os.path.relpath(REGISTRY_PATH, PROJECT_ROOT)}")
    print("=" * 65)
    return True

def main():
    parser = argparse.ArgumentParser(description="Promote evaluated model if it exceeds current best.")
    parser.add_argument("--report", type=str, required=True, help="Path to evaluation report JSON")
    parser.add_argument("--min_score", type=float, default=60.0, help="Minimum composite quality floor")
    args = parser.parse_args()

    evaluate_and_promote(candidate_report_path=args.report, min_score_threshold=args.min_score)

if __name__ == "__main__":
    main()
