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

from training.common.checkpoint import (
    verify_checkpoint_provenance,
    compute_file_sha256,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from evaluation.security.evaluation_attestation import verify_evaluation_attestation

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "model_registry.json")

def load_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads or initializes model registry."""
    reg_p = registry_path or REGISTRY_PATH
    os.makedirs(os.path.dirname(os.path.abspath(reg_p)), exist_ok=True)
    if os.path.isfile(reg_p):
        try:
            with open(reg_p, "r", encoding="utf-8") as f:
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

def save_registry(data: Dict[str, Any], registry_path: Optional[str] = None):
    """Saves model registry to disk."""
    reg_p = registry_path or REGISTRY_PATH
    os.makedirs(os.path.dirname(os.path.abspath(reg_p)), exist_ok=True)
    with open(reg_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def evaluate_and_promote(
    candidate_report_path: str,
    min_score_threshold: float = 60.0,
    registry_path: Optional[str] = None,
    evaluators_path: Optional[str] = None,
    verifiers_path: Optional[str] = None,
    base_dir: str = PROJECT_ROOT
) -> bool:
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

    registry = load_registry(registry_path)
    current_best = registry.get("current_best_model", {})
    current_best_score = current_best.get("composite_score", 0.0)

    cand_name = cand.get("model_name", "Unknown Candidate")
    cand_score = cand.get("composite_quality_score", 0.0)
    cand_non_empty = cand.get("non_empty_rate_pct", 0.0)
    cand_syntax = cand.get("code_syntax_pct", cand.get("coding_syntax_valid_pct", 0.0))

    print("=" * 65)
    print("   MODEL PROMOTION GATE & REGRESSION VERIFICATION")
    print("=" * 65)
    print(f"Current Best Model:   {current_best.get('name')} (Score: {current_best_score})")
    print(f"Candidate Model:      {cand_name} (Score: {cand_score})")
    print(f"Candidate Non-Empty:  {cand_non_empty}% (Required: 100%)")
    print(f"Candidate Syntax:     {cand_syntax}% (Required: 100%)")
    print("-" * 65)

    rejection_reasons = []

    # Checkpoint provenance & training isolation gate
    cand_mode = cand.get("training_mode")
    cand_eligible = cand.get("is_production_eligible")
    if cand_mode == "DEVELOPMENT_SYNTHETIC":
        rejection_reasons.append("Candidate model is marked 'training_mode: DEVELOPMENT_SYNTHETIC' and cannot be promoted to production.")
    if cand_eligible is False:
        rejection_reasons.append("Candidate model has 'is_production_eligible: false'. Rejected from production.")

    has_dataset_manifest = "dataset_manifest_sha256" in cand or "dataset_manifest_path" in cand
    has_test_results = "test_results_summary_sha256" in cand or "test_results" in cand or "test_results_path" in cand

    # Task 4 & Stage 2.0.6: Cryptographic evaluation provenance and tamper-proof attestation
    eval_ok, eval_err = verify_evaluation_attestation(
        cand, evaluators_path=evaluators_path, enforce_production_hashes=(has_dataset_manifest or has_test_results)
    )
    if not eval_ok:
        rejection_reasons.append(
            f"Evaluation report failed cryptographic attestation verification: {eval_err} "
            "Manually edited or unsigned evaluation reports are strictly prohibited."
        )

    # Recompute evaluation manifest hash from disk
    if has_dataset_manifest:
        cand_manifest_path = cand.get("dataset_manifest_path")
        cand_manifest_sha = str(cand.get("dataset_manifest_sha256", "")).strip().lower()
        if not cand_manifest_sha or cand_manifest_sha in ("none", "null", ""):
            rejection_reasons.append("Missing or placeholder 'dataset_manifest_sha256' in candidate evaluation report. Genuine 64-char hex SHA-256 required.")
        elif cand_manifest_path:
            full_manifest = os.path.normpath(os.path.join(PROJECT_ROOT, cand_manifest_path)) if not os.path.isabs(cand_manifest_path) else cand_manifest_path
            if not os.path.isfile(full_manifest):
                rejection_reasons.append(f"Evaluation dataset manifest not found on disk: '{cand_manifest_path}'")
            else:
                actual_manifest_sha = compute_file_sha256(full_manifest)
                if actual_manifest_sha.lower() != cand_manifest_sha.lower():
                    rejection_reasons.append(
                        f"Evaluation dataset manifest SHA-256 mismatch! Report specifies '{cand_manifest_sha}', but actual file yields '{actual_manifest_sha}'."
                    )

    # Recompute test results summary hash and verify aggregate metrics
    if has_test_results:
        cand_test_sha = str(cand.get("test_results_summary_sha256", "")).strip().lower()
        if not cand_test_sha or cand_test_sha in ("none", "null", ""):
            rejection_reasons.append("Missing or placeholder 'test_results_summary_sha256' in candidate evaluation report. Genuine 64-char hex SHA-256 required.")
        else:
            test_results = cand.get("test_results")
            if not test_results and cand.get("test_results_path"):
                tr_path = cand["test_results_path"]
                full_tr_path = os.path.normpath(os.path.join(PROJECT_ROOT, tr_path)) if not os.path.isabs(tr_path) else tr_path
                if os.path.isfile(full_tr_path):
                    try:
                        with open(full_tr_path, "r", encoding="utf-8") as f:
                            test_results = json.load(f)
                    except Exception as e:
                        rejection_reasons.append(f"Failed to read test_results_path '{tr_path}': {e}")
            if test_results and isinstance(test_results, list):
                from evaluation.security.evaluation_attestation import (
                    compute_test_results_summary_sha256,
                    recompute_aggregate_metrics
                )
                comp_test_sha = compute_test_results_summary_sha256(test_results)
                if comp_test_sha.lower() != cand_test_sha.lower():
                    rejection_reasons.append(
                        f"Test results summary hash mismatch! Report specifies '{cand_test_sha}', but actual test results yield '{comp_test_sha}'."
                    )
                recomputed = recompute_aggregate_metrics(test_results)
                if abs(recomputed["composite_quality_score"] - cand_score) > 0.01:
                    rejection_reasons.append(
                        f"Candidate composite_quality_score mismatch: report declared {cand_score}, but individual test results calculate to {recomputed['composite_quality_score']}."
                    )
                if abs(recomputed["non_empty_rate_pct"] - cand_non_empty) > 0.01:
                    rejection_reasons.append(
                        f"Candidate non_empty_rate_pct mismatch: report declared {cand_non_empty}%, but individual test results calculate to {recomputed['non_empty_rate_pct']}%."
                    )
                if abs(recomputed["code_syntax_pct"] - cand_syntax) > 0.01:
                    rejection_reasons.append(
                        f"Candidate code_syntax_pct mismatch: report declared {cand_syntax}%, but individual test results calculate to {recomputed['code_syntax_pct']}%."
                    )

    ckpt_path = cand.get("checkpoint_path") or cand.get("checkpoint")
    cand_ckpt_sha = str(cand.get("checkpoint_sha256", "")).strip().lower()

    if not ckpt_path or not str(ckpt_path).strip():
        rejection_reasons.append("Missing required 'checkpoint_path' in candidate evaluation report.")
    else:
        full_ckpt = os.path.normpath(os.path.join(PROJECT_ROOT, ckpt_path)) if not os.path.isabs(ckpt_path) else ckpt_path
        if not os.path.isfile(full_ckpt):
            rejection_reasons.append(f"Candidate checkpoint file not found on disk: '{ckpt_path}'")
        else:
            if not cand_ckpt_sha or len(cand_ckpt_sha) != 64:
                rejection_reasons.append("Missing or invalid 'checkpoint_sha256' in candidate evaluation report (must be 64-char hex SHA-256).")
            else:
                actual_ckpt_sha = compute_file_sha256(full_ckpt)
                if actual_ckpt_sha != cand_ckpt_sha:
                    rejection_reasons.append(
                        f"Candidate checkpoint_sha256 mismatch! Report specifies '{cand_ckpt_sha}', "
                        f"but actual file bytes yield '{actual_ckpt_sha}'."
                    )

            valid_ckpt, ckpt_meta, ckpt_errs = verify_checkpoint_provenance(
                full_ckpt,
                mode=TRAINING_MODE_PRODUCTION,
                recompute_manifests=True,
                base_dir=base_dir,
                verifiers_path=verifiers_path
            )
            if not valid_ckpt:
                rejection_reasons.append(f"Underlying checkpoint failed production provenance verification: {'; '.join(ckpt_errs)}")
            else:
                ckpt_model_name = ckpt_meta.get("model_name")
                if ckpt_model_name and cand_name and ckpt_model_name != cand_name:
                    rejection_reasons.append(
                        f"Candidate model name '{cand_name}' does not match model name inside checkpoint '{ckpt_model_name}'."
                    )

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
        save_registry(registry, registry_path)
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
    save_registry(registry, registry_path)
    reg_display = registry_path or REGISTRY_PATH
    try:
        reg_display_str = os.path.relpath(reg_display, PROJECT_ROOT)
    except ValueError:
        reg_display_str = reg_display
    print(f"Model registry updated: {reg_display_str}")
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
