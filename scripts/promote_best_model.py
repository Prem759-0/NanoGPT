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
from typing import Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.common.checkpoint import (
    verify_checkpoint_provenance,
    compute_file_sha256,
    TRAINING_MODE_PRODUCTION,
    TRAINING_MODE_DEVELOPMENT
)
from evaluation.security.evaluation_attestation import (
    verify_evaluation_attestation,
    compute_test_results_summary_sha256,
    recompute_aggregate_metrics
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "model_registry.json")

def verify_path_containment(target_path: str, base_dir: str = PROJECT_ROOT) -> Tuple[bool, str, Optional[str]]:
    """
    Enforces canonical path containment within the approved directory root (SEC-04).
    1. Rejects empty or whitespace paths.
    2. Rejects raw '..' directory traversal components.
    3. Resolves symlinks and canonical absolute paths using os.path.realpath.
    4. Enforces commonpath containment against canonical approved root.
    Returns: (is_contained, canonical_resolved_path, error_reason)
    """
    if not target_path or not str(target_path).strip():
        return False, "", "Path is empty or whitespace."

    clean_target = str(target_path).strip()

    # Reject raw '..' traversal patterns
    norm_parts = os.path.normpath(clean_target).split(os.sep)
    if ".." in norm_parts:
        return False, "", f"Directory traversal attempt ('..') detected in path: '{clean_target}'."

    canonical_base = os.path.realpath(os.path.abspath(base_dir))

    if not os.path.isabs(clean_target):
        joined = os.path.join(canonical_base, clean_target)
    else:
        joined = clean_target

    canonical_target = os.path.realpath(os.path.abspath(joined))

    try:
        common = os.path.commonpath([canonical_base, canonical_target])
    except ValueError:
        return False, "", f"Path '{clean_target}' is on a separate filesystem drive or outside approved root '{canonical_base}'."

    if common != canonical_base:
        return False, "", f"Path '{clean_target}' escapes approved directory root '{canonical_base}'."

    return True, canonical_target, None

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
    ok_rep, full_rep_path, err_rep = verify_path_containment(candidate_report_path, base_dir=base_dir)
    if not ok_rep:
        print(f"❌ Evaluation report path security violation: {err_rep}")
        return False

    if not os.path.isfile(full_rep_path):
        print(f"❌ Evaluation report not found: {candidate_report_path}")
        return False

    with open(full_rep_path, "r", encoding="utf-8") as f:
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

    # Recompute evaluation manifest hash from disk with path containment (SEC-04)
    if has_dataset_manifest:
        cand_manifest_path = cand.get("dataset_manifest_path")
        cand_manifest_sha = str(cand.get("dataset_manifest_sha256", "")).strip().lower()
        if not cand_manifest_sha or cand_manifest_sha in ("none", "null", ""):
            rejection_reasons.append("Missing or placeholder 'dataset_manifest_sha256' in candidate evaluation report. Genuine 64-char hex SHA-256 required.")
        elif cand_manifest_path:
            ok_m, full_manifest, err_m = verify_path_containment(cand_manifest_path, base_dir=base_dir)
            if not ok_m:
                rejection_reasons.append(f"Evaluation dataset manifest path security violation: {err_m}")
            elif not os.path.isfile(full_manifest):
                rejection_reasons.append(f"Evaluation dataset manifest not found on disk: '{cand_manifest_path}'")
            else:
                actual_manifest_sha = compute_file_sha256(full_manifest)
                if actual_manifest_sha.lower() != cand_manifest_sha.lower():
                    rejection_reasons.append(
                        f"Evaluation dataset manifest SHA-256 mismatch! Report specifies '{cand_manifest_sha}', but actual file yields '{actual_manifest_sha}'."
                    )

    # Recompute test results summary hash and verify aggregate metrics (SEC-02, SEC-03, SEC-04)
    if has_test_results:
        cand_test_sha = str(cand.get("test_results_summary_sha256", "")).strip().lower()
        if not cand_test_sha or cand_test_sha in ("none", "null", ""):
            rejection_reasons.append("Missing or placeholder 'test_results_summary_sha256' in candidate evaluation report. Genuine 64-char hex SHA-256 required.")
        else:
            test_results = cand.get("test_results")
            if test_results is None and cand.get("test_results_path"):
                tr_path = cand["test_results_path"]
                ok_tr, full_tr_path, err_tr = verify_path_containment(tr_path, base_dir=base_dir)
                if not ok_tr:
                    rejection_reasons.append(f"Test results path security violation: {err_tr}")
                elif not os.path.isfile(full_tr_path):
                    rejection_reasons.append(f"Test results file not found on disk: '{tr_path}'")
                else:
                    try:
                        with open(full_tr_path, "r", encoding="utf-8") as f:
                            test_results = json.load(f)
                    except Exception as e:
                        rejection_reasons.append(f"Failed to read or parse JSON in test_results_path '{tr_path}': {e}")
            
            # SEC-02 & SEC-03: Explicitly reject None, non-list, or empty test_results
            if test_results is None:
                rejection_reasons.append("Missing test results in candidate evaluation report. Either 'test_results' list or valid 'test_results_path' is required.")
            elif not isinstance(test_results, list) or len(test_results) == 0:
                rejection_reasons.append(f"Candidate evaluation 'test_results' must be a non-empty list of test outcome objects (got {type(test_results).__name__} with length {len(test_results) if isinstance(test_results, list) else 0}).")
            else:
                try:
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
                except ValueError as e:
                    rejection_reasons.append(f"Candidate individual test results validation failed: {e}")

    ckpt_path = cand.get("checkpoint_path") or cand.get("checkpoint")
    cand_ckpt_sha = str(cand.get("checkpoint_sha256", "")).strip().lower()

    if not ckpt_path or not str(ckpt_path).strip():
        rejection_reasons.append("Missing required 'checkpoint_path' in candidate evaluation report.")
    else:
        ok_ckpt, full_ckpt, err_ckpt = verify_path_containment(ckpt_path, base_dir=base_dir)
        if not ok_ckpt:
            rejection_reasons.append(f"Checkpoint path security violation: {err_ckpt}")
        elif not os.path.isfile(full_ckpt):
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
