"""
evaluation/security/evaluation_attestation.py
Strict Asymmetric Cryptographic Evaluation Provenance & Tamper-Proof Attestation Engine for Moon AI.
Binds candidate model evaluation scores cryptographically to checkpoint SHA-256,
test suite hashes, evaluation manifests, individual test results, and independent evaluator identities via Ed25519.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: RFC 8785 Compatible Canonical JSON Ed25519 Evaluation Attestation v2.0
"""
import os
import sys
import re
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EVALUATORS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "security", "trusted_evaluators.json")

EVAL_SUITE_VERSION_CURRENT = "1.0"
EVAL_SOFTWARE_VERSION_CURRENT = "2.0.6"
MAX_ATTESTATION_AGE_DAYS = 730
MAX_CLOCK_SKEW_SECONDS = 60

def load_trusted_evaluators(evaluators_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads registered trusted evaluators from secure keystore."""
    path = evaluators_path or DEFAULT_EVALUATORS_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("trusted_evaluators", {})
    except Exception:
        return {}

def compute_test_results_summary_sha256(test_results: List[Dict[str, Any]]) -> str:
    """
    Computes deterministic SHA-256 summary of individual evaluation test results.
    Canonical format: RFC 8785 style sorted keys, no whitespace separators, UTF-8.
    """
    if not isinstance(test_results, list) or len(test_results) == 0:
        return "none"
    canonical_lines = []
    for item in test_results:
        if isinstance(item, dict):
            canonical_lines.append(json.dumps(item, sort_keys=True, separators=(',', ':')))
        else:
            canonical_lines.append(str(item))
    payload = "\n".join(canonical_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().lower()

def recompute_aggregate_metrics(test_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Recomputes aggregate benchmark metrics from individual test results.
    Returns: dict with composite_quality_score, non_empty_rate_pct, code_syntax_pct.
    """
    if not isinstance(test_results, list) or len(test_results) == 0:
        return {
            "composite_quality_score": 0.0,
            "non_empty_rate_pct": 0.0,
            "code_syntax_pct": 0.0
        }
    total = len(test_results)
    non_empty_count = sum(
        1 for t in test_results
        if isinstance(t, dict) and bool(str(t.get("output", "") or t.get("response", "") or t.get("text", "")).strip())
    )
    syntax_count = sum(
        1 for t in test_results
        if isinstance(t, dict) and t.get("syntax_valid") is True
    )
    total_score = sum(
        float(t.get("score", 0.0)) for t in test_results
        if isinstance(t, dict) and t.get("score") is not None
    )

    return {
        "composite_quality_score": round(total_score / total, 4),
        "non_empty_rate_pct": round((non_empty_count / total) * 100.0, 4),
        "code_syntax_pct": round((syntax_count / total) * 100.0, 4)
    }

def construct_canonical_evaluation_payload(
    evaluator_id: str,
    checkpoint_sha256: str,
    model_name: str,
    architecture_version: str,
    composite_quality_score: float,
    non_empty_rate_pct: float,
    code_syntax_pct: float,
    dataset_manifest_sha256: str,
    evaluation_suite_version: str,
    evaluation_software_version: str,
    test_results_summary_sha256: str,
    timestamp: str
) -> bytes:
    """
    Constructs deterministic canonical JSON byte payload for cryptographic binding.
    Canonical format: RFC 8785 style sorted keys, no whitespace separators, UTF-8.
    Never uses ambiguous delimiter concatenation.
    """
    canonical_dict = {
        "architecture_version": str(architecture_version).strip(),
        "checkpoint_sha256": str(checkpoint_sha256).strip().lower(),
        "code_syntax_pct": round(float(code_syntax_pct), 4),
        "composite_quality_score": round(float(composite_quality_score), 4),
        "dataset_manifest_sha256": str(dataset_manifest_sha256).strip().lower(),
        "evaluation_software_version": str(evaluation_software_version).strip(),
        "evaluation_suite_version": str(evaluation_suite_version).strip(),
        "evaluator_id": str(evaluator_id).strip(),
        "model_name": str(model_name).strip(),
        "non_empty_rate_pct": round(float(non_empty_rate_pct), 4),
        "test_results_summary_sha256": str(test_results_summary_sha256).strip().lower(),
        "timestamp": str(timestamp).strip()
    }
    return json.dumps(canonical_dict, sort_keys=True, separators=(',', ':')).encode('utf-8')

def sign_evaluation_attestation(
    evaluator_id: str,
    private_key: ed25519.Ed25519PrivateKey,
    checkpoint_sha256: str,
    model_name: str,
    architecture_version: str,
    composite_quality_score: float,
    non_empty_rate_pct: float,
    code_syntax_pct: float,
    dataset_manifest_sha256: str = "none",
    test_results_summary_sha256: str = "none",
    timestamp: Optional[str] = None,
    evaluation_suite_version: str = EVAL_SUITE_VERSION_CURRENT,
    evaluation_software_version: str = EVAL_SOFTWARE_VERSION_CURRENT
) -> Dict[str, Any]:
    """
    Generates an Ed25519 cryptographically signed evaluation attestation bundle.
    Used by authorized independent evaluators to certify model benchmarks.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    payload_bytes = construct_canonical_evaluation_payload(
        evaluator_id=evaluator_id,
        checkpoint_sha256=checkpoint_sha256,
        model_name=model_name,
        architecture_version=architecture_version,
        composite_quality_score=composite_quality_score,
        non_empty_rate_pct=non_empty_rate_pct,
        code_syntax_pct=code_syntax_pct,
        dataset_manifest_sha256=dataset_manifest_sha256,
        evaluation_suite_version=evaluation_suite_version,
        evaluation_software_version=evaluation_software_version,
        test_results_summary_sha256=test_results_summary_sha256,
        timestamp=ts
    )
    sig_bytes = private_key.sign(payload_bytes)

    return {
        "evaluator_id": evaluator_id,
        "checkpoint_sha256": checkpoint_sha256.lower().strip(),
        "model_name": model_name.strip(),
        "architecture_version": architecture_version.strip(),
        "composite_quality_score": round(float(composite_quality_score), 4),
        "non_empty_rate_pct": round(float(non_empty_rate_pct), 4),
        "code_syntax_pct": round(float(code_syntax_pct), 4),
        "dataset_manifest_sha256": dataset_manifest_sha256.lower().strip(),
        "evaluation_suite_version": evaluation_suite_version,
        "evaluation_software_version": evaluation_software_version,
        "test_results_summary_sha256": test_results_summary_sha256.lower().strip(),
        "timestamp": ts,
        "algorithm": "Ed25519",
        "signature": sig_bytes.hex()
    }

def verify_evaluation_attestation(
    report_data: Dict[str, Any],
    evaluators_path: Optional[str] = None,
    enforce_production_hashes: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Cryptographically verifies an evaluation report's attestation against trusted public keys.
    Returns: (is_valid, error_reason)
    """
    att = report_data.get("evaluation_attestation")
    if not att or not isinstance(att, dict):
        return False, "Missing 'evaluation_attestation' dictionary in candidate report. Unsigned evaluation reports are strictly rejected."

    algo = str(att.get("algorithm", "")).strip()
    if algo != "Ed25519":
        return False, f"Evaluation attestation algorithm '{algo}' is strictly rejected. Ed25519 required."

    required_fields = [
        "evaluator_id", "checkpoint_sha256", "model_name", "architecture_version",
        "composite_quality_score", "non_empty_rate_pct", "code_syntax_pct",
        "timestamp", "signature"
    ]
    for f in required_fields:
        if f not in att or att[f] is None or (isinstance(att[f], str) and not att[f].strip()):
            return False, f"Evaluation attestation is missing required field: '{f}'"

    ev_id = str(att["evaluator_id"]).strip()
    att_ckpt_sha = str(att["checkpoint_sha256"]).strip().lower()
    att_name = str(att["model_name"]).strip()
    att_arch = str(att.get("architecture_version", "moon_arch_v1.0")).strip()
    att_score = round(float(att["composite_quality_score"]), 4)
    att_non_empty = round(float(att["non_empty_rate_pct"]), 4)
    att_syntax = round(float(att["code_syntax_pct"]), 4)
    att_dataset_sha = str(att.get("dataset_manifest_sha256", "none")).strip().lower()
    att_suite_ver = str(att.get("evaluation_suite_version", EVAL_SUITE_VERSION_CURRENT)).strip()
    att_soft_ver = str(att.get("evaluation_software_version", EVAL_SOFTWARE_VERSION_CURRENT)).strip()
    att_test_sha = str(att.get("test_results_summary_sha256", "none")).strip().lower()
    att_ts = str(att["timestamp"]).strip()
    att_sig_hex = str(att["signature"]).strip().lower()

    # Signature format and length check (Task 2)
    if len(att_sig_hex) != 128:
        return False, f"Evaluation attestation signature has invalid length ({len(att_sig_hex)} chars; exactly 128 hex chars required)."
    if not re.fullmatch(r"[0-9a-f]{128}", att_sig_hex):
        return False, "Evaluation attestation signature contains invalid non-hex characters."

    # Checkpoint SHA-256 format check (Task 2)
    if len(att_ckpt_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", att_ckpt_sha):
        return False, f"Malformed checkpoint_sha256 in evaluation attestation '{att_ckpt_sha}' (must be 64-char lowercase hex)."

    # Enforce production hash non-placeholder check (Task 2)
    if enforce_production_hashes:
        if att_dataset_sha in ("none", "null", "") or len(att_dataset_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", att_dataset_sha):
            return False, f"Production promotion rejects placeholder or malformed dataset_manifest_sha256 '{att_dataset_sha}'. 64-char hex SHA-256 required."
        if att_test_sha in ("none", "null", "") or len(att_test_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", att_test_sha):
            return False, f"Production promotion rejects placeholder or malformed test_results_summary_sha256 '{att_test_sha}'. 64-char hex SHA-256 required."

    # Strict timestamp validation
    try:
        if not (att_ts.endswith("Z") or "+00:00" in att_ts or att_ts.endswith("+00")):
            return False, f"Evaluation attestation timestamp '{att_ts}' lacks explicit UTC timezone."
        att_dt = datetime.fromisoformat(att_ts.replace("Z", "+00:00"))
        if att_dt.tzinfo is None or att_dt.utcoffset() != timedelta(0):
            return False, f"Evaluation attestation timestamp '{att_ts}' lacks explicit UTC timezone."
    except Exception as e:
        return False, f"Malformed evaluation attestation timestamp '{att_ts}': {e}"

    now_utc = datetime.now(timezone.utc)
    if att_dt > (now_utc + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)):
        return False, f"Evaluation attestation timestamp '{att_ts}' is in the future."

    if (now_utc - att_dt) > timedelta(days=MAX_ATTESTATION_AGE_DAYS):
        return False, f"Evaluation attestation has expired: age ({((now_utc - att_dt).days)} days) exceeds {MAX_ATTESTATION_AGE_DAYS} days limit."

    # Bindings check against candidate report metrics
    rep_ckpt_sha = str(report_data.get("checkpoint_sha256", "")).strip().lower()
    if att_ckpt_sha != rep_ckpt_sha:
        return False, f"Evaluation attestation checkpoint_sha256 mismatch: attestation '{att_ckpt_sha}', report '{rep_ckpt_sha}'."

    rep_name = str(report_data.get("model_name", "")).strip()
    if att_name != rep_name:
        return False, f"Evaluation attestation model_name mismatch: attestation '{att_name}', report '{rep_name}'."

    rep_score = round(float(report_data.get("composite_quality_score", 0.0)), 4)
    if att_score != rep_score:
        return False, f"Evaluation attestation score mismatch (tampered score detected): attestation {att_score}, report {rep_score}."

    rep_non_empty = round(float(report_data.get("non_empty_rate_pct", 0.0)), 4)
    if att_non_empty != rep_non_empty:
        return False, f"Evaluation attestation non_empty_rate_pct mismatch: attestation {att_non_empty}, report {rep_non_empty}."

    rep_syntax = round(float(report_data.get("code_syntax_pct", report_data.get("coding_syntax_valid_pct", 0.0))), 4)
    if att_syntax != rep_syntax:
        return False, f"Evaluation attestation code_syntax_pct mismatch: attestation {att_syntax}, report {rep_syntax}."

    # Verify individual test results if present in report
    if "test_results" in report_data and isinstance(report_data["test_results"], list):
        test_results = report_data["test_results"]
        if len(test_results) > 0:
            computed_test_sha = compute_test_results_summary_sha256(test_results)
            if att_test_sha != "none" and att_test_sha != computed_test_sha:
                return False, f"Individual test results hash mismatch: attestation specifies '{att_test_sha}', but actual test results yield '{computed_test_sha}'."

            recomputed_metrics = recompute_aggregate_metrics(test_results)
            if abs(recomputed_metrics["composite_quality_score"] - rep_score) > 0.01:
                return False, f"Aggregate composite_quality_score {rep_score} does not match individual test results average {recomputed_metrics['composite_quality_score']}."
            if abs(recomputed_metrics["non_empty_rate_pct"] - rep_non_empty) > 0.01:
                return False, f"Aggregate non_empty_rate_pct {rep_non_empty} does not match individual test results {recomputed_metrics['non_empty_rate_pct']}."
            if abs(recomputed_metrics["code_syntax_pct"] - rep_syntax) > 0.01:
                return False, f"Aggregate code_syntax_pct {rep_syntax} does not match individual test results {recomputed_metrics['code_syntax_pct']}."

    # Keystore verification
    trusted_registry = load_trusted_evaluators(evaluators_path)
    if not trusted_registry:
        return False, "Trusted evaluator keystore is empty or unconfigured. Model promotion blocked."

    evaluator_info = trusted_registry.get(ev_id)
    if not evaluator_info:
        return False, f"Unknown or untrusted evaluator_id: '{ev_id}'. Not in trusted evaluators keystore."

    ev_status = str(evaluator_info.get("status", "")).strip().upper()
    if ev_status in ("REVOKED", "SUSPENDED"):
        return False, f"Evaluator '{ev_id}' is {ev_status} and cannot certify candidate models."
    if ev_status != "ACTIVE":
        return False, f"Evaluator '{ev_id}' is not ACTIVE (status: '{ev_status}')."

    if evaluator_info.get("algorithm") != "Ed25519":
        return False, f"Evaluator '{ev_id}' algorithm '{evaluator_info.get('algorithm')}' is invalid. Ed25519 required."

    # Validity window
    valid_from = evaluator_info.get("valid_from")
    if not valid_from or not isinstance(valid_from, str) or not valid_from.strip():
        return False, f"Evaluator '{ev_id}' is missing required 'valid_from' timestamp in keystore."

    valid_until = evaluator_info.get("valid_until")
    if not valid_until or not isinstance(valid_until, str) or not valid_until.strip():
        return False, f"Evaluator '{ev_id}' is missing required 'valid_until' timestamp in keystore."

    try:
        if not (valid_from.endswith("Z") or "+00:00" in valid_from or valid_from.endswith("+00")):
            return False, f"Evaluator '{ev_id}' valid_from timestamp '{valid_from}' lacks explicit UTC timezone."
        from_dt = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
        if from_dt.tzinfo is None or from_dt.utcoffset() != timedelta(0):
            return False, f"Evaluator '{ev_id}' valid_from timestamp '{valid_from}' lacks explicit UTC timezone."
    except Exception as e:
        return False, f"Evaluator '{ev_id}' has malformed valid_from in keystore: {e}"

    try:
        if not (valid_until.endswith("Z") or "+00:00" in valid_until or valid_until.endswith("+00")):
            return False, f"Evaluator '{ev_id}' valid_until timestamp '{valid_until}' lacks explicit UTC timezone."
        until_dt = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
        if until_dt.tzinfo is None or until_dt.utcoffset() != timedelta(0):
            return False, f"Evaluator '{ev_id}' valid_until timestamp '{valid_until}' lacks explicit UTC timezone."
    except Exception as e:
        return False, f"Evaluator '{ev_id}' has malformed valid_until in keystore: {e}"

    if from_dt >= until_dt:
        return False, f"Evaluator '{ev_id}' has inverted validity window: valid_from '{valid_from}' >= valid_until '{valid_until}'."

    if now_utc < from_dt:
        return False, f"Evaluator '{ev_id}' trust certificate is not yet valid (valid_from: '{valid_from}')."

    if now_utc > until_dt:
        return False, f"Evaluator '{ev_id}' trust certificate expired on {valid_until}."

    if att_dt < from_dt:
        return False, f"Evaluation timestamp '{att_ts}' predates evaluator validity start '{valid_from}'."

    if att_dt > until_dt:
        return False, f"Evaluation timestamp '{att_ts}' exceeds evaluator validity end '{valid_until}'."

    pub_hex = evaluator_info.get("public_key") or evaluator_info.get("public_key_hex") or evaluator_info.get("verification_key")
    if not pub_hex or len(pub_hex) != 64 or not re.fullmatch(r"[0-9a-f]{64}", str(pub_hex).lower()):
        return False, f"Evaluator '{ev_id}' has invalid or missing 64-char public_key in keystore."

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
    except Exception as e:
        return False, f"Failed to initialize Ed25519 public key for evaluator '{ev_id}': {e}"

    try:
        sig_bytes = bytes.fromhex(att_sig_hex)
    except Exception as e:
        return False, f"Evaluation attestation signature is not valid hex: {e}"

    payload_bytes = construct_canonical_evaluation_payload(
        evaluator_id=ev_id,
        checkpoint_sha256=att_ckpt_sha,
        model_name=att_name,
        architecture_version=att_arch,
        composite_quality_score=att_score,
        non_empty_rate_pct=att_non_empty,
        code_syntax_pct=att_syntax,
        dataset_manifest_sha256=att_dataset_sha,
        evaluation_suite_version=att_suite_ver,
        evaluation_software_version=att_soft_ver,
        test_results_summary_sha256=att_test_sha,
        timestamp=att_ts
    )

    try:
        public_key.verify(sig_bytes, payload_bytes)
    except InvalidSignature:
        return False, f"Ed25519 evaluation signature verification failed for evaluator '{ev_id}'. Signature forged or altered."
    except Exception as e:
        return False, f"Ed25519 signature verification error: {e}"

    return True, None
