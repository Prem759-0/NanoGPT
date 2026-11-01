"""
evaluation/security/evaluation_attestation.py
Asymmetric Cryptographic Evaluation Provenance & Tamper-Proof Attestation Engine for Moon AI.
Binds candidate model evaluation scores cryptographically to checkpoint SHA-256,
test suite hashes, evaluation manifests, and independent evaluator identities via Ed25519.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EVALUATORS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "security", "trusted_evaluators.json")

EVAL_SUITE_VERSION_CURRENT = "1.0"
EVAL_SOFTWARE_VERSION_CURRENT = "2.0.5"

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
    evaluators_path: Optional[str] = None
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

    # Strict timestamp validation
    try:
        att_dt = datetime.fromisoformat(att_ts.replace("Z", "+00:00"))
        if att_dt.tzinfo is None:
            return False, f"Evaluation attestation timestamp '{att_ts}' lacks explicit UTC timezone."
    except Exception as e:
        return False, f"Malformed evaluation attestation timestamp '{att_ts}': {e}"

    now_utc = datetime.now(timezone.utc)
    if att_dt > (now_utc + timedelta(seconds=60)):
        return False, f"Evaluation attestation timestamp '{att_ts}' is in the future."

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

    # Keystore verification
    trusted_registry = load_trusted_evaluators(evaluators_path)
    if not trusted_registry:
        return False, "Trusted evaluator keystore is empty or unconfigured. Model promotion blocked."

    evaluator_info = trusted_registry.get(ev_id)
    if not evaluator_info:
        return False, f"Unknown or untrusted evaluator_id: '{ev_id}'. Not in trusted evaluators keystore."

    if evaluator_info.get("status") != "ACTIVE":
        return False, f"Evaluator '{ev_id}' is not ACTIVE (status: '{evaluator_info.get('status')}')."

    if evaluator_info.get("algorithm") != "Ed25519":
        return False, f"Evaluator '{ev_id}' algorithm '{evaluator_info.get('algorithm')}' is invalid. Ed25519 required."

    valid_from = evaluator_info.get("valid_from")
    if valid_from:
        try:
            from_dt = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
            if att_dt < from_dt:
                return False, f"Evaluation timestamp '{att_ts}' predates evaluator validity start '{valid_from}'."
        except Exception:
            return False, f"Evaluator '{ev_id}' has malformed valid_from in keystore."

    valid_until = evaluator_info.get("valid_until")
    if valid_until:
        try:
            until_dt = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            if att_dt > until_dt:
                return False, f"Evaluation timestamp '{att_ts}' is after evaluator expiration '{valid_until}'."
            if now_utc > until_dt:
                return False, f"Evaluator '{ev_id}' trust certificate expired on {valid_until}."
        except Exception:
            return False, f"Evaluator '{ev_id}' has malformed valid_until in keystore."

    pub_hex = evaluator_info.get("public_key") or evaluator_info.get("public_key_hex") or evaluator_info.get("verification_key")
    if not pub_hex or len(pub_hex) != 64:
        return False, f"Evaluator '{ev_id}' has invalid public_key_hex in keystore."

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
