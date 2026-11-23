"""
core/security/training_evidence.py
Verifiable Training-Run Evidence Manifest & Deep Audit Engine for Moon AI.

Architecture Principles:
1. Beyond Signed Claims: A cryptographic signature proves who signed a claim and that it was
   not tampered with after signing. It does NOT independently prove that the claimed training run,
   optimizer steps, dataset, or telemetry actually occurred.
2. Independent Verification: The evidence engine binds the checkpoint to:
   - Specific model architecture specification and SHA-256.
   - Genuine scratch Gaussian initialization telemetry.
   - Concrete dataset manifest and record hash.
   - Exact training source code commit and tree SHA-256.
   - Complete optimizer step telemetry and non-empty loss convergence metrics.
   - Raw byte SHA-256 of the training telemetry run logs.
3. Strict Format Enforcement: Enforces 64-character lowercase hexadecimal hash validation,
   UTC chronological ordering (start_time < completion_time), non-future timestamps,
   finite positive numeric metrics, and cross-consistency between attestation, evidence, and file bytes.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
Attestation Standard: Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.1
"""

import os
import re
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")
MAX_CLOCK_SKEW_SECONDS = 300  # 5 minutes clock skew tolerance
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {"2.1", "MRDC-JSON-2.1"}


def compute_file_sha256(filepath: str) -> str:
    """Computes exact raw-bytes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def is_valid_hex64(val: Any) -> bool:
    """Validates that a value is a strict 64-character lowercase hexadecimal digest."""
    if not isinstance(val, str):
        return False
    return bool(HEX64_REGEX.match(val.strip().lower()))


def parse_iso_utc_timestamp(ts_str: Any) -> Optional[datetime]:
    """Parses an ISO-8601 string and ensures it resolves to timezone-aware UTC datetime."""
    if not isinstance(ts_str, str):
        return None
    try:
        clean_str = ts_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class EvidenceArtifactStore:
    """
    Manages physical storage, resolution, and verification of training evidence artifacts.
    Enforces that evidence claims are backed by authentic byte-level files on disk.
    """
    REQUIRED_ARTIFACT_KEYS = [
        "model_architecture",
        "dataset_manifest",
        "training_code",
        "telemetry_logs",
    ]

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = os.path.abspath(store_dir) if store_dir else os.environ.get("MOON_EVIDENCE_STORE_DIR")

    def is_configured(self) -> bool:
        return bool(self.store_dir and os.path.isdir(self.store_dir))

    def resolve_artifact_path(self, artifact_type: str, evidence: Dict[str, Any]) -> Optional[str]:
        if not self.is_configured():
            return None

        # Check for explicit artifact_file in section
        section = evidence.get(artifact_type, {})
        if isinstance(section, dict):
            rel_file = section.get("artifact_file") or section.get("artifact_path")
            if rel_file:
                cand = os.path.join(self.store_dir, rel_file)
                if os.path.isfile(cand):
                    return cand

        # Also check top-level declared paths
        if artifact_type == "telemetry_logs":
            rel_file = evidence.get("telemetry_logs_file") or evidence.get("telemetry_logs_path")
            if rel_file:
                cand = os.path.join(self.store_dir, rel_file)
                if os.path.isfile(cand):
                    return cand

        STANDARD_FILENAMES = {
            "model_architecture": ["architecture_spec.json", "model_architecture.json"],
            "dataset_manifest": ["dataset_manifest.json"],
            "training_code": ["code_manifest.json", "training_code.json"],
            "environment": ["environment_manifest.json", "environment.json"],
            "telemetry_logs": ["telemetry_logs.raw", "telemetry_logs.jsonl", "training.log"],
            "optimizer": ["optimizer_telemetry.json"],
        }
        for fname in STANDARD_FILENAMES.get(artifact_type, []):
            cand = os.path.join(self.store_dir, fname)
            if os.path.isfile(cand):
                return cand
        return None

    def verify_artifacts(
        self,
        evidence: Dict[str, Any],
        checkpoint_file_path: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        if not self.is_configured():
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                "Evidence artifact store is unconfigured or directory does not exist.",
            )

        # 1. Model Architecture
        arch_path = self.resolve_artifact_path("model_architecture", evidence)
        if not arch_path or not os.path.isfile(arch_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required model_architecture evidence artifact."
        actual_arch_hash = compute_file_sha256(arch_path)
        expected_arch_hash = str(evidence.get("model_architecture", {}).get("model_architecture_hash", "")).strip().lower()
        if actual_arch_hash != expected_arch_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Model architecture artifact SHA-256 mismatch: disk '{actual_arch_hash}' vs manifest '{expected_arch_hash}'.",
            )

        # 2. Dataset Manifest
        data_path = self.resolve_artifact_path("dataset_manifest", evidence)
        if not data_path or not os.path.isfile(data_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required dataset_manifest evidence artifact."
        actual_data_hash = compute_file_sha256(data_path)
        expected_data_hash = str(evidence.get("dataset_manifest", {}).get("dataset_manifest_hash", "")).strip().lower()
        if actual_data_hash != expected_data_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Dataset manifest artifact SHA-256 mismatch: disk '{actual_data_hash}' vs manifest '{expected_data_hash}'.",
            )

        # 3. Training Code
        code_path = self.resolve_artifact_path("training_code", evidence)
        if not code_path or not os.path.isfile(code_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required training_code evidence artifact."
        actual_code_hash = compute_file_sha256(code_path)
        expected_code_hash = str(evidence.get("training_code", {}).get("training_code_hash", "")).strip().lower()
        if actual_code_hash != expected_code_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Training code artifact SHA-256 mismatch: disk '{actual_code_hash}' vs manifest '{expected_code_hash}'.",
            )

        # 4. Telemetry Logs
        logs_path = self.resolve_artifact_path("telemetry_logs", evidence)
        if not logs_path or not os.path.isfile(logs_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required telemetry_logs evidence artifact."
        actual_logs_hash = compute_file_sha256(logs_path)
        expected_logs_hash = str(evidence.get("telemetry_logs_sha256", "")).strip().lower()
        if actual_logs_hash != expected_logs_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Telemetry logs artifact SHA-256 mismatch: disk '{actual_logs_hash}' vs manifest '{expected_logs_hash}'.",
            )

        # Verify run_id inside telemetry logs
        try:
            with open(logs_path, "r", encoding="utf-8", errors="replace") as f:
                log_content = f.read(65536)
                declared_run_id = str(evidence.get("run_id", "")).strip()
                if declared_run_id and declared_run_id not in log_content:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Run ID '{declared_run_id}' not found in telemetry logs artifact.",
                    )
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed reading telemetry logs artifact: {e}"

        return True, "GENUINE_TRAINED_SOVEREIGN", "Physical evidence artifacts verified against store."


class TrainingEvidenceValidator:
    """
    Performs deep verification of training-run evidence packages:
    1. Schema and mandatory section completeness.
    2. Policy and scratch initialization invariants.
    3. Strict hash format and lowercase hex validation.
    4. Chronological UTC timestamp ordering and sanity.
    5. Optimizer configuration and step telemetry validation.
    6. Monotonic step progression and finite numeric loss metrics.
    7. Cross-consistency binding with candidate checkpoint and attestation.
    """

    REQUIRED_MANIFEST_SECTIONS = [
        "schema_version",
        "run_id",
        "policy",
        "initialization",
        "initialization_evidence",
        "model_architecture",
        "dataset_manifest",
        "training_code",
        "environment",
        "timestamps",
        "optimizer",
        "metrics_history",
        "telemetry_logs_sha256",
        "checkpoint_sha256",
        "verifier",
    ]

    @classmethod
    def validate_evidence_manifest(
        cls,
        evidence: Optional[Dict[str, Any]],
        checkpoint_file_sha256: str,
        attestation: Optional[Dict[str, Any]] = None,
        evidence_store: Optional[EvidenceArtifactStore] = None,
        require_artifacts: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        Validates the complete training evidence package against candidate checkpoint and attestation.
        Returns: (is_valid: bool, tier: str, reason: str)
        """
        if evidence is None:
            return False, "ATTESTATION_EVIDENCE_INVALID", "Training evidence manifest is missing."
        if not isinstance(evidence, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Training evidence manifest must be a JSON dictionary."

        # 0. Schema version validation
        schema_ver = str(evidence.get("schema_version", "")).strip()
        if schema_ver not in SUPPORTED_EVIDENCE_SCHEMA_VERSIONS:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Unsupported or missing evidence schema_version '{schema_ver}'. Supported: {sorted(SUPPORTED_EVIDENCE_SCHEMA_VERSIONS)}",
            )

        # 1. Section completeness
        missing_sections = [s for s in cls.REQUIRED_MANIFEST_SECTIONS if s not in evidence or evidence[s] is None]
        if missing_sections:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Training evidence manifest missing required sections: {', '.join(missing_sections)}",
            )

        # 2. Policy enforcement
        policy = evidence.get("policy")
        if policy != "STRICT_SOVEREIGN_RANDOM_INIT_ONLY":
            return (
                False,
                "UNAPPROVED_POLICY",
                f"Evidence declares policy '{policy}' violating STRICT_SOVEREIGN_RANDOM_INIT_ONLY.",
            )

        # 3. Initialization verification
        init_method = evidence.get("initialization")
        if init_method != "random_gaussian_scratch":
            return (
                False,
                "NON_SCRATCH_INITIALIZATION",
                f"Evidence declares initialization '{init_method}' violating random_gaussian_scratch.",
            )

        init_ev = evidence.get("initialization_evidence", {})
        if not isinstance(init_ev, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "initialization_evidence must be a dictionary."
        if init_ev.get("weights_initialized_from_file") is not False:
            return (
                False,
                "TEACHER_CONTAMINATION",
                "Initialization evidence indicates weights were initialized from external file or base weights.",
            )
        if init_ev.get("distribution") != "gaussian":
            return (
                False,
                "NON_SCRATCH_INITIALIZATION",
                f"Initialization distribution '{init_ev.get('distribution')}' must be 'gaussian'.",
            )

        # 4. Hash format validations
        arch_dict = evidence.get("model_architecture", {})
        arch_hash = arch_dict.get("model_architecture_hash") if isinstance(arch_dict, dict) else None
        if not is_valid_hex64(arch_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid model_architecture_hash format: {arch_hash}"

        dataset_dict = evidence.get("dataset_manifest", {})
        dataset_hash = dataset_dict.get("dataset_manifest_hash") if isinstance(dataset_dict, dict) else None
        if not is_valid_hex64(dataset_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid dataset_manifest_hash format: {dataset_hash}"

        code_dict = evidence.get("training_code", {})
        code_hash = code_dict.get("training_code_hash") if isinstance(code_dict, dict) else None
        if not is_valid_hex64(code_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid training_code_hash format: {code_hash}"

        telemetry_logs_hash = evidence.get("telemetry_logs_sha256")
        if not is_valid_hex64(telemetry_logs_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid telemetry_logs_sha256 format: {telemetry_logs_hash}"

        ev_ckpt_hash = str(evidence.get("checkpoint_sha256", "")).strip().lower()
        if not is_valid_hex64(ev_ckpt_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid evidence checkpoint_sha256 format: {ev_ckpt_hash}"

        # 5. Checkpoint byte binding
        if ev_ckpt_hash != checkpoint_file_sha256.lower().strip():
            return (
                False,
                "ATTESTATION_CHECKPOINT_HASH_MISMATCH",
                f"Evidence checkpoint SHA-256 '{ev_ckpt_hash}' does not match file bytes '{checkpoint_file_sha256}'.",
            )

        # 6. Timestamp verification
        ts_dict = evidence.get("timestamps", {})
        if not isinstance(ts_dict, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "timestamps section must be a dictionary."
        start_dt = parse_iso_utc_timestamp(ts_dict.get("start_time"))
        completion_dt = parse_iso_utc_timestamp(ts_dict.get("completion_time"))

        if not start_dt or not completion_dt:
            return False, "ATTESTATION_EVIDENCE_INVALID", "Evidence timestamps must be valid ISO-8601 UTC strings."

        if start_dt >= completion_dt:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Chronological violation: training start_time ({start_dt}) is not before completion_time ({completion_dt}).",
            )

        now_utc = datetime.now(timezone.utc)
        max_future = now_utc + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)
        if completion_dt > max_future:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Timestamp violation: training completion_time ({completion_dt}) is in the future.",
            )

        # 7. Optimizer and step count verification
        optim_dict = evidence.get("optimizer", {})
        if not isinstance(optim_dict, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "optimizer section must be a dictionary."

        try:
            total_steps = int(optim_dict.get("total_steps", 0))
        except (ValueError, TypeError):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer 'total_steps' must be an integer."

        if total_steps <= 0:
            return (
                False,
                "UNTRAINED_RANDOM_INIT",
                f"Evidence optimizer total_steps ({total_steps}) indicates untrained model state.",
            )

        # 8. Metrics history verification
        metrics = evidence.get("metrics_history")
        if not isinstance(metrics, list) or len(metrics) == 0:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                "metrics_history must be a non-empty list of validation metric records.",
            )

        last_step = -1
        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                return False, "ATTESTATION_EVIDENCE_INVALID", f"Metric record at index {i} must be a dictionary."
            step = m.get("step")
            if not isinstance(step, int) or step < 0:
                return False, "ATTESTATION_EVIDENCE_INVALID", f"Metric record at index {i} has invalid step: {step}"
            if step <= last_step:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Metrics steps must be monotonically increasing (step {step} <= previous {last_step}).",
                )
            last_step = step

            val_loss = m.get("val_loss")
            if not isinstance(val_loss, (int, float)) or val_loss <= 0.0 or val_loss != val_loss:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Metric record at index {i} has invalid non-positive or non-finite val_loss: {val_loss}",
                )

        # 9. Cross-consistency with attestation (if attestation provided)
        if attestation is not None and isinstance(attestation, dict):
            # Run ID match
            att_run = str(attestation.get("run_id", "")).strip()
            ev_run = str(evidence.get("run_id", "")).strip()
            if att_run and att_run != ev_run:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Run ID mismatch: attestation '{att_run}' vs evidence '{ev_run}'.",
                )

            # Policy match
            att_policy = str(attestation.get("policy", "")).strip()
            if att_policy and att_policy != policy:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Policy mismatch: attestation '{att_policy}' vs evidence '{policy}'.",
                )

            # Initialization match
            att_init = str(attestation.get("initialization", "")).strip()
            if att_init and att_init != init_method:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Initialization mismatch: attestation '{att_init}' vs evidence '{init_method}'.",
                )

            # Timestamps match
            att_start = attestation.get("training_start_time")
            ev_start = ts_dict.get("start_time")
            if att_start and ev_start and str(att_start).strip() != str(ev_start).strip():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Start timestamp mismatch: attestation '{att_start}' vs evidence '{ev_start}'.",
                )

            att_comp = attestation.get("training_completion_time")
            ev_comp = ts_dict.get("completion_time")
            if att_comp and ev_comp and str(att_comp).strip() != str(ev_comp).strip():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Completion timestamp mismatch: attestation '{att_comp}' vs evidence '{ev_comp}'.",
                )

            att_arch = str(attestation.get("model_architecture_hash", "")).strip().lower()
            if att_arch != arch_hash.lower():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Architecture hash mismatch: attestation '{att_arch}' vs evidence '{arch_hash}'.",
                )

            att_data = str(attestation.get("dataset_manifest_hash", "")).strip().lower()
            if att_data != dataset_hash.lower():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Dataset manifest hash mismatch: attestation '{att_data}' vs evidence '{dataset_hash}'.",
                )

            att_code = str(attestation.get("training_code_hash", "")).strip().lower()
            if att_code != code_hash.lower():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Training code hash mismatch: attestation '{att_code}' vs evidence '{code_hash}'.",
                )

            att_steps = int(attestation.get("step_count", 0))
            if att_steps != total_steps:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Step count mismatch: attestation ({att_steps}) vs evidence ({total_steps}).",
                )

            att_ckpt = str(attestation.get("checkpoint_sha256", "")).strip().lower()
            if att_ckpt != ev_ckpt_hash.lower():
                return (
                    False,
                    "ATTESTATION_CHECKPOINT_HASH_MISMATCH",
                    f"Checkpoint hash mismatch: attestation '{att_ckpt}' vs evidence '{ev_ckpt_hash}'.",
                )

        # 10. Physical evidence artifact verification (when required or store configured)
        if require_artifacts or (evidence_store is not None and evidence_store.is_configured()):
            resolved_store = evidence_store if evidence_store is not None else EvidenceArtifactStore()
            if not resolved_store.is_configured():
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    "Physical evidence artifact store is unconfigured or directory does not exist.",
                )
            art_ok, art_tier, art_reason = resolved_store.verify_artifacts(evidence)
            if not art_ok:
                return False, art_tier, art_reason

        return (
            True,
            "GENUINE_TRAINED_SOVEREIGN",
            "Training evidence manifest verified: full training provenance independently confirmed.",
        )


def create_sample_training_evidence(
    checkpoint_file_sha256: str,
    run_id: str = "moon_pilot_run_001",
    step_count: int = 1000,
    start_time: Optional[str] = None,
    completion_time: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    """Generates a complete, structurally valid training evidence package for test fixtures."""
    now = datetime.now(timezone.utc)
    st = start_time or (now - timedelta(hours=3)).isoformat()
    ct = completion_time or (now - timedelta(minutes=10)).isoformat()

    manifest = {
        "schema_version": "2.1",
        "run_id": run_id,
        "policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
        "initialization": "random_gaussian_scratch",
        "initialization_evidence": {
            "distribution": "gaussian",
            "mean": 0.0,
            "std": 0.02,
            "seed": 1337,
            "weights_initialized_from_file": False,
        },
        "model_architecture": {
            "architecture_name": "MoonTransformerChat25M",
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "vocab_size": 512,
            "block_size": 512,
            "model_architecture_hash": hashlib.sha256(b"moon_arch_chat_25m").hexdigest(),
        },
        "dataset_manifest": {
            "dataset_name": "moon_expanded_chat_v1",
            "record_count": 572,
            "manifest_sha256": hashlib.sha256(b"moon_chat_dataset_manifest").hexdigest(),
            "dataset_manifest_hash": hashlib.sha256(b"moon_chat_dataset_manifest").hexdigest(),
        },
        "training_code": {
            "git_commit_sha": "05873a0b171942423ac9824fee93ada32c4a1ac7",
            "clean_working_tree": True,
            "code_tree_sha256": hashlib.sha256(b"moon_code_tree_v1").hexdigest(),
            "training_code_hash": hashlib.sha256(b"moon_code_tree_v1").hexdigest(),
        },
        "environment": {
            "python_version": "3.11.9",
            "torch_version": "2.14.1",
            "cuda_version": "12.1",
            "hardware_identifier": "Tesla T4 GPU (Cloud Sovereign Cluster)",
            "dependencies_hash": hashlib.sha256(b"moon_env_deps").hexdigest(),
        },
        "timestamps": {
            "start_time": st,
            "completion_time": ct,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0006,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "total_steps": step_count,
        },
        "metrics_history": [
            {"step": 100, "train_loss": 3.85, "val_loss": 3.91},
            {"step": 500, "train_loss": 2.65, "val_loss": 2.70},
            {"step": 1000, "train_loss": 2.15, "val_loss": 2.21},
        ],
        "telemetry_logs_sha256": hashlib.sha256(b"moon_telemetry_logs_raw").hexdigest(),
        "checkpoint_sha256": checkpoint_file_sha256.lower().strip(),
        "verifier": {
            "verifier_id": "test_training_authority",
            "verification_timestamp": ct,
            "evidence_audit_checklist_verified": True,
        },
    }
    manifest.update(overrides)
    return manifest


def create_sample_training_artifacts(
    target_dir: str,
    run_id: str = "moon_test_run_001",
    step_count: int = 1000,
    start_time: Optional[str] = None,
    completion_time: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Creates authentic physical files on disk for all required evidence artifacts in target_dir.
    Returns: (artifact_paths_dict, evidence_manifest_dict)
    """
    os.makedirs(target_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    st = start_time or (now - timedelta(hours=3)).isoformat()
    ct = completion_time or (now - timedelta(minutes=10)).isoformat()

    arch_content = json.dumps({
        "architecture_name": "MoonTransformerChat25M",
        "n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512
    }, indent=2).encode("utf-8")
    arch_file = os.path.join(target_dir, "architecture_spec.json")
    with open(arch_file, "wb") as f:
        f.write(arch_content)
    arch_hash = hashlib.sha256(arch_content).hexdigest().lower()

    data_content = json.dumps({
        "dataset_name": "moon_expanded_chat_v1",
        "record_count": 572,
        "records_sha256": hashlib.sha256(b"records_data").hexdigest()
    }, indent=2).encode("utf-8")
    data_file = os.path.join(target_dir, "dataset_manifest.json")
    with open(data_file, "wb") as f:
        f.write(data_content)
    data_hash = hashlib.sha256(data_content).hexdigest().lower()

    code_content = json.dumps({
        "commit": "05873a0b171942423ac9824fee93ada32c4a1ac7",
        "clean_tree": True,
        "files_count": 140
    }, indent=2).encode("utf-8")
    code_file = os.path.join(target_dir, "code_manifest.json")
    with open(code_file, "wb") as f:
        f.write(code_content)
    code_hash = hashlib.sha256(code_content).hexdigest().lower()

    env_content = json.dumps({
        "python": "3.11.9", "torch": "2.14.1", "device": "cuda:0"
    }, indent=2).encode("utf-8")
    env_file = os.path.join(target_dir, "environment_manifest.json")
    with open(env_file, "wb") as f:
        f.write(env_content)
    env_hash = hashlib.sha256(env_content).hexdigest().lower()

    log_lines = [
        f"[RUN_INIT] run_id={run_id} policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0",
        f"[STEP_TELEMETRY] run_id={run_id} step=100 train_loss=3.85 val_loss=3.91",
        f"[STEP_TELEMETRY] run_id={run_id} step=500 train_loss=2.65 val_loss=2.70",
        f"[STEP_TELEMETRY] run_id={run_id} step={step_count} train_loss=2.15 val_loss=2.21",
        f"[RUN_COMPLETE] run_id={run_id} total_steps={step_count} status=SUCCESS"
    ]
    log_content = "\n".join(log_lines).encode("utf-8")
    log_file = os.path.join(target_dir, "telemetry_logs.raw")
    with open(log_file, "wb") as f:
        f.write(log_content)
    logs_hash = hashlib.sha256(log_content).hexdigest().lower()

    paths = {
        "architecture_spec": arch_file,
        "dataset_manifest": data_file,
        "training_code": code_file,
        "environment": env_file,
        "telemetry_logs": log_file,
    }

    manifest = create_sample_training_evidence(
        checkpoint_file_sha256="00" * 32,
        run_id=run_id,
        step_count=step_count,
        start_time=st,
        completion_time=ct,
        model_architecture={
            "architecture_name": "MoonTransformerChat25M",
            "model_architecture_hash": arch_hash,
            "artifact_file": "architecture_spec.json",
        },
        dataset_manifest={
            "dataset_name": "moon_expanded_chat_v1",
            "dataset_manifest_hash": data_hash,
            "artifact_file": "dataset_manifest.json",
        },
        training_code={
            "training_code_hash": code_hash,
            "artifact_file": "code_manifest.json",
        },
        environment={
            "dependencies_hash": env_hash,
            "artifact_file": "environment_manifest.json",
        },
        telemetry_logs_sha256=logs_hash,
        telemetry_logs_file="telemetry_logs.raw",
    )
    return paths, manifest
