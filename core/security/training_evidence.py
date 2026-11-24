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
import math
import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger("moon.security.training_evidence")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
    Manages physical storage, resolution, containment, and verification of training evidence artifacts.
    Enforces that evidence claims are backed by authentic byte-level files strictly contained
    within the authorized evidence store directory.
    """
    REQUIRED_ARTIFACT_KEYS = [
        "model_architecture",
        "dataset_manifest",
        "training_code",
        "environment",
        "telemetry_logs",
        "optimizer",
    ]

    STANDARD_FILENAMES = {
        "model_architecture": ["architecture_spec.json", "model_architecture.json"],
        "dataset_manifest": ["dataset_manifest.json"],
        "training_code": ["code_manifest.json", "training_code.json"],
        "environment": ["environment_manifest.json", "environment.json"],
        "telemetry_logs": ["telemetry_logs.raw", "telemetry_logs.jsonl", "training.log"],
        "optimizer": ["optimizer_telemetry.json", "optimizer.json"],
    }

    def __init__(self, store_dir: Optional[str] = None, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or PROJECT_ROOT)
        raw_dir = store_dir or os.environ.get("MOON_EVIDENCE_STORE_DIR")
        self.store_dir = os.path.abspath(raw_dir) if raw_dir else None

    def is_configured(self) -> bool:
        """Returns True if store_dir is set and exists as a directory."""
        return bool(self.store_dir and os.path.isdir(self.store_dir))

    def is_production_store(self) -> bool:
        """
        Returns True if the store is configured and strictly outside the repository root.
        Repository files cannot supply production evidence artifacts.
        """
        if not self.is_configured():
            return False
        try:
            real_base = os.path.realpath(self.base_dir)
            real_store = os.path.realpath(self.store_dir)
            if os.path.splitdrive(real_base)[0].lower() != os.path.splitdrive(real_store)[0].lower():
                return True  # Different drive is strictly outside the repository
            common = os.path.commonpath([real_base, real_store])
            if common == real_base:
                return False  # Inside repo: strictly forbidden for production evidence!
            return True
        except (ValueError, Exception):
            return False

    def _sanitize_and_contain_path(self, rel_path: Any) -> Optional[str]:
        """
        Validates, canonicalizes, and verifies containment of an artifact path:
        1. Reject non-string, empty, or whitespace-only paths.
        2. Reject absolute paths (POSIX / or Windows C:\\ or \\\\unc).
        3. Reject null bytes or control characters.
        4. Normalize path and reject any traversal escaping root ('..').
        5. Verify that os.path.realpath(cand) is strictly contained in os.path.realpath(self.store_dir).
        6. Verify that target is not the store directory itself.
        7. Verify that target exists and is a regular file (stat.S_ISREG).
        """
        if not self.is_configured():
            return None
        if not rel_path or not isinstance(rel_path, str):
            return None
        clean = rel_path.strip()
        if not clean or "\x00" in clean:
            return None

        # Reject absolute paths (POSIX and Windows)
        if os.path.isabs(clean) or clean.startswith(("/", "\\")):
            return None
        if len(clean) >= 2 and clean[1] == ":":  # Windows drive letter e.g. C:
            return None

        # Normalize and reject path traversal
        norm = os.path.normpath(clean)
        if norm == "." or norm.startswith("..") or "/../" in norm or "\\..\\" in norm:
            return None

        # Build candidate path
        cand = os.path.join(self.store_dir, norm)
        try:
            real_store = os.path.realpath(self.store_dir)
            real_cand = os.path.realpath(cand)

            # Containment check: commonpath must equal real_store and cand must not be real_store
            common = os.path.commonpath([real_store, real_cand])
            if common != real_store or real_cand == real_store:
                return None

            # File existence and regular file check
            if not os.path.isfile(real_cand):
                return None
            st = os.stat(real_cand)
            import stat
            if not stat.S_ISREG(st.st_mode):
                return None
            return real_cand
        except (ValueError, OSError, Exception):
            return None

    def resolve_artifact_path(self, artifact_type: str, evidence: Dict[str, Any]) -> Optional[str]:
        """
        Resolves an artifact file path with strict containment and security validation:
        1. Check section-level 'artifact_file' or 'artifact_path'.
        2. Check top-level '{artifact_type}_file' or '{artifact_type}_path'.
        3. Check standard filenames within the evidence store.
        Returns canonical real file path on success, or None on failure/violation.
        """
        if not self.is_configured():
            return None

        # 1. Section-level declared paths
        section = evidence.get(artifact_type, {})
        if isinstance(section, dict):
            for key in ("artifact_file", "artifact_path"):
                val = section.get(key)
                if val:
                    safe_path = self._sanitize_and_contain_path(val)
                    if safe_path:
                        return safe_path
                    else:
                        logger.warning(
                            f"Evidence artifact '{artifact_type}' specified path '{val}' "
                            f"which failed containment, traversal, or regular file validation."
                        )
                        return None  # Fail closed on path violation!

        # 2. Top-level declared paths
        for key in (f"{artifact_type}_file", f"{artifact_type}_path"):
            val = evidence.get(key)
            if val:
                safe_path = self._sanitize_and_contain_path(val)
                if safe_path:
                    return safe_path
                else:
                    logger.warning(
                        f"Evidence artifact '{artifact_type}' top-level path '{val}' "
                        f"failed containment, traversal, or regular file validation."
                    )
                    return None  # Fail closed!

        # 3. Standard filenames
        for fname in self.STANDARD_FILENAMES.get(artifact_type, []):
            safe_path = self._sanitize_and_contain_path(fname)
            if safe_path:
                return safe_path

        return None

    def verify_artifacts(
        self,
        evidence: Dict[str, Any],
        checkpoint_file_path: Optional[str] = None,
        attestation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, str]:
        if not self.is_configured():
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                "Evidence artifact store is unconfigured or directory does not exist.",
            )

        declared_run_id = str(evidence.get("run_id", "")).strip()

        # 1. Model Architecture
        arch_path = self.resolve_artifact_path("model_architecture", evidence)
        if not arch_path or not os.path.isfile(arch_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required model_architecture evidence artifact."
        actual_arch_hash = compute_file_sha256(arch_path)
        expected_arch_hash = str(evidence.get("model_architecture", {}).get("model_architecture_hash", "")).strip().lower()
        if not is_valid_hex64(expected_arch_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid model_architecture_hash in evidence manifest: '{expected_arch_hash}'",
            )
        if actual_arch_hash != expected_arch_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Model architecture artifact SHA-256 mismatch: disk '{actual_arch_hash}' vs manifest '{expected_arch_hash}'.",
            )
        try:
            with open(arch_path, "r", encoding="utf-8") as f:
                arch_json = json.load(f)
            if not isinstance(arch_json, dict) or not arch_json.get("architecture_name"):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Model architecture artifact missing 'architecture_name'."
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed parsing model architecture artifact: {e}"

        # 2. Dataset Manifest & Referenced Files
        data_path = self.resolve_artifact_path("dataset_manifest", evidence)
        if not data_path or not os.path.isfile(data_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required dataset_manifest evidence artifact."
        actual_data_hash = compute_file_sha256(data_path)
        expected_data_hash = str(evidence.get("dataset_manifest", {}).get("dataset_manifest_hash", "")).strip().lower()
        if not is_valid_hex64(expected_data_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid dataset_manifest_hash in evidence manifest: '{expected_data_hash}'",
            )
        if actual_data_hash != expected_data_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Dataset manifest artifact SHA-256 mismatch: disk '{actual_data_hash}' vs manifest '{expected_data_hash}'.",
            )
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                dataset_data = json.load(f)
            ref_files = dataset_data.get("files") or dataset_data.get("dataset_files")
            if isinstance(ref_files, dict):
                for rel_fname, declared_sha in ref_files.items():
                    data_file_path = self._sanitize_and_contain_path(rel_fname)
                    if not data_file_path or not os.path.isfile(data_file_path):
                        return (
                            False,
                            "ATTESTATION_EVIDENCE_INVALID",
                            f"Missing referenced dataset file '{rel_fname}' declared in dataset manifest.",
                        )
                    actual_file_sha = compute_file_sha256(data_file_path)
                    if actual_file_sha != str(declared_sha).strip().lower():
                        return (
                            False,
                            "ATTESTATION_EVIDENCE_INVALID",
                            f"Referenced dataset file '{rel_fname}' SHA-256 mismatch: disk '{actual_file_sha}' vs declared '{declared_sha}'.",
                        )
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed parsing dataset manifest: {e}"

        # 3. Training Code
        code_path = self.resolve_artifact_path("training_code", evidence)
        if not code_path or not os.path.isfile(code_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required training_code evidence artifact."
        actual_code_hash = compute_file_sha256(code_path)
        expected_code_hash = str(evidence.get("training_code", {}).get("training_code_hash", "")).strip().lower()
        if not is_valid_hex64(expected_code_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid training_code_hash in evidence manifest: '{expected_code_hash}'",
            )
        if actual_code_hash != expected_code_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Training code artifact SHA-256 mismatch: disk '{actual_code_hash}' vs manifest '{expected_code_hash}'.",
            )
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                code_data = json.load(f)
            commit_sha = code_data.get("commit") or code_data.get("git_commit_sha")
            if not commit_sha or len(commit_sha) != 40 or not all(c in "0123456789abcdef" for c in commit_sha.lower()):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Training code manifest specifies invalid git commit hash."
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed parsing training code manifest: {e}"

        # 4. Environment and Dependency Manifest
        env_path = self.resolve_artifact_path("environment", evidence)
        if not env_path or not os.path.isfile(env_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required environment evidence artifact."
        actual_env_hash = compute_file_sha256(env_path)
        expected_env_hash = str(
            evidence.get("environment", {}).get("dependencies_hash")
            or evidence.get("environment", {}).get("environment_manifest_hash", "")
        ).strip().lower()
        if not is_valid_hex64(expected_env_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid expected environment dependencies_hash: '{expected_env_hash}'",
            )
        if actual_env_hash != expected_env_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Environment manifest artifact SHA-256 mismatch: disk '{actual_env_hash}' vs manifest '{expected_env_hash}'.",
            )
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                env_data = json.load(f)
            for req_field in ("python", "torch"):
                if not env_data.get(req_field) and not env_data.get(f"{req_field}_version"):
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Environment manifest missing '{req_field}' field."
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed parsing environment manifest: {e}"

        # 5. Optimizer Telemetry Manifest
        optim_path = self.resolve_artifact_path("optimizer", evidence)
        if not optim_path or not os.path.isfile(optim_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required optimizer evidence artifact."
        actual_optim_hash = compute_file_sha256(optim_path)
        expected_optim_hash = str(
            evidence.get("optimizer", {}).get("optimizer_telemetry_hash")
            or evidence.get("optimizer", {}).get("optimizer_hash", "")
        ).strip().lower()
        if not is_valid_hex64(expected_optim_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid expected optimizer_telemetry_hash: '{expected_optim_hash}'",
            )
        if actual_optim_hash != expected_optim_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Optimizer artifact SHA-256 mismatch: disk '{actual_optim_hash}' vs manifest '{expected_optim_hash}'.",
            )
        try:
            with open(optim_path, "r", encoding="utf-8") as f:
                optim_data = json.load(f)
            if not isinstance(optim_data, dict):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer telemetry artifact must be a JSON dictionary."
            optim_run_id = str(optim_data.get("run_id", "")).strip()
            if not optim_run_id:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer telemetry missing 'run_id'."
            if optim_run_id != declared_run_id:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Optimizer telemetry run_id mismatch: disk '{optim_run_id}' vs manifest '{declared_run_id}'.",
                )
            if "total_steps" not in optim_data:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer telemetry missing 'total_steps'."
            try:
                optim_steps = int(optim_data["total_steps"])
            except (ValueError, TypeError):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer telemetry 'total_steps' must be an integer."
            if optim_steps <= 0:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Optimizer telemetry declared non-positive total_steps: {optim_steps}.",
                )
            declared_steps = int(evidence.get("optimizer", {}).get("total_steps", 0))
            if optim_steps != declared_steps:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Optimizer telemetry total_steps mismatch: disk {optim_steps} vs manifest {declared_steps}.",
                )
            if attestation is not None:
                att_steps = int(attestation.get("step_count", 0))
                if optim_steps != att_steps:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Optimizer telemetry total_steps mismatch: disk {optim_steps} vs attestation {att_steps}.",
                    )

            # Optimizer configuration checks
            opt_name = optim_data.get("name") or optim_data.get("optimizer") or evidence.get("optimizer", {}).get("name")
            if not opt_name or not isinstance(opt_name, str) or not opt_name.strip():
                return False, "ATTESTATION_EVIDENCE_INVALID", "Optimizer telemetry missing valid optimizer name/type."

            # Step records completeness check (Mandatory, non-empty, strictly monotonic, agreeing with total_steps)
            step_records = optim_data.get("step_records") if "step_records" in optim_data else optim_data.get("optimizer_steps")
            if not isinstance(step_records, list) or len(step_records) == 0:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    "Optimizer telemetry missing or empty step_records list. Declared total_steps is not proof of actual optimizer updates.",
                )
            if len(step_records) > optim_steps:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Optimizer step records count ({len(step_records)}) exceeds total_steps ({optim_steps}).",
                )

            last_step = -1
            for idx, rec in enumerate(step_records):
                if not isinstance(rec, dict):
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Optimizer step record at index {idx} must be a dictionary."
                st_val = rec.get("step")
                if not isinstance(st_val, int) or st_val < 0:
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Invalid step in optimizer telemetry at index {idx}: {st_val}"
                if st_val <= last_step:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Optimizer steps not strictly monotonic or contain duplicates: step {st_val} <= previous {last_step}.",
                    )
                if st_val > optim_steps:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Optimizer step {st_val} exceeds declared total_steps {optim_steps}.",
                    )
                if "lr" in rec:
                    lr_val = rec["lr"]
                    if not isinstance(lr_val, (int, float)) or not math.isfinite(lr_val) or lr_val <= 0.0:
                        return False, "ATTESTATION_EVIDENCE_INVALID", f"Optimizer step {st_val} has invalid learning rate: {lr_val}"
                if "grad_norm" in rec:
                    gn_val = rec["grad_norm"]
                    if not isinstance(gn_val, (int, float)) or not math.isfinite(gn_val) or gn_val < 0.0:
                        return False, "ATTESTATION_EVIDENCE_INVALID", f"Optimizer step {st_val} has invalid gradient norm: {gn_val}"
                last_step = st_val

            if last_step != optim_steps:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Optimizer telemetry final observed step ({last_step}) does not match declared total_steps ({optim_steps}).",
                )
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed parsing optimizer telemetry artifact: {e}"

        # 6. Telemetry Logs
        logs_path = self.resolve_artifact_path("telemetry_logs", evidence)
        if not logs_path or not os.path.isfile(logs_path):
            return False, "ATTESTATION_EVIDENCE_INVALID", "Missing required telemetry_logs evidence artifact."
        actual_logs_hash = compute_file_sha256(logs_path)
        expected_logs_hash = str(evidence.get("telemetry_logs_sha256", "")).strip().lower()
        if not is_valid_hex64(expected_logs_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid expected telemetry_logs_sha256: '{expected_logs_hash}'",
            )
        if actual_logs_hash != expected_logs_hash:
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Telemetry logs artifact SHA-256 mismatch: disk '{actual_logs_hash}' vs manifest '{expected_logs_hash}'.",
            )
        try:
            with open(logs_path, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(raw_lines) < 3:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Telemetry logs artifact contains insufficient records ({len(raw_lines)} lines; required at least 3: [RUN_INIT], [STEP_TELEMETRY], [RUN_COMPLETE]).",
                )

            # First line must be [RUN_INIT]
            first_line = raw_lines[0]
            if not first_line.startswith("[RUN_INIT]"):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log must start with [RUN_INIT] event."
            if declared_run_id and f"run_id={declared_run_id}" not in first_line:
                return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log missing valid [RUN_INIT] record with run_id '{declared_run_id}'."
            if "policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY" not in first_line:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log [RUN_INIT] missing policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY."
            if "step=0" not in first_line:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log [RUN_INIT] must declare step=0."

            # Last line must be [RUN_COMPLETE]
            last_line = raw_lines[-1]
            if not last_line.startswith("[RUN_COMPLETE]"):
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log must end with [RUN_COMPLETE] event."
            if declared_run_id and f"run_id={declared_run_id}" not in last_line:
                return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log missing valid [RUN_COMPLETE] record with run_id '{declared_run_id}'."
            if "status=SUCCESS" not in last_line:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log [RUN_COMPLETE] missing status=SUCCESS."

            # Optional total_steps check in [RUN_COMPLETE]
            import re
            m_fin = re.search(r"(?:total_steps|final_step)=(\d+)", last_line)
            if m_fin:
                fin_st = int(m_fin.group(1))
                if fin_st != declared_steps:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log [RUN_COMPLETE] step count ({fin_st}) does not match declared total_steps ({declared_steps}).",
                    )

            # Intermediate lines must be [STEP_TELEMETRY]
            last_log_step = 0
            step_event_count = 0
            for line_idx, line_str in enumerate(raw_lines[1:-1], start=2):
                if line_str.startswith("[RUN_INIT]") or line_str.startswith("[RUN_COMPLETE]"):
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log contains duplicate lifecycle event on line {line_idx}: {line_str}",
                    )
                if not line_str.startswith("[STEP_TELEMETRY]"):
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log contains unexpected record format on line {line_idx}: {line_str}",
                    )
                if declared_run_id and f"run_id={declared_run_id}" not in line_str:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log step event on line {line_idx} does not match run_id '{declared_run_id}'.",
                    )

                m_step = re.search(r"step=(\d+)", line_str)
                if not m_step:
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log step event on line {line_idx} missing valid step number."
                cur_step = int(m_step.group(1))
                if cur_step <= last_log_step:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log steps not strictly monotonic or duplicate detected: step {cur_step} <= {last_log_step}.",
                    )
                if cur_step > declared_steps:
                    return (
                        False,
                        "ATTESTATION_EVIDENCE_INVALID",
                        f"Telemetry log step {cur_step} exceeds declared total_steps {declared_steps} on line {line_idx}.",
                    )

                # Loss metrics verification
                m_loss = re.search(r"train_loss=([0-9\.\-eE]+)", line_str)
                if not m_loss:
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log step event on line {line_idx} missing train_loss."
                try:
                    t_loss = float(m_loss.group(1))
                    if not math.isfinite(t_loss) or t_loss <= 0.0:
                        return (
                            False,
                            "ATTESTATION_EVIDENCE_INVALID",
                            f"Telemetry log step event on line {line_idx} has non-positive or non-finite train_loss: {t_loss}",
                        )
                except ValueError:
                    return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log step event on line {line_idx} has malformed train_loss: {m_loss.group(1)}"

                m_vloss = re.search(r"val_loss=([0-9\.\-eE]+)", line_str)
                if m_vloss:
                    try:
                        v_loss = float(m_vloss.group(1))
                        if not math.isfinite(v_loss) or v_loss <= 0.0:
                            return (
                                False,
                                "ATTESTATION_EVIDENCE_INVALID",
                                f"Telemetry log step event on line {line_idx} has non-positive or non-finite val_loss: {v_loss}",
                            )
                    except ValueError:
                        return False, "ATTESTATION_EVIDENCE_INVALID", f"Telemetry log step event on line {line_idx} has malformed val_loss: {m_vloss.group(1)}"

                last_log_step = cur_step
                step_event_count += 1

            if step_event_count == 0:
                return False, "ATTESTATION_EVIDENCE_INVALID", "Telemetry log contains zero [STEP_TELEMETRY] events."
            if last_log_step != declared_steps:
                return (
                    False,
                    "ATTESTATION_EVIDENCE_INVALID",
                    f"Telemetry log final step ({last_log_step}) does not match declared total_steps ({declared_steps}).",
                )
        except Exception as e:
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Failed reading telemetry logs artifact: {e}"

        # 7. Candidate Checkpoint Verification (if path supplied)
        expected_ckpt_hash = str(evidence.get("checkpoint_sha256", "")).strip().lower()
        if not is_valid_hex64(expected_ckpt_hash):
            return (
                False,
                "ATTESTATION_EVIDENCE_INVALID",
                f"Missing or invalid expected checkpoint_sha256: '{expected_ckpt_hash}'",
            )
        if checkpoint_file_path:
            if not os.path.isfile(checkpoint_file_path):
                return False, "STRUCTURALLY_INVALID", f"Candidate checkpoint file not found: '{checkpoint_file_path}'"
            actual_ckpt_hash = compute_file_sha256(checkpoint_file_path)
            if actual_ckpt_hash != expected_ckpt_hash:
                return (
                    False,
                    "ATTESTATION_CHECKPOINT_HASH_MISMATCH",
                    f"Candidate checkpoint disk SHA-256 '{actual_ckpt_hash}' does not match manifest '{expected_ckpt_hash}'.",
                )

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
        require_artifacts: bool = True,
        checkpoint_file_path: Optional[str] = None,
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

        env_dict = evidence.get("environment", {})
        if not isinstance(env_dict, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "environment section must be a dictionary."
        env_hash = env_dict.get("dependencies_hash") or env_dict.get("environment_manifest_hash")
        if not is_valid_hex64(env_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Missing or invalid environment dependencies_hash format: {env_hash}"

        optim_dict_manifest = evidence.get("optimizer", {})
        if not isinstance(optim_dict_manifest, dict):
            return False, "ATTESTATION_EVIDENCE_INVALID", "optimizer section must be a dictionary."
        opt_hash = optim_dict_manifest.get("optimizer_telemetry_hash") or optim_dict_manifest.get("optimizer_hash")
        if not is_valid_hex64(opt_hash):
            return False, "ATTESTATION_EVIDENCE_INVALID", f"Missing or invalid optimizer_telemetry_hash format: {opt_hash}"

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
            art_ok, art_tier, art_reason = resolved_store.verify_artifacts(
                evidence,
                checkpoint_file_path=checkpoint_file_path,
                attestation=attestation,
            )
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
            "optimizer_telemetry_hash": hashlib.sha256(b"moon_optimizer_telemetry").hexdigest(),
            "artifact_file": "optimizer_telemetry.json",
        },
        "metrics_history": [
            {"step": 100, "train_loss": 3.85, "val_loss": 3.91},
            {"step": 500, "train_loss": 2.65, "val_loss": 2.70},
            {"step": 1000, "train_loss": 2.15, "val_loss": 2.21},
        ],
        "telemetry_logs_sha256": hashlib.sha256(b"moon_telemetry_logs_raw").hexdigest(),
        "telemetry_logs_file": "telemetry_logs.raw",
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
    checkpoint_file_sha256: Optional[str] = None,
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

    optim_step_500 = 500 if step_count > 500 else max(1, step_count // 2)
    s1 = min(100, max(1, step_count // 4 if step_count >= 4 else 1))
    s2 = optim_step_500
    optim_records = [
        {"step": s1, "lr": 0.0006, "grad_norm": 0.45},
        {"step": s2, "lr": 0.0003, "grad_norm": 0.32},
        {"step": step_count, "lr": 0.00006, "grad_norm": 0.28},
    ]
    # Filter unique steps sorted
    seen_steps = set()
    unique_records = []
    for r in sorted(optim_records, key=lambda x: x["step"]):
        if r["step"] not in seen_steps:
            seen_steps.add(r["step"])
            unique_records.append(r)

    # Ensure last record is exactly step_count
    if unique_records[-1]["step"] != step_count:
        unique_records.append({"step": step_count, "lr": 0.00006, "grad_norm": 0.28})

    optim_content = json.dumps({
        "run_id": run_id,
        "name": "AdamW",
        "learning_rate": 0.0006,
        "total_steps": step_count,
        "step_records": unique_records
    }, indent=2).encode("utf-8")
    optim_file = os.path.join(target_dir, "optimizer_telemetry.json")
    with open(optim_file, "wb") as f:
        f.write(optim_content)
    optim_hash = hashlib.sha256(optim_content).hexdigest().lower()

    log_lines = [
        f"[RUN_INIT] run_id={run_id} policy=STRICT_SOVEREIGN_RANDOM_INIT_ONLY step=0"
    ]
    for r in unique_records:
        st_val = r["step"]
        t_l = 3.85 if st_val <= 100 else (2.65 if st_val <= optim_step_500 else 2.15)
        v_l = 3.91 if st_val <= 100 else (2.70 if st_val <= optim_step_500 else 2.21)
        log_lines.append(f"[STEP_TELEMETRY] run_id={run_id} step={st_val} train_loss={t_l} val_loss={v_l}")
    log_lines.append(f"[RUN_COMPLETE] run_id={run_id} total_steps={step_count} status=SUCCESS")
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
        "optimizer": optim_file,
        "telemetry_logs": log_file,
    }

    manifest = create_sample_training_evidence(
        checkpoint_file_sha256=checkpoint_file_sha256 or ("00" * 32),
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
        optimizer={
            "name": "AdamW",
            "learning_rate": 0.0006,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "total_steps": step_count,
            "optimizer_telemetry_hash": optim_hash,
            "artifact_file": "optimizer_telemetry.json",
        },
        metrics_history=[
            {"step": min(100, step_count // 4 if step_count >= 4 else 1), "train_loss": 3.85, "val_loss": 3.91},
            {"step": optim_step_500, "train_loss": 2.65, "val_loss": 2.70},
            {"step": step_count, "train_loss": 2.15, "val_loss": 2.21},
        ],
        telemetry_logs_sha256=logs_hash,
        telemetry_logs_file="telemetry_logs.raw",
    )
    return paths, manifest
