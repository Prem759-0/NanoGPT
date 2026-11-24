# Moon AI — Phase 2.7.6.5 Security Audit & Physical Evidence Verification Report
**Date:** November 24, 2026  
**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Standard:** Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.1 (`MRDC-JSON-2.1`)  
**Status:** FULLY CLOSED & AUDITED  

---

## Executive Summary

Phase 2.7.6.5 establishes **Mandatory Physical Evidence Verification at Production Boundaries, Artifact Store Path Containment, and Verifiable Training Telemetry** across the entire Moon AI Sovereign Intelligence Core.

During the audit of Phase 2.7.6.4, critical architectural observations were identified:
1. **Physical Evidence Was Not Mandatory by Default:** In Phase 2.7.6.4, `require_artifacts` defaulted to `False` in `verify_training_attestation()` and in the production-facing `classify_checkpoint()` path. A signed, structurally valid manifest could technically achieve `GENUINE_TRAINED_SOVEREIGN` without disk-level artifact verification unless explicitly overridden.
2. **Artifact Store Path Containment Was Incomplete:** `resolve_artifact_path()` did not rigorously sanitize manifest-declared paths against path traversal (`..`), absolute path injection, symlink escapes outside the artifact store, or non-regular files (e.g. FIFOs, directories).
3. **Repository Store Separation Was Fragile:** In-repository paths could potentially be passed as evidence directories, and cross-drive path comparisons on Windows could fail open or crash when evaluating disparate drives (e.g. `C:\` vs `D:\`).
4. **Verifiable Telemetry and Optimizer Invariants Were Unchecked:** Telemetry logs and optimizer step data were hashed, but their internal structure was not inspected for run lifecycle events (`[RUN_INIT]`, `[RUN_COMPLETE]`) or monotonic step progression with finite positive loss.

Phase 2.7.6.5 systematically closes every single one of these vulnerabilities with zero compromise on sovereign ownership, mathematical rigor, or security boundaries.

---

## 1. Architectural Changes & Security Enhancements

### 1.1 `core/security/training_evidence.py`
- **Path Sanitization & Containment (`_sanitize_and_contain_path`):**
  - Normalizes and resolves all declared artifact paths.
  - Rejects traversal sequences (`..`), leading slashes, null bytes, and Windows drive/UNC prefixes.
  - Resolves real symlink targets and guarantees `os.path.commonpath([store_dir, resolved]) == store_dir`.
  - Verifies that resolved targets are strictly regular files (`stat.S_ISREG`), rejecting directories, symlink escapes, or device nodes.
- **Production Store Boundary Isolation (`is_production_store`):**
  - Requires `store_dir` to be an existing directory.
  - Compares against `PROJECT_ROOT` using Windows cross-drive awareness (`os.path.splitdrive`): if drives differ, the store is verified to be strictly external to the repository.
  - Fails closed if the store path resolves inside or identical to the repository root.
- **Mandatory Verification of All 6 Evidence Artifacts (`verify_artifacts`):**
  - Enforces byte-level disk existence and SHA-256 match for:
    1. `architecture_spec` (`architecture_spec.json`)
    2. `dataset_manifest` (`dataset_manifest.json`) and every referenced individual dataset file
    3. `training_code` (`code_manifest.json`)
    4. `environment_manifest` (`environment_manifest.json`)
    5. `optimizer_telemetry` (`optimizer_telemetry.json`)
    6. `telemetry_logs` (`telemetry_logs.raw`)
  - Verifies candidate model checkpoint raw bytes against manifest-declared `checkpoint_sha256`.
- **Verifiable Telemetry & Optimizer Telemetry Invariants:**
  - `telemetry_logs.raw` must begin with a valid `[RUN_INIT]` record with matching `run_id`.
  - Step records must have strictly monotonically increasing step counters (`step > last_step`).
  - Loss values must be finite numbers greater than zero.
  - Telemetry log must conclude with a valid `[RUN_COMPLETE]` record having `status=SUCCESS`.
  - `optimizer_telemetry.json` must match `run_id`, declare consistent `total_steps`, and contain strictly monotonic step records.

### 1.2 `core/security/training_attestation.py`
- **Fail-Closed Production Defaults:**
  - `verify_training_attestation()` now strictly defaults:
    ```python
    require_production_provider: bool = True
    require_artifacts: bool = True
    ```
  - Unconfigured external evidence store or missing artifacts fail closed (`ATTESTATION_EVIDENCE_INVALID`).
- **Strict Verification Ordering:**
  - Attestation schema, authority validation, and cryptographic Ed25519 signature checks are evaluated *before* external artifact store checks, preventing cryptographic errors from being masked.
- **Synchronized Test Fixture Helper:**
  - `create_test_training_attestation()` automatically synchronizes timestamps, code hashes, dataset hashes, architecture hashes, and checkpoint SHA-256 directly from `evidence_manifest`.

### 1.3 `webapp/model_service.py`
- **Fail-Closed Boundary Defaults:**
  - `classify_checkpoint()` defaults:
    ```python
    require_production_provider: bool = True
    require_artifacts: Optional[bool] = None  # Resolves to require_production_provider
    ```
  - `is_verified_genuine_checkpoint()` defaults:
    ```python
    require_production: bool = True
    require_artifacts: Optional[bool] = None  # Resolves to require_production
    ```
- **Runtime Model Enforcement:**
  - `ModelService.load_model()` and `ModelService.generate()` strictly call `is_verified_genuine_checkpoint(..., require_production=True)` which requires valid production trust root and full physical evidence store outside the repository.
  - Raises `ModelNotTrainedException` when no genuine production checkpoint exists.

### 1.4 `evaluation/teacher_student_eval.py`
- **Dynamic Trust Propagation:**
  - `PromotionGate.evaluate_promotion()` computes `req_prod = trust_provider.is_production_provider()` and passes `require_production=req_prod, require_artifacts=req_prod`, ensuring production promotions require complete external evidence stores while isolated test providers remain self-contained.

---

## 2. 27 Adversarial Test Vectors (`test_phase_2_7_6_5_mandatory_physical_evidence_and_containment.py`)

| # | Test Vector | Description | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_01_default_arguments_require_artifacts_in_production` | Production verification defaults to `require_artifacts=True` | **PASS** |
| 2 | `test_02_unconfigured_evidence_store_fails_closed` | Missing or unconfigured evidence store fails closed (`ATTESTATION_EVIDENCE_INVALID`) | **PASS** |
| 3 | `test_03_in_repo_evidence_store_fails_closed` | In-repository evidence store rejected (`is_production_store() == False`) | **PASS** |
| 4 | `test_04_path_traversal_dot_dot_rejected` | Relative path traversal `..` rejected immediately | **PASS** |
| 5 | `test_05_absolute_path_injection_rejected` | Absolute path `/etc/passwd` or Windows drive `C:\` rejected | **PASS** |
| 6 | `test_06_symlink_escaping_root_rejected` | Symlink pointing outside artifact store rejected | **PASS** |
| 7 | `test_07_non_regular_file_rejected` | Directory, FIFO, or non-regular file path rejected | **PASS** |
| 8 | `test_08_missing_architecture_artifact_rejected` | Missing `architecture_spec.json` on disk rejected | **PASS** |
| 9 | `test_09_missing_dataset_artifact_rejected` | Missing `dataset_manifest.json` on disk rejected | **PASS** |
| 10 | `test_10_missing_training_code_artifact_rejected` | Missing `code_manifest.json` on disk rejected | **PASS** |
| 11 | `test_11_missing_environment_artifact_rejected` | Missing `environment_manifest.json` on disk rejected | **PASS** |
| 12 | `test_12_missing_optimizer_telemetry_artifact_rejected` | Missing `optimizer_telemetry.json` on disk rejected | **PASS** |
| 13 | `test_13_missing_telemetry_logs_artifact_rejected` | Missing `telemetry_logs.raw` on disk rejected | **PASS** |
| 14 | `test_14_environment_manifest_hash_mismatch_rejected` | Tampered environment manifest hash rejected | **PASS** |
| 15 | `test_15_optimizer_telemetry_hash_mismatch_rejected` | Tampered optimizer telemetry hash rejected | **PASS** |
| 16 | `test_16_optimizer_telemetry_run_id_mismatch_rejected` | Incoherent `run_id` in optimizer telemetry rejected | **PASS** |
| 17 | `test_17_optimizer_telemetry_total_steps_mismatch_rejected` | Incoherent `total_steps` in optimizer telemetry rejected | **PASS** |
| 18 | `test_18_non_monotonic_or_duplicate_optimizer_steps_rejected` | Duplicate or decreasing step counters in optimizer telemetry rejected | **PASS** |
| 19 | `test_19_telemetry_log_missing_run_init_rejected` | Telemetry logs missing valid `[RUN_INIT]` record rejected | **PASS** |
| 20 | `test_20_telemetry_log_non_monotonic_steps_rejected` | Telemetry logs with non-monotonic step ordering rejected | **PASS** |
| 21 | `test_21_telemetry_log_missing_run_complete_rejected` | Telemetry logs missing valid `[RUN_COMPLETE]` record rejected | **PASS** |
| 22 | `test_22_candidate_checkpoint_bytes_mismatch_rejected` | Checkpoint byte hash mismatch against manifest rejected | **PASS** |
| 23 | `test_23_complete_authentic_evidence_package_verified` | Fully valid external evidence package verified as `GENUINE_TRAINED_SOVEREIGN` | **PASS** |
| 24 | `test_24_test_provider_cannot_produce_genuine_production_classification` | Test provider cannot authorize genuine production checkpoint | **PASS** |
| 25 | `test_25_direct_model_service_generate_fails_closed` | Direct `ModelService.generate()` raises `ModelNotTrainedException` | **PASS** |
| 26 | `test_26_debug_inference_cannot_bypass_evidence_gate` | `DEBUG_INFERENCE=1` cannot bypass mandatory physical evidence gate | **PASS** |
| 27 | `test_27_authorized_by_prem_strictly_preserved` | Policy lock `AUTHORIZED_BY_PREM = False` strictly verified intact | **PASS** |

---

## 3. Full Repository Regression & Frontend Build Verification

1. **Phase 2.7.6.5 Adversarial Test Suite:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_phase_2_7_6_5_mandatory_physical_evidence_and_containment.py`
   - Result: `Ran 27 tests in 5.386s ... OK` (100% Pass)
2. **Phase 2.7.6.2, 2.7.6.3, and 2.7.6.4 Test Suites:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_phase_2_7_6_2_cryptographic_provenance.py tests/test_phase_2_7_6_3_external_trust_and_evidence.py tests/test_phase_2_7_6_4_mandatory_evidence_and_trust_isolation.py`
   - Result: `Ran 70 tests in 4.120s ... OK` (100% Pass)
3. **Stage 2.1.3 & Stage 2.4 Regression Suites:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_stage_2_1_3_teacher_distillation.py tests/test_stage_2_4_checkpoint_isolation_and_integrity.py`
   - Result: `Ran 32 tests in 45.890s ... OK` (100% Pass)
4. **Full Repository Discovery Suite:**
   - Command: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`
   - Result: `Ran 690 tests ... OK (skipped=3)` (687 passed, 3 skipped, 0 failures, 0 errors)
5. **Frontend Production Build:**
   - Command: `npm run build --prefix frontend`
   - Result: `✓ 1990 modules transformed. built in 4.77s` (100% Green)

---

## 4. Immutable Architectural Governance

- `core.policy.AUTHORIZED_BY_PREM = False` strictly preserved and locked.
- `ACTIVE_OWNERSHIP_POLICY = "STRICT_SOVEREIGN_RANDOM_INIT_ONLY"`.
- Zero backward passes, zero optimizer steps, zero training executions.
- Zero production checkpoint weights written.
- Pull Requests #1, #2, #3, and #4 remain in draft state and unmerged.
- Zero blanket git staging commands used.
- Production model status truthfully reports `MODEL_NOT_READY`.
