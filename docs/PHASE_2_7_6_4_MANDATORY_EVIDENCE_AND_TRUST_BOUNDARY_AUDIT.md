# Moon AI — Phase 2.7.6.4 Security Audit & Trust Boundary Verification Report
**Date:** November 24, 2026  
**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Standard:** Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.1 (`MRDC-JSON-2.1`)  
**Status:** FULLY CLOSED & AUDITED  

---

## Executive Summary

Phase 2.7.6.4 establishes **Mandatory Evidence Enforcement, Physical Artifact Verification & Production Trust Boundary Closure** across the entire Moon AI Sovereign Intelligence Core.

Prior to Phase 2.7.6.4, cryptographic attestations verified that a registered Ed25519 authority key had signed a canonical JSON payload, but the training evidence manifest remained optional. Furthermore, test-isolated trust providers could theoretically be instantiated without a strict compile-time and runtime barrier forbidding them from claiming production authority.

Phase 2.7.6.4 closes all remaining gaps:
1. **Mandatory Evidence Manifest:** Cryptographic verification strictly fails closed (`ATTESTATION_EVIDENCE_INVALID`) if the evidence manifest is absent, empty, or declares an unapproved schema.
2. **Physical Byte-Level Artifact Verification:** `EvidenceArtifactStore` resolves actual physical files on disk (`architecture_spec.json`, `dataset_manifest.json`, `code_manifest.json`, `environment_manifest.json`, `telemetry_logs.raw`, `optimizer_telemetry.json`), recomputes SHA-256 digests from raw disk bytes, and validates telemetry contents (e.g. `run_id` physical presence).
3. **Production Trust Isolation:** `ExternalTrustRootProvider.is_production_provider()` returns `True` **only if** `self.is_configured()` is `True`. An unconfigured provider fails closed (`UnconfiguredFailClosed`) and cannot claim production authority. `TestIsolatedTrustProvider` returns `is_production_provider() == False` and is unconditionally rejected whenever production verification is requested (`ATTESTATION_UNTRUSTED_SIGNER`).
4. **Runtime Production Gate:** `is_verified_genuine_checkpoint()` strictly defaults to `require_production=True`. `ModelService.load_model()` and `ModelService.generate()` strictly enforce production verification, blocking test providers from authorizing production inference.

---

## 1. Architectural Changes & Security Enhancements

### 1.1 `core/security/training_evidence.py`
- **Supported Schemas:** `SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {"2.1", "MRDC-JSON-2.1"}`. Unsupported schema versions are rejected immediately.
- **`EvidenceArtifactStore` Class:**
  - Manages artifact directory resolution and validation.
  - Implements `resolve_artifact_path()` with standard filename conventions.
  - Implements `verify_artifacts()`: reads disk bytes, computes SHA-256 for model architecture, dataset manifest, training code, and telemetry logs, and compares against manifest claims.
  - Verifies physical presence of `run_id` inside raw telemetry log bytes.
- **Cross-Consistency Validation:**
  - `TrainingEvidenceValidator.validate_evidence_manifest()` rigorously cross-validates:
    - `run_id`
    - `policy` (`STRICT_SOVEREIGN_RANDOM_INIT_ONLY`)
    - `initialization` (`random_gaussian_scratch`)
    - `timestamps` (`training_start_time`, `training_completion_time`)
    - `model_architecture_hash`
    - `dataset_manifest_hash`
    - `training_code_hash`
    - `step_count` vs optimizer `total_steps`
    - `checkpoint_sha256` vs candidate checkpoint raw bytes
- **`create_sample_training_artifacts()` Helper:**
  - Generates authentic sample artifact files and manifests for isolated test fixtures.

### 1.2 `core/security/trust_provider.py`
- **`BaseTrustRootProvider` Interface:**
  - Added abstract method `is_configured() -> bool`.
  - Added concrete inspection method `get_configuration_status() -> Dict[str, Any]`.
- **`ExternalTrustRootProvider` Implementation:**
  - `is_configured()` checks for valid non-empty authorities and absence of fail-closed states.
  - `is_production_provider()` strictly returns `self.is_configured()`. Unconfigured external providers cannot claim production status.
- **`TestIsolatedTrustProvider` Implementation:**
  - `is_production_provider()` returns `False`.
  - `is_configured()` returns `bool(self.test_authorities and len(self.test_authorities) > 0)`.

### 1.3 `core/security/training_attestation.py`
- **Mandatory Evidence:**
  - In `verify_training_attestation()`:
    ```python
    ev_data = evidence_manifest or attestation.get("evidence_manifest")
    if ev_data is None:
        return False, "ATTESTATION_EVIDENCE_INVALID", "Training evidence manifest is strictly mandatory..."
    ```
- **Production Provider Enforcement:**
  - Added `require_production_provider: bool = False`. If True, enforces `resolved_provider.is_production_provider()` and `resolved_provider.is_configured()`.
- **Expiration Check Defense-in-Depth:**
  - Validates `valid_until` ISO UTC timestamp directly against current UTC time on authority records, returning `ATTESTATION_EXPIRED_AUTHORITY` if expired.
- **Fixture Generation Alignment:**
  - `create_test_training_attestation()` accepts `training_start_time` / `start_time` and `training_completion_time` / `completion_time`, ensuring perfect coherence between attestation and evidence manifest.

### 1.4 `webapp/model_service.py`
- **`classify_checkpoint()`:**
  - Supports `require_production_provider`, `evidence_store`, and `require_artifacts`.
  - Passes these parameters to `verify_training_attestation()`.
- **`is_verified_genuine_checkpoint()`:**
  - Strictly defaults to `require_production=True`.
  - Passes `require_production_provider=require_production` to `classify_checkpoint()`.
- **`ModelService` Production Gate:**
  - `load_model()` and `find_pilot_checkpoint()` use `is_verified_genuine_checkpoint(..., require_production=True)`.
  - `_generate_internal()` enforces that the active checkpoint passes `is_verified_genuine_checkpoint(..., require_production=True)` unless `allow_fixture=True`.

---

## 2. 24 Adversarial Test Vectors (`test_phase_2_7_6_4_mandatory_evidence_and_trust_isolation.py`)

| # | Test Vector | Description | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_01_missing_evidence_manifest_rejected` | Checkpoint attestation lacking evidence manifest rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 2 | `test_02_empty_evidence_manifest_rejected` | Checkpoint attestation with empty `{}` evidence rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 3 | `test_03_unknown_evidence_schema_rejected` | Evidence declaring unsupported schema version `"1.0"` rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 4 | `test_04_missing_dataset_artifact_rejected` | Missing `dataset_manifest.json` on disk rejected when `require_artifacts=True` | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 5 | `test_05_dataset_artifact_hash_mismatch_rejected` | Tampered `dataset_manifest.json` bytes on disk rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 6 | `test_06_missing_architecture_artifact_rejected` | Missing `architecture_spec.json` on disk rejected when `require_artifacts=True` | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 7 | `test_07_architecture_artifact_hash_mismatch_rejected` | Tampered `architecture_spec.json` bytes on disk rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 8 | `test_08_missing_training_code_artifact_rejected` | Missing `code_manifest.json` on disk rejected when `require_artifacts=True` | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 9 | `test_09_training_code_artifact_hash_mismatch_rejected` | Tampered `code_manifest.json` bytes on disk rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 10 | `test_10_missing_telemetry_logs_artifact_rejected` | Missing `telemetry_logs.raw` on disk rejected when `require_artifacts=True` | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 11 | `test_11_telemetry_logs_artifact_hash_mismatch_rejected` | Tampered `telemetry_logs.raw` bytes on disk rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 12 | `test_12_fabricated_optimizer_step_count_rejected` | Attestation step count (1000) vs evidence optimizer steps (99999) rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 13 | `test_13_run_id_mismatch_between_attestation_and_evidence_rejected` | Attestation `run_authentic` vs evidence `run_forged` rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 14 | `test_14_attestation_evidence_timestamp_mismatch_rejected` | Incoherent start/completion timestamps rejected | **PASS** (`ATTESTATION_EVIDENCE_INVALID`) |
| 15 | `test_15_checkpoint_byte_modification_rejected` | Modifying candidate checkpoint bytes rejected | **PASS** (`ATTESTATION_CHECKPOINT_HASH_MISMATCH`) |
| 16 | `test_16_test_provider_rejected_by_production_verification` | `TestIsolatedTrustProvider` rejected when `require_production_provider=True` | **PASS** (`ATTESTATION_UNTRUSTED_SIGNER`) |
| 17 | `test_17_test_provider_cannot_authorize_direct_model_service_generate` | Direct `ModelService.generate()` raises `ModelNotTrainedException` | **PASS** (`ModelNotTrainedException`) |
| 18 | `test_18_allow_untrusted_for_test_cannot_produce_production_genuine_classification` | `allow_test_fixture_classification` produces `TEST_FIXTURE_VALID_STRUCTURE`, never genuine | **PASS** |
| 19 | `test_19_missing_external_trust_root_fails_closed` | Missing external trust store returns `{}` and `is_production_provider() == False` | **PASS** (`UnconfiguredFailClosed`) |
| 20 | `test_20_repository_trust_root_injection_rejected` | Specifying repository path for production trust root fails closed | **PASS** (`InRepoSecurityViolationFailClosed`) |
| 21 | `test_21_revoked_signer_rejected` | Signer status `"revoked"` rejected | **PASS** (`ATTESTATION_REVOKED_AUTHORITY`) |
| 22 | `test_22_expired_signer_rejected` | Signer with expired `valid_until` timestamp rejected | **PASS** (`ATTESTATION_EXPIRED_AUTHORITY`) |
| 23 | `test_23_debug_inference_cannot_bypass_production_gate` | `DEBUG_INFERENCE=1` cannot bypass model readiness gate | **PASS** (`MODEL_NOT_READY`) |
| 24 | `test_24_no_training_or_checkpoint_writing_side_effects` | Confirms `AUTHORIZED_BY_PREM = False` locked and zero temporary checkpoint files | **PASS** |

---

## 3. Full Repository Regression & Frontend Build Verification

1. **Phase 2.7.6.4 Test Suite:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_phase_2_7_6_4_mandatory_evidence_and_trust_isolation.py`
   - Result: `Ran 24 tests in 2.146s ... OK` (100% Pass)
2. **Phase 2.7.6.2 & Phase 2.7.6.3 Test Suites:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_phase_2_7_6_2_cryptographic_provenance.py tests/test_phase_2_7_6_3_external_trust_and_evidence.py`
   - Result: `Ran 46 tests in 1.976s ... OK` (100% Pass)
3. **Stage 2.1.3 & Stage 2.4 Test Suites:**
   - Command: `.venv\Scripts\python.exe -m unittest tests/test_stage_2_1_3_teacher_distillation.py tests/test_stage_2_4_checkpoint_isolation_and_integrity.py`
   - Result: `Ran 32 tests in 46.143s ... OK` (100% Pass)
4. **Full Repository Discovery Suite:**
   - Command: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`
   - Result: `Ran 663 tests in 662.999s ... OK` (660 passed, 3 skipped, 0 failures, 0 errors)
5. **Frontend Production Build:**
   - Command: `npm run build --prefix frontend`
   - Result: `✓ 1990 modules transformed. built in 6.85s` (100% Green)

---

## 4. Immutable Architectural Governance

- `core.policy.AUTHORIZED_BY_PREM = False` strictly preserved and locked.
- `ACTIVE_OWNERSHIP_POLICY = "STRICT_SOVEREIGN_RANDOM_INIT_ONLY"`.
- Zero backward passes, zero optimizer steps, zero training executions.
- PRs #1, #2, #3, and #4 remain in draft state and unmerged.
- Zero blanket git staging commands used.
