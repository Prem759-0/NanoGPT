# Moon AI — Phase 2.7.6.3 Security Audit & External Trust Root Certification

**Date:** 22 November 2026  
**Chief ML Systems Architect & Repository Owner:** Prem Gaikwad (India 🇮🇳)  
**Target Branch:** `feature/phase-2-7-5-training-lock-audit-closure`  
**Base Commit:** `8e17358d89d1a8b6399bcfe87fa102032b336797`  
**Active Sovereign Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Authorization Lock:** `core.policy.AUTHORIZED_BY_PREM = False` (LOCKED)  
**Attestation Specification:** Moon Restricted Deterministic Canonical JSON Training Attestation Specification v2.1 (MRDC-JSON-2.1)

---

## Executive Summary

Phase 2.7.6.3 establishes the cryptographic boundary between in-repository code/test fixtures and production training verification. It resolves critical architectural vulnerabilities discovered during the Phase 2.7.6.2 independent audit:

1. **Complete Closure of Test-Only Bypass:**
   - Removed any caller-controlled mechanism (`allow_unattested_for_test`, `allow_test_fixture_classification`) from producing `GENUINE_TRAINED_SOVEREIGN` or `PROMOTED_PRODUCTION`.
   - Structural test fixtures strictly receive `TEST_FIXTURE_VALID_STRUCTURE` and are permanently prevented from serving production inference.
   - `is_verified_genuine_checkpoint()` accepts **zero** bypass flags and fails closed to `False` on all fixtures, mocks, unverified, and unattested models.

2. **Strict Separation of In-Repository Keystore and Production Trust Roots:**
   - In-repository file `core/security/trusted_training_authorities.json` is maintained as an empty placeholder (`{ "trusted_authorities": {} }`).
   - `ExternalTrustRootProvider` strictly rejects any trust store located inside `PROJECT_ROOT`, failing closed to empty authorities.
   - Production authority keys must be supplied via `MOON_EXTERNAL_TRUST_STORE_PATH` (strictly outside repository root) or protected secret `MOON_TRUSTED_TRAINING_AUTHORITIES_JSON`.
   - Comprehensive status enforcement distinguishes between `active`, `revoked` (`ATTESTATION_REVOKED_AUTHORITY`), and `expired` (`ATTESTATION_EXPIRED_AUTHORITY`) authorities.
   - Test suites use `TestIsolatedTrustProvider` (`is_production_provider() == False`), preventing CI test keys from ever contaminating production trust roots.

3. **Verifiable Training Evidence Manifest Validation (MRDC-JSON-2.1):**
   - Introduced `TrainingEvidenceValidator` enforcing cryptographic binding between candidate checkpoint SHA-256, random Gaussian scratch initialization, architecture hash, dataset manifest hash, code tree hash, and verifiable telemetry.
   - Enforces chronological ordering (`start_time < completion_time`), rejects future timestamps (>5 minutes clock skew), verifies strict lowercase 64-character hexadecimal hashes, and validates non-empty monotonic training telemetry histories.

4. **Honest and Accurate Frontend Architecture Display:**
   - Separated web platform status (`Online`) from neural model status (`NOT READY` / `MODEL_NOT_READY`).
   - Added `.grok-badge-amber` and `.grok-badge-dot-amber` to `frontend/src/App.css`.
   - Updated System Architecture card in `frontend/src/App.jsx` to render an amber badge (`NOT READY`) when `model_status !== 'READY'`.
   - Updated model descriptions to explicitly state that the 25.7M scratch transformer is in `MODEL_NOT_READY` awaiting authorized training run and sovereign cryptographic attestation.

5. **Elimination of Debug Inference Bypass:**
   - Eliminated `self.debug_mode` from `eff_allow_fixture` in `ModelService._generate_internal()`. `DEBUG_INFERENCE=1` enables diagnostic logging only, never bypassing the production fixture gate.

---

## Mutually Exclusive Provenance Tiers (Phase 2.7.6.3)

| Provenance Tier | Classification Description | Promotable to Prod? |
| :--- | :--- | :---: |
| `STRUCTURALLY_INVALID` | Nonexistent, empty, malformed, non-dict, non-tensor, or unreadable checkpoint. | **NO** |
| `TEST_FIXTURE` | Explicitly marked as CI fixture or smoke test (`is_test_fixture: True`). | **NO** |
| `TEST_FIXTURE_VALID_STRUCTURE` | Valid model structure and metadata in test mode, lacking cryptographic verification. | **NO** |
| `HISTORICAL_POLICY_B` | Quarantined Stage 2.1.6 pilot checkpoint with teacher-student policy. | **NO** |
| `UNAPPROVED_POLICY` | Ownership policy does not match `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`. | **NO** |
| `NON_SCRATCH_INITIALIZATION` | Initialized from pretrained weights or non-scratch Gaussian method. | **NO** |
| `TEACHER_CONTAMINATION` | Teacher weights transferred, teacher model referenced, or base weights present. | **NO** |
| `UNTRAINED_RANDOM_INIT` | Random Gaussian weights without training progress (`iter_num <= 0` or `val_loss <= 0`). | **NO** |
| `SYNTHETIC_MOCK` | Synthetically generated random weights without authentic training telemetry. | **NO** |
| `ATTESTATION_MISSING` | Checkpoint lacks an asymmetric cryptographic training attestation. | **NO** |
| `ATTESTATION_INVALID` | Attestation is malformed, missing required fields, or Ed25519 signature is invalid. | **NO** |
| `ATTESTATION_CHECKPOINT_HASH_MISMATCH` | Attestation binds a SHA-256 that does not match disk checkpoint file bytes. | **NO** |
| `ATTESTATION_UNTRUSTED_SIGNER` | Attestation signed by an unregistered authority or no trust root is configured. | **NO** |
| `ATTESTATION_REVOKED_AUTHORITY` | Attestation signed by an authority marked as revoked in external trust root. | **NO** |
| `ATTESTATION_EXPIRED_AUTHORITY` | Attestation signed by an authority whose registration has expired. | **NO** |
| `ATTESTATION_EVIDENCE_INVALID` | Training evidence manifest contains telemetry contradictions, invalid hashes, or skew. | **NO** |
| `GENUINE_TRAINED_SOVEREIGN` | Verified scratch-trained model satisfying all strict provenance pillars & Ed25519 signature. | **ELIGIBLE** |
| `PROMOTED_PRODUCTION` | Fully certified sovereign checkpoint promoted for production serving with cryptographic attestation. | **CERTIFIED** |

---

## External Trust Root vs In-Repo Separation Architecture

```
+--------------------------------------------------------------------------+
|                      PRODUCTION TRUST ROOT HIERARCHY                     |
+--------------------------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-------------------------------+         +-------------------------------+
|  MOON_EXTERNAL_TRUST_STORE_PATH|         | MOON_TRUSTED_TRAINING_        |
|  (Strictly Outside Repo Root)  |         | AUTHORITIES_JSON (Secret)     |
+-------------------------------+         +-------------------------------+
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
                  +-------------------------------------+
                  |     ExternalTrustRootProvider       |
                  |  - Validates path is OUTSIDE repo   |
                  |  - Fails closed if inside repo root |
                  |  - Checks active/revoked/expired    |
                  +-------------------------------------+
                                     |
        +----------------------------+----------------------------+
        |                                                         |
        v                                                         v
+--------------------------------+               +--------------------------------+
| In-Repo Keystore Placeholder   |               | TestIsolatedTrustProvider      |
| core/security/trusted_         |               | (In-memory dict for CI/tests)  |
| training_authorities.json      |               | is_production_provider()=False |
| { "trusted_authorities": {} }  |               | Strictly isolated from prod    |
+--------------------------------+               +--------------------------------+
```

---

## 27-Vector Adversarial Verification Matrix

| Vector # | Adversarial Test Scenario | Expected Outcome | Result |
| :---: | :--- | :--- | :---: |
| **01** | Test-only bypass cannot return genuine | `TEST_FIXTURE_VALID_STRUCTURE` | **PASSED** |
| **02** | Test-only bypass cannot enable production inference | `ModelNotTrainedException` | **PASSED** |
| **03** | In-repo trust store cannot be production root | Fails closed to `{}` | **PASSED** |
| **04** | Missing external trust root fails closed | Empty `{}` | **PASSED** |
| **05** | Invalid external trust root fails closed | Empty `{}` | **PASSED** |
| **06** | Revoked authority is rejected | `ATTESTATION_REVOKED_AUTHORITY` | **PASSED** |
| **07** | Expired authority is rejected | `ATTESTATION_EXPIRED_AUTHORITY` | **PASSED** |
| **08** | Test key is rejected in production | `is_production_provider() == False` | **PASSED** |
| **09** | Unknown signer is rejected | `ATTESTATION_UNTRUSTED_SIGNER` | **PASSED** |
| **10** | Tampered signature is rejected | `ATTESTATION_INVALID` | **PASSED** |
| **11** | Tampered payload is rejected | `ATTESTATION_INVALID` | **PASSED** |
| **12** | Checkpoint byte modification is detected | `ATTESTATION_CHECKPOINT_HASH_MISMATCH` | **PASSED** |
| **13** | Invalid timestamp ordering is rejected (start >= completion) | `ATTESTATION_INVALID` | **PASSED** |
| **14** | Future timestamp is rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **15** | Invalid hash formats rejected (non-64 hex chars) | `ATTESTATION_INVALID` | **PASSED** |
| **16** | Missing training evidence is rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **17** | Empty training metrics are rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **18** | Non-monotonic step numbers rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **19** | Dataset hash mismatch rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **20** | Architecture hash mismatch rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **21** | Training code hash mismatch rejected | `ATTESTATION_EVIDENCE_INVALID` | **PASSED** |
| **22** | Direct `ModelService.generate()` cannot bypass production gate | `ModelNotTrainedException` | **PASSED** |
| **23** | API `/api/chat` production gate | `MODEL_NOT_READY` | **PASSED** |
| **24** | Startup / discovery blocks unverified | `find_pilot_checkpoint() == None` | **PASSED** |
| **25** | `DEBUG_INFERENCE=1` cannot bypass production gate | `ModelNotTrainedException` | **PASSED** |
| **26** | `core.policy.AUTHORIZED_BY_PREM` invariant | `False` (Strictly Locked) | **PASSED** |
| **27** | Strict sovereign policy invariant | `STRICT_SOVEREIGN_RANDOM_INIT_ONLY` | **PASSED** |

---

## Invariant Guarantees
- Zero model training executed during audit (`iter_num=0`, zero optimizer steps, zero backprop).
- Zero production checkpoints written.
- In-repository placeholder keystore remains empty (`{ "trusted_authorities": {} }`).
- Pull requests #1, #2, #3, and #4 remain unmerged in draft state.
- No blanket `git add .` or `git commit -a` used.
