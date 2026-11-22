# Moon AI — Phase 2.7.6.2: Cryptographic Training Provenance & Production Gate Closure

**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Repository:** `Prem759-0/NanoGPT`  
**Active Branch:** `feature/phase-2-7-5-training-lock-audit-closure`  
**Active Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Authorization Lock:** `AUTHORIZED_BY_PREM = False` (Permanent & Non-Negotiable)  
**Attestation Standard:** Restricted Deterministic Canonical JSON Ed25519 Training Attestation v2.0  

---

## 1. Executive Summary

Phase 2.7.6.2 eliminates the final checkpoint provenance fallthrough vulnerability identified in independent repository audits. Previously, `classify_checkpoint()` permitted checkpoints claiming strict metadata (`STRICT_SOVEREIGN_RANDOM_INIT_ONLY`, `random_gaussian_scratch`, and `teacher_checkpoint_weights_transferred = False`) with positive iteration and loss telemetry to fall through to `GENUINE_TRAINED_SOVEREIGN` without cryptographic verification.

Under Phase 2.7.6.2:
1. **Zero-Trust Checkpoint Provenance:** A checkpoint is never classified as genuine or promoted for production serving without an asymmetric Ed25519 cryptographic training attestation signed by an authorized, registered training authority.
2. **Deterministic Canonical JSON Specification:** Attestation payload strictly binds:
   - `run_id`
   - `policy` (must be `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`)
   - `initialization` (must be `random_gaussian_scratch`)
   - `model_architecture_hash`
   - `dataset_manifest_hash`
   - `training_code_hash`
   - `training_start_time` & `training_completion_time`
   - `step_count` (> 0)
   - `metrics` (telemetry dictionary)
   - `checkpoint_sha256` (matching raw file bytes on disk)
   - `verifier_id` (registered in secure keystore)
   - `signature` (64-byte Ed25519 signature)
3. **Fail-Closed Root Keystore:** `core/security/trusted_training_authorities.json` defaults to an empty authority store (`"trusted_authorities": {}`). Production inference honestly remains `MODEL_NOT_READY` until an authorized, scratch-trained sovereign model is cryptographically attested.
4. **15 Mutually Exclusive Provenance Tiers:** Checkpoint classifications are partitioned into 15 strict tiers:
   - `STRUCTURALLY_INVALID`
   - `STRUCTURALLY_VALID_UNVERIFIED`
   - `TEST_FIXTURE`
   - `HISTORICAL_POLICY_B`
   - `UNAPPROVED_POLICY`
   - `NON_SCRATCH_INITIALIZATION`
   - `TEACHER_CONTAMINATION`
   - `UNTRAINED_RANDOM_INIT`
   - `SYNTHETIC_MOCK`
   - `ATTESTATION_MISSING`
   - `ATTESTATION_INVALID`
   - `ATTESTATION_CHECKPOINT_HASH_MISMATCH`
   - `ATTESTATION_UNTRUSTED_SIGNER`
   - `GENUINE_TRAINED_SOVEREIGN`
   - `PROMOTED_PRODUCTION`
5. **Strict Discovery Defaults:** `find_checkpoint(allow_fixture=False)`, `find_pilot_checkpoint(allow_fixture=False)`, and `load_model(allow_fixture=False)` strictly reject test fixtures and retrospective checkpoints by default. `DEBUG_INFERENCE=1` cannot bypass production gates in live serving.
6. **Elimination of Stale Claims:** Removed outdated references to LLaMA-3.2/GPT-2 support and "frontier quality" claims in `webapp/model_service.py`. Refactored system reflexes to accurately declare architecture capabilities without asserting training completion (`tokens_generated: 0`). Procedural image/video utilities in `webapp/main.py` are explicitly labelled as deterministic procedural engines.
7. **Frontend UI Enhancement:** Added responsive sidebar toggle controls (`PanelLeftClose` / `PanelLeftOpen` from `lucide-react`) in `frontend/src/App.jsx` with full state persistence in `localStorage`.

---

## 2. Technical Architecture & Cryptographic Engine

### 2.1 Attestation Engine (`core/security/training_attestation.py`)
- Employs asymmetric Ed25519 cryptography (`cryptography.hazmat.primitives.asymmetric.ed25519`).
- Canonical payload serialization follows RFC 8785 (Restricted Deterministic Canonical JSON):
  - Keys sorted in lexicographical order.
  - Zero whitespace separators (`separators=(',', ':')`).
  - Strict UTF-8 byte encoding.
- Recomputes exact SHA-256 byte digest from candidate checkpoint file on disk.

### 2.2 Authority Keystore (`core/security/trusted_training_authorities.json`)
```json
{
  "schema_version": "2.0",
  "authority": "Moon AI Sovereign Training Governance",
  "updated_at": "2026-11-22T00:00:00Z",
  "policy": "ASYMMETRIC_ED25519_ONLY",
  "notes": "Cryptographic trust root for sovereign training provenance attestations. Private signing keys are held strictly by authorized independent training audit authorities outside this repository. If no genuine authority is configured, model promotion and production classification remain fail-closed.",
  "trusted_authorities": {}
}
```

### 2.3 Verification Flow
```
Candidate Checkpoint
       │
       ▼
validate_checkpoint_structure() ──► Invalid ──► STRUCTURALLY_INVALID
       │ Valid
       ▼
Negative Screening (Mock / Fixture / Policy / Init / Teacher / Telemetry)
       │ Failed Screening ──► [SYNTHETIC_MOCK, TEST_FIXTURE, HISTORICAL_POLICY_B,
       │                       UNAPPROVED_POLICY, NON_SCRATCH_INITIALIZATION,
       │                       TEACHER_CONTAMINATION, UNTRAINED_RANDOM_INIT]
       │ Passed Screening
       ▼
Attestation Discovery (In-bundle or sibling .attestation.json)
       │ Not Found ──► ATTESTATION_MISSING
       │ Found
       ▼
verify_training_attestation()
       ├── Missing required schema fields ────────► ATTESTATION_INVALID
       ├── Policy != STRICT_SOVEREIGN... ────────► UNAPPROVED_POLICY
       ├── Init != random_gaussian_scratch ──────► NON_SCRATCH_INITIALIZATION
       ├── Step count <= 0 ──────────────────────► UNTRAINED_RANDOM_INIT
       ├── Checkpoint file SHA-256 mismatch ─────► ATTESTATION_CHECKPOINT_HASH_MISMATCH
       ├── Verifier not in trusted_authorities ──► ATTESTATION_UNTRUSTED_SIGNER
       ├── Invalid Ed25519 signature ────────────► ATTESTATION_INVALID
       └── Valid Signature & Trusted Key
               ├── Status == PROMOTED_PRODUCTION ─► PROMOTED_PRODUCTION
               └── Standard Status ──────────────► GENUINE_TRAINED_SOVEREIGN
```

---

## 3. Verification & Test Evidence

### 3.1 Dedicated Adversarial Test Suite
Executed `tests/test_phase_2_7_6_2_cryptographic_provenance.py` covering all 19 adversarial vectors:
- **Test 01:** Empty trust store defaults fail closed (`PASS`)
- **Test 02:** Checkpoint without attestation classified as `ATTESTATION_MISSING` (`PASS`)
- **Test 03:** Corrupt or malformed checkpoint classified as `STRUCTURALLY_INVALID` (`PASS`)
- **Test 04:** Synthetic mock classified as `SYNTHETIC_MOCK` (`PASS`)
- **Test 05:** CI test fixture classified as `TEST_FIXTURE` (`PASS`)
- **Test 06:** Historical Policy B classified as `HISTORICAL_POLICY_B` (`PASS`)
- **Test 07:** Unapproved policy classified as `UNAPPROVED_POLICY` (`PASS`)
- **Test 08:** Non-scratch init classified as `NON_SCRATCH_INITIALIZATION` (`PASS`)
- **Test 09:** Teacher weights transferred classified as `TEACHER_CONTAMINATION` (`PASS`)
- **Test 10:** Untrained iteration zero classified as `UNTRAINED_RANDOM_INIT` (`PASS`)
- **Test 11:** Attestation missing required fields classified as `ATTESTATION_INVALID` (`PASS`)
- **Test 12:** Attestation hash mismatch classified as `ATTESTATION_CHECKPOINT_HASH_MISMATCH` (`PASS`)
- **Test 13:** Attestation untrusted signer classified as `ATTESTATION_UNTRUSTED_SIGNER` (`PASS`)
- **Test 14:** Attestation tampered signature classified as `ATTESTATION_INVALID` (`PASS`)
- **Test 15:** Attestation tampered payload detected and rejected (`PASS`)
- **Test 16:** Valid attestation with trusted authority classified as `GENUINE_TRAINED_SOVEREIGN` (`PASS`)
- **Test 17:** Promoted production checkpoint with valid attestation (`PASS`)
- **Test 18:** `find_checkpoint` and `find_pilot_checkpoint` default to no fixtures (`PASS`)
- **Test 19:** `DEBUG_INFERENCE` cannot bypass production verification (`PASS`)

**Result:** `Ran 19 tests in 10.029s — OK`

### 3.2 Regression Suite Summary
- `tests/test_phase_2_7_6_runtime_sovereignty.py`: 18 tests passed (`OK`)
- `tests/test_model_inference.py`: 7 tests passed (`OK`)
- `tests/test_webapp.py`: 9 tests passed (`OK`)
- `tests/test_phase_2_6_4_5_consumer_closure.py`: 9 tests passed (`OK`)
- `tests/test_phase_2_6_4_8_attestation_and_fixture.py`: 11 tests passed (`OK`)
- **Frontend Build (`npm run build --prefix frontend`):** `✓ built in 24.33s` (0 errors)
- **Live Health Endpoint (`/api/health`):** `{"status":"ok","model_status":"MODEL_NOT_READY","checkpoint_exists":true,"model_loaded":false,"device":"cpu","torch_available":true,"torch_version":"2.14.1+cpu","is_sovereign_ready":false,"sovereign_checkpoint_path":null}`

---

## 4. Governance Constraints & Sign-off

- `core.policy.AUTHORIZED_BY_PREM = False`: strictly preserved and locked.
- Pull Requests #1, #2, #3, and #4 remain unmerged in draft state.
- Zero neural training steps executed; zero weights modified.
- All code changes strictly bounded to provenance gating, cryptographic verification, diagnostic honesty, and UI layout.
