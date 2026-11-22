# Moon AI — Phase 2.7.6.1: Runtime Gate & Checkpoint Provenance Closure Report

**Project & Repository:** Moon AI Sovereign Intelligence Core (`Prem759-0/NanoGPT`)  
**Branch:** `feature/phase-2-7-5-training-lock-audit-closure`  
**Base Commit:** `edddf3036e3b818c0e3921015cffda24004e3e55`  
**Chief ML Systems Architect & Repository Owner:** Prem Gaikwad (India 🇮🇳)  
**Active Sovereign Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Owner Authorization Lock:** `core.policy.AUTHORIZED_BY_PREM = False` (Strict Invariant)  
**Verification Date:** November 22, 2026  

---

## 1. Executive Summary & Audit Overview

Phase 2.7.6.1 performs a focused follow-up audit to Phase 2.7.6 to close all remaining bypass vectors, weak assertions, and misleading identity claims in runtime inference and test infrastructure.

### Critical Vulnerabilities Closed in Phase 2.7.6.1:
1. **The `scratch` Model Bypass Closed**:
   - Previously, selecting or switching to `scratch` relied solely on filesystem file presence (`find_checkpoint()`), bypassing the strict sovereign provenance verification gate.
   - If an unverified or legacy checkpoint existed in `out-assistant/ckpt.pt` or via `NANOGPT_CHECKPOINT_PATH`, inference proceeded with `is_neural_generation=True` and `checkpoint_verified=True`.
   - In Phase 2.7.6.1, `find_checkpoint(allow_fixture=False)` and `load_model("scratch", allow_fixture=False)` strictly require `is_verified_genuine_checkpoint()`. If unverified, `get_status()` returns `MODEL_NOT_READY`, `get_available_models()` reports `available: False`, and `/api/switch-model` rejects switches with HTTP 403 Forbidden.
2. **Checkpoint Provenance Categorization**:
   - Introduced `classify_checkpoint(ckpt_path)` to formally separate:
     - `STRUCTURALLY_INVALID` (corrupted, unreadable, non-dict, non-tensor)
     - `TEST_FIXTURE` (explicit CI fixture or smoke test)
     - `HISTORICAL_POLICY_B` (Stage 2.1.6 pilot checkpoint with quarantined Policy B provenance)
     - `UNTRAINED_RANDOM_INIT` (random scratch weights lacking training steps, `iter_num <= 0`)
     - `SYNTHETIC_MOCK` (synthetic tensors from `torch.randn` lacking training telemetry)
     - `GENUINE_TRAINED_SOVEREIGN` (fully verified scratch-trained student model)
     - `PROMOTED_PRODUCTION` (certified production checkpoint)
   - Self-declared metadata labels and random tensors are strictly rejected by `is_verified_genuine_checkpoint()`.
3. **Deterministic Response Truthfulness**:
   - Eliminated misleading claims in creator/identity reflex that stated the model was "trained purely from scratch". Accurately reports design for scratch initialization with training pending explicit owner authorization.
   - Fixed token count reporting in greeting and creator reflexes: changed from `len(reply.split())` to `tokens_generated = 0`.
   - Fixed procedural image generation route: changed from `tokens_generated = 35` to `tokens_generated = 0`, classified as `deterministic_utility`.
4. **Health Endpoint Assertion Hardening**:
   - Fixed `test_08` to reject contradictory statuses (`READY`, `READY (ON DEMAND)`). Asserts strictly `model_status == "MODEL_NOT_READY"`.
5. **Zero-Optimizer & Zero-Backward Pass Instrumentation**:
   - Instrumented `torch.optim.Optimizer.__init__` and `torch.Tensor.backward` to verify 0 optimizer instantiations and 0 backward passes during all audit queries.

> [!IMPORTANT]
> **Zero Model Training Statement**:
> No model training, fine-tuning, distillation, backward passes, or optimizer steps were executed during Phase 2.7.6.1. Zero production checkpoints were created, modified, or promoted. `core.policy.AUTHORIZED_BY_PREM = False` remained permanently intact.

---

## 2. The Scratch-Model Bypass Investigation & Closure

### Investigation Finding
An investigation into `webapp/model_service.py` and `webapp/main.py` revealed:
- `find_checkpoint()` probed disk for `ckpt.pt` in `out-assistant`, `out-nanogpt-scratch`, and `NANOGPT_CHECKPOINT_PATH` without validating provenance.
- In `load_model("scratch")`, the file was loaded directly and `self.is_test_fixture` was not set.
- In `/api/chat`, the `MODEL_NOT_READY` gate only inspected `model_service.active_model_name == "moon-chat-pilot-v0"`. If a client requested `scratch`, the gate was bypassed.
- `/api/switch-model` permitted switching to `scratch` because `available` was based on mere file existence.

### Remediations Applied
| Subsystem | File & Function | Remediation |
| :--- | :--- | :--- |
| Checkpoint Locator | `webapp/model_service.py::find_checkpoint()` | Added `allow_fixture: bool = False`. When `False`, returns a path ONLY if `is_verified_genuine_checkpoint(cand)` is True. |
| Model Loader | `webapp/model_service.py::load_model()` | For `target == "scratch"`, checks `is_verified_genuine_checkpoint()`. Fails closed if unverified. Sets `is_test_fixture = True` if loaded under `allow_fixture=True`. |
| Status Reporter | `webapp/model_service.py::get_status()` | For `scratch`, returns `MODEL_NOT_READY` unless positively verified genuine. |
| Model Catalog | `webapp/model_service.py::get_available_models()` | `scratch` reports `available: False`, `status: "MODEL_NOT_READY"` unless verified genuine. |
| Model Switch Route | `webapp/main.py::switch_model()` | Rejects switching to `scratch` with HTTP 403 Forbidden if status is `MODEL_NOT_READY` or uncertified. |
| Chat Route Gate | `webapp/main.py::chat()` | Unified `MODEL_NOT_READY` gate across ALL neural model targets (`moon-chat-pilot-v0`, `scratch`, `pilot`). Returns honest system notice. |

---

## 3. Checkpoint Provenance Verification Architecture

A checkpoint file must satisfy six strict pillars before becoming production-eligible:

```
+-----------------------------------------------------------------------------+
|               6 PILLARS OF STRICT SOVEREIGN PROVENANCE                      |
+-----------------------------------------------------------------------------+
| 1. Structural Integrity     | Valid PyTorch dictionary, safe deserialization|
| 2. Policy Conformance       | STRICT_SOVEREIGN_RANDOM_INIT_ONLY strictly     |
| 3. Scratch Initialization   | random_gaussian_scratch (zero teacher weights)|
| 4. Zero Teacher Intrusion   | teacher_checkpoint_weights_transferred: False |
| 5. Verifiable Telemetry     | iter_num > 0, val_loss > 0.0, training_done   |
| 6. Production Promotion     | Status is PROMOTED_PRODUCTION / PILOT_TRAINED |
+-----------------------------------------------------------------------------+
```

### Mutually Exclusive Checkpoint Provenance Tiers
1. `STRUCTURALLY_INVALID`: Missing file, 0 bytes, non-dict, non-tensor, or missing required keys (`model`, `model_args`).
2. `TEST_FIXTURE`: Explicitly flagged as CI fixture (`is_test_fixture=True`, `status="CI_TEST_FIXTURE"`).
3. `HISTORICAL_POLICY_B`: Stage 2.1.6 pilot checkpoint with quarantined Policy B provenance (`TEACHER_STUDENT_RANDOM_INIT`).
4. `UNTRAINED_RANDOM_INIT`: Random Gaussian initialization lacking training progress (`iter_num <= 0` or missing).
5. `SYNTHETIC_MOCK`: Tensors generated synthetically with mock labels but lacking genuine training telemetry or run ID.
6. `GENUINE_TRAINED_SOVEREIGN`: Positively satisfies all 6 pillars with verified training metrics.
7. `PROMOTED_PRODUCTION`: Officially certified production checkpoint promoted for live serving.

---

## 4. Route Classification Matrix

All routes returning `ChatResponse` adhere strictly to the following contract:

| Route / Intent | Trigger / Pattern | `engine_type` | `tokens_generated` | `is_neural_generation` | `is_deterministic_utility` | `model_status` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Creator Reflex** | "who made you", "who is your developer" | `system_reflex` | `0` | `False` | `True` | `READY` |
| **Greeting Reflex** | "hello", "hi", "good morning" | `system_reflex` | `0` | `False` | `True` | `READY` |
| **Snake Game** | "snake game in html" | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Chess Game** | "chess in html" | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Racing Game** | "racing car game in html" | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Procedural Image** | `/imagine`, "generate image of..." | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Procedural Video** | `/video`, "what about video" | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Science Knowledge** | Biological / chemical query | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Memory Intent** | "remember that...", memory check | `deterministic_utility` | `0` | `False` | `True` | `READY` |
| **Untrained Neural Notice** | General query (no genuine checkpoint) | `system_notice` | `0` | `False` | `False` | `MODEL_NOT_READY` |
| **Web Retrieval** | Live search with verifiable citations | `web_retrieval` | `0` | `False` | `False` | `READY` |
| **Verified Neural Inference** | Real verified checkpoint loaded | `sovereign_neural` | `> 0` | `True` | `False` | `READY` |

---

## 5. Verification & Test Suite Details

### Phase 2.7.6.1 Dedicated Test Suite (`tests/test_phase_2_7_6_runtime_sovereignty.py`)
- **Total Scenarios:** 18
- **Passed:** 18
- **Failed / Errored:** 0
- **Execution Time:** ~22.8s

1. `test_01_qwen_cannot_become_default_moon_production_model`: **PASSED**
2. `test_02_llama_cannot_become_default_moon_production_model`: **PASSED**
3. `test_03_gpt2_cannot_become_default_moon_production_model`: **PASSED**
4. `test_04_unverified_checkpoint_rejected`: **PASSED**
5. `test_05_test_fixture_rejected_for_production_inference`: **PASSED**
6. `test_06_historical_policy_b_checkpoint_rejected`: **PASSED**
7. `test_07_synthetic_checkpoint_fails_genuine_provenance_validation`: **PASSED**
8. `test_08_health_endpoint_reports_exact_model_not_ready`: **PASSED**
9. `test_09_scratch_checkpoint_mere_presence_does_not_enable_production_inference`: **PASSED**
10. `test_10_switching_to_scratch_blocked_without_verified_checkpoint`: **PASSED**
11. `test_11_environment_variable_override_does_not_bypass_provenance`: **PASSED**
12. `test_12_qwen_llama_gpt2_remain_blocked`: **PASSED**
13. `test_13_greetings_and_creator_reflexes_report_zero_neural_tokens`: **PASSED**
14. `test_14_creator_identity_does_not_claim_untrained_model_is_trained`: **PASSED**
15. `test_15_deterministic_utilities_report_zero_neural_tokens`: **PASSED**
16. `test_16_general_chat_returns_model_not_ready_for_scratch_and_pilot`: **PASSED**
17. `test_17_authorized_by_prem_remains_strictly_false`: **PASSED**
18. `test_18_instrumented_zero_optimizers_zero_backward_zero_checkpoint_writes`: **PASSED**

### Combined Regression Verification
- `test_phase_2_7_4_adversarial_training_routes.py`: **PASSED**
- `test_phase_2_7_5_audit_closure.py`: **PASSED**
- `test_phase_2_6_4_7_inference_and_provenance.py`: **PASSED**
- `test_webapp.py`: **PASSED**
- **Total Combined Tests:** 46
- **Passed:** 46
- **Failed / Errored:** 0

### Frontend Production Build
- Command: `npm run build --prefix frontend`
- Modules Transformed: 1,990
- Output: `dist/index.html` (0.91 kB), `dist/assets/index-CWH3yDma.js` (366.05 kB), `dist/assets/index-DDaEwjP4.css` (15.49 kB)
- Status: **PASSED** (0 errors)

---

## 6. Remaining Limitations & Path to Authorized Training

1. **Current Production Readiness Status: `MODEL_NOT_READY`**:
   - Moon AI possesses zero promoted production neural checkpoints on disk.
   - The existing pilot checkpoint (`models/checkpoints/chat/moon_chat_pilot_v0.pt`) is an unpromoted historical Policy B fixture.
2. **Authorized Training Path**:
   - Production training requires explicit owner authorization (`core.policy.AUTHORIZED_BY_PREM = True`).
   - Training must be preceded by preflight validation (`python scripts/preflight_pilot_training.py`).
   - The training run must stream strictly from verified sovereign datasets (`datasets/chat/sovereign_chat_manifest.jsonl`) with assistant loss masking.
   - When training completes, the resulting checkpoint will record verified telemetry (`iter_num`, `val_loss`, `training_completed`) and be attested as `PROMOTED_PRODUCTION`.
