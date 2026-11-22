# Moon AI — Phase 2.7.6: Runtime Sovereignty, Genuine Inference & Hardcoded-Answer Audit

**Repository:** `Prem759-0/NanoGPT`  
**Branch:** `feature/phase-2-7-5-training-lock-audit-closure`  
**Chief ML Systems Architect & Repository Owner:** Prem Gaikwad (India 🇮🇳)  
**Active Sovereign Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Owner Authorization Invariant:** `core.policy.AUTHORIZED_BY_PREM = False` (Strictly Maintained)  
**Date:** November 22, 2026  

---

## 1. Executive Summary

Phase 2.7.5.2 achieved a critical milestone by achieving a 100% clean GitHub Actions CI run (`#44939078926`: 572 passed, 3 skipped). However, deep architectural inspection of the runtime inference service revealed severe ownership and truthfulness violations:

1. **Third-Party Pretrained Model Masquerading**: `webapp/model_service.py` automatically searched disk for third-party GGUF files (`Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf`, `Llama-3.2-1B-Instruct.Q4_K_M.gguf`) or defaulted to OpenAI's pretrained `gpt2` (124M), setting third-party weights as Moon AI's default runtime brain.
2. **Deceptive Hardcoded Knowledge Answers**: `webapp/main.py` included `_fast_knowledge_reply()`, intercepting user prompts with static canned responses (pancake recipes in Hindi/English, Grand Theft Auto VI game summaries, world capitals, simple math) pretending to be fast model generation.
3. **Deceptive Token Generation Counts**: Procedural web game generators (Snake, Chess, Cyber Racing) reported generating 850 or 920 neural tokens despite being static deterministic template strings.
4. **False Provenance Claims in Identity Reflexes**: Identity responses claimed Moon AI was "Qwen-2.5-Coder-7B (Super-AI)" created by Prem Gaikwad, falsely attributing Alibaba's pretrained model as Moon's own creation.

**Phase 2.7.6 resolves all runtime sovereignty and truthfulness violations.** Moon AI now strictly enforces that:
- Moon AI defaults strictly to its own sovereign student model: `moon-chat-pilot-v0`.
- Third-party models (Qwen, LLaMA, GPT-2) are quarantined as `external_pretrained_reference` (`available: False`, `sovereign_owned: False`, `status: "BLOCKED_EXTERNAL_PRETRAINED"`) and cannot be loaded as Moon's production engine.
- Deceptive hardcoded knowledge shortcuts are permanently removed from the chat route.
- Deterministic procedural utilities are explicitly tagged: `engine_type="deterministic_utility"`, `tokens_generated=0`.
- When no verified genuine strict sovereign checkpoint exists on disk (as is currently the case, since `moon_chat_pilot_v0.pt` is an unpromoted historical Policy B fixture), Moon AI honestly reports **`MODEL_NOT_READY`** with transparent diagnostic notices.

---

## 2. Absolute Safety Restrictions & Invariants Maintained

The following safety and architectural invariants are strictly enforced:
- **`core.policy.AUTHORIZED_BY_PREM = False`**: Unmodified and permanently locked.
- **Zero Real Model Training**: No training loops, no dataset streaming, no weight modifications.
- **Zero Production Backward Passes / Optimizers**: No `torch.optim` allocations or `.backward()` calls.
- **Zero Checkpoint Writes**: `models/checkpoints/chat/` was completely unmodified during Phase 2.7.6.
- **Pull Requests Untouched**: PR #1, PR #2, PR #3, and PR #4 remain unmerged.
- **Zero Git Rewrite**: No rebases, force-pushes, or blanket `git add .` commands.

---

## 3. Audited Files & Remediations

### 3.1 `webapp/model_service.py`
| Path / Method | Violation Found | Remediation Applied |
| :--- | :--- | :--- |
| `__init__` (lines 247–254) | Scanned disk for Qwen/LLaMA GGUF and set active model to Qwen, LLaMA, or GPT-2 | Removed auto-discovery; hard-coded default strictly to `moon-chat-pilot-v0` |
| `get_status()` | Returned `"READY"` if any model was loaded or `"READY (ON DEMAND)"` | Returns `"MODEL_NOT_READY"` unless a verified genuine strict sovereign checkpoint is loaded |
| `get_available_models()` | Listed Qwen/LLaMA as active available models with false creator attributions | Classified Qwen, LLaMA, GPT-2 as `external_pretrained_reference` (`available: False`, `sovereign_owned: False`) |
| `load_model()` | Allowed loading Qwen, LLaMA, and GPT-2 into memory as Moon's brain | Blocks all third-party models with `Runtime Sovereignty Violation` error |
| `switch_model()` | Allowed switching to third-party models | Fails closed when attempting to switch to external pretrained weights |
| `_generate_internal()` | Creator response claimed "I am Qwen-2.5-Coder (Super-AI)" | Replaced with truthful Moon AI Sovereign Core statement reflecting scratch initialization |

### 3.2 `webapp/main.py`
| Path / Endpoint | Violation Found | Remediation Applied |
| :--- | :--- | :--- |
| `_fast_knowledge_reply()` | Returned canned pancake recipes, GTA 6 summaries, capitals, and math | Permanently removed from the chat routing pipeline |
| Procedural Game Routes | Snake, Chess, and Racing reported generating 850/920 fake neural tokens | Explicitly classified as `engine_type="deterministic_utility"`, `tokens_generated=0` |
| Greeting Reflexes | "Hello", "hi", "good morning" returned as generic model outputs | Explicitly classified as `engine_type="system_reflex"`, `is_deterministic_utility=True` |
| Neural Model Dispatch | Could fall through to untrained generation or unverified fixtures | Checks positive genuine checkpoint verification; returns `ChatResponse` with `model_status="MODEL_NOT_READY"` |
| `health_check` (`/api/health`) | Did not report sovereign readiness or checkpoint verification | Updated to include `is_sovereign_ready=False` and `sovereign_checkpoint_path=None` |
| `switch_model` (`/api/switch-model`) | Permitted switching to unverified or third-party models | Rejects uncertified fixtures and external models with HTTP 403 |

### 3.3 `webapp/schemas.py`
Updated `ChatResponse` and `HealthResponse` to establish a strict, truthful response contract:
```python
class ChatResponse(BaseModel):
    reply: str
    model_name: str
    prompt_length: int
    tokens_generated: int
    device: str
    session_id: str
    engine_type: str = Field(default="sovereign_neural")       # "sovereign_neural", "deterministic_utility", "system_reflex", "system_notice", "web_retrieval"
    model_id: str = Field(default="moon-chat-pilot-v0")
    model_status: str = Field(default="MODEL_NOT_READY")
    is_neural_generation: bool = Field(default=False)
    is_deterministic_utility: bool = Field(default=False)
    is_web_retrieval: bool = Field(default=False)
    checkpoint_verified: bool = Field(default=False)
    provenance_status: str = Field(default="UNVERIFIED_OR_ABSENT")
    sources: List[str] = Field(default_factory=list)
```

### 3.4 `frontend/src/App.jsx`
- Default model set to `moon-chat-pilot-v0`.
- Catalog model titles updated:
  - `🌙 Moon Chat — Model Not Ready`
  - `🌐 Moon Web — Retrieval Available`
  - `⚙️ Utility Engine — Deterministic`
- Topbar model badge updated to display truthful status (`🌙 Moon Chat — Model Not Ready`).
- Hero subtitle updated: Removed claims attributing Qwen as Moon AI; updated to state strict sovereign random initialization (`STRICT_SOVEREIGN_RANDOM_INIT_ONLY`).

---

## 4. Route Classification Matrix

Every route in Moon AI is now classified into one of four mutually exclusive, transparent categories:

| Route / Trigger | Classification | `engine_type` | `is_neural_generation` | `is_deterministic_utility` | `tokens_generated` |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `/imagine`, `/image` | Procedural Art | `deterministic_utility` | `False` | `True` | `0` |
| `/video` | Procedural Video | `deterministic_utility` | `False` | `True` | `0` |
| Creator / Identity Query | Identity Reflex | `system_reflex` | `False` | `True` | Word count |
| Scientific / Biological KB | Knowledge Base | `deterministic_utility` | `False` | `True` | `0` |
| Snake, Chess, Racing Game | Interactive Canvas | `deterministic_utility` | `False` | `True` | `0` |
| Greetings ("hi", "hello") | Conversational Reflex | `system_reflex` | `False` | `True` | Word count |
| Diagnostics / Memory | System Utility | `system_notice` | `False` | `True` | `0` |
| Web Search / Citations | Retrieval Engine | `web_retrieval` | `False` | `False` | `0` |
| General Queries (No Model) | Model Status Notice | `system_notice` | `False` | `False` | `0` |
| Scratch / Sovereign Student | Real Model Inference | `sovereign_neural` | `True` | `False` | Actual tokens |

---

## 5. Test Suite Verification: 14 Core Scenarios

The dedicated Phase 2.7.6 test suite (`tests/test_phase_2_7_6_runtime_sovereignty.py`) certifies all 14 mandatory scenarios:

| # | Test Scenario | Method / Assertion | Status |
| :---: | :--- | :--- | :---: |
| **1** | Qwen cannot become default Moon model | `ms.active_model_name == "moon-chat-pilot-v0"` | **PASSED** |
| **2** | LLaMA cannot become default Moon model | `ms.active_model_name != "llama-3.2"` | **PASSED** |
| **3** | GPT-2 cannot become default Moon model | `ms.active_model_name != "gpt2"` | **PASSED** |
| **4** | Unverified checkpoint rejected | `is_verified_genuine_checkpoint() == False` | **PASSED** |
| **5** | Test fixture rejected for production | `allow_fixture=False` fails closed | **PASSED** |
| **6** | Historical Policy B checkpoint rejected | `TEACHER_STUDENT_RANDOM_INIT` rejected | **PASSED** |
| **7** | Verified eligible checkpoint byte invariant | SHA-256 before == SHA-256 after loading | **PASSED** |
| **8** | Missing eligible checkpoints produce `MODEL_NOT_READY` | `ms.get_status() == "MODEL_NOT_READY"` | **PASSED** |
| **9** | No silent pretrained fallback | `load_model("qwen-...")` returns `False` | **PASSED** |
| **10** | Hardcoded knowledge not labelled neural | General queries report `is_neural_generation=False` | **PASSED** |
| **11** | Deterministic utilities correctly labelled | Games return `tokens_generated=0`, `engine_type="deterministic_utility"` | **PASSED** |
| **12** | Web retrieval separately labelled | Citations preserved; no fabricated sources | **PASSED** |
| **13** | `AUTHORIZED_BY_PREM` remains `False` | Strict boolean invariant verified | **PASSED** |
| **14** | Zero optimizers, backward passes, or checkpoint writes | Verified zero new files in `models/checkpoints/` | **PASSED** |

**Phase 2.7.6 Test Suite Execution Result:**
```
Ran 14 tests in 32.774s
OK
```

---

## 6. Current Sovereignty State & Next Steps

### 6.1 Honest Model Readiness Statement
Moon AI currently has **NO** promoted production neural checkpoint on disk. The existing checkpoint `models/checkpoints/chat/moon_chat_pilot_v0.pt` was created under historical Policy B (`TEACHER_STUDENT_RANDOM_INIT`) and is properly quarantined as an unpromotable test fixture. Therefore:
- **Honest Runtime Status**: **`MODEL_NOT_READY`**
- **Sovereign Checkpoint Verified**: **`False`**
- **Production Inference on Third-Party Weights**: **`BLOCKED`**

### 6.2 Preflight Checklist Before Authorized Training
Before starting Phase 2.8 / Phase 3 (the first authorized scratch training run):
1. Keep PR #1, PR #2, PR #3, and PR #4 unmerged.
2. Confirm training dataset provenance under `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`.
3. Require explicit, deliberate owner authorization: `core.policy.AUTHORIZED_BY_PREM = True`.
4. Run preflight validation script (`scripts/preflight_pilot_training.py`).
5. Execute training strictly from Gaussian random scratch initialization with zero teacher weights.
