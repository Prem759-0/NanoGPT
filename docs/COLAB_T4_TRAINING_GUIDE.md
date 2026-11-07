# Moon AI — Google Colab NVIDIA T4 Hardware Validation Runbook

**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT` (Policy B)  
**Active Branch:** `feature/stage-2-3-multimodal-ui-sandbox-and-accelerated-training`  
**Execution Phase:** Phase 2.4.5 — Physical Hardware Benchmark & Validation  
**Target Hardware:** Google Colab Free Tier — NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM, Turing Architecture, Compute Capability 7.5)  

---

> [!CAUTION]
> **STRICT STOP CONDITION: HARDWARE VALIDATION ONLY**  
> This runbook is strictly for **empirical hardware validation** of the NVIDIA T4 GPU runtime.  
> **DO NOT start the 100-step training run.**  
> **DO NOT train a production model.**  
> **DO NOT execute `training/chat/train.py`.**  
> Model training is reserved for Phase 2.5 and requires Prem Gaikwad's explicit review and approval of the empirical validation report.

---

## 1. Objectives & Hardware Governance

Before committing computational resources to student model training, Moon AI enforces strict pre-flight hardware certification:

1. **Physical T4 Device Authentication:** Confirms the runtime environment reports an authentic NVIDIA Tesla T4 GPU with Compute Capability **7.5** and at least **14,000 MB** physical VRAM. Non-T4 GPUs (A100, L4, V100, CPU) fail closed.
2. **FP16 AMP & GradScaler Integrity:** Verifies automatic mixed precision (AMP) autocast and numerical scaling without underflow or overflow.
3. **Execution Dynamics:** Validates forward/backward passes, 4-step gradient accumulation, gradient norm clipping (`max_norm=1.0`), and AdamW optimizer parameter updates using synthetic test tensors.
4. **Numerical Stability:** Detects and guards against `NaN` or `Inf` anomalies across losses, gradients, and model parameters.
5. **Bitwise Checkpoint Roundtrip:** Verifies isolated temporary checkpoint save, load, and 100% bit-for-bit parameter reconstruction using secure `weights_only=True` deserialization.
6. **Empirical VRAM Accounting:** Records actual peak allocated and reserved VRAM measurements on physical silicon (differentiating empirical results from analytical planning estimates).

---

## 2. Analytical Planning Budget vs. Empirical Validation

| Dimension | Analytical Planning Estimate (`data/reports/t4_compatibility_benchmark.json`) | Empirical GPU Validation (`data/reports/t4_empirical_benchmark.json`) |
| :--- | :--- | :--- |
| **Execution Environment** | Analytical formula / CPU simulation | Physical NVIDIA Tesla T4 in Google Colab |
| **Model Parameters** | 25,698,816 parameters (25.7M student) | 25,698,816 parameters (25.7M student) |
| **Estimated Weights (FP16)** | 51.4 MB | Measured directly on device |
| **Estimated Optimizer States (FP32)** | 308.4 MB (master + $m_t$ + $v_t$) | Measured directly on device |
| **Estimated CUDA Overhead** | 600.0 MB | Measured directly on device |
| **Total VRAM Planning Budget** | ~1,312 MB (~1.3 GB) | Measured peak allocated / reserved |
| **Expected Safety Headroom** | > 13.8 GB (> 90% free) | Measured empirical headroom |
| **Status Classification** | `COMPATIBILITY_ESTIMATE` (is_empirical: false) | `T4_EMPIRICAL_BENCHMARK_VERIFIED` (is_empirical: true) |

---

## 3. Step-by-Step Google Colab Execution Runbook

Open [Google Colab](https://colab.research.google.com/), create a new notebook, navigate to **Runtime $\rightarrow$ Change runtime type**, select **T4 GPU**, and execute the following cells sequentially:

### Cell 1: Hardware & Driver Verification
```bash
# Check physical GPU device and driver allocation
!nvidia-smi
```
*Expected: Tesla T4, 15,360 MiB (or ~15,109 MiB) VRAM, Driver Version 535+.*

---

### Cell 2: Clone Repository & Checkout Active Feature Branch
```bash
# Clone repository and checkout the active Stage 2.3 / Phase 2.4 feature branch
!git clone https://github.com/Prem759-0/NanoGPT.git
%cd NanoGPT

# Checkout the active feature branch (NOT obsolete stage branches)
!git checkout feature/stage-2-3-multimodal-ui-sandbox-and-accelerated-training

# Verify branch and latest commit
!git branch --show-current
!git log -1 --oneline
```
*Expected branch: `feature/stage-2-3-multimodal-ui-sandbox-and-accelerated-training`.*

---

### Cell 3: Environment Setup (Preserving Colab Pre-installed CUDA PyTorch)
```bash
# Install required dependencies WITHOUT upgrading or overwriting Colab's pre-installed CUDA PyTorch
!pip install -q pyyaml pydantic fastapi uvicorn httpx cryptography tiktoken

# Verify PyTorch, CUDA, and cuDNN environment
import sys
import torch

print(f"Python Version:   {sys.version.split()[0]}")
print(f"PyTorch Version:  {torch.__version__}")
print(f"CUDA Available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:      {torch.cuda.get_device_name(0)}")
    print(f"Compute Cap:      {torch.cuda.get_device_capability(0)}")
    print(f"CUDA Version:     {torch.version.cuda}")
    print(f"cuDNN Version:    {torch.backends.cudnn.version()}")
    print(f"Device VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**2):,.1f} MB")
```

---

### Cell 4: Execute Dedicated T4 Empirical Hardware Validation
```bash
# Execute the strict empirical validation benchmark (Hardware tests only; zero dataset training)
!python scripts/validate_t4_empirical.py
```
*What this validates:*
- Enforces strict Turing Tesla T4 device name matching.
- Enforces compute capability exactly `7.5`.
- Enforces physical VRAM $\ge 14,000\text{ MB}$.
- Instantiates 25.7M parameter Moon Chat architecture from random Gaussian scratch ($\mathcal{N}(0, 0.02^2)$).
- Tests FP16 AMP autocast and GradScaler under 4-step gradient accumulation.
- Tests gradient norm clipping and AdamW optimizer updates.
- Verifies absence of NaN/Inf values.
- Performs isolated temporary checkpoint roundtrip with `weights_only=True` bitwise verification.
- Exports report to `data/reports/t4_empirical_benchmark.json`.

---

### Cell 5: Inspect and Download Empirical Validation Artifacts
```python
# Inspect canonical empirical benchmark report
!cat data/reports/t4_empirical_benchmark.json
!sha256sum data/reports/t4_empirical_benchmark.json

# Download report to local machine for cryptographic attestation and audit
from google.colab import files
files.download('data/reports/t4_empirical_benchmark.json')
```

---

### Cell 6: Run Regression & Checkpoint Security Suites
```bash
# Run Phase 2.4 hardware identity and deserialization security test suite
!python -m unittest tests/test_stage_2_4_hardware_identity_and_checkpoint_security.py

# Run Phase 2.4 checkpoint isolation and integrity test suite
!python -m unittest tests/test_stage_2_4_checkpoint_isolation_and_integrity.py
```
*Expected: All 12 hardware identity tests and all 9 checkpoint isolation tests pass cleanly.*

---

## 4. Policy B Compliance & Dataset Isolation

During this validation phase:
1. **Zero Pretrained Weights:** The student model is initialized strictly from random Gaussian scratch ($\mathcal{N}(0, 0.02^2)$).
2. **Zero Teacher Checkpoint Weights:** No teacher weights or embeddings are transferred or loaded.
3. **Hardware Test Inputs Only:** The gradient accumulation test uses synthetic token IDs solely to exercise CUDA execution units. No training data from `datasets/manifests/` is consumed or updated.
4. **Isolated Checkpointing:** Temporary checkpoint roundtrip creates and verifies an ephemeral file inside a temporary directory and deletes it immediately in a `finally` block. Existing checkpoints on disk are untouched.
5. **Fail-Closed Gate:** If executed on CPU or non-T4 GPU, the script exits with non-zero code and reports `CUDA_UNAVAILABLE` or `HARDWARE_MISMATCH`, never falsifying empirical verification.

---

## 5. Next Steps Following Hardware Validation

1. **Save Output Artifacts:** Preserve `t4_empirical_benchmark.json` and download it locally.
2. **Inspect Measured VRAM:** Verify that peak reserved VRAM is $\le 2.0\text{ GB}$ and headroom is $\ge 13.0\text{ GB}$.
3. **Submit Report to Prem Gaikwad:** Present the empirical Colab run log, GPU details, and report SHA-256 for architectural review.
4. **Phase 2.5 Authorization:** Only after Prem's explicit sign-off will Phase 2.5 (the 100-step student pilot training run on the 1,100-record curriculum) be initiated.
