# Moon AI — Teacher Model Qualification Report
**Stage:** 2.1.5 — Local Teacher Qualification & Offline Corpus Foundation  
**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT` (Policy B)  
**Date:** November 4, 2026  

---

## 1. Executive Summary

This qualification report establishes the selection, verification, and hardware feasibility of the primary local offline teacher model for the Moon AI project. Under Policy B (`TEACHER_STUDENT_RANDOM_INIT`), the Moon student model is initialized strictly from random Gaussian scratch weights and acquires general capabilities via structured, verified teacher demonstrations.

To ensure reproducible, zero-API, sovereign distillation:
1. The teacher model runs **100% offline via local inference** (e.g. `llama.cpp` / GGUF).
2. The teacher weights reside in `models/teachers/checkpoints/` and **never cross into student directories** (`models/chat/`).
3. Model files are acquired via public HTTPS without any API keys, login credentials, or tokens.
4. All generated outputs undergo human-in-the-loop and automated validation (AST syntax checking, PII filtering, prompt injection screening, and duplication analysis) before entering production training manifests.

---

## 2. Candidate Specification: Qwen2.5-7B-Instruct-GGUF

### 2.1 Model Identity & Repository
* **Repository ID:** `Qwen/Qwen2.5-7B-Instruct-GGUF`
* **Official URL:** `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF`
* **Immutable Commit SHA:** `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
* **Publisher:** Alibaba Qwen Team
* **License:** `Apache-2.0` (Permissive open-source license permitting commercial use, modification, and distillation)

### 2.2 Quantization & Checkpoint Artifacts
The `Q4_K_M` (4-bit medium k-quant) format was selected as offering the optimal balance between perplexity retention (>99.2% of FP16 benchmark scores) and memory footprint:

| Shard File Name | Size (Bytes) | Size (GB) | Git-LFS SHA-256 Checksum |
|---|---|---|---|
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3,993,201,344 | 3.99 GB | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689,872,288 | 0.69 GB | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |
| **Combined Checkpoint Total** | **4,683,073,632** | **4.68 GB** | Verified via Multi-Shard GGUF Reader |

### 2.3 Model Architecture & Hyperparameters
* **Base Architecture:** `Qwen2ForCausalLM`
* **Parameters:** 7.61 Billion
* **Context Length:** 32,768 tokens (native) / up to 131,072 with YaRN
* **Vocabulary Size:** 152,064 tokens
* **Hidden Size ($d_{\text{model}}$):** 3,584
* **Intermediate Size ($d_{\text{ffn}}$):** 18,944
* **Transformer Layers:** 28
* **Attention Heads ($Q$):** 28
* **KV Attention Heads ($KV$):** 4 (Grouped-Query Attention with 7:1 ratio)
* **Head Dimension ($d_{\text{head}}$):** 128
* **RoPE Base Frequency:** 1,000,000
* **Chat Template:** ChatML
  ```
  <|im_start|>system
  {system_message}<|im_end|>
  <|im_start|>user
  {prompt}<|im_end|>
  <|im_start|>assistant
  ```

---

## 3. Hardware Requirements & VRAM Budget on NVIDIA Tesla T4

Google Colab provides an NVIDIA Tesla T4 GPU with 16,384 MB total GDDR6 memory (15,360 MB addressable in user space).

### 3.1 Mathematical VRAM Breakdown (Context Length: 4,096 tokens)

1. **Model Weights:**
   $$\text{VRAM}_{\text{weights}} = 4,683 \text{ MB}$$

2. **CUDA Context & Driver Runtime:**
   $$\text{VRAM}_{\text{runtime}} \approx 600 \text{ MB}$$

3. **Key-Value (KV) Cache (FP16):**
   $$\text{KV}_{\text{bytes}} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{context\_len} \times 2 \text{ bytes}$$
   $$\text{KV}_{\text{bytes}} = 2 \times 28 \times 4 \times 128 \times 4096 \times 2 = 234,881,024 \text{ bytes} \approx 235 \text{ MB}$$

4. **Working Scratch & Activation Buffer:**
   $$\text{VRAM}_{\text{scratch}} \approx 350 \text{ MB}$$

5. **Total Peak VRAM Allocation:**
   $$\text{VRAM}_{\text{peak}} = 4683 + 600 + 235 + 350 = 5,868 \text{ MB} \approx 5.73 \text{ GB}$$

6. **Headroom Analysis on Tesla T4 (15,360 MB):**
   $$\text{VRAM}_{\text{headroom}} = 15,360 \text{ MB} - 5,868 \text{ MB} = 9,492 \text{ MB} \ (61.8\% \text{ Free})$$

**Verdict:** The `Qwen2.5-7B-Instruct-GGUF` model in `Q4_K_M` quantization fits easily on a standard Colab Tesla T4 GPU with over 9.4 GB of reserve headroom. Out-Of-Memory (OOM) risk is negligible.

---

## 4. Candidate Comparison: Qwen2.5-7B-Instruct vs. DeepSeek-R1-Distill-Qwen-7B

| Evaluation Dimension | Qwen2.5-7B-Instruct (Selected) | DeepSeek-R1-Distill-Qwen-7B (Alternative) |
|---|---|---|
| **Primary Specialty** | General-purpose instruction, coding, multi-turn dialogue, tool use | Deep chain-of-thought mathematical & logical reasoning |
| **Output Style** | Direct, concise, structured responses following system constraints | Verbose reasoning trace (`<think>...</think>`) before final answer |
| **General Conversation** | ⭐⭐⭐⭐⭐ Natural, coherent, highly adaptable | ⭐⭐⭐ Often overthinks conversational prompts |
| **Code & Architecture** | ⭐⭐⭐⭐⭐ Superior HumanEval (85+), multi-language software design | ⭐⭐⭐⭐ Strong algorithmic coding, verbose solutions |
| **Tool Calling & JSON** | ⭐⭐⭐⭐⭐ Native JSON schema and tool execution tags | ⭐⭐⭐ Inconsistent adherence to rigid tool call syntax |
| **License** | Apache-2.0 | Apache-2.0 |
| **T4 VRAM Usage** | ~5.8 GB (Q4_K_M) | ~5.8 GB (Q4_K_M) |
| **Selection Rationale** | **Selected as Primary Teacher:** Moon AI requires a balanced foundation across all 10 curriculum domains. | Reserved as an optional reasoning specialist for pure math benchmarks. |

---

## 5. Offline Public Downloader & Security Governance

The download process enforces strict engineering rules:
1. **Zero Credential Dependency:** Requests are standard unauthenticated HTTPS GETs to official Hugging Face CDN endpoints.
2. **Resumable HTTP Range Protocol:** Downloader queries existing byte lengths and resumes using `Range: bytes={pos}-`.
3. **Magic Byte Verification:** Inspects the first 4 bytes (`GGUF` / `0x46554747`) to prevent saving HTML login redirects or 404 error pages.
4. **Storage Safety:** Requires $\ge 10 \text{ GB}$ of verified free disk space prior to beginning download.
5. **Git Protection:** Multi-GB weights reside in `models/teachers/checkpoints/`, which is ignored by `.gitignore`.
6. **Student Isolation:** Checkpoint verification asserts that no teacher files reside in or link to student model directories (`models/chat/`).

---

## 6. Verification Checklist

- [x] Official model card and revision verified (`bb5d59e06d9551d752d08b292a50eb208b07ab1f`).
- [x] Model license confirmed as `Apache-2.0`.
- [x] Checkpoint files, exact byte sizes, and Git-LFS SHA-256 hashes recorded.
- [x] Grouped-Query Attention (GQA) and KV cache memory analytically verified on T4.
- [x] ChatML formatting compatibility verified.
- [x] Directory isolation and `.gitignore` rules enforced.

