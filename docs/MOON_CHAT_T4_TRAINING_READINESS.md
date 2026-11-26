# Moon AI — Moon Chat v0.1 T4 Training Readiness & Architecture Review

**Phase:** Phase 2.7.7 (Year-End Acceleration & Hardware Feasibility)  
**Date:** November 2026  
**Hardware Target:** Google Colab NVIDIA Tesla T4 (15.3 GB VRAM)  
**Model Target:** Moon Chat GPT (25.7M Parameters, Pure Gaussian Scratch Initialization)  
**Configuration File:** `training/chat/v0_1_t4_release_config.yaml`  
**Colab Notebook:** `notebooks/moon_chat_v0_1_t4_training.ipynb`  
**Active Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY` (`core.policy.AUTHORIZED_BY_PREM = False`)  

---

## 1. Executive Summary

This engineering review establishes the architectural specifications, tokenizer efficiency, hardware feasibility, and training schedules required to achieve a first genuinely trained Moon Chat v0.1 model on a single NVIDIA Tesla T4 GPU before December 31, 2026.

**Key Readiness Conclusions:**
1. **Architecture Confirmed:** The 25.7M parameter causal transformer (`8 layers, 8 heads, 512 embedding dim, Pre-LayerNorm, GELU, weight-tied embeddings, bias=False`) is mathematically verified at **`25,698,816` exact parameters**.
2. **Tokenizer Evaluated:** The 512-vocabulary byte-level BPE tokenizer (`models/chat/tokenizer.json`) is confirmed with 100% UTF-8 roundtrip symmetry. While it yields ~1.48–1.56 characters/token compression on English/Python, **it is recommended to retain `vocab_size=512` for the v0.1 release** to prevent parameter bloat and embedding table sparsity on small initial datasets (<500k tokens).
3. **T4 Memory & Headroom Verified:** Analytical peak VRAM is **1,312.1 MB** for micro-batch 4 and **1,632.1 MB** for micro-batch 8. Empirical live T4 benchmarking allocates only **601.37 MB** (4.1% of available 14.9 GB VRAM), leaving over **13.7 GB of safety headroom**. Out-of-memory (OOM) risk is rated **LOW**.
4. **Selected v0.1 Step Budget:** A 600-step cosine decay schedule with micro-batch 8 and gradient accumulation 4 (effective batch size: 32 sequences = 16,384 tokens/step) provides a total training budget of **9,830,400 tokens seen**, completing in **15 to 22 minutes** on a single T4 GPU.
5. **Execution Gated:** While the hardware and training configuration are fully validated, training execution remains strictly locked (`AUTHORIZED_BY_PREM = False` and `DATASET_NOT_READY`).

---

## 2. Model Architecture & Exact Parameter Count

The Moon Chat student architecture is a decoder-only autoregressive transformer following GPT-2 Pre-LayerNorm conventions with modern stability enhancements:

```
Input Tokens (x) 
  --> Embedding Lookup (wte: 512 x 512) + Learned Positional Encoding (wpe: 512 x 512)
  --> Dropout (0.1)
  --> Transformer Block x 8:
        [LayerNorm 1] --> Causal Self-Attention (8 heads, dim 512, Pre-LN) --> Residual (+)
        [LayerNorm 2] --> MLP (512 -> 2048 -> 512, GELU activation, Pre-LN) --> Residual (+)
  --> Final LayerNorm (512)
  --> Linear Projection Head (weight-tied with wte: 512 x 512)
  --> Logits (Batch, Seq_Len, 512)
```

### Exact Parameter Breakdown

Calculated via `training.chat.validate_hardware.calculate_exact_parameters()`:

| Component | Dimensions / Formula | Parameter Count |
| :--- | :--- | :---: |
| **Token Embeddings (`wte`)** | $512 \times 512$ | 262,144 |
| **Position Embeddings (`wpe`)** | $512 \times 512$ | 262,144 |
| **Block Attention (`c_attn` QKV)** | $512 \times (3 \times 512)$ | 786,432 per block |
| **Block Attention Projection (`c_proj`)** | $512 \times 512$ | 262,144 per block |
| **Block MLP Expansion (`c_fc`)** | $512 \times 2048$ | 1,048,576 per block |
| **Block MLP Contraction (`c_proj`)** | $2048 \times 512$ | 1,048,576 per block |
| **Block LayerNorms (`ln_1`, `ln_2`)** | $2 \times 512$ (weights only, `bias=False`) | 1,024 per block |
| **Per-Block Total** | $786,432 + 262,144 + 1,048,576 + 1,048,576 + 1,024$ | **3,146,752** |
| **All 8 Blocks** | $8 \times 3,146,752$ | **25,174,016** |
| **Final LayerNorm (`ln_f`)** | $512$ (weight only, `bias=False`) | 512 |
| **LM Head (`lm_head`)** | Weight-tied to `wte` | 0 (tied) |
| **Total Model Parameters** | $262,144 + 262,144 + 25,174,016 + 512$ | **25,698,816** (25.70M) |
| **Non-Embedding Parameters** | $25,174,016 + 512$ | **25,174,528** (25.17M) |

---

## 3. Sovereign Tokenizer Evaluation & Vocabulary Sizing

The sovereign tokenizer (`models/chat/tokenizer.json`) is an independently trained Byte-Pair Encoding (BPE) model:
- **Base vocabulary:** 256 raw byte tokens (ensures 100% coverage of arbitrary UTF-8 without `<unk>` out-of-vocabulary errors).
- **Special tokens (7 tokens):** `<|im_start|>`, `<|im_end|>`, `<|pad|>`, `<|endoftext|>`, `<|thought_start|>`, `<|thought_end|>`, `<|system_override|>`.
- **Learned merges:** 249 merges.
- **Total Vocabulary Size:** **512 tokens**.

### Compression & Context Window Trade-Off

Empirical testing across domains demonstrates the following compression ratios:
- **ASCII English Prose:** ~1.48 characters per token.
- **Python Source Code:** ~1.56 characters per token.
- **Multilingual / UTF-8:** ~0.89 characters per token.

With `block_size = 512`, the effective context length in characters is approximately:
$$\text{Context Length} \approx 512 \times 1.50 \approx 768\text{ characters}$$

### Architecture Recommendation: v0.1 vs v0.2+

| Option | Vocab Size | Embedding Params | Total Params | Effective Context (Chars) | Pros / Cons | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **v0.1 Release** | **512** | **524,288** | **25.70M** | ~768 chars | Maximum parameter allocation to transformer blocks; zero embedding sparsity on small corpora; preserves existing CI & hardware tests. | **Selected for v0.1** |
| **v0.2 Extended** | 2,048 | 2,097,152 | 27.27M | ~1,600 chars | Better compression (~3.2 chars/token); requires at least 2M characters of training text to prevent unobserved token embeddings. | Target for Phase 2.8+ |
| **v1.0 Standard** | 4,096 | 4,194,304 | 29.37M | ~2,000 chars | Standard small-LLM compression (~3.8 chars/token); causes severe embedding underfitting on small datasets (<1M tokens). | Deferred |

**Decision:** Retain `vocab_size=512` and `block_size=512` for the first Moon Chat v0.1 release.

---

## 4. Hardware Sizing & Empirical T4 Benchmarks

### Analytical VRAM Model

For FP16 mixed precision training with the AdamW optimizer:
- **FP16 Model Weights:** $25,698,816 \times 2\text{ bytes} = 49.0\text{ MB}$
- **FP16 Gradients:** $25,698,816 \times 2\text{ bytes} = 49.0\text{ MB}$
- **FP32 AdamW Optimizer States:** $25,698,816 \times 12\text{ bytes} = 294.1\text{ MB}$ (FP32 master weights + 1st moment + 2nd moment)
- **PyTorch / CUDA Driver Context:** $600.0\text{ MB}$
- **Activation Memory:**
  - Micro-batch 4: $\approx 320.0\text{ MB} \implies \text{Peak VRAM} \approx 1,312.1\text{ MB}$ (8.5% of 15.3 GB)
  - Micro-batch 8: $\approx 640.0\text{ MB} \implies \text{Peak VRAM} \approx 1,632.1\text{ MB}$ (10.6% of 15.3 GB)

### Empirical T4 GPU Benchmark

Independent verification on an NVIDIA Tesla T4 GPU (`data/reports/t4_empirical_benchmark.json`) measured:
- **Device:** Tesla T4 (Turing, Compute Capability 7.5, 14,912.69 MB total VRAM).
- **Live Peak Memory Allocated:** **601.37 MB**.
- **Live Peak Memory Reserved:** **612.00 MB** (4.1% utilization).
- **Available Headroom:** **14,300.69 MB** (95.9% free).
- **Step Latency:** **1,124.08 ms** per step (batch size 4, gradient accumulation 4, fp16 AMP enabled).
- **OOM Risk Assessment:** **LOW**.

---

## 5. Candidate Step Budgets & Selected v0.1 Configuration

Because the model occupies less than 11% of T4 VRAM, we can safely increase micro-batch size from 4 to 8. Combined with gradient accumulation 4, this yields an effective batch size of **32 sequences** ($32 \times 512 = 16,384\text{ tokens per optimizer step}$).

### Step Budget Comparison

| Schedule Option | Optimizer Steps | Batch / Accum | Effective Batch (Seqs) | Tokens Per Step | Total Tokens Seen | Estimated Wall-Clock (T4) | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sanity Pilot** | 300 | $8 \times 4$ | 32 | 16,384 | 4,915,200 | 8 – 12 min | Fast pre-release sanity check |
| **Selected v0.1 Release** | **600** | **$8 \times 4$** | **32** | **16,384** | **9,830,400** | **15 – 22 min** | **First genuine Moon Chat v0.1 release** |
| **Extended v0.2 Pilot** | 1,200 | $8 \times 4$ | 32 | 16,384 | 19,660,800 | 30 – 44 min | Multi-epoch convergence on scaled corpus |

### Hyperparameters (`training/chat/v0_1_t4_release_config.yaml`)

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  max_iters: 600
  eval_interval: 50
  eval_iters: 20
  save_interval: 100
  warmup_iters: 60
  lr_decay_iters: 600
  learning_rate: 0.0006
  min_lr: 0.00006
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  mixed_precision: "fp16"
  seed: 1337
  loss_masking: "assistant_only"
```

---

## 6. Training Pipeline Hardening in Phase 2.7.7

1. **Bug Fix in Learning Rate Schedule (`get_lr`):**  
   Fixed an edge-case zero-division and negative cosine ratio in `training/chat/train.py:100` when `max_iters <= warmup_iters` or `warmup_iters == 0`.
2. **Dedicated Configuration Validator (`training/chat/config_validator.py`):**  
   Provides a shared, importable validation suite used by `train.py`, `notebooks/moon_chat_v0_1_t4_training.ipynb`, and unit tests. Automatically validates dimensions, step budgets, split isolation, T4 VRAM safety, and production dataset truth.
3. **Dynamic Checkpoint Metadata:**  
   Replaced hardcoded dataset provenance strings with dynamic provenance analysis (`_train_audit = audit_single_manifest(...)`).
4. **Reproducible Colab Notebook:**  
   `notebooks/moon_chat_v0_1_t4_training.ipynb` includes preflight checks, GPU identification, artifact hashing, and an owner-controlled launch cell that enforces `AUTHORIZED_BY_PREM = True` and `DATASET_READY`.
