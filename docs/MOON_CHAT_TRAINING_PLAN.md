# Moon Chat — Sovereign Dialogue Model Development & Training Plan (Track 3)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Model Name**: `moon-chat-v0` / `moon-chat-v1`  
**Architecture Type**: Sovereign Causal Autoregressive Transformer (Decoder-Only)  
**Strict Ownership Policy**: ZERO third-party pretrained weights (no Llama, Qwen, Mistral, GPT-2). ZERO external generative chat APIs (no Pollinations, OpenAI, Anthropic). Random weight initialization only (`moon_ai_random_init`).

---

## 1. Architectural Truthfulness & Blueprint

### 1.1 Active Implementation (Stage 2.1 Pilot)
The active, verified code implementation in `model.py` and `training/chat/` is a **Decoder-Only Causal Transformer with Pre-LayerNorm, GELU activation, and learned positional embeddings (`wpe`)**:

```
Input Tokens ──► [Token Embedding (wte)] ──+──► [Transformer Blocks x L] ──► [LayerNorm (ln_f)] ──► [LM Head] ──► Next-Token Logits
                                           ▲             │ (Causal Self-Attention)
Positional Indices ──► [Position Embedding (wpe)] ───────┘
```

- **Layer Normalization**: Pre-LayerNorm placed before self-attention and MLP blocks for stable backpropagation.
- **Feedforward Layer**: Multi-layer perceptron with Gaussian Error Linear Unit (GELU) non-linearity.
- **Positional Representation**: Learned absolute positional embeddings (`wpe`) of length `block_size`.
- **Weight Tying**: Embedding matrix `wte` is tied with the output projection `lm_head`.
- **Weight Initialization**: All linear and embedding weights initialized strictly from $\mathcal{N}(0, 0.02^2)$ without external priors.

### 1.2 Target Scale Roadmap (Post-Pilot v1.0+)
For multi-billion parameter multi-node scaling (Track 3 v1.0+), Moon Chat plans migration to Rotary Position Embeddings (RoPE), SwiGLU activation functions, and RMSNorm. However, all active Stage 2.1 pilot checkpoints adhere truthfully to the verified Pre-LayerNorm GELU architecture in `model.py`.

### 1.3 Architectural Specifications
| Parameter | Moon Chat Pilot (25.7M) | Moon Chat Baseline (85.5M / 124M) | Moon Chat Target (350M) |
|---|---|---|---|
| **Layers ($N_L$)** | 8 | 12 | 24 |
| **Attention Heads ($N_H$)** | 8 | 12 | 16 |
| **Embedding Dimension ($d_{\text{model}}$)** | 512 | 768 | 1024 |
| **Context Window ($T$)** | 512 tokens | 1024 tokens | 2048 tokens |
| **Vocabulary Size ($V$)** | 512 (Byte BPE) | 512 (Byte BPE) | 32,000 (Byte BPE) |
| **Exact Parameters** | **25,690,624** | **85,568,256** | **~350 Million** |
| **Active Normalization** | Pre-LayerNorm | Pre-LayerNorm | RMSNorm (Roadmap) |
| **Active Activation** | GELU | GELU | SwiGLU (Roadmap) |
| **Positional Encoding** | Learned (`wpe`) | Learned (`wpe`) | RoPE (Roadmap) |
| **Initialization** | `moon_ai_random_init` ($\mathcal{N}(0, 0.02^2)$) | `moon_ai_random_init` | `moon_ai_random_init` |

---

## 2. Tokenization & Sovereign Identity Alignment

1. **Pure-Sovereign Byte-Level BPE Tokenizer (`SovereignTokenizer`)**:
   - Trained from scratch from raw byte statistics and pair frequency counting.
   - Base alphabet: 256 byte values (`b'\x00'` through `b'\xff'`) guaranteeing 100% vocabulary coverage with zero `<unk>` tokens.
   - Zero pretrained Hugging Face tokenizers or external vocabulary files.
2. **ChatML Boundary Protocol**:
   ```
   <|im_start|>system
   You are Moon AI, a sovereign intelligence system engineered from first principles by Chief ML Systems Architect Prem Gaikwad in India 🇮🇳.
   <|im_end|>
   <|im_start|>user
   {prompt}
   <|im_end|>
   <|im_start|>assistant
   {response}
   <|im_end|>
   ```

---

## 3. Supervised Fine-Tuning & Assistant-Only Loss Masking

In Supervised Fine-Tuning (SFT) for multi-turn dialogue, standard causal language modeling loss is modified with **Assistant-Only Response Loss Masking**:
$$\mathcal{L}_{\text{SFT}} = - \frac{1}{\sum_{i=1}^T \mathcal{M}_i} \sum_{i=1}^T \mathcal{M}_i \log P(x_i \mid x_1, \dots, x_{i-1})$$

Where the mask $\mathcal{M}_i$ is defined as:
$$\mathcal{M}_i = \begin{cases} 1 & \text{if } x_i \text{ is an assistant response token or closing } <|\text{im\_end}|>\backslash\text{n} \\ 0 & \text{if } x_i \text{ is a user prompt, system prompt, turn header, or padding} \end{cases}$$

- Tokens with $\mathcal{M}_i = 0$ are assigned target label `-1` (`ignore_index` in PyTorch `F.cross_entropy`).
- User and system prompt tokens do not contribute to response loss or gradient updates.
- Padding tokens are strictly masked out.
- Empty validation sets fail closed with a descriptive exception, preventing misleading zero validation loss metrics.

---

## 4. Hardware Sizing & Google Colab T4 Feasibility

- **GPU**: NVIDIA T4 (15.3 GB VRAM).
- **Peak VRAM Allocated (25.7M Pilot)**: ~1.46 GB (~9.5% of total VRAM).
- **VRAM Headroom**: ~13.84 GB (~90.5% free).
- **Realistic OOM Assessment**: Risk is **LOW** under standard micro-batch 4 and gradient accumulation 8. Memory risk elevates if micro-batch exceeds 32 or block size exceeds 2048 without gradient checkpointing.
- **Estimated Pilot Duration**: 500 iterations take ~45–60 seconds on T4 GPU (FP16 AMP).
