# Moon Chat — Sovereign Dialogue Model Development & Training Plan (Track 3)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Model Name**: `moon-chat-v0` / `moon-chat-v1`  
**Architecture Type**: Sovereign Causal Autoregressive Transformer (Decoder-Only)  
**Strict Ownership Policy**: ZERO third-party pretrained weights (no Llama, Qwen, Mistral, GPT-2). ZERO external generative chat APIs (no Pollinations, OpenAI, Anthropic). Random weight initialization only (`moon_ai_random_init`).

---

## 1. Architectural Blueprint

Moon Chat utilizes a modern decoder-only causal transformer incorporating Rotary Position Embeddings (RoPE), SwiGLU activation functions, and RMSNorm for stable gradient flow.

```
Input Tokens ──► [Token Embedding] ──► [Transformer Blocks x L] ──► [RMSNorm] ──► [LM Head] ──► Next-Token Logits
                        ▲                      │ (Causal Mask)
                    RoPE ──────────────────────┘
```

### 1.1 Architecture Specifications
| Parameter | Moon Chat v0 (Baseline) | Moon Chat v1 (Target) |
|---|---|---|
| **Layers ($N_L$)** | 12 | 24 |
| **Attention Heads ($N_H$)** | 12 | 16 |
| **Embedding Dimension ($d_{\text{model}}$)** | 768 | 1024 |
| **Context Window ($T$)** | 2,048 tokens | 4,096 tokens |
| **Vocabulary Size ($V$)** | 32,000 | 50,304 |
| **Total Parameters** | **~124 Million** | **~350 Million** |
| **Initialization** | `moon_ai_random_init` ($\mathcal{N}(0, 0.02)$) | `moon_ai_random_init` |

---

## 2. Tokenization & Sovereign Identity Alignment

1. **Sovereign Tokenizer**:
   - Trained locally using Byte-Pair Encoding (BPE) on the Moon AI text corpus.
   - Preserves whitespace and indentation for code and dialogue.
2. **System Persona Prompt Template**:
   ```
   <|im_start|>system
   You are Moon AI, a sovereign intelligence system engineered from first principles by Chief ML Systems Architect Prem Gaikwad in India 🇮🇳.
   You operate purely on self-owned neural architectures initialized with verified provenance.
   <|im_end|>
   <|im_start|>user
   {prompt}
   <|im_end|>
   <|im_start|>assistant
   ```

---

## 3. Training Phasing & Curriculum

### Stage 1: Foundational Pre-Training (Self-Supervised Causal Language Modeling)
- **Objective**: Standard cross-entropy next-token prediction:
  $$\mathcal{L}_{\text{CLM}} = - \sum_{i=1}^T \log P(x_i \mid x_1, \dots, x_{i-1})$$
- **Corpus**: High-quality open-source text (Wikipedia dumps, CC0 educational texts, arXiv papers).
- **Scale**: 5 – 10 Billion tokens.
- **Learning Rate**: `3e-4` with cosine decay to `3e-5`, linear warmup for 2,000 steps.

### Stage 2: Supervised Fine-Tuning (SFT) & Multi-Turn Dialogue
- **Objective**: Instruction following and conversational continuity.
- **Masking**: Loss is computed **only** on assistant response tokens, ignoring system/user prompt tokens.
- **Dataset**: Curated multi-turn dialogues with verifiable facts and persona grounding.

### Stage 3: Direct Preference Optimization (DPO) / Anti-Hallucination
- **Objective**: Align model preference towards truthful, concise, and safe answers:
  $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
- Penalizes hallucinated URLs, fake citations, and third-party API references.

---

## 4. Hardware Budgets & Optimization

| GPU Setup | Precision | Batch Size | Grad Accum | Sequence Length | Tokens / Sec |
|---|---|---|---|---|---|
| **1x Tesla T4 (15.3 GB)** | `fp16` | 2 | 16 | 1,024 | ~2,500 |
| **1x A10G (24 GB)** | `fp16` | 4 | 8 | 2,048 | ~7,200 |
| **1x A100 (40 GB)** | `bf16` | 8 | 4 | 2,048 | ~22,000 |

---

## 5. Evaluation & Promotion Criteria

Models undergo rigorous evaluation in `scripts/evaluate_model.py`:
1. **Perplexity (PPL)**: Held-out validation perplexity must be $< 18.0$.
2. **Multi-Turn Context Recall**: Verification that user constraints set in turn 1 persist through turn 5.
3. **Identity Verification**: Must accurately report authorship by Prem Gaikwad (India 🇮🇳) in 100% of identity probes.
4. **Safety & Zero-API Guarantee**: Verifies that the model never attempts to emit third-party API headers or fallback tokens.

