# Moon Code — Sovereign Coding Engine Development & Training Plan (Track 4)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Model Name**: `moon-code-v0` / `moon-code-v1`  
**Architecture Type**: Sovereign Causal Transformer with Fill-in-the-Middle (FIM)  
**Strict Ownership Policy**: ZERO third-party pretrained weights (no Codex, DeepSeek-Coder, StarCoder, Qwen-Coder). ZERO external code generation APIs. Random initialization only (`moon_ai_random_init`).

---

## 1. Architectural Architecture & FIM Transformations

To power inline code completion, full function generation, and automated refactoring, Moon Code implements **Fill-in-the-Middle (FIM)** training alongside causal left-to-right generation.

```
Document D = [Prefix, Middle, Suffix]
FIM Transformation ──► [ <|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> ] ──► Middle Prediction
```

### 1.1 Model Configurations
| Spec | Baseline v0 | Production v1 Target |
|---|---|---|
| **Parameters** | 124M (12 layers, 768 dim, 12 heads) | 500M (28 layers, 1280 dim, 20 heads) |
| **Context Length** | 4,096 tokens | 8,192 tokens |
| **Tokenization** | Sovereign BPE preserving indentations & symbols | Sovereign BPE (50,304 vocab) |
| **Special Tokens** | `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|file_sep|>` | Same |

---

## 2. Dataset Engineering & Deduplication

### 2.1 Permissive Open Source Corpus
- **Permissive Licenses Only**: Strictly filters for `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`. Proprietary, copyleft (GPL), or unverified code is stripped.
- **Deduplication**: MinHash LSH deduplication at 85% Jaccard similarity threshold to prevent memorization of boilerplate.
- **Language Distribution**:
  - Python: 40%
  - TypeScript / JavaScript: 25%
  - Rust & C++: 15%
  - SQL & Shell: 10%
  - Synthetic Execution Proofs & Tests: 10%

---

## 3. Sandboxed Execution-Based Evaluation Suite

Superficial keyword-matching benchmarks are prohibited. All coding benchmarks must execute code in isolated test harnesses:

1. **AST Syntax Tree Validation**:
   ```python
   import ast
   try:
       ast.parse(generated_code)
       is_valid = True
   except SyntaxError:
       is_valid = False
   ```
2. **Functional Correctness (Pass@k)**:
   Generated functions are injected into unit test suites and executed with strict timeout boundaries (2.0s per test).
3. **Multi-Turn Code Editing**:
   Tests the model's ability to locate and fix deliberate bugs across multi-file contexts.

---

## 4. Training Schedule & Hardware Budgets

| Hardware | Sequence Length | Batch Size | Grad Accum | Precision | Steps / Hour |
|---|---|---|---|---|---|
| **Tesla T4 (15.3 GB)** | 2,048 | 2 | 16 | `fp16` | ~1,200 |
| **A10G (24 GB)** | 4,096 | 4 | 8 | `fp16` | ~3,800 |
| **A100 (40 GB)** | 8,192 | 8 | 4 | `bf16` | ~12,500 |

---

## 5. Deployment & Inference Contracts

- When a verified trained checkpoint exists: Loaded directly via local PyTorch inference runtime.
- When checkpoint is missing or untrained: The engine truthfully reports `status: "MODEL_NOT_TRAINED"`. Under strict ownership rules, it **never** falls back to proprietary hosted coding APIs.

