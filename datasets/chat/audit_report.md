# Moon AI — Sovereign Chat Dataset Audit Report

**Generated:** 2026-11-01T18:08:56.912238+00:00  
**Chief ML Systems Architect:** Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Model Track:** `moon-chat-v0`  
**Status:** **READY_FOR_PILOT_TRAINING**  

---

## 1. Executive Summary

| Metric | Count |
|---|---|
| **Total Genuine Records** | **10** |
| **Train Records (80%)** | 8 |
| **Validation Records (10%)** | 1 |
| **Test Records (10%)** | 1 |
| **Total Tokens (ChatML)** | **4,296** |
| **Mean Tokens / Conversation** | 429.6 |
| **Median Tokens / Conversation** | 431.0 |
| **Token Range [Min, Max]** | [311, 494] |
| **Total Conversation Turns** | 40 |

---

## 2. Cryptographic Integrity & Anti-Leakage Gate

- **Hash Verification:** 10 / 10 records verified matching canonical SHA-256 (100.0%).
- **Cross-Split Leakage:** **NO LEAKAGE (PASSED)**
  - Train/Val Overlap: 0
  - Train/Test Overlap: 0
  - Val/Test Overlap: 0
- **Synthetic Fixture Contamination:** 0 synthetic dev fixtures in production manifests (Strictly isolated in `synthetic_chat_dev.jsonl`).

---

## 3. License Distribution

| License | Count | Percentage |
|---|---|---|
| `Public Domain` | 4 | 40.0% |
| `CustomSovereign` | 2 | 20.0% |
| `CC0-1.0` | 2 | 20.0% |
| `MIT` | 2 | 20.0% |

---

## 4. Category Breakdown

| Category | Count | Percentage |
|---|---|---|
| `cryptography` | 1 | 10.0% |
| `mathematics` | 1 | 10.0% |
| `physics` | 1 | 10.0% |
| `science` | 1 | 10.0% |
| `machine_learning` | 1 | 10.0% |
| `computer_science` | 2 | 20.0% |
| `software_engineering` | 1 | 10.0% |
| `operating_systems` | 1 | 10.0% |
| `sovereign_identity` | 1 | 10.0% |

---

## 5. Sovereign Tokenizer Specifications

- **Architecture:** Pure Byte-Level BPE trained from scratch on sovereign corpus.
- **Vocabulary Size:** 512 tokens (including 256 base bytes, special chat boundaries, and learned subword merges).
- **Special Tokens:** `<|endoftext|>, <|pad|>, <|im_start|>, <|im_end|>, <|system|>, <|user|>, <|assistant|>`
- **Pretrained Independence:** Zero third-party Hugging Face tokenizers or external vocabularies.
