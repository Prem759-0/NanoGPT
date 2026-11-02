# Moon AI — Sovereign Chat Dataset Audit Report

**Generated:** 2026-11-02T16:06:18.076090+00:00  
**Chief ML Systems Architect:** Prem Gaikwad (Chief ML Systems Architect, India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Model Track:** `moon-chat-v0`  
**Status:** **READY_FOR_PILOT_TRAINING**  

---

## 1. Executive Summary

| Metric | Count |
|---|---|
| **Total Genuine Records** | **22** |
| **Train Records (80%)** | 18 |
| **Validation Records (10%)** | 2 |
| **Test Records (10%)** | 2 |
| **Total Tokens (ChatML)** | **9,445** |
| **Mean Tokens / Conversation** | 429.32 |
| **Median Tokens / Conversation** | 431.5 |
| **Token Range [Min, Max]** | [304, 576] |
| **Total Conversation Turns** | 88 |

---

## 2. Cryptographic Integrity & Anti-Leakage Gate

- **Hash Verification:** 22 / 22 records verified matching canonical SHA-256 (100.0%).
- **Cross-Split Leakage:** **NO LEAKAGE (PASSED)**
  - Train/Val Overlap: 0
  - Train/Test Overlap: 0
  - Val/Test Overlap: 0
- **Synthetic Fixture Contamination:** 0 synthetic dev fixtures in production manifests (Strictly isolated in `synthetic_chat_dev.jsonl`).

---

## 3. License Distribution

| License | Count | Percentage |
|---|---|---|
| `CustomSovereign` | 6 | 27.3% |
| `CC-BY-SA-4.0` | 15 | 68.2% |
| `CC0-1.0` | 1 | 4.5% |

---

## 4. Category Breakdown

| Category | Count | Percentage |
|---|---|---|
| `beginner_educational` | 1 | 4.5% |
| `physics` | 1 | 4.5% |
| `computer_science` | 3 | 13.6% |
| `cryptography` | 1 | 4.5% |
| `mathematics` | 3 | 13.6% |
| `programming_fundamentals` | 2 | 9.1% |
| `distributed_systems` | 1 | 4.5% |
| `instruction_following` | 1 | 4.5% |
| `science` | 2 | 9.1% |
| `software_engineering` | 1 | 4.5% |
| `general_conversation` | 2 | 9.1% |
| `operating_systems` | 1 | 4.5% |
| `sovereign_identity` | 2 | 9.1% |
| `reasoning_explanations` | 1 | 4.5% |

---

## 5. Sovereign Tokenizer Specifications

- **Architecture:** Pure Byte-Level BPE trained from scratch on sovereign corpus.
- **Vocabulary Size:** 512 tokens (including 256 base bytes, special chat boundaries, and learned subword merges).
- **Special Tokens:** `<|endoftext|>, <|pad|>, <|im_start|>, <|im_end|>, <|system|>, <|user|>, <|assistant|>`
- **Pretrained Independence:** Zero third-party Hugging Face tokenizers or external vocabularies.
