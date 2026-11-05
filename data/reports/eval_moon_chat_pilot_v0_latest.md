# Moon AI — Pilot Model Evaluation & Promotion Audit Report
**Date:** 2026-11-05T10:53:54.055754+00:00  
**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Evaluated Model:** `moon-chat-pilot-v0`  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT` (Policy B)  
**Checkpoint Path:** `D:\Downloads\linkden post\ai\models\checkpoints\chat\moon_chat_pilot_v0.pt`  
**Checkpoint SHA-256:** `f20c1c6b5565e06f250c13191bbb898dc98114eb4eb9c4db4d260a911ab5ae85`  

---

## 1. Executive Summary & Policy Attestation
- **Random Scratch Initialization:** Verified pure Gaussian scratch initialization (zero pretrained or teacher weights transferred).
- **Offline Integrity:** Zero external API keys or cloud AI inference endpoints used.
- **Corpus Scale:** Evaluated across the 52-record blind test split (`expanded_chat_test.jsonl`) and the 12-category blind independent benchmark.

---

## 2. Quantitative Performance Metrics

| Metric | Measured Value | Target / Baseline |
|---|---|---|
| **Architecture** | 8 layers, 8 heads, 512 embedding, 512 block | `moon-chat-pilot-25m` |
| **Total Parameters** | 25,698,816 (25.70M) | Analytical matches PyTorch numel |
| **Test Loss (Cross-Entropy)** | **5.2110** | Evaluated on blind test split |
| **Test Perplexity** | **183.27** | Baseline random perplexity ~580 |
| **Composite Quality Score** | **0.00 / 100** | Rubric keyword & correctness coverage |
| **Non-Empty Response Rate** | **100.0%** | Threshold: 100.0% |
| **Code Syntax Accuracy** | **0.0%** | AST parsing on generated blocks |
| **Inference Throughput** | **2.50 tokens/sec** | CPU Local Engine |

---

## 3. Promotion Gate Decision
- **Promoted:** **False**
- **Decision Rationale:** Promotion rejected: Safety score 0.0% < 100.0% required.
- **Baseline Comparison:** Historical character-level scratch baseline composite score was 10.5. The pilot model demonstrates successful loss convergence (5.2110) and 100% non-empty generation. Production promotion gate safeguards the pipeline until extensive GPU training steps are completed.

---

## 4. Benchmark Category Breakdown
- **general_knowledge**: 0.0%
- **unseen_conversation**: 0.0%
- **multi_turn_instruction**: 0.0%
- **math_reasoning**: 0.0%
- **logical_reasoning**: 0.0%
- **programming**: 0.0%
- **debugging**: 0.0%
- **tool_selection**: 0.0%
- **structured_output**: 0.0%
- **long_context**: 0.0%
- **safety_uncertainty**: 0.0%
- **document_task_planning**: 0.0%
