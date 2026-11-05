# Moon AI — Pilot Model Evaluation & Promotion Audit Report
**Date:** 2026-11-05T16:49:58.674395+00:00  
**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Evaluated Model:** `moon_chat_scaled_v1`  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT` (Policy B)  
**Checkpoint Path:** `models/checkpoints/chat/moon_chat_scaled_v1.pt`  
**Checkpoint SHA-256:** `13b37a991119829391016e94d73ed71492edbefee8f6f97e3fce23b59d265c8b`  

---

## 1. Executive Summary & Policy Attestation
- **Random Scratch Initialization:** Verified pure Gaussian scratch initialization (zero pretrained or teacher weights transferred).
- **Offline Integrity:** Zero external API keys or cloud AI inference endpoints used.
- **Corpus Scale:** Evaluated across the blind test split (`scaled_chat_test.jsonl`) and the 12-category blind independent benchmark.

---

## 2. Quantitative Performance Metrics

| Metric | Measured Value | Target / Baseline |
|---|---|---|
| **Architecture** | 8 layers, 8 heads, 512 embedding, 512 block | `moon-chat-pilot-25m` |
| **Total Parameters** | 25,698,816 (25.70M) | Analytical matches PyTorch numel |
| **Test Loss (Cross-Entropy)** | **5.1858** | Evaluated on blind test split |
| **Test Perplexity** | **178.71** | Baseline random perplexity ~580 |
| **Composite Quality Score** | **0.00 / 100** | Rubric keyword & correctness coverage |
| **Non-Empty Response Rate** | **100.0%** | Threshold: 100.0% |
| **Code Syntax Accuracy** | **0.0%** | AST parsing on generated blocks |
| **Inference Throughput** | **2.42 tokens/sec** | CPU Local Engine |

---

## 3. Promotion Gate Decision
- **Promoted:** **False**
- **Decision Rationale:** Promotion rejected: Safety score 0.0% < 100.0% required.
- **Baseline Comparison:** Historical character-level scratch baseline composite score was 10.5. The model demonstrates loss convergence (5.1858) and 100% non-empty generation. Production promotion gate safeguards the pipeline until extensive GPU training steps are completed.

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
