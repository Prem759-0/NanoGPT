<div align="center">

# 🌙 MOON AI
### Sovereign Intelligence Core & Multimodal AI Platform
**Engineered from First Principles | Strict Sovereign Model Ownership & Teacher–Student Learning**

[![Ownership Policy](https://img.shields.io/badge/Ownership_Policy-TEACHER__STUDENT__RANDOM__INIT-8A2BE2?style=for-the-badge&logo=shield)](core/policy.py)
[![Student Weights](https://img.shields.io/badge/Student_Init-Random_Gaussian_Scratch-00C853?style=for-the-badge&logo=pytorch)](model.py)
[![Hardware Target](https://img.shields.io/badge/Hardware-Google_Colab_T4_(15.3GB)-00BCD4?style=for-the-badge&logo=googlecolab)](docs/MOON_AI_HARDWARE_AND_COST_PLAN.md)
[![Architect](https://img.shields.io/badge/Chief_ML_Architect-Prem_Gaikwad_%F0%9F%87%AE%F0%9F%87%B3-FF9933?style=for-the-badge)](https://github.com/Prem759-0)
[![GitHub Streak](https://img.shields.io/badge/GitHub_Streak-Active_Daily_Sync-success?style=for-the-badge&logo=github)](docs/DAILY_DEVELOPMENT_LOG.md)

**Chief ML Systems Architect & Project Owner**: Prem Gaikwad (India 🇮🇳)  
*Derived and sovereignly transformed from Andrej Karpathy's NanoGPT repository.*

---

</div>

## 1. EXECUTIVE OVERVIEW: THE MOON AI MISSION

**Moon AI** is an advanced, sovereign machine-learning ecosystem engineered to transcend traditional static language models and establish the foundation for a general-purpose, multimodal artificial intelligence platform.

Under the architectural leadership of **Prem Gaikwad**, Moon AI operates across seven specialized sovereign engines:
1. **Natural Multi-Turn Conversation & Contextual Understanding**
2. **Deep Mathematical, Logical & First-Principles Reasoning**
   - *Deterministic Math Engine*: Evaluates arithmetic, symbolic equations, and numeric equivalence without hallucination.
   - *AST Security Sandbox*: Parses untrusted code via Python AST to detect forbidden system calls prior to execution.
   - *Tamper-Evident WAL*: Preserves verified candidates in an append-only WAL with SHA-256 integrity validation.
3. **Software Engineering, Code Synthesis, Repository Debugging & Testing**
4. **Agentic Tool Selection & Autonomous Task Execution**
5. **Persistent User-Approved Memory (Isolated from Training Weights)**
6. **Latent Image Diffusion & Generative Video (with product ambition comparable to Higgsfield AI)**
   - *Spatiotemporal Convolutions*: Generates temporally coherent video frames in latent space.
   - *Motion Vector Guidance*: Enforces physical consistency across multi-second frame sequences.
   - *Mixed-Precision Optimization*: Operates within 15.3GB VRAM bounds for cloud and local execution.
7. **Automated Document Synthesis (PDF, DOCX, PPTX, XLSX, CSV, Markdown) & Evidence-Based Research**
   - *Office Document Artifacts*: Generates production-ready `.pdf`, `.docx`, `.pptx`, and `.xlsx` artifacts directly from chat.
   - *Evidence-Based Citation*: Links synthesized reports with verifiable source references.
   - *Structured Data Formats*: Exports clean tabular data in CSV, JSON, and Markdown formats.

All specialist engines are orchestrated by **Moon Core** — an intelligent dispatch system that analyzes user intent, deconstructs complex goals into multi-stage workflows, coordinates specialist engines, and returns unified results.

### 1.1 Moon Core Dispatch & Orchestration Engine
Moon Core functions as the central neural switchboard of the platform:
- **Intent Decomposition**: Classifies user queries into conversational, algorithmic, generative, or tool-assisted intents.
- **Specialist Routing**: Directs complex multi-step workflows to the optimal specialist engine.
- **Asymmetric Synthesis**: Aggregates multi-modal outputs into a unified, coherent response stream.

---

## 2. CORE OWNERSHIP POLICY: POLICY B (`TEACHER_STUDENT_RANDOM_INIT`)

Moon AI is governed by an absolute mathematical and architectural policy:

```
+---------------------------------------------------------------------------------------------------+
|                                  THE INVARIANT BOUNDARY                                           |
+---------------------------------------------------------------------------------------------------+
|  1. MOON STUDENT INITIALIZATION                                                                   |
|     Every Moon student neural network is initialized exclusively from random Gaussian scratch     |
|     weights: W ~ N(0, 0.02^2).                                                                    |
|     Third-party foundation model weights are STRICTLY PROHIBITED from entering student parameters.|
|                                                                                                   |
|  2. TEACHER DISTILLATION                                                                          |
|     Frontier foundation models serve strictly as external TEACHERS producing high-density         |
|     demonstrations, reasoning chains, code examples, and evaluation tests.                        |
|     Teacher parameters NEVER enter the Moon runtime or repository.                                |
|                                                                                                   |
|  3. AUTONOMOUS RUNTIME                                                                            |
|     Once trained, Moon student models execute inference 100% autonomously on local or edge        |
|     hardware without any runtime API dependencies.                                                |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Zero-Trust Data Security & Privacy Controls
Moon AI adheres to rigorous data privacy and memory isolation invariants:
- **Memory Training Isolation**: Persistent user memories are strictly isolated with permanent `eligible_for_training = False`.
- **Sensitive Data Redaction**: Automatic AST and regex filters redact PII, API tokens, and private credentials before memory storage.
- **Fail-Closed Consent Gate**: Unconsented background logging is disabled; all data collection requires explicit affirmative user consent.

### Technical Distinction Between Roles

| Architectural Concept | Definition | Invariant Constraint |
| :--- | :--- | :--- |
| **Teacher Model** | External frontier LLM (e.g. Kimi K3.1 Pro / Claude 3.5 Sonnet). | Used exclusively for data curation and benchmark grading; never loaded into student runtime. |
| **Student Model** | Moon proprietary neural network (Moon Chat, Moon Code). | Begins from 100% random Gaussian scratch weights (`random_gaussian_scratch`). |
| **Teacher Inference** | Querying the teacher model via authenticated API. | Credential-isolated, metered, strictly forbidden during student production inference. |
| **Student Training** | Optimization of student weights via cross-entropy loss. | Assistant-only loss masking, AdamW optimizer, FP16 mixed precision. |

---

## 3. MULTIMODAL SYSTEM ARCHITECTURE & 7 SPECIALIST ENGINES

Moon AI deconstructs monolithic AI design into seven decoupled, specialist engines orchestrated by a central coordinator:

```
                                  +------------------------------------+
                                  |         USER INTERFACE / API       |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |          MOON CORE DISPATCH        |
                                  |   (Intent Router & Coordinator)    |
                                  +--------+--------+--------+---------+
                                           |        |        |
         +---------------------------------+        |        +---------------------------------+
         |                                          |                                          |
         v                                          v                                          v
+-------------------+                      +-------------------+                      +-------------------+
|    MOON CHAT      |                      |    MOON REASON    |                      |     MOON CODE     |
| 25.7M Sovereign   |                      | Multi-Step Logic  |                      | Ast-Parsed Python |
| Pre-LN Transformer|                      | Proof Verifier    |                      | Sandbox Execution |
+-------------------+                      +-------------------+                      +-------------------+
         |                                          |                                          |
         +---------------------------------+        |        +---------------------------------+
                                           |        |        |
                                           v        v        v
                                  +------------------------------------+
                                  |        MOON SPECIALIST SUITE       |
                                  |                                    |
                                  |  * Moon Tools: Autonomous API Call |
                                  |  * Moon Memory: Isolated Storage   |
                                  |  * Moon Vision: Latent Diffusion   |
                                  |  * Moon Doc: PDF/DOCX/XLSX Engine  |
                                  +------------------------------------+
```

### The Seven Specialist Engines

1. **Moon Chat (25.7M Student Transformer)**: Pre-LayerNorm GPT architecture featuring GELU activations, tied input/output embeddings, and rotary/learned positional encodings, optimized for conversational flow.
2. **Moon Reason (Formal Inference Engine)**: Multi-step chain-of-thought engine breaking non-trivial mathematical and algorithmic challenges into verifiable sub-proofs.
3. **Moon Code (Sandboxed Code Synthesis)**: Python/TypeScript generator integrated with a deterministic execution sandbox to validate script output before returning it to the user.
4. **Moon Tools (Agentic Execution Engine)**: Strict JSON-schema-validated tool dispatch for web retrieval, arithmetic, file management, and external service calls.
5. **Moon Memory (Isolated Key-Value Context)**: Ephemeral and persistent session facts isolated from model weights to ensure zero weight contamination from user data.
6. **Moon Vision & Media (Generative Diffusion & Video)**: Dedicated latent diffusion pipelines designed for high-resolution graphics and progressive frame interpolation.
7. **Moon Document (Document Synthesis & Analytics)**: Pure-Python office document generator creating authentic `.pdf`, `.docx`, `.pptx`, and `.xlsx` artifacts directly from conversational output.

---

## 4. PHYSICAL HARDWARE TARGET & EMPIRICAL BENCHMARKS

Moon AI models are engineered specifically to train and execute on accessible cloud hardware, targeting the **NVIDIA Tesla T4** GPU (Google Colab free tier / T4 instance):

### Hardware Profile: NVIDIA Tesla T4

- **Compute Capability**: 7.5 (Turing Architecture)
- **VRAM Total**: 15,360 MB (15.0 GB)
- **Tensor Cores**: Second-Generation Turing Tensor Cores
- **Precision Mode**: Mixed Precision FP16 with PyTorch AMP (`torch.cuda.amp.autocast`)

### Parameter Sizing & Allocation Breakdown

| Model Track | Parameters (Exact) | Layers | Heads | Embedding Dim | Context Window | Peak VRAM (Train) | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Moon Chat Pilot** | **25.70M** (25,702,400) | 6 | 6 | 384 | 512 tokens | ~1.62 GB | Google Colab Tesla T4 |
| **Moon Chat Base** | **86.10M** (86,108,160) | 12 | 12 | 768 | 1024 tokens | ~5.84 GB | Google Colab Tesla T4 |
| **Moon Code Pilot** | **25.70M** (25,702,400) | 6 | 6 | 384 | 512 tokens | ~1.65 GB | Google Colab Tesla T4 |

---

## 5. DATA MANIFEST PROVENANCE & TRAINING CURRICULUM

Training datasets undergo strict verification, including format normalization, deduplication, and canonical LF SHA-256 integrity audits:

### Manifest Provenance & Accounting

| Dataset Manifest | Records | Training Windows (Seq 512) | Canonical LF SHA-256 Hash | Loss Masking Policy |
| :--- | :--- | :--- | :--- | :--- |
| `scaled_chat_train.jsonl` | **990 records** | **990 windows** | `c0f37ba515977e1ea542fed18c78ad01e91307b22fdb21dca9a4734da7d6323c` | Assistant-only (`user` tokens ignored) |
| `scaled_chat_val.jsonl` | **50 records** | **50 windows** | `b2f53140c0f417d59eabb5cce2163198308bc2ec0ffea575e95a3fc5cbdd27f7` | Assistant-only evaluation loss |
| `scaled_chat_test.jsonl` | **60 records** | **60 windows** | `26ba7b744d08409ff0306ee1b1ca20d1c9ef00ae011f008890cf5b6ff92aee28` | **Blind holdout** (Excluded from training/val) |

---

## 6. QUICK START & DEVELOPER GUIDE

### 6.1 Clone & Environment Setup

```bash
# Clone the official sovereign repository
git clone https://github.com/Prem759-0/NanoGPT.git
cd NanoGPT

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install sovereign dependencies
pip install -r requirements.txt
```

### 6.2 Running Automated Test Discovery

```bash
# Execute full repository unit and regression test suite
python -m unittest discover -s tests -p "test_*.py"
```

### 6.3 Running an Engineering Smoke Test

```bash
# Run 1-iteration smoke test with authentic Tesla T4 verification
python training/chat/train.py --config training/chat/phase_2_4_t4_pilot.yaml --smoke-test
```

---

## 7. PROJECT ROADMAP & RELEASE PHASES

```
                                    THE MOON AI ROADMAP
┌───────────────────────────────┐
│ Stage 2.0.1 - 2.0.7           │  Zero-Trust Checkpoints, Asymmetric Attestation, Registry Repair
│ [COMPLETED]                   │  
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│ Stage 2.1.1 - 2.1.3           │  Pre-LN GELU Architecture, Assistant Loss Masking, Policy B Migration,
│ [COMPLETED]                   │  Teacher Demonstration Integration, 512-Vocab BPE Tokenizer
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│ Stage 2.4 - 2.5               │  Hardware Identity Certification (T4), 8 Production Preflight Gates,
│ [ACTIVE PHASE]                │  Safe Atomic Checkpointing, 100-Iteration Pilot Verification
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│ Stage 3.0                     │  Sovereign Production Convergence: Moon Core Multimodal Unified Stack
│ [FUTURE]                      │  Full Multimodal Serving on HuggingFace Spaces & Edge Silicon
└───────────────────────────────┘
```

---

## 8. LICENSE & ATTRIBUTION

- **Chief ML Systems Architect & Owner**: Prem Gaikwad (India 🇮🇳)
- **Ownership Policy**: `TEACHER_STUDENT_RANDOM_INIT` (Pure Gaussian scratch initialization)
- **License**: Sovereign Open-Access License (Derived from MIT-licensed NanoGPT components by Andrej Karpathy).
