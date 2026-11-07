# Moon AI — Multimodal Platform Roadmap
### Specialist Engines, Unified Core Orchestration & Capability Progression

**Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT`  
**Date:** November 2026

---

## 1. THE VISION: BEYOND A TEXT CHATBOT

Moon AI is designed as a unified multimodal intelligence platform coordinating seven specialist engines under the command of **Moon Core**:

```
                              +--------------------+
                              |     MOON CORE      |
                              | Unified Dispatcher |
                              +--------------------+
                                         |
     +---------------+-------------------+---------------+---------------+
     |               |                   |               |               |
     v               v                   v               v               v
+----------+   +----------+        +----------+    +----------+    +-----------+
| MOON     |   | MOON     |        | MOON     |    | MOON     |    | MOON      |
| CHAT     |   | CODE     |        | IMAGE    |    | VIDEO    |    | DOCUMENTS |
+----------+   +----------+        +----------+    +----------+    +-----------+
                                                         |
                                                         v
                                                 [Higgsfield AI
                                                  Product Ambition]
```

---

## 2. SPECIALIST ENGINE SPECIFICATIONS & STATUSES

### 1. Moon Chat (`moon-chat-v0`)
- **Focus**: Conversation, deep reasoning, knowledge synthesis, instruction following.
- **Current Status**: `MODEL_TRAINING_PENDING` (Engineering Smoke Test Verified).
- **Architecture**: Causal Decoder-Only Transformer (Pre-LayerNorm, GELU, 25.7M parameters).

### 2. Moon Code (`moon-code-v0`)
- **Focus**: Repository refactoring, bug fixing, test synthesis, software architecture.
- **Current Status**: `ARCHITECTURE_DEFINED`.
- **Architecture**: Fill-In-The-Middle (FIM) Transformer (12L, 12H, 768D).

### 3. Moon Image (`moon-image-v0`)
- **Focus**: Text-to-image, editing, inpainting, variations, super-resolution.
- **Current Status**: `ARCHITECTURE_DEFINED`.
- **Architecture**: Latent Diffusion (MoonUNet2D + MoonTextEncoder + MoonImageAutoencoder, 23.7M parameters).

### 4. Moon Video (`moon-video-v0`)
- **Focus**: Text-to-video, image-to-video, camera control, character consistency, dynamic motion.
- **Product Ambition**: Comparable in quality and cinematic control to **Higgsfield AI**.
- **Current Status**: `ARCHITECTURE_DEFINED`.
- **Architecture**: Spatiotemporal Diffusion (MoonUNet3D + MoonVideoTextEncoder + MoonVideoAutoencoder, 17.0M parameters).

### 5. Moon Documents (`moon-documents-v0`)
- **Focus**: Programmatic generation and deep parsing of PDF, DOCX, PPTX, XLSX, CSV, and Markdown.
- **Current Status**: `PROVIDER_CONNECTED`.
- **Operation**: Local CPU processing with structured report generation templates.

### 6. Moon Research (`moon-research-v0`)
- **Focus**: Multi-source web search, citation verification, fact-checking, and objective summaries.
- **Current Status**: `PROVIDER_CONNECTED`.
- **Operation**: Agentic query synthesis and cross-verification.

### 7. Moon Core (`moon-core-v0`)
- **Focus**: User intent deconstruction, specialist engine dispatch, tool permission management, and unified aggregation.
- **Current Status**: `PROVIDER_CONNECTED`.

---

## 3. MULTIMODAL PROGRESSION PHASES

1. **Phase 1 (Current — Stage 2.1.3)**:
   - Establish capability registry, interface contracts, orchestrator routing, and teacher distillation pipeline.
2. **Phase 2 (Stage 2.1.4)**:
   - Curate 500+ teacher distillation records across all 10 curriculum domains.
   - Execute first legitimate 500-iteration Moon Chat pilot training on Google Colab T4.
3. **Phase 3 (Stage 2.2)**:
   - Train Moon Code pilot on synthetic and licensed code repositories.
   - Connect Moon Documents for automated PDF/PPTX briefing exports.
4. **Phase 4 (Stage 2.3)**:
   - Collect genuine visual image/video datasets with license tracking.
   - Train Moon Image 256x256 diffusion model and Moon Video 16-frame spatiotemporal preview model.

