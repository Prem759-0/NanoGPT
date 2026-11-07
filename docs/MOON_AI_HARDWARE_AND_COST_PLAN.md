# Moon AI — Hardware & Cost Planning Blueprint
### Practical, Phased Infrastructure Sizing for Student Training, Teacher Distillation & Multimodal Engines

**Project Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT`  
**Date:** November 2026  
**Document Revision:** 1.0.0

---

## 1. EXECUTIVE SUMMARY & INFRASTRUCTURE PRINCIPLES

Moon AI is engineered to transition from a focused conversational pilot to a full-scale multimodal intelligence platform (Chat, Code, Image, Video, Documents, Research, and Core Orchestration).

To achieve maximum cost efficiency and accessibility for the project, infrastructure is planned across practical tiers:
1. **Tier 1 (Zero-Cost Baseline)**: Local Windows Laptop (development, unit tests, schema validation) + Google Colab Free T4 (pilot student training and engineering verification).
2. **Tier 2 (Affordable Cloud Compute)**: Google Colab Pro / Pro+ ($10 - $50/mo) with NVIDIA L4 (24GB) or A100 (40GB) for accelerated student training.
3. **Tier 3 (Production Cloud Serving)**: Hugging Face Spaces / Serverless GPU endpoints for public deployment.
4. **Tier 4 (Dedicated Multimodal Cluster)**: Multi-node A100/H100 clusters for large-scale video diffusion and foundation training.

---

## 2. TEACHER DISTILLATION COST & HARDWARE SIZING

### 2.1 Kimi K3.1 Pro Teacher API Usage (Recommended Distillation Strategy)

Using Kimi K3.1 Pro via authenticated API is orders of magnitude more affordable and practical than self-hosting a multi-trillion parameter model:

| Stage | Demonstrations Target | Estimated Total Tokens | API Rate (Input/Output avg) | Estimated Cost (USD) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 Pilot Dataset** | 500 records | ~250,000 tokens | \$0.005 / 1k tokens | **\$1.25** |
| **Phase 2 Expansion** | 2,500 records | ~1,500,000 tokens | \$0.005 / 1k tokens | **\$7.50** |
| **Phase 3 Baseline Scale** | 10,000 records | ~7,500,000 tokens | \$0.005 / 1k tokens | **\$37.50** |
| **Full SFT Corpus** | 50,000 records | ~40,000,000 tokens | \$0.005 / 1k tokens | **\$200.00** |

*Verdict*: Generating a rich, 10,000-example distillation dataset across all 10 curriculum domains costs under **\$40 USD**.

### 2.2 Local Teacher Inference Reality Check (Why a 3T Model Cannot Run on a 16GB T4)

> [!CAUTION]
> **Fundamental Physics & VRAM Invariant**:  
> A foundation model of the scale of Kimi K3.1 Pro (~hundreds of billions to 3T total mixture-of-experts parameters) requires between **350 GB to 1.5 TB of high-bandwidth memory** merely to hold the weights in memory at FP16/INT8.  
> Attempting to load this model on a single 16 GB Google Colab NVIDIA T4 is physically impossible. Even at aggressive 4-bit quantization (INT4), a 70B parameter model requires 40+ GB VRAM, and a 300B+ model requires 200+ GB VRAM.  
> Therefore, **API distillation is the only viable, cost-effective route** until official quantized checkpoints or local distilled adapters are formally evaluated.

---

## 3. MOON STUDENT MODEL TRAINING (MOON CHAT & MOON CODE)

### 3.1 Pilot Model (25.7M Parameters)
- **Architecture**: 8 layers, 8 heads, 512 embedding dimension, 512 context window.
- **Hardware Target**: Google Colab Free NVIDIA T4 (15.3 GB VRAM).
- **Analytical Peak VRAM**: **1,312.1 MB** (8.5% utilization, 14,047.9 MB headroom).
- **OOM Risk**: `LOW` (Zero risk with batch size 4 and gradient accumulation 8).
- **Training Time (500 iters)**: 45–60 seconds on T4 GPU; 25–30 minutes on CPU.
- **Cost**: **\$0.00 (Free Colab)**.

### 3.2 Baseline Model (86.1M Parameters)
- **Architecture**: 12 layers, 12 heads, 768 embedding dimension, 1024 context window.
- **Hardware Target**: Google Colab Free NVIDIA T4 (15.3 GB VRAM).
- **Analytical Peak VRAM**: **2,730.3 MB** (17.8% utilization, 12,629.7 MB headroom).
- **OOM Risk**: `LOW`.
- **Training Time (2,000 iters)**: ~8–12 minutes on T4 GPU.
- **Cost**: **\$0.00 (Free Colab)**.

### 3.3 Intermediate Model (124M Parameters)
- **Architecture**: 12 layers, 12 heads, 768 embedding dimension, 2048 context window.
- **Hardware Target**: Google Colab T4 or L4 (24GB).
- **Analytical Peak VRAM**: ~4.2 GB VRAM.
- **Training Time (10,000 iters)**: ~45–60 minutes on T4; ~15 minutes on L4.
- **Cost**: **\$0.00 (Free T4) or ~\$1.00 (Colab Pro compute units)**.

---

## 4. MOON IMAGE & MOON VIDEO COMPUTE SIZING

### 4.1 Moon Image (Latent Diffusion — 23.7M Parameters)
- **Inference**: Can run in ~1.8 GB VRAM at 256x256 resolution on a local laptop (NVIDIA GTX/RTX) or Free Colab T4.
- **Training (100k steps)**: Requires ~6–8 GB VRAM. Feasible on Colab T4 over 4–6 hours.
- **Cost**: **\$0.00 (Free Colab)** or **\$5–10 on RunPod/Vast.ai RTX 4090**.

### 4.2 Moon Video (Spatiotemporal Diffusion — Ambition Comparable to Higgsfield AI)
- **Understanding Higgsfield Scale**: Commercial video generation models (like Higgsfield, Runway Gen-2, or Sora) employ multi-billion parameter 3D spatiotemporal transformers and latent video autoencoders trained on millions of high-resolution video clips with optical flow supervision.
- **Moon Video Phase 1 (16-frame 128x128 preview clip)**:
  - Architecture: MoonUNet3D (17.0M parameters).
  - VRAM: ~8.5 GB VRAM on Colab T4.
  - Compute: ~8 hours training on Colab T4.
- **Moon Video Phase 2 (720p 4-second video generation at 24fps)**:
  - Requires dedicated multi-GPU setup: 4x NVIDIA A100 (80GB) or 4x RTX 4090.
  - Training Cost: ~$150 – $400 on decentralized GPU clouds (RunPod/Lambda).

---

## 5. DOCUMENT PROCESSING & RESEARCH ENGINES

- **Moon Documents (PDF, DOCX, PPTX, XLSX, CSV)**:
  - Runs 100% on CPU using standard Python parsing and synthesis libraries (`pypdf`, `python-docx`, `python-pptx`, `openpyxl`).
  - Compute: Negligible (< 200 MB RAM, < 0.5s per document).
  - Cost: **\$0.00**.
- **Moon Research (Web Search, Citation Verification, Summarization)**:
  - Operates via agentic search and Moon Chat summarization.
  - Compute: Standard CPU/light GPU inference.
  - Cost: Free search APIs (DuckDuckGo / SearXNG) or low-cost search APIs ($5 / 10,000 queries).

---

## 6. PRACTICAL ENVIRONMENT FEASIBILITY MATRIX

| Environment | Supported Workloads | Unsupported Workloads | Monthly Cost |
| :--- | :--- | :--- | :--- |
| **Local Windows Laptop (16GB RAM, CPU / light GPU)** | Unit testing, frontend build, schema validation, document parsing, small CPU dry runs. | Live full model training, large video rendering. | **\$0.00** |
| **Google Colab Free (1x NVIDIA T4 16GB)** | Moon Chat Pilot (25.7M) training, Baseline (86.1M) training, Image pilot inference. | Full 3T teacher inference, commercial 1080p video generation. | **\$0.00** |
| **Google Colab Pro ($10/mo, L4 24GB / A100 40GB)** | 124M-350M Chat/Code training, Moon Image SFT training, short video clip training. | Large distributed multi-node pre-training. | **\$10.00 / mo** |
| **Hugging Face Spaces (Free / Basic GPU)** | Web backend deployment, live user chat demo, API hosting. | Heavy batch training runs. | **\$0.00 - \$9.00 / mo** |
| **Decentralized GPU Cloud (RunPod / Vast.ai RTX 4090)** | High-throughput video generation, fast image diffusion training. | None (highly cost-effective at $0.40/hr). | **Pay-as-you-go ($10-$50)** |

---

## 7. SUMMARY OF RECOMMENDATIONS

1. **Do not attempt to self-host Kimi K3.1 Pro weights on a T4 GPU.** Use the Kimi API integration for teacher demonstration generation.
2. **Execute Moon Chat pilot student training on Google Colab Free T4** once the 500+ record teacher distillation dataset is curated.
3. **Keep document processing completely local and CPU-bound** ($0 cost).
4. **Advance Moon Video systematically**: master 16-frame architectural previews before renting expensive multi-A100 clusters.

