# Moon Image — Sovereign Model Development & Training Plan (Track 1)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Model Name**: `moon-image-v0` / `moon-image-v1`  
**Architecture Type**: Sovereign Latent Diffusion Model (LDM)  
**Strict Ownership Policy**: ZERO third-party pretrained weights (no SDXL, Flux, Midjourney, Stable Diffusion). All weights initialize from random distribution (`moon_ai_random_init`). ZERO external generation APIs.

---

## 1. Architectural Blueprint

The Moon Image engine operates in latent space to maximize training throughput and reduce GPU memory consumption by 8x compared to pixel-space diffusion.

```
Text Prompt ──► [MoonTextEncoder] ──► Token Embeddings [B, L, 256]
                                              │ (Cross-Attention)
Raw Image   ──► [MoonImageAutoencoder] ──► Latents [B, 4, H/8, W/8] ──► [MoonUNet2D] ──► Predicted Noise
                                                                               ▲
                                                      Timestep t ──────────────┘
```

### 1.1 Sub-module Specifications (Baseline v0 - 23.7M Parameters)
1. **Text Conditioning Module (`MoonTextEncoder`)**:
   - Parameters: **5,276,416**
   - Vocabulary: 1,000 subword/character tokens
   - Embedding Dimension: 256
   - Transformer Encoder: 4 causal/bidirectional attention blocks, 4 heads
2. **Latent Autoencoder (`MoonImageAutoencoder`)**:
   - Parameters: **8,032,587**
   - Encoder: 3 Downsampling Residual Stages (channels 64 -> 128 -> 256), stride 2
   - Decoder: 3 Upsampling Residual Stages with nearest-neighbor interpolation + conv
   - Latent Channels: 4
3. **Denoising Core (`MoonUNet2D`)**:
   - Parameters: **10,404,740**
   - UNet Backbone: 3 down-blocks, middle block with cross-attention, 3 up-blocks with skip connections
   - Time Embedding: Sinusoidal Positional Encoding projected to 256-dim via MLP

### 1.2 Scaling Roadmap to v1 (85M Parameters)
- Latent Channels: 4 -> 8
- UNet Channels: [64, 128, 256] -> [128, 256, 512]
- Attention Heads: 4 -> 8
- Native Resolution: 128x128 -> 256x256 -> 512x512

---

## 2. Loss Formulations & Training Objectives

The training objective is standard noise-prediction mean squared error (MSE) conditioned on text representations:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{z \sim \mathcal{E}(x), \epsilon \sim \mathcal{N}(0, I), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\phi(y)) \|_2^2 \right]$$

Where:
- $z = \mathcal{E}(x)$ is the latent representation from the Moon Autoencoder.
- $z_t = \sqrt{\bar{\alpha}_t} z + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is the noisy latent at timestep $t$.
- $\tau_\phi(y)$ is the text embedding produced by `MoonTextEncoder`.
- $\epsilon_\theta$ is the `MoonUNet2D` noise estimation network.

Autoencoder fine-tuning pairs L1 pixel reconstruction loss with Structural Similarity (SSIM):
$$\mathcal{L}_{\text{AE}} = \| x - \hat{x} \|_1 + \lambda (1 - \text{SSIM}(x, \hat{x}))$$

---

## 3. Data Curriculum & Phasing

| Phase | Steps | Target Resolution | Dataset Source | Objectives |
|---|---|---|---|---|
| **Phase 1: Geometric Primitives** | 0 – 20,000 | 128x128 | Procedural primitives, shapes, gradients | Learn latent encoding stability and basic color/shape alignment. |
| **Phase 2: Open Iconic Datasets** | 20,000 – 100,000 | 128x128 / 256x256 | Public domain open images (CC0, OpenData) | Learn textures, lighting, foreground-background separation. |
| **Phase 3: High-Detail Photorealism** | 100,000 – 300,000 | 256x256 / 512x512 | Curated high-resolution open datasets | Compositional fidelity, fine details, text adherence. |

---

## 4. Hardware & VRAM Optimization Budget

| Hardware Tier | Max Latent Resolution | Batch Size | Grad Accum | Precision | Steps/Day |
|---|---|---|---|---|---|
| **Tesla T4 (15.3 GB)** | 128x128 | 2 | 8 | `fp16` (AMP) | ~40,000 |
| **A10G (24 GB)** | 256x256 | 4 | 4 | `fp16` (AMP) | ~90,000 |
| **A100 (40 GB)** | 512x512 | 8 | 2 | `bf16` (AMP) | ~220,000 |

---

## 5. Checkpointing & Evaluation Rigor

1. **Atomic Checkpointing**: Optimizer state, GradScaler state, Learning rate scheduler, RNG states (PyTorch CPU/CUDA, NumPy, Python `random`), and manifest SHA-256 hash.
2. **Provenance Validation**: Every checkpoint must have `"provenance": "moon_ai_random_init"`.
3. **Benchmarking Suite**:
   - Validation Denoising Loss on held-out split (`datasets/image/manifest.jsonl`).
   - Sample Pixel Diversity Score (identifies mode collapse).
   - Fixed prompt panel generation at `evaluation/image/`.

