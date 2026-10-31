# Moon Video — Sovereign Spatiotemporal Model Development & Training Plan (Track 2)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Model Name**: `moon-video-v0` / `moon-video-v1`  
**Architecture Type**: Sovereign Spatiotemporal Latent Diffusion Model  
**Strict Ownership Policy**: ZERO third-party pretrained weights (no ModelScope, AnimateDiff, Kling, Runway, CogVideo). ZERO external video generation APIs. Random initialization only (`moon_ai_random_init`).

---

## 1. Architectural Architecture

Moon Video operates across spatiotemporal dimensions $[B, C, F, H, W]$ by factorizing spatial 2D convolutions and temporal 1D convolutions + self-attention.

```
Text & Motion Prompt ──► [MoonVideoTextEncoder] ──► Text Context [B, L, 256]
Camera Motion Tag    ──► [CameraMotionEmbedding] ──► Motion Vector [B, 1, 256]
                                                           │ (Cross-Attention)
Raw Video [B, 3, F, H, W] ──► [3D Latent Autoencoder] ──► [MoonUNet3D] ──► Predicted Noise
                                    │                             ▲
                          Latents [B, 4, F, H/8, W/8] ────────────┘
```

### 1.1 Baseline v0 Sub-module Specifications (17.0M Parameters)
1. **Text & Motion Conditioning**:
   - Parameters: **5,278,976**
   - Text Encoder: 4 transformer layers, 256 hidden dimension, 4 attention heads.
   - Camera Motion Embedder: 12 camera motion classes (`static`, `orbit_left`, `orbit_right`, `dolly_in`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `crane_up`, `crane_down`) mapped to 256-dimensional vectors.
2. **Spatiotemporal Autoencoder (`MoonVideoAutoencoder`)**:
   - Parameters: **4,521,519**
   - Spatial downsampling (8x) paired with temporal feature alignment.
3. **Spatiotemporal Denoising Core (`MoonUNet3D`)**:
   - Parameters: **7,208,516**
   - Interleaved 2D Spatial Convolutions + 1D Temporal Convolutions (`TemporalConvBlock`).
   - Temporal Self-Attention (`TemporalSelfAttention`) attending across frame sequences $[B \times HW, F, C]$ to ensure optical flow stability.

### 1.2 Scaling Roadmap to v1 (65M Parameters)
- Frame count: 8 frames -> 16 frames -> 24 frames
- Native Resolution: 128x128 -> 256x256
- Video Autoencoder: 3D Causal Autoencoder with temporal compression factor of 4.

---

## 2. Real Video Data Integrity & Dataset Pipeline

### 2.1 Strict Separation of Real Video vs Synthetic Data
- **Production Video Training**: Must ONLY use genuine video containers (`.mp4`, `.webm`) containing real frame progressions, timestamps, and camera motions. The pipeline strictly halts with `ValueError` if 0 real video records exist.
- **Synthetic Development Dataset**: Quarantined strictly for fast unit testing and CI pipeline validation.

### 2.2 Key Video Dataset Fields
```json
{
  "id": "real_vid_sample_01",
  "media_path": "data/videos/sample_real_video.mp4",
  "caption": "Cinematic camera orbiting a gleaming hypercar on a mountain pass",
  "camera_motion": "orbit_left",
  "fps": 8.0,
  "duration_sec": 3.0,
  "num_frames": 24,
  "sha256": "4b92...",
  "split": "train",
  "license": "CustomSovereign"
}
```

---

## 3. Training Objectives & Loss Formulation

The spatiotemporal objective optimizes 3D noise estimation and temporal consistency:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{3D\_noise}} + \lambda_{\text{flow}} \mathcal{L}_{\text{temporal\_coherence}}$$

1. **3D Denoising Loss**:
   $$\mathcal{L}_{\text{3D\_noise}} = \frac{1}{B \cdot F \cdot C \cdot H \cdot W} \sum_{b, f, c, h, w} (\epsilon - \epsilon_\theta(z_t, t, c_{\text{text}}, c_{\text{motion}}))^2$$
2. **Temporal Coherence Regularization**:
   Penalizes unnatural inter-frame flickering by aligning temporal gradients between predicted denoised latents and target latents:
   $$\mathcal{L}_{\text{temporal\_coherence}} = \| (z_t^{f} - z_t^{f-1}) - (z_{t-1}^f - z_{t-1}^{f-1}) \|_1$$

---

## 4. Hardware Budgets & Optimization

| GPU Model | Video Resolution | Frames | Batch Size | Grad Accum | Precision | VRAM Used |
|---|---|---|---|---|---|---|
| **Tesla T4 (15.3 GB)** | 128x128 | 8 frames | 1 | 8 | `fp16` | 6.8 GB |
| **A10G (24 GB)** | 128x128 / 256x256 | 16 frames | 2 | 4 | `fp16` | 14.2 GB |
| **A100 (40 GB)** | 256x256 | 24 frames | 4 | 2 | `bf16` | 28.5 GB |

---

## 5. Temporal Verification & Benchmark Metrics

Evaluated through `training/video/evaluate.py`:
1. **Validation Loss**: Computed over held-out real MP4 video clips (`datasets/video/manifest.jsonl`).
2. **Inter-Frame Jitter Metric**:
   $$\Delta_{\text{jitter}} = \frac{1}{F-1} \sum_{f=1}^{F-1} \frac{\| I_{f} - I_{f-1} \|_1}{H \times W \times C}$$
   Measures perceptual temporal smoothness (lower delta = smoother continuity).
3. **Cross-Clip Diversity Score**: Verifies that the model generates unique scenes across different camera motion conditioning rather than collapsing to static frames.

