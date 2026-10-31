# 🎬 MOON VIDEO ARCHITECTURE SPECIFICATION
**Engine Name**: Moon Video (`moon-video-v0`)  
**Chief Architect**: Prem Gaikwad (India 🇮🇳)  
**Weight Provenance**: `moon_ai_random_init` (Strict Sovereign Model Ownership)  
**Status**: Implemented & Verified in `backend/engines/moon_video/`

---

## 1. Architectural Overview

Moon Video is a sovereign spatiotemporal latent diffusion network designed for controllable, high-definition video synthesis. It integrates spatial 2D feature synthesis with 1D temporal self-attention across frames, conditioned on both natural language text and explicit 8-DoF camera motion vectors.

The model is initialized entirely from scratch with zero pretrained weights or external APIs.

```
+--------------------------------------------------------------------------+
|                       MOON VIDEO v0 ARCHITECTURE                         |
+--------------------------------------------------------------------------+

  [ Text Prompt ] ──> [ MoonVideoTextEncoder ] ──┐
  [ Camera Motion ] ──> [ Motion Embedding ] ────┼──> [ Conditioning Context ]
                                                 │                 │
  [ Video (BxCxTxHxW) ]                          │                 │
            │                                    │                 │
            ▼                                    │                 │
  [ Spatiotemporal VAE ]                         │                 │
            │                                    │                 │
            ▼                                    │                 │
  [ Latents (Bx4xTx16x16) ] ─────────────────────┼─────────────────┤
            ▲                                    │                 │
            │                                    ▼                 ▼
     [ Noise (t) ] ───────────────> [ MoonUNet3D (Spatial 2D + Temporal 1D) ]
                                                 │
                                                 ▼
                                        [ Predicted Noise ]
```

---

## 2. Parameter Distribution

The baseline configuration of `moon-video-v0` contains **17,009,011 trainable parameters**:

| Component | Sub-modules | Parameter Count |
| :--- | :--- | :--- |
| **`MoonVideoTextEncoder`** | Token embedding (8,192 vocab, 512 dim), 4 Transformer layers, Camera motion embedding (16 categories, 512 dim) | **5,278,976** |
| **`MoonVideoAutoencoder`** | 3D-aware convolutional encoder (channels: 64->128->256), temporal smoothing convs, decoder | **4,521,519** |
| **`MoonUNet3D`** | 3 Down Blocks (Spatial 2D Conv + Cross-Attention + Temporal 1D Transformer), Bottleneck, 3 Up Blocks | **7,208,516** |
| **Total Trainable Parameters** | | **17,009,011** |

---

## 3. Factorized Spatiotemporal Attention

Video generation is computationally demanding. Moon Video uses a **factorized 2D+1D architecture**:
1. **Spatial Blocks (2D)**:
   - Treat each frame independently to extract spatial features and align with text semantics via cross-attention.
   - Preserves high spatial detail without cubic computational explosion.
2. **Temporal Blocks (1D)**:
   - Reshape feature tensors from $(B, C, F, H, W)$ to $(B \times H \times W, F, C)$ and pass through multi-head self-attention along the frame axis $F$.
   - Enforces physical motion continuity, temporal smoothness, and eliminates frame flickering.

$$\text{Attention}_{\text{Temporal}}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V, \quad Q, K, V \in \mathbb{R}^{(B \cdot H \cdot W) \times F \times C}$$

---

## 4. 8-DoF Camera Motion Conditioning

Moon Video allows direct steering of the virtual camera. The camera trajectory is indexed into 16 discrete motion primitives:
- `static`: Locked camera position.
- `pan_left`, `pan_right`: Horizontal angular camera sweep.
- `tilt_up`, `tilt_down`: Vertical pitch rotation.
- `zoom_in`, `zoom_out`: Focal length scaling / dolly trajectory.
- `orbit_left`, `orbit_right`: Circular orbital rotation around the central focus point.
- `crane_up`, `crane_down`: Vertical pedestal translation.

The camera motion ID is mapped to a learned continuous embedding vector and concatenated to the text conditioning sequence:

$$c_{\text{multimodal}} = [e_{\text{text}}; e_{\text{camera}}] \in \mathbb{R}^{B \times (L + 1) \times D}$$

---

## 5. Training Objective

The spatiotemporal network $\epsilon_\theta$ is trained via diffusion MSE loss across all frames:

$$\mathcal{L}_{\text{MoonVideo}}(\theta) = \mathbb{E}_{z_0, \epsilon, t, c_{\text{text}}, c_{\text{cam}}} \left[ \frac{1}{F} \sum_{f=1}^F \left\| \epsilon_f - \epsilon_\theta(z_{t, f}, t, c_{\text{text}}, c_{\text{cam}}) \right\|^2 \right]$$

Where:
- $\epsilon \sim \mathcal{N}(0, I)$ is added independently per frame or with smooth temporal correlation.
- $t \in [0, T-1]$ is the shared diffusion timestep.

---

## 6. Usage & Inference API

```python
from backend.engines.moon_video import MoonVideoInferenceService

service = MoonVideoInferenceService(checkpoint_path="models/checkpoints/moon_video_v0.pt")
result = service.generate_video(
    prompt="a sports car accelerating on highway, rainy evening reflections",
    camera_motion="zoom_in",
    num_frames=16,
    export_gif_path="output.gif"
)

if result["success"]:
    print(f"Generated video: {result['output_path']}")
else:
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
```
