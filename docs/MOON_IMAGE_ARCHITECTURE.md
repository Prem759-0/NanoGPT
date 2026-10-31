# 🎨 MOON IMAGE ARCHITECTURE SPECIFICATION
**Engine Name**: Moon Image (`moon-image-v0`)  
**Chief Architect**: Prem Gaikwad (India 🇮🇳)  
**Weight Provenance**: `moon_ai_random_init` (Strict Sovereign Model Ownership)  
**Status**: Implemented & Verified in `backend/engines/moon_image/`

---

## 1. Architectural Overview

Moon Image is a self-contained, sovereign latent diffusion model built entirely from scratch with PyTorch. It does not use any third-party pretrained weights, APIs, or Hugging Face diffusers pipelines.

The architecture comprises four sub-networks:
1. **`MoonTextEncoder`**: Original learned Transformer text conditioning encoder.
2. **`MoonImageAutoencoder`**: Spatial convolutional Variational Autoencoder (VAE) for continuous latent representation.
3. **`MoonUNet2D`**: Multi-scale spatial UNet with cross-attention for iterative denoising in latent space.
4. **`MoonDiffusionScheduler`**: Noise schedule implementing DDPM/DDIM forward diffusion and reverse sampling.

```
+--------------------------------------------------------------------------+
|                       MOON IMAGE v0 ARCHITECTURE                         |
+--------------------------------------------------------------------------+

  [ Text Prompt ] ──────> [ SimpleTextTokenizer ]
                                   │
                                   ▼
                       [ MoonTextEncoder (5.28M) ] ───────────────┐
                                                                  │
                                                                  ▼
  [ Image (256x256x3) ]                                [ Cross-Attention ]
            │                                                     │
            ▼                                                     ▼
  [ Autoencoder (8.03M) ] ──> [ Latents (32x32x4) ] ──> [ MoonUNet2D (10.40M) ]
                                      ▲                           │
                                      │                           ▼
                               [ Noise (t) ]            [ Predicted Noise ]
```

---

## 2. Parameter Distribution

The baseline configuration of `moon-image-v0` contains **23,713,743 trainable parameters**:

| Component | Sub-modules | Parameter Count |
| :--- | :--- | :--- |
| **`MoonTextEncoder`** | Embedding (8,192 vocab, 512 dim), 4 Transformer layers, MultiheadAttention (8 heads) | **5,276,416** |
| **`MoonImageAutoencoder`** | Encoder (Residual Conv Blocks 64->128->256), Quant/Moments, Decoder (256->128->64) | **8,032,587** |
| **`MoonUNet2D`** | Time Embedding (MLP 256 dim), 3 Down Blocks (Cross-Attn), Bottleneck, 3 Up Blocks | **10,404,740** |
| **Total Trainable Parameters** | | **23,713,743** |

---

## 3. Mathematical Foundations

### 3.1 Latent Space Compression
Instead of operating directly on high-dimensional pixel space ($256 \times 256 \times 3 = 196,608$ values), the spatial autoencoder compresses RGB inputs into a compact 4-channel latent space:

$$z = \mathcal{E}(x), \quad z \in \mathbb{R}^{B \times 4 \times 32 \times 32}$$

This achieves a spatial downsampling factor of $f = 8$, reducing the computational burden by $64\times$ while preserving perceptual features.

### 3.2 Forward Diffusion Process
At timestep $t \in \{1, \dots, T\}$, Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ is added according to the cosine-beta schedule:

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

Where:
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

### 3.3 Training Objective
The UNet $\epsilon_\theta$ is trained using Mean Squared Error (MSE) loss to predict the noise vector $\epsilon$:

$$\mathcal{L}_{\text{MoonImage}}(\theta) = \mathbb{E}_{z_0, \epsilon, t, c} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, \tau(c)) \right\|^2 \right]$$

Where:
- $\tau(c)$ is the text conditioning representation produced by `MoonTextEncoder`.
- $t$ is sampled uniformly from $\{0, \dots, T - 1\}$.

### 3.4 Classifier-Free Guidance (CFG)
During reverse generation, classifier-free guidance amplifies text alignment:

$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)$$

Where $s$ is the guidance scale (default: $7.5$).

---

## 4. Key Implementation Details

1. **Self-Attention & Cross-Attention**:
   - `MoonUNet2D` uses spatial cross-attention layers where queries $Q$ are derived from latent feature maps, and keys $K$ and values $V$ are projected from the text encoder token context.
2. **Group Normalization & SiLU**:
   - Every residual block employs `nn.GroupNorm(num_groups=8, ...)` and `nn.SiLU()` activations to guarantee numerical stability across varying batch sizes.
3. **Random Weight Initialization**:
   - All weights are initialized using truncated Gaussian distributions ($\mu = 0, \sigma = 0.02$).
   - Checkpoints enforce `"provenance": "moon_ai_random_init"`.

---

## 5. Usage & Inference API

```python
from backend.engines.moon_image import MoonImageInferenceService

# Truthful inference: returns MODEL_NOT_TRAINED until checkpoint is trained
service = MoonImageInferenceService(checkpoint_path="models/checkpoints/moon_image_v0.pt")
result = service.generate(
    prompt="cybernetic chrome sphere on reflective neon grid",
    steps=25,
    guidance=7.5
)

if result["success"]:
    result["image"].save("output.png")
else:
    print(f"Status: {result['status']}")
    print(f"Diagnostics: {result['message']}")
```
