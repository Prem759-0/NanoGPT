# 🚀 MOON MULTIMODAL TRAINING GUIDE
**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Standard**: Strict Sovereign Model Ownership (`moon_ai_random_init`)  
**Engines**: Moon Image (`moon-image-v0`) & Moon Video (`moon-video-v0`)

---

## 1. Prerequisites & Environment Setup

All training pipelines run on Python 3.10+ using PyTorch. No proprietary cloud API keys or external weights are required.

```bash
# Activate your Python virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Verify PyTorch installation and device detection
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Dataset Preparation & Manifest Validation

Every multimodal training dataset must be formatted as a JSON Lines (`.jsonl`) file conforming to Moon AI's strict manifest schema.

### 2.1 Image Manifest Schema (`datasets/image/manifest.jsonl`)
```json
{
  "id": "sample_unique_id",
  "media_path": "data/images_3d/sample_0_0.png",
  "caption": "<moon-3d> cybernetic chrome sphere floating over reflective grid",
  "source": "Moon AI Synthetic Seed Collection",
  "license": "MIT",
  "split": "train",
  "width": 256,
  "height": 256,
  "channels": 3
}
```

### 2.2 Video Manifest Schema (`datasets/video/manifest.jsonl`)
```json
{
  "id": "vid_sample_01",
  "media_path": "data/images_3d/sample_0_0.png",
  "caption": "<moon-video> chrome sphere with orbital camera tracking",
  "source": "Moon AI Spatiotemporal Collection",
  "license": "MIT",
  "split": "train",
  "width": 128,
  "height": 128,
  "duration_s": 2.0,
  "fps": 8,
  "num_frames": 16,
  "camera_motion": "orbit_left"
}
```

### 2.3 Strict Manifest Validation
Before launching any training run, validate your manifest to guarantee 0 missing files, 0 corrupted media, 0 duplicates, and 0 train/test split leakage:

```bash
# Validate Image Manifest
python datasets/validate_manifest.py --manifest datasets/image/manifest.jsonl

# Validate Video Manifest
python datasets/validate_manifest.py --manifest datasets/video/manifest.jsonl
```

---

## 3. Training Moon Image (`moon-image-v0`)

### 3.1 Quick Verification (CPU Dry-Run)
Verifies the complete forward pass, loss calculation, backward pass, and checkpoint saving in 2 steps:
```bash
python training/image/train.py --dry_run
```

### 3.2 Production Training Run (GPU)
```bash
python training/image/train.py \
  --manifest datasets/image/manifest.jsonl \
  --image_size 256 \
  --batch_size 8 \
  --lr 1e-4 \
  --max_steps 5000 \
  --log_every 50 \
  --output_dir models/checkpoints \
  --device cuda
```

### Recommended Hyperparameters:
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `image_size` | `256` | Input resolution (downsampled to 32x32 latents) |
| `batch_size` | `8` (GPU) / `2` (CPU) | Batch size per optimizer step |
| `lr` | `1e-4` | AdamW initial learning rate |
| `weight_decay` | `1e-2` | L2 weight regularization |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `timesteps` | `1000` | Diffusion noise schedule discretization |

---

## 4. Training Moon Video (`moon-video-v0`)

### 4.1 Quick Verification (CPU Dry-Run)
```bash
python training/video/train.py --dry_run
```

### 4.2 Production Video Training Run (GPU)
```bash
python training/video/train.py \
  --manifest datasets/video/manifest.jsonl \
  --frame_size 128 \
  --num_frames 16 \
  --batch_size 2 \
  --lr 8e-5 \
  --max_steps 8000 \
  --log_every 50 \
  --output_dir models/checkpoints \
  --device cuda
```

### Recommended Hyperparameters:
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `frame_size` | `128` | Frame resolution (downsampled to 16x16 latents) |
| `num_frames` | `16` | Frames per video clip (2.0s at 8 FPS) |
| `batch_size` | `2` | Batch size per GPU step (high VRAM usage) |
| `lr` | `8e-5` | AdamW learning rate |
| `camera_motion` | Conditioned | Learned embedding concatenated to text token sequence |

---

## 5. Checkpointing & Verification

Checkpoints are saved to `models/checkpoints/` and must pass strict provenance verification:
- `moon_image_v0.pt`: Contains model weights, optimizer state, step counter, loss tracker, and `"provenance": "moon_ai_random_init"`.
- `moon_video_v0.pt`: Contains spatiotemporal weights and motion conditioning parameters.

### Inspecting Checkpoint Provenance:
```python
import torch

ckpt = torch.load("models/checkpoints/moon_image_v0.pt", map_location="cpu")
print("Provenance:", ckpt.get("provenance"))  # Must print: moon_ai_random_init
print("Step:", ckpt.get("step"))
print("Architect:", ckpt.get("architect"))     # Prem Gaikwad (India 🇮🇳)
```

---

## 6. Hardware & GPU Compute Matrix

| Target Engine | Minimum Hardware | Recommended Production Hardware | VRAM |
| :--- | :--- | :--- | :--- |
| **Image Verification (Dry Run)** | 4-Core CPU | Any Modern Laptop / Desktop | < 2 GB RAM |
| **Image Convergence (256x256)** | NVIDIA T4 / RTX 3060 | NVIDIA A10G / RTX 4090 | 12 - 16 GB VRAM |
| **Video Verification (Dry Run)** | 4-Core CPU | Any Modern Laptop / Desktop | < 4 GB RAM |
| **Video Convergence (16 Frames)** | NVIDIA A10G (24GB) | NVIDIA A100 (40GB/80GB) | 24 - 40 GB VRAM |
