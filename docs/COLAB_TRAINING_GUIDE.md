# Google Colab GPU Training & Fine-Tuning Guide (Sovereign AI Edition)

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Repository**: [Prem759-0/NanoGPT](https://github.com/Prem759-0/NanoGPT)  
**Hardware Targets**: Google Colab Free **NVIDIA Tesla T4** (15.3 GB VRAM), **NVIDIA A10G**, or **NVIDIA A100**  
**Ownership Policy**: STRICT SOVEREIGN — Random weight initialization (`moon_ai_random_init`), zero third-party pretrained weights, zero hosted external APIs.

---

## 1. Sovereign Training Architecture

Unlike conventional AI wrappers that depend on proprietary APIs or download billions of weights from foreign corporations, Moon AI is built from first principles:
- **Image Generation**: Moon Latent Diffusion (`MoonUNet2D` + `MoonImageAutoencoder` + `MoonTextEncoder`, 23.7M parameters).
- **Video Generation**: Spatiotemporal Latent Diffusion (`MoonUNet3D` + `TemporalConv` + `TemporalSelfAttention`, 17.0M parameters).
- **Text & Code Generation**: NanoGPT Autoregressive Causal Transformer.
- **Hardware Optimization**: PyTorch Automatic Mixed Precision (AMP `fp16`/`bf16`), dynamic gradient accumulation, and GPU VRAM scaling.

---

## 2. Step-by-Step Google Colab Execution

### Step 1: Open Google Colab & Select Hardware Accelerator
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **File** -> **New Notebook** (or open `notebooks/NanoGPT_Training_Pipeline.ipynb`).
3. Set accelerator:
   **Runtime** -> **Change runtime type** -> Select **T4 GPU** (or **A100 GPU** if using Colab Pro) -> Click **Save**.

### Step 2: Verify GPU Hardware
In a code cell, run:
```bash
!nvidia-smi
```
Verify that an NVIDIA GPU (e.g. Tesla T4 with ~15.3 GB VRAM) is active.

### Step 3: Mount Google Drive (Persistent Checkpoints)
To ensure checkpoints survive Colab 90-minute disconnects:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Your checkpoints will automatically be mirrored to `/content/drive/MyDrive/MoonAI_Checkpoints`.

### Step 4: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/Prem759-0/NanoGPT.git
%cd NanoGPT
!pip install -r requirements.txt
```

### Step 5: Run Automated Dataset Audit
Validate that manifests, SHA-256 hashes, media files, and splits are 100% clean:
```bash
!python datasets/tools/dataset_manager.py
```

### Step 6: Execute Sovereign Training Runner

#### A. Smoke-Test Mode (Fast 5-step verification)
```bash
!python scripts/colab_train.py --track image --smoke_test
!python scripts/colab_train.py --track video --smoke_test
```

#### B. Full Production Training Mode (Image Track)
```bash
!python scripts/colab_train.py --track image --production --steps 500 --drive_dir /content/drive/MyDrive/MoonAI_Checkpoints
```

#### C. Full Production Training Mode (Video Track)
```bash
!python scripts/colab_train.py --track video --production --steps 250 --drive_dir /content/drive/MyDrive/MoonAI_Checkpoints
```

#### D. Automatic Resumption After Disconnect
If Colab disconnects or restarts:
```bash
!python scripts/colab_train.py --track image --resume --drive_dir /content/drive/MyDrive/MoonAI_Checkpoints
```
The runner will automatically detect the latest checkpoint in Google Drive and restore optimizer, scheduler, scaler, RNG state, and global step.

---

## 3. Evaluation & Quality Metrics

Run the comprehensive evaluation suite directly in Colab:
```bash
# Image Evaluation (computes validation denoising loss, prompt synthesis, pixel diversity)
!python training/image/evaluate.py --checkpoint models/checkpoints/best_moon_image.pt

# Video Evaluation (computes validation loss, inter-frame temporal jitter, cross-clip diversity)
!python training/video/evaluate.py --checkpoint models/checkpoints/best_moon_video.pt
```

All evaluation reports, generated benchmark images, and animated GIFs are exported to `data/reports/`.

---

## 4. Hardware Optimization Reference Matrix

| Accelerator | VRAM | Recommended Precision | Image Batch Size | Video Batch Size | Gradient Accumulation Steps |
|---|---|---|---|---|---|
| **Tesla T4** | 15.3 GB | `fp16` | 2 | 1 | 8 |
| **A10G** | 24.0 GB | `fp16` | 4 | 2 | 4 |
| **A100** | 40/80 GB | `bf16` | 8 | 4 | 2 |
| **CPU Fallback** | RAM | `fp32` | 1 | 1 | 4 |
