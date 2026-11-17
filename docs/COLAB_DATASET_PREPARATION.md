# Moon AI — Google Colab Dataset Preparation & T4 Verification Guide

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Target Hardware**: Google Colab Free **NVIDIA Tesla T4** (15.3 GB VRAM)  
**Script**: [`scripts/colab_dataset_prep.py`](file:///d:/Downloads/linkden%20post/ai/scripts/colab_dataset_prep.py)  
**Ownership Policy**: STRICT SOVEREIGN — 100% Random Initialization, Permissible Open Licenses Only.

---

## 1. Overview

Before starting any pilot training runs on Google Colab, you must run the dataset preparation and quality gate audit. This ensures that:
1. Manifests are technically clean (`HYGIENE_VALID = True`).
2. Training adequacy status is transparently reported (`TRAINING_DATA_ADEQUATE`).
3. Dataset manifests and reports are securely mirrored to persistent Google Drive storage.

---

## 2. Step-by-Step Colab Preparation

### Step 1: Open Google Colab and Attach T4 GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** -> **Change runtime type** -> **T4 GPU** -> **Save**.

### Step 2: Mount Google Drive
Run in a Colab code cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
This mounts your Google Drive at `/content/drive/MyDrive`.

### Step 3: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/Prem759-0/NanoGPT.git
%cd NanoGPT
!pip install -r requirements.txt
```

### Step 4: Execute Dataset Preparation Runner
```bash
!python scripts/colab_dataset_prep.py --drive_dir /content/drive/MyDrive/MoonAI_Datasets
```

The script will:
- Detect the NVIDIA Tesla T4 GPU and available VRAM.
- Run complete SHA-256 and perceptual hash audits across all manifests.
- Verify 0 duplicate files, 0 corrupted containers, 0 split leakage.
- Evaluate the two quality gates:
  - `HYGIENE_VALID`: (Verifies technical compliance)
  - `TRAINING_DATA_ADEQUATE`: (Reports genuine sample count vs pilot target of 500 images / 50 videos)
- Atomically sync manifests and reports to `/content/drive/MyDrive/MoonAI_Datasets/manifests/`.

---

## 3. Interpreting Quality Gate Statuses

| Quality Gate | Value | Meaning |
|---|---|---|
| **`HYGIENE_VALID`** | `✅ PASS` | Zero missing files, zero corrupted images/videos, zero duplicate SHA-256 hashes, zero near-duplicates (dHash $\le 4$), zero train/val/test leakage, and all licenses verified. |
| **`TRAINING_DATA_ADEQUATE`** | `⚠️ INSUFFICIENT` | The current dataset is in the pilot phase (e.g. 20/500 images, 5/50 videos). Training can run for pilot/unit testing, but is not yet adequate for broad general-purpose generalization. |
| **`TRAINING_DATA_ADEQUATE`** | `✅ ADEQUATE` | The dataset meets or exceeds the configured 500-image / 50-video pilot targets across all 10 balanced categories. |

---

## 4. Launching Pilot Training Runs on T4

Once `HYGIENE_VALID` passes, you can launch T4-optimized training runs:

```bash
# Image Pilot (FP16, batch=2, grad_accum=8, 50 steps):
!python scripts/colab_train.py --track image --steps 50 --precision fp16 --batch_size 2 --grad_accum 8 --drive_dir /content/drive/MyDrive/MoonAI_Checkpoints

# Video Pilot (FP16, batch=1, grad_accum=8, 25 steps):
!python scripts/colab_train.py --track video --steps 25 --precision fp16 --batch_size 1 --grad_accum 8 --drive_dir /content/drive/MyDrive/MoonAI_Checkpoints
```
Checkpoints will automatically sync to Google Drive after each run!

