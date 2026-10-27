# Google Colab GPU Training & Fine-Tuning Guide

**Author & AI Architect**: Prem Gaikwad  
**Notebook**: [`notebooks/NanoGPT_Training_Pipeline.ipynb`](file:///d:/Downloads/linkden%20post/ai/notebooks/NanoGPT_Training_Pipeline.ipynb)  
**Target Hardware**: Google Colab Free **NVIDIA T4 GPU** (15.3 GB VRAM)  
**Cost**: $0.00 (Zero paid cloud resources or proprietary API keys required)

---

## 1. Introduction

This guide describes how to train and fine-tune your sovereign AI assistant using Google Colab's free GPU compute.

Instead of retraining an entire model from scratch (which requires millions of dollars in compute), we use **QLoRA (Quantized Low-Rank Adaptation)**:
- The base instruct model (e.g., `Llama-3.2-1B-Instruct` or `Qwen2.5-Coder-7B-Instruct`) is loaded in 4-bit NormalFloat (`NF4`) quantization.
- The base weights remain frozen.
- Lightweight adapter matrices ($r=16$) are attached to all projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Training only optimizes ~0.5% of total parameters, allowing frontier models to be trained within the 15GB VRAM limit of free Colab GPUs.

---

## 2. Step-by-Step Training Walkthrough

### Step 1: Open Notebook in Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/NanoGPT_Training_Pipeline.ipynb`.
3. In the Colab menu, navigate to:
   **Runtime** -> **Change runtime type** -> Select **T4 GPU** -> Click **Save**.

### Step 2: Verify GPU Hardware
Run Step 1 cell:
```bash
!nvidia-smi
```
Confirm that you see an **NVIDIA Tesla T4** with approximately 15,360 MiB VRAM.

### Step 3: Run High-Speed Stack Install
Run Step 2 cell. This installs:
- `unsloth` (5x faster backpropagation, 70% less VRAM)
- `trl` (Transformer Reinforcement Learning / SFTTrainer)
- `peft` (Parameter-Efficient Fine-Tuning)
- `bitsandbytes` (4-bit matrix quantization)

### Step 4: Clone Repo & Prepare Data
Step 3 clones `https://github.com/Prem759-0/NanoGPT.git` and runs:
```bash
python scripts/prepare_dataset.py --seed 42
```
This automatically deduplicates all Q&A pairs, validates Python syntax, and creates clean `train.json` (90%) and `val.json` (5%) splits.

### Step 5: Base Model Baseline Evaluation
Step 5 runs a pre-training evaluation on a test question ("Who created you?") to record how the base model responds *before* fine-tuning.

### Step 6: Train with Validation Monitoring
Step 6 trains the model for 150–300 steps.
- **Batch Size**: 2
- **Gradient Accumulation**: 4 (Effective batch size = 8)
- **Learning Rate**: `2e-4` with cosine decay and warmup
- **Validation Loss**: Evaluated every 30 steps against `data/splits/val.json` to prevent overfitting.

### Step 7: Post-Training Evaluation
Step 7 executes the same test questions. You will observe the model correctly adopt the **Prem Gaikwad** sovereign persona, output clean Python code, and answer science questions.

### Step 8: Export 4-bit GGUF Model
Step 8 quantizes the trained adapter and base weights into `Moon-AI-v2.Q4_K_M.gguf`.
1. Download this file from the Colab file explorer (left sidebar).
2. Place it into your local `Prem759-0/NanoGPT/` root folder.
3. The local `webapp` and `chat_cli.py` will automatically detect and load it!

---

## 3. Serving Directly from Colab GPU (FastAPI + ngrok)

If your local computer has a slow CPU or no GPU, you can keep the Colab GPU running as your private cloud inference backend:
1. In Step 9, paste your free ngrok token from [ngrok dashboard](https://dashboard.ngrok.com/).
2. Run Step 9. Colab will display a live URL, for example:
   `https://abcdef123.ngrok-free.app`
3. In your local webapp (or at `https://moon-gpt.vercel.app/`), click **⚡ Connect Colab GPU**, paste the URL, and click **Connect**.
4. All messages will now be processed directly on the Colab NVIDIA T4 GPU in < 2 seconds, completely bypassing local CPU latency!

---

## 4. Troubleshooting & Session Recovery

| Issue | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Sequence length too long or batch size too large | Reduce `max_seq_length` to 1024 or 512, or set `per_device_train_batch_size = 1`. |
| Session Disconnected | Colab idle timeout (90 min free tier limit) | Re-run cells from Step 2. Datasets remain saved in the GitHub repository. |
| ngrok tunnel failed | Missing or invalid auth token | Create a free account at `ngrok.com`, get your token, and paste it into `NGROK_AUTH_TOKEN`. |
| Loss not decreasing | Learning rate too high or dataset noisy | Keep `learning_rate = 2e-4` and run `scripts/audit_dataset.py` to verify clean inputs. |
