# NanoGPT Lab — Frontier Multimodal AI Suite

Welcome to **NanoGPT Lab**! An entirely local, 100% private, API-key-free AI system spanning all three frontier domains:
1. 💻 **Coding AI (Codex & Claude Caliber)**: Production code generation, algorithmic reasoning, debugging, and live sandboxed Python execution with 1-click running.
2. 🎨 **Image AI (Higgsfield & Midjourney Caliber)**: High-resolution image synthesis with automatic aesthetic prompt enhancement (Cinematic 8K, Cyberpunk Neon, Photorealistic Studio, Anime, and Quantum 3D styles).
3. 🎬 **Video AI (Kling & Runway Caliber)**: Spatio-temporal text-to-video synthesis engine compiling motion sequences into downloadable `.mp4` video clips with embedded HTML5 video playback.

---

## 🔒 100% Local & Honest Execution Guarantee
- **Zero External Paid AI APIs**: No OpenAI, Anthropic, Gemini, Grok, or Hugging Face hosted inference endpoints.
- **Local Neural Computation**: Fine-tuned 1.23B LLaMA-3.2 running via `llama-cpp-python` (AVX2 4-bit quantization) on your laptop CPU at ~20 tokens/sec.
- **Free Cloud GPU Training**: Pre-configured, 1-click Google Colab notebooks for scaling models on free NVIDIA Tesla T4 GPUs (15GB VRAM).

---

## 🚀 The Three Frontier Pillars

### 1. 💻 Coding AI (Codex & Claude Caliber)
- **Local Engine**: Fine-tuned LLaMA-3.2 equipped with Codex & Claude system prompts for clean, idiomatic code generation with syntax highlighting.
- **Interactive Python Sandbox**: Execute Python scripts safely with execution timers, stdout/stderr capture, and return code tracking.
- **1-Click Sandbox Execution**: Click **`▶ Run Code`** on any generated code block in the chat stream to execute it locally.
- **Cloud Fine-Tuning**: [`Train_Coding_AI_Colab.ipynb`](Train_Coding_AI_Colab.ipynb) trains LLaMA-3.2 on CodeAlpaca datasets and exports quantized `.gguf` models directly.

### 2. 🎨 Image Generation (Higgsfield & Midjourney Caliber)
- **Prompt Enhancer**: Automatically enriches simple user prompts with cinematic lighting, depth-of-field, volumetric smoke, 8K resolution, and lens specifications.
- **5 Aesthetic Styles**:
  - `cinematic`: 8K IMAX 70mm, volumetric fog, octane render
  - `cyberpunk`: Neon reflections, rainy pavement, holograms
  - `photorealistic`: Hasselblad 85mm f/1.4 studio lighting
  - `anime`: Makoto Shinkai aesthetic, ArtStation trending
  - `quantum`: Holographic particle core, neural network nodes
- **Cloud Diffusion Studio**: [`Train_And_Generate_Images_Colab.ipynb`](Train_And_Generate_Images_Colab.ipynb) runs Stable Diffusion XL / SD-Turbo for 1024x1024 photorealistic rendering on Colab GPU.

### 3. 🎬 Video Generation (Kling & Runway Caliber)
- **Local MP4 Synthesis**: Generates spatio-temporal animated video clips locally and encodes them into standard H.264 MP4 files.
- **Motion Simulation Modes**: Matrix digital rain, neural network pulse, cosmic starfield warp, cyberpunk synthwave horizon, and kinetic particle vortex.
- **Embedded Player**: Video clips play directly in the chat stream with `<video controls autoplay loop>` and direct MP4 download links.
- **Cloud Video Diffusion**: [`Generate_AI_Video_Colab.ipynb`](Generate_AI_Video_Colab.ipynb) runs Text-to-Video spatio-temporal diffusion on Colab GPU to render high-frame video clips.

---

## 🛠️ Quickstart Guide

### Starting the Web Application
```powershell
# Open terminal in project root
cd "d:\Downloads\linkden post\ai"

# Run Uvicorn server
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### Chat & Command Shortcuts
| Command | Domain | Description |
| :--- | :--- | :--- |
| `[text]` | **Chat** | Converse with active model (LLaMA-3.2, GPT-2, or Scratch) |
| `/code [task]` | **Coding AI** | Generate production code with Claude/Codex persona |
| `/imagine [prompt]` | **Image AI** | Generate Midjourney/Higgsfield caliber image |
| `/video [prompt]` | **Video AI** | Generate Kling/Runway caliber MP4 video |
| `/run [code]` | **Sandbox** | Execute Python code snippet directly |
| `/model [name]` | **Engine** | Switch between `llama-3.2`, `gpt2`, and `scratch` |

### Running the Terminal CLI Assistant
```powershell
.\.venv\Scripts\python.exe chat_cli.py
```

---

## ☁️ Google Colab Notebooks (Free GPU Training)
| Notebook | Domain | GPU Type | Output |
| :--- | :--- | :--- | :--- |
| [`Train_Super_AI_Colab.ipynb`](Train_Super_AI_Colab.ipynb) | General LLM | T4 GPU | Fine-tuned LLaMA-3.2 GGUF |
| [`Train_Coding_AI_Colab.ipynb`](Train_Coding_AI_Colab.ipynb) | Coding AI | T4 GPU | Code-specialized GGUF |
| [`Train_And_Generate_Images_Colab.ipynb`](Train_And_Generate_Images_Colab.ipynb) | Image Studio | T4 GPU | 1024x1024 Photorealistic Images |
| [`Generate_AI_Video_Colab.ipynb`](Generate_AI_Video_Colab.ipynb) | Video Studio | T4 GPU | Downloadable `.mp4` Video Clips |
