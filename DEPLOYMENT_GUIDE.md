# 🚀 Moon AI: Complete Production Deployment Guide

This guide will walk you through deploying **Moon AI** to the cloud so it runs at lightning speed (1–2 second responses) with **zero load on your laptop**.

---

## 🏛️ System Architecture

```
[ User Mobile / Laptop Browser ]
              │
              ▼
   [ Vercel Edge Network ]
   (Frontend: React + Vite + Clerk Auth)
   URL: https://moon-ai.vercel.app
              │
              │ REST API Calls (VITE_API_URL)
              ▼
    [ Cloud GPU Backend ]
   (FastAPI + Python + CUDA / NVIDIA GPU)
   • Free Colab T4 GPU (via ngrok)
   • Hugging Face Spaces Docker
   • RunPod / Render Cloud Pod
```

---

## Part 1: Deploy Frontend to Vercel (Free)

1. **Push your code to GitHub**:
   If not already pushed, commit and push your repo to GitHub:
   ```bash
   git add .
   git commit -m "Configure Vercel and CORS for cloud deployment"
   git push origin main
   ```

2. **Open Vercel**:
   * Go to [vercel.com](https://vercel.com) and Sign In (with GitHub).
   * Click **"Add New..."** → **"Project"**.
   * Import your **`NanoGPT`** / Moon AI repository.

3. **Configure Project Settings**:
   * **Framework Preset**: `Vite`
   * **Root Directory**: Click *Edit* and select **`frontend`** (Important!).
   * **Build Command**: `npm run build`
   * **Output Directory**: `dist`

4. **Add Environment Variables**:
   Under **Environment Variables**, add:
   * `VITE_CLERK_PUBLISHABLE_KEY`: `pk_test_cG9ldGljLXJhcHRvci00MjM0LmNsZXJrLmFjY291bnRzLmRldiQ`
   * `VITE_API_URL`: *(Leave empty for now, or paste your Cloud Backend URL from Part 2)*

5. Click **"Deploy"**!
   * Within 60 seconds, your frontend will be live at `https://your-app-name.vercel.app`!

---

## Part 2: Deploy Backend on Cloud GPU (Free via Google Colab + ngrok)

Because 7.6B models need an NVIDIA GPU to respond in seconds, you can run the backend for **100% free** using Google Colab's cloud GPU:

1. Open [Google Colab](https://colab.research.google.com).
2. Upload and open the provided notebook:
   **[`Deploy_Moon_AI_Cloud_GPU.ipynb`](file:///d:/Downloads/linkden%20post/ai/Deploy_Moon_AI_Cloud_GPU.ipynb)**
3. Go to **Runtime** → **Change runtime type** → Select **T4 GPU**.
4. Get a free ngrok token:
   * Sign up free at [ngrok.com](https://dashboard.ngrok.com/signup).
   * Copy your **Authtoken** from the ngrok dashboard.
   * Paste it into Step 5 in the Colab notebook.
5. Click **Run All** in Google Colab.
6. The notebook will display your public HTTPS backend link:
   ```
   🚀 MOON AI PUBLIC CLOUD BACKEND URL:
   https://xxxx-xx-xx-xx.ngrok-free.app
   ```

---

## Part 3: Connect Frontend to Backend

1. Go back to your [Vercel Dashboard](https://vercel.com).
2. Click your project → **Settings** → **Environment Variables**.
3. Add or update:
   * **Key**: `VITE_API_URL`
   * **Value**: `https://xxxx-xx-xx-xx.ngrok-free.app` (your Colab backend URL)
4. Go to **Deployments** → Click **"Redeploy"**.
5. Open your live Vercel URL:
   * Chat queries will now execute on the **Cloud GPU** at **60+ tokens/second** (1–2 seconds per answer)!
   * Games, code execution, image/video generation, and memory will work worldwide.

---

## Alternative Backend: Hugging Face Spaces (24/7 Hosting)

If you prefer 24/7 hosting without keeping Colab open:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a **Docker Space**.
2. Push the files in this repository (it already includes the pre-configured [Dockerfile](file:///d:/Downloads/linkden%20post/ai/Dockerfile) and [README_SPACES.md](file:///d:/Downloads/linkden%20post/ai/README_SPACES.md)).
3. Hugging Face will automatically build and host the FastAPI server with a permanent HTTPS URL!
