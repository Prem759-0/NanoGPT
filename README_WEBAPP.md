# NanoGPT Lab — User Guide & Setup Manual

Welcome to **NanoGPT Lab**! This project provides a 100% local, API-key-free Web Application interface around Andrej Karpathy's [`nanoGPT`](https://github.com/karpathy/nanoGPT) transformer neural network.

---

## 🔒 100% Local & Honest Execution Guarantee
- **Zero API Keys Required**: No OpenAI, Anthropic, Gemini, Grok, or Hugging Face cloud APIs are used.
- **Local Neural Computation**: All models run and train on your machine's hardware using PyTorch.
- **Honest State Tracking**: If no trained checkpoint exists on disk, the system explicitly reports **`NOT TRAINED`** and returns a `409 Conflict` error when chat is requested. No simulated or fake AI answers are fabricated.

---

## 📁 Repository Structure
```
nanoGPT/
├── config/                     # Original nanoGPT training configurations
│   └── train_shakespeare_char.py
├── data/                       # Dataset preparation scripts
│   └── shakespeare_char/
├── model.py                    # Original Transformer GPT neural network
├── train.py                    # Original training loop
├── sample.py                   # Original generation / sampling script
├── webapp/                     # Web Application Layer
│   ├── main.py                 # FastAPI application and route endpoints
│   ├── model_service.py        # Local model loading, safety, & inference
│   ├── schemas.py              # Pydantic request and response models
│   └── static/
│       ├── index.html          # Responsive Dark Developer-Lab UI
│       ├── styles.css          # Vanilla CSS design tokens & animations
│       └── app.js              # Client controller & state polling
├── tests/
│   └── test_webapp.py          # Automated verification test suite
├── requirements-webapp.txt     # Web dependencies (FastAPI, Uvicorn, Pydantic, HTTPX, NumPy)
└── README_WEBAPP.md            # This documentation file
```

---

## 🚀 Quickstart Instructions (Windows PowerShell)

### Step 1: Open PowerShell in the Project Directory
Make sure your terminal is opened in `d:\Downloads\linkden post\ai`:
```powershell
cd "d:\Downloads\linkden post\ai"
```

### Step 2: Activate the Virtual Environment
A virtual environment has been created in `.venv` using Python 3.13:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Web Dependencies (if not already installed)
```powershell
python -m pip install -r requirements-webapp.txt
```

### Step 4: Start the Web Application Server
```powershell
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 5: Open in Your Browser
- **Web Application UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### How to Stop the Server
Press `Ctrl + C` in the PowerShell terminal where Uvicorn is running.

---

## 🧪 Running Automated Tests
To run the automated test suite and verify endpoints, rejection of empty/oversized messages, and honest status:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_webapp.py
```

---

## 🧠 Understanding Current vs. Future Capabilities

### What Works Right Now (Pre-Training Phase):
1. ✅ **Local Web Server**: FastAPI serving the responsive Developer-Lab interface at `127.0.0.1:8000`.
2. ✅ **Health & Status Reporting**: `/api/health` and `/api/model-info` report the accurate status: `NOT TRAINED`.
3. ✅ **Input Validation**: Rejects empty prompts, whitespace-only messages, and inputs exceeding 1,000 characters with HTTP 422.
4. ✅ **Honest Conflict Error**: Submitting a prompt before training returns HTTP 409 (`NOT TRAINED`) with instructions rather than fake text.
5. ✅ **Security Protections**: Safe model loading paths, no `eval`/`exec`, bound to localhost.

### What Requires Training (Next Phase):
- **Checkpoint Generation**: Running dataset preparation and model training creates `out-shakespeare-char/ckpt.pt`.
- **Text Generation**: Once `ckpt.pt` exists, the web application automatically transitions from `NOT TRAINED` to `READY`, allowing you to generate text directly from your local transformer.
