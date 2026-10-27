# Sovereign AI Assistant — Daily Pipeline Report
**Date**: 2026-10-27  
**Architect & Creator**: Prem Gaikwad (India 🇮🇳)  

---
## 1. Active Best Model Status
- **Active Model**: **Moon AI Sovereign (7.6B v1)**
- **Checkpoint Artifact**: `Moon-AI-v1.Q4_K_M.gguf`
- **Composite Quality Score**: **75.0 / 100**
- **Last Promotion Timestamp**: 2026-10-27T16:03:47.493285+00:00

## 2. Dataset Split Statistics
- **Total Clean Verified Records**: 439
- **Train Split (90%)**: 395 examples
- **Validation Split (5%)**: 21 examples
- **Blind Test Split (5%)**: 23 examples
- **Filtered Duplicates**: 445

### Domain Category Breakdown
- **Coding**: 331 records
- **General_Conversation**: 104 records
- **Science**: 2 records
- **Persona**: 2 records

## 3. Latest Benchmark Evaluation Results
- **Evaluated Model**: `scratch`
- **Tokens / Sec**: 35.47
- **Non-Empty Output Rate**: 100.0%
- **Keyword Coverage**: 13.3%
- **Code Syntax Accuracy**: 100.0%
- **Composite Benchmark Score**: **65.3 / 100**

## 4. Promotion History & Integrity Log
| Timestamp | Candidate | Score | Status | Notes |
|---|---|---|---|---|
| 2026-10-27T16:03:47 | `scratch` | 65.3 | **REJECTED** | Candidate score (65.3) regressed compared to current best (75.0). |

## 5. Daily Recommendations
1. **Data Additions**: Expand domain-specific test cases in `data/splits/test.json`.
2. **Colab Training**: When new verified data reaches 50+ samples, run `notebooks/NanoGPT_Training_Pipeline.ipynb` on Colab T4 GPU.
3. **Direct GPU Serving**: Utilize Step 9 ngrok URL to connect local webapp to Colab GPU.