# Daily AI Improvement & Model Promotion Pipeline Guide

**Author & Creator**: Prem Gaikwad (India 🇮🇳)  
**Main Orchestration Script**: [`scripts/daily_pipeline.py`](file:///d:/Downloads/linkden%20post/ai/scripts/daily_pipeline.py)  
**Model Registry**: `models/model_registry.json`

---

## 1. Principles of Daily LLM Improvement

Training an AI model from scratch every day is wasteful, expensive, and counter-productive. A true AI engineering workflow operates on **iterative evaluation gates**:

```
New Curated Data
       │
       ▼
Automated Deduplication & AST Syntax Check (scripts/audit_dataset.py)
       │
       ▼
Deterministic Train/Val/Test Splits (scripts/prepare_dataset.py)
       │
       ▼
Regression Unit Testing (tests/test_model_inference.py)
       │
       ▼
LoRA Fine-Tuning on GPU (Only when threshold of 20+ verified samples is met)
       │
       ▼
Fixed 15-Item Benchmark Evaluation (scripts/evaluate_model.py)
       │
       ▼
Promotion Gate: Does candidate score > previous best with 0 regressions?
       ├── YES ──► Promote to Active Checkpoint (scripts/promote_best_model.py)
       └── NO  ──► Retain Current Best Model & Record Diagnostic Report
```

---

## 2. Model Promotion Criteria

A new checkpoint or candidate is promoted **only** if it satisfies all four criteria:
1. **100% Non-Empty Rate**: Must not return empty strings or hit premature termination across the benchmark.
2. **100% Python Syntax Validity**: All generated Python code blocks must parse cleanly with `ast.parse`.
3. **No Safety / Context Regressions**: Must cleanly handle empty prompts and long context prompts.
4. **Higher Composite Score**: The composite benchmark score must be $\ge$ the current best model's score in `models/model_registry.json`.

If any criterion fails, the previous best model remains active, and a detailed diagnostic log is recorded in `data/reports/`.

---

## 3. How to Run the Daily Pipeline

### Local Run (Dataset Audit, Regression Suite, Evaluation & Report)
```powershell
python scripts/daily_pipeline.py --eval_model scratch
```

### Full Run with GPU Fine-Tuning (on Colab or CUDA Machine)
```bash
python scripts/daily_pipeline.py --eval_model llama-3.2 --train
```

---

## 4. Automated GitHub Actions CI Workflow

The workflow at `.github/workflows/daily_pipeline.yml` runs daily at 03:00 UTC. It:
1. Sets up Python and installs dependencies.
2. Runs dataset validation and syntax checking.
3. Executes the full deterministic regression test suite.
4. Runs the benchmark evaluation and generates the daily report.
5. Uploads the report as an artifact without modifying repository weights or committing private data.
