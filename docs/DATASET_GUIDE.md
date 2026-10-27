# NanoGPT & Moon AI — Reproducible Dataset Engineering Guide

**Owner / Creator**: Prem Gaikwad  
**Last Updated**: 2026-10-27  
**Directory**: `data/splits/` and `scripts/`

---

## 1. Overview & Principles

Training high-quality LLMs requires strict data hygiene. Adding thousands of corrupted or near-identical examples harms reasoning ability and causes model hallucinations.

Our sovereign dataset pipeline adheres to five non-negotiable principles:
1. **Never Destroy Originals**: Raw files (`moon_ai_master_dataset.json`, `dataset.json`, `chatgpt_dataset.json`) remain permanently immutable as reference seeds.
2. **Deterministic Splitting**: Train (90%), Validation (5%), and Test (5%) sets are partitioned using a fixed random seed (`seed=42`). Test samples are **strictly isolated** and never seen during training.
3. **Automated Deduplication**: Exact duplicate queries and near-duplicates (whitespace, punctuation, casing variations) are filtered out automatically.
4. **AST Syntax Validation**: All Python code examples inside triple backtick blocks are parsed with Python's Abstract Syntax Tree (`ast.parse`). Examples with invalid syntax are rejected.
5. **Universal ChatML / Instruction Format**: Every sample is exported in standard OpenAI/ChatML conversation structure:
   ```json
   {
     "id": "moon_ai_00001",
     "instruction": "Write a Python function to reverse a string.",
     "response": "def reverse_string(s):\n    return s[::-1]",
     "messages": [
       {"role": "system", "content": "You are Moon AI..."},
       {"role": "user", "content": "Write a Python function to reverse a string."},
       {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}
     ],
     "category": "coding",
     "source": "moon_ai_master_dataset.json"
   }
   ```

---

## 2. Directory Structure

```
data/
├── assistant_chat/          # Scratch NanoGPT character-level training data
│   ├── meta.pkl             # Character vocabulary mapping (stoi/itos, 97 chars)
│   ├── train.bin            # Binary encoded tokens (uint16)
│   └── val.bin              # Binary encoded validation tokens
├── reports/                 # Automated audit logs & regression reports
│   └── dataset_audit_report.json
├── splits/                  # Cleaned, deduplicated, reproducible dataset splits
│   ├── manifest.json        # Version, counts, SHA256 hashes, category breakdown
│   ├── train.json           # 90% training split
│   ├── val.json             # 5% validation split
│   └── test.json            # 5% blind benchmark test split
└── user_learned_data.jsonl  # Live runtime chat turns captured from local app
```

---

## 3. How to Audit Datasets

To run a full integrity audit on all source datasets:

```powershell
python scripts/audit_dataset.py
```

To audit a specific file:
```powershell
python scripts/audit_dataset.py --file my_new_data.json
```

The script reports:
- Total records and empty record count
- Exact duplicates and near-duplicates
- Python syntax errors
- Multilingual content (Hindi/Hinglish)
- Average prompt and response lengths in characters

---

## 4. How to Generate Clean Splits

Whenever you add new curated records or want to rebuild the training splits:

```powershell
python scripts/prepare_dataset.py --seed 42
```

This updates `data/splits/train.json`, `data/splits/val.json`, and `data/splits/test.json`, and regenerates `data/splits/manifest.json`.

---

## 5. Adding New Training Examples

When adding new Q&A pairs:
1. Append your examples to `moon_ai_master_dataset.json` or create a new file in `data/`.
2. Format as a JSON list:
   ```json
   [
     {
       "instruction": "What is photosynthesis?",
       "response": "Photosynthesis is the biological process by which green plants convert light energy, water, and carbon dioxide into glucose and oxygen."
     }
   ]
   ```
3. Run `python scripts/audit_dataset.py` to verify syntax and absence of duplicates.
4. Run `python scripts/prepare_dataset.py` to update the versioned training splits.
