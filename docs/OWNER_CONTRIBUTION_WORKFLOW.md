# Moon Chat — Owner-Authored Contribution & Public-Source Review Workflow

**Phase:** Phase 2.7.8 (Genuine Dataset Acquisition & Quality Control)  
**Active Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY` (`AUTHORIZED_BY_PREM = False`)  

This guide documents how the project owner adds original human-authored Q&A examples or reviews public-source-derived summaries without editing JSONL manifests manually.

> [!IMPORTANT]
> No newly submitted or fetched record ever enters production manifests automatically.
> Every imported record defaults strictly to **`PENDING_REVIEW`** until validated and explicitly promoted via `python -m datasets.chat.review_queue --approve`.

---

## 1. Supported 13 Target Categories

| Category Slug | Human-Readable Domain |
| :--- | :--- |
| `general_knowledge` | General knowledge |
| `basic_science` | Basic science |
| `mathematics` | Mathematics and beginner explanations |
| `programming_fundamentals` | Programming fundamentals |
| `python` | Python |
| `javascript` | JavaScript |
| `html_css` | HTML and CSS |
| `web_development` | Web development |
| `computer_science` | Computer science |
| `ai_ml_fundamentals` | AI and machine learning fundamentals |
| `instruction_following` | Instruction following |
| `reasoning_problem_solving` | Reasoning and problem-solving |
| `helpful_conversational` | Helpful conversational responses |

---

## 2. Quick Owner Commands

### A. Generate 13 Empty Category Templates or Browse Category Guidance
```bash
python -m datasets.chat.contribute --categories
python -m datasets.chat.contribute --init-templates
python -m datasets.chat.contribute --init-example
python -m datasets.chat.contribute --interactive
```
This creates:
- `datasets/contributions/owner_templates.jsonl` (13 empty category skeletons with per-category authoring guidance, zero fabricated answers)
- `datasets/contributions/example_owner_contribution.json` (documented structure)
- `datasets/contributions/SCHEMA.json` (JSON Schema specification)
- `--interactive`: launches a multi-line guided wizard with live syntax, PII, duplicate, and token-count checks (always staged as `PENDING_REVIEW`)

### B. Submit a Single Original Q&A Without Editing JSONL
```bash
python -m datasets.chat.contribute --submit ^
  --category python ^
  --author "Prem Gaikwad" ^
  --prompt "What is a Python list comprehension and when should you use one?" ^
  --answer "A list comprehension is a concise syntax for constructing a new list from an iterable..."
```

### C. Validate and Import a Filled Contribution File (Dry-Run Preview + Stage)
```bash
# Preview accepted-to-pending vs rejected records without writing:
python -m datasets.chat.contribute --import-file datasets/contributions/my_batch.jsonl --dry-run

# Stage valid entries as PENDING_REVIEW:
python -m datasets.chat.contribute --import-file datasets/contributions/my_batch.jsonl
```

---

## 3. Public-Source Acquisition & Review Queue Commands

### A. Fetch Permitted Open Sources (CC-BY-SA-4.0 / CC0 / PSF-2.0 / Public Domain)
```bash
python -m datasets.chat.acquisition --fetch-open-batch
python -m datasets.chat.acquisition --list-sources
```
Every imported source logs original URL/permalink, author/publisher, exact license, attribution text, SHA-256 of raw fetched bytes, retrieval timestamp, derived record IDs, transformation notes, and reviewer approval status in `datasets/sources/source_registry.json`.

### B. Review, Approve, Reject & Regenerate Production Manifests
```bash
# Check overall progress, pending count, approved count, and shortfall:
python -m datasets.chat.review_queue --status

# Inspect pending items:
python -m datasets.chat.review_queue --list

# Inspect full details & source extract for a staged draft:
python -m datasets.chat.review_queue --show <DRAFT_ID>

# Author an original educational explanation for a staged source draft (stays PENDING_REVIEW):
python -m datasets.chat.review_queue --complete-draft <DRAFT_ID> --answer "..."

# Approve verified records (fails closed if answer is empty, copied verbatim, or unreviewed):
python -m datasets.chat.review_queue --approve <ID_1> <ID_2> --reviewer "Prem Gaikwad"

# Reject invalid/low-quality items:
python -m datasets.chat.review_queue --reject <ID> --reviewer "Prem Gaikwad" --reason "..."

# Deterministically rebuild prod_chat_{train,val,test}.jsonl + provenance + progress JSON:
python -m datasets.chat.review_queue --rebuild-manifests
```
