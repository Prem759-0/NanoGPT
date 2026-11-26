# Moon AI Development / Scaled Curriculum Corpora (Quarantined from Production)

**Classification:** `DETERMINISTIC_TEMPLATE` (Development / Engineering Benchmark / Legacy Regression Only)  
**Production Release Eligibility:** **DISALLOWED (`False`)**  

## Why This Directory Exists

Previous pilot scaling stages (`Stage 2.1.6` / `Stage 2.1.7` / `Stage 2.4`) generated `expanded_chat_*.jsonl` (572 records) and `scaled_chat_*.jsonl` (1,100 records) using deterministic Python curriculum templates (`teacher/factory/curriculum_generator.py` and `teacher/factory/curator.py`).

While those corpora remain temporarily referenced at their legacy paths (`datasets/manifests/expanded_chat_*.jsonl` and `datasets/manifests/scaled_chat_*.jsonl`) solely so backward-compatible unit and hardware benchmark tests continue to function:

1. **100% of `scaled_chat_*.jsonl` (1,100 / 1,100)** consists of programmatic template variants (`Variant N:` prompts + `"Overview of ... Key Takeaway:"` boilerplate replies) with `local://scaled_curriculum_generator` provenance and zero external URLs.
2. There are only **261 unique first-turn assistant replies** across 1,100 records (the top template answer repeats **110 times**).
3. There is severe **cross-split assistant reply leakage** (`train ∩ val = 36`, `train ∩ test = 37`, `val ∩ test = 32`), meaning train, validation, and test splits measure memorization of repeated template answers rather than real generalization.
4. `expanded_chat_*.jsonl` contains **550 deterministic template records** mixed with the **22** genuine baseline records.

## Rule for Moon Chat v0.1 Production Training

No `DETERMINISTIC_TEMPLATE`, `SYNTHETIC_FIXTURE`, `AI_GENERATED`, or `UNVERIFIED` corpus may ever be used for Moon Chat v0.1 production fine-tuning (`training/chat/v0_1_t4_release_config.yaml`). Production training manifests must be built exclusively through `datasets/chat/genuine_ingestion.py` and pass `verify_production_manifest_group()` (`HUMAN_AUTHORED_VERIFIED` + `PUBLIC_SOURCE_VERIFIED` only, zero cross-split hash/prompt/reply leakage).
