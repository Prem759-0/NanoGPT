# MOON AI — STAGE 2.0.1: CRITICAL DATASET PROVENANCE REPAIR & PRODUCTION TRAINING GATES

**Repository:** `Prem759-0/NanoGPT`  
**Branch:** `feature/stage-2-0-1-provenance-repair`  
**Base Commit:** `434df8a8192e4a15b9e9c5efdebb60934f0c0cac`  
**Project Owner & Lead ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`

---

## 1. Executive Summary & Root Cause Audit

During the Stage 2.0 review, a critical dataset provenance and misclassification issue was discovered in `scripts/build_pilot_datasets.py`:
- 30 images generated via procedural Pillow drawing commands (`create_distinct_pilot_image()`) and 8 video clips generated via procedural geometric animations (`create_distinct_pilot_video_clip()`) had been registered with `is_synthetic: false` in production manifests (`image_train.jsonl`, `video_train.jsonl`, etc.).
- Furthermore, self-referential internal repository folder paths (e.g. `https://github.com/Prem759-0/NanoGPT/tree/main/data/...`) were populated as `source_url`.
- Production training manifests were incorrectly presented as containing genuine real-world media, and models were marked `DATA_READY`.

### Corrective Actions Implemented in Stage 2.0.1:
1. **Reclassification & Quarantine:**
   - All 30 procedural images and 8 procedural videos were reclassified with `is_synthetic: true`, `source_type: "procedural_synthetic_fixture"`, `provenance_type: "synthetic_dev_fixture"`, and quarantined exclusively to `synthetic_image_dev.jsonl` (37 total) and `synthetic_video_dev.jsonl` (10 total).
   - Production manifests (`image_train.jsonl`, `image_val.jsonl`, `image_test.jsonl`, `video_train.jsonl`, `video_val.jsonl`, `video_test.jsonl`) were emptied to **0 records**.
   - Chat (4 records) and Code (3 records) were reclassified as development fixtures in `synthetic_chat_dev.jsonl` and `synthetic_code_dev.jsonl`.
2. **Separated Workflows:**
   - Created `scripts/build_synthetic_fixtures.py` dedicated exclusively to generating synthetic test fixtures. It writes only to `synthetic_*_dev.jsonl` and is prohibited from touching production manifests.
   - Refactored `scripts/build_pilot_datasets.py` with deprecation guards against writing to production manifests.
   - Ingestion tools (`collect_images.py`, `collect_videos.py`) enforce verifiable external URLs (`http://`, `https://`, `owner://`) and reject self-referential repo paths.
3. **Strict Production Training Gates:**
   - Hardened `MoonImageDataset` and `MoonVideoDataset`: production mode (`allow_synthetic=False`) strictly rejects `is_synthetic=True`, dev manifests, missing provenance metadata, and empty manifests.
   - Added `--allow_synthetic` development mode for pipeline debugging; development mode disables production checkpoint emission.
4. **Aggregate Dataset Auditing:**
   - Overhauled `datasets/tools/dataset_manager.py` to aggregate all four splits: `train`, `val`, `test`, and `dev`.
   - Reports truthful counts: **0 Genuine Images, 0 Genuine Videos**.
   - Decoupled `HYGIENE_VALID: PASS` from `GENUINE_DATA_PRESENT: NO` and `TRAINING_DATA_ADEQUATE: INSUFFICIENT`.
5. **Truthful Model Registry:**
   - Updated `models/registry.json`: `moon-image-v0`, `moon-video-v0`, `moon-chat-v0`, and `moon-code-v0` are honestly marked `DATA_INSUFFICIENT` (`training_ready: false`, `genuine_samples_count: 0`).
   - Production training is blocked until genuine real-world media is collected and validated.

---

## 2. Manifest Inventory & Asset Breakdown

| Modality | Train Manifest | Val Manifest | Test Manifest | Synthetic Dev Manifest | Genuine Count | Synthetic Count | Pilot Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image** | `image_train.jsonl` (0) | `image_val.jsonl` (0) | `image_test.jsonl` (0) | `synthetic_image_dev.jsonl` (37) | **0** | **37** | 500 | `DATA_INSUFFICIENT` |
| **Video** | `video_train.jsonl` (0) | `video_val.jsonl` (0) | `video_test.jsonl` (0) | `synthetic_video_dev.jsonl` (10) | **0** | **10** | 50 | `DATA_INSUFFICIENT` |
| **Chat** | `chat_train.jsonl` (0) | `chat_val.jsonl` (0) | — | `synthetic_chat_dev.jsonl` (4) | **0** | **4** | 1,000 | `DATA_INSUFFICIENT` |
| **Code** | `code_train.jsonl` (0) | `code_val.jsonl` (0) | — | `synthetic_code_dev.jsonl` (3) | **0** | **3** | 1,000 | `DATA_INSUFFICIENT` |

---

## 3. Strict Sovereign Ownership Assurances

- ZERO third-party pretrained weights (no SDXL, Flux, ModelScope, CLIP, T5).
- ZERO external generative APIs (no Pollinations, OpenAI, Anthropic, Replicate).
- 100% Random Weight Initialization for all architectures.
- All checkpoints enforce `provenance: "moon_ai_random_init"`.
- Production training fails closed if genuine real-world media is missing.
