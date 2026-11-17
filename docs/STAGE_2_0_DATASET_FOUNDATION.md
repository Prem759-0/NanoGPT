# Moon AI — Stage 2.0 Balanced Dataset Foundation & Quality Gate Report

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Repository**: `Prem759-0/NanoGPT`  
**Base Commit**: `6f7736dcffc3e85df811207fa1eac54275caa80c`  
**Branch**: `feature/stage-2-dataset-foundation`  
**Date**: November 1, 2026  
**Status**: COMPLETE & VERIFIED  
**Ownership Policy**: STRICT SOVEREIGN — 100% Random Weight Initialization, Permissible Open Licenses Only, Zero External Generative/Vision APIs.

---

## Executive Summary

Stage 2.0 establishes the foundational dataset engineering infrastructure, ingestion tooling, and quality gates for Moon AI. Rather than relying on unverified web scraping or conflating tiny development fixtures with production adequacy, Stage 2.0 provides:
1. **Strict Data Classification**: Complete separation of legacy procedural fixtures from genuine source media.
2. **Comprehensive Ingestion Tools**: Built tools for image and video ingestion, normalization, perceptual deduplication, frame extraction, and license verification.
3. **Versioned Separated Manifests**: Separate `train`, `val`, and `test` manifests with source-level leakage prevention.
4. **Decoupled Quality Gates**: Transparent evaluation of technical hygiene (`HYGIENE_VALID: PASS`) separate from volume adequacy (`TRAINING_DATA_ADEQUATE: INSUFFICIENT`).
5. **Google Colab T4 Preparation**: Standalone dataset verification runner with Google Drive persistent sync.

---

## 1. Audit of Existing Repository Media

| Asset Path | Records Count | Genuine Source or Fixture? | Classification & Action |
|---|---|---|---|
| `data/images_3d/` | 7 images | **Procedural Fixture** | Procedural 3D geometric seed spheres generated during early testing. **Quarantined** into `datasets/manifests/synthetic_image_dev.jsonl`. |
| `data/videos/` | 2 MP4 videos | **Synthetic Benchmark** | Procedural rotating spheres rendered via script. Valid H.264 MP4 containers, but synthetic benchmark footage. **Quarantined** into `datasets/manifests/synthetic_video_dev.jsonl`. |
| `datasets/video/synthetic_dev_manifest.jsonl` | 2 records | **Dev Simulation** | Crop-shift simulated motion on still images. Isolated exclusively for pipeline unit tests. |

---

## 2. Implemented Ingestion & Quality Tooling (`datasets/tools/`)

1. **`datasets/tools/deduplicate_images.py`**:
   - Exact SHA-256 duplicate detection.
   - Near-duplicate detection via 64-bit difference hash (dHash) and average hash (aHash) with configurable Hamming distance threshold ($\le 4$ bits).
   - Implemented natively using Pillow & NumPy with zero external pretrained models.
2. **`datasets/tools/validate_image_sources.py`**:
   - Validates permissible open licenses (`CustomSovereign`, `CC0`, `CC-BY-4.0`, `MIT`, `Apache-2.0`, `PublicDomain`).
   - Validates the 10 balanced categories.
   - Enforces minimum caption quality ($\ge 10$ characters; rejects generic prefixes).
3. **`datasets/tools/normalize_images.py`**:
   - Verifies image integrity and decodability.
   - Converts to sRGB 3-channel mode.
   - Performs aspect-ratio-preserving center cropping and Lanczos resizing to 256x256.
4. **`datasets/tools/collect_images.py`**:
   - End-to-end ingestion orchestrator: archives raw originals in `data/raw/images/`, normalizes copies to `data/images/`, generates deterministic IDs, and writes schema 2.0.0 records.
5. **`datasets/tools/inspect_video.py`**:
   - Uses FFmpeg/imageio to inspect container codec, spatial dimensions, duration, FPS, and frame count.
   - Rejects clips shorter than 1.0s or with fewer than 8 frames.
6. **`datasets/tools/extract_video_frames.py`**:
   - Decodes genuine ordered frames with exact timestamps in seconds.
7. **`datasets/tools/validate_video_sources.py`**:
   - Validates video licensing, creator, scene caption, motion description, and camera motion vocabulary.
8. **`datasets/tools/collect_videos.py`**:
   - End-to-end video ingestion engine: archives raw originals in `data/raw/videos/`, copies to `data/videos/`, and writes schema 2.0.0 video records.
9. **`datasets/tools/dataset_manager.py`**:
   - Master quality gate auditor evaluating `HYGIENE_VALID` and `TRAINING_DATA_ADEQUATE`.

---

## 3. Versioned Manifest Schema (Version 2.0.0)

All manifests in `datasets/manifests/` adhere to the version 2.0.0 schema:

### 3.1 Image Records
```json
{
  "schema_version": "2.0.0",
  "id": "img_landscapes_6539fe82bc",
  "media_path": "data/images/img_landscapes_6539fe82bc.png",
  "caption": "Alpine mountain peaks bathed in golden hour twilight with violet haze",
  "category": "landscapes",
  "source_url": "https://github.com/Prem759-0/NanoGPT/tree/main/data/images/landscapes",
  "creator": "Prem Gaikwad / Moon AI Project",
  "license": "CustomSovereign",
  "attribution": "Original media authored by Prem Gaikwad / Moon AI Project for Moon AI Project (CustomSovereign)",
  "sha256": "44db3e72eb4025d2b781da6194b1a437a3c3933c1f01fc63eb52ba1ebcfb603a",
  "dhash": "3999999999b99918",
  "split": "train",
  "width": 256,
  "height": 256,
  "channels": 3,
  "is_synthetic": false
}
```

### 3.2 Video Records
```json
{
  "schema_version": "2.0.0",
  "id": "vid_vehicles_88a0417209",
  "media_path": "data/videos/vid_vehicles_88a0417209.mp4",
  "caption": "Cinematic video clip in vehicles domain demonstrating smooth orbit_left movement",
  "motion_description": "Camera continuously orbits counter-clockwise around central subject",
  "category": "vehicles",
  "creator": "Prem Gaikwad / Moon AI Project",
  "license": "CustomSovereign",
  "camera_motion": "orbit_left",
  "sha256": "38a846fdb75ba95955683b5443fa626e255ee988fa37d2f95b5e3962d3a95ab9",
  "split": "train",
  "width": 128,
  "height": 128,
  "fps": 8.0,
  "duration_sec": 2.0,
  "num_frames": 16,
  "timestamps": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
  "codec": "h264",
  "is_synthetic": false
}
```

---

## 4. Quality Gate Audit Results

Executing [`datasets/tools/dataset_manager.py`](file:///d:/Downloads/linkden%20post/ai/datasets/tools/dataset_manager.py) yields:

- **`HYGIENE_VALID: ✅ PASS`**:
  - Missing media files: **0**
  - Corrupted files: **0**
  - Exact duplicate SHA-256 hashes: **0**
  - Near-duplicate dHash matches: **0**
  - Split leakages (train/val/test overlap): **0**
  - Invalid licenses: **0**
- **`TRAINING_DATA_ADEQUATE: ⚠️ INSUFFICIENT (PILOT IN PROGRESS)`**:
  - Genuine image samples: **20 / 500** (Shortfall: 480 required for full pilot target)
  - Genuine video clips: **5 / 50** (Shortfall: 45 required for full pilot target)
  - Category coverage: **10/10 categories represented**

---

## 5. Next Steps

1. **Stage 2.1 — Moon Image Pilot Training**:
   Execute pilot training run with `datasets/manifests/image_train.jsonl` on Google Colab T4 (`scripts/colab_train.py --track image --steps 50`).
2. **Stage 3.0 — Moon Video Pilot Training**:
   Execute spatiotemporal pilot run with `datasets/manifests/video_train.jsonl` on Colab T4 (`scripts/colab_train.py --track video --steps 25`).
3. **Batch Collection Ingestion**:
   Use `collect_images.py` and `collect_videos.py` to ingest the next batch of licensed assets toward the 500/50 targets.

