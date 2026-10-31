# Moon AI — Sovereign Dataset Collection & Curation Guide

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Strict Ownership Policy**: Permissive open-license and public-domain data only. ZERO scraping of copyrighted or proprietary assets.

---

## 1. Permissible Licenses & Data Sources

All training assets included in Moon AI manifests must hold one of the following permissible licenses:
- `CC0` (Public Domain Dedication)
- `CC-BY-4.0` / `CC-BY-SA-4.0` (Attribution required)
- `MIT` / `Apache-2.0` / `BSD-3-Clause`
- `OpenData` / `PublicDomain`
- `CustomSovereign` (Authored directly by Prem Gaikwad / Moon AI project)

---

## 2. Manifest Schema Specifications

### 2.1 Image Manifest (`datasets/image/manifest.jsonl`)
Every line is a valid JSON object:
```json
{
  "id": "img_sample_01",
  "media_path": "data/images/neon_landscape.png",
  "caption": "A futuristic city with gleaming cybernetic towers and neon reflections in the rain",
  "sha256": "8f31b...",
  "split": "train",
  "license": "CC0"
}
```

### 2.2 Video Manifest (`datasets/video/manifest.jsonl`)
Must point to genuine video containers (`.mp4`, `.webm`):
```json
{
  "id": "real_vid_01",
  "media_path": "data/videos/sample_real_video.mp4",
  "caption": "Drone camera orbiting a mountain pass at sunset",
  "camera_motion": "orbit_left",
  "fps": 8.0,
  "duration_sec": 3.0,
  "num_frames": 24,
  "sha256": "4b92c...",
  "split": "train",
  "license": "CustomSovereign"
}
```

---

## 3. Step-by-Step Dataset Ingestion Workflow

### Step 1: Place Media in Designated Folder
- Place new images in `data/images/`
- Place new video files in `data/videos/`

### Step 2: Calculate File SHA-256 Checksum
In Python:
```python
import hashlib
def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
```

### Step 3: Append Record to JSONL Manifest
Ensure the record has: `id`, `media_path`, `caption`, `sha256`, `split` (`train`, `val`, or `test`), and `license`.

### Step 4: Run Automated Hygiene Audit
Execute the automated dataset manager:
```bash
python datasets/tools/dataset_manager.py
```
This guarantees:
1. No missing media paths.
2. No corrupted image or video headers.
3. No duplicate SHA-256 files.
4. No split leakage (an asset appearing in both train and val).
5. All licenses are permissible.
6. Refreshed `datasets/dataset_report.json` and `datasets/dataset_report.md`.

