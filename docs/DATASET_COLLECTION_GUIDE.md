# Moon AI — Sovereign Dataset Collection & Ingestion Guide

**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Version**: `2.0.0` (Stage 2.0 Balanced Foundation)  
**Ownership Policy**: STRICT SOVEREIGN — Permissible Open Licenses Only, Zero AI-Assisted Auto-Labeling.

---

## 1. Balanced Category Taxonomy

All collected images and videos must be classified into one of the 10 balanced categories:
1. `landscapes`: Mountains, oceans, forests, beaches, sunsets, natural terrain.
2. `architecture`: Modern skyscrapers, historic temples, bridges, interior design, pavilions.
3. `products`: Watches, electronics, fragrances, footwear, crafted goods.
4. `technology`: Semiconductors, circuit boards, fiber optics, datacenters, robotics.
5. `people`: Studio portraits, silhouettes, hands crafting, athletic movement.
6. `animals`: Wildlife, birds in flight, marine life, insects.
7. `vehicles`: Cars, trains, aircraft, ships, bicycles.
8. `food`: Baked goods, culinary bowls, fresh fruits, vegetables, beverages.
9. `illustration_design`: Bauhaus graphics, geometric posters, ukiyo-e woodblock, synthwave.
10. `abstract_scientific`: Molecular orbitals, fractal mathematics, astronomy, crystal lattices.

---

## 2. Ingestion Commands

### 2.1 Ingesting an Image File
Use [`datasets/tools/collect_images.py`](file:///d:/Downloads/linkden%20post/ai/datasets/tools/collect_images.py):
```python
from datasets.tools.collect_images import ImageCollector

collector = ImageCollector()
record = collector.ingest_image(
    src_file_path="path/to/my_photo.jpg",
    caption="A snow-covered alpine mountain summit glowing in the morning sunrise",
    category="landscapes",
    creator="Prem Gaikwad",
    source_url="https://github.com/Prem759-0/NanoGPT",
    license_id="CustomSovereign",
    split="train"  # "train", "val", or "test"
)
```
**Automated Safeguards**:
- Rejects missing/invalid licenses.
- Rejects exact SHA-256 duplicates.
- Rejects near-duplicates using dHash perceptual hashing (Hamming distance $\le 4$).
- Preserves raw original in `data/raw/images/`.
- Normalizes processed copy to 256x256 RGB PNG in `data/images/`.

### 2.2 Ingesting a Video Clip
Use [`datasets/tools/collect_videos.py`](file:///d:/Downloads/linkden%20post/ai/datasets/tools/collect_videos.py):
```python
from datasets.tools.collect_videos import VideoCollector

collector = VideoCollector()
record = collector.ingest_video(
    src_video_path="path/to/my_clip.mp4",
    caption="Drone footage orbiting a coastal lighthouse at sunset",
    motion_description="Camera orbits 180 degrees counter-clockwise around lighthouse",
    category="architecture",
    creator="Prem Gaikwad",
    source_url="https://github.com/Prem759-0/NanoGPT",
    license_id="CustomSovereign",
    camera_motion="orbit_left",
    split="train"
)
```
**Automated Safeguards**:
- Rejects corrupted or truncated video streams.
- Rejects clips shorter than 1.0 second or with fewer than 8 frames.
- Decodes real ordered frames with timestamps.
- Preserves raw original in `data/raw/videos/`.

---

## 3. Manifest Splitting & Leakage Prevention Rules

1. **Source / Creator-Level Grouping**: Never place assets from the same specific creator or source batch across both `train` and `test` splits.
2. **Versioned Manifest Locations**:
   - `datasets/manifests/image_train.jsonl`
   - `datasets/manifests/image_val.jsonl`
   - `datasets/manifests/image_test.jsonl`
   - `datasets/manifests/video_train.jsonl`
   - `datasets/manifests/video_val.jsonl`
   - `datasets/manifests/video_test.jsonl`

---

## 4. Running the Quality Gate Audit

Always run the quality gate audit after adding media:
```bash
python datasets/tools/dataset_manager.py
```
This produces:
- `datasets/dataset_report.json`
- `datasets/dataset_report.md`
With explicit reporting of both `HYGIENE_VALID` and `TRAINING_DATA_ADEQUATE`.
