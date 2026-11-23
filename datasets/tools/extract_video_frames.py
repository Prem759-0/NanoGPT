"""
datasets/tools/extract_video_frames.py
Ordered Frame and Timestamp Extraction Engine for Moon Video.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Extracts real ordered video frames from genuine MP4/WebM files with exact timestamps.
"""
import os
import sys
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def extract_ordered_frames(
    video_path: str,
    target_num_frames: int = 16,
    target_size: int = 128
) -> Dict[str, Any]:
    """
    Decodes genuine video frames in strict temporal order.
    Returns:
    - frames: list of PIL Images (RGB, target_size x target_size)
    - timestamps: list of float timestamps in seconds
    - source_fps: float
    - total_source_frames: int
    - duration_sec: float
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import imageio

    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 8.0))
        duration = float(meta.get("duration", 0.0))

        raw_frames = []
        for i, frame in enumerate(reader):
            raw_frames.append(frame)
        reader.close()
    except Exception as e:
        raise ValueError(f"Failed to decode video '{video_path}': {str(e)}")

    total_frames = len(raw_frames)
    if total_frames < 2:
        raise ValueError(f"Video '{video_path}' has fewer than 2 frames ({total_frames} frames).")

    # Sample exactly target_num_frames uniformly
    if total_frames >= target_num_frames:
        indices = np.linspace(0, total_frames - 1, target_num_frames, dtype=int)
    else:
        # Loop / repeat if slightly short
        indices = np.pad(np.arange(total_frames), (0, target_num_frames - total_frames), mode="edge")

    sampled_frames: List[Image.Image] = []
    timestamps: List[float] = []

    for idx in indices:
        f_arr = raw_frames[idx]
        im = Image.fromarray(f_arr).convert("RGB")

        # Center crop to 1:1 and resize
        w, h = im.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        im_cropped = im.crop((left, top, left + min_dim, top + min_dim))
        im_resized = im_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        sampled_frames.append(im_resized)
        ts = round(float(idx / fps), 3) if fps > 0 else 0.0
        timestamps.append(ts)

    return {
        "frames": sampled_frames,
        "timestamps": timestamps,
        "source_fps": fps,
        "duration_sec": duration,
        "total_source_frames": total_frames,
        "sampled_num_frames": len(sampled_frames)
    }

