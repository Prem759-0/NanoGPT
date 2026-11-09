"""
datasets/tools/inspect_video.py
Video Inspection and Validation Utility for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Uses local FFmpeg / imageio to inspect:
- Container format and video codec
- Spatial dimensions (width, height)
- Temporal duration (seconds), FPS, and total frame count
- Rejects truncated or unreadable containers
"""
import os
import sys
import subprocess
import json
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_VIDEO_CONTAINERS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

def inspect_video_file(video_path: str, min_duration_sec: float = 1.0, min_frames: int = 8) -> Dict[str, Any]:
    """
    Inspects a video file and returns its technical metadata.
    Raises ValueError if corrupted, too short, or unsupported.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_CONTAINERS:
        raise ValueError(f"Unsupported video container '{ext}'. Allowed: {sorted(list(SUPPORTED_VIDEO_CONTAINERS))}")

    # Inspect using imageio / imageio_ffmpeg
    try:
        import imageio.v3 as iio
        props = iio.improps(video_path, plugin="pyav")
        shape = props.shape  # typically (num_frames, H, W, C) or (H, W, C)
        n_frames = shape[0] if len(shape) == 4 else 0
        h = shape[1] if len(shape) == 4 else shape[0]
        w = shape[2] if len(shape) == 4 else shape[1]
    except Exception:
        # Fallback to imageio ffmpeg reader
        try:
            import imageio
            reader = imageio.get_reader(video_path, "ffmpeg")
            meta = reader.get_meta_data()
            w, h = meta.get("size", (0, 0))
            fps = float(meta.get("fps", 0.0))
            duration = float(meta.get("duration", 0.0))
            n_frames = reader.count_frames() if hasattr(reader, "count_frames") else int(fps * duration)
            reader.close()
            return {
                "valid": True,
                "path": video_path,
                "width": int(w),
                "height": int(h),
                "fps": round(fps, 2),
                "duration_sec": round(duration, 2),
                "num_frames": int(n_frames),
                "codec": meta.get("codec", "unknown")
            }
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable video file '{video_path}': {str(e)}")

    try:
        import imageio
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 0.0))
        duration = float(meta.get("duration", 0.0))
        reader.close()
    except Exception:
        fps = 8.0
        duration = n_frames / fps if n_frames > 0 else 0.0

    if duration < min_duration_sec:
        raise ValueError(f"Video duration ({duration:.2f}s) is below required minimum ({min_duration_sec}s).")

    if n_frames < min_frames:
        raise ValueError(f"Video frame count ({n_frames}) is below required minimum ({min_frames} frames).")

    return {
        "valid": True,
        "path": video_path,
        "width": int(w),
        "height": int(h),
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "num_frames": int(n_frames),
        "codec": meta.get("codec", "unknown") if "meta" in locals() else "unknown"
    }

