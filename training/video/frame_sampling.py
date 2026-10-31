"""
training/video/frame_sampling.py
Video Frame Extraction and Temporal Sampling Utilities for Moon Video.
Supports:
1. Real video decoding from actual video files (.mp4, .webm, .mov, etc.) via imageio / FFmpeg.
2. Ordered frame extraction with timestamp and FPS preservation.
3. Isolated synthetic motion generation for development and unit testing only.
"""
import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class CorruptVideoError(Exception):
    """Raised when a video file cannot be opened, decoded, or contains zero valid frames."""
    pass


def decode_real_video_clip(
    video_path: str,
    target_frames: int = 16,
    frame_size: int = 128,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Decodes an actual video file into a standardized tensor [C, F, H, W] normalized to [-1, 1].
    Preserves true temporal ordering, source FPS, and timestamp information.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at '{video_path}'.")

    if not IMAGEIO_AVAILABLE:
        raise RuntimeError("imageio is required for real video decoding.")

    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
    except Exception as e:
        raise CorruptVideoError(f"Failed to open video '{video_path}': {e}")

    source_fps = meta.get("fps", 24.0) or 24.0
    total_source_frames = 0
    try:
        total_source_frames = reader.count_frames()
    except Exception:
        # Some formats don't support count_frames, we estimate from duration
        duration = meta.get("duration", 0.0)
        if duration > 0:
            total_source_frames = int(duration * source_fps)

    # Read frames into list
    raw_frames = []
    try:
        for i, frame in enumerate(reader):
            raw_frames.append(frame)
            # Cap reading if total frames is huge (> 500) to conserve memory
            if duration_sec and i > int((start_sec + duration_sec + 1.0) * source_fps):
                break
    except Exception as e:
        # If reader fails midway but read enough frames
        if len(raw_frames) == 0:
            raise CorruptVideoError(f"Corrupt video file '{video_path}': could not read any frames ({e})")
    finally:
        try:
            reader.close()
        except Exception:
            pass

    actual_read_count = len(raw_frames)
    if actual_read_count == 0:
        raise CorruptVideoError(f"Video file '{video_path}' contained 0 decodeable frames.")

    # Calculate temporal slice
    start_frame = max(0, int(start_sec * source_fps))
    if duration_sec:
        end_frame = min(actual_read_count, start_frame + int(duration_sec * source_fps))
    else:
        end_frame = actual_read_count

    if start_frame >= actual_read_count:
        start_frame = 0
        end_frame = actual_read_count

    selected_slice = raw_frames[start_frame:end_frame]
    slice_count = len(selected_slice)
    if slice_count == 0:
        selected_slice = raw_frames
        slice_count = len(selected_slice)

    # Sample exactly target_frames uniformly along the time axis
    if slice_count >= target_frames:
        indices = np.linspace(0, slice_count - 1, target_frames, dtype=int)
    else:
        # Loop / repeat if video is shorter than target_frames
        indices = [i % slice_count for i in range(target_frames)]

    processed_frames = []
    timestamps = []
    for idx in indices:
        f = selected_slice[idx]
        t_sec = round((start_frame + idx) / source_fps, 3)
        timestamps.append(t_sec)

        im = Image.fromarray(f).convert("RGB")
        im = im.resize((frame_size, frame_size), Image.Resampling.BILINEAR)
        arr = np.array(im).astype(np.float32) / 127.5 - 1.0  # Normalized to [-1, 1]
        processed_frames.append(arr)

    # [F, H, W, C] -> [C, F, H, W]
    video_np = np.stack(processed_frames, axis=0)
    video_tensor = torch.tensor(video_np).permute(3, 0, 1, 2)

    metadata = {
        "source_fps": round(float(source_fps), 2),
        "total_source_frames": actual_read_count,
        "sampled_frames": target_frames,
        "timestamps_sec": timestamps,
        "duration_sec": round(actual_read_count / source_fps, 2) if source_fps > 0 else 0.0,
        "is_synthetic": False
    }

    return video_tensor, metadata


def create_synthetic_motion_clip(
    base_image_path: str,
    num_frames: int = 16,
    image_size: int = 128,
    motion_type: str = "orbit_left"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generates a continuous synthetic camera movement clip from a base image.
    ISOLATED UTILITY FOR UNIT TESTS AND DEVELOPMENT PIPELINE VALIDATION ONLY.
    """
    if not os.path.isfile(base_image_path):
        raise FileNotFoundError(f"Base image not found at '{base_image_path}'.")

    with Image.open(base_image_path) as im:
        im = im.convert("RGB")
        im_large = im.resize((int(image_size * 1.3), int(image_size * 1.3)), Image.Resampling.BILINEAR)

    frames = []
    max_offset = int(image_size * 0.3)

    for i in range(num_frames):
        alpha = i / max(1, num_frames - 1)
        if motion_type == "pan_right":
            left = int(alpha * max_offset)
            top = max_offset // 2
        elif motion_type == "dolly_in":
            crop_shrink = int((1.0 - alpha * 0.5) * max_offset)
            left = crop_shrink // 2
            top = crop_shrink // 2
        else:  # orbit_left / default
            left = int((1.0 - alpha) * max_offset)
            top = max_offset // 2

        crop = im_large.crop((left, top, left + image_size, top + image_size))
        arr = np.array(crop).astype(np.float32) / 127.5 - 1.0
        frames.append(arr)

    video_np = np.stack(frames, axis=0)
    video_tensor = torch.tensor(video_np).permute(3, 0, 1, 2)

    metadata = {
        "source_fps": 8.0,
        "total_source_frames": num_frames,
        "sampled_frames": num_frames,
        "motion_type": motion_type,
        "is_synthetic": True
    }
    return video_tensor, metadata
