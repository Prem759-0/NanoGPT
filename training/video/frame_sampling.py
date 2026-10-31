"""
training/video/frame_sampling.py
Video Frame Extraction and Temporal Sampling Utilities for Moon Video.
"""
import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch

def sample_frames_from_sequence(image_paths: List[str], target_frames: int = 16, image_size: int = 128) -> torch.Tensor:
    """
    Loads and resizes an ordered sequence of image frames into a video tensor [C, F, H, W].
    """
    total = len(image_paths)
    if total == 0:
        raise ValueError("Empty image sequence provided.")

    if total >= target_frames:
        indices = np.linspace(0, total - 1, target_frames, dtype=int)
    else:
        # Loop / pad if fewer frames available
        indices = [i % total for i in range(target_frames)]

    frames = []
    for idx in indices:
        path = image_paths[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.array(im).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            frames.append(arr)

    # [F, H, W, C] -> [C, F, H, W]
    video_np = np.stack(frames, axis=0)
    video_tensor = torch.tensor(video_np).permute(3, 0, 1, 2)
    return video_tensor

def create_synthetic_motion_clip(base_image_path: str, num_frames: int = 16, image_size: int = 128, motion_type: str = "orbit_left") -> torch.Tensor:
    """
    Generates a continuous synthetic camera movement clip from a base image for development testing.
    """
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
            # Crop tighter over time
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
    return torch.tensor(video_np).permute(3, 0, 1, 2)
