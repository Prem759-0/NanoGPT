"""
datasets/tools/normalize_images.py
Image Normalization and Integrity Pipeline for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Verifies:
1. Image can be opened and decoded without corruption.
2. Color mode converted cleanly to 3-channel sRGB.
3. Resizes with high-quality Lanczos resampling and aspect-preserving center crop.
4. Computes SHA-256 and perceptual hash of normalized output.
"""
import os
import sys
import hashlib
from typing import Dict, Any, Tuple, Optional
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.deduplicate_images import compute_sha256, compute_dhash

def normalize_single_image(
    src_path: str,
    dest_path: str,
    target_size: int = 256,
    quality: int = 95
) -> Dict[str, Any]:
    """
    Normalizes a source image:
    - Verifies file integrity
    - Converts to RGB
    - Performs aspect-ratio-preserving center crop and resize to (target_size, target_size)
    - Saves normalized PNG or JPG
    - Returns detailed metadata
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source image not found: {src_path}")

    # 1. Integrity Verification
    with Image.open(src_path) as im:
        im.verify()

    # Re-open after verify()
    with Image.open(src_path) as im:
        orig_w, orig_h = im.size
        orig_format = im.format
        orig_mode = im.mode

        # Convert to RGB
        im_rgb = im.convert("RGB")

        # Aspect-preserving center crop to 1:1
        min_dim = min(orig_w, orig_h)
        left = (orig_w - min_dim) // 2
        top = (orig_h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped = im_rgb.crop((left, top, right, bottom))

        # Resize to target resolution
        resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        # Save as PNG for lossless training stability
        resized.save(dest_path, format="PNG", optimize=True)

    norm_sha256 = compute_sha256(dest_path)
    norm_dhash = compute_dhash(dest_path)

    return {
        "original_width": orig_w,
        "original_height": orig_h,
        "original_format": orig_format,
        "original_mode": orig_mode,
        "normalized_width": target_size,
        "normalized_height": target_size,
        "channels": 3,
        "normalized_path": dest_path,
        "sha256": norm_sha256,
        "dhash": norm_dhash
    }

