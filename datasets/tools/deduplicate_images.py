"""
datasets/tools/deduplicate_images.py
Exact and Near-Duplicate Image Detection Suite for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Uses:
1. Exact SHA-256 cryptographic hashing.
2. Perceptual Difference Hashing (dHash) and Average Hashing (aHash) in pure Python/NumPy/PIL.
Zero external pretrained vision models or third-party embeddings.
"""
import os
import sys
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def compute_sha256(file_path: str) -> str:
    """Computes exact cryptographic SHA-256 checksum."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image_input, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) tracking horizontal luminance gradients.
    Resizes image to (hash_size + 1, hash_size) in grayscale, compares adjacent pixels.
    Produces a (hash_size * hash_size) bit integer formatted as hexadecimal.
    """
    if isinstance(image_input, str):
        with Image.open(image_input) as img:
            return compute_dhash(img, hash_size=hash_size)

    # Convert to grayscale and resize to (hash_size + 1, hash_size)
    img_gray = image_input.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS
    )
    pixels = np.array(img_gray, dtype=np.int32)

    # Difference between adjacent horizontal pixels
    diff = pixels[:, 1:] > pixels[:, :-1]

    # Convert boolean 2D array to binary string and then hex
    bits = "".join(["1" if b else "0" for b in diff.flatten()])
    hex_str = f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    return hex_str

def compute_ahash(image_input, hash_size: int = 8) -> str:
    """
    Computes average hash (aHash) comparing pixels against the mean luminance.
    """
    if isinstance(image_input, str):
        with Image.open(image_input) as img:
            return compute_ahash(img, hash_size=hash_size)

    img_gray = image_input.convert("L").resize(
        (hash_size, hash_size),
        Image.Resampling.LANCZOS
    )
    pixels = np.array(img_gray, dtype=np.float32)
    avg = pixels.mean()
    diff = pixels > avg
    bits = "".join(["1" if b else "0" for b in diff.flatten()])
    hex_str = f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    return hex_str

def hamming_distance(hex1: str, hex2: str) -> int:
    """Calculates Hamming distance between two equal-length hex hash strings."""
    if len(hex1) != len(hex2):
        raise ValueError(f"Hash lengths differ: {len(hex1)} vs {len(hex2)}")
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")

def find_duplicates(
    image_paths: List[str],
    phash_threshold: int = 4
) -> Dict[str, Any]:
    """
    Scans a collection of image paths and identifies:
    - Exact SHA-256 duplicates
    - Near-duplicates where dHash Hamming distance <= phash_threshold
    """
    sha_map: Dict[str, str] = {}
    dhash_map: Dict[str, Tuple[str, str]] = {}  # path -> (sha256, dhash)

    exact_duplicates: List[Dict[str, str]] = []
    near_duplicates: List[Dict[str, Any]] = []

    for path in image_paths:
        if not os.path.isfile(path):
            continue
        try:
            sha = compute_sha256(path)
            dh = compute_dhash(path)
        except Exception:
            continue

        # Check exact duplicate
        if sha in sha_map:
            exact_duplicates.append({
                "original": sha_map[sha],
                "duplicate": path,
                "sha256": sha
            })
        else:
            sha_map[sha] = path

        # Check near duplicate across existing unique images
        for prev_path, (prev_sha, prev_dh) in dhash_map.items():
            if prev_sha == sha:
                continue  # already flagged as exact
            dist = hamming_distance(dh, prev_dh)
            if dist <= phash_threshold:
                near_duplicates.append({
                    "original": prev_path,
                    "near_duplicate": path,
                    "hamming_distance": dist,
                    "threshold": phash_threshold,
                    "dhash_1": prev_dh,
                    "dhash_2": dh
                })

        dhash_map[path] = (sha, dh)

    return {
        "total_scanned": len(image_paths),
        "unique_exact": len(sha_map),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates
    }

