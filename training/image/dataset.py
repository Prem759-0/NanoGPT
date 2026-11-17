"""
training/image/dataset.py
Strict Manifest-driven PyTorch Dataset for Moon Image Training.
Enforces strict production data integrity and provenance verification.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Any

from backend.engines.moon_image.text_encoder import SimpleTextTokenizer
from datasets.tools.validate_image_sources import VALID_IMAGE_LICENSES

class MoonImageDataset(Dataset):
    """
    Manifest-driven dataset for Moon Image Training.
    In production mode (allow_synthetic=False), strictly rejects:
    - Synthetic records (is_synthetic=True)
    - Development fixtures or manifests containing 'synthetic'/'dev'
    - Missing provenance (creator, source_url, license, sha256)
    - Empty manifests (fails closed with ValueError)
    """
    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        split: str = "train",
        image_size: int = 256,
        max_length: int = 77,
        allow_synthetic: bool = False
    ):
        self.manifest_path = manifest_path
        self.base_dir = base_dir
        self.split = split
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
        self.tokenizer = SimpleTextTokenizer(max_length=max_length)
        self.records: List[Dict[str, Any]] = []
        self.is_synthetic_dev = False

        self._load_and_validate_records()

    def _load_and_validate_records(self):
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        norm_path = self.manifest_path.replace("\\", "/").lower()
        if not self.allow_synthetic and ("synthetic" in norm_path or "dev" in norm_path):
            raise ValueError(
                f"STRICT IMAGE INTEGRITY ERROR: Attempted to load synthetic development manifest "
                f"'{self.manifest_path}' in production mode (allow_synthetic=False). "
                "Synthetic development manifests cannot be used for production training."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    raise ValueError(f"JSON decode error in '{self.manifest_path}' line {line_no}: {e}")

                rec_split = data.get("split", "train").lower()
                if rec_split != self.split.lower():
                    continue

                is_synth = data.get("is_synthetic", False) or data.get("source_type") == "procedural_synthetic_fixture"

                if not self.allow_synthetic:
                    if is_synth:
                        raise ValueError(
                            f"STRICT IMAGE INTEGRITY ERROR: Synthetic record '{data.get('id')}' found in production "
                            f"manifest '{self.manifest_path}' line {line_no}. "
                            "Synthetic procedural fixtures are strictly prohibited in production training."
                        )
                    # Verify provenance fields
                    creator = str(data.get("creator", "")).strip()
                    source_url = str(data.get("source_url", "")).strip()
                    license_id = str(data.get("license", "")).strip()
                    sha = str(data.get("sha256", "")).strip()

                    if not creator:
                        raise ValueError(f"STRICT PROVENANCE ERROR: Record '{data.get('id')}' missing 'creator'.")
                    if not source_url:
                        raise ValueError(f"STRICT PROVENANCE ERROR: Record '{data.get('id')}' missing 'source_url'.")
                    if not license_id or license_id not in VALID_IMAGE_LICENSES:
                        raise ValueError(f"STRICT PROVENANCE ERROR: Record '{data.get('id')}' invalid license '{license_id}'.")
                    if not sha or len(sha) != 64:
                        raise ValueError(f"STRICT PROVENANCE ERROR: Record '{data.get('id')}' missing valid SHA-256 checksum.")

                    # Verify media file exists
                    rel_p = data.get("media_path") or data.get("image_path")
                    if not rel_p:
                        raise ValueError(f"STRICT PROVENANCE ERROR: Record '{data.get('id')}' missing media_path.")
                    full_p = os.path.normpath(os.path.join(self.base_dir, rel_p))
                    if not os.path.isfile(full_p):
                        raise FileNotFoundError(f"STRICT IMAGE INTEGRITY ERROR: Image file not found: {full_p}")

                self.records.append(data)

        if not self.allow_synthetic:
            if len(self.records) == 0:
                raise ValueError(
                    f"STRICT IMAGE INTEGRITY ERROR: Production manifest '{self.manifest_path}' contains "
                    f"0 genuine records for split='{self.split}'. "
                    "Production training cannot proceed without verified genuine media."
                )
        else:
            self.is_synthetic_dev = True
            if len(self.records) == 0:
                raise ValueError(
                    f"No records found in dev manifest '{self.manifest_path}' for split='{self.split}'."
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rel_path = rec.get("media_path") or rec.get("image_path")
        img_path = os.path.normpath(os.path.join(self.base_dir, rel_path))

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            arr = np.array(im).astype(np.float32) / 127.5 - 1.0  # Normalized to [-1, 1]
            pixel_values = torch.tensor(arr).permute(2, 0, 1)    # [3, H, W]

        caption = rec.get("caption", "")
        input_ids = self.tokenizer.encode(caption)

        return {
            "id": rec.get("id", f"sample_{idx}"),
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": caption,
            "is_synthetic": rec.get("is_synthetic", False)
        }
