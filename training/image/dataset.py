"""
training/image/dataset.py
Strict Manifest-driven PyTorch Dataset for Moon Image Training.
Enforces strict production data integrity, owner declarations, and cryptographic file verification.
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
from datasets.tools.provenance_validator import (
    validate_provenance_record,
    validate_safe_path,
    verify_file_checksum,
    ALLOWED_IMAGE_EXTENSIONS
)

class MoonImageDataset(Dataset):
    """
    Manifest-driven dataset for Moon Image Training.
    In production mode (allow_synthetic=False), strictly enforces:
    - Explicit is_synthetic=false (never defaulted)
    - Full provenance and verifiable license / owner declaration
    - Actual file byte SHA-256 match
    - Path safety (rejection of directory traversal)
    - Non-empty manifest (fails closed if 0 eligible records)
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

                if not self.allow_synthetic:
                    audit = validate_provenance_record(
                        record=data,
                        modality="image",
                        base_dir=self.base_dir,
                        allow_synthetic=False,
                        verify_file=True
                    )
                    if not audit["training_eligible"]:
                        raise ValueError(
                            f"STRICT PROVENANCE ERROR / STRICT IMAGE INTEGRITY ERROR: Record '{data.get('id')}' in '{self.manifest_path}' "
                            f"line {line_no} failed production verification ({audit['status']}): {'; '.join(audit['errors'])}"
                        )

                self.records.append(data)

        if not self.allow_synthetic:
            if len(self.records) == 0:
                raise ValueError(
                    f"STRICT PROVENANCE ERROR / STRICT IMAGE INTEGRITY ERROR: Production manifest '{self.manifest_path}' contains "
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
        safe_ok, full_path, path_err = validate_safe_path(
            rel_path,
            base_dir=self.base_dir,
            allowed_extensions=ALLOWED_IMAGE_EXTENSIONS
        )
        if not safe_ok:
            raise ValueError(f"Security error opening image: {path_err}")

        with Image.open(full_path) as im:
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
