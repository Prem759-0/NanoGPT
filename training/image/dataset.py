"""
training/image/dataset.py
Manifest-driven PyTorch Dataset for Moon Image Training.
Reads verified JSONL manifests, resizes images, and tokenizes captions.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Any

from backend.engines.moon_image.text_encoder import SimpleTextTokenizer

class MoonImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        split: str = "train",
        image_size: int = 256,
        max_length: int = 77
    ):
        self.manifest_path = manifest_path
        self.base_dir = base_dir
        self.split = split
        self.image_size = image_size
        self.tokenizer = SimpleTextTokenizer(max_length=max_length)
        self.records: List[Dict[str, Any]] = []

        self._load_records()

    def _load_records(self):
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                if data.get("split", "train").lower() == self.split.lower():
                    self.records.append(data)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = os.path.normpath(os.path.join(self.base_dir, rec["media_path"]))
        
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
            "caption": caption
        }
