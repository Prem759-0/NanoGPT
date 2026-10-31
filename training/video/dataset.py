"""
training/video/dataset.py
Manifest-driven PyTorch Dataset for Moon Video Training.
Loads video records, extracts sampled frame sequences, and encodes prompts and camera motion.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any

from backend.engines.moon_video.text_encoder import MoonVideoTextEncoder
from backend.engines.moon_image.text_encoder import SimpleTextTokenizer
from .frame_sampling import sample_frames_from_sequence, create_synthetic_motion_clip

class MoonVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        split: str = "train",
        num_frames: int = 16,
        image_size: int = 128,
        max_length: int = 77
    ):
        self.manifest_path = manifest_path
        self.base_dir = base_dir
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.tokenizer = SimpleTextTokenizer(max_length=max_length)
        self.records: List[Dict[str, Any]] = []

        self._load_records()

    def _load_records(self):
        if not os.path.isfile(self.manifest_path):
            return

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
        rel_path = rec.get("media_path", "")
        full_path = os.path.normpath(os.path.join(self.base_dir, rel_path))

        # If pointing to an image, create synthetic camera motion sequence
        motion_type = rec.get("camera_motion", "orbit_left")
        video_tensor = create_synthetic_motion_clip(
            base_image_path=full_path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            motion_type=motion_type
        )

        caption = rec.get("caption", "")
        input_ids = self.tokenizer.encode(caption)

        camera_motions = ("static", "orbit_left", "orbit_right", "dolly_in", "dolly_out", "pan_left", "pan_right", "tilt_up", "tilt_down")
        cam_id = camera_motions.index(motion_type) + 1 if motion_type in camera_motions else 0

        return {
            "id": rec.get("id", f"vid_{idx}"),
            "video_values": video_tensor,
            "input_ids": input_ids,
            "camera_id": torch.tensor(cam_id, dtype=torch.long),
            "caption": caption
        }
