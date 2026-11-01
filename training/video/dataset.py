"""
training/video/dataset.py
Strict Video Datasets for Moon Video Engine.
Separates real video training data from synthetic development data.
Prohibits silent substitution of synthetic still-image animations in production training.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import json
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Tuple

from backend.engines.moon_image.text_encoder import SimpleTextTokenizer
from datasets.tools.validate_video_sources import VALID_VIDEO_LICENSES, VALID_CAMERA_MOTIONS
from .frame_sampling import decode_real_video_clip, create_synthetic_motion_clip, CorruptVideoError

CAMERA_MOTION_CLASSES = (
    "static",
    "orbit_left",
    "orbit_right",
    "dolly_in",
    "dolly_out",
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "crane_up",
    "crane_down"
)

REAL_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

class MoonRealVideoDataset(Dataset):
    """
    Dataset that loads and decodes ACTUAL video files (.mp4, .webm, etc.).
    Preserves real temporal ordering, source FPS, and frame timestamps.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        base_dir: str,
        num_frames: int = 16,
        frame_size: int = 128,
        max_length: int = 77
    ):
        self.records = records
        self.base_dir = base_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.tokenizer = SimpleTextTokenizer(max_length=max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rel_path = rec.get("video_path") or rec.get("media_path") or ""
        full_path = os.path.normpath(os.path.join(self.base_dir, rel_path))

        try:
            video_tensor, meta = decode_real_video_clip(
                video_path=full_path,
                target_num_frames=self.num_frames,
                target_size=self.frame_size
            )
        except Exception as e:
            raise CorruptVideoError(f"Failed to decode video file '{full_path}': {e}")

        caption = rec.get("caption", "")
        input_ids = self.tokenizer.encode(caption)

        motion_str = rec.get("camera_motion", "static")
        cam_id = (
            CAMERA_MOTION_CLASSES.index(motion_str) + 1
            if motion_str in CAMERA_MOTION_CLASSES
            else 0
        )

        return {
            "id": rec.get("id", f"real_vid_{idx}"),
            "video_values": video_tensor,
            "input_ids": input_ids,
            "camera_id": torch.tensor(cam_id, dtype=torch.long),
            "caption": caption,
            "is_synthetic": False,
            "fps": meta["source_fps"],
            "duration_sec": meta["duration_sec"],
            "timestamps": meta["timestamps_sec"]
        }


class MoonSyntheticVideoDataset(Dataset):
    """
    Development-only dataset that creates synthetic camera movement from still images.
    ISOLATED EXCLUSIVELY FOR UNIT TESTS AND PIPELINE DEBUGGING.
    Every sample is explicitly marked with is_synthetic=True.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        base_dir: str,
        num_frames: int = 16,
        frame_size: int = 128,
        max_length: int = 77
    ):
        self.records = records
        self.base_dir = base_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.tokenizer = SimpleTextTokenizer(max_length=max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rel_path = rec.get("media_path") or rec.get("video_path") or ""
        full_path = os.path.normpath(os.path.join(self.base_dir, rel_path))

        motion_type = rec.get("camera_motion", "orbit_left")
        video_tensor, meta = create_synthetic_motion_clip(
            base_image_path=full_path,
            num_frames=self.num_frames,
            image_size=self.frame_size,
            motion_type=motion_type
        )

        caption = rec.get("caption", "")
        input_ids = self.tokenizer.encode(caption)

        cam_id = (
            CAMERA_MOTION_CLASSES.index(motion_type) + 1
            if motion_type in CAMERA_MOTION_CLASSES
            else 0
        )

        return {
            "id": rec.get("id", f"synth_vid_{idx}"),
            "video_values": video_tensor,
            "input_ids": input_ids,
            "camera_id": torch.tensor(cam_id, dtype=torch.long),
            "caption": caption,
            "is_synthetic": True,
            "fps": 8.0,
            "duration_sec": self.num_frames / 8.0,
            "timestamps": [i / 8.0 for i in range(self.num_frames)]
        }


class MoonVideoDataset(Dataset):
    """
    Unified Video Dataset with strict production integrity validation.
    Enforces that production training ONLY uses genuine real video files.
    """
    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 128,
        max_length: int = 77,
        allow_synthetic: bool = False
    ):
        self.manifest_path = manifest_path
        self.base_dir = base_dir
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.allow_synthetic = allow_synthetic
        self.is_synthetic_dev = False

        self.real_records: List[Dict[str, Any]] = []
        self.synthetic_records: List[Dict[str, Any]] = []

        self._load_and_partition_records()

        # Enforce strict production data integrity
        if not self.allow_synthetic:
            if len(self.real_records) == 0:
                raise ValueError(
                    f"STRICT VIDEO INTEGRITY ERROR: Production video training requires real video files (.mp4, .webm). "
                    f"Manifest '{manifest_path}' for split='{split}' contained 0 real video records. "
                    f"Found {len(self.synthetic_records)} synthetic records. "
                    "Synthetic development data is strictly prohibited in production training."
                )
            self._underlying_dataset = MoonRealVideoDataset(
                records=self.real_records,
                base_dir=self.base_dir,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                max_length=max_length
            )
        else:
            self.is_synthetic_dev = True
            # Dev / unit testing mode
            if len(self.synthetic_records) > 0:
                self._underlying_dataset = MoonSyntheticVideoDataset(
                    records=self.synthetic_records,
                    base_dir=self.base_dir,
                    num_frames=self.num_frames,
                    frame_size=self.frame_size,
                    max_length=max_length
                )
            elif len(self.real_records) > 0:
                self._underlying_dataset = MoonRealVideoDataset(
                    records=self.real_records,
                    base_dir=self.base_dir,
                    num_frames=self.num_frames,
                    frame_size=self.frame_size,
                    max_length=max_length
                )
            else:
                raise ValueError(f"No records found in manifest '{manifest_path}' for split='{split}'.")

    def _load_and_partition_records(self):
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Video manifest not found at '{self.manifest_path}'.")

        norm_path = self.manifest_path.replace("\\", "/").lower()
        if not self.allow_synthetic and ("synthetic" in norm_path or "dev" in norm_path):
            raise ValueError(
                f"STRICT VIDEO INTEGRITY ERROR: Attempted to load synthetic development manifest "
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
                except Exception:
                    continue

                if data.get("split", "train").lower() != self.split.lower():
                    continue

                # Determine if real video or synthetic
                path_val = (data.get("video_path") or data.get("media_path") or "").lower()
                is_synth = (
                    data.get("is_synthetic", False)
                    or data.get("source_type") == "procedural_synthetic_fixture"
                )
                is_real = (
                    any(path_val.endswith(ext) for ext in REAL_VIDEO_EXTENSIONS)
                    and not is_synth
                )

                if is_real:
                    # Provenance verification in production mode
                    if not self.allow_synthetic:
                        creator = str(data.get("creator", "")).strip()
                        source_url = str(data.get("source_url", "")).strip()
                        license_id = str(data.get("license", "")).strip()
                        if not creator or not source_url or not license_id or license_id not in VALID_VIDEO_LICENSES:
                            raise ValueError(
                                f"STRICT PROVENANCE ERROR: Video record '{data.get('id')}' in '{self.manifest_path}' "
                                "missing or invalid provenance metadata."
                            )
                    self.real_records.append(data)
                else:
                    self.synthetic_records.append(data)

    def __len__(self) -> int:
        return len(self._underlying_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._underlying_dataset[idx]
