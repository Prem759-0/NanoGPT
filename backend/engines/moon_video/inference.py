"""
backend/engines/moon_video/inference.py
Production Inference Service for Moon Video Engine.
Strict Ownership Mode: Never silently falls back to external APIs or procedural canvas fakes.
"""
import os
import torch
from typing import Dict, Any, Optional, List
from PIL import Image

from .config import MoonVideoConfig
from .model import MoonVideoModel
from .pipeline import MoonVideoPipeline

class MoonVideoInferenceService:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or os.environ.get("MOON_VIDEO_CHECKPOINT_PATH", "models/checkpoints/moon_video_v0.pt")
        self.pipeline: Optional[MoonVideoPipeline] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        self._attempt_load()

    def _attempt_load(self):
        if os.path.isfile(self.checkpoint_path):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = MoonVideoPipeline.from_checkpoint(self.checkpoint_path, device=device)
                self.is_loaded = True
                self.load_error = None
            except Exception as e:
                self.is_loaded = False
                self.load_error = f"Failed to load checkpoint: {str(e)}"
        else:
            self.is_loaded = False
            self.load_error = f"No trained checkpoint found at '{self.checkpoint_path}'."

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        camera_motion: str = "static",
        num_frames: Optional[int] = None,
        steps: int = 25,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        export_gif_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes video frames using Moon Video. If model is not trained, returns a truthful diagnostic
        instead of pretending or falling back to third-party hosted video services.
        """
        if not self.is_loaded or self.pipeline is None:
            return {
                "success": False,
                "status": "MODEL_NOT_TRAINED",
                "model_name": "moon-video-v0",
                "error": self.load_error or "Model is not trained.",
                "message": (
                    "Moon Video (our sovereign spatiotemporal diffusion model) requires completed training weights before inference. "
                    "Under strict model ownership rules, third-party hosted APIs (ModelScope, AnimateDiff, Runway, Kling, CogVideo) are prohibited. "
                    "Run 'python training/video/train.py' to produce a verified checkpoint."
                ),
                "frames": [],
                "output_path": None,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }

        try:
            frames = self.pipeline.text_to_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                camera_motion=camera_motion,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )
            out_file = None
            if export_gif_path:
                self.pipeline.export_to_gif(frames, export_gif_path, fps=self.pipeline.config.fps)
                out_file = export_gif_path

            return {
                "success": True,
                "status": "COMPLETED",
                "model_name": "moon-video-v0",
                "error": None,
                "frames": frames,
                "num_frames": len(frames),
                "output_path": out_file,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }
        except Exception as e:
            return {
                "success": False,
                "status": "INFERENCE_ERROR",
                "model_name": "moon-video-v0",
                "error": str(e),
                "frames": [],
                "output_path": None,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }
