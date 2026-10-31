"""
backend/engines/moon_image/inference.py
Production Inference Service for Moon Image Engine.
Strict Ownership Mode: Never silently falls back to external APIs.
"""
import os
import torch
from typing import Dict, Any, Optional
from PIL import Image

from .config import MoonImageConfig
from .model import MoonImageModel
from .pipeline import MoonImagePipeline

class MoonImageInferenceService:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or os.environ.get("MOON_IMAGE_CHECKPOINT_PATH", "models/checkpoints/moon_image_v0.pt")
        self.pipeline: Optional[MoonImagePipeline] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        self._attempt_load()

    def _attempt_load(self):
        if os.path.isfile(self.checkpoint_path):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = MoonImagePipeline.from_checkpoint(self.checkpoint_path, device=device)
                self.is_loaded = True
                self.load_error = None
            except Exception as e:
                self.is_loaded = False
                self.load_error = f"Failed to load checkpoint: {str(e)}"
        else:
            self.is_loaded = False
            self.load_error = f"No trained checkpoint found at '{self.checkpoint_path}'."

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 25,
        guidance: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates an image using Moon Image. If model is not trained, returns a truthful diagnostic
        instead of pretending or silently falling back to a third-party service.
        """
        if not self.is_loaded or self.pipeline is None:
            return {
                "success": False,
                "status": "MODEL_NOT_TRAINED",
                "model_name": "moon-image-v0",
                "error": self.load_error or "Model is not trained.",
                "message": (
                    "Moon Image (our sovereign latent diffusion model) requires completed training weights before inference. "
                    "Under strict model ownership rules, third-party hosted APIs (Pollinations, Replicate, fal.ai, SDXL) are prohibited. "
                    "Run 'python training/image/train.py' to produce a verified checkpoint."
                ),
                "image": None,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }

        try:
            image = self.pipeline.text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )
            return {
                "success": True,
                "status": "COMPLETED",
                "model_name": "moon-image-v0",
                "error": None,
                "image": image,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }
        except Exception as e:
            return {
                "success": False,
                "status": "INFERENCE_ERROR",
                "model_name": "moon-image-v0",
                "error": str(e),
                "image": None,
                "provenance": "moon_ai_random_init",
                "external_fallback_used": False
            }
