import os
import math
import time
import hashlib
import random
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

try:
    from backend.engines.moon_image import MoonImageInferenceService
    MOON_IMAGE_ENGINE_AVAILABLE = True
except ImportError:
    MOON_IMAGE_ENGINE_AVAILABLE = False


class ImageService:
    """
    Moon Image Sovereign Synthesis Engine (moon-image-v0).
    Features:
    - Sovereign Latent Diffusion Architecture (MoonUNet2D + MoonTextEncoder + MoonImageAutoencoder)
    - Strict Model Ownership: Zero external third-party generation APIs (no Pollinations, Replicate, fal.ai)
    - Weight Provenance: Strictly initialized from random weights ('moon_ai_random_init')
    - Truthful MODEL_NOT_TRAINED diagnostics when checkpoint is not yet trained
    - Architect: Prem Gaikwad (India 🇮🇳)
    """

    STYLES = {
        "cinematic": {
            "name": "Cinematic Sovereign",
            "primary": (245, 158, 11),
            "secondary": (59, 130, 246),
            "bg": (10, 15, 25),
            "tags": "cinematic lighting, volumetric atmosphere, balanced composition, high definition"
        },
        "cyberpunk": {
            "name": "Cyberpunk Neon",
            "primary": (6, 182, 212),
            "secondary": (236, 72, 153),
            "bg": (8, 10, 20),
            "tags": "neon illumination, cyberpunk aesthetic, high tech, raytracing, sharp focus"
        },
        "photorealistic": {
            "name": "Photorealistic Studio",
            "primary": (226, 232, 240),
            "secondary": (217, 119, 6),
            "bg": (15, 20, 30),
            "tags": "studio softbox lighting, shallow depth of field, balanced exposure, photorealistic textures"
        },
        "anime": {
            "name": "Anime & Digital Art",
            "primary": (168, 85, 247),
            "secondary": (244, 63, 94),
            "bg": (18, 12, 32),
            "tags": "cel shading, vibrant colors, detailed sky, glowing particles, anime concept art"
        },
        "cartography": {
            "name": "Geographic Cartography",
            "primary": (14, 165, 233),
            "secondary": (245, 158, 11),
            "bg": (8, 18, 32),
            "tags": "cartographic map, topography, geographic boundaries, oceanic bathymetry"
        }
    }

    def __init__(self, output_dir: str, checkpoint_path: Optional[str] = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "MOON_IMAGE_CHECKPOINT_PATH", "models/checkpoints/moon_image_v0.pt"
        )
        self.inference_service: Optional[MoonImageInferenceService] = None
        if MOON_IMAGE_ENGINE_AVAILABLE:
            try:
                self.inference_service = MoonImageInferenceService(checkpoint_path=self.checkpoint_path)
            except Exception:
                self.inference_service = None

    def enhance_prompt(self, raw_prompt: str, style_key: str = "cinematic") -> str:
        """Expands raw prompt with sovereign style conditioning tags."""
        style_info = self.STYLES.get(style_key, self.STYLES["cinematic"])
        return f"{raw_prompt}, {style_info['tags']}"

    def generate_image(self, prompt: str, style: str = "auto") -> Dict[str, Any]:
        """
        Executes sovereign image synthesis.
        If a trained checkpoint exists, runs MoonImagePipeline.
        If untrained, returns a truthful diagnostic response with a local diagnostic artifact.
        """
        prompt_hash = hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:12]
        filename = f"gen_{prompt_hash}.png"
        filepath = os.path.join(self.output_dir, filename)

        width, height = 512, 384
        prompt_lower = prompt.lower()

        # Style resolution
        if style in self.STYLES:
            style_key = style
        elif any(k in prompt_lower for k in ["map", "india", "geography", "world"]):
            style_key = "cartography"
        elif any(k in prompt_lower for k in ["cyber", "neon", "robot", "future"]):
            style_key = "cyberpunk"
        elif any(k in prompt_lower for k in ["anime", "cartoon", "illustration"]):
            style_key = "anime"
        elif any(k in prompt_lower for k in ["photo", "real", "portrait"]):
            style_key = "photorealistic"
        else:
            style_key = "cinematic"

        style_cfg = self.STYLES[style_key]
        enhanced_prompt = self.enhance_prompt(prompt, style_key)

        # 1. Attempt trained neural diffusion checkpoint
        if self.inference_service and self.inference_service.is_loaded:
            res = self.inference_service.generate(prompt=prompt)
            if res.get("success") and res.get("image"):
                res["image"].save(filepath, format="PNG")
                return {
                    "image_url": f"/static/generated/{filename}",
                    "filename": filename,
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "style": style_cfg["name"],
                    "status": "READY",
                    "model_name": "moon-image-v0",
                    "width": width,
                    "height": height,
                    "message": "Synthesized with trained sovereign latent diffusion weights.",
                    "provenance": "moon_ai_random_init"
                }

        # 2. Sovereign Untrained Diagnostic Mode
        # Strictly local: 0 third-party calls, 0 Wikipedia scraping
        seed = sum(ord(c) for c in prompt)
        rng = random.Random(seed)

        img = self._render_sovereign_diagnostic_image(prompt, width, height, style_cfg, rng)
        img.save(filepath, format="PNG", quality=95)

        return {
            "image_url": f"/static/generated/{filename}",
            "filename": filename,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "style": style_cfg["name"],
            "status": "MODEL_NOT_TRAINED",
            "model_name": "moon-image-v0",
            "width": width,
            "height": height,
            "message": "Moon Image model initialized (Random Init). Requires training checkpoint to synthesize learned neural features.",
            "checkpoint_required": self.checkpoint_path,
            "training_command": "python training/image/train.py --manifest datasets/image/manifest.jsonl --output_dir checkpoints/moon_image",
            "provenance": "moon_ai_random_init"
        }

    def _render_sovereign_diagnostic_image(
        self, prompt: str, width: int, height: int, style_cfg: dict, rng: random.Random
    ) -> Image.Image:
        """
        Renders a truthful local diagnostic visualization without external dependencies.
        Clearly indicates that this is an architectural preview, not pretending to be trained diffusion.
        """
        img = Image.new("RGB", (width, height), color=style_cfg.get("bg", (10, 15, 25)))
        draw = ImageDraw.Draw(img)

        c1 = style_cfg.get("primary", (245, 158, 11))
        c2 = style_cfg.get("secondary", (59, 130, 246))

        # Geometric neural architecture background
        for i in range(12):
            x1 = rng.randint(0, width)
            y1 = rng.randint(0, height - 70)
            x2 = rng.randint(0, width)
            y2 = rng.randint(0, height - 70)
            draw.line([(x1, y1), (x2, y2)], fill=(c1[0] // 5, c1[1] // 5, c1[2] // 5), width=1)

        # Architectural core frame
        margin = 30
        draw.rectangle(
            [(margin, margin), (width - margin, height - 70)],
            outline=(c2[0] // 2, c2[1] // 2, c2[2] // 2),
            width=2
        )

        # Crosshairs
        cx, cy = width // 2, (height - 70) // 2
        draw.line([(cx - 20, cy), (cx + 20, cy)], fill=c1, width=2)
        draw.line([(cx, cy - 20), (cx, cy + 20)], fill=c1, width=2)

        # Status & watermark
        draw.text((margin + 15, margin + 15), "MOON IMAGE v0 — SOVEREIGN LATENT DIFFUSION", fill=(240, 240, 245))
        draw.text((margin + 15, margin + 35), "Status: MODEL_NOT_TRAINED (Random Initialization Mode)", fill=(245, 158, 11))
        draw.text((margin + 15, margin + 55), "Architecture: MoonUNet2D + Cross-Attention + VAE", fill=(148, 163, 184))
        draw.text((margin + 15, margin + 75), "Provenance: Strictly Self-Owned (0 External Weights)", fill=(148, 163, 184))

        # Prompt display
        p_disp = prompt[:60] + ("..." if len(prompt) > 60 else "")
        draw.text((margin + 15, cy + 30), f"Prompt: \"{p_disp}\"", fill=(226, 232, 240))
        draw.text((margin + 15, cy + 50), f"Style: {style_cfg['name']}", fill=(100, 116, 139))

        # Studio footer banner
        draw.rectangle([(0, height - 60), (width, height)], fill=(8, 12, 20))
        draw.line([(0, height - 60), (width, height - 60)], fill=c1, width=2)
        draw.text((15, height - 50), "🌙 Moon AI Sovereign Creative Studio • Chief Architect: Prem Gaikwad (India 🇮🇳)", fill=(241, 245, 249))
        draw.text((15, height - 30), "Train weights: python training/image/train.py --manifest datasets/image/manifest.jsonl", fill=(148, 163, 184))

        return img
