import os
import math
import time
import hashlib
import random
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

class ImageService:
    """
    Local Procedural Image Generation Engine.
    Generates high-tech AI artwork, neural networks, cyberpunk graphics,
    and visual designs locally without external APIs.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_image(self, prompt: str) -> Dict[str, Any]:
        prompt_hash = hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:12]
        filename = f"gen_{prompt_hash}.png"
        filepath = os.path.join(self.output_dir, filename)

        width, height = 768, 512
        img = Image.new("RGB", (width, height), color=(10, 14, 23))
        draw = ImageDraw.Draw(img)

        # Seed based on prompt for consistent creativity
        seed = sum(ord(c) for c in prompt)
        rng = random.Random(seed)

        # 1. Background atmospheric gradient
        primary_hue = rng.choice([(6, 182, 212), (139, 92, 246), (16, 185, 129), (244, 63, 94)])
        secondary_hue = rng.choice([(59, 130, 246), (236, 72, 153), (245, 158, 11)])

        for y in range(height):
            ratio = y / height
            r = int(10 + (primary_hue[0] * 0.15 * (1 - ratio)) + (secondary_hue[0] * 0.1 * ratio))
            g = int(14 + (primary_hue[1] * 0.15 * (1 - ratio)) + (secondary_hue[1] * 0.1 * ratio))
            b = int(23 + (primary_hue[2] * 0.15 * (1 - ratio)) + (secondary_hue[2] * 0.1 * ratio))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # 2. Glowing Neural nodes & interconnects
        nodes = []
        for _ in range(32):
            nx = rng.randint(40, width - 40)
            ny = rng.randint(40, height - 120)
            nodes.append((nx, ny))

        # Connect nearby nodes with glowing lines
        for i, (x1, y1) in enumerate(nodes):
            for x2, y2 in nodes[i+1:]:
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 160:
                    alpha = int(max(20, (1 - dist / 160) * 180))
                    line_color = (
                        min(255, primary_hue[0] + rng.randint(-20, 20)),
                        min(255, primary_hue[1] + rng.randint(-20, 20)),
                        min(255, primary_hue[2] + rng.randint(-20, 20)),
                    )
                    draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)

        # Draw pulsing node circles
        for nx, ny in nodes:
            radius = rng.randint(3, 7)
            glow_color = primary_hue if rng.random() > 0.4 else secondary_hue
            draw.ellipse(
                [(nx - radius, ny - radius), (nx + radius, ny + radius)],
                fill=glow_color,
                outline=(255, 255, 255)
            )

        # 3. Futuristic Grid & Horizon
        grid_y = height - 100
        for gx in range(0, width, 40):
            draw.line([(gx, grid_y), (int(width/2 + (gx - width/2) * 1.8), height)], fill=(30, 41, 59), width=1)
        for gy in range(grid_y, height, 18):
            draw.line([(0, gy), (width, gy)], fill=(30, 41, 59), width=1)

        # 4. Central Geometric Focus (Holographic Core)
        cx, cy = width // 2, height // 2 - 20
        for ring_r in range(70, 10, -15):
            draw.ellipse(
                [(cx - ring_r, cy - ring_r), (cx + ring_r, cy + ring_r)],
                outline=primary_hue,
                width=2
            )

        # 5. Banner Overlay with Prompt metadata
        draw.rectangle([(0, height - 60), (width, height)], fill=(15, 23, 42))
        draw.line([(0, height - 60), (width, height - 60)], fill=primary_hue, width=2)

        title_text = f"GENERATED: {prompt[:55]}"
        draw.text((20, height - 48), title_text, fill=(241, 245, 249))
        draw.text((20, height - 26), f"Engine: NanoGPT Local Neural Visualizer | Seed: {seed}", fill=(148, 163, 184))

        img.save(filepath, format="PNG")
        return {
            "image_url": f"/static/generated/{filename}",
            "filename": filename,
            "prompt": prompt,
            "width": width,
            "height": height
        }
