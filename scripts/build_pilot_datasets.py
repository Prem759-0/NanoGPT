"""
scripts/build_pilot_datasets.py
Constructs, Normalizes, and Manifests the Stage 2.0 Balanced Pilot Dataset Foundation.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
"""
import os
import sys
import json
import math
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.collect_images import ImageCollector
from datasets.tools.collect_videos import VideoCollector
from datasets.tools.deduplicate_images import compute_sha256

MANIFESTS_DIR = os.path.join(PROJECT_ROOT, "datasets", "manifests")
os.makedirs(MANIFESTS_DIR, exist_ok=True)

def create_distinct_pilot_image(category: str, variant: int, width: int = 256, height: int = 256) -> Tuple[Image.Image, str, str]:
    """
    Creates perceptually distinct images for each category and variant.
    """
    img = Image.new("RGB", (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    creator = "Prem Gaikwad / Moon AI Project"

    if category == "landscapes":
        if variant == 0:
            # Alpine sunset
            for y in range(height // 2):
                draw.line([(0, y), (width, y)], fill=(int(255 - y*0.8), int(120 + y*0.4), int(50 + y*1.2)))
            draw.ellipse([width//2 - 25, height//3 - 25, width//2 + 25, height//3 + 25], fill=(255, 230, 150))
            pts = [(0, height), (0, 140), (80, 90), (140, 150), (200, 80), (256, 160), (256, 256)]
            draw.polygon(pts, fill=(50, 40, 70))
            caption = "Alpine mountain peaks bathed in golden hour twilight with violet haze"
        elif variant == 1:
            # Tropical beach coast
            for y in range(height):
                if y < 130:
                    draw.line([(0, y), (width, y)], fill=(80, 180, 240))
                elif y < 190:
                    draw.line([(0, y), (width, y)], fill=(20, 140, 190)) # ocean
                else:
                    draw.line([(0, y), (width, y)], fill=(230, 210, 160)) # sand
            draw.ellipse([40, 30, 90, 80], fill=(255, 250, 200)) # sun
            caption = "Vibrant tropical coastline with turquoise ocean waves and golden sand"
        else:
            # Autumn pine forest misty path
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(120 - y//4, 130 - y//4, 140 - y//4))
            for tr in range(6):
                tx = 25 + tr * 40
                draw.polygon([(tx, 60), (tx - 25, 200), (tx + 25, 200)], fill=(180, 80 + tr*15, 30))
            caption = "Misty autumn pine forest with golden foliage and winding woodland path"

    elif category == "architecture":
        if variant == 0:
            # Modern high-rise towers
            draw.rectangle([0, 0, width, height], fill=(15, 20, 35))
            draw.rectangle([30, 50, 100, height], fill=(50, 65, 95))
            draw.rectangle([120, 80, 220, height], fill=(40, 50, 75))
            for y in range(60, height - 20, 16):
                draw.rectangle([45, y, 85, y + 8], fill=(240, 220, 130))
                draw.rectangle([135, y + 20, 205, y + 28], fill=(220, 240, 255))
            caption = "Glass skyscraper facades illuminated by ambient city night glow"
        elif variant == 1:
            # Classical neoclassical colonnade arch
            draw.rectangle([0, 0, width, height], fill=(230, 225, 215))
            draw.rectangle([20, 40, width - 20, 65], fill=(140, 130, 120))
            for col in range(4):
                cx = 40 + col * 55
                draw.rectangle([cx, 65, cx + 22, height - 30], fill=(180, 175, 165))
            draw.rectangle([10, height - 30, width - 10, height], fill=(120, 115, 105))
            caption = "Ancient neoclassical marble temple colonnade with majestic stone columns"
        else:
            # Futuristic organic parametric pavilion
            draw.rectangle([0, 0, width, height], fill=(30, 35, 45))
            for i in range(7):
                r = 30 + i * 20
                draw.arc([width//2 - r, height//2 - r + 30, width//2 + r, height//2 + r + 30], 180, 360, fill=(240, 240, 250), width=6)
            caption = "Parametric architectural pavilion with fluid white vaulted arches"

    elif category == "products":
        if variant == 0:
            # Luxury chronograph watch
            draw.rectangle([0, 0, width, height], fill=(245, 245, 248))
            cx, cy = width//2, height//2
            draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=(25, 30, 40), outline=(210, 175, 70), width=6)
            draw.line([(cx, cy), (cx + 30, cy - 20)], fill=(255, 255, 255), width=3)
            caption = "Luxury chronograph watch with gold bezel and dark ceramic dial face"
        elif variant == 1:
            # Glass perfume flacon
            draw.rectangle([0, 0, width, height], fill=(20, 25, 35))
            cx, cy = width//2, height//2 + 20
            draw.rectangle([cx - 45, cy - 60, cx + 45, cy + 60], fill=(230, 190, 120), outline=(255, 255, 255), width=3)
            draw.rectangle([cx - 18, cy - 90, cx + 18, cy - 60], fill=(200, 160, 60)) # gold cap
            caption = "Elegantly sculpted amber perfume bottle with polished metallic golden stopper"
        else:
            # Sleek wireless over-ear headphones
            draw.rectangle([0, 0, width, height], fill=(240, 240, 242))
            cx, cy = width//2, height//2
            draw.arc([cx - 70, cy - 90, cx + 70, cy + 30], 180, 360, fill=(40, 40, 45), width=12) # band
            draw.ellipse([cx - 75, cy - 10, cx - 40, cy + 60], fill=(30, 30, 35)) # ear cups
            draw.ellipse([cx + 40, cy - 10, cx + 75, cy + 60], fill=(30, 30, 35))
            caption = "Minimalist matte black over-ear wireless headphones with memory foam cushions"

    elif category == "technology":
        if variant == 0:
            # Silicon chip die
            draw.rectangle([0, 0, width, height], fill=(10, 20, 15))
            cx, cy = width//2, height//2
            draw.rectangle([cx - 45, cy - 45, cx + 45, cy + 45], fill=(30, 35, 40), outline=(200, 160, 50), width=4)
            for i in range(8):
                draw.line([(cx - 45, cy - 35 + i*10), (20, cy - 35 + i*10)], fill=(40, 220, 160), width=2)
                draw.line([(cx + 45, cy - 35 + i*10), (width - 20, cy - 35 + i*10)], fill=(40, 220, 160), width=2)
            caption = "High-performance silicon microprocessor die with intricate bus interconnects"
        elif variant == 1:
            # Fiber optic illumination
            draw.rectangle([0, 0, width, height], fill=(5, 5, 12))
            for i in range(12):
                bx = int(20 + i * 19)
                draw.arc([bx - 40, 30, bx + 80, height + 40], 190, 330, fill=(0, 200, 255), width=3)
                draw.ellipse([bx - 3, 35, bx + 5, 43], fill=(255, 255, 255))
            caption = "Glowing cyan fiber optic cable filaments transmitting optical data pulses"
        else:
            # Server rack datacenter corridor
            draw.rectangle([0, 0, width, height], fill=(12, 14, 20))
            pts_left = [(0, 0), (70, 40), (70, height - 40), (0, height)]
            draw.polygon(pts_left, fill=(25, 30, 45))
            pts_right = [(width, 0), (width - 70, 40), (width - 70, height - 40), (width, height)]
            draw.polygon(pts_right, fill=(25, 30, 45))
            for y in range(50, height - 50, 14):
                draw.ellipse([75, y, 80, y + 4], fill=(80, 255, 120)) # server LEDs
                draw.ellipse([width - 80, y, width - 75, y + 4], fill=(80, 180, 255))
            caption = "Cloud computing enterprise datacenter corridor lined with illuminated server blades"

    elif category == "people":
        if variant == 0:
            # Silhouette portrait with warm rim lighting
            draw.rectangle([0, 0, width, height], fill=(15, 15, 20))
            cx, cy = width//2, height//2 + 15
            draw.ellipse([cx - 30, cy - 40, cx + 30, cy + 30], fill=(200, 150, 120))
            draw.ellipse([cx - 35, cy - 50, cx + 35, cy - 5], fill=(30, 20, 15))
            draw.ellipse([cx - 65, cy + 35, cx + 65, cy + 120], fill=(35, 45, 65))
            draw.arc([cx - 32, cy - 42, cx + 32, cy + 32], 180, 300, fill=(255, 180, 80), width=4)
            caption = "Studio headshot silhouette featuring warm directional rim light on dark slate background"
        elif variant == 1:
            # Artisan hands sculpting wet clay
            draw.rectangle([0, 0, width, height], fill=(220, 210, 195))
            cx, cy = width//2, height//2 + 20
            draw.ellipse([cx - 40, cy - 50, cx + 40, cy + 50], fill=(160, 100, 60)) # clay pot
            draw.ellipse([cx - 60, cy - 20, cx - 20, cy + 20], fill=(210, 160, 130)) # left hand
            draw.ellipse([cx + 20, cy - 20, cx + 60, cy + 20], fill=(210, 160, 130)) # right hand
            caption = "Close-up artisan hands shaping wet terracotta pottery clay on revolving wheel"
        else:
            # Mountaineer looking out over clouds
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(200 - y//3, 220 - y//4, 250 - y//5))
            draw.ellipse([width//3, height//2, width - 20, height + 80], fill=(40, 45, 60)) # peak
            draw.rectangle([80, height//2 - 40, 105, height//2], fill=(220, 60, 40)) # hiker jacket
            draw.ellipse([85, height//2 - 55, 100, height//2 - 40], fill=(200, 150, 120)) # head
            caption = "Adventurous mountaineer standing upon alpine peak overlooking morning sea of clouds"

    elif category == "animals":
        if variant == 0:
            # Majestic eagle in flight
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(120 - y//4, 180 - y//4, 240 - y//5))
            cx, cy = width//2, height//2 - 15
            pts = [(cx, cy), (cx - 75, cy - 35), (cx - 40, cy), (cx, cy - 8), (cx + 40, cy), (cx + 75, cy - 35), (cx, cy)]
            draw.polygon(pts, fill=(40, 30, 25))
            caption = "Soaring golden eagle banking gracefully across boundless open blue sky"
        elif variant == 1:
            # Monarch butterfly on blooming flower
            draw.rectangle([0, 0, width, height], fill=(240, 245, 230))
            cx, cy = width//2, height//2
            draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(240, 200, 40)) # flower core
            draw.ellipse([cx - 45, cy - 50, cx - 5, cy], fill=(240, 120, 20), outline=(20, 20, 20), width=2) # wing
            draw.ellipse([cx + 5, cy - 50, cx + 45, cy], fill=(240, 120, 20), outline=(20, 20, 20), width=2)
            caption = "Vibrant monarch butterfly resting upon yellow blossoming wildflower petals"
        else:
            # Playful swimming dolphin
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(10, 80 + y//3, 140 + y//3))
            cx, cy = width//2, height//2
            pts = [(cx - 70, cy + 10), (cx - 20, cy - 25), (cx + 40, cy - 15), (cx + 70, cy), (cx + 45, cy + 10), (cx - 20, cy + 5)]
            draw.polygon(pts, fill=(90, 120, 145))
            draw.polygon([(cx - 5, cy - 25), (cx + 10, cy - 45), (cx + 15, cy - 20)], fill=(90, 120, 145)) # fin
            caption = "Bottlenose dolphin leaping effortlessly through deep azure ocean water"

    elif category == "vehicles":
        if variant == 0:
            # Aerodynamic supercar profile
            draw.rectangle([0, 0, width, height], fill=(25, 25, 30))
            cx, cy = width//2, height//2 + 15
            car_pts = [(cx - 85, cy + 15), (cx - 70, cy - 5), (cx - 30, cy - 22), (cx + 25, cy - 22), (cx + 65, cy - 5), (cx + 85, cy + 15)]
            draw.polygon(car_pts, fill=(225, 25, 35))
            draw.ellipse([cx - 60, cy + 10, cx - 35, cy + 35], fill=(15, 15, 20), outline=(180, 180, 190), width=3)
            draw.ellipse([cx + 35, cy + 10, cx + 60, cy + 35], fill=(15, 15, 20), outline=(180, 180, 190), width=3)
            caption = "Crimson red high-performance hypercar profile parked on dark asphalt circuit"
        elif variant == 1:
            # High-speed bullet train crossing bridge
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(180 - y//5, 210 - y//5, 240 - y//5))
            draw.rectangle([0, height//2 + 30, width, height//2 + 45], fill=(80, 85, 95)) # bridge
            draw.rectangle([30, height//2 + 5, 220, height//2 + 30], fill=(245, 245, 250), outline=(30, 80, 180), width=3)
            caption = "Aerodynamic modern electric bullet train gliding across elevated concrete railway viaduct"
        else:
            # Classic wooden sailing yacht on ocean
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(40, 110 + y//3, 170 + y//3) if y > height//2 else (180, 220, 245))
            cx, cy = width//2, height//2 + 10
            draw.polygon([(cx - 50, cy), (cx + 50, cy), (cx + 35, cy + 25), (cx - 35, cy + 25)], fill=(120, 70, 40)) # hull
            draw.polygon([(cx, cy - 70), (cx + 40, cy - 5), (cx, cy - 5)], fill=(250, 250, 252)) # sail
            caption = "Polished mahogany wooden sailing yacht cutting through sunlit ocean waves"

    elif category == "food":
        if variant == 0:
            # Artisanal sourdough bread loaf
            draw.rectangle([0, 0, width, height], fill=(240, 235, 225))
            cx, cy = width//2, height//2
            draw.ellipse([cx - 70, cy - 35, cx + 70, cy + 35], fill=(160, 95, 45), outline=(90, 50, 25), width=3)
            draw.arc([cx - 50, cy - 25, cx + 50, cy + 15], 200, 340, fill=(240, 210, 160), width=4) # scoring
            caption = "Crusty golden artisanal sourdough boule loaf resting on rustic wooden baker's table"
        elif variant == 1:
            # Fresh ceramic bowl of steaming ramen
            draw.rectangle([0, 0, width, height], fill=(30, 25, 25))
            cx, cy = width//2, height//2 + 10
            draw.ellipse([cx - 65, cy - 40, cx + 65, cy + 40], fill=(210, 205, 195))
            draw.ellipse([cx - 55, cy - 30, cx + 55, cy + 30], fill=(225, 140, 50))
            draw.ellipse([cx - 30, cy - 15, cx - 10, cy + 5], fill=(255, 245, 200))
            draw.ellipse([cx - 23, cy - 10, cx - 17, cy - 2], fill=(240, 120, 10))
            caption = "Steaming bowl of traditional tonkotsu ramen with soft boiled ajitsuke tamago"
        else:
            # Fresh colorful seasonal fruit composition
            draw.rectangle([0, 0, width, height], fill=(245, 245, 240))
            cx, cy = width//2, height//2
            draw.ellipse([cx - 50, cy - 30, cx + 10, cy + 30], fill=(220, 40, 40)) # apple
            draw.ellipse([cx, cy - 20, cx + 55, cy + 35], fill=(255, 160, 20)) # orange
            draw.ellipse([cx - 20, cy + 5, cx + 20, cy + 45], fill=(140, 190, 40)) # lime
            caption = "Vibrant still life composition of ripe red apple, juicy orange, and fresh lime"

    elif category == "illustration_design":
        if variant == 0:
            # Bauhaus geometric constructivism
            draw.rectangle([0, 0, width, height], fill=(245, 240, 230))
            draw.ellipse([25, 25, 115, 115], fill=(220, 50, 40))
            draw.rectangle([90, 80, 190, 180], fill=(30, 80, 160))
            draw.polygon([(160, 225), (220, 115), (240, 225)], fill=(235, 190, 40))
            caption = "Bauhaus constructivist graphic layout pairing vibrant primary geometry with asymmetry"
        elif variant == 1:
            # Retro cyberpunk 80s synthwave sunset grid
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(int(20 + y*0.5), int(10), int(40 + y*0.6)))
            draw.ellipse([width//2 - 40, 60, width//2 + 40, 140], fill=(255, 80, 140))
            for y in range(140, height, 18):
                draw.line([(0, y), (width, y)], fill=(0, 240, 255), width=2)
            caption = "Retro synthwave neon landscape featuring glowing magenta horizon and cyan perspective grid"
        else:
            # Japanese ukiyo-e wave wave art
            for y in range(height):
                draw.line([(0, y), (width, y)], fill=(240, 235, 220))
            pts = [(0, height), (0, 160), (60, 120), (120, 190), (180, 90), (256, 170), (256, 256)]
            draw.polygon(pts, fill=(25, 65, 120))
            draw.arc([140, 50, 220, 130], 160, 340, fill=(255, 255, 255), width=8)
            caption = "Japanese ukiyo-e woodblock style ocean wave cresting with decorative white seafoam"

    else: # abstract_scientific
        if variant == 0:
            # Quantum orbital electron density cloud
            draw.rectangle([0, 0, width, height], fill=(5, 8, 18))
            cx, cy = width//2, height//2
            for r in range(15, 85, 14):
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 140 + r, 255), width=2)
            caption = "Quantum atomic orbital probability distribution model radiating electromagnetic contours"
        elif variant == 1:
            # Double helix DNA strand diagram
            draw.rectangle([0, 0, width, height], fill=(10, 18, 25))
            for i in range(10):
                y = 30 + i * 20
                x1 = int(width//2 + 40 * math.sin(i * 0.8))
                x2 = int(width//2 - 40 * math.sin(i * 0.8))
                draw.line([(x1, y), (x2, y)], fill=(80, 220, 180), width=3)
                draw.ellipse([x1 - 4, y - 4, x1 + 4, y + 4], fill=(255, 100, 140))
                draw.ellipse([x2 - 4, y - 4, x2 + 4, y + 4], fill=(100, 160, 255))
            caption = "Scientific molecular double helix DNA ladder model showing base pair nucleotide bridges"
        else:
            # Astrophysical spiral galaxy cluster
            draw.rectangle([0, 0, width, height], fill=(3, 3, 8))
            cx, cy = width//2, height//2
            draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(255, 245, 220)) # galactic core
            for arm in range(2):
                phase = arm * math.pi
                for step in range(30):
                    theta = step * 0.2 + phase
                    rad = 12 + step * 2.5
                    gx = cx + int(rad * math.cos(theta))
                    gy = cy + int(rad * math.sin(theta))
                    draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=(140, 180, 255))
            caption = "Astrophysical spiral galaxy system with radiant stellar nucleus and swirling star arms"

    return img, caption, creator

def create_distinct_pilot_video_clip(
    output_path: str,
    category: str,
    camera_motion: str,
    num_frames: int = 16,
    fps: float = 8.0,
    size: int = 128
) -> Tuple[str, str]:
    import imageio
    frames = []
    w, h = size, size

    for t in range(num_frames):
        im = Image.new("RGB", (w, h), color=(15, 15, 25))
        draw = ImageDraw.Draw(im)
        p = t / float(num_frames - 1)

        if camera_motion == "orbit_left":
            angle = p * math.pi
            cx = w//2 + int(32 * math.cos(angle))
            cy = h//2 + int(12 * math.sin(angle))
            draw.ellipse([w//2 - 15, h//2 - 15, w//2 + 15, h//2 + 15], fill=(60, 70, 90))
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(240, 80, 120), outline=(255, 255, 255), width=2)
            motion_desc = "Camera continuously orbits counter-clockwise around central subject"
        elif camera_motion == "orbit_right":
            angle = -p * math.pi
            cx = w//2 + int(32 * math.cos(angle))
            cy = h//2 + int(12 * math.sin(angle))
            draw.ellipse([w//2 - 15, h//2 - 15, w//2 + 15, h//2 + 15], fill=(60, 70, 90))
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(80, 200, 240), outline=(255, 255, 255), width=2)
            motion_desc = "Camera continuously orbits clockwise around central subject"
        elif camera_motion == "dolly_in":
            scale = 0.5 + p * 0.9
            rad = int(24 * scale)
            draw.ellipse([w//2 - rad, h//2 - rad, w//2 + rad, h//2 + rad], fill=(220, 180, 50), outline=(255, 255, 255), width=2)
            motion_desc = "Camera smoothly dollies forward toward the central object"
        elif camera_motion == "pan_right":
            offset_x = int(-40 + p * 80)
            draw.rectangle([offset_x + 30, 30, offset_x + 70, h - 30], fill=(70, 160, 120))
            motion_desc = "Camera pans smoothly across the horizontal horizon to the right"
        elif camera_motion == "pan_left":
            offset_x = int(40 - p * 80)
            draw.rectangle([offset_x + 30, 30, offset_x + 70, h - 30], fill=(160, 90, 180))
            motion_desc = "Camera pans smoothly across the horizontal horizon to the left"
        elif camera_motion == "tilt_up":
            offset_y = int(35 - p * 70)
            draw.polygon([(w//2, offset_y + 20), (w//2 - 35, offset_y + 80), (w//2 + 35, offset_y + 80)], fill=(230, 110, 40))
            motion_desc = "Camera tilts upwards revealing the summit"
        elif camera_motion == "zoom_in":
            scale = 0.6 + p * 0.8
            rad = int(20 * scale)
            draw.ellipse([w//2 - rad, h//2 - rad, w//2 + rad, h//2 + rad], fill=(90, 130, 250), outline=(255, 255, 255), width=2)
            motion_desc = "Lens zooms in continuously focusing on the focal sphere"
        else: # static
            draw.ellipse([w//2 - 20, h//2 - 20, w//2 + 20, h//2 + 20], fill=(200, 200, 210))
            motion_desc = "Camera remains static on stationary subject"

        frames.append(np.array(im))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    caption = f"Cinematic video clip in {category} domain demonstrating smooth {camera_motion} movement"
    return caption, motion_desc

def build_all():
    print("=" * 70)
    print("   BUILDING MOON AI STAGE 2.0 BALANCED PILOT DATASETS")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("=" * 70)

    collector_img = ImageCollector(
        base_dir=PROJECT_ROOT,
        raw_dir="data/raw/images",
        processed_dir="data/images",
        target_resolution=256
    )

    collector_vid = VideoCollector(
        base_dir=PROJECT_ROOT,
        raw_dir="data/raw/videos",
        processed_dir="data/videos"
    )

    categories = [
        "landscapes", "architecture", "products", "technology", "people",
        "animals", "vehicles", "food", "illustration_design", "abstract_scientific"
    ]

    tmp_staging = os.path.join(PROJECT_ROOT, "data", "staging")
    os.makedirs(tmp_staging, exist_ok=True)

    # 1. Ingest Pilot Images
    image_records = []
    print("\n[1/4] Ingesting balanced category pilot images...")
    for cat_idx, cat in enumerate(categories):
        for variant in range(3):
            img, caption, creator = create_distinct_pilot_image(cat, variant=variant, width=256, height=256)
            staging_path = os.path.join(tmp_staging, f"staging_{cat}_{variant}.png")
            img.save(staging_path)

            split = "train" if variant < 2 else ("val" if cat_idx % 2 == 0 else "test")
            source_url = f"https://github.com/Prem759-0/NanoGPT/tree/main/data/images/{cat}"

            rec = collector_img.ingest_image(
                src_file_path=staging_path,
                caption=caption,
                category=cat,
                creator=creator,
                source_url=source_url,
                license_id="CustomSovereign",
                attribution=f"Original media authored by {creator} for Moon AI Project (CustomSovereign)",
                split=split,
                phash_threshold=2 # sensitive near-duplicate threshold
            )
            image_records.append(rec)
            print(f"  • Ingested image [{cat} v{variant}]: {rec['id']} (Split: {split})")

    # 2. Ingest Pilot Video Clips
    video_records = []
    camera_motions = ["orbit_left", "orbit_right", "dolly_in", "pan_right", "pan_left", "tilt_up", "zoom_in", "static"]
    vid_categories = ["vehicles", "landscapes", "technology", "architecture", "abstract_scientific"]

    print("\n[2/4] Ingesting genuine video clips with temporal motions...")
    for vid_idx, motion in enumerate(camera_motions):
        cat = vid_categories[vid_idx % len(vid_categories)]
        staging_vid = os.path.join(tmp_staging, f"staging_vid_{motion}_{vid_idx}.mp4")
        caption, motion_desc = create_distinct_pilot_video_clip(
            output_path=staging_vid,
            category=cat,
            camera_motion=motion,
            num_frames=16,
            fps=8.0,
            size=128
        )

        split = "train" if vid_idx < 5 else ("val" if vid_idx < 7 else "test")
        v_rec = collector_vid.ingest_video(
            src_video_path=staging_vid,
            caption=caption,
            motion_description=motion_desc,
            category=cat,
            creator="Prem Gaikwad / Moon AI Project",
            source_url=f"https://github.com/Prem759-0/NanoGPT/tree/main/data/videos/{motion}",
            license_id="CustomSovereign",
            camera_motion=motion,
            attribution="Original spatiotemporal video clip authored by Prem Gaikwad (CustomSovereign)",
            split=split,
            min_duration_sec=1.0,
            min_frames=8
        )
        video_records.append(v_rec)
        print(f"  • Ingested video [{motion}]: {v_rec['id']} (Split: {split})")

    import shutil
    shutil.rmtree(tmp_staging, ignore_errors=True)

    # 3. Write Separated Versioned Manifests
    print("\n[3/4] Writing separated train, val, and test manifests to datasets/manifests/...")
    for split_name in ["train", "val", "test"]:
        img_split_records = [r for r in image_records if r["split"] == split_name]
        img_out = os.path.join(MANIFESTS_DIR, f"image_{split_name}.jsonl")
        with open(img_out, "w", encoding="utf-8") as f:
            for r in img_split_records:
                f.write(json.dumps(r) + "\n")
        print(f"  • {os.path.relpath(img_out, PROJECT_ROOT)}: {len(img_split_records)} records")

        vid_split_records = [r for r in video_records if r["split"] == split_name]
        vid_out = os.path.join(MANIFESTS_DIR, f"video_{split_name}.jsonl")
        with open(vid_out, "w", encoding="utf-8") as f:
            for r in vid_split_records:
                f.write(json.dumps(r) + "\n")
        print(f"  • {os.path.relpath(vid_out, PROJECT_ROOT)}: {len(vid_split_records)} records")

    # 4. Chat and Code Datasets
    print("\n[4/4] Writing sovereign Chat and Code corpora...")
    chat_train_records = [
        {"id": "chat_01", "prompt": "Who created you?", "response": "I am Moon AI, a sovereign intelligence system engineered from first principles by Chief ML Systems Architect Prem Gaikwad in India 🇮🇳.", "category": "identity", "license": "CustomSovereign", "split": "train"},
        {"id": "chat_02", "prompt": "What are your core architectural components?", "response": "Moon AI consists of four sovereign specialist engines: Moon Image (Latent Diffusion), Moon Video (Spatiotemporal UNet3D), Moon Chat (Causal Autoregressive Transformer), and Moon Code (Fill-in-the-Middle Coding Transformer), unified via Moon Core.", "category": "architecture", "license": "CustomSovereign", "split": "train"},
        {"id": "chat_03", "prompt": "Explain photosynthesis in simple terms.", "response": "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into chemical energy (glucose) while releasing oxygen as a byproduct.", "category": "science", "license": "CustomSovereign", "split": "train"}
    ]
    chat_val_records = [
        {"id": "chat_val_01", "prompt": "What is the speed of light in a vacuum?", "response": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 kilometers per second).", "category": "physics", "license": "CustomSovereign", "split": "val"}
    ]

    with open(os.path.join(MANIFESTS_DIR, "chat_train.jsonl"), "w", encoding="utf-8") as f:
        for r in chat_train_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(MANIFESTS_DIR, "chat_val.jsonl"), "w", encoding="utf-8") as f:
        for r in chat_val_records:
            f.write(json.dumps(r) + "\n")

    code_train_records = [
        {"id": "code_01", "language": "python", "prompt": "Write a Python function to compute the Fibonacci sequence up to n terms.", "code": "def fibonacci(n: int):\n    if n <= 0:\n        return []\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]", "test_case": "assert fibonacci(5) == [0, 1, 1, 2, 3]", "license": "MIT", "split": "train"},
        {"id": "code_02", "language": "typescript", "prompt": "Write a TypeScript function to check if a string is a palindrome.", "code": "function isPalindrome(str: string): boolean {\n  const clean = str.toLowerCase().replace(/[^a-z0-9]/g, '');\n  return clean === clean.split('').reverse().join('');\n}", "test_case": "assert(isPalindrome('Racecar') === true)", "license": "MIT", "split": "train"}
    ]
    code_val_records = [
        {"id": "code_val_01", "language": "python", "prompt": "Write a binary search implementation.", "code": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1", "test_case": "assert binary_search([1, 2, 4, 8], 4) == 2", "license": "MIT", "split": "val"}
    ]

    with open(os.path.join(MANIFESTS_DIR, "code_train.jsonl"), "w", encoding="utf-8") as f:
        for r in code_train_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(MANIFESTS_DIR, "code_val.jsonl"), "w", encoding="utf-8") as f:
        for r in code_val_records:
            f.write(json.dumps(r) + "\n")

    # 5. Isolate Legacy Synthetic Benchmarks
    legacy_img_manifest = os.path.join(PROJECT_ROOT, "datasets", "image", "manifest.jsonl")
    if os.path.isfile(legacy_img_manifest):
        with open(legacy_img_manifest, "r", encoding="utf-8") as f_in, \
             open(os.path.join(MANIFESTS_DIR, "synthetic_image_dev.jsonl"), "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.strip():
                    item = json.loads(line.strip())
                    item["is_synthetic"] = True
                    item["source_type"] = "procedural_synthetic_seed"
                    f_out.write(json.dumps(item) + "\n")

    legacy_vid_manifest = os.path.join(PROJECT_ROOT, "datasets", "video", "manifest.jsonl")
    if os.path.isfile(legacy_vid_manifest):
        with open(legacy_vid_manifest, "r", encoding="utf-8") as f_in, \
             open(os.path.join(MANIFESTS_DIR, "synthetic_video_dev.jsonl"), "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.strip():
                    item = json.loads(line.strip())
                    item["is_synthetic"] = True
                    item["source_type"] = "procedural_benchmark_fixture"
                    f_out.write(json.dumps(item) + "\n")

    print("\n✅ Stage 2.0 Balanced Pilot Dataset Foundation successfully built!")

if __name__ == "__main__":
    build_all()
