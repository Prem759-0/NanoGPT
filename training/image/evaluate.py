"""
training/image/evaluate.py
Evaluation and Quality Verification Suite for Moon Image Engine.
Computes validation loss, evaluates fixed held-out benchmark prompts,
measures sample diversity (mode-collapse detection), and verifies provenance.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_image.pipeline import MoonImagePipeline
from backend.engines.moon_image.model import MoonImageModel
from training.image.dataset import MoonImageDataset
from training.common.checkpoint import load_checkpoint, compute_file_sha256

BENCHMARK_PROMPTS = [
    "A cybernetic chrome sphere floating over reflective grid with neon magenta lighting",
    "Close up portrait of an astronaut on Mars looking at Earth in the sky",
    "A serene Japanese zen garden with glowing cherry blossoms in the rain",
    "Futuristic sports car driving along a highway at sunset with volumetric lighting"
]

def calculate_sample_diversity(images: list) -> float:
    """Computes pixel-level variance across generated samples to detect mode collapse."""
    if len(images) < 2:
        return 0.0
    arrays = [np.array(im).astype(np.float32) / 255.0 for im in images]
    stacked = np.stack(arrays, axis=0)  # [N, H, W, C]
    # Variance across images
    variance = np.var(stacked, axis=0).mean()
    return float(variance)

def evaluate(args):
    print("=" * 70)
    print("   MOON IMAGE BENCHMARK EVALUATION SUITE")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    ckpt_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.checkpoint)) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if not os.path.isfile(ckpt_path):
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        return {"status": "FAILED", "error": f"Checkpoint missing at '{ckpt_path}'"}

    # 1. Enforce Provenance Verification
    try:
        raw_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        provenance = raw_meta.get("provenance")
        if provenance != "moon_ai_random_init":
            print(f"❌ STRICT OWNERSHIP VIOLATION: Unknown provenance '{provenance}'.")
            return {"status": "FAILED", "error": "Provenance violation"}
    except Exception as e:
        print(f"❌ Failed to inspect checkpoint: {e}")
        return {"status": "FAILED", "error": str(e)}

    pipeline = MoonImagePipeline.from_checkpoint(ckpt_path, device=device)
    model = pipeline.model
    model.eval()

    # 2. Validation Loss across held-out split
    manifest_full = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest))
    val_dataset = MoonImageDataset(manifest_path=manifest_full, base_dir=PROJECT_ROOT, split="val")
    if len(val_dataset) == 0:
        val_dataset = MoonImageDataset(manifest_path=manifest_full, base_dir=PROJECT_ROOT, split="test")

    val_loss = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                t = torch.randint(0, model.config.timesteps, (1,), device=device).long()
                pred, target = model(pv, ids, t)
                total_loss += F.mse_loss(pred, target).item()
        val_loss = float(total_loss / len(val_dataset))
        print(f"• Validation Denoising Loss: {val_loss:.4f} (across {len(val_dataset)} held-out samples)")
    else:
        print("• Validation Loss: Skipped (no explicit val/test split in manifest)")

    # 3. Fixed Benchmark Prompt Generation
    out_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.output_dir))
    os.makedirs(out_dir, exist_ok=True)
    generated_images = []
    sample_records = []

    print(f"\nSynthesizing {len(BENCHMARK_PROMPTS)} benchmark samples (steps={args.steps}, guidance={args.guidance})...")
    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        seed = 42 + idx
        img = pipeline.text_to_image(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed
        )
        img_filename = f"eval_sample_{idx}.png"
        img_path = os.path.join(out_dir, img_filename)
        img.save(img_path)
        generated_images.append(img)
        sample_records.append({
            "index": idx,
            "prompt": prompt,
            "seed": seed,
            "image_file": img_filename
        })
        print(f"  [{idx + 1}/{len(BENCHMARK_PROMPTS)}] Prompt: \"{prompt[:45]}...\" -> {img_filename}")

    # 4. Diversity / Mode Collapse Metric
    diversity_score = calculate_sample_diversity(generated_images)
    print(f"\n• Sample Pixel Diversity Score: {diversity_score:.4f} (higher is better, 0 indicates complete collapse)")

    # 5. Save Evaluation Report
    report = {
        "status": "PASSED",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT),
        "provenance": provenance,
        "checkpoint_step": raw_meta.get("step", 0),
        "checkpoint_train_loss": raw_meta.get("loss", 0.0),
        "validation_loss": val_loss,
        "sample_diversity_score": diversity_score,
        "manifest_path": args.manifest,
        "manifest_hash": compute_file_sha256(manifest_full),
        "benchmark_samples": sample_records,
        "total_parameters": model.count_parameters()["total_trainable_params"]
    }

    report_path = os.path.join(out_dir, "eval_image_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Evaluation completed. Summary written to: {os.path.relpath(report_path, PROJECT_ROOT)}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Moon Image Model.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/moon_image_v0.pt")
    parser.add_argument("--manifest", type=str, default="datasets/image/manifest.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/reports/image_eval")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate(args)
