"""
training/video/train.py
Runnable Sovereign Training Script for Moon Video Engine.
Trains spatiotemporal latent diffusion network from scratch with random initialization.
Zero pretrained weights imported.
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_video.config import MoonVideoConfig
from backend.engines.moon_video.model import MoonVideoModel
from training.common.seed import set_seed
from training.common.distributed import setup_device, get_vram_usage_mb
from training.common.metrics import MetricTracker
from training.common.checkpoint import save_checkpoint
from training.video.dataset import MoonVideoDataset

def train(args):
    print("=" * 65)
    print("   MOON VIDEO SOVEREIGN TRAINING (Random Initialization)")
    print("=" * 65)

    set_seed(args.seed)
    device, dtype = setup_device(args.device)
    print(f"Device:            {device}")
    print(f"Compute Precision: {dtype}")
    print(f"Manifest:          {args.manifest}")
    print(f"Num Frames:        {args.num_frames}\n")

    # 1. Initialize Video Model from Scratch
    config = MoonVideoConfig(
        num_frames=args.num_frames,
        frame_height=args.frame_size,
        frame_width=args.frame_size
    )
    model = MoonVideoModel(config=config).to(device)
    param_counts = model.count_parameters()
    print("Spatiotemporal Model Architecture Initialized (Random Gaussian Initialization):")
    for k, v in param_counts.items():
        print(f"  • {k}: {v:,} parameters")
    print(f"  • Provenance Tag: {model.provenance}\n")

    # 2. Setup Dataset
    manifest_full = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest))
    # If video manifest is empty, fall back to image manifest for synthetic camera motion tests
    if not os.path.isfile(manifest_full) or os.path.getsize(manifest_full) == 0:
        manifest_full = os.path.normpath(os.path.join(PROJECT_ROOT, "datasets", "image", "manifest.jsonl"))

    train_dataset = MoonVideoDataset(
        manifest_path=manifest_full,
        base_dir=PROJECT_ROOT,
        split="train",
        num_frames=args.num_frames,
        image_size=args.frame_size
    )

    if len(train_dataset) == 0:
        raise ValueError(f"No samples found in manifest '{manifest_full}' with split='train'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False
    )

    # 3. Optimizer & Training Loop
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    tracker = MetricTracker()
    total_steps = args.max_steps if not args.dry_run else 2
    step = 0
    t0 = time.time()

    model.train()
    print(f"Starting training run: max_steps={total_steps}, batch_size={args.batch_size}...")

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            step += 1

            video_values = batch["video_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            cam_ids = batch["camera_id"].to(device)
            b = video_values.shape[0]

            timesteps = torch.randint(0, config.timesteps, (b,), device=device).long()

            pred_noise, target_noise = model(video_values, input_ids, timesteps, camera_id=cam_ids)
            loss = F.mse_loss(pred_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            tracker.update("loss", loss.item())

            if step % args.log_every == 0 or step == total_steps:
                avg_loss = tracker.get_recent_avg("loss")
                vram_mb = get_vram_usage_mb()
                print(f"[Step {step:4d}/{total_steps:4d}] Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | VRAM: {vram_mb:.1f} MB")

    elapsed = time.time() - t0
    final_loss = tracker.get_recent_avg("loss")
    print(f"\n✅ Video training completed in {elapsed:.2f}s | Final Loss: {final_loss:.4f}")

    ckpt_path = os.path.join(PROJECT_ROOT, args.output_dir, "moon_video_v0.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        epoch=1,
        loss=final_loss,
        config_dict=config.__dict__,
        output_path=ckpt_path
    )
    print(f"✅ Video Checkpoint saved to: {os.path.relpath(ckpt_path, PROJECT_ROOT)}")
    return ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Moon Video Spatiotemporal Model.")
    parser.add_argument("--manifest", type=str, default="datasets/image/manifest.jsonl")
    parser.add_argument("--frame_size", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=8)  # Lightweight 8 frames default for rapid testing
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2 quick steps to verify pipeline")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    train(args)
