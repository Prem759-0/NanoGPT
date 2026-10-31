"""
scripts/colab_train.py
Google Colab Sovereign Training Runner for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Supports:
- Automatic hardware accelerator detection (NVIDIA Tesla T4, A10G, A100, V100, or CPU)
- Dynamic VRAM adaptation (precision, batch size, gradient accumulation)
- Google Drive continuous checkpoint synchronization
- Session resumption from Google Drive
- Smoke-test mode (--smoke_test) and Production mode (--production)
- Sovereign training tracks: image, video, llm
"""
import os
import sys
import shutil
import argparse
import subprocess
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def detect_colab_environment():
    """Detects whether running inside Google Colab environment."""
    return "COLAB_GPU" in os.environ or os.path.exists("/content")

def get_hardware_profile():
    """Inspects CUDA device, VRAM capacity, and determines optimal training configuration."""
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "name": "CPU",
            "vram_gb": 0.0,
            "recommended_precision": "fp32",
            "batch_size_image": 1,
            "batch_size_video": 1,
            "grad_accum": 4
        }

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Determine recommended precision and batch parameters based on GPU tier
    if "A100" in device_name:
        precision = "bf16"
        batch_img = 8
        batch_vid = 4
        grad_accum = 2
    elif "A10G" in device_name or "V100" in device_name:
        precision = "fp16"
        batch_img = 4
        batch_vid = 2
        grad_accum = 4
    elif "T4" in device_name:
        precision = "fp16"
        batch_img = 2
        batch_vid = 1
        grad_accum = 8
    else:
        precision = "fp16" if total_vram_gb >= 8.0 else "fp32"
        batch_img = 2
        batch_vid = 1
        grad_accum = 4

    return {
        "device": "cuda",
        "name": device_name,
        "vram_gb": round(total_vram_gb, 2),
        "recommended_precision": precision,
        "batch_size_image": batch_img,
        "batch_size_video": batch_vid,
        "grad_accum": grad_accum
    }

def sync_checkpoint_to_drive(src_path: str, drive_dir: str):
    """Copies checkpoint atomically to Google Drive directory."""
    if not os.path.isdir(drive_dir):
        try:
            os.makedirs(drive_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Could not create Drive directory: {e}")
            return
    try:
        fname = os.path.basename(src_path)
        dest_path = os.path.join(drive_dir, fname)
        shutil.copy2(src_path, dest_path)
        print(f"☁️ Synced checkpoint to Google Drive: {dest_path}")
    except Exception as e:
        print(f"⚠️ Drive sync error: {e}")

def run_image_track(args, hw, drive_dir: str):
    print("\n" + "=" * 65)
    print("   RUNNING MOON IMAGE TRAINING TRACK")
    print("=" * 65)
    train_script = os.path.join(PROJECT_ROOT, "training", "image", "train.py")
    steps = 5 if args.smoke_test else (args.steps or 200)
    precision = args.precision or hw["recommended_precision"]
    batch_size = args.batch_size or hw["batch_size_image"]
    grad_accum = args.grad_accum or hw["grad_accum"]

    cmd = [
        sys.executable, train_script,
        "--max_steps", str(steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--precision", precision,
        "--manifest", "datasets/image/manifest.jsonl"
    ]

    # Resume handling from Drive or local checkpoints
    if args.resume:
        drive_ckpt = os.path.join(drive_dir, "best_moon_image.pt") if drive_dir else None
        local_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_moon_image.pt")
        if drive_ckpt and os.path.isfile(drive_ckpt):
            print(f"🔄 Resuming from Drive checkpoint: {drive_ckpt}")
            cmd.extend(["--resume", drive_ckpt])
        elif os.path.isfile(local_ckpt):
            print(f"🔄 Resuming from local checkpoint: {local_ckpt}")
            cmd.extend(["--resume", local_ckpt])

    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"❌ Image training failed with exit code {ret}")
        return ret

    # Sync back to Drive
    if drive_dir:
        for f in ["best_moon_image.pt", "moon_image_v0.pt"]:
            local_f = os.path.join(PROJECT_ROOT, "models", "checkpoints", f)
            if os.path.isfile(local_f):
                sync_checkpoint_to_drive(local_f, drive_dir)
    return 0

def run_video_track(args, hw, drive_dir: str):
    print("\n" + "=" * 65)
    print("   RUNNING MOON VIDEO TRAINING TRACK")
    print("=" * 65)
    train_script = os.path.join(PROJECT_ROOT, "training", "video", "train.py")
    steps = 5 if args.smoke_test else (args.steps or 100)
    precision = args.precision or hw["recommended_precision"]
    batch_size = args.batch_size or hw["batch_size_video"]
    grad_accum = args.grad_accum or hw["grad_accum"]

    cmd = [
        sys.executable, train_script,
        "--max_steps", str(steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--precision", precision,
        "--manifest", "datasets/video/manifest.jsonl"
    ]

    if args.smoke_test:
        cmd.append("--allow_synthetic")

    # Resume handling
    if args.resume:
        drive_ckpt = os.path.join(drive_dir, "best_moon_video.pt") if drive_dir else None
        local_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_moon_video.pt")
        if drive_ckpt and os.path.isfile(drive_ckpt):
            print(f"🔄 Resuming from Drive checkpoint: {drive_ckpt}")
            cmd.extend(["--resume", drive_ckpt])
        elif os.path.isfile(local_ckpt):
            print(f"🔄 Resuming from local checkpoint: {local_ckpt}")
            cmd.extend(["--resume", local_ckpt])

    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"❌ Video training failed with exit code {ret}")
        return ret

    if drive_dir:
        for f in ["best_moon_video.pt", "moon_video_v0.pt"]:
            local_f = os.path.join(PROJECT_ROOT, "models", "checkpoints", f)
            if os.path.isfile(local_f):
                sync_checkpoint_to_drive(local_f, drive_dir)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Moon AI Google Colab Sovereign Training Runner")
    parser.add_argument("--track", choices=["image", "video", "all"], default="image", help="Training track to execute")
    parser.add_argument("--smoke_test", action="store_true", help="Execute short 5-step validation run")
    parser.add_argument("--production", action="store_true", help="Execute full production run")
    parser.add_argument("--steps", type=int, default=None, help="Explicit step count override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None, help="Precision override")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/MoonAI_Checkpoints", help="Google Drive sync directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest available checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("   MOON AI — SOVEREIGN TRAINING RUNNER")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("=" * 70)

    is_colab = detect_colab_environment()
    print(f"• Environment: {'Google Colab' if is_colab else 'Local Workstation'}")

    hw = get_hardware_profile()
    print(f"• Accelerator: {hw['name']} (Device: {hw['device']})")
    print(f"• Total VRAM: {hw['vram_gb']} GB")
    print(f"• Recommended Precision: {hw['recommended_precision']}")
    print(f"• Mode: {'SMOKE_TEST (5 steps)' if args.smoke_test else 'PRODUCTION'}")

    drive_dir = args.drive_dir if (is_colab and os.path.isdir("/content/drive")) else ""
    if drive_dir:
        print(f"• Google Drive Sync: ENABLED ({drive_dir})")
    else:
        print("• Google Drive Sync: Disabled (Local checkpoint saving only)")

    # Execute Tracks
    if args.track in ("image", "all"):
        code = run_image_track(args, hw, drive_dir)
        if code != 0:
            sys.exit(code)

    if args.track in ("video", "all"):
        code = run_video_track(args, hw, drive_dir)
        if code != 0:
            sys.exit(code)

    print("\n" + "=" * 70)
    print("✅ All requested training tracks finished successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
