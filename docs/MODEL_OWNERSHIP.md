# 🌙 MOON AI — STRICT MODEL OWNERSHIP STANDARD
**Chief ML Systems Architect**: Prem Gaikwad (India 🇮🇳)  
**Standard Version**: 1.0.0 (Stage 0 & Stage 1 Foundation)  
**Status**: ACTIVE & ENFORCED

---

## 1. Core Principle & Philosophy

Moon AI is committed to becoming a **genuinely self-owned, sovereign artificial intelligence studio**.

We firmly reject the illusion of "wrapper AI"—systems that present third-party cloud APIs (such as Pollinations, Replicate, fal.ai, Midjourney, Kling, or Runway) under custom branding. Every neural generative capability in Moon AI must be mathematically and computationally sovereign.

### The Non-Negotiable Rules of Strict Model Ownership:
1. **Zero Third-Party Generative APIs**:
   - No requests to `text.pollinations.ai`, `image.pollinations.ai`, `replicate.com`, `fal.ai`, or commercial inference gateways.
   - External hosted generation fallbacks are strictly prohibited and purged from the codebase.
2. **Zero External Pretrained Weights for Sovereign Engines**:
   - Moon Image (`moon-image-v0`) and Moon Video (`moon-video-v0`) are initialized **strictly from random Gaussian weights** (`moon_ai_random_init`).
   - Importing external weights (e.g., Stable Diffusion, SDXL, Flux, AnimateDiff, ModelScope, CogVideo) is strictly prohibited in sovereign tiers.
3. **Truthful State Reporting**:
   - When a sovereign model has not been trained or a checkpoint is missing, the system must return a truthful diagnostic response with status `MODEL_NOT_TRAINED`.
   - The platform will **never** silently fall back to an external API or claim procedural Canvas/SVG animations are neural diffusion.
4. **Verified Weight Provenance**:
   - Every saved checkpoint must contain an immutable metadata envelope with `"provenance": "moon_ai_random_init"`.
   - Any checkpoint missing this tag or containing foreign weights will fail validation and be rejected by `training/common/checkpoint.py`.

---

## 2. Inventory of Sovereign Engines

| Engine | ID | Architecture | Weight Provenance | Target Resolution | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Moon Image** | `moon-image-v0` | `MoonUNet2D` + `MoonTextEncoder` + `MoonImageAutoencoder` | `moon_ai_random_init` | 256×256 (expandable to 512×512) | Scaffolded & Verified |
| **Moon Video** | `moon-video-v0` | `MoonUNet3D` (Spatial Cross-Attention + 1D Temporal Self-Attention) | `moon_ai_random_init` | 128×128, 16 frames, 8 FPS | Scaffolded & Verified |
| **Moon LLM** | `moon-llm-scratch` | Autoregressive Decoder Transformer (`nanoGPT` core) | `moon_ai_random_init` | Context length: 256 tokens | Scaffolded & Active |

---

## 3. Provenance Verification in Code

Checkpoint saving and loading enforce strict provenance checks:

```python
# training/common/checkpoint.py
REQUIRED_PROVENANCE = "moon_ai_random_init"

def save_checkpoint(model, optimizer, step, epoch, loss, config_dict, output_path):
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict,
        "provenance": REQUIRED_PROVENANCE,
        "architect": "Prem Gaikwad (India 🇮🇳)",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    torch.save(checkpoint, output_path)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if ckpt.get("provenance") != REQUIRED_PROVENANCE:
        raise ValueError(
            f"Provenance validation failed for {checkpoint_path}! "
            f"Expected '{REQUIRED_PROVENANCE}', found '{ckpt.get('provenance')}'. "
            "External pretrained weights are prohibited under Strict Model Ownership."
        )
    model.load_state_dict(ckpt["model_state_dict"])
```

---

## 4. Deceptive Patterns Purged

During the Stage 0 + Stage 1 refactor, all deceptive patterns were audited and permanently eliminated:

1. **Wikipedia Scraping**:
   - `webapp/image_service.py` previously fetched Wikipedia thumbnails via `urllib` and added a footer banner. This was **completely removed**.
2. **Third-Party Pollinations Forwarding**:
   - `api/chat.js` and `frontend/api/chat.js` routed image generation to `https://image.pollinations.ai/prompt/` and chat generation to `https://text.pollinations.ai/`. These have been **completely purged**.
   - `app.py` previously constructed `pollinations.ai` image URLs. This has been **completely purged**.
3. **Deceptive Canvas Simulations**:
   - HTML5 canvas particle animations were previously described as "dynamic spatio-temporal video simulations". These are now truthfully identified as architectural diagnostics when models are in `MODEL_NOT_TRAINED` state.

---

## 5. Audit Checklist for Pull Requests & Commits

Before any pull request or deployment is approved, the following checklist must be verified:

- [x] No imports of `diffusers.StableDiffusionPipeline`, `fal_client`, or `replicate`.
- [x] No HTTP requests to `*.pollinations.ai`, `api.openai.com`, or other hosted LLM/image endpoints.
- [x] All new model classes inherit from native `torch.nn.Module` and initialize with random weights.
- [x] Training scripts support `--dry_run` for CI verification without GPU requirements.
- [x] Missing checkpoints return `MODEL_NOT_TRAINED` with explicit training instructions.
- [x] Attribution explicitly states Chief Architect Prem Gaikwad (India 🇮🇳).
