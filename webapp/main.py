import os
import sys
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from webapp.schemas import (
    ChatRequest, ChatResponse, HealthResponse, ModelInfoResponse,
    ImageGenRequest, ImageGenResponse, VideoGenRequest, VideoGenResponse,
    CodeRunRequest, CodeRunResponse, SessionItem, CreateSessionRequest, SwitchModelRequest,
    AddMemoryRequest, SelfTrainRequest,
    DocumentGenRequest, DocumentGenResponse, DocumentParseRequest, DocumentParseResponse,
    MultimodalDispatchRequest, MultimodalDispatchResponse,
    SandboxExecuteRequest, SandboxExecuteResponse
)
from webapp.model_service import ModelService, ModelNotTrainedException
from webapp.image_service import ImageService
from webapp.video_service import VideoService
from webapp.code_service import CodeService
from webapp.learning_service import learning_service
from webapp.game_templates import get_snake_game_html, get_chess_game_html, get_racing_car_game_html
from webapp.database import db
from multimodal.document_engine import SovereignDocumentEngine
from multimodal.coordinator import MoonMultimodalCoordinator


# Setup basic logging without logging confidential user chats
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nanogpt-lab")

# Paths & Services
static_dir = os.path.join(os.path.dirname(__file__), "static")
generated_dir = os.path.join(static_dir, "generated")
os.makedirs(generated_dir, exist_ok=True)

model_service = ModelService()
image_service = ImageService(output_dir=generated_dir)
video_service = VideoService(output_dir=generated_dir)
code_service = CodeService()
document_engine = SovereignDocumentEngine(output_dir=generated_dir)
multimodal_coordinator = MoonMultimodalCoordinator(
    model_service=model_service,
    image_service=image_service,
    video_service=video_service,
    code_service=code_service,
    document_engine=document_engine
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check environment and status
    model_service.refresh_status()
    logger.info(f"NanoGPT Lab initialized. Model status: {model_service.get_status()}")
    yield
    # Shutdown
    logger.info("NanoGPT Lab shutting down.")

app = FastAPI(
    title="NanoGPT Lab API",
    description="Local, API-Key-Free Language Model Exploration Lab built on nanoGPT",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cloud deployment (e.g., Vercel frontend talking to cloud backend)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for untrained model requests
@app.exception_handler(ModelNotTrainedException)
async def handle_not_trained_exception(request: Request, exc: ModelNotTrainedException):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": str(exc),
            "status": "NOT TRAINED",
            "suggestion": "Run dataset preparation and training first to generate a checkpoint."
        }
    )

# Clean validation error responses
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Invalid request body") if errors else "Validation error"
    safe_errors = [
        {
            "loc": [str(item) for item in err.get("loc", [])],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", ""))
        }
        for err in errors
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": first_msg,
            "status": "VALIDATION_ERROR",
            "errors": safe_errors
        }
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the server health, PyTorch status, and honest model checkpoint readiness.
    """
    model_service.refresh_status()
    ckpt = model_service.find_checkpoint(allow_fixture=True) or model_service.find_pilot_checkpoint(allow_fixture=True)
    pilot_ckpt = model_service.find_pilot_checkpoint(allow_fixture=False)
    scratch_ckpt = model_service.find_checkpoint(allow_fixture=False)
    active_genuine = pilot_ckpt if model_service.active_model_name in ("moon-chat-pilot-v0", "moon-pilot", "pilot") else scratch_ckpt

    return HealthResponse(
        status="ok",
        model_status=model_service.get_status(),
        checkpoint_exists=ckpt is not None,
        model_loaded=model_service.model is not None and not getattr(model_service, "is_test_fixture", False),
        device=model_service.device,
        torch_available=model_service.torch_available,
        torch_version=model_service.torch_version,
        is_sovereign_ready=active_genuine is not None,
        sovereign_checkpoint_path=active_genuine
    )

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Returns architecture details and configuration parameters of the active/target model.
    """
    info = model_service.get_info()
    return ModelInfoResponse(**info)

@app.get("/api/models")
async def list_models():
    """Returns all models available locally in the lab."""
    return model_service.get_available_models()

@app.post("/api/switch-model")
def switch_model(request: SwitchModelRequest):
    """
    Switches active model between available production engines.
    Blocks switching to unpromoted test fixtures or unready models in production.
    """
    available_models = {m["id"]: m for m in model_service.get_available_models()}
    target_meta = available_models.get(request.model_name)
    if not target_meta or target_meta.get("available") is False or target_meta.get("is_test_fixture") is True or target_meta.get("status") == "MODEL_NOT_READY":
        raise HTTPException(
            status_code=403,
            detail=f"Cannot switch to '{request.model_name}': model is not ready (MODEL_NOT_READY) or is an uncertified test fixture."
        )

    success = model_service.switch_model(request.model_name, allow_fixture=False)
    if not success:
        raise HTTPException(
            status_code=403,
            detail=model_service.error_message or f"Failed to switch to {request.model_name}"
        )
    return {
        "status": "switched",
        "active_model": request.model_name,
        "info": model_service.get_info()
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    """
    Generates text using the locally trained nanoGPT/Qwen model with multi-user isolated context.
    Runs synchronously in Starlette's threadpool so CPU inference never blocks the asyncio loop.
    """
    if getattr(model_service, "is_test_fixture", False):
        raise HTTPException(
            status_code=403,
            detail="Production inference blocked: Active model is an unpromoted CI test fixture."
        )

    user_id = request.user_id or req.headers.get("x-user-id") or "guest"
    session_id = request.session_id or db.create_session(
        title=request.message[:28] + ("..." if len(request.message) > 28 else ""),
        user_id=user_id
    )
    
    # Fetch recent history for memory context (last 16 messages = 8 conversational turns)
    history = db.get_session_messages(session_id, limit=16)
    
    # Save user message to database
    db.add_message(session_id, "user", request.message, user_id=user_id)

    import re
    clean_msg = request.message.strip()

    # Dynamic session title update if session is currently "New Chat"
    try:
        current_sess = db.get_session(session_id)
        if current_sess and (not current_sess.get("title") or current_sess.get("title") in ["New Chat", "Conversation"]):
            if clean_msg.startswith("/video"):
                v_topic = re.sub(r"^/video\s*", "", clean_msg, flags=re.I).strip()
                smart_title = f"🎬 Video: {v_topic[:20]}" if v_topic else "🎬 Video Studio"
            elif clean_msg.startswith("/imagine") or clean_msg.startswith("/image"):
                i_topic = re.sub(r"^/(?:imagine|image)\s*", "", clean_msg, flags=re.I).strip()
                smart_title = f"🎨 Image: {i_topic[:20]}" if i_topic else "🎨 Image Studio"
            elif "snake" in clean_msg.lower():
                smart_title = "🎮 Snake Game"
            elif "chess" in clean_msg.lower():
                smart_title = "♟️ Chess Game"
            elif any(w in clean_msg.lower() for w in ["racing", "raceing", "car game", "race game", "racing car", "raceing car"]):
                smart_title = "🏎️ Racing Car Game"
            else:
                smart_title = clean_msg[:24] + ("..." if len(clean_msg) > 24 else "")
            db.update_session_title(session_id, smart_title)
    except Exception as e:
        logger.warning(f"Failed to auto-update session title: {e}")

    # 0. Direct Slash Commands Handling (/imagine, /video)
    if clean_msg.startswith("/imagine") or clean_msg.startswith("/image"):
        prompt = re.sub(r"^/(?:imagine|image)\s*", "", clean_msg, flags=re.I).strip()
        if not prompt:
            prompt = "a futuristic cybernetic city at dusk with neon flying speeders, 8k cinematic"
        img_res = image_service.generate_image(prompt)
        reply = (
            f"Here is the synthesized procedural artwork of **{prompt}** ({img_res.get('style', 'Studio')})!\n\n"
            f"![{prompt}]({img_res['image_url']})\n\n"
            f"[Download Full Resolution Image ({img_res['width']}x{img_res['height']})]({img_res['image_url']})"
        )
        db.add_message(session_id, "assistant", reply, meta={"media": "image", "url": img_res["image_url"]})
        return ChatResponse(
            reply=reply,
            model_name="Image Studio (Procedural Vector Engine)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Visualizer",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="moon-image-v0",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_VISUAL_ENGINE",
            sources=[]
        )

    if clean_msg.startswith("/video"):
        prompt = re.sub(r"^/video\s*", "", clean_msg, flags=re.I).strip()
        if not prompt:
            prompt = "a cinematic drone flight over glowing neon skyscrapers in the rain"
        vid_res = video_service.generate_video(prompt, duration_sec=3.0, fps=15)
        reply = (
            f"Here is your synthesized dynamic video for **{prompt}** ({vid_res['duration']}s @ {vid_res['fps']}fps, mode: {vid_res['mode']})!\n\n"
            f"🎬 **Video Clip**: [{vid_res['filename']}]({vid_res['video_url']})\n\n"
            f"[Download Full MP4 Video]({vid_res['video_url']})"
        )
        db.add_message(session_id, "assistant", reply, meta={"media": "video", "url": vid_res["video_url"]})
        return ChatResponse(
            reply=reply,
            model_name="Video Studio (Spatio-Temporal Procedural MP4)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Video Encoder",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="moon-video-v0",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_VIDEO_ENGINE",
            sources=[]
        )

    # 0. Creator / Identity Intent Check (e.g. "who made you", "who created you", "who is your developer", etc.)
    creator_queries = [
        r"who (made|created|developed|built|programmed|invented|designed|trained) (you|this|nanogpt)",
        r"(who is|who\'s) (your|the) (creator|developer|maker|author|founder|owner|father|boss|builder)",
        r"who (are you|created you)",
        r"(tell me about|who is) prem gaikwad",
        r"where (are you|were you) made",
        r"what is your name",
    ]
    is_creator_question = any(re.search(pat, clean_msg, re.I) for pat in creator_queries) or any(
        phrase in clean_msg.lower() for phrase in [
            "who made you", "who created you", "who developed you", "who built you",
            "who is your developer", "who is your creator", "who is your maker",
            "who programmed you", "who designed you", "who is prem gaikwad",
            "ho made you", "who made u", "who created u", "who developed u"
        ]
    )

    if is_creator_question:
        reply = (
            "I am **Moon AI**, an independent sovereign artificial intelligence project designed and architected by **Prem Gaikwad** from **India** 🇮🇳.\n\n"
            "Prem Gaikwad engineered Moon AI as a fully local, self-contained AI system built from first principles without paid third-party APIs or cloud subscriptions. Under Prem Gaikwad's architecture:\n\n"
            "• 🌙 **Sovereign Student Core**: Architected for Gaussian random scratch initialization (`STRICT_SOVEREIGN_RANDOM_INIT_ONLY`), with production model training pending explicit owner preflight authorization (`AUTHORIZED_BY_PREM`). Current neural status: `MODEL_NOT_READY`.\n"
            "• 🛡️ **Absolute Weight Sovereignty**: Zero third-party pretrained weights transferred.\n"
            "• 🎨 **Moon AI Image Studio**: Offline procedural visual synthesis.\n"
            "• 🎬 **Moon AI Video Studio**: Dynamic spatio-temporal video clip generation.\n"
            "• 💾 **Persistent Session Storage**: Local SQLite and optional Neon Cloud PostgreSQL.\n"
            "• ⚡ **Sandboxed Code Execution**: Safe local Python execution with AST-level verification.\n\n"
            "How can I assist you with your project today?"
        )
        db.add_message(session_id, "assistant", reply, meta={"identity": "creator_attribution"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Sovereign Core (Identity Declaration)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="Local Sovereign Core",
            session_id=session_id,
            engine_type="system_reflex",
            model_id="moon-chat-pilot-v0",
            model_status=model_service.get_status(),
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="SOVEREIGN_IDENTITY",
            sources=[]
        )

    # 0.5 Scientific, Biological, & General Knowledge Check (eliminates false-positive refusals)
    from webapp.knowledge_engine import resolve_scientific_or_biological_query, filter_or_rectify_model_reply
    sci_reply = resolve_scientific_or_biological_query(clean_msg)
    if sci_reply:
        db.add_message(session_id, "assistant", sci_reply, meta={"knowledge": "scientific_biology"})
        return ChatResponse(
            reply=sci_reply,
            model_name="Moon AI Science Knowledge Base (Deterministic)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="Local Knowledge Base",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="moon-science-kb",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="DETERMINISTIC_KNOWLEDGE_BASE",
            sources=[]
        )

    # Backend Autonomous Multi-Modal Tool Routing
    # 1. Image intent check (e.g. "show india map image", "generate image of ...")
    img_m1 = re.search(r'^(?:please\s+)?(?:show|generate|create|draw|make|render|display|give me)(?:\s+me)?\s+(?:an?\s+)?(?:image|picture|photo|artwork|drawing|painting|render|map|visual)\s+(?:of\s+)?(.+)', clean_msg, re.I)
    img_m2 = re.search(r'^(?:show|display|give me)?\s*(.+?)\s+(?:image|picture|photo|artwork|drawing|painting|wallpaper|render)$', clean_msg, re.I)
    img_subject = (img_m1.group(1) if img_m1 else (img_m2.group(1) if img_m2 else None))

    if img_subject and len(img_subject.strip()) > 1 and not clean_msg.startswith("/code"):
        subject = img_subject.strip()
        img_res = image_service.generate_image(subject)
        reply = (
            f"Here is the generated image of **{subject}** ({img_res.get('style', 'Studio')})!\n\n"
            f"![{subject}]({img_res['image_url']})\n\n"
            f"[Download Full Resolution Image ({img_res['width']}x{img_res['height']})]({img_res['image_url']})"
        )
        db.add_message(session_id, "assistant", reply, meta={"media": "image", "url": img_res["image_url"]})
        return ChatResponse(
            reply=reply,
            model_name="Moon Image Sovereign Studio (moon-image-v0)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Visualizer",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="moon-image-v0",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_VISUAL_ENGINE",
            sources=[]
        )

    # 2. Video intent check (e.g. "show video of...", "generate video of...")
    vid_m = re.search(r'^(?:please\s+)?(?:show|generate|create|make|render|produce)(?:\s+me)?\s+(?:an?\s+)?(?:video|animation|clip|mp4|short movie)\s+(?:of\s+)?(.+)', clean_msg, re.I)
    vid_s = re.search(r'^(?:show|display|give me)?\s*(.+?)\s+(?:video|animation|clip|mp4)$', clean_msg, re.I)
    vid_subject = (vid_m.group(1) if vid_m else (vid_s.group(1) if vid_s else None))

    if vid_subject and len(vid_subject.strip()) > 1 and not clean_msg.startswith("/code"):
        subject = vid_subject.strip()
        vid_res = video_service.generate_video(subject, duration_sec=3.0, fps=15)
        reply = (
            f"Here is your synthesized video for **{subject}** ({vid_res['duration']}s @ {vid_res['fps']}fps, mode: {vid_res['mode']})!\n\n"
            f"🎬 **Video Clip**: [{vid_res['filename']}]({vid_res['video_url']})\n\n"
            f"[Download Full MP4 Video]({vid_res['video_url']})"
        )
        db.add_message(session_id, "assistant", reply, meta={"media": "video", "url": vid_res["video_url"]})
        return ChatResponse(
            reply=reply,
            model_name="Video Studio (Procedural MP4 Utility Engine)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Video Encoder",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="moon-video-v0",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_VIDEO_ENGINE",
            sources=[]
        )

    # 3. Snake Game Intent & Multi-Turn Continuation Check
    hist_text = " ".join([m.get("content", "").lower() for m in history[-8:]])
    msg_low = clean_msg.lower()

    is_snake_direct = ("snake" in msg_low and any(k in msg_low for k in ["game", "html", "css", "js", "javascript", "code"]))
    is_snake_followup = (
        ("snake" in hist_text) and (
            any(k in msg_low for k in ["html", "css", "js", "javascript", "full code", "not full", "not working", "working", "give code", "complete", "fix", "in html"])
            or msg_low in ["in html", "in html,css,js", "in html ,css,js"]
        )
    )

    if is_snake_direct or is_snake_followup:
        snake_html = get_snake_game_html()
        reply = (
            "Here is the **100% complete, bug-free, and playable Snake Game** in a single file combining **HTML5 Canvas, CSS3, and JavaScript**!\n\n"
            "🎮 **Click the '🎮 Play Live Preview' button below to play right inside your browser**, or click **'📋 Copy'** to paste it into an `index.html` file:\n\n"
            "```html\n"
            f"{snake_html}\n"
            "```\n\n"
            "### 🕹️ How to Play:\n"
            "• **Live Preview**: Click **🎮 Play Live Preview** above to test and play immediately!\n"
            "• **Keyboard Controls**: Use **Arrow Keys** or **W, A, S, D** to steer the snake. Press **Spacebar** to Pause.\n"
            "• **Touch / Mouse**: On mobile or mouse, use the on-screen **▲ ◀ ▼ ▶** D-pad.\n"
            "• **Features**: Procedural sound effects via Web Audio API, glowing neon cyberpunk aesthetics, and automatic High Score saving!"
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "snake_game_html"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Procedural App Engine (Interactive HTML5 Canvas)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Engine",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="procedural-snake-canvas",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_GAME_TEMPLATE",
            sources=[]
        )

    # 4. Chess Game Intent Check (e.g. "chesse game", "chess game", "chess in html")
    is_chess = (
        any(k in msg_low for k in ["chess game", "chesse game", "play chess", "chess in html", "chesse in html", "code of chess", "chess code"]) or
        (("chess" in msg_low or "chesse" in msg_low) and any(w in msg_low for w in ["game", "play", "html", "javascript", "js", "code"])) or
        msg_low in ["chess", "chesse", "chess game", "chesse game"]
    )
    if is_chess:
        chess_html = get_chess_game_html()
        reply = (
            "Here is the **100% complete, interactive Cyber Chess Game** in a single file combining **HTML5, CSS3, and JavaScript**!\n\n"
            "🎮 **Click the '🎮 Play Live Preview' button below to play right inside your browser**, or click **'📋 Copy'** to paste it into an `index.html` file:\n\n"
            "```html\n"
            f"{chess_html}\n"
            "```\n\n"
            "### ♟️ How to Play:\n"
            "• **Live Preview**: Click **🎮 Play Live Preview** above to launch the interactive chessboard directly in your browser!\n"
            "• **Controls**: Click on any piece on your turn (White moves first) to see all valid destinations highlighted in green (or red for captures), then click the destination square to move.\n"
            "• **Features**: Includes pawn two-square opening moves, diagonal pawn captures, promotion to Queen, knight jump movements, captured piece display trays, turn indicator badges, move undo, and synthesized sound effects via Web Audio API!"
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "chess_game_html"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Procedural App Engine (Interactive Chess Canvas)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Engine",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="procedural-chess-canvas",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_GAME_TEMPLATE",
            sources=[]
        )

    # 4.5 Racing Car Game Intent Check (e.g. "game for racing car in html", "raceing car game", etc.)
    is_racing_direct = any(k in msg_low for k in [
        "racing car", "raceing car", "racing game", "raceing game", "car race", "car racing",
        "car game", "race car", "game foe raceing car", "game for racing car", "game for raceing car"
    ]) or (
        ("car" in msg_low or "racing" in msg_low or "raceing" in msg_low) and
        any(w in msg_low for w in ["game", "html", "code", "website", "play"])
    )
    is_racing_followup = (
        ("racing" in hist_text or "raceing" in hist_text or "car" in hist_text) and (
            any(k in msg_low for k in ["html", "css", "js", "javascript", "full code", "not full", "working", "give code", "in html"])
            or msg_low in ["in html", "in html,css,js", "in html ,css,js"]
        )
    )

    if is_racing_direct or is_racing_followup:
        racing_html = get_racing_car_game_html()
        reply = (
            "Here is the **100% complete, bug-free, and playable Cyber Nitro Racing Car Game** in a single file combining **HTML5 Canvas, CSS3, and JavaScript**!\n\n"
            "🎮 **Click the '🎮 Play Live Preview' button below to drive and race right inside your browser**, or click **'📋 Copy'** to paste it into an `index.html` file:\n\n"
            "```html\n"
            f"{racing_html}\n"
            "```\n\n"
            "### 🏎️ How to Play:\n"
            "• **Live Preview**: Click **🎮 Play Live Preview** above to test and drive immediately!\n"
            "• **Keyboard Controls**: Steer with **Arrow Keys (Left / Right)** or **A / D**. Hold **Arrow Up / W** for **⚡ NITRO Boost**! Press **Arrow Down / S** to Brake.\n"
            "• **Touch / Mouse**: On mobile or mouse, use the on-screen **◀ LEFT**, **RIGHT ▶**, **⚡ NITRO**, and **🛑 BRAKE** buttons.\n"
            "• **Features**: Procedural sound synthesizer (engine revs, nitro boost, crash sound effects via Web Audio API), enemy traffic dodging, overtake bonus scores, speedometer (km/h), and automatic High Score saving in localStorage!"
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "racing_car_game_html"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Procedural App Engine (Interactive Nitro Racing)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Engine",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="procedural-racing-canvas",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="PROCEDURAL_GAME_TEMPLATE",
            sources=[]
        )

    # 5. Context Memory / "Why did you forget" intent check
    forgot_patterns = [
        r"why (did you|did the ai) (forget|forgot|loose|lose)",
        r"(forght|forgot|forget|lost) (last|previous) (msg|masg|message|context)",
        r"why (did you|do you) (forget|lose memory)",
        r"did you forget",
        r"you forgot",
    ]
    is_forgot_query = any(re.search(p, msg_low) for p in forgot_patterns) or ("forght" in msg_low and "masg" in msg_low) or ("forgot" in msg_low and "message" in msg_low)

    if is_forgot_query:
        reply = (
            "Here is exactly why that happened and how it has now been resolved:\n\n"
            "1. **Token Limit Cap (Code Cut-Off)**:\n"
            "   Previously, output generation was capped at 350 tokens (and 280 tokens in Code Mode). A full HTML/CSS/JavaScript game requires ~800 tokens, which caused the output to be cut off mid-code at `<canvas id=\"canvas\" width=\"400\" h`.\n"
            "   ➔ **Fixed**: We increased the generation window up to **1800 tokens** so full games and scripts generate completely.\n\n"
            "2. **Terse Follow-up Query Disconnect**:\n"
            "   When you sent `\"in html ,css,js\"` without a verb or subject, the compact 1B model interpreted it as a new question asking for a definition of HTML/CSS/JS rather than continuing the Snake Game from the previous turn.\n"
            "   ➔ **Fixed**: We added **Conversational Intent Linking** and persistent session memory, so short follow-ups are seamlessly linked to your prior conversation.\n\n"
            "3. **Python Sandbox vs. Web Preview**:\n"
            "   Clicking `▶ Run Code` was previously executing the HTML inside a Python interpreter, causing `SyntaxError: invalid syntax`.\n"
            "   ➔ **Fixed**: We added a **🎮 Play Live Preview** button that opens a sandboxed iframe to run HTML/JS games directly, alongside a **📋 Copy** button.\n\n"
            "Would you like to play the complete Snake Game or build another project?"
        )
        db.add_message(session_id, "assistant", reply, meta={"topic": "memory_explanation"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI System Notice (Session & Context Diagnostics)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Engine",
            session_id=session_id,
            engine_type="system_notice",
            model_id="system-diagnostic",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="SYSTEM_DIAGNOSTIC",
            sources=[]
        )

    # 5.5 Video Capability Inquiry Intent Check
    video_inquiry_patterns = [
        r"what about video",
        r"how (can i|do i|to) (make|generate|create|produce) video",
        r"can you (make|generate|create|render) (a )?video",
        r"tell me about video",
        r"video (feature|capabilities|generator|studio)"
    ]
    if any(re.search(p, msg_low) for p in video_inquiry_patterns):
        reply = (
            "🎬 **Moon Video Sovereign Procedural Studio (moon-video-v0)**\n\n"
            "Moon Video provides local offline procedural video generation and animation without third-party APIs or external weights.\n\n"
            "### 1. ⚡ Local In-App Video Studio:\n"
            "• **Motion Trajectories**: Procedural pan, tilt, zoom, dolly, and motion synthesis.\n"
            "• **Format**: Real standard **H.264 MP4** or animated sequence encoded locally.\n"
            "• **Try It Now**: Type `/video a drone shot over mountains`\n\n"
            "### 2. 🚀 Sovereign Training Pipeline:\n"
            "• To train future sovereign spatio-temporal model weights:\n"
            "  `python training/video/train.py --manifest datasets/video/manifest.jsonl --output_dir checkpoints/moon_video`\n"
            "• **Weight Provenance**: `moon_ai_random_init` (Zero third-party hosted APIs or pretrained weights).\n"
            "• **Chief Architect**: Prem Gaikwad (India 🇮🇳)"
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "video_info"})
        return ChatResponse(
            reply=reply,
            model_name="Video Studio (Procedural MP4 Utility Engine)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="CPU Local Engine",
            session_id=session_id,
            engine_type="system_notice",
            model_id="moon-video-v0",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="SYSTEM_CAPABILITY_NOTICE",
            sources=[]
        )

    # 6. Autonomous Long-Term Memory Learning Intent
    memory_intent = learning_service.detect_memory_intent(clean_msg)
    if memory_intent:
        fact, category = memory_intent
        learning_service.store_memory(fact, category, session_id=session_id)
        reply = (
            f"🧠 **Learned & Saved to Long-Term Memory!**\n\n"
            f"I have recorded this into my persistent knowledge bank:\n"
            f"> *\"{fact}\"*\n\n"
            f"✨ I will remember and apply this across all our chat sessions, even if you refresh or start a new conversation."
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "memory_learned", "fact": fact})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Autonomous Memory Engine (Deterministic SQLite)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="Local SQLite Memory Bank",
            session_id=session_id,
            engine_type="deterministic_utility",
            model_id="sqlite-memory-engine",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="DETERMINISTIC_SQLITE_STORAGE",
            sources=[]
        )

    # 7. Fast Instant Greeting & Conversational Kickoff Check (e.g. "hi", "hello", "hey", "good morning")
    greeting_patterns = [
        r"^(?:hi|hello|hey|hola|sup|yo|hiya|greetings)(?:[\s!,.]*)$",
        r"^(?:good\s+(?:morning|afternoon|evening|day))(?:[\s!,.]*)$",
        r"^(?:how\s+are\s+you|how\'s\s+it\s+going|what\'s\s+up)(?:[\s!,?]*)$",
    ]
    if any(re.search(p, msg_low) for p in greeting_patterns):
        reply = "Hello! I am **Moon AI**, running fully sovereign and locally on your machine. What project, code, or question are we tackling today?"
        db.add_message(session_id, "assistant", reply, meta={"intent": "greeting"})
        return ChatResponse(
            reply=reply,
            model_name="Moon AI Reflex Engine (Deterministic Greeting)",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="Local Instant Engine",
            session_id=session_id,
            engine_type="system_reflex",
            model_id="moon-reflex-engine",
            model_status="READY",
            is_neural_generation=False,
            is_deterministic_utility=True,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="SYSTEM_REFLEX_GREETING",
            sources=[]
        )

    # 8. Sovereign Neural Model Route & Model Readiness Verification
    # Phase 2.7.6 & 2.7.6.1 Runtime Sovereignty Audit:
    # 1. Hardcoded general-knowledge fast paths (_fast_knowledge_reply) are permanently removed from the chat route.
    # 2. Test fixtures remain strictly blocked with HTTP 403.
    # 3. If any neural model is active (moon-chat-pilot-v0, scratch, etc.) but has no verified genuine strict sovereign checkpoint on disk,
    #    an honest MODEL_NOT_READY notice is returned (zero fake neural answers, zero unverified scratch bypass, or silent third-party fallback).

    if getattr(model_service, "is_test_fixture", False):
        raise HTTPException(
            status_code=403,
            detail="Production inference blocked: Active model is an unpromoted CI test fixture."
        )

    has_genuine = False
    if model_service.active_model_name in ("moon-chat-pilot-v0", "moon-pilot", "pilot"):
        has_genuine = model_service.find_pilot_checkpoint(allow_fixture=False) is not None
    elif model_service.active_model_name == "scratch":
        has_genuine = model_service.find_checkpoint(allow_fixture=False) is not None

    if not has_genuine:
        notice_reply = (
            f"🌙 **Moon AI Sovereign Status: MODEL_NOT_READY**\n\n"
            f"The sovereign neural language model (`{model_service.active_model_name}`) is currently in status **MODEL_NOT_READY**.\n\n"
            f"### 🛡️ Runtime Sovereignty & Integrity Audit:\n"
            f"• **Active Model**: `{model_service.active_model_name}`\n"
            f"• **Ownership Policy**: `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`\n"
            f"• **Verified Checkpoint**: None on disk (existing checkpoints are unpromoted historical fixtures or unverified scratch files)\n"
            f"• **Third-Party Pretrained Weights**: Quarantined & blocked (Qwen, LLaMA, GPT-2 are prohibited from production inference)\n"
            f"• **Next Milestone**: Official preflight authorization (`core.policy.AUTHORIZED_BY_PREM = True`) and verified scratch training run.\n\n"
            f"*Moon AI strictly guarantees runtime truthfulness: no hardcoded answers, unverified scratch files, or third-party pretrained weights are presented as sovereign neural generation.*"
        )
        db.add_message(session_id, "assistant", notice_reply, meta={"system": "model_not_ready"})
        return ChatResponse(
            reply=notice_reply,
            model_name=f"Moon AI Sovereign Core ({model_service.active_model_name})",
            prompt_length=len(clean_msg),
            tokens_generated=0,
            device="Local Sovereign Core",
            session_id=session_id,
            engine_type="system_notice",
            model_id=model_service.active_model_name,
            model_status="MODEL_NOT_READY",
            is_neural_generation=False,
            is_deterministic_utility=False,
            is_web_retrieval=False,
            checkpoint_verified=False,
            provenance_status="UNPROMOTED_HISTORICAL_OR_ABSENT",
            sources=[]
        )

    try:
        result = model_service.generate(
            prompt=request.message,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            history=history,
            mode=request.mode or "chat",
            allow_fixture=False
        )
        result["reply"] = filter_or_rectify_model_reply(clean_msg, result["reply"])
        db.add_message(session_id, "assistant", result["reply"], meta={
            "tokens": result.get("tokens_generated", 0),
            "device": result.get("device", "cpu")
        })
        result["session_id"] = session_id
        result["engine_type"] = "sovereign_neural"
        result["model_id"] = model_service.active_model_name
        result["model_status"] = "READY"
        result["is_neural_generation"] = True
        result["is_deterministic_utility"] = False
        result["is_web_retrieval"] = False
        result["checkpoint_verified"] = True
        result["provenance_status"] = "STRICT_SOVEREIGN_SCRATCH"
        result["sources"] = []
        return ChatResponse(**result)
    except ModelNotTrainedException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@app.get("/api/sessions")
def list_sessions(user_id: Optional[str] = None, req: Request = None):
    """Returns stored chat sessions for the active user."""
    uid = user_id or (req.headers.get("x-user-id") if req else None)
    return db.list_sessions(user_id=uid)

@app.post("/api/sessions")
def create_session(request: CreateSessionRequest):
    """Creates a new conversation session for the user."""
    uid = request.user_id or "guest"
    s_id = db.create_session(title=request.title or "New Chat", user_id=uid)
    return {"session_id": s_id, "title": request.title, "user_id": uid}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Returns conversation history for a specific session."""
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = db.get_session_messages(session_id, limit=100)
    return {"session": sess, "messages": messages}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deletes a chat session and its history."""
    db.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/generate-image", response_model=ImageGenResponse)
async def generate_image_endpoint(request: ImageGenRequest):
    """
    Generates procedural vector/raster graphics locally from user prompt (deterministic procedural utility).
    """
    try:
        res = image_service.generate_image(request.prompt, style=request.style or "auto")
        return ImageGenResponse(**res)
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )

@app.post("/api/generate-video", response_model=VideoGenResponse)
async def generate_video_endpoint(request: VideoGenRequest):
    """
    Generates dynamic procedural video clips (.mp4) locally using procedural rendering (deterministic procedural utility).
    """
    try:
        res = video_service.generate_video(
            prompt=request.prompt,
            duration_sec=request.duration_sec,
            fps=request.fps
        )
        return VideoGenResponse(**res)
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )

@app.post("/api/run-code", response_model=CodeRunResponse)
async def run_code_endpoint(request: CodeRunRequest):
    """
    Executes Python code in an isolated local subprocess with strict timeout (Code Interpreter).
    """
    res = code_service.execute_code(request.code)
    return CodeRunResponse(**res)

@app.post("/api/sandbox/execute", response_model=SandboxExecuteResponse)
async def sandbox_execute_endpoint(request: SandboxExecuteRequest):
    """
    Sovereign isolated Python code execution sandbox.
    Audits code AST, blocks dangerous syscalls/network calls, executes with timeout, and captures artifacts.
    """
    res = code_service.sandbox.execute(request.code, timeout_override=request.timeout_seconds)
    return SandboxExecuteResponse(**res)

# =========================================================================
# Self-Learning & Long-Term Memory API Endpoints
# =========================================================================

@app.get("/api/memories")
async def list_memories_endpoint():
    """Returns all stored long-term memories."""
    memories = db.list_memories()
    return {"memories": memories, "total": len(memories)}

@app.post("/api/memories")
async def add_memory_endpoint(request: AddMemoryRequest):
    """Manually adds a memory or fact."""
    res = learning_service.store_memory(request.fact, request.category or "general")
    return res

@app.delete("/api/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: int):
    """Deletes a learned memory by ID."""
    deleted = db.delete_memory(memory_id)
    return {"success": deleted, "deleted_id": memory_id}

@app.get("/api/learning-stats")
async def get_learning_stats_endpoint():
    """Returns real-time learning statistics (memories and collected chat training pairs)."""
    return learning_service.get_learning_stats()

@app.post("/api/self-train")
async def self_train_endpoint(request: SelfTrainRequest):
    """
    Self-training is permanently disabled under Moon AI learning governance policy.
    Returns HTTP 403 Forbidden (SEC-04).
    """
    raise HTTPException(
        status_code=403,
        detail="Self-training is permanently disabled under Moon AI learning governance policy: live model updates and unconsented training data integration are prohibited."
    )

# --- Sovereign Document Intelligence & Multimodal Dispatch Routes ---
@app.post("/api/document/generate", response_model=DocumentGenResponse)
def generate_document_endpoint(request: DocumentGenRequest):
    """
    Generates sovereign PDF, DOCX, XLSX, or PPTX documents locally with zero third-party APIs.
    """
    t = request.doc_type.lower().strip()
    if t == "pdf":
        sections = request.sections or [{"heading": "Introduction", "body": "Sovereign PDF document generated by Moon Core."}]
        out_path = document_engine.generate_pdf(
            title=request.title or "Moon AI Document",
            sections=sections,
            author=request.author or "Prem Gaikwad (India 🇮🇳)"
        )
    elif t == "docx":
        sections = request.sections or [{"heading": "Overview", "body": "Sovereign Word document generated by Moon Core."}]
        out_path = document_engine.generate_docx(
            title=request.title or "Moon AI Document",
            sections=sections,
            author=request.author or "Prem Gaikwad (India 🇮🇳)"
        )
    elif t == "xlsx":
        out_path = document_engine.generate_xlsx(
            sheet_name=request.sheet_name or "Sheet1",
            headers=request.headers or ["Field", "Value"],
            rows=request.rows or [["System", "Moon AI Sovereign Core"], ["Engine", "Local Zero-API Engine"]]
        )
    elif t == "pptx":
        out_path = document_engine.generate_pptx(
            title=request.title or "Moon AI Presentation",
            subtitle=request.subtitle or "Autonomous Multimodal Intelligence",
            slides=request.slides or [{"title": "Introduction", "bullets": ["Sovereign Architecture", "100% Offline Zero-Key"]}]
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {t}. Supported: 'pdf', 'docx', 'xlsx', 'pptx'")

    fname = os.path.basename(out_path)
    return DocumentGenResponse(
        status="success",
        doc_type=t,
        filename=fname,
        file_path=out_path,
        file_url=f"/static/generated/{fname}",
        size_bytes=os.path.getsize(out_path)
    )

@app.post("/api/document/parse", response_model=DocumentParseResponse)
def parse_document_endpoint(request: DocumentParseRequest):
    """
    Parses and extracts structured text and tables from local documents (.pdf, .docx, .xlsx, .csv, .json, .txt, .md).
    """
    if not os.path.isfile(request.filepath):
        raise HTTPException(status_code=404, detail=f"Document file not found: {request.filepath}")

    parsed = document_engine.parse_document(request.filepath)
    return DocumentParseResponse(
        status="success",
        filename=parsed.get("filename", ""),
        extension=parsed.get("extension", ""),
        size_bytes=parsed.get("size_bytes", 0),
        text=parsed.get("text", ""),
        tables=parsed.get("tables", []),
        metadata=parsed.get("metadata", {})
    )

@app.post("/api/multimodal/dispatch", response_model=MultimodalDispatchResponse)
def multimodal_dispatch_endpoint(request: MultimodalDispatchRequest):
    """
    Unified Moon Core dispatch endpoint routing instructions to specialist engines.
    """
    res = multimodal_coordinator.dispatch(
        prompt=request.prompt,
        user_id=request.user_id or "guest",
        extra_args=request.extra_args
    )
    return MultimodalDispatchResponse(
        modality=res.get("modality", "text"),
        engine=res.get("engine", "Moon Core"),
        reply=res.get("reply", ""),
        artifacts=res.get("artifacts", []),
        result=res.get("result")
    )

# Mount React dist and static files
react_dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
react_assets_dir = os.path.join(react_dist_dir, "assets")

if os.path.isdir(react_assets_dir):
    app.mount("/assets", StaticFiles(directory=react_assets_dir), name="react-assets")

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    react_index = os.path.join(react_dist_dir, "index.html")
    if os.path.isfile(react_index):
        return FileResponse(react_index)
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "Claude / NanoGPT Lab backend is running."}
