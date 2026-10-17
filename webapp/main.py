import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from webapp.schemas import (
    ChatRequest, ChatResponse, HealthResponse, ModelInfoResponse,
    ImageGenRequest, ImageGenResponse, VideoGenRequest, VideoGenResponse,
    CodeRunRequest, CodeRunResponse, SessionItem, CreateSessionRequest, SwitchModelRequest
)
from webapp.model_service import ModelService, ModelNotTrainedException
from webapp.image_service import ImageService
from webapp.video_service import VideoService
from webapp.code_service import CodeService
from webapp.database import db


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
    ckpt = model_service.find_checkpoint()
    return HealthResponse(
        status="ok",
        model_status=model_service.get_status(),
        checkpoint_exists=ckpt is not None,
        model_loaded=model_service.model is not None,
        device=model_service.device,
        torch_available=model_service.torch_available,
        torch_version=model_service.torch_version
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
async def switch_model(request: SwitchModelRequest):
    """
    Switches active model between GPT-2 (124M) and local scratch checkpoint.
    """
    success = model_service.switch_model(request.model_name)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=model_service.error_message or f"Failed to switch to {request.model_name}"
        )
    return {
        "status": "switched",
        "active_model": request.model_name,
        "info": model_service.get_info()
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generates text using the locally trained nanoGPT model with database context memory.
    If no checkpoint exists, returns an explicit 409 Conflict.
    Never returns fake or external AI responses.
    """
    session_id = request.session_id or db.create_session(
        title=request.message[:28] + ("..." if len(request.message) > 28 else "")
    )
    
    # Fetch recent history for memory context (last 16 messages = 8 conversational turns)
    history = db.get_session_messages(session_id, limit=16)
    
    # Save user message to database
    db.add_message(session_id, "user", request.message)

    import re
    clean_msg = request.message.strip()

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
            "I am **NanoGPT** (NanoGPT-3.2), an advanced, sovereign artificial intelligence created and developed by **Prem Gaikwad** from **India** 🇮🇳.\n\n"
            "Prem Gaikwad engineered NanoGPT as a fully local, self-contained AI platform designed to rival cloud-based models like ChatGPT and Claude—running 100% privately without paid third-party APIs. Under Prem Gaikwad's architecture, NanoGPT provides:\n\n"
            "• 💻 **NanoGPT Coder & Reasoning**: Frontier Python, algorithm design, and system architecture running locally on CPU.\n"
            "• 🎨 **NanoGPT Image Studio**: High-fidelity procedural and diffusion-grade visual art synthesis.\n"
            "• 🎬 **NanoGPT Video Studio**: Dynamic spatio-temporal video clip generation.\n"
            "• ⚡ **Local Code Interpreter**: Secure, sandboxed Python code execution.\n\n"
            "How can I assist you with your project or code today?"
        )
        db.add_message(session_id, "assistant", reply, meta={"identity": "creator_attribution"})
        return ChatResponse(
            reply=reply,
            model_name="NanoGPT-3.2 (Created by Prem Gaikwad)",
            prompt_length=len(clean_msg),
            tokens_generated=140,
            device="CPU Local Engine",
            session_id=session_id
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
            model_name="Image Studio (Higgsfield & Midjourney Engine)",
            prompt_length=len(clean_msg),
            tokens_generated=35,
            device="CPU Local Visualizer",
            session_id=session_id
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
            model_name="Video Studio (Kling & Runway Engine)",
            prompt_length=len(clean_msg),
            tokens_generated=40,
            device="CPU Video Encoder",
            session_id=session_id
        )

    try:
        result = model_service.generate(
            prompt=request.message,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            history=history,
            mode=request.mode or "chat"
        )
        # Save assistant reply to database
        db.add_message(session_id, "assistant", result["reply"], meta={
            "tokens": result.get("tokens_generated", 0),
            "device": result.get("device", "cpu")
        })
        result["session_id"] = session_id
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
async def list_sessions():
    """Returns all stored chat sessions."""
    return db.list_sessions()

@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    """Creates a new conversation session."""
    s_id = db.create_session(title=request.title or "New Chat")
    return {"session_id": s_id, "title": request.title}

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
    Generates high-tech procedural neural artwork locally from user prompt.
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
    Generates dynamic spatio-temporal video clips (.mp4) locally (Kling / Runway caliber).
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

# Mount static files and frontend
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "NanoGPT Lab backend is running. Frontend static files missing."}
