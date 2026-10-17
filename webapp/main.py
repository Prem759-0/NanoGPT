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
    
    # Fetch recent history for memory context
    history = db.get_session_messages(session_id, limit=6)
    
    # Save user message to database
    db.add_message(session_id, "user", request.message)

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
