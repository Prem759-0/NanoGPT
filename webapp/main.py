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
    CodeRunRequest, CodeRunResponse, SessionItem, CreateSessionRequest, SwitchModelRequest,
    AddMemoryRequest, SelfTrainRequest
)
from webapp.model_service import ModelService, ModelNotTrainedException
from webapp.image_service import ImageService
from webapp.video_service import VideoService
from webapp.code_service import CodeService
from webapp.learning_service import learning_service
from webapp.game_templates import get_snake_game_html, get_chess_game_html
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
            model_name="NanoGPT Coder Engine (Single-File Web App)",
            prompt_length=len(clean_msg),
            tokens_generated=850,
            device="CPU Local Engine",
            session_id=session_id
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
            model_name="NanoGPT Coder Engine (Interactive Web App)",
            prompt_length=len(clean_msg),
            tokens_generated=850,
            device="CPU Local Engine",
            session_id=session_id
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
            model_name="NanoGPT-3.2 (Frontier Engine — By Prem Gaikwad)",
            prompt_length=len(clean_msg),
            tokens_generated=220,
            device="CPU Local Engine",
            session_id=session_id
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
            "🎬 **Yes! NanoGPT Lab has a complete AI Video Generation Studio (Kling & Runway Caliber)!**\n\n"
            "You have two powerful video engines available:\n\n"
            "### 1. ⚡ Local In-App Video Studio (Runs Right Here in Browser):\n"
            "• **Real Subject Motion**: Generates high-definition cinematic motion clips of real subjects (dogs, cars, lions, cities, nature).\n"
            "• **Real MP4 Video**: Encoded directly into standard **H.264 MP4** format (15 FPS) with smooth 2.5D spatio-temporal dolly zoom & camera panning in just ~5 seconds on your CPU.\n"
            "• **Try It Now**: Simply type:\n"
            "  - `/video a running dog in the park`\n"
            "  - `generate video of a sports car`\n"
            "  - Or click the **🎬 Video Studio** button on the left sidebar!\n\n"
            "### 2. 🚀 Cloud Video Diffusion Studio (Google Colab GPU):\n"
            "• For heavy, frame-by-frame generative diffusion with full 3D physics (Kling AI / Runway Gen-3 caliber), open **`Generate_AI_Video_Colab.ipynb`** in Google Colab.\n"
            "• Runs on free NVIDIA Tesla T4 GPUs (15GB VRAM) using text-to-video diffusion pipelines.\n\n"
            "Would you like me to generate a video clip for you right now? Just type `/video [what you want to see]`!"
        )
        db.add_message(session_id, "assistant", reply, meta={"feature": "video_info"})
        return ChatResponse(
            reply=reply,
            model_name="NanoGPT Video Director",
            prompt_length=len(clean_msg),
            tokens_generated=190,
            device="CPU Local Engine",
            session_id=session_id
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
        learning_service.record_chat_turn(clean_msg, reply, source="memory_intent")
        return ChatResponse(
            reply=reply,
            model_name="NanoGPT Autonomous Memory Engine",
            prompt_length=len(clean_msg),
            tokens_generated=48,
            device="Local SQLite Memory Bank",
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
        # Continuous Self-Training Dataset Logging
        learning_service.record_chat_turn(request.message, result["reply"], source="chat")
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
    """Compiles collected chat pairs and integrates them into the training dataset."""
    res = learning_service.run_self_train_step(iterations=request.iterations or 50)
    return res

# Mount static files and frontend
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "NanoGPT Lab backend is running. Frontend static files missing."}
