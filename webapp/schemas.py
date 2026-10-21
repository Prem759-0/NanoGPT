from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Prompt or input message for the language model"
    )
    mode: Optional[str] = Field(
        default="chat",
        description="Generation persona/mode: 'chat' or 'code'"
    )
    max_new_tokens: int = Field(
        default=180,
        ge=10,
        le=2048,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (lower = more deterministic/precise, higher = more creative)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory"
    )
    user_id: Optional[str] = Field(
        default="guest",
        description="User identity or Clerk user ID for isolated private storage"
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message cannot be empty or whitespace only.")
        return trimmed

class ChatResponse(BaseModel):
    reply: str
    model_name: str
    prompt_length: int
    tokens_generated: int
    device: str
    session_id: str

class SessionItem(BaseModel):
    id: str
    title: str
    user_id: Optional[str] = "guest"
    created_at: str
    updated_at: str

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    user_id: Optional[str] = "guest"

class HealthResponse(BaseModel):
    status: str
    model_status: str
    checkpoint_exists: bool
    model_loaded: bool
    device: str
    torch_available: bool
    torch_version: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_type: str
    checkpoint_path: Optional[str] = None
    model_status: str
    dataset: Optional[str] = None
    vocab_size: Optional[int] = None
    parameters: Optional[int] = None
    block_size: Optional[int] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    description: str

class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Image generation prompt")
    style: Optional[str] = Field(default="auto", description="Style preset: auto, cinematic, cyberpunk, photorealistic, anime, quantum")

class ImageGenResponse(BaseModel):
    image_url: str
    filename: str
    prompt: str
    enhanced_prompt: Optional[str] = None
    style: Optional[str] = None
    width: int
    height: int

class VideoGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Video generation prompt")
    duration_sec: float = Field(default=3.0, ge=1.0, le=10.0, description="Video duration in seconds")
    fps: int = Field(default=15, ge=8, le=30, description="Frames per second")

class VideoGenResponse(BaseModel):
    video_url: str
    filename: str
    prompt: str
    mode: str
    duration: float
    fps: int
    total_frames: int
    width: int
    height: int
    render_time_ms: float

class CodeRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=5000, description="Python code snippet to execute")

class CodeRunResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int
    execution_time_ms: float

class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., description="Target model name: 'llama-3.2', 'gpt2', or 'scratch'")

# --- Self-Learning & Memory Schemas ---
class MemoryItem(BaseModel):
    id: int
    fact: str
    category: str
    created_at: str

class AddMemoryRequest(BaseModel):
    fact: str = Field(..., min_length=2, max_length=500, description="Fact or rule to remember")
    category: Optional[str] = Field(default="general", description="Category for the memory")

class MemoryListResponse(BaseModel):
    memories: List[Dict[str, Any]]
    total: int

class SelfTrainRequest(BaseModel):
    iterations: Optional[int] = Field(default=50, ge=10, le=200)

class SelfTrainResponse(BaseModel):
    success: bool
    message: str
    training_pairs_count: int
    dataset_ready: bool = True
